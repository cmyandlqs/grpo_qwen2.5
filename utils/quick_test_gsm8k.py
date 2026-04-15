#!/usr/bin/env python3
"""
GSM8K 快速测试脚本
专门处理GSM8K数据集格式
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict, Any
import re


def print_separator(title=""):
    """打印分隔线"""
    width = 80
    if title:
        print(f"\n{'=' * (width - len(title) - 2)} {title} {'=' * 2}")
    else:
        print("=" * width)


def load_model_and_tokenizer(model_path: str):
    """加载模型和 tokenizer"""
    print_separator("加载模型")

    print(f"模型路径: {model_path}")

    # 加载 tokenizer
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    print(f"✓ Tokenizer 已加载")

    # 加载模型
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print(f"✓ 模型已加载")

    return model, tokenizer


def load_gsm8k_simple(data_path: str) -> List[Dict]:
    """简单加载GSM8K数据集"""
    print_separator("加载 GSM8K 数据集")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)

        # 获取测试集
        if isinstance(dataset, dict):
            test_dataset = dataset['test']
        else:
            test_dataset = dataset

        print(f"✓ 数据集大小: {len(test_dataset)}")

        # 转换为标准字典列表
        standard_data = []
        for item in test_dataset:
            # 处理不同的列名
            if isinstance(item, dict):
                standard_data.append({
                    'question': item.get('question', item.get('prompt', '')),
                    'answer': item.get('answer', item.get('completion', ''))
                })
            else:
                # 处理Arrow格式
                item_dict = dict(item)
                standard_data.append({
                    'question': item_dict.get('question', item_dict.get('prompt', '')),
                    'answer': item_dict.get('answer', item_dict.get('completion', ''))
                })

        print(f"✓ 转换完成: {len(standard_data)} 个样本")

        # 显示前3个样本
        print("\n前3个样本:")
        for i in range(min(3, len(standard_data))):
            item = standard_data[i]
            print(f"\n样本 {i+1}:")
            print(f"  问题: {item['question'][:100]}...")
            print(f"  答案: {item['answer']}")

        return standard_data

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

        # 返回示例数据
        return [
            {
                'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
                'answer': '72'
            },
            {
                'question': 'What is the sum of the first 10 prime numbers?',
                'answer': '129'
            }
        ]


def format_prompt(question: str) -> str:
    """格式化提示词"""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def extract_answer(text: str) -> str:
    """从生成的文本中提取最终答案"""
    patterns = [
        # LaTeX boxed 格式（优先级最高，Qwen 常用）
        r'\\boxed\{([^}]+)\}',
        r'\\\([^)]*\\boxed\{([^}]+)\}[^(]*\\\)',  # \(\boxed{...}\)
        r'\$[^$]*\\boxed\{([^}]+)\}[^$]*\$',      # $\boxed{...}$

        # 简单 boxed 格式
        r'boxed\{([^}]+)\}',

        # 文本描述格式
        r'(?:answer|结果|答案是?)\s*[:：]?\s*[=-]?\s*(\d+(?:\.\d+)?)',
        r'(?:therefore|thus|so)\s*,?\s*(?:the answer is)?\s*[=-]?\s*(\d+(?:\.\d+)?)',

        # 符号格式
        r'[=-]\s*(\d+(?:\.\d+)?)\s*$',
        r'(\d+(?:\.\d+)?)\s*(?:is the answer|$)',
    ]

    for pattern in patterns:
        try:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        except re.error:
            # 跳过有问题的正则表达式
            continue

    # 备选方案：提取所有数字，取最后一个
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    return "无法提取答案"


def extract_gsm8k_answer(gsm8k_answer: str) -> str:
    """从 GSM8K 数据集的答案格式中提取最终答案

    GSM8K 格式:
    推理过程...
    #### 数字
    """
    # 尝试提取 #### 后的数字
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', gsm8k_answer)
    if match:
        return match.group(1)

    # 备选：提取 <<...>> 格式中的最后一个数字
    matches = re.findall(r'<<.*?=(\d+(?:\.\d+)?)>>', gsm8k_answer)
    if matches:
        return matches[-1]

    # 备选：提取所有数字，取最后一个
    numbers = re.findall(r'\d+(?:\.\d+)?', gsm8k_answer)
    if numbers:
        return numbers[-1]

    return gsm8k_answer.strip()


def run_batch_inference(model, tokenizer, dataset: List[Dict], batch_size: int = 16, debug_samples: int = 5, max_new_tokens: int = 512):
    """批处理推理

    Args:
        model: 模型
        tokenizer: 分词器
        dataset: 数据集
        batch_size: 批大小
        debug_samples: 调试显示的样本数量（前N个样本显示详细信息）
        max_new_tokens: 最大生成 token 数（默认 512）
    """
    print_separator(f"调试模式 - 前 {debug_samples} 个样本详细信息")
    print(f"将在前 {debug_samples} 个样本中显示完整信息，然后开始批量测试")
    print(f"最大生成 token 数: {max_new_tokens}\n")

    results = []
    correct = 0
    total = len(dataset)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = dataset[batch_start:batch_end]

        # 准备数据
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]

        # 格式化和编码
        prompts = [format_prompt(q) for q in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048, pad_token_id=tokenizer.eos_token_id).to(model.device)

        # 生成（使用贪婪解码，更确定）
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id)

        # 解码和处理结果
        for j, (question, answer, output) in enumerate(zip(questions, answers, outputs)):
            # 只解码新生成的部分（去掉输入部分）
            input_length = inputs['input_ids'].shape[1]
            new_tokens = output[input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            predicted = extract_answer(response)
            ground_truth = extract_gsm8k_answer(answer)

            try:
                is_correct = abs(float(predicted) - float(ground_truth)) < 0.001
            except:
                is_correct = predicted.strip() == ground_truth.strip()

            if is_correct:
                correct += 1

            sample_idx = batch_start + j

            # 显示调试样本的详细信息
            if sample_idx < debug_samples:
                input_length = inputs['input_ids'].shape[1]
                new_tokens_count = len(output) - input_length

                print(f"\n{'─' * 80}")
                print(f"📝 调试样本 {sample_idx + 1}/{debug_samples}")
                print(f"{'─' * 80}")
                print(f"\n🔸 问题:")
                print(f"  {question}")
                print(f"\n🔸 GSM8K 标准答案（原始）:")
                print(f"  {answer[:200]}{'...' if len(answer) > 200 else ''}")
                print(f"\n🔸 提取的标准答案:")
                print(f"  {ground_truth}")
                print(f"\n🔸 提示词格式（debug）:")
                print(f"  {format_prompt(question)[:150]}...")
                print(f"\n🔸 生成信息:")
                print(f"  输入 token 数: {input_length}")
                print(f"  生成 token 数: {new_tokens_count}")
                print(f"\n🔸 模型完整输出:")
                print(f"  ──────────────────────────────────────────────────────────────────")
                print(f"  {response}")
                print(f"  ──────────────────────────────────────────────────────────────────")
                print(f"\n🔸 提取的预测答案:")
                print(f"  {predicted}")
                print(f"\n🔸 答案对比:")
                print(f"  预测: {predicted} | 标准: {ground_truth}")
                print(f"\n{'✅ 正确' if is_correct else '❌ 错误'}")
                print(f"{'─' * 80}")

                # 最后一个调试样本后显示分隔
                if sample_idx == debug_samples - 1:
                    print(f"\n{'=' * 80}")
                    print("调试完成，开始批量测试...")
                    print(f"{'=' * 80}\n")

            # 显示进度（非调试样本或所有样本）
            if sample_idx >= debug_samples:
                print(f"\r处理: {sample_idx + 1}/{total} ({(sample_idx + 1)/total*100:.1f}%) - 当前准确率: {correct/(sample_idx + 1)*100:.1f}%", end="")

            results.append({
                'question': question,
                'ground_truth_raw': answer,      # 原始答案
                'ground_truth': ground_truth,     # 提取的答案
                'predicted': predicted,
                'response': response,
                'correct': is_correct
            })

    print()  # 换行

    # 统计结果
    print_separator("结果统计")
    accuracy = correct / total * 100
    print(f"总样本数: {total}")
    print(f"正确数量: {correct}")
    print(f"错误数量: {total - correct}")
    print(f"准确率: {accuracy:.2f}%")

    return results


'''
qwen2.5-3B-Instruct
总样本数: 1319
正确数量: 761
错误数量: 558
准确率: 57.70%
'''
def main():
    """主函数"""
    MODEL_PATH = "/mnt/workspace/qwen2.5-3B-Instruct"
    DATA_PATH = "/mnt/workspace/dataset_eval/gsm8k_data"

    BATCH_SIZE = 32

    print_separator("GSM8K 快速测试")

    try:
        # 加载模型
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

        # 显示显存
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}")
            print(f"显存: {props.total_memory / (1024**3):.2f} GB")

        # 加载数据
        dataset = load_gsm8k_simple(DATA_PATH)

        # 运行推理（前5个样本为调试模式，显示详细信息）
        results = run_batch_inference(model, tokenizer, dataset, batch_size=BATCH_SIZE, debug_samples=5)

        # 保存结果
        output_path = "/mnt/workspace/output/gsm8k_quick_results.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 结果已保存到: {output_path}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
