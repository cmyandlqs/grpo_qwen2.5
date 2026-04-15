#!/usr/bin/env python3
"""
GSM8K 评测流程验证脚本
验证数据加载、提示词格式、模型输出、答案提取等各个环节
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict, Any
import re


def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def load_and_inspect_dataset(data_path: str, num_samples: int = 5):
    """检查数据集加载"""
    print_section("1. 数据集加载检查")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)

        # 获取测试集
        if isinstance(dataset, dict):
            test_dataset = dataset['test']
        else:
            test_dataset = dataset

        print(f"✓ 数据集大小: {len(test_dataset)}")
        print(f"✓ 数据集列: {test_dataset.column_names}")

        # 显示原始数据样本
        print(f"\n原始数据样本 (前{num_samples}个):")
        for i in range(min(num_samples, len(test_dataset))):
            item = test_dataset[i]
            print(f"\n--- 样本 {i+1} ---")
            print(f"数据类型: {type(item)}")
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str):
                        print(f"{key}: {value[:200]}{'...' if len(value) > 200 else ''}")
                    else:
                        print(f"{key}: {value}")
            else:
                # Arrow 格式
                item_dict = dict(item)
                for key, value in item_dict.items():
                    if isinstance(value, str):
                        print(f"{key}: {value[:200]}{'...' if len(value) > 200 else ''}")
                    else:
                        print(f"{key}: {value}")

        return test_dataset

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_prompt_format(question: str):
    """验证提示词格式"""
    print_section("2. 提示词格式检查")

    # 当前脚本使用的格式
    current_format = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    print("当前脚本使用的提示词格式:")
    print("-" * 80)
    print(current_format)
    print("-" * 80)

    # 检查是否是 Qwen2.5 推荐格式
    print("\nQwen2.5 标准对话格式应为:")
    print("-" * 80)
    standard_format = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    print(standard_format)
    print("-" * 80)

    # Qwen2.5 Instruct 模型通常需要 system prompt
    print("\n⚠️  注意: Qwen2.5-Instruct 模型通常需要 system prompt")
    print("   当前格式缺少 system prompt，可能影响模型性能")

    return current_format, standard_format


def load_model_and_tokenizer(model_path: str):
    """加载模型"""
    print_section("3. 模型加载检查")

    print(f"模型路径: {model_path}")

    try:
        print("正在加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"✓ Tokenizer 已加载")
        print(f"  词汇表大小: {len(tokenizer)}")
        print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

        print("\n正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"✓ 模型已加载")

        # 检查特殊 token
        special_tokens = tokenizer.special_tokens_map
        print(f"\n特殊 tokens:")
        for key, value in special_tokens.items():
            print(f"  {key}: {value}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_generation_with_details(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """测试生成并显示详细信息"""
    print_section("4. 模型生成测试")

    print("测试提示词:")
    print("-" * 80)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 80)

    # 编码
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        pad_token_id=tokenizer.eos_token_id
    ).to(model.device)

    input_length = inputs['input_ids'].shape[1]
    print(f"\n输入 token 数: {input_length}")
    print(f"最大生成 token 数: {max_new_tokens}")
    print(f"总 token 限制: 2048 (输入) + {max_new_tokens} (生成) = {2048 + max_new_tokens}")

    # 检查是否有截断风险
    if input_length > 1900:
        print(f"\n⚠️  警告: 输入长度 {input_length} 接近 2048 限制，")
        print(f"   可能导致输出被截断！")

    # 生成
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    output_length = outputs.shape[1]
    new_tokens = output_length - input_length

    print(f"\n生成结果:")
    print(f"  输出总 token 数: {output_length}")
    print(f"  新生成 token 数: {new_tokens}")

    # 检查是否达到长度限制
    if new_tokens >= max_new_tokens:
        print(f"  ⚠️  警告: 生成达到 max_new_tokens 限制，输出可能被截断！")
    else:
        print(f"  ✓ 输出完整，未达到长度限制")

    # 解码
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens_only = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print(f"\n完整输出:")
    print("-" * 80)
    print(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)
    print("-" * 80)

    print(f"\n新生成部分:")
    print("-" * 80)
    print(new_tokens_only)
    print("-" * 80)

    return full_text, new_tokens_only


def test_answer_extraction(response: str, ground_truth: str):
    """测试答案提取"""
    print_section("5. 答案提取测试")

    print(f"模型输出:")
    print("-" * 80)
    print(response)
    print("-" * 80)

    print(f"\n标准答案 (原始):")
    print("-" * 80)
    print(ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth)
    print("-" * 80)

    # 测试各种提取模式
    patterns = [
        (r'\\boxed\{([^}]+)\}', "LaTeX \\boxed{}"),
        (r'\\\([^)]*\\boxed\{([^}]+)\}[^(]*\\\)', "LaTeX \\(\\boxed{}\\)"),
        (r'\$[^$]*\\boxed\{([^}]+)\}[^$]*\$', "LaTeX $\\boxed{}$"),
        (r'boxed\{([^}]+)\}', "plain boxed{}"),
        (r'(?:answer|结果|答案是?)\s*[:：]?\s*[=-]?\s*(\d+(?:\.\d+)?)', "文本描述"),
        (r'(?:therefore|thus|so)\s*,?\s*(?:the answer is)?\s*[=-]?\s*(\d+(?:\.\d+)?)', "英文描述"),
        (r'[=-]\s*(\d+(?:\.\d+)?)\s*$', "符号结尾"),
        (r'(\d+(?:\.\d+)?)\s*(?:is the answer|$)', "答案结尾"),
    ]

    print("\n正则模式匹配结果:")
    for pattern, name in patterns:
        try:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                print(f"  ✓ {name:20s}: 提取到 '{match.group(1)}'")
            else:
                print(f"  ✗ {name:20s}: 未匹配")
        except re.error as e:
            print(f"  ✗ {name:20s}: 正则错误 {e}")

    # 当前脚本使用的提取函数
    def extract_answer(text: str) -> str:
        """从生成的文本中提取最终答案"""
        for pattern, _ in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1)
            except re.error:
                continue

        # 备选方案：提取所有数字，取最后一个
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return "无法提取答案"

    def extract_gsm8k_answer(gsm8k_answer: str) -> str:
        """从 GSM8K 数据集的答案格式中提取最终答案"""
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

    predicted = extract_answer(response)
    ground_truth_extracted = extract_gsm8k_answer(ground_truth)

    print(f"\n最终提取结果:")
    print(f"  预测答案: {predicted}")
    print(f"  标准答案: {ground_truth_extracted}")

    # 检查是否正确
    try:
        is_correct = abs(float(predicted) - float(ground_truth_extracted)) < 0.001
        print(f"  结果: {'✓ 正确' if is_correct else '✗ 错误'}")
    except:
        is_correct = predicted.strip() == ground_truth_extracted.strip()
        print(f"  结果: {'✓ 正确' if is_correct else '✗ 错误'}")

    return predicted, ground_truth_extracted, is_correct


def compare_prompt_formats(model, tokenizer, question: str):
    """比较不同提示词格式的效果"""
    print_section("6. 提示词格式对比")

    formats = {
        "当前格式 (无 system)": f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        "标准格式 (有 system)": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        "CoT 格式": f"<|im_start|>system\nYou are a helpful assistant. Please think step by step and show your work.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
    }

    results = {}

    for name, prompt in formats.items():
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            pad_token_id=tokenizer.eos_token_id
        ).to(model.device)

        input_length = inputs['input_ids'].shape[1]
        print(f"输入 token 数: {input_length}")

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        print(f"输出:\n{response[:500]}{'...' if len(response) > 500 else ''}")

        results[name] = response

    return results


def main():
    """主函数"""
    print_section("GSM8K 评测流程验证")

    # 配置
    MODEL_PATH = "/mnt/workspace/qwen2.5-3B-Instruct"  # 使用 3B Instruct 模型
    DATA_PATH = "/mnt/workspace/dataset_eval/gsm8k_data"
    NUM_TEST_SAMPLES = 3

    print(f"模型路径: {MODEL_PATH}")
    print(f"数据路径: {DATA_PATH}")
    print(f"测试样本数: {NUM_TEST_SAMPLES}")

    # 1. 检查数据集
    dataset = load_and_inspect_dataset(DATA_PATH, NUM_TEST_SAMPLES)
    if dataset is None:
        print("❌ 数据集加载失败，终止验证")
        return

    # 2. 检查提示词格式
    sample_question = dataset[0]['question']
    current_fmt, standard_fmt = verify_prompt_format(sample_question)

    # 3. 加载模型
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    if model is None or tokenizer is None:
        print("❌ 模型加载失败，终止验证")
        return

    # 4. 测试第一个样本的完整流程
    print_section("完整流程测试 - 样本 1")

    sample = dataset[0]
    question = sample['question']
    answer = sample['answer']

    print(f"问题: {question}")
    print(f"答案: {answer[:100]}...")

    # 使用当前脚本的格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    full_text, response = test_generation_with_details(model, tokenizer, prompt, max_new_tokens=512)

    # 5. 测试答案提取
    predicted, ground_truth, is_correct = test_answer_extraction(response, answer)

    # 6. 对比不同提示词格式
    print("\n是否要测试不同提示词格式？(这会花费更多时间)")
    print("在服务器上运行时可以添加这个测试")

    # 总结
    print_section("验证总结")

    issues = []

    # 检查 1: system prompt
    if "system" not in prompt:
        issues.append("⚠️  当前提示词格式缺少 system prompt")

    # 检查 2: 输入长度
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs['input_ids'].shape[1] > 1900:
        issues.append("⚠️  输入长度接近限制，可能需要更短的提示词")

    # 检查 3: 答案提取
    if not is_correct:
        issues.append("⚠️  答案提取可能有问题，或模型输出格式不符合预期")

    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ 基本流程正常")

    print("\n建议:")
    print("  1. 添加 system prompt 到提示词格式中")
    print("  2. 检查更多样本，确认答案提取的准确性")
    print("  3. 考虑使用 CoT (Chain of Thought) 提示提高准确率")


if __name__ == "__main__":
    main()
