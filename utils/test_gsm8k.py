#!/usr/bin/env python3
"""
GSM8K 数据集推理测试脚本
加载 Qwen2.5 模型对 GSM8K 测试集进行推理
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


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
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
    print(f"✓ Tokenizer 已加载 (词汇表大小: {len(tokenizer)})")

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

    # 显示参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 总参数量: {total_params / 1e6:.2f} M")

    return model, tokenizer


def load_gsm8k_dataset(data_path: str, split: str = "test") -> List[Dict[str, Any]]:
    """加载 GSM8K 数据集"""
    print_separator(f"加载 GSM8K 数据集 ({split} 集)")

    # 检查数据集路径
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    # 尝试加载 dataset_dict.json
    dataset_dict_path = os.path.join(data_path, "dataset_dict.json")
    split_path = os.path.join(data_path, split)

    if os.path.exists(dataset_dict_path):
        with open(dataset_dict_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        print(f"✓ 数据集信息: {dataset_info}")

    # 加载 split 数据
    if os.path.exists(split_path):
        # Hugging Face dataset 格式
        data_file = os.path.join(split_path, "data-00000-of-00001.arrow")
        if os.path.exists(data_file):
            print(f"✓ 找到数据文件: {data_file}")
            # 使用 datasets 库加载
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(data_path)
                dataset_size = len(dataset[split])
                print(f"✓ 数据集大小: {dataset_size}")
                print(f"ℹ 将测试全部 {dataset_size} 个样本")
                return dataset[split]
            except ImportError:
                print("⚠ datasets 库未安装，尝试手动加载...")

        # 尝试从 JSON 加载
        json_path = os.path.join(split_path, "dataset_info.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print(f"✓ Split 信息: {info}")

    # 如果是标准格式，尝试直接加载
    # 假设数据已经转换为标准格式
    print("⚠ 无法直接加载数据集，使用示例数据")

    # 返回示例数据
    sample_data = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "72"
        },
        {
            "question": "What is the sum of the first 10 prime numbers?",
            "answer": "129"
        },
        {
            "question": "A baker has 120 loaves of bread. If he sells 3/5 of them in the morning and 1/4 of the remainder in the afternoon, how many loaves are left?",
            "answer": "36"
        }
    ]
    return sample_data


def format_prompt(question: str) -> str:
    """格式化提示词"""
    # Qwen2.5 的对话格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def extract_answer(text: str) -> str:
    """从生成的文本中提取最终答案"""
    # 尝试多种模式提取数字答案
    patterns = [
        r'(?:answer|结果|答案是?)\s*[:：]?\s*[=-]?\s*(\d+(?:\.\d+)?)',
        r'(?:therefore|thus|so)\s*,?\s*(?:the answer is)?\s*[=-]?\s*(\d+(?:\.\d+)?)',
        r'[=-]\s*(\d+(?:\.\d+)?)\s*$',
        r'(\d+(?:\.\d+)?)\s*(?:is the answer|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)

    # 如果没有找到模式，尝试获取最后一个数字
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    return "无法提取答案"


def check_answer(predicted: str, ground_truth: str) -> bool:
    """检查答案是否正确"""
    try:
        # 转换为数字进行比较
        pred_num = float(predicted.strip())
        true_num = float(ground_truth.strip())
        return abs(pred_num - true_num) < 0.001
    except (ValueError, TypeError):
        # 如果无法转换为数字，进行字符串比较
        return predicted.strip().lower() == ground_truth.strip().lower()


def calculate_optimal_batch_size(model, tokenizer, max_new_tokens: int = 512) -> int:
    """根据显存自动计算最优batch size"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
        available_memory = total_memory - reserved_memory

        # 估算每个样本需要的显存（包括模型、KV cache、中间激活）
        # Qwen2.5-0.5B 约1-2GB，每个推理样本约0.3-0.5GB
        model_memory = reserved_memory
        memory_per_sample = 0.5  # GB per sample (保守估计)
        safe_memory = available_memory * 0.8  # 留20%余量

        optimal_batch_size = int(safe_memory / memory_per_sample)

        # 限制在合理范围
        optimal_batch_size = max(1, min(optimal_batch_size, 32))

        print(f"✓ 总显存: {total_memory:.2f} GB")
        print(f"✓ 已用显存: {reserved_memory:.2f} GB")
        print(f"✓ 可用显存: {available_memory:.2f} GB")
        print(f"✓ 推荐 Batch Size: {optimal_batch_size}")

        return optimal_batch_size
    else:
        return 1  # CPU fallback


def run_inference_test(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    num_samples: int = -1,
    max_new_tokens: int = 512,
    batch_size: int = 8
):
    """运行推理测试（支持批处理）

    Args:
        num_samples: 测试样本数量，-1 表示测试全部样本
        batch_size: 批处理大小，24G显存建议8-16
    """
    # 确定实际测试数量
    if num_samples == -1:
        actual_num = len(dataset)
        print_separator(f"推理测试 (测试全部 {actual_num} 个样本)")
    else:
        actual_num = min(num_samples, len(dataset))
        print_separator(f"推理测试 (测试 {actual_num} 个样本)")

    print(f"✓ Batch Size: {batch_size}")
    print(f"✓ 预计加速比: ~{batch_size}x")

    results = []
    correct = 0

    # 批处理推理
    for batch_start in range(0, actual_num, batch_size):
        batch_end = min(batch_start + batch_size, actual_num)
        batch_items = dataset[batch_start:batch_end]
        current_batch_size = len(batch_items)

        # 准备批处理数据
        questions = []
        answers = []
        indices = []

        for idx, item in enumerate(batch_items):
            i = batch_start + idx
            if isinstance(item, dict):
                question = item.get("question", "")
                answer = item.get("answer", "")
            else:
                question = item["question"]
                answer = item["answer"]

            questions.append(question)
            answers.append(answer)
            indices.append(i)

        # 显示进度
        print(f"\r处理中: {batch_end}/{actual_num} ({batch_end/actual_num*100:.1f}%) - 当前准确率: {correct/(batch_end)*100:.1f}%", end="")

        # 格式化提示词
        prompts = [format_prompt(q) for q in questions]

        # 编码输入（批处理）
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id
        ).to(model.device)

        # 生成（批处理）
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码输出
        for j in range(current_batch_size):
            full_text = tokenizer.decode(outputs[j], skip_special_tokens=True)

            # 提取 assistant 的回复
            if "<|im_start|>assistant" in full_text:
                response = full_text.split("<|im_start|>assistant")[-1].strip()
            else:
                prompt = prompts[j]
                response = full_text[len(prompt):]

            # 提取答案
            predicted_answer = extract_answer(response)

            # 检查正确性
            is_correct = check_answer(predicted_answer, answers[j])
            if is_correct:
                correct += 1

            # 统计信息
            input_len = inputs["input_ids"][j].shape[0]
            output_len = outputs[j].shape[0] - input_len

            # 显示前几个样本的详细信息
            if indices[j] < 3:
                print(f"\n【样本 {indices[j]+1}】")
                print(f"问题: {questions[j][:80]}...")
                print(f"标准答案: {answers[j]}")
                print(f"预测答案: {predicted_answer}")
                print(f"{'✓ 正确' if is_correct else '✗ 错误'}")

            results.append({
                "question": questions[j],
                "ground_truth": answers[j],
                "predicted": predicted_answer,
                "response": response,
                "correct": is_correct,
                "input_tokens": input_len,
                "output_tokens": output_len
            })

    print()  # 换行

    # 显示统计结果
    print_separator("测试结果统计")
    accuracy = correct / actual_num * 100
    print(f"测试样本数: {actual_num}")
    print(f"正确数量: {correct}")
    print(f"错误数量: {actual_num - correct}")
    print(f"准确率: {accuracy:.2f}%")

    return results


def show_memory_usage():
    """显示显存使用情况"""
    print_separator("显存使用情况")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = props.total_memory / (1024 ** 3)

            print(f"GPU {i} ({props.name}):")
            print(f"  总显存:     {total:.2f} GB")
            print(f"  已分配:     {allocated:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"  已预留:     {reserved:.2f} GB ({reserved/total*100:.1f}%)")
            print(f"  可用显存:   {total - reserved:.2f} GB")
    else:
        print("未检测到 CUDA 设备")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存测试结果"""
    print_separator("保存结果")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_path}")


def main():
    """主函数"""
    # 配置
    MODEL_PATH = "/mnt/workspace/qwen2.5-0.5B-base"
    DATA_PATH = "/mnt/workspace/dataset_eval/gsm8k_data"

    # 测试参数
    NUM_SAMPLES = -1  # -1 表示测试全部样本，也可以设置为具体数字如 100
    MAX_NEW_TOKENS = 512
    AUTO_BATCH_SIZE = True  # 是否自动计算batch size，False则使用下面的手动设置
    MANUAL_BATCH_SIZE = 16  # 手动batch size，24G显存建议16

    print_separator("GSM8K 数据集推理测试")
    print(f"ℹ 测试模式: {'全部样本' if NUM_SAMPLES == -1 else f'{NUM_SAMPLES} 个样本'}")

    try:
        # 1. 加载模型和 tokenizer
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

        # 2. 显示显存使用
        show_memory_usage()

        # 3. 计算最优batch size
        print_separator("Batch Size 优化")
        if AUTO_BATCH_SIZE:
            BATCH_SIZE = calculate_optimal_batch_size(model, tokenizer, MAX_NEW_TOKENS)
        else:
            BATCH_SIZE = MANUAL_BATCH_SIZE
            print(f"ℹ 使用手动设置的 Batch Size: {BATCH_SIZE}")

        # 4. 加载数据集
        dataset = load_gsm8k_dataset(DATA_PATH, split="test")

        # 4. 运行推理测试
        results = run_inference_test(
            model,
            tokenizer,
            dataset,
            num_samples=NUM_SAMPLES,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE
        )

        # 5. 保存结果
        output_path = "/mnt/workspace/output/gsm8k_test_results.json"
        save_results(results, output_path)

        # 6. 最终显存使用
        show_memory_usage()

        print_separator("测试完成")
        print(f"✓ 详细结果已保存到: {output_path}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
