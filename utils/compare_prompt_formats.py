#!/usr/bin/env python3
"""
快速对比测试：验证 prompt 格式对模型性能的影响
只测试前 10 个样本，快速验证
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def print_sep(title=""):
    width = 60
    if title:
        print(f"\n{'='* (width-len(title)-2)} {title} {'='*2}")
    else:
        print("="*width)


def test_single_sample(model, tokenizer, question: str, answer: str, config_name: str, prompt_template: str):
    """测试单个样本"""

    # 构造提示词
    prompt = prompt_template.format(question=question)

    # 编码
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        pad_token_id=tokenizer.eos_token_id
    ).to(model.device)

    input_len = inputs['input_ids'].shape[1]

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # 提取答案
    import re
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
        r'(?:answer|结果)\s*[:：]?\s*[=-]?\s*(\d+(?:\.\d+)?)',
        r'[=-]\s*(\d+(?:\.\d+)?)\s*$',
    ]
    predicted = None
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            predicted = match.group(1)
            break

    if not predicted:
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            predicted = numbers[-1]
        else:
            predicted = "无法提取"

    # 提取标准答案
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer)
    if match:
        ground_truth = match.group(1)
    else:
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        ground_truth = numbers[-1] if numbers else answer

    # 检查正确性
    try:
        is_correct = abs(float(predicted) - float(ground_truth)) < 0.001
    except:
        is_correct = predicted.strip() == ground_truth.strip()

    return {
        'config': config_name,
        'response': response,
        'predicted': predicted,
        'ground_truth': ground_truth,
        'correct': is_correct,
        'input_tokens': input_len,
        'output_tokens': outputs.shape[1] - input_len
    }


def main():
    print_sep("GSM8K Prompt 格式快速对比测试")

    # 配置
    MODEL_PATH = "/mnt/workspace/qwen2.5-3B-Instruct"
    DATA_PATH = "/mnt/workspace/dataset_eval/gsm8k_data"
    NUM_SAMPLES = 10  # 只测试 10 个样本

    print(f"\n模型: {MODEL_PATH}")
    print(f"数据: {DATA_PATH}")
    print(f"测试样本: {NUM_SAMPLES}")

    # 加载模型
    print_sep("加载模型")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 模型已加载")

    # 加载数据
    print_sep("加载数据")
    from datasets import load_from_disk
    dataset = load_from_disk(DATA_PATH)
    if isinstance(dataset, dict):
        dataset = dataset['test']
    print(f"✓ 数据集大小: {len(dataset)}")

    # 定义不同的 prompt 模板
    prompt_templates = {
        "原始格式(无system)": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        "标准格式(有system)": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        "CoT格式": "<|im_start|>system\nYou are a helpful assistant. Please think step by step and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
    }

    # 测试每个配置
    results_by_config = {name: [] for name in prompt_templates.keys()}

    for i in range(min(NUM_SAMPLES, len(dataset))):
        item = dataset[i]
        question = item['question']
        answer = item['answer']

        print(f"\n样本 {i+1}/{NUM_SAMPLES}")
        print(f"问题: {question[:80]}...")

        for config_name, template in prompt_templates.items():
            result = test_single_sample(model, tokenizer, question, answer, config_name, template)
            results_by_config[config_name].append(result)

            status = "✓" if result['correct'] else "✗"
            print(f"  {config_name:20s}: {result['predicted']:10s} vs {result['ground_truth']:10s} {status}")

    # 统计结果
    print_sep("统计结果")
    print(f"\n{'配置':<25s} {'正确率':<10s} {'正确数':<10s}")
    print("-"*60)

    for config_name, results in results_by_config.items():
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results) * 100
        print(f"{config_name:<25s} {accuracy:.1f}%       {correct_count}/{len(results)}")

    # 显示详细输出对比
    print_sep("详细输出对比 (前3个样本)")

    for i in range(min(3, NUM_SAMPLES)):
        print(f"\n{'='*60}")
        print(f"样本 {i+1}")
        print(f"问题: {dataset[i]['question'][:80]}...")
        print(f"标准答案: {results_by_config[list(prompt_templates.keys())[0]][i]['ground_truth']}")
        print()

        for config_name, results in results_by_config.items():
            r = results[i]
            print(f"{config_name}:")
            print(f"  预测: {r['predicted']} ({'✓' if r['correct'] else '✗'})")
            print(f"  输出: {r['response'][:200]}{'...' if len(r['response']) > 200 else ''}")
            print()

    # 保存结果
    output_path = "/mnt/workspace/output/prompt_comparison_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_by_config, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
