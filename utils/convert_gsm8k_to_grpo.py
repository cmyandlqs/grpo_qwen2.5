#!/usr/bin/env python3
"""
将GSM8K数据集转换为ms-swift GRPO训练格式

原始格式（用于评测）:
- question: 数学问题
- answer: 推理过程 + #### <答案>

GRPO格式要求:
- messages: 对话消息列表，包含user角色的问题
- solution: 答案（用于accuracy奖励函数）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from datasets import load_from_disk, Dataset
except ImportError:
    print("错误: 需要安装 datasets 库")
    print("请运行: pip install datasets")
    sys.exit(1)


def extract_gsm8k_answer(gsm8k_answer: str) -> str:
    """从GSM8K数据集的答案格式中提取最终答案"""
    import re
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', gsm8k_answer)
    if match:
        return match.group(1)

    # Fallback: 提取最后一个数字
    numbers = re.findall(r'\d+(?:\.\d+)?', gsm8k_answer)
    if numbers:
        return numbers[-1]

    return gsm8k_answer.strip()


def convert_to_grpo_format(
    dataset_path: str,
    output_path: str,
    split: str = 'train',
    num_samples: int = -1,
    system_prompt: str = None
):
    """
    转换GSM8K数据集为GRPO格式

    Args:
        dataset_path: 原始数据集路径
        output_path: 输出路径
        split: 数据集划分 (train/test)
        num_samples: 转换样本数，-1表示全部
        system_prompt: 可选的system prompt
    """

    print("=" * 80)
    print("GSM8K → GRPO 格式转换")
    print("=" * 80)
    print(f"输入路径: {dataset_path}")
    print(f"输出路径: {output_path}")
    print(f"数据划分: {split}")
    if num_samples > 0:
        print(f"样本数量: {num_samples}")
    print("=" * 80)
    print()

    # 加载数据集
    print("正在加载数据集...")
    try:
        full_dataset = load_from_disk(dataset_path)

        # 处理DatasetDict
        if hasattr(full_dataset, 'keys'):
            if split not in full_dataset.keys():
                print(f"错误: 数据集中没有 '{split}' 划分")
                print(f"可用划分: {list(full_dataset.keys())}")
                return False
            dataset = full_dataset[split]
        else:
            dataset = full_dataset

        print(f"✓ 数据集已加载: {len(dataset)} 个样本")
        print(f"  列名: {dataset.column_names}")

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False

    # 限制样本数
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"  限制样本数: {len(dataset)}")

    print()

    # 转换格式
    print("正在转换格式...")
    grpo_data = []

    for idx, item in enumerate(dataset):
        question = item.get('question', '')
        answer = item.get('answer', '')

        # 提取最终答案作为solution
        solution = extract_gsm8k_answer(answer)

        # 构建messages
        messages = []

        # 添加system prompt（如果提供）
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 添加user消息
        messages.append({
            "role": "user",
            "content": question
        })

        # 构建GRPO格式数据
        grpo_item = {
            "messages": messages,
            "solution": solution
        }

        grpo_data.append(grpo_item)

        # 显示进度
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  处理进度: {idx + 1}/{len(dataset)}")

    print(f"✓ 转换完成: {len(grpo_data)} 个样本")
    print()

    # 保存为JSONL格式
    print(f"正在保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✓ 保存完成")
    print()

    # 显示示例
    print("=" * 80)
    print("数据格式示例 (前3个样本):")
    print("=" * 80)
    for i, item in enumerate(grpo_data[:3]):
        print(f"\n样本 {i + 1}:")
        print(json.dumps(item, ensure_ascii=False, indent=2))
    print()

    # 统计信息
    print("=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"总样本数: {len(grpo_data)}")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="将GSM8K数据集转换为ms-swift GRPO格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 转换训练集（全部样本）
  python convert_gsm8k_to_grpo.py \\
      --input /mnt/workspace/dataset_eval/gsm8k_data \\
      --output /mnt/workspace/dataset_eval/gsm8k_grpo_train.jsonl \\
      --split train

  # 转换测试集（100个样本）
  python convert_gsm8k_to_grpo.py \\
      --input /mnt/workspace/dataset_eval/gsm8k_data \\
      --output /mnt/workspace/dataset_eval/gsm8k_grpo_test.jsonl \\
      --split test \\
      --num-samples 100

  # 添加system prompt（数学推理专用）
  python convert_gsm8k_to_grpo.py \\
      --input /mnt/workspace/dataset_eval/gsm8k_data \\
      --output /mnt/workspace/dataset_eval/gsm8k_grpo_train.jsonl \\
      --split train \\
      --system "You are a helpful math assistant. Solve step by step."
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入数据集路径（原始GSM8K格式）'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出JSONL文件路径（GRPO格式）'
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='数据集划分 (默认: train)'
    )

    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=-1,
        help='转换样本数，-1表示全部 (默认: -1)'
    )

    parser.add_argument(
        '--system',
        type=str,
        default=None,
        help='可选的system prompt'
    )

    args = parser.parse_args()

    # 执行转换
    success = convert_to_grpo_format(
        dataset_path=args.input,
        output_path=args.output,
        split=args.split,
        num_samples=args.num_samples,
        system_prompt=args.system
    )

    if success:
        print("=" * 80)
        print("✓ 转换成功！")
        print("=" * 80)
        print()
        print("使用方法:")
        print(f"  swift rlhf \\")
        print(f"    --rlhf_type grpo \\")
        print(f"    --dataset {args.output} \\")
        print(f"    --reward_funcs accuracy \\")
        print(f"    ...")
        print()
        return 0
    else:
        print("=" * 80)
        print("❌ 转换失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
