#!/usr/bin/env python3
"""
GSM8K 模型评测脚本
系统化评测模型在 GSM8K 数据集上的表现，支持命令行参数和完整日志记录
"""

import os
import sys
import json
import time
import torch
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import re

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM


class GSM8KEvaluator:
    """GSM8K 评测器"""

    # Prompt 模板定义
    PROMPT_TEMPLATES = {
        "no_system": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        "standard": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        "cot": "<|im_start|>system\nYou are a helpful assistant. Please think step by step and show your work. Put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
    }

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """初始化评测器

        Args:
            config: 评测配置字典
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志
        self.log_file = self.output_dir / "log.txt"
        self.logger = open(self.log_file, 'w', encoding='utf-8')

        # 记录开始时间
        self.start_time = time.time()

        # 显存统计
        self.memory_stats = {
            "peak_allocated_gb": 0,
            "peak_reserved_gb": 0,
        }

        self.log("=" * 80)
        self.log("GSM8K 模型评测")
        self.log("=" * 80)
        self.log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

    def log(self, message: str, also_print: bool = True):
        """记录日志"""
        self.logger.write(message + "\n")
        if also_print:
            print(message)

    def log_config(self):
        """记录配置信息"""
        self.log("=" * 80)
        self.log("配置信息")
        self.log("=" * 80)

        # 模型信息
        self.log(f"\n模型配置:")
        self.log(f"  模型路径: {self.config['model_path']}")
        self.log(f"  提示词格式: {self.config['prompt_format']}")

        # 获取 system prompt
        template = self.PROMPT_TEMPLATES[self.config['prompt_format']]
        if "<|im_start|>system" in template:
            # 提取 system prompt
            match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', template, re.DOTALL)
            if match:
                system_prompt = match.group(1).strip()
                self.log(f"  System Prompt: {system_prompt}")

        # 超参数
        self.log(f"\n超参数:")
        self.log(f"  Batch Size: {self.config['batch_size']}")
        self.log(f"  Max New Tokens: {self.config['max_new_tokens']}")
        self.log(f"  Temperature: {self.config['temperature']}")
        self.log(f"  Top P: {self.config['top_p']}")

        # 评测设置
        self.log(f"\n评测设置:")
        self.log(f"  测试样本数: {self.config['num_samples'] if self.config['num_samples'] > 0 else '全部'}")
        self.log(f"  数据集路径: {self.config['data_path']}")
        self.log("")

    def load_model(self):
        """加载模型和 tokenizer"""
        self.log("=" * 80)
        self.log("加载模型")
        self.log("=" * 80)

        model_path = self.config['model_path']

        # 加载 tokenizer
        self.log("正在加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            padding_side='left'  # decoder-only 架构需要左侧 padding
        )
        # 设置 pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.log(f"✓ Tokenizer 已加载")
        self.log(f"  词汇表大小: {len(self.tokenizer)}")

        # 加载模型
        self.log("正在加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.log(f"✓ 模型已加载")

        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        self.config['model_params'] = total_params / 1e6  # 转换为百万
        self.log(f"  模型参数量: {self.config['model_params']:.2f} M")

        # GPU 信息
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.config['gpu_name'] = props.name
            self.config['gpu_total_memory_gb'] = props.total_memory / (1024**3)
            self.log(f"\nGPU 信息:")
            self.log(f"  GPU: {props.name}")
            self.log(f"  总显存: {self.config['gpu_total_memory_gb']:.2f} GB")

        self.log("")

    def load_dataset(self) -> List[Dict]:
        """加载数据集"""
        self.log("=" * 80)
        self.log("加载数据集")
        self.log("=" * 80)

        try:
            from datasets import load_from_disk
            dataset = load_from_disk(self.config['data_path'])

            if isinstance(dataset, dict):
                dataset = dataset['test']

            self.log(f"✓ 数据集已加载: {len(dataset)} 个样本")

            # 限制测试样本数
            if self.config['num_samples'] > 0:
                # Dataset 切片返回字典，需要转换
                sliced = dataset[:self.config['num_samples']]
                # 将字典转换为列表
                dataset = [
                    {key: values[i] for key, values in sliced.items()}
                    for i in range(len(list(sliced.values())[0]))
                ]
                self.log(f"  限制测试样本数: {len(dataset)}")
            else:
                # 转换整个数据集为列表
                dataset = [dict(item) for item in dataset]

            self.log("")
            return dataset

        except Exception as e:
            self.log(f"❌ 数据集加载失败: {e}")
            raise

    def update_memory_stats(self):
        """更新显存统计"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)

            self.memory_stats['peak_allocated_gb'] = max(
                self.memory_stats['peak_allocated_gb'], allocated
            )
            self.memory_stats['peak_reserved_gb'] = max(
                self.memory_stats['peak_reserved_gb'], reserved
            )

    def extract_answer(self, text: str) -> str:
        """从生成的文本中提取最终答案"""
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
            r'(?:answer|结果|答案是?)\s*[:：]?\s*[=-]?\s*(\d+(?:\.\d+)?)',
            r'(?:therefore|thus|so)\s*,?\s*(?:the answer is)?\s*[=-]?\s*(\d+(?:\.\d+)?)',
            r'[=-]\s*(\d+(?:\.\d+)?)\s*$',
            r'(\d+(?:\.\d+)?)\s*(?:is the answer|$)',
        ]

        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1)
            except re.error:
                continue

        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return "无法提取答案"

    def extract_gsm8k_answer(self, gsm8k_answer: str) -> str:
        """从 GSM8K 数据集的答案格式中提取最终答案"""
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', gsm8k_answer)
        if match:
            return match.group(1)

        numbers = re.findall(r'\d+(?:\.\d+)?', gsm8k_answer)
        if numbers:
            return numbers[-1]

        return gsm8k_answer.strip()

    def run_evaluation(self, dataset: List[Dict]):
        """运行评测"""
        self.log("=" * 80)
        self.log("开始评测")
        self.log("=" * 80)

        template = self.PROMPT_TEMPLATES[self.config['prompt_format']]
        batch_size = self.config['batch_size']
        max_new_tokens = self.config['max_new_tokens']

        results = []
        correct = 0
        total = len(dataset)

        inference_start = time.time()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = dataset[batch_start:batch_end]

            questions = [item['question'] for item in batch]
            answers = [item['answer'] for item in batch]

            # 格式化提示词
            prompts = [template.format(question=q) for q in questions]

            # 编码
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                pad_token_id=self.tokenizer.eos_token_id
            ).to(self.model.device)

            # 生成
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False if self.config['temperature'] == 0 else True,
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 更新显存统计
            self.update_memory_stats()

            # 解码和处理
            for j, (question, answer, output) in enumerate(zip(questions, answers, outputs)):
                input_length = inputs['input_ids'].shape[1]
                new_tokens = output[input_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                predicted = self.extract_answer(response)
                ground_truth = self.extract_gsm8k_answer(answer)

                try:
                    is_correct = abs(float(predicted) - float(ground_truth)) < 0.001
                except:
                    is_correct = predicted.strip() == ground_truth.strip()

                if is_correct:
                    correct += 1

                sample_idx = batch_start + j
                new_tokens_count = len(output) - input_length

                # 显示进度
                print(f"\r处理: {sample_idx + 1}/{total} ({(sample_idx + 1)/total*100:.1f}%) - 当前准确率: {correct/(sample_idx + 1)*100:.1f}%", end="")

                results.append({
                    'question': question,
                    'ground_truth_raw': answer,
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'response': response,
                    'correct': is_correct,
                    'input_tokens': input_length,
                    'output_tokens': new_tokens_count,
                    'truncated': new_tokens_count >= max_new_tokens
                })

        inference_time = time.time() - inference_start

        print()  # 换行
        self.log("")

        # 保存结果
        self.config['inference_time_seconds'] = inference_time
        self.config['avg_time_per_sample'] = inference_time / total

        return results

    def save_results(self, results: List[Dict]):
        """保存结果"""
        self.log("=" * 80)
        self.log("保存结果")
        self.log("=" * 80)

        # 统计指标
        total = len(results)
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / total * 100
        truncated_count = sum(1 for r in results if r['truncated'])

        # 创建指标字典
        metrics = {
            'total_samples': total,
            'correct_count': correct_count,
            'incorrect_count': total - correct_count,
            'accuracy': accuracy,
            'truncated_count': truncated_count,
            'inference_time_seconds': self.config['inference_time_seconds'],
            'avg_time_per_sample': self.config['avg_time_per_sample'],
            'memory_stats': self.memory_stats,
        }

        # 添加配置信息
        metrics['config'] = {
            'model_path': self.config['model_path'],
            'prompt_format': self.config['prompt_format'],
            'batch_size': self.config['batch_size'],
            'max_new_tokens': self.config['max_new_tokens'],
            'temperature': self.config['temperature'],
            'top_p': self.config['top_p'],
        }

        # 保存指标
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.log(f"✓ 指标已保存: {metrics_file}")

        # 保存详细结果
        results_file = self.output_dir / "detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.log(f"✓ 详细结果已保存: {results_file}")

        # 保存配置
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        self.log(f"✓ 配置已保存: {config_file}")

        self.log("")

    def print_summary(self, results: List[Dict]):
        """打印总结"""
        total_time = time.time() - self.start_time

        self.log("=" * 80)
        self.log("评测总结")
        self.log("=" * 80)

        # 基本指标
        total = len(results)
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / total * 100

        self.log(f"\n评测结果:")
        self.log(f"  总样本数: {total}")
        self.log(f"  正确数量: {correct_count}")
        self.log(f"  错误数量: {total - correct_count}")
        self.log(f"  准确率: {accuracy:.2f}%")

        # 时间统计
        self.log(f"\n时间统计:")
        self.log(f"  总耗时: {total_time:.2f} 秒")
        self.log(f"  推理耗时: {self.config['inference_time_seconds']:.2f} 秒")
        self.log(f"  平均每样本: {self.config['avg_time_per_sample']:.2f} 秒")

        # 显存统计
        self.log(f"\n显存统计:")
        self.log(f"  峰值分配显存: {self.memory_stats['peak_allocated_gb']:.2f} GB")
        self.log(f"  峰值预留显存: {self.memory_stats['peak_reserved_gb']:.2f} GB")
        if 'gpu_total_memory_gb' in self.config:
            self.log(f"  显存利用率: {self.memory_stats['peak_reserved_gb'] / self.config['gpu_total_memory_gb'] * 100:.1f}%")

        # 配置回顾
        self.log(f"\n配置回顾:")
        self.log(f"  模型: {self.config['model_path']}")
        self.log(f"  Prompt 格式: {self.config['prompt_format']}")
        self.log(f"  Batch Size: {self.config['batch_size']}")
        self.log(f"  Max New Tokens: {self.config['max_new_tokens']}")

        # 截断警告
        truncated_count = sum(1 for r in results if r['truncated'])
        if truncated_count > 0:
            self.log(f"\n⚠️  警告: {truncated_count} 个样本的输出可能被截断")

        self.log("")
        self.log("=" * 80)
        self.log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"结果目录: {self.output_dir}")
        self.log("=" * 80)

    def close(self):
        """关闭日志文件"""
        self.logger.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GSM8K 模型评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 评测 3B-Instruct 模型，使用标准 prompt
  python run_gsm8k_eval.py --model /mnt/workspace/qwen2.5-3B-Instruct --prompt-format standard

  # 评测 0.5B-base 模型，使用 CoT prompt，只测试 100 个样本
  python run_gsm8k_eval.py --model /mnt/workspace/qwen2.5-0.5B-base --prompt-format cot --num-samples 100

  # 自定义 batch size 和 max_new_tokens
  python run_gsm8k_eval.py --model /mnt/workspace/qwen2.5-3B-Instruct --batch-size 16 --max-new-tokens 1024
        """
    )

    # 必需参数
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型路径'
    )

    # Prompt 格式
    parser.add_argument(
        '--prompt-format', '-p',
        type=str,
        choices=['no_system', 'standard', 'cot'],
        default='standard',
        help='提示词格式 (默认: standard)'
    )

    # 数据集
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/mnt/workspace/dataset_eval/gsm8k_data',
        help='数据集路径'
    )

    # 超参数
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='批处理大小 (默认: 32)'
    )

    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=1024,
        help='最大生成 token 数 (默认: 1024)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='温度参数 (默认: 0.0，贪婪解码)'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=1.0,
        help='Top-p 采样参数 (默认: 1.0)'
    )

    # 评测设置
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=-1,
        help='测试样本数，-1 表示全部 (默认: -1)'
    )

    # 输出
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='输出目录 (默认: tests/gsm8k_eval_<timestamp>)'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(args.model).name
        output_dir = f"tests/gsm8k_eval_{model_name}_{args.prompt_format}_{timestamp}"
    else:
        output_dir = args.output_dir

    # 创建配置字典
    config = {
        'model_path': args.model,
        'prompt_format': args.prompt_format,
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'num_samples': args.num_samples,
    }

    # 创建评测器
    evaluator = GSM8KEvaluator(config, output_dir)

    try:
        # 记录配置
        evaluator.log_config()

        # 加载模型
        evaluator.load_model()

        # 加载数据集
        dataset = evaluator.load_dataset()

        # 运行评测
        results = evaluator.run_evaluation(dataset)

        # 保存结果
        evaluator.save_results(results)

        # 打印总结
        evaluator.print_summary(results)

    except Exception as e:
        evaluator.log(f"\n❌ 错误: {e}")
        import traceback
        evaluator.log(traceback.format_exc())
        sys.exit(1)

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
