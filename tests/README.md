# GSM8K 模型评测

本目录包含用于系统化评测模型在 GSM8K 数学数据集上表现的工具。

## 使用方法

### 基本用法

```bash
# 评测 3B-Instruct 模型，使用标准 prompt
python run_gsm8k_eval.py --model /mnt/workspace/qwen2.5-3B-Instruct --prompt-format standard

# 评测 0.5B-base 模型，使用 CoT prompt
python run_gsm8k_eval.py --model /mnt/workspace/qwen2.5-0.5B-base --prompt-format cot

# 快速测试（只测试 100 个样本）
python run_gsm8k_eval.py --model /mnt/workspace/qwen2.5-3B-Instruct --num-samples 100
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `-m` | 模型路径（必需） | - |
| `--prompt-format` | `-p` | 提示词格式: `no_system`, `standard`, `cot` | `standard` |
| `--data-path` | `-d` | 数据集路径 | `/mnt/workspace/dataset_eval/gsm8k_data` |
| `--batch-size` | `-b` | 批处理大小 | `32` |
| `--max-new-tokens` | - | 最大生成 token 数 | `1024` |
| `--temperature` | - | 温度参数 | `0.0` |
| `--top-p` | - | Top-p 采样参数 | `1.0` |
| `--num-samples` | `-n` | 测试样本数，`-1` 表示全部 | `-1` |
| `--output-dir` | `-o` | 输出目录 | 自动生成 |

## Prompt 格式说明

### no_system
不使用 system prompt，只有 user-assistant 格式：
```
<|im_start|>user
{问题}
<|im_end|>
<|im_start|>assistant
```

### standard
使用标准 system prompt：
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{问题}
<|im_end|>
<|im_start|>assistant
```

### cot
使用 Chain of Thought prompt，要求模型逐步思考：
```
<|im_start|>system
You are a helpful assistant. Please think step by step and show your work. Put your final answer within \boxed{}.
<|im_end|>
<|im_start|>user
{问题}
<|im_end|>
<|im_start|>assistant
```

## 输出结构

每次评测会创建一个新的输出目录：

```
tests/
└── gsm8k_eval_qwen2.5-3B-Instruct_standard_20240415_123456/
    ├── config.json              # 配置参数
    ├── metrics.json             # 评测指标
    ├── detailed_results.json    # 详细结果（每个样本）
    └── log.txt                  # 运行日志
```

### metrics.json 格式

```json
{
  "total_samples": 1319,
  "correct_count": 761,
  "incorrect_count": 558,
  "accuracy": 57.70,
  "truncated_count": 0,
  "inference_time_seconds": 123.45,
  "avg_time_per_sample": 0.094,
  "memory_stats": {
    "peak_allocated_gb": 2.34,
    "peak_reserved_gb": 5.67
  },
  "config": {
    "model_path": "/mnt/workspace/qwen2.5-3B-Instruct",
    "prompt_format": "standard",
    "batch_size": 32,
    "max_new_tokens": 1024
  }
}
```

## 示例

### 对比不同模型

```bash
# 评测 0.5B-base 模型
python run_gsm8k_eval.py -m /mnt/workspace/qwen2.5-0.5B-base -p no_system
python run_gsm8k_eval.py -m /mnt/workspace/qwen2.5-0.5B-base -p standard
python run_gsm8k_eval.py -m /mnt/workspace/qwen2.5-0.5B-base -p cot

# 评测 3B-Instruct 模型
python run_gsm8k_eval.py -m /mnt/workspace/qwen2.5-3B-Instruct -p standard
python run_gsm8k_eval.py -m /mnt/workspace/qwen2.5-3B-Instruct -p cot
```

### 快速验证

修改脚本前先在小数据集上验证：
```bash
python run_gsm8k_eval.py -m /mnt/workspace/qwen2.5-3B-Instruct -n 10
```
