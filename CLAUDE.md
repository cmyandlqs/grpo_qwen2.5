# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRPO (Group Relative Policy Optimization) training project for Qwen2.5 language models using the ms-swift RLHF framework. The project trains mathematical reasoning capabilities on the GSM8K dataset.

### Training Stack

- **Framework**: ms-swift (ModelScope Swift) for RLHF training
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Models**: Qwen2.5-0.5B-base, Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct
- **Dataset**: GSM8K (Grade School Math 8K)
- **Acceleration**: vLLM for inference acceleration during training
- **Hardware**: CUDA GPU with bfloat16 precision
### Attention
这个项目是在服务器上训练的，并且模型权重、数据集等都在服务器上。本地仓库只包含工具脚本。本地负责写脚本和修改脚本，修改完毕后需要git add\commit\push,方便服务器更新代码。
### Directory Structure

```
grpo_qwen2.5/
├── utils/                      # Utility scripts
│   ├── check_env.py           # Environment validation
│   ├── quick_test_gsm8k.py    # Fast GSM8K evaluation
│   ├── test_gsm8k.py          # Full batch inference testing
│   ├── infer_qwen.py          # Interactive inference
│   ├── check_qwen2.5.py       # Model validation
│   └── print_tree.py          # Directory visualization
├── dataset_eval/              # Evaluation datasets (server only)
├── models/                    # Model checkpoints (server only)
└── output/                    # Training outputs and results
```

**Note**: Models, datasets, and training outputs reside on the server at `/mnt/workspace/`. Local repo contains only utility scripts.

## Common Commands

### Environment Validation

```bash
# Check GRPO training environment (dependencies, CUDA, models, datasets)
python utils/check_env.py

# Check both virtual and global environments
python utils/check_env.py --check-global

# Show uv installation guide
python utils/check_env.py --uv-guide
```

### Model Validation

```bash
# Validate model loading and structure
python utils/check_qwen2.5.py

# Interactive inference testing
python utils/infer_qwen.py
```

### GSM8K Evaluation

```bash
# Quick evaluation with detailed debug output (first 5 samples)
python utils/quick_test_gsm8k.py

# Full batch inference testing
python utils/test_gsm8k.py
```

### GRPO Training (Server)

```bash
# Standard GRPO training with ms-swift
python -m torch.distributed.run --nproc_per_node 1 \
  $(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")') \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-0.5B \
  --dataset AI-ModelScope/gsm8k \
  --train_type lora \
  --lora_rank 64 \
  --torch_dtype bfloat16 \
  --use_vllm true \
  --num_generations 8 \
  --reward_funcs accuracy \
  --output_dir ./output/grpo_qwen2.5_0.5b
```

## Architecture

### Answer Extraction System

The project uses a sophisticated multi-pattern answer extraction system for GSM8K evaluation:

**Priority Order**:
1. LaTeX `\boxed{}` format (highest priority - Qwen commonly uses this)
2. Plain `boxed{}` format
3. Text descriptions ("answer:", "结果:", etc.)
4. Symbolic patterns (`= number`, `- number`)
5. Fallback: Last number in response

**File**: `utils/quick_test_gsm8k.py` - `extract_answer()` function

### GSM8K Dataset Format

The GSM8K dataset uses a specific answer format requiring extraction:

```
[Reasoning steps...]
#### 72
```

**Extraction**: The `extract_gsm8k_answer()` function extracts the number after `####` for ground truth comparison.

### Batch Processing

Both evaluation scripts support automatic batch size calculation based on available GPU memory:

- `quick_test_gsm8k.py`: Fixed batch size (default: 32)
- `test_gsm8k.py`: Auto-calculation or manual setting (24GB GPU → 16)

### Model Loading

All scripts use consistent model loading patterns:
- `torch_dtype=torch.bfloat16` for memory efficiency
- `device_map="auto"` for automatic device placement
- `trust_remote_code=True` for Qwen compatibility
- `low_cpu_mem_usage=True` for CPU memory optimization

## Key Dependencies

**Core Training**:
- `ms-swift` - RLHF training framework with GRPO support
- `torch` - PyTorch with CUDA support
- `transformers` - Hugging Face transformers
- `modelscope` - ModelScope model hub integration

**Evaluation**:
- `datasets` - Hugging Face datasets (for GSM8K loading)
- `vllm` - High-throughput inference acceleration

**Monitoring** (optional):
- `nvitop` - GPU monitoring
- `swanlab` - Experiment tracking

## Environment Requirements

- Python 3.10+
- CUDA-capable GPU (24GB+ recommended for training)
- 50GB+ disk space
- Virtual environment (venv, conda, or uv)

## GSM8K Answer Format Notes

When working with GSM8K evaluation:
- Ground truth answers use `#### number` format in the dataset
- Qwen models tend to output `\boxed{number}` LaTeX format
- The extraction system handles multiple fallback patterns
- For numerical comparison, tolerance of 0.001 is used

## Server vs Local Development

- **Server** (`/mnt/workspace/`): Models, datasets, training runs
- **Local**: Utility scripts, code development, testing

When running evaluation scripts locally, update paths in `main()` functions:
- `MODEL_PATH`: Point to local model directory if available
- `DATA_PATH`: Point to local GSM8K dataset if available
- `output_path`: Update for local output directory
