# GRPO 训练 Qwen 数学推理完整指南

> 基于 ms-swift 框架，使用 GSM8K 数据集训练 Qwen 基座模型
>
> 目标：1) 提升数学性能 2) 激活思考能力 3) 学习 LLM RL 范式

---

## 目录

1. [GRPO 算法原理](#1-grpo-算法原理)
2. [环境准备](#2-环境准备)
3. [数据准备](#3-数据准备)
4. [训练配置](#4-训练配置)
5. [训练执行](#5-训练执行)
6. [评估与对比](#6-评估与对比)
7. [LLM RL 范式学习](#7-llm-rl-范式学习)
8. [常见问题](#8-常见问题)

---

## 1. GRPO 算法原理

### 1.1 什么是 GRPO？

**GRPO (Group Relative Policy Optimization)** 是 DeepSeek 团队提出的强化学习算法，用于训练 DeepSeekMath 和 DeepSeek-R1。

**核心创新**：
- 不需要单独训练 Reward Model
- 通过**组内相对比较**计算优势
- 计算效率高，适合大模型训练

### 1.2 GRPO vs 传统 RLHF

| 特性 | 传统 PPO | GRPO |
|------|---------|------|
| Reward Model | 需要单独训练 | ❌ 不需要 |
| 优势计算 | 使用 Value Model | 组内相对比较 |
| 计算成本 | 高（4个模型） | 低（2个模型） |
| 样本效率 | 中等 | 高 |

### 1.3 GRPO 工作流程

```
1. 对每个 prompt 生成 G 个输出 (Group)
                    ↓
2. 计算每个输出的 reward
                    ↓
3. 组内比较：计算相对优势
                    ↓
4. 基于优势更新策略
                    ↓
5. 重复迭代
```

**关键公式**：
```
Advantage = R - mean(Group_Rewards)
```

每个输出相对于组平均的优势。

---

## 2. 环境准备

### 2.1 服务器配置要求

- **GPU**: NVIDIA A10/A100 (24GB+ 显存推荐)
- **CUDA**: 11.8+
- **Python**: 3.10+
- **内存**: 64GB+

### 2.2 安装依赖

```bash
# 安装 ms-swift
pip install ms-swift

# 或使用 uv（推荐）
uv pip install ms-swift

# 验证安装
python -c "import swift; print(swift.__version__)"
```

### 2.3 环境验证

```bash
# 使用项目中的验证脚本
python utils/check_env.py
```

---

## 3. 数据准备

### 3.1 GSM8K 数据集格式

GSM8K (Grade School Math 8K) 包含 8000+ 小学数学应用题。

**标准格式**：
```json
{
  "question": "Janet's ducks lay 16 eggs per day...",
  "answer": "#### 72"
}
```

**answer 字段说明**：
- 前面是推理过程
- `####` 后是最终答案

### 3.2 ms-swift 数据格式

创建 `data/gsm8k_train.json`:
```json
[
  {
    "prompt": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at a market for $2 per duck egg. How much money in dollars does Janet make every day at the market?",
    "response": "To solve this, we first calculate the eggs Janet sells.\nEach day, Janet starts with 16 eggs.\nShe eats 3 eggs, so she has 16 - 3 = 13 eggs left.\nShe uses 4 eggs for muffins, so she has 13 - 4 = 9 eggs left to sell.\nAt $2 per egg, she makes 9 × $2 = $18.\n#### 18"
  }
]
```

---

## 4. 训练配置

### 4.1 基础训练脚本

创建 `scripts/train_grpo_gsm8k.sh`:

```bash
#!/bin/bash
# GRPO 训练 Qwen2.5-0.5B on GSM8K

# 配置
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"
OUTPUT_DIR="/mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k"

# GRPO 参数
NUM_GENERATIONS=8          # G 值：每个 prompt 生成 8 个输出
REWARD_FUNCS="accuracy"    # 奖励函数：准确率
TEMPERATURE=1.0            # 采样温度
TOP_P=0.9                 # Top-p 采样

# 训练参数
BATCH_SIZE=8              # 每设备批大小
GRAD_ACCUM=2              # 梯度累积步数
LEARNING_RATE=1e-5        # 学习率
NUM_EPOCHS=1              # 训练轮数

# LoRA 参数
LORA_RANK=64              # LoRA 秩
LORA_ALPHA=128            # LoRA alpha

# 执行训练
python -m torch.distributed.run \
    --nproc_per_node 1 \
    $(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")') \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --dataset $DATA_PATH \
    --train_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --num_generations $NUM_GENERATIONS \
    --reward_funcs $REWARD_FUNCS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs
```

### 4.2 关键参数说明

#### GRPO 核心参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--num_generations` | G 值：每个 prompt 生成的样本数 | 8-16 |
| `--reward_funcs` | 奖励函数列表 | accuracy,format |
| `--temperature` | 采样温度 | 0.7-1.0 |
| `--top_p` | Top-p 采样 | 0.9-0.95 |

#### 奖励函数选项

- `accuracy`: 答案准确性（数学题必需）
- `format`: 输出格式合规性
- `cosine`: 与参考答案的余弦相似度
- `repetition`: 惩罚重复
- `soft_overlong`: 惩罚过长输出

**推荐组合**：
```bash
--reward_funcs accuracy format
--reward_weights 0.8 0.2  # 80% 准确率，20% 格式
```

---

## 5. 训练执行

### 5.1 完整训练流程

```bash
# 1. 基座评测（训练前）
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/qwen2.5-0.5B-base \
    --prompt-format no_system \
    --output-dir tests/baseline_0.5b_base

# 2. GRPO 训练
bash scripts/train_grpo_gsm8k.sh

# 3. 训练后评测
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k/checkpoint-xxx \
    --prompt-format standard \
    --output-dir tests/after_grpo_0.5b
```

### 5.2 监控训练

**查看日志**：
```bash
tail -f /mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k/logs/trainer.log
```

**关键指标**：
- `reward`: 平均奖励（应该上升）
- `policy_loss`: 策略损失
- `value_loss`: 价值损失（GRPO 中为 0）
- `learning_rate`: 学习率

### 5.3 使用 SwanLab 实时监控（推荐）

**为什么使用 SwanLab？**
- 🇨🇳 国产平台，访问快速
- 📊 实时可视化训练曲线
- 🔄 支持多实验对比
- 💾 云端存储，随时查看

**快速配置**：

1. **安装 SwanLab**：
```bash
pip install swanlab
```

2. **获取 API Key**：
   - 访问 https://swanlab.cn/
   - 注册并获取 API Key

3. **使用 SwanLab 版本训练脚本**：
```bash
# 设置 API Key
export SWANLAB_API_KEY="your-api-key"

# 运行训练
bash scripts/train_grpo_gsm8k_swlab.sh
```

**在 SwanLab 中监控**：

打开 https://swanlab.cn/ 查看：

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| `train/reward` | 训练奖励 | ⬆️ 持续上升 |
| `train/policy_loss` | 策略损失 | ⬇️ 下降后稳定 |
| `train/reward/accuracy` | 准确率奖励 | ⬆️ 上升 |
| `train/grad_norm` | 梯度范数 | ➡️ 保持稳定 |

**详细监控指南**：查看 `docs/GRPO_Monitoring_Guide.md`

---

## 6. 评估与对比

### 6.1 使用项目评测工具

```bash
# 对比不同阶段
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/qwen2.5-0.5B-base \
    --prompt-format cot \
    --num-samples 100 \
    --output-dir tests/baseline_100

python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k \
    --prompt-format cot \
    --num-samples 100 \
    --output-dir tests/grpo_100
```

### 6.2 分析结果

查看 `tests/*/metrics.json` 对比：

```python
import json

baseline = json.load(open('tests/baseline_100/metrics.json'))
grpo = json.load(open('tests/grpo_100/metrics.json'))

print(f"基座准确率: {baseline['accuracy']:.2f}%")
print(f"GRPO 准确率: {grpo['accuracy']:.2f}%")
print(f"提升: {grpo['accuracy'] - baseline['accuracy']:.2f}%")
```

---

## 7. LLM RL 范式学习

### 7.1 为什么 GRPO 适合数学推理？

**传统 SFT 的问题**：
- 只学习"标准答案"
- 不会探索多种解法
- 缺乏推理深度

**GRPO 的优势**：
```
Prompt: "1+1=?"

传统 SFT:
- 只学习 "答案是 2"

GRPO (G=8):
- 探索 8 种不同推理路径
- 学习哪种路径更有效
- 自然涌现"思考"能力
```

### 7.2 RL 训练的核心概念

#### 7.2.1 Policy（策略）

在 LLM 中，Policy = 模型本身

```
输入 → Policy (LLM) → 输出
```

#### 7.2.2 Reward（奖励）

数学题的奖励设计：
```python
def calculate_reward(question, response, ground_truth):
    reward = 0

    # 1. 答案正确性（最重要）
    if extract_answer(response) == ground_truth:
        reward += 1.0

    # 2. 推理步骤质量
    if has_reasoning_steps(response):
        reward += 0.3

    # 3. 格式合规
    if follows_format(response):
        reward += 0.2

    return reward
```

#### 7.2.3 Advantage（优势）

GRPO 中优势的计算：
```python
# 对一个 prompt 生成 8 个输出
outputs = generate_outputs(prompt, G=8)
rewards = [calculate_r(o) for o in outputs]

# 计算组平均
mean_reward = sum(rewards) / len(rewards)

# 每个输出的优势
advantages = [r - mean_reward for r in rewards]

# 优势为正 → 加强这个输出
# 优势为负 → 减弱这个输出
```

### 7.3 训练过程可视化

```
Epoch 0 (随机):
  Prompt: "1+1=?"
  Outputs: ["=2", "是2", "2", "answer is 2", ...]
  Rewards: [1.0, 0.8, 0.9, 0.7, ...]
  ↑
  学习哪种回答方式更好

Epoch 10:
  Prompt: "1+1=?"
  Outputs: ["=2", "答案是 2", "#### 2", ...]
  Rewards: [1.0, 1.0, 1.0, ...]
  ↑
  开始收敛到高质量回答

Epoch 50:
  Prompt: "1+1=?"
  Outputs: ["首先计算 1+1，结果是 2\n#### 2", ...]
  ↑
  自然涌现推理能力！
```

### 7.4 从 GRPO 学习到的 RL 范式

1. **探索与利用**
   - 生成多个输出（探索）
   - 选择最好的（利用）

2. **延迟奖励**
   - 完整推理后才得到奖励
   - 学习整个推理链

3. **策略迭代**
   - 当前策略 → 采样 → 评估 → 更新策略
   - 持续改进

4. **无需显式监督**
   - 不需要"标准推理过程"
   - 只需要"正确答案"

---

## 8. 常见问题

### Q1: 训练多久合适？

**A**: 建议 1-3 个 epoch
- GSM8K: ~9000 样本
- 1 epoch ≈ 2-4 小时（A100 GPU）
- 观察 reward 曲线，收敛即可停止

### Q2: G 值（num_generations）如何选择？

**A**: 权衡计算资源和效果
- G=4: 快速，效果一般
- G=8: 推荐（默认）
- G=16: 效果好，慢 2 倍

### Q3: 如何检查训练是否成功？

**A**: 三种方法
1. **训练日志**: reward 应该上升
2. **生成质量**: 随机抽样看推理过程
3. **评测指标**: 对比训练前后准确率

### Q4: 内存不足怎么办？

**A**: 调整以下参数
```bash
# 减小批大小
--per_device_train_batch_size 4

# 使用梯度checkpointing
--gradient_checkpointing true

# 使用 ZeRO
--deepspeed zero2
```

### Q5: 如何处理多 GPU？

**A**: 调整 nproc_per_node
```bash
# 4 GPU
--nproc_per_node 4 \
--per_device_train_batch_size 8  # 每个GPU的批大小
```

---

## 9. 推荐训练流程

### 阶段 1: 快速验证（1-2 小时）

```bash
# 小规模测试
--num_train_epochs 0.1
--num_generations 4
--num_samples 100
```

### 阶段 2: 正式训练（4-8 小时）

```bash
# 完整训练
--num_train_epochs 1
--num_generations 8
```

### 阶段 3: 评估与迭代（1 小时）

```bash
# 完整评估
python tests/run_gsm8k_eval.py --model $OUTPUT_DIR
```

---

## 10. 进阶技巧

### 10.1 Curriculum Learning

从简单到困难：
```python
# 先训练简单题
easy_data = filter_by_difficulty(gsm8k, level='easy')

# 再训练难题
hard_data = filter_by_difficulty(gsm8k, level='hard')
```

### 10.2 混合奖励函数

```bash
--reward_funcs accuracy format cosine repetition
--reward_weights 0.6 0.2 0.1 0.1
```

### 10.3 多轮训练

```bash
# 第1轮：激活推理能力
--reward_funcs accuracy

# 第2轮：优化推理格式
--reward_funcs accuracy format

# 第3轮：精调
--reward_funcs accuracy format cosine
```

---

## 参考资料

1. **ms-swift 文档**: https://swift.readthedocs.io/
2. **DeepSeekMath 论文**: https://arxiv.org/abs/2406.06661
3. **DeepSeek-R1 论文**: https://github.com/deepseek-ai/DeepSeek-R1
4. **项目评测工具**: `tests/run_gsm8k_eval.py`

---

**最后更新**: 2025-04-15
**作者**: Claude Code + ms-swift 社区
