# DPO 训练 Qwen 数学推理 - 快速指南

> 使用直接偏好优化在 GSM8K 上训练 Qwen

## 核心概念

**DPO** = Direct Preference Optimization（直接偏好优化）

- 不需要 Reward Model
- 需要 (chosen, rejected) 数据对
- 比 PPO 简单，比 GRPO 需要更多数据准备

---

## 快速开始

### 1. 准备偏好数据

DPO 需要成对的数据：

```python
{
  "prompt": "1+1=?",
  "chosen": "首先计算 1+1=2\n#### 2",      # 好的回答
  "rejected": "1+1=3\n#### 3"             # 差的回答
}
```

### 2. 生成偏好数据（脚本）

```bash
python scripts/prepare_dpo_data.py \
    --base-model /mnt/workspace/qwen2.5-0.5B-base \
    --data-path /mnt/workspace/dataset_eval/gsm8k_data \
    --output-path /mnt/workspace/data/gsm8k_dpo.json
```

### 3. DPO 训练

```bash
bash scripts/train_dpo.sh
```

**预期时间**：3-6 小时

### 4. 评测

```bash
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/output/dpo_qwen2.5_0.5b/checkpoint-final
```

---

## DPO vs GRPO vs PPO

| 特性 | DPO | GRPO | PPO |
|------|-----|------|-----|
| Reward Model | ❌ 不需要 | ❌ 不需要 | ✅ 需要 |
| Value Model | ❌ 不需要 | ❌ 不需要 | ✅ 需要 |
| 数据格式 | chosen/rejected 对 | prompt + reward | prompt + reward |
| 训练稳定性 | ⭐⭐ 好 | ⭐⭐⭐ 很好 | ⭐ 一般 |
| 数据准备 | ⭐⭐⭐ 复杂 | ⭐ 简单 | ⭐⭐ 中等 |

---

## 数据准备方法

### 方法 1: 使用 GRPO 输出

```python
# 1. 用 GRPO 生成多个输出
outputs = generate(prompt, num_samples=8)

# 2. 按奖励排序
sorted_outputs = sort_by_reward(outputs)

# 3. 选最好和最差
chosen = sorted_outputs[0]      # 奖励最高的
rejected = sorted_outputs[-1]   # 奖励最低的
```

### 方法 2: 人工构造

```python
# 对于数学题
{
    "prompt": "解方程 2x + 3 = 7",
    "chosen": "2x + 3 = 7\n2x = 4\nx = 2\n#### 2",
    "rejected": "x = 5\n#### 5"
}
```

---

## 训练配置

编辑 `scripts/train_dpo.sh`：

```bash
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
DATA_PATH="/mnt/workspace/data/gsm8k_dpo.json"
BATCH_SIZE=4              # DPO 批大小通常较小
LEARNING_RATE=5e-6        # DPO 用较低学习率
NUM_EPOCHS=2
BETA=0.1                  # DPO 温度参数
```

---

## 原理简述

### DPO 损失函数

```
L = -log σ(β × (log P(chosen) - log P(rejected)))
```

- 直接优化模型偏好
- 不需要训练 Reward Model
- β 控制偏好强度

### 直观理解

```
Prompt: "1+1=?"

模型输出:
- "答案是 2"    → 给高奖励
- "答案是 3"    → 给低奖励

DPO 直接学习：给 "答案是 2" 更高的概率
```

---

## 什么时候用 DPO？

✅ **适合场景**：
- 有偏好数据（chosen/rejected）
- 不想训练 Reward Model
- 需要对齐人类偏好

❌ **不适合场景**：
- 只有答案没有偏好数据
- 需要实时训练（GRPO 更灵活）
- 数据准备成本高

---

## 推荐训练流程

### 流程 1: SFT → DPO

```bash
# 1. 先 SFT 获得基础能力
bash scripts/train_sft.sh

# 2. 准备 DPO 数据（基于 SFT 模型）
python scripts/prepare_dpo_data.py \
    --model /mnt/workspace/output/sft_qwen2.5_0.5b/checkpoint-final

# 3. DPO 训练
bash scripts/train_dpo.sh
```

### 流程 2: 直接 GRPO

如果只需要答案准确率，GRPO 更简单：
```bash
bash scripts/train_grpo.sh
```

---

**DPO 适合有偏好数据的场景，否则用 GRPO！**
