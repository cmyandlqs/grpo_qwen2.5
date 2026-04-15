# SFT 训练 Qwen 数学推理 - 快速指南

> 使用监督学习在 GSM8K 上训练 Qwen 基座

## 核心概念

**SFT** = Supervised Fine-Tuning（监督微调）

- 最简单、最基础的训练方法
- 模型学习"标准答案"
- 适合快速获得基础能力

---

## 快速开始

### 1. 基座评测

```bash
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/qwen2.5-0.5B-base \
    --num-samples 100
```

### 2. SFT 训练

```bash
bash scripts/train_sft.sh
```

**预期时间**：2-4 小时

### 3. 训练后评测

```bash
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/output/sft_qwen2.5_0.5b/checkpoint-final \
    --num-samples 100
```

**预期提升**：准确率 +15-30%

---

## SFT vs GRPO

| 特性 | SFT | GRPO |
|------|-----|------|
| 训练数据 | 需要标准答案 | 只需要正确答案 |
| 推理能力 | 学习模仿 | 自然涌现 |
| 训练时间 | 快（2-4h） | 慢（4-8h） |
| 适用场景 | 快速获得基础能力 | 激活深度思考 |

---

## 数据格式

GSM8K 天然适合 SFT：

```json
{
  "prompt": "1+1=?",
  "response": "首先计算 1+1=2\n#### 2"
}
```

模型学习完整的推理过程。

---

## 训练配置

编辑 `scripts/train_sft.sh`：

```bash
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
BATCH_SIZE=8          # 批大小
LEARNING_RATE=2e-5    # SFT 用较高学习率
NUM_EPOCHS=3          # SFT 可以多训练几轮
```

---

## 原理简述

### SFT 做什么？

```
输入: "1+1=?"
目标: "首先计算 1+1=2\n#### 2"
      ↓
模型学习这个映射
      ↓
输出: "首先计算 1+1=2\n#### 2"
```

### 损失函数

```
Loss = -log P(正确答案 | 输入)
```

模型只需要最大化正确答案的概率。

---

## 什么时候用 SFT？

✅ **适合场景**：
- 数据有高质量的标准答案
- 需要快速获得基础能力
- 数据量充足（GSM8K 有 8000+ 样本）

❌ **不适合场景**：
- 想要模型自己探索推理方法
- 数据只有答案没有过程
- 需要超越训练数据的性能

---

## 推荐训练流程

### 阶段 1: SFT（2-4h）
获得基础推理能力

### 阶段 2: GRPO（4-8h）
在 SFT 基础上激活深度思考

```bash
# 先 SFT
bash scripts/train_sft.sh

# 再 GRPO（基于 SFT 的 checkpoint）
bash scripts/train_grpo.sh
```

---

**快速简洁，首选 SFT 入门！**
