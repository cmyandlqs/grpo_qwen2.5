# 训练方法选择指南

> 在 GSM8K 上训练 Qwen 基座 - 4 种方法对比

## 快速选择

```
需要基础能力？
├─ 是 → SFT（最简单，2-4小时）
│
└─ 否 → 需要深度思考？
    ├─ 是 → GRPO（推荐，4-8小时）
    │
    └─ 否 → 有偏好数据？
        ├─ 是 → DPO（3-6小时）
        │
        └─ 否 → PPO（不推荐，用GRPO）
```

---

## 方法对比

| 方法 | 复杂度 | 时间 | 效果 | 推荐度 |
|------|--------|------|------|--------|
| **SFT** | ⭐ | 2-4h | ⭐⭐ | ⭐⭐⭐ |
| **GRPO** | ⭐⭐ | 4-8h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DPO** | ⭐⭐⭐ | 3-6h | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **PPO** | ⭐⭐⭐⭐ | 8-12h | ⭐⭐⭐ | ⭐ |

---

## 详细说明

### SFT - 监督微调

**适合**：
- ✅ 快速获得基础能力
- ✅ 数据有标准答案
- ✅ 计算资源有限

**不适合**：
- ❌ 需要超越训练数据
- ❌ 想要模型自己探索

**脚本**：`scripts/train_sft.sh`

---

### GRPO - 组相对策略优化 ⭐ 推荐

**适合**：
- ✅ 数学推理（专门优化）
- ✅ 只要答案准确率
- ✅ 激活深度思考

**优势**：
- 不需要 Reward Model
- 训练稳定
- DeepSeek-R1 同款算法

**脚本**：`scripts/train_grpo.sh`

---

### DPO - 直接偏好优化

**适合**：
- ✅ 有 chosen/rejected 数据
- ✅ 不想训练 Reward Model

**劣势**：
- 需要准备偏好数据
- 数据准备成本高

**脚本**：`scripts/train_dpo.sh`

---

### PPO - 近端策略优化

**适合**：
- ⚠️  已有 Reward Model
- ⚠️  需要复杂奖励

**劣势**：
- 需要 3-4 个模型
- 训练时间长
- 不如 GRPO 稳定

**脚本**：`scripts/train_ppo.sh`

---

## 推荐训练流程

### 流程 1: 快速上手

```
SFT (2-4h) → 评测
```

### 流程 2: 最佳效果 ⭐

```
SFT (2-4h) → GRPO (4-8h) → 评测
```

### 流程 3: 有偏好数据

```
SFT → 准备 DPO 数据 → DPO → 评测
```

---

## 何时用哪种方法？

### 场景 1: 从零开始

```
使用 SFT 快速获得基础能力
bash scripts/train_sft.sh
```

### 场景 2: 追求最佳数学性能

```
SFT + GRPO 两阶段训练
bash scripts/train_sft.sh
bash scripts/train_grpo.sh
```

### 场景 3: 有高质量偏好数据

```
使用 DPO 利用偏好信号
python scripts/prepare_dpo_data.py
bash scripts/train_dpo.sh
```

### 场景 4: 学习 RLHF

```
可以用 PPO，但建议先用 GRPO 理解概念
bash scripts/train_ppo.sh  # 不推荐
```

---

## 文件清单

### 训练脚本
- `scripts/train_sft.sh` - SFT 训练
- `scripts/train_grpo.sh` - GRPO 训练
- `scripts/train_dpo.sh` - DPO 训练
- `scripts/train_ppo.sh` - PPO 训练

### 文档
- `docs/SFT_QuickStart.md` - SFT 详细说明
- `docs/GRPO_QuickStart.md` - GRPO 详细说明
- `docs/DPO_QuickStart.md` - DPO 详细说明
- `docs/PPO_QuickStart.md` - PPO 详细说明

---

## 一句话总结

> **SFT 入门，GRPO 出效果，DPO 需要数据，PPO 过时了。**

**从 SFT 开始，用 GRPO 追求最佳效果！**
