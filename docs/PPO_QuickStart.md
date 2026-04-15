# PPO 训练 Qwen 数学推理 - 快速指南

> 使用 PPO 在 GSM8K 上训练 Qwen（不推荐，GRPO 更好）

## ⚠️ 重要提示

**PPO 可行但不推荐！**

GRPO 是 DeepSeek 团队专门为数学推理优化的算法，比 PPO：
- 更简单（不需要 Reward Model）
- 更快（少训练 2 个模型）
- 更稳定（组内相对比较）

**除非你有特殊需求，否则用 GRPO！**

---

## 核心概念

**PPO** = Proximal Policy Optimization（近端策略优化）

- 需要 Reward Model（给输出打分）
- 需要 Value Model（估计价值）
- OpenAI 的经典 RLHF 算法

---

## 为什么 PPO 更复杂？

### PPO 训练流程

```
1. 训练 Reward Model  ← 需要标注数据
                    ↓
2. 训练 Value Model   ← 需要额外训练
                    ↓
3. PPO 训练 Policy    ← 同时优化 3 个模型
```

### GRPO 训练流程

```
1. 直接 GRPO 训练     ← 只需要答案正确性
```

---

## 如果一定要用 PPO

### 1. 训练 Reward Model

```bash
python scripts/train_reward_model.sh
```

**需要数据**：
```python
{
    "prompt": "1+1=?",
    "response": "答案是 2",
    "reward": 1.0  # 人工标注或规则生成
}
```

### 2. PPO 训练

```bash
bash scripts/train_ppo.sh
```

**预期时间**：8-12 小时（包括 Reward Model 训练）

---

## PPO vs GRPO 对比

| 特性 | PPO | GRPO |
|------|-----|------|
| 训练 Reward Model | ✅ 需要 | ❌ 不需要 |
| 训练 Value Model | ✅ 需要 | ❌ 不需要 |
| 总模型数 | 3-4 个 | 1-2 个 |
| 训练时间 | 8-12h | 4-8h |
| 显存占用 | 高 | 低 |
| 稳定性 | 一般 | 很好 |
| 数学推理 | 可行 | ✅ 专门优化 |

---

## PPO 适用场景

✅ **PPO 适合**：
- 有复杂的多维度奖励
- 需要 Reward Model 泛化
- 不止一个优化目标
- 已有训练好的 Reward Model

❌ **PPO 不适合**：
- 只需要答案准确率
- 计算资源有限
- 快速迭代
- 数学推理（GRPO 专门优化）

---

## 推荐流程

### 对于数学推理，推荐：

```
SFT → GRPO
```

### 如果需要复杂奖励：

```
SFT → 训练 Reward Model → PPO
```

但即使这样，也可以考虑：
```
SFT → 自定义 Reward GRPO
```

GRPO 支持自定义奖励函数，可以模拟复杂 Reward。

---

## 结论

**本项目推荐顺序**：

1. **SFT** - 快速获得基础能力（2-4h）
2. **GRPO** - 激活深度思考（4-8h）

**使用 PPO 的情况**：
- 你已经有训练好的 Reward Model
- 需要 GRPO 无法表达的复杂奖励
- 学习 PPO 算法本身

**否则用 GRPO，它就是为了数学推理优化的！**

---

**详细对比**：查看 `docs/GRPO_QuickStart.md`
