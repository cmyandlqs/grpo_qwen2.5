# GRPO 训练 Qwen 数学推理 - 快速指南

> 使用 ms-swift + GSM8K 训练 Qwen 基座模型

## 核心概念

**GRPO** = Group Relative Policy Optimization（DeepSeek-R1 使用的算法）

### 为什么 GRPO 适合数学推理？

1. **不需要 Reward Model** - 直接用答案准确率作为奖励
2. **组内比较** - 每个 prompt 生成 8 个输出，学习哪个更好
3. **激活思考** - 自然涌现推理能力

---

## 快速开始

### 1. 基座评测（建立基准）

```bash
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/qwen2.5-0.5B-base \
    --prompt-format cot \
    --num-samples 100
```

**预期**：准确率 ~10-20%

### 2. GRPO 训练

```bash
# 安装 SwanLab（推荐，用于监控）
pip install swanlab
# 访问 https://swanlab.cn/ 获取 API Key

# 设置 API Key
export SWANLAB_API_KEY="your-key"

# 运行训练
bash scripts/train_grpo.sh
```

**预期时间**：4-8 小时

### 3. 训练后评测

```bash
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/output/grpo_qwen2.5_0.5b/checkpoint-final \
    --num-samples 100
```

**预期提升**：准确率 +20-40%

---

## 训练脚本配置

编辑 `scripts/train_grpo.sh` 中的关键参数：

```bash
# 模型
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"

# GRPO 核心参数
NUM_GENERATIONS=8         # G 值：每个 prompt 生成几个输出
REWARD_FUNCS="accuracy"   # 奖励函数

# 训练参数
BATCH_SIZE=8              # 批大小
LEARNING_RATE=1e-5        # 学习率
NUM_EPOCHS=1              # 训练轮数

# SwanLab（可选）
SWANLAB_KEY="your-key"    # 留空则不使用
```

---

## 监控训练

### 使用 SwanLab（推荐）

访问 https://swanlab.cn/ 查看：

**关键指标**：
- `train/reward` - 训练奖励（应上升）
- `train/policy_loss` - 策略损失（应下降）

### 本地日志

```bash
tail -f /mnt/workspace/output/grpo_qwen2.5_0.5b/logs/trainer.log
```

---

## 常见问题

**Q: 内存不足？**
→ 减小 `BATCH_SIZE` 到 4

**Q: 训练太慢？**
→ 减小 `NUM_GENERATIONS` 到 4

**Q: 效果不好？**
→ 增加 `NUM_GENERATIONS` 到 16

---

## 原理简述

### GRPO 工作流程

```
1. Prompt: "1+1=?"
                    ↓
2. 生成 8 个输出（Group）
   - "= 2"
   - "答案是 2"
   - "首先计算...结果是 2"
   ...
                    ↓
3. 计算每个输出的 Reward
                    ↓
4. 组内比较：相对优势 = Reward - 平均
                    ↓
5. 更新策略（加强好的，减弱坏的）
```

### 核心公式

```
Advantage = R - mean(Group_Rewards)
```

不需要单独的 Reward Model，这是 GRPO 的最大优势。

---

**完整文档**：查看 `docs/GRPO_Training_Guide.md`
