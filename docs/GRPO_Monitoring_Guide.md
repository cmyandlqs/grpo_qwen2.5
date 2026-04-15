# RL 训练监控指南 - 使用 SwanLab

## 为什么需要监控？

RL 训练比传统训练更复杂，需要关注：
- **Reward 曲线** - 是否在上升？
- **Policy Loss** - 是否稳定？
- **生成质量** - 推理能力是否改善？
- **资源使用** - 显存、GPU 利用率

---

## SwanLab 快速配置

### 1. 安装 SwanLab

```bash
pip install swanlab
# 或
uv pip install swanlab
```

### 2. 注册并获取 API Key

1. 访问 https://swanlab.cn/
2. 注册账号
3. 创建项目并获取 API Key

### 3. 配置环境变量

```bash
# 临时设置
export SWANLAB_API_KEY="your-api-key"

# 或永久设置
echo 'export SWANLAB_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

---

## 训练脚本配置

### 更新训练脚本使用 SwanLab

创建 `scripts/train_grpo_gsm8k_swlab.sh`:

```bash
#!/bin/bash
# GRPO 训练 + SwanLab 监控

# ============================================
# SwanLab 配置
# ============================================
export SWANLAB_API_KEY="your-api-key"
SWANLAB_PROJECT="GRPO-Qwen-GSM8K"
SWANLAB_EXPERIMENT_NAME="qwen2.5-0.5b-grpo-$(date +%Y%m%d_%H%M%S)"
SWANLAB_MODE="online"  # online: 上传云端, local: 仅本地

# ============================================
# 其他配置（与之前相同）
# ============================================
MODEL_PATH="/mnt/workspace/qwen2.5-0.5B-base"
DATA_PATH="/mnt/workspace/dataset_eval/gsm8k_data"
OUTPUT_DIR="/mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k_swlab"

# ... (其他参数与之前相同)

# ============================================
# 执行训练
# ============================================

python -m torch.distributed.run \
    --nproc_per_node 1 \
    $(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + "/cli/rlhf.py")') \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --dataset $DATA_PATH \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --num_generations 8 \
    --reward_funcs accuracy format \
    --temperature 1.0 \
    --top_p 0.9 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --report_to swanlab \  # ← 关键：使用 SwanLab
    --project_name $SWANLAB_PROJECT \  # ← 项目名称
    --experiment_name $SWANLAB_EXPERIMENT_NAME \  # ← 实验名称
    --logging_steps 10 \
    --log_completions true
```

---

## 关键监控指标

### 1. Reward 相关

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| `train/reward` | 训练平均奖励 | ⬆️ 持续上升 |
| `train/reward/accuracy` | 准确率奖励 | ⬆️ 上升 |
| `train/reward/format` | 格式奖励 | ⬆️ 上升 |

### 2. Loss 相关

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| `train/policy_loss` | 策略损失 | ⬇️ 下降后稳定 |
| `train/entropy` | 策略熵 | ➡️ 保持多样性 |

### 3. 学习相关

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| `train/learning_rate` | 学习率 | ➡️ 按schedule变化 |
| `train/grad_norm` | 梯度范数 | ⬇️ 保持稳定 |

### 4. 生成质量

| 指标 | 说明 | 如何观察 |
|------|------|---------|
| `train/completion_length` | 生成长度 | SwanLab 文本日志 |
| `train/sample_outputs` | 样本输出 | SwanLab 文本/图表 |

---

## SwanLab 界面使用

### 1. 实时监控

打开浏览器访问：
```
https://swanlab.cn/@username/project
```

### 2. 关键图表

**Reward 曲线**：
```python
# 在 SwanLab 中查看
# 左侧面板 → Metrics → train/reward
```

应该看到：
- 初期：波动较大
- 中期：逐渐上升
- 后期：趋于稳定

### 3. 对比实验

在 SwanLab 中可以：
- 创建多个实验
- 实时对比不同配置
- 导出对比图表

```bash
# 运行不同配置的实验
bash scripts/train_grpo_gsm8k_swlab.sh  # 实验1
# 修改参数后再次运行
bash scripts/train_grpo_gsm8k_swlab.sh  # 实验2
```

在 SwanLab 界面中：
```
实验对比 → 选择多个实验 → Compare
```

### 4. 查看生成样本

SwanLab 可以记录每个 batch 的生成样本：

```python
# 在训练日志中
# Train → Step 100 → Media → 查看生成的文本
```

---

## 本地训练 + 云端监控

### 方案 1: 完全云端

```bash
export SWANLAB_API_KEY="xxx"
SWANLAB_MODE="online"  # 所有数据上传云端
```

**优点**：
- 随时随地查看
- 团队协作
- 永久保存

**缺点**：
- 需要网络
- 大量日志可能慢

### 方案 2: 混合模式

```bash
# 只上传关键指标
SWANLAB_MODE="online"
LOGGING_INTERVAL=100  # 每100步上传一次
```

### 方案 3: 本地优先

```bash
SWANLAB_MODE="local"  # 只保存本地
# 本地查看：swanlab watch
```

---

## 高级监控技巧

### 1. 自定义指标

创建 `scripts/train_with_custom_metrics.py`:

```python
import swanlab
from swift import RLHFTrainer

# 初始化 SwanLab
swanlab.init(
    project="GRPO-Qwen-GSM8K",
    experiment_name="qwen-0.5b-grpo-test",
    mode="online"
)

# 自定义指标记录
def log_custom_metrics(step, trainer):
    # 计算并记录自定义指标
    metrics = {
        "custom/reward_per_token": trainer.total_reward / trainer.total_tokens,
        "custom/diversity_score": calculate_diversity(trainer.outputs),
        "custom/reasoning_depth": calculate_reasoning_depth(trainer.outputs)
    }
    swanlab.log(metrics, step=step)

# 在训练循环中调用
# trainer.train(callback=log_custom_metrics)
```

### 2. 可视化推理过程

```python
# 记录推理链
swanlab.log({
    "reasoning_chain": swanlab.Text(
        "Step 1: 计算 1+1=2\n"
        "Step 2: 乘以 10\n"
        "Step 3: 结果是 20"
    )
}, step=step)
```

### 3. 对比基座 vs GRPO

```python
# 在 SwanLab 中创建对比图表
import matplotlib.pyplot as plt
import swanlab

# 假设有两个实验的数据
baseline_rewards = [0.1, 0.15, 0.2, 0.25]
grpo_rewards = [0.1, 0.25, 0.4, 0.55]

fig, ax = plt.subplots()
ax.plot(baseline_rewards, label='Baseline')
ax.plot(grpo_rewards, label='GRPO')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward')
ax.legend()

swanlab.log({"comparison/reward_curve": fig}, step=0)
```

---

## 常见监控问题

### Q1: Reward 不上升？

**可能原因**：
1. 学习率太大或太小
2. G 值（num_generations）太小
3. 奖励函数设计问题

**调试方法**：
```bash
# 检查日志
grep "reward" $OUTPUT_DIR/logs/trainer.log

# 在 SwanLab 中查看
# Metrics → train/reward → 查看曲线
```

### Q2: 训练不稳定？

**观察指标**：
- `train/policy_loss` 剧烈波动
- `train/grad_norm` 突然增大

**解决方法**：
```bash
# 降低学习率
--learning_rate 5e-6

# 增加梯度裁剪
--max_grad_norm 1.0

# 检查梯度累积
--gradient_accumulation_steps 4
```

### Q3: 显存不足？

**在 SwanLab 中监控**：
```python
# 记录显存使用
import torch
swanlab.log({
    "memory/allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
    "memory/reserved_gb": torch.cuda.memory_reserved(0) / 1e9
})
```

---

## 最佳实践

### 1. 训练前检查清单

- [ ] SwanLab API key 已配置
- [ ] 项目名称有意义
- [ ] 实验名称包含关键参数
- [ ] logging_steps 设置合理
- [ ] 确认网络连接

### 2. 训练中监控

**每 10 分钟检查**：
```bash
# 本地快速检查
tail -f $OUTPUT_DIR/logs/trainer.log

# SwanLab 网页查看
# https://swanlab.cn/@username/project
```

**关注重点**：
1. Reward 是否在上升？
2. Loss 是否稳定？
3. 显存使用是否正常？

### 3. 训练后分析

```bash
# 1. 导出实验数据
swanlab export $EXPERIMENT_ID --format csv

# 2. 生成报告
python scripts/generate_training_report.py \
    --experiment $EXPERIMENT_ID \
    --output report.html
```

---

## 与其他工具对比

| 工具 | 优点 | 缺点 |
|------|------|------|
| **SwanLab** | 国产、快速、免费、中文 | 生态较新 |
| TensorBoard | 标准、兼容性好 | 界面老旧、无云端 |
| WandB | 功能强大 | 服务器在国外、慢 |

**推荐**：使用 SwanLab 作为主要工具，TensorBoard 作为备用。

---

## 快速开始命令

```bash
# 1. 安装
pip install swanlab

# 2. 配置
export SWANLAB_API_KEY="your-key"

# 3. 运行训练
bash scripts/train_grpo_gsm8k_swlab.sh

# 4. 打开浏览器
# https://swanlab.cn/@username/project
```

---

**相关文档**：
- SwanLab 官方文档: https://docs.swanlab.cn/
- 项目完整指南: `docs/GRPO_Training_Guide.md`
