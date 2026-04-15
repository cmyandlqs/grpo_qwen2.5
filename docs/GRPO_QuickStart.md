# GRPO 训练快速开始

## 5 分钟快速上手

### 1. 基座评测（建立基准）

```bash
# 评测训练前的基座模型
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/qwen2.5-0.5B-base \
    --prompt-format cot \
    --num-samples 100 \
    --output-dir tests/baseline_0.5b_base_100
```

**预期输出**：
```
准确率: ~10-20%  (base 模型未经指令微调)
```

### 2. GRPO 训练（两种方式）

#### 方式 A: 基础训练（无监控）

```bash
# 运行训练脚本
bash scripts/train_grpo_gsm8k.sh
```

#### 方式 B: 使用 SwanLab 监控（推荐）⭐

```bash
# 1. 安装 SwanLab
pip install swanlab

# 2. 获取 API Key
# 访问 https://swanlab.cn/ 注册

# 3. 设置 API Key
export SWANLAB_API_KEY="your-api-key"

# 4. 运行训练（带实时监控）
bash scripts/train_grpo_gsm8k_swlab.sh

# 5. 打开浏览器查看
# https://swanlab.cn
```

**预期时间**：4-8 小时（A100 GPU，1 epoch）

### 3. 训练后评测

```bash
# 评测训练后的模型
python tests/run_gsm8k_eval.py \
    --model /mnt/workspace/output/grpo_qwen2.5_0.5b_gsm8k_xxx/checkpoint-final \
    --prompt-format cot \
    --num-samples 100 \
    --output-dir tests/after_grpo_0.5b_100
```

**预期提升**：
- 准确率提升: +20-40%
- 推理能力: 显著增强
- 输出格式: 更结构化

---

## 关键参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_generations` | 8 | G 值，每个 prompt 生成的样本数 |
| `reward_funcs` | accuracy format | 奖励函数 |
| `learning_rate` | 1e-5 | 学习率 |
| `num_train_epochs` | 1 | 训练轮数 |
| `temperature` | 1.0 | 采样温度 |

---

## 常见问题速查

**Q: 内存不足？**
→ 减小 `PER_DEVICE_BATCH_SIZE` 到 4

**Q: 训练太慢？**
→ 减小 `NUM_GENERATIONS` 到 4，或减少 `NUM_TRAIN_EPOCHS`

**Q: 效果不好？**
→ 增加 `NUM_GENERATIONS` 到 16，或调整学习率

---

## 完整文档

查看 `docs/GRPO_Training_Guide.md` 获取详细信息。
