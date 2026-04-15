# GSM8K 评测脚本问题分析与改进

## 原始脚本 (`quick_test_gsm8k.py`) 的问题

### 1. ⚠️ 提示词格式缺少 system prompt（关键问题）

**问题**: Qwen2.5-Instruct 模型在训练时使用了 system prompt，但当前脚本只使用了 user-assistant 格式。

**当前格式**:
```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

**推荐格式**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

**影响**: 这可能导致模型性能下降 5-15%

### 2. ⚠️ max_new_tokens 可能不够

**当前值**: 512 tokens
**问题**: 对于复杂的数学推理问题，512 tokens 可能不够，导致输出被截断
**建议**: 增加到 1024 或更高

### 3. ⚠️ 缺少 Chain of Thought 提示

**问题**: 没有明确要求模型逐步思考并展示工作过程
**建议**: 在 system prompt 中添加 "Please think step by step and show your work"

### 4. ⚠️ 缺少答案格式引导

**问题**: 没有要求模型使用特定格式输出答案
**建议**: 要求模型使用 `\boxed{答案}` 格式，方便提取

### 5. ✅ 答案提取逻辑基本正确

当前脚本的答案提取逻辑有多个 fallback 模式，覆盖了大多数情况。
但可以增加更多模式以提高鲁棒性。

## 改进建议

### 短期改进（立即可用）

在 `format_prompt` 函数中添加 system prompt:

```python
def format_prompt(question: str) -> str:
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
```

### 中期改进（更高准确率）

使用 Chain of Thought 提示:

```python
def format_prompt(question: str) -> str:
    return f"""<|im_start|>system
You are a helpful assistant. Please think step by step and show your work.
Put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
```

## 验证脚本

我创建了三个脚本来帮助验证和改进：

### 1. `verify_gsm8k_pipeline.py` - 完整验证脚本

**功能**:
- 检查数据集加载是否正确
- 验证提示词格式
- 测试模型生成并检查是否截断
- 测试答案提取逻辑
- 对比不同提示词格式

**运行**:
```bash
python utils/verify_gsm8k_pipeline.py
```

### 2. `quick_test_gsm8k_fixed.py` - 改进版评测脚本

**改进**:
- ✅ 添加 system prompt 选项
- ✅ 增加 max_new_tokens 到 1024
- ✅ 添加 CoT 提示选项
- ✅ 检测输出截断并警告
- ✅ 在结果中记录是否截断

**运行**:
```bash
python utils/quick_test_gsm8k_fixed.py
```

**可配置参数**:
```python
USE_SYSTEM_PROMPT = True    # 强烈推荐
USE_COT = False             # 可选，可能提高准确率但更慢
MAX_NEW_TOKENS = 1024       # 避免截断
```

### 3. `compare_prompt_formats.py` - 快速对比脚本

**功能**: 只测试前 10 个样本，快速对比不同 prompt 格式的效果

**运行**:
```bash
python utils/compare_prompt_formats.py
```

**输出**:
- 每种格式的准确率对比
- 详细输出对比（前3个样本）
- JSON 结果文件

## 建议的测试流程

1. **快速验证**: 先运行 `compare_prompt_formats.py` (10 样本，快速)
2. **详细检查**: 运行 `verify_gsm8k_pipeline.py` (3-5 样本，详细输出)
3. **完整评测**: 使用 `quick_test_gsm8k_fixed.py` (全部样本)

## 预期改进效果

根据类似任务的经验，添加 system prompt 预计能提高准确率 5-15%。
使用 CoT 提示可能再提高 5-10%，但会增加推理时间和 token 消耗。

## 其他检查项

- [ ] 确认数据集加载后，`question` 和 `answer` 字段正确
- [ ] 检查输入长度是否接近 2048 限制
- [ ] 确认答案提取对各种输出格式都能正确处理
- [ ] 验证 tokenizer 的 special tokens 配置正确
