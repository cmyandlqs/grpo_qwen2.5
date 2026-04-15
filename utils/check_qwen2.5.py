#!/usr/bin/env python3
"""
Qwen2.5 模型加载与推理脚本
用于 GRPO 训练前的模型验证和测试
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path


def print_separator(title=""):
    """打印分隔线"""
    width = 80
    if title:
        print(f"\n{'=' * (width - len(title) - 2)} {title} {'=' * 2}")
    else:
        print("=" * width)


def load_model_info(model_path):
    """加载并显示模型基本信息"""
    print_separator("模型基本信息")

    # 读取 config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"✓ 模型类型: {config.get('_name_or_path', 'N/A')}")
        print(f"✓ 架构: {config.get('architectures', ['N/A'])[0]}")
        print(f"✓ 隐藏层大小: {config.get('hidden_size', 'N/A')}")
        print(f"✓ 注意力头数: {config.get('num_attention_heads', 'N/A')}")
        print(f"✓ 隐藏层数: {config.get('num_hidden_layers', 'N/A')}")
        print(f"✓ 词汇表大小: {config.get('vocab_size', 'N/A')}")
        print(f"✓ 最大位置嵌入: {config.get('max_position_embeddings', 'N/A')}")
        print(f"✓ 中间层大小: {config.get('intermediate_size', 'N/A')}")
        print(f"✓ RMS 归一化 eps: {config.get('rms_norm_eps', 'N/A')}")
        print(f"✓ 使用 Flash Attention: {config.get('use_flash_attn', 'N/A')}")

    # 列出模型文件
    print_separator("模型文件列表")
    model_files = list(Path(model_path).glob("*"))
    for file in sorted(model_files):
        size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
        print(f"  {file.name:40} {size_mb:>8.2f} MB")


def load_model_and_tokenizer(model_path, device="cuda"):
    """加载模型和 tokenizer"""
    print_separator("加载模型和 Tokenizer")

    # 加载 tokenizer
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    print(f"✓ Tokenizer 已加载")
    print(f"  - 词汇表大小: {len(tokenizer)}")
    print(f"  - Special tokens: {tokenizer.special_tokens_map}")

    # 加载模型配置
    print("\n正在加载模型配置...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"✓ 配置已加载")

    # 加载模型
    print("\n正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省显存
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print(f"✓ 模型已加载到设备: {model.device}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ 参数统计:")
    print(f"  - 总参数量: {total_params / 1e6:.2f} M")
    print(f"  - 可训练参数: {trainable_params / 1e6:.2f} M")

    return model, tokenizer, config


def print_model_structure(model, max_depth=3):
    """打印模型结构"""
    print_separator("模型结构")

    print(f"\n模型类型: {model.__class__.__name__}")
    print(f"\n层级结构 (深度={max_depth}):")
    print_model_tree(model, max_depth=max_depth, current_depth=0)


def print_model_tree(module, max_depth, current_depth, prefix=""):
    """递归打印模型树结构"""
    if current_depth >= max_depth:
        return

    for name, child in list(module.named_children())[:20]:  # 限制显示数量
        # 获取模块类型
        module_type = child.__class__.__name__

        # 获取参数信息
        params = []
        for param_name, param in child.named_parameters():
            params.append(f"{param_name}: {param.shape}")
            if len(params) >= 3:  # 限制显示参数数量
                params.append("...")
                break

        param_str = ", ".join(params) if params else "无参数"

        # 打印当前模块
        connector = "├──" if prefix else ""
        print(f"{prefix}{connector} {name}: {module_type}")
        if param_str != "无参数":
            print(f"{prefix}│   ({param_str})")

        # 递归打印子模块
        new_prefix = prefix + ("│   " if prefix else "")
        print_model_tree(child, max_depth, current_depth + 1, new_prefix)

        # 只显示第一个子模块的详细信息
        if current_depth > 0:
            break


def run_inference(model, tokenizer, prompts, max_new_tokens=256):
    """运行推理测试"""
    print_separator("推理测试")

    model.eval()

    for i, prompt in enumerate(prompts, 1):
        print(f"\n【示例 {i}】")
        print(f"输入: {prompt}")

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输出: {response}")

        # 统计信息
        input_len = inputs["input_ids"].shape[1]
        output_len = outputs.shape[1] - input_len
        print(f"  (输入 {input_len} tokens, 生成 {output_len} tokens)")


def show_memory_usage():
    """显示显存使用情况"""
    print_separator("显存使用情况")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = props.total_memory / (1024 ** 3)

            print(f"GPU {i} ({props.name}):")
            print(f"  总显存:     {total:.2f} GB")
            print(f"  已分配:     {allocated:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"  已预留:     {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    else:
        print("未检测到 CUDA 设备")


def main():
    """主函数"""
    # 配置
    MODEL_PATH = "/mnt/workspace/qwen2.5-0.5B-base"

    # 测试提示词
    TEST_PROMPTS = [
        "请解释什么是机器学习。",
        "计算: 123 + 456 = ?",
        "Python 中列表和元组的区别是什么？"
    ]

    print_separator("Qwen2.5 模型加载与推理工具")

    try:
        # 1. 加载模型信息
        load_model_info(MODEL_PATH)

        # 2. 加载模型和 tokenizer
        model, tokenizer, config = load_model_and_tokenizer(MODEL_PATH)

        # 3. 打印模型结构
        print_model_structure(model, max_depth=2)

        # 4. 显示显存使用
        show_memory_usage()

        # 5. 运行推理测试
        run_inference(model, tokenizer, TEST_PROMPTS, max_new_tokens=128)

        print_separator("完成")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
