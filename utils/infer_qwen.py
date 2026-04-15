#!/usr/bin/env python3
"""
Qwen2.5-3B-Instruct 简单推理脚本
输入问题，打印模型输出
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


def load_model(model_path: str):
    """加载模型和 tokenizer"""
    print(f"正在加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("✓ 模型加载完成\n")
    return model, tokenizer


def format_prompt(query: str) -> str:
    """格式化为 ChatML 格式"""
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


def generate_response(model, tokenizer, query: str, max_new_tokens: int = 512):
    """生成回复"""
    prompt = format_prompt(query)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # 贪婪解码，更确定性
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只解码新生成的部分
    input_length = inputs['input_ids'].shape[1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


def main():
    """主函数"""
    MODEL_PATH = "/mnt/workspace/qwen2.5-3B-Instruct"

    # 加载模型
    model, tokenizer = load_model(MODEL_PATH)

    print("=" * 60)
    print("Qwen2.5-3B-Instruct 推理终端")
    print("输入问题按回车，输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            # 获取用户输入
            query = input("\n🔸 你的问题: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            # 生成回复
            print("\n🤖 模型回复:")
            print("─" * 60)
            response = generate_response(model, tokenizer, query)
            print(response)
            print("─" * 60)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
