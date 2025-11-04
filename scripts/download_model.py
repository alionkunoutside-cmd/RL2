#!/usr/bin/env python
"""
下载 HuggingFace 模型到本地
用法: python scripts/download_model.py [model_name]
"""
import sys
import os
from pathlib import Path

def download_model(model_name, cache_dir=None):
    """下载模型到指定目录"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"正在下载模型: {model_name}")
        print(f"缓存目录: {cache_dir or '默认缓存目录'}")
        print("这可能需要几分钟时间，取决于网络速度...\n")

        # 下载 tokenizer
        print("1/2 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer 下载完成")

        # 下载模型
        print("\n2/2 下载模型权重...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map="cpu"  # 下载时不占用GPU
        )
        print(f"✓ 模型下载完成")

        print(f"\n✅ 成功下载模型: {model_name}")
        print(f"模型大小: {model.num_parameters() / 1e9:.2f}B 参数")

        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 设置 HuggingFace 代理:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 使用本地模型路径")
        return False

def main():
    # 默认模型列表
    models = {
        "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "3b": "Qwen/Qwen2.5-3B-Instruct",
    }

    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
        model_name = models.get(model_arg, sys.argv[1])
    else:
        print("可用的模型:")
        for key, value in models.items():
            print(f"  {key}: {value}")
        print("\n用法: python scripts/download_model.py [0.5b|1.5b|3b|model_name]")
        print("示例: python scripts/download_model.py 1.5b")
        return

    # 设置缓存目录
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    # 下载模型
    success = download_model(model_name, cache_dir)

    if success:
        print(f"\n现在可以运行训练脚本了!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
