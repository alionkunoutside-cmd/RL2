# 模型设置指南

## 问题

训练时出现错误：
```
OSError: Can't load the model for 'Qwen/Qwen2.5-1.5B-Instruct'
```

这表示模型还没有下载到本地。

## 解决方案

### 方案 1: 自动下载（推荐）

训练脚本会在首次运行时自动从 HuggingFace 下载模型。只需确保：

1. **网络连接正常**
2. **有足够的磁盘空间**（1.5B 模型约 3GB，3B 模型约 6GB）

直接运行训练脚本，等待下载完成：
```bash
bash examples/countdown_reinforce_single_gpu.sh
```

### 方案 2: 使用国内镜像（如果网络慢）

如果 HuggingFace 下载很慢，使用国内镜像：

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行训练
bash examples/countdown_reinforce_single_gpu.sh
```

### 方案 3: 预先下载模型

使用提供的下载脚本：

```bash
# 进入虚拟环境
source /data/workspace/.rl2_venv/bin/activate

# 下载 1.5B 模型（推荐）
python scripts/download_model.py 1.5b

# 或下载 0.5B 模型（更小）
python scripts/download_model.py 0.5b

# 或下载 3B 模型（更大）
python scripts/download_model.py 3b
```

### 方案 4: 使用本地模型路径

如果你已经有下载好的模型，修改训练脚本中的模型路径：

```bash
# 将这一行：
actor.model_name=Qwen/Qwen2.5-1.5B-Instruct

# 改为本地路径：
actor.model_name=/path/to/your/local/model
```

## 模型大小对比

| 模型 | 参数量 | 模型文件大小 | 训练显存 | 推荐场景 |
|------|--------|------------|----------|----------|
| Qwen2.5-0.5B | 0.5B | ~1GB | ~6-8GB | 快速测试 |
| Qwen2.5-1.5B | 1.5B | ~3GB | ~10-12GB | 推荐 ✅ |
| Qwen2.5-3B | 3B | ~6GB | ~14-15GB | 最佳性能 |

## 常见问题

### Q: 下载很慢怎么办？
A: 使用方案 2 的国内镜像，或者在网络好的时候使用方案 3 预先下载。

### Q: 显示 "Connection timeout" 怎么办？
A:
1. 检查网络连接
2. 设置 HTTP 代理（如果有）：
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```
3. 使用国内镜像站

### Q: 磁盘空间不足怎么办？
A:
1. 使用更小的模型（0.5B）
2. 清理旧的模型缓存：
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--*
   ```
3. 设置自定义缓存目录：
   ```bash
   export HF_HOME=/path/to/large/disk
   ```

## 验证模型下载

下载完成后，可以验证：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"✅ 模型加载成功: {model_name}")
print(f"参数量: {model.num_parameters() / 1e9:.2f}B")
```

## 模型缓存位置

默认情况下，模型会缓存在：
- Linux: `~/.cache/huggingface/hub/`
- 或环境变量 `HF_HOME` 指定的位置

查看缓存：
```bash
ls -lh ~/.cache/huggingface/hub/
```
