# Tesla T4 单GPU训练指南

## 硬件配置
- GPU: Tesla T4 (15GB 显存)
- CUDA: 12.2
- Driver: 535.161.07

## 模型推荐

### 1. 🟢 Qwen2.5-1.5B-Instruct (推荐)
**最佳平衡选项**

- **模型大小**: ~3GB
- **训练显存**: ~10-12GB
- **脚本**: `examples/countdown_reinforce_single_gpu.sh`
- **优点**:
  - 性能较好,适合大多数任务
  - 显存占用适中,训练稳定
  - 可以使用较大的 batch size
- **配置**:
  - prompts_per_rollout: 32
  - max_length_per_device: 4096
  - max_new_tokens: 512

```bash
bash examples/countdown_reinforce_single_gpu.sh
```

### 2. 🟡 Qwen2.5-3B-Instruct (原始模型)
**最大性能,需要激进优化**

- **模型大小**: ~6GB
- **训练显存**: ~14-15GB (接近上限)
- **脚本**: `examples/countdown_reinforce_3b_single_gpu.sh`
- **优点**: 性能最好
- **缺点**:
  - 显存占用高,可能OOM
  - 需要减小 batch size
  - 训练速度较慢
- **配置** (激进优化):
  - prompts_per_rollout: 16 (从128降低)
  - responses_per_prompt: 2 (从4降低)
  - max_length_per_device: 2048 (从8192降低)
  - max_new_tokens: 512 (从1024降低)
  - mem_fraction_static: 0.4 (降低推理服务器内存占用)

```bash
bash examples/countdown_reinforce_3b_single_gpu.sh
```

**⚠️ 注意**: 如果遇到 OOM (Out of Memory) 错误,建议使用 1.5B 模型。

### 3. 🟢 Qwen2.5-0.5B-Instruct (最小模型)
**最快训练,显存占用最少**

- **模型大小**: ~1GB
- **训练显存**: ~6-8GB
- **脚本**: `examples/countdown_reinforce_0.5b_single_gpu.sh`
- **优点**:
  - 显存占用极少
  - 训练速度快
  - 可以使用更大的 batch size
- **缺点**: 性能相对较弱
- **配置**:
  - prompts_per_rollout: 64
  - max_length_per_device: 4096
  - max_new_tokens: 1024

```bash
bash examples/countdown_reinforce_0.5b_single_gpu.sh
```

## 关键配置变化说明

### 原始配置 vs 单GPU优化

| 配置项 | 原始值 (4 GPU) | 1.5B 单GPU | 3B 单GPU | 0.5B 单GPU |
|--------|---------------|-----------|----------|-----------|
| nproc_per_node | 4 | 1 | 1 | 1 |
| 模型 | 3B | 1.5B | 3B | 0.5B |
| prompts_per_rollout | 128 | 32 | 16 | 64 |
| responses_per_prompt | 4 | 4 | 2 | 4 |
| max_length_per_device | 8192 | 4096 | 2048 | 4096 |
| max_new_tokens | 1024 | 512 | 512 | 1024 |
| mem_fraction_static | 0.6 | 0.5 | 0.4 | 0.5 |

### 优化技术说明

1. **Gradient Checkpointing** (`enable_gradient_checkpointing=true`)
   - 减少激活值显存占用
   - 代价是增加约20%的训练时间

2. **Optimizer Offload** (`offload_optimizer=true`)
   - 将优化器状态从GPU转移到CPU
   - 减少约50%的显存占用
   - 轻微增加训练时间

3. **Model Offload** (`offload_model=true`)
   - 在不训练时将模型转移到CPU
   - 进一步减少显存占用

4. **Memory Fraction** (`mem_fraction_static`)
   - 控制推理服务器(SGLang)的显存占用
   - 降低此值为训练预留更多显存

## 其他可选的小模型

如果 Qwen 系列不满足需求,可以尝试:

1. **microsoft/phi-2** (2.7B)
   ```bash
   actor.model_name=microsoft/phi-2
   ```

2. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B)
   ```bash
   actor.model_name=TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ```

3. **google/gemma-2b-it** (2B)
   ```bash
   actor.model_name=google/gemma-2b-it
   ```

## 监控显存使用

训练时可以在另一个终端监控GPU使用情况:

```bash
watch -n 1 nvidia-smi
```

## 故障排除

### 遇到 OOM 错误

如果训练时遇到显存不足 (CUDA out of memory):

1. **立即方案**: 切换到更小的模型 (1.5B 或 0.5B)
2. **减小 batch size**: 降低 `prompts_per_rollout`
3. **减小序列长度**: 降低 `max_length_per_device`
4. **降低推理服务器内存**: 减小 `mem_fraction_static` 到 0.3

### 训练速度慢

1. 适当增大 `prompts_per_rollout` (如果显存允许)
2. 确保使用 bfloat16 (默认已启用)
3. 考虑关闭一些 offload 选项 (如果显存充足)

## 预期训练时间

在 Tesla T4 上的大致训练时间 (每个 epoch):

- Qwen2.5-0.5B: ~1-2小时
- Qwen2.5-1.5B: ~3-4小时
- Qwen2.5-3B: ~6-8小时

实际时间取决于数据集大小和具体配置。
