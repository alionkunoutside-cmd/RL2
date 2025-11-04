#!/bin/bash
# 单GPU训练脚本 - 保持 Qwen2.5-3B 模型 (更激进的内存优化)

# 禁用SGLang自定义kernel以兼容Tesla T4 (Compute Capability 7.5)
export SGLANG_DISABLE_CUSTOM_KERNEL=1

torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.ppo \
    train_data.path=train@Chenmien/Countdown \
    train_data.prompts_per_rollout=16 \
    train_data.responses_per_prompt=2 \
    test_data.path=test@Chenmien/Countdown \
    actor.model_name=Qwen/Qwen2.5-3B-Instruct \
    actor.max_length_per_device=2048 \
    rollout.server_args.mem_fraction_static=0.4 \
    rollout.train_sampling_params.max_new_tokens=512 \
    "rollout.train_sampling_params.stop=['</answer>']" \
    rollout.env_path=envs/countdown.py \
    trainer.project=Countdown \
    trainer.experiment_name=qwen2.5-3b_reinforce_single_gpu \
    trainer.test_freq=8 \
    trainer.save_freq=32
