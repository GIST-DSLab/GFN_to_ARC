#!/bin/bash

# GPU 5와 6을 사용하여 병렬 학습 실행
# 기본 설정: GPU 5, 6 사용, 2개 프로세스

echo "Starting multi-GPU training on GPUs 5 and 6..."

python main_parallel.py \
    --gpu_ids 5 6 \
    --num_processes 2 \
    --num_trajectories 10000 \
    --num_epochs 1 \
    --output_dir "trajectories_multi_gpu"

echo "Training completed!"