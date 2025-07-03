#!/bin/bash

echo "🚀 Starting GPU 5 training (Problems 149, 154)..."

# Conda 환경 활성화
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gflownet

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 학습 실행
python run_experiment.py --config configs/config_gpu5.yaml

echo "✅ GPU 5 training completed!"