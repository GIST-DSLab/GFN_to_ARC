#!/bin/bash

echo "🚀 Starting GPU 6 training (Problems 178, 240)..."

# Conda 환경 활성화
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gflownet

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 학습 실행
python run_experiment.py --config configs/config_gpu6.yaml

echo "✅ GPU 6 training completed!"