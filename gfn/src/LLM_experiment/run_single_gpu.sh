#!/bin/bash

# 단일 GPU 학습을 위한 실행 스크립트
# 이 스크립트는 메모리 최적화된 단일 GPU 학습을 실행합니다

echo "🚀 Starting single GPU training..."

# Conda 환경 활성화
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gflownet

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=4  # 사용 가능한 GPU 사용
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 학습 실행
python training.py

echo "✅ Training completed!"