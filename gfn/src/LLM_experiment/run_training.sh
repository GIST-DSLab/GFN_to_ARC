#!/bin/bash
# 학습 실행 스크립트

# Flash attention 문제 해결
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# GPU 설정
export CUDA_VISIBLE_DEVICES=6

# 실행
echo "Starting training with GPU $CUDA_VISIBLE_DEVICES..."
python training.py --gpus 1 "$@"