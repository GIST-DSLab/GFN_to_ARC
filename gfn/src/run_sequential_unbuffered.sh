#!/bin/bash

# Run sequential training with unbuffered output
export PYTHONUNBUFFERED=1

echo "🚀 Starting Sequential GFN Training (Unbuffered)"
echo "Time: $(date)"
echo "="

cd /home/ubuntu/GFN_to_ARC/gfn/src

# Run with explicit unbuffering
python -u main_sequential_improved.py \
    --gpu_id 6 \
    --problems 86 139 149 154 178 240 379 \
    --num_trajectories 100000 \
    --accuracy_threshold 0.75 \
    --max_training_steps 50000 \
    --evaluation_interval 500 \
    --evaluation_samples 100 \
    --min_exploration_rate 0.1 \
    --lr_decay_factor 0.95 \
    --patience 5 \
    --use_is_correct