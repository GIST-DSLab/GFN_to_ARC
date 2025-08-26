#!/bin/bash

# GFN Training and Augmentation with is_correct filtering
# GPU 6, 7에서 실행

echo "🚀 Starting GFN Training and Augmentation with is_correct filtering"
echo "Time started: $(date)"
echo "========================================================================================"

cd /home/ubuntu/GFN_to_ARC/gfn/src

# Activate conda environment (using miniforge3)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate base

# Set PYTHONPATH
export PYTHONPATH=/home/ubuntu/GFN_to_ARC/gfn/src:$PYTHONPATH

# Run the training with is_correct filtering
python main_parallel_enhanced_correct_only.py \
    --problems 86 139 149 154 178 240 379 \
    --gpu_ids 6 7 \
    --num_trajectories 100000 \
    --accuracy_threshold 0.75 \
    --max_training_steps 50000 \
    --evaluation_interval 1000 \
    --evaluation_samples 100 \
    --output_dir /data/gflownet-llm-additional \
    --checkpoint_interval 1000 \
    --batch_size 32 \
    --num_epochs 1 \
    --env_mode "entire" \
    --num_actions 5 \
    --ep_len 10 \
    --use_offpolicy

echo "========================================================================================"
echo "Training completed at: $(date)"
echo "Results saved to: /data/gflownet-llm-additional"
echo "Check experiment_summary_correct_only.json for detailed results"