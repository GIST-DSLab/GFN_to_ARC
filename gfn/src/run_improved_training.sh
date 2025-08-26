#!/bin/bash

# Improved GFN Training and Augmentation
# GPU 6, 7에서 실행

echo "🚀 Starting Improved GFN Training and Augmentation"
echo "Time started: $(date)"
echo "========================================================================================"

cd /home/ubuntu/GFN_to_ARC/gfn/src

# Activate conda environment
source /data/miniforge3/etc/profile.d/conda.sh
conda activate base

# Set PYTHONPATH
export PYTHONPATH=/home/ubuntu/GFN_to_ARC/gfn/src:$PYTHONPATH

# Run the improved training
python main_parallel_improved.py \
    --problems 86 139 149 154 178 240 379 \
    --gpu_ids 6 7 \
    --num_trajectories 100000 \
    --accuracy_threshold 0.75 \
    --max_training_steps 50000 \
    --evaluation_interval 500 \
    --evaluation_samples 100 \
    --output_dir /data/gflownet-llm-additional \
    --checkpoint_interval 1000 \
    --batch_size 32 \
    --num_epochs 1 \
    --env_mode "entire" \
    --num_actions 5 \
    --ep_len 10 \
    --min_exploration_rate 0.1 \
    --lr_decay_factor 0.95 \
    --patience 5 \
    --use_is_correct

echo "========================================================================================"
echo "Training completed at: $(date)"
echo "Results saved to: /data/gflownet-llm-additional"
echo "Check experiment_summary_improved.json for detailed results"