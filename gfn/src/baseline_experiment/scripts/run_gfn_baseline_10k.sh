#!/bin/bash

# Kill any existing tmux sessions
tmux kill-session -t gfn_baseline_10k 2>/dev/null || true

# Create log directory
mkdir -p /data/gfn_baseline_10k/logs

# Launch training with GPU 7
echo "Starting GFN baseline training with 35k samples on GPU 7..."
echo "  - Model: meta-llama/Llama-3.1-8B-Instruct"
echo "  - Dataset: /data/gfn_baseline_10k/data/massive_7problem_baseline.json"
echo "  - Output: /data/gfn_baseline_10k/models"

# Set environment variables to help with download
export HF_HUB_ENABLE_HF_TRANSFER=0  # Disable hf_transfer for better error handling
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=${HF_TOKEN}  # Set this environment variable before running

# Using tmux to run training
tmux new-session -d -s gfn_baseline_10k \
    "source /data/miniforge3/bin/activate gflow-llm && \
     export CUDA_VISIBLE_DEVICES=7 && \
     cd /home/ubuntu/GFN_to_ARC/gfn/src/baseline_experiment/scripts && \
     python train_gfn_baseline_10k.py \
        --data_path /data/gfn_baseline_10k/data/massive_7problem_baseline.json \
        --output_dir /data/gfn_baseline_10k/models \
        --num_epochs 10 \
        --batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 5e-5 \
        2>&1 | tee /data/gfn_baseline_10k/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Training started in tmux session 'gfn_baseline_10k'"
echo "Use 'tmux attach -t gfn_baseline_10k' to monitor"
echo "Logs are saved to /data/gfn_baseline_10k/logs/"