# ARC Trajectory Transformer Experiment

This experiment trains a Trajectory Transformer model on GFlowNet trajectory data to predict action sequences for ARC puzzles, similar to the LLM experiment but using the trajectory transformer architecture.

## Overview

The trajectory transformer treats reinforcement learning as sequence modeling, converting trajectories into sequences of observations, actions, and rewards. It uses a GPT-style causal transformer to predict the next token in the sequence.

### Objectives
- Train Trajectory Transformer on GFlowNet trajectory data
- Generate action sequences for ReARC test problems  
- Compare performance with LLM experiment using same evaluation methodology

### Data Transformation
- **Input**: GFlowNet trajectories (states, actions, rewards)
- **Output**: Transformer sequences (observation, action, reward tokens)
- **Grid Representation**: 3x3 grid → 9-dimensional flattened vector
- **Actions**: 0-4 integers (left_rotate, right_rotate, horizontal_flip, vertical_flip, submit)

### Model Architecture
- GPT-style transformer with causal attention
- Sequence format: [obs_tokens(9), action_token(1), reward_token(1), ...]
- Vocabulary: colors (0-9), actions (11-15), rewards (16-19), special tokens (20-21)
- Autoregressive generation for action sequence prediction

## File Structure

```
trajectory_transformer_experiment/
├── README.md                    # This documentation
├── configs/
│   └── arc_config.py           # Model and training configurations
├── models/
│   └── arc_transformer.py     # GPT-style transformer model
├── utils/
│   └── data_utils.py          # Data conversion utilities
├── data_preprocessing.py       # GFlowNet → Transformer format conversion
├── training.py                 # Model training script
├── inference.py                # Model inference and evaluation
├── run_experiment.py          # Complete experiment pipeline
├── processed_data/            # Preprocessed trajectory data
├── models/                    # Trained model checkpoints
├── results/                   # Training logs and statistics
└── experiments/               # Complete experiment outputs
```

## Quick Start

### 1. Run Complete Experiment
```bash
# Full pipeline with small model (recommended for testing)
python run_experiment.py --config small --wandb

# Full pipeline with base model
python run_experiment.py --config base --wandb
```

### 2. Run Individual Steps

#### Data Preprocessing
```bash
python data_preprocessing.py --config_name base --analyze
```

#### Training Only
```bash
python training.py --config base --wandb
```

#### Inference Only
```bash
python inference.py --config base --model_path ./models/checkpoint_best.pt
```

## Configuration

### Base Configuration
- **Layers**: 6 transformer layers
- **Embedding**: 128 dimensions
- **Heads**: 8 attention heads
- **Batch Size**: 32
- **Sequence Length**: 64 tokens
- **Epochs**: 50

### Small Configuration (for testing)
- **Layers**: 3 transformer layers
- **Embedding**: 64 dimensions  
- **Heads**: 4 attention heads
- **Batch Size**: 16
- **Sequence Length**: 32 tokens
- **Epochs**: 10

## Data Pipeline

1. **Load GFlowNet Trajectories**: From `../LLM_experiment/data/trajectories_output`
2. **Filter Successful Trajectories**: Only trajectories with final reward > 0
3. **Convert to Sequence Format**: Flatten grids, tokenize actions/rewards
4. **Create Train/Val Split**: 90% train, 10% validation
5. **Tokenization**: Convert to transformer-compatible sequences

## Usage Examples

### Run with Weights & Biases Logging
```bash
python run_experiment.py --config base --wandb
```

### Skip Preprocessing (if already done)
```bash
python run_experiment.py --config base --skip_preprocessing
```

### Only Training
```bash
python run_experiment.py --config base --training_only --wandb
```

### Inference on Specific Problems
```bash
python inference.py --config base --model_path ./models/checkpoint_best.pt --problems 86 139 178
```

## Expected Results

The trajectory transformer should learn to:
- Encode spatial grid patterns in token embeddings
- Predict appropriate action sequences for ARC puzzles
- Achieve competitive performance compared to LLM approach

## Comparison with LLM Experiment

| Aspect | LLM Experiment | Trajectory Transformer |
|--------|----------------|------------------------|
| Architecture | DialoGPT/LLaMA | Custom GPT-style |
| Input Format | Text sequences | Tokenized trajectories |
| Training | Language modeling | Sequence prediction |
| Inference | Text generation | Token generation |
| Evaluation | Action parsing | Direct token decoding |
| Vocabulary | Natural language | Domain-specific tokens |

## Troubleshooting

### Common Issues
1. **Missing trajectory data**: Ensure LLM experiment data exists
2. **CUDA out of memory**: Use `--config small` or reduce batch size
3. **No successful trajectories**: Check GFlowNet data quality
4. **Import errors**: Ensure all dependencies installed

### Performance Tips
- Start with `small` configuration for faster iteration
- Use `--wandb` for experiment tracking and visualization
- Monitor validation loss to prevent overfitting
- Use gradient checkpointing for memory efficiency