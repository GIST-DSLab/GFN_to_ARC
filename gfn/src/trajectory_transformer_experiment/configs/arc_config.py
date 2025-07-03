"""
ARC Trajectory Transformer Configuration
"""

import os

# Base configuration
base = {
    'train': {
        # Model architecture
        'n_layer': 6,           # Transformer layers
        'n_head': 8,            # Attention heads  
        'n_embd': 128,          # Embedding dimension
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'n_epochs': 50,
        'warmup_steps': 1000,
        'lr_decay': True,
        'weight_decay': 0.01,
        
        # Regularization
        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1, 
        'attn_pdrop': 0.1,
        
        # Sequence parameters
        'max_sequence_length': 64,  # Max trajectory length
        'step': 1,                  # Subsampling step
        
        # Data parameters
        'observation_dim': 9,       # 3x3 grid flattened
        'action_dim': 1,           # Single action index
        'reward_dim': 1,           # Single reward value
        'vocab_size': 22,          # 0-9 colors + 10 pad + 11-15 actions + 16-19 rewards + 20-21 special
        
        # Loss weights
        'action_weight': 5.0,      # Emphasize action prediction
        'reward_weight': 1.0,
        'observation_weight': 1.0,
        
        # Paths
        'trajectory_data_dir': "../LLM_experiment/data/trajectories_output",
        'processed_data_dir': "./processed_data", 
        'model_save_dir': "./models",
        'results_dir': "./results",
        'device': 'cuda',
        'seed': 42,
        
        # Logging
        'log_interval': 100,
        'eval_interval': 1000,
        'save_interval': 5000,
    },
    
    'inference': {
        # Generation parameters
        'temperature': 1.0,
        'top_k': None,
        'top_p': 0.9,
        'max_new_tokens': 32,      # Max action sequence length
        'num_return_sequences': 1,
        
        # Planning parameters  
        'horizon': 20,             # Planning horizon
        'beam_width': 64,          # Beam search width
        'n_samples': 1,            # Number of trajectory samples
        
        # Evaluation parameters
        'eval_problems': [86, 139, 178, 149, 154, 240, 379],
        'max_test_samples': 50,
        
        # Paths
        'rearc_data_dir': "../LLM_experiment/data/re-arc",
        'model_load_path': "./models/arc_transformer_best.pt",
        'results_dir': "./results",
    }
}

# Small model for testing
arc_small = {
    'train': {
        **base['train'],
        'n_layer': 3,
        'n_head': 4,
        'n_embd': 64,
        'batch_size': 16,
        'n_epochs': 10,
        'max_sequence_length': 128,  # Increased to accommodate longer sequences
    }
}

# Problem-specific configurations
problem_configs = {
    86: {"max_actions": 15},
    139: {"max_actions": 20}, 
    178: {"max_actions": 12},
    149: {"max_actions": 18},
    154: {"max_actions": 16},
    240: {"max_actions": 22},
    379: {"max_actions": 14},
}