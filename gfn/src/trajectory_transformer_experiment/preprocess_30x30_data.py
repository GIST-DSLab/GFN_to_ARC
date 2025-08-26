#!/usr/bin/env python3
"""
Preprocess data for 30x30 Trajectory Transformer
"""

import json
import os
import sys
from tqdm import tqdm
from typing import List, Dict
import argparse

sys.path.append('.')
from utils.data_utils import (
    load_gflownet_trajectories, 
    convert_rearc_to_30x30_sequence,
    pad_grid_to_30x30,
    flatten_30x30_grid
)

def preprocess_rearc_data(rearc_data_dir: str, problem_ids: List[int], output_file: str):
    """
    Preprocess Re-ARC data to 30x30 format for training
    
    Args:
        rearc_data_dir: Re-ARC data directory
        problem_ids: List of problem IDs to process
        output_file: Output JSON file path
    """
    print(f"Preprocessing Re-ARC data for 30x30 Trajectory Transformer...")
    
    # Mapping from problem ID to hex filename
    id_to_hex = {
        86: "25ff71a9",
        128: "5582e5ca", 
        139: "6150a2bd",
        149: "67a3c6ac",
        154: "68b16354",
        178: "74dd1130",
        240: "9dfd6313",
        379: "ed36ccf7"
    }
    
    training_sequences = []
    
    for problem_id in tqdm(problem_ids, desc="Processing problems"):
        if problem_id not in id_to_hex:
            print(f"Warning: No hex mapping for problem ID {problem_id}")
            continue
            
        hex_filename = id_to_hex[problem_id]
        problem_file = os.path.join(rearc_data_dir, f"{hex_filename}.json")
        
        if not os.path.exists(problem_file):
            print(f"Warning: Problem file not found: {problem_file}")
            continue
        
        # Load problem data
        with open(problem_file, 'r') as f:
            examples = json.load(f)
        
        print(f"Processing problem {problem_id} ({hex_filename}) with {len(examples)} examples")
        
        # Convert each example to 30x30 sequence
        problem_sequences = []
        for i, example in enumerate(tqdm(examples, desc=f"Problem {problem_id} examples", leave=False)):
            sequence_data = convert_rearc_to_30x30_sequence(
                input_grid=example['input'],
                output_grid=example['output'],
                problem_id=problem_id
            )
            
            if sequence_data:
                sequence_data['example_id'] = i
                problem_sequences.append(sequence_data)
        
        training_sequences.extend(problem_sequences)
        print(f"Converted {len(problem_sequences)} examples from problem {problem_id}")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        'sequences': training_sequences,
        'total_sequences': len(training_sequences),
        'grid_size': '30x30',
        'observation_dim': 900,  # 30x30 flattened
        'problems_processed': problem_ids
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(training_sequences)} training sequences to {output_file}")
    return training_sequences

def create_30x30_config():
    """Create configuration for 30x30 model"""
    config = {
        # Data paths
        'rearc_data_dir': '../LLM_experiment/data/re-arc/re_arc_extracted/re_arc/tasks',
        'processed_data_dir': './processed_data_30x30',
        'model_save_dir': './models_30x30',
        'results_dir': './results_30x30',
        
        # Model architecture - updated for 30x30 grids
        'n_layer': 8,           # Increased layers for larger input
        'n_head': 12,           # Increased attention heads
        'n_embd': 256,          # Increased embedding size
        'batch_size': 32,       # Reduced batch size due to larger input
        'learning_rate': 0.0001,
        'n_epochs': 10,
        'warmup_steps': 10000,
        'lr_decay': True,
        'weight_decay': 0.01,
        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,
        
        # Sequence settings for 30x30
        'max_sequence_length': 1024,  # 900 (obs) + additional tokens
        'observation_dim': 900,       # 30x30 flattened
        'action_dim': 1,
        'reward_dim': 1,
        'value_dim': 1,
        'vocab_size': 26,
        
        # Loss weights
        'action_weight': 2.0,
        'reward_weight': 1.0,
        'value_weight': 1.5,
        'observation_weight': 1.0,
        
        # Training settings
        'device': 'cuda',
        'seed': 42,
        'log_interval': 100,
        'eval_interval': 1000,
        'save_interval': 5000,
        
        # Generation settings
        'temperature': 1.0,
        'top_k': None,
        'top_p': 0.9,
        'max_new_tokens': 64,
        'num_return_sequences': 1,
        
        # Evaluation settings
        'eval_problems': [86, 139, 149, 154, 178, 240, 379],
        'max_test_samples': None,  # Use all examples
        'problem_max_actions': {
            '86': 15, '139': 20, '149': 18, '154': 16,
            '178': 12, '240': 22, '379': 20
        }
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for 30x30 Trajectory Transformer")
    parser.add_argument("--problems", type=int, nargs="+", 
                       default=[86, 139, 149, 154, 178, 240, 379],
                       help="Problem IDs to process")
    parser.add_argument("--output_dir", type=str, default="./processed_data_30x30",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess Re-ARC data
    rearc_data_dir = '../LLM_experiment/data/re-arc/re_arc_extracted/re_arc/tasks'
    output_file = os.path.join(args.output_dir, 'rearc_30x30_training_data.json')
    
    training_sequences = preprocess_rearc_data(
        rearc_data_dir=rearc_data_dir,
        problem_ids=args.problems,
        output_file=output_file
    )
    
    # Create and save 30x30 config
    config = create_30x30_config()
    config_file = os.path.join(args.output_dir, 'config_30x30.yaml')
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f, indent=2)
    
    print(f"Created config file: {config_file}")
    
    # Summary
    print(f"\n=== Preprocessing Summary ===")
    print(f"Total training sequences: {len(training_sequences)}")
    print(f"Problems processed: {args.problems}")
    print(f"Grid size: 30x30 (900 tokens per observation)")
    print(f"Output directory: {args.output_dir}")
    print(f"Training data file: {output_file}")
    print(f"Config file: {config_file}")

if __name__ == "__main__":
    main()