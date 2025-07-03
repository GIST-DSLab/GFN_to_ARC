#!/usr/bin/env python3
"""
Data preprocessing for ARC Trajectory Transformer
GFlowNet trajectory 데이터를 Trajectory Transformer 형식으로 변환
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm

from utils.data_utils import (
    load_gflownet_trajectories, 
    convert_trajectory_to_sequence,
    create_vocabulary,
    pad_sequence,
    create_attention_mask
)
from configs.arc_config import base, arc_small

class ARCTrajectoryDataset(Dataset):
    """ARC Trajectory Dataset for Transformer"""
    
    def __init__(self, sequences: List[Dict], max_length: int = 64, vocab_size: int = 22):
        self.sequences = sequences
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = create_vocabulary()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        sequence = seq_data['sequence']
        
        # Padding
        padded_sequence = pad_sequence(sequence, self.max_length)
        attention_mask = create_attention_mask(padded_sequence)
        
        # Input과 target 생성 (autoregressive)
        input_ids = padded_sequence[:-1]
        labels = padded_sequence[1:]
        attention_mask = attention_mask[:-1]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long), 
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'trajectory_id': seq_data['trajectory_id'],
            'problem_id': seq_data['problem_id']
        }

def preprocess_data(config: Dict) -> Tuple[ARCTrajectoryDataset, ARCTrajectoryDataset]:
    """
    전체 데이터 전처리 파이프라인
    """
    print("=== ARC Trajectory Data Preprocessing ===")
    
    # 1. GFlowNet trajectory 로드
    print("Loading GFlowNet trajectories...")
    trajectories = load_gflownet_trajectories(
        config['trajectory_data_dir'],
        problem_ids=[86, 139, 178, 149, 154, 240, 379]
    )
    
    if len(trajectories) == 0:
        raise ValueError("No trajectories loaded!")
    
    # 2. Trajectory를 sequence로 변환
    print("Converting trajectories to sequences...")
    sequences = []
    failed_conversions = 0
    
    for traj in tqdm(trajectories, desc="Converting trajectories", unit="traj"):
        seq_data = convert_trajectory_to_sequence(traj)
        if seq_data is not None:
            sequences.append(seq_data)
        else:
            failed_conversions += 1
    
    print(f"Successfully converted {len(sequences)} trajectories")
    print(f"Failed conversions: {failed_conversions}")
    
    # 3. 시퀀스 길이 분석
    seq_lengths = [len(seq['sequence']) for seq in sequences]
    print(f"Sequence length stats:")
    print(f"  Mean: {np.mean(seq_lengths):.1f}")
    print(f"  Std: {np.std(seq_lengths):.1f}")
    print(f"  Min: {np.min(seq_lengths)}")
    print(f"  Max: {np.max(seq_lengths)}")
    print(f"  95th percentile: {np.percentile(seq_lengths, 95):.1f}")
    
    # 4. 너무 긴 시퀀스 필터링
    max_length = config['max_sequence_length']
    sequences = [seq for seq in sequences if len(seq['sequence']) <= max_length]
    print(f"After filtering by max length ({max_length}): {len(sequences)} sequences")
    
    # 5. Train/validation split
    np.random.seed(config['seed'])
    indices = np.random.permutation(len(sequences))
    train_size = int(0.9 * len(sequences))
    
    train_sequences = [sequences[i] for i in indices[:train_size]]
    val_sequences = [sequences[i] for i in indices[train_size:]]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # 6. Dataset 생성
    train_dataset = ARCTrajectoryDataset(
        train_sequences, 
        max_length=max_length,
        vocab_size=config['vocab_size']
    )
    
    val_dataset = ARCTrajectoryDataset(
        val_sequences,
        max_length=max_length, 
        vocab_size=config['vocab_size']
    )
    
    # 7. 데이터 저장
    os.makedirs(config['processed_data_dir'], exist_ok=True)
    
    processed_data = {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'vocab': create_vocabulary(),
        'config': config,
        'stats': {
            'total_trajectories': len(trajectories),
            'successful_conversions': len(sequences),
            'train_size': len(train_sequences),
            'val_size': len(val_sequences),
            'seq_length_stats': {
                'mean': float(np.mean(seq_lengths)),
                'std': float(np.std(seq_lengths)),
                'min': int(np.min(seq_lengths)),
                'max': int(np.max(seq_lengths))
            }
        }
    }
    
    save_path = os.path.join(config['processed_data_dir'], 'arc_trajectory_data.json')
    with open(save_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed data saved to {save_path}")
    
    return train_dataset, val_dataset

def analyze_sample_data(dataset: ARCTrajectoryDataset, num_samples: int = 3):
    """
    샘플 데이터 분석
    """
    print("\n=== Sample Data Analysis ===")
    vocab = create_vocabulary()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Problem ID: {sample['trajectory_id']}")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Label shape: {sample['labels'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Non-padding tokens: {sample['attention_mask'].sum().item()}")
        
        # 토큰 분석
        input_tokens = sample['input_ids'][:20].tolist()  # First 20 tokens
        print(f"  First 20 input tokens: {input_tokens}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess ARC trajectory data")
    parser.add_argument("--config_name", type=str, default="base", 
                       help="Configuration name")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze sample data")
    
    args = parser.parse_args()
    
    # Configuration 로드
    if args.config_name == "base":
        config = base['train']
    elif args.config_name == "small":
        config = arc_small['train']
    else:
        raise ValueError(f"Unknown config: {args.config_name}")
    
    try:
        # 데이터 전처리
        train_dataset, val_dataset = preprocess_data(config)
        
        if args.analyze:
            analyze_sample_data(train_dataset)
            
        print("\n=== Preprocessing Complete ===")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()