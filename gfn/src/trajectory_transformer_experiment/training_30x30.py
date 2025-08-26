#!/usr/bin/env python3
"""
Training script for 30x30 ARC Trajectory Transformer
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml
import argparse
import logging
from datetime import datetime

from models.arc_transformer_30x30 import create_30x30_model
from utils.data_utils import pad_sequence, create_attention_mask

class ARC30x30Dataset(Dataset):
    """Dataset for 30x30 ARC sequences"""
    
    def __init__(self, data_file, max_length=1024):
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.sequences = data['sequences']
        self.max_length = max_length
        
        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Grid size: {data.get('grid_size', 'unknown')}")
        print(f"Observation dim: {data.get('observation_dim', 'unknown')}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        sequence = sequence_data['sequence']
        
        # Pad sequence
        padded_sequence = pad_sequence(sequence, self.max_length, pad_token=10)
        attention_mask = create_attention_mask(padded_sequence, pad_token=10)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_sequence[:-1], dtype=torch.long)
        targets = torch.tensor(padded_sequence[1:], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'targets': targets,
            'attention_mask': attention_mask,
            'problem_id': sequence_data['problem_id']
        }

def setup_logging(log_dir):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_30x30_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, scheduler, device, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, targets=targets)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        total_tokens += attention_mask.sum().item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': optimizer.param_groups[0]['lr']})
        
        # Log every N batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            logger.info(f"Batch {batch_idx}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train 30x30 ARC Trajectory Transformer")
    parser.add_argument("--config", type=str, default="./processed_data_30x30/config_30x30.yaml",
                       help="Config file path")
    parser.add_argument("--data_file", type=str, default="./processed_data_30x30/rearc_30x30_training_data.json",
                       help="Training data file")
    parser.add_argument("--gpu", type=int, default=5,
                       help="GPU device to use")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    logger = setup_logging(config['model_save_dir'])
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Config: {config}")
    
    # Create dataset and dataloader
    dataset = ARC30x30Dataset(args.data_file, max_length=config['max_sequence_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_30x30_model(config)
    model.to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    total_steps = len(dataloader) * config['n_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    logger.info(f"Starting training for {config['n_epochs']} epochs")
    logger.info(f"Total batches per epoch: {len(dataloader)}")
    
    best_loss = float('inf')
    
    for epoch in range(config['n_epochs']):
        logger.info(f"\n=== Epoch {epoch+1}/{config['n_epochs']} ===")
        
        # Train
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, logger)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            
            os.makedirs(config['model_save_dir'], exist_ok=True)
            checkpoint_path = os.path.join(config['model_save_dir'], 'checkpoint_30x30_best.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config['model_save_dir'], f'checkpoint_30x30_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()