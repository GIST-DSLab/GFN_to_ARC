#!/usr/bin/env python3
"""
ARC Trajectory Transformer Training Script
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse
from tqdm import tqdm
import wandb
import logging

from models.arc_transformer import create_model
from data_preprocessing import ARCTrajectoryDataset, preprocess_data
import yaml

class ARCTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup logging
        self.setup_logging()
        
        # Create directories
        os.makedirs(config['model_save_dir'], exist_ok=True)
        os.makedirs(config['results_dir'], exist_ok=True)
        
        # Initialize model
        self.model = create_model(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        self.logger.info(f"Using device: {self.device}")
    
    def setup_logging(self):
        """Setup logging"""
        log_file = os.path.join(self.config['results_dir'], 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_wandb(self):
        """Initialize Weights & Biases"""
        wandb.init(
            project="arc-trajectory-transformer",
            config=self.config,
            name=f"arc_transformer_{self.config['n_layer']}L_{self.config['n_embd']}d"
        )
        wandb.watch(self.model, log_freq=100)
    
    def load_data(self):
        """Load training and validation data"""
        self.logger.info("Loading data...")
        
        # Check if processed data exists
        processed_file = os.path.join(self.config['processed_data_dir'], 'arc_trajectory_data.json')
        
        if os.path.exists(processed_file):
            self.logger.info("Loading preprocessed data...")
            with open(processed_file, 'r') as f:
                data = json.load(f)
            
            train_dataset = ARCTrajectoryDataset(
                data['train_sequences'],
                max_length=self.config['max_sequence_length'],
                vocab_size=self.config['vocab_size']
            )
            val_dataset = ARCTrajectoryDataset(
                data['val_sequences'],
                max_length=self.config['max_sequence_length'], 
                vocab_size=self.config['vocab_size']
            )
        else:
            self.logger.info("Preprocessing data...")
            train_dataset, val_dataset = preprocess_data(self.config)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Validation batches: {len(self.val_loader)}")
        
        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * self.config['n_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=self.config['learning_rate'] * 0.1
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['n_epochs']}", unit="batch")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if hasattr(self, 'use_wandb') and self.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + batch_idx
                })
            
            # Periodic logging
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", unit="batch"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Log to wandb
        if hasattr(self, 'use_wandb') and self.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'epoch': epoch
            })
        
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['model_save_dir'], 'checkpoint_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['model_save_dir'], 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.config['model_save_dir'], f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def train(self, use_wandb=False):
        """Main training loop"""
        self.use_wandb = use_wandb
        
        if use_wandb:
            self.setup_wandb()
        
        self.load_data()
        
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {self.config}")
        
        for epoch in range(self.config['n_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['n_epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                f"{' (Best!)' if is_best else ''}"
            )
        
        # Save final training stats
        stats = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        stats_path = os.path.join(self.config['results_dir'], 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train ARC Trajectory Transformer")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--device", type=str, default=None, help="Device to use (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    # Initialize trainer
    trainer = ARCTrainer(config)
    
    try:
        # Start training
        trainer.train(use_wandb=args.wandb)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()