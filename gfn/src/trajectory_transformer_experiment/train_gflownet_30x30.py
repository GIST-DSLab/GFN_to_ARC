#!/usr/bin/env python3
"""
GFlowNet 방식 30x30 ARC Trajectory Transformer 학습 스크립트
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import yaml
import logging
from typing import Dict, List

from models.arc_transformer_gflownet_30x30 import create_gflownet_30x30_model
from data_preprocessing import ARCTrajectoryDataset

def setup_logging(log_file: str):
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class GFlowNet30x30TrajectoryDataset(ARCTrajectoryDataset):
    """GFlowNet 30x30 데이터셋"""
    
    def __init__(self, sequences: List[Dict], max_length: int = 920, vocab_size: int = 26):
        self.sequences = sequences
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        sequence = seq_data['sequence']
        
        # 패딩
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [10] * (self.max_length - len(sequence))  # 패딩 토큰 10
        
        # 어텐션 마스크 생성
        attention_mask = [1 if token != 10 else 0 for token in sequence]
        
        # Input과 target 생성 (autoregressive)
        input_ids = sequence[:-1]
        labels = sequence[1:]
        attention_mask = attention_mask[:-1]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }

def load_processed_data(data_file: str) -> tuple:
    """전처리된 데이터 로드"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    train_sequences = data['train']
    val_sequences = data['validation']
    config_data = data.get('config', {})
    
    return train_sequences, val_sequences, config_data

def train_epoch(model, dataloader, optimizer, scheduler, device, logger, epoch, log_interval=50):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping으로 안정성 확보
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 로깅
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
            progress_bar.set_postfix(loss=f"{loss.item():.6f}", avg_loss=f"{avg_loss:.6f}")
    
    return total_loss / num_batches

def validate(model, dataloader, device, logger):
    """검증"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, config):
    """체크포인트 저장"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, save_path)

def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet 30x30 ARC Trajectory Transformer")
    parser.add_argument("--config", type=str, default="configs/config_gflownet_30x30.yaml",
                       help="Configuration file path")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Preprocessed data file path")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 디렉토리 생성
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 로깅 설정
    log_file = os.path.join(config['results_dir'], 'training.log')
    logger = setup_logging(log_file)
    
    logger.info("=== GFlowNet 30x30 ARC Trajectory Transformer Training ===")
    logger.info(f"Configuration: {args.config}")
    
    # 디바이스 설정
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 데이터 로드
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = os.path.join(config['processed_data_dir'], 'arc_trajectory_data_30x30.json')
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run preprocess_gflownet_30x30.py first")
        return
    
    logger.info(f"Loading data from: {data_file}")
    train_sequences, val_sequences, data_config = load_processed_data(data_file)
    
    logger.info(f"Training sequences: {len(train_sequences)}")
    logger.info(f"Validation sequences: {len(val_sequences)}")
    
    # 데이터셋 생성
    train_dataset = GFlowNet30x30TrajectoryDataset(
        train_sequences, 
        max_length=config['max_sequence_length'],
        vocab_size=config['vocab_size']
    )
    val_dataset = GFlowNet30x30TrajectoryDataset(
        val_sequences,
        max_length=config['max_sequence_length'], 
        vocab_size=config['vocab_size']
    )
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # multiprocessing 오류 방지
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 모델 생성
    logger.info("Creating model...")
    model = create_gflownet_30x30_model(config)
    model.to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 옵티마이저와 스케줄러
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 학습률 스케줄러
    total_steps = len(train_dataloader) * config['n_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 체크포인트에서 재시작
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # 학습 루프
    logger.info("Starting training...")
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(start_epoch, config['n_epochs']):
            logger.info(f"\n=== Epoch {epoch+1}/{config['n_epochs']} ===")
            
            # 학습
            train_loss = train_epoch(
                model, train_dataloader, optimizer, scheduler, 
                device, logger, epoch+1, config['log_interval']
            )
            train_losses.append(train_loss)
            
            # 검증
            val_loss = validate(model, val_dataloader, device, logger)
            val_losses.append(val_loss)
            
            # 체크포인트 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(config['model_save_dir'], 'checkpoint_best.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_checkpoint_path, config)
                logger.info(f"New best model saved: {best_checkpoint_path}")
            
            # 정기 체크포인트
            if (epoch + 1) % 1 == 0:  # 매 에포크마다 저장
                checkpoint_path = os.path.join(config['model_save_dir'], f'checkpoint_epoch_{epoch}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path, config)
            
            # 최신 체크포인트
            latest_checkpoint_path = os.path.join(config['model_save_dir'], 'checkpoint_latest.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, latest_checkpoint_path, config)
            
            logger.info(f"Epoch {epoch+1} completed - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # 학습 결과 저장
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': config,
        'total_epochs': len(train_losses)
    }
    
    results_file = os.path.join(config['results_dir'], 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()