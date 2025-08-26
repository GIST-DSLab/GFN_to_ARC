#!/usr/bin/env python3
"""
Baseline 모델 학습: 동일한 문제에 대해 같은 답을 반복 학습
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from typing import List, Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineDataset(Dataset):
    """Baseline 학습 데이터셋"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 프롬프트와 completion 결합
        full_text = item['prompt'] + item['completion']
        
        # 토크나이징
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # labels 설정 (프롬프트 부분은 -100으로 마스킹)
        prompt_length = len(self.tokenizer.encode(item['prompt'], add_special_tokens=False))
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def train_baseline_model(config: Dict):
    """Baseline 모델 학습"""
    
    # 모델과 토크나이저 로드
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터셋 로드
    logger.info("Loading datasets...")
    train_dataset = BaselineDataset(
        config['train_data_path'],
        tokenizer,
        max_length=config['max_length']
    )
    
    val_dataset = BaselineDataset(
        config['val_data_path'],
        tokenizer,
        max_length=config['max_length']
    )
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        warmup_steps=config['warmup_steps'],
        weight_decay=0.01,
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        report_to=["tensorboard"],
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # 학습 시작
    logger.info("Starting training...")
    trainer.train()
    
    # 모델 저장
    logger.info(f"Saving model to {config['final_model_dir']}")
    trainer.save_model(config['final_model_dir'])
    tokenizer.save_pretrained(config['final_model_dir'])
    
    # 학습 통계 저장
    with open(f"{config['output_dir']}/training_stats.json", 'w') as f:
        json.dump({
            'model_name': config['model_name'],
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'num_epochs': config['num_epochs'],
            'final_loss': trainer.state.best_metric,
            'training_completed': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info("Training completed!")

def main():
    # Baseline 학습 설정
    config = {
        'model_name': 'meta-llama/Llama-3.2-1B',  # 더 작은 모델로 테스트
        'train_data_path': './processed_data/baseline_train_data.json',
        'val_data_path': './processed_data/baseline_val_data.json',
        'output_dir': './models/baseline_model',
        'final_model_dir': './models/baseline_model/final',
        'max_length': 512,
        'batch_size': 4,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'warmup_steps': 100,
        'gradient_accumulation_steps': 4,
    }
    
    # 학습 실행
    train_baseline_model(config)

if __name__ == "__main__":
    main()