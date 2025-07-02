#!/usr/bin/env python3
"""
LLM 학습: trajectory 데이터를 사용하여 action sequence 예측 모델 학습
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import numpy as np
from typing import List, Dict, Any, Tuple
from utils import *
import logging
import wandb
from sklearn.model_selection import train_test_split

class ARCActionDataset(Dataset):
    """ARC Action Sequence 데이터셋"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 입력 텍스트 구성
        prompt = item['prompt']
        completion = item['completion']
        
        # 전체 텍스트 (prompt + completion)
        full_text = prompt + completion
        
        # 토큰화
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 프롬프트 부분만 토큰화 (마스킹용)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # 라벨 생성 (프롬프트 부분은 -100으로 마스킹)
        labels = input_ids.clone()
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class ARCActionTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # GPU 개수 확인
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {self.num_gpus}")
        else:
            self.num_gpus = 0
        
        # 토크나이저와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name']
        )
        
        # 단일 GPU 사용 (멀티 GPU는 일단 비활성화)
        self.model.to(self.device)
        
    def load_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """학습 데이터 로드"""
        train_file = os.path.join(self.config['processed_data_dir'], "train_data.json")
        val_file = os.path.join(self.config['processed_data_dir'], "val_data.json")
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            # 전체 데이터에서 분할
            all_data_file = os.path.join(self.config['processed_data_dir'], "all_training_data.json")
            if not os.path.exists(all_data_file):
                raise FileNotFoundError(f"Training data not found. Run preprocessing first.")
                
            all_data = load_json(all_data_file)
            train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
            
            # 저장
            save_json(train_data, train_file)
            save_json(val_data, val_file)
        else:
            train_data = load_json(train_file)
            val_data = load_json(val_file)
            
        self.logger.info(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data
    
    def create_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        """데이터셋 생성"""
        train_dataset = ARCActionDataset(train_data, self.tokenizer, self.config['max_length'])
        val_dataset = ARCActionDataset(val_data, self.tokenizer, self.config['max_length'])
        
        return train_dataset, val_dataset
    
    def setup_training_arguments(self):
        """학습 인자 설정"""
        model_name_safe = self.config['model_name'].split('/')[-1].replace('.', '_')
        output_dir = os.path.join(self.config.get('model_save_dir', './models'), f"arc_action_model_{model_name_safe}")
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            learning_rate=self.config['learning_rate'],
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            eval_steps=200,
            save_steps=200,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_drop_last=False,
            fp16=False,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 2),
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb",
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        predictions, labels = eval_pred
        
        # 예측값에서 실제 토큰 ID 추출
        predictions = np.argmax(predictions, axis=-1)
        
        # 마스킹된 부분 제외하고 정확도 계산
        mask = labels != -100
        correct = (predictions == labels) & mask
        accuracy = correct.sum() / mask.sum()
        
        return {"accuracy": accuracy}
    
    def train(self):
        """모델 학습"""
        self.logger.info("Starting model training...")
        
        # wandb 로그인 및 초기화
        wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
        wandb.init(
            project="arc-action-sequence",
            config=self.config,
            name=f"arc_llm_{self.config['model_name'].split('/')[-1]}",
            tags=["llama3.1", "action_sequence", "arc"]
        )
        
        # 데이터 로드
        train_data, val_data = self.load_training_data()
        train_dataset, val_dataset = self.create_datasets(train_data, val_data)
        
        # 학습 인자 설정
        training_args = self.setup_training_arguments()
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM이므로 MLM 사용 안함
        )
        
        # 트레이너 설정
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        # 학습 실행
        trainer.train()
        
        # 최종 모델 저장
        final_model_path = os.path.join(training_args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        self.logger.info(f"Training completed. Model saved to {final_model_path}")
        
        # 평가 결과 로깅
        eval_results = trainer.evaluate()
        self.logger.info(f"Final evaluation results: {eval_results}")
        
        wandb.finish()
        
        return trainer
    
    def generate_action_sequence(self, input_grid: List[List[int]], 
                                output_grid: List[List[int]], 
                                max_new_tokens: int = 50) -> List[int]:
        """주어진 입력-출력 쌍에 대해 액션 시퀀스 생성"""
        
        # 프롬프트 생성 (BARC 형식 사용)
        prompt = create_inference_prompt(input_grid, output_grid, use_barc_format=True)
        
        # 토큰화
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = generated_text[len(prompt):].strip()
        
        # 액션 시퀀스 파싱
        actions = parse_action_sequence_from_llm(response)
        
        return actions

def main():
    # 설정 로드
    config = load_config("configs/config.yaml")
    
    # 로깅 설정
    log_dir = config.get('results_dir', './results')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logger = setup_logging(log_file)
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 트레이너 초기화 및 학습
    trainer = ARCActionTrainer(config)
    trained_model = trainer.train()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()