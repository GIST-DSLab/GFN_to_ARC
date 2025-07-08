#!/usr/bin/env python3
"""
LLM 학습: Unsloth를 사용한 고속 LoRA 파인튜닝
"""

import os
import json
import torch
import argparse
import yaml
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
import wandb

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth is available")
except ImportError:
    print("❌ Unsloth not available, falling back to regular training")
    UNSLOTH_AVAILABLE = False

from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset

def load_config(config_path: str) -> Dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_json(file_path: str) -> List[Dict]:
    """JSON 파일 로드"""
    with open(file_path, 'r') as f:
        return json.load(f)

def setup_logging(log_file: str) -> logging.Logger:
    """로깅 설정"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_training_data(config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """학습 데이터 로드"""
    import glob
    from sklearn.model_selection import train_test_split
    
    # 모든 problem_*_processed.json 파일들 찾기
    problem_files = glob.glob(os.path.join(config['processed_data_dir'], "problem_*_processed.json"))
    
    if not problem_files:
        raise FileNotFoundError(f"No training data found in {config['processed_data_dir']}")
    
    all_data = []
    for problem_file in tqdm(problem_files, desc="Loading problem files", unit="file"):
        try:
            problem_data = load_json(problem_file)
            if problem_data:
                all_data.extend(problem_data)
                print(f"Loaded {len(problem_data)} samples from {os.path.basename(problem_file)}")
        except Exception as e:
            print(f"Error loading {problem_file}: {e}")
            continue
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # train/val 분할
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
    return train_data, val_data

def format_data_for_unsloth(data: List[Dict]) -> List[Dict]:
    """Unsloth 형식으로 데이터 변환"""
    formatted_data = []
    
    for item in tqdm(data, desc="Formatting data for Unsloth", unit="sample"):
        # Unsloth는 "text" 키를 사용
        full_text = item['prompt'] + item['completion']
        formatted_data.append({
            "text": full_text
        })
    
    return formatted_data

def create_unsloth_model_and_tokenizer(config: Dict):
    """Unsloth 모델과 토크나이저 생성"""
    lora_config = config.get('lora', {})
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        max_seq_length=config['max_length'],
        dtype=None,  # Unsloth가 자동으로 선택
        load_in_4bit=lora_config.get('load_in_4bit', True),  # 4bit 양자화로 메모리 절약
    )
    
    # LoRA 어댑터 추가
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get('r', 16),  # LoRA rank
        target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', "none"),
        use_gradient_checkpointing="unsloth",  # Unsloth의 최적화된 체크포인팅
        random_state=lora_config.get('random_state', 3407),
        use_rslora=lora_config.get('use_rslora', False),
        loftq_config=None,
    )
    
    return model, tokenizer

def train_with_unsloth(config: Dict, logger: logging.Logger) -> None:
    """Unsloth를 사용한 고속 학습"""
    logger.info("Starting Unsloth-accelerated training...")
    
    # 데이터 로드
    logger.info("Loading training data...")
    train_data, val_data = load_training_data(config)
    
    # Unsloth 형식으로 변환
    logger.info("Converting data to Unsloth format...")
    train_formatted = format_data_for_unsloth(train_data)
    val_formatted = format_data_for_unsloth(val_data)
    
    # Dataset 객체 생성
    logger.info("Creating Dataset objects...")
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    # 모델과 토크나이저 생성
    logger.info("Creating Unsloth model and tokenizer...")
    model, tokenizer = create_unsloth_model_and_tokenizer(config)
    
    # 학습 설정 가져오기
    training_config = config.get('training', {})
    optimizer_config = config.get('optimizer', {})
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=config.get('batch_size', 8),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 2),
        warmup_steps=config.get('warmup_steps', 50),
        num_train_epochs=config.get('num_epochs', 1),
        learning_rate=config.get('learning_rate', 2e-4),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=training_config.get('logging_steps', 10),
        optim=optimizer_config.get('name', "adamw_8bit"),
        weight_decay=optimizer_config.get('weight_decay', 0.01),
        lr_scheduler_type=optimizer_config.get('lr_scheduler_type', "linear"),
        seed=training_config.get('random_seed', 3407),
        output_dir=config['model_save_dir'],
        save_steps=training_config.get('save_steps', 100),
        eval_steps=training_config.get('eval_steps', 100),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    # SFT Trainer 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config['max_length'],
        dataset_num_proc=optimizer_config.get('dataset_num_proc', 2),
        packing=optimizer_config.get('packing', False),
        args=training_args,
    )
    
    # 학습 실행
    logger.info("Starting training...")
    trainer.train()
    
    # 모델 저장
    model_save_path = os.path.join(config['model_save_dir'], "unsloth_lora_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    logger.info(f"Unsloth training completed. Model saved to: {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description="ARC Action Sequence LLM Training with Unsloth")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Unsloth 가용성 체크
    if not UNSLOTH_AVAILABLE:
        print("❌ Unsloth is not available. Please install it first:")
        print("pip install unsloth")
        return
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    log_dir = config.get('results_dir', './results')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "unsloth_training.log")
    logger = setup_logging(log_file)
    
    logger.info("Starting Unsloth training with config:")
    logger.info(f"  Model: {config['model_name']}")
    logger.info(f"  Max length: {config['max_length']}")
    logger.info(f"  Batch size: {config.get('batch_size', 8)}")
    logger.info(f"  Learning rate: {config.get('learning_rate', 2e-4)}")
    logger.info(f"  Num epochs: {config.get('num_epochs', 1)}")
    logger.info(f"  Gradient accumulation steps: {config.get('gradient_accumulation_steps', 2)}")
    
    try:
        train_with_unsloth(config, logger)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()