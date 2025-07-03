#!/usr/bin/env python3
"""
LLM 학습: trajectory 데이터를 사용하여 action sequence 예측 모델 학습
"""

import os
import json

# Flash attention 비활성화
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
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
import argparse
from tqdm import tqdm

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
    def __init__(self, config: Dict, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger(__name__)
        
        # DDP 설정
        if world_size > 1:
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Rank {rank}: Using device: {self.device}")
        
        # GPU 개수 확인
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {self.num_gpus}, World size: {world_size}")
        else:
            self.num_gpus = 0
        
        # 토크나이저와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Llama 8B인 경우 8비트 양자화 사용
        if "Llama" in config['model_name']:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    config['model_name'],
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            except ImportError:
                self.logger.warning("BitsAndBytesConfig not available, using regular loading")
                self.model = AutoModelForCausalLM.from_pretrained(
                    config['model_name'],
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # DDP 래핑 (멀티 GPU인 경우)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
    def load_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """학습 데이터 로드"""
        train_file = os.path.join(self.config['processed_data_dir'], "train_data.json")
        val_file = os.path.join(self.config['processed_data_dir'], "val_data.json")
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            # 전체 데이터에서 분할
            all_data_file = os.path.join(self.config['processed_data_dir'], "all_training_data.json")
            if os.path.exists(all_data_file):
                print("Loading training data from all_training_data.json...")
                all_data = load_json(all_data_file)
            else:
                # all_training_data.json이 없으면 개별 problem 파일들을 찾아서 로드
                print("all_training_data.json not found. Loading from individual problem files...")
                all_data = []
                
                # processed_data_dir에서 problem_*_processed.json 파일들 찾기
                import glob
                problem_files = glob.glob(os.path.join(self.config['processed_data_dir'], "problem_*_processed.json"))
                
                if not problem_files:
                    raise FileNotFoundError(f"No training data found in {self.config['processed_data_dir']}. Run preprocessing first.")
                
                problem_files.sort()  # 파일 순서 정렬
                print(f"Found {len(problem_files)} problem files to load:")
                
                for problem_file in problem_files:
                    problem_name = os.path.basename(problem_file)
                    print(f"  Loading {problem_name}...")
                    try:
                        problem_data = load_json(problem_file)
                        if problem_data:  # 빈 배열이 아닌 경우만 추가
                            all_data.extend(problem_data)
                            print(f"    Added {len(problem_data)} samples from {problem_name}")
                        else:
                            print(f"    Skipped {problem_name} (empty)")
                    except Exception as e:
                        print(f"    Error loading {problem_name}: {e}")
                        continue
                
                if not all_data:
                    raise FileNotFoundError(f"No valid training data found. All problem files are empty or invalid.")
                
                print(f"Total samples loaded: {len(all_data)}")
            
            print(f"Splitting {len(all_data)} samples into train/validation...")
            train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
            
            # 저장
            print("Saving train/val splits...")
            save_json(train_data, train_file)
            save_json(val_data, val_file)
        else:
            print("Loading existing train/val data...")
            train_data = load_json(train_file)
            val_data = load_json(val_file)
            
        self.logger.info(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data
    
    def create_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        """데이터셋 생성"""
        print("Creating training dataset...")
        train_dataset = ARCActionDataset(train_data, self.tokenizer, self.config['max_length'])
        print("Creating validation dataset...")
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
            per_device_eval_batch_size=1,  # 평가 배치 크기를 1로 고정
            warmup_steps=self.config['warmup_steps'],
            learning_rate=self.config['learning_rate'],
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            eval_steps=500,  # 평가 주기를 늘림
            save_steps=500,  # 저장 주기를 늘림
            eval_strategy="no",  # 평가 비활성화
            save_strategy="steps",
            load_best_model_at_end=False,  # 평가가 없으므로 비활성화
            # metric_for_best_model="eval_loss",  # 평가가 없으므로 주석 처리
            # greater_is_better=False,  # 평가가 없으므로 주석 처리
            dataloader_drop_last=False,
            fp16=False,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 2),
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb" if self.rank == 0 else None,
            dataloader_pin_memory=False,  # 메모리 사용량 줄이기
            skip_memory_metrics=True,  # 메모리 메트릭 건너뛰기
            torch_empty_cache_steps=5,  # 더 자주 캐시 비우기
            ddp_timeout=7200,  # DDP 타임아웃 2시간
            ddp_bucket_cap_mb=25,  # DDP 버킷 크기 줄이기
            ddp_broadcast_buffers=False,  # 브로드캐스트 비활성화
            gradient_checkpointing=False,  # DDP에서는 비활성화
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
        
        # wandb 로그인 및 초기화 (rank 0에서만)
        if self.rank == 0:
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
            eval_dataset=None,  # 평가 비활성화
            data_collator=data_collator,
            compute_metrics=None,  # 평가 메트릭 비활성화
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Early stopping 비활성화
        )
        
        # 학습 실행
        trainer.train()
        
        # 최종 모델 저장
        final_model_path = os.path.join(training_args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        self.logger.info(f"Training completed. Model saved to {final_model_path}")
        
        # 평가 결과 로깅 (평가 비활성화로 인해 주석 처리)
        # eval_results = trainer.evaluate()
        # self.logger.info(f"Final evaluation results: {eval_results}")
        
        if self.rank == 0:
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

def setup_ddp(rank: int, world_size: int):
    """DDP 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    # NCCL 설정 최적화
    os.environ['NCCL_DEBUG'] = 'WARN'  # 로그 레벨 낮춤
    os.environ['NCCL_TIMEOUT'] = '3600'  # 1시간 타임아웃
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '3600'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA 동기화
    
    # CUDA 백엔드 사용 (GPU용)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """DDP 정리"""
    destroy_process_group()

def train_ddp(rank: int, world_size: int, config: Dict):
    """DDP 학습 함수"""
    setup_ddp(rank, world_size)
    
    # 로깅 설정 (rank 0에서만)
    if rank == 0:
        log_dir = config.get('results_dir', './results')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        logger = setup_logging(log_file)
        logger.info(f"Starting DDP training with {world_size} GPUs")
    
    # 트레이너 초기화 및 학습
    trainer = ARCActionTrainer(config, rank, world_size)
    trained_model = trainer.train()
    
    if rank == 0:
        logger.info("DDP Training completed successfully!")
    
    cleanup_ddp()

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="ARC Action Sequence LLM Training")
    parser.add_argument("--gpus", type=int, default=1, 
                       help="Number of GPUs to use (default: 1)")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=None,
                       help="Specific GPU IDs to use (e.g., --gpu_ids 0 1 2)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--download_data", action="store_true",
                       help="Download re-arc data before training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 데이터 다운로드 (필요한 경우)
    if args.download_data:
        print("Downloading data...")
        import subprocess
        result = subprocess.run([sys.executable, "download_data.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Data download failed: {result.stderr}")
            return
        print("Data download completed!")
    
    # 설정 로드
    config = load_config(args.config)
    
    # GPU 설정
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 사용할 GPU 개수 결정
        if args.gpu_ids:
            # 특정 GPU ID 지정
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
            world_size = len(args.gpu_ids)
            print(f"Using specific GPUs: {args.gpu_ids}")
        else:
            # GPU 개수 지정
            world_size = min(args.gpus, num_gpus)
            print(f"Using {world_size} GPUs")
    else:
        world_size = 1
        print("CUDA not available, using CPU")
    
    if world_size == 1:
        # 단일 GPU/CPU 학습
        print("Starting single GPU/CPU training")
        log_dir = config.get('results_dir', './results')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        logger = setup_logging(log_file)
        
        trainer = ARCActionTrainer(config)
        trained_model = trainer.train()
        
        logger.info("Training completed successfully!")
    else:
        # 멀티 GPU 학습 (DDP)
        print(f"Starting multi-GPU training with {world_size} GPUs")
        mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    import sys
    main()