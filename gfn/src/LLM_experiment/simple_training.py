#!/usr/bin/env python3
"""
간단한 학습 스크립트 (의존성 최소화)
"""

import os
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

print("Starting simple training script...")

try:
    import torch
    print(f"✅ PyTorch loaded: {torch.__version__}")
    
    # 기본 transformers만 사용
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import Trainer, TrainingArguments
    
    print("✅ Transformers loaded successfully")
    
    # 간단한 설정
    model_name = "microsoft/DialoGPT-small"
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델과 토크나이저 로드
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model and tokenizer loaded")
    
    # 간단한 학습 데이터
    train_texts = [
        "Input: [[0,1,2],[1,0,1]] Output: [[2,1,0],[1,0,1]] Actions: [left_rotate,submit]",
        "Input: [[1,2],[0,1]] Output: [[2,1],[1,0]] Actions: [horizontal_flip,submit]"
    ]
    
    # 토큰화
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    
    # 간단한 데이터셋
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings['input_ids'])
        
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    train_dataset = SimpleDataset(train_encodings)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./simple_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10,
        logging_steps=1,
        warmup_steps=10,
        learning_rate=5e-5,
        logging_dir='./logs',
    )
    
    # 트레이너
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    print("✅ Starting training...")
    trainer.train()
    
    print("✅ Training completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()