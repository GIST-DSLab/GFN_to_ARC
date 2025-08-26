#!/usr/bin/env python3
"""
Simple training script using standard transformers instead of unsloth
"""

import os
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
from peft import LoraConfig, get_peft_model, TaskType

def load_data(data_path):
    """Load training data"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to simple text format
    texts = []
    for item in data:
        prompt = item['prompt']
        completion = item['completion']
        # Simple format: prompt + completion
        text = f"{prompt}\n{completion}"
        texts.append(text)
    
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    
    # Set HF token from environment
    if 'HF_TOKEN' not in os.environ:
        raise ValueError("HF_TOKEN environment variable must be set")
    
    print(f"Loading data from: {args.data_path}")
    texts = load_data(args.data_path)
    print(f"Total samples: {len(texts)}")
    
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        token=os.environ['HF_TOKEN']
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ['HF_TOKEN'],
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("Configuring LoRA...")
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for model
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Tokenize data
    print("Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=2048)
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        gradient_checkpointing=False,  # Disabled since we manually enable it on model
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()