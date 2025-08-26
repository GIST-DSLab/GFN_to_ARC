#!/usr/bin/env python3
"""
프롬프트 테스트 - 간단한 액션 요청
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('/home/ubuntu/GFN_to_ARC/gfn/src')
from utils import parse_action_sequence_from_llm

def test_simple_action_prompt():
    print("=== Testing Simple Action Prompt ===")
    
    # Fine-tuned 모델 로드
    model_path = "/data/gflownet-llm/models/models_gpu456/unsloth_lora_model/"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 여러 가지 프롬프트 테스트
    prompts = [
        # 1. 매우 간단한 액션 요청
        "List these actions: left_rotate, right_rotate, horizontal_flip, vertical_flip, submit",
        
        # 2. 직접적인 액션 요청
        "What actions can you perform? Please list: left_rotate, right_rotate, horizontal_flip, vertical_flip, submit",
        
        # 3. 액션 시퀀스 요청
        "Generate an action sequence using: left_rotate, right_rotate, horizontal_flip, vertical_flip, submit",
        
        # 4. Llama 포맷
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease list these actions: left_rotate, right_rotate, horizontal_flip, vertical_flip, submit<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt[:100]}...")
        
        inputs = tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'].to(model.device),
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Response: {response}")
        
        actions = parse_action_sequence_from_llm(response)
        print(f"Parsed actions: {actions}")

if __name__ == "__main__":
    test_simple_action_prompt()