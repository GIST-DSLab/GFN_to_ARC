#!/usr/bin/env python3
"""
모델을 직접 테스트해서 CUDA 오류 원인 파악
"""

import torch
import os
import json
import numpy as np
from models.arc_transformer import create_model
from utils.data_utils import create_vocabulary
import yaml

def test_model_loading():
    """모델 로딩 테스트"""
    print("=== Testing Model Loading ===")
    
    # Config 로드
    with open('configs/config_gpu456.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = "/data/gflownet-llm/models/rl-models/checkpoint_best.pt"
    print(f"Model path: {model_path}")
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    model_config = checkpoint.get('config', config)
    print(f"Model config vocab_size: {model_config.get('vocab_size', 'not found')}")
    
    # 모델 생성
    model = create_model(model_config)
    print(f"Model vocab_size: {model.vocab_size}")
    print(f"Model embedding size: {model.token_embedding.num_embeddings}")
    
    return model, checkpoint

def test_simple_forward():
    """간단한 forward 테스트"""
    print("\n=== Testing Simple Forward Pass ===")
    
    model, checkpoint = test_model_loading()
    
    # 간단한 입력 생성
    vocab = create_vocabulary()
    input_ids = torch.tensor([[vocab['sos'], 0, 1, 2, vocab['eos']]])  # 작은 시퀀스
    
    print(f"Input IDs: {input_ids}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Max input ID: {input_ids.max().item()}")
    print(f"Min input ID: {input_ids.min().item()}")
    
    try:
        # Forward pass
        outputs = model(input_ids)
        print(f"✅ Forward pass successful!")
        print(f"Output logits shape: {outputs['logits'].shape}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_problematic_inputs():
    """문제가 될 수 있는 입력들 테스트"""
    print("\n=== Testing Problematic Inputs ===")
    
    model, checkpoint = test_model_loading()
    vocab = create_vocabulary()
    
    test_cases = [
        ([vocab['sos'], 22, vocab['eos']], "Token 22 (exceeds vocab)"),
        ([vocab['sos'], -1, vocab['eos']], "Token -1 (negative)"),
        ([vocab['sos']] + list(range(22, 30)) + [vocab['eos']], "Multiple tokens exceeding vocab"),
        ([vocab['sos']] + [10] * 65 + [vocab['eos']], "Very long sequence"),
    ]
    
    for test_input, description in test_cases:
        print(f"\nTesting: {description}")
        input_ids = torch.tensor([test_input])
        print(f"Input IDs: {input_ids[:5].tolist()}...")
        print(f"Max ID: {input_ids.max().item()}, Min ID: {input_ids.min().item()}")
        
        try:
            outputs = model(input_ids)
            print(f"✅ Success")
        except Exception as e:
            print(f"❌ Failed: {e}")

def test_with_cuda():
    """CUDA에서 테스트"""
    print("\n=== Testing with CUDA ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    device = torch.device('cuda')
    model, checkpoint = test_model_loading()
    model = model.to(device)
    
    vocab = create_vocabulary()
    input_ids = torch.tensor([[vocab['sos'], 0, 1, 2, vocab['eos']]]).to(device)
    
    print(f"Input IDs on CUDA: {input_ids}")
    
    try:
        outputs = model(input_ids)
        print(f"✅ CUDA forward pass successful!")
        return True
    except Exception as e:
        print(f"❌ CUDA forward pass failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_forward()
    if success:
        test_problematic_inputs()
        test_with_cuda()
    else:
        print("Basic test failed, skipping advanced tests")