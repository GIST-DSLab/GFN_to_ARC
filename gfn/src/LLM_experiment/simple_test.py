#!/usr/bin/env python3
"""
간단한 import 테스트
"""

print("Testing imports...")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    # Flash attention 비활성화
    import os
    os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers imported successfully")
    
    # 간단한 모델 로드 테스트
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    print("✅ Tokenizer loaded")
    
except Exception as e:
    print(f"❌ Transformers import failed: {e}")
    
print("\nTrying to import without flash attention...")
try:
    # transformers 버전 확인
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Error: {e}")