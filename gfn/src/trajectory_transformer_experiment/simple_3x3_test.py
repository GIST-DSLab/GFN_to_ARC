#!/usr/bin/env python3
"""
간단한 3x3 ReARC 문제 테스트
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from models.arc_transformer import ARCTrajectoryTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = "cuda"):
    """모델 로드"""
    config = {
        'n_layer': 6, 'n_head': 8, 'n_embd': 128,
        'observation_dim': 9, 'action_dim': 1, 'reward_dim': 1,
        'vocab_size': 26, 'max_sequence_length': 128
    }
    
    model = ARCTrajectoryTransformer(config)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def encode_grid(grid: List[List[int]]) -> List[int]:
    """3x3 그리드를 9개 토큰으로 인코딩"""
    tokens = []
    for row in grid:
        for cell in row:
            tokens.append(cell)
    return tokens

def generate_actions(model, input_grid: List[List[int]], device: str = "cuda") -> List[int]:
    """입력 그리드에서 액션 시퀀스 생성"""
    input_tokens = encode_grid(input_grid)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.9
        )
        
        # 액션 토큰 추출 (11-15 범위)
        actions = []
        generated_tokens = generated[0].cpu().numpy()
        
        for token in generated_tokens[len(input_tokens):]:
            if 11 <= token <= 15:  # 액션 토큰
                action = token - 11  # 0-4로 변환
                actions.append(action)
                if action == 4:  # submit 액션
                    break
                    
    return actions

def execute_actions(grid: List[List[int]], actions: List[int]) -> List[List[int]]:
    """액션 시퀀스를 그리드에 적용"""
    current_grid = [row[:] for row in grid]  # 복사
    
    for action in actions:
        if action == 0:  # 회전 없음
            continue
        elif action == 1:  # 왼쪽 회전 (90도)
            current_grid = [[current_grid[j][2-i] for j in range(3)] for i in range(3)]
        elif action == 2:  # 수평 뒤집기
            current_grid = [row[::-1] for row in current_grid]
        elif action == 3:  # 수직 뒤집기
            current_grid = current_grid[::-1]
        elif action == 4:  # 제출
            break
            
    return current_grid

def find_3x3_examples():
    """실제 3x3 예제 찾기"""
    rearc_dir = "../LLM_experiment/data/re-arc/re_arc_extracted/re_arc/tasks"
    
    # 실제로 3x3가 있는 파일들 확인
    found_3x3 = []
    
    test_files = [
        "25ff71a9.json",  # 문제 178 관련
        "74dd1130.json",  # 문제 178 관련  
        "67a3c6ac.json",  # 문제 149 관련
        "9dfd6313.json"   # 문제 240 관련
    ]
    
    for filename in test_files:
        filepath = os.path.join(rearc_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                for i, example in enumerate(data):
                    input_grid = example['input']
                    output_grid = example['output']
                    
                    if (len(input_grid) == 3 and len(input_grid[0]) == 3 and
                        len(output_grid) == 3 and len(output_grid[0]) == 3):
                        found_3x3.append({
                            'problem_id': filename.replace('.json', ''),
                            'example_id': i,
                            'input': input_grid,
                            'output': output_grid
                        })
                        break  # 파일당 하나씩만
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
    return found_3x3

def main():
    """메인 실행"""
    logger.info("=== 3x3 ReARC 간단 테스트 ===")
    
    # 모델 로드
    model_path = "./models/checkpoint_best.pt"
    device = "cuda"
    model = load_model(model_path, device)
    logger.info("Model loaded successfully")
    
    # 3x3 예제 찾기
    examples = find_3x3_examples()
    logger.info(f"Found {len(examples)} 3x3 examples")
    
    if not examples:
        logger.error("No 3x3 examples found")
        return
        
    # 평가 실행
    correct = 0
    total = len(examples)
    
    for example in examples:
        logger.info(f"Testing {example['problem_id']}, example {example['example_id']}")
        
        # 액션 생성
        actions = generate_actions(model, example['input'], device)
        logger.info(f"Generated actions: {actions}")
        
        # 액션 실행
        predicted = execute_actions(example['input'], actions)
        
        # 정확도 확인
        is_correct = predicted == example['output']
        if is_correct:
            correct += 1
            logger.info("✓ CORRECT")
        else:
            logger.info("✗ INCORRECT")
            logger.info(f"Expected: {example['output']}")
            logger.info(f"Predicted: {predicted}")
            
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info("=" * 50)
    logger.info(f"Final Results:")
    logger.info(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    logger.info(f"Total Examples: {total}")
    logger.info(f"Correct: {correct}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()