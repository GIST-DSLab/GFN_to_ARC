#!/usr/bin/env python3
"""
Vocabulary 문제 디버깅
"""

import torch
import json
import os
import numpy as np
from utils.data_utils import create_vocabulary

def debug_data():
    """실제 데이터의 토큰 값 범위 확인"""
    print("=== Debugging Vocabulary Issues ===")
    
    # Vocabulary 확인
    vocab = create_vocabulary()
    print(f"Vocabulary size: {max(vocab.values()) + 1}")
    print(f"Vocabulary: {vocab}")
    
    # ReARC 데이터 파일 경로
    rearc_dir = "../LLM_experiment/data/re-arc/re_arc_extracted/re_arc/tasks"
    
    id_to_hex = {
        86: "25ff71a9",
        139: "6150a2bd",
        149: "67a3c6ac",
        178: "74dd1130",
        240: "9dfd6313"
    }
    
    all_values = set()
    
    for problem_id, hex_id in id_to_hex.items():
        file_path = os.path.join(rearc_dir, f"{hex_id}.json")
        if os.path.exists(file_path):
            print(f"\nChecking problem {problem_id} ({hex_id})")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 첫 3개 예제만 확인
            for i, example in enumerate(data[:3]):
                input_grid = example['input']
                output_grid = example['output']
                
                # Grid 값들 확인
                input_values = set(np.array(input_grid).flatten())
                output_values = set(np.array(output_grid).flatten())
                
                all_values.update(input_values)
                all_values.update(output_values)
                
                print(f"  Example {i}: input values {sorted(input_values)}, output values {sorted(output_values)}")
                
                # 위험한 값 체크 (vocab size 초과)
                dangerous_input = [v for v in input_values if v >= 22]
                dangerous_output = [v for v in output_values if v >= 22]
                
                if dangerous_input or dangerous_output:
                    print(f"    ⚠️  DANGEROUS VALUES: input {dangerous_input}, output {dangerous_output}")
    
    print(f"\n=== Summary ===")
    print(f"All unique values found: {sorted(all_values)}")
    print(f"Max value: {max(all_values) if all_values else 'None'}")
    print(f"Values >= vocab_size (22): {[v for v in all_values if v >= 22]}")
    
    return all_values

def test_encoding():
    """인코딩 함수 테스트"""
    print("\n=== Testing Encoding Function ===")
    
    # 문제가 될 수 있는 그리드 생성
    test_grids = [
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],  # 정상 범위
        [[9, 10, 11], [12, 13, 14], [15, 16, 17]],  # vocab 초과
        [[20, 21, 22], [23, 24, 25], [26, 27, 28]]  # 매우 큰 값
    ]
    
    for i, grid in enumerate(test_grids):
        print(f"\nTest grid {i+1}: {grid}")
        
        # 현재 인코딩 방식
        flat_grid = np.array(grid).flatten()
        tokens = [int(x) if x < 10 else 10 for x in flat_grid]
        
        print(f"Flattened: {flat_grid.tolist()}")
        print(f"Encoded tokens: {tokens}")
        print(f"Token range: {min(tokens)} - {max(tokens)}")
        
        # 문제 체크
        if max(flat_grid) >= 22:
            print(f"  ⚠️  Original values exceed vocab_size (22)")
        if max(tokens) >= 22:
            print(f"  🚨 CRITICAL: Encoded tokens exceed vocab_size!")

def fix_encoding():
    """수정된 인코딩 함수 제안"""
    print("\n=== Fixed Encoding Function ===")
    
    def encode_initial_state_fixed(grid_state):
        """수정된 인코딩 함수"""
        flat_grid = np.array(grid_state).flatten()
        
        # 모든 값을 vocab 범위 내로 클램핑 (0-9만 사용)
        tokens = [min(int(x), 9) for x in flat_grid]
        
        return tokens
    
    # 테스트
    test_grid = [[20, 21, 22], [23, 24, 25], [26, 27, 28]]
    print(f"Test grid: {test_grid}")
    
    original_tokens = [int(x) if x < 10 else 10 for x in np.array(test_grid).flatten()]
    fixed_tokens = encode_initial_state_fixed(test_grid)
    
    print(f"Original encoding: {original_tokens}")
    print(f"Fixed encoding: {fixed_tokens}")
    print(f"Max token (original): {max(original_tokens)}")
    print(f"Max token (fixed): {max(fixed_tokens)}")

if __name__ == "__main__":
    all_values = debug_data()
    test_encoding()
    fix_encoding()