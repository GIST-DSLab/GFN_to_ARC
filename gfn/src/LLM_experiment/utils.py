#!/usr/bin/env python3
"""
유틸리티 함수들
"""

import json
import numpy as np
import torch
import yaml
from typing import Dict, List, Tuple, Any
import os

def load_config(config_path: str) -> Dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_json(data: Any, filepath: str):
    """JSON 파일 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str) -> Any:
    """JSON 파일 로드"""
    with open(filepath, 'r') as f:
        return json.load(f)

def trim_padding(grid: List[List[int]]) -> List[List[int]]:
    """30x30 그리드에서 실제 사용되는 영역만 추출"""
    grid = np.array(grid)
    
    # 0이 아닌 영역 찾기
    non_zero_rows = np.any(grid != 0, axis=1)
    non_zero_cols = np.any(grid != 0, axis=0)
    
    if not np.any(non_zero_rows) or not np.any(non_zero_cols):
        # 모두 0인 경우 최소 1x1 반환
        return [[0]]
    
    # 경계 찾기
    min_row = np.argmax(non_zero_rows)
    max_row = len(non_zero_rows) - 1 - np.argmax(non_zero_rows[::-1])
    min_col = np.argmax(non_zero_cols)
    max_col = len(non_zero_cols) - 1 - np.argmax(non_zero_cols[::-1])
    
    # 실제 크기 영역 추출
    trimmed = grid[min_row:max_row+1, min_col:max_col+1]
    return trimmed.tolist()

def get_actual_grid_size(grid: List[List[int]]) -> Tuple[int, int]:
    """실제 그리드 크기 계산"""
    trimmed = trim_padding(grid)
    return len(trimmed), len(trimmed[0]) if trimmed else (0, 0)

def map_gflownet_action_to_arc_action(gfn_action: int) -> int:
    """
    GFlowNet action ID를 ARC action ID로 매핑
    
    GFlowNet actions:
    0: left_rotate (ARC: 25)
    1: right_rotate (ARC: 24) 
    2: horizontal_flip (ARC: 26)
    3: vertical_flip (ARC: 27)
    4: submit (ARC: 34)
    """
    mapping = {
        0: 25,  # left rotate
        1: 24,  # right rotate  
        2: 26,  # horizontal flip
        3: 27,  # vertical flip
        4: 34   # submit
    }
    return mapping.get(gfn_action, gfn_action)

def map_arc_action_to_gflownet_action(arc_action: int) -> int:
    """ARC action ID를 GFlowNet action ID로 매핑"""
    mapping = {
        25: 0,  # left rotate
        24: 1,  # right rotate
        26: 2,  # horizontal flip  
        27: 3,  # vertical flip
        34: 4   # submit
    }
    return mapping.get(arc_action, arc_action)

def format_grid_for_llm(grid: List[List[int]], use_colors: bool = False) -> str:
    """그리드를 LLM이 이해할 수 있는 텍스트 형태로 변환"""
    if not grid or not grid[0]:
        return "[]"
    
    if use_colors:
        # BARC 스타일: 색상 이름 사용
        color_map = {
            0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
            5: "Grey", 6: "Pink", 7: "Orange", 8: "Teal", 9: "Maroon"
        }
        rows = []
        for row in grid:
            row_colors = [color_map.get(cell, str(cell)) for cell in row]
            rows.append(" ".join(row_colors))
        return "\n".join(rows)
    else:
        # 기존 형식: 숫자 배열
        rows = []
        for row in grid:
            row_str = "[" + ",".join(map(str, row)) + "]"
            rows.append(row_str)
        return "[" + ",".join(rows) + "]"

def parse_grid_from_llm(grid_str: str) -> List[List[int]]:
    """LLM 출력에서 그리드 파싱"""
    try:
        # 간단한 eval 사용 (보안상 실제 운영에서는 더 안전한 파싱 필요)
        grid = eval(grid_str)
        return grid
    except:
        return []

def format_action_sequence_for_llm(actions: List[int]) -> str:
    """액션 시퀀스를 LLM이 이해할 수 있는 형태로 변환"""
    action_names = {
        0: "left_rotate",
        1: "right_rotate", 
        2: "horizontal_flip",
        3: "vertical_flip",
        4: "submit"
    }
    
    action_list = [action_names.get(a, f"unknown_{a}") for a in actions]
    return "[" + ",".join(action_list) + "]"

def parse_action_sequence_from_llm(action_str: str) -> List[int]:
    """LLM 출력에서 액션 시퀀스 파싱"""
    name_to_id = {
        "left_rotate": 0,
        "right_rotate": 1,
        "horizontal_flip": 2, 
        "vertical_flip": 3,
        "submit": 4
    }
    
    try:
        # 문자열에서 액션 이름들 추출
        action_str = action_str.strip()
        if action_str.startswith('[') and action_str.endswith(']'):
            action_str = action_str[1:-1]
        
        action_names = [name.strip().strip('"\'') for name in action_str.split(',')]
        actions = [name_to_id.get(name, -1) for name in action_names]
        
        # 유효하지 않은 액션 제거
        actions = [a for a in actions if a != -1]
        return actions
    except:
        return []

def create_training_prompt(input_grid: List[List[int]], 
                          output_grid: List[List[int]], 
                          action_sequence: List[int],
                          use_barc_format: bool = True) -> str:
    """학습용 프롬프트 생성"""
    if use_barc_format:
        # BARC/Llama-3.1 스타일 프롬프트
        input_str = format_grid_for_llm(input_grid, use_colors=True)
        output_str = format_grid_for_llm(output_grid, use_colors=True)
        action_str = format_action_sequence_for_llm(action_sequence)
        
        system_prompt = "You are a world-class puzzle solver who is extremely good at spotting patterns and solving puzzles by applying transformations like rotations and flips."
        user_prompt = f"""Given an input grid, predict the sequence of actions (rotations and flips) needed to transform it into the output grid.

Input:
{input_str}

Target Output:
{output_str}

Actions: {action_str}"""
        
        # Llama-3.1 format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    else:
        # 기존 형식
        input_str = format_grid_for_llm(input_grid)
        output_str = format_grid_for_llm(output_grid) 
        action_str = format_action_sequence_for_llm(action_sequence)
        
        prompt = f"Input: {input_str}\nOutput: {output_str}\nActions: {action_str}"
        return prompt

def create_inference_prompt(input_grid: List[List[int]], 
                           output_grid: List[List[int]],
                           train_examples: List[Dict] = None,
                           use_barc_format: bool = True) -> str:
    """추론용 프롬프트 생성 (few-shot learning 지원)"""
    if use_barc_format:
        # BARC/Llama-3.1 스타일 프롬프트
        system_prompt = "You are a world-class puzzle solver who is extremely good at spotting patterns and solving puzzles by applying transformations like rotations and flips."
        
        # Few-shot examples 구성
        examples_text = ""
        if train_examples:
            for i, example in enumerate(train_examples[:3]):  # 최대 3개 예제 사용
                train_input_str = format_grid_for_llm(example['input'], use_colors=True)
                train_output_str = format_grid_for_llm(example['output'], use_colors=True)
                
                # 실제 액션 시퀀스가 있다면 사용, 없으면 예시
                if 'actions' in example:
                    actions_str = format_action_sequence_for_llm(example['actions'])
                else:
                    # 예시 액션 (실제로는 학습 데이터에서 가져와야 함)
                    actions_str = "[left_rotate,submit]"
                
                examples_text += f"""Example {i+1}:
Input:
{train_input_str}

Target Output:
{train_output_str}

Actions: {actions_str}

"""
        
        # 실제 문제
        input_str = format_grid_for_llm(input_grid, use_colors=True)
        output_str = format_grid_for_llm(output_grid, use_colors=True)
        
        user_prompt = f"""Given input grids and their target outputs, predict the sequence of actions (rotations and flips) needed to transform the input into the output.

{examples_text}Now solve this:
Input:
{input_str}

Target Output:
{output_str}

Predict the sequence of actions needed. Use action names: left_rotate, right_rotate, horizontal_flip, vertical_flip, submit.
Actions:"""
        
        # Llama-3.1 format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    else:
        # 기존 형식
        examples_text = ""
        if train_examples:
            for example in train_examples[:3]:
                train_input_str = format_grid_for_llm(example['input'])
                train_output_str = format_grid_for_llm(example['output'])
                examples_text += f"Input: {train_input_str}\nOutput: {train_output_str}\nActions: [left_rotate,submit]\n\n"
        
        input_str = format_grid_for_llm(input_grid)
        output_str = format_grid_for_llm(output_grid)
        
        prompt = f"{examples_text}Input: {input_str}\nOutput: {output_str}\nActions:"
        return prompt

def validate_action_sequence(actions: List[int]) -> bool:
    """액션 시퀀스 유효성 검증"""
    if not actions:
        return False
    
    # 마지막은 submit이어야 함
    if actions[-1] != 4:
        return False
    
    # 모든 액션이 유효한 범위 내에 있어야 함
    valid_actions = {0, 1, 2, 3, 4}
    return all(a in valid_actions for a in actions)

def calculate_accuracy(predicted_grids: List[List[List[int]]], 
                      target_grids: List[List[List[int]]]) -> float:
    """정확도 계산"""
    if len(predicted_grids) != len(target_grids):
        return 0.0
    
    correct = 0
    total = len(predicted_grids)
    
    for pred, target in zip(predicted_grids, target_grids):
        if np.array_equal(np.array(pred), np.array(target)):
            correct += 1
    
    return correct / total if total > 0 else 0.0

def setup_logging(log_file: str):
    """로깅 설정"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)