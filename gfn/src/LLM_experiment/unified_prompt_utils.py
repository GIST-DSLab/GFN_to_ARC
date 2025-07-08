#!/usr/bin/env python3
"""
통합된 프롬프트 유틸리티:
Training과 Inference에서 일관된 프롬프트 형식 사용
"""

from typing import List, Dict

def format_grid_for_llm(grid: List[List[int]], use_colors: bool = True) -> str:
    """그리드를 LLM용 텍스트로 변환"""
    if use_colors:
        color_map = {
            0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
            5: "Grey", 6: "Pink", 7: "Orange", 8: "Teal", 9: "Maroon"
        }
        
        rows = []
        for row in grid:
            color_row = [color_map.get(cell, "Black") for cell in row]
            rows.append(" ".join(color_row))
        return "\n".join(rows)
    else:
        return '\n'.join(' '.join(map(str, row)) for row in grid)

def create_examples_section(train_examples: List[Dict]) -> str:
    """예시 섹션 생성 (training과 inference 공통)"""
    examples_text = ""
    for i, example in enumerate(train_examples[:4]):  # 최대 4개
        example_input_str = format_grid_for_llm(example['input'], use_colors=True)
        example_output_str = format_grid_for_llm(example['output'], use_colors=True)
        
        examples_text += f"""Example {i+1}:
Input:
{example_input_str}

Output:
{example_output_str}

"""
    return examples_text

def create_training_prompt(train_examples: List[Dict]) -> str:
    """학습용 프롬프트 생성 (패턴 이해 후 action sequence 제시)"""
    
    examples_text = create_examples_section(train_examples)
    
    prompt = f"""{examples_text}Look at the examples to understand the transformation pattern. Then predict the sequence of actions needed to solve these types of problems.

Actions:"""
    
    return prompt

def create_inference_prompt(train_examples: List[Dict], test_input: List[List[int]]) -> str:
    """추론용 프롬프트 생성 (특정 test input에 대한 action 찾기)"""
    
    system_prompt = "You are a puzzle solver. Look at the training examples to understand the pattern, then solve the test case using rotations and flips."
    
    examples_text = create_examples_section(train_examples)
    test_input_str = format_grid_for_llm(test_input, use_colors=True)
    
    user_prompt = f"""Look at the examples to understand the transformation pattern. Then predict the sequence of actions needed to transform the test input.

{examples_text}Now solve this test case:
Input:
{test_input_str}

Actions:"""
    
    # Llama-3.1 format
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def parse_action_sequence_from_llm(action_str: str) -> List[int]:
    """LLM 응답에서 액션 시퀀스 파싱"""
    name_to_id = {
        "left_rotate": 0,
        "right_rotate": 1,
        "horizontal_flip": 2,
        "vertical_flip": 3,
        "submit": 4
    }
    
    try:
        action_str = action_str.strip()
        
        if '[' in action_str and ']' in action_str:
            start = action_str.find('[')
            end = action_str.find(']', start)
            if start < end:
                bracket_content = action_str[start+1:end]
            else:
                bracket_content = action_str
        else:
            bracket_content = action_str
        
        action_names = [name.strip().strip('"\'') for name in bracket_content.split(',')]
        actions = [name_to_id.get(name, -1) for name in action_names if name.strip()]
        
        actions = [a for a in actions if a != -1]
        return actions
    except:
        return []

def format_action_sequence_for_training(actions: List[int]) -> str:
    """학습용 액션 시퀀스 형식화"""
    action_names = ['left_rotate', 'right_rotate', 'horizontal_flip', 'vertical_flip', 'submit']
    action_sequence = [action_names[action] for action in actions]
    return f"[{','.join(action_sequence)}]"