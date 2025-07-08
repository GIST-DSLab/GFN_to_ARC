#!/usr/bin/env python3
"""
GPU456 모델 결과 재현용 추론 스크립트
utils.py의 create_inference_prompt와 동일한 형식 사용
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def load_json(file_path: str) -> Any:
    """JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_grid_for_llm(grid: List[List[int]], use_colors: bool = True) -> str:
    """그리드를 LLM용 텍스트로 변환 (utils.py와 동일)"""
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

def create_inference_prompt(input_grid: List[List[int]], 
                           train_examples: List[Dict] = None) -> str:
    """추론용 프롬프트 생성 (utils.py와 동일한 BARC 형식)"""
    # BARC/Llama-3.1 스타일 프롬프트
    system_prompt = "You are a world-class puzzle solver who is extremely good at spotting patterns and solving puzzles by applying transformations like rotations and flips."
    
    # Few-shot examples 구성 (input-output pair만, 액션 제외)
    examples_text = ""
    if train_examples:
        for i, example in enumerate(train_examples[:2]):  # 최대 2개 예제 사용
            train_input_str = format_grid_for_llm(example['input'], use_colors=True)
            train_output_str = format_grid_for_llm(example['output'], use_colors=True)
            
            examples_text += f"""Example {i+1}:
Input:
{train_input_str}

Target Output:
{train_output_str}

"""
    
    # 실제 문제 (input만 제공, output은 숨김)
    input_str = format_grid_for_llm(input_grid, use_colors=True)
    
    user_prompt = f"""Look at the examples to understand the transformation pattern. Then predict the sequence of actions (rotations and flips) needed to transform the given input.

{examples_text}Now solve this:
Input:
{input_str}

Based on the pattern from the examples, predict the sequence of actions needed. Use action names: left_rotate, right_rotate, horizontal_flip, vertical_flip, submit.
Respond with ONLY the action sequence in this exact format: [action1,action2,submit]
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

class ARCActionExecutor:
    """ARC 액션 실행기"""
    
    def apply_action_to_grid(self, grid: List[List[int]], action: int) -> List[List[int]]:
        """그리드에 액션 적용"""
        if not grid or not grid[0]:
            return grid
            
        grid_array = np.array(grid)
        
        if action == 0:  # left_rotate
            result = np.rot90(grid_array, k=1)
        elif action == 1:  # right_rotate
            result = np.rot90(grid_array, k=-1)
        elif action == 2:  # horizontal_flip
            result = np.fliplr(grid_array)
        elif action == 3:  # vertical_flip
            result = np.flipud(grid_array)
        else:
            result = grid_array
            
        return result.tolist()
    
    def execute_action_sequence(self, initial_grid: List[List[int]], actions: List[int]) -> List[List[int]]:
        """액션 시퀀스 실행"""
        current_grid = [row[:] for row in initial_grid]  # 복사
        
        for action in actions:
            if action == 4:  # submit
                break
            current_grid = self.apply_action_to_grid(current_grid, action)
        
        return current_grid

class GPU456ModelTester:
    """GPU456 모델 테스터"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logging()
        
        # 모델과 토크나이저 로드
        self.tokenizer, self.model = self.load_model_and_tokenizer()
        
        # 액션 실행기
        self.executor = ARCActionExecutor()
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        self.logger.info(f"Loading model from {self.model_path}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA 모델 로드
        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.eval()
        
        return tokenizer, model
    
    def generate_action_sequence(self, arc_train_examples, test_input, max_new_tokens=50):
        """액션 시퀀스 생성"""
        # utils.py의 create_inference_prompt 사용
        prompt = create_inference_prompt(test_input, arc_train_examples)
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # 생성 (원본 inference.py와 동일한 파라미터)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 디코딩 (프롬프트 부분만 먼저 제거)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 액션 시퀀스 파싱
        actions = parse_action_sequence_from_llm(response)
        
        return actions, response
    
    def evaluate_test_case(self, test_case: Dict, arc_train_examples: List[Dict]) -> Dict:
        """단일 테스트 케이스 평가"""
        input_grid = test_case['input']
        target_output = test_case['output']
        
        # 액션 시퀀스 생성
        predicted_actions, raw_response = self.generate_action_sequence(
            arc_train_examples, input_grid
        )
        
        # 액션 시퀀스 실행
        if predicted_actions:
            predicted_grid = self.executor.execute_action_sequence(input_grid, predicted_actions)
        else:
            predicted_grid = input_grid
        
        # 정확도 계산
        is_correct = predicted_grid == target_output
        
        # 픽셀 단위 정확도
        if predicted_grid and target_output:
            total_pixels = len(target_output) * len(target_output[0])
            correct_pixels = sum(
                1 for i in range(len(target_output))
                for j in range(len(target_output[0]))
                if i < len(predicted_grid) and j < len(predicted_grid[0]) 
                and predicted_grid[i][j] == target_output[i][j]
            )
            pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
        else:
            pixel_accuracy = 0
        
        return {
            'input_grid': input_grid,
            'target_output': target_output,
            'predicted_actions': predicted_actions,
            'predicted_grid': predicted_grid,
            'is_correct': is_correct,
            'pixel_accuracy': pixel_accuracy,
            'raw_response': raw_response
        }
    
    def evaluate_problem(self, problem_id: int, arc_id: str) -> Dict:
        """문제 평가"""
        self.logger.info(f"Evaluating problem {problem_id} ({arc_id})")
        
        # ARC 데이터 로드
        arc_file = f"data/re-arc/arc_original/training/{arc_id}.json"
        if not os.path.exists(arc_file):
            self.logger.error(f"ARC file not found: {arc_file}")
            return None
            
        with open(arc_file, 'r') as f:
            arc_data = json.load(f)
        
        arc_train_examples = arc_data['train']
        arc_test_cases = arc_data['test']
        
        # 각 테스트 케이스 평가
        test_results = []
        for i, test_case in enumerate(arc_test_cases):
            result = self.evaluate_test_case(test_case, arc_train_examples)
            result['test_idx'] = i
            test_results.append(result)
        
        # 문제 전체 통계
        correct_count = sum(1 for r in test_results if r['is_correct'])
        total_count = len(test_results)
        exact_accuracy = correct_count / total_count if total_count > 0 else 0
        pixel_accuracy = sum(r['pixel_accuracy'] for r in test_results) / total_count if total_count > 0 else 0
        
        return {
            'problem_id': problem_id,
            'arc_id': arc_id,
            'test_results': test_results,
            'exact_accuracy': exact_accuracy,
            'pixel_accuracy': pixel_accuracy,
            'correct_count': correct_count,
            'total_count': total_count
        }

def main():
    logger = setup_logging()
    
    # GPU456 모델 경로
    model_path = "/opt/dlami/nvme/seungpil/models_gpu456/unsloth_lora_model"
    
    # 테스터 초기화
    tester = GPU456ModelTester(model_path)
    
    # 평가할 문제들 (gpu456에서 평가한 것과 동일)
    problems = [
        (86, "25ff71a9"),
        (139, "6150a2bd"), 
        (178, "74dd1130"),
        (149, "6773b310"),
        (154, "6855a6e4"),
        (240, "9d9215db"),
        (379, "ecdecbb3")
    ]
    
    # 전체 평가 결과
    problem_results = {}
    
    for problem_id, arc_id in problems:
        result = tester.evaluate_problem(problem_id, arc_id)
        if result:
            problem_results[f"{problem_id}_{arc_id}"] = result
    
    # 전체 통계 계산
    total_correct = sum(r['correct_count'] for r in problem_results.values())
    total_tests = sum(r['total_count'] for r in problem_results.values())
    overall_exact_accuracy = total_correct / total_tests if total_tests > 0 else 0
    overall_pixel_accuracy = sum(r['pixel_accuracy'] * r['total_count'] for r in problem_results.values()) / total_tests if total_tests > 0 else 0
    
    # 결과 저장
    results = {
        "trained_problems": {
            "model_path": model_path,
            "evaluation_type": "trained_problems_reproduction",
            "problem_results": problem_results,
            "overall_stats": {
                "exact_accuracy": overall_exact_accuracy,
                "pixel_accuracy": overall_pixel_accuracy,
                "total_correct": total_correct,
                "total_tests": total_tests,
                "problems_evaluated": len(problem_results)
            }
        }
    }
    
    # 결과 파일 저장
    output_file = "gpu456_reproduction_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Overall accuracy: {overall_exact_accuracy:.1%} ({total_correct}/{total_tests})")
    
    return results

if __name__ == "__main__":
    main()