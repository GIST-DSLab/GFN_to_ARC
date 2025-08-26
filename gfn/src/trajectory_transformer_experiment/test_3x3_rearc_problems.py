#!/usr/bin/env python3
"""
3x3 ReARC 문제에 대한 Trajectory Transformer 정확도 테스트
가장 적합한 10개의 3x3 문제를 선별하여 평가
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm

from models.arc_transformer import ARCTrajectoryTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARC3x3Evaluator:
    """3x3 ARC 문제 전용 평가기"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        
        # 모델 로드
        self.model = self._load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # 선별된 10개 3x3 문제 (순수 3x3 + 높은 예제 수)
        self.test_problems = [
            "794b24be",  # 10 training + 2 test (최대)
            "6ea4a07e",  # 6 training + 2 test (evaluation)
            "0d3d703e",  # 4 training + 1 test
            "25d8a9c8",  # 4 training + 1 test
            "25ff71a9",  # 4 training + 2 test
            "3c9b0459",  # 4 training + 1 test
            "ed36ccf7",  # 4 training + 1 test
            "67a3c6ac",  # training set
            "74dd1130",  # training set
            "9dfd6313"   # training set (Problem 240)
        ]
        
    def _load_model(self, model_path: str) -> ARCTrajectoryTransformer:
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # 설정에서 모델 생성
        config = {
            'n_layer': 6, 'n_head': 8, 'n_embd': 128,
            'observation_dim': 9, 'action_dim': 1, 'reward_dim': 1,
            'vocab_size': 26, 'max_sequence_length': 128
        }
        
        model = ARCTrajectoryTransformer(config)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        return model
        
    def load_problem_data(self, problem_id: str) -> Dict[str, Any]:
        """ReARC 문제 데이터 로드"""
        # 통합된 tasks 디렉토리에서 찾기
        data_path = f"../LLM_experiment/data/re-arc/re_arc_extracted/re_arc/tasks/{problem_id}.json"
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                return json.load(f)
                
        raise FileNotFoundError(f"Problem {problem_id} not found")
        
    def filter_3x3_examples(self, problem_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """3x3 예제만 필터링"""
        filtered_examples = []
        
        # ReARC 데이터는 리스트 형태
        for i, example in enumerate(problem_data):
            input_grid = example['input']
            output_grid = example['output']
            
            if (len(input_grid) == 3 and len(input_grid[0]) == 3 and
                len(output_grid) == 3 and len(output_grid[0]) == 3):
                filtered_examples.append({
                    'input': input_grid,
                    'output': output_grid,
                    'type': f'example_{i}'
                })
                
        return filtered_examples
        
    def encode_grid(self, grid: List[List[int]]) -> List[int]:
        """3x3 그리드를 9개 토큰으로 인코딩"""
        tokens = []
        for row in grid:
            for cell in row:
                tokens.append(cell)
        return tokens
        
    def generate_actions(self, input_grid: List[List[int]], max_actions: int = 10) -> List[int]:
        """입력 그리드에서 액션 시퀀스 생성"""
        input_tokens = self.encode_grid(input_grid)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 액션 시퀀스 생성
            generated = self.model.generate(
                input_tensor,
                max_new_tokens=max_actions * 4,  # obs + action + reward + value
                temperature=1.0,
                top_p=0.9
            )
            
            # 액션 토큰 추출 (11-15 범위)
            actions = []
            generated_tokens = generated[0].cpu().numpy()
            
            for i, token in enumerate(generated_tokens[len(input_tokens):]):
                if 11 <= token <= 15:  # 액션 토큰
                    action = token - 11  # 0-4로 변환
                    actions.append(action)
                    if action == 4:  # submit 액션
                        break
                        
        return actions
        
    def execute_actions(self, grid: List[List[int]], actions: List[int]) -> List[List[int]]:
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
        
    def evaluate_problem(self, problem_id: str) -> Dict[str, Any]:
        """단일 문제 평가"""
        logger.info(f"Evaluating problem {problem_id}")
        
        try:
            # 문제 데이터 로드
            problem_data = self.load_problem_data(problem_id)
            examples = self.filter_3x3_examples(problem_data)
            
            if not examples:
                return {
                    'problem_id': problem_id,
                    'error': 'No 3x3 examples found',
                    'accuracy': 0.0,
                    'total_examples': 0,
                    'correct_examples': 0
                }
                
            results = []
            correct_count = 0
            
            for i, example in enumerate(examples):
                # 액션 생성
                actions = self.generate_actions(example['input'])
                
                # 액션 실행
                predicted_output = self.execute_actions(example['input'], actions)
                
                # 정확도 확인
                is_correct = predicted_output == example['output']
                if is_correct:
                    correct_count += 1
                    
                results.append({
                    'example_index': i,
                    'type': example['type'],
                    'input': example['input'],
                    'expected_output': example['output'],
                    'predicted_output': predicted_output,
                    'actions': actions,
                    'correct': is_correct
                })
                
            accuracy = correct_count / len(examples) if examples else 0.0
            
            return {
                'problem_id': problem_id,
                'accuracy': accuracy,
                'total_examples': len(examples),
                'correct_examples': correct_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating problem {problem_id}: {e}")
            return {
                'problem_id': problem_id,
                'error': str(e),
                'accuracy': 0.0,
                'total_examples': 0,
                'correct_examples': 0
            }
            
    def run_evaluation(self) -> Dict[str, Any]:
        """전체 평가 실행"""
        logger.info("Starting 3x3 ReARC evaluation")
        
        all_results = []
        total_correct = 0
        total_examples = 0
        
        for problem_id in tqdm(self.test_problems, desc="Evaluating problems"):
            result = self.evaluate_problem(problem_id)
            all_results.append(result)
            
            if 'error' not in result:
                total_correct += result['correct_examples']
                total_examples += result['total_examples']
                
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        
        summary = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_examples': total_examples,
            'problems_evaluated': len([r for r in all_results if 'error' not in r]),
            'problems_with_errors': len([r for r in all_results if 'error' in r]),
            'problem_results': all_results
        }
        
        return summary

def main():
    """메인 실행 함수"""
    # 모델 경로
    model_path = "./models/checkpoint_best.pt"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
        
    # 평가 실행
    evaluator = ARC3x3Evaluator(model_path)
    results = evaluator.run_evaluation()
    
    # 결과 저장
    output_file = "./results/3x3_rearc_evaluation.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # 결과 출력
    logger.info("=" * 50)
    logger.info("3x3 ReARC Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    logger.info(f"Total Correct: {results['total_correct']}/{results['total_examples']}")
    logger.info(f"Problems Evaluated: {results['problems_evaluated']}")
    logger.info(f"Problems with Errors: {results['problems_with_errors']}")
    
    # 문제별 결과
    logger.info("\nPer-Problem Results:")
    for result in results['problem_results']:
        if 'error' not in result:
            logger.info(f"  {result['problem_id']}: {result['accuracy']:.1%} "
                       f"({result['correct_examples']}/{result['total_examples']})")
        else:
            logger.info(f"  {result['problem_id']}: ERROR - {result['error']}")
            
    logger.info(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()