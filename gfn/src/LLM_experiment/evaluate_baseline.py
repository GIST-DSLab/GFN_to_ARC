#!/usr/bin/env python3
"""
Baseline vs GFlowNet 비교 평가
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import numpy as np
from utils import (
    parse_action_sequence_from_llm,
    create_inference_prompt,
    load_config,
    load_json
)
from ARCenv.wrapper import ARCEnvWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparativeEvaluator:
    """Baseline과 GFlowNet 모델 비교 평가"""
    
    def __init__(self, baseline_model_path: str, gflownet_model_path: str, config_path: str):
        self.config = load_config(config_path)
        
        # 모델 로드
        logger.info("Loading baseline model...")
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Loading GFlowNet model...")
        self.gflownet_tokenizer = AutoTokenizer.from_pretrained(gflownet_model_path)
        self.gflownet_model = AutoModelForCausalLM.from_pretrained(
            gflownet_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 패딩 토큰 설정
        if self.baseline_tokenizer.pad_token is None:
            self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
        if self.gflownet_tokenizer.pad_token is None:
            self.gflownet_tokenizer.pad_token = self.gflownet_tokenizer.eos_token
            
        # ARC 환경 초기화
        self.env = ARCEnvWrapper()
        
    def evaluate_model(self, model, tokenizer, model_name: str, test_problems: List[Dict]) -> Dict:
        """단일 모델 평가"""
        results = {
            'model_name': model_name,
            'total_problems': 0,
            'correct_actions': 0,
            'correct_grids': 0,
            'problem_results': []
        }
        
        for problem in tqdm(test_problems, desc=f"Evaluating {model_name}"):
            problem_result = self.evaluate_single_problem(model, tokenizer, problem)
            results['problem_results'].append(problem_result)
            
            results['total_problems'] += 1
            if problem_result['action_correct']:
                results['correct_actions'] += 1
            if problem_result['grid_correct']:
                results['correct_grids'] += 1
                
        # 정확도 계산
        results['action_accuracy'] = results['correct_actions'] / results['total_problems']
        results['grid_accuracy'] = results['correct_grids'] / results['total_problems']
        
        return results
    
    def evaluate_single_problem(self, model, tokenizer, problem: Dict) -> Dict:
        """단일 문제 평가"""
        # Few-shot 프롬프트 생성
        prompt = create_inference_prompt(
            problem['train_examples'][:3],  # 3개 예제 사용
            problem['test_input'],
            use_barc_format=True
        )
        
        # 모델 추론
        inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'].to(model.device),
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 생성된 텍스트 디코딩
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Action sequence 파싱
        predicted_actions = parse_action_sequence_from_llm(generated_text)
        
        # 환경에서 action 실행
        grid_correct = False
        if predicted_actions:
            try:
                # 환경 초기화
                self.env.reset(problem['problem_id'])
                
                # Action 실행
                for action in predicted_actions[:-1]:  # submit 제외
                    if action in [0, 1, 2, 3]:  # 유효한 action
                        self.env.step(action)
                
                # 최종 그리드 비교
                final_grid = self.env.get_current_grid()
                grid_correct = np.array_equal(final_grid, problem['test_output'])
                
            except Exception as e:
                logger.error(f"Error executing actions: {e}")
        
        # Action sequence 정확도 (정답과 비교)
        action_correct = False
        if 'true_actions' in problem:
            action_correct = predicted_actions == problem['true_actions']
        
        return {
            'problem_id': problem['problem_id'],
            'predicted_actions': predicted_actions,
            'generated_text': generated_text,
            'action_correct': action_correct,
            'grid_correct': grid_correct
        }
    
    def run_comparison(self, test_data_path: str) -> Dict:
        """비교 평가 실행"""
        # 테스트 데이터 로드
        test_problems = self.load_test_problems(test_data_path)
        
        # Baseline 평가
        logger.info("Evaluating baseline model...")
        baseline_results = self.evaluate_model(
            self.baseline_model, 
            self.baseline_tokenizer,
            "Baseline (Repeated Solutions)",
            test_problems
        )
        
        # GFlowNet 평가
        logger.info("Evaluating GFlowNet model...")
        gflownet_results = self.evaluate_model(
            self.gflownet_model,
            self.gflownet_tokenizer,
            "GFlowNet (Diverse Trajectories)",
            test_problems
        )
        
        # 비교 결과
        comparison = {
            'baseline': baseline_results,
            'gflownet': gflownet_results,
            'comparison_summary': {
                'baseline_action_accuracy': baseline_results['action_accuracy'],
                'gflownet_action_accuracy': gflownet_results['action_accuracy'],
                'baseline_grid_accuracy': baseline_results['grid_accuracy'],
                'gflownet_grid_accuracy': gflownet_results['grid_accuracy'],
                'improvement_action': gflownet_results['action_accuracy'] - baseline_results['action_accuracy'],
                'improvement_grid': gflownet_results['grid_accuracy'] - baseline_results['grid_accuracy']
            }
        }
        
        return comparison
    
    def load_test_problems(self, test_data_path: str) -> List[Dict]:
        """테스트 문제 로드"""
        # 실제 구현에서는 ARC evaluation set 사용
        # 여기서는 예시로 간단한 구조 사용
        test_problems = []
        
        # 훈련에 사용된 문제들의 테스트 예제
        problem_ids = [86, 139, 178, 149, 154, 240, 379]
        
        for problem_id in problem_ids:
            # 실제로는 ARC 데이터에서 로드
            test_problems.append({
                'problem_id': problem_id,
                'train_examples': [],  # Few-shot examples
                'test_input': [],  # Test input grid
                'test_output': [],  # Expected output
                'true_actions': []  # Ground truth actions (if available)
            })
            
        return test_problems

def main():
    # 경로 설정
    baseline_model_path = "./models/baseline_model/final"
    gflownet_model_path = "./models/gflownet_model/final"  # 실제 GFlowNet 모델 경로
    config_path = "./configs/config.yaml"
    test_data_path = "./data/test_problems.json"
    
    # 평가자 초기화
    evaluator = ComparativeEvaluator(
        baseline_model_path,
        gflownet_model_path,
        config_path
    )
    
    # 비교 평가 실행
    results = evaluator.run_comparison(test_data_path)
    
    # 결과 저장
    output_path = "./results/baseline_vs_gflownet_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 결과 출력
    print("\n=== Comparison Results ===")
    print(f"Baseline Action Accuracy: {results['comparison_summary']['baseline_action_accuracy']:.2%}")
    print(f"GFlowNet Action Accuracy: {results['comparison_summary']['gflownet_action_accuracy']:.2%}")
    print(f"Improvement: {results['comparison_summary']['improvement_action']:.2%}")
    print(f"\nBaseline Grid Accuracy: {results['comparison_summary']['baseline_grid_accuracy']:.2%}")
    print(f"GFlowNet Grid Accuracy: {results['comparison_summary']['gflownet_grid_accuracy']:.2%}")
    print(f"Improvement: {results['comparison_summary']['improvement_grid']:.2%}")
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()