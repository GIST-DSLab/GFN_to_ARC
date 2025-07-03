#!/usr/bin/env python3
"""
추론 및 평가: 학습된 모델로 ReARC 데이터셋에서 성능 평가
"""

import os
import json
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Any
import sys
import copy
import argparse

# 현재 디렉토리의 상위 디렉토리들을 path에 추가
sys.path.append('/home/ubuntu/GFN_to_ARC/gfn/src')
sys.path.append('/home/ubuntu/GFN_to_ARC/gfn/src/ARCenv')

from utils import *
import logging
from tqdm import tqdm

# ARCenv 임포트
from ARCenv.EntireARCEnv import DiagonalARCEnv

class ARCActionExecutor:
    """ARC 액션 실행기 - 실제로 액션을 적용하여 결과 확인"""
    
    def __init__(self):
        # ARC 환경 초기화
        self.env = DiagonalARCEnv()
        
    def execute_action_sequence(self, 
                               initial_grid: List[List[int]], 
                               actions: List[int]) -> List[List[int]]:
        """액션 시퀀스를 실행하여 최종 그리드 반환"""
        try:
            # 환경 초기화
            # 더미 옵션으로 환경 리셋 (실제 문제 ID는 중요하지 않음)
            state, info = self.env.reset(options={
                "prob_index": 178,  # 더미 문제 ID
                "adaptation": True,
                "subprob_index": 0
            })
            
            # 초기 그리드를 환경에 설정
            # 이 부분은 환경의 구체적인 구현에 따라 달라질 수 있음
            current_grid = copy.deepcopy(initial_grid)
            
            # 각 액션 실행
            for action in actions:
                if action == 4:  # submit
                    break
                    
                # GFlowNet action을 ARC action으로 변환
                arc_action = map_gflownet_action_to_arc_action(action)
                
                # 액션 적용 (직접 그리드 변환)
                current_grid = self.apply_action_to_grid(current_grid, action)
            
            return current_grid
            
        except Exception as e:
            logging.error(f"Error executing action sequence: {e}")
            return initial_grid  # 실패시 원본 반환
    
    def apply_action_to_grid(self, grid: List[List[int]], action: int) -> List[List[int]]:
        """그리드에 액션 직접 적용"""
        grid_array = np.array(grid)
        
        if action == 0:  # left rotate (90도 반시계 방향)
            result = np.rot90(grid_array, k=1)
        elif action == 1:  # right rotate (90도 시계 방향)
            result = np.rot90(grid_array, k=-1)
        elif action == 2:  # horizontal flip
            result = np.fliplr(grid_array)
        elif action == 3:  # vertical flip
            result = np.flipud(grid_array)
        else:
            result = grid_array  # 다른 액션은 변화 없음
            
        return result.tolist()

class ARCInferenceEvaluator:
    """ARC 추론 및 평가기"""
    
    def __init__(self, config: Dict, model_path: str, gpu_ids: List[int] = None):
        self.config = config
        self.gpu_ids = gpu_ids
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {self.num_gpus}")
            
            if gpu_ids:
                # 특정 GPU 사용
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
                self.logger.info(f"Using specified GPUs: {gpu_ids}")
            else:
                self.device = torch.device("cuda")
        else:
            self.num_gpus = 0
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # GPU 병렬화 설정
        if torch.cuda.is_available() and self.num_gpus > 1:
            if gpu_ids and len(gpu_ids) > 1:
                # 특정 GPU들로 DataParallel
                self.model = DataParallel(self.model, device_ids=gpu_ids)
                self.logger.info(f"Using DataParallel on GPUs: {gpu_ids}")
            elif not gpu_ids:
                # 모든 GPU 사용
                self.model = DataParallel(self.model)
                self.logger.info(f"Using DataParallel on all {self.num_gpus} GPUs")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 액션 실행기
        self.executor = ARCActionExecutor()
        
        self.logger.info(f"Loaded model from {model_path}")
    
    def load_rearc_data(self) -> Dict[str, Dict]:
        """ReARC 데이터 로드"""
        rearc_data = {}
        
        # 학습에 사용된 문제들의 ARC ID
        for problem_id, arc_id in self.config['problem_mapping'].items():
            # 우선순위로 여러 경로 시도
            possible_paths = [
                # re_arc_extracted에서 먼저 시도
                os.path.join(self.config['rearc_data_dir'], "re_arc_extracted", "re_arc", "tasks", f"{arc_id}.json"),
                # 기존 arc_original 경로
                os.path.join(self.config['rearc_data_dir'], "arc_original", "training", f"{arc_id}.json"),
                # 다른 가능한 경로들
                os.path.join(self.config['rearc_data_dir'], "arc_original", "evaluation", f"{arc_id}.json")
            ]
            
            arc_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    arc_file = path
                    break
            
            if arc_file:
                with open(arc_file, 'r') as f:
                    data = json.load(f)
                rearc_data[arc_id] = data
                self.logger.info(f"Loaded ReARC data for {arc_id} from {arc_file}")
            else:
                self.logger.warning(f"ReARC file not found for {arc_id}. Tried paths: {possible_paths}")
        
        return rearc_data
    
    def generate_action_sequence(self, 
                                input_grid: List[List[int]], 
                                train_examples: List[Dict] = None,
                                max_new_tokens: int = 100) -> List[int]:
        """입력 그리드에 대해 액션 시퀀스 생성 (output은 숨김)"""
        
        # 프롬프트 생성 (BARC 형식 사용, few-shot examples 포함, test output 없음)
        prompt = create_inference_prompt(input_grid, train_examples, use_barc_format=True)
        
        # 토큰화
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt',
            max_length=self.config['max_length'],
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.1,  # 낮은 temperature로 더 확정적인 출력
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = generated_text[len(prompt):].strip()
        
        # 액션 시퀀스 파싱
        actions = parse_action_sequence_from_llm(response)
        
        return actions, response
    
    def evaluate_single_problem(self, arc_id: str, problem_data: Dict) -> Dict:
        """단일 문제에 대한 평가"""
        results = {
            'arc_id': arc_id,
            'train_examples': [],
            'test_results': []
        }
        
        # re-arc tasks 형식 vs ARC 원본 형식 구분
        if isinstance(problem_data, list):
            # re-arc tasks 형식: 1000개 예제 리스트
            import random
            
            # 전체 데이터에서 샘플링
            all_examples = problem_data
            random.shuffle(all_examples)
            
            # few-shot용 train examples (5개)
            train_examples = all_examples[:5]
            
            # 평가용 test examples (최대 20개)
            test_examples = all_examples[5:25] if len(all_examples) > 25 else all_examples[5:]
            
            self.logger.info(f"Evaluating {arc_id} (re-arc format): {len(train_examples)} train, {len(test_examples)} test from {len(all_examples)} total examples")
            
        else:
            # ARC 원본 형식: train/test 구조
            train_examples = problem_data.get('train', [])
            test_examples = problem_data.get('test', [])
            
            self.logger.info(f"Evaluating {arc_id} (ARC format): {len(train_examples)} train, {len(test_examples)} test")
        
        # 각 테스트 예제에 대해 평가
        for test_idx, test_example in enumerate(tqdm(test_examples, desc=f"Testing {arc_id}")):
            test_input = test_example['input']
            test_output = test_example['output']
            
            # 액션 시퀀스 생성 (test input만 제공, output은 숨김)
            predicted_actions, raw_response = self.generate_action_sequence(
                test_input, train_examples
            )
            
            # 액션 시퀀스 실행
            predicted_grid = self.executor.execute_action_sequence(
                test_input, predicted_actions
            )
            
            # 정확도 계산
            is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
            
            test_result = {
                'test_idx': test_idx,
                'input_grid': test_input,
                'target_output': test_output,
                'predicted_actions': predicted_actions,
                'predicted_grid': predicted_grid,
                'is_correct': is_correct,
                'raw_response': raw_response
            }
            
            results['test_results'].append(test_result)
            
            self.logger.info(f"Test {test_idx}: {'✓' if is_correct else '✗'} Actions: {predicted_actions}")
        
        # 정확도 계산
        correct_count = sum(1 for r in results['test_results'] if r['is_correct'])
        total_count = len(results['test_results'])
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        results['accuracy'] = accuracy
        results['correct_count'] = correct_count
        results['total_count'] = total_count
        
        return results
    
    def evaluate_all_problems(self) -> Dict:
        """모든 문제에 대한 평가"""
        # ReARC 데이터 로드
        rearc_data = self.load_rearc_data()
        
        all_results = {
            'model_path': self.model.config.name_or_path,
            'config': self.config,
            'problem_results': {},
            'overall_stats': {}
        }
        
        total_correct = 0
        total_tests = 0
        
        # 각 문제별 평가
        for arc_id, problem_data in tqdm(rearc_data.items(), desc="Evaluating problems"):
            self.logger.info(f"\\nEvaluating problem {arc_id}")
            
            problem_results = self.evaluate_single_problem(arc_id, problem_data)
            all_results['problem_results'][arc_id] = problem_results
            
            total_correct += problem_results['correct_count']
            total_tests += problem_results['total_count']
            
            self.logger.info(f"Problem {arc_id}: {problem_results['accuracy']:.3f} "
                           f"({problem_results['correct_count']}/{problem_results['total_count']})")
        
        # 전체 통계
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        all_results['overall_stats'] = {
            'total_correct': total_correct,
            'total_tests': total_tests,
            'overall_accuracy': overall_accuracy,
            'num_problems': len(rearc_data)
        }
        
        self.logger.info(f"\\n=== FINAL RESULTS ===")
        self.logger.info(f"Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_tests})")
        self.logger.info(f"Problems Evaluated: {len(rearc_data)}")
        
        return all_results
    
    def save_results(self, results: Dict, output_file: str):
        """평가 결과 저장"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_json(results, output_file)
        self.logger.info(f"Results saved to {output_file}")

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="ARC Action Sequence LLM Inference")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=None,
                       help="Specific GPU IDs to use (e.g., --gpu_ids 0 1 2)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model (auto-detected if not provided)")
    parser.add_argument("--download_data", action="store_true",
                       help="Download re-arc data before inference")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 데이터 다운로드 (필요한 경우)
    if args.download_data:
        print("Downloading data...")
        import subprocess
        result = subprocess.run([sys.executable, "download_data.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Data download failed: {result.stderr}")
            return
        print("Data download completed!")
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    log_dir = config.get('results_dir', './results')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inference.log")
    logger = setup_logging(log_file)
    
    # GPU 설정 정보 출력
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if args.gpu_ids:
            logger.info(f"Using specific GPUs: {args.gpu_ids}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
        else:
            logger.info("Using all available GPUs")
    else:
        logger.info("CUDA not available, using CPU")
    
    # 모델 경로 확인
    if args.model_path:
        model_path = args.model_path
    else:
        model_name_safe = config['model_name'].split('/')[-1].replace('.', '_')
        model_path = os.path.join(config.get('model_save_dir', './models'), f"arc_action_model_{model_name_safe}", "final_model")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run training first.")
        return
    
    # 평가기 초기화 및 실행
    evaluator = ARCInferenceEvaluator(config, model_path, gpu_ids=args.gpu_ids)
    
    logger.info("Starting evaluation on ReARC dataset...")
    results = evaluator.evaluate_all_problems()
    
    # 결과 저장
    results_dir = config.get('results_dir', './results')
    results_file = os.path.join(results_dir, "evaluation_results.json")
    evaluator.save_results(results, results_file)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()