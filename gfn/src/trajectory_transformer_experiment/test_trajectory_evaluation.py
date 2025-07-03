#!/usr/bin/env python3
"""
Trajectory Transformer 정답/오답 구분 테스트 스크립트
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.arc_transformer import create_model
from utils.data_utils import (
    create_vocabulary, 
    convert_trajectory_to_sequence, 
    pad_sequence, 
    create_attention_mask
)
from configs.arc_config import base


class TrajectoryEvaluationTester:
    """Trajectory Transformer의 정답/오답 구분 능력 테스트"""
    
    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.vocab = create_vocabulary()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Load model
        self.model = self.load_model(model_path)
        print(f"Model loaded on {self.device}")
        
    def load_model(self, model_path: str):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('config', self.config)
        
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def create_synthetic_trajectories(self) -> Dict[str, List[Dict]]:
        """
        정답/오답 구분 테스트를 위한 합성 trajectory 생성
        """
        trajectories = {
            'correct': [],
            'incorrect': []
        }
        
        # 정답 trajectory 패턴 (성공적인 sequence)
        correct_patterns = [
            {
                'states': [
                    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],  # 초기 상태
                    [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]],  # 회전 후
                    [[[0, 0, 1], [1, 0, 0], [0, 1, 0]]],  # 다시 회전
                    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]   # 최종 상태
                ],
                'actions': [1, 1, 4],  # right_rotate, right_rotate, submit
                'rewards': [0.0, 0.0, 1.0],
                'problem_id': 1,
                'trajectory_id': 1
            },
            {
                'states': [
                    [[[2, 3, 4], [5, 6, 7], [8, 9, 0]]],  # 초기 상태
                    [[[8, 5, 2], [9, 6, 3], [0, 7, 4]]],  # 좌회전
                    [[[2, 3, 4], [5, 6, 7], [8, 9, 0]]]   # 최종 상태
                ],
                'actions': [0, 4],  # left_rotate, submit
                'rewards': [0.0, 1.0],
                'problem_id': 2,
                'trajectory_id': 2
            }
        ]
        
        # 오답 trajectory 패턴 (실패한 sequence)
        incorrect_patterns = [
            {
                'states': [
                    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],  # 초기 상태
                    [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]],  # 회전 후
                    [[[0, 0, 1], [1, 0, 0], [0, 1, 0]]]   # 잘못된 최종 상태
                ],
                'actions': [1, 4],  # right_rotate, submit (너무 빠른 submit)
                'rewards': [0.0, 0.0],  # 실패 - 보상 없음
                'problem_id': 1,
                'trajectory_id': 3
            },
            {
                'states': [
                    [[[2, 3, 4], [5, 6, 7], [8, 9, 0]]],  # 초기 상태
                    [[[2, 3, 4], [5, 6, 7], [8, 9, 0]]],  # 변화 없음
                    [[[4, 3, 2], [7, 6, 5], [0, 9, 8]]]   # 잘못된 변환
                ],
                'actions': [2, 4],  # horizontal_flip, submit (잘못된 변환)
                'rewards': [0.0, 0.0],  # 실패 - 보상 없음
                'problem_id': 2,
                'trajectory_id': 4
            }
        ]
        
        # Convert to sequence format
        for pattern in correct_patterns:
            converted = convert_trajectory_to_sequence(pattern)
            if converted:
                trajectories['correct'].append(converted)
        
        for pattern in incorrect_patterns:
            converted = convert_trajectory_to_sequence(pattern)
            if converted:
                trajectories['incorrect'].append(converted)
        
        return trajectories
    
    def compute_trajectory_likelihood(self, trajectory: Dict) -> float:
        """
        주어진 trajectory에 대한 모델의 likelihood 계산
        """
        sequence = trajectory['sequence']
        
        # Add start token and pad sequence
        full_sequence = [self.vocab['sos']] + sequence + [self.vocab['eos']]
        max_length = self.config.get('max_sequence_length', 64)
        
        if len(full_sequence) > max_length:
            full_sequence = full_sequence[:max_length]
        
        padded_sequence = pad_sequence(full_sequence, max_length, self.vocab['pad'])
        attention_mask = create_attention_mask(padded_sequence, self.vocab['pad'])
        
        # Convert to tensors
        input_ids = torch.tensor([padded_sequence], device=self.device)
        attention_mask = torch.tensor([attention_mask], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # Calculate likelihood for each token
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get log probabilities of actual tokens
            token_log_probs = []
            for i in range(len(full_sequence) - 1):
                if i < logits.size(1) - 1:
                    current_token = full_sequence[i + 1]
                    log_prob = log_probs[0, i, current_token].item()
                    token_log_probs.append(log_prob)
            
            # Average log probability
            avg_log_prob = np.mean(token_log_probs) if token_log_probs else -float('inf')
            
            return avg_log_prob
    
    def evaluate_trajectory_discrimination(self, trajectories: Dict) -> Dict:
        """
        정답/오답 trajectory 구분 성능 평가
        """
        correct_trajectories = trajectories['correct']
        incorrect_trajectories = trajectories['incorrect']
        
        print(f"Evaluating {len(correct_trajectories)} correct and {len(incorrect_trajectories)} incorrect trajectories")
        
        # Calculate likelihoods
        correct_likelihoods = []
        incorrect_likelihoods = []
        
        print("Computing likelihoods for correct trajectories...")
        for traj in tqdm(correct_trajectories):
            likelihood = self.compute_trajectory_likelihood(traj)
            correct_likelihoods.append(likelihood)
        
        print("Computing likelihoods for incorrect trajectories...")
        for traj in tqdm(incorrect_trajectories):
            likelihood = self.compute_trajectory_likelihood(traj)
            incorrect_likelihoods.append(likelihood)
        
        # Statistics
        correct_mean = np.mean(correct_likelihoods)
        correct_std = np.std(correct_likelihoods)
        incorrect_mean = np.mean(incorrect_likelihoods)
        incorrect_std = np.std(incorrect_likelihoods)
        
        # Classification accuracy using likelihood threshold
        all_likelihoods = correct_likelihoods + incorrect_likelihoods
        all_labels = [1] * len(correct_likelihoods) + [0] * len(incorrect_likelihoods)
        
        # Find optimal threshold
        thresholds = np.linspace(min(all_likelihoods), max(all_likelihoods), 100)
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = [1 if ll >= threshold else 0 for ll in all_likelihoods]
            accuracy = np.mean([pred == label for pred, label in zip(predictions, all_labels)])
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Final evaluation with best threshold
        predictions = [1 if ll >= best_threshold else 0 for ll in all_likelihoods]
        
        # Confusion matrix
        tp = sum(1 for pred, label in zip(predictions, all_labels) if pred == 1 and label == 1)
        tn = sum(1 for pred, label in zip(predictions, all_labels) if pred == 0 and label == 0)
        fp = sum(1 for pred, label in zip(predictions, all_labels) if pred == 1 and label == 0)
        fn = sum(1 for pred, label in zip(predictions, all_labels) if pred == 0 and label == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'correct_trajectories': {
                'count': len(correct_trajectories),
                'mean_likelihood': correct_mean,
                'std_likelihood': correct_std,
                'likelihoods': correct_likelihoods
            },
            'incorrect_trajectories': {
                'count': len(incorrect_trajectories),
                'mean_likelihood': incorrect_mean,
                'std_likelihood': incorrect_std,
                'likelihoods': incorrect_likelihoods
            },
            'classification': {
                'accuracy': best_accuracy,
                'threshold': best_threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': {
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                }
            },
            'statistical_test': {
                'difference_in_means': correct_mean - incorrect_mean,
                'pooled_std': np.sqrt(((len(correct_likelihoods) - 1) * correct_std**2 + 
                                     (len(incorrect_likelihoods) - 1) * incorrect_std**2) / 
                                    (len(correct_likelihoods) + len(incorrect_likelihoods) - 2)),
                'effect_size': (correct_mean - incorrect_mean) / np.sqrt((correct_std**2 + incorrect_std**2) / 2) if correct_std > 0 and incorrect_std > 0 else 0
            }
        }
        
        return results
    
    def run_test(self) -> Dict:
        """
        전체 테스트 실행
        """
        print("=== Trajectory Transformer 정답/오답 구분 테스트 ===")
        
        # Generate synthetic trajectories
        trajectories = self.create_synthetic_trajectories()
        
        # Evaluate discrimination ability
        results = self.evaluate_trajectory_discrimination(trajectories)
        
        # Print results
        print("\n=== 테스트 결과 ===")
        print(f"정답 trajectory 평균 likelihood: {results['correct_trajectories']['mean_likelihood']:.4f} ± {results['correct_trajectories']['std_likelihood']:.4f}")
        print(f"오답 trajectory 평균 likelihood: {results['incorrect_trajectories']['mean_likelihood']:.4f} ± {results['incorrect_trajectories']['std_likelihood']:.4f}")
        print(f"평균 차이: {results['statistical_test']['difference_in_means']:.4f}")
        print(f"효과 크기 (Cohen's d): {results['statistical_test']['effect_size']:.4f}")
        
        print(f"\n분류 성능:")
        print(f"정확도: {results['classification']['accuracy']:.4f}")
        print(f"정밀도: {results['classification']['precision']:.4f}")
        print(f"재현율: {results['classification']['recall']:.4f}")
        print(f"F1 점수: {results['classification']['f1_score']:.4f}")
        print(f"최적 임계값: {results['classification']['threshold']:.4f}")
        
        print(f"\n혼동 행렬:")
        cm = results['classification']['confusion_matrix']
        print(f"  예측\\실제    정답    오답")
        print(f"  정답        {cm['tp']:4d}    {cm['fp']:4d}")
        print(f"  오답        {cm['fn']:4d}    {cm['tn']:4d}")
        
        # 해석
        print(f"\n=== 해석 ===")
        if results['statistical_test']['difference_in_means'] > 0:
            print("✓ 모델이 정답 trajectory에 더 높은 likelihood를 할당합니다.")
        else:
            print("✗ 모델이 오답 trajectory에 더 높은 likelihood를 할당합니다.")
        
        if results['classification']['accuracy'] > 0.7:
            print("✓ 모델이 정답/오답을 잘 구분합니다.")
        elif results['classification']['accuracy'] > 0.5:
            print("△ 모델이 정답/오답을 어느 정도 구분하지만 개선이 필요합니다.")
        else:
            print("✗ 모델이 정답/오답을 구분하지 못합니다.")
        
        if abs(results['statistical_test']['effect_size']) > 0.8:
            print("✓ 정답과 오답 사이의 차이가 큽니다.")
        elif abs(results['statistical_test']['effect_size']) > 0.5:
            print("△ 정답과 오답 사이의 차이가 중간 정도입니다.")
        else:
            print("✗ 정답과 오답 사이의 차이가 작습니다.")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Trajectory Transformer 정답/오답 구분 테스트")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="학습된 모델 체크포인트 경로")
    parser.add_argument("--config", type=str, default="small",
                       choices=["base", "small"], help="설정 이름")
    parser.add_argument("--output_dir", type=str, default="./test_evaluation",
                       help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "base":
        config = base['train'].copy()
    else:  # small
        config = base['train'].copy()
        config.update({
            'max_sequence_length': 32,
            'batch_size': 8
        })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test
    tester = TrajectoryEvaluationTester(args.model_path, config)
    results = tester.run_test()
    
    # Save results
    results_file = os.path.join(args.output_dir, 'trajectory_discrimination_test.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n결과가 {results_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()