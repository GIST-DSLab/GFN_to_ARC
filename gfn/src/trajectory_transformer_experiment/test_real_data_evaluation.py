#!/usr/bin/env python3
"""
실제 GFlowNet 데이터를 사용한 Trajectory Transformer 정답/오답 구분 테스트
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
import glob

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.arc_transformer import create_model
from utils.data_utils import (
    create_vocabulary, 
    convert_trajectory_to_sequence, 
    pad_sequence, 
    create_attention_mask,
    load_gflownet_trajectories
)
from configs.arc_config import base


class RealDataEvaluationTester:
    """실제 GFlowNet 데이터를 사용한 Trajectory Transformer 평가"""
    
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
    
    def load_real_trajectories(self, data_dir: str) -> Dict[str, List[Dict]]:
        """
        실제 GFlowNet 데이터에서 성공/실패 trajectory 로드
        """
        trajectories = {
            'successful': [],
            'failed': []
        }
        
        # Find all trajectory files
        trajectory_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json') and 'trajectories' in file:
                    trajectory_files.append(os.path.join(root, file))
        
        print(f"Found {len(trajectory_files)} trajectory files")
        
        for filepath in tqdm(trajectory_files, desc="Loading trajectory files"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for traj in data:
                    if 'rewards' in traj and len(traj['rewards']) > 0:
                        final_reward = traj['rewards'][-1]
                        
                        # Convert to sequence format
                        converted = convert_trajectory_to_sequence(traj)
                        if converted:
                            if final_reward > 0:  # 성공한 trajectory
                                trajectories['successful'].append(converted)
                            else:  # 실패한 trajectory
                                trajectories['failed'].append(converted)
                        
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        print(f"Loaded {len(trajectories['successful'])} successful trajectories")
        print(f"Loaded {len(trajectories['failed'])} failed trajectories")
        
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
    
    def sample_trajectories(self, trajectories: Dict, max_samples: int = 100) -> Dict:
        """
        메모리 효율성을 위해 trajectory 샘플링
        """
        sampled = {}
        
        for category, trajs in trajectories.items():
            if len(trajs) > max_samples:
                # 다양한 길이의 trajectory를 골고루 선택
                trajs_sorted = sorted(trajs, key=lambda x: len(x['sequence']))
                indices = np.linspace(0, len(trajs_sorted)-1, max_samples, dtype=int)
                sampled[category] = [trajs_sorted[i] for i in indices]
            else:
                sampled[category] = trajs
        
        return sampled
    
    def evaluate_trajectory_discrimination(self, trajectories: Dict) -> Dict:
        """
        실제 데이터에서 성공/실패 trajectory 구분 성능 평가
        """
        successful_trajectories = trajectories['successful']
        failed_trajectories = trajectories['failed']
        
        print(f"Evaluating {len(successful_trajectories)} successful and {len(failed_trajectories)} failed trajectories")
        
        # Sample trajectories for efficiency
        sampled_trajectories = self.sample_trajectories(trajectories, max_samples=50)
        successful_trajectories = sampled_trajectories['successful']
        failed_trajectories = sampled_trajectories['failed']
        
        print(f"Sampled to {len(successful_trajectories)} successful and {len(failed_trajectories)} failed trajectories")
        
        # Calculate likelihoods
        successful_likelihoods = []
        failed_likelihoods = []
        
        print("Computing likelihoods for successful trajectories...")
        for traj in tqdm(successful_trajectories):
            likelihood = self.compute_trajectory_likelihood(traj)
            successful_likelihoods.append(likelihood)
        
        print("Computing likelihoods for failed trajectories...")
        for traj in tqdm(failed_trajectories):
            likelihood = self.compute_trajectory_likelihood(traj)
            failed_likelihoods.append(likelihood)
        
        # Statistics
        successful_mean = np.mean(successful_likelihoods)
        successful_std = np.std(successful_likelihoods)
        failed_mean = np.mean(failed_likelihoods)
        failed_std = np.std(failed_likelihoods)
        
        # Classification accuracy using likelihood threshold
        all_likelihoods = successful_likelihoods + failed_likelihoods
        all_labels = [1] * len(successful_likelihoods) + [0] * len(failed_likelihoods)
        
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
        
        # Statistical significance test (t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(successful_likelihoods, failed_likelihoods)
        
        # Convert numpy types to Python types for JSON serialization
        t_stat = float(t_stat)
        p_value = float(p_value)
        
        results = {
            'successful_trajectories': {
                'count': len(successful_trajectories),
                'mean_likelihood': successful_mean,
                'std_likelihood': successful_std,
                'likelihoods': successful_likelihoods
            },
            'failed_trajectories': {
                'count': len(failed_trajectories),
                'mean_likelihood': failed_mean,
                'std_likelihood': failed_std,
                'likelihoods': failed_likelihoods
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
                'difference_in_means': successful_mean - failed_mean,
                'pooled_std': np.sqrt(((len(successful_likelihoods) - 1) * successful_std**2 + 
                                     (len(failed_likelihoods) - 1) * failed_std**2) / 
                                    (len(successful_likelihoods) + len(failed_likelihoods) - 2)),
                'effect_size': (successful_mean - failed_mean) / np.sqrt((successful_std**2 + failed_std**2) / 2) if successful_std > 0 and failed_std > 0 else 0,
                't_statistic': t_stat,
                'p_value': p_value
            }
        }
        
        return results
    
    def analyze_trajectory_patterns(self, trajectories: Dict) -> Dict:
        """
        성공/실패 trajectory의 패턴 분석
        """
        analysis = {}
        
        for category, trajs in trajectories.items():
            if not trajs:
                continue
            
            # Length statistics
            lengths = [len(traj['sequence']) for traj in trajs]
            
            # Action statistics
            all_actions = []
            for traj in trajs:
                all_actions.extend(traj['actions'])
            
            action_counts = {}
            for action in all_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Reward statistics
            final_rewards = [traj['rewards'][-1] if traj['rewards'] else 0 for traj in trajs]
            
            analysis[category] = {
                'sequence_length': {
                    'mean': float(np.mean(lengths)),
                    'std': float(np.std(lengths)),
                    'min': int(np.min(lengths)),
                    'max': int(np.max(lengths))
                },
                'action_distribution': {k: int(v) for k, v in action_counts.items()},
                'final_reward': {
                    'mean': float(np.mean(final_rewards)),
                    'std': float(np.std(final_rewards)),
                    'min': float(np.min(final_rewards)),
                    'max': float(np.max(final_rewards))
                }
            }
        
        return analysis
    
    def run_test(self, data_dir: str) -> Dict:
        """
        전체 테스트 실행
        """
        print("=== 실제 데이터 Trajectory Transformer 평가 ===")
        
        # Load real trajectories
        trajectories = self.load_real_trajectories(data_dir)
        
        if not trajectories['successful'] or not trajectories['failed']:
            print("Error: No successful or failed trajectories found")
            return {}
        
        # Analyze trajectory patterns
        pattern_analysis = self.analyze_trajectory_patterns(trajectories)
        
        # Evaluate discrimination ability
        results = self.evaluate_trajectory_discrimination(trajectories)
        results['pattern_analysis'] = pattern_analysis
        
        # Print results
        print("\n=== 테스트 결과 ===")
        print(f"성공 trajectory 평균 likelihood: {results['successful_trajectories']['mean_likelihood']:.4f} ± {results['successful_trajectories']['std_likelihood']:.4f}")
        print(f"실패 trajectory 평균 likelihood: {results['failed_trajectories']['mean_likelihood']:.4f} ± {results['failed_trajectories']['std_likelihood']:.4f}")
        print(f"평균 차이: {results['statistical_test']['difference_in_means']:.4f}")
        print(f"효과 크기 (Cohen's d): {results['statistical_test']['effect_size']:.4f}")
        print(f"t-검정 p-value: {results['statistical_test']['p_value']:.4f}")
        
        print(f"\n분류 성능:")
        print(f"정확도: {results['classification']['accuracy']:.4f}")
        print(f"정밀도: {results['classification']['precision']:.4f}")
        print(f"재현율: {results['classification']['recall']:.4f}")
        print(f"F1 점수: {results['classification']['f1_score']:.4f}")
        print(f"최적 임계값: {results['classification']['threshold']:.4f}")
        
        print(f"\n혼동 행렬:")
        cm = results['classification']['confusion_matrix']
        print(f"  예측\\실제    성공    실패")
        print(f"  성공        {cm['tp']:4d}    {cm['fp']:4d}")
        print(f"  실패        {cm['fn']:4d}    {cm['tn']:4d}")
        
        # Pattern analysis
        print(f"\n=== 패턴 분석 ===")
        for category, analysis in pattern_analysis.items():
            print(f"{category} trajectories:")
            print(f"  시퀀스 길이: {analysis['sequence_length']['mean']:.1f} ± {analysis['sequence_length']['std']:.1f}")
            print(f"  최종 보상: {analysis['final_reward']['mean']:.3f} ± {analysis['final_reward']['std']:.3f}")
            print(f"  액션 분포: {analysis['action_distribution']}")
        
        # 해석
        print(f"\n=== 해석 ===")
        if results['statistical_test']['difference_in_means'] > 0:
            print("✓ 모델이 성공한 trajectory에 더 높은 likelihood를 할당합니다.")
        else:
            print("✗ 모델이 실패한 trajectory에 더 높은 likelihood를 할당합니다.")
        
        if results['statistical_test']['p_value'] < 0.05:
            print("✓ 성공/실패 trajectory 간의 차이가 통계적으로 유의합니다.")
        else:
            print("✗ 성공/실패 trajectory 간의 차이가 통계적으로 유의하지 않습니다.")
        
        if results['classification']['accuracy'] > 0.7:
            print("✓ 모델이 성공/실패를 잘 구분합니다.")
        elif results['classification']['accuracy'] > 0.5:
            print("△ 모델이 성공/실패를 어느 정도 구분하지만 개선이 필요합니다.")
        else:
            print("✗ 모델이 성공/실패를 구분하지 못합니다.")
        
        if abs(results['statistical_test']['effect_size']) > 0.8:
            print("✓ 성공과 실패 사이의 차이가 큽니다.")
        elif abs(results['statistical_test']['effect_size']) > 0.5:
            print("△ 성공과 실패 사이의 차이가 중간 정도입니다.")
        else:
            print("✗ 성공과 실패 사이의 차이가 작습니다.")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="실제 데이터 Trajectory Transformer 평가")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="학습된 모델 체크포인트 경로")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="GFlowNet trajectory 데이터 디렉토리")
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
            'max_sequence_length': 64,
            'batch_size': 8
        })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test
    tester = RealDataEvaluationTester(args.model_path, config)
    results = tester.run_test(args.data_dir)
    
    if results:
        # Save results
        results_file = os.path.join(args.output_dir, 'real_data_evaluation_test.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n결과가 {results_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()