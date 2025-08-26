#!/usr/bin/env python3
"""
Detailed ARC Problem-Specific Accuracy Evaluation
"""

import os
import json
import torch
import numpy as np
import argparse
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from inference import ARCTrajectoryInference
import yaml


class DetailedARCEvaluator:
    """Detailed evaluation for ARC problems with problem-specific accuracy analysis"""
    
    def __init__(self, config, model_path: str):
        self.config = config
        self.inference_engine = ARCTrajectoryInference(config, model_path)
        self.logger = self.inference_engine.logger
        
        # Problem metadata
        self.problem_metadata = {
            86: {"name": "25ff71a9", "type": "rotation", "difficulty": "medium"},
            128: {"name": "5582e5ca", "type": "pattern", "difficulty": "hard"}, 
            139: {"name": "6150a2bd", "type": "transformation", "difficulty": "medium"},
            149: {"name": "67a3c6ac", "type": "symmetry", "difficulty": "easy"},
            154: {"name": "68b16354", "type": "copy", "difficulty": "easy"},
            178: {"name": "74dd1130", "type": "flip", "difficulty": "medium"},
            240: {"name": "9dfd6313", "type": "pattern", "difficulty": "hard"},
            379: {"name": "ed36ccf7", "type": "complex", "difficulty": "hard"}
        }
    
    def load_original_arc_problems(self, problem_ids: List[int]) -> List[Dict]:
        """Load original ARC training problems if available"""
        # This would require the original ARC dataset
        # For now, return empty list since we're focusing on Re-ARC
        return []
    
    def evaluate_problem_with_full_analysis(self, problem_data: Dict) -> Dict:
        """Evaluate a single problem with detailed analysis"""
        problem_id = problem_data['id']
        test_cases = problem_data.get('test', [])
        if not test_cases:
            # Use all Re-ARC examples for evaluation
            test_cases = problem_data.get('train', [])
        
        self.logger.info(f"Evaluating problem {problem_id} with {len(test_cases)} test cases")
        
        results = []
        grid_size_stats = []
        action_pattern_stats = []
        
        for test_idx, test_case in enumerate(tqdm(test_cases, desc=f"Problem {problem_id}", leave=False)):
            test_input = test_case['input']
            test_output = test_case['output']
            
            # Record grid statistics
            input_grid = np.array(test_input)
            output_grid = np.array(test_output)
            grid_size_stats.append({
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'size_match': input_grid.shape == output_grid.shape
            })
            
            try:
                # Generate action sequence
                predicted_actions, confidence = self.inference_engine.generate_action_sequence(
                    test_input, 
                    max_actions=self.config.get('problem_max_actions', {}).get(str(problem_id), 20)
                )
                
                # Execute actions on the grid
                predicted_grid = self.inference_engine.executor.execute_action_sequence(
                    test_input, predicted_actions
                )
                
                # Check correctness
                is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
                
                # Analyze action patterns
                action_pattern_stats.append({
                    'actions': predicted_actions,
                    'num_actions': len(predicted_actions),
                    'action_types': list(set(predicted_actions)),
                    'has_submit': 4 in predicted_actions
                })
                
                result = {
                    'test_idx': test_idx,
                    'predicted_actions': predicted_actions,
                    'predicted_grid': predicted_grid,
                    'expected_grid': test_output,
                    'input_grid': test_input,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'num_actions': len(predicted_actions),
                    'grid_size_changed': input_grid.shape != np.array(predicted_grid).shape
                }
                
                results.append(result)
                
                if test_idx % 100 == 0:
                    current_accuracy = sum(1 for r in results if r['is_correct']) / len(results)
                    self.logger.info(f"Problem {problem_id}, Test {test_idx}: Current accuracy: {current_accuracy:.3f} ({sum(1 for r in results if r['is_correct'])}/{len(results)})")
                
            except Exception as e:
                self.logger.error(f"Error evaluating problem {problem_id}, test {test_idx}: {e}")
                result = {
                    'test_idx': test_idx,
                    'error': str(e),
                    'is_correct': False,
                    'confidence': 0.0,
                    'predicted_actions': [],
                    'num_actions': 0
                }
                results.append(result)
        
        # Calculate detailed statistics
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        accuracy = correct_count / len(results) if results else 0.0
        
        # Grid size analysis
        unique_input_shapes = list(set(stat['input_shape'] for stat in grid_size_stats))
        unique_output_shapes = list(set(stat['output_shape'] for stat in grid_size_stats))
        size_consistency = all(stat['size_match'] for stat in grid_size_stats)
        
        # Action pattern analysis
        avg_actions = np.mean([stat['num_actions'] for stat in action_pattern_stats]) if action_pattern_stats else 0
        most_common_actions = {}
        for stat in action_pattern_stats:
            for action in stat['actions']:
                most_common_actions[action] = most_common_actions.get(action, 0) + 1
        
        problem_result = {
            'problem_id': problem_id,
            'problem_metadata': self.problem_metadata.get(problem_id, {}),
            'test_results': results,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(results),
            'grid_analysis': {
                'unique_input_shapes': unique_input_shapes,
                'unique_output_shapes': unique_output_shapes,
                'size_consistency': size_consistency,
                'avg_input_size': np.mean([np.prod(stat['input_shape']) for stat in grid_size_stats]) if grid_size_stats else 0
            },
            'action_analysis': {
                'avg_actions_per_test': avg_actions,
                'most_common_actions': most_common_actions,
                'submit_rate': sum(1 for stat in action_pattern_stats if stat['has_submit']) / len(action_pattern_stats) if action_pattern_stats else 0
            }
        }
        
        return problem_result
    
    def run_detailed_evaluation(self, problem_ids: List[int] = None) -> Dict:
        """Run detailed evaluation with problem-specific analysis"""
        if problem_ids is None:
            problem_ids = self.config.get('eval_problems', [86, 139, 149, 154, 178, 240, 379])
        
        self.logger.info(f"Starting detailed evaluation on problems: {problem_ids}")
        
        # Load ReARC problems
        problems = self.inference_engine.load_rearc_problems(problem_ids)
        if not problems:
            raise ValueError("No problems loaded for evaluation")
        
        # Evaluate each problem with detailed analysis
        all_results = []
        total_correct = 0
        total_tests = 0
        problem_accuracies = {}
        
        for problem in tqdm(problems, desc="Evaluating problems", unit="problem"):
            problem_result = self.evaluate_problem_with_full_analysis(problem)
            all_results.append(problem_result)
            
            total_correct += problem_result['correct_count']
            total_tests += problem_result['total_count']
            problem_accuracies[problem_result['problem_id']] = problem_result['accuracy']
        
        # Calculate overall statistics
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        # Problem difficulty analysis
        easy_problems = [pid for pid in problem_ids if self.problem_metadata.get(pid, {}).get('difficulty') == 'easy']
        medium_problems = [pid for pid in problem_ids if self.problem_metadata.get(pid, {}).get('difficulty') == 'medium']
        hard_problems = [pid for pid in problem_ids if self.problem_metadata.get(pid, {}).get('difficulty') == 'hard']
        
        difficulty_analysis = {
            'easy_avg_accuracy': np.mean([problem_accuracies[pid] for pid in easy_problems]) if easy_problems else 0.0,
            'medium_avg_accuracy': np.mean([problem_accuracies[pid] for pid in medium_problems]) if medium_problems else 0.0,
            'hard_avg_accuracy': np.mean([problem_accuracies[pid] for pid in hard_problems]) if hard_problems else 0.0,
            'easy_problems': easy_problems,
            'medium_problems': medium_problems,
            'hard_problems': hard_problems
        }
        
        # Create comprehensive evaluation summary
        evaluation_summary = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_tests': total_tests,
            'problem_results': all_results,
            'problem_accuracies': problem_accuracies,
            'difficulty_analysis': difficulty_analysis,
            'model_config': self.config,
            'evaluation_problems': problem_ids,
            'evaluation_type': 'rearc_detailed'
        }
        
        # Save detailed results
        results_file = os.path.join(self.config['results_dir'], 'detailed_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(evaluation_summary)
        
        return evaluation_summary
    
    def generate_summary_report(self, evaluation_summary: Dict):
        """Generate a detailed summary report"""
        report_file = os.path.join(self.config['results_dir'], 'evaluation_summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=== ARC Trajectory Transformer Detailed Evaluation Report ===\n\n")
            
            # Overall Results
            f.write(f"Overall Accuracy: {evaluation_summary['overall_accuracy']:.3f}\n")
            f.write(f"Total Correct: {evaluation_summary['total_correct']}\n")
            f.write(f"Total Tests: {evaluation_summary['total_tests']}\n\n")
            
            # Problem-by-Problem Results
            f.write("=== Problem-Specific Results ===\n")
            for result in evaluation_summary['problem_results']:
                pid = result['problem_id']
                metadata = result['problem_metadata']
                f.write(f"\nProblem {pid} ({metadata.get('name', 'unknown')}):\n")
                f.write(f"  Type: {metadata.get('type', 'unknown')}\n")
                f.write(f"  Difficulty: {metadata.get('difficulty', 'unknown')}\n")
                f.write(f"  Accuracy: {result['accuracy']:.3f} ({result['correct_count']}/{result['total_count']})\n")
                f.write(f"  Avg Actions: {result['action_analysis']['avg_actions_per_test']:.1f}\n")
                f.write(f"  Submit Rate: {result['action_analysis']['submit_rate']:.3f}\n")
                f.write(f"  Grid Shapes: {result['grid_analysis']['unique_input_shapes']}\n")
            
            # Difficulty Analysis
            f.write(f"\n=== Difficulty Analysis ===\n")
            diff_analysis = evaluation_summary['difficulty_analysis']
            f.write(f"Easy Problems Avg Accuracy: {diff_analysis['easy_avg_accuracy']:.3f}\n")
            f.write(f"Medium Problems Avg Accuracy: {diff_analysis['medium_avg_accuracy']:.3f}\n")
            f.write(f"Hard Problems Avg Accuracy: {diff_analysis['hard_avg_accuracy']:.3f}\n")
            
            # Top and Bottom Performing Problems
            f.write(f"\n=== Performance Rankings ===\n")
            sorted_problems = sorted(evaluation_summary['problem_accuracies'].items(), 
                                   key=lambda x: x[1], reverse=True)
            f.write("Best Performing Problems:\n")
            for pid, acc in sorted_problems[:3]:
                f.write(f"  Problem {pid}: {acc:.3f}\n")
            f.write("Worst Performing Problems:\n")
            for pid, acc in sorted_problems[-3:]:
                f.write(f"  Problem {pid}: {acc:.3f}\n")
        
        self.logger.info(f"Summary report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Detailed ARC Problem-Specific Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--problems", type=int, nargs="+", default=None,
                       help="Problem IDs to evaluate (default: all)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.model_path:
        config['model_load_path'] = args.model_path
    else:
        config['model_load_path'] = os.path.join(config['model_save_dir'], "checkpoint_best.pt")
    
    # Initialize detailed evaluator
    evaluator = DetailedARCEvaluator(config, config['model_load_path'])
    
    try:
        # Run detailed evaluation
        results = evaluator.run_detailed_evaluation(args.problems)
        
        print(f"\n=== Detailed Evaluation Summary ===")
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Total Correct: {results['total_correct']}/{results['total_tests']}")
        
        # Print problem-specific results
        print(f"\n=== Problem-Specific Accuracies ===")
        for pid, acc in results['problem_accuracies'].items():
            metadata = evaluator.problem_metadata.get(pid, {})
            print(f"Problem {pid} ({metadata.get('type', 'unknown')}): {acc:.3f}")
        
        # Print difficulty analysis
        diff_analysis = results['difficulty_analysis']
        print(f"\n=== Difficulty Analysis ===")
        print(f"Easy Problems: {diff_analysis['easy_avg_accuracy']:.3f}")
        print(f"Medium Problems: {diff_analysis['medium_avg_accuracy']:.3f}")
        print(f"Hard Problems: {diff_analysis['hard_avg_accuracy']:.3f}")
        
        print(f"\nDetailed results saved to: {config['results_dir']}/detailed_evaluation_results.json")
        print(f"Summary report saved to: {config['results_dir']}/evaluation_summary_report.txt")
        
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()