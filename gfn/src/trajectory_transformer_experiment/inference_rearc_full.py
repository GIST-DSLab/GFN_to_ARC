#!/usr/bin/env python3
"""
Test Re-ARC evaluation with existing 3x3 model using grid downsampling
"""

import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging
import yaml
from sklearn.cluster import KMeans

from inference import ARCTrajectoryInference

class ARCReARCFullEvaluation(ARCTrajectoryInference):
    """Extended evaluation using full Re-ARC dataset with grid downsampling"""
    
    def downsample_grid_to_3x3(self, grid: List[List[int]]) -> List[List[int]]:
        """Downsample any grid to 3x3 using clustering"""
        grid_array = np.array(grid)
        rows, cols = grid_array.shape
        
        if rows == 3 and cols == 3:
            return grid
        
        # Create 3x3 result grid
        result = np.zeros((3, 3), dtype=int)
        
        # Simple averaging approach
        for i in range(3):
            for j in range(3):
                # Calculate source region
                row_start = int(i * rows / 3)
                row_end = int((i + 1) * rows / 3)
                col_start = int(j * cols / 3)
                col_end = int((j + 1) * cols / 3)
                
                # Extract region
                region = grid_array[row_start:row_end, col_start:col_end]
                
                # Find most common color in region
                if region.size > 0:
                    unique, counts = np.unique(region, return_counts=True)
                    result[i, j] = unique[np.argmax(counts)]
                else:
                    result[i, j] = 0  # Default to 0 if empty region
        
        return result.tolist()
    
    def upsample_3x3_to_original(self, grid_3x3: List[List[int]], target_shape: Tuple[int, int]) -> List[List[int]]:
        """Upsample 3x3 grid back to target shape"""
        grid_array = np.array(grid_3x3)
        target_rows, target_cols = target_shape
        
        # Create result grid
        result = np.zeros(target_shape, dtype=int)
        
        # Simple nearest neighbor upsampling
        for i in range(target_rows):
            for j in range(target_cols):
                # Map to 3x3 coordinates
                src_i = min(2, int(i * 3 / target_rows))
                src_j = min(2, int(j * 3 / target_cols))
                result[i, j] = grid_array[src_i, src_j]
        
        return result.tolist()
    
    def evaluate_single_problem(self, problem_data: Dict) -> Dict:
        """Evaluate model on a single ARC problem with full dataset"""
        problem_id = problem_data['id']
        
        # Use all examples
        test_cases = problem_data.get('train', [])
        
        # Optional: limit for faster testing
        max_test_samples = self.config.get('max_test_samples')
        if max_test_samples:
            test_cases = test_cases[:max_test_samples]
        
        results = []
        
        for test_idx, test_case in enumerate(tqdm(test_cases, desc=f"Problem {problem_id}", leave=False)):
            test_input = test_case['input']
            test_output = test_case['output']
            
            try:
                # Downsample input to 3x3
                input_3x3 = self.downsample_grid_to_3x3(test_input)
                
                # Generate action sequence using 3x3 model
                predicted_actions, confidence = self.generate_action_sequence(
                    input_3x3, 
                    max_actions=self.config.get('max_actions', 20)
                )
                
                # Execute actions on 3x3 grid
                predicted_3x3 = self.executor.execute_action_sequence(
                    input_3x3, predicted_actions
                )
                
                # Upsample back to original size
                predicted_grid = self.upsample_3x3_to_original(
                    predicted_3x3, 
                    (len(test_input), len(test_input[0]))
                )
                
                # Check correctness
                is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
                
                result = {
                    'test_idx': test_idx,
                    'predicted_actions': predicted_actions,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'num_actions': len(predicted_actions),
                    'original_size': f"{len(test_input)}x{len(test_input[0])}",
                    'downsampled_input': input_3x3,
                    'predicted_3x3': predicted_3x3
                }
                
                results.append(result)
                
                # Log progress every 100 examples
                if test_idx % 100 == 0:
                    correct_so_far = sum(1 for r in results if r['is_correct'])
                    accuracy_so_far = correct_so_far / len(results)
                    self.logger.info(
                        f"Problem {problem_id}, Test {test_idx}: "
                        f"Current accuracy: {accuracy_so_far:.3f} "
                        f"({correct_so_far}/{len(results)})"
                    )
                
            except Exception as e:
                self.logger.error(f"Error evaluating problem {problem_id}, test {test_idx}: {e}")
                result = {
                    'test_idx': test_idx,
                    'error': str(e),
                    'is_correct': False,
                    'confidence': 0.0
                }
                results.append(result)
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        accuracy = correct_count / len(results) if results else 0.0
        
        problem_result = {
            'problem_id': problem_id,
            'test_results': results,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(results)
        }
        
        return problem_result

def main():
    parser = argparse.ArgumentParser(description="Re-ARC Full Evaluation with 3x3 Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Config file path")
    parser.add_argument("--model_path", type=str, default="models/checkpoint_best.pt",
                       help="Model checkpoint path")
    parser.add_argument("--problems", type=int, nargs="+", default=[86, 139, 149, 154, 178, 240, 379],
                       help="Problem IDs to evaluate")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples per problem for testing")
    parser.add_argument("--gpu", type=int, default=5,
                       help="GPU device to use")
    
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set max samples for testing
    config['max_test_samples'] = args.max_samples
    
    # Initialize inference engine
    inference_engine = ARCReARCFullEvaluation(config, args.model_path)
    
    # Run evaluation
    results = inference_engine.run_evaluation(args.problems)
    
    print(f"\n=== Re-ARC Full Evaluation Results ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    print(f"Correct: {results['total_correct']}/{results['total_tests']}")
    
    # Print per-problem summary
    print(f"\nDetailed Results:")
    for result in results['problem_results']:
        print(f"Problem {result['problem_id']}: {result['accuracy']:.1%} ({result['correct_count']}/{result['total_count']})")

if __name__ == "__main__":
    main()