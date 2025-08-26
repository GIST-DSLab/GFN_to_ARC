#!/usr/bin/env python3
"""
Test inference on 3x3 grids to verify model is working correctly
"""

import torch
import numpy as np
from inference import ARCTrajectoryInference
import yaml
import json

def test_3x3_inference():
    """Test inference on simple 3x3 grids"""
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inference engine
    model_path = 'models/checkpoint_best.pt'
    inference_engine = ARCTrajectoryInference(config, model_path)
    
    # Test cases: 3x3 grids
    test_cases = [
        {
            'name': 'All same color',
            'input': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            'expected_pattern': 'Should apply transformations to uniform grid'
        },
        {
            'name': 'Simple pattern',
            'input': [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
            'expected_pattern': 'Should apply transformations to patterned grid'
        },
        {
            'name': 'Mixed colors',
            'input': [[3, 7, 2], [5, 4, 8], [1, 6, 9]],
            'expected_pattern': 'Should apply transformations to mixed grid'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {test_case['name']} ===")
        print(f"Input grid: {test_case['input']}")
        
        try:
            # Generate actions
            actions, confidence = inference_engine.generate_action_sequence(
                test_case['input'], 
                max_actions=10
            )
            
            print(f"Generated actions: {actions}")
            print(f"Confidence: {confidence:.3f}")
            
            # Execute actions
            result_grid = inference_engine.executor.execute_action_sequence(
                test_case['input'], 
                actions
            )
            
            print(f"Result grid: {result_grid}")
            
            # Check if result is different from input
            input_array = np.array(test_case['input'])
            result_array = np.array(result_grid)
            is_changed = not np.array_equal(input_array, result_array)
            
            print(f"Grid changed: {is_changed}")
            
            result = {
                'test_case': test_case['name'],
                'input': test_case['input'],
                'actions': actions,
                'confidence': confidence,
                'result_grid': result_grid,
                'changed': is_changed,
                'num_actions': len(actions)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'test_case': test_case['name'],
                'input': test_case['input'],
                'error': str(e),
                'changed': False
            }
            results.append(result)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total test cases: {len(test_cases)}")
    
    successful_cases = [r for r in results if 'error' not in r]
    changed_cases = [r for r in results if r.get('changed', False)]
    
    print(f"Successful inferences: {len(successful_cases)}")
    print(f"Cases with grid changes: {len(changed_cases)}")
    
    if successful_cases:
        avg_confidence = np.mean([r['confidence'] for r in successful_cases])
        avg_actions = np.mean([r['num_actions'] for r in successful_cases])
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average number of actions: {avg_actions:.1f}")
    
    # Save results
    with open('results/3x3_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/3x3_test_results.json")
    
    return results

if __name__ == "__main__":
    test_3x3_inference()