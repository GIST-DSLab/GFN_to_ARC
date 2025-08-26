#!/usr/bin/env python3
"""
Quick Problem-Specific Accuracy Evaluation
"""

import os
import json
import numpy as np
from inference import ARCTrajectoryInference
import yaml

def quick_problem_evaluation():
    """Quick evaluation on small subset for immediate results"""
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['model_load_path'] = os.path.join(config['model_save_dir'], "checkpoint_best.pt")
    
    # Initialize inference engine
    inference_engine = ARCTrajectoryInference(config, config['model_load_path'])
    
    # Problem metadata
    problem_metadata = {
        86: {"name": "25ff71a9", "type": "rotation", "difficulty": "medium"},
        139: {"name": "6150a2bd", "type": "transformation", "difficulty": "medium"},
        149: {"name": "67a3c6ac", "type": "symmetry", "difficulty": "easy"},
        154: {"name": "68b16354", "type": "copy", "difficulty": "easy"},
        178: {"name": "74dd1130", "type": "flip", "difficulty": "medium"},
        240: {"name": "9dfd6313", "type": "pattern", "difficulty": "hard"},
        379: {"name": "ed36ccf7", "type": "complex", "difficulty": "hard"}
    }
    
    # Test subset of problems with limited examples
    test_problems = [86, 139, 149, 154, 178]
    max_examples_per_problem = 50  # Quick test with 50 examples each
    
    results = {}
    
    print("=== Quick Problem-Specific Evaluation ===\n")
    
    for problem_id in test_problems:
        print(f"Evaluating Problem {problem_id} ({problem_metadata[problem_id]['name']})...")
        
        # Load problem data
        problems = inference_engine.load_rearc_problems([problem_id])
        if not problems:
            continue
        
        problem_data = problems[0]
        test_cases = problem_data.get('train', [])[:max_examples_per_problem]
        
        correct_count = 0
        total_count = len(test_cases)
        grid_sizes = []
        action_counts = []
        
        for i, test_case in enumerate(test_cases):
            try:
                test_input = test_case['input']
                test_output = test_case['output']
                
                # Record grid size
                grid_sizes.append(f"{len(test_input)}x{len(test_input[0])}")
                
                # Generate action sequence
                predicted_actions, confidence = inference_engine.generate_action_sequence(
                    test_input, 
                    max_actions=config.get('problem_max_actions', {}).get(str(problem_id), 20)
                )
                
                action_counts.append(len(predicted_actions))
                
                # Execute actions
                predicted_grid = inference_engine.executor.execute_action_sequence(
                    test_input, predicted_actions
                )
                
                # Check correctness
                is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
                if is_correct:
                    correct_count += 1
                
                if i % 10 == 0:
                    print(f"  Progress: {i+1}/{total_count}, Current accuracy: {correct_count/(i+1):.3f}")
                    
            except Exception as e:
                print(f"  Error on test {i}: {e}")
                continue
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        unique_grid_sizes = list(set(grid_sizes))
        avg_actions = np.mean(action_counts) if action_counts else 0
        
        results[problem_id] = {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': total_count,
            'grid_sizes': unique_grid_sizes,
            'avg_actions': avg_actions,
            'metadata': problem_metadata[problem_id]
        }
        
        print(f"  Result: {accuracy:.3f} ({correct_count}/{total_count})")
        print(f"  Grid sizes: {unique_grid_sizes}")
        print(f"  Avg actions: {avg_actions:.1f}\n")
    
    # Overall statistics
    all_correct = sum(r['correct'] for r in results.values())
    all_total = sum(r['total'] for r in results.values())
    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    
    # Difficulty analysis
    easy_problems = [pid for pid, r in results.items() if r['metadata']['difficulty'] == 'easy']
    medium_problems = [pid for pid, r in results.items() if r['metadata']['difficulty'] == 'medium']
    hard_problems = [pid for pid, r in results.items() if r['metadata']['difficulty'] == 'hard']
    
    easy_acc = np.mean([results[pid]['accuracy'] for pid in easy_problems]) if easy_problems else 0.0
    medium_acc = np.mean([results[pid]['accuracy'] for pid in medium_problems]) if medium_problems else 0.0
    hard_acc = np.mean([results[pid]['accuracy'] for pid in hard_problems]) if hard_problems else 0.0
    
    print("=== Summary ===")
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({all_correct}/{all_total})")
    print(f"Easy Problems Avg: {easy_acc:.3f}")
    print(f"Medium Problems Avg: {medium_acc:.3f}")
    print(f"Hard Problems Avg: {hard_acc:.3f}")
    
    print("\n=== Problem Rankings ===")
    sorted_problems = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for pid, result in sorted_problems:
        meta = result['metadata']
        print(f"Problem {pid} ({meta['type']}, {meta['difficulty']}): {result['accuracy']:.3f}")
    
    # Save results
    output_file = os.path.join(config['results_dir'], 'quick_evaluation_results.json')
    summary = {
        'overall_accuracy': overall_accuracy,
        'total_correct': all_correct,
        'total_tests': all_total,
        'problem_results': results,
        'difficulty_analysis': {
            'easy_avg': easy_acc,
            'medium_avg': medium_acc, 
            'hard_avg': hard_acc
        },
        'evaluation_settings': {
            'max_examples_per_problem': max_examples_per_problem,
            'test_problems': test_problems
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary

if __name__ == "__main__":
    quick_problem_evaluation()