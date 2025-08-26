#!/usr/bin/env python3
"""
Script to find ARC problems with 3x3 input/output grids in the ReARC dataset.
"""

import json
import os
from typing import Dict, List, Tuple, Any

def analyze_grid_dimensions(grid: List[List[int]]) -> Tuple[int, int]:
    """Return (height, width) of a grid."""
    if not grid:
        return 0, 0
    return len(grid), len(grid[0]) if grid[0] else 0

def analyze_problem(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single problem and return its grid dimensions."""
    result = {
        'train_examples': [],
        'test_examples': [],
        'has_3x3': False
    }
    
    # Analyze training examples
    for i, example in enumerate(problem_data.get('train', [])):
        input_dims = analyze_grid_dimensions(example['input'])
        output_dims = analyze_grid_dimensions(example['output'])
        
        example_info = {
            'example_num': i,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'is_3x3': input_dims == (3, 3) and output_dims == (3, 3)
        }
        result['train_examples'].append(example_info)
        
        if example_info['is_3x3']:
            result['has_3x3'] = True
    
    # Analyze test examples
    for i, example in enumerate(problem_data.get('test', [])):
        input_dims = analyze_grid_dimensions(example['input'])
        output_dims = analyze_grid_dimensions(example['output'])
        
        example_info = {
            'example_num': i,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'is_3x3': input_dims == (3, 3) and output_dims == (3, 3)
        }
        result['test_examples'].append(example_info)
        
        if example_info['is_3x3']:
            result['has_3x3'] = True
    
    return result

def scan_directory(directory: str) -> Dict[str, Dict[str, Any]]:
    """Scan a directory of JSON files and analyze each problem."""
    results = {}
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return results
    
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    for filename in sorted(json_files):
        filepath = os.path.join(directory, filename)
        problem_id = filename[:-5]  # Remove .json extension
        
        try:
            with open(filepath, 'r') as f:
                problem_data = json.load(f)
            
            analysis = analyze_problem(problem_data)
            results[problem_id] = analysis
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return results

def main():
    # Paths to the ReARC dataset
    base_path = "/home/ubuntu/GFN_to_ARC/gfn/src/LLM_experiment/data/re-arc/arc_original"
    training_dir = os.path.join(base_path, "training")
    evaluation_dir = os.path.join(base_path, "evaluation")
    
    print("Scanning ReARC dataset for 3x3 problems...")
    print("=" * 50)
    
    # Analyze training set
    print("\nAnalyzing training set...")
    train_results = scan_directory(training_dir)
    
    # Analyze evaluation set
    print("\nAnalyzing evaluation set...")
    eval_results = scan_directory(evaluation_dir)
    
    # Find problems with 3x3 grids
    train_3x3_problems = {pid: data for pid, data in train_results.items() if data['has_3x3']}
    eval_3x3_problems = {pid: data for pid, data in eval_results.items() if data['has_3x3']}
    
    print(f"\n3x3 Problems Found:")
    print("=" * 50)
    print(f"Training set: {len(train_3x3_problems)} problems")
    print(f"Evaluation set: {len(eval_3x3_problems)} problems")
    print(f"Total: {len(train_3x3_problems) + len(eval_3x3_problems)} problems")
    
    # Print detailed information about 3x3 problems
    all_3x3_problems = {}
    all_3x3_problems.update({f"train_{pid}": data for pid, data in train_3x3_problems.items()})
    all_3x3_problems.update({f"eval_{pid}": data for pid, data in eval_3x3_problems.items()})
    
    print(f"\nDetailed 3x3 Problem Analysis:")
    print("=" * 50)
    
    for problem_id, analysis in all_3x3_problems.items():
        dataset = "Training" if problem_id.startswith("train_") else "Evaluation"
        clean_id = problem_id[6:] if problem_id.startswith("train_") else problem_id[5:]
        
        print(f"\nProblem: {clean_id} ({dataset})")
        print(f"  Training examples with 3x3:")
        for ex in analysis['train_examples']:
            if ex['is_3x3']:
                print(f"    Example {ex['example_num']}: {ex['input_dims']} → {ex['output_dims']}")
        
        print(f"  Test examples with 3x3:")
        for ex in analysis['test_examples']:
            if ex['is_3x3']:
                print(f"    Example {ex['example_num']}: {ex['input_dims']} → {ex['output_dims']}")
    
    # Show some examples of the actual problem data
    print(f"\nExample Problem Data:")
    print("=" * 50)
    
    # Show a few examples
    shown_count = 0
    for problem_id, analysis in all_3x3_problems.items():
        if shown_count >= 3:  # Show only first 3 examples
            break
            
        dataset = "training" if problem_id.startswith("train_") else "evaluation"
        clean_id = problem_id[6:] if problem_id.startswith("train_") else problem_id[5:]
        
        # Load and show the actual problem data
        filepath = os.path.join(base_path, dataset, f"{clean_id}.json")
        try:
            with open(filepath, 'r') as f:
                problem_data = json.load(f)
            
            print(f"\nProblem {clean_id} ({dataset}):")
            
            # Show training examples that are 3x3
            for i, example in enumerate(problem_data.get('train', [])):
                input_dims = analyze_grid_dimensions(example['input'])
                output_dims = analyze_grid_dimensions(example['output'])
                
                if input_dims == (3, 3) and output_dims == (3, 3):
                    print(f"  Training example {i}:")
                    print(f"    Input:  {example['input']}")
                    print(f"    Output: {example['output']}")
                    break  # Show only first 3x3 example
            
            # Show test examples that are 3x3
            for i, example in enumerate(problem_data.get('test', [])):
                input_dims = analyze_grid_dimensions(example['input'])
                output_dims = analyze_grid_dimensions(example['output'])
                
                if input_dims == (3, 3) and output_dims == (3, 3):
                    print(f"  Test example {i}:")
                    print(f"    Input:  {example['input']}")
                    print(f"    Output: {example['output']}")
                    break  # Show only first 3x3 example
            
            shown_count += 1
            
        except Exception as e:
            print(f"Error reading {clean_id}: {e}")
    
    # Save results to file
    results_file = "/home/ubuntu/GFN_to_ARC/gfn/src/trajectory_transformer_experiment/3x3_problems_analysis.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_3x3_problems': len(all_3x3_problems),
                'training_3x3_problems': len(train_3x3_problems),
                'evaluation_3x3_problems': len(eval_3x3_problems)
            },
            'training_problems': train_3x3_problems,
            'evaluation_problems': eval_3x3_problems
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()