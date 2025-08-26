#!/usr/bin/env python3
"""
Baseline 데이터 생성: 동일한 문제에 대해 같은 답을 반복 학습
"""

import os
import json
import random
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm

def load_gflownet_data(file_path: str) -> List[Dict]:
    """GFlowNet 학습 데이터 로드"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_problem_solutions(train_data: List[Dict]) -> Dict[int, List[Dict]]:
    """각 문제별로 모든 solution 추출"""
    problem_solutions = defaultdict(list)
    
    for item in train_data:
        problem_id = item['metadata']['problem_id']
        problem_solutions[problem_id].append({
            'prompt': item['prompt'],
            'completion': item['completion'],
            'action_count': item['metadata']['action_count']
        })
    
    return problem_solutions

def find_best_solution(solutions: List[Dict]) -> Dict:
    """가장 짧은 action sequence를 가진 solution 선택"""
    # action_count가 가장 작은 solution 찾기
    best_solution = min(solutions, key=lambda x: x['action_count'])
    return best_solution

def create_baseline_data(train_data_path: str, output_path: str, samples_per_problem: int = 1000):
    """Baseline 데이터 생성: 각 문제당 하나의 정답을 반복"""
    
    # GFlowNet 데이터 로드
    print("Loading GFlowNet training data...")
    train_data = load_gflownet_data(train_data_path)
    
    # 문제별 solution 추출
    print("Extracting solutions for each problem...")
    problem_solutions = extract_problem_solutions(train_data)
    
    # 사용된 문제 ID들
    problem_ids = [86, 139, 178, 149, 154, 240, 379]
    
    baseline_data = []
    
    print("Creating baseline data...")
    for problem_id in tqdm(problem_ids, desc="Processing problems"):
        if problem_id not in problem_solutions:
            print(f"Warning: Problem {problem_id} not found in training data")
            continue
            
        # 해당 문제의 모든 solution 중 가장 짧은 것 선택
        solutions = problem_solutions[problem_id]
        best_solution = find_best_solution(solutions)
        
        print(f"Problem {problem_id}: Using solution with {best_solution['action_count']} actions")
        
        # 동일한 solution을 samples_per_problem번 반복
        for i in range(samples_per_problem):
            baseline_item = {
                'prompt': best_solution['prompt'],
                'completion': best_solution['completion'],
                'metadata': {
                    'problem_id': problem_id,
                    'solution_type': 'repeated_best',
                    'action_count': best_solution['action_count'],
                    'repetition_idx': i
                }
            }
            baseline_data.append(baseline_item)
    
    # 데이터 셔플 (학습시 다양성을 위해)
    random.shuffle(baseline_data)
    
    # 저장
    print(f"\nSaving {len(baseline_data)} baseline samples to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    # 통계 출력
    print("\nBaseline data statistics:")
    print(f"Total samples: {len(baseline_data)}")
    print(f"Problems included: {len(problem_ids)}")
    print(f"Samples per problem: {samples_per_problem}")
    
    # 각 문제별 사용된 action sequence 출력
    print("\nSelected solutions for each problem:")
    for problem_id in problem_ids:
        if problem_id in problem_solutions:
            best = find_best_solution(problem_solutions[problem_id])
            print(f"Problem {problem_id}: {best['completion']}")

def analyze_gflownet_diversity(train_data_path: str):
    """GFlowNet 데이터의 다양성 분석 (비교용)"""
    train_data = load_gflownet_data(train_data_path)
    problem_solutions = extract_problem_solutions(train_data)
    
    print("\nGFlowNet data diversity analysis:")
    for problem_id in [86, 139, 178, 149, 154, 240, 379]:
        if problem_id in problem_solutions:
            solutions = problem_solutions[problem_id]
            unique_completions = set(s['completion'] for s in solutions)
            action_counts = [s['action_count'] for s in solutions]
            
            print(f"\nProblem {problem_id}:")
            print(f"  Total trajectories: {len(solutions)}")
            print(f"  Unique solutions: {len(unique_completions)}")
            print(f"  Min actions: {min(action_counts)}")
            print(f"  Max actions: {max(action_counts)}")
            print(f"  Avg actions: {sum(action_counts)/len(action_counts):.1f}")

if __name__ == "__main__":
    # 경로 설정
    gflownet_train_path = "./processed_data/train_data.json"
    baseline_output_path = "./processed_data/baseline_train_data.json"
    
    # GFlowNet 데이터 다양성 분석
    print("=== GFlowNet Data Analysis ===")
    analyze_gflownet_diversity(gflownet_train_path)
    
    # Baseline 데이터 생성
    print("\n=== Creating Baseline Data ===")
    create_baseline_data(
        gflownet_train_path, 
        baseline_output_path,
        samples_per_problem=1000  # 7개 문제 * 1000 = 7000 samples
    )
    
    print("\nBaseline data creation completed!")
    print(f"Output saved to: {baseline_output_path}")