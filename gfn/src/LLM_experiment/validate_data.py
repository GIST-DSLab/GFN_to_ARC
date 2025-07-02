#!/usr/bin/env python3
"""
데이터 검증: 훈련 데이터의 품질과 일관성 확인
"""

import os
import json
import numpy as np
from collections import Counter, defaultdict
from utils import *

def validate_training_data():
    """훈련 데이터 검증"""
    config = load_config("configs/config.yaml")
    
    # 전체 훈련 데이터 로드
    data_file = os.path.join(config['processed_data_dir'], "all_training_data.json")
    if not os.path.exists(data_file):
        print(f"Training data file not found: {data_file}")
        return
    
    data = load_json(data_file)
    print(f"Total training samples: {len(data)}")
    
    # 액션 분포 확인
    action_counter = Counter()
    problem_action_count = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        problem_id = item['metadata']['problem_id']
        completion = item['completion']
        
        # 액션 파싱
        try:
            actions = eval(completion)  # [submit] 형태
            for action in actions:
                action_counter[action] += 1
                problem_action_count[problem_id][action] += 1
        except:
            print(f"Failed to parse completion: {completion}")
    
    print("\n=== Action Distribution ===")
    for action, count in action_counter.most_common():
        print(f"{action}: {count}")
    
    print("\n=== Problem-wise Action Distribution ===")
    for problem_id in sorted(problem_action_count.keys()):
        actions = problem_action_count[problem_id]
        print(f"Problem {problem_id}: {dict(actions)}")
    
    # 문제별 샘플 수 확인
    problem_counter = Counter()
    for item in data:
        problem_id = item['metadata']['problem_id']
        problem_counter[problem_id] += 1
    
    print(f"\n=== Problem Distribution ===")
    for problem_id, count in sorted(problem_counter.items()):
        arc_id = config['problem_mapping'].get(str(problem_id), 'Unknown')
        print(f"Problem {problem_id} (ARC: {arc_id}): {count} samples")
    
    # ARC ID 중복 확인
    arc_id_problems = defaultdict(list)
    for problem_id, arc_id in config['problem_mapping'].items():
        arc_id_problems[arc_id].append(problem_id)
    
    print(f"\n=== ARC ID Mapping Issues ===")
    for arc_id, problem_list in arc_id_problems.items():
        if len(problem_list) > 1:
            print(f"ARC {arc_id} has multiple problems: {problem_list}")
    
    # 액션 시퀀스 길이 분석
    action_lengths = []
    for item in data:
        try:
            actions = eval(item['completion'])
            action_lengths.append(len(actions))
        except:
            pass
    
    if action_lengths:
        print(f"\n=== Action Sequence Length Stats ===")
        print(f"Mean: {np.mean(action_lengths):.2f}")
        print(f"Min: {min(action_lengths)}")
        print(f"Max: {max(action_lengths)}")
        print(f"Median: {np.median(action_lengths):.2f}")
    
    # 그리드 크기 분석
    grid_sizes = []
    for item in data:
        grid_size = item['metadata']['grid_size']
        grid_sizes.append(tuple(grid_size))
    
    size_counter = Counter(grid_sizes)
    print(f"\n=== Grid Size Distribution ===")
    for size, count in size_counter.most_common():
        print(f"{size}: {count}")

def check_trajectory_files():
    """원본 trajectory 파일들 확인"""
    config = load_config("configs/config.yaml")
    
    print("=== Trajectory Files Check ===")
    for problem_id in config['problem_mapping'].keys():
        trajectory_file = os.path.join(
            config['trajectory_data_dir'],
            f"problem_{problem_id}",
            "trajectories_0_1000.json"
        )
        
        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'r') as f:
                trajectories = json.load(f)
            
            # 첫 번째 trajectory의 액션 분석
            if trajectories:
                first_traj = trajectories[0]
                actions = first_traj.get('actions', [])
                unique_actions = set(actions)
                
                print(f"Problem {problem_id}: {len(trajectories)} trajectories")
                print(f"  First trajectory actions: {len(actions)} total, unique: {sorted(unique_actions)}")
                
                # 액션 분포
                action_counter = Counter(actions)
                print(f"  Action distribution: {dict(action_counter.most_common()[:5])}")
        else:
            print(f"Problem {problem_id}: File not found")

if __name__ == "__main__":
    print("Validating training data...")
    validate_training_data()
    
    print("\n" + "="*50)
    check_trajectory_files()