#!/usr/bin/env python3
"""
GFlowNet 데이터를 30x30 패딩으로 전처리하는 스크립트
GFlowNet의 실제 패딩 방식을 그대로 사용
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import yaml

def pad_grid_to_30x30(grid: List[List[int]], pad_value: int = 10) -> List[List[int]]:
    """
    GFlowNet 방식으로 그리드를 30x30으로 패딩
    
    Args:
        grid: 입력 그리드 (가변 크기)
        pad_value: 패딩 값 (기본 10, GFlowNet과 동일)
    
    Returns:
        30x30 패딩된 그리드
    """
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    
    # 30x30보다 크면 크롭
    if rows > 30 or cols > 30:
        grid_array = grid_array[:30, :30]
        rows, cols = min(rows, 30), min(cols, 30)
    
    # 패딩 계산
    pad_rows = 30 - rows
    pad_cols = 30 - cols
    
    # 중앙에 배치하도록 양쪽에 균등하게 패딩
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left
    
    # 패딩 적용
    padded = np.pad(grid_array, 
                   ((pad_top, pad_bottom), (pad_left, pad_right)), 
                   mode='constant', 
                   constant_values=pad_value)
    
    return padded.tolist()

def flatten_30x30_grid(grid: List[List[int]]) -> List[int]:
    """30x30 그리드를 900차원 벡터로 평탄화"""
    padded_grid = pad_grid_to_30x30(grid)
    return [cell for row in padded_grid for cell in row]

def create_30x30_vocabulary():
    """30x30용 어휘 생성"""
    vocab = {
        # 그리드 색상 (0-9)
        'color_0': 0, 'color_1': 1, 'color_2': 2, 'color_3': 3, 'color_4': 4,
        'color_5': 5, 'color_6': 6, 'color_7': 7, 'color_8': 8, 'color_9': 9,
        'pad': 10,  # 패딩 토큰
        
        # 액션 (11-15)
        'action_0': 11,  # left_rotate
        'action_1': 12,  # right_rotate  
        'action_2': 13,  # horizontal_flip
        'action_3': 14,  # vertical_flip
        'action_4': 15,  # submit
        
        # 리워드 (16-20)
        'reward_0': 16, 'reward_1': 17, 'reward_2': 18, 'reward_3': 19, 'reward_4': 20,
        
        # 값 (21-25)
        'value_0': 21, 'value_1': 22, 'value_2': 23, 'value_3': 24, 'value_4': 25,
    }
    return vocab

def discretize_reward_30x30(reward: float) -> int:
    """리워드를 이산화 (0-4 범위)"""
    if reward <= -1.0:
        return 0
    elif reward <= -0.5:
        return 1
    elif reward <= 0.0:
        return 2
    elif reward <= 0.5:
        return 3
    else:
        return 4

def discretize_value_30x30(value: float) -> int:
    """값을 이산화 (0-4 범위)"""
    if value <= -2.0:
        return 0
    elif value <= -1.0:
        return 1
    elif value <= 0.0:
        return 2
    elif value <= 1.0:
        return 3
    else:
        return 4

def convert_gflownet_trajectory_to_30x30_sequence(trajectory: Dict) -> Dict:
    """
    GFlowNet 궤적을 30x30 Trajectory Transformer 시퀀스로 변환
    
    시퀀스 형식: [obs(900), action(1), reward(1), value(1)] * steps
    """
    # GFlowNet 데이터 구조 처리
    if 'states_full' in trajectory and trajectory['states_full']:
        # states_full에서 grid 정보 추출
        states_data = []
        for state_full in trajectory['states_full']:
            if 'grid' in state_full:
                states_data.append(state_full['grid'])
            else:
                return None
    elif 'states' in trajectory and trajectory['states']:
        # states에서 grid 정보 추출 (첫 번째 형태의 nested 구조)
        states_data = []
        for state in trajectory['states']:
            if isinstance(state, list) and len(state) > 0:
                # 첫 번째 원소가 실제 그리드
                grid = state[0] if isinstance(state[0], list) else state
                states_data.append(grid)
            else:
                return None
    else:
        return None
    
    actions = trajectory.get('actions', [])
    rewards = trajectory.get('rewards', [])
    
    if len(states_data) == 0:
        return None
    
    # TD returns로 값 계산
    returns = []
    G = 0
    gamma = 0.99
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    sequence = []
    
    for i in range(len(states_data)):
        grid = states_data[i]
        
        # 그리드가 30x30이 아닌 경우 처리
        if isinstance(grid, list) and len(grid) > 0:
            # 30x30 형태로 변환
            obs_tokens = flatten_30x30_grid(grid)
            # 토큰 값 클램핑 (0-9는 그대로, 나머지는 패딩 값 10)
            obs_tokens = [int(x) if 0 <= x <= 9 else 10 for x in obs_tokens]
            sequence.extend(obs_tokens)
            
            # 액션이 있는 경우에만 추가
            if i < len(actions):
                action = actions[i]
                reward = rewards[i] if i < len(rewards) else 0.0
                value = returns[i] if i < len(returns) else 0.0
                
                # 액션 토큰 (11-15)
                action_token = 11 + int(action) if 0 <= action <= 4 else 15
                sequence.append(action_token)
                
                # 리워드 토큰 (16-20)
                reward_token = 16 + discretize_reward_30x30(reward)
                sequence.append(reward_token)
                
                # 값 토큰 (21-25)
                value_token = 21 + discretize_value_30x30(value)
                sequence.append(value_token)
    
    return {
        'sequence': sequence,
        'length': len(sequence),
        'num_states': len(states_data),
        'num_actions': len(actions),
        'total_reward': sum(rewards) if rewards else 0.0,
        'problem_id': trajectory.get('problem_id', -1)
    }

def load_gflownet_trajectories_30x30(data_dir: str) -> List[Dict]:
    """GFlowNet 궤적 데이터 로드"""
    trajectories = []
    
    print(f"Loading GFlowNet trajectories from {data_dir}")
    
    # 178번 문제만 처리
    target_problems = ['problem_178']
    for problem_dir in os.listdir(data_dir):
        if problem_dir not in target_problems:
            continue
            
        problem_path = os.path.join(data_dir, problem_dir)
        if not os.path.isdir(problem_path):
            continue
            
        print(f"Processing problem {problem_dir}")
        
        # 궤적 파일들 로드
        trajectory_files = [f for f in os.listdir(problem_path) if f.endswith('.json')]
        
        for traj_file in tqdm(trajectory_files[:1000], desc=f"Problem {problem_dir}"):  # 문제당 1000개 제한
            traj_path = os.path.join(problem_path, traj_file)
            try:
                with open(traj_path, 'r') as f:
                    trajectory = json.load(f)
                    # problem_dir에서 숫자 추출 (예: problem_154 -> 154)
                    if problem_dir.startswith('problem_'):
                        problem_id = int(problem_dir.split('_')[1])
                    else:
                        problem_id = int(problem_dir) if problem_dir.isdigit() else 0
                    trajectory['problem_id'] = problem_id
                    trajectories.append(trajectory)
            except Exception as e:
                print(f"Error loading {traj_path}: {e}")
                continue
    
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories

def main():
    parser = argparse.ArgumentParser(description="Preprocess GFlowNet data with 30x30 padding")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Configuration file")
    parser.add_argument("--data_dir", type=str, default="/data/gflownet-llm",
                       help="GFlowNet data directory")
    parser.add_argument("--output_dir", type=str, default="./processed_data_30x30",
                       help="Output directory")
    parser.add_argument("--max_trajectories", type=int, default=350000,
                       help="Maximum number of trajectories to process")
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== GFlowNet 30x30 Data Preprocessing ===")
    
    # 1. GFlowNet 궤적 데이터 로드
    trajectories = load_gflownet_trajectories_30x30(args.data_dir)
    
    if len(trajectories) > args.max_trajectories:
        print(f"Limiting to {args.max_trajectories} trajectories")
        trajectories = trajectories[:args.max_trajectories]
    
    # 2. 30x30 시퀀스로 변환
    print("Converting trajectories to 30x30 sequences...")
    sequences = []
    
    for trajectory in tqdm(trajectories, desc="Converting trajectories"):
        sequence_data = convert_gflownet_trajectory_to_30x30_sequence(trajectory)
        if sequence_data and sequence_data['length'] > 0:
            sequences.append(sequence_data)
    
    print(f"Converted {len(sequences)} valid sequences")
    
    # 3. 통계 계산
    sequence_lengths = [seq['length'] for seq in sequences]
    avg_length = np.mean(sequence_lengths)
    max_length = max(sequence_lengths)
    
    print(f"Sequence statistics:")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Maximum length: {max_length}")
    print(f"  Total sequences: {len(sequences)}")
    
    # 4. 훈련/검증 분할
    np.random.seed(42)
    np.random.shuffle(sequences)
    
    train_size = int(0.9 * len(sequences))
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]
    
    print(f"Data split:")
    print(f"  Training: {len(train_sequences)}")
    print(f"  Validation: {len(val_sequences)}")
    
    # 5. 저장
    output_file = os.path.join(args.output_dir, "arc_trajectory_data_30x30.json")
    data = {
        'train': train_sequences,
        'validation': val_sequences,
        'config': {
            'vocab_size': 26,
            'observation_dim': 900,  # 30x30 flattened
            'action_dim': 1,
            'reward_dim': 1,
            'value_dim': 1,
            'max_sequence_length': max_length,
            'avg_sequence_length': avg_length,
            'step_size': 903,  # obs(900) + action(1) + reward(1) + value(1)
            'padding_value': 10
        },
        'vocabulary': create_30x30_vocabulary(),
        'statistics': {
            'total_sequences': len(sequences),
            'train_sequences': len(train_sequences),
            'val_sequences': len(val_sequences),
            'avg_length': avg_length,
            'max_length': max_length
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to: {output_file}")
    
    # 6. 샘플 시퀀스 검증
    if sequences:
        sample = sequences[0]
        print(f"\nSample sequence:")
        print(f"  Length: {sample['length']}")
        print(f"  Problem ID: {sample['problem_id']}")
        print(f"  First 10 tokens: {sample['sequence'][:10]}")
        print(f"  Last 10 tokens: {sample['sequence'][-10:]}")

if __name__ == "__main__":
    main()