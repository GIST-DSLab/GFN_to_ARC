"""
Data utilities for ARC Trajectory Transformer
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
import os
from tqdm import tqdm

def load_gflownet_trajectories(data_dir: str, problem_ids: List[int] = None) -> List[Dict]:
    """
    GFlowNet trajectory 데이터 로드
    """
    trajectories = []
    
    if problem_ids is None:
        # 모든 문제 ID 찾기
        problem_ids = []
        for item in os.listdir(data_dir):
            if item.startswith('problem_') and os.path.isdir(os.path.join(data_dir, item)):
                problem_id = int(item.split('_')[1])
                problem_ids.append(problem_id)
    
    for problem_id in tqdm(problem_ids, desc="Loading problems", unit="problem"):
        problem_dir = os.path.join(data_dir, f"problem_{problem_id}")
        if not os.path.exists(problem_dir):
            print(f"Warning: Problem {problem_id} directory not found")
            continue
            
        # trajectory 파일들 찾기
        trajectory_files = [f for f in os.listdir(problem_dir) if f.endswith('.json') and 'trajectories' in f]
        
        for filename in tqdm(trajectory_files, desc=f"Loading trajectory files for problem {problem_id}", unit="file", leave=False):
            filepath = os.path.join(problem_dir, filename)
            print(f"Loading {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # 성공한 trajectory만 필터링
            for traj in tqdm(data, desc="Filtering successful trajectories", unit="traj", leave=False):
                if 'rewards' in traj and len(traj['rewards']) > 0:
                    final_reward = traj['rewards'][-1]
                    if final_reward > 0:  # 성공한 trajectory만
                        trajectories.append(traj)
    
    print(f"Loaded {len(trajectories)} successful trajectories")
    return trajectories

def flatten_grid_state(grid_state: List[List[List[float]]]) -> np.ndarray:
    """
    3x3 grid state를 9차원 벡터로 변환
    """
    if isinstance(grid_state, list) and len(grid_state) == 1:
        grid_state = grid_state[0]  # Remove extra dimension
    
    grid = np.array(grid_state)
    if grid.shape == (3, 3):
        return grid.flatten()
    elif grid.shape == (1, 3, 3):
        return grid[0].flatten()
    else:
        raise ValueError(f"Unexpected grid shape: {grid.shape}")

def convert_trajectory_to_sequence(trajectory: Dict) -> Dict:
    """
    GFlowNet trajectory를 Trajectory Transformer 형식으로 변환
    
    Returns:
        {
            'observations': List[np.ndarray],  # Flattened grid states
            'actions': List[int],              # Action indices
            'rewards': List[float],            # Reward values
            'sequence': List[int],             # Tokenized sequence
            'trajectory_id': int,
            'problem_id': int
        }
    """
    try:
        states = trajectory['states']
        actions = trajectory['actions'] 
        rewards = trajectory['rewards']
        
        # Grid states를 flattened observations로 변환
        observations = []
        for state in states[:-1]:  # 마지막 state 제외 (action 없음)
            flat_obs = flatten_grid_state(state)
            observations.append(flat_obs)
        
        # Reward shaping: 마지막 step에만 reward 집중 → 분산
        shaped_rewards = shape_rewards(rewards, actions)
        
        # Sequence 생성: [obs, action, reward, obs, action, reward, ...]
        sequence = []
        for i in range(len(actions)):
            # Observation tokens (0-9 for colors, 10 for padding)
            obs_tokens = [int(x) if x < 10 else 10 for x in observations[i]]
            sequence.extend(obs_tokens)
            
            # Action token (offset by observation vocab)
            action_token = 11 + int(actions[i])  # 11-15 for actions 0-4
            sequence.append(action_token)
            
            # Reward token (discretized)
            reward_token = discretize_reward(shaped_rewards[i])
            sequence.append(reward_token)
        
        return {
            'observations': [obs.tolist() if isinstance(obs, np.ndarray) else obs for obs in observations],
            'actions': actions,
            'rewards': shaped_rewards,
            'sequence': sequence,
            'trajectory_id': trajectory.get('trajectory_id', -1),
            'problem_id': trajectory.get('problem_id', -1),
            'sequence_length': len(sequence)
        }
        
    except Exception as e:
        print(f"Error converting trajectory: {e}")
        return None

def shape_rewards(rewards: List[float], actions: List[int]) -> List[float]:
    """
    희소 보상을 조금 더 dense하게 변환
    """
    shaped = rewards.copy()
    
    # 성공 시 마지막 보상을 일부 분산
    if len(rewards) > 0 and rewards[-1] > 0:
        final_reward = rewards[-1]
        
        # Submit action (4)에 대한 보상 강화
        for i, action in enumerate(actions):
            if action == 4:  # submit
                shaped[i] = final_reward * 0.5  # 최종 보상의 50%
            elif i >= len(actions) - 3:  # 마지막 3 steps
                shaped[i] = final_reward * 0.1  # 최종 보상의 10%
    
    return shaped

def discretize_reward(reward: float) -> int:
    """
    Reward를 discrete token으로 변환
    """
    if reward <= 0:
        return 16  # Negative/zero reward token
    elif reward <= 0.1:
        return 17  # Small positive reward
    elif reward <= 0.5:
        return 18  # Medium positive reward  
    else:
        return 19  # Large positive reward

def create_vocabulary():
    """
    Vocabulary 생성
    """
    vocab = {
        # Grid colors (0-9)
        'color_0': 0, 'color_1': 1, 'color_2': 2, 'color_3': 3, 'color_4': 4,
        'color_5': 5, 'color_6': 6, 'color_7': 7, 'color_8': 8, 'color_9': 9,
        'pad': 10,  # Padding token
        
        # Actions (11-15) 
        'action_0': 11,  # left_rotate
        'action_1': 12,  # right_rotate  
        'action_2': 13,  # horizontal_flip
        'action_3': 14,  # vertical_flip
        'action_4': 15,  # submit
        
        # Rewards (16-19)
        'reward_neg': 16,   # Negative/zero reward
        'reward_small': 17, # Small positive reward
        'reward_med': 18,   # Medium positive reward
        'reward_large': 19, # Large positive reward
        
        # Special tokens
        'sos': 20,  # Start of sequence
        'eos': 21,  # End of sequence
    }
    
    return vocab

def pad_sequence(sequence: List[int], max_length: int, pad_token: int = 10) -> List[int]:
    """
    Sequence를 고정 길이로 패딩
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        return sequence + [pad_token] * (max_length - len(sequence))

def create_attention_mask(sequence: List[int], pad_token: int = 10) -> List[int]:
    """
    Attention mask 생성 (padding은 0, 실제 토큰은 1)
    """
    return [1 if token != pad_token else 0 for token in sequence]