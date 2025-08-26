#!/usr/bin/env python3
"""
대용량 trajectory 데이터 전처리 - 모든 성공한 trajectory 활용
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from utils import *
import logging
from tqdm import tqdm
import random

def count_successful_trajectories():
    """각 문제별 성공한 trajectory 수 계산"""
    base_dir = "/data/gflownet-llm"
    problem_stats = {}
    
    for prob in [86, 139, 149, 154, 178, 240, 379]:
        problem_dir = os.path.join(base_dir, f"problem_{prob}")
        if not os.path.exists(problem_dir):
            continue
            
        trajectory_files = [f for f in os.listdir(problem_dir) 
                          if f.startswith("trajectories_") and f.endswith(".json")]
        
        total_trajectories = 0
        successful_trajectories = 0
        
        # 첫 번째 파일만 확인 (전체 패턴 파악용)
        if trajectory_files:
            sample_file = os.path.join(problem_dir, trajectory_files[0])
            with open(sample_file, 'r') as f:
                data = json.load(f)
            
            total_trajectories = len(data)
            successful_trajectories = len([t for t in data if t.get('rewards', []) and t['rewards'][-1] > 0])
            success_rate = successful_trajectories / total_trajectories * 100
            
            problem_stats[prob] = {
                'total_files': len(trajectory_files),
                'trajectories_per_file': total_trajectories,
                'successful_per_file': successful_trajectories,
                'success_rate': success_rate,
                'estimated_total_successful': successful_trajectories * len(trajectory_files)
            }
    
    return problem_stats

def preprocess_large_scale(max_files_per_problem: int = 50, target_samples: int = 20000):
    """대용량 전처리"""
    
    # 설정 로드
    config = load_config("configs/config_ddp_456.yaml")
    
    # 로깅 설정
    log_file = os.path.join(config['processed_data_dir'], "large_scale_preprocessing.log")
    os.makedirs(config['processed_data_dir'], exist_ok=True)
    logger = setup_logging(log_file)
    
    logger.info("=== Large Scale Trajectory Preprocessing ===")
    
    # 먼저 통계 확인
    stats = count_successful_trajectories()
    logger.info("Problem statistics:")
    total_estimated = 0
    for prob, stat in stats.items():
        logger.info(f"  Problem {prob}: {stat['success_rate']:.1f}% success rate, "
                   f"~{stat['estimated_total_successful']} successful trajectories")
        total_estimated += stat['estimated_total_successful']
    
    logger.info(f"Total estimated successful trajectories: {total_estimated}")
    
    # 각 문제별로 사용할 파일 수 결정 (성공률 기반)
    all_training_data = []
    
    # 성공한 trajectory가 있는 문제들만 처리
    valid_problems = [prob for prob, stat in stats.items() if stat['successful_per_file'] > 0]
    logger.info(f"Processing problems with successful trajectories: {valid_problems}")
    
    samples_per_problem = target_samples // len(valid_problems)
    
    for problem_id in tqdm(valid_problems, desc="Processing problems"):
        problem_dir = os.path.join("/data/gflownet-llm", f"problem_{problem_id}")
        
        trajectory_files = [f for f in os.listdir(problem_dir) 
                          if f.startswith("trajectories_") and f.endswith(".json")]
        trajectory_files.sort()
        
        # 성공률에 따라 파일 수 조정
        success_rate = stats[problem_id]['success_rate']
        if success_rate > 20:
            files_to_use = min(max_files_per_problem, len(trajectory_files))
        elif success_rate > 10:
            files_to_use = min(max_files_per_problem * 2, len(trajectory_files))
        else:
            files_to_use = min(max_files_per_problem * 3, len(trajectory_files))
        
        selected_files = random.sample(trajectory_files, files_to_use)
        logger.info(f"Problem {problem_id}: Using {files_to_use}/{len(trajectory_files)} files "
                   f"(success rate: {success_rate:.1f}%)")
        
        problem_samples = []
        
        for filename in tqdm(selected_files, desc=f"Processing problem {problem_id}", leave=False):
            filepath = os.path.join(problem_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    trajectories = json.load(f)
                
                # 성공한 trajectory만 필터링
                successful_trajectories = []
                for traj in trajectories:
                    if 'rewards' in traj and len(traj['rewards']) > 0:
                        if traj['rewards'][-1] > 0:
                            successful_trajectories.append(traj)
                
                # 전처리 및 LLM 형식 변환
                for traj in successful_trajectories:
                    processed = preprocess_single_trajectory(traj)
                    if processed:
                        llm_data = convert_to_llm_format(processed)
                        if llm_data:
                            problem_samples.append(llm_data)
                
                logger.info(f"  {filename}: {len(successful_trajectories)} successful → {len(problem_samples)} samples")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        
        all_training_data.extend(problem_samples)
        logger.info(f"Problem {problem_id}: {len(problem_samples)} total samples")
    
    logger.info(f"Total training samples generated: {len(all_training_data)}")
    
    # Train/Val split
    random.shuffle(all_training_data)
    split_idx = int(len(all_training_data) * 0.9)
    train_data = all_training_data[:split_idx]
    val_data = all_training_data[split_idx:]
    
    # 저장
    train_file = os.path.join(config['processed_data_dir'], "train_data_large_scale.json")
    val_file = os.path.join(config['processed_data_dir'], "val_data_large_scale.json")
    all_file = os.path.join(config['processed_data_dir'], "all_training_data_large_scale.json")
    
    save_json(train_data, train_file)
    save_json(val_data, val_file)
    save_json(all_training_data, all_file)
    
    logger.info(f"Saved {len(train_data)} training samples")
    logger.info(f"Saved {len(val_data)} validation samples")
    logger.info(f"Saved {len(all_training_data)} total samples")
    
    return len(all_training_data)

def preprocess_single_trajectory(trajectory: Dict) -> Dict:
    """단일 trajectory 전처리"""
    try:
        # 초기/최종 상태 추출
        states = trajectory.get('states', [])
        if len(states) < 2:
            return None
            
        initial_state = states[0]
        final_state = states[-1]
        
        # 상태가 nested list인 경우 처리
        if isinstance(initial_state, list) and len(initial_state) > 0:
            if isinstance(initial_state[0], list) and len(initial_state[0]) > 0:
                initial_grid = initial_state[0]
            else:
                initial_grid = initial_state
        else:
            return None
            
        if isinstance(final_state, list) and len(final_state) > 0:
            if isinstance(final_state[0], list) and len(final_state[0]) > 0:
                final_grid = final_state[0]
            else:
                final_grid = final_state
        else:
            return None
        
        # padding 제거
        initial_trimmed = trim_padding(initial_grid)
        final_trimmed = trim_padding(final_grid)
        
        # action sequence 변환
        raw_actions = trajectory.get('actions', [])
        converted_actions = convert_actions(raw_actions)
        
        if not converted_actions:
            return None
            
        return {
            'trajectory_id': trajectory.get('trajectory_id', 0),
            'problem_id': trajectory.get('problem_id', 0),
            'initial_grid': initial_trimmed,
            'final_grid': final_trimmed,
            'action_sequence': converted_actions,
            'original_actions': raw_actions
        }
        
    except Exception as e:
        return None

def convert_actions(actions: List[int]) -> List[int]:
    """액션 변환"""
    converted_actions = []
    for action in actions:
        if action in [0, 1, 2, 3, 4]:
            converted_actions.append(action)
            
    if converted_actions and converted_actions[-1] != 4:
        converted_actions.append(4)
    elif not converted_actions:
        converted_actions = [4]
        
    return converted_actions

def convert_to_llm_format(processed_traj: Dict) -> Dict:
    """LLM 학습용 형식으로 변환"""
    try:
        # BARC 형식 프롬프트 생성
        prompt = create_training_prompt(
            processed_traj['initial_grid'],
            processed_traj['final_grid'], 
            processed_traj['action_sequence'],
            use_barc_format=True
        )
        
        # 정답 action sequence
        answer = format_action_sequence_for_llm(processed_traj['action_sequence'])
        
        return {
            'prompt': prompt,
            'completion': answer,
            'metadata': {
                'trajectory_id': processed_traj['trajectory_id'],
                'problem_id': processed_traj['problem_id'],
                'grid_size': get_actual_grid_size(processed_traj['initial_grid']),
                'action_count': len(processed_traj['action_sequence'])
            }
        }
        
    except Exception as e:
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_samples", type=int, default=10000)
    parser.add_argument("--max_files_per_problem", type=int, default=30)
    args = parser.parse_args()
    
    # 시드 설정
    random.seed(42)
    np.random.seed(42)
    
    print("=== Large Scale Trajectory Preprocessing ===")
    print(f"Target samples: {args.target_samples}")
    print(f"Max files per problem: {args.max_files_per_problem}")
    
    # 통계 먼저 확인
    print("\nAnalyzing trajectory statistics...")
    stats = count_successful_trajectories()
    for prob, stat in stats.items():
        print(f"Problem {prob}: {stat['success_rate']:.1f}% success, "
              f"~{stat['estimated_total_successful']} successful trajectories")
    
    print(f"\nEstimated total successful trajectories: {sum(s['estimated_total_successful'] for s in stats.values())}")
    
    # 전처리 실행
    print("\nStarting preprocessing...")
    total_samples = preprocess_large_scale(args.max_files_per_problem, args.target_samples)
    print(f"\n✅ Preprocessing completed! Generated {total_samples} samples.")