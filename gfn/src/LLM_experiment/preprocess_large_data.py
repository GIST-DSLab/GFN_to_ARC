#!/usr/bin/env python3
"""
대용량 trajectory 데이터 전처리 (700,000개 trajectory 활용)
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from utils import *
import logging
from tqdm import tqdm
import random

class LargeTrajectoryProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.action_mapping = config['action_mapping']
        self.problem_mapping = config['problem_mapping']
        self.logger = logging.getLogger(__name__)
        
    def load_trajectory_files_efficiently(self, max_files_per_problem: int = 20, max_samples: int = 50000):
        """메모리 효율적으로 trajectory 로드"""
        all_training_data = []
        
        problem_ids = list(self.problem_mapping.keys())
        self.logger.info(f"Processing {len(problem_ids)} problems: {problem_ids}")
        
        samples_per_problem = max_samples // len(problem_ids)
        
        for problem_id in tqdm(problem_ids, desc="Processing problems"):
            problem_dir = os.path.join(self.config['trajectory_data_dir'], f"problem_{problem_id}")
            
            if not os.path.exists(problem_dir):
                self.logger.warning(f"Problem directory not found: {problem_dir}")
                continue
                
            # trajectory 파일들 찾기
            trajectory_files = [f for f in os.listdir(problem_dir) 
                              if f.startswith("trajectories_") and f.endswith(".json")]
            trajectory_files.sort()
            
            # 랜덤하게 파일 선택
            selected_files = random.sample(trajectory_files, 
                                         min(max_files_per_problem, len(trajectory_files)))
            
            self.logger.info(f"Problem {problem_id}: Using {len(selected_files)}/{len(trajectory_files)} files")
            
            problem_samples = []
            
            for filename in tqdm(selected_files, desc=f"Loading files for problem {problem_id}", leave=False):
                filepath = os.path.join(problem_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        trajectories = json.load(f)
                    
                    # 성공한 trajectory만 필터링
                    successful_trajectories = []
                    for traj in trajectories:
                        if 'rewards' in traj and len(traj['rewards']) > 0:
                            if traj['rewards'][-1] > 0:  # 성공한 trajectory
                                successful_trajectories.append(traj)
                    
                    # 랜덤 샘플링
                    if len(successful_trajectories) > 0:
                        sample_count = min(len(successful_trajectories), 
                                         max(1, samples_per_problem // len(selected_files)))
                        sampled = random.sample(successful_trajectories, sample_count)
                        
                        # LLM 학습용 데이터로 변환
                        for traj in sampled:
                            processed = self.preprocess_single_trajectory(traj)
                            if processed:
                                llm_data = self.convert_to_llm_format(processed)
                                if llm_data:
                                    problem_samples.append(llm_data)
                    
                    self.logger.info(f"  {filename}: {len(successful_trajectories)} successful, sampled {len(sampled) if 'sampled' in locals() else 0}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {filename}: {e}")
            
            all_training_data.extend(problem_samples)
            self.logger.info(f"Problem {problem_id}: {len(problem_samples)} training samples created")
        
        return all_training_data
    
    def preprocess_single_trajectory(self, trajectory: Dict) -> Dict:
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
            converted_actions = self.convert_actions(raw_actions)
            
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
            self.logger.error(f"Error processing trajectory: {e}")
            return None
    
    def convert_actions(self, actions: List[int]) -> List[int]:
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
    
    def convert_to_llm_format(self, processed_traj: Dict) -> Dict:
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
            self.logger.error(f"Error converting to LLM format: {e}")
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess large trajectory data")
    parser.add_argument("--config", type=str, default="configs/config_ddp_456.yaml",
                       help="Path to config file")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum training samples to generate")
    parser.add_argument("--max_files_per_problem", type=int, default=10,
                       help="Maximum files per problem")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    log_file = os.path.join(config['processed_data_dir'], "large_preprocessing.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logging(log_file)
    
    # 랜덤 시드 설정
    random.seed(42)
    np.random.seed(42)
    
    # 전처리 실행
    processor = LargeTrajectoryProcessor(config)
    
    logger.info(f"Starting large-scale trajectory preprocessing...")
    logger.info(f"Target samples: {args.max_samples}")
    logger.info(f"Max files per problem: {args.max_files_per_problem}")
    
    # 모든 데이터 처리
    training_data = processor.load_trajectory_files_efficiently(
        max_files_per_problem=args.max_files_per_problem,
        max_samples=args.max_samples
    )
    
    logger.info(f"Generated {len(training_data)} training samples")
    
    # train/val split
    np.random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.9)
    
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # 저장
    train_file = os.path.join(config['processed_data_dir'], "train_data_large.json")
    val_file = os.path.join(config['processed_data_dir'], "val_data_large.json")
    all_file = os.path.join(config['processed_data_dir'], "all_training_data_large.json")
    
    save_json(train_data, train_file)
    save_json(val_data, val_file)
    save_json(training_data, all_file)
    
    logger.info(f"Saved {len(train_data)} training samples to {train_file}")
    logger.info(f"Saved {len(val_data)} validation samples to {val_file}")
    logger.info(f"Saved {len(training_data)} total samples to {all_file}")
    
    # 통계 출력
    logger.info(f"Problems processed: {len(config['problem_mapping'])}")
    logger.info(f"Final training samples: {len(train_data)}")
    logger.info(f"Final validation samples: {len(val_data)}")

if __name__ == "__main__":
    main()