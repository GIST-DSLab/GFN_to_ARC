#!/usr/bin/env python3
"""
데이터 전처리: GFlowNet trajectory 데이터를 LLM 학습용으로 변환
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from utils import *
import logging

class TrajectoryProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.action_mapping = config['action_mapping']
        self.problem_mapping = config['problem_mapping']
        self.logger = logging.getLogger(__name__)
        
    def load_trajectory_data(self, trajectory_file: str) -> List[Dict]:
        """trajectory JSON 파일 로드"""
        try:
            with open(trajectory_file, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} trajectories from {trajectory_file}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading {trajectory_file}: {e}")
            return []
    
    def extract_initial_and_final_states(self, trajectory: Dict) -> Tuple[List[List[int]], List[List[int]]]:
        """trajectory에서 초기 상태와 최종 상태 추출"""
        states = trajectory.get('states', [])
        if len(states) < 2:
            return None, None
            
        # 첫 번째 상태 (초기)
        initial_state = states[0]
        if isinstance(initial_state, list) and len(initial_state) > 0:
            if isinstance(initial_state[0], list) and len(initial_state[0]) > 0:
                initial_grid = initial_state[0]  # 첫 번째 차원 제거
            else:
                initial_grid = initial_state
        else:
            return None, None
            
        # 마지막 상태 (최종)
        final_state = states[-1]
        if isinstance(final_state, list) and len(final_state) > 0:
            if isinstance(final_state[0], list) and len(final_state[0]) > 0:
                final_grid = final_state[0]  # 첫 번째 차원 제거
            else:
                final_grid = final_state
        else:
            return None, None
            
        return initial_grid, final_grid
    
    def convert_actions(self, actions: List[int]) -> List[int]:
        """원본 action을 새로운 action 매핑으로 변환"""
        # trajectory 파일의 액션은 이미 0,1,2,3,4 형태이므로 직접 사용
        # 0: left_rotate, 1: right_rotate, 2: h_flip, 3: v_flip, 4: submit
        
        converted_actions = []
        for action in actions:
            if action in [0, 1, 2, 3, 4]:  # 유효한 액션만 포함
                converted_actions.append(action)
                
        # 마지막에 submit이 없으면 추가
        if converted_actions and converted_actions[-1] != 4:
            converted_actions.append(4)
        elif not converted_actions:
            converted_actions = [4]  # submit만 있는 경우
            
        return converted_actions
    
    def preprocess_single_trajectory(self, trajectory: Dict) -> Dict:
        """단일 trajectory 전처리 (성공한 trajectory만)"""
        try:
            # 성공 여부 확인 (마지막 reward > 0)
            rewards = trajectory.get('rewards', [])
            if not rewards or rewards[-1] <= 0:
                return None  # 실패한 trajectory는 제외
                
            # 초기/최종 상태 추출
            initial_grid, final_grid = self.extract_initial_and_final_states(trajectory)
            if initial_grid is None or final_grid is None:
                return None
                
            # padding 제거하여 실제 크기로 변환
            initial_trimmed = trim_padding(initial_grid)
            final_trimmed = trim_padding(final_grid)
            
            # action sequence 변환
            raw_actions = trajectory.get('actions', [])
            converted_actions = self.convert_actions(raw_actions)
            
            if not converted_actions:
                return None
                
            # 처리된 데이터 반환
            processed = {
                'trajectory_id': trajectory.get('trajectory_id', 0),
                'problem_id': trajectory.get('problem_id', 0),
                'initial_grid': initial_trimmed,
                'final_grid': final_trimmed,
                'action_sequence': converted_actions,
                'original_actions': raw_actions
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing trajectory: {e}")
            return None
    
    def create_llm_training_data(self, processed_trajectories: List[Dict]) -> List[Dict]:
        """LLM 학습용 데이터 생성"""
        training_data = []
        
        for traj in processed_trajectories:
            if traj is None:
                continue
                
            # 학습용 프롬프트 생성 (BARC 형식 사용)
            prompt = create_training_prompt(
                traj['initial_grid'],
                traj['final_grid'], 
                traj['action_sequence'],
                use_barc_format=True
            )
            
            # 정답 action sequence
            answer = format_action_sequence_for_llm(traj['action_sequence'])
            
            training_item = {
                'prompt': prompt,
                'completion': answer,
                'metadata': {
                    'trajectory_id': traj['trajectory_id'],
                    'problem_id': traj['problem_id'],
                    'grid_size': get_actual_grid_size(traj['initial_grid']),
                    'action_count': len(traj['action_sequence'])
                }
            }
            
            training_data.append(training_item)
            
        return training_data
    
    def process_all_problems(self):
        """모든 문제에 대해 전처리 수행"""
        all_training_data = []
        
        for problem_id in self.problem_mapping.keys():
            problem_dir = os.path.join(
                self.config['trajectory_data_dir'],
                f"problem_{problem_id}"
            )
            
            if not os.path.exists(problem_dir):
                self.logger.warning(f"Problem directory not found: {problem_dir}")
                continue
                
            self.logger.info(f"Processing problem {problem_id}")
            
            # 해당 문제의 모든 trajectory 파일 찾기
            trajectory_files = []
            for file in os.listdir(problem_dir):
                if file.startswith("trajectories_") and file.endswith(".json"):
                    trajectory_files.append(os.path.join(problem_dir, file))
            
            if not trajectory_files:
                self.logger.warning(f"No trajectory files found in {problem_dir}")
                continue
                
            trajectory_files.sort()  # 파일 순서 정렬
            self.logger.info(f"Found {len(trajectory_files)} trajectory files for problem {problem_id}")
            
            # 각 trajectory 파일 처리
            all_processed_trajectories = []
            total_trajectories_loaded = 0
            
            for trajectory_file in trajectory_files:
                self.logger.info(f"Processing file: {os.path.basename(trajectory_file)}")
                
                # trajectory 데이터 로드
                trajectories = self.load_trajectory_data(trajectory_file)
                total_trajectories_loaded += len(trajectories)
                
                # 각 trajectory 전처리
                for traj in trajectories:
                    processed = self.preprocess_single_trajectory(traj)
                    if processed:
                        all_processed_trajectories.append(processed)
            
            self.logger.info(f"Successfully processed {len(all_processed_trajectories)}/{total_trajectories_loaded} trajectories for problem {problem_id}")
            
            # LLM 학습용 데이터 생성
            training_data = self.create_llm_training_data(all_processed_trajectories)
            all_training_data.extend(training_data)
            
            # 문제별로 중간 저장
            problem_output_file = os.path.join(
                self.config['processed_data_dir'],
                f"problem_{problem_id}_processed.json"
            )
            os.makedirs(os.path.dirname(problem_output_file), exist_ok=True)
            save_json(training_data, problem_output_file)
            
        # 전체 데이터 저장
        output_file = os.path.join(self.config['processed_data_dir'], "all_training_data.json")
        save_json(all_training_data, output_file)
        
        self.logger.info(f"Total training samples: {len(all_training_data)}")
        return all_training_data
    
    def create_validation_split(self, training_data: List[Dict], split_ratio: float = 0.1):
        """검증 데이터 분할"""
        np.random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - split_ratio))
        
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # 저장
        train_file = os.path.join(self.config['processed_data_dir'], "train_data.json")
        val_file = os.path.join(self.config['processed_data_dir'], "val_data.json")
        
        save_json(train_data, train_file)
        save_json(val_data, val_file)
        
        self.logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        return train_data, val_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess trajectory data")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    log_file = os.path.join(config['processed_data_dir'], "preprocessing.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logging(log_file)
    
    # 전처리 실행
    processor = TrajectoryProcessor(config)
    
    logger.info("Starting trajectory data preprocessing...")
    
    # 모든 문제 처리
    training_data = processor.process_all_problems()
    
    # 검증 데이터 분할
    train_data, val_data = processor.create_validation_split(training_data)
    
    logger.info("Preprocessing completed successfully!")
    
    # 통계 출력
    logger.info(f"Total problems processed: {len(config['problem_mapping'])}")
    logger.info(f"Total training samples: {len(train_data)}")
    logger.info(f"Total validation samples: {len(val_data)}")

if __name__ == "__main__":
    main()