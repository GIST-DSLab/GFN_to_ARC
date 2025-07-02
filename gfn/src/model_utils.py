#!/usr/bin/env python3
"""
GFlowNet 모델 저장/로딩 유틸리티 함수들
"""

import torch
import json
import os
from gflow.gflownet_target import GFlowNet
from policy_target import EnhancedMLPForwardPolicy
from ARCenv.wrapper import env_return
from arcle.loaders import ARCLoader

def load_trained_model(model_path, device=None):
    """학습된 GFlowNet 모델 로딩"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 체크포인트 로딩
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 설정 가져오기
    training_args = checkpoint['training_args']
    problem_id = checkpoint['problem_id']
    
    print(f"Loading model for problem {problem_id}")
    print(f"Training config: {training_args}")
    
    # 환경 초기화
    loader = ARCLoader()
    env = env_return(render=None, data=loader, options=None, batch_size=1, mode=training_args['env_mode'])
    
    # 모델 아키텍처 재구성
    forward_policy = EnhancedMLPForwardPolicy(
        state_dim=30, 
        hidden_dim=256, 
        num_actions=training_args['num_actions'],
        batch_size=1, 
        embedding_dim=32, 
        ep_len=training_args['ep_len']
    ).to(device)

    model = GFlowNet(
        forward_policy=forward_policy,
        backward_policy=None,
        total_flow=torch.nn.parameter.Parameter(torch.tensor(1.0).to(device)),
        env=env, 
        device=device, 
        env_style=training_args['env_mode'], 
        num_actions=training_args['num_actions'], 
        ep_len=training_args['ep_len']
    ).to(device)
    
    # 학습된 가중치 로딩
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Trajectories used in training: {checkpoint['num_trajectories']}")
    print(f"📅 Training timestamp: {checkpoint['timestamp']}")
    
    return model, env, checkpoint

def list_saved_models(output_dir="trajectories_output"):
    """저장된 모델들 목록 출력"""
    
    model_files = []
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return model_files
    
    for problem_dir in os.listdir(output_dir):
        problem_path = os.path.join(output_dir, problem_dir)
        if not os.path.isdir(problem_path):
            continue
            
        models_path = os.path.join(problem_path, "models")
        if not os.path.exists(models_path):
            continue
            
        for file in os.listdir(models_path):
            if file.endswith(".pt"):
                model_path = os.path.join(models_path, file)
                info_path = os.path.join(models_path, file.replace(".pt", "_info.json").replace("gflownet_", "model_info_"))
                
                model_info = {
                    'problem_dir': problem_dir,
                    'model_file': file,
                    'model_path': model_path,
                    'info_path': info_path if os.path.exists(info_path) else None
                }
                
                # 모델 정보 로딩
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            info_data = json.load(f)
                            model_info.update(info_data)
                    except Exception as e:
                        print(f"Warning: Failed to load info for {file}: {e}")
                
                model_files.append(model_info)
    
    return model_files

def test_model_inference(model, env, problem_id, num_samples=5):
    """모델 추론 테스트"""
    
    print(f"\n🧪 Testing model inference for problem {problem_id}")
    print(f"Generating {num_samples} sample trajectories...")
    
    results = []
    
    for i in range(num_samples):
        try:
            # 환경 리셋
            state, info = env.reset(options={
                "prob_index": problem_id, 
                "adaptation": True, 
                "subprob_index": 0
            })
            
            # 모델로 trajectory 생성
            final_state, log = model.sample_states(state, info, return_log=True, batch_size=1)
            
            # 결과 정리
            result = {
                'sample_id': i,
                'trajectory_length': len(log.traj),
                'final_reward': float(log.rewards[-1]),
                'total_actions': len(log.actions)
            }
            
            results.append(result)
            print(f"  Sample {i+1}: {result['trajectory_length']} steps, reward: {result['final_reward']:.4f}")
            
        except Exception as e:
            print(f"  Sample {i+1}: Failed - {e}")
            
    return results

if __name__ == "__main__":
    print("=== GFlowNet Model Utility ===")
    
    # 저장된 모델들 출력
    models = list_saved_models()
    
    if not models:
        print("No saved models found.")
    else:
        print(f"\nFound {len(models)} saved models:")
        for i, model_info in enumerate(models):
            print(f"{i+1}. {model_info['model_file']}")
            print(f"   Path: {model_info['model_path']}")
            if 'model_parameters' in model_info:
                print(f"   Parameters: {model_info['model_parameters']:,}")
            print()