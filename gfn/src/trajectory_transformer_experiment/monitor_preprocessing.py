#!/usr/bin/env python3
"""
전처리 진행 상황 모니터링 및 완료 시 자동 학습 시작
"""

import time
import subprocess
import os

def check_preprocessing_status():
    """전처리 상황 확인"""
    try:
        result = subprocess.run(['tmux', 'capture-pane', '-t', 'gflownet_30x30_preprocessing', '-p'], 
                              capture_output=True, text=True)
        output = result.stdout
        
        # 마지막 몇 줄 확인
        lines = output.strip().split('\n')
        last_lines = lines[-5:] if len(lines) >= 5 else lines
        
        return '\n'.join(last_lines)
    except Exception as e:
        return f"Error checking status: {e}"

def check_if_completed():
    """전처리 완료 여부 확인"""
    status = check_preprocessing_status()
    
    # 완료 신호들 확인
    completion_signals = [
        "Preprocessing completed",
        "Data saved to:",
        "Converting trajectories to 30x30 sequences"
    ]
    
    for signal in completion_signals:
        if signal in status:
            return True
    
    # 프로세스 종료 확인
    try:
        result = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
        if 'gflownet_30x30_preprocessing' not in result.stdout:
            return True
    except:
        pass
    
    return False

def start_training():
    """학습 시작"""
    print("Starting multi-GPU training...")
    
    # 데이터 파일 경로 확인
    data_file = "./processed_data_gflownet_30x30/arc_trajectory_data_30x30.json"
    
    if os.path.exists(data_file):
        print(f"Data file found: {data_file}")
        
        # 멀티 GPU 학습 시작
        cmd = [
            'tmux', 'new-session', '-d', '-s', 'gflownet_30x30_training',
            f'CUDA_VISIBLE_DEVICES=5,6 python train_gflownet_30x30_multi_gpu.py --config configs/config_gflownet_30x30.yaml --data_file {data_file}'
        ]
        
        subprocess.run(cmd)
        print("Multi-GPU training started in tmux session: gflownet_30x30_training")
        print("Monitor with: tmux attach -t gflownet_30x30_training")
        
    else:
        print(f"Data file not found: {data_file}")
        print("Starting single GPU training instead...")
        
        cmd = [
            'tmux', 'new-session', '-d', '-s', 'gflownet_30x30_training_single',
            f'CUDA_VISIBLE_DEVICES=5 python train_gflownet_30x30.py --config configs/config_gflownet_30x30.yaml'
        ]
        
        subprocess.run(cmd)
        print("Single GPU training started in tmux session: gflownet_30x30_training_single")

def main():
    print("=== Monitoring GFlowNet 30x30 Preprocessing ===")
    print("Checking every 30 seconds...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print(f"\n[{time.strftime('%H:%M:%S')}] Checking preprocessing status...")
            
            status = check_preprocessing_status()
            print(status)
            
            if check_if_completed():
                print("\n🎉 Preprocessing completed!")
                start_training()
                break
            
            print("Still processing... waiting 30 seconds")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        
        # 현재 상황 체크
        status = check_preprocessing_status()
        print(f"Current status:\n{status}")
        
        response = input("Start training anyway? (y/n): ")
        if response.lower() == 'y':
            start_training()

if __name__ == "__main__":
    main()