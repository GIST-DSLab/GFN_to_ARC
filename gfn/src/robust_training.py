#!/usr/bin/env python3
"""
안정적인 GFN 학습 래퍼 스크립트
오류 발생 시 자동 재시작하고 로그를 남김
"""

import subprocess
import time
import sys
import os
from datetime import datetime
import traceback

def log_message(message):
    """타임스탬프와 함께 로그 메시지 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_training():
    """메인 학습 실행"""
    log_message("Starting GFN ARC parallel training...")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '5,6',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'CUDA_LAUNCH_BLOCKING': '0'
    })
    
    cmd = [
        'python', 'main_parallel.py',
        '--gpu_ids', '5', '6',
        '--num_processes', '2',
        '--num_trajectories', '10000',
        '--output_dir', 'trajectories_gpu56'
    ]
    
    attempt = 1
    max_attempts = 3
    
    while attempt <= max_attempts:
        try:
            log_message(f"Attempt {attempt}/{max_attempts} - Running: {' '.join(cmd)}")
            
            # subprocess 실행
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                cwd='/home/ubuntu/GFN_to_ARC/gfn/src'
            )
            
            # 실시간 출력
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    sys.stdout.flush()
            
            return_code = process.poll()
            
            if return_code == 0:
                log_message("✅ Training completed successfully!")
                break
            else:
                log_message(f"❌ Training failed with return code: {return_code}")
                
        except KeyboardInterrupt:
            log_message("🛑 Training interrupted by user")
            if 'process' in locals():
                process.terminate()
            break
            
        except Exception as e:
            log_message(f"❌ Error occurred: {str(e)}")
            log_message(f"Traceback: {traceback.format_exc()}")
            
        if attempt < max_attempts:
            wait_time = 30 * attempt  # 30초, 60초, 90초 대기
            log_message(f"⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            
        attempt += 1
    
    log_message("🏁 Training session ended")
    
    # 세션을 살려두기 위해 무한 대기
    log_message("📋 Keeping session alive. Press Ctrl+C to exit or type 'exit' to close.")
    try:
        while True:
            user_input = input(">>> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'restart':
                log_message("🔄 Restarting training...")
                run_training()
                break
            else:
                log_message(f"Unknown command: {user_input}")
    except KeyboardInterrupt:
        log_message("👋 Session ended by user")

if __name__ == "__main__":
    run_training()