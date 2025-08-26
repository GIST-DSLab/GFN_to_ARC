#!/usr/bin/env python3
import os
import time
import subprocess
from datetime import datetime

def get_training_progress():
    """Get training progress from all log files"""
    log_dir = "/data/gflownet-llm-additional/logs"
    problems = [86, 139, 149, 154, 178, 240, 379]
    
    print(f"\n{'='*60}")
    print(f"Training Progress - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    for problem in problems:
        log_file = f"{log_dir}/training_{problem}.log"
        if os.path.exists(log_file):
            try:
                # Get last few lines with progress
                result = subprocess.run(['tail', '-50', log_file], 
                                      capture_output=True, text=True)
                lines = result.stdout.split('\n')
                
                # Find the most recent tqdm progress line
                progress_lines = [line for line in lines if f"Problem {problem} Training:" in line]
                if progress_lines:
                    latest_progress = progress_lines[-1]
                    print(f"Problem {problem}: {latest_progress.split('Problem')[1].strip()}")
                else:
                    print(f"Problem {problem}: No progress data yet")
            except Exception as e:
                print(f"Problem {problem}: Error reading log - {e}")
        else:
            print(f"Problem {problem}: Log file not found")
    
    print(f"{'='*60}\n")

def check_processes():
    """Check if training processes are running"""
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    training_processes = [line for line in result.stdout.split('\n') 
                         if 'train_problem_' in line and 'python' in line]
    
    if training_processes:
        print(f"Active training processes: {len(training_processes)}")
        for proc in training_processes:
            parts = proc.split()
            if len(parts) > 10:
                script_name = [p for p in parts if 'train_problem_' in p]
                if script_name:
                    print(f"  - {script_name[0]}")
    else:
        print("No training processes found!")

def main():
    try:
        while True:
            os.system('clear')
            check_processes()
            get_training_progress()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()