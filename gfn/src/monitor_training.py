#!/usr/bin/env python3
"""Monitor GFN training progress."""

import os
import time
import json
from datetime import datetime

def monitor_training(output_dir="/data/gflownet-llm-additional", interval=30):
    """Monitor training progress by checking saved models and logs."""
    print(f"🔍 Monitoring training progress in {output_dir}")
    print(f"📊 Checking every {interval} seconds...\n")
    
    while True:
        try:
            # Check for saved models
            models_dir = os.path.join(output_dir, "models")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(model_files)} model files:")
                
                for model_file in sorted(model_files):
                    model_path = os.path.join(models_dir, model_file)
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                    print(f"  📄 {model_file} ({size_mb:.1f} MB) - {mod_time.strftime('%H:%M:%S')}")
            
            # Check for trajectory files
            for problem_id in [86, 139, 149, 154, 178, 240, 379]:
                problem_dir = os.path.join(output_dir, f"problem_{problem_id}")
                if os.path.exists(problem_dir):
                    traj_files = [f for f in os.listdir(problem_dir) if f.startswith('trajectories_batch')]
                    if traj_files:
                        print(f"  📊 Problem {problem_id}: {len(traj_files)} trajectory batches")
                    
                    # Check summary
                    summary_path = os.path.join(problem_dir, "summary.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                        print(f"     ✅ Completed with accuracy: {summary.get('model_accuracy', 'N/A')}")
            
            # Check experiment summary
            summary_files = [f for f in os.listdir(output_dir) if f.startswith('experiment_summary')]
            if summary_files:
                print(f"\n📝 Found experiment summaries: {', '.join(summary_files)}")
            
            print("\n" + "-" * 60)
            
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    monitor_training()