#!/usr/bin/env python3
import torch
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from config import CONFIG

# ARC 문제 리스트
ARC_PROBLEMS = [86, 139, 149, 154, 178, 240, 379]

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs='+', type=int, default=ARC_PROBLEMS)
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[6, 7])
    parser.add_argument("--output_dir", type=str, default="/data/gflownet-llm-additional")
    parser.add_argument("--num_trajectories", type=int, default=30000)
    return parser.parse_args()

def create_single_training_script(problem_id, gpu_id, output_dir, script_path):
    """Create a single training script that uses main.py approach."""
    script_content = f'''#!/usr/bin/env python3
import sys
sys.path.append('/home/ubuntu/GFN_to_ARC/gfn/src')

import torch
import os
import json
from datetime import datetime
from config import CONFIG
from gflow.utils import seed_everything
from train import train_model, evaluate_model, save_gflownet_trajectories_batch
from arcle.loaders import ARCLoader

print(f"🚀 GPU {gpu_id}: Training problem {problem_id} with original main.py method")

# When using CUDA_VISIBLE_DEVICES, the specified GPU becomes cuda:0
device = torch.device("cuda:0")
print(f"Using device: {{device}} (GPU {gpu_id} via CUDA_VISIBLE_DEVICES)")

# Seed everything
seed_everything(777)

# Create args object like main.py
class Args:
    def __init__(self):
        self.batch_size = CONFIG["BATCH_SIZE"]
        self.num_epochs = CONFIG["NUM_EPOCHS"] 
        self.env_mode = CONFIG["ENV_MODE"]
        self.num_actions = CONFIG["ACTIONNUM"]
        self.ep_len = CONFIG["EP_LEN"]
        self.use_offpolicy = False
        self.sampling_method = "prt"
        self.subtask_num = CONFIG["SUBTASKNUM"]
        self.prob_index = {problem_id}

args = Args()

try:
    print(f"[GPU {gpu_id}] Problem {problem_id}: Starting 30k step training...")
    
    # Use exact same method as main.py
    model, env = train_model(
        num_epochs=CONFIG["NUM_EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"], 
        device=device,
        env_mode=CONFIG["ENV_MODE"],
        prob_index={problem_id},
        num_actions=CONFIG["ACTIONNUM"],
        args=args,
        use_offpolicy=False,
        sub_task=CONFIG["SUBTASKNUM"]
    )
    
    # Final evaluation using original method
    final_accuracy = evaluate_model(model, env, num_samples=100, prob_index={problem_id}, subtask=CONFIG["SUBTASKNUM"])
    print(f"[GPU {gpu_id}] Problem {problem_id}: Final accuracy {{final_accuracy:.3f}}")
    
    # Save model
    model_path = os.path.join("{output_dir}", f"models/problem_{problem_id}_gpu_{gpu_id}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({{
        'model_state_dict': model.state_dict(),
        'final_accuracy': final_accuracy,
        'problem_id': {problem_id},
        'gpu_id': {gpu_id},
        'training_method': 'original_main_py',
        'timestamp': datetime.now().isoformat()
    }}, model_path)
    
    print(f"✅ [GPU {gpu_id}] Problem {problem_id}: Training completed, accuracy {{final_accuracy:.3f}}")
    
    # Save result
    result = {{
        "prob_index": {problem_id},
        "success": True,
        "final_accuracy": final_accuracy,
        "model_path": model_path,
        "gpu_id": {gpu_id},
        "training_method": "original_main_py"
    }}
    
    result_path = os.path.join("{output_dir}", f"results/training_result_{problem_id}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
except Exception as e:
    print(f"❌ [GPU {gpu_id}] Problem {problem_id}: Training failed - {{e}}")
    result = {{"prob_index": {problem_id}, "success": False, "reason": str(e), "gpu_id": {gpu_id}}}
    result_path = os.path.join("{output_dir}", f"results/training_result_{problem_id}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

def create_trajectory_script(problem_id, gpu_id, output_dir, num_trajectories, script_path):
    """Create trajectory generation script."""
    script_content = f'''#!/usr/bin/env python3
import sys
sys.path.append('/home/ubuntu/GFN_to_ARC/gfn/src')

import torch
import os
import json
from datetime import datetime
from config import CONFIG
from train import initialize_env, initialize_model, save_gflownet_trajectories_batch
from arcle.loaders import ARCLoader

print(f"🔄 GPU {gpu_id}: Generating {num_trajectories} trajectories for problem {problem_id}")

# When using CUDA_VISIBLE_DEVICES, the specified GPU becomes cuda:0
device = torch.device("cuda:0")
print(f"Using device: {{device}} (GPU {gpu_id} via CUDA_VISIBLE_DEVICES)")

try:
    # Load trained model
    model_path = os.path.join("{output_dir}", f"models/problem_{problem_id}_gpu_{gpu_id}.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {{model_path}}")
        exit(1)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize environment and model
    loader = ARCLoader()
    env = initialize_env(CONFIG["ENV_MODE"], {problem_id}, loader)
    
    class Args:
        def __init__(self):
            self.batch_size = CONFIG["BATCH_SIZE"]
            self.num_epochs = CONFIG["NUM_EPOCHS"]
            self.env_mode = CONFIG["ENV_MODE"]
            self.num_actions = CONFIG["ACTIONNUM"]
            self.ep_len = CONFIG["EP_LEN"]
            self.use_offpolicy = False
            self.sampling_method = "prt"
            self.subtask_num = CONFIG["SUBTASKNUM"]
    
    args = Args()
    model, _, _ = initialize_model(env, CONFIG["ACTIONNUM"], CONFIG["BATCH_SIZE"], device, args)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate trajectories using original batch method
    save_path_prefix = os.path.join("{output_dir}", f"problem_{problem_id}", "trajectories")
    
    print(f"[GPU {gpu_id}] Problem {problem_id}: Generating trajectories...")
    save_gflownet_trajectories_batch(
        model, env, {num_trajectories}, save_path_prefix, 
        {problem_id}, CONFIG["SUBTASKNUM"], batch_size=1000
    )
    
    print(f"✅ [GPU {gpu_id}] Problem {problem_id}: Generated {num_trajectories} trajectories")
    
    # Save summary
    summary = {{
        "problem_id": {problem_id},
        "num_trajectories": {num_trajectories},
        "model_accuracy": checkpoint.get('final_accuracy', 0),
        "generation_method": "original_batch",
        "timestamp": datetime.now().isoformat()
    }}
    
    summary_path = os.path.join("{output_dir}", f"problem_{problem_id}", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    result = {{
        "prob_index": {problem_id},
        "success": True,
        "trajectories_generated": {num_trajectories},
        "save_path": save_path_prefix,
        "gpu_id": {gpu_id}
    }}
    
    result_path = os.path.join("{output_dir}", f"results/trajectory_result_{problem_id}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
        
except Exception as e:
    print(f"❌ [GPU {gpu_id}] Problem {problem_id}: Trajectory generation failed - {{e}}")
    result = {{"prob_index": {problem_id}, "success": False, "reason": str(e), "gpu_id": {gpu_id}}}
    result_path = os.path.join("{output_dir}", f"results/trajectory_result_{problem_id}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

def main():
    args = parse_arguments()
    
    print("🚀 Starting Original Main.py Style Training with GPU Control")
    print(f"📊 Problems: {args.problems}")
    print(f"🎯 GPU IDs: {args.gpu_ids}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Method: Original main.py (30k steps hardcoded)")
    print(f"📈 Trajectories per problem: {args.num_trajectories}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Phase 1: Training
    print(f"\\n🔥 Phase 1: Training models...")
    training_processes = []
    
    for i, problem_id in enumerate(args.problems):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        script_path = os.path.join(args.output_dir, "scripts", f"train_problem_{problem_id}.py")
        log_path = os.path.join(args.output_dir, "logs", f"training_{problem_id}.log")
        
        print(f"📝 Creating training script for problem {problem_id} on GPU {gpu_id}")
        create_single_training_script(problem_id, gpu_id, args.output_dir, script_path)
        
        # Start process
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script_path} > {log_path} 2>&1"
        print(f"🚀 Starting: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        training_processes.append((problem_id, gpu_id, process, log_path))
        
        time.sleep(2)  # Small delay between launches
    
    # Monitor training
    print(f"\\n📊 Monitoring {len(training_processes)} training processes...")
    while True:
        all_done = True
        print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] Training Status:")
        
        for problem_id, gpu_id, process, log_path in training_processes:
            poll = process.poll()
            if poll is None:
                all_done = False
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if "Step" in last_line or "Accuracy" in last_line:
                                print(f"  Problem {problem_id} (GPU {gpu_id}): {last_line}")
                            else:
                                print(f"  Problem {problem_id} (GPU {gpu_id}): Running...")
                except:
                    print(f"  Problem {problem_id} (GPU {gpu_id}): Starting...")
            else:
                print(f"  Problem {problem_id} (GPU {gpu_id}): Completed (exit {poll})")
        
        if all_done:
            break
        time.sleep(30)
    
    # Check training results
    training_results = []
    successful_models = []
    
    for problem_id in args.problems:
        result_path = os.path.join(args.output_dir, f"results/training_result_{problem_id}.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result = json.load(f)
                training_results.append(result)
                if result["success"]:
                    successful_models.append(result)
    
    print(f"\\n📊 Training Results:")
    print(f"✅ Successful: {len(successful_models)}/{len(args.problems)}")
    
    for result in training_results:
        if result["success"]:
            print(f"  Problem {result['prob_index']}: Final accuracy {result.get('final_accuracy', 0):.3f}")
        else:
            print(f"  Problem {result['prob_index']}: {result.get('reason', 'Unknown error')}")
    
    if not successful_models:
        print("❌ No models were successfully trained. Exiting.")
        return
    
    # Phase 2: Trajectory generation
    print(f"\\n📈 Phase 2: Generating trajectories from {len(successful_models)} successful models...")
    trajectory_processes = []
    
    for i, result in enumerate(successful_models):
        problem_id = result['prob_index']
        gpu_id = result['gpu_id']
        script_path = os.path.join(args.output_dir, "scripts", f"trajectory_problem_{problem_id}.py")
        log_path = os.path.join(args.output_dir, "logs", f"trajectory_{problem_id}.log")
        
        print(f"📝 Creating trajectory script for problem {problem_id} on GPU {gpu_id}")
        create_trajectory_script(problem_id, gpu_id, args.output_dir, args.num_trajectories, script_path)
        
        # Start process
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script_path} > {log_path} 2>&1"
        print(f"🚀 Starting: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        trajectory_processes.append((problem_id, gpu_id, process, log_path))
        
        time.sleep(2)
    
    # Monitor trajectory generation
    print(f"\\n📊 Monitoring {len(trajectory_processes)} trajectory processes...")
    while True:
        all_done = True
        print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] Trajectory Status:")
        
        for problem_id, gpu_id, process, log_path in trajectory_processes:
            poll = process.poll()
            if poll is None:
                all_done = False
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            print(f"  Problem {problem_id} (GPU {gpu_id}): {last_line}")
                except:
                    print(f"  Problem {problem_id} (GPU {gpu_id}): Starting...")
            else:
                print(f"  Problem {problem_id} (GPU {gpu_id}): Completed (exit {poll})")
        
        if all_done:
            break
        time.sleep(30)
    
    print("\\n✅ All processes completed!")

if __name__ == "__main__":
    main()