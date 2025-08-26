#!/usr/bin/env python3
import torch
import argparse
import json
import os
from datetime import datetime
import subprocess
import time
import numpy as np
from config import CONFIG

# ARC 문제 리스트
ARC_PROBLEMS = [86, 139, 149, 154, 178, 240, 379]

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs='+', type=int, default=ARC_PROBLEMS)
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[6, 7])
    parser.add_argument("--output_dir", type=str, default="/data/gflownet-llm-augmented")
    
    # Training hyperparameters with augmentation support
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--env_mode", type=str, default="entire")
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--ep_len", type=int, default=10)
    parser.add_argument("--num_trajectories", type=int, default=100000)
    parser.add_argument("--accuracy_threshold", type=float, default=0.75)
    parser.add_argument("--max_training_steps", type=int, default=50000)
    parser.add_argument("--evaluation_interval", type=int, default=500)
    parser.add_argument("--evaluation_samples", type=int, default=100)
    parser.add_argument("--min_exploration_rate", type=float, default=0.3, help="Higher min exploration")
    parser.add_argument("--lr_decay_factor", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=10, help="More patience before decay")
    parser.add_argument("--use_is_correct", action="store_true", default=True)
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for learning rate")
    
    # Augmentation parameters
    parser.add_argument("--use_augmentation", action="store_true", default=True, help="Enable data augmentation")
    parser.add_argument("--augmentation_prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--rotation_prob", type=float, default=0.25, help="Probability of rotation augmentation")
    parser.add_argument("--flip_prob", type=float, default=0.25, help="Probability of flip augmentation")
    
    return parser.parse_args()

def create_training_script(problem_id, gpu_id, args, script_path):
    """Create a training script for a single problem with augmentation."""
    script_content = f'''#!/usr/bin/env python3
import sys
sys.path.append('/home/ubuntu/GFN_to_ARC/gfn/src')

import torch
import os
import json
from datetime import datetime
import numpy as np
import random
from train import initialize_env, initialize_model, update_on_policy, evaluate_model_is_correct
from arcle.loaders import ARCLoader

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"🚀 Training problem {problem_id} on GPU {gpu_id} with augmentation")
print(f"Device: {{device}}")

# Initialize
loader = ARCLoader()
env = initialize_env("{args.env_mode}", {problem_id}, loader)

# Create args object with stable hyperparameters
class Args:
    def __init__(self):
        self.batch_size = {args.batch_size}
        self.num_epochs = {args.num_epochs}
        self.env_mode = "{args.env_mode}"
        self.num_actions = {args.num_actions}
        self.ep_len = {args.ep_len}
        self.use_offpolicy = False
        self.sampling_method = "prt"
        self.subtask_num = 0
        self.learning_rate = {args.learning_rate}
        self.gradient_clip = {args.gradient_clip}
        self.warmup_steps = {args.warmup_steps}

# Augmentation functions
def rotate_state(state, k=1):
    """Rotate state by k*90 degrees."""
    if isinstance(state, torch.Tensor):
        if state.dim() >= 2:
            return torch.rot90(state, k, dims=(-2, -1))
        else:
            return state
    else:
        if isinstance(state, np.ndarray) and state.ndim >= 2:
            return np.rot90(state, k, axes=(-2, -1))
        else:
            return state

def flip_state(state, axis):
    """Flip state along axis."""
    if isinstance(state, torch.Tensor):
        if state.dim() >= abs(axis):
            return torch.flip(state, dims=(axis,))
        else:
            return state
    else:
        if isinstance(state, np.ndarray) and state.ndim >= abs(axis):
            return np.flip(state, axis=axis)
        else:
            return state

def augment_state(state, info):
    """Apply random augmentation to state."""
    if not {args.use_augmentation} or random.random() > {args.augmentation_prob}:
        return state, info
    
    # Check if state has sufficient dimensions
    if isinstance(state, torch.Tensor):
        if state.dim() < 2:
            return state, info
    elif isinstance(state, np.ndarray):
        if state.ndim < 2:
            return state, info
    else:
        return state, info
    
    # Deep copy to avoid modifying original
    augmented_state = state.clone() if isinstance(state, torch.Tensor) else state.copy()
    augmented_info = info.copy()
    
    # Apply rotation
    if random.random() < {args.rotation_prob}:
        k = random.randint(1, 3)  # 90, 180, or 270 degrees
        augmented_state = rotate_state(augmented_state, k)
        if 'input' in augmented_info and isinstance(augmented_info['input'], (torch.Tensor, np.ndarray)):
            augmented_info['input'] = rotate_state(augmented_info['input'], k)
        if 'target' in augmented_info and isinstance(augmented_info['target'], (torch.Tensor, np.ndarray)):
            augmented_info['target'] = rotate_state(augmented_info['target'], k)
    
    # Apply flip
    if random.random() < {args.flip_prob}:
        axis = random.choice([-2, -1])  # Horizontal or vertical flip
        augmented_state = flip_state(augmented_state, axis)
        if 'input' in augmented_info and isinstance(augmented_info['input'], (torch.Tensor, np.ndarray)):
            augmented_info['input'] = flip_state(augmented_info['input'], axis)
        if 'target' in augmented_info and isinstance(augmented_info['target'], (torch.Tensor, np.ndarray)):
            augmented_info['target'] = flip_state(augmented_info['target'], axis)
    
    return augmented_state, augmented_info

args = Args()
model, optimizer, scheduler = initialize_model(env, args.num_actions, args.batch_size, device, args)

# Override gradient clipping in update function
torch.nn.utils.clip_grad_norm_ = lambda params, max_norm: torch.nn.utils.clip_grad_norm_(params, {args.gradient_clip})

# Training variables
step = 0
best_accuracy = 0.0
patience_counter = 0
exploration_rate = 1.0
initial_lr = {args.learning_rate}

# Initial evaluation
initial_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                           prob_index={problem_id}, 
                                           subtask=0)
print(f"Initial accuracy: {{initial_accuracy:.3f}}")

# Training log
training_log = []
augmentation_stats = {{"rotations": 0, "flips": 0, "total": 0}}

while step < {args.max_training_steps}:
    # Warmup learning rate
    if step < {args.warmup_steps}:
        warmup_factor = step / {args.warmup_steps}
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * warmup_factor
    
    # Adjust exploration rate more gradually
    exploration_rate = max({args.min_exploration_rate}, 
                          1.0 - (step / {args.max_training_steps}) * (1.0 - {args.min_exploration_rate}))
    
    # Training step
    state, info = env.reset(options={{
        "prob_index": {problem_id}, 
        "adaptation": True, 
        "subprob_index": 0
    }})
    
    # Apply augmentation with proper error handling
    try:
        if isinstance(state, (torch.Tensor, np.ndarray)):
            original_state = state.clone() if isinstance(state, torch.Tensor) else state.copy()
            state, info = augment_state(state, info)
            
            # Track augmentation usage
            if isinstance(state, torch.Tensor):
                if not torch.equal(state, original_state):
                    augmentation_stats["total"] += 1
            else:
                if not np.array_equal(state, original_state):
                    augmentation_stats["total"] += 1
    except Exception as e:
        print(f"Augmentation error at step {{step}}: {{e}}")
        # Continue without augmentation
    
    # Set exploration rate
    if hasattr(model, 'exploration_rate'):
        model.exploration_rate = exploration_rate
    
    # Update with gradient monitoring
    try:
        log = update_on_policy(model, optimizer, scheduler, state, info, args)
        
        # Monitor gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if step % 100 == 0:
            print(f"Step {{step}}: grad_norm={{total_grad_norm:.4f}}, lr={{optimizer.param_groups[0]['lr']:.2e}}, explore={{exploration_rate:.3f}}, aug_count={{augmentation_stats['total']}}")
            
    except Exception as e:
        print(f"Error at step {{step}}: {{e}}")
        break
    
    # Evaluation
    if step % {args.evaluation_interval} == 0:
        accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                           prob_index={problem_id}, 
                                           subtask=0)
        
        # Save to log
        training_log.append({{
            'step': step,
            'accuracy': accuracy,
            'best_accuracy': best_accuracy,
            'lr': optimizer.param_groups[0]['lr'],
            'exploration_rate': exploration_rate,
            'grad_norm': total_grad_norm,
            'augmentations_used': augmentation_stats['total']
        }})
        
        # Check for improvement
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join("{args.output_dir}", 
                                         f"models/best_model_problem_{problem_id}.pt")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({{
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'step': step,
                'problem_id': {problem_id},
                'augmentation_stats': augmentation_stats
            }}, best_model_path)
        else:
            patience_counter += 1
            
            # Decay learning rate if no improvement
            if patience_counter > {args.patience}:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= {args.lr_decay_factor}
                patience_counter = 0
                print(f"Learning rate decayed to {{optimizer.param_groups[0]['lr']}}")
        
        print(f"Step {{step}}, Accuracy: {{accuracy:.3f}} (Best: {{best_accuracy:.3f}})")
        
        # Check if threshold reached
        if accuracy >= {args.accuracy_threshold}:
            print(f"✅ Reached target accuracy {{accuracy:.3f}} >= {args.accuracy_threshold:.3f}")
            break
        
        # Early stopping only if accuracy is very bad for too long
        if step > 5000 and best_accuracy < 0.05:
            print(f"⚠️ Training not improving after 5000 steps. Best accuracy only {{best_accuracy:.3f}}")
            break
    
    step += 1

# Final evaluation
final_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                         prob_index={problem_id}, 
                                         subtask=0)
print(f"Final accuracy: {{final_accuracy:.3f}}")
print(f"Augmentation stats: {{augmentation_stats}}")

# Save training log
log_path = os.path.join("{args.output_dir}", f"logs/training_log_problem_{problem_id}.json")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, 'w') as f:
    json.dump(training_log, f, indent=2)

# Save final results
result = {{
    "problem_id": {problem_id},
    "gpu_id": {gpu_id},
    "initial_accuracy": initial_accuracy,
    "final_accuracy": final_accuracy,
    "best_accuracy": best_accuracy,
    "training_steps": step,
    "success": final_accuracy >= {args.accuracy_threshold} * 0.8,
    "augmentation_stats": augmentation_stats,
    "timestamp": datetime.now().isoformat()
}}

result_path = os.path.join("{args.output_dir}", f"results/result_problem_{problem_id}.json")
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"✅ Training completed for problem {problem_id}")
print(f"Results: Initial={{initial_accuracy:.3f}}, Final={{final_accuracy:.3f}}, Best={{best_accuracy:.3f}}")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

def main():
    args = parse_arguments()
    
    print("🚀 Starting Parallel GFlowNet Training with Augmentation on GPUs 6 and 7")
    print(f"📊 Problems: {args.problems}")
    print(f"🎯 GPU IDs: {args.gpu_ids}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎓 Learning rate: {args.learning_rate} with {args.warmup_steps} warmup steps")
    print(f"✂️ Gradient clipping: {args.gradient_clip}")
    print(f"🔍 Min exploration rate: {args.min_exploration_rate}")
    print(f"⏱️ Patience: {args.patience} intervals")
    print(f"🔄 Augmentation: {'Enabled' if args.use_augmentation else 'Disabled'}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Create training scripts for each problem
    processes = []
    for i, problem_id in enumerate(args.problems):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        script_path = os.path.join(args.output_dir, "scripts", f"train_problem_{problem_id}.py")
        
        print(f"\n📝 Creating training script for problem {problem_id} on GPU {gpu_id}")
        create_training_script(problem_id, gpu_id, args, script_path)
        
        # Start training process
        log_file = os.path.join(args.output_dir, "logs", f"training_problem_{problem_id}.log")
        cmd = f"python {script_path} > {log_file} 2>&1"
        
        print(f"🚀 Starting: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        processes.append((problem_id, gpu_id, process, log_file))
        
        # Small delay between launches
        time.sleep(2)
    
    print(f"\n📊 Monitoring {len(processes)} training processes...")
    
    # Monitor processes
    while True:
        all_done = True
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")
        
        for problem_id, gpu_id, process, log_file in processes:
            poll = process.poll()
            if poll is None:
                all_done = False
                # Check last line of log
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if "Step" in last_line and "Accuracy" in last_line:
                                print(f"  Problem {problem_id} (GPU {gpu_id}): {last_line}")
                            else:
                                print(f"  Problem {problem_id} (GPU {gpu_id}): Running...")
                except:
                    print(f"  Problem {problem_id} (GPU {gpu_id}): Starting...")
            else:
                print(f"  Problem {problem_id} (GPU {gpu_id}): Completed (exit code: {poll})")
        
        if all_done:
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    print("\n✅ All training processes completed!")
    
    # Collect results
    print("\n📊 Collecting results...")
    all_results = []
    for problem_id in args.problems:
        result_path = os.path.join(args.output_dir, "results", f"result_problem_{problem_id}.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result = json.load(f)
                all_results.append(result)
                print(f"  Problem {problem_id}: Initial={result['initial_accuracy']:.3f}, "
                      f"Final={result['final_accuracy']:.3f}, Best={result['best_accuracy']:.3f}")
                if 'augmentation_stats' in result:
                    print(f"    Augmentations used: {result['augmentation_stats']['total']}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": all_results
    }
    
    summary_path = os.path.join(args.output_dir, "training_summary_augmented.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to {summary_path}")

if __name__ == "__main__":
    main()