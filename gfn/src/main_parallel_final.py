import torch
import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool, Process
import multiprocessing
import time
import hashlib
import numpy as np
from config import CONFIG
from gflow.utils import seed_everything, setup_wandb
from train import train_model, save_gflownet_trajectories_batch
import wandb

# ARC 문제 리스트 (요청된 문제들)
ARC_PROBLEMS = [86, 139, 149, 154, 178, 240, 379]

def parse_arguments():
    """Parse command-line arguments for parallel training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--num_epochs", type=int, default=CONFIG["NUM_EPOCHS"])
    parser.add_argument("--env_mode", type=str, default=CONFIG["ENV_MODE"])
    parser.add_argument("--problems", nargs='+', type=int, default=ARC_PROBLEMS,
                        help="List of ARC problem indices to train on")
    parser.add_argument("--num_actions", type=int, default=CONFIG["ACTIONNUM"])
    parser.add_argument("--ep_len", type=int, default=CONFIG["EP_LEN"])
    parser.add_argument("--device", type=int, default=CONFIG["CUDANUM"])
    parser.add_argument("--use_offpolicy", action="store_true", default=False)
    parser.add_argument("--sampling_method", type=str, default="prt", 
                        choices=["prt", "fixed_ratio", "egreedy"])
    parser.add_argument("--num_trajectories", type=int, default=100000,
                        help="Number of trajectories per problem")
    parser.add_argument("--subtask_num", type=int, default=CONFIG["SUBTASKNUM"])
    parser.add_argument("--output_dir", type=str, default="/data/gflownet-llm-additional",
                        help="Directory to save trajectories")
    parser.add_argument("--num_processes", type=int, default=2,
                        help="Number of parallel processes")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N trajectories")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[2, 3],
                        help="List of GPU IDs to use (e.g., --gpu_ids 2 3)")
    parser.add_argument("--accuracy_threshold", type=float, default=0.75,
                        help="Accuracy threshold to start augmentation (default: 75%)")
    parser.add_argument("--max_training_steps", type=int, default=50000,
                        help="Maximum training steps before giving up")
    parser.add_argument("--evaluation_interval", type=int, default=1000,
                        help="Evaluate accuracy every N steps")
    parser.add_argument("--evaluation_samples", type=int, default=100,
                        help="Number of samples for accuracy evaluation")
    return parser.parse_args()

def action_sequence_to_key(actions):
    """Convert action sequence to hashable key for uniqueness tracking."""
    # Convert actions to tuple (more efficient than MD5 hash)
    if hasattr(actions[0], 'tolist'):
        return tuple(a.tolist() if hasattr(a, 'tolist') else a for a in actions)
    else:
        return tuple(actions)

def load_trained_model(model_path, device):
    """Load a trained GFlowNet model for reproduction."""
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"Loading model trained on: {checkpoint['timestamp']}")
    print(f"Problem: {checkpoint['model_config']['prob_index']}")
    print(f"Final accuracy: {checkpoint['training_results']['final_accuracy']:.1%}")
    print(f"Training steps: {checkpoint['training_results']['training_steps']}")
    
    # Initialize model with saved config
    from train import initialize_env, initialize_model
    from arcle.loaders import ARCLoader
    
    config = checkpoint['model_config']
    loader = ARCLoader()
    env = initialize_env(config['env_mode'], config['prob_index'], loader)
    
    # Create dummy args object
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(ep_len=config['ep_len'])
    model, optimizer, scheduler = initialize_model(
        env, config['num_actions'], config['batch_size'], device, args
    )
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, env, checkpoint

def evaluate_model_accuracy(model, env, num_samples, prob_index, subtask_num=0):
    """Evaluate model accuracy and return success rate."""
    correct = 0
    for _ in range(num_samples):
        eval_state, eval_info = env.reset(options={
            "prob_index": prob_index, 
            "adaptation": True, 
            "subprob_index": subtask_num
        })
        eval_s, _ = model.sample_states(eval_state, eval_info, return_log=True, batch_size=1)
        
        eval_s = eval_s.cpu().detach().numpy()[:,:eval_info["input_dim"][0], :eval_info["input_dim"][1]][0]
        answer = np.array(env.unwrapped.answer)
        
        if eval_s.shape != answer.shape:
            eval_s = eval_s[0]
        if np.array_equal(eval_s, answer):
            correct += 1
    
    return correct / num_samples

def train_until_accuracy_threshold(args_dict):
    """Phase 1: Train model until accuracy threshold is reached."""
    prob_index = args_dict['prob_index']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    # Set specific GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"[Process {process_id}] Phase 1: Training problem {prob_index} on {device}")
    
    # Initialize wandb for this process (commented out for tmux display)
    # wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
    # run = wandb.init(
    #     project="gflownet time evaluation",
    #     name=f"problem_{prob_index}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #     config={
    #         "problem_id": prob_index,
    #         "process_id": process_id,
    #         "gpu_id": gpu_id,
    #         "device": str(device),
    #         "phase": "training",
    #         "accuracy_threshold": args.accuracy_threshold,
    #         "max_training_steps": args.max_training_steps,
    #         "evaluation_interval": args.evaluation_interval,
    #         "num_trajectories_target": args.num_trajectories,
    #         "env_mode": args.env_mode,
    #         "num_actions": args.num_actions,
    #         "ep_len": args.ep_len
    #     },
    #     tags=["gfn", "arc", "training", f"problem_{prob_index}"],
    #     reinit=True
    # )
    
    # Set random seed for reproducibility
    seed_everything(777 + prob_index)
    
    # Create output directory for this problem
    problem_dir = os.path.join(args.output_dir, f"problem_{prob_index}")
    os.makedirs(problem_dir, exist_ok=True)
    
    # Initialize model and environment
    from train import initialize_env, initialize_model
    from arcle.loaders import ARCLoader
    
    loader = ARCLoader()
    env = initialize_env(args.env_mode, prob_index, loader)
    model, optimizer, scheduler = initialize_model(env, args.num_actions, args.batch_size, device, args)
    
    # Training loop with accuracy monitoring
    training_start_time = time.time()
    step = 0
    accuracy = 0.0
    
    print(f"[Process {process_id}] Starting training phase for problem {prob_index}")
    print(f"[Process {process_id}] Target accuracy: {args.accuracy_threshold:.1%}")
    
    while step < args.max_training_steps and accuracy < args.accuracy_threshold:
        # Reset environment
        state, info = env.reset(options={
            "prob_index": prob_index, 
            "adaptation": True, 
            "subprob_index": args.subtask_num
        })
        
        # Training step
        from train import update_on_policy
        log = update_on_policy(model, optimizer, scheduler, state, info, args)
        
        step += 1
        
        # Evaluate accuracy periodically
        if step % args.evaluation_interval == 0:
            eval_start_time = time.time()
            accuracy = evaluate_model_accuracy(
                model, env, args.evaluation_samples, prob_index, args.subtask_num
            )
            eval_time = time.time() - eval_start_time
            
            print(f"[Process {process_id}] Step {step}: Accuracy = {accuracy:.3f} ({accuracy:.1%}) - Eval time: {eval_time:.2f}s")
            
            # Log to wandb
            # wandb.log({
            print(f"[GPU-{gpu_id} Problem-{prob_index}] Step {step+1}: Loss={loss:.4f}, Accuracy={acc:.4f}", flush=True)
            # Original wandb log: {
            #     "step": step,
            #     "accuracy": accuracy,
            #     "training_time": time.time() - training_start_time,
            #     "evaluation_time": eval_time,
            #     "loss": log.rewards[-1].item() if hasattr(log, 'rewards') and len(log.rewards) > 0 else 0.0,
            #     "total_flow": log.total_flow.exp().item() if hasattr(log, 'total_flow') else 0.0,
            # })
            
            # Check if threshold reached
            if accuracy >= args.accuracy_threshold:
                training_end_time = time.time()
                training_duration = training_end_time - training_start_time
                print(f"[Process {process_id}] ✅ Accuracy threshold reached! {accuracy:.1%} >= {args.accuracy_threshold:.1%}")
                print(f"[Process {process_id}] Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
                
                # wandb.log({
                #     "threshold_reached": True,
                #     "final_accuracy": accuracy,
                #     "training_duration": training_duration,
                #     "training_steps": step
                # })
                break
    
    if accuracy < args.accuracy_threshold:
        print(f"[Process {process_id}] ⚠️ Training stopped at max steps ({args.max_training_steps}). Final accuracy: {accuracy:.1%}")
        # wandb.log({
        #     "threshold_reached": False,
        #     "final_accuracy": accuracy,
        #     "training_duration": time.time() - training_start_time,
        #     "training_steps": step
        # })
        # wandb.finish()
        return None  # Failed to reach threshold
    
    # Save training results
    training_results = {
        "problem_id": prob_index,
        "final_accuracy": accuracy,
        "training_steps": step,
        "training_duration": time.time() - training_start_time,
        "threshold_reached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(problem_dir, "training_results.json"), 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Save trained model with full reproduction info
    model_path = os.path.join(problem_dir, "trained_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_results': training_results,
        'model_config': {
            'num_actions': args.num_actions,
            'env_mode': args.env_mode,
            'ep_len': args.ep_len,
            'batch_size': args.batch_size,
            'prob_index': prob_index,
            'subtask_num': args.subtask_num
        },
        'training_hyperparameters': {
            'learning_rate': 0.0001,
            'scheduler_T_max': 10000,
            'scheduler_eta_min': 0.00001,
            'gradient_clip_norm': 0.1,
            'accuracy_threshold': args.accuracy_threshold,
            'max_training_steps': args.max_training_steps,
            'evaluation_interval': args.evaluation_interval
        },
        'random_seed': 777 + prob_index,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    # wandb.finish()
    return {"model": model, "env": env, "problem_dir": problem_dir, "training_results": training_results}

def generate_trajectories_with_stats(args_dict):
    """Phase 2: Generate high-quality trajectories with statistics tracking."""
    training_result = args_dict['training_result']
    if training_result is None:
        print(f"[Process {args_dict['process_id']}] Skipping trajectory generation - training failed")
        return
    
    prob_index = args_dict['prob_index']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    model = training_result["model"]
    env = training_result["env"]
    problem_dir = training_result["problem_dir"]
    
    print(f"[Process {process_id}] Phase 2: Generating {args.num_trajectories} trajectories for problem {prob_index}")
    
    # Initialize wandb for augmentation phase
    # wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
    # run = wandb.init(
    #     project="gflownet time evaluation",
    #     name=f"problem_{prob_index}_augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #     config={
    #         "problem_id": prob_index,
    #         "process_id": process_id,
    #         "gpu_id": gpu_id,
    #         "phase": "augmentation",
    #         "num_trajectories": args.num_trajectories,
    #         "training_results": training_result["training_results"]
    #     },
    #     tags=["gfn", "arc", "augmentation", f"problem_{prob_index}"],
    #     reinit=True
    # )
    
    # Statistics tracking - 분리된 추적
    generation_start_time = time.time()
    all_unique_sequences = set()  # 모든 trajectory의 unique sequence
    successful_unique_sequences = set()  # 성공한 trajectory만의 unique sequence
    successful_trajectories = 0
    total_generated = 0
    batch_size = args.checkpoint_interval
    
    def serialize_dict(d):
        """Convert dictionary values to JSON-serializable format."""
        if isinstance(d, dict):
            return {k: serialize_dict(v) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d.cpu().tolist()
        if isinstance(d, np.ndarray):
            return d.tolist()
        return d
    
    # Check for existing progress
    checkpoint_file = os.path.join(problem_dir, "augmentation_checkpoint.json")
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('completed_trajectories', 0)
            all_unique_sequences = set(checkpoint.get('all_unique_sequences', []))
            successful_unique_sequences = set(checkpoint.get('successful_unique_sequences', []))
            successful_trajectories = checkpoint.get('successful_trajectories', 0)
            print(f"[Process {process_id}] Resuming from trajectory {start_idx}")
    
    if start_idx >= args.num_trajectories:
        print(f"[Process {process_id}] Trajectory generation already completed")
        # wandb.finish()
        return
    
    print(f"[Process {process_id}] Generating trajectories {start_idx} to {args.num_trajectories}")
    
    for batch_start in range(start_idx, args.num_trajectories, batch_size):
        batch_end = min(batch_start + batch_size, args.num_trajectories)
        batch_trajectories = []
        batch_start_time = time.time()
        
        for traj_idx in range(batch_start, batch_end):
            state, info = env.reset(options={
                "prob_index": prob_index, 
                "adaptation": True, 
                "subprob_index": args.subtask_num
            })
            _, log = model.sample_states(state, info, return_log=True, batch_size=1)
            
            # Check if trajectory reaches correct answer
            final_state = log.traj[-1] if len(log.traj) > 0 else state
            final_state_np = final_state.cpu().detach().numpy()[:info["input_dim"][0], :info["input_dim"][1]]
            answer = np.array(env.unwrapped.answer)
            
            is_successful = np.array_equal(final_state_np, answer)
            if is_successful:
                successful_trajectories += 1
            
            # Track unique action sequences - 개선된 방식
            action_key = action_sequence_to_key(log.actions)
            all_unique_sequences.add(action_key)
            
            if is_successful:
                successful_unique_sequences.add(action_key)
            
            # Store trajectory data
            trajectory = {
                "trajectory_id": traj_idx,
                "problem_id": prob_index,
                "is_successful": is_successful,
                "action_sequence": [a.cpu().tolist() if hasattr(a, 'cpu') else a for a in log.actions],
                "states": [serialize_dict(t[:5, :5]) for t in log.traj],
                "actions": [a.cpu().tolist() if hasattr(a, 'cpu') else a for a in log.actions],
                "rewards": [r.cpu().tolist() if hasattr(r, 'cpu') else r for r in log.rewards],
                "states_full": [serialize_dict(s) for s in log.tstates],
                "generation_time": time.time()
            }
            batch_trajectories.append(trajectory)
            total_generated += 1
        
        # Save batch
        batch_file = os.path.join(problem_dir, f"trajectories_{batch_start}_{batch_end}.json")
        with open(batch_file, 'w') as f:
            json.dump(batch_trajectories, f)
        
        # Update statistics
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - generation_start_time
        success_rate = successful_trajectories / total_generated if total_generated > 0 else 0
        all_unique_rate = len(all_unique_sequences) / total_generated if total_generated > 0 else 0
        successful_unique_rate = len(successful_unique_sequences) / successful_trajectories if successful_trajectories > 0 else 0
        
        print(f"[Process {process_id}] Batch {batch_start}-{batch_end}: {batch_time:.2f}s")
        print(f"[Process {process_id}] Progress: {total_generated}/{args.num_trajectories}")
        print(f"[Process {process_id}] Success rate: {success_rate:.3f} ({successful_trajectories}/{total_generated})")
        print(f"[Process {process_id}] All unique sequences: {len(all_unique_sequences)} ({all_unique_rate:.3f})")
        print(f"[Process {process_id}] Successful unique sequences: {len(successful_unique_sequences)} ({successful_unique_rate:.3f})")
        
        # Log to wandb - 개선된 메트릭
        # wandb.log({
        print(f"[GPU-{gpu_id} Problem-{prob_index}] Batch {batch_start}-{batch_end}: Success={success_rate:.2%}, Unique={unique_in_batch}/{len(batch_trajectories)}, Total={total_generated}", flush=True)
        # Original wandb log: {
        #     "batch_start": batch_start,
        #     "batch_end": batch_end,
        #     "batch_time": batch_time,
        #     "total_elapsed_time": elapsed_time,
        #     "trajectories_generated": total_generated,
        #     "successful_trajectories": successful_trajectories,
        #     "success_rate": success_rate,
        #     "all_unique_sequences": len(all_unique_sequences),
        #     "all_unique_rate": all_unique_rate,
        #     "successful_unique_sequences": len(successful_unique_sequences),
        #     "successful_unique_rate": successful_unique_rate,
        #     "trajectories_per_second": total_generated / elapsed_time if elapsed_time > 0 else 0
        # })
        
        # Update checkpoint
        checkpoint_data = {
            "problem_id": prob_index,
            "completed_trajectories": batch_end,
            "total_trajectories": args.num_trajectories,
            "successful_trajectories": successful_trajectories,
            "all_unique_sequences": list(all_unique_sequences),
            "successful_unique_sequences": list(successful_unique_sequences),
            "generation_time": elapsed_time,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Clear GPU memory periodically
        if batch_end % 5000 == 0:
            torch.cuda.empty_cache()
    
    # Final statistics
    total_time = time.time() - generation_start_time
    final_success_rate = successful_trajectories / args.num_trajectories
    final_all_unique_rate = len(all_unique_sequences) / args.num_trajectories
    final_successful_unique_rate = len(successful_unique_sequences) / successful_trajectories if successful_trajectories > 0 else 0
    
    # Save final summary
    summary = {
        "problem_id": prob_index,
        "total_trajectories": args.num_trajectories,
        "successful_trajectories": successful_trajectories,
        "success_rate": final_success_rate,
        "all_unique_sequences": len(all_unique_sequences),
        "all_unique_rate": final_all_unique_rate,
        "successful_unique_sequences": len(successful_unique_sequences),
        "successful_unique_rate": final_successful_unique_rate,
        "total_generation_time": total_time,
        "trajectories_per_second": args.num_trajectories / total_time,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(problem_dir, "generation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[Process {process_id}] ✅ Completed trajectory generation for problem {prob_index}")
    print(f"[Process {process_id}] Final stats:")
    print(f"  - Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  - Success rate: {final_success_rate:.3f} ({successful_trajectories}/{args.num_trajectories})")
    print(f"  - All unique sequences: {len(all_unique_sequences)} ({final_all_unique_rate:.3f})")
    print(f"  - Successful unique sequences: {len(successful_unique_sequences)} ({final_successful_unique_rate:.3f})")
    print(f"  - Speed: {args.num_trajectories/total_time:.2f} trajectories/second")
    
    wandb.log({
        "final_success_rate": final_success_rate,
        "final_all_unique_sequences": len(all_unique_sequences),
        "final_all_unique_rate": final_all_unique_rate,
        "final_successful_unique_sequences": len(successful_unique_sequences),
        "final_successful_unique_rate": final_successful_unique_rate,
        "total_generation_time": total_time,
        "final_trajectories_per_second": args.num_trajectories / total_time
    })
    
    # wandb.finish()
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

def train_single_problem(args_dict):
    """Combined function: Train model then generate trajectories."""
    # Phase 1: Training
    training_result = train_until_accuracy_threshold(args_dict)
    
    if training_result is None:
        print(f"[Process {args_dict['process_id']}] Problem {args_dict['prob_index']} failed to reach accuracy threshold")
        return
    
    # Phase 2: Trajectory Generation
    args_dict['training_result'] = training_result
    generate_trajectories_with_stats(args_dict)

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare arguments for each problem with GPU assignment
    problem_args = []
    for i, prob_index in enumerate(args.problems):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]  # Cycle through available GPUs
        problem_args.append({
            'prob_index': prob_index,
            'args': args,
            'process_id': i,
            'gpu_id': gpu_id
        })
    
    # Allow more processes than GPUs (processes will share GPUs)
    if args.num_processes > len(args.gpu_ids):
        print(f"Info: num_processes ({args.num_processes}) > available GPUs ({len(args.gpu_ids)})")
        print(f"Multiple processes will share GPUs through cycling")
    
    # Run parallel training and augmentation
    print(f"Starting 2-phase process on {len(args.problems)} problems with {args.num_processes} processes")
    print(f"Problems: {args.problems}")
    print(f"Available GPUs: {args.gpu_ids}")
    print(f"Accuracy threshold: {args.accuracy_threshold:.1%}")
    print(f"Trajectories per problem: {args.num_trajectories}")
    print(f"Output directory: {args.output_dir}")
    
    # Show GPU assignment
    print("\nGPU Assignment:")
    for i, prob_index in enumerate(args.problems):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        print(f"  Problem {prob_index} -> GPU {gpu_id}")
    
    print("\n🔍 Uniqueness Tracking:")
    print("  - All trajectories: Total unique action sequences")
    print("  - Successful only: Unique sequences from successful trajectories")
    print("  - Improved hash method: Direct tuple conversion (faster than MD5)")
    
    start_time = time.time()
    
    # Use multiprocessing Pool for parallel execution
    with Pool(processes=args.num_processes) as pool:
        # Start all processes simultaneously
        results = []
        for arg_dict in problem_args:
            result = pool.apply_async(train_single_problem, (arg_dict,))
            results.append(result)
        
        # Wait for all processes to complete
        for result in results:
            result.get()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"🎉 Completed all problems!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Target trajectories: {len(args.problems) * args.num_trajectories}")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    # Import numpy here to avoid multiprocessing issues
    import numpy as np
    # Set the multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()