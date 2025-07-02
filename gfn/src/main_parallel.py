import torch
import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool, Process
import multiprocessing
import time
from config import CONFIG
from gflow.utils import seed_everything, setup_wandb
from train import train_model, save_gflownet_trajectories_batch

# ARC 문제 리스트
ARC_PROBLEMS = [178, 52, 86, 128, 139, 149, 154, 240, 379]

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
    parser.add_argument("--save_trajectories", action="store_true", 
                        help="Save trajectories for each problem")
    parser.add_argument("--num_trajectories", type=int, default=10000,
                        help="Number of trajectories per problem")
    parser.add_argument("--subtask_num", type=int, default=CONFIG["SUBTASKNUM"])
    parser.add_argument("--output_dir", type=str, default="trajectories_output",
                        help="Directory to save trajectories")
    parser.add_argument("--num_processes", type=int, default=2,
                        help="Number of parallel processes")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N trajectories")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[5, 6],
                        help="List of GPU IDs to use (e.g., --gpu_ids 5 6)")
    return parser.parse_args()

def train_single_problem(args_dict):
    """Train GFlowNet on a single ARC problem and save trajectories."""
    prob_index = args_dict['prob_index']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    # Set specific GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"[Process {process_id}] Starting problem {prob_index} on {device}")
    
    # Initialize wandb for this process
    import wandb
    from datetime import datetime
    
    wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
    run = wandb.init(
        project="gfn-arc-parallel",
        name=f"arc_problem_{prob_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "problem_id": prob_index,
            "process_id": process_id,
            "gpu_id": gpu_id,
            "device": str(device),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "num_trajectories": args.num_trajectories,
            "env_mode": args.env_mode,
            "num_actions": args.num_actions,
            "ep_len": args.ep_len
        },
        tags=["gfn", "arc", "parallel", f"problem_{prob_index}"],
        reinit=True
    )
    
    # Set random seed for reproducibility
    seed_everything(777 + prob_index)
    
    # Create output directory for this problem
    problem_dir = os.path.join(args.output_dir, f"problem_{prob_index}")
    os.makedirs(problem_dir, exist_ok=True)
    
    # Check if we already have some trajectories
    checkpoint_file = os.path.join(problem_dir, "checkpoint.json")
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('completed_trajectories', 0)
            print(f"[Process {process_id}] Resuming from trajectory {start_idx}")
    
    if start_idx >= args.num_trajectories:
        print(f"[Process {process_id}] Problem {prob_index} already completed")
        return
    
    # Train model
    print(f"[Process {process_id}] Training model for problem {prob_index}")
    model, env = train_model(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=device,
        env_mode=args.env_mode,
        prob_index=prob_index,
        num_actions=args.num_actions,
        args=args,
        use_offpolicy=args.use_offpolicy,
        sub_task=args.subtask_num
    )
    
    # Generate trajectories in batches
    print(f"[Process {process_id}] Generating {args.num_trajectories - start_idx} trajectories")
    batch_size = args.checkpoint_interval
    all_trajectories = []
    
    for batch_start in range(start_idx, args.num_trajectories, batch_size):
        batch_end = min(batch_start + batch_size, args.num_trajectories)
        batch_trajectories = []
        
        print(f"[Process {process_id}] Problem {prob_index}: Generating trajectories {batch_start} to {batch_end}")
        
        for traj_idx in range(batch_start, batch_end):
            state, info = env.reset(options={
                "prob_index": prob_index, 
                "adaptation": True, 
                "subprob_index": args.subtask_num
            })
            _, log = model.sample_states(state, info, return_log=True, batch_size=1)
            
            def serialize_dict(d):
                """Convert dictionary values to JSON-serializable format."""
                if isinstance(d, dict):
                    return {k: serialize_dict(v) for k, v in d.items()}
                if isinstance(d, torch.Tensor):
                    return d.cpu().tolist()
                if isinstance(d, np.ndarray):
                    return d.tolist()
                return d
            
            trajectory = {
                "trajectory_id": traj_idx,
                "problem_id": prob_index,
                "states": [serialize_dict(t[:5, :5]) for t in log.traj],
                "actions": [a.cpu().tolist() for a in log.actions],
                "rewards": [r.cpu().tolist() for r in log.rewards],
                "states_full": [serialize_dict(s) for s in log.tstates],
            }
            batch_trajectories.append(trajectory)
        
        # Save batch
        batch_file = os.path.join(problem_dir, f"trajectories_{batch_start}_{batch_end}.json")
        with open(batch_file, 'w') as f:
            json.dump(batch_trajectories, f)
        
        # Update checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'problem_id': prob_index,
                'completed_trajectories': batch_end,
                'total_trajectories': args.num_trajectories,
                'last_updated': datetime.now().isoformat()
            }, f)
        
        all_trajectories.extend(batch_trajectories)
        
        # Clear GPU memory periodically
        if batch_end % 5000 == 0:
            torch.cuda.empty_cache()
    
    # Save summary
    summary_file = os.path.join(problem_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'problem_id': prob_index,
            'total_trajectories': len(all_trajectories),
            'num_batches': (args.num_trajectories + batch_size - 1) // batch_size,
            'completed': datetime.now().isoformat()
        }, f)
    
    print(f"[Process {process_id}] Completed problem {prob_index}: {len(all_trajectories)} trajectories")
    
    # Close wandb run
    wandb.finish()
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

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
    
    # Update num_processes to match number of available GPUs if not specified
    if args.num_processes > len(args.gpu_ids):
        print(f"Warning: num_processes ({args.num_processes}) > available GPUs ({len(args.gpu_ids)})")
        print(f"Setting num_processes to {len(args.gpu_ids)}")
        args.num_processes = len(args.gpu_ids)
    
    # Run parallel training
    print(f"Starting parallel training on {len(args.problems)} problems with {args.num_processes} processes")
    print(f"Problems: {args.problems}")
    print(f"Available GPUs: {args.gpu_ids}")
    print(f"Trajectories per problem: {args.num_trajectories}")
    print(f"Output directory: {args.output_dir}")
    
    # Show GPU assignment
    print("\nGPU Assignment:")
    for i, prob_index in enumerate(args.problems):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        print(f"  Problem {prob_index} -> GPU {gpu_id}")
    
    start_time = time.time()
    
    # Use multiprocessing Pool for parallel execution
    with Pool(processes=args.num_processes) as pool:
        pool.map(train_single_problem, problem_args)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Completed all problems!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Total trajectories generated: {len(args.problems) * args.num_trajectories}")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Import numpy here to avoid multiprocessing issues
    import numpy as np
    # Set the multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()