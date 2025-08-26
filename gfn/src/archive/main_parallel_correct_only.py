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
from train import train_model_with_threshold, save_gflownet_trajectories_batch, evaluate_model_is_correct
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
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[6, 7],
                        help="List of GPU IDs to use (e.g., --gpu_ids 6 7)")
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

def evaluate_model_accuracy_is_correct(model, env, num_samples, prob_index, subtask_num=0):
    """Evaluate model accuracy using is_correct field and return success rate."""
    correct = 0
    for _ in range(num_samples):
        eval_state, eval_info = env.reset(options={
            "prob_index": prob_index, 
            "adaptation": True, 
            "subprob_index": subtask_num
        })
        _, log = model.sample_states(eval_state, eval_info, return_log=True, batch_size=1)
        
        # Check is_correct in the final state
        if log.tstates and len(log.tstates) > 0:
            final_state = log.tstates[-1]
            if final_state.get('is_correct', 0) == 1:
                correct += 1
    
    return correct / num_samples

def save_gflownet_trajectories_batch_correct_only(model, env, num_trajectories, save_path_prefix, prob_index, subtask_num=0, batch_size=1000):
    """Save only trajectories with is_correct=1 in any state."""
    import os
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    
    def serialize_dict(d):
        """Convert dictionary values to JSON-serializable format."""
        if isinstance(d, dict):
            return {k: serialize_dict(v) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d.cpu().tolist()
        if isinstance(d, np.ndarray):
            return d.tolist()
        return d
    
    total_saved = 0
    batch_idx = 0
    total_generated = 0
    
    while total_saved < num_trajectories:
        current_batch_size = min(batch_size, num_trajectories - total_saved)
        trajectories = []
        
        while len(trajectories) < current_batch_size:
            state, info = env.reset(options={
                "prob_index": prob_index, 
                "adaptation": True, 
                "subprob_index": subtask_num
            })
            _, log = model.sample_states(state, info, return_log=True, batch_size=1)
            total_generated += 1
            
            # Check if any state has is_correct = 1
            is_successful = False
            for t_state in log.tstates:
                if t_state.get('is_correct', 0) == 1:
                    is_successful = True
                    break
            
            # Only save successful trajectories
            if is_successful:
                trajectory = {
                    "trajectory_id": total_saved + len(trajectories),
                    "problem_id": prob_index,
                    "states": [serialize_dict(t[:5, :5]) for t in log.traj],
                    "actions": [a.cpu().tolist() for a in log.actions],
                    "rewards": [r.cpu().tolist() for r in log.rewards],
                    "states_full": [serialize_dict(s) for s in log.tstates],
                }
                trajectories.append(trajectory)
        
        # Save batch
        batch_file = f"{save_path_prefix}_batch_{batch_idx}.json"
        with open(batch_file, 'w') as f:
            json.dump(trajectories, f)
        
        total_saved += len(trajectories)
        batch_idx += 1
        
        success_rate = len(trajectories) / total_generated * 100 if total_generated > 0 else 0
        print(f"Saved batch {batch_idx}: {len(trajectories)} correct trajectories from {total_generated} generated ({success_rate:.1f}% success rate)")
        print(f"Total saved: {total_saved}/{num_trajectories}")

def train_until_accuracy_threshold(args_dict):
    """Phase 1: Train model until accuracy threshold is reached using is_correct."""
    prob_index = args_dict['prob_index']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    # Set specific GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Process {process_id} starting training for problem {prob_index} on GPU {gpu_id}")
    
    # Initialize wandb for this process
    wandb.init(
        project="gflownet-arc-training-correct",
        name=f"problem_{prob_index}_gpu_{gpu_id}_correct_only",
        config={
            "problem_id": prob_index,
            "gpu_id": gpu_id,
            "process_id": process_id,
            "accuracy_threshold": args.accuracy_threshold,
            "max_training_steps": args.max_training_steps,
            "evaluation_interval": args.evaluation_interval,
            "use_is_correct": True
        }
    )
    
    try:
        # Train model
        model, training_results = train_model_with_threshold(
            prob_index=prob_index,
            args=args,
            device=device,
            accuracy_threshold=args.accuracy_threshold,
            max_steps=args.max_training_steps,
            evaluation_interval=args.evaluation_interval,
            evaluation_fn=evaluate_model_is_correct  # Use is_correct evaluation
        )
        
        if model is None:
            print(f"❌ Failed to train model for problem {prob_index} (couldn't reach {args.accuracy_threshold:.1%} accuracy)")
            return {"prob_index": prob_index, "success": False, "reason": "training_failed"}
        
        # Save trained model
        model_save_path = os.path.join(args.output_dir, f"models/problem_{prob_index}_gpu_{gpu_id}_correct.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_results': training_results,
            'model_config': {
                'prob_index': prob_index,
                'num_actions': args.num_actions,
                'batch_size': args.batch_size,
                'ep_len': args.ep_len,
                'env_mode': args.env_mode
            },
            'timestamp': datetime.now().isoformat(),
            'gpu_id': gpu_id,
            'use_is_correct': True
        }, model_save_path)
        
        print(f"✅ Model for problem {prob_index} trained successfully and saved to {model_save_path}")
        
        result = {
            "prob_index": prob_index,
            "success": True,
            "model_path": model_save_path,
            "final_accuracy": training_results["final_accuracy"],
            "training_steps": training_results["training_steps"],
            "gpu_id": gpu_id
        }
        
        wandb.log({
            "final_accuracy": training_results["final_accuracy"],
            "training_steps": training_results["training_steps"],
            "success": 1
        })
        
        return result
        
    except Exception as e:
        print(f"❌ Error training problem {prob_index}: {e}")
        wandb.log({"success": 0, "error": str(e)})
        return {"prob_index": prob_index, "success": False, "reason": str(e)}
    
    finally:
        wandb.finish()

def generate_trajectories_from_model(args_dict):
    """Phase 2: Generate trajectories from trained model using is_correct filter."""
    prob_index = args_dict['prob_index']
    model_path = args_dict['model_path']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"🔄 Process {process_id} generating trajectories for problem {prob_index} on GPU {gpu_id}")
    
    try:
        # Load trained model
        model, env, checkpoint = load_trained_model(model_path, device)
        
        # Generate trajectories using is_correct filter
        save_path_prefix = os.path.join(args.output_dir, f"problem_{prob_index}", "trajectories")
        
        print(f"📊 Generating {args.num_trajectories} correct trajectories for problem {prob_index}...")
        save_gflownet_trajectories_batch_correct_only(
            model, env, args.num_trajectories, save_path_prefix, prob_index, args.subtask_num
        )
        
        print(f"✅ Successfully generated trajectories for problem {prob_index}")
        
        return {
            "prob_index": prob_index,
            "success": True,
            "trajectories_generated": args.num_trajectories,
            "save_path": save_path_prefix,
            "gpu_id": gpu_id
        }
        
    except Exception as e:
        print(f"❌ Error generating trajectories for problem {prob_index}: {e}")
        return {"prob_index": prob_index, "success": False, "reason": str(e)}

def main():
    """Main execution with correct-only filtering."""
    args = parse_arguments()
    
    print("🚀 Starting GFlowNet Training and Augmentation with is_correct filtering")
    print(f"📊 Problems: {args.problems}")
    print(f"🎯 GPU IDs: {args.gpu_ids}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Accuracy threshold: {args.accuracy_threshold:.1%}")
    print(f"✅ Using is_correct field for success filtering")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: Train models until accuracy threshold
    print(f"\n🔥 Phase 1: Training models to {args.accuracy_threshold:.1%} accuracy...")
    
    # Prepare training tasks
    training_tasks = []
    for i, prob_index in enumerate(args.problems):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        training_tasks.append({
            'prob_index': prob_index,
            'args': args,
            'process_id': i,
            'gpu_id': gpu_id
        })
    
    # Execute training in parallel
    with Pool(processes=len(args.gpu_ids)) as pool:
        training_results = pool.map(train_until_accuracy_threshold, training_tasks)
    
    # Filter successful training results
    successful_models = [r for r in training_results if r["success"]]
    failed_models = [r for r in training_results if not r["success"]]
    
    print(f"\n📊 Training Results:")
    print(f"✅ Successful: {len(successful_models)}/{len(args.problems)}")
    print(f"❌ Failed: {len(failed_models)}/{len(args.problems)}")
    
    if failed_models:
        print("❌ Failed problems:")
        for failure in failed_models:
            print(f"  Problem {failure['prob_index']}: {failure.get('reason', 'Unknown error')}")
    
    if not successful_models:
        print("❌ No models were successfully trained. Exiting.")
        return
    
    # Phase 2: Generate trajectories from successful models
    print(f"\n📈 Phase 2: Generating trajectories from {len(successful_models)} successful models...")
    
    trajectory_tasks = []
    for i, result in enumerate(successful_models):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        trajectory_tasks.append({
            'prob_index': result['prob_index'],
            'model_path': result['model_path'],
            'args': args,
            'process_id': i,
            'gpu_id': gpu_id
        })
    
    # Execute trajectory generation in parallel
    with Pool(processes=len(args.gpu_ids)) as pool:
        trajectory_results = pool.map(generate_trajectories_from_model, trajectory_tasks)
    
    # Summary
    successful_trajectories = [r for r in trajectory_results if r["success"]]
    failed_trajectories = [r for r in trajectory_results if not r["success"]]
    
    print(f"\n📊 Trajectory Generation Results:")
    print(f"✅ Successful: {len(successful_trajectories)}/{len(successful_models)}")
    print(f"❌ Failed: {len(failed_trajectories)}/{len(successful_models)}")
    
    # Save final summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "training_results": training_results,
        "trajectory_results": trajectory_results,
        "summary": {
            "total_problems": len(args.problems),
            "successful_training": len(successful_models),
            "successful_trajectories": len(successful_trajectories),
            "use_is_correct": True
        }
    }
    
    summary_path = os.path.join(args.output_dir, "experiment_summary_correct_only.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Experiment completed! Summary saved to {summary_path}")
    print(f"📁 Data saved to: {args.output_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()