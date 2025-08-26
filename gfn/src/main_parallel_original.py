import torch
import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool, Process
import multiprocessing
import time
import numpy as np
from tqdm import tqdm
from config import CONFIG
from gflow.utils import seed_everything, setup_wandb
from train import train_model, save_gflownet_trajectories_batch, evaluate_model
import wandb

# ARC 문제 리스트
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
    parser.add_argument("--num_trajectories", type=int, default=30000,
                        help="Number of trajectories per problem")
    parser.add_argument("--subtask_num", type=int, default=CONFIG["SUBTASKNUM"])
    parser.add_argument("--output_dir", type=str, default="/data/gflownet-llm-additional",
                        help="Directory to save trajectories")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[6, 7],
                        help="List of GPU IDs to use (e.g., --gpu_ids 6 7)")
    return parser.parse_args()

def train_single_problem(args_dict):
    """Train a single problem using original main.py approach."""
    prob_index = args_dict['prob_index']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    # Set specific GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 [GPU {gpu_id}] Training problem {prob_index} with original main.py method")
    
    try:
        # Use original training approach
        seed_everything(777)
        
        # Create args object compatible with train_model
        class TrainArgs:
            def __init__(self):
                self.batch_size = args.batch_size
                self.num_epochs = args.num_epochs
                self.env_mode = args.env_mode
                self.num_actions = args.num_actions
                self.ep_len = args.ep_len
                self.use_offpolicy = args.use_offpolicy
                self.sampling_method = args.sampling_method
                self.subtask_num = args.subtask_num
                self.prob_index = prob_index
        
        train_args = TrainArgs()
        
        print(f"[GPU {gpu_id}] Problem {prob_index}: Starting 30k step training...")
        
        # Use original train_model function (30k steps hardcoded)
        model, env = train_model(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=device,
            env_mode=args.env_mode,
            prob_index=prob_index,
            num_actions=args.num_actions,
            args=train_args,
            use_offpolicy=args.use_offpolicy,
            sub_task=args.subtask_num
        )
        
        # Final evaluation using original method
        final_accuracy = evaluate_model(model, env, num_samples=100, prob_index=prob_index, subtask=args.subtask_num)
        print(f"[GPU {gpu_id}] Problem {prob_index}: Final accuracy {final_accuracy:.3f}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f"models/problem_{prob_index}_gpu_{gpu_id}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_accuracy': final_accuracy,
            'problem_id': prob_index,
            'gpu_id': gpu_id,
            'training_method': 'original_main_py',
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        result = {
            "prob_index": prob_index,
            "success": True,
            "final_accuracy": final_accuracy,
            "model_path": model_path,
            "gpu_id": gpu_id,
            "training_method": "original_main_py"
        }
        
        print(f"✅ [GPU {gpu_id}] Problem {prob_index}: Training completed, accuracy {final_accuracy:.3f}")
        return result
        
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] Problem {prob_index}: Training failed - {e}")
        return {"prob_index": prob_index, "success": False, "reason": str(e), "gpu_id": gpu_id}

def generate_trajectories_original(args_dict):
    """Generate trajectories using original batch method."""
    prob_index = args_dict['prob_index']
    model_path = args_dict['model_path']
    args = args_dict['args']
    gpu_id = args_dict['gpu_id']
    
    # Set specific GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"🔄 [GPU {gpu_id}] Generating {args.num_trajectories} trajectories for problem {prob_index}")
    
    try:
        # Load trained model
        from train import initialize_env, initialize_model
        from arcle.loaders import ARCLoader
        
        checkpoint = torch.load(model_path, map_location=device)
        
        loader = ARCLoader()
        env = initialize_env(args.env_mode, prob_index, loader)
        
        # Create args object for model initialization
        class TrainArgs:
            def __init__(self):
                self.batch_size = args.batch_size
                self.num_epochs = args.num_epochs
                self.env_mode = args.env_mode
                self.num_actions = args.num_actions
                self.ep_len = args.ep_len
                self.use_offpolicy = args.use_offpolicy
                self.sampling_method = args.sampling_method
                self.subtask_num = args.subtask_num
        
        train_args = TrainArgs()
        model, _, _ = initialize_model(env, args.num_actions, args.batch_size, device, train_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate trajectories using original batch method
        save_path_prefix = os.path.join(args.output_dir, f"problem_{prob_index}", "trajectories")
        
        print(f"[GPU {gpu_id}] Problem {prob_index}: Generating trajectories...")
        save_gflownet_trajectories_batch(
            model, env, args.num_trajectories, save_path_prefix, 
            prob_index, args.subtask_num, batch_size=1000
        )
        
        print(f"✅ [GPU {gpu_id}] Problem {prob_index}: Generated {args.num_trajectories} trajectories")
        
        # Save summary
        summary = {
            "problem_id": prob_index,
            "num_trajectories": args.num_trajectories,
            "model_accuracy": checkpoint.get('final_accuracy', 0),
            "generation_method": "original_batch",
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = os.path.join(args.output_dir, f"problem_{prob_index}", "summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return {
            "prob_index": prob_index,
            "success": True,
            "trajectories_generated": args.num_trajectories,
            "save_path": save_path_prefix,
            "gpu_id": gpu_id
        }
        
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] Problem {prob_index}: Trajectory generation failed - {e}")
        return {"prob_index": prob_index, "success": False, "reason": str(e), "gpu_id": gpu_id}

def main():
    """Main execution using original main.py approach."""
    args = parse_arguments()
    
    print("🚀 Starting Original Main.py Style Training and Augmentation")
    print(f"📊 Problems: {args.problems}")
    print(f"🎯 GPU IDs: {args.gpu_ids}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Method: Original main.py (30k steps hardcoded)")
    print(f"📈 Trajectories per problem: {args.num_trajectories}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: Train models using original approach
    print(f"\n🔥 Phase 1: Training models with original main.py method...")
    
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
    print("Training models in parallel on available GPUs...")
    with Pool(processes=len(args.gpu_ids)) as pool:
        training_results = pool.map(train_single_problem, training_tasks)
    
    # Filter successful training results
    successful_models = [r for r in training_results if r["success"]]
    failed_models = [r for r in training_results if not r["success"]]
    
    print(f"\n📊 Training Results:")
    print(f"✅ Successful: {len(successful_models)}/{len(args.problems)}")
    print(f"❌ Failed: {len(failed_models)}/{len(args.problems)}")
    
    for result in training_results:
        if result["success"]:
            print(f"  Problem {result['prob_index']}: Final accuracy {result.get('final_accuracy', 0):.3f}")
        else:
            print(f"  Problem {result['prob_index']}: {result.get('reason', 'Unknown error')}")
    
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
            'gpu_id': gpu_id
        })
    
    # Execute trajectory generation in parallel
    print("Generating trajectories in parallel on available GPUs...")
    with Pool(processes=len(args.gpu_ids)) as pool:
        trajectory_results = pool.map(generate_trajectories_original, trajectory_tasks)
    
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
        "method": "original_main_py",
        "training_steps": 30000,  # Hardcoded in original
        "summary": {
            "total_problems": len(args.problems),
            "successful_training": len(successful_models),
            "successful_trajectories": len(successful_trajectories)
        }
    }
    
    summary_path = os.path.join(args.output_dir, "experiment_summary_original.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Experiment completed! Summary saved to {summary_path}")
    print(f"📁 Data saved to: {args.output_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()