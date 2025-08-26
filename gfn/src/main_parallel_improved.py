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
from tqdm import tqdm
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
    parser.add_argument("--evaluation_interval", type=int, default=500,
                        help="Evaluate accuracy every N steps")
    parser.add_argument("--evaluation_samples", type=int, default=100,
                        help="Number of samples for accuracy evaluation")
    parser.add_argument("--min_exploration_rate", type=float, default=0.1,
                        help="Minimum exploration rate to maintain diversity")
    parser.add_argument("--lr_decay_factor", type=float, default=0.95,
                        help="Learning rate decay factor when accuracy plateaus")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping (in evaluation intervals)")
    parser.add_argument("--use_is_correct", action="store_true", default=True,
                        help="Use is_correct field for evaluation and filtering")
    return parser.parse_args()

def save_gflownet_trajectories_batch_filtered(model, env, num_trajectories, save_path_prefix, 
                                               prob_index, subtask_num=0, batch_size=1000, 
                                               filter_correct=True):
    """Save trajectories with optional filtering for correct ones."""
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
    unique_sequences = set()  # Track unique action sequences
    
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
            
            # Check if we should save this trajectory
            should_save = True
            
            if filter_correct:
                # Check if any state has is_correct = 1
                is_successful = False
                for t_state in log.tstates:
                    if t_state.get('is_correct', 0) == 1:
                        is_successful = True
                        break
                should_save = is_successful
            
            # Also check for uniqueness
            if should_save:
                action_seq = tuple(a.cpu().tolist() for a in log.actions)
                if action_seq in unique_sequences:
                    continue  # Skip duplicate sequences
                unique_sequences.add(action_seq)
            
            # Save trajectory
            if should_save:
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
        
        success_rate = len(unique_sequences) / total_generated * 100 if total_generated > 0 else 0
        uniqueness_rate = len(unique_sequences) / total_saved * 100 if total_saved > 0 else 0
        
        print(f"Saved batch {batch_idx}: {len(trajectories)} trajectories")
        print(f"  Total saved: {total_saved}/{num_trajectories}")
        print(f"  Success rate: {success_rate:.1f}% ({len(unique_sequences)}/{total_generated})")
        print(f"  Uniqueness: {uniqueness_rate:.1f}% ({len(unique_sequences)} unique sequences)")

def train_with_early_stopping(args_dict):
    """Phase 1: Train model with early stopping and regularization."""
    prob_index = args_dict['prob_index']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    # Set specific GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Process {process_id} starting training for problem {prob_index} on GPU {gpu_id}")
    
    # Initialize wandb for this process (disabled due to server errors)
    use_wandb = False  # Disabled due to wandb server 500 errors
    if use_wandb:
        wandb.init(
            project="gflownet-arc-training-improved",
            name=f"problem_{prob_index}_gpu_{gpu_id}_improved",
            config={
                "problem_id": prob_index,
                "gpu_id": gpu_id,
                "process_id": process_id,
                "accuracy_threshold": args.accuracy_threshold,
                "max_training_steps": args.max_training_steps,
                "evaluation_interval": args.evaluation_interval,
                "use_is_correct": args.use_is_correct,
                "min_exploration_rate": args.min_exploration_rate,
                "lr_decay_factor": args.lr_decay_factor,
                "patience": args.patience
            }
        )
    
    try:
        # Modified training with early stopping and regularization
        from train import initialize_env, initialize_model, update_on_policy
        from arcle.loaders import ARCLoader
        
        loader = ARCLoader()
        env = initialize_env(args.env_mode, prob_index, loader)
        model, optimizer, scheduler = initialize_model(env, args.num_actions, args.batch_size, device, args)
        
        # Training with improved regularization
        step = 0
        best_accuracy = 0.0
        patience_counter = 0
        initial_lr = optimizer.param_groups[0]['lr']
        exploration_rate = 1.0
        
        # Initial evaluation
        initial_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                                     prob_index=prob_index, 
                                                     subtask=args.subtask_num)
        print(f"Initial accuracy: {initial_accuracy:.3f}")
        if use_wandb:
            wandb.log({"initial_accuracy": initial_accuracy})
        
        # Use tqdm for progress tracking
        pbar = tqdm(range(args.max_training_steps), desc=f"Problem {prob_index} (GPU {gpu_id})")
        
        for step in pbar:
            # Adjust exploration rate
            exploration_rate = max(args.min_exploration_rate, 
                                   1.0 - (step / args.max_training_steps) * (1.0 - args.min_exploration_rate))
            
            # Training step with exploration
            state, info = env.reset(options={
                "prob_index": prob_index, 
                "adaptation": True, 
                "subprob_index": args.subtask_num
            })
            
            # Add exploration noise
            if hasattr(model, 'exploration_rate'):
                model.exploration_rate = exploration_rate
            log = update_on_policy(model, optimizer, scheduler, state, info, args)
            
            # Calculate loss for display
            current_loss = log.rewards[-1].item() if hasattr(log.rewards[-1], 'item') else log.rewards[-1]
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{current_loss:.4f}",
                'Best_Acc': f"{best_accuracy:.3f}",
                'Explore': f"{exploration_rate:.3f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Evaluation
            if step % args.evaluation_interval == 0 and step > 0:
                accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                                     prob_index=prob_index, 
                                                     subtask=args.subtask_num)
                
                # Check for improvement
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(args.output_dir, 
                                                   f"models/best_model_problem_{prob_index}.pt")
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'accuracy': best_accuracy,
                        'step': step
                    }, best_model_path)
                else:
                    patience_counter += 1
                    
                    # Decay learning rate if no improvement
                    if patience_counter > args.patience:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= args.lr_decay_factor
                        patience_counter = 0
                        print(f"\n[GPU {gpu_id}] Problem {prob_index}: Learning rate decayed to {optimizer.param_groups[0]['lr']:.2e}")
                
                print(f"\n[GPU {gpu_id}] Problem {prob_index} - Step {step}: Accuracy {accuracy:.3f} (Best: {best_accuracy:.3f})")
                if use_wandb:
                    wandb.log({
                        "step": step,
                        "accuracy": accuracy,
                        "best_accuracy": best_accuracy,
                        "exploration_rate": exploration_rate,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })
                
                # Check if threshold reached
                if accuracy >= args.accuracy_threshold:
                    print(f"\n✅ [GPU {gpu_id}] Problem {prob_index}: Reached target accuracy {accuracy:.3f} >= {args.accuracy_threshold:.3f}")
                    break
                
                # Early stopping if accuracy drops too much
                if best_accuracy > 0.3 and accuracy < best_accuracy * 0.7:
                    print(f"\n⚠️ [GPU {gpu_id}] Problem {prob_index}: Accuracy dropped significantly. Reverting to best model.")
                    # Load best model
                    checkpoint = torch.load(best_model_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    break
        
        pbar.close()
        
        # Final evaluation
        final_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                                   prob_index=prob_index, 
                                                   subtask=args.subtask_num)
        print(f"Final accuracy: {final_accuracy:.3f}")
        
        # Use best model if it's better
        if best_accuracy > final_accuracy and os.path.exists(best_model_path):
            print(f"Loading best model with accuracy {best_accuracy:.3f}")
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            final_accuracy = best_accuracy
        
        result = {
            "prob_index": prob_index,
            "success": final_accuracy >= args.accuracy_threshold * 0.8,  # 80% of threshold
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "training_steps": step,
            "gpu_id": gpu_id
        }
        
        if use_wandb:
            wandb.log({
                "final_accuracy": final_accuracy,
                "training_steps": step,
                "success": result["success"]
            })
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, 
                                        f"models/problem_{prob_index}_gpu_{gpu_id}_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_results': result,
            'model_config': {
                'prob_index': prob_index,
                'num_actions': args.num_actions,
                'batch_size': args.batch_size,
                'ep_len': args.ep_len,
                'env_mode': args.env_mode
            },
            'timestamp': datetime.now().isoformat(),
            'gpu_id': gpu_id,
            'use_is_correct': args.use_is_correct
        }, final_model_path)
        
        result['model_path'] = final_model_path
        return result
        
    except Exception as e:
        print(f"❌ Error training problem {prob_index}: {e}")
        if use_wandb:
            wandb.log({"success": 0, "error": str(e)})
        return {"prob_index": prob_index, "success": False, "reason": str(e)}
    
    finally:
        if use_wandb:
            wandb.finish()

def generate_trajectories_from_model(args_dict):
    """Phase 2: Generate trajectories from trained model."""
    prob_index = args_dict['prob_index']
    model_path = args_dict['model_path']
    args = args_dict['args']
    process_id = args_dict['process_id']
    gpu_id = args_dict['gpu_id']
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"🔄 Process {process_id} generating trajectories for problem {prob_index} on GPU {gpu_id}")
    
    try:
        # Load trained model
        from train import initialize_env, initialize_model
        from arcle.loaders import ARCLoader
        
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']
        
        loader = ARCLoader()
        env = initialize_env(config['env_mode'], config['prob_index'], loader)
        model, _, _ = initialize_model(env, config['num_actions'], config['batch_size'], device, args)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate trajectories with filtering
        save_path_prefix = os.path.join(args.output_dir, f"problem_{prob_index}", "trajectories")
        
        print(f"📊 Generating {args.num_trajectories} trajectories for problem {prob_index}...")
        
        # Use tqdm for trajectory generation progress
        total_saved = 0
        batch_size = 1000
        total_generated = 0
        unique_sequences = set()
        
        pbar = tqdm(total=args.num_trajectories, desc=f"Trajectories P{prob_index} (GPU {gpu_id})")
        
        while total_saved < args.num_trajectories:
            current_batch_size = min(batch_size, args.num_trajectories - total_saved)
            trajectories = []
            
            while len(trajectories) < current_batch_size:
                state, info = env.reset(options={
                    "prob_index": prob_index, 
                    "adaptation": True, 
                    "subprob_index": args.subtask_num
                })
                _, log = model.sample_states(state, info, return_log=True, batch_size=1)
                total_generated += 1
                
                # Check if we should save this trajectory
                should_save = True
                
                if args.use_is_correct:
                    # Check if any state has is_correct = 1
                    is_successful = False
                    for t_state in log.tstates:
                        if t_state.get('is_correct', 0) == 1:
                            is_successful = True
                            break
                    should_save = is_successful
                
                # Also check for uniqueness
                if should_save:
                    action_seq = tuple(a.cpu().tolist() for a in log.actions)
                    if action_seq in unique_sequences:
                        continue  # Skip duplicate sequences
                    unique_sequences.add(action_seq)
                
                # Save trajectory
                if should_save:
                    def serialize_dict(d):
                        if isinstance(d, dict):
                            return {k: serialize_dict(v) for k, v in d.items()}
                        if isinstance(d, torch.Tensor):
                            return d.cpu().tolist()
                        if isinstance(d, np.ndarray):
                            return d.tolist()
                        return d
                    
                    trajectory = {
                        "trajectory_id": total_saved + len(trajectories),
                        "problem_id": prob_index,
                        "states": [serialize_dict(t[:5, :5]) for t in log.traj],
                        "actions": [a.cpu().tolist() for a in log.actions],
                        "rewards": [r.cpu().tolist() for r in log.rewards],
                        "states_full": [serialize_dict(s) for s in log.tstates],
                    }
                    trajectories.append(trajectory)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': f"{len(unique_sequences)}/{total_generated}",
                        'Rate': f"{len(unique_sequences)/total_generated*100:.1f}%"
                    })
            
            # Save batch
            os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
            batch_file = f"{save_path_prefix}_batch_{total_saved//batch_size}.json"
            with open(batch_file, 'w') as f:
                json.dump(trajectories, f)
            
            total_saved += len(trajectories)
        
        pbar.close()
        
        print(f"✅ Successfully generated trajectories for problem {prob_index}")
        
        # Save summary
        summary = {
            "problem_id": prob_index,
            "num_trajectories": args.num_trajectories,
            "filter_correct": args.use_is_correct,
            "model_accuracy": checkpoint.get('training_results', {}).get('final_accuracy', 0),
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = os.path.join(args.output_dir, f"problem_{prob_index}", "summary.json")
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
        print(f"❌ Error generating trajectories for problem {prob_index}: {e}")
        return {"prob_index": prob_index, "success": False, "reason": str(e)}

def main():
    """Main execution with improved training strategy."""
    args = parse_arguments()
    
    print("🚀 Starting Improved GFlowNet Training and Augmentation")
    print(f"📊 Problems: {args.problems}")
    print(f"🎯 GPU IDs: {args.gpu_ids}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Accuracy threshold: {args.accuracy_threshold:.1%}")
    print(f"✅ Using is_correct field: {args.use_is_correct}")
    print(f"🔍 Min exploration rate: {args.min_exploration_rate}")
    print(f"📉 LR decay factor: {args.lr_decay_factor}")
    print(f"⏱️ Patience: {args.patience} intervals")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: Train models with improved strategy
    print(f"\n🔥 Phase 1: Training models with early stopping and regularization...")
    
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
    
    # Execute training in parallel with real-time logging
    print("Training models in parallel on available GPUs...")
    with Pool(processes=len(args.gpu_ids)) as pool:
        training_results = pool.map(train_with_early_stopping, training_tasks)
    
    # Filter successful training results
    successful_models = [r for r in training_results if r["success"]]
    failed_models = [r for r in training_results if not r["success"]]
    
    print(f"\n📊 Training Results:")
    print(f"✅ Successful: {len(successful_models)}/{len(args.problems)}")
    print(f"❌ Failed: {len(failed_models)}/{len(args.problems)}")
    
    for result in training_results:
        if result["success"]:
            print(f"  Problem {result['prob_index']}: "
                  f"Initial: {result.get('initial_accuracy', 0):.3f}, "
                  f"Final: {result.get('final_accuracy', 0):.3f}, "
                  f"Best: {result.get('best_accuracy', 0):.3f}")
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
            'process_id': i,
            'gpu_id': gpu_id
        })
    
    # Execute trajectory generation in parallel
    print("Generating trajectories in parallel on available GPUs...")
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
            "use_is_correct": args.use_is_correct,
            "improvements": [
                "Early stopping to prevent overfitting",
                "Learning rate decay on plateau",
                "Minimum exploration rate maintained",
                "Best model checkpointing",
                "Unique trajectory filtering"
            ]
        }
    }
    
    summary_path = os.path.join(args.output_dir, "experiment_summary_improved.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Experiment completed! Summary saved to {summary_path}")
    print(f"📁 Data saved to: {args.output_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()