import torch
import argparse
import json
import os
from datetime import datetime
import time
import numpy as np
from config import CONFIG
from gflow.utils import seed_everything
from train import train_model_with_threshold, save_gflownet_trajectories_batch, evaluate_model_is_correct
from train import initialize_env, initialize_model, update_on_policy
from arcle.loaders import ARCLoader

# ARC 문제 리스트 (요청된 문제들)
ARC_PROBLEMS = [86, 139, 149, 154, 178, 240, 379]

def parse_arguments():
    """Parse command-line arguments."""
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
    parser.add_argument("--sampling_method", type=str, default="prt")
    parser.add_argument("--num_trajectories", type=int, default=100000)
    parser.add_argument("--subtask_num", type=int, default=CONFIG["SUBTASKNUM"])
    parser.add_argument("--output_dir", type=str, default="/data/gflownet-llm-additional")
    parser.add_argument("--gpu_id", type=int, default=6, help="GPU ID to use")
    parser.add_argument("--accuracy_threshold", type=float, default=0.75)
    parser.add_argument("--max_training_steps", type=int, default=50000)
    parser.add_argument("--evaluation_interval", type=int, default=500)
    parser.add_argument("--evaluation_samples", type=int, default=100)
    parser.add_argument("--min_exploration_rate", type=float, default=0.1)
    parser.add_argument("--lr_decay_factor", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_is_correct", action="store_true", default=True)
    return parser.parse_args()

def save_gflownet_trajectories_batch_filtered(model, env, num_trajectories, save_path_prefix, 
                                               prob_index, subtask_num=0, batch_size=1000, 
                                               filter_correct=True):
    """Save trajectories with optional filtering for correct ones."""
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

def train_single_problem(prob_index, args, device):
    """Train a single problem with early stopping and regularization."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for problem {prob_index} on GPU {args.gpu_id}")
    print(f"{'='*60}\n")
    
    try:
        loader = ARCLoader()
        env = initialize_env(args.env_mode, prob_index, loader)
        model, optimizer, scheduler = initialize_model(env, args.num_actions, args.batch_size, device, args)
        
        # Training with improved regularization
        step = 0
        best_accuracy = 0.0
        patience_counter = 0
        exploration_rate = 1.0
        
        # Initial evaluation
        initial_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                                     prob_index=prob_index, 
                                                     subtask=args.subtask_num)
        print(f"Initial accuracy: {initial_accuracy:.3f}")
        
        while step < args.max_training_steps:
            # Adjust exploration rate
            exploration_rate = max(args.min_exploration_rate, 
                                   1.0 - (step / args.max_training_steps) * (1.0 - args.min_exploration_rate))
            
            # Training step
            state, info = env.reset(options={
                "prob_index": prob_index, 
                "adaptation": True, 
                "subprob_index": args.subtask_num
            })
            
            # Add exploration noise
            model.exploration_rate = exploration_rate
            log = update_on_policy(model, optimizer, scheduler, state, info, args)
            
            # Evaluation
            if step % args.evaluation_interval == 0:
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
                        print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")
                
                print(f"Step {step}, Accuracy: {accuracy:.3f} (Best: {best_accuracy:.3f})")
                
                # Check if threshold reached
                if accuracy >= args.accuracy_threshold:
                    print(f"✅ Reached target accuracy {accuracy:.3f} >= {args.accuracy_threshold:.3f}")
                    break
                
                # Early stopping if accuracy drops too much
                if best_accuracy > 0.3 and accuracy < best_accuracy * 0.7:
                    print(f"⚠️ Accuracy dropped significantly. Reverting to best model.")
                    # Load best model
                    checkpoint = torch.load(best_model_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    break
            
            step += 1
        
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
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, 
                                        f"models/problem_{prob_index}_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_results': {
                "prob_index": prob_index,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": final_accuracy,
                "best_accuracy": best_accuracy,
                "training_steps": step
            },
            'model_config': {
                'prob_index': prob_index,
                'num_actions': args.num_actions,
                'batch_size': args.batch_size,
                'ep_len': args.ep_len,
                'env_mode': args.env_mode
            },
            'timestamp': datetime.now().isoformat()
        }, final_model_path)
        
        success = final_accuracy >= args.accuracy_threshold * 0.8
        return {
            "prob_index": prob_index,
            "success": success,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "training_steps": step,
            "model_path": final_model_path
        }
        
    except Exception as e:
        print(f"❌ Error training problem {prob_index}: {e}")
        import traceback
        traceback.print_exc()
        return {"prob_index": prob_index, "success": False, "reason": str(e)}

def generate_trajectories_for_problem(prob_index, model_path, args, device):
    """Generate trajectories from trained model."""
    print(f"\n📊 Generating trajectories for problem {prob_index}...")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']
        
        loader = ARCLoader()
        env = initialize_env(config['env_mode'], config['prob_index'], loader)
        model, _, _ = initialize_model(env, config['num_actions'], config['batch_size'], device, args)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate trajectories with filtering
        save_path_prefix = os.path.join(args.output_dir, f"problem_{prob_index}", "trajectories")
        
        save_gflownet_trajectories_batch_filtered(
            model, env, args.num_trajectories, save_path_prefix, 
            prob_index, args.subtask_num, filter_correct=args.use_is_correct
        )
        
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
        
        return {"prob_index": prob_index, "success": True}
        
    except Exception as e:
        print(f"❌ Error generating trajectories for problem {prob_index}: {e}")
        import traceback
        traceback.print_exc()
        return {"prob_index": prob_index, "success": False, "reason": str(e)}

def main():
    """Main execution with sequential training."""
    args = parse_arguments()
    
    print("🚀 Starting Sequential GFlowNet Training and Augmentation")
    print(f"📊 Problems: {args.problems}")
    print(f"🎯 GPU ID: {args.gpu_id}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Accuracy threshold: {args.accuracy_threshold:.1%}")
    print(f"✅ Using is_correct field: {args.use_is_correct}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Phase 1: Train models sequentially
    print(f"\n🔥 Phase 1: Training models sequentially...")
    training_results = []
    
    for prob_index in args.problems:
        result = train_single_problem(prob_index, args, device)
        training_results.append(result)
        
        # Clear GPU cache between problems
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
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
    
    # Phase 2: Generate trajectories
    print(f"\n📈 Phase 2: Generating trajectories from {len(successful_models)} successful models...")
    trajectory_results = []
    
    for result in successful_models:
        traj_result = generate_trajectories_for_problem(
            result['prob_index'], result['model_path'], args, device
        )
        trajectory_results.append(traj_result)
        
        # Clear GPU cache between problems
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    successful_trajectories = [r for r in trajectory_results if r["success"]]
    
    print(f"\n📊 Trajectory Generation Results:")
    print(f"✅ Successful: {len(successful_trajectories)}/{len(successful_models)}")
    
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
            "use_is_correct": args.use_is_correct
        }
    }
    
    summary_path = os.path.join(args.output_dir, "experiment_summary_sequential.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Experiment completed! Summary saved to {summary_path}")
    print(f"📁 Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()