#!/usr/bin/env python3
"""Test accuracy of saved checkpoint models."""

import torch
import os
import json
import numpy as np
from train import initialize_env, initialize_model, evaluate_model, evaluate_model_is_correct
from arcle.loaders import ARCLoader
import argparse

def test_checkpoint(checkpoint_path, num_samples=100, gpu_id=4):
    """Test a single checkpoint's accuracy."""
    print(f"\n🔍 Testing checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU first
    
    # Move to target device after loading
    if 'model_state_dict' in checkpoint:
        for key in checkpoint['model_state_dict']:
            if torch.is_tensor(checkpoint['model_state_dict'][key]):
                checkpoint['model_state_dict'][key] = checkpoint['model_state_dict'][key].to(device)
    
    # Extract model config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        prob_index = config['prob_index']
        num_actions = config['num_actions']
        batch_size = config['batch_size']
        ep_len = config['ep_len']
        env_mode = config['env_mode']
    else:
        # Try to extract problem ID from filename
        import re
        match = re.search(r'problem_(\d+)', os.path.basename(checkpoint_path))
        if match:
            prob_index = int(match.group(1))
            print(f"⚠️ No model config found, extracted problem ID from filename: {prob_index}")
            # Use default values
            num_actions = 5
            batch_size = 32
            ep_len = 10
            env_mode = "entire"
        else:
            print("❌ Could not determine problem ID")
            return None
    
    print(f"Problem ID: {prob_index}")
    print(f"Device: {device}")
    
    # Initialize environment and model
    loader = ARCLoader()
    env = initialize_env(env_mode, prob_index, loader)
    
    # Create args object with all necessary attributes
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(
        ep_len=ep_len,
        env_mode=env_mode,
        batch_size=batch_size,
        num_actions=num_actions,
        use_offpolicy=False,
        sampling_method="prt"
    )
    model, _, _ = initialize_model(env, num_actions, batch_size, device, args)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test with both evaluation methods
    print("\n📊 Evaluating accuracy...")
    
    # Method 1: Original evaluation (grid comparison)
    accuracy_grid = evaluate_model(model, env, num_samples=num_samples, 
                                   prob_index=prob_index, subtask=0)
    
    # Method 2: is_correct evaluation
    accuracy_is_correct = evaluate_model_is_correct(model, env, num_samples=num_samples,
                                                    prob_index=prob_index, subtask=0)
    
    # Get training results if available
    training_results = checkpoint.get('training_results', {})
    saved_accuracy = checkpoint.get('accuracy', training_results.get('final_accuracy', 'N/A'))
    
    results = {
        'checkpoint_path': checkpoint_path,
        'problem_id': prob_index,
        'saved_accuracy': saved_accuracy,
        'current_accuracy_grid': accuracy_grid,
        'current_accuracy_is_correct': accuracy_is_correct,
        'training_steps': training_results.get('training_steps', 'N/A'),
        'timestamp': checkpoint.get('timestamp', 'N/A')
    }
    
    print(f"\n📈 Results:")
    print(f"  Saved accuracy: {saved_accuracy}")
    print(f"  Current accuracy (grid): {accuracy_grid:.3f}")
    print(f"  Current accuracy (is_correct): {accuracy_is_correct:.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, 
                        default="/data/gflownet-llm-additional/models",
                        help="Directory containing checkpoint files")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for evaluation")
    parser.add_argument("--gpu", type=int, default=4,
                        help="GPU ID to use")
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print(f"🚀 Testing checkpoint accuracy on GPU {args.gpu}")
    print(f"📁 Checkpoint directory: {args.checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = []
    if os.path.exists(args.checkpoint_dir):
        for file in os.listdir(args.checkpoint_dir):
            if file.endswith('.pt'):
                checkpoint_files.append(os.path.join(args.checkpoint_dir, file))
    
    checkpoint_files.sort()
    print(f"\n📋 Found {len(checkpoint_files)} checkpoints")
    
    # Test each checkpoint
    all_results = []
    for checkpoint_path in checkpoint_files:
        try:
            result = test_checkpoint(checkpoint_path, args.num_samples, args.gpu)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Error testing {checkpoint_path}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for result in all_results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"  File: {os.path.basename(result['checkpoint_path'])}")
        print(f"  Saved accuracy: {result['saved_accuracy']}")
        print(f"  Current accuracy (grid): {result['current_accuracy_grid']:.3f}")
        print(f"  Current accuracy (is_correct): {result['current_accuracy_is_correct']:.3f}")
        
        # Check for accuracy drop
        if isinstance(result['saved_accuracy'], float):
            grid_drop = result['saved_accuracy'] - result['current_accuracy_grid']
            correct_drop = result['saved_accuracy'] - result['current_accuracy_is_correct']
            if grid_drop > 0.1:
                print(f"  ⚠️ Accuracy drop (grid): {grid_drop:.3f}")
            if correct_drop > 0.1:
                print(f"  ⚠️ Accuracy drop (is_correct): {correct_drop:.3f}")
    
    # Save results
    output_file = os.path.join(args.checkpoint_dir, "../checkpoint_accuracy_test.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()