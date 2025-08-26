#!/usr/bin/env python3
"""Simple training with fixed gradient clipping."""

import torch
import os
import numpy as np
from train import initialize_env, initialize_model, evaluate_model_is_correct
from arcle.loaders import ARCLoader
from gflow.log import Log

# Problem to train
PROBLEM_ID = 86
GPU_ID = 6

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"🚀 Training problem {PROBLEM_ID} with fixed gradient clipping")
print(f"Device: {device}")

# Initialize
loader = ARCLoader()
env = initialize_env("entire", PROBLEM_ID, loader)

# Create args object
class Args:
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 1
        self.env_mode = "entire"
        self.num_actions = 5
        self.ep_len = 10
        self.use_offpolicy = False
        self.sampling_method = "prt"
        self.subtask_num = 0

args = Args()
model, optimizer, scheduler = initialize_model(env, args.num_actions, args.batch_size, device, args)

# Initial evaluation
initial_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                           prob_index=PROBLEM_ID, 
                                           subtask=0)
print(f"Initial accuracy: {initial_accuracy:.3f}")

# Training with manual update (avoiding the 0.1 clipping in update_on_policy)
best_accuracy = initial_accuracy
patience_counter = 0
exploration_rate = 1.0

for step in range(10000):
    # Adjust exploration rate
    exploration_rate = max(0.3, 1.0 - (step / 10000) * 0.7)
    
    # Get batch of trajectories
    state, info = env.reset(options={
        "prob_index": PROBLEM_ID, 
        "adaptation": True, 
        "subprob_index": 0
    })
    
    # Sample trajectories
    log = Log()
    model.train()
    
    # Forward pass
    _, sample_log = model.sample_states(state, info, return_log=True, batch_size=args.batch_size)
    
    # Calculate loss (from gflownet_target.py)
    loss = model.compute_loss(sample_log)
    
    # Backward pass with proper gradient clipping
    optimizer.zero_grad()
    loss.backward()
    
    # Monitor gradients before clipping
    total_grad_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_grad_norm_before += param_norm.item() ** 2
    total_grad_norm_before = total_grad_norm_before ** 0.5
    
    # Clip with reasonable value
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Much higher than 0.1
    
    # Monitor gradients after clipping
    total_grad_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_grad_norm_after += param_norm.item() ** 2
    total_grad_norm_after = total_grad_norm_after ** 0.5
    
    optimizer.step()
    scheduler.step()
    
    # Logging
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, grad_before={total_grad_norm_before:.4f}, "
              f"grad_after={total_grad_norm_after:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
    
    # Evaluation
    if step % 500 == 0:
        accuracy = evaluate_model_is_correct(model, env, num_samples=100,
                                           prob_index=PROBLEM_ID, 
                                           subtask=0)
        
        print(f"Step {step}, Accuracy: {accuracy:.3f} (Best: {best_accuracy:.3f})")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'step': step
            }, f"/data/gflownet-llm-additional/models/best_model_problem_{PROBLEM_ID}_fixed.pt")
        else:
            patience_counter += 1
            
            if patience_counter > 10:
                # Decay learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
                patience_counter = 0
                print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")
        
        # Early stopping
        if accuracy >= 0.75:
            print(f"✅ Reached target accuracy!")
            break

print(f"\n✅ Training completed!")
print(f"Initial accuracy: {initial_accuracy:.3f}")
print(f"Best accuracy: {best_accuracy:.3f}")