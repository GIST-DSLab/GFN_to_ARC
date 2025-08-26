#!/usr/bin/env python3
"""Debug why accuracy drops to 0."""

import torch
import numpy as np
from train import initialize_env, initialize_model, update_on_policy, evaluate_model_is_correct
from arcle.loaders import ARCLoader
import matplotlib.pyplot as plt

# Problem to debug
PROBLEM_ID = 86
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

print(f"🔍 Debugging training for problem {PROBLEM_ID}")
print(f"Device: {device}")

# Initialize
loader = ARCLoader()
env = initialize_env("entire", PROBLEM_ID, loader)

# Create args object
class Args:
    def __init__(self):
        self.batch_size = 1  # Start with batch size 1
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
print("\n📊 Initial evaluation...")
initial_accuracy = evaluate_model_is_correct(model, env, num_samples=100, 
                                           prob_index=PROBLEM_ID, 
                                           subtask=0)
print(f"Initial accuracy: {initial_accuracy:.3f}")

# Analyze model outputs before training
print("\n🔍 Analyzing initial model outputs...")
with torch.no_grad():
    for i in range(5):
        state, info = env.reset(options={
            "prob_index": PROBLEM_ID, 
            "adaptation": True, 
            "subprob_index": 0
        })
        
        # Get model predictions
        _, log = model.sample_states(state, info, return_log=True, batch_size=1)
        
        print(f"\nSample {i+1}:")
        print(f"  Actions taken: {len(log.actions)}")
        print(f"  Final reward: {log.rewards[-1].item() if log.rewards else 'None'}")
        print(f"  Is correct: {log.tstates[-1].get('is_correct', 0) if log.tstates else 'None'}")
        
        # Check logits
        if hasattr(model, 'actor_model'):
            actor_out = model.actor_model(state.to(device))
            logits = actor_out.squeeze(0).cpu().numpy()
            print(f"  Initial logits: {logits}")
            print(f"  Logit stats: mean={logits.mean():.3f}, std={logits.std():.3f}, max={logits.max():.3f}, min={logits.min():.3f}")

# Train for a few steps and monitor
print("\n🔄 Training and monitoring...")
losses = []
grad_norms = []
accuracies = []

for step in range(200):
    # Training step
    state, info = env.reset(options={
        "prob_index": PROBLEM_ID, 
        "adaptation": True, 
        "subprob_index": 0
    })
    
    # Get loss before update
    with torch.no_grad():
        if hasattr(model, 'actor_model'):
            actor_out = model.actor_model(state.to(device))
            pre_logits = actor_out.clone()
    
    # Update
    log = update_on_policy(model, optimizer, scheduler, state, info, args)
    
    # Monitor gradients
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    grad_norms.append(total_grad_norm)
    
    # Get loss value if available
    if hasattr(log, 'loss') and log.loss is not None:
        losses.append(log.loss.item() if torch.is_tensor(log.loss) else log.loss)
    else:
        losses.append(0)
    
    # Check logits after update
    with torch.no_grad():
        if hasattr(model, 'actor_model'):
            actor_out = model.actor_model(state.to(device))
            post_logits = actor_out
            logit_change = (post_logits - pre_logits).abs().max().item()
            
            if step % 20 == 0:
                print(f"\nStep {step}:")
                print(f"  Grad norm: {total_grad_norm:.4f}")
                print(f"  Loss: {losses[-1]:.4f}")
                print(f"  Logit change: {logit_change:.6f}")
                print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Evaluate every 50 steps
    if step % 50 == 0:
        accuracy = evaluate_model_is_correct(model, env, num_samples=20,  # Fewer samples for speed
                                           prob_index=PROBLEM_ID, 
                                           subtask=0)
        accuracies.append((step, accuracy))
        print(f"  Accuracy: {accuracy:.3f}")

# Plot results
print("\n📊 Saving diagnostic plots...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot losses
ax1.plot(losses)
ax1.set_title('Loss over steps')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.grid(True)

# Plot gradient norms
ax2.plot(grad_norms)
ax2.set_title('Gradient norm over steps')
ax2.set_xlabel('Step')
ax2.set_ylabel('Gradient norm')
ax2.set_yscale('log')
ax2.grid(True)

# Plot accuracies
if accuracies:
    steps, accs = zip(*accuracies)
    ax3.plot(steps, accs, 'o-')
    ax3.set_title('Accuracy over steps')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True)

plt.tight_layout()
plt.savefig('/data/gflownet-llm-additional/debug_training_plots.png')
print("Plots saved to /data/gflownet-llm-additional/debug_training_plots.png")

# Final analysis
print("\n📊 Final analysis:")
print(f"Average gradient norm: {np.mean(grad_norms):.4f}")
print(f"Max gradient norm: {np.max(grad_norms):.4f}")
print(f"Average loss: {np.mean(losses):.4f}")
print(f"Final accuracy: {accuracies[-1][1] if accuracies else 'N/A'}")

# Check if model weights changed
print("\n🔍 Checking if model weights changed...")
initial_weights = []
for name, param in model.named_parameters():
    initial_weights.append(param.data.clone())

# Train one more step
state, info = env.reset(options={
    "prob_index": PROBLEM_ID, 
    "adaptation": True, 
    "subprob_index": 0
})
log = update_on_policy(model, optimizer, scheduler, state, info, args)

# Check weight changes
total_change = 0
for i, (name, param) in enumerate(model.named_parameters()):
    change = (param.data - initial_weights[i]).abs().max().item()
    total_change += change
    if i < 5:  # Print first 5 layers
        print(f"  {name}: max change = {change:.6f}")

print(f"\nTotal weight change: {total_change:.6f}")

print("\n✅ Debug complete!")