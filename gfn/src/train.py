import torch
import json
import numpy as np
import wandb

from replay_buffer import ReplayBuffer
from gflow.utils import compute_reward_with_penalty, detect_cycle, trajectory_balance_loss
from policy_target import EnhancedMLPForwardPolicy
from gflow.gflownet_target import GFlowNet
from ARCenv.wrapper import env_return
from arcle.loaders import ARCLoader
from config import CONFIG


def save_gflownet_trajectories(num_trajectories, save_path, args):
    """Save GFlowNet trajectories to a file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, env = train_model(
        num_epochs=1, batch_size=1, device=device, 
        env_mode=args.env_mode, prob_index=args.prob_index, 
        num_actions=args.num_actions, args=args, use_offpolicy=False
    )

    trajectories = []
    for _ in range(num_trajectories):
        state, info = env.reset(options={"prob_index": args.prob_index, "adaptation": True, "subprob_index": args.subtask_num})
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

        trajectories.append({
            "states": [serialize_dict(t[:5, :5]) for t in log.traj],
            "actions": [a.cpu().tolist() for a in log.actions],
            "rewards": [r.cpu().tolist() for r in log.rewards],
            "states_full": [serialize_dict(s) for s in log.tstates],
        })

    with open(save_path, 'w') as f:
        json.dump(trajectories, f)
    print(f"Saved {num_trajectories} trajectories to {save_path}")


def initialize_env(env_mode, prob_index, loader):  
    """Initialize ARC environment."""
    return env_return(render=None, data=loader, options=None, batch_size=1, mode=env_mode)


def initialize_model(env, num_actions, batch_size, device, args):
    """Initialize model and optimizer."""
    forward_policy = EnhancedMLPForwardPolicy(
        state_dim=30, hidden_dim=256, num_actions=num_actions,
        batch_size=batch_size, embedding_dim=32, ep_len=args.ep_len
    ).to(device)

    model = GFlowNet(
        forward_policy=forward_policy,
        backward_policy=None,
        total_flow=torch.nn.parameter.Parameter(torch.tensor(1.0).to(device)),
        env=env, device=device, env_style=args.env_mode, num_actions=num_actions, ep_len=args.ep_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0.00001)
    return model, optimizer, scheduler


def update_on_policy(model, optimizer, scheduler, state, info, args):
    """Perform an on-policy training update."""
    result = model.sample_states(state, info, return_log=True, batch_size=1)
    log = result[1]

    # Compute loss
    rewards = compute_reward_with_penalty(log.traj, log.rewards[-1])
    loss, _, _ = trajectory_balance_loss(
        log.total_flow, rewards, log.fwd_probs, log.back_probs
    )

    # Model update
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    # scheduler.step()
    optimizer.zero_grad()

    return log


def update_off_policy(model, optimizer, scheduler, replay_buffer, batch_size, args):
    """Perform an off-policy training update."""
    if len(replay_buffer) < batch_size:
        return  # Skip if not enough data in buffer

    states, actions, rewards, log_probs, back_probs, trajs = replay_buffer.batch_sample(batch_size)
    loss, _, _ = trajectory_balance_loss(
        model.total_flow, rewards, log_probs, back_probs
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    scheduler.step()


def evaluate_model(model, env, num_samples=100, epoch=0, prob_index=178, subtask = 0):
    """Evaluate model accuracy."""
    correct = 0
    for _ in range(num_samples):
        eval_state, eval_info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": subtask})
        eval_s, _ = model.sample_states(eval_state, eval_info, return_log=True, batch_size=1)

        eval_s = eval_s.cpu().detach().numpy()[:,:eval_info["input_dim"][0], :eval_info["input_dim"][1]][0]
        answer = np.array(env.unwrapped.answer)
        
        if eval_s.shape != answer.shape:
            eval_s = eval_s[0]
        if np.array_equal(eval_s, answer):
            correct += 1
    return correct / num_samples


def train_model(num_epochs, batch_size, device, env_mode, prob_index, num_actions, args, use_offpolicy=False, sub_task = 0):
    """Main training loop for GFlowNet."""
    loader = ARCLoader()
    env = initialize_env(env_mode, prob_index, loader)
    model, optimizer, scheduler = initialize_model(env, num_actions, batch_size, device, args)

    replay_buffer = ReplayBuffer(capacity=CONFIG["REPLAY_BUFFER_CAPACITY"], device=device) if use_offpolicy else None

    for epoch in range(num_epochs):
        state, info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": sub_task})

        for step in range(30000):
            # Perform on-policy or off-policy update
            if use_offpolicy:
                result = model.sample_states(state, info, return_log=True, batch_size=1)
                log = result[1]
                rewards = compute_reward_with_penalty(log.traj, log.rewards[-1])
                replay_buffer.add(log.traj, log.actions, rewards, log.fwd_probs, log.back_probs)
                if step % 200 == 0:
                    update_off_policy(model, optimizer, scheduler, replay_buffer, batch_size, args)
            else:
                log = update_on_policy(model, optimizer, scheduler, state, info, args)

            # wandb logging
            if CONFIG["WANDB_USE"]:
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "loss": log.rewards[-1].item(),
                    "total_flow": log.total_flow.exp().item(),
                })

            # Evaluation
            if step % 1000 == 0:
                accuracy = evaluate_model(model, env, epoch=epoch, prob_index=prob_index, subtask=sub_task)
                print(f"Epoch {epoch}, Step {step}, Accuracy: {accuracy}")
                if CONFIG["WANDB_USE"]:
                    wandb.log({"accuracy": accuracy})

            # Move to next state
            state, info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": sub_task})
    return model, env