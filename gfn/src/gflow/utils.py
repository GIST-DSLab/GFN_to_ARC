import torch
import pdb
import torch.nn.functional as F
import numpy as np
import random
import wandb

def setup_wandb(project_name: str, entity: str, config: dict, run_name: str = None):
    """
    Initialize wandb settings.
    Args:
        project_name (str): Project name.
        entity (str): Team or user name.
        config (dict): Configuration to store in wandb.
        run_name (str): Optional run name.
    """
    wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=run_name
    )

def normalize_probabilities(x):
    return x / x.sum()

# def process_rewards(rewards, device):
#     b = rewards[0].shape[0]
#     if isinstance(rewards, list):
#         # For batch training: take the last reward of each episode
#         last_rewards = rewards[-1]
#     else:
#         # If rewards is already a single value
#         last_rewards = [rewards]
#     return torch.tensor(last_rewards, device=device)

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs, batch_size=1):
    """
    Computes the mean trajectory balance loss for a collection of samples.
    See Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    """
    total_flow = total_flow.clamp(min=0)
    if isinstance(rewards, int or float):
        rewards = torch.tensor([rewards], device=total_flow.device, dtype=torch.float)

    if batch_size > 1:
        fwd_probs = torch.stack([torch.sum(torch.stack(probs)) for probs in fwd_probs])
        back_probs = torch.stack([torch.sum(torch.stack(probs)) for probs in back_probs])
        rewards = torch.tensor([r[-1] for r in rewards], device=total_flow.device)
    else:
        fwd_probs = torch.sum(torch.stack(fwd_probs))
        back_probs = torch.sum(torch.stack(back_probs))
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards[-1], device=total_flow.device)


    log_rewards = torch.log(rewards).clip(-1)
    loss = torch.square(total_flow + fwd_probs - log_rewards - back_probs).clamp(max=10000)
    return loss, total_flow, rewards

def detailed_balance_loss(total_flow, rewards, fwd_probs, back_probs, answer):
    """
    Loss focusing on matching forward and backward probabilities, like FM Loss.
    Does not use the reward directly.
    """
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)
    rewards = torch.tensor([rewards[-1]], device=total_flow.device)

    forward_term = torch.sum(fwd_probs) * torch.exp(fwd_probs)
    backward_term = torch.sum(back_probs) * torch.exp(back_probs)

    loss = torch.square(torch.log(forward_term) - torch.log(backward_term))
    loss = loss.mean()
    return loss, total_flow, rewards

def subtrajectory_balance_loss(trajectories, fwd_probs, back_probs):
    """
    Calculate the Subtrajectory Balance Loss for given trajectories.
    """
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)

    losses = []
    for trajectory in trajectories:
        log_pf_product = 0
        log_pb_product = 0

        for i in range(len(trajectory) - 1):
            log_pf_product += fwd_probs[i]
            log_pb_product += back_probs[i]

        flow_start = torch.exp(torch.sum(fwd_probs[:len(trajectory)//2]))
        flow_end = torch.exp(torch.sum(back_probs[len(trajectory)//2:]))

        log_ratio = (torch.log(flow_start) + log_pf_product) - (torch.log(flow_end) + log_pb_product)
        losses.append(log_ratio ** 2)

    return torch.mean(torch.stack(losses))

def guided_TB_loss(total_flow, rewards, fwd_probs, back_probs, answer):
    """
    Guided version of trajectory balance loss using rewards.
    """
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)
    rewards = torch.tensor([rewards[-1]], device=total_flow.device)

    loss = torch.square(torch.log(total_flow) + torch.sum(fwd_probs) - torch.log(rewards).clip(0) - torch.sum(back_probs))
    loss = loss.sum().clamp(max=1e+6)

    return loss, total_flow, rewards

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def detect_cycle(traj):
    """
    Detects cycles in a trajectory by checking for repeated states.
    """
    visited_states = set()
    detect_count = 0
    for state in traj:
        state_tuple = tuple(state.cpu().detach().numpy().flatten())
        if state_tuple in visited_states:
            detect_count += 1
        visited_states.add(state_tuple)
    return detect_count

def compute_reward_with_penalty(traj, base_reward, penalty=1.0):
    """
    Compute reward with penalty applied if cycle is detected.
    
    Args:
        traj: Trajectory (list of states)
        base_reward: Base reward value
        penalty: Penalty applied per detected cycle (default 0.1)
    """
    detect_count = detect_cycle(traj)
    if detect_count > 0:
        reward = base_reward - penalty * detect_count
        if reward < 0:
            return torch.tensor(0.0, device=reward.device)  # Negative reward not allowed
        return reward
    return base_reward
