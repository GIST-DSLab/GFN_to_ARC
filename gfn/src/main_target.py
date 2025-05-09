import torch
from torch.nn.parameter import Parameter
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

from gflow.gflownet_target import GFlowNet
from policy_target import MLPForwardPolicy, MLPBackwardPolicy, EnhancedMLPForwardPolicy
from gflow.utils import trajectory_balance_loss, detailed_balance_loss, subtrajectory_balance_loss
from ARCenv.wrapper import env_return  

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from tqdm import tqdm
from collections import deque

import os
import sys
import copy
import json
from typing import Tuple

import arcle
from arcle.loaders import ARCLoader, MiniARCLoader

import pdb
import random
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = None  # ANSI rendering mode if needed

TASKNUM = 178
CUDANUM = 0
ACTIONNUM = 5
LOSS = "trajectory_balance_loss"  # Options: "trajectory_balance_loss", "subtb_loss", "detailed_balance_loss"

WANDB_USE = True
FILENAME = f"geometric_10,5_taskg_rscale10_{TASKNUM}"

if WANDB_USE:
    wandb.init(project="gflow_re", 
               entity="hsh6449", 
               name=f"local_cuda{CUDANUM},ep10,a{ACTIONNUM},enhanced_reward, task {TASKNUM}, onpolicy")


# ... [No changes to classes ReplayBuffer, TrajLengthRegularization, TrajectoryBuffer] ...

def detect_cycle(traj):
    visited_states = set()
    detect_count = 0 
    for state in traj:
        state_tuple = tuple(state.cpu().detach().numpy().flatten())  # Convert state to a unique tuple
        if state_tuple in visited_states:
            detect_count += 1  # Cycle detected
        visited_states.add(state_tuple)
    return detect_count  # 0 if no cycle

def compute_reward_with_penalty(traj, base_reward, penalty=0.1):
    """
    Args:
        traj: List of states in the episode trajectory
        base_reward: Original reward value
        penalty: Penalty applied when cycles are detected (default 0.1)
    """
    detect_count = detect_cycle(traj)
    if detect_count > 0:
        reward = base_reward - penalty * detect_count
        if reward < 0:
            return torch.tensor(0.0, device=reward.device)  # Negative rewards not allowed
        return reward
    return base_reward  # If no cycle, return base reward

def save_trajectory_and_rewards(trajectory, rewards, filename='data.json'):
    data = {
        'trajectory': [t.detach().cpu().numpy().tolist() for t in trajectory],
        'rewards': rewards.detach().cpu().numpy().tolist()
    } 

    with open(filename, 'w') as f:
        json.dump(data, f)

def save_gflownet_trajectories(num_trajectories, save_path, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, env = train(
        num_epochs=3, batch_size=1, device=device, 
        env_mode=args.env_mode, prob_index=args.prob_index, 
        num_actions=args.num_actions, args=args, use_offpolicy=False
    )
    trajectories = []
    for _ in range(num_trajectories):
        state, info = env.reset(options={"prob_index": args.prob_index, "adaptation": True})
        _, log = model.sample_states(state, info, return_log=True, batch_size=1)
        
        def serialize_dict(d):
            """Convert dictionary values to JSON-serializable format"""
            serialized = {}
            for key, value in d.items():
                if isinstance(value, np.ndarray):
                    serialized[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    serialized[key] = value.detach().cpu().tolist()
                elif isinstance(value, dict):
                    serialized[key] = serialize_dict(value)
                else:
                    serialized[key] = value
            return serialized
        
        trajectories.append({
            "states": [traj.detach().cpu().tolist()[:10][:10] for traj in log.traj],
            "actions": [a.detach().cpu().tolist() for a in log.actions],
            "rewards": [r.detach().cpu().tolist() for r in log.rewards],
            "states_full": [serialize_dict(s) for s in log.tstates]
        })
    
    with open(save_path, 'w') as f:
        json.dump(trajectories, f)
    print(f"Saved {num_trajectories} trajectories to {save_path}")

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_importance_weights(current_log_probs, old_log_probs, batch_size):
    ratios = []
    for cur_probs, old_probs in zip(current_log_probs, old_log_probs):
        stepwise_ratios = []
        for cur_prob, old_prob in zip(cur_probs, old_probs):
            stepwise_ratio = torch.exp(cur_prob - old_prob)
            stepwise_ratios.append(stepwise_ratio)
        episode_ratio = torch.mean(torch.stack(stepwise_ratios))
        ratios.append(episode_ratio)

    final_ratios = torch.mean(torch.stack(ratios))
    return torch.clamp(final_ratios, 0.8, 1.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1) 
    parser.add_argument("--env_mode", type=str, default="entire")
    parser.add_argument("--prob_index", type=int, default=TASKNUM)
    parser.add_argument("--num_actions", type=int, default=ACTIONNUM)
    parser.add_argument("--ep_len", type=int, default=10)
    parser.add_argument("--device", type=int, default=CUDANUM)
    parser.add_argument("--use_offpolicy", action="store_true", help="Enable off-policy training", default=False)
    parser.add_argument("--sampling_method", type=str, default="prt", choices=["prt", "fixed_ratio", "egreedy"],
                        help="Sampling method for replay buffer")
    
    parser.add_argument("--save_trajectories", type=str, default=None, help="Path to save GFlowNet trajectories")
    parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to save")

    args = parser.parse_args()
    
    seed_everything(48)  # Changed from 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save_trajectories:
        save_gflownet_trajectories(args.num_trajectories, args.save_trajectories, args)
    else:
        model, env = train(
            args.num_epochs,
            args.batch_size,
            device,
            args.env_mode,
            args.prob_index,
            args.num_actions,
            args,
            use_offpolicy=args.use_offpolicy
        )
