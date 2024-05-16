import torch
from torch.nn.parameter import Parameter
from torch.optim import AdamW
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from gflow.gflownet_target import GFlowNet, RewardModel
from policy_target import MLPForwardPolicy, MLPBackwardPolicy
from gflow.utils import trajectory_balance_loss, detailed_balance_loss, subtrajectory_balance_loss
from PointARCEnv import env_return # Point는 PointARCEnv로 변경

import argparse
import time
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from collections import deque

import arcle
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, MiniARCLoader

import pdb

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = None # None  # ansi

LOSS = "trajectory_balance_loss" # "trajectory_balance_loss", "subtb_loss", "detailed_balance_loss"
WANDB_USE = False

if WANDB_USE:
    wandb.init(project="gflow_re", entity="hsh6449", name="TB_loss_step10_redOnly_H100_MSErewardmodel+LSTM_3(logz)")

class RewardBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def add(self, trajectory, reward):
        if len(self.buffer) >= self.capacity:
            # 가장 낮은 보상을 가진 항목을 제거
            min_reward_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
            self.buffer.pop(min_reward_idx)
        self.buffer.append((trajectory, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        trajectories, rewards = zip(*[self.buffer[idx] for idx in indices])
        
        # 모든 trajectory가 동일한 크기를 가지는지 확인하고 텐서로 변환
        trajectories = [torch.tensor(traj) for traj in trajectories]
        
        return torch.stack(trajectories), torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return len(self.buffer)

def train(num_epochs, device):
    env = env_return(render_mode, miniloader, options=None)
    
    forward_policy = MLPForwardPolicy(30, hidden_dim=32, num_actions=2).to(device)
    backward_policy = MLPBackwardPolicy(30, hidden_dim=32, num_actions=2).to(device)

    reward_model = RewardModel().to(device)
    reward_buffer = RewardBuffer(capacity=1000)

    model = GFlowNet(forward_policy, backward_policy, env=env, reward_model=reward_model, reward_buffer=reward_buffer, device = DEVICE).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=0.01)
    reward_opt = AdamW(reward_model.parameters(), lr=0.01)

    for i in tqdm(range(num_epochs)):
        state, info = env.reset(options={"prob_index": 101, "adaptation": True, "subprob_index": i}) 
        
        for step in tqdm(range(50000)):
            result = model.sample_states(state, info, return_log=True, i=i) 
            
            if len(result) == 2:
                s, log = result # s : tensor, log : GFlowNetLog
            else:
                s = result # s : tensor

            # Use reward_model for the loss calculation
            if step > 11:
                predicted_rewards = reward_model(log.traj[-1][-1])
                if predicted_rewards < 1 : 
                    predicted_rewards = torch.tensor(1.0, device=predicted_rewards.device)

                if LOSS == "trajectory_balance_loss":
                    loss, total_flow, re = trajectory_balance_loss(
                        log.total_flow,
                        predicted_rewards,
                        log.fwd_probs,
                        log.back_probs,
                        torch.tensor(env.unwrapped.answer).to("cuda")
                    )
                    if WANDB_USE:
                        wandb.log({"loss": loss.item()})
                        wandb.log({"total_flow": total_flow.item()})
                        wandb.log({"reward": re.item()})
                elif LOSS == "detailed_balance_loss":
                    loss, total_flow, re = detailed_balance_loss(
                        log.total_flow,
                        predicted_rewards,
                        log.fwd_probs,
                        log.back_probs,
                        torch.tensor(env.unwrapped.answer).to("cuda")
                    )
                    if WANDB_USE:
                        wandb.log({"loss": loss.item()})
                        wandb.log({"total_flow": total_flow.item()})
                        wandb.log({"reward": re.item()})
                elif LOSS == "subtb_loss":
                    loss = subtrajectory_balance_loss(log.traj, log.fwd_probs, log.back_probs)
                    re = log.rewards[-1]
                    if WANDB_USE:
                        wandb.log({"loss": loss.item()})
                        wandb.log({"total_flow": total_flow.item()})
                        wandb.log({"reward": re.item()})
            
            
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                tqdm.write(f"Loss: {loss.item():.3f}")

            # Calculate reward and store in the buffer
            reward = log.rewards[-1]
            reward_buffer.add(log.traj[-1][-1], reward)

            # Train reward model
            if len(reward_buffer.buffer) > 10:
                trajectories, rewards = reward_buffer.sample(batch_size=10)
                reward_preds = reward_model(trajectories)
                reward_loss = F.mse_loss(reward_preds, rewards)
                reward_opt.zero_grad()
                reward_loss.backward()
                reward_opt.step()

            state, info = env.reset(options={"prob_index": 101, "adaptation": True, "subprob_index": i})

            ## evaluation 
            if i % 10 == 0:
                s, _ = model.sample_states(state, info, return_log=True)
                print("initial state : \n")
                print(state["input"][:info["input_dim"][0], :info["input_dim"][1]])
                print("Final state : \n")
                print(s[:info["input_dim"][0], :info["input_dim"][1]].long())
                print("=============")
                print("Answer : \n")
                print(env.unwrapped.answer)
                print("=============")

                correct = np.equal(s.cpu().detach().numpy()[:info["input_dim"][0], :info["input_dim"][1]], env.unwrapped.answer)
                acc = np.sum(correct) / (correct.shape[0] * correct.shape[1])
                
                if WANDB_USE:
                    wandb.log({"accuracy": acc})
                    wandb.log({"true accuracy": 0 if acc < 1 else 1})

            if i % 100 == 0:
                torch.save(model.state_dict(), f"model_{i}.pt")
    
    env.post_adaptation()
    state, _ = env.reset()
    s, log = model.sample_states(state, return_log=True)
    print("initial state : \n")
    print(state["input"])
    print("Final state : \n")
    print(s)

    return model, env

def eval(model):
    env = env_return(render_mode, miniloader, options=None)
    env.post_adaptation()
        
    state, _ = env.reset(options={"prob_index": 101, "adaptation": False})
    s0 = torch.tensor(state["input"]).to(device)
    s, log = model.sample_states(state, return_log=True)
    print("initial state : \n")
    print(state["input"])
    print("Final state : \n")
    print(s)
    print("=============")
    print("Answer : \n")
    print(env.unwrapped.answer)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=3)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(num_epochs, device)
    eval(model)
