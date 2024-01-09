import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW

from gflow.gflownet import GFlowNet
from policy import ForwardPolicy, BackwardPolicy
from gflow.utils import trajectory_balance_loss

from ARCenv import MiniArcEnv
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import arcle
import gymnasium as gym
import time
import pickle
from typing import Dict, List, Tuple

# import wandb

# wandb.init(project="gflow", entity="hsh6449")


from arcle.loaders import ARCLoader, Loader, MiniARCLoader

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = "ansi" # None  # ansi


def train(num_epochs, device):

    env = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=render_mode, data_loader=loader,
                   max_grid_size=(30, 30), colors=10, max_episode_steps=None)

    forward_policy = ForwardPolicy(
        30, hidden_dim=32, num_actions=35).to(device)

    backward_policy = BackwardPolicy(30, hidden_dim=32, num_actions=35).to(device)

    model = GFlowNet(forward_policy, backward_policy,
                     env=env).to(device)

    opt = AdamW(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
        state, info = env.reset()
        # s0 = torch.tensor(state, dtype=torch.float32).to(device)
        
        for _ in range(20):
            result = model.sample_states(state, return_log=True)
            
            if len(result) == 2:
                s, log = result
            else:
                s = result

            # probs = model.forward_probs(s)

            # model.actions["operation"] = int(torch.argmax(probs).item())
            # model.actions["selection"] = np.zeros((30, 30))

            # result = env.step(model.actions)
            # state, reward, is_done, _, info = result

            loss = trajectory_balance_loss(log.total_flow,
                                        log.rewards,
                                        log.fwd_probs,
                                        log.back_probs)
            
            # wandb.log({"loss": loss.item()})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()


            opt.zero_grad()
            # if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
    
    state, _ = env.reset()
    s0 = torch.tensor(state["input"]).to(device)
    s = model.sample_states(s0, return_log=False)
    print("initial state\n" + s0)
    print("Final state\n" + s)

    return model

def eval(model):
    env = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=render_mode, data_loader=loader,
                   max_grid_size=(30, 30), colors=10, max_episode_steps=None)
    

    for i in (p := tqdm(range(num_epochs))):
        state, info = env.reset()
        s0 = torch.tensor(state, dtype=torch.float32).to(device)
        
        
        result = model.sample_states(s0, return_log=True)
        
        if len(result) == 2:
            s, log = result
        else:
            s = result

        probs = model.forward_probs(s)

        model.actions["operation"] = int(torch.argmax(probs).item())
        model.actions["selection"] = np.zeros((30, 30))

        result = env.step(model.actions)
        state, reward, is_done, _, info = result


        # if i % 10 == 0:
        # p.set_description(f"{loss.item():.3f}")
        
    state, _ = env.reset()
    s0 = torch.tensor(state["input"]).to(device)
    s = model.sample_states(s0, return_log=False)
    print("initial state\n" + s0)
    print("Final state\n" + s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=28280)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """dataset = ARCdataset(
        "/home/jovyan/Gflownet/ARCdataset/diagonal_flip_augment.data_2/")
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)"""

    model = train(num_epochs, device)
    eval(model)
