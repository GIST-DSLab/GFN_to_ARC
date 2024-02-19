import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
torch.autograd.set_detect_anomaly(True)


from gflow.gflownet import GFlowNet
from policy import ForwardPolicy, BackwardPolicy
from gflow.utils import trajectory_balance_loss
from ColorARCEnv import env_return

# from ARCenv import MiniArcEnv
# import matplotlib.pyplot as plt
import argparse
import time
import pickle
import pdb
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Dict, List, Any, Tuple, SupportsInt, Tuple, Callable

import arcle
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader, MiniARCLoader

import wandb
wandb.init(project="gflow_collas", entity="hsh6449")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = None # None  # ansi



def train(num_epochs, device):

    # env = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=render_mode, data_loader=loader,
    #                max_grid_size=(30, 30), colors=10, max_episode_steps=None)

    env = env_return(render_mode, miniloader)

    forward_policy = ForwardPolicy(
        5, hidden_dim=32, num_actions=10).to(device)

    backward_policy = BackwardPolicy(5, hidden_dim=32, num_actions=10).to(device)

    model = GFlowNet(forward_policy, backward_policy,
                     env=env).to(device)
    model.train()

    # for param in model.parameters():
    #     print(param.requires_grad)

    opt = AdamW(model.parameters(), lr=1e-4)
    # state = env.env.reset(options= {'adaptation':False, 'prob_index':env.findbyname(env.traces_info[idx][0]), 'subprob_index': env.traces_info[idx][1]})

    for i in (p := tqdm(range(num_epochs))):
        state, info = env.reset(options = {"prob_index" : 98, "adaptation" : False})
        # s0 = torch.tensor(state, dtype=torch.float32).to(device)
        
        for _ in tqdm(range(100000)):
            result = model.sample_states(state, return_log=True)
            
            if len(result) == 2:
                s, log = result
            else:
                s = result


            loss, re = trajectory_balance_loss(log.total_flow,
                                        log.rewards,
                                        log.fwd_probs,
                                        log.back_probs,
                                        torch.tensor(env.unwrapped.answer).to("cuda")) # answer는 빼도됨 (MSE를 쓸경우 )
            
            
            wandb.log({"loss": loss.item()})
            wandb.log({"reward": re})

            ## parameter update check
            # a = list(model.parameters())[0]
            # print("forward_prev : ", a.grad)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()

            # b = list(model.parameters())[0]
            # print("forward_equal : ", torch.equal(a.data, b.data))


            # opt.zero_grad()
            # if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
    
    state, _ = env.reset()
    # s0 = torch.tensor(state["input"]).to(device)
    s, log = model.sample_states(state, return_log=False)
    print("initial state\n" + state["input"])
    print("Final state\n" + s)

    return model

def eval(model):

    env = env_return()

    state, _ = env.reset(options = {"prob_index" : 98, "adaptation" : False})
    s0 = torch.tensor(state["input"]).to(device)
    s, log = model.sample_states(s0, return_log=False)

    print("initial state\n" + s0)
    print("Final state\n" + s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=3)

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
