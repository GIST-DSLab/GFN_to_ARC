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
wandb.init(project="gflow_re", entity="hsh6449")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = None # None  # ansi



def train(num_epochs, device):

    env = env_return(render_mode, miniloader, options= None)
    

    forward_policy = ForwardPolicy(
        30, hidden_dim=32, num_actions=25).to(device)

    backward_policy = BackwardPolicy(30, hidden_dim=32, num_actions=25).to(device)

    model = GFlowNet(forward_policy, backward_policy,
                     env=env).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
<<<<<<< Updated upstream
        state, info = env.reset(options = {"prob_index" : 101, "adaptation" : True, "subprob_index" : i})
        # s0 = torch.tensor(state, dtype=torch.float32).to(device)
        
        for _ in tqdm(range(100)):
            result = model.sample_states(state, return_log=True)
=======
        state, info = env.reset(options = {"prob_index" : 101, "adaptation" : True, "subprob_index" : i}) 
        """ 4/13 수정
        state : dict , info : dict
        prob index 바뀌는거 확인함
        """    
        for _ in tqdm(range(10000)):
            result = model.sample_states(state, info, return_log=True) 
>>>>>>> Stashed changes
            
            if len(result) == 2:
                s, log = result # s : tensor, log : GFlowNetLog
            else:
                s = result # s : tensor

            # probs = model.forward_probs(s)

            # model.actions["operation"] = int(torch.argmax(probs).item())
            # model.actions["selection"] = np.zeros((30, 30))

            # result = env.step(model.actions)
            # state, reward, is_done, _, info = result

            # pdb.set_trace()

            loss, total_flow, re = trajectory_balance_loss(log.total_flow,
                                        log.rewards,
                                        log.fwd_probs,
                                        log.back_probs,
                                        torch.tensor(env.unwrapped.answer).to("cuda"))
            
            wandb.log({"loss": loss.item()})
            wandb.log({"total_flow": total_flow.item()})
            wandb.log({"reward": re.item()})

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            # if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
    
    env.post_adaptation()
    state, _ = env.reset()
    # s0 = torch.tensor(state["input"]).to(device)
    s, log = model.sample_states(state, return_log=True)
    print("initial state : \n")
    print(state["input"])
    print("Final state : \n")
    print(s)

    return model, env

def eval(model):

    env = env_return(render_mode, miniloader)

    env.post_adaptation()
        
    state, _ = env.reset(options = {"prob_index" : 101, "adaptation" : False})
    s0 = torch.tensor(state["input"]).to(device)
    s, log = model.sample_states(state, return_log=True)
    print("initial state : \n")
    print(state["input"])
    print("Final state : \n")
    print(s)

    print("=============")
    print("Answer : \n")
    print(env.unwrapped.answer)


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
