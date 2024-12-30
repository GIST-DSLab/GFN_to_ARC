import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
torch.autograd.set_detect_anomaly(True)


from gflow.gflownet_entire import GFlowNet
from policy_entire import MLPForwardPolicy, MLPBackwardPolicy, LSTMForwardPolicy
from gflow.utils import trajectory_balance_loss, detailed_balance_loss, subtrajectory_balance_loss

# from ColorARCEnv import env_return # BBox는 ColorARCEnv로 변경
from EntireARCEnv import env_return # Point는 PointARCEnv로 변경

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
wandb.init(project="gflow_re", entity="hsh6449", name="TB_loss_step10_redOnly_local_MSE+LSTM_3(logz)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()


render_mode = None # None  # ansi

LOSS = "trajectory_balance_loss" # "trajectory_balance_loss", "subtb_loss", "detailed_balance_loss"

def train(num_epochs, device):

    env = env_return(render_mode, loader)
    
    # seed_everything(42)

    forward_policy = MLPForwardPolicy(3, hidden_dim=32, num_actions=4).to(device)
    backward_policy = MLPBackwardPolicy(3, hidden_dim=32, num_actions=4).to(device)

    clip_grad_norm_pram = [forward_policy.parameters(), backward_policy.parameters()]

    model = GFlowNet(forward_policy, backward_policy,
                     env=env).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
        info = env.reset(options = {"prob_index" : 178, "adaptation" : True, "subprob_index" : i}) 
        state = info["grid"]
        """ 4/13 수정
        state : dict , info : dict
        prob index 바뀌는거 확인함
        """    
        for _ in tqdm(range(1000)):
            result = model.sample_states(state,info, return_log=True, i=i) 
            
            if len(result) == 2:
                s, log = result # s : tensor, log : GFlowNetLog
            else:
                s = result # s : tensor

            if LOSS ==  "trajectory_balance_loss":
                loss, total_flow, re = trajectory_balance_loss(log.total_flow,
                                        log.rewards,
                                        log.fwd_probs,
                                        log.back_probs,
                                        torch.tensor(env.unwrapped.answer).to("cuda"))
                wandb.log({"loss": loss.item()})
                wandb.log({"total_flow": total_flow.item()})
                wandb.log({"reward": re.item()})
            elif LOSS == "detailed_balance_loss":
            
                loss, total_flow, re = detailed_balance_loss(log.total_flow,
                                        log.rewards,
                                        log.fwd_probs,
                                        log.back_probs,
                                        torch.tensor(env.unwrapped.answer).to("cuda"))
                
                #wandb.log({"loss": loss.item()})
                #wandb.log({"total_flow": total_flow.item()})
                #wandb.log({"reward": re.item()})

            elif LOSS == "subtb_loss":
                loss = subtrajectory_balance_loss(log.traj, log.fwd_probs, log.back_probs)

                re = log.rewards[-1]

                # wandb.log({"loss": loss.item()})
                # #wandb.log({"total_flow": total_flow.item()})
                # wandb.log({"reward": re.item()})

            log.total_flow = log.total_flow.clamp(min=0)
            loss.backward()

            for param in clip_grad_norm_pram:
                torch.nn.utils.clip_grad_norm_(param, 5)
            
            opt.step()
            # if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
            info = env.reset(options = {"prob_index" : 178, "adaptation" : True, "subprob_index" : i})
            state = info["grid"]

            ## evaluation 
            ah, aw = env.unwrapped.answer.shape
            if i % 10 == 0:
                s, _ = model.sample_states(state,info, return_log=True)
                print("initial state : \n")
                print(state[:state.shape[0],:state.shape[1]])
                print("Final state : \n")
                print(s[:ah,:aw].long())
                print("=============")
                print("Answer : \n")
                print(env.unwrapped.answer)
                print("=============")

                correct = np.equal(s.cpu().detach().numpy()[:ah,:aw], env.unwrapped.answer)
                acc = np.sum(correct) / (correct.shape[0]*correct.shape[1])
                wandb.log({"accuracy": acc})
                wandb.log({"true accuracy": 0 if acc < 1 else 1})

            if i % 100 == 0:
                torch.save(model.state_dict(), f"model_{i}.pt")

    
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

    env = env_return(render_mode, miniloader, options= None)

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

    """dataset = ARCdataset(
        "/home/jovyan/Gflownet/ARCdataset/diagonal_flip_augment.data_2/")
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)"""

    model = train(num_epochs, device)
    eval(model)
