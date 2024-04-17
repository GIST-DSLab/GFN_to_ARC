import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
torch.autograd.set_detect_anomaly(True)


from gflow.gflownet_random import GFlowNet
from policy_random import ForwardPolicy, BackwardPolicy
from gflow.utils import trajectory_balance_loss

# from ColorARCEnv import env_return # BBox는 ColorARCEnv로 변경
from PointARCEnv import env_return # Point는 PointARCEnv로 변경

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
# wandb.init(project="gflow_re", entity="hsh6449", name="Step_length_30")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = None # None  # ansi

class TestLoader(Loader):
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.rng = np.random.default_rng(12345)
        super().__init__(rng=self.rng)
    def get_path(self):
        return ['']

    def parse(self):

        ti= np.zeros((self.size_x,self.size_y), dtype=np.uint8)
        to = np.zeros((self.size_x,self.size_y), dtype=np.uint8)
        ei = np.zeros((self.size_x,self.size_y), dtype=np.uint8)
        eo = np.zeros((self.size_x,self.size_y), dtype=np.uint8)

        ti[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        to[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        ei[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        eo[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        return [([ti],[to],[ei],[eo], {'desc': "just for test"})]
    def pick(self):
        return self.parse()
    
def train(num_epochs, device, env):

    forward_policy = ForwardPolicy(30, hidden_dim=32, num_actions=10).to(device)
    backward_policy = BackwardPolicy(30, hidden_dim=32, num_actions=10).to(device)

    model = GFlowNet(forward_policy, backward_policy,
                     env=env).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
        state, info = env.reset() 
        """ 4/13 수정
        state : dict , info : dict
        prob index 바뀌는거 확인함
        """    
        for _ in tqdm(range(1000000)):
            result = model.sample_states(state,info, return_log=True) 
            
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
            
            # wandb.log({"loss": loss.item()})
            # wandb.log({"total_flow": total_flow.item()})
            # wandb.log({"reward": re.item()})

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            # if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
            state, info = env.reset()
    
    # env.post_adaptation()
    state, _ = env.reset()
    # s0 = torch.tensor(state["input"]).to(device)
    s, log = model.sample_states(state, return_log=True)
    print("initial state : \n")
    print(state["input"])
    print("Final state : \n")
    print(s)

    return model, env

def eval(model):

    env = gym.make(
            'ARCLE/O2ARCv2Env-v0', 
            data_loader = TestLoader(5, 5), 
            max_trial = 3,
            max_grid_size=(5, 5), 
            colors= 10)

    # env.post_adaptation()
        
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

    env = gym.make(
            'ARCLE/O2ARCv2Env-v0', 
            data_loader = TestLoader(5, 5), 
            max_trial = 3,
            max_grid_size=(5, 5), 
            colors=10)
    
    """dataset = ARCdataset(
        "/home/jovyan/Gflownet/ARCdataset/diagonal_flip_augment.data_2/")
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)"""

    model = train(num_epochs, device, env)
    eval(model)
