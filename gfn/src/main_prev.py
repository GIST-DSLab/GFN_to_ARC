import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW

from gflow.gflownet import GFlowNet
from policy import ForwardPolicy, BackwardPolicy
from gflow.utils import trajectory_balance_loss
# from grid import Grid
# from gfn.ARCdataset import ARCdataset
from ARCenv import MiniArcEnv
import numpy as np

import arcle
import gymnasium as gym
import time
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class examplearc:
    INITIAL = torch.tensor(np.array(
        [[0, 4, 4, 0, 0],
         [0, 4, 6, 6, 0],
         [0, 4, 4, 6, 0],
         [0, 4, 4, 0, 0],
         [0, 6, 4, 0, 0]]), dtype=torch.long).to(device)
    GOAL = torch.tensor(np.array(
        [[0, 6, 6, 0, 0],
         [0, 6, 4, 4, 0],
         [0, 6, 6, 4, 0],
         [0, 6, 6, 0, 0],
         [0, 4, 6, 0, 0]]), dtype=torch.long).to(device)
    OBJ_MAP = torch.tensor(np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 2, 2, 0],
         [0, 1, 1, 2, 0],
         [0, 1, 1, 0, 0],
         [0, 3, 1, 0, 0]]), dtype=torch.long).to(device)

    @ classmethod
    def get_args(cls):
        return cls.INITIAL.clone().detach(), cls.GOAL.clone().detach(), cls.OBJ_MAP.clone().detach()


def train(batch_size, num_epochs, device):

    env = MiniArcEnv(*examplearc.get_args())

    forward_policy = ForwardPolicy(
        5, hidden_dim=32, num_actions=700).to(device)

    backward_policy = BackwardPolicy(
        32, num_actions=700).to(device)

    model = GFlowNet(forward_policy, backward_policy,
                     env=env).to(device)

    opt = AdamW(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
        s0 = torch.tensor(env.initial, dtype=torch.float32).to(device)
        # import pdb; pdb.set_trace()
        result = model.sample_states(s0, return_log=True)
        if len(result) == 2:
            s, log = result
        else:
            s = result

        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs,
                                       log.back_probs)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")

    s0 = torch.tensor(env.initial).to(device)
    s = model.sample_states(s0, return_log=False)
    print("Final state\n" + s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """dataset = ARCdataset(
        "/home/jovyan/Gflownet/ARCdataset/diagonal_flip_augment.data_2/")
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)"""

    train(batch_size, num_epochs, device)
