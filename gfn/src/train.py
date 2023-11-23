import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.nn.functional import one_hot
from gflow.gflownet import GFlowNet
from policy import ForwardPolicy, BackwardPolicy
from gflow.utils import trajectory_balance_loss
from torch.optim import Adam

# from grid import Grid
import numpy as np
# from gfn.ARCdataset import ARCdataset


def plot(samples, env):
    _, ax = plt.subplots(1, 2)
    s = samples.sum(0).view(25, 25)
    e = env.reward(torch.eye(env.state_dim)).view(25, 25)

    ax[0].matshow(s.numpy())
    ax[0].set_title("Samples")
    ax[1].matshow(e.numpy())
    ax[1].set_title("Environment")

    plt.show()


def train(batch_size, num_epochs, train_dataloader):

    forward_policy = ForwardPolicy(64, hidden_dim=32, num_actions=5).to(
        device=torch.device("cuda"))
    backward_policy = BackwardPolicy(64, num_actions=5)
    model = GFlowNet(forward_policy, backward_policy, env=None)
    opt = Adam(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
        for batch in train_dataloader:
            s0 = batch["state"]
            s, log = model.sample_states(s0, return_log=True)
            loss = trajectory_balance_loss(log.total_flow,
                                           log.rewards,
                                           log.fwd_probs,
                                           log.back_probs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 10 == 0:
                p.set_description(f"{loss.item():.3f}")

    s0 = torch.tensor([[4, 4, 4, 0, 4],
                       [4, 4, 0, 0, 0],
                       [4, 4, 4, 4, 4],
                       [4, 4, 4, 4, 0],
                       [0, 4, 4, 0, 4]]).float()
    s = model.sample_states(s0, return_log=False)
    # plot(s, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    train(batch_size, num_epochs)
