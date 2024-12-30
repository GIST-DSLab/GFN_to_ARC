import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
torch.autograd.set_detect_anomaly(True)


from gflow.gflownet_s import GFlowNet
from policy_s import ForwardPolicy, BackwardPolicy
from gflow.utils_s import trajectory_balance_loss

import argparse

import pdb
import numpy as np
import matplotlib.pyplot as pp

from tqdm import tqdm
import wandb
wandb.init(project="gflow_smile", entity="hsh6449")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_face = lambda: (pp.gca().add_patch(pp.Circle((0.5,0.5),0.5,fc=(.9,.9,0))),
                     pp.gca().add_patch(pp.Circle((0.25,0.6),0.1,fc=(0,0,0))),
                     pp.gca().add_patch(pp.Circle((0.75,0.6),0.1,fc=(0,0,0))))
patches = {
  'smile': lambda: pp.gca().add_patch(pp.Polygon(np.stack([np.linspace(0.2,0.8), 0.3-np.sin(np.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
  'frown': lambda: pp.gca().add_patch(pp.Polygon(np.stack([np.linspace(0.2,0.8), 0.15+np.sin(np.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
  'left_eb_down': lambda: pp.gca().add_line(pp.Line2D([0.15, 0.35], [0.75,0.7], color=(0,0,0))),
  'right_eb_down': lambda: pp.gca().add_line(pp.Line2D([0.65, 0.85], [0.7,0.75], color=(0,0,0))),
  'left_eb_up': lambda: pp.gca().add_line(pp.Line2D([0.15, 0.35], [0.7,0.75], color=(0,0,0))),
  'right_eb_up': lambda: pp.gca().add_line(pp.Line2D([0.65, 0.85], [0.75,0.7], color=(0,0,0))),
}
sorted_keys = sorted(patches.keys())




def face_parents(state):
  parent_states = []  # states that are parents of state
  parent_actions = []  # actions that lead from those parents to state
  for face_part in state:
    # For each face part, there is a parent without that part
    parent_states.append([i for i in state if i != face_part])
    # The action to get there is the corresponding index of that face part
    parent_actions.append(sorted_keys.index(face_part))
  return parent_states, parent_actions

def face_to_tensor(face):
  return torch.tensor([i in face for i in sorted_keys]).float().to("cuda")


def train(num_epochs, device):
    

    forward_policy = ForwardPolicy(
        6, hidden_dim=512, num_actions=6).to(device)

    backward_policy = ForwardPolicy(6, hidden_dim=512, num_actions=6).to(device)

    model = GFlowNet(forward_policy, backward_policy,
                     env=None).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=5e-3)

    # Let's keep track of the losses and the faces we sample
    losses = []
    sampled_faces = []

    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode.
    minibatch_loss = 0
    update_freq = 4

    for episode in (p :=tqdm(range(50000), ncols=40)):
        state = []
        
        for _ in tqdm(range(3)):
            result = model.sample_states(state, return_log=True)
            
            if len(result) == 2:
                s, log = result
            else:
                s = result

            loss, re, total_flow = trajectory_balance_loss(log.total_flow,
                                        log.rewards,
                                        log.fwd_probs,
                                        log.back_probs)
            losses.append(loss.item())
            
            wandb.log({"loss": loss.item()})
            wandb.log({"reward": re.item()})
            wandb.log({"total_flow": np.exp(total_flow.item())})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            opt.zero_grad()
            # if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
    
    
    s, log = model.sample_states(state, return_log=True)
    print("initial state : \n")
    print(state["input"])
    print("Final state : \n")
    print(s)

    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=3)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, losses = train(num_epochs, device)

    pp.figure(figsize=(10,3))
    pp.plot(losses)
    pp.yscale('log')
