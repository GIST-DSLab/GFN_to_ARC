import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
torch.autograd.set_detect_anomaly(True)


from gflow.gflownet_point import GFlowNet
from policy_point import MLPForwardPolicy, MLPBackwardPolicy, LSTMForwardPolicy


from arcle.loaders import ARCLoader, Loader, MiniARCLoader

import numpy as np
import matplotlib.pyplot as plt

from EntireARCEnv import env_return

def infer():

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  loader = ARCLoader()
  render_mode = None # None  # ansi

  env = env_return(render_mode, loader)

  forward_policy = LSTMForwardPolicy(30, hidden_dim=32, num_actions=3).to(device)
  backward_policy = MLPBackwardPolicy(30, hidden_dim=32, num_actions=3).to(device)

  model = GFlowNet(forward_policy, backward_policy,
                    env=env).to(device)
  
  model.state_dict(torch.load('model_0.pt'))
  model.eval()

  state, info = env.reset(options={"prob_index" : 178, "adaptation" : False})
  s, log = model.sample_states(s, info, return_log=True)
  visualize(log.traj)

  return model

def visualize(traj):
  for i in range(traj.shape[0]):
    plt.imshow(traj[i])
    plt.show()
    plt.savefig(f'gfn/result/visualize_{i}.png')

infer()
