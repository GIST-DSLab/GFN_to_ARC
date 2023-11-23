import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax
# import pdb


class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, s):
        x = torch.flatten(s, 0)
        # 이부분 바꾸는게 좋다고 함 clone().detach()로? 근데 dtype때문에 이렇게 한거라서 일단 놔둠
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        return softmax(x, dim=0)


class BackwardPolicy(nn.Module):  # 여기도 바꿔야함
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        # self.size = int(state_dim**0.5)
        self.size = state_dim

    def __call__(self, fwd_probs):
        half_length = len(fwd_probs) // 2
        
        probs = torch.cat(fwd_probs, dim=0)[half_length:]
        # probs = 0.5 * torch.ones(len(s), self.num_actions)

        # probs[:, -1] = torch.zeros(probs[:,0].shape)  # disregard termination

        return probs
