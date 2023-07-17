import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax
import pdb


class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, s):
        # import pdb; pdb.set_trace()
        x = self.dense1(s)
        x = relu(x)
        x = self.dense2(x)
        return softmax(x, dim=1)


class BackwardPolicy(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.size = int(state_dim**0.5)

    def __call__(self, s):
        idx = s.argmax(-1)
        at_top_edge = idx < self.size
        at_left_edge = (idx > 0) & (idx % self.size == 0)

        probs = 0.5 * torch.ones(len(s), self.num_actions)
        probs[at_left_edge] = torch.Tensor([1, 0, 0])
        probs[at_top_edge] = torch.Tensor([0, 1, 0])
        probs[:, -1] = 0  # disregard termination

        return probs
