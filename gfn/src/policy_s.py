import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax


class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)

        self.num_action = num_actions


    def forward(self, s):
        # check tensor device
        if s.device.type != 'cuda':
            s = s.to('cuda')
        
        # forward pass
        x = torch.flatten(s, 0)
        x = x.to(torch.float32)
        x = self.dense1(x)
        # x = relu(x)
        # x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)

        # x = softmax(x, dim=0)
    
        return x

class BackwardPolicy(nn.Module):
    def __init__(self, state_dim,hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, s):
        x = torch.flatten(s, 0)
            # 이부분 바꾸는게 좋다고 함 clone().detach()로? 근데 dtype때문에 이렇게 한거라서 일단 놔둠
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        # x = self.dense2(x)
        # x = relu(x)
        x = self.dense3(x)
        # x =softmax(x, dim=0)
        return x

