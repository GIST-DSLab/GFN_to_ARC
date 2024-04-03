import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax

import numpy as np
import pdb


class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim*20, hidden_dim*hidden_dim)
        self.dense2 = nn.Linear(hidden_dim*hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)

        self.num_action = num_actions

    def forward(self, s, mask):
        

        x = torch.flatten(s, 0)
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)

        x = softmax(x, dim=0)
        
        coordinate = self.select_mask(s, mask)

        return x, coordinate
        
        
    def select_mask(self, s, mask):
        available_pixels = torch.nonzero(~mask).to(mask.device)
        if len(available_pixels) > 0:
            selected_idx = torch.randint(0, len(available_pixels), (1,)).item()
            selected_pixel = available_pixels[selected_idx]
            coordinate = (selected_pixel[0].item(), selected_pixel[1].item())
        else:
            coordinate = None  
        # 근데 policy에서 선택하게 해야하나? 이건 고민됨 
        return coordinate

class BackwardPolicy(nn.Module):
    def __init__(self, state_dim,hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim*20, hidden_dim*hidden_dim)
        self.dense2 = nn.Linear(hidden_dim*hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, s):
        x = torch.flatten(s, 0)
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)

        return softmax(x, dim=0)
