import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax

from torch.distributions import Categorical

import numpy as np
import pdb


class MLPForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim*10, hidden_dim*hidden_dim)
        self.dense2 = nn.Linear(hidden_dim*hidden_dim, hidden_dim*hidden_dim)
        self.dense3 = nn.Linear(hidden_dim*hidden_dim, hidden_dim)
        
        ## action mlp와 selection mlp를 따로 만들기
        ## action mlp는 action을 선택하는데 사용
        ## selection mlp는 선택된 action의 좌표를 반환하는데 사용 (이전의 select된 좌표는 선택되지 않아야 함, mask가 있으므로 제어하기)

        ## action mlp
        self.action_mlp = nn.Linear(hidden_dim, num_actions)

        ## selection mlp
        self.selection_mlp = nn.Linear(hidden_dim, state_dim)


        self.num_action = num_actions

    def forward(self, s, mask = None, iter = None):
        
        x = torch.flatten(s, 0)
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)
        x = relu(x)

        ## action mlp
        x_action = self.action_mlp(x)
        x_action = softmax(x_action, dim=0)

        ## selection mlp
        x_selection = self.selection_mlp(x)
        x_selection = relu(x_selection)

        # # selection Categorical 분포화
        x_selection = softmax(x_selection, dim=0)
        x_selection = Categorical(x_selection)

        if mask is not None:
            # x_selection = x_selection.logits.masked_fill(mask.flatten(), -np.inf)
            # x_selection = Categorical(softmax(x_selection, dim=0))
            coordinate = self.select_mask(x_selection, mask)
            return x_action, coordinate
        
        else:
            return x_action


    def select_mask(self, x, mask):
        h, w = mask.shape  # mask의 높이와 너비를 가져옵니다.
        
        # mask에서 False인 부분, 즉 선택 가능한 위치의 인덱스를 가져옵니다.
        available_idx = torch.nonzero(~mask).to(mask.device)
        
        # 선택 가능한 인덱스가 없다면 예외 처리
        if len(available_idx) == 0:
            raise ValueError("No available positions left to select.")
        
        # 선택 가능한 인덱스들 중에서 무작위로 하나를 선택합니다.
        selected = available_idx[torch.randint(0, len(available_idx), (1,)).item()]
        
        # 선택된 인덱스의 좌표를 반환합니다.
        return tuple(selected.tolist())

class MLPBackwardPolicy(nn.Module):
    def __init__(self, state_dim,hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim*10, hidden_dim*hidden_dim)
        self.dense2 = nn.Linear(hidden_dim*hidden_dim, hidden_dim*hidden_dim)
        self.dense3 = nn.Linear(hidden_dim*hidden_dim, num_actions)

    def forward(self, s):
        x = torch.flatten(s, 0)
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)

        return softmax(x, dim=0)
    

class LSTMForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # lstm cell 3개
        self.lstm1 = nn.LSTM(state_dim*state_dim, hidden_dim*hidden_dim) # 9000,900
        self.lstm2 = nn.LSTM(hidden_dim*hidden_dim, hidden_dim*hidden_dim)
        self.lstm3 = nn.LSTM(hidden_dim*hidden_dim, hidden_dim)

        self.action = nn.Linear(hidden_dim, num_actions)
        self.selection = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, s, mask = None, iter = None):
        # x = torch.flatten(s, 0)

        x, _ = self.lstm1(s.reshape(-1,1, 900))
        x = nn.Tanh()(x)
        x, _ = self.lstm2(x)
        x = nn.Tanh()(x)
        x, _ = self.lstm3(x)
        x = nn.Tanh()(x)

        ## selection mlp
        x_selection = self.selection(x)
        x_selection = relu(x_selection)
        # selection Categorical 분포화
        x_selection = softmax(x_selection, dim=0)
        x_selection = Categorical(x_selection)

        coordinate = self.select_mask(x_selection, mask)

        ac = self.action(x[iter,])
        ac =  softmax(ac, dim=0).squeeze(0)

        ac_prob = (ac + 1e-8) /  torch.sum((ac + 1e-8), dim=-1, keepdims=True)
        return ac_prob, coordinate
    
    def select_mask(self, x, mask):
        h, w = mask.shape  # mask의 높이와 너비를 가져옵니다.
        
        # mask에서 False인 부분, 즉 선택 가능한 위치의 인덱스를 가져옵니다.
        available_idx = torch.nonzero(~mask).to(mask.device)
        
        # 선택 가능한 인덱스가 없다면 예외 처리
        if len(available_idx) == 0:
            raise ValueError("No available positions left to select.")
        
        # 선택 가능한 인덱스들 중에서 무작위로 하나를 선택합니다.
        selected = available_idx[torch.randint(0, len(available_idx), (1,)).item()]
        
        # 선택된 인덱스의 좌표를 반환합니다.
        return tuple(selected.tolist())