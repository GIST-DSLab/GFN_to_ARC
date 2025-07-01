import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax, log_softmax
import torch.nn.functional as F

from torch.distributions import Categorical

import numpy as np
import pdb
import math

def make_mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.LeakyReLU())
            # layers.append(nn.ReLU()) # LeakyReLU -> ReLU 8/13 실험 
    return nn.Sequential(*layers)

class EnhancedMLPForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, batch_size, embedding_dim = 64, ep_len=3, use_selection=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.use_selection = use_selection
        self.episode_length = ep_len+1

        self.positional_encoding = self.create_positional_encoding(state_dim, embedding_dim)
        
        f_d = self.state_dim * self.state_dim * self.embedding_dim #* self.episode_length
        ff_d = self.embedding_dim * hidden_dim
        self.feature_extractor = make_mlp([f_d, ff_d, ff_d//4, hidden_dim])
        self.action_mlp = nn.Linear(hidden_dim, num_actions*2)
        self.selection_mlp = nn.Linear(hidden_dim, state_dim * state_dim)

    def create_positional_encoding(self, size, dim):
        encoding = torch.zeros(size, size, dim)
        position = torch.arange(0, size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        encoding[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(0).repeat(size, 1, 1)
        encoding[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(0).repeat(size, 1, 1)
        return encoding

    def forward(self, state, mask=None):
        
        # if state.dim() == 4:
        #     batch_size, episode_length, height, width = state.size()
        # else : 
        #     episode_length, height, width = state.size()
        #     batch_size = 1

        # state = state.view(-1, height, width)
        
        batch_size, height, width = state.size()
        
        # Apply positional encoding
        pos_encoding = self.positional_encoding.to(state.device)
        state_encoded = state.unsqueeze(-1) * pos_encoding.unsqueeze(0)
        
        # Flatten and pass through feature extractor
        x = state_encoded.view(batch_size, -1)
        features = self.feature_extractor(x)
        
        # Action probabilities
        action_logits = self.action_mlp(features)
        # print(f"action_logits: {action_logits}")
        action_probs = F.softmax(action_logits, dim=-1)
        # print(f"action_logits shape: {action_logits.shape}, requires_grad: {action_logits.requires_grad}")
        
        if self.use_selection:
            # Selection probabilities
            selection_logits = self.selection_mlp(features)
            coordinate = self.select_mask(selection_logits, mask)
            
            return action_probs, coordinate
        else :
             return action_probs, mask

    def select_action(self, state):
        action_probs, selection_probs = self.forward(state)
        
        action_dist = torch.distributions.Categorical(action_probs)
        selected_action = action_dist.sample()

        selection_dist = torch.distributions.Categorical(selection_probs)
        selected_coordinate = selection_dist.sample()

        return selected_action, self.flat_index_to_2d(selected_coordinate, self.state_dim)

    def select_mask(self, x_selection, mask):
        
        # x_selection is a 1D tensor of probabilities after masking and softmax
        # mask is a 2D tensor where False indicates selectable positions
        batch_size, _ = x_selection.size()
        device = x_selection.device

        coordinates = []
        for i in range(batch_size):
            valid_indices = (~mask[i]).nonzero(as_tuple=False)
            if valid_indices.numel() == 0:
                raise ValueError(f"No available positions left to select for batch {i}.")
            
            valid_probs = x_selection[i][~mask[i].view(-1)]
            if torch.all(valid_probs == 0):
                valid_probs = torch.ones_like(valid_probs) / valid_probs.numel()
            selection_dist = Categorical(valid_probs)
            selected_index = selection_dist.sample()
            selected_coordinate = valid_indices[selected_index]
            
            coordinates.append(selected_coordinate)

    # Stack the coordinates into a single tensor
        coordinates = torch.stack(coordinates).to(device)
        
        return coordinates
    
    def flat_index_to_2d(self, index, dim):
        return torch.div(index, dim, rounding_mode='floor'), index % dim


class MLPForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, batch_size, use_selection=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.use_selection = use_selection

        self.dense1 = nn.Linear(state_dim * state_dim * 64, hidden_dim*hidden_dim) # 64 is the embedding dim
        self.dense2 = nn.Linear(hidden_dim*hidden_dim, hidden_dim*hidden_dim)
        self.dense3 = nn.Linear(hidden_dim * hidden_dim, hidden_dim*hidden_dim)
        
        # action mlp
        self.action_mlp = nn.Linear(hidden_dim*hidden_dim - state_dim*state_dim, num_actions)
        
        # selection mlp
        self.selection_mlp = nn.Linear(state_dim*state_dim, state_dim * state_dim)
        
        self.num_action = num_actions
        self.state_dim = state_dim

    def forward(self, s, mask=None, iter=None):
        x = s.view(s.size(0), -1).to(torch.float32) 

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))

        # action mlp
        x_action = self.action_mlp(x[:, :self.hidden_dim*self.hidden_dim - self.state_dim*self.state_dim])
        x_action = F.softmax(x_action, dim=1)

        if self.use_selection:
            # selection mlp
            x_selection = self.selection_mlp(x[:, self.hidden_dim*self.hidden_dim - self.state_dim*self.state_dim:]) ## 900개 
            x_selection = F.softmax(x_selection, dim=1)
            
            if mask is not None:
                # Masking the logits for selection
                x_selection = x_selection.masked_fill(mask.view(s.size(0), -1), -1e9)
                x_selection = F.softmax(x_selection, dim=1)  # Re-normalize probabilities
                
                # Select coordinate based on masked probabilities
                coordinate = self.select_mask(x_selection, mask)
                
                # Get the probability of the selected coordinate
                selection_prob = x_selection.gather(1, (coordinate[:, 0] * mask.size(2) + coordinate[:, 1]).unsqueeze(1))
                
                # Final action probability for each action
                final_action_prob = x_action * selection_prob
                
                return final_action_prob, coordinate
        else:
            action_logits = self.action_mlp(x)
            action_probs = F.softmax(action_logits, dim=1)
            return action_probs

    def select_mask(self, x_selection, mask):
        
        # x_selection is a 1D tensor of probabilities after masking and softmax
        # mask is a 2D tensor where False indicates selectable positions
        batch_size, _ = x_selection.size()
        device = x_selection.device

        coordinates = []
        for i in range(batch_size):
            valid_indices = (~mask[i]).nonzero(as_tuple=False)
            if valid_indices.numel() == 0:
                raise ValueError(f"No available positions left to select for batch {i}.")
            
            valid_probs = x_selection[i][~mask[i].view(-1)]
            if torch.all(valid_probs == 0):
                valid_probs = torch.ones_like(valid_probs) / valid_probs.numel()
            selection_dist = Categorical(valid_probs)
            selected_index = selection_dist.sample()
            selected_coordinate = valid_indices[selected_index]
            
            coordinates.append(selected_coordinate)

    # Stack the coordinates into a single tensor
        coordinates = torch.stack(coordinates).to(device)
        
        return coordinates
    
    def flat_index_to_2d(self, index, dim):
        return torch.div(index, dim, rounding_mode='floor'), index % dim
    
class MLPBackwardPolicy(nn.Module):
    def __init__(self, state_dim,hidden_dim, num_actions, batch_size):
        super().__init__()
        self.dense1 = nn.Linear(state_dim * state_dim, hidden_dim*hidden_dim)
        self.dense2 = nn.Linear(hidden_dim*hidden_dim, hidden_dim*hidden_dim)
        self.dense3 = nn.Linear(hidden_dim*hidden_dim, num_actions)

    def forward(self, s):
        x = s.view(s.size(0), -1).to(torch.float32)

        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)

        return softmax(x, dim=0)
    

