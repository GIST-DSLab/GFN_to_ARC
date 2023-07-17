import torch
from torch.nn.functional import one_hot
from gflow.env import Env
import copy
# import torch.nn as nn


class Grid(Env):
    def __init__(self, size, device):
        self.size = size
        self.state_dim = size**2
        self.num_actions = 10  # down, right, terminate
        self.device = device

    def update(self, s, actions):
        idx = s.argmax(1)
        down, right = actions == 0, actions == 1
        idx[down] = idx[down] + self.size
        idx[right] = idx[right] + 1
        return one_hot(idx, self.state_dim).float().to(self.device)

    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions).to(self.device)
        idx = s.argmax(1) + 1
        right_edge = (idx > 0) & (idx % (self.size) == 0)
        bottom_edge = idx > self.size*(self.size-1)
        mask[right_edge, 1] = 0
        mask[bottom_edge, 0] = 0
        return mask

    def reward(self, s):
        grid = s.view(len(s), self.size, self.size)
        coord = (grid == 1).nonzero()[:, 1:].view(len(s), 2)
        R0, R1, R2 = 1e-2, 0.5, 2
        norm = torch.abs(coord / (self.size-1) - 0.5)
        R1_term = torch.prod(0.25 < norm, dim=1)
        R2_term = torch.prod((0.3 < norm) & (norm < 0.4), dim=1)
        return (R0 + R1*R1_term + R2*R2_term).to(self.device)


def rotate_left(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[j][4-i])
        rotate_state.append(temp)
    return rotate_state


def rotate_right(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[4-j][i])
        rotate_state.append(temp)
    return rotate_state


def horizontal_flip(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[4-i][j])
        rotate_state.append(temp)
    return rotate_state


def vertical_flip(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[i][4-j])
        rotate_state.append(temp)
    return rotate_state


def stop_action(state):
    return state


"""if a['action_sequence']['action_sequence'][i]['action']['tool'] in ['start', 'end', 'none']:
        target_action = 'stop_action'
      elif a['action_sequence']['action_sequence'][i]['action']['tool'] == 'reflectx':
        target_action = 'horizontal_flip'
      elif a['action_sequence']['action_sequence'][i]['action']['tool'] == 'reflecty':
        target_action = 'vertical_flip'
      elif a['action_sequence']['action_sequence'][i]['action']['tool'] == 'rotate':
        target_action = 'rotate_right'"""
