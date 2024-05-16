import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Categorical

import numpy as np
from .log import Log
from collections import OrderedDict

class RewardModel(nn.Module):
    def __init__(self):
        """
        Learnable reward model for GFlowNet
        using x (trajectory)
        label y (reward)
        """
        super().__init__()
        self.embedding = nn.Embedding(11, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc3 = nn.Linear(900,1)
    
    def forward(self, x):

        if len(x.shape) > 2 :
            b,_,_ = x.shape
        elif len(x.shape) == 2 :
            x = x.unsqueeze(0)
            b,_,_ = x.shape

        x = self.embedding(x.reshape(b,1,-1).long())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.fc3(x.squeeze())
        return x

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env=None, reward_model=None, reward_buffer = None, device='cuda'):
        super().__init__()
        self.total_flow = Parameter(torch.tensor(1.0).to(device))
        self.forward_policy = forward_policy.to(device)
        self.backward_policy = backward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}
        self.reward_model = reward_model

        self.env_style = "point" # "point","bbox" ...
        
        self.mask = None
        self.dag = None
        self.emb_dag = None

        self.reward_buffer = reward_buffer

        # self.embedding = nn.Sequential(
        #     nn.Conv2d(1, 30, kernel_size=1, stride=1, padding=0), 
        #     nn.ReLU(),
        #     nn.Conv2d(30, 30, kernel_size=1, stride=1, padding=0), 
        #     nn.ReLU(),
        #     nn.Flatten(), 
        #     nn.Linear(30*30, 100)
        # )
        self.embedding = nn.Embedding(11, 64)

    def forward_probs(self, s, mask, iter):
        """
        Returns a vector of probabilities over actions in a given state.
        """
        probs, selection = self.forward_policy(s, mask, iter)
        return probs, selection

    def sample_states(self, s0, info=None, return_log=True, i=None):
        iter = 0
        hreward = torch.tensor(0.0).to(self.device)

        if self.dag is not None:
            self.dag = None
            
        state = s0
        s = torch.tensor(s0["input"], dtype=torch.float).to(self.device)

        t_ = torch.tensor(self.env.unwrapped.answer).to(self.device)
        pad_terminal = torch.zeros_like(s) 
        pad_terminal[:t_.shape[0], :t_.shape[1]] = t_
        
        self.dag = torch.zeros(10, 30, 30).to(self.device)
        self.dag[0] = s
        # self.dag[9] = torch.tensor(pad_terminal).to(self.device)

        self.reward_buffer.add(pad_terminal, self.task_specific_reward(t_,t_,i))
        
        h, w = info["input_dim"]
        s[h:, w:] = 10
        
        pad_terminal_2 = torch.ones_like(s) * 10
        pad_terminal_2[:t_.shape[0], :t_.shape[1]] = t_

        self.dag[-1] = pad_terminal_2.to(self.device)

        # emb_s = self.embedding(s.long().flatten().unsqueeze(0))
        # emb_terminal = self.embedding(pad_terminal_2.long().flatten().unsqueeze(0))

        self.emb_dag = self.embedding(self.dag.long().flatten().unsqueeze(0)) 

        if self.mask is None or iter == 0:
            self.mask = torch.zeros((30,30), dtype=torch.bool).to(self.device)
            self.mask[h:, w:] = True

        log = Log(s, self.backward_policy, self.total_flow, self.env, emb_s=None) if return_log else None
        is_done = False

        while not is_done:
            iter += 1
            probs_s, selection = self.forward_probs(self.emb_dag, self.mask.clone(), iter)
            prob = Categorical(logits=probs_s)
            ac = prob.sample()

            self.actions = {"operation": ac, "selection": selection}
            result = self.env.step(self.actions)
        
            state, reward, is_done, _, info = result
            
            s = torch.tensor(state["grid"], dtype=torch.float).to(self.device)
            s[state["input_dim"][0]:, state["input_dim"][1]:] = 10
            
            # emb_s = self.embedding(s.long().flatten().unsqueeze(0))
            # mse = self.MSE_reward(emb_s, emb_terminal)
            reward = self.task_specific_reward(s, t_, i)

            # if torch.all(torch.eq(s, pad_terminal_2)):
            #     reward = mse
            # else:
            #     reward = torch.tensor(0.0).to(self.device)

            self.mask[selection[0], selection[1]] = True
            self.dag.clone()[iter] = s.clone()
            self.emb_dag = self.embedding(self.dag.long().flatten().unsqueeze(0))

            if return_log:
                log.log(s=self.dag.clone(), probs=prob.log_prob(ac).squeeze(), actions=ac, rewards=reward, embedding=self.emb_dag.clone(), done=is_done)

            if iter >= 9:
                return (s, log) if return_log else s

            if is_done:
                break

        return s, log if return_log else s

    def evaluate_trajectories(self, traj, actions):
        num_samples = len(traj)
        traj = traj.reshape(-1, traj.shape[-1])
        actions = actions.flatten()
        finals = traj[actions == len(actions) - 1]
        zero_to_n = torch.arange(len(actions))

        fwd_probs, selection = self.forward_probs(traj, mask=None, iter=None)
        fwd_probs = torch.where(actions == -1, 1, fwd_probs[zero_to_n, actions])
        fwd_probs = fwd_probs.reshape(num_samples, -1)

        actions = actions.reshape(num_samples, -1)[:, :-1].flatten()

        back_probs = self.backward_policy(traj)
        back_probs = back_probs.reshape(num_samples, -1, back_probs.shape[1])
        back_probs = back_probs[:, 1:, :].reshape(-1, back_probs.shape[2])
        back_probs = torch.where((actions == -1) | (actions == 2), 1, back_probs[zero_to_n[:-num_samples], actions])
        back_probs = back_probs.reshape(num_samples, -1)

        rewards = self.reward_model(finals)

        return fwd_probs, back_probs, rewards

    def MSE_reward(self, s, pad_terminal, scalefactor=100):
        s = self.normalize(s)
        pad_terminal = self.normalize(pad_terminal)
        r = ((pad_terminal.squeeze() - s.squeeze())**2 + 1e-6)
        r[torch.isinf(r)] = 0
        r = torch.mean(r)
        r = torch.exp(-r)
        if r == 0:
            r = torch.tensor(1e-10).to(self.device)
        return r
    
    def boltzman_reward(self, s, pad_terminal):
        s = self.normalize(s)
        pad_terminal = self.normalize(pad_terminal)
        w_ij = 1.5
        b_i = -torch.abs(pad_terminal - s)
        energy = -torch.sum(w_ij * (s * pad_terminal)) - torch.sum(b_i * s)
        if energy == float("inf"):
            energy = torch.tensor(-1e+10, device=energy.device)
        return -energy
    
    def pixel_Reward(self, s, pad_terminal):
        reward = 0
        for i in range(len(s)):
            for j in range(len(s[0])):
                if s[i][j] == pad_terminal[i][j]:
                    reward += 1
        return torch.tensor(reward, dtype=torch.float32).to(self.device)
    
    def human_reward(self):
        r = int(input("input reward: "))
        return torch.tensor(r, dtype=torch.float32).to(self.device)

    def task_specific_reward(self, s, answer, i):
        reward = 0
        gray_positions = [(x, y) for x in range(answer.shape[0])
                        for y in range(answer.shape[1]) if answer[x, y] == 5]
        black_positions = [(x, y) for x in range(answer.shape[0])
                       for y in range(answer.shape[1]) if answer[x, y] == 0]

        positions_dict = {
            0: [(3, 2), (4, 2), (3, 3), (4, 3), (10, 3)],
            1: [(2, 3), (9, 1), (9, 2), (10, 2), (10, 1), (4, 7), (5, 7), (6, 7), (7, 7), (4, 8), (5, 8), (6, 8), (7, 8), (4, 9), (5, 9), (6, 9), (7, 9), (4, 10), (5, 10), (6, 10), (7, 10)],
            2: [(4, 8), (4, 9), (4, 8), (5, 9)],
            3: []
        }

        positions_to_check = positions_dict.get(i, [])

        for pos in positions_to_check:
            if s[pos] == 2:
                reward += 150
            elif s[pos] == 0:
                reward += 10

        for x in range(s.shape[0]):
            for y in range(s.shape[1]):
                if (x, y) not in positions_to_check and s[x, y] == 2:
                    reward += 1

        for pos in gray_positions:
            if s[pos] == 0:
                reward += 1
            if s[pos] == 5:
                reward += 10
        for pos in black_positions:
            if s[pos] == 0:
                reward += 10

        return torch.exp(torch.tensor(reward/1000.0, dtype=torch.float32).to(self.device))

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    
    # def mahalanobis_reward(self, emb_s, emb_pad_terminal):
    #     s = self.normalize(emb_s)
    #     pad_terminal = self.normalize(emb_pad_terminal)
    #     if s.dim() == 3:
    #         s = s.squeeze(0)
    #     if pad_terminal.dim() == 3:
    #         pad_terminal = pad_terminal.squeeze(0)
    #     diff = s - pad_terminal
    #     cov_matrix = torch.mm(diff.T, diff) / (diff.size(0) - 1)
    #     inv_cov_matrix = torch.inverse(cov_matrix)
    #     mahalanobis_dist = torch.sqrt(torch.mm(torch.mm(diff, inv_cov_matrix), diff.T))
    #     return torch.exp(-mahalanobis_dist.squeeze())
            