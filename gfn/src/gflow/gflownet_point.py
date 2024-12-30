import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Categorical, Normal

import numpy as np
from scipy.spatial import distance

from .log import Log
from collections import OrderedDict


class reward_model(nn.Module):
    def __init__(self):
        """
        learnable reward model for GFlowNet
        using x (trajectory)
        label y (reward)
        """
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)


    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
        

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env=None, device='cuda'):
        super().__init__()
        self.total_flow = Parameter(torch.tensor(1.0).to(device))
        # self.std = Parameter(torch.tensor([0.5]*10, dtype = torch.float32).to(device))
        self.forward_policy = forward_policy.to(device)
        self.backward_policy = backward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}

        self.env_style = "point" # "point","bbox" ...
        
        self.mask = None
        self.dag = None
        self.emb_dag = None

        self.embedding = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(30, 30, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(30*30, 100) 
        )
        # self.embedding = nn.Embedding(11,32)
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        """
        Initializes a GFlowNet using the specified forward and backward policies
        acting over an environment, i.e. a state space and a reward function.

        Args:
            forward_policy: A policy network taking as input a state and
            outputting a vector of probabilities over actions

            backward_policy: A policy network (or fixed function) taking as
            input a state and outputting a vector of probabilities over the
            actions which led to that state

            env: An environment defining a state space and an associated reward
            function"""
        
    def forward_probs(self, s, mask, iter):
        """
        Returns a vector of probabilities over actions in a given state.

        Args:
            s: An NxD matrix representing N states
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
        self.dag[0] = s # 첫 번째 상태를 설정합니다.
        self.dag[9] = torch.tensor(pad_terminal).to(self.device) # 마지막 상태를 설정합니다.
        
        h, w = info["input_dim"]
        ## Embedding을 해보자 
        s[h:, w:] = 10  # input size 밖의 값은 10으로 채워넣기
        
        pad_terminal_2 = torch.ones_like(s) * 10
        pad_terminal_2[:t_.shape[0], :t_.shape[1]] = t_

        # embedding 행렬을 만드는 방식 ep_len,size*size,embedding_dim 
        emb_s = self.embedding(s.long().flatten().unsqueeze(0))
        emb_terminal = self.embedding(pad_terminal_2.long().flatten().unsqueeze(0))

        # self.emb_dag = torch.zeros(10,900,128).to(self.device)
        # self.emb_dag[0,] = emb_s
        # self.emb_dag[9] = emb_terminal

        # dag 행렬 자체를 pixel by pixel으로 embedding 하는 방식
        self.emb_dag = self.embedding(self.dag.long().flatten().unsqueeze(0)) 


        # 마스크 행렬 초기화, info에서 input dimension 조회해서 input dimension 밖은 True로 설정
        if self.mask is None or iter == 0:
            self.mask = torch.zeros((30,30), dtype=torch.bool) 
            self.mask[h:, w:] = True

        # grid_dim = info["input_dim"]

        log = Log(s, self.backward_policy, self.total_flow,
                  self.env, emb_s=None) if return_log else None
        is_done = False

        while not is_done:
            iter += 1
            probs_s, selection = self.forward_probs(self.emb_dag, self.mask, iter)
            prob = Categorical(logits = probs_s)
            ac = prob.sample()

            
            self.actions = {"operation": ac, "selection": selection}
            result = self.env.step(self.actions)
        
            state, reward, is_done, _, info = result
            
            s = torch.tensor(state["grid"], dtype = torch.float).to(self.device)
            s[state["input_dim"][0]:, state["input_dim"][1]:] = 10  # input size 밖의 값은 10으로 채워넣기
            
            ### s와 answer를 embedding space에 올려서 계산
            emb_s = self.embedding(s.long().flatten().unsqueeze(0))  

            ## s와 pad_terminal이 같으면 계산 아니면 0 
            mse = self.MSE_reward(emb_s, emb_terminal)
            if torch.all(torch.eq(s, pad_terminal_2)):
                reward = mse
            else:
                reward = torch.tensor(0.0).to(self.device)

            # mse = self.MSE_reward(emb_s, emb_terminal)
            # mse = self.mahalanobis_reward(emb_s, emb_terminal)
            # mse = self.pixel_Reward(s, pad_terminal_2)
            # boltzman = self.boltzman_reward(emb_s, emb_terminal)

            # alpha = 0.7
            # ime_reward = alpha*mse + (1-alpha)*reward  #reward 조합 생각해보기 
            # # ime_reward = alpha*boltzman + (1-alpha)*reward
            # # ime_reward = mse

            # if iter == 9:
            #     # hreward = self.human_reward()
            #     hreward = self.task_specific_reward(s, pad_terminal_2, i)

            # 마스크 & DAG 업데이트
            self.mask[selection[0], selection[1]] = True
            self.dag.clone()[iter] = s.clone()

            # 첫번째 임베딩 방식
            # self.emb_dag.clone()[iter] = emb_s.clone()

            # 두번째 임베딩 방식
            self.emb_dag = self.embedding(self.dag.long().flatten().unsqueeze(0))

            if return_log:
                log.log(s=self.dag, probs=prob.log_prob(ac).squeeze(), actions = ac, rewards=reward, embedding=self.emb_dag, done=is_done)  # log에 저장

            if iter >= 9:  # max_length miniarc = 25, arc = 900
                return (s, log) if return_log else s
            
            # if selection[0] == 4 & selection[1] == 4:
            #     return (s, log) if return_log else s

            if is_done:
                break

        # return (s, log) if return_log else s
        return s, log if return_log else s

    def evaluate_trajectories(self, traj, actions):
        """
        Returns the GFlowNet's estimated forward probabilities, backward
        probabilities, and rewards for a collection of trajectories. This is
        useful in an offline learning context where samples drawn according to
        another policy (e.g. a random one) are used to train the model.

        Args:
            traj: The trajectory of each sample

            actions: The actions that produced the trajectories in traj
        """
        num_samples = len(traj)
        traj = traj.reshape(-1, traj.shape[-1])
        actions = actions.flatten()
        finals = traj[actions == len(actions) - 1]
        zero_to_n = torch.arange(len(actions))

        fwd_probs = self.forward_probs(traj)
        fwd_probs = torch.where(
            actions == -1, 1, fwd_probs[zero_to_n, actions])
        fwd_probs = fwd_probs.reshape(num_samples, -1)

        actions = actions.reshape(num_samples, -1)[:, :-1].flatten()

        back_probs = self.backward_policy(traj)
        back_probs = back_probs.reshape(num_samples, -1, back_probs.shape[1])
        back_probs = back_probs[:, 1:, :].reshape(-1, back_probs.shape[2])
        back_probs = torch.where((actions == -1) | (actions == 2), 1,
                                 back_probs[zero_to_n[:-num_samples], actions])
        back_probs = back_probs.reshape(num_samples, -1)

        rewards = self.reward(finals)

        return fwd_probs, back_probs, rewards

    
    def MSE_reward(self, s, pad_terminal, scalefactor=100):
        """
        Returns the reward associated with a given state.

        Args:
            s: An NxD matrix representing N states
        """

        # MSE 
        # r = ((pad_terminal.squeeze() - s.squeeze())**2 + 1e-6) # pad_terminal은 ARC용
        # r[torch.isinf(r)] = 0
        # mse_reward = 1 / (r.sum() + 1) 

        # 만약 값이 inf 면 적당히 10000으로 대체
        # if mse_reward == float("inf"):
        #     mse_reward = 10000
        s = self.normalize(s)
        pad_terminal = self.normalize(pad_terminal)

        # r = torch.exp(-torch.norm(s - pad_terminal)**2 / scalefactor)

        r = ((pad_terminal.squeeze() - s.squeeze())**2 + 1e-6)
        r[torch.isinf(r)] = 0

        r = torch.mean(r) 
        r = torch.exp(-r)

        if r == 0 :
            r = torch.tensor(1e-10).to(self.device)

        return r
    
    def boltzman_reward(self, s, pad_terminal):
        """
        Returns the reward associated with a given state.

        Args:
            s: embedding된 상태
        """
        s = self.normalize(s)
        pad_terminal = self.normalize(pad_terminal)
        #boltzman energy
        w_ij = 1.5  # 연결 강도
        b_i = -torch.abs(pad_terminal - s)   # 외부 필드

        # 볼츠만 에너지 계산
        energy = -torch.sum(w_ij * (s * pad_terminal)) - torch.sum(b_i * s)

        # 에너지를 기반으로 한 보상
        # 에너지가 낮을수록 보상이 높아집니다.
        # e_reward = torch.exp(-energy) # nan이 발생하므로 일단 주석처리
        
        if energy == float("inf"):
            energy = torch.tensor(-1e+10, device= energy.device)
        return -energy # log를 씌웠다고 가정한 값
    
    def pixel_Reward(self, s, pad_terminal):
    
        # 맞는 픽셀 맞췄을 때 보상
        reward = 0
        for i in range(len(s)):
            for j in range(len(s[0])):
                if s[i][j] == pad_terminal[i][j]:
                    reward += 1
        
        return torch.tensor(reward, dtype = torch.float32).to(self.device)
    
    def human_reward(self):
        #리워드를 지급해야할 때 사람이 직접 입력해서 리워드를 줌
        r = int(input("input reward : "))
        reward = torch.tensor(r, dtype = torch.float32).to(self.device)

        return reward
    def task_specific_reward(self, s, answer, i):
        reward = 0
        gray_positions = [(x, y) for x in range(answer.shape[0])
                        for y in range(answer.shape[1]) if answer[x, y] == 5]
        black_positions = [(x, y) for x in range(answer.shape[0])
                       for y in range(answer.shape[1]) if answer[x, y] == 0]

        positions_dict = {
            0: [(3, 2), (4, 2), (3, 3), (4, 3), (10, 3)],
            1: [(2,3), (9,1), (9,2), (10,2), (10,1), (4,7), (5,7), (6,7), (7,7), (4,8), (5,8), (6,8), (7,8), (4,9), (5,9), (6,9), (7,9), (4,10), (5,10), (6,10), (7,10)],
            2: [(4,8), (4,9), (4,8), (5,9)],
            3: []
        }

        positions_to_check = positions_dict.get(i, [])

        for pos in positions_to_check:
            if s[pos] == 2:
                reward += 1000
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
            if s[pos] == 0 :
                reward +=10

        return torch.tensor(reward, dtype = torch.float32).to(self.device)
    
    def mahalanobis_reward(self, emb_s, emb_pad_terminal):
        # emb_s와 emb_pad_terminal은 torch.Tensor 객체라고 가정
        s = self.normalize(emb_s)
        pad_terminal = self.normalize(emb_pad_terminal)
        
        # 차원 확인 (N, D) 형태가 되어야 함
        if s.dim() == 3:
            s = s.squeeze(0)
        if pad_terminal.dim() == 3:
            pad_terminal = pad_terminal.squeeze(0)
        
        # 공분산 행렬 계산
        diff = s - pad_terminal
        cov_matrix = torch.mm(diff.T, diff) / (diff.size(0) - 1)
        inv_cov_matrix = torch.inverse(cov_matrix)

        # 마할라노비스 거리 계산
        mahalanobis_dist = torch.sqrt(torch.mm(torch.mm(diff, inv_cov_matrix), diff.T))

        # 보상 계산
        return torch.exp(-mahalanobis_dist.squeeze())
            
    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())