import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical, Normal

import numpy as np
from .log import Log
from collections import OrderedDict

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

        self.embedding = nn.Embedding(10, 30)
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
        
    def forward_probs(self, s, mask):
        """
        Returns a vector of probabilities over actions in a given state.

        Args:
            s: An NxD matrix representing N states
        """
        probs, selection = self.forward_policy(s, mask)

        return probs, selection

    def sample_states(self, s0, info=None, return_log=True):
        """
        Samples and returns a collection of final states from the GFlowNet.

        Args:
            s0: An NxD matrix of initial states

            return_log: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.)
        """
        iter = 0
        if self.dag is not None:
            self.dag = None
            
        if type(s0) is not torch.float :
            s0 = s0["input"]
        s = torch.tensor(s0, dtype=torch.float).to(self.device)

        self.dag = torch.zeros(30, 30, 30).to(self.device)
        self.dag[0] = s # 첫 번째 상태를 설정합니다.

        # 마스크 행렬 초기화, info에서 input dimension 조회해서 input dimension 밖은 True로 설정
        h, w = info["input_dim"]
        if self.mask is None or iter == 0:
            self.mask = torch.zeros((30,30), dtype=torch.bool) 
            self.mask[h:, w:] = True

        # grid_dim = info["input_dim"]

        log = Log(s, self.backward_policy, self.total_flow,
                  self.env) if return_log else None
        is_done = False

        while not is_done:
            iter += 1
            # probs = self.forward_probs(s)  # 랜덤액션?
            probs_s, selection = self.forward_probs(self.dag, self.mask)
            prob = Categorical(logits = probs_s)
            ac = prob.sample()

            if self.env_style == "point":
                self.actions["operation"] = ac
                self.actions["selection"] = selection  # selection 어떻게
                result = self.env.step(self.actions)
        
            # reward 는 spase reward 이기 때문에 따로 reward 함수를 만들어서 log에 저장하는 함수를 만들어야함
            state, reward, is_done, _, info = result
            s = torch.tensor(state["grid"], dtype = torch.float).to(self.device)
            
            # mse = self.reward(s)
            boltzman = self.boltzman_reward(s)

            alpha = 0.7
            # ime_reward = alpha*mse + (1-alpha)*reward  #reward 조합 생각해보기 
            ime_reward = alpha*boltzman + (1-alpha)*reward

            # 마스크 & DAG 업데이트
            self.mask[selection[0], selection[1]] = True
            self.dag.clone()[iter] = s.clone() 

            if return_log:
                log.log(s=self.dag.clone(), probs=prob.log_prob(ac).squeeze(), actions = ac, rewards=ime_reward, total_flow=self.total_flow, done=is_done)  # log에 저장

            if iter >= 29:  # max_length miniarc = 25, arc = 900
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
    
    def reward(self, s):
        """
        Returns the reward associated with a given state.

        Args:
            s: An NxD matrix representing N states
        """
        terminal = torch.tensor(self.env.unwrapped.answer).to("cuda")
        pad_terminal = torch.zeros_like(s)
        pad_terminal[:terminal.shape[0], :terminal.shape[1]] = terminal

        # MSE 
        r = ((pad_terminal - s)**2 + 1e-6) # pad_terminal은 ARC용
        for i in range(len(r)):
            for j in range(len(r[0])):
                if r[i][j] == float("inf"):
                    r[i][j] = 0
        mse_reward = 1 / (r.sum() + 1) 

        # 만약 값이 inf 면 적당히 10000으로 대체
        # if mse_reward == float("inf"):
        #     mse_reward = 10000

        return mse_reward
    
    def boltzman_reward(self, s):
        """
        Returns the reward associated with a given state.

        Args:
            s: An NxD matrix representing N states
        """
        terminal = torch.tensor(self.env.unwrapped.answer).to("cuda")
        pad_terminal = torch.zeros_like(s)
        pad_terminal[:terminal.shape[0], :terminal.shape[1]] = terminal

        #boltzman energy
        w_ij = 1.0  # 연결 강도
        b_i = -torch.abs(pad_terminal - s)   # 외부 필드

        # 볼츠만 에너지 계산
        energy = -torch.sum(w_ij * (s * pad_terminal)) - torch.sum(b_i * s)

        # 에너지를 기반으로 한 보상
        # 에너지가 낮을수록 보상이 높아집니다.
        e_reward = torch.exp(-energy)
        if e_reward == float("inf"):
            e_reward = torch.tensor(10000, device= energy.device)
        return e_reward
    

    def human_reward(self):
        #리워드를 지급해야할 때 사람이 직접 입력해서 리워드를 줌
        reward = torch.tensor(input("input reward : "), dtype = torch.float32).to(self.device)

        return reward