import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Geometric, OneHotCategorical

import numpy as np
from .log import Log
from collections import OrderedDict


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy= None, total_flow=None, env=None, device='cuda', env_style="point", num_actions=12, ep_len=10):
        super().__init__()
        # print("Initializing GFlowNet")

        if total_flow is None:
            self.total_flow = nn.Parameter(torch.tensor(1.0).to(device))
        else:
            self.total_flow = total_flow # Log Z 

        if backward_policy is not None:
            self.backward_policy = backward_policy.to(device)
        else :
            self.backward_policy = backward_policy

        self.forward_policy = forward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}
        self.num_actions = num_actions
        self.max_length = ep_len
        
        self.env_style = env_style  # "point", "bbox", ...
        
        self.mask = None
        self.emb_dag = None
        self.dag_s = None
        self.ac = None

        # self.embedding = nn.Embedding(10, 64)

    def set_env(self, env):
        self.env = env

    def forward_probs(self, s, mask, sample=True, action = None):
        """
        Returns a vector of probabilities over actions in a given state.
        """
        if sample : 
            # print(f"Forward probs called.")
            probs, selection = self.forward_policy(s, mask)
            # print(f"Forward probs completed. Probs shape: {probs.shape}, Selection shape: {selection.shape}")
            return probs, selection
        
        else : 
            probs, selection = self.forward_policy(s, mask)

            l = int(probs.shape[1]/2)
            
            logpf = self.logit_to_pf(probs[:,:l], sample=True, action= action)
            logpb = self.logit_to_pb(probs[:,l:])

            return logpf, logpb 
        
    def logit_to_pf(self, logits, sample=True, action = None):
        # fwd_prob = Categorical(logits=logits[0]) # 8/9 probs_s[0]이 맞는지 probs_s(원래)가 맞는지 확인
        # ac = fwd_prob.sample()
        # fwd_prob_s = fwd_prob.log_prob(ac)

        if 0.0 in logits[0] :
            logits = logits.clone()
            logits[0] = logits[0] + 1e-20
            
        if sample:
            fwd_prob = Geometric(probs=logits[0])
            # fwd_prob = Categorical(logits=logits[0]) # prob으로 해야함
            self.ac = fwd_prob.sample().argmin()
            # self.ac = fwd_prob.sample()
            fwd_prob_s = fwd_prob.log_prob(self.ac)[self.ac]
            # fwd_prob_s = fwd_prob.log_prob(self.ac)
        else : 
            fwd_prob = Geometric(probs=logits[0])
            self.ac = action
            fwd_prob_s = fwd_prob.log_prob(self.ac)


        if fwd_prob_s.dim() == 0:
            fwd_prob_s = fwd_prob_s.unsqueeze(0)
        return fwd_prob_s

    def logit_to_pb(self, logits):
        back_probs = logits
        back_probs_s = Categorical(logits=back_probs).log_prob(self.ac)
        # back_probs_s = Geometric(probs=back_probs[0]).log_prob(self.ac)[self.ac]

        if back_probs_s.dim() == 0:
            back_probs_s = back_probs_s.unsqueeze(0)
        return back_probs_s

    def sample_states(self, s0, info=None, return_log=True, batch_size=128, use_selection=False):

        if self.env is None:
            raise ValueError("Environment is not set. Please call set_env() before using this method.")

        iter = 0

        # Set initial state
        s = torch.tensor(s0["input"], dtype=torch.float).unsqueeze(0).to(self.device)
        answers = np.array(self.env.unwrapped.answer)
        t_ = torch.from_numpy(answers).unsqueeze(0).to(self.device)


        # pad_terminal = torch.zeros((batch_size, 30, 30), device=self.device)
        # pad_terminal[:, :t_.shape[1], :t_.shape[2]] = t_        
        
        # dag 행렬 

        # self.dag_s = torch.zeros((self.max_length+1, 30, 30), device=self.device)
        # self.dag_s[0] = s

        h, w = t_.shape[1], t_.shape[2]

        if use_selection: # Mask 수정해야함 (dag_s에 대한 mask)
            if self.mask is None or iter == 0:
                # print("Initializing mask")
                self.mask = torch.zeros((batch_size, 30, 30), dtype=torch.bool).to(self.device)
                self.mask[:, :h, :w] = True
                # print(f"Mask shape: {self.mask.shape}")
        else : 
            if batch_size > 1:
                self.mask = torch.zeros((batch_size, 30, 30), dtype=torch.bool).to(self.device)
                self.mask[:, :h, :w] = True  # 전체 배치에 대해 mask 설정
            else:
                self.mask = torch.zeros((30, 30), dtype=torch.bool).to(self.device)
                self.mask[:h, :w] = True  # 단일 배치에 대해 mask 설정

        log = Log(s[:,:3,:3], self.backward_policy, self.total_flow, self.env,tstate=s0, emb_s=None, num_actions = self.num_actions) if return_log else None
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        while not is_done.all():
            iter += 1
            
            if use_selection:
                active_mask = (~is_done).unsqueeze(1).unsqueeze(2)
                probs_s, selection = self.forward_probs(s * active_mask, self.mask.clone() * active_mask, iter)
            
                prob = Categorical(probs=probs_s)
                ac = prob.sample()
                
                actions = tuple([selection[i] for i in range(batch_size)] + [ac.cpu().numpy()])

            else:
                probs, selection = self.forward_probs(s, self.mask, sample=True)
                l = int(probs.shape[1]/2)

                fwd_prob_s = self.logit_to_pf(probs[:,:l])
                back_probs_s = self.logit_to_pb(probs[:,l:])



                actions = {
                    "operation" : self.ac.cpu().numpy(),
                    "selection" : self.mask.cpu().numpy()
                }


            results = self.env.step(actions)
            states, rewards, dones, truncated, infos = results

            gamma = 0.9

            is_done = torch.tensor(dones, device=self.device) 
            # rewards = torch.tensor(rewards * (gamma**iter) , dtype=torch.float, device=self.device)  # discount factor 적용
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)  # discount factor 적용 안함

 

            s = torch.where(~is_done, torch.tensor(states['grid'], dtype=torch.float, device=self.device).unsqueeze(0),s)

            # self.dag_s[iter] = s

            # if not is_done :
            #     s = torch.cat([s, torch.tensor(states['grid'], device=s.device)])
            # else  :
            #     break

            if use_selection:
                for i, sel in enumerate(selection):
                    if not is_done[i]:
                        self.mask[i, sel[0], sel[1]] = True

            if return_log:
                log.log(s=s.clone(), probs=fwd_prob_s,back_probs= back_probs_s, actions=self.ac, tstate=states, rewards=rewards, done=is_done)

            if (iter >= self.max_length) or is_done.all() :
                
                break

        return s, log if return_log else s
    

    """def MSE_reward(self, s, pad_terminal, scalefactor=100):
        s = self.normalize(s)
        pad_terminal = self.normalize(pad_terminal)
        r = ((pad_terminal.squeeze() - s.squeeze())**2 + 1e-6)
        r[torch.isinf(r)] = 0
        r = torch.mean(r)
        r = torch.exp(-r)
        if r == 0:
            r = torch.tensor(1e-10).to(self.device)
        return r*scalefactor

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
        gray_positions = [(x, y) for x in range(answer.shape[0]) for y in range(answer.shape[1]) if answer[x, y] == 5]
        black_positions = [(x, y) for x in range(answer.shape[0]) for y in range(answer.shape[1]) if answer[x, y] == 0]

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

        return torch.exp(torch.tensor(reward / 1000.0, dtype=torch.float32).to(self.device))

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())
"""