import torch
from torch.nn.parameter import Parameter
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

from gflow.gflownet_target import GFlowNet
from policy_target import MLPForwardPolicy, MLPBackwardPolicy, EnhancedMLPForwardPolicy
from gflow.utils import trajectory_balance_loss, detailed_balance_loss, subtrajectory_balance_loss
from ARCenv.wrapper import env_return  

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from tqdm import tqdm
from collections import deque

import os
import sys
import copy
import json
from typing import Tuple

import arcle
from arcle.loaders import ARCLoader, MiniARCLoader

import pdb
import random
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = ARCLoader()
miniloader = MiniARCLoader()

render_mode = None  # None  # ansi

TASKNUM = 178
CUDANUM = 0
ACTIONNUM = 5
LOSS = "trajectory_balance_loss"  # "trajectory_balance_loss", "subtb_loss", "detailed_balance_loss"

WANDB_USE = True                
FILENAME = f"geometric_rewardsum_10,10 task {TASKNUM}"                              

if WANDB_USE:
    wandb.init(project="gflow_re", 
               entity="hsh6449", 
               name=f"local+offpolicy cuda{CUDANUM}, ep10, a{ACTIONNUM}, reward, task {TASKNUM}, onpolicy")


class ReplayBuffer:
    def __init__(self, capacity: int, initial_threshold: float = 1.0, 
                 max_threshold: float = 10.0, threshold_increase_rate: float = 0.01,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.device = device
        self.capacity = capacity
        self.reward_threshold = initial_threshold
        self.max_threshold = max_threshold
        self.threshold_increase_rate = threshold_increase_rate
        self.reward_sum = 0
        self.reward_count = 0

    def add(self, state, action, reward, log_prob, back_prob, traj):
        last_reward = reward[-1]
        
        # capacity 초과 시 낮은 리워드 샘플을 우선적으로 제거
        if len(self.buffer) >= self.capacity:
            self._remove_oldest_samples()
        
        self.buffer.append((state, action, reward, log_prob, back_prob, traj))
        self.priorities.append(last_reward)
        
        # 평균 리워드 업데이트
        self.reward_sum += last_reward
        self.reward_count += 1
        
        # reward threshold 업데이트
        self.update_reward_threshold()

    def _remove_oldest_samples(self):
        # 오래된 샘플들을 우선적으로 제거
        del self.buffer[0]
        del self.priorities[0]

    def update_reward_threshold(self):
        if self.reward_count % 100 == 0:  # 100번의 경험마다 threshold를 업데이트
            new_threshold = min(self.reward_threshold + self.threshold_increase_rate, self.max_threshold)
            self.reward_threshold = new_threshold

    def batch_sample(self, batch_size: int):
        # if not self.is_ready_for_sampling():
        #     return None
        
        high_reward_threshold = 5  # 이 값은 필요에 따라 조정 가능
        high_reward_samples = [(state, action, reward, log_prob, back_prob, traj, priority) 
                               for (state, action, reward, log_prob, back_prob, traj), priority 
                               in zip(self.buffer, self.priorities) if priority >= high_reward_threshold]
        low_reward_samples = [(state, action, reward, log_prob, back_prob, traj, priority) 
                              for (state, action, reward, log_prob, back_prob, traj), priority 
                              in zip(self.buffer, self.priorities) if priority < high_reward_threshold]

        # 높은 리워드 샘플의 비율 조정 (예: 60% 높은 리워드, 40% 낮은 리워드)
        high_reward_sample_size = int(batch_size * 0.6)
        low_reward_sample_size = batch_size - high_reward_sample_size

        high_reward_batch = random.sample(high_reward_samples, k=min(high_reward_sample_size, len(high_reward_samples)))
        low_reward_batch = random.sample(low_reward_samples, k=min(low_reward_sample_size, len(low_reward_samples)))

        # 샘플 수가 부족한 경우 다른 그룹에서 채워넣기
        if len(high_reward_batch) < high_reward_sample_size:
            low_reward_batch.extend(random.sample(low_reward_samples, k=high_reward_sample_size - len(high_reward_batch)))
        elif len(low_reward_batch) < low_reward_sample_size:
            high_reward_batch.extend(random.sample(high_reward_samples, k=low_reward_sample_size - len(low_reward_batch)))

        # 최종 배치 생성 및 섞기
        batch = high_reward_batch + low_reward_batch
        random.shuffle(batch)

        # 샘플 언패킹
        states, actions, rewards, log_probs, back_probs, traj,  _ = zip(*batch)

        states = [state.to(self.device) for state in states]
        # actions = [action if isinstance(action, int) else action.to(self.device) for action in actions]
        actions = list(actions)
        rewards = [torch.tensor(episode_rewards, device=self.device) for episode_rewards in rewards] 
        log_probs = [torch.tensor(episode_log_probs, device=self.device) for episode_log_probs in log_probs]
        back_probs = [torch.tensor(episode_back_probs, device=self.device) for episode_back_probs in back_probs]
        traj = [[grid.to(self.device) if isinstance(grid, torch.Tensor) else torch.tensor(grid, device=self.device) 
              for grid in episode_traj] 
             for episode_traj in traj]


        return (states, actions, rewards, log_probs, back_probs, traj)

    
    def e_sample(self, epsilon: float, reward_threshold: float = 5.0):

        # if not self.is_ready_for_sampling():
        #     return None
        
        if random.random() < epsilon:  # exploration
            sample = random.choice(self.buffer)
        else:  # exploitation: sample from high-reward experiences
        # 리워드가 threshold 이상인 샘플들 필터링
            high_reward_samples = [(state, action, reward, log_prob, back_prob, traj) 
                                for (state, action, reward, log_prob, back_prob, traj), priority 
                                in zip(self.buffer, self.priorities) if priority >= reward_threshold]
        
            if len(high_reward_samples) > 0:
                sample = random.choice(high_reward_samples)
            else:
                # threshold 이상의 리워드가 없으면 exploration으로 대체
                sample = random.choice(self.buffer)
        state, action, reward, log_prob, back_prob, traj = sample

        state = state.to(self.device)
        reward = torch.tensor(reward, device=self.device)
        log_prob = torch.tensor(log_prob, device=self.device)
        back_prob = torch.tensor(back_prob, device=self.device)
        traj = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in traj]

        return (state, action, reward, log_prob, back_prob, traj)
    
    def priority_based_sampling(self, batch_size: int):
        # 우선순위 확률 분포 생성 (높은 우선순위일수록 확률 증가)
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()  # 우선순위의 비율로 샘플링 확률 계산

        # 우선순위 기반 샘플링 수행
        sampled_indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in sampled_indices]

        # 샘플 언패킹
        states, actions, rewards, log_probs, back_probs, traj = zip(*batch)
        states = [state.to(self.device) for state in states]
        actions = list(actions)
        rewards = [torch.tensor(episode_rewards, device=self.device) for episode_rewards in rewards]
        log_probs = [torch.tensor(episode_log_probs, device=self.device) for episode_log_probs in log_probs]
        back_probs = [torch.tensor(episode_back_probs, device=self.device) for episode_back_probs in back_probs]
        traj = [[grid.to(self.device) if isinstance(grid, torch.Tensor) else torch.tensor(grid, device=self.device) 
              for grid in episode_traj] 
             for episode_traj in traj]
        # pdb.set_trace()
        return (states, actions, rewards, log_probs, back_probs, traj)

    def is_ready_for_sampling(self):
        if self.reward_count == 0:
            return False
        return (self.reward_sum / self.reward_count) >= self.reward_threshold

    def get_average_reward(self):
        if self.reward_count == 0:
            return 0
        return self.reward_sum / self.reward_count

    def __len__(self):
        return len(self.buffer)

def train(num_epochs, batch_size, device, env_mode, prob_index, num_actions, args, use_offpolicy=False):
    if env_mode == "entire":
        use_selection = False
    else:
        use_selection = True

    train_start = False
    env = env_return(render=render_mode, data=loader, options=None, batch_size=1, mode=env_mode)

    forward_policy = EnhancedMLPForwardPolicy(30, hidden_dim=256, num_actions=num_actions, batch_size=batch_size, embedding_dim=32, ep_len=args.ep_len, use_selection=use_selection).to(device)
    # backward_policy = MLPBackwardPolicy(30, hidden_dim=512, num_actions=num_actions, batch_size=batch_size).to(device)
    total_flow = Parameter(torch.tensor(1.0).to(device)) # Log Z 

    model = GFlowNet(forward_policy, backward_policy=None,total_flow = total_flow, env=env, device=device, env_style=env_mode, num_actions=num_actions, ep_len=args.ep_len).to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=0.0001)
    scheduler = CosineAnnealingLR(opt, T_max=10000, eta_min=0.00001)    

    sampling_model = None
    sampling_model_use = False
    sampling_method = args.sampling_method

    traj_length_reg = TrajLengthRegularization(target_length=args.ep_len, lambda_reg=0.01)
    eval_buffer = TrajectoryBuffer()

    if sampling_method == "egreedy" or use_offpolicy == False:
        batch_size = 1

    if use_offpolicy:
        replay_buffer = ReplayBuffer(capacity=10000, device = device)
        min_buffer_size = 100  # 충분한 데이터를 모은 후에 학습 시작
        update_freq = 200  # 매 update_freq 스텝마다 학습을 진행
        num_replay_steps = 50  # 버퍼 내에서 학습할 횟수
        sampling_model_use = True

    if sampling_model_use :
        sampling_model = copy.deepcopy(model)
        sampling_model.set_env(env)
        sampling_model.train()
        sampling_opt = AdamW(sampling_model.parameters(), lr=0.0001)
        # sampling_tau = 0.001  # 샘플링 모델 업데이트 속도 조절

    if sampling_model_use: 
        # Sampling 모델 학습 과정이 오히려 초기파라미터와의 차이 때문에 좋은 trajectory가 들어갔음에도 
        print("Sampling model Training start")
        
        val_acc_s = 0
        average_acc = []
        for i in tqdm(range(100)) : 
            state, info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": 0})
            if not train_start:
                # 초기 온라인 학습: sampling_model 사용
                result, sample_log = sampling_model.sample_states(state, info, return_log=True, batch_size=1, use_selection=use_selection)
                s, log = result if len(result) == 2 else (result, None)
                reward = sample_log.rewards[-1] if sample_log else None
                reward = compute_reward_with_penalty(sample_log.traj, reward, penalty=1.0)

                if LOSS == "trajectory_balance_loss":
                    loss, total_flow, re = trajectory_balance_loss(
                        sample_log.total_flow,
                        reward,
                        sample_log.fwd_probs,
                        sample_log.back_probs,
                        batch_size=1
                    )
                alpha = 0.2
                traj_length_reg_loss = traj_length_reg(sample_log.traj, batch_size=1)
                loss = loss*alpha + (1-alpha)*traj_length_reg_loss
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sampling_model.parameters(), 0.5)
                sampling_opt.step()
                sampling_opt.zero_grad()

                if WANDB_USE:
                    wandb.log({"Sample Model loss": loss.item(), "Sample Model total_flow": total_flow.exp().item(), "Sample Model reward": re.item()})

                ## Evaluation : average training accuracy
                correct = 0
                num_samples = 100

                if i % 20 == 0:
                    for k in range(num_samples):

                        eval_state, eval_info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": 0})
                        eval_s, eval_log = sampling_model.sample_states(eval_state, eval_info, return_log=True, batch_size=1, use_selection=use_selection)
                        eval_s = eval_s.cpu().detach().numpy()[0][:eval_info["answer_dim"][0], :eval_info["answer_dim"][1]]
                        answer = np.array(env.unwrapped.answer)

                        if np.array_equal(eval_s, answer):
                            correct += 1

                    val_acc_s = correct / num_samples
                    average_acc.append(val_acc_s)
                    if WANDB_USE:
                        wandb.log({"Sample Model val_accuracy": val_acc_s})

                    if np.mean(average_acc) > 0.5:
                        break


    if use_offpolicy:
        print(f"Using off-policy learning - Sampling Method : {sampling_method}")

    for epoch in tqdm(range(num_epochs)):
        state, info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": epoch})

        for step in tqdm(range(120000)):
            if use_offpolicy:
                
                if sampling_model_use:
                # Sampling model 활용 Off-policy 학습 로직
                    with torch.no_grad():
                        result = sampling_model.sample_states(state, info, return_log=True, batch_size=1, use_selection=use_selection)

                    s, sample_log = result 

                    rewards_ = [compute_reward_with_penalty(sample_log.traj, reward, penalty=1.0).cpu().detach() for reward in sample_log.rewards]
                    fwd_probs_ = [fwd_probs.cpu().detach() for fwd_probs in sample_log.fwd_probs]
                    back_probs_ = [back_probs.cpu().detach() for back_probs in sample_log.back_probs]
                    traj_ = [t.cpu().detach() for t in sample_log.traj]
                    
                    replay_buffer.add(s.cpu().detach(), sample_log.actions, rewards_, fwd_probs_, back_probs_, traj_)
                else :
                # 메인 모델 활용 버퍼 수집 
                    result = model.sample_states(state, info, return_log=True, batch_size = 1, use_selection=use_selection)
                    s, log = result 
                    
                    reward = log.rewards[-1]
                    
                    rewards_ = [reward.cpu().detach() for reward in log.rewards]
                    fwd_probs_ = [fwd_probs.cpu().detach() for fwd_probs in log.fwd_probs]
                    back_probs_ = [back_probs.cpu().detach() for back_probs in log.back_probs]
                    traj_ = [t.cpu().detach() for t in log.traj]

                    replay_buffer.add(s.cpu().detach(), log.actions, rewards_, fwd_probs_, back_probs_, traj_)

                if len(replay_buffer) >= min_buffer_size and step % update_freq == 0 : #and replay_buffer.is_ready_for_sampling():
                    print(f"Training from buffer at step {step+1}...")
                    if train_start == False:
                        train_start = True
                    
                    for _ in tqdm(range(num_replay_steps)): 
                        if sampling_method == "prt": # "prt", "fixed_ratio", "egreedy"
                            terminal_state, actions, rewards, log_probs_s, back_probs_s, traj = replay_buffer.priority_based_sampling(batch_size)
                        elif sampling_method == "fixed_ratio":
                            terminal_state, actions, rewards, log_probs_s, back_probs_s, traj = replay_buffer.batch_sample(batch_size)
                        elif sampling_method == "egreedy":
                            terminal_state, actions, rewards, log_probs_s, back_probs_s, traj = replay_buffer.e_sample(0.1)

                        current_log_probs = []
                        current_back_probs = []
                        
                        # pdb.set_trace()

                        if batch_size > 1:
                            for i in range(batch_size):
                                # 이전 구현 
                                # state, info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": epoch})
                                # current_result, current_log = model.sample_states(state, info, return_log=True, batch_size=1, use_selection=use_selection)

                                # current_log_probs.append(current_log.fwd_probs)
                                # current_rewards.append(current_log.rewards[-1])
                                # current_back_probs.append(current_log.back_probs)

                                # 9/5 새로운 구현 
                                fwd_probs_s = []
                                back_probs_s = []
                                for j in range(len(traj[i])-1):
                                    fwd_probs, back_probs  = model.forward_probs(traj[i][j], model.mask, sample=False, action=actions[i][j]) 
                                    fwd_probs_s.append(fwd_probs)
                                    back_probs_s.append(back_probs)

                                current_log_probs.append(fwd_probs_s)
                                current_back_probs.append(back_probs_s)
                        else : 
                            
                            for j in range(len(traj)-1):
                                # pdb.set_trace()
                                fwd_probs, back_probs  = model.forward_probs(traj[j], model.mask, sample=False, action=actions[j])
                                current_log_probs.append(fwd_probs)
                                current_back_probs.append(back_probs)

                        # current_log_probs.append(fwd_probs_s)
                        # current_back_probs.append(back_probs_s)
                    # current_log_probs = torch.stack(current_log_probs).squeeze()
                    # current_rewards = torch.tensor(current_rewards, device=model.device)
                    # current_back_probs = torch.stack(current_back_probs).squeeze()

                    # ratios = (torch.tensor([sum(i) for i in current_log_probs], device=model.device).mean() - torch.tensor([sum(j) for j in log_probs_s]).mean()).exp() 


                        if LOSS == "trajectory_balance_loss":
                            loss, total_flow, re = trajectory_balance_loss(
                                model.total_flow,
                                rewards,
                                current_log_probs,
                                current_back_probs,
                                batch_size=batch_size
                            )
                        elif LOSS == "detailed_balance_loss":
                            loss, total_flow, re = detailed_balance_loss(
                                log.total_flow,
                                rewards,
                                current_log_probs,
                                log.back_probs,
                                torch.tensor(env.unwrapped.answer).to(device)
                            )
                        elif LOSS == "subtb_loss":
                            loss = subtrajectory_balance_loss(log.traj, current_log_probs, log.back_probs)
                            re = rewards[-1]

                        if batch_size > 1 : 
                            importance_weights = compute_importance_weights(current_log_probs, log_probs_s, batch_size)
                        else : 
                            current_log_probs = [current_log_probs]
                            log_probs_s = [log_probs_s]
                            importance_weights = compute_importance_weights(current_log_probs, log_probs_s, batch_size)
                        
                        traj_length_reg_loss = traj_length_reg(traj, batch_size=batch_size)
                        clipped_loss = (loss * importance_weights * traj_length_reg_loss).sum() # batch 만큼 나옴 mean()을 취하면 scalar

                        # if (step % update_freq == 0) & (len(replay_buffer) >= min_buffer_size) : #& replay_buffer.is_ready_for_sampling():
                        clipped_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        opt.step()
                        opt.zero_grad()

                        scheduler.step()

                        tqdm.write(f"Loss: {clipped_loss.item():.3f}")

                    # if sampling_model_use:
                    #     # 샘플링 모델 업데이트
                    #     with torch.no_grad():
                    #         for param, target_param in zip(model.parameters(), sampling_model.parameters()):
                    #             target_param.data.copy_(sampling_tau * param.data + (1 - sampling_tau) * target_param.data)
            else:
                if train_start == False:
                    train_start = True
                    print("on-policy training start")
                # 기존 on-policy 학습 로직
                result = model.sample_states(state, info, return_log=True, batch_size=1, use_selection=use_selection)
                s, log = result if len(result) == 2 else s
                rewards = compute_reward_with_penalty(log.traj, log.rewards[-1], penalty=1.0)

                if LOSS == "trajectory_balance_loss":
                    loss, total_flow, re = trajectory_balance_loss(
                        log.total_flow,
                        # log.rewards,
                        rewards,
                        log.fwd_probs,
                        log.back_probs,
                        batch_size=batch_size
                    )
                elif LOSS == "detailed_balance_loss":
                    loss, total_flow, re = detailed_balance_loss(
                        log.total_flow,
                        log.rewards,
                        log.fwd_probs,
                        log.back_probs,
                        torch.tensor(env.unwrapped.answer).to(device)
                    )
                elif LOSS == "subtb_loss":
                    loss = subtrajectory_balance_loss(log.traj, log.fwd_probs, log.back_probs)
                    re = log.rewards[-1]

                alpha = 0.2
                traj_length_reg_loss = traj_length_reg(log.traj, batch_size=1)
                clipped_loss = loss*alpha + (1-alpha)*traj_length_reg_loss
                # clipped_loss = loss

                clipped_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                opt.zero_grad()

            if train_start:
                if WANDB_USE:
                    wandb.log({"loss": clipped_loss.item(), "total_flow": total_flow.exp().item(), "reward": re.mean().item()})
            

            ## Evaluation : average training accuracy
            ## 다시 만들어야함 
                # [.]100개중에 몇개 맞았는지
                # [.]맞은 것의 trajectory는 어떻게 되는지 

            if step % 100 == 0:
                correct = 0
                num_samples = 100

                for k in range(num_samples):
                    eval_state, eval_info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": epoch})
                    """
                    adaptation true면 examples
                    adaptation false면 tests
                    """

                    eval_s, eval_log = model.sample_states(eval_state, eval_info, return_log=True, batch_size=1, use_selection=use_selection)
                

                    if step % 10000 == 0:
                        eval_buffer.add(eval_log.traj, eval_log.rewards[-1])
                        
                    # if batch_size == 1:
                    eval_s = eval_s.cpu().detach().numpy()[:,:eval_info["input_dim"][0], :eval_info["input_dim"][1]][0]
                    answer = np.array(env.unwrapped.answer)
                    # else:
                    #     eval_s = eval_s.cpu().detach().numpy()[:,:eval_info["input_dim"][0][0], :eval_info["input_dim"][0][1]]
                        # answer = np.array(env.call('get_answer'))
                    
                    
                    # if batch_size == 1:
                    if eval_s.shape != answer.shape:
                        eval_s = eval_s[0]
                    if np.array_equal(eval_s, answer):
                        correct += 1
                    # else:
                    #     for i in range(batch_size):
                    #         if np.array_equal(eval_s[i], answer[i]):
                    #             correct += 1

                val_acc = correct / num_samples #* batch_size)
                if WANDB_USE:
                    wandb.log({"val_accuracy": val_acc})

            if step % 10 == 0:
                torch.save(model.state_dict(), f"model_{epoch}.pt")
            if step % 10000 == 0 :
                eval_buffer.save(f"eval_samples_{FILENAME}_step_{step}.json")
                eval_buffer.clear()

            state, next_info = env.reset(options={"prob_index": prob_index, "adaptation": True, "subprob_index": epoch})

    return model, sampling_model if use_offpolicy else None, env

def compute_importance_weights(current_log_probs, old_log_probs, batch_size):

    # current_sums  = torch.stack([torch.sum(torch.stack(probs)) for probs in current_log_probs])
    # old_sums  = torch.stack([torch.sum(probs) for probs in old_log_probs])
    # ratios = torch.exp(current_sums - old_sums)
    # # return torch.clamp(ratios, 0.8, 1.2)
    ratios = []
    for cur_probs, old_probs in zip(current_log_probs, old_log_probs):
        # 에피소드 내의 각 스텝에 대해 log-prob의 차이를 계산
        stepwise_ratios = []
        for cur_prob, old_prob in zip(cur_probs, old_probs):
            # 각 스텝의 log-prob 차이에 대해 exp를 적용하여 비율 계산
            stepwise_ratio = torch.exp(cur_prob - old_prob)
            stepwise_ratios.append(stepwise_ratio)

        # 각 에피소드에 대해 중요도 비율을 평균으로 계산
        episode_ratio = torch.mean(torch.stack(stepwise_ratios))
        ratios.append(episode_ratio)

    # 최종적으로 배치 내 모든 에피소드에 대해 평균을 구함
    final_ratios = torch.mean(torch.stack(ratios))

    return torch.clamp(final_ratios, 0.8, 1.2)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def detect_cycle(traj):
    visited_states = set()
    detect_count = 0 
    for state in traj:
        state_tuple = tuple(state.cpu().detach().numpy().flatten())  # state를 고유하게 변환하기 위해 tuple로 변환
        if state_tuple in visited_states:
            detect_count += 1  # cycle이 발견된 경우
        visited_states.add(state_tuple)
    return detect_count  # cycle이 없는 경우

def compute_reward_with_penalty(traj, base_reward, penalty=0.1):
    """
    traj: 현재 에피소드의 trajectory (상태들의 리스트)
    base_reward: 기본 보상
    penalty: cycle이 감지되었을 때 적용할 페널티 (기본값 0.1)
    """
    detect_count = detect_cycle(traj)
    if detect_count > 0:
        # print("Cycle detected! Applying penalty.")
        reward = base_reward - penalty*detect_count
        if reward < 0:
            return torch.tensor(0.0, device=reward.device) # 음수 보상은 허용하지 않음
        return reward # 페널티 적용
    return base_reward  # cycle이 없으면 기본 보상 유지


class TrajLengthRegularization(nn.Module):
    def __init__(self, target_length, lambda_reg=0.1):
        super().__init__()
        self.target_length = 3
        self.lambda_reg = lambda_reg

    def forward(self, trajectories, batch_size=None):
        if batch_size > 1 :
            traj_lengths = torch.tensor([len(traj) for traj in trajectories], device=trajectories[0][0].device, dtype=torch.float)
        else:
            traj_lengths = torch.tensor(len(trajectories), device=trajectories[0][0].device, dtype=torch.float)
        length_diff = traj_lengths - self.target_length
        reg_loss = torch.mean(length_diff**2)  # MSE loss
        return self.lambda_reg * reg_loss
    
class TrajectoryBuffer:
    def __init__(self):
        self.trajectories = []
        self.rewards = []

    def add(self, trajectory, reward):
        self.trajectories.append([t.detach().cpu().numpy().tolist() for t in trajectory])
        self.rewards.append(reward.detach().cpu().numpy().tolist())

    def save(self, filename='data.json'):
        data = {
            'trajectories': self.trajectories,
            'rewards': self.rewards
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def clear(self):
        self.trajectories = []
        self.rewards = []

def save_trajectory_and_rewards(trajectory, rewards, filename='data.json'):

    data = {
        'trajectory': [t.detach().cpu().numpy().tolist() for t in trajectory],
        'rewards': rewards.detach().cpu().numpy().tolist()
    } 

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=1) 
    parser.add_argument("--env_mode", type=str, default= "entire")
    parser.add_argument("--prob_index", type=int, default=TASKNUM) # prob index는 0~399까지 있음 (o2arc문제 번호에서 -1 해야 함)
    parser.add_argument("--num_actions", type=int, default=ACTIONNUM)
    parser.add_argument("--ep_len", type=int, default=10)
    parser.add_argument("--device", type=int, default=CUDANUM)
    parser.add_argument("--use_offpolicy", action="store_true", help="Use off-policy learning", default=False)
    parser.add_argument("--sampling_method", type=str, default="prt", choices=["prt", "fixed_ratio", "egreedy"],
                        help="Sampling method to use in replay buffer")
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    env_mode = args.env_mode
    prob_index = args.prob_index
    num_actions = args.num_actions
    # ep_len = args.ep_len

    seed_everything(42) # seed 바꿔봄 원래 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, env = train(num_epochs, batch_size, device, env_mode, prob_index,num_actions, args, use_offpolicy= args.use_offpolicy)#args.use_offpolicy)
    # eval(model)