import torch
import pdb
import torch.nn.functional as F
import numpy as np
import random
import wandb

def setup_wandb(project_name: str, entity: str, config: dict, run_name: str = None):
    """
    wandb 초기화 설정.
    Args:
        project_name (str): 프로젝트 이름.
        entity (str): 팀 또는 사용자 이름.
        config (dict): wandb에 저장할 설정값.
        run_name (str): 실행 이름 (optional).
    """
    wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=run_name
    )

def normalize_probabilities(x):
        return x / x.sum()

# def process_rewards(rewards, device):
#     b = rewards[0].shape[0]
#     if isinstance(rewards, list):
#         # 배치 학습의 경우: 각 에피소드의 마지막 리워드를 가져옴
#         last_rewards = rewards[-1]
#     else:
#         # rewards가 이미 단일 값인 경우
#         last_rewards = [rewards]

#     # 텐서로 변환
#     return torch.tensor(last_rewards, device=device)

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs, batch_size=1):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259

    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed

        rewards: The rewards associated with the final state of each of the
        samples

        fwd_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actually taken in
        each trajectory)

        back_probs: The backward probabilities associated with each trajectory
    """
    
    # back_probs = normalize_probabilities(torch.cat(back_probs, dim=0).reshape(-1, b).transpose(0, 1))
    # fwd_probs = normalize_probabilities(torch.cat(fwd_probs, dim=0).reshape(-1, b).transpose(0, 1))
    total_flow = total_flow.clamp(min=0)
    if isinstance(rewards, int or float):
        rewards = torch.tensor([rewards], device=total_flow.device, dtype=torch.float)

    if batch_size > 1:
        # fwd_probs = torch.stack([torch.sum(torch.stack(probs)) for probs in fwd_probs])
        # back_probs = torch.stack([torch.sum(torch.stack(probs)) for probs in back_probs])
        fwd_probs = torch.stack([torch.sum(torch.stack(probs)) for probs in fwd_probs])
        back_probs = torch.stack([torch.sum(torch.stack(probs)) for probs in back_probs])
        rewards = torch.tensor([r[-1] for r in rewards], device=total_flow.device)
    else:
        fwd_probs = torch.sum(torch.stack(fwd_probs))
        back_probs = torch.sum(torch.stack(back_probs))

        if isinstance(rewards, list):
            rewards = torch.tensor(rewards[-1], device=total_flow.device)
        # else:
        #     rewards = torch.tensor(rewards, device=total_flow.device)

    log_rewards = torch.log(rewards).clip(-10)
    loss = torch.square(total_flow + fwd_probs - log_rewards  - back_probs).clamp(max=1e+6)

    # if torch.isnan(loss):
    #     loss = loss.sum()
#     loss = loss.clamp(max=1e+6)

    return loss, total_flow, rewards


def detailed_balance_loss(total_flow, rewards, fwd_probs, back_probs, answer):
    """
    FM Loss와 마찬가지로, 전후방 확률을 매칭하는 것에 중심을 두는 loss입니다.
    따라서 reward가 사용되지 않는다.
    """
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)
    rewards = torch.tensor([rewards[-1]], device=total_flow.device)

    # total_flow 값을 이용하여 forward_term과 backward_term 계산
    forward_term = torch.sum(fwd_probs) * torch.exp(fwd_probs)  # F(s) * exp(log P_F(s'|s)) = F(s) * P_F(s'|s)
    backward_term = torch.sum(back_probs) * torch.exp(back_probs)  # F(s') * exp(log P_B(s|s')) = F(s') * P_B(s|s')

    loss = torch.square(torch.log(forward_term) - torch.log(backward_term))
    loss = loss.mean()  # 모든 상태 쌍에 대한 평균을 계산

    return loss, total_flow, rewards


def subtrajectory_balance_loss(trajectories, fwd_probs, back_probs):
    """
    Calculate the Subtrajectory Balance Loss for given trajectories.
    
    Parameters:
    - trajectories: List of tuples (start_index, end_index, trajectory)
    - flow: Function that returns the flow for a given state
    - PF: Function that returns the forward transition probability log P_F(s'|s)
    - PB: Function that returns the backward transition probability log P_B(s|s')
    
    Returns:
    - Average subtrajectory balance loss

    """
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)
    
    losses = []
    for trajectory in trajectories:
        # Initialize products of probabilities
        log_pf_product = 0
        log_pb_product = 0
        
        # Extract the log probabilities for the trajectory
        for i in range(len(trajectory) - 1):
            log_pf_product += fwd_probs[i]
            log_pb_product += back_probs[i]
        
        # Estimate flow for the start and end of the trajectory
        flow_start = torch.exp(torch.sum(fwd_probs[:len(trajectory)//2]))  # Cumulative product approximation
        flow_end = torch.exp(torch.sum(back_probs[len(trajectory)//2:]))   # Cumulative product approximation

        # Calculate the log of the ratio of products of probabilities and flows
        log_ratio = (torch.log(flow_start) + log_pf_product) - (torch.log(flow_end) + log_pb_product)
        
        # Square the log ratio and add to losses
        losses.append(log_ratio ** 2)

    # Return the mean of the losses
    return torch.mean(torch.stack(losses))


def guided_TB_loss(total_flow, rewards, fwd_probs, back_probs, answer):
    
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)
    rewards = torch.tensor([rewards[-1]], device=total_flow.device)

    # if rewards == 0 :
    #     pass
    # else:
    #     rewards = torch.log(rewards*10)
    # reward에 지수 함수 씌었다고 가정하고 log 붙이면 원래 값이 나오므로 일단 주석처리  
    loss = torch.square(torch.log(total_flow) + torch.sum(fwd_probs) - torch.log(rewards).clip(0) - torch.sum(back_probs))

    # 만약 loss가 nan이면 100으로 대체
    loss = loss.sum()
    loss = loss.clamp(max=1e+6)

    return loss, total_flow, rewards

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