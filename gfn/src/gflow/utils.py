import torch
import pdb
import torch.nn.functional as F

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs, answer):
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