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
    
        # half_length = len(fwd_probs) // 2

        # back_probs = torch.cat(fwd_probs[half_length:], dim=0)
    back_probs = torch.cat(back_probs, dim=0)
    fwd_probs = torch.cat(fwd_probs, dim=0)
    rewards = torch.tensor([rewards[-1]], device=total_flow.device)

    # if rewards == 0 :
    #     pass
    # else:
    #     rewards = torch.log(rewards*10)
    # reward에 지수 함수 씌었다고 가정하고 log 붙이면 원래 값이 나오므로 일단 주석처리 

    ### TODO 길이가 안맞아서 일단 51개씩 맞춰놨는데 (100개기준) 나중에 변경하기 
    ### TODO 35개의 길이를 가진 분포 형태로 나옴 (35개의 action) -> 이게 맞는지 확인 필요

    loss = (torch.log(total_flow) + torch.sum(fwd_probs) - rewards.clip(0) - torch.sum(back_probs)).pow(2)
    # reward shape 확인하고 clip을 해야할지 안해야할지 확인

    return loss.sum(), total_flow, rewards
    