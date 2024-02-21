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
    rewards = torch.tensor(rewards[-1]).to("cuda")
    total_flow = torch.tensor(total_flow).to("cuda")
    ### TODO 길이가 안맞아서 일단 51개씩 맞춰놨는데 (100개기준) 나중에 변경하기 
    ### TODO 35개의 길이를 가진 분포 형태로 나옴 (35개의 action) -> 이게 맞는지 확인 필요

    # ce_reward = torch.log(F.cross_entropy(fwd_probs, torch.flatten(answer,0), reduction="sum"))
    # rewards = (1 / (ce_reward + 1e-5)) + rewards
    # reward = rewards

    # lhs = total_flow * torch.prod(fwd_probs, dim=1)
    # rhs = sum(rewards) * torch.prod(back_probs, dim=1)
    loss = torch.square(torch.log(total_flow) + torch.sum(torch.log(fwd_probs), dim=1) - torch.sum(torch.log(rewards)) - torch.sum(torch.log(back_probs), dim=1))
    

    return loss.sum(), rewards
    