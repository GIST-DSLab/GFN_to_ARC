import torch


def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs):
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
    half_length = len(fwd_probs) // 2
    
    fwd_probs = torch.cat(fwd_probs[:half_length+1], dim=0) ### TODO 길이가 안맞아서 일단 51개씩 맞춰놨는데 (100개기준) 나중에 변경하기 
    rewards = torch.tensor(rewards).to("cuda")[half_length:]
    total_flow = torch.tensor(total_flow).to("cuda")

    # lhs = total_flow * torch.prod(fwd_probs, dim=1)
    # rhs = sum(rewards) * torch.prod(back_probs, dim=1)
    loss = torch.log(total_flow) + torch.sum(torch.log(fwd_probs), dim=1) - torch.log(rewards) - torch.sum(torch.log(back_probs), dim=1)
    # loss = torch.log(lhs / rhs+1e-8)**2
    return loss.mean()