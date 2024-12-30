import torch
import pdb
import torch.nn.functional as F

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs, answer = None):
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

    if rewards == 0 :
        pass
    else:
        rewards = torch.log(rewards)

    # device check
    if total_flow.device.type != 'cuda':
        total_flow = torch.tensor(total_flow).to("cuda")
    
    loss = (torch.log(total_flow) + torch.sum(fwd_probs, dim=0) - rewards.clip(-20) - torch.sum(back_probs, dim=0)).pow(2)
    

    return loss.sum(), torch.exp(rewards), total_flow
    