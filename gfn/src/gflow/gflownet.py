import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from .log import Log


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env=None, device='cpu'):
        super().__init__()
        self.total_flow = Parameter(torch.ones(1).to(device))
        self.forward_policy = forward_policy.to(device)
        self.backward_policy = backward_policy.to(device)
        self.env = env
        self.device = device

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

    def forward_probs(self, s):
        """
        Returns a vector of probabilities over actions in a given state.

        Args:
            s: An NxD matrix representing N states
        """
        probs = self.forward_policy(s)

        return probs

    def sample_states(self, s0, return_log=False):
        """
        Samples and returns a collection of final states from the GFlowNet.

        Args:
            s0: An NxD matrix of initial states

            return_log: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.)
        """
        s = s0.clone()
        done = False
        log = Log(s0, self.backward_policy, self.total_flow,
                  self.env) if return_log else None
        while not done:
            probs = self.forward_probs(s)  # 여기서 에러남
            actions = Categorical(probs).sample()
            s, reward, done, _ = self.env.step(s, actions)

            if return_log:
                log = Log(s, probs, actions, done)  # 의심 1 : done이 log에서 바껴야함?

            if done:
                break

        return (s, log) if return_log else s

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

        rewards = self.env.reward(finals)

        return fwd_probs, back_probs, rewards
