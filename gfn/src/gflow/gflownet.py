import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical

import numpy as np
from .log import Log


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env=None, device='cuda'):
        super().__init__()
        self.total_flow = Parameter(torch.ones(1000).to(device))
        self.forward_policy = forward_policy.to(device)
        self.backward_policy = backward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}

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

    def sample_states(self, s0, return_log=True):
        """
        Samples and returns a collection of final states from the GFlowNet.

        Args:
            s0: An NxD matrix of initial states

            return_log: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.)
        """
        s = torch.tensor(s0["input"]).to(self.device)
        log = Log(s, self.backward_policy, self.total_flow,
                  self.env) if return_log else None
        is_done = False

        iter = 0
        while not is_done:
            iter += 1
            probs = self.forward_probs(s)  # 랜덤액션?
            selection = np.zeros((30, 30), dtype=bool)
            selection[:s0["grid_dim"][0], :s0["grid_dim"][1]] = np.ones(
                s0["grid_dim"], dtype=bool)

            self.actions["operation"] = int(Categorical(probs).sample())
            self.actions["selection"] = selection  # selection 어떻게
            result = self.env.step(self.actions)

            # reward 는 spaset reward 이기 때문에 따로 reward 함수를 만들어서 log에 저장하는 함수를 만들어야함
            state, reward, is_done, _, info = result
            s = torch.tensor(state["grid"], dtype = torch.long).to(self.device)

            ime_reward = self.reward(s)

            if return_log:
                log.log(s=state, probs=probs, actions = self.actions, done=is_done, rewards=ime_reward)  # log에 저장

            if iter > 100:  # max_length
                return (s, log) if return_log else s

            if is_done:
                break

        # return (s, log) if return_log else s
        return s, log

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
        terminal = torch.tensor(self.env.unwrapped.answer)
        pad_terminal = torch.zeros_like(s)
        pad_terminal[:terminal.shape[0], :terminal.shape[1]] = terminal

        # MSE 
        r = ((pad_terminal - s)**2 + 1e-6)
        for i in range(len(r)):
            for j in range(len(r[0])):
                if r[i][j] == float("inf"):
                    r[i][j] = 0
        mse_reward = 1 / (r.sum() + 1e-6) 

        # sparse reward
        sparse_reward = torch.where((pad_terminal - s).sum(dim=1) == 0, 1, 0.1)
        return mse_reward
