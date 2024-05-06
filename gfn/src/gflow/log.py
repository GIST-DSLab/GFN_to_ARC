import torch
import numpy as np
from torch.distributions import Categorical, Uniform

class Log:
    def __init__(self, s0, backward_policy, total_flow, env, emb_s=None):
        """
        Initializes a Stats object to record sampling statistics from a
        GFlowNet (e.g. trajectories, forward and backward probabilities,
        actions, etc.)

        Args:
            s0: The initial state of collection of samples

            backward_policy: The backward policy used to estimate the backward
            probabilities associated with each sample's trajectory

            total_flow: The estimated total flow used by the GFlowNet during
            sampling

            env: The environment (i.e. state space and reward function) from
            which samples are drawn
        """
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self.env = env
        self._traj = []  # state value
        self._fwd_probs = []
        self._back_probs = []
        self._actions = []
        self._state_colors = []
        self.rewards = []
        self.num_samples = len(self._traj)
        self.is_terminals = []
        self.masks = []
        self._emb_traj = []
        
        self._traj.append(s0)

        if emb_s is not None:
            self._emb_traj.append(emb_s)

    def log(self, s, probs, actions, rewards=None, embedding=None, done=None):
        """
        Logs relevant information about each sampling step

        Args:
            s: An NxD matrix containing he current state of complete and
            incomplete samples

            probs: An NxA matrix containing the forward probabilities output by the
            GFlowNet for the given states

            actions: A Nx1 vector containing the actions taken by the GFlowNet
            in the given states

            done: An Nx1 Boolean vector indicating which samples are complete
            (True) and which are incomplete (False)
        """
        self._emb_traj.append(embedding.detach())
        self._traj.append(s.detach())
        self._fwd_probs.append(probs.unsqueeze(0))
        self._actions.append(actions)
        

        if rewards is not None:
            if isinstance(rewards, np.float64) or isinstance(rewards, np.float32):
                self.rewards.append(rewards)
            else:
                self.rewards.append(rewards.detach())
        if done is not None:
            self.is_terminals.append(done)
        # Note: Assuming total_flow and other properties are handled correctly elsewhere
        self._back_probs_computed = False  # Invalidate cached back_probs

    @property
    def emb_traj(self):
        if type(self._emb_traj) is list:
            pass

        return self._emb_traj
    @property
    def traj(self):
        if type(self._traj) is list:
            pass

        return self._traj

    @property
    def fwd_probs(self):
        if type(self._fwd_probs) is list:
            pass

        return self._fwd_probs

    @property
    def actions(self):
        if type(self._actions) is list:
            # self._actions = torch.cat(self._actions, dim=1)
            pass
        return self._actions

    @property
    def back_probs(self):

        if not self._back_probs_computed:
            self._compute_back_probs()

        return self._back_probs
    
    def _compute_back_probs(self):

        """
        Computes the backward probabilities for the logged trajectories and actions.
        This function is called lazily to ensure that it's computed only once.
        """
        self._back_probs = []
        for t, (traj, action) in enumerate(zip(reversed(self._traj), reversed(self._actions))):
            # action은 1차원 벡터 (0~9)
            # traj는 3차원 텐서 (E_len, 30, 30) 즉 state
            # pb_s는 (1,10)

            pb_s = self.backward_policy(traj.to("cuda")).unsqueeze(0)
            pb = Categorical(logits=pb_s).log_prob(action)
            # pb = Uniform(0, 9).log_prob(action)

            if pb.dim() == 0:
                pb = pb.unsqueeze(0)
            pb = pb.clamp(min=-5, max=5)

            self._back_probs.append(pb.detach())
        self._back_probs_computed = True
