import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """Replay Buffer for storing and sampling experience tuples."""
    def __init__(self, capacity: int, device=torch.device("cpu")):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.device = device
        self.capacity = capacity

    def add(self, state, action, reward, log_prob, back_prob, traj):
        """Add a new experience tuple to the buffer."""
        last_reward = reward[-1]
        self.buffer.append((state, action, reward, log_prob, back_prob, traj))
        self.priorities.append(last_reward)

    def batch_sample(self, batch_size: int):
        """Sample a batch of experiences."""
        indices = random.sample(range(len(self.buffer)), k=batch_size)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, log_probs, back_probs, trajs = zip(*batch)
        states = [state.to(self.device) for state in states]
        rewards = [torch.tensor(r, device=self.device) for r in rewards]
        log_probs = [torch.tensor(lp, device=self.device) for lp in log_probs]
        back_probs = [torch.tensor(bp, device=self.device) for bp in back_probs]
        trajs = [[t.to(self.device) if isinstance(t, torch.Tensor) else t for t in traj] for traj in trajs]

        return states, actions, rewards, log_probs, back_probs, trajs

    def priority_sample(self, batch_size: int, alpha: float = 0.6):
        """Priority-based sampling."""
        priorities = np.array(self.priorities, dtype=np.float32) ** alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, log_probs, back_probs, trajs = zip(*batch)
        states = [state.to(self.device) for state in states]
        rewards = [torch.tensor(r, device=self.device) for r in rewards]
        log_probs = [torch.tensor(lp, device=self.device) for lp in log_probs]
        back_probs = [torch.tensor(bp, device=self.device) for bp in back_probs]
        trajs = [[t.to(self.device) if isinstance(t, torch.Tensor) else t for t in traj] for traj in trajs]

        return states, actions, rewards, log_probs, back_probs, trajs

    def is_ready_for_sampling(self, min_samples: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= min_samples

    def __len__(self):
        return len(self.buffer)
