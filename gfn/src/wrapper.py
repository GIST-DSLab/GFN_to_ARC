import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from gymnasium import spaces


class PointWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(len(self.operations)),
            )
        )
    
    def action(self, action:dict):
        # 5-tuple: (x, y, op)
        op = action["operation"]
        x, y = action["selection"]
        
        selection = np.zeros((5, 5), dtype=np.uint8)
        # selection = np.zeros(self.env.unwrapped.max_grid_size, dtype=np.uint8)
        selection[x, y] = 1
        return {'selection': selection, 'operation': op}