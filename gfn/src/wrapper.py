import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from gymnasium import spaces
from typing import Any, Callable, List, SupportsInt, Tuple
from arcle.envs import AbstractARCEnv
from arcle.envs import O2ARCv2Env

from arcle.loaders import ARCLoader, Loader


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


class BBoxWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(len(self.operations)),
            )
        )
        self.max_grid_size = (30,30) # mini-arc
    def action(self,action: tuple):
        # 5-tuple: (x1, y1, x2, y2, op)
        x1, y1, x2, y2, op = action

        x1 = int(x1.cpu().detach().numpy())
        y1 = int(y1.cpu().detach().numpy())
        x2 = int(x2.cpu().detach().numpy())
        y2 = int(y2.cpu().detach().numpy())
        
        selection = np.zeros(self.max_grid_size, dtype=bool)
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        
        if (x1==x2) & (y1 == y2):
            selection[x1,y1] = 1
        
        elif (x1 == x2) & (y1 != y2) :
            selection[x1, y1:y2+1] = 1
        elif (y1 == y2) & (x1 != x2) :
            selection[x1:x2+1, y1] = 1
        
        else:
            selection[x1:x2+1, y1:y2+1] = 1
        return {'selection': selection, 'operation': op}
    

class O2ARCNoFillEnv(O2ARCv2Env):

    def create_operations(self) -> List[Callable[..., Any]]:
        ops= super().create_operations()
        return ops[0:10] + ops[20:] 

from gymnasium.envs.registration import register

register(
    id='ARCLE/O2ARCNoFillEnv',
    entry_point=__name__+':O2ARCNoFillEnv',
    max_episode_steps=300,
)
env = gym.make('ARCLE/O2ARCNoFillEnv')
env = PointWrapper(env)
obs, info = env.reset()
print(env.action_space)