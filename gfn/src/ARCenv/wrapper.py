import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from gymnasium import spaces
from typing import Any, Callable, List, SupportsInt, Tuple
from arcle.envs import AbstractARCEnv
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader

from .PointARCEnv import *
from .ColorARCEnv import *
from .EntireARCEnv import *
from gymnasium.vector import AsyncVectorEnv
from gymnasium.envs.registration import register


class PointWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(len(self.unwrapped.operations)),
            )
        )

        self.batch_size = None
        self.env_mode = 'Point'

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        if hasattr(self.env, 'set_batch_size'):
            self.env.set_batch_size(batch_size)
    

    def action(self, action):
        x, y, op = action
        selection = np.zeros((30, 30), dtype=np.uint8)
        selection[x, y] = 1
        return {'selection': selection, 'operation': op}

    def step(self, actions):

        return self.env.step(self.action(actions))

    # def reset(self, **kwargs):
    #     options = kwargs.get('options')
    #     if isinstance(options, (list, tuple)):
    #         kwargs['options'] = dict(zip(['prob_index', 'adaptation', 'subprob_index'], options))
        
    #     if self.batch_size is None or self.batch_size == 1:
    #         return self.env.reset(**kwargs)
    #     else:
    #         results = [self.env.reset(**kwargs) for _ in range(self.batch_size)]
    #         return list(zip(*results))

class Entirewrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # self.action_space = spaces.Tuple(
        #     (
        #         spaces.Discrete(self.H),
        #         spaces.Discrete(self.W),
        #         spaces.Discrete(len(self.operations)),
        #     )
        # )
        self.env_mode = 'entire'
        
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        if hasattr(self.env, 'set_batch_size'):
            self.env.set_batch_size(batch_size)
    
    def action(self, action):
        # op = action
        # selection = np.zeros((30, 30), dtype=np.uint8)
        return action #{'selection': selection, 'operation': op}

    def step(self, action):
        return self.env.step(self.action(action))
    
    def reset(self, **kwargs):
        return super().reset(**kwargs)


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
        self.env_mode = 'bbox'
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
    


def make_env(render=None, data=None, options=None, batch_size=8, mode='Point'):
    def _init():
        if mode == 'point':
            env = gym.make('ARCLE/O2ARCv2Env', render_mode=render, data_loader=data, max_grid_size=(30, 30), colors=10)
            env = CustomO2ARCEnv(render_mode=render, data_loader=data, max_trial=100, options=options)
            env = PointWrapper(env)
        elif mode == 'bbox':
            env = gym.make('ARCLE/O2ARCv2Env', render_mode=render, data_loader=data, max_grid_size=(30, 30), colors=10)
            env = CustomO2ARCEnv(render_mode=render, data_loader=data, max_trial=100, options=options)
            env = BBoxWrapper(env)
        elif mode == 'entire':
            env = gym.make('ARCLE/O2ARCv2Env', render_mode=render, data_loader=data, max_grid_size=(30, 30), colors=10)
            env = DiagonalARCEnv(data_loader=data, max_grid_size=(30, 30), colors=10, render_mode=render)
            env = Entirewrapper(env)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        env.set_batch_size(batch_size)
        return env
    return _init

def env_return(render, data, options, batch_size=1, mode='Point', batch=False):
    # 환경 등록
    register(
        id='O2ARCv2Env',
        entry_point='arcle.envs:O2ARCv2Env',  # path of Env class as string
        max_episode_steps=300, 
    )
    
    if batch_size > 1 :
        env_fns = [make_env(render, data, options, batch_size, mode) for _ in range(batch_size)]
        envs = AsyncVectorEnv(env_fns)
    else : 
        env = make_env(render, data, options, batch_size, mode)()
        envs = env

    return envs