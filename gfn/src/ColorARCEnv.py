from typing import List, Any, Callable

import arcle
from arcle.envs import O2ARCv2Env

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import numpy as np
import wrapper
from wrapper import PointWrapper
from arcle.loaders import ARCLoader, Loader, MiniARCLoader


class ColorARCEnv(O2ARCv2Env):
    """
    One-pixel coloring Env for ARC
    """
    def create_operations(self) -> List[Callable[..., Any]]:
        ops= super().create_operations()
        return ops[0:10] # + ops[20:] 

register(
    id='ARCLE/ColorARCEnv',
    entry_point=__name__+':ColorARCEnv',
    max_episode_steps=25,
)

# env = gym.make('ARCLE/ColorARCEnv', )

def env_return(render, data):
    env = gym.make('ARCLE/ColorARCEnv', render_mode=render, data_loader=data, max_grid_size=(5, 5), colors=10, max_episode_steps=None )
    wrapped_env = PointWrapper(env)
    return wrapped_env


# env_return(env)