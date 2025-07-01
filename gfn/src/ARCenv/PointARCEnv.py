from typing import List, Any, Callable,  SupportsFloat, SupportsInt, Tuple

from collections import OrderedDict

import arcle
from arcle.envs import O2ARCv2Env

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
# from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.envs.registration import register

import torch
import numpy as np
from arcle.loaders import ARCLoader, Loader, MiniARCLoader


class ColorARCEnv(O2ARCv2Env):
    """
    One-pixel coloring Env for ARC
    """
    def create_operations(self) -> List[Callable[..., Any]]:
        ops= super().create_operations()
        return ops[0:10]

register(
    id='ARCLE/ColorARCEnv',
    entry_point=__name__+':ColorARCEnv',
    max_episode_steps=25,
)


class CustomO2ARCEnv(O2ARCv2Env):
    
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None, options : dict = {}) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)

        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': None, #options["prob_index"],
            # 'subprob_index' : 0
        }

        self.batch_size = None
        self.curerent_states = None
        self.last_actions_ops = None
        self.last_actions = None
        self.action_steps = None
        self.submit_counts = None
        self.answer = None
    
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self.current_states = [None] * batch_size
        self.last_action_ops = [None] * batch_size
        self.last_actions = [None] * batch_size
        self.action_steps = 0 
        self.submit_counts = 0
        print(f"---------------Complete to set batch size {batch_size}-----------------")


    def create_operations(self) :
        from arcle.actions.critical import crop_grid
        from arcle.actions.object import reset_sel
        ops = super().create_operations()
        ops[33] = reset_sel(crop_grid)
 
        return [ops[0], ops[2], ops[34]] # ops[0:10] #+ ops[20:]
    '''
        0~9 : color
        1~19 : flood fill
        20~23 : moving (U D R L)
        24~25 : rotate (right, left)
        26~27 : flip (horizontal, vertical)
        28~29 : copy (input, output)
        30 : paste
        31 : copy from input
        32 : reset grid
        33 : resize grid
        34 : submit

        0 : black
        1 : blue
        2 : red
        3 : green
        4 : yellow
        5 : grey
        6 : purple
        7 : orange
        8 : skyblq
        9 : brown
    ''' 
    
    
    def get_current_problem_index(self):
        return self.current_prob_index

  

    def reset(self, seed = None, options= None):
        obs, info = super().reset(seed, options)
        self.reset_options = options if options is not None else self.reset_options
        
        # rotate_k = np.random.randint(0,4)
        # permute = np.random.permutation(10)
        # f = lambda x: permute[int(x)]
        # ffv = np.vectorize(f)
        # # augment
        
        # self.input_ = np.copy(np.rot90(ffv(self.input_),k=rotate_k).astype(np.int8))
        # self.answer = np.copy(np.rot90(ffv(self.answer),k=rotate_k).astype(np.int8))
        # self.input = np.copy(obs)
        
        self.init_state(self.input_.copy(),options)

        # if self.batch_size is not None:
        # copy obs and info to match batch size
        #     obs = [obs.copy() for _ in range(self.batch_size)]
        #     info = [info.copy() for _ in range(self.batch_size)]
        #     self.current_states = [self.current_state.copy() for _ in range(self.batch_size)]

        return obs, info
        
    # def reset(self, seed=None, options=None):
    #     if isinstance(options, (list, tuple)):
    #         options = dict(zip(['prob_index', 'adaptation', 'subprob_index'], options))
    #     if self.batch_size is not None:
    #         obs_list = []
    #         info_list = []
    #         for _ in range(self.batch_size):
    #             obs, info = super().reset(seed, options)
    #             obs_list.append(obs)
    #             info_list.append(info)
    #         self.current_states = obs_list
    #         return obs_list, info_list
    #     else:
    #         obs, info = super().reset(seed, options)
    #         self.current_states = [obs]
    #         return [obs], [info]
    
    def step(self, action):
        # print("In step, initial actions", action)
        if not isinstance(action, dict) or 'selection' not in action or 'operation' not in action:
            raise ValueError("Action must be a dictionary with 'selection' and 'operation' keys")
        
        selection = action['selection']
        operation = action['operation']
        
        # if selection.shape != (30, 30) or not isinstance(operation, int):
        #     raise ValueError("Selection must be a (30, 30) numpy array and operation must be an integer")

        self.transition(self.current_state, action)
        self.last_action_ops = operation
        self.last_actions = action

        state = self.current_state
        reward = self.reward(state)
        done = bool(state["terminated"][0])
        print("Is Done? :", done)

        self.action_steps += 1
        if operation == len(self.operations) - 1:
            # for i in range(len(self.submit_counts)):
            self.submit_counts += 1
        print("submit counts : ", self.submit_counts)

        result = (state, reward, done, self.truncated, {
            'steps': self.action_steps,
            'submit_count': self.submit_counts
        })

        print("Step result generated !", result)

        if self.render_mode:
            self.render()

        return result
    
    def reward(self, state) -> SupportsFloat:
        sparse_reward = super().reward(state)
        
        h = state["grid_dim"][0]
        w = state["grid_dim"][1]
        H, W = self.answer.shape
        minh, minw = min(h,H), min(w,W)
        total_size = minh*minw
        correct = np.sum(state["grid"][:minh,:minw]==self.answer[:minh,:minw])
        if (h <= H) == (w <= W):
            total_size += abs(H*W - h*w)
        else:
            total_size += abs(h-H)*minw + abs(w-W)*minh

        print("reward calculated.")
    
        return sparse_reward*100 + correct / total_size

        
    #TaskSettableEnv API
    def sample_tasks(self, n_tasks: int):
        return np.random.choice(len(self.loader.data),n_tasks,replace=False)

    def get_task(self):
        return super().get_task()
    
    def set_task(self, task) -> None:
        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': task
        }
        super().reset(options=self.reset_options)

    def init_adaptation(self): 
        self.adaptation = True
        self.reset_options['adaptation'] = True
        super().reset(options=self.reset_options)
        
    def post_adaptation(self):
        self.adaptation = False
        self.reset_options['adaptation'] = False
        self.reset_options['prob_index'] = self.options['prob_index']
        self.reset_options['subprob_index'] = self.options['subprob_index']
        super().reset(options=self.reset_options)

    def get_answer(self):
        return self.answer
    
    def get_current_problem_index(self):
        return self.current_prob_index

class FilterO2ARC(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "trials_remain": spaces.Box(-1, self.max_trial, shape=(1,), dtype=np.int8),

            "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "grid_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "clip_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),

            "active": spaces.MultiBinary(1),
            "object": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "object_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),
            "object_pos": spaces.Box(low=np.array([-128,-128]), high=np.array([127,127]), dtype=np.int8), 

            }
        )

    def observation(self, observation) :
        obs = observation
        o2s = obs["object_states"]
        return OrderedDict([
            
            ("trials_remain",obs["trials_remain"]),
            
            ("grid",obs["grid"]),
            ("grid_dim",obs["grid_dim"]),

            ("clip",obs["clip"]),
            ("clip_dim",obs["clip_dim"]),

            ("active",o2s["active"]),
            ("object",o2s["object"]),
            ("object_dim",o2s["object_dim"]),
            ("object_pos",o2s["object_pos"]), 
        ])
