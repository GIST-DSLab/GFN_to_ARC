from arcle.envs import O2ARCv2Env, AbstractARCEnv
from arcle.loaders import ARCLoader, Loader, MiniARCLoader
from gymnasium.core import ObsType, ActType
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from io import BytesIO
import gymnasium as gym
import copy
import pdb
from PIL import Image
import random
import torch
from numpy.typing import NDArray
from typing import Dict, Optional, Union, Callable, List, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any

from functools import wraps
from numpy import ma
import json



def create_img(state): # ARC -> PIL Image
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]     

    """
    [Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Light blue, Brown]
    """
    
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    output_folder = './'
    fig, axs = plt.subplots(1, 1, figsize=(3 * 0.7, 3 * 0.7))
    rows, cols = np.array(state).shape
    axs.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    axs.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    axs.tick_params(which='minor', size=0)
    axs.grid(True, which='minor', color='#555555', linewidth=1)
    axs.set_xticks([]); axs.set_yticks([])
    axs.imshow(np.array(state), cmap=cmap, vmin=0, vmax=9)

    plt.tight_layout()
    tmpfile = BytesIO()
    plt.savefig(tmpfile, bbox_inches='tight', format='png', dpi=300)
    plt.close()

    return tmpfile

class DiagonalARCEnv(O2ARCv2Env):
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None, options : dict = {}):
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)

        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': None, #options["prob_index"],
            # 'subprob_index' : 0
        }

        self._observation_space = None
        self._action_space = None
        self.answer = None
        self.env_mode = 'entire'       
        self.batch_size = None
        self.curerent_states = None
        self.last_actions_ops = None
        self.last_actions = None
        self.action_steps = None
        self.submit_counts = None
    
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self.current_states = [None] * batch_size
        self.last_action_ops = [None] * batch_size
        self.last_actions = [None] * batch_size
        self.action_steps = 0 
        self.submit_counts = 0
        self.num_envs = batch_size
        print(f"---------------Complete to set batch size {batch_size}-----------------")


    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_state(initial_grid, options)
        
        self.current_state.update({

            "selected": np.zeros((self.H, self.W), dtype=np.int8),
            "clip": np.zeros((self.H, self.W), dtype=np.int8),
            "clip_dim": np.zeros((2,), dtype=np.int8),
            "submit_terminated": np.array([0], dtype=np.int8),
            "is_correct": np.array([0], dtype=np.int8),
            "object_states": {
                "active": np.zeros((1,), dtype=np.int8),
                "object": np.zeros((self.H, self.W), dtype=np.int8),
                "object_sel": np.zeros((self.H, self.W), dtype=np.int8),
                "object_dim": np.zeros((2,), dtype=np.int8),
                "object_pos": np.zeros((2,), dtype=np.int8),
                "background": np.zeros((self.H, self.W), dtype=np.int8),
                "rotation_parity": np.zeros((1,), dtype=np.int8),
            }
        })

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    
    def create_operations(self) :
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

        '''
        from arcle.actions.object import (
            reset_sel, keep_sel,
            gen_move, gen_rotate, gen_flip,
            gen_copy, gen_paste
        )
        from arcle.actions.color import (
            gen_color, gen_flood_fill
        )
        from arcle.actions.critical import (
            copy_from_input,reset_grid,resize_grid,crop_grid
        )
        # ops = [None] * 35

        # # color ops (20)
        # ops[0:10] = [reset_sel(gen_color(i)) for i in range(10)]
        # ops[10:20] = [reset_sel(gen_flood_fill(i)) for i in range(10)]

        # # obj ops (8)
        # ops[20:24] = [gen_move(i) for i in range(4)] # [U,D,R,L]
        # ops[24] = gen_rotate(1) # left rotate
        # ops[25] = gen_rotate(3) # right rotate
        # ops[26] = gen_flip("H")
        # ops[27] = gen_flip("V")
        
        # # clipboard ops (3)
        # ops[28] = reset_sel(gen_copy("I"))
        # ops[29] = reset_sel(gen_copy("O"))  
        # ops[30] = reset_sel(gen_paste(paste_blank=True))

        # # critical ops (3)
        # ops[31] = reset_sel(copy_from_input)
        # ops[32] = reset_sel(reset_grid)
        # ops[33] = reset_sel(resize_grid)

        # # submit op (1)

        ops = super().create_operations()
        ops[33] = reset_sel(crop_grid)
        ops[34] = self.submit

        return ops[24:28] + [ops[34]]
        # return ops[24:28] + [ops[34]] + [ops[4]] + [ops[6]] + [ops[8]] + [ops[9]] + [ops[21]] + ops[29:31]
        # return [ops[24]] + [ops[26]] + [ops[34]]
        # return ops[26:31] + [ops[34]]
    
    def render_ansi(self):
        if self.rendering is None:
            self.rendering = True
            print('\033[2J',end='')

        print(f'\033[{self.H+3}A\033[K', end='')


        state = self.current_state
        grid = state['grid']
        grid_dim = state['grid_dim']

        for i,dd in enumerate(grid):
            for j,d in enumerate(dd):
                
                if i >= grid_dim[0] or j>= grid_dim[1]:
                    print('\033[47m  ', end='')
                else:
                    print("\033[48;5;"+str(self.ansi256arc[d])+"m  ", end='')

            print('\033[0m')

        print('Dimension : '+ str(grid_dim), end=' ')
        print('Action : ' + str(self.op_names[self.last_action_op] if self.last_action_op is not None else '') , end=' ')

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        state = self.current_state
        # grid = state['grid']
        # image = Image.open(create_img(grid)).convert('RGB')
        # image = image.resize(self._size)
        # image = np.array(image)

        grid = torch.tensor(state['grid'])
        bottom_pad_size = 30 - grid.shape[0]
        right_pad_size = 30 - grid.shape[1] 
        image = torch.nn.functional.pad(grid, (0, right_pad_size, 0, bottom_pad_size), 'constant', 0)#.unsqueeze(-1)

        return (
            {"grid": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        )
    def get_answer(self):
        return self.unwrapped.answer
    
    def get_current_problem_index(self):
        return self.current_prob_index
    

    def step(self, action):

        if self.env_mode == 'entire':
            pass
        else:
            if not isinstance(action, dict) or 'selection' not in action or 'operation' not in action:
                raise ValueError("Action must be a dictionary with 'selection' and 'operation' keys")

        operation = action['operation']
        selection = action['selection']
        
        # if selection.shape != (30, 30) or not isinstance(operation, int):
        #     raise ValueError("Selection must be a (30, 30) numpy array and operation must be an integer")

        self.transition(self.current_state, action)
        self.last_action_ops = operation
        self.last_actions = action

        state = self.current_state
        reward = self.reward(state)
        done = bool(state["terminated"][0])

        self.action_steps += 1
        if operation == len(self.operations) - 1:
            # for i in range(len(self.submit_counts)):
            self.submit_counts += 1
        # print("submit counts : ", self.submit_counts)

        if done:
            state["submit_terminated"] = 1
        else :
            state["submit_terminated"] = 0

        result = (state, reward, done, self.truncated, {
            'steps': self.action_steps,
            'submit_count': self.submit_counts
        })


        # print("Step result generated !", result)

        if self.render_mode:
            self.render()

        return result


    def transition(self, state, action):
        op = int(action['operation'])
        self.operations[op](state, action)

    def submit(self, state, action) -> None:
        self.submit_count += 1
        h, w = state["grid_dim"][0], state["grid_dim"][1]
        
        if state["trials_remain"][0] != -1:
            state["trials_remain"][0] -= 1

        is_correct = self.answer.shape == (h,w) and np.all(self.answer == state["grid"][:h,:w])
        
        if is_correct:
            state["terminated"][0] = 1  # Always terminate after submit

        state["terminated"][0] = 1 # Always terminate after submit

        state["submit_terminated"] = 1  # Flag to indicate termination by submit action
        state["is_correct"] = int(is_correct) 

    def reward(self, state) -> SupportsFloat:        
        # Check if the answer is correct
        # if tuple(state['grid_dim']) == self.answer.shape:
        h, w = self.answer.shape
        if state["submit_terminated"] == 1 : 
            if np.all(state['grid'][0:h, 0:w] == self.answer):
                # If terminated by submit action, return 2, else return 1
                # return 10 if int(self.last_action_ops) == len(self.operations) - 1 else 1
                return 10
            else:
                return 0
        else : 
            if np.all(state['grid'][0:h, 0:w] == self.answer):
                return 5 
            else : 
                return 0
