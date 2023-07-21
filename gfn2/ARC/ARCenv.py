import numpy as np
from enum import Enum

from typing import ClassVar, Literal, Tuple, Union, cast

import torch
from einops import rearrange
from gymnasium.spaces import Discrete
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.env import Env
from gfn.envs.preprocessors import (
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
)

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]

preprocessors_dict = {
    "KHot": KHotPreprocessor,
    "OneHot": OneHotPreprocessor,
    "Identity": IdentityPreprocessor,
}


NUM_COLORS = 10
DEFAULT_COLOR = 0
"""
유효한 properties를 넣어야함 
"""
class ActionType(Enum):
    TRANSLATE = 0
    SELECT_FILL = 1

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class MiniArcEnv:
    def __init__(self, initial, goal, use_reward_shaping=True, 
                 device_str: Literal["cpu", "cuda"] = "cpu",
                 preprocessor_name: Literal["KHot", "OneHot", "Identity"] = "KHot",):
        self.initial = initial
        self.ndim, self.height = self.initial.shape
        self.goal = goal
        #self.initial_obj_map = initial_obj_map
        self.state = None
        self.obj_map = None
        self.use_reward_shaping = use_reward_shaping
        self.step_func = {
            ActionType.TRANSLATE: self.translate,
            ActionType.SELECT_FILL: self.select_fill
        }
        self.translate_x = {
            Direction.UP: 0, Direction.DOWN: 0, Direction.LEFT: -1, Direction.RIGHT: 1
        }
        self.translate_y = {
            Direction.UP: -1, Direction.DOWN: 1, Direction.LEFT: 0, Direction.RIGHT: 0
        }
        self.reset()

        self.n_actions = 2 * (len(Direction) + NUM_COLORS)
        self.num_pixels = self.ndim * self.height

        self.action_space = Discrete(self.n_actions) # ***일단 이렇게 해놓고 수정 
        
        self.s0 = torch.tensor(self.initial).float()
        self.sf = torch.tensor(self.goal).float

        if preprocessor_name == "Identity":
            preprocessor = IdentityPreprocessor(output_shape=(self.ndim,))
        elif preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(
                height=self.height, ndim=self.ndim, get_states_indices=self.get_states_indices
            )
        elif preprocessor_name == "OneHot":
            preprocessor = OneHotPreprocessor(
                n_states=self.n_states,
                get_states_indices=self.get_states_indices,
            )

        self.preprocessor = preprocessor
        """super().__init__(
            action_space = self.action_space,
            s0 = s0,
            sf = sf,
            device_str = device_str,
            preprocessor=preprocessor
            )"""

    def step(self, action):
        pixel = action // self.num_actions
        is_obj = action % self.num_actions // (len(Direction) + NUM_COLORS)
        row, col = pixel // self.max_rows, pixel % self.max_rows
        if is_obj:
            obj_index = self.obj_map[row, col]
            selected_rows, selected_cols = np.nonzero(self.obj_map == obj_index)
        else:
            selected_rows = np.array([row])
            selected_cols = np.array([col])
        action_arg = action % (self.num_actions // 2)
        is_translate = action_arg < len(Direction)

        if is_translate:
            direction = action % (self.num_actions // 2)
            return self.translate(selected_rows, selected_cols, action_arg)
        else:
            return self.select_fill(selected_rows, selected_cols, action_arg - len(Direction))

    def translate(self, selected_rows, selected_cols, direction):
        prev_state = self.state.copy()

        target_rows = selected_rows + self.translate_y[Direction(direction%4)]
        target_cols = selected_cols + self.translate_x[Direction(direction%4)]

        row_excluder = np.logical_and(target_rows >= 0, target_rows < self.max_rows)
        col_excluder = np.logical_and(target_cols >= 0, target_cols < self.max_cols)
        excluder = np.logical_and(row_excluder, col_excluder)
        target_rows = target_rows[excluder]
        moved_rows = selected_rows[excluder]
        target_cols = target_cols[excluder]
        moved_cols = selected_cols[excluder]

        temp = self.state[moved_rows, moved_cols]
        self.state[selected_rows, selected_cols] = DEFAULT_COLOR
        self.state[target_rows, target_cols] = temp

        temp = self.obj_map[moved_rows, moved_cols]
        self.obj_map[selected_rows, selected_cols] = DEFAULT_COLOR
        self.obj_map[target_rows, target_cols] = temp

        return np.ravel(self.state), np.ravel(self.obj_map), *self.reward_done(prev_state, self.state)

    def select_fill(self, selected_rows, selected_cols, color):
        prev_state = self.state.copy()
        self.state[selected_rows, selected_cols] = color

        return np.ravel(self.state), np.ravel(self.obj_map), *self.reward_done(prev_state, self.state)

    def reward_done(self, state, next_state):
        is_done = int((next_state == self.goal).all())
        if not self.use_reward_shaping:
            return is_done, is_done
        potential = (state == self.goal).sum() ** 2
        next_potential = (next_state == self.goal).sum() ** 2
        initial_distance = (self.initial == self.goal).sum()
        distance = ((next_state == self.goal).sum() - initial_distance)
        return 0.99 * next_potential - potential + is_done, is_done
        return distance + is_done * 50000, is_done

    def reset(self):
        self.state = self.initial.copy()
        #self.obj_map = self.initial_obj_map.copy()
        return np.ravel(self.state) #, np.ravel(self.obj_map)
    
    """def reset(
        self, batch_shape: Union[int, Tuple[int]], random: bool = False
    ) -> States:
        "Instantiates a batch of initial states."
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.States.from_batch_shape(batch_shape=batch_shape, random=random)""" 
    
    def get_states_indices(self, states: States) -> TensorLong:
        states_raw = states.states_tensor

        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices