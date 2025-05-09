import torch
from gflow.env import Env
import numpy as np
from enum import Enum
import copy
# import torch.nn as nn

NUM_COLORS = 10
DEFAULT_COLOR = 0


class ActionType(Enum):
    TRANSLATE = 0
    SELECT_FILL = 1


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class MiniArcEnv:
    def __init__(self, initial, goal, initial_obj_map, use_reward_shaping=True):
        self.initial = initial
        self.max_rows, self.max_cols = self.initial.shape
        self.goal = goal
        self.initial_obj_map = initial_obj_map
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

        self.num_actions = 2 * (len(Direction) + NUM_COLORS)
        self.num_pixels = self.max_rows * self.max_cols
        self.Done = False

    def step(self, action):
        pixel = action // self.num_actions  # action is sigle int
        is_obj = action % self.num_actions // (len(Direction) + NUM_COLORS)
        row, col = pixel // self.max_rows, pixel % self.max_rows
        if is_obj:
            obj_index = self.obj_map[row, col]
            selected = torch.nonzero(self.obj_map == obj_index)

            selected_rows, selected_cols = selected[:, 0], selected[:, 1]
        else:
            selected_rows = torch.tensor([row])
            selected_cols = torch.tensor([col])
        action_arg = action % (self.num_actions // 2)
        is_translate = action_arg < len(Direction)

        if is_translate:
            direction = action % (self.num_actions // 2)
            return self.translate(selected_rows, selected_cols, action_arg)
        else:
            return self.select_fill(selected_rows, selected_cols, action_arg - len(Direction))

        # next_state = np.ravel(self.state), np.ravel(self.obj_map)
        # reward = *self.reward_done(prev_state, self.state)  # you need to implement this
        # done = False  # change this as needed
        # info = {}  # change this as needed
        # return torch.from_numpy(next_state), reward, done, info

    def translate(self, selected_rows, selected_cols, direction):
        prev_state = torch.tensor(self.state)

        target_rows = selected_rows + \
            self.translate_y[Direction(direction.item() % 4)]
        target_cols = selected_cols + \
            self.translate_x[Direction(direction.item() % 4)]

        row_excluder = torch.logical_and(
            target_rows >= 0, target_rows < self.max_rows)
        col_excluder = torch.logical_and(
            target_cols >= 0, target_cols < self.max_cols)
        excluder = torch.logical_and(row_excluder, col_excluder)
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

        return torch.ravel(self.state), torch.ravel(self.obj_map), *self.reward_done(prev_state, self.state)

    def select_fill(self, selected_rows, selected_cols, color):
        prev_state = torch.tensor(self.state, dtype=torch.long)
        self.state[selected_rows, selected_cols] = color

        return torch.ravel(self.state), torch.ravel(self.obj_map), *self.reward_done(prev_state, self.state)

    def reward_done(self, state, next_state):
        is_done = int((next_state == self.goal).all())
        if not self.use_reward_shaping:
            return is_done, is_done
        # potential = (state == self.goal).sum() ** 2
        # next_potential = (next_state == self.goal).sum() ** 2
        # initial_distance = (self.initial == self.goal).sum()
        distance = (next_state - self.goal).square().sum()
        # return 0.99 * next_potential - potential + is_done, is_done
        return 1/distance + is_done * 50000, is_done

    def reset(self):
        self.state = self.initial.clone().detach()
        self.obj_map = self.initial_obj_map.clone().detach()
        return torch.ravel(self.state), torch.ravel(self.obj_map)

    def reward(self):
        # placeholder implementation
        return ((self.state == self.goal).sum() - (self.initial == self.goal).sum()).item()
