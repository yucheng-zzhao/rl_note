import time
from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType
np.random.seed(1)
import render


def arr_in_list(array, _list):
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


class GridEnv(gym.Env):

    def __init__(self, size: int, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: str):
        self.agent_location = np.array([0, 0])
        self.time_steps = 0
        self.size = size
        self.render_mode = render_mode
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)
        self.forbidden_location = []
        for fob in forbidden:
            self.forbidden_location.append(np.array(fob))
        self.target_location = np.array(target)
        self.action_space, self.action_space_size = spaces.Discrete(5), spaces.Discrete(5).n

        # reward_list[0]: Represents the reward for a normal action (e.g., moving to a regular grid cell that is neither the target nor forbidden).
        # reward_list[1]: Represents the reward for reaching the target location.
        # reward_list[2]: Represents the penalty for moving into a forbidden state.
        # reward_list[3]: Represents the penalty for taking an invalid action, such as trying to move outside the grid boundaries.
        self.reward_list = [0, 1, -10, -10]
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "barrier": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }
        # Rsa represents the reward structure in the environment. 
        # It is a 3D array where:
        # - The first dimension corresponds to the current state.
        # - The second dimension corresponds to the action taken.
        # - The third dimension corresponds to the reward type (e.g., normal action, reaching the target, forbidden area, invalid action).
        # This variable will be initialized later in the `psa_rsa_init` method.
        self.Rsa = None
        # Psa represents the transition probability matrix in the environment.
        # It is a 3D array where:
        # - The first dimension corresponds to the current state.
        # - The second dimension corresponds to the action taken from the current state.
        # - The third dimension corresponds to the probability of transitioning to the next state.
        # For example, Psa[s, a, s'] stores the probability of moving to state s' after taking action a in state s.
        # This matrix is initialized later in the `psa_rsa_init` method.
        self.Psa = None
        self.psa_rsa_init()
        self.state_space_dim = 2

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_location = np.array([0, 0])
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        reward = self.reward_list[self.Rsa[self.pos2state(self.agent_location), action].tolist().index(1)]
        direction = self.action_to_direction[action]
        self.render_.upgrade_agent(self.agent_location, direction, self.agent_location + direction)
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self.agent_location, self.target_location)
        observation = self.get_obs()
        info = self.get_info()
        return observation, reward, terminated, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))
        self.render_.show_frame(0.3)
        return None

    def get_obs(self) -> ObsType:
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def get_info(self) -> dict:
        return {"time_steps": self.time_steps}

    def state2pos(self, state: int) -> np.ndarray:
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index)
                next_pos = pos + self.action_to_direction[action_index]
                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                    # # In a grid-world environment, if the agent tries to move beyond the grid (e.g., left at the left edge), it stays in the same state.
                    self.Psa[state_index, action_index, state_index] = 1
                    # The reward_list in the class is defined as [0, 1, -10, -10], where the fourth element (-10) corresponds to the penalty for invalid actions.
                    self.Rsa[state_index, action_index, 3] = 1
                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1
                    if np.array_equal(next_pos, self.target_location):
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location):
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        self.Rsa[state_index, action_index, 0] = 1


if __name__ == "__main__":
    grid = GridEnv(size=5, target=[1, 2], forbidden=[[2, 2]], render_mode='')
    grid.render()
