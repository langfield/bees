"""Environment with Bees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard imports.
import os
import math
import random
import itertools
from typing import Tuple, Dict

# Third-party imports.
import numpy as np

# Package imports.
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Bees imports.
from agent import Agent
from utils import convert_obs_to_tuple

# HARDCODE
REPR_LOG = "logs/repr_log.txt"


class Env(MultiAgentEnv):
    """Environment with bees in it."""

    def __init__(self, env_config: dict) -> None:

        # Parse ``env_config``.
        self.height = env_config["height"]
        self.width = env_config["width"]
        self.sight_len = env_config["sight_len"]
        self.obj_types = env_config["obj_types"]
        self.num_agents = env_config["num_agents"]
        self.food_density = env_config["food_density"]
        self.food_size_mean = env_config["food_size_mean"]
        self.food_size_stddev = env_config["food_size_stddev"]
        self.aging_rate = env_config["aging_rate"]

        # Get constants.
        self.consts = env_config["constants"]
        self.heaven = tuple(self.consts["BEE_HEAVEN"])

        # Construct object identifier dictionary
        self.obj_type_id = {"agent": 0, "food": 1}

        # Compute number of foods.
        num_squares = self.width * self.height
        self.initial_num_foods = math.floor(self.food_density * num_squares)
        self.num_foods = self.initial_num_foods

        # Construct ``grid``.
        self.grid = np.zeros((self.width, self.height, self.obj_types))

        # Construct observation and action spaces
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(5), gym.spaces.Discrete(2))
        )

        # Each observation is a k * k matrix with values from a discrete
        # space of size self.obj_types + 1, where k = 2 * self.sight_len + 1
        outer_list = []
        for _x in range(-self.sight_len, self.sight_len + 1):
            inner_list = []
            for _y in range(-self.sight_len, self.sight_len + 1):
                agent_space = gym.spaces.Discrete(2)
                food_space = gym.spaces.Discrete(2)
                inner_list.append(gym.spaces.Tuple((agent_space, food_space)))
            inner_space = gym.spaces.Tuple(tuple(inner_list))
            outer_list.append(inner_space)
        self.observation_space = gym.spaces.Tuple(tuple(outer_list))

        # Construct agents
        self.agents = [Agent() for i in range(self.num_agents)]

        # Misc settings
        self.dones = set()
        self.resetted = False
        self.iteration = 0

    def fill(self):
        """Populate the environment with food and agents."""
        # Reset ``self.grid``.
        self.grid = np.zeros((self.width, self.height, self.obj_types))

        # Set unique agent positions.
        grid_positions = list(itertools.product(range(self.height), range(self.width)))
        agent_positions = random.sample(grid_positions, self.num_agents)
        for agent, agent_pos in zip(self.agents, agent_positions):
            self._place(self.obj_type_id["agent"], agent_pos)
            agent.pos = agent_pos

        # Set unique food positions
        food_positions = random.sample(grid_positions, self.initial_num_foods)
        for food_pos in food_positions:
            self._place(self.obj_type_id["food"], food_pos)

    def reset(self):
        """ Reset the entire environment. """
        self.iteration = 0
        self.resetted = True
        self.dones = set()
        self.fill()

        # Set initial agent observations
        for _, agent in enumerate(self.agents):
            agent.observation = self._get_obs(agent.pos)

        return {i: a.reset() for i, a in enumerate(self.agents)}

    def _update_pos(self, pos: Tuple[int], move: int) -> Tuple[int]:
        """Compute new position from a given move."""
        if move == self.consts["UP"]:
            new_pos = tuple([pos[0], pos[1] + 1])
        elif move == self.consts["DOWN"]:
            new_pos = tuple([pos[0], pos[1] - 1])
        elif move == self.consts["LEFT"]:
            new_pos = tuple([pos[0] - 1, pos[1]])
        elif move == self.consts["RIGHT"]:
            new_pos = tuple([pos[0] + 1, pos[1]])
        elif move == self.consts["STAY"]:
            new_pos = pos
        else:
            raise ValueError("'%s' is not a valid action.")

        return new_pos

    def _remove(self, obj_type_id: int, pos: Tuple[int]) -> None:

        grid_idx = pos + (obj_type_id,)
        self.grid[grid_idx] = 0

    def _place(self, obj_type_id: int, pos: Tuple[int]) -> None:

        grid_idx = pos + (obj_type_id,)
        self.grid[grid_idx] = 1

    def _obj_exists(self, obj_type_id: int, pos: Tuple[int]) -> bool:

        grid_idx = pos + (obj_type_id,)
        return self.grid[grid_idx] == 1

    def _move(self, action_dict: Dict[int, Tuple[int]]) -> Dict[int, Tuple[int]]:
        """ Identify collisions and update ``action_dict``,
            ``self.grid``, and ``agent.pos``.
        """
        # Shuffle the keys.
        shuffled_items = list(action_dict.items())
        random.shuffle(shuffled_items)
        for agent_id, action in shuffled_items:
            agent = self.agents[agent_id]
            pos = agent.pos
            move, consume = action
            new_pos = self._update_pos(pos, move)

            # Validate new position.
            out_of_bounds = False
            if new_pos[0] < 0 or new_pos[0] >= self.width:
                out_of_bounds = True
            if new_pos[1] < 0 or new_pos[1] >= self.height:
                out_of_bounds = True

            if out_of_bounds or self._obj_exists(self.obj_type_id["agent"], new_pos):
                action_dict[agent_id] = (self.consts["STAY"], consume)
            else:
                self._remove(self.obj_type_id["agent"], pos)
                self._place(self.obj_type_id["agent"], new_pos)
                agent.pos = new_pos

        return action_dict

    def _consume(self, action_dict: Dict[str, Dict[str, str]]) -> None:
        """ Takes as input a collision-free ``action_dict`` and
            executes the ``consume`` action for all agents.
        """
        rew = {}
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            pos = agent.pos

            # If they try to eat when there's nothing there, do nothing.
            _, consume = action
            if (
                self._obj_exists(self.obj_type_id["food"], pos)
                and consume == self.consts["EAT"]
            ):
                self._remove(self.obj_type_id["food"], pos)
                self.num_foods -= 1

            food_size = np.random.normal(self.food_size_mean, self.food_size_stddev)
            original_health = agent.health
            agent.health = min(1, agent.health + food_size)
            rew[agent_id] = agent.health - original_health

        return rew

    def _get_obs(self, pos: Tuple[int]) -> np.ndarray:
        """ Returns a ``np.ndarray`` of observations given an agent ``pos``. """

        # Calculate bounds of field of vision.
        x = pos[0]
        y = pos[1]
        sight_left = x - self.sight_len
        sight_right = x + self.sight_len
        sight_bottom = y - self.sight_len
        sight_top = y + self.sight_len

        # Calculate length of zero-padding in case sight goes out of bounds.
        pad_left = max(-sight_left, 0)
        pad_right = max(sight_right - self.width + 1, 0)
        pad_bottom = max(-sight_bottom, 0)
        pad_top = max(sight_top - self.height + 1, 0)

        # Constrain field of vision within grid bounds.
        sight_left = max(sight_left, 0)
        sight_right = min(sight_right, self.width - 1)
        sight_bottom = max(sight_bottom, 0)
        sight_top = min(sight_top, self.height - 1)

        # Construct observation.
        obs_len = 2 * self.sight_len + 1
        obs = np.zeros((obs_len, obs_len, self.obj_types))
        pad_x_len = obs_len - pad_left - pad_right
        pad_y_len = obs_len - pad_top - pad_bottom
        obs[
            pad_left : pad_left + pad_x_len, pad_bottom : pad_bottom + pad_y_len
        ] = self.grid[sight_left : sight_right + 1, sight_bottom : sight_top + 1]
        obs = convert_obs_to_tuple(obs)

        return obs

    def get_action_dict(self) -> Dict[str, Tuple[int]]:
        """
        Constructs ``action_dict`` by querying individual agents for
        their actions based on their observations.
        """
        action_dict = {}

        for agent_id, agent in enumerate(self.agents):
            action_dict[agent_id] = agent.get_action()

        return action_dict

    def step(self, action_dict: Dict[str, Tuple[int]]):
        """
        ``action_dict`` has agent indices as keys and a dict of the form
        ``{"move": <move>, "consume": <consume>)`` where the dict values
        are strings from the sets
            ``movements = set(["up", "down", "left", "right", "stay"])``
            ``consumptions = set(["eat", "noeat"])``.
        """

        # Execute move actions and consume actions, and calculate reward
        action_dict = self._move(action_dict)
        obs, rew, done, info = {}, {}, {}, {}
        rew = self._consume(action_dict)

        # Decrease agent health, compute observations and dones.
        for agent_id, agent in enumerate(self.agents):
            if agent.health > 0.0:
                agent.health -= self.aging_rate
                obs[agent_id] = self._get_obs(agent.pos)
                agent.observation = obs[agent_id]
                done[agent_id] = self.num_foods == 0 or agent.health <= 0.0

                # Kill agent if ``done[agent_id]`` and remove from ``self.grid``.
                if done[agent_id]:
                    self._remove(self.obj_type_id["agent"], agent.pos)
                    agent.pos = self.heaven

        agents_done = [done[agent_id] for agent_id in range(len(self.agents))]
        done["__all__"] = all(agents_done)

        # Write environment representation to log
        self._log_state()

        self.iteration += 1
        return obs, rew, done, info

    def __repr__(self):
        """
        Returns a representation of the environment state.
        """

        output = ""
        for y in range(self.height):
            for x in range(self.width):

                pos = (x, y)
                object_id = "_"

                # Check if there is an agent in ``pos``.
                if self._obj_exists(self.obj_type_id["agent"], pos):
                    object_id = "B"
                # NOTE: ``B`` currently overwrites ``*``.
                # Check if there is a food in ``pos``.
                elif self._obj_exists(self.obj_type_id["food"], pos):
                    object_id = "*"

                output += object_id + " "

            output += "\n"

        return output

    def _log_state(self):
        """
        Logs the state of the environment as a string to a
        prespecified log file path.
        """

        log_dir = os.path.dirname(REPR_LOG)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        with open(REPR_LOG, "a+") as f:
            f.write("Iteration %d:\n" % self.iteration)
            f.write(self.__repr__())
            f.write(",\n")
