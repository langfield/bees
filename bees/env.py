"""Environment with Bees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import itertools
from typing import Tuple, Dict

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from agent import Agent


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

        # Construct object identifier dictionary
        self.obj_id = {"agent": 0, "food": 1}

        # Compute number of foods.
        num_squares = self.width * self.height
        self.num_foods = math.floor(self.food_density * num_squares)
        # Foods are not currently unique.
        self.food_id = 0
        self.max_obj_count = max(self.num_agents, self.num_foods)

        # Construct ``grid``.
        self.grid = np.zeros(
            (self.width, self.height, self.obj_types, self.max_obj_count)
        )

        # Construct observation and action spaces
        self.action_space = gym.spaces.Dict(
            {"move": gym.spaces.Discrete(5), "consume": gym.spaces.Discrete(2)}
        )

        # Each observation is a k * k matrix with values from a discrete
        # space of size self.obj_types + 1, where k = 2 * self.sight_len - 1
        outer_list = []
        for _x in range(-self.sight_len + 1, self.sight_len):
            inner_list = []
            for _y in range(-self.sight_len + 1, self.sight_len):
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

    def fill(self):
        """Populate the environment with food and agents."""
        # Reset ``self.grid``.
        self.grid = np.zeros(
            (self.width, self.height, self.obj_types, self.max_obj_count)
        )

        # Set unique agent positions.
        grid_positions = list(itertools.product(range(self.height), range(self.width)))
        agent_positions = random.sample(grid_positions, self.num_agents)
        for agent_id, agent in enumerate(self.agents):
            # Shape: (4,).
            grid_idx = agent_positions[agent_id] + (self.obj_id["agent"], agent_id)
            self.grid[grid_idx] = 1
            agent.pos = agent_positions[agent_id]

        # Set unique food positions
        food_positions = random.sample(grid_positions, self.num_foods)
        for agent_id, agent in enumerate(range(self.num_foods)):
            # Shape: (4,).
            grid_idx = food_positions[agent_id] + (self.obj_id["food"], self.food_id)
            self.grid[grid_idx] = 1

    def reset(self):
        self.resetted = True
        self.dones = set()
        self.fill()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    @staticmethod
    def _update_pos(pos: Tuple[int], move: str) -> Tuple[int]:
        """Compute new position from a given move."""
        if move == "up":
            new_pos = tuple([pos[0], pos[1] + 1])
        elif move == "down":
            new_pos = tuple([pos[0], pos[1] - 1])
        elif move == "left":
            new_pos = tuple([pos[0] - 1, pos[1]])
        elif move == "right":
            new_pos = tuple([pos[0] + 1, pos[1]])
        else:
            new_pos = pos
        return new_pos

    def _move(
        self, action_dict: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Identify collisions and update ``action_dict``,
        ``self.grid``, and ``agent.pos``.
        """
        # Shuffle the keys.
        shuffled_items = list(action_dict.items())
        random.shuffle(shuffled_items)
        for agent_id, action in shuffled_items:
            agent = self.agents[agent_id]
            pos = agent.pos
            # Shape: (4,).
            grid_idx = pos + (self.obj_id["agent"], agent_id)
            move = action["move"]
            new_pos = Env._update_pos(pos, move)

            # Validate new position.
            out_of_bounds = False
            if new_pos[0] < 0 or new_pos[0] >= self.width:
                out_of_bounds = True
            if new_pos[1] < 0 or new_pos[1] >= self.height:
                out_of_bounds = True
            # Shape: (4,).
            new_grid_idx = new_pos + (self.obj_id["agent"], agent_id)
            if out_of_bounds or self.grid[new_grid_idx] == 1:
                consume = action["consume"]
                action_dict[agent_id] = {"move": "stay", "consume": consume}
            else:
                self.grid[grid_idx] = 0
                self.grid[new_grid_idx] = 1
                agent.pos = new_pos

        return action_dict

    def _consume(self, action_dict: Dict[str, Dict[str, str]]) -> None:
        """
        Takes as input a collision-free ``action_dict`` and
        executes the ``consume`` action for all agents.
        """
        rew = {}
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            pos = agent.pos
            grid_idx = pos + (self.obj_id["agent"], self.food_id)
            # If they try to eat when there's nothing there, do nothing.
            consume = action["consume"]
            if self.grid[grid_idx] == 1 and consume == "eat":
                self.grid[grid_idx] = 0
            food_size = np.random.normal(self.food_size_mean, self.food_size_stddev)
            original_health = agent.health
            agent.health = min(1, agent.health + food_size)
            rew[agent_id] = agent.health - original_health

        return rew

    def _get_obs(self, pos: Tuple[int]) -> np.ndarray:
        x = pos[0]
        y = pos[1]
        sight_left = x - self.sight_len + 1 
        sight_right = x + self.sight_len
        sight_bottom = y - self.sight_len + 1
        sight_top = y + self.sight_len
        sight_left = max(sight_left, 0)
        sight_right = min(sight_right, self.width) 
        sight_bottom = max(sight_bottom, 0)
        sight_top = min(sight_top, self.height)
        agent_obs = self.grid[sight_left:sight_right, sight_bottom: sight_top, self.obj_id["agent"]]
        food_obs = self.grid[sight_left:sight_right, sight_bottom: sight_top, self.obj_id["food"]]
        # if agent -> agent
        # if ~agent and food -> food
        # else 0
        # TODO: Consolidate ``agent_obs`` and ``food_obs``.
        
        return obs

    def get_action_dict(self) -> Dict[str, Dict[str, str]]:
        """
        Constructs ``action_dict`` by querying individual agents for
        their actions based on their observations.
        """
        action_dict = {}

        for agent_id, agent in enumerate(self.agents):
            action_dict[agent_id] = agent.get_action()

        return action_dict

    def step(self, action_dict: Dict[str, Dict[str, str]]):
        """
        ``action_dict`` has agent indices as keys and a dict of the form
        ``{"move": <move>, "consume": <consume>)`` where the dict values
        are strings from the sets
            ``movements = set(["up", "down", "left", "right", "stay"])``
            ``consumptions = set(["eat", "noeat"])``.
        """
        # Compute collisions and update ``self.grid``.
        action_dict = self._move(action_dict)
        rew = self._consume(action_dict)
        obs, rew, done, info = {}, {}, {}, {}
        for agent_id, agent in enumerate(self.agents):

            # Compute observation.
            obs[agent_id] = self._get_obs(agent.pos)
            agent.observation = obs[agent_id]
            done[agent_id] = False

        return obs, rew, done, info

    def __repr__(self):
        """
        Returns a representation of the environment state.
        """

        output = ""
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                grid_agent_vec_idx = pos + (self.obj_id["agent"],)
                grid_food_vec_idx = pos + (self.obj_id["food"],)
                object_id = "_"
                # Check if nonzero vals in ``self.grid[grid_agent_vec_idx]``.
                if len(np.nonzero(self.grid[grid_agent_vec_idx])[0]) > 0:
                    object_id = "B"
                # NOTE: ``B`` currently overwrites ``*``.
                # Check if nonzero vals in ``self.grid[grid_food_vec_idx]``.
                elif len(np.nonzero(self.grid[grid_food_vec_idx])[0]) > 0:
                    object_id = "*"

                output += object_id + " "

            output += "\n"

        return output
