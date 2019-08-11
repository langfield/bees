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
        self.rows = env_config["rows"]
        self.cols = env_config["cols"]
        self.sight_len = env_config["sight_len"]
        self.obj_types = env_config["obj_types"]
        self.num_agents = env_config["num_agents"]
        self.food_density = env_config["food_density"]
        self.food_size_mean = env_config["food_size_mean"]
        self.food_size_stddev = env_config["food_size_stddev"]

        # Construct ``grid``.
        grid = {}
        for y in range(self.rows):
            for x in range(self.cols):
                # Dict of object indices. Keys are objtype strings.
                objects = {}
                key = tuple([x, y])
                grid.update({key: objects})
        self.grid = grid

        # Compute number of foods.
        self.num_foods = math.floor(self.food_density * len(self.grid))

        # Construct observation and action spaces
        self.action_space = gym.spaces.Dict({
            "move": gym.spaces.Discrete(5),
            "consume": gym.spaces.Discrete(2)
        })

        obs_dict = {}
        for x in range(-self.sight_len + 1, self.sight_len):
            for y in range(-self.sight_len + 1, self.sight_len):
                obs_dict[(x, y)] = gym.spaces.Discrete(self.obj_types)
        self.observation_space = gym.spaces.Dict(obs_dict)

        # Construct agents
        self.agents = [Agent() for i in range(self.num_agents)]

        # Misc settings
        self.dones = set()
        self.resetted = False

        # Fill environment
        self.fill()

    def fill(self):
        """Populate the environment with food and agents."""

        # Set unique agent positions
        grid_positions = itertools.product(range(self.rows), range(self.cols))
        agent_positions = random.sample(grid_positions, self.num_agents)
        for agent_id, agent in enumerate(self.agents):
            pos = agent_positions[agent_id]
            self.grid[pos]["agent"] = agent_id
            agent.pos = agent_positions[agent_id]

        # Set unique food positions
        food_positions = random.sample(grid_positions, self.num_foods)
        for agent_id, agent in enumerate(range(self.num_foods)):
            pos = food_positions[agent_id]
            self.grid[pos]["food"] = agent_id

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

    def _move(self, action_dict: Dict[Tuple[str]]) -> Dict[Tuple[str]]:
        """
        Identify collisions and update ``action_dict``,
        ``self.grid``, and ``agent.pos``.
        """
        # Shuffle the keys.
        for agent_id, action in random.shuffle(action_dict.items()):
            agent = self.agents[agent_id]
            pos = agent.pos
            move = action["move"]
            new_pos = Env._update_pos(pos, move)

            # Validate new position.
            out_of_bounds = False
            if new_pos[0] < 0 or new_pos[0] >= self.cols:
                out_of_bounds = True
            if new_pos[1] < 0 or new_pos[1] >= self.rows:
                out_of_bounds = True
            if "agent" in self.grid[new_pos] or out_of_bounds:
                consume = action[1]
                action_dict[agent_id] = tuple(["stay", consume])
            else:
                del self.grid[pos]["agent"]
                self.grid[new_pos]["agent"] = agent_id
                agent.pos = new_pos

        return action_dict

    def _consume(self, action_dict: Dict[Tuple[str]]) -> None:
        """
        Takes as input a collision-free ``action_dict`` and
        executes the ``consume`` action for all agents.
        """
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            pos = agent.pos
            # If they try to eat when there's nothing there, do nothing.
            consume = action[1]
            if "food" in self.grid[pos] and consume == "eat":
                del self.grid[pos]["food"]
            food_size = np.random.normal(self.food_size_mean,
                                         self.food_size_stddev)
            agent.health = min(1, agent.health + food_size)

    def _get_obs(self, pos: Tuple[int]) -> np.array:
        obs_size = 2 * self.sight_len - 1
        one_hot_dim = max(len(self.agents), self.num_foods)
        obs = np.zeros((obs_size, obs_size, one_hot_dim))
        return obs

    def step(self, action_dict: Dict[Tuple[str]]):
        """
        ``action_dict`` has agent indices as keys and a tuple of the form
        ``(<move>, <consume>)`` where the tuple elements are strings
        from the sets
            ``movements = set(["up", "down", "left", "right", "stay"])``
            ``consumptions = set(["eat", "noeat"])``.
        """
        # Compute collisions and update ``self.grid``.
        action_dict = self._move(action_dict)
        self._consume(action_dict)
        obs, rew, done, info = {}, {}, {}, {}
        for agent_id, agent in enumerate(self.agents):
            # Compute ovservation.
            obs[agent_id] = self._get_obs(agent.pos)
            rew[agent_id] = 1  # TODO: implement.
            done[agent_id] = False
        """
        for i, action in action_dict.items():
            # Updated agent info should contain new grid positions.
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        """
        return obs, rew, done, info
