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

    def fill(self):
        """Populate the environment with food and agents."""

        # Set unique agent positions
        grid_positions = list(itertools.product(range(self.rows), range(self.cols)))
        agent_positions = random.sample(grid_positions, self.num_agents)
        for agent_id, agent in enumerate(self.agents):
            pos = agent_positions[agent_id]
            self.grid[pos]["agent"] = agent_id
            agent.pos = pos

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

    def _move(self, action_dict: Dict[str,Dict[str,str]]) -> Dict[str,Dict[str,str]]:
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
            move = action["move"]
            new_pos = Env._update_pos(pos, move)

            # Validate new position.
            out_of_bounds = False
            if new_pos[0] < 0 or new_pos[0] >= self.cols:
                out_of_bounds = True
            if new_pos[1] < 0 or new_pos[1] >= self.rows:
                out_of_bounds = True
            if out_of_bounds or "agent" in self.grid[new_pos]:
                consume = action["consume"]
                action_dict[agent_id] = {"move": "stay", "consume": consume}
            else:
                del self.grid[pos]["agent"]
                self.grid[new_pos]["agent"] = agent_id
                agent.pos = new_pos

        return action_dict

    def _consume(self, action_dict: Dict[str,Dict[str,str]]) -> None:
        """
        Takes as input a collision-free ``action_dict`` and
        executes the ``consume`` action for all agents.
        """
        rew = {}
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            pos = agent.pos
            # If they try to eat when there's nothing there, do nothing.
            consume = action["consume"]
            if "food" in self.grid[pos] and consume == "eat":
                del self.grid[pos]["food"]
            food_size = np.random.normal(self.food_size_mean,
                                         self.food_size_stddev)
            original_health = agent.health
            agent.health = min(1, agent.health + food_size)
            rew[agent_id] = agent.health - original_health

        return rew

    def _get_obs(self, pos: Tuple[int]) -> np.array:
        obs_size = 2 * self.sight_len - 1
        one_hot_dim = max(len(self.agents), self.num_foods)
        obs = np.zeros((obs_size, obs_size, one_hot_dim))
        return obs

    def get_action_dict(self) -> Dict[str,Dict[str,str]]:
        """
        Constructs ``action_dict`` by querying individual agents for
        their actions based on their observations.
        """
        action_dict = {}
        
        for agent_id, agent in enumerate(self.agents):
            action_dict[agent_id] = agent.get_action()

        return action_dict

    def step(self, action_dict: Dict[str,Dict[str,str]]):
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
        for y in range(self.rows):
            for x in range(self.cols):
                pos = (x, y)
                object_id = '_'
                if 'agent' in self.grid[pos]:
                    object_id = 'B'
                elif 'food' in self.grid[pos]:
                    object_id = '*'

                output += object_id + ' '

            output += "\n"

        return output

