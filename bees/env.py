"""Environment with Bees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import itertools
from typing import List, Tuple

import gym
import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.pg.pg_policy import PGTFPolicy
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.rllib.optimizers import (SyncSamplesOptimizer, SyncReplayOptimizer,
                                  AsyncGradientsOptimizer)
from ray.rllib.tests.test_rollout_worker import (MockEnv, MockEnv2, MockPolicy)
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.env.base_env import _MultiAgentEnvToBaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from agent import Agent 

class Env(MultiAgentEnv):
    """Environment with bees in it."""

    def __init__(self, env_config: dict) -> None:

        # Parse env_config
        self.rows = env_config["rows"]
        self.cols = env_config["cols"]
        self.sight_len = env_config["sight_len"]
        self.obj_types = env_config["obj_types"]
        self.num_agents = env_config["num_agents"]
        self.food_density = env_config["food_density"]

        # Construct ``grid``. 
        grid = {}
        for i in range(self.rows):
            for j in range(self.cols):
                # Dict of object indices. Keys are objtype strings.
                objects = {}
                key = tuple([i,j])
                grid.update({key:objects})
        self.grid = grid

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
        grid_positions = itertools.product(
                range(self.rows),
                range(self.cols)
        )
        agent_positions = random.sample(grid_positions, self.num_agents)
        for i, agent in enumerate(self.agents):
            pos = agent_positions[i]
            self.grid[pos]['agent'] = i 
            agent.pos = agent_positions[i]

        # Set unique food positions
        num_foods = math.floor(self.food_density * len(self.grid))
        food_positions = random.sample(grid_positions, num_foods)
        for i, agent in enumerate(range(num_foods)):
            pos = food_positions[i]
            self.grid[pos]['food'] = i 

    def reset(self):
        self.resetted = True
        self.dones = set()
        self.fill()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict: Dict[Tuple[str]]):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            # Updated agent info should contain new grid positions.
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            # TODO: Update agent positions in env grid.
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

