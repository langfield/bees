"""Environment with Bees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import itertools

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

def one_hot(identifier):

    assert identifier in ['agent', 'food']


class Env(MultiAgentEnv):
    """Environment with bees in it."""

    def __init__(self, env_config: dict) -> None:

        # Parse env_config
        self.rows = env_config["rows"]
        self.cols = env_config["cols"]
        self.sight_len = env_config["sight_len"]
        self.obj_types = env_config["obj_types"]
        self.num_agents = env_config["num_agents"]

        # Construct ``grid``. 
        grid = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                objects = []
                row.append(objects)
            grid.append(row)
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
                range(self.rows)
        )
        agent_positions = random.sample(grid_positions, self.num_agents)

        for i, agent in enumerate(self.agents):
            pos = agent_positions[i]
            self.grid[pos] = one_hot('agent')
            agent.pos = agent_positions[i]

        # Set unique food positions

        pass

    def reset(self):
        self.resetted = True
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

