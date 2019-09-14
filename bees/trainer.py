""" Example of using two different training methods at once in multi-agent. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

from typing import Dict, Tuple, Any

import gym
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from env import Env

# pylint: disable=invalid-name
if __name__ == "__main__":
    ray.init()

    # Get ``settings`` file for now.
    settings_file = sys.argv[1]
    with open(settings_file, "r") as f:
        settings = json.load(f)

    # Parse settings
    env_config = settings["env"]
    width = env_config["width"]
    height = env_config["height"]
    sight_len = env_config["sight_len"]
    num_obj_types = env_config["num_obj_types"]
    num_agents = env_config["num_agents"]
    aging_rate = env_config["aging_rate"]
    food_density = env_config["food_density"]
    food_size_mean = env_config["food_size_mean"]
    food_size_stddev = env_config["food_size_stddev"]
    time_steps = env_config["time_steps"]

    rew_config = settings["rew"]
    n_layers = rew_config["n_layers"]
    hidden_dim = rew_config["hidden_dim"]
    reward_weight_mean = rew_config["weight_mean"]
    reward_weight_stddev = rew_config["weight_stddev"]

    consts = settings["constants"]

    # Register environment
    register_env(
        "bee_world",
        lambda _: Env(
            width,
            height,
            sight_len,
            num_obj_types,
            num_agents,
            aging_rate,
            food_density,
            food_size_mean,
            food_size_stddev,
            n_layers,
            hidden_dim,
            reward_weight_mean,
            reward_weight_stddev,
            consts,
        ),
    )

    # Build environment instance to get ``obs_space``
    env = Env(
        width,
        height,
        sight_len,
        num_obj_types,
        num_agents,
        aging_rate,
        food_density,
        food_size_mean,
        food_size_stddev,
        n_layers,
        hidden_dim,
        reward_weight_mean,
        reward_weight_stddev,
        consts,
    )
    obs_space = env.observation_space
    act_space = env.action_space

    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies: Dict[str, Tuple[Any, gym.Space, gym.Space, Dict[Any, Any]]] = {
        "ppo_policy": (PPOTFPolicy, obs_space, act_space, {})
    }

    def policy_mapping_fn(_agent_id):
        """ Returns the given agent's policy identifier. """
        return "ppo_policy"

    ppo_trainer = PPOTrainer(
        env="bee_world",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["ppo_policy"],
            },
            # Disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent.
            "observation_filter": "NoFilter",
            "num_workers": 1,
        },
    )

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(time_steps):
        print("== Iteration", i, "==")

        # Improve the PPO policy.
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))
