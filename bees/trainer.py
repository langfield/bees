"""Example of using two different training methods at once in multi-agent.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import argparse

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from env import Env

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    # Get ``settings`` file for now.
    settings_file = sys.argv[1]
    with open(settings_file, "r") as f:
        settings = json.load(f)

    # Simple environment with 4 independent cartpole entities
    register_env("bee_world", lambda _: Env(settings["env_config"]))
    # TODO: Do we really need to construct twice to get ``obs_space``?
    env = Env(settings["env_config"])
    obs_space = env.observation_space
    act_space = env.action_space

    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies = {"ppo_policy": (PPOTFPolicy, obs_space, act_space, {})}

    def policy_mapping_fn(agent_id):
        return "ppo_policy"

    ppo_trainer = PPOTrainer(
        env="bee_world",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["ppo_policy"],
            },
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "NoFilter",
        },
    )

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the PPO policy
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))
