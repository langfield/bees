""" Runs the environment and trains the agents for a number of timesteps. """
import os
import sys
import time
import json
from typing import Dict, Any, Tuple

import gym
from env import Env


def create_env(
    settings: Dict[str, Any],
) -> Env:
    """ Create an instance of ``Env`` and return it. """

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
    plant_foods_mean = env_config["plant_foods_mean"]
    plant_foods_stddev = env_config["plant_foods_stddev"]
    food_plant_retries = env_config["food_plant_retries"]
    mating_cooldown_len = env_config["mating_cooldown_len"]
    min_mating_health = env_config["min_mating_health"]
    agent_init_x_upper_bound = env_config["agent_init_x_upper_bound"]
    agent_init_y_upper_bound = env_config["agent_init_y_upper_bound"]

    rew_config = settings["rew"]
    n_layers = rew_config["n_layers"]
    hidden_dim = rew_config["hidden_dim"]
    reward_weight_mean = rew_config["weight_mean"]
    reward_weight_stddev = rew_config["weight_stddev"]

    genetics_config = settings["genetics"]
    mut_sigma = genetics_config["mut_sigma"]
    mut_p = genetics_config["mut_p"]

    consts = settings["constants"]

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
        plant_foods_mean,
        plant_foods_stddev,
        food_plant_retries,
        mating_cooldown_len,
        min_mating_health,
        agent_init_x_upper_bound,
        agent_init_y_upper_bound,
        n_layers,
        hidden_dim,
        reward_weight_mean,
        reward_weight_stddev,
        mut_sigma,
        mut_p,
        consts,
    )
    return env


def main(settings: Dict[str, Any]) -> None:
    """ Main training loop. """

    env_config = settings["env"]
    time_steps = env_config["time_steps"]

    env = create_env(settings)
    env.reset()
    print(env)
    time.sleep(0.2)

    # DEBUG
    print("Beginning loop.")

    for _ in range(time_steps):

        action_dict = env.get_action_dict()
        _obs, _rew, done, _info = env.step(action_dict)

        # Print out environment state
        os.system("clear")
        print(env)
        time.sleep(0)
        if all(done.values()):
            print("All agents have died.")
            break


# pylint: disable=invalid-name
if __name__ == "__main__":

    settings_file = sys.argv[1]
    with open(settings_file, "r") as f:
        SETTINGS = json.load(f)
    main(SETTINGS)
