""" Analyzes the reward network of an agent. """

import os
import glob
import argparse
import json
import random
import pickle

import numpy as np
from np.random import normal

from agent import Agent

EAT_PROB = 0.1
REWARD_SAMPLE_SIZE = 32


def one_hot(n, k):
    """ Get one-hot vector of length n with k-th bit set as numpy array. """

    vec = np.zeros((n,))
    vec[k] = 1
    return vec


def search_model_dir(modelDir, template):
    """ Search model results directory for results file. """

    results = glob.glob(os.path.join(modelDir, template))
    if len(results) == 0:
        raise ValueError(
            "No files matching template '%s' in %s." % (template, modelDir)
        )
    elif len(results) > 1:
        raise ValueError(
            "More than one file matching template '%s' in %s"
            % (template, args.modelDir)
        )
    return results[0]


def main(main):

    # Read in env.pkl.
    env_path = search_model_dir(args.modelDir, "*_env.pkl")
    with open(env_path, "rb") as env_file:
        env = pickle.load(env_file)

    # Select agent to analyze.
    if args.agent != -1:
        agent_id = args.agent
    else:
        agent_id = random.choice(list(env["agents"].keys()))

    # Construct agent parameters from settings file and environment.
    settings_path = search_model_dir(args.modelDir, "*_settings.json")
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)
    action_space = env["action_space"]
    subaction_sizes = [action_space[i].n for i in range(len(action_space))]
    agent_args = {}
    agent_args["sight_len"] = settings["env"]["sight_len"]
    agent_args["num_obj_types"] = settings["env"]["num_obj_types"]
    agent_args["consts"] = settings["constants"]
    agent_args["n_layers"] = settings["rew"]["n_layers"]
    agent_args["hidden_dim"] = settings["rew"]["hidden_dim"]
    agent_args["num_actions"] = sum(subaction_sizes)
    agent_args["reward_inputs"] = settings["rew"]["reward_inputs"]
    agent_args["reward_weights"] = env["agents"][agent_id]["reward_weights"]
    agent_args["reward_biases"] = env["agents"][agent_id]["reward_biases"]
    agent = Agent(**agent_args)

    # Get settings for sampling health and observation distributions.
    aging_rate = settings["env"]["aging_rate"]
    food_size_mean = settings["env"]["food_size_mean"]
    food_size_stddev = settings["env"]["food_size_stddev"]
    sight_len = settings["env"]["sight_len"]

    # Define function to sample from food distribution.
    def get_food():
        return food_size_mean + normal() * food_size_stddev

    # HARDCODE
    AGENT_OBJ_TYPE = 0

    # Compute the distribution of rewards for each fixed action as observation and
    # health vary.
    distributions = {}

    for subaction_index, subaction_size in enumerate(subaction_sizes):
        for subaction in range(subaction_size):
            for i in range(REWARD_SAMPLE_SIZE):

                # Sample and set health.
                health = random.random()
                prev_health = (
                    health - get_food()
                    if random.random() < EAT_PROB
                    else max(health + aging_rate, 1.0)
                )

                # Sample action.
                action = []
                for j in range(len(subaction_sizes)):
                    if j == subaction_index:
                        action.append(subaction)
                    else:
                        action.append(random.choice(list(range(subaction_sizes[j]))))
                action = tuple(action)

                # Sample and set observation.
                obs_length = 2 * sight_len + 1
                observation = np.zeros((num_obj_types, obs_length, obs_length))
                for x in range(-sight_len, sight_len + 1):
                    for y in range(-sight_len, sight_len + 1):

                        if x == 0 and y == 0:
                            object_type = AGENT_OBJ_TYPE
                        else:
                            object_type = random.choice(num_obj_types)

                        observation[:, x, y] = one_hot(object_type, num_obj_types)

                # Compute reward.
                # DEBUG
                reward_input = (observation, action, health, prev_health)
                print(reward_input)

                # TOMORROW:
                # - Compute reward for each (obs, action, health, prev_health).
                # - Compute average and standard deviation of each reward distribution.
                # - Only sample reward inputs corresponding to
                #   settings["rew"]["reward_inputs"].


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelDir", type=str, help="Directory containing environment state and logs."
    )
    parser.add_argument(
        "--agent",
        type=int,
        default=-1,
        help="Agent id whose reward "
        "network to analyze. If none is provided, agent is chosen randomly.",
    )
    args = parser.parse_args()

    main(args)
