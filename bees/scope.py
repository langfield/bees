""" Analyzes the reward network of an agent. """
import os
import json
import glob
import random
import pickle
import argparse
import functools
from pprint import pprint

import numpy as np

from bees.agent import Agent

EAT_PROB = 0.1
OBS_DENSITY = 0.3
REWARD_SAMPLE_SIZE = 3200


def one_hot(n: int, k: int) -> np.ndarray:
    """ Get one-hot vector of length n with k-th bit set as numpy array. """

    vec = np.zeros((n,))
    vec[k] = 1
    return vec


def search_model_dir(model_dir: str, template: str) -> str:
    """ Search model results directory for results file. """

    results = glob.glob(os.path.join(model_dir, template))
    if len(results) == 0:
        raise ValueError(
            "No files matching template '%s' in %s." % (template, model_dir)
        )
    elif len(results) > 1:
        raise ValueError(
            "More than one file matching template '%s' in %s"
            % (template, model_dir)
        )
    return results[0]


def scope(args: argparse.Namespace) -> None:

    # Read in env.pkl.
    env_path = search_model_dir(args.model_dir, "*_env.pkl")
    with open(env_path, "rb") as env_file:
        env = pickle.load(env_file)

    # Select agent to analyze.
    if args.agent != -1:
        agent_id = args.agent
    else:
        agent_id = random.choice(list(env["agents"].keys()))

    # Construct agent parameters from settings file and environment.
    settings_path = search_model_dir(args.model_dir, "*_settings.json")
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
    num_obj_types = settings["env"]["num_obj_types"]

    # HARDCODE
    AGENT_OBJ_TYPE = 0

    # Compute the distribution of rewards for each fixed action as observation and
    # health vary.
    distributions = {}

    num_actions = functools.reduce(lambda a, b: a * b, subaction_sizes)
    for action in range(num_actions):
        rewards = []

        # HARDCODE
        # Since the observation and health spaces are large, we randomly sample from
        # them instead of iterating over them. If these values aren't inputs to the
        # reward network, there is no need to sample, so take sample_size = 1.
        sample_size = (
            1 if settings["rew"]["reward_inputs"] == ["actions"] else REWARD_SAMPLE_SIZE
        )
        for _ in range(sample_size):

            # Sample and set health.
            health = random.random()
            agent.health = health

            # Sample and set observation.
            obs_length = 2 * sight_len + 1
            observation = np.zeros((num_obj_types, obs_length, obs_length))
            for x in range(obs_length):
                for y in range(obs_length):

                    # x == sight_len == y iff (x, y) is in center of agent's
                    # field of vision, AKA the agent's position.
                    if random.random() < OBS_DENSITY or (x == sight_len == y):
                        if x == sight_len == y:
                            object_type = AGENT_OBJ_TYPE
                        else:
                            object_type = random.choice(list(range(num_obj_types)))
                        observation[:, x, y] = one_hot(num_obj_types, object_type)
            agent.observation = observation

            # Compute reward.
            rewards.append(agent.compute_reward(action))

        mean = np.mean(rewards)
        std = np.std(rewards)
        distributions[action] = {"mean": mean, "std": std}

    pprint(distributions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, help="Directory containing environment state and logs."
    )
    parser.add_argument(
        "--agent",
        type=int,
        default=-1,
        help="Agent id whose reward "
        "network to analyze. If none is provided, agent is chosen randomly.",
    )
    args = parser.parse_args()

    scope(args)


if __name__ == "__main__":
    main()
