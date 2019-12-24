""" Analyzes the reward network of an agent. """

import os
import glob
import argparse
import json
import random
import pickle

from agent import Agent


def search_model_dir(modelDir, template):
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
    agent_args = {}
    agent_args["sight_len"] = settings["env"]["sight_len"]
    agent_args["num_obj_types"] = settings["env"]["num_obj_types"]
    agent_args["consts"] = settings["constants"]
    agent_args["n_layers"] = settings["rew"]["n_layers"]
    agent_args["hidden_dim"] = settings["rew"]["hidden_dim"]
    agent_args["num_actions"] = 9  # HARDCODE
    agent_args["reward_inputs"] = settings["rew"]["reward_inputs"]
    agent_args["reward_weights"] = env["agents"][agent_id]["reward_weights"]
    agent_args["reward_biases"] = env["agents"][agent_id]["reward_biases"]
    agent = Agent(**agent_args)

    # Compute the distribution of rewards for each fixed action as observation and
    # health vary.
    distributions = {}
    action_space = env["action_space"]
    subaction_sizes = [action_space[i].n for i in range(len(action_space))]

    for subaction_index, subaction_size in enumerate(subaction_sizes):
        for subaction in range(subaction_size):
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelDir", type=str, help="Directory containing environment " "state and logs."
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
