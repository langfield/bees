#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Plot data from environment log. """
import os
import time
import argparse
from typing import List, Dict, Any, TextIO

import pandas as pd

from plotplotplot.draw import graph


EMA_ALPHA = 0.999


ACTION_NAMES = [
    ["stay", "left", "right", "up", "down"],
    ["eat", "no_eat"],
    ["mate", "no_mate"],
]


def parse_agent_data(steps: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[Any]]]:
    """ Parse log data to be indexed by agent instead of by step. """

    agent_data: Dict[int, Dict[str, List[Any]]] = {}
    for step in steps:
        agents: Dict[int, Dict[str, Any]] = step["agents"]

        # For each agent alive at this timestep.
        for agent_id, agent_info in agents.items():

            # Create container if one does not exist in ``agent_data``.
            if agent_id not in agent_data:
                agent_info_list = {
                    field: [value] for field, value in agent_info.items()
                }
                agent_data[agent_id] = agent_info_list

            # Otherwise, append to the list for each field.
            else:
                for field in agent_data[agent_id]:
                    value_list = agent_data[agent_id][field]
                    if field in agent_info:
                        value_list.append(agent_info[field])
                        agent_data[agent_id][field] = value_list

    return agent_data


def get_EMA(seq: List[float]) -> List[float]:
    """ Computes and returns the EMA of a list of floats. """

    ema = []
    current_avg = 0.0
    for i, element in enumerate(seq):
        if i == 0:
            current_avg = element
        else:
            current_avg = EMA_ALPHA * current_avg + (1.0 - EMA_ALPHA) * element

        ema.append(current_avg)

    return ema


def get_rewards(agent_data: Dict[int, Dict[str, List[Any]]]) -> pd.DataFrame:
    """ Parses ``agent_data`` into a DataFrame with rewards for each agent. """

    # Map ``agent_id`` to lists of last rewards.
    reward_key = "last_reward"
    reward_log_map: Dict[int, List[float]] = {}
    max_length = 0
    for agent_id, value_list_dict in agent_data.items():
        max_length = max(max_length, len(value_list_dict[reward_key]))
        reward_log_map[agent_id] = value_list_dict[reward_key]

    # Replace ``reward_log_map[agent_id]`` with EMA of itself.
    for agent_id in agent_data:
        reward_log_map[agent_id] = get_EMA(reward_log_map[agent_id])

    # Add padding to each list so they all have length ``max_length``.
    for agent_id in reward_log_map:
        reward_log = reward_log_map[agent_id]
        if len(reward_log) < max_length:
            reward_log.extend([float("nan")] * (max_length - len(reward_log)))
            reward_log_map[agent_id] = reward_log

    reward_df = pd.DataFrame.from_dict(reward_log_map)
    return reward_df


def get_child_count_map(agent_data: Dict[int, Dict[str, List[Any]]]) -> Dict[int, int]:
    """ Gets histogram-like data for number of children per agent. """

    children_per_agent = {
        agent_id: agent_data[agent_id]["num_children"][-1]
        for agent_id in agent_data.keys()
    }
    children_values = list(set(children_per_agent.values()))
    num_children = {}
    for children_value in children_values:
        num_children[children_value] = len(
            [
                agent_id
                for agent_id in children_per_agent
                if children_per_agent[agent_id] == children_value
            ]
        )

    return num_children


def get_action_dists(agent_data: Dict[int, Dict[str, List[Any]]]) -> pd.DataFrame:
    """ Compute a moving distribution of actions. """

    """
    ACTION_NAMES = [
        ["stay", "left", "right", "up", "down"],
        ["eat", "no_eat"],
        ["mate", "no_mate"],
    ]
    """

    # Aggregate subaction names.
    aggregated_subactions = []
    for sublist in ACTION_NAMES:
        aggregated_subactions += sublist

    # Initialize moving distributions of actions.
    action_dists: Dict[int, Dict[str, List[float]]] = {
        agent_id: {subaction: [] for subaction in aggregated_subactions} for agent_id in agent_data
    }

    # Compute action distributions for each agent.
    num_subspaces = len(agent_data[0]["last_action"][0])
    for agent_id in agent_data:

        # This is an EMA estimate of the probability of each subaction over time.
        action_probs: List[Dict[str, float]] = [{action_name: -1.0 for action_name in action_sublist} for action_sublist in ACTION_NAMES]

        for chosen_action in agent_data[agent_id]["last_action"]:
            for subaction_index, action_sublist in enumerate(ACTION_NAMES):
                for i, subaction in enumerate(action_sublist):

                    # Update action distribution for a single agent, single subaction.
                    next_prob = float(chosen_action[subaction_index] == i)

                    # Perform EMA update on action_probs.
                    if action_probs[subaction_index][subaction] == -1:
                        action_probs[subaction_index][subaction] = next_prob
                    else:
                        action_probs[subaction_index][subaction] = action_probs[subaction_index][subaction] * EMA_ALPHA + next_prob * (1 - EMA_ALPHA)

                    # Store intermediate EMA.
                    action_dists[agent_id][subaction].append(action_probs[subaction_index][subaction])

    # Convert action_dists to DataFrame format.
    aggregated_action_dists: Dict[str, List[float]] = {}
    for agent_id in action_dists:
        for subaction in action_dists[agent_id]:
            aggregated_action_dists["%d_%s" % (agent_id, subaction)] = action_dists[agent_id][subaction]
    return pd.DataFrame.from_dict(aggregated_action_dists)


# pylint: disable=too-many-locals
def main(args: argparse.Namespace) -> None:
    """ Plot rewards from a log. """

    def readlines(log_file: TextIO) -> List[Dict[str, Any]]:
        """ Construct steps. """
        steps: List[Dict[str, Any]] = []
        for line in log_file:
            stripped: Dict[str, Any] = eval(line.strip())
            steps.append(stripped)
        return steps

    with open(args.log_path, "r") as log_file:
        steps = readlines(log_file)

    # Read and parse log.
    agent_data = parse_agent_data(steps)

    # Get individual metrics from parsed data.
    reward_df = get_rewards(agent_data)
    action_df = get_action_dists(agent_data)
    # child_count_map = get_child_count_map(agent_data)
    food_df = pd.DataFrame({"food": [step["num_foods"] for step in steps]})

    # Plot or write out individual metrics.
    save_dir = os.path.dirname(args.log_path)
    plot_path = os.path.join(save_dir, "metric_log.svg")
    graph(
        dfs=[reward_df, food_df, action_df],
        y_labels=["reward", "food", "action"],
        column_counts=[len(reward_df.columns), 1, len(action_df.columns)],
        save_path=plot_path,
        settings_path=args.settings_path,
    )

    """
    children_path = os.path.join(save_dir, "num_children.txt")
    with open(children_path, "w") as children_file:
        children_file.write(str(child_count_map))
    """


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--log-path", required=True, type=str, help="Path to log.")
    PARSER.add_argument("--settings-path", required=True, type=str)
    ARGS = PARSER.parse_args()
    main(ARGS)
