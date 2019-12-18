""" Plot data from environment log. """
import os
import pprint
import argparse
from typing import List, Dict, Any

import pandas as pd

from plotplotplot.draw import graph


# pylint: disable=too-many-locals
def main(args: argparse.Namespace) -> None:
    """ Plot rewards from a log. """
    with open(args.log_path, "r") as log_file:
        steps: List[Dict[str, Any]] = [
            eval(line.strip()) for line in log_file.readlines()
        ]
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

    # Map ``agent_id`` to lists of last rewards.
    reward_key = "last_reward"
    reward_log_map: Dict[int, List[float]] = {}
    max_length = 0
    for agent_id, value_list_dict in agent_data.items():
        max_length = max(max_length, len(value_list_dict[reward_key]))
        reward_log_map[agent_id] = value_list_dict[reward_key]

    # Add padding to each list so they all have length ``max_length``.
    for agent_id in reward_log_map:
        reward_log = reward_log_map[agent_id]
        if len(reward_log) < max_length:
            reward_log.extend([float("nan")] * (max_length - len(reward_log)))
            reward_log_map[agent_id] = reward_log

    reward_df = pd.DataFrame.from_dict(reward_log_map)
    save_path = os.path.join(os.path.dirname(args.log_path), 'reward_log.svg')
    graph(
        dfs=[reward_df],
        y_labels=["rewards"],
        column_counts=[len(reward_log_map)],
        save_path=save_path,
        settings_path=args.settings_path,
    )
    print(reward_df)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--log-path", required=True, type=str, help="Path to log.")
    PARSER.add_argument("--settings-path", required=True, type=str)
    ARGS = PARSER.parse_args()
    main(ARGS)
