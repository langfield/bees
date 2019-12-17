""" Plot data from environment log. """
import json
from typing import List


def main() -> None:
    with open(log_path, "r") as log_file:
        steps = [json.loads(line.strip()) for line in log_file.readlines()]
    agent_data: Dict[int, Dict[str, List[Any]]] = {}
    for i, step in steps.items():
        iteration = step["iteration"]
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

    reward_key = "last_reward"
    agent_reward_data: Dict[int, List[float]] = {}
    # for agent_id, value_list_dict in agent_data.it


if __name__ == "__main__":
    main()
