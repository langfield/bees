""" Runs the environment and trains the agents for a number of timesteps. """
import sys
import os
import json
import time

from env import Env


def main(settings):
    """ Main training loop. """

    env = Env(settings["env_config"])
    env.reset()
    print(env)
    time.sleep(0.2)

    for _ in range(settings["time_steps"]):

        action_dict = env.get_action_dict()
        _obs, _rew, done, _info = env.step(action_dict)

        # Print out environment state
        os.system("clear")
        print(env)
        time.sleep(1)
        if all(done.values()):
            print("All agents have died.")
            break


# pylint: disable=invalid-name
if __name__ == "__main__":

    settings_file = sys.argv[1]
    with open(settings_file, "r") as f:
        SETTINGS = json.load(f)
    main(SETTINGS)
