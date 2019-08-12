import sys
import os
import json
import time

from env import Env

def main(settings):

    env = Env(settings['env_config'])
    env.reset()
    print(env)
    time.sleep(2)

    for timestep in range(settings['time_steps']):

        action_dict = env.get_action_dict()
        obs, rew, done, info = env.step(action_dict)

        # Print out environment state
        os.system('clear')
        print(env)
        time.sleep(2)

if __name__ == "__main__":

    settings_file = sys.argv[1]
    with open(settings_file, 'r') as f:
        settings = json.load(f)

    main(settings)
