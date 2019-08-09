import json

from env import Env

def main(settings):

    env = Env(settings['env_config'])
    env.reset()

    for timestep in range(settings['time_steps']):

        action_dict = {}
        obs, rew, done, info = env.step(action_dict)

if __name__ == "__main__":

    settings_file = sys.argv[1]
    with open(settings_file, 'r') as f:
        settings = json.load(f)

    main(settings)
