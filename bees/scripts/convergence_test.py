import os
import json
import itertools


DEFAULT_CONFIG_PATH = os.path.join("bees", "settings", "convergence_test.json")
CURRENT_CONFIG_PATH = os.path.join("bees", "settings", "convergence_config.json")


def main():

    # Load default config.
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        default_config = json.load(f)

    # Define variable config values.
    config_variables = {
        "n_layers": [1, 2],
        "hidden_dim": [1, 4, 16],
        "reward_inputs": [["actions"], ["actions", "obs"]],
    }

    # Loop over variable config combinations.
    for current_config_values in itertools.product(*[val for val in config_variables.values()]):

        # Build current config.
        current_config = dict(default_config)
        for i, key in enumerate(config_variables.keys()):
            current_config[key] = current_config_values[i]

        # Save out config and run training.
        with open(CURRENT_CONFIG_PATH, "w") as f:
            json.dump(current_config, f)
        os.system("python3 main.py --settings %s" % CURRENT_CONFIG_PATH)

    # Clean up.
    os.remove(CURRENT_CONFIG_PATH)


if __name__ == "__main__":
    main()
