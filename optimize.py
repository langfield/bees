#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Script for optimizing GPST model hyperparameters via Optuna. """
import os
import json
import shutil
import logging
import argparse
import tempfile
import datetime

import optuna

from bees.trainer import train

# pylint: disable=bad-continuation

LOG_DIR = "logs"

def main() -> None:
    """ Run an Optuna study. """
    datestring = str(datetime.datetime.now())
    datestring = datestring.replace(" ", "_")
    logging.getLogger().setLevel(logging.INFO)  # Setup the root logger.
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_path = os.path.join(LOG_DIR, "optuna_%s.log" % datestring)
    logging.getLogger().addHandler(
        logging.FileHandler(log_path)

    )
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in stderr.

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    logging.getLogger().info("Start optimization.")
    study.optimize(objective, n_trials=3e9)


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function. Should never be called explicitly.

    Parameters
    ----------
    trial : ``optuna.Trial``, required.
        The trial with which we define our hyperparameter suggestions.

    Returns
    -------
    loss : ``float``.
        The output from the model call after the timeout value specified in ``snow.sh``.
    """

    # Get settings and create environment.
    # HARDCODE
    settings_path = "bees/settings/settings.json"
    with open(settings_path, "r") as json_file:
        settings = json.load(json_file)

    # Suggestions for policy hyperparameters.
    settings["algo"] = trial.suggest_categorical("algo", ["ppo"])
    settings["lr"] = trial.suggest_loguniform("lr", 5e-4, 5e-3)
    settings["eps"] = trial.suggest_loguniform("eps", 1e-6, 1e-4)
    settings["alpha"] = trial.suggest_loguniform("alpha", 1e-6, 1e-4)
    settings["gamma"] = trial.suggest_uniform("gamma", 0.98, 0.999)
    settings["use_gae"] = trial.suggest_categorical("use_gae", [True, False])
    settings["gae_lambda"] = trial.suggest_uniform("gae_lambda", 0.9, 1.0)
    settings["entropy_coef"] = trial.suggest_loguniform("entropy_coef", 1e-4, 0.1)
    settings["value_loss_coef"] = trial.suggest_uniform("value_loss_coef", 0.5, 1.0)
    settings["seed"] = trial.suggest_int("seed", 1, 3)
    settings["num_steps"] = trial.suggest_int("num_steps", 32, 512)
    settings["ppo_epoch"] = trial.suggest_int("ppo_epoch", 2, 8)
    settings["clip_param"] = trial.suggest_categorical("clip_param", [0.1, 0.2, 0.3])
    settings["recurrent_policy"] = trial.suggest_categorical(
        "recurrent_policy", [True, False]
    )

    logging.getLogger().info("Settings: %s", str(settings))

    # Hardcoded settings for optimization runs.
    settings["print_repr"] = False
    # settings["trial"] = trial # Add back in for early pruning.
    settings["time_steps"] = 20480
    settings["aging_rate"] = 0.0001
    settings["mating_cooldown_len"] = 51200

    # Print out settings to temp file.
    temp_dir = tempfile.mkdtemp()
    temp_settings_path = os.path.join(temp_dir, "settings.json")
    with open(temp_settings_path, "w") as temp_settings_file:
        json.dump(settings, temp_settings_file)

    # Get ``args`` object to pass to train().
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from", default="", help="Saved directory to load from.")
    parser.add_argument("--settings", default="", help="Settings file to use.")
    parser.add_argument(
        "--save-root",
        default="./models/",
        help="directory to save agent logs (default: ./models/)",
    )
    args = parser.parse_args()
    args.settings = settings_path
    args.trial = trial

    # Print settings and run training.
    print(settings)
    loss = train(args)

    # Cleanup
    shutil.rmtree(temp_dir)
    shutil.rmtree(args.save_root)

    return loss


if __name__ == "__main__":
    main()
