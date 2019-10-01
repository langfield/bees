""" Script for optimizing GPST model hyperparameters via Optuna. """

import json
import logging
import datetime

import optuna

from torch_trainer import train

# pylint: disable=bad-continuation


def main() -> None:
    """ Run an Optuna study. """
    datestring = str(datetime.datetime.now())
    datestring = datestring.replace(" ", "_")
    logging.getLogger().setLevel(logging.INFO)  # Setup the root logger.
    logging.getLogger().addHandler(
        logging.FileHandler("logs/optuna_" + datestring + ".log")
    )
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in stderr.

    study = optuna.create_study()
    logging.getLogger().info("Start optimization.")
    study.optimize(objective, n_trials=100)


def compute_loss(
    num_env_steps: int,
    max_num_env_steps: int,
    avg_agent_lifetime: float,
    aging_rate: float,
    num_agents: int,
    width: int,
    height: int,
) -> float:
    """
    Computes the optuna loss function.

    Parameters
    ----------
    num_env_steps : ``int``.
        Number of environment steps completed so far.
    max_num_env_steps : ``int``.
        Number of environment steps to attempt.
    avg_agent_lifetime : ``float``.
        Average agent lifetime over all done agents measured in environment steps.
    aging_rate : ``float``.
        Health loss for all agents at each environment step.
    num_agents : ``int``.
        Number of living agents.
    width : ``int``.
        Width of the grid.
    height : ``int``.
        Height of the grid.

    Returns
    -------
    loss : ``float``.
        Loss as computed for an optuna trial.
    """
    # Constants.
    # HARDCODE
    optimal_density = 0.05
    optimal_lifetime = 5

    agent_density = num_agents / (width * height)
    lifetime_loss = (avg_agent_lifetime / (1 / aging_rate) - optimal_lifetime) ** 2
    density_loss = (agent_density - optimal_density) ** 2
    step_loss = (max_num_env_steps - num_env_steps) ** 2
    loss = lifetime_loss + density_loss + step_loss
    return loss


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
    settings_file = "settings/torch.json"
    with open(settings_file, "r") as json_file:
        settings = json.load(json_file)

    settings["env"]["sight_len"] = trial.suggest_int("sight_len", 2, 10)
    settings["env"]["num_agents"] = trial.suggest_int("num_agents", 2, 30)
    settings["env"]["food_density"] = trial.suggest_uniform("food_density", 0.05, 0.3)
    settings["env"]["food_size_mean"] = trial.suggest_uniform("foodsz_meann", 0.01, 0.3)
    settings["env"]["food_size_stddev"] = trial.suggest_uniform("foodsz_std", 0.01, 0.3)
    settings["env"]["plant_foods_mean"] = trial.suggest_uniform("plant_mean", -0.2, 1)
    settings["env"]["plant_foods_stddev"] = trial.suggest_uniform("plant_std", 0.0, 1)
    settings["env"]["mating_cooldown_len"] = trial.suggest_int("mate_cooldown", 2, 40)
    settings["env"]["min_mating_health"] = trial.suggest_uniform("min_mate_hth", 0.0, 1)

    settings["env"]["print"] = False
    settings["trial"] = trial

    loss = train(settings)

    return loss


if __name__ == "__main__":
    main()
