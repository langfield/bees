""" Script for optimizing GPST model hyperparameters via Optuna. """

import os
import time
import json
import shutil
import logging
import tempfile
import datetime

import optuna


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
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    # Set arguments.
    args.num_train_epochs = 10000
    args.stationarize = True
    args.normalize = False
    args.seq_norm = True
    args.seed = 42
    args.max_grad_norm = 3
    args.adam_epsilon = 7.400879524874149e-08

    batch_size = 192
    agg_size = 5

    # Commented-out trial suggestions should be placed at top of block.
    # args.stationarize = trial.suggest_categorical("stationarize", [True, False])
    # batch_size = trial.suggest_discrete_uniform("train_batch_size", 32, 256, 32)
    # agg_size = trial.suggest_discrete_uniform("agg_size", 1, 40, 5)
    args.weight_decay = trial.suggest_loguniform("weight_decay", 0.0001, 0.01)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 8e-7, 5e-4)
    args.warmup_proportion = trial.suggest_uniform("warmup_proportion", 0.05, 0.4)
    args.train_batch_size = int(batch_size)
    args.aggregation_size = int(agg_size)
    logging.getLogger().info(str(args))

    # Set config.
    config = {}
    config["initializer_range"] = 0.02
    config["n_head"] = 8
    config["vocab_size"] = 33
    n_positions = 30

    # Commented-out trial suggestions should be placed at top of block.
    # n_positions = int(trial.suggest_discrete_uniform("n_ctx", 10, 40, 5))
    # config["n_head"] = int(trial.suggest_discrete_uniform("n_head", 4, 16, 4))
    config["layer_norm_epsilon"] = trial.suggest_loguniform("layer_eps", 1e-5, 1e-3)
    config["resid_pdrop"] = trial.suggest_loguniform("resid_pdrop", 0.01, 0.15)
    config["attn_pdrop"] = trial.suggest_loguniform("attn_pdrop", 0.1, 0.3)
    config["n_embd"] = int(trial.suggest_discrete_uniform("n_embd", 32, 768, 64))
    config["n_layer"] = trial.suggest_int("n_layer", 4, 10)
    config["n_positions"] = n_positions
    config["n_ctx"] = n_positions

    dirpath = tempfile.mkdtemp()
    config_filename = str(time.time()) + ".json"
    config_filepath = os.path.join(dirpath, config_filename)
    with open(config_filepath, "w") as path:
        json.dump(config, path)
    args.gpst_model = config_filepath
    args.model_name = "optuna"
    args.trial = trial

    loss = train(args)

    shutil.rmtree(dirpath)
    return loss


if __name__ == "__main__":
    main()
