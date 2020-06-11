#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Trainer initialization class. """
import os
import json
import glob
import pickle
import shutil
import argparse
import datetime
from typing import TextIO, Any, Dict
from bees.utils import get_token, validate_args
from bees.config import Config

# pylint: disable=too-few-public-methods


class Setup:
    """
    Object to setup training process. Meant to be run at the start of
    ``bees.trainer.train()``.

    Parameters
    ----------
    args : ``argparse.Namespace``.
        Args determining whether or not to load saved model, where to save models, and
        what settings file to use.


    Raises
    ------
    ValueError
        In the case where both ``--settings`` and ``--load_from`` are not passed.
    """

    def __init__(self, args: argparse.Namespace):

        # Convert to abspaths.
        if args.load_from:
            args.load_from = os.path.abspath(os.path.expanduser(args.load_from))
            args.load_from = glob.glob(args.load_from)[0]
        if args.save_root:
            args.save_root = os.path.abspath(os.path.expanduser(args.save_root))
        if args.save_path:
            args.save_path = os.path.abspath(os.path.expanduser(args.save_path))
        if args.settings:
            args.settings = os.path.abspath(os.path.expanduser(args.settings))

        validate_args(args)

        trainer_state: Dict[str, Any] = {}
        trainer_state_path: str = ""
        env_state_path: str = ""

        if not args.settings:
            raise ValueError("You must pass a value for ``--settings``.")

        # Resume from previous run.
        if args.load_from:

            # Construct new codename.
            # NOTE: we were going to have the basename be just the token, but this seems
            # ill-advised since you'd have to go into each folder to determine which is
            # newest.
            codename = os.path.basename(os.path.abspath(args.load_from))
            token = codename.split("_")[0]

            # Construct paths.
            # TODO: Do we want to use glob here? Dangerous in any way?
            env_filename = codename + "*_env.pkl"
            trainer_filename = codename + "*_trainer.pkl"
            settings_filename = codename + "*_settings.json"
            env_state_path = os.path.join(args.load_from, env_filename)
            trainer_state_path = os.path.join(args.load_from, trainer_filename)
            settings_path = os.path.join(args.load_from, settings_filename)

            print("Listdir:", os.listdir(args.load_from))

            # Glob.
            env_state_path = glob.glob(env_state_path)[0]
            trainer_state_path = glob.glob(trainer_state_path)[0]
            settings_path = glob.glob(settings_path)[0]

            # Load trainer state.
            print("DEBUG: trying to load:", trainer_state_path)
            with open(trainer_state_path, "rb") as trainer_file:
                trainer_state = pickle.load(trainer_file)

        if args.save_path:
            args.save_root = os.path.dirname(args.save_path)
            token = os.path.basename(os.path.abspath(args.save_path))
        else:
            token = get_token(args.save_root)

        # New training run.
        date = str(datetime.datetime.now())
        date = date.replace(" ", "_")
        codename = "%s_%s" % (token, date)
        settings_path = args.settings

        # Load settings dict into Config object.
        with open(settings_path, "r") as settings_file:
            settings = json.load(settings_file)
        config = Config(settings)

        # Construct a new ``save_dir`` in either case.
        if args.save_path:
            save_dir = args.save_path
        else:
            save_dir = os.path.join(args.save_root, codename)

        # Only allow saving to an existing directory if we are continuing training.
        if os.path.isdir(save_dir) and not args.load_from:
            raise ValueError(f"Save directory already exists: {save_dir}")
        os.makedirs(save_dir)

        # Construct log paths.
        env_log_filename = codename + "_env_log.txt"
        visual_log_filename = codename + "_visual_log.txt"
        metrics_log_filename = codename + "_metrics.txt"
        env_log_path = os.path.join(save_dir, env_log_filename)
        visual_log_path = os.path.join(save_dir, visual_log_filename)
        metrics_log_path = os.path.join(save_dir, metrics_log_filename)

        # If ``save_dir`` is not the same as ``load_from`` we must copy the existing logs
        # into the new save directory, then contine to append to them.
        if args.load_from and save_dir not in env_log_path:
            new_env_log_filename = codename + "_env_log.txt"
            new_visual_log_filename = codename + "_visual_log.txt"
            new_metrics_log_filename = codename + "_metrics.txt"
            new_env_log_path = os.path.join(save_dir, new_env_log_filename)
            new_visual_log_path = os.path.join(save_dir, new_visual_log_filename)
            new_metrics_log_path = os.path.join(save_dir, new_metrics_log_filename)
            shutil.copyfile(env_log_path, new_env_log_path)
            shutil.copyfile(visual_log_path, new_visual_log_path)
            shutil.copyfile(metrics_log_path, new_metrics_log_path)
            env_log_path = new_env_log_path
            visual_log_path = new_visual_log_path
            metrics_log_path = new_metrics_log_path

        # Open logs.
        env_log = open(env_log_path, "a+")
        visual_log = open(visual_log_path, "a+")
        metrics_log = open(metrics_log_path, "a+")

        # Load setup state.
        self.config: Config = config
        self.save_dir: str = save_dir
        self.codename: str = codename
        self.env_log: TextIO = env_log
        self.visual_log: TextIO = visual_log
        self.metrics_log: TextIO = metrics_log
        self.env_state_path: str = env_state_path
        self.trainer_state: Dict[str, Any] = trainer_state
