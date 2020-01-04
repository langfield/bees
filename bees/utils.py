""" Various functions for use in ``env.py``. """
import os
import inspect
import argparse
from typing import Tuple, Any

import numpy as np


def one_hot(k: int, dim: int) -> np.ndarray:
    """
    Returns a one-hot vector of length dim with a set bit of k.

    Parameters
    ----------
    k : ``int``.
        Which bit to set in the one-hot vector.
    dim : ``int``.
        The dimension of the one-hot vector.

    Returns
    -------
    vec : ``np.ndarray``.
        The resultant vector.
        Shape: ``(dim,)``.
    """
    vec = np.zeros([dim])
    vec[k] = 1
    return vec


# UNUSED
def convert_obs_to_tuple(obs: np.ndarray) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """
    Convert an observation to a tuple.

    Parameters
    ----------
    obs : ``np.ndarray``.
        Observation from some grid position.
        Shape: ``(obs_len, obs_len, num_obj_types)``.

    Returns
    -------
    observation_tuple : ``Tuple[Tuple[Tuple[int, ...], ...], ...]``.
        The converted observation with the same shape.
    """
    # TODO: reconcile naming of obs in trainer and obs of a single agent.

    outer_list = []
    for x in range(obs.shape[0]):

        inner_list = []
        for y in range(obs.shape[1]):

            # Convert floats to int.
            point_obs = list(obs[x, y])
            for i, _ in enumerate(point_obs):
                point_obs[i] = int(point_obs[i])
            inner_list.append(tuple(point_obs))
        outer_list.append(tuple(inner_list))
    observation_tuple = tuple(outer_list)

    return observation_tuple


def get_token(save_root: str) -> str:
    """
    Creates and returns a new token for saving and loading runs.

    Returns
    -------
    token : ``str``.
        The training run token name.
    """

    # HARDCODE
    with open(
        "bees/settings/google-10000-english.txt", "r", encoding="utf-8"
    ) as english:
        tokens = [word.rstrip() for word in english.readlines()]
    tokens.sort()
    tokens = [word for word in tokens if len(word) > 5]

    if not os.path.isdir(save_root):
        print("Save root: '%s' does not exist. Creating directories." % save_root)
        os.makedirs(save_root)

    dirlist = os.listdir(save_root)
    token_idx = 0
    while 1:
        token = tokens[token_idx]
        already_used = False
        for filename in dirlist:
            if token in filename:
                already_used = True
                break
        if already_used:
            token_idx += 1
            continue
        break

    return token


def validate_args(args: argparse.Namespace) -> None:
    """ Validates ``args``. Will raise ValueError if invalid arguments are given. """

    # Check for settings file or loading path.
    if not args.settings and not args.load_from:
        raise ValueError("Must either provide argument --settings or --load-from.")

    # Validate paths.
    if args.load_from and not os.path.isdir(args.load_from):
        raise ValueError(
            "Invalid load directory for argument --load-from: '%s'." % args.load_from
        )
    if args.settings and not os.path.isfile(args.settings):
        raise ValueError(
            "Invalid settings file for argument --settings: '%s'." % args.settings
        )

    # Check for missing --settings argument.
    if args.load_from and not args.settings:
        print(
            "Warning: Argument --settings not provided, loading from '%s'."
            % args.load_from
        )


# pylint: disable=invalid-name
def DEBUG(var: Any) -> None:
    """
    Debugging tool to print a variable's name and value.

    Paramters
    ---------
    var : ``Any``.
        Any variable.
    """
    name = ""
    delimit = False
    string_repr = repr(var)
    lines = string_repr.split("\n")
    if len(lines) > 1 or len(string_repr) > 80:
        delimit = True

    found_name = False
    frame = inspect.currentframe()
    for key, val in frame.f_back.f_locals.items():
        if var is val:
            name = key
            found_name = True

    if not found_name:
        raise ValueError("DEBUG() was not able to find the name of the variable.")

    if delimit:
        print(
            "vvvvvvvvvv||VARIABLE NAME: '%s' | TYPE: '%s'||vvvvvvvvvv"
            % (name, type(var))
        )
        print(var)
        print(
            "^^^^^^^^^^||VARIABLE NAME: '%s' | TYPE: '%s'||^^^^^^^^^^"
            % (name, type(var))
        )
    else:
        print("'%s':" % name, var)
        print("Type of '%s':" % name, type(var))
