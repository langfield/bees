""" Various functions for use in ``env.py``. """
import os
import inspect
import argparse
from typing import List, Tuple, Any
import functools

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
    for key, val in frame.f_back.f_locals.items():  # type: ignore
        if var is val:
            name = key
            found_name = True

    if not found_name:
        raise ValueError("DEBUG() was not able to find the name of the variable.")

    def printattrs(var: Any, name: str) -> None:
        """ Print those attrs. """
        if hasattr(var, "shape"):
            print("||Shape for '%s':" % name, var.shape)
        if hasattr(var, "device"):
            print("||Device for '%s':" % name, var.device)

    if delimit:
        print(
            "vvvvvvvvvv||VARIABLE NAME: '%s' | TYPE: '%s'||vvvvvvvvvv"
            % (name, type(var))
        )
        print(var)
        printattrs(var, name)
        print(
            "^^^^^^^^^^||VARIABLE NAME: '%s' | TYPE: '%s'||^^^^^^^^^^"
            % (name, type(var))
        )
    else:
        print("'%s':" % name, var)
        print("Type of '%s':" % name, type(var))
        printattrs(var, name)


def flat_action_to_tuple(
    flat_action: int, subaction_sizes: List[int]
) -> Tuple[int, ...]:
    """
    Converts a flat action to a tuple action. Example: If action space is made up of
    three categorical action spaces of sizes (5, 2, 2), then a flat action is an
    int between 0 and 20 (5 * 2 * 2). 0 will be converted to (0, 0, 0), 1 will be
    converted to (0, 0, 1), 4 will be converted to (1, 0, 0), 19 will be converted to
    (4, 1, 1).

    Parameters
    ----------
    flat_action : ``int``.
        Integer index of action sampled from a categorical distribution.
    subaction_sizes : ``Tuple[int, ...]``.
        Sizes of subaction spaces whose product makes up the action space.

    Returns
    -------
    action_tuple : ``Tuple[int, ...]``.
        Representation of ``flat_action`` in the equivalent tuple action space.
    """

    current_flat_action = flat_action
    action_list = []

    for i, _subaction_size in enumerate(subaction_sizes):
        if i == len(subaction_sizes) - 1:
            subspace_size = 1
        else:
            subspace_size = functools.reduce(
                lambda a, b: a * b, subaction_sizes[i + 1 :]
            )

        action_list.append(current_flat_action // subspace_size)
        current_flat_action = current_flat_action % subspace_size

    action_tuple = tuple(action_list)
    return action_tuple
