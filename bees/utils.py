""" Various functions for use in ``env.py``. """
import os
import datetime
from typing import Tuple

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
    with open("settings/google-10000-english.txt", "r", encoding="utf-8") as english:
        tokens = [word.rstrip() for word in english.readlines()]
    tokens.sort()
    tokens = [word for word in tokens if len(word) > 5]

    # TODO: Add warning when path doesn't exist.
    if not os.path.isdir(save_root):
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
