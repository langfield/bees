""" Various functions for use in ``env.py``. """
import os
import datetime
from typing import Tuple, Any

import numpy as np


def one_hot(k: int, dim: int) -> np.ndarray:
    """ Returns a one-hot vector of length dim with a set bit of k. """
    vec = np.zeros([dim])
    vec[k] = 1
    return vec


def convert_obs_to_tuple(obs: np.ndarray) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """ Convert an observation to a tuple. """

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

    return tuple(outer_list)


def get_logs() -> Tuple[Any, Any]:
    """
    Creates and returns log objects for repr and reward logs.
    """

    # HARDCODE
    with open("settings/google-10000-english.txt", "r", encoding="utf-8") as english:
        tokens = [word.rstrip() for word in english.readlines()]
    tokens.sort()
    tokens = [word for word in tokens if len(word) > 5]
    dirlist = os.listdir("logs/")
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
    date = str(datetime.datetime.now())
    date = date.replace(" ", "_")
    repr_log_path = "logs/%s_%s_repr_log.txt" % (token, date)
    rew_log_path = "logs/%s_%s_rew_log.txt" % (token, date)
    for log in [repr_log_path, rew_log_path]:
        log_dir = os.path.dirname(log)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    repr_log = open(repr_log_path, "a+")
    rew_log = open(rew_log_path, "a+")

    return repr_log, rew_log
