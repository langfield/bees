""" Various functions for use in ``env.py``. """
import numpy as np

# pylint: disable=invalid-name
def one_hot(k: int, n: int) -> np.ndarray:
    """ Returns a one-hot vector of length n with a set bit of k """
    vec = np.zeros([n])
    vec[k] = 1
    return vec


def convert_obs_to_tuple(obs: np.ndarray):
    """ Convert an observation to a tuple. """

    outer_list = []
    for x in range(obs.shape[0]):

        inner_list = []
        for y in range(obs.shape[1]):

            # Convert floats to int
            point_obs = list(obs[x, y])
            for i, _ in enumerate(point_obs):
                point_obs[i] = int(point_obs[i])

            inner_list.append(tuple(point_obs))

        outer_list.append(tuple(inner_list))

    return tuple(outer_list)
