import numpy as np


def one_hot(k: int, n: int) -> np.ndarray:
    """ Returns a one-hot vector of length n with a set bit of k """
    vec = np.zeros([n])
    vec[k] = 1
    return vec

def convert_obs_to_tuple(obs: np.ndarray):

    outerList = []
    for x in range(obs.shape[0]):

        innerList = []
        for y in range(obs.shape[1]):

            # Convert floats to int
            pointObs = obs[x, y]
            for c in range(pointObs.shape[0]):
                pointObs[c] = int(pointObs[c])

            innerList.append(tuple(pointObs))

        outerList.append(tuple(innerList))

    return tuple(outerList)

