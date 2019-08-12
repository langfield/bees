import numpy as np

def one_hot(k: int, n: int) -> np.ndarray:
    """ Returns a one-hot vector of length n with a set bit of k """
    vec = np.zeros([n])
    vec[k] = 1
    return vec
