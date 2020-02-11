import numpy as np
import hypothesis.strategies as st
from hypothesis import given

from bees.tests import strategies as bst

# pylint: disable=no-value-for-parameter


@given(st.data())
def test_get_obs_has_correct_shape(data: st.DataObject) -> None:
    """ Make sure that a returned observation has the correct shape. """
    env = data.draw(bst.envs())
    env.reset()
    pos = data.draw(bst.positions(env=env))
    ob = env._get_obs(pos)
    assert ob.shape == env.observation_space.shape


@given(st.data())
def test_get_obs_has_correct_objects(data: st.DataObject) -> None:
    """ Make sure that a returned observation is accurate w.r.t. ``env.grid``. """
    env = data.draw(bst.envs())
    env.reset()
    pos = data.draw(bst.positions(env=env))
    ob = env._get_obs(pos)
    for i in range(ob.shape[1]):
        for j in range(ob.shape[2]):
            ob_pos = (pos[0] + i - env.sight_len, pos[1] + j - env.sight_len)
            print("=========================================")
            if 0 <= ob_pos[0] < env.width and 0 <= ob_pos[1] < env.height:
                print(f"Shape of 'ob': {ob.shape}")
                print(f"Shape of 'grid': {env.grid.shape}")
                ob_square = ob[:, i, j]
                env_square = env.grid[ob_pos]
                print(f"Shape of 'ob_square': {ob_square.shape}")
                print(f"Shape of 'env_square': {env_square.shape}")
                assert np.all(ob_square == env_square)
            else:
                assert np.all(ob[:, i, j] == np.zeros((env.num_obj_types,)))
