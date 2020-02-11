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
            if 0 <= ob_pos[0] < env.width and 0 <= ob_pos[1] < env.height:
                print("START")
                print(ob[:, i, j])
                print(env.grid[ob_pos, :])
                print(env.grid.shape)
                print(ob_pos)
                assert np.all(ob[:, i, j] == env.grid[ob_pos, :])
            else:
                print("flaG")
                print(ob[:, i, j])
                assert np.all(ob[:, i, j] == np.zeros((env.num_obj_types,)))
