""" Test that ``Env.reset()`` works correctly. """
from itertools import product

from hypothesis import given

from bees.env import Env
from bees.tests import strategies
from bees.utils import DEBUG


# TODO: Everything.
@given(strategies.grid_positions.flatmap(strategies.envs())
def test_env_update_pos(env: Env) -> None:
    """ Tests that each observation has the corret number of each object type. """

    env.reset()
    valid_moves = [
        env.config.STAY,
        env.config.LEFT,
        env.config.RIGHT,
        env.config.UP,
        env.config.DOWN,
    ]
    raise NotImplementedError
