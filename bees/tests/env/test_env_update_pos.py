""" Test that ``Env.reset()`` works correctly. """
from hypothesis import given

from bees.env import Env


# TODO: Everything.
@given()
def env_update_pos(env: Env) -> None:
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
