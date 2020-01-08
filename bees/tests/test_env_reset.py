""" Test that ``Env.reset()`` works correctly. """
from hypothesis import given
from bees.env import Env
from bees.tests import strategies


# TODO: Everything.
@given(strategies.envs())
def test_env_reset_places_correct_number_of_agents(env: Env) -> None:
    """ Tests that we find a place for each agent in ``self.agents``. """
    num_grid_agents = 0
    env.fill()
    for row in env.grid:
        for square in row:
            if square[env.obj_type_ids["agent"]] == 1:
                num_grid_agents += 1
    assert num_grid_agents == len(env.agents)
