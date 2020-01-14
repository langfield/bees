""" Test that ``Env.reset()`` works correctly. """
from typing import Tuple
from hypothesis import given
from bees.env import Env
from bees.tests import strategies

# pylint: disable=no-value-for-parameter, protected-access


@given(strategies.grid_positions())
def test_env_place_without_id(place_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests placement of an object without id (such as food). """

    # Env setup.
    env, pos = place_args
    env.reset()
    obj_type_id = env.obj_type_ids["food"]
    if not env._obj_exists(obj_type_id, pos):
        env._place(obj_type_id, pos)

    grid_idx = pos + (obj_type_id,)
    assert env.grid[grid_idx] == 1


@given(strategies.grid_positions())
def test_env_place_with_id(place_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests placement of an object with id (such as an agent). """

    # Env setup.
    env, pos = place_args
    x, y = pos
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]
    if not env._obj_exists(obj_type_id, pos):
        agent_id = env._new_agent_id()
        env._place(obj_type_id, pos, agent_id)

        assert agent_id in env.id_map[x][y][obj_type_id]

    grid_idx = pos + (obj_type_id,)
    assert env.grid[grid_idx] == 1
