""" Test that ``Env.reset()`` works correctly. """
from typing import Tuple
from hypothesis import given
from bees.env import Env
from bees.tests import strategies

@given(strategies.grid_positions())
def test_env_remove_without_id(remove_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests removal of an object without id (such as food). """

    # Env setup.
    env, pos = remove_args
    env.reset()
    obj_type_id = env.obj_type_ids["food"]
    env._place(obj_type_id, pos)

    # Remove from ``pos``.
    env._remove(obj_type_id, pos)

    grid_idx = pos + (obj_type_id,)
    assert env.grid[grid_idx] == 0


@given(strategies.grid_positions())
def test_env_remove_with_id(remove_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests removal of an object with id (such as an agent). """

    # Env setup.
    env, pos = remove_args
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]
    if not env._obj_exists(obj_type_id, pos):
        agent_id = env._new_agent_id()
        env._place(obj_type_id, pos, agent_id)
    else:
        x, y = pos
        agent_id = next(iter(env.id_map[x][y][obj_type_id]))

    # Remove from ``pos``.
    env._remove(obj_type_id, pos, agent_id)

    grid_idx = pos + (obj_type_id,)
    assert env.grid[grid_idx] == 0
