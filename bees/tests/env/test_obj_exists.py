from typing import Tuple
import hypothesis.strategies as st
from hypothesis import given

import bees.tests.strategies as bst

# pylint: disable=no-value-for-parameter


@given(data=st.data())
def test_obj_exists_handles_out_of_grid_positions(data: st.DataObject) -> None:
    """ Make sure the correct error is raised. """
    raised_index_error = False
    raised_value_error = False
    index_error = None
    value_error = None
    env = data.draw(bst.envs())
    obj_type_id = data.draw(bst.obj_type_ids(env=env))
    pos_indices = st.integers(min_value=-100, max_value=100)
    pos_strategy = st.tuples(pos_indices, pos_indices)
    pos: Tuple[int, int] = data.draw(pos_strategy)  # type: ignore
    try:
        existence = env._obj_exists(obj_type_id, pos)
    except IndexError as err:
        raised_index_error = True
        index_error = err
    except ValueError as err:
        raised_value_error = True
        value_error = err

    # If pos in in the grid, should raise no errors.
    if 0 <= pos[0] < env.width and 0 <= pos[1] < env.height:
        try:
            assert not raised_index_error
        except AssertionError:
            raise index_error  # type: ignore

    # If negative, should catch and raise a ValueError.
    elif pos[0] < 0 or pos[1] < 0:
        assert raised_value_error

    # Otherwise, currently raises an IndexError.
    # TODO: Add an error message for this?
    else:
        assert raised_index_error


@given(data=st.data())
def test_obj_exists_handles_invalid_obj_type_ids(data: st.DataObject) -> None:
    """ Make sure the correct error is raised. """
    env = data.draw(bst.envs())
    obj_type_id = data.draw(st.integers(min_value=-10, max_value=10))
    pos = data.draw(bst.positions(env=env))
    raised_value_error = False
    try:
        existence = env._obj_exists(obj_type_id, pos)
    except ValueError as err:
        raised_value_error = True
        value_error = err

    # If the obj_type_id is invalid, should raise ValueError.
    if obj_type_id not in env.obj_type_names:
        assert raised_value_error

    # Otherwise, should raise no error.
    else:
        try:
            assert not raised_value_error
        except AssertionError:
            raise value_error


@given(data=st.data())
def test_obj_exists_detects_bad_id_map(data: st.DataObject) -> None:
    """ Make sure error is raised when ``id_map`` disagrees with ``grid``. """
    env = data.draw(bst.envs())
    obj_type_id = env.obj_type_ids["agent"]
    pos = data.draw(bst.positions(env=env))
    x = pos[0]
    y = pos[1]
    grid_idx = pos + (obj_type_id,)
    singleton_id_set = set([data.draw(st.integers(min_value=0, max_value=3))])
    env.id_map[x][y][obj_type_id] = singleton_id_set
    raised_value_error = False
    try:
        existence = env._obj_exists(obj_type_id, pos)
    except ValueError as err:
        raised_value_error = True
        value_error = err
    if env.grid[grid_idx] == 0:
        assert raised_value_error
    else:
        assert not raised_value_error


@given(data=st.data())
def test_obj_exists_has_correct_num_agents(data: st.DataObject) -> None:
    """ Make sure iterating over grid with obj_exists sees all agents. """
    env = data.draw(bst.envs())
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]

    num_agents = 0
    for x in range(env.width):
        for y in range(env.height):
            if env._obj_exists(obj_type_id, (x, y)):
                num_agents += 1
    assert num_agents == len(env.agents)


@given(data=st.data())
def test_obj_exists_has_correct_num_foods(data: st.DataObject) -> None:
    """ Make sure iterating over grid with obj_exists sees all food. """
    env = data.draw(bst.envs())
    env.reset()
    obj_type_id = env.obj_type_ids["food"]

    num_food = 0
    for x in range(env.width):
        for y in range(env.height):
            if env._obj_exists(obj_type_id, (x, y)):
                num_food += 1
    assert num_food == env.num_foods


@given(data=st.data())
def test_obj_exists_marks_grid_squares_correctly(data: st.DataObject) -> None:
    """ Make sure emptiness of grid matches function output. """
    env = data.draw(bst.envs())
    env.reset()
    obj_type_id = data.draw(bst.obj_type_ids(env=env))

    for x in range(env.width):
        for y in range(env.height):
            if env.grid[x][y][obj_type_id] == 0:
                assert not env._obj_exists(obj_type_id, (x, y))
            else:
                assert env._obj_exists(obj_type_id, (x, y))


@given(data=st.data())
def test_obj_exists_marks_id_map_emptiness_correctly(data: st.DataObject) -> None:
    """ Make sure emptiness of grid matches function output. """
    env = data.draw(bst.envs())
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]

    for x in range(env.width):
        for y in range(env.height):
            if env.id_map[x][y][obj_type_id]:
                assert env._obj_exists(obj_type_id, (x, y))
            else:
                assert not env._obj_exists(obj_type_id, (x, y))
