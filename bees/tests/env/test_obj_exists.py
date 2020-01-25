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
    pos: Tuple[int, int] = data.draw(st.tuples(st.integers(min_value=-100, max_value=100), st.integers(min_value=-100, max_value=100)))  # type: ignore
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
    """ Make sure the correct error is raised. """
    raise NotImplementedError
    env = data.draw(bst.envs())
    obj_type_id = data.draw(bst.obj_type_ids(env=env))
    pos = data.draw(bst.positions(env=env))
    grid_idx = pos + (obj_type_id,)
