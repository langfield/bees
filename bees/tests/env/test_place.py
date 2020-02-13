#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test that ``Env.reset()`` works correctly. """
from typing import Tuple

import pytest
import hypothesis.strategies as st
from hypothesis import given

from bees.env import Env
from bees.tests import strategies

# pylint: disable=no-value-for-parameter, protected-access

# TODO: Convert to ``@given(st.data())``.
@given(strategies.grid_positions())
def test_env_place_without_id(place_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests placement of an object without id (such as food). """
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
    env, pos = place_args
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]
    if not env._obj_exists(obj_type_id, pos):
        agent_id = env._new_agent_id()
        env._place(obj_type_id, pos, agent_id)
    grid_idx = pos + (obj_type_id,)
    assert env.grid[grid_idx] == 1


@given(strategies.grid_positions())
def test_env_place_with_id_requires_id(place_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests that env gets angry if you don't pass an object id for het object. """
    env, pos = place_args
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]
    if not env._obj_exists(obj_type_id, pos):
        with pytest.raises(TypeError):
            env._place(obj_type_id, pos)


@given(strategies.grid_positions())
def test_env_place_no_double_place(place_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests that env gets angry if you try to double up het objs with same id. """
    env, pos = place_args
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]
    if not env._obj_exists(obj_type_id, pos):
        agent_id = env._new_agent_id()
        env._place(obj_type_id, pos, agent_id)
        with pytest.raises(ValueError):
            env._place(obj_type_id, pos, agent_id)


@given(st.data())
def test_env_place_no_double_place_homo(data: st.DataObject) -> None:
    """ Tests that env gets angry if you try to double up homo objs. """
    env = data.draw(strategies.envs())
    pos = data.draw(strategies.positions(env=env))
    env.reset()
    homo_obj_type_ids = set(env.obj_type_ids.values()) - env.heterogeneous_obj_type_ids
    obj_type_id = data.draw(st.sampled_from(list(homo_obj_type_ids)))

    if not env._obj_exists(obj_type_id, pos):
        env._place(obj_type_id, pos)
        with pytest.raises(ValueError):
            env._place(obj_type_id, pos)


@given(strategies.grid_positions())
def test_env_place_with_id_in_id_map(place_args: Tuple[Env, Tuple[int, int]]) -> None:
    """ Tests placement of an object with id (such as an agent). """
    env, pos = place_args
    x, y = pos
    env.reset()
    obj_type_id = env.obj_type_ids["agent"]
    if not env._obj_exists(obj_type_id, pos):
        agent_id = env._new_agent_id()
        env._place(obj_type_id, pos, agent_id)
        assert agent_id in env.id_map[x][y][obj_type_id]
