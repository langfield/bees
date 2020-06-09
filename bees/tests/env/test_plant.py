#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hypothesis.strategies as st
from hypothesis import given

from bees.env import Env
from bees.tests import strategies as bst

# pylint: disable=no-value-for-parameter, protected-access


# TODO: Can we even test anything else here (density calculation)?
@given(st.data())
def test_env_plant_doesnt_remove_food(data: st.DataObject) -> None:
    env: Env = data.draw(bst.envs())
    old_num_foods = env.num_foods
    env._plant()
    assert env.num_foods >= old_num_foods


@given(st.data())
def test_env_plant_does_nothing_when_zero_prob(data: st.DataObject) -> None:
    env: Env = data.draw(bst.envs())
    env.adaptive_food_type = None
    env.food_regen_prob = 0.0
    old_num_foods = env.num_foods
    env._plant()
    assert env.num_foods == old_num_foods


@given(st.data())
def test_env_plant_fills_grid_when_one_prob(data: st.DataObject) -> None:
    env: Env = data.draw(bst.envs())
    env.adaptive_food_type = None
    env.food_regen_prob = 1.0
    env._plant()
    assert env.num_foods == env.width * env.height
