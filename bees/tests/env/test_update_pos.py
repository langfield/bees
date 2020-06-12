#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test that ``Env.reset()`` works correctly. """
from typing import Tuple

from hypothesis import given

from bees.env import Env
from bees.tests import strategies

# pylint: disable=no-value-for-parameter, protected-access


@given(strategies.grid_positions_and_moves())
def test_env_update_pos(env_and_pos_and_move: Tuple[Env, Tuple[int, int], int]) -> None:
    """ Tests that changes are always delta 1 and only one coordinate is changed. """
    env, pos, move = env_and_pos_and_move
    env.reset()
    new_pos = env._update_pos(pos, move)

    if pos[0] != new_pos[0]:
        assert pos[1] == new_pos[1]
        assert abs(pos[0] - new_pos[0]) == 1
    if pos[1] != new_pos[1]:
        assert pos[0] == new_pos[0]
        assert abs(pos[1] - new_pos[1]) == 1
