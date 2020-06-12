#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import product

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

from bees.env import Env
from bees.tests import strategies as bst
from bees.utils import timing

# pylint: disable=no-value-for-parameter, protected-access


class EnvironmentMachine(RuleBasedStateMachine):
    """ Finite-state machine for testing ``Env`` multi agent environment. """

    @timing
    @given(env=bst.envs())
    def __init__(self, env: Env):
        super(EnvironmentMachine, self).__init__()
        self.env = env

    @initialize()
    @timing
    def reset(self) -> None:
        env = self.env
        obs = env.reset()
        for agent_id, agent_obs in obs.items():

            # Calculate correct number of each object type.
            correct_obj_nums = {obj_type: 0 for obj_type in env.obj_type_ids.values()}
            for dx, dy in product(range(-env.sight_len, env.sight_len + 1), repeat=2):
                x = env.agents[agent_id].pos[0] + dx
                y = env.agents[agent_id].pos[1] + dy
                if (x, y) not in product(range(env.width), range(env.height)):
                    continue
                for obj_type in env.obj_type_ids.values():
                    correct_obj_nums[obj_type] += int(env.grid[x][y][obj_type])

            # Calculate number of each object type in returned observations.
            observed_obj_nums = {obj_type: 0 for obj_type in env.obj_type_ids.values()}
            for dx, dy in product(range(-env.sight_len, env.sight_len + 1), repeat=2):
                for obj_type in env.obj_type_ids.values():
                    observed_obj_nums[obj_type] += int(agent_obs[obj_type][dx][dy])

            assert correct_obj_nums == observed_obj_nums

    @rule()
    @timing
    @given(data=st.data())
    def update_pos(self, data: st.DataObject) -> None:
        pos = data.draw(bst.positions(env=self.env))
        move = data.draw(bst.moves(env=self.env))
        new_pos = self.env._update_pos(pos, move)

        if pos[0] != new_pos[0]:
            assert pos[1] == new_pos[1]
            assert abs(pos[0] - new_pos[0]) == 1
        if pos[1] != new_pos[1]:
            assert pos[0] == new_pos[0]
            assert abs(pos[1] - new_pos[1]) == 1

    @rule()
    def dummy(self) -> None:
        assert True


esm = EnvironmentMachine.TestCase
esm.settings = settings(max_examples=20, stateful_step_count=10, deadline=None)
