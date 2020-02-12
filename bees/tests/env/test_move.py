#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Tuple

import hypothesis.strategies as st
from hypothesis import given
from bees.tests import strategies as bst

# pylint: disable=no-value-for-parameter, protected-access


# TODO: Do we need to test the grid as well?
@given(st.data())
def test_move_correctly_modifies_agent_state(data: st.DataObject) -> None:
    """ Makes sure they actually move or STAY. """
    # TODO: Handle out-of-bounds errors.
    # TODO: Consider making the environment toroidal.
    env = data.draw(bst.envs())
    env.reset()
    old_locations: Dict[int, Tuple[int, int]] = {}
    for agent_id, agent in env.agents.items():
        old_locations[agent_id] = agent.pos
    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    executed_dict = env._move(tuple_action_dict)

    # TODO: Consider making ``env.LEFT``, etc tuples which can be added to existing
    # positions rather than just integers.
    for agent_id, action in executed_dict.items():
        agent = env.agents[agent_id]
        move = action[0]
        old_pos = old_locations[agent_id]
        if move == env.UP or move == env.DOWN:
            assert agent.pos[0] == old_pos[0]
        if move == env.LEFT or move == env.RIGHT:
            assert agent.pos[1] == old_pos[1]
        if move == env.UP:
            assert agent.pos[1] == old_pos[1] + 1
        if move == env.DOWN:
            assert agent.pos[1] == old_pos[1] - 1
        if move == env.RIGHT:
            assert agent.pos[0] == old_pos[0] + 1
        if move == env.LEFT:
            assert agent.pos[0] == old_pos[0] - 1
        if move == env.STAY:
            assert agent.pos == old_pos


@given(st.data())
def test_move_holds_other_actions_invariant(data: st.DataObject) -> None:
    """ Makes sure the returned action dict only modifies move subaction space. """
    env = data.draw(bst.envs())
    env.reset()

    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    executed_dict = env._move(tuple_action_dict)

    pairs = zip(list(tuple_action_dict.values()), list(executed_dict.values()))
    for attempted_action, executed_action in pairs:
        assert attempted_action[1:] == executed_action[1:]


@given(st.data())
def test_move_only_changes_to_stay(data: st.DataObject) -> None:
    """ Makes sure the returned action dict only changes to STAY if at all. """
    env = data.draw(bst.envs())
    env.reset()

    tuple_action_dict = data.draw(bst.tuple_action_dicts(env=env))
    executed_dict = env._move(tuple_action_dict)

    pairs = zip(list(tuple_action_dict.values()), list(executed_dict.values()))
    for attempted_action, executed_action in pairs:
        if attempted_action[0] != executed_action[0]:
            assert executed_action[0] == env.STAY
