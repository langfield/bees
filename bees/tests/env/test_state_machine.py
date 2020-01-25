import datetime
import unittest
from typing import Callable, Any, Tuple

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.stateful import rule, initialize, Bundle, RuleBasedStateMachine

from bees.env import Env
from bees.utils import timing
from bees.tests import strategies

# pylint: disable=no-value-for-parameter


@st.composite
def positions(draw: Callable[[st.SearchStrategy], Any], env: Env) -> Tuple[int, int]:
    pos: Tuple[int, int] = draw(
        st.tuples(
            st.integers(min_value=0, max_value=env.width - 1),
            st.integers(min_value=0, max_value=env.height - 1),
        )
    )
    return pos


@st.composite
def moves(draw: Callable[[st.SearchStrategy], Any], env: Env) -> int:
    valid_moves = [
        env.config.STAY,
        env.config.LEFT,
        env.config.RIGHT,
        env.config.UP,
        env.config.DOWN,
    ]
    move: int = draw(st.sampled_from(valid_moves))
    return move


class EnvironmentMachine(RuleBasedStateMachine):
    """ Finite-state machine for testing ``Env`` multi agent environment. """

    @timing
    @given(env=strategies.envs())
    def __init__(self, env: Env):
        super(EnvironmentMachine, self).__init__()
        self.env = env

    @initialize()
    @timing
    def reset(self) -> None:
        self.env.reset()

    @rule()
    @timing
    @given(data=st.data())
    def update_pos(self, data: st.DataObject) -> None:
        pos = data.draw(positions(env=self.env))
        move = data.draw(moves(env=self.env))
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
