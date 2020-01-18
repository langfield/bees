import unittest
from hypothesis import given, settings
from hypothesis.stateful import rule, initialize, Bundle, RuleBasedStateMachine

from bees.env import Env
from bees.utils import timing
from bees.tests import strategies

# pylint: disable=no-value-for-parameter


class EnvironmentMachine(RuleBasedStateMachine):
    """ Finite-state machine for testing ``Env`` multi agent environment. """

    @timing
    @given(env=strategies.envs())
    def __init__(self, env: Env):
        super(EnvironmentMachine, self).__init__()
        self.env = env

    @timing
    @initialize()
    def reset(self) -> None:
        self.env.reset()

    @rule()
    def dummy(self) -> None:
        assert True


env_state_machine = EnvironmentMachine.TestCase
env_state_machine.settings = settings(max_examples=10, stateful_step_count=10)

if __name__ == "__main__":
    unittest.main()
