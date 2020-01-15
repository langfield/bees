import unittest
from hypothesis.stateful import rule, initialize, Bundle, RuleBasedStateMachine
from bees.tests import strategies

class EnvironmentMachine(RuleBasedStateMachine):
    """ Finite-state machine for testing ``Env`` multi agent environment. """
    def __init__(self):
        super(EnvironmentMachine, self).__init__()

        # TODO: Figure out how to generate.
        self.env = strategies.envs()

    @initialize()
    def reset(self):
        self.env.reset()

    @rule()
    def dummy(self):
        assert True

TestEnv = EnvironmentMachine.TestCase

if __name__ == "__main__":
    unittest.main()
