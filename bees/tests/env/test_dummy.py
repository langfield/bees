import json
import datetime
import unittest
from typing import Callable, Any, Tuple, Dict

import hypothesis.strategies as st
from hypothesis import HealthCheck as hc
from hypothesis import given, settings
from hypothesis.stateful import rule, initialize, Bundle, RuleBasedStateMachine

from bees.utils import timing
from bees.config import Config
from bees.env import Env

# pylint: disable=no-value-for-parameter


@st.composite
def envs(draw: Callable[[st.SearchStrategy], Any]) -> Env:
    """ A hypothesis strategy for generating ``Env`` objects. """

    sample: Dict[str, Any] = {}

    sample["width"] = draw(st.integers(min_value=1, max_value=9))
    sample["height"] = draw(st.integers(min_value=1, max_value=9))

    # Get variable names. It is important that the call to locals() stays at the top
    # of this function, before any other local variables are made.
    # arg_names = list(locals())

    # Read settings file for defaults.
    settings_path = "bees/settings/settings.json"
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)

    # Fill settings with values from arguments.
    for key, value in sample.items():
        settings[key] = value

    config = Config(settings)
    env = Env(config)

    return env


class EnvironmentMachine(RuleBasedStateMachine):
    """ Finite-state machine for testing ``Env`` multi agent environment. """

    @timing
    @settings(suppress_health_check=[hc.too_slow])
    @given(env=envs())
    def __init__(self, env: Env):
        super(EnvironmentMachine, self).__init__()
        self.env = env

    """
    @initialize()
    @timing
    def reset(self) -> None:
        pass
    """

    @rule()
    def dummy(self) -> None:
        assert True


env_state_machine = EnvironmentMachine.TestCase
env_state_machine.settings = settings(
    max_examples=1,
    stateful_step_count=1,
    deadline=None,
    suppress_health_check=[hc.too_slow],
)
