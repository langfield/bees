""" Test that ``Env.reset()`` works correctly. """
from typing import Tuple

from hypothesis import given

from bees.env import Env
from bees.analysis import update_food_scores, Metrics
from bees.tests import strategies

# pylint: disable=no-value-for-parameter, protected-access

# TODO: Implement this strategy.
@given(strategies.env_and_metrics())
def test_analysis_update_food_scores_returns_new_object(
    env_and_metrics: Tuple[Env, Metrics]
) -> None:
    """ Tests ``new_metric`` is a different object from ``metric``. """

    env, metrics = env_and_metrics
    env.reset()

    new_metrics = update_food_scores(env, metrics)

    assert id(metrics) != id(new_metrics)
