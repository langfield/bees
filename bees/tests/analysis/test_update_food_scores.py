""" Test that ``Env.reset()`` works correctly. """
from typing import Dict, Tuple
from math import log

import numpy as np
from scipy.special import softmax
from hypothesis import given

from bees.env import Env
from bees.config import Config
from bees.analysis import update_food_scores, Metrics
from bees.utils import flat_action_to_tuple
from bees.tests import strategies
from bees.tests.test_utils import get_default_settings

# pylint: disable=no-value-for-parameter, protected-access


@given(strategies.env_and_metrics())
def test_analysis_update_food_scores_returns_new_object(
    env_and_metrics: Tuple[Env, Metrics]
) -> None:
    """ Tests ``new_metric`` is a different object from ``metric``. """

    env, metrics = env_and_metrics
    env.reset()

    new_metrics = update_food_scores(env, metrics)

    assert id(metrics) != id(new_metrics)


@given(strategies.env_and_metrics())
def test_analysis_update_food_scores_computes_mean_corectly(
    env_and_metrics: Tuple[Env, Metrics]
) -> None:
    """
    Tests ``new_metrics.food_score`` is the mean of the values in
    ``new_metrics.food_scores``.
    """

    env, metrics = env_and_metrics
    env.reset()

    new_metrics = update_food_scores(env, metrics)

    assert new_metrics.food_score == np.mean(list(new_metrics.food_scores.values()))


def test_analysis_update_food_scores_computes_uniform_dist_correctly() -> None:
    """
    Tests that ``new_metrics.food_scores`` is computed correctly when the reward
    network has a single layer, only actions as reward inputs, outputs zero as a
    reward for any action, and there is only a single agent in the environment.
    """

    # Get settings and create environment.
    settings = get_default_settings()
    assert settings["n_layers"] == 1
    assert settings["num_agents"] == 1
    assert settings["reward_inputs"] == ["actions"]
    config = Config(settings)
    env = Env(config)
    env.reset()

    # Set reward weights and biases.
    weight_shape = env.agents[0].reward_weights[0].shape
    env.agents[0].reward_weights[0] = np.zeros(weight_shape)

    # Compute expected food score.
    num_correct_actions = env.num_actions / 2.0
    expected_food_score = num_correct_actions / env.num_actions

    # Compare expected vs. actual.
    metrics = Metrics()
    new_metrics = update_food_scores(env, metrics)
    np.testing.assert_almost_equal(new_metrics.food_score, expected_food_score)


@given(strategies.envs())
def test_analysis_update_food_scores_computes_scores_correctly(env: Env) -> None:
    """
    Tests that ``new_metrics.food_scores`` is computed correctly when the reward
    network has a single layer, and only actions as inputs.
    """

    # Janky fix because wasn't sure how to modify strategies.envs() to constrain
    # ``n_layers`` to 1 and ``reward_inputs`` to ["actions"].
    if env.n_layers != 1 or env.reward_inputs != ["actions"]:
        return
    env.reset()

    # Compute expected food score for each agent.
    expected_food_scores: Dict[int, float] = {}
    for agent_id in env.agents:
        optimal_dist = softmax(env.agents[agent_id].reward_weights[0])

        correct_actions = []
        EAT_INDEX = 1
        for action in range(env.num_actions):
            tuple_action = flat_action_to_tuple(action, env.subaction_sizes)
            if tuple_action[EAT_INDEX] == env.config.EAT:
                correct_actions.append(action)

        expected_food_scores[agent_id] = 0.0
        for action in correct_actions:
            expected_food_scores[agent_id] += optimal_dist[action]

    expected_food_score = np.mean(list(expected_food_scores.values()))

    # Compare expected vs. actual.
    metrics = Metrics()
    new_metrics = update_food_scores(env, metrics)
    np.testing.assert_almost_equal(new_metrics.food_score, expected_food_score)
