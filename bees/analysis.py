""" Print live training debug output and do reward analysis. """
from typing import Dict, Any, Tuple
import copy

import numpy as np
import torch
import torch.nn.functional as F

from bees.env import Env
from bees.config import Config
from bees.utils import flat_action_to_tuple

# pylint: disable=too-few-public-methods


class Metrics:
    """ Struct-like object to hold metric values for analysis of training. """

    def __init__(self) -> None:
        """ __init__ function for metrics class. """

        self.policy_scores: Dict[int, float] = {}
        self.value_losses: Dict[int, float] = {}
        self.action_losses: Dict[int, float] = {}
        self.dist_entropies: Dict[int, float] = {}
        self.total_losses: Dict[int, float] = {}
        self.policy_score: float = float("inf")
        self.value_loss: float = float("inf")
        self.action_loss: float = float("inf")
        self.dist_entropy: float = float("inf")
        self.total_loss: float = float("inf")
        self.initial_policy_score = float("inf")

        # Task specific score (for now, zero-to-food).
        self.food_scores: Dict[int, float] = {}
        self.food_score: float = float("inf")


def aggregate_loss(env: Env, losses: Dict[int, float]) -> float:
    """
    Aggregates loss data over all agents, by taking a weighted average of the values
    in ``losses``, weighted by the age of the corresponding agent in ``env``.
    """

    # Get all agent ages and compute their sum.
    ages = {agent_id: agent.age for agent_id, agent in env.agents.items()}
    age_sum = sum(ages.values())

    # Normalize agent ages.
    normalized_ages: Dict[int, float] = {}
    for agent_id, age in ages.items():
        normalized_ages[agent_id] = age / age_sum

    # Get only agents which are alive and have losses computed.
    valid_ids = set(env.agents.keys()).intersection(set(losses.keys()))

    # Sum the weighted losses.
    loss: float = 0.0
    for agent_id in valid_ids:
        loss += losses[agent_id] * normalized_ages[agent_id]

    return loss


def update_policy_score(
    env: Env,
    config: Config,
    infos: Dict[int, Any],
    agent_action_dists: Dict[int, torch.Tensor],
    metrics: Metrics,
) -> Metrics:
    """
    Computes updated policy score metrics from agent actions.

    Parameters
    ----------
    env : ``Env``.
        Training environment.
    config : ``config``.
        Training configuration.
    infos : ``Dict[int, Any]``.
        Returned per-agent information from the ``env.step()`` function.
    agent_action_dists : ``Dict[int, torch.Tensor]``.
        The policy distributions over actions for all agents.
    metrics : ``Metrics``.
        The state of the training analysis metrics. Not mutated.

    Returns
    -------
    new_metrics : ``Metrics``.
        Updated version of ``metrics``. This is not the same object.
    """

    new_metrics = copy.deepcopy(metrics)

    # Update policy score estimates.
    valid_ids = set(env.agents.keys()).intersection(set(agent_action_dists.keys()))
    for agent_id in valid_ids:
        optimal_action_dist = infos[agent_id]["optimal_action_dist"]
        agent_action_dist = agent_action_dists[agent_id]
        agent_action_dist = agent_action_dist.cpu()

        timestep_score = float(
            F.kl_div(torch.log(agent_action_dist), optimal_action_dist)
        )

        # Update policy score with exponential moving average.
        if agent_id in metrics.policy_scores:
            new_metrics.policy_scores[agent_id] = (
                config.ema_alpha * metrics.policy_scores[agent_id]
                + (1.0 - config.ema_alpha) * timestep_score
            )
        else:
            new_metrics.policy_scores[agent_id] = timestep_score

    # Compute aggregate policy score across all agents (weighted average by age).
    new_metrics.policy_score = aggregate_loss(env, new_metrics.policy_scores)

    # Set initial policy score, if necessary.
    if new_metrics.policy_score != float("inf") and metrics.policy_score == float(
        "inf"
    ):
        new_metrics.initial_policy_score = new_metrics.policy_score

    return new_metrics


def update_losses(
    env: Env,
    config: Config,
    losses: Tuple[Dict[int, float], Dict[int, float], Dict[int, float]],
    metrics: Metrics,
) -> Metrics:
    """
    Computes updated loss metrics from training losses.

    Parameters
    ----------
    env : ``Env``.
        Training environment.
    config : ``config``.
        Training configuration.
    losses : ``Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]``.
        Losses returned from call to ``agent.update()`` in trainer.py.
    metrics : ``Metrics``.
        The state of the training analysis metrics. Not mutated.

    Returns
    -------
    new_metrics : ``Metrics``.
        Updated version of ``metrics``. This is not the same object.
    """

    new_metrics = copy.deepcopy(metrics)

    # Store training losses in ``new_metrics``.
    new_metrics.value_losses = dict(losses[0])
    new_metrics.action_losses = dict(losses[1])
    new_metrics.dist_entropies = dict(losses[2])

    # Compute total loss as a function of training losses.
    new_metrics.total_losses = {}
    for agent_id in env.agents:
        new_metrics.total_losses[agent_id] = (
            new_metrics.value_losses[agent_id] * config.value_loss_coef
            + new_metrics.action_losses[agent_id]
            - new_metrics.dist_entropies[agent_id] * config.entropy_coef
        )

    # Compute aggregate losses over all agents (weighted average by age).
    new_metrics.value_loss = aggregate_loss(env, new_metrics.value_losses)
    new_metrics.action_loss = aggregate_loss(env, new_metrics.action_losses)
    new_metrics.dist_entropy = aggregate_loss(env, new_metrics.dist_entropies)
    new_metrics.total_loss = aggregate_loss(env, new_metrics.total_losses)

    return new_metrics


def update_food_scores(env: Env, metrics: Metrics) -> Metrics:
    """
    Computes update food scores from agent reward networks.

    Parameters
    ----------
    env : ``Env``.
        Training environment.
    metrics : ``Metrics``.
        The state of the training analysis metrics. Not mutated.

    Returns
    -------
    new_metrics : ``Metrics``.
        Updated version of ``metrics``. This is not the same object.
    """

    new_metrics = copy.deepcopy(metrics)
    previous_agent_ids = set(new_metrics.food_scores.keys())

    # Compute individual food scores for any new agents.
    # TODO: Fix this inefficiency.
    # Calling ``env.get_optimal_action_dists()`` is pretty inefficient, because this
    # function will compute the optimal action distribution for each agent, even though
    # we only need it for newly born agents.
    optimal_action_dists = env.get_optimal_action_dists()
    target_dist = get_food_target_dist(env)
    for agent_id in env.agents:
        if agent_id not in new_metrics.food_scores:
            # TODO: Compute KL-divergence with non-greedy optimal distribution.
            # Currently, since the KL-divergence isn't defined when q(x) = 0 and
            # p(x) != 0 for some x, we can't use the argmax optimal distribution. But
            # the two computations (policy score loss and food score loss) should be
            # decoupled, so that we can use the non-greedy version here and the greedy
            # version for the policy score loss, if we choose.
            new_metrics.food_scores[agent_id] = float(
                F.kl_div(torch.log(optimal_action_dists[agent_id]), target_dist)
            )

    # Remove food scores for any agents that have died.
    intermediate_agent_ids = new_metrics.food_scores.keys()
    for agent_id in intermediate_agent_ids:
        if agent_id not in env.agents:
            del new_metrics.food_scores[agent_id]

    # Recompute average food scores, if relevant agents have changed.
    current_agent_ids = set(new_metrics.food_scores.keys())
    if current_agent_ids != previous_agent_ids:
        new_metrics.food_score = np.mean(list(new_metrics.food_scores.values()))

    return new_metrics


def get_food_target_dist(env: Env) -> torch.Tensor:
    """
    Computes target action distribution for the zero-to-food task. This distribution is
    uniform over all actions which involve the 'eat' subaction, and zero everywhere
    else.

    Parameters
    ----------
    env : ``Env``.
        Training environment.

    Returns
    -------
    target_dist : ``torch.Tensor``.
        Optimal action distribution for the zero-to-food task.
    """

    # HARDCODE
    EAT_INDEX = 1
    target_dist = torch.zeros((env.num_actions,))
    num_eat_actions = 0

    # Count actions that involve eating, and set values in ``target_dist``.
    for flat_action in range(env.num_actions):
        action_tuple = flat_action_to_tuple(flat_action, env.subaction_sizes)

        if action_tuple[EAT_INDEX] == 1:
            target_dist[flat_action] = 1.0
            num_eat_actions += 1

    # Normalize values in ``target_dist``.
    target_dist /= num_eat_actions

    return target_dist
