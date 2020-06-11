#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Print live training debug output and do reward analysis. """
import copy
from pprint import pformat
from typing import Dict, Any, Tuple, Set

import torch
import torch.nn.functional as F

from asta import Tensor, dims, typechecked

from bees.rl.algo.algo import Algo
from bees.env import Env
from bees.agent import Agent
from bees.config import Config

# pylint: disable=too-few-public-methods

N_ACTS = dims.N_ACTS


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

    def __eq__(self, other: object) -> bool:
        """ Comparison function for ``Metrics`` class. """

        if not isinstance(other, Metrics):
            raise NotImplementedError
        eq = True
        for key in self.__dict__:
            if getattr(self, key) != getattr(other, key):
                eq = False
                break
        return eq

    def get_summary(self) -> Dict[str, float]:
        """ Returns a summary of the current metric values. """

        summary = {}
        attrs = ["policy_score", "total_loss", "food_score"]
        for attr in attrs:
            summary[attr] = getattr(self, attr)

        edians = []
        NUM_EDIANS = 5
        sorted_scores = sorted(list(self.policy_scores.values()))
        for i in range(NUM_EDIANS):
            edian_index = int((i / (NUM_EDIANS - 1)) * len(sorted_scores))
            if edian_index == len(sorted_scores):
                edian_index -= 1
            if edian_index not in range(len(sorted_scores)):
                break

            edians.append(sorted_scores[edian_index])
        summary["edians"] = list(edians)

        return summary

    def __repr__(self) -> str:
        """ Return string representation of object. """

        # Try to use ``sort_dicts`` option, only available in Python 3.8.
        try:
            # pylint: disable=unexpected-keyword-arg
            formatted = pformat(self.get_summary(), sort_dicts=False)  # type: ignore
        except TypeError:
            formatted = pformat(self.get_summary())
        return formatted


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


def update_losses(
    env: Env,
    agents: Dict[int, Algo],
    config: Config,
    losses: Tuple[Dict[int, float], Dict[int, float], Dict[int, float]],
    metrics: Metrics,
    minted_agents: Set[int],
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
    for agent_id in set(agents.keys()) - minted_agents:
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
