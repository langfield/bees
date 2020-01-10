""" Print live training debug output and do reward analysis. """
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F

from bees.env import Env
from bees.config import Config


def aggregate_loss(env: Env, losses: Dict[int, float]) -> float:
    """ Aggregates loss data over all agents. """

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
    loss = 0
    for agent_id in valid_ids:
        loss += losses[agent_id] * normalized_ages[agent_id]

    return loss


def live_analysis(
    env: Env,
    config: Config,
    infos: Dict[int, Any],
    agents: Dict[int, "AgentAlgo"],
    policy_scores: Dict[int, float],
    value_losses: Dict[int, float],
    action_losses: Dict[int, float],
    dist_entropies: Dict[int, float],
    agent_action_dists: Dict[int, torch.Tensor],
    first_policy_score_loss: float,
    policy_score_loss: float,
    loss: float,
) -> Tuple[Dict[int, float], float]:
    """ TODO: Everything here. """

    updated_policy_scores: Dict[int, float] = policy_scores.copy()
    updated_loss: float = loss
    end = "\r" if not config.print_repr else "\n"

    # NOTE: we assume ``config.num_processes`` is ``1``.
    # Update policy score estimates.
    if env.iteration % config.policy_score_frequency == 0:
        for agent_id in env.agents:
            optimal_action_dist = infos[agent_id]["optimal_action_dist"]
            agent_action_dist = agent_action_dists[agent_id]
            agent_action_dist = agent_action_dist.cpu()

            timestep_score = float(
                F.kl_div(torch.log(agent_action_dist), optimal_action_dist)
            )

            # Update policy score with exponential moving average.
            if agent_id in policy_scores:
                updated_policy_scores[agent_id] = (
                    config.ema_alpha * policy_scores[agent_id]
                    + (1.0 - config.ema_alpha) * timestep_score
                )
            else:
                updated_policy_scores[agent_id] = timestep_score

        # For this sum, we iterate over ``env.agents``, not ``agents``. This
        # is because env.agents has been updated in the call to env.step(), so
        # that if any agents die during that step, they are removed from
        # ``env.agents``. But they aren't removed from ``agents`` until after
        # we compute the policy score loss, so we use env.agents instead.

        # Compute policy score loss, and a weighted average of RL training losses.
        policy_score_loss = aggregate_loss(env, policy_scores)
        agg_action_loss = aggregate_loss(env, action_losses)
        agg_value_loss = aggregate_loss(env, value_losses)
        agg_dist_entropy = aggregate_loss(env, dist_entropies)
        updated_loss = (
            agg_value_loss * config.value_loss_coef
            + agg_action_loss
            - agg_dist_entropy * config.entropy_coef
        )

    agg_value_loss = 0
    agg_action_loss = 0
    agg_dist_entropy = 0
    print("Iteration: %d| " % env.iteration, end="")
    print("Num agents: %d| " % len(agents), end="")
    print(
        "Policy score loss: %.6f/%.6f| "
        % (policy_score_loss, first_policy_score_loss),
        end="",
    )
    print("Losses (action, value, entropy, total): ", end="")
    print(
        "%.6f, %.6f, %.6f, %.6f|||||"
        % (agg_action_loss, agg_value_loss, agg_dist_entropy, updated_loss),
        end=end,
    )

    return updated_policy_scores, policy_score_loss, updated_loss
