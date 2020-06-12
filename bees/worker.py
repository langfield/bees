#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Distributed training function for a single agent worker. """
import copy
from typing import Any, Dict, Tuple, Optional
from multiprocessing.connection import Connection

import numpy as np
import torch
import torch.nn.functional as F
from asta import Array, Tensor, dims, shapes, typechecked

from bees.rl import utils
from bees.config import Config
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

# pylint: disable=duplicate-code

STOP_FLAG = 999

N_ACTS = dims.N_ACTS
FloatTensor = Tensor[float]

# TODO: Consider using Ray for multiprocessing, which is supposedly around 10x faster.


@typechecked
def get_policy_score(action_dist: Tensor[1, N_ACTS], info: Dict[str, Any]) -> float:
    """ Compute the policy score given current and optimal distributions. """
    optimal_action_dist = info["optimal_action_dist"]
    action_dist = action_dist.cpu()
    timestep_score = float(
        F.kl_div(torch.log(action_dist), optimal_action_dist, reduction="sum")
    )

    return timestep_score


@typechecked
def get_masks(
    done: bool, info: Dict[str, Any]
) -> Tuple[FloatTensor[1, 1], FloatTensor[1, 1]]:
    """ Compute masks to insert into ``rollouts``. """
    # If done then clean the history of observations.
    if done:
        masks = torch.FloatTensor([[0.0]])
    else:
        masks = torch.FloatTensor([[1.0]])
    if "bad_transition" in info.keys():
        bad_masks = torch.FloatTensor([[0.0]])
    else:
        bad_masks = torch.FloatTensor([[1.0]])

    return masks, bad_masks


# TODO: Consider adding ``age`` as an attribute of ``agent: Algo``.
@typechecked
def act(
    iteration: int,
    decay: bool,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
    age: int,
    action_funnel: Optional[Connection],
) -> Tuple[
    FloatTensor[1, 1],
    Tensor[torch.int64, 1, 1],
    FloatTensor[1, 1],
    FloatTensor,
    FloatTensor[1, N_ACTS],
]:
    """ Make a forward pass and send the env action to the leader process. """
    # Should execute only when trainer would make an update/backward pass.
    if decay:
        min_agent_lifetime = 1.0 / config.aging_rate

        # Decrease learning rate linearly.
        learning_rate = utils.update_linear_schedule(
            agent.optimizer,
            age,
            min_agent_lifetime,
            agent.optimizer.lr if config.algo == "acktr" else config.lr,
            config.min_lr,
        )
        agent.lr = learning_rate

    # Rollout tensors have dimension ``0`` size of ``config.num_steps``.
    rollout_index = iteration % config.num_steps
    with torch.no_grad():
        act_returns = agent.actor_critic.act(
            rollouts.obs[rollout_index],
            rollouts.recurrent_hidden_states[rollout_index],
            rollouts.masks[rollout_index],
        )

        # Get integer action to pass to ``env.step()``.
        env_action: int = int(act_returns[1][0])

    # Send ``env_action: int`` back to leader to execute step.
    if config.mp and action_funnel:
        action_funnel.send(env_action)

    return act_returns


def worker_loop(
    device: torch.device,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
    initial_age: int,
    initial_iteration: int,
    initial_ob: np.ndarray,
    env_spout: Connection,
    action_funnel: Connection,
    action_dist_funnel: Connection,
    loss_funnel: Connection,
    save_funnel: Connection,
) -> None:
    """ Training loop for a single agent worker. """
    age: int = initial_age
    iteration: int = initial_iteration

    # Copy first observations to rollouts, and send to device.
    initial_observation: torch.Tensor = torch.FloatTensor([initial_ob])
    rollouts.obs[0].copy_(initial_observation)
    rollouts.to(device)

    decay: bool = config.use_linear_lr_decay

    # Initial forward pass.
    fwds = act(iteration, decay, agent, rollouts, config, age, action_funnel)

    while True:

        # These are all CUDA tensors (on device).
        value: torch.Tensor = fwds[0]
        action: torch.Tensor = fwds[1]
        action_log_prob: torch.Tensor = fwds[2]
        recurrent_hidden_states: torch.Tensor = fwds[3]
        action_dist: torch.Tensor = fwds[4]

        # Grab iteration index and env output from leader (no tensors included).
        iteration, ob, reward, done, info, backward_pass = env_spout.recv()

        # Get updated age from env.
        age = info["age"]

        decay = config.use_linear_lr_decay and backward_pass

        # If done then remove from environment.
        if done:
            break

        # Shape correction and casting.
        # TODO: Change names so everything is statically-typed.
        observation = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        masks, bad_masks = get_masks(done, info)

        # Add to rollouts.
        rollouts.insert(
            observation,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            reward,
            masks,
            bad_masks,
        )

        # Only when trainer would make an update/backward pass.
        if backward_pass:
            with torch.no_grad():
                next_value = agent.actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()
            rollouts.compute_returns(
                next_value,
                config.use_gae,
                config.gamma,
                config.gae_lambda,
                config.use_proper_time_limits,
            )

            # Compute weight updates.
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            rollouts.after_update()

            # Send losses back to leader for ``update_losses()``.
            loss_funnel.send((value_loss, action_loss, dist_entropy))

        # Send state back to the leader.
        save_state: bool = iteration % config.save_interval == 0
        if save_state or iteration == config.time_steps - 1:

            # This is becuase torch.multiprocessing will not allow you to send a tensor
            # created in another process to a different process.
            agent_copy = copy.deepcopy(agent)
            rollouts_copy = copy.deepcopy(rollouts)
            save_funnel.send((agent_copy, rollouts_copy))

        # Make a forward pass.
        fwds = act(iteration, decay, agent, rollouts, config, age, action_funnel)
