""" Distributed training function for a single agent worker. """
import sys
import time
from typing import Dict, Tuple, Any
from multiprocessing.connection import Connection

import torch
import torch.nn.functional as F

import numpy as np

from bees.rl import utils
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.utils import DEBUG, timing
from bees.config import Config

# pylint: disable=duplicate-code

STOP_FLAG = 999

# TODO: Consider using Ray for multiprocessing, which is supposedly around 10x faster.


def get_policy_score(action_dist: torch.Tensor, info: Dict[str, Any]) -> float:
    """ Compute the policy score given current and optimal distributions. """
    optimal_action_dist = info["optimal_action_dist"]
    action_dist = action_dist.cpu()

    timestep_score = float(
        F.kl_div(torch.log(action_dist), optimal_action_dist, reduction="sum")
    )

    return timestep_score


def get_masks(done: bool, info: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
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


def act(
    step: int,
    decay: bool,
    agent_id: int,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
    age: int,
    action_funnel: Connection,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # TODO: Consider moving this block and the above to the bottom so that
    # ``env_spout.recv()`` is the first statement in the loop.
    # This would require running it once at the top on agent's first iteration
    # when step == initial_step, which is gross.
    with torch.no_grad():
        act_returns = agent.actor_critic.act(
            rollouts.obs[step],
            rollouts.recurrent_hidden_states[step],
            rollouts.masks[step],
        )

        # Get integer action to pass to ``env.step()``.
        env_action: int = int(act_returns[1][0])

    # TODO: Send ``env_action: int`` back to leader to execute step.
    action_funnel.send(env_action)

    print("Action send time: %f" % (time.time()))

    return act_returns


def worker_loop(
    device: torch.device,
    agent_id: int,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
    initial_step: int,
    initial_ob: np.ndarray,
    env_spout: Connection,
    action_funnel: Connection,
    action_dist_funnel: Connection,
    loss_funnel: Connection,
) -> None:
    """ Training loop for a single agent worker. """

    age: int = 0
    step: int = initial_step

    # Copy first observations to rollouts, and send to device.
    initial_observation: torch.Tensor = torch.FloatTensor([initial_ob])
    rollouts.obs[0].copy_(initial_observation)
    rollouts.to(device)

    decay: bool = config.use_linear_lr_decay

    # Initial forward pass.
    fwds = act(step, decay, agent_id, agent, rollouts, config, age, action_funnel)

    while True:

        # These are all CUDA tensors (on device).
        value: torch.Tensor = fwds[0]
        action: torch.Tensor = fwds[1]
        action_log_prob: torch.Tensor = fwds[2]
        recurrent_hidden_states: torch.Tensor = fwds[3]
        action_dist: torch.Tensor = fwds[4]

        t_0 = time.time()

        # Execute environment step.
        # TODO: Grab step index and output from leader (no tensors included).
        step, ob, reward, done, info, backward_pass = env_spout.recv()

        print("Received step %d in %fs" % (step, time.time() - t_0))
        sys.stdout.flush()

        decay = config.use_linear_lr_decay and backward_pass

        # Update the policy score.
        # TODO: Send ``action_dist`` back to leader to update_policy_score.
        # TODO: This should be done every k steps on workers instead of leader.
        # Then we just send the floats back to leader, which is cheaper.

        # TODO: Only compute on policy_score_frequency.
        """
        timestep_score = get_policy_score(action_dist, info)
        action_dist_funnel.send(timestep_score)
        """

        # If done then remove from environment.
        if done:
            action_funnel.send(STOP_FLAG)

        t_0 = time.time()

        # Shape correction and casting.
        # TODO: Change names so everything is statically-typed.
        observation = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        masks, bad_masks = get_masks(done, info)

        print("Tensor creation: %fs" % (time.time() - t_0,))
        sys.stdout.flush()
        t_0 = time.time()

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

        print("Rollout insertion: %fs" % (time.time() - t_0,))
        sys.stdout.flush()

        # Only when trainer would make an update/backward pass.
        # TODO: Environment is updating age, but we can't see it because of a shared
        # memory issue.
        age = info["age"]
        # TODO: Will age always be positive here?
        if backward_pass and age > 0:

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

            # TODO: Send losses back to leader for ``update_losses()``.
            loss_funnel.send((value_loss, action_loss, dist_entropy))

        # Make a forward pass.
        fwds = act(step, decay, agent_id, agent, rollouts, config, age, action_funnel)
