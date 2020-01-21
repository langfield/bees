""" Distributed training function for a single agent worker. """
import collections
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from bees.rl import utils
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.config import Config

# pylint: disable=duplicate-code

STOP_FLAG = 999

# TODO: Consider using Ray for multiprocessing, which is supposedly around 10x faster.


def update_learning_rate(agent_id: int, agent: Algo, config: Config, env: Env) -> None:
    """ Compute age and minimum agent lifetime. """
    # TODO: Figure out how to share ``env`` in memory.
    env_agent = env.agents[agent_id]
    age = env_agent.age
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


def get_policy_score(action_dist: torch.Tensor, info: Dict[str, Any]) -> float:
    """ Compute the policy score given current and optimal distributions. """
    optimal_action_dist = info["optimal_action_dist"]
    action_dist = action_dist.cpu()

    timestep_score = float(
        F.kl_div(torch.log(action_dist), optimal_action_dist, reduction="sum")
    )

    return timestep_score


def single_agent_loop(
    device: torch.device,
    agent_id: int,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
    env: Env,
    num_updates: int,
    initial_step: int,
    initial_ob: np.ndarray,
    env_pipe: mp.Pipe,
    action_pipe: mp.Pipe,
    action_dist_pipe: mp.Pipe,
    loss_pipe: mp.Pipe,
) -> None:
    """ Training loop for a single agent worker. """

    step = initial_step
    agent_episode_rewards: collections.deque = collections.deque(maxlen=10)

    # Copy first observations to rollouts, and send to device.
    initial_observation: torch.Tensor = torch.FloatTensor([initial_ob])
    rollouts.obs[0].copy_(initial_observation)
    rollouts.to(device)

    # TODO: Compute/pass this!!!
    num_updates_condition = float("inf")

    while True:

        # Only when trainer would make an update/backward pass.
        if step % num_updates_condition == 0 and config.use_linear_decay:
            update_learning_rate(agent_id, agent, config, env)

        # TODO: Consider moving this block and the above to the bottom so that
        # ``env_pipe.get()`` is the first statement in the loop.
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

            # These are all CUDA tensors (on device).
            value: torch.Tensor = act_returns[0]
            action: torch.Tensor = act_returns[1]
            action_log_prob: torch.Tensor = act_returns[2]
            recurrent_hidden_states: torch.Tensor = act_returns[3]
            action_dist: torch.Tensor = act_returns[4]

        # TODO: Send ``env_action: int`` back to leader to execute step.
        action_pipe.put(env_action)

        # Execute environment step.
        # TODO: Grab step index and output from leader (no tensors included).
        step, ob, reward, done, info = env_pipe.get()

        # Update the policy score.
        # TODO: Send ``action_dist`` back to leader to update_policy_score.
        # TODO: This should be done every k steps on workers instead of leader.
        # Then we just send the floats back to leader, which is cheaper.
        timestep_score = get_policy_score(action_dist, info)
        action_dist_pipe.put(timestep_score)

        # Rollout stacking.
        # TODO: Figure out this condition.
        if agent_id in agents:
            if "episode" in info.keys():
                agent_episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            if done:
                masks = torch.FloatTensor([[0.0]])
            else:
                masks = torch.FloatTensor([[1.0]])
            if "bad_transition" in info.keys():
                bad_masks = torch.FloatTensor([[0.0]])
            else:
                bad_masks = torch.FloatTensor([[1.0]])

            # If done then remove from environment.
            # TODO: Tell leader to remove from environment.
            if done:
                action_pipe.put(STOP_FLAG)
                pass

            # Shape correction and casting.
            # TODO: Change names so everything is statically-typed.
            observation = torch.FloatTensor([ob])
            reward = torch.FloatTensor([reward])

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
        if step % num_updates_condition == 0 and env.agent.age > 0:

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
            loss_pipe.put((value_loss, action_loss, dist_entropy))
