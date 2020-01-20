""" Distributed training function for a single agent worker. """
import collections

import torch

from bees.rl import utils
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.config import Config

# pylint: disable=duplicate-code


def single_agent_loop(
    agent_id: int,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
    env: Env,
    num_updates: int,
):
    """ Training loop for a single agent worker. """

    agent_episode_rewards = collections.deque(maxlen=10)

    for j in range(num_updates):

        if config.use_linear_lr_decay:

            # Compute age and minimum agent lifetime.
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

        for step in range(config.num_steps):

            # Sample actions.
            with torch.no_grad():
                ac_tuple = agent.actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

                value: torch.Tensor = ac_tuple[0]
                env_action: int = int(ac_tuple[1][0])
                action: torch.Tensor = ac_tuple[1]
                action_log_prob: torch.Tensor = ac_tuple[2]
                recurrent_hidden_states: torch.Tensor = ac_tuple[3]
                agent_action_dist: torch.Tensor = ac_tuple[4]

            # TODO: Send ``env_action`` back to leader to execute step.

            # Execute environment step.
            # TODO: Grab step output from leader.
            action_dict = {}
            obs, rewards, dones, infos = env.step(action_dict)

            # Update the policy score.
            # TODO: Send agent_action_dist back to leader to update_policy_score.

            # Agent creation and termination, rollout stacking.
            agent_obs = obs[agent_id]
            agent_reward = rewards[agent_id]
            agent_done = dones[agent_id]
            agent_info = infos[agent_id]

            if agent_id in agents:
                if "episode" in agent_info.keys():
                    agent_episode_rewards[agent_id].append(agent_info["episode"]["r"])

                # If done then clean the history of observations.
                if agent_done:
                    masks = torch.FloatTensor([[0.0]])
                else:
                    masks = torch.FloatTensor([[1.0]])
                if "bad_transition" in agent_info.keys():
                    bad_masks = torch.FloatTensor([[0.0]])
                else:
                    bad_masks = torch.FloatTensor([[1.0]])

                # If done then remove from environment.
                # TODO: Tell leader to remove from environment.
                if agent_done:
                    pass

                # Shape correction and casting.
                observation = torch.FloatTensor([agent_obs])
                reward = torch.FloatTensor([agent_reward])

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

        if env.agent.age > 0:

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

        # +++++CONVERT+++++
        # TODO: Send losses back to leader for ``update_losses()``.
        # TODO: Break if done.
        """
        metrics = update_losses(
            env=env,
            config=config,
            losses=(value_losses, action_losses, dist_entropies),
            metrics=metrics,
        )
        if env_done:
            break
        """
        # +++++CONVERT+++++
