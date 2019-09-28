import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = make_vec_envs(
        args.env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        args.log_dir,
        device,
        False,
    )

    def get_agent(
        args: args.Namespace,
        obs_space: gym.space,
        act_space: gym.space,
        device: torch.device,
    ) -> Tuple["AgentAlgo", Policy, RolloutStorage]:

        actor_critic = Policy(
            obs_space.shape, act_space, base_kwargs={"recurrent": args.recurrent_policy}
        )
        actor_critic.to(device)

        if args.algo == "a2c":
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == "ppo":
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == "acktr":
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True
            )

        rollouts = RolloutStorage(
            args.num_steps,
            args.num_processes,
            obs_space.shape,
            act_space,
            actor_critic.recurrent_hidden_state_size,
        )

    # Create multiagent maps.
    actor_critics: Dict[str, Policy] = {}
    agents: Dict[str, "AgentAlgo"] = {}
    rollout_map: Dict[str, RolloutStorage] = {}
    episode_rewards: Dict[str, collections.deque] = {}
    minted_agents: Set[str] = set()
    value_losses: Dict[str, float] = {}
    action_losses: Dict[str, float] = {}
    dist_entropies: Dict[str, float] = {}

    obs = env.reset()

    # Initialize first policies.
    for agent_id, agent_obs in obs.items():
        if agent_id not in agents:
            agent, actor_critic, rollouts = get_agent(
                args, env.observation_space, env.action_space, device
            )
            agents[agent_id] = agent
            actor_critics[agent_id] = actor_critic
            rollout_map[agent_id] = rollouts
            episode_rewards[agent_id] = deque(maxlen=10)

        # Copy first observations to rollouts, and send to device.
        rollouts = rollout_map[agent_id]
        rollouts.obs[0].copy_(agent_obs)
        rollouts.to(device)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        for step in range(args.num_steps):

            minted_agents = set()

            actor_critic_return_dict: Dict[
                str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = {}

            # Sample actions
            with torch.no_grad():
                for agent_id, actor_critic in actor_critics.items():
                    rollouts = rollout_map[agent_id]
                    actor_critic_return_dict[agent_id] = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )

            # Obser reward and next obs
            obs, rewards, dones, infos = env.step(action)

            # NOTE: we assume ``args.num_processes`` is ``1``.

            for agent_id in obs.keys():
                agent_obs = obs[agent_id]
                agent_reward = rewards[agent_id]
                agent_done = dones[agent_id]
                agent_info = infos[agent_id]

                # Initialize new policies.
                if agent_id not in agents:
                    minted_agents.add(agent_id)
                    agent, actor_critic, rollouts = get_agent(
                        args, env.observation_space, env.action_space, device
                    )

                    # Copy first observations to rollouts, and send to device.
                    rollouts = rollout_map[agent_id]
                    rollouts.obs[0].copy_(agent_obs)
                    rollouts.to(device)

                    # Update dicts.
                    agents[agent_id] = agent
                    actor_critics[agent_id] = actor_critic
                    rollout_map[agent_id] = rollouts

                else:
                    if "episode" in agent_info.keys():
                        episode_rewards[agent_id].append(agent_info["episode"]["r"])

                    # If done then clean the history of observations.
                    if agent_done:
                        masks = torch.FloatTensor([[0.0]])
                    else:
                        masks = torch.FloatTensor([[1.0]])
                    if "bad_transition" in agent_info.keys():
                        bad_masks = torch.FloatTensor([[0.0]])
                    else:
                        bad_masks = torch.FloatTensor([[1.0]])

                    # Shape correction and casting.
                    obs_tensor = torch.FloatTensor([[obs]])
                    reward_array = np.array([agent_reward])

                    # Add to rollouts.
                    value_tuple = actor_critic_return_dict[agent_id]
                    value, action, action_log_prob, recurrent_hidden_states = (
                        value_tuple
                    )
                    rollout_map[agent_id].insert(
                        obs_tensor,
                        recurrent_hidden_states,
                        action,
                        action_log_prob,
                        value,
                        reward_array,
                        masks,
                        bad_masks,
                    )

        for agent_id, agent in agents.items():
            if agent_id not in minted_agents:

                actor_critic = actor_critics[agent_id]
                rollouts = rollout_map[agent_id]

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1],
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1],
                    ).detach()

                rollouts.compute_returns(
                    next_value,
                    args.use_gae,
                    args.gamma,
                    args.gae_lambda,
                    args.use_proper_time_limits,
                )

                value_loss, action_loss, dist_entropy = agent.update(rollouts)
                value_losses[agent_id] = value_loss
                action_losses[agent_id] = action_loss
                dist_entropies[agent_id] = dist_entropy

                rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (
            j % args.save_interval == 0 or j == num_updates - 1
        ) and args.save_dir != "":
            for agent_id, agent in agents.items():
                if agent_id not in minted_agents:
                    actor_critic = actor_critics[agent_id]
                    save_path = os.path.join(args.save_dir, args.algo)
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass

                    # TODO: implement ``ob_rms`` from ``VecNormalize`` in baselines in our env.
                    # TODO: Add ``agent_id`` in save path.
                    torch.save(
                        [actor_critic, getattr(env, "ob_rms", None)],
                        os.path.join(save_path, args.env_name + ".pt"),
                    )

        for agent_id, agent in agents.items():
            if agent_id not in minted_agents:
                agent_episode_rewards = episode_rewards[agent_id]
                if j % args.log_interval == 0 and len(agent_episode_rewards) > 1:
                    total_num_steps = (j + 1) * args.num_processes * args.num_steps
                    end = time.time()
                    print(
                        "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                            j,
                            total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(agent_episode_rewards),
                            np.mean(agent_episode_rewards),
                            np.median(agent_episode_rewards),
                            np.min(agent_episode_rewards),
                            np.max(agent_episode_rewards),
                            dist_entropies[agent_id],
                            value_losses[agent_id],
                            action_losses[agent_id],
                        )
                    )

                """
                if (
                    args.eval_interval is not None
                    and len(agent_episode_rewards) > 1
                    and j % args.eval_interval == 0
                ):
                    ob_rms = utils.get_vec_normalize(env).ob_rms
                    evaluate(
                        actor_critic,
                        ob_rms,
                        args.env_name,
                        args.seed,
                        args.num_processes,
                        eval_log_dir,
                        device,
                    )
                """


if __name__ == "__main__":
    main()
