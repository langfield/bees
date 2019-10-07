""" PyTorch environment trainer. """
import os
import sys
import json
import time
import logging
import argparse
import collections
from collections import deque
from typing import Dict, Tuple, Set, List, Any

import gym
import torch
import optuna
import numpy as np

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

# from evaluation import evaluate

from main import create_env

# pylint: disable=bad-continuation


def train(settings: Dict[str, Any]) -> float:
    """
    Runs the environment.

    Parameters
    ----------
    settings : ``Dict[str, Any]``.
        Global settings file.

    Returns
    -------
    loss : ``float``.
        The optuna loss.
    """
    args = get_args()
    args.num_env_steps = settings["env"]["time_steps"]
    print("Arguments:", str(args))
    env = create_env(settings)

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

    # Create multiagent maps.
    actor_critics: Dict[int, Policy] = {}
    agents: Dict[int, "AgentAlgo"] = {}
    rollout_map: Dict[int, RolloutStorage] = {}
    episode_rewards: Dict[int, collections.deque] = {}
    minted_agents: Set[int] = set()
    value_losses: Dict[int, float] = {}
    action_losses: Dict[int, float] = {}
    dist_entropies: Dict[int, float] = {}
    agent_lifetimes: List[int] = []

    obs = env.reset()

    # Initialize first policies.
    env_done = False
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
        obs_tensor = torch.FloatTensor([agent_obs])
        rollouts.obs[0].copy_(obs_tensor)
        rollouts.to(device)

    steps_completed = 0

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly.
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        for step in range(args.num_steps):

            minted_agents = set()
            value_dict: Dict[int, float] = {}
            action_dict: Dict[int, Tuple[int, int, int]] = {}
            action_tensor_dict: Dict[int, torch.Tensor] = {}
            action_log_prob_dict: Dict[int, float] = {}
            recurrent_hidden_states_dict: Dict[int, float] = {}

            # Sample actions
            with torch.no_grad():
                for agent_id, actor_critic in actor_critics.items():
                    rollouts = rollout_map[agent_id]
                    # Right now our new distribution object returns a
                    # tuple of tensors instead of a tensor as the action.
                    # This is where we should start tomorrow, then continue
                    # reverting changes to MultiBinary action space. 
                    ac_tuple = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )
                    value_dict[agent_id] = ac_tuple[0]
                    action_dict[agent_id] = ac_tuple[1][0].cpu().numpy()
                    action_tensor_dict[agent_id] = ac_tuple[1]
                    action_log_prob_dict[agent_id] = ac_tuple[2]
                    recurrent_hidden_states_dict[agent_id] = ac_tuple[3]

            # Observe reward and next obs
            obs, rewards, dones, infos = env.step(action_dict)

            # NOTE: we assume ``args.num_processes`` is ``1``.

            for agent_id in obs:
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
                    # Update dicts.
                    agents[agent_id] = agent
                    actor_critics[agent_id] = actor_critic
                    rollout_map[agent_id] = rollouts
                    episode_rewards[agent_id] = deque(maxlen=10)

                    # Copy first observations to rollouts, and send to device.
                    rollouts = rollout_map[agent_id]
                    obs_tensor = torch.FloatTensor([agent_obs])
                    rollouts.obs[0].copy_(obs_tensor)
                    rollouts.to(device)

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

                    # If done then remove from environment.
                    if agent_done:
                        actor_critic = actor_critics.pop(agent_id)
                        del actor_critic
                        # TODO: should we remove from ``rollout_map`` and ``agents``?
                        agent_lifetimes.append(agent_info["age"])
                        agent = agents.pop(agent_id)
                        del agent

                    # Shape correction and casting.
                    obs_tensor = torch.FloatTensor([agent_obs])
                    reward_tensor = torch.FloatTensor([agent_reward])

                    # Add to rollouts.
                    value = value_dict[agent_id]
                    action_tensor = action_tensor_dict[agent_id]
                    action_log_prob = action_log_prob_dict[agent_id]
                    recurrent_hidden_states = recurrent_hidden_states_dict[agent_id]

                    rollout_map[agent_id].insert(
                        obs_tensor,
                        recurrent_hidden_states,
                        action_tensor,
                        action_log_prob,
                        value,
                        reward_tensor,
                        masks,
                        bad_masks,
                    )

            # Print out environment state.
            if settings["env"]["print"]:
                os.system("clear")
                print(env)
            if all(dones.values()):
                if settings["env"]["print"]:
                    print("All agents have died.")
                env_done = True
            steps_completed += 1

            avg_agent_lifetime = np.mean(agent_lifetimes)
            loss = compute_loss(
                steps_completed,
                args.num_env_steps,
                avg_agent_lifetime,
                settings["env"]["aging_rate"],
                len(agents),
                settings["env"]["width"],
                settings["env"]["height"],
            )

            if "trial" in settings:
                width = settings["env"]["width"]
                height = settings["env"]["height"]
                trial = settings["trial"]
                trial.report(loss, steps_completed)
                """
                if steps_completed % 50 == 0 and trial.should_prune():
                    print("Pruning automatically.")
                    raise optuna.structs.TrialPruned()
                """
                agent_density = len(agents) / (width * height)
                # HARDCODE
                if agent_density > 0.2:
                    print("Density too high:", agent_density)
                    raise optuna.structs.TrialPruned()

        # DEBUG
        print("\n\n")
        t0_list = []
        t1_list = []

        for agent_id, agent in agents.items():
            if agent_id not in minted_agents:

                # DEBUG
                t0 = time.time()

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

                # DEBUG
                t0_list.append(time.time() - t0)
                t1 = time.time()

                value_loss, action_loss, dist_entropy = agent.update(rollouts)

                # DEBUG
                t1_list.append(time.time() - t1)

                value_losses[agent_id] = value_loss
                action_losses[agent_id] = action_loss
                dist_entropies[agent_id] = dist_entropy

                rollouts.after_update()

        print("t0 avg:", np.mean(t0_list))
        print("t1 avg:", np.mean(t1_list))

        """
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
        """

        for agent_id, agent in agents.items():
            if agent_id not in minted_agents:
                agent_episode_rewards = episode_rewards[agent_id]
                if j % args.log_interval == 0 and len(agent_episode_rewards) > 1:
                    total_num_steps = (j + 1) * args.num_processes * args.num_steps
                    end = time.time()
                    """
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

        if env_done:
            break

    logging.getLogger().info(
        "Steps completed during episode: %d / %d"
        % (steps_completed, args.num_env_steps)
    )

    return loss


def get_agent(
    args: argparse.Namespace,
    obs_space: gym.Space,
    act_space: gym.Space,
    device: torch.device,
) -> Tuple["AgentAlgo", Policy, RolloutStorage]:
    """
    Spins up a new agent/policy.

    Parameters
    ----------
    args : ``argparse.Namespace``.
        Command-line arguments from a2c-ppo-acktr.
    obs_space : ``gym.Space``.
        Observation space from the environment.
    act_space : ``gym.Space``.
        Action space from the environment.
    device : ``torch.device``.
        The GPU/TPU/CPU.

    Returns
    -------
    agent : ``AgentAlgo``.
        Agent object from a2c-ppo-acktr.
    actor_critic : ``Policy``.
        The policy object.
    rollouts : ``RolloutStorage``.
        The rollout object.
    """

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
    return agent, actor_critic, rollouts


def compute_loss(
    num_env_steps: int,
    max_num_env_steps: int,
    avg_agent_lifetime: float,
    aging_rate: float,
    num_agents: int,
    width: int,
    height: int,
) -> float:
    """
    Computes the optuna loss function.

    Parameters
    ----------
    num_env_steps : ``int``.
        Number of environment steps completed so far.
    max_num_env_steps : ``int``.
        Number of environment steps to attempt.
    avg_agent_lifetime : ``float``.
        Average agent lifetime over all done agents measured in environment steps.
    aging_rate : ``float``.
        Health loss for all agents at each environment step.
    num_agents : ``int``.
        Number of living agents.
    width : ``int``.
        Width of the grid.
    height : ``int``.
        Height of the grid.

    Returns
    -------
    loss : ``float``.
        Loss as computed for an optuna trial.
    """
    # Constants.
    # HARDCODE
    optimal_density = 0.05
    optimal_lifetime = 5

    agent_density = num_agents / (width * height)
    lifetime_loss = ((avg_agent_lifetime / (1 / aging_rate)) - optimal_lifetime) ** 2
    density_loss = (agent_density - optimal_density) ** 2
    step_loss = (max_num_env_steps - num_env_steps) ** 2

    # Normalize losses.
    # HARDCODE
    lifetime_loss = lifetime_loss / (optimal_lifetime ** 2)
    density_loss = density_loss / (4 * optimal_density)
    step_loss = 2 * step_loss

    print(
        "Steps completed: %d\t Avg lifetime: %f\t Agent density: %f\r"
        % (num_env_steps, avg_agent_lifetime, agent_density),
        end="",
    )

    loss = lifetime_loss + density_loss + step_loss
    return loss


if __name__ == "__main__":
    # Get settings and create environment.
    # HARDCODE
    SETTINGS_FILE = "settings/settings.json"
    with open(SETTINGS_FILE, "r") as f:
        SETTINGS = json.load(f)
    train(SETTINGS)
