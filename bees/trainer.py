""" PyTorch environment trainer. """
import os
import json
import copy
import random
import logging
import argparse
import collections
from collections import deque, OrderedDict
from typing import Dict, Tuple, Set, List, Any, TextIO
import pickle

import gym
import torch
import numpy as np

from bees.rl import algo, utils
from bees.rl.model import Policy, CNNBase, MLPBase
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.config import Config
from bees.analysis import update_policy_score, update_losses, Metrics
from bees.initialization import Setup

# pylint: disable=bad-continuation, too-many-branches
# pylint: disable=too-many-statements, too-many-locals


def train(args: argparse.Namespace) -> float:
    """
    Runs the environment.

    Three command line arguments ``--settings``, ``--load-from``, ``--save-root``.

    If you want to run training from scratch, you must pass ``--settings``, so that
    the program knows what to do, and you may optionally pass ``--save-root`` in the
    case where you would like to save elsewhere from the canonical directory.

    Passing ``--load-from`` will tell the program to attempt to load from a previously
    saved training run at the directory specified by the value of the argument. If
    ``--save-root`` is passed, then all saves during the current run will be saved in
    that root directory, regardless of the root directory implied by ``--load-from``.
    If ``--settings`` is passed, then the settings file, if any, in the ``--load-from``
    directory will be ignored. If no ``--settings`` argument is passed and there is no
    settings file in the ``--load-from`` directory, then the program will raise an
    error. If no ``--save-root`` is passed, the root will be implicitly set to the
    parent directory of ``--load-from``, i.e. the ``--save-root`` from the training run
    being loaded in. It will NOT default to the canonical root unless this is also the
    parent directory of ``--load-from``.

    Since there are a lot of cases, we should add an ``validate_args()`` function which
    raises errors when needed.

    --save-root : ALWAYS OPTIONAL -> Canonical rootdir default.

    --load-from : ALWAYS OPTIONAL -> Empty default.

    --settings : OPTIONAL IF --load-from ELSE REQUIRED -> Empty default.

    Parameters
    ----------
    args : ``argparse.Namespace``.
        Contains arguments as described above.
    """

    # TODO: Replace instances of these locals with ``setup.<var_name>``.
    setup = Setup(args)
    config: Config = setup.config
    trainer_state: Dict[str, Any] = setup.trainer_state
    env_log: TextIO = setup.env_log
    visual_log: TextIO = setup.visual_log
    save_dir: str = setup.save_dir
    settings_path: str = setup.settings_path
    trainer_state_path: str = setup.trainer_state_path
    env_state_path: str = setup.env_state_path
    codename: str = setup.codename

    # Create environment.
    if config.print_repr:
        print("Arguments:", str(config))
    env = Env(config)

    if not config.reuse_state_dicts:
        print(
            "Warning: ``config.reuse_state_dicts`` is False. This is slower, but the "
            "alternative bounds the number of unique policy initializations, i.e. "
            "policy initializations will be reused for multiple agents."
        )

    # Set random seed for all packages.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(config.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(2)
    device = torch.device("cuda:0" if config.cuda else "cpu")

    # Create multiagent maps.
    actor_critics: Dict[int, Policy] = {}
    agents: Dict[int, Algo] = {}
    rollout_map: Dict[int, RolloutStorage] = {}
    episode_rewards: Dict[int, collections.deque] = {}
    minted_agents: Set[int] = set()
    metrics = Metrics()
    agent_action_dists: Dict[int, torch.Tensor] = {}

    # Save dead objects to make creation faster.
    dead_critics: Set[Policy] = set()
    dead_agents: Set[Algo] = set()
    state_dicts: List[OrderedDict] = []
    optim_state_dicts: List[OrderedDict] = []

    if args.load_from:

        # Load the environment state from file.
        env.load(env_state_path)

        # Load in multiagent maps.
        agents = trainer_state["agents"]
        rollout_map = trainer_state["rollout_map"]
        episode_rewards = trainer_state["episode_rewards"]
        minted_agents = trainer_state["minted_agents"]
        metrics = trainer_state["metrics"]
        agent_action_dists = trainer_state["agent_action_dists"]

        # Load in dead objects.
        dead_critics = trainer_state["dead_critics"]
        dead_agents = trainer_state["dead_agents"]
        state_dicts = trainer_state["state_dicts"]
        optim_state_dicts = trainer_state["optim_state_dicts"]

        # Create actor_critics object from state in ``agents``.
        actor_critics = {
            agent_id: agent.actor_critic for agent_id, agent in agents.items()
        }

        # Don't reset environment if we are resuming a previous run.
        obs = {agent_id: agent.observation for agent_id, agent in env.agents.items()}

    else:

        obs = env.reset()

    # Initialize first policies.
    env_done = False
    for agent_id, agent_obs in obs.items():
        if agent_id not in agents:
            agent, actor_critic, rollouts = get_agent(
                config, env.observation_space, env.action_space, device
            )
            agents[agent_id] = agent
            actor_critics[agent_id] = actor_critic
            rollout_map[agent_id] = rollouts
            episode_rewards[agent_id] = deque(maxlen=10)

            if config.reuse_state_dicts:
                # Save a copy of the state dict.
                state_dicts.append(copy.deepcopy(actor_critic.state_dict()))
                optim_state_dicts.append(copy.deepcopy(agent.optimizer.state_dict()))

        # Copy first observations to rollouts, and send to device.
        rollouts = rollout_map[agent_id]
        obs_tensor = torch.FloatTensor([agent_obs])
        rollouts.obs[0].copy_(obs_tensor)
        rollouts.to(device)

    num_updates = (
        int(config.time_steps - env.iteration)
        // config.num_steps
        // config.num_processes
    )
    for j in range(num_updates):

        if config.use_linear_lr_decay:

            # Decrease learning rate linearly.
            for agent_id, agent in agents.items():

                # Compute age and minimum agent lifetime.
                env_agent = env.agents[agent_id]
                age = env_agent.age
                min_agent_lifetime = 1.0 / config.aging_rate

                learning_rate = utils.update_linear_schedule(
                    agent.optimizer,
                    age,
                    min_agent_lifetime,
                    agent.optimizer.lr if config.algo == "acktr" else config.lr,
                    config.min_lr,
                )

                agent.lr = learning_rate

        for step in range(config.num_steps):

            minted_agents = set()
            value_dict: Dict[int, torch.Tensor] = {}
            # action_dict: Dict[int, Tuple[int, int, int]] = {}
            action_dict: Dict[int, int] = {}
            action_tensor_dict: Dict[int, torch.Tensor] = {}
            action_log_prob_dict: Dict[int, torch.Tensor] = {}
            recurrent_hidden_states_dict: Dict[int, torch.Tensor] = {}

            # Sample actions.
            with torch.no_grad():
                for agent_id, actor_critic in actor_critics.items():
                    rollouts = rollout_map[agent_id]
                    ac_tuple = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )
                    value_dict[agent_id] = ac_tuple[0]
                    action_dict[agent_id] = int(ac_tuple[1][0])
                    action_tensor_dict[agent_id] = ac_tuple[1]
                    action_log_prob_dict[agent_id] = ac_tuple[2]
                    recurrent_hidden_states_dict[agent_id] = ac_tuple[3]
                    agent_action_dists[agent_id] = ac_tuple[4]

            # Execute environment step.
            obs, rewards, dones, infos = env.step(action_dict)

            # Write env state to log.
            env.log_state(env_log, visual_log)

            # Update the policy score.
            if env.iteration % config.policy_score_frequency == 0:
                metrics = update_policy_score(
                    env=env,
                    config=config,
                    infos=infos,
                    agent_action_dists=agent_action_dists,
                    metrics=metrics,
                )

                # This block will run if train() was called with optuna for parameter
                # optimization. If policy score loss explodes, end the training run
                # early.
                if hasattr(args, "trial"):
                    args.trial.report(metrics.policy_score, env.iteration)
                    if args.trial.should_prune() or metrics.policy_score == float(
                        "inf"
                    ):
                        print(
                            "\nEnding training because ``policy_score_loss`` diverged."
                        )
                        return metrics.policy_score

            # Print debug output.
            end = "\n" if config.print_repr else "\r"
            print("Iteration: %d| " % env.iteration, end="")
            print("Num agents: %d| " % len(agents), end="")
            print("Policy score loss: %.6f" % metrics.policy_score, end="")
            print("/%.6f| " % metrics.initial_policy_score, end="")
            print("Losses (action, value, entropy, total): ", end="")
            print("%.6f, " % metrics.action_loss, end="")
            print("%.6f, " % metrics.value_loss, end="")
            print("%.6f, " % metrics.dist_entropy, end="")
            print("%.6f" % metrics.total_loss, end="")
            print("||||||", end=end)

            # Agent creation and termination, rollout stacking.
            for agent_id in obs:
                agent_obs = obs[agent_id]
                agent_reward = rewards[agent_id]
                agent_done = dones[agent_id]
                agent_info = infos[agent_id]

                # Initialize new policies.
                if agent_id not in agents:
                    minted_agents.add(agent_id)

                    # Test whether we can reuse previously instantiated policy
                    # objects, or if we need to create new ones.
                    if len(dead_critics) > 0:

                        actor_critic = dead_critics.pop()
                        agent = dead_agents.pop()

                        if config.reuse_state_dicts:
                            state_dict = copy.deepcopy(random.choice(state_dicts))
                            optim_state_dict = copy.deepcopy(
                                random.choice(optim_state_dicts)
                            )

                            # Load initialized state dicts.
                            actor_critic.load_state_dict(state_dict)
                            agent.optimizer.load_state_dict(optim_state_dict)
                        else:

                            # Reinitialize the policy of ``actor_critic``.
                            if isinstance(actor_critic.base, CNNBase):
                                (
                                    actor_critic.base.main,
                                    actor_critic.base.critic_linear,
                                ) = CNNBase.init_weights(
                                    actor_critic.base.main,
                                    actor_critic.base.critic_linear,
                                )
                            elif isinstance(actor_critic.base, MLPBase):
                                (
                                    actor_critic.base.actor,
                                    actor_critic.base.critic,
                                    actor_critic.base.critic_linear,
                                ) = MLPBase.init_weights(
                                    actor_critic.base.actor,
                                    actor_critic.base.critic,
                                    actor_critic.base.critic_linear,
                                )
                            else:
                                raise NotImplementedError

                        # Create new RolloutStorage object.
                        rollouts = RolloutStorage(
                            config.num_steps,
                            config.num_processes,
                            env.observation_space.shape,
                            env.action_space,
                            actor_critic.recurrent_hidden_state_size,
                        )

                    else:
                        agent, actor_critic, rollouts = get_agent(
                            config, env.observation_space, env.action_space, device
                        )

                        # Save a copy of the state dict.
                        if config.reuse_state_dicts:
                            state_dicts.append(copy.deepcopy(actor_critic.state_dict()))
                            optim_state_dicts.append(
                                copy.deepcopy(agent.optimizer.state_dict())
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
                        dead_critics.add(actor_critic)
                        # TODO: should we remove from ``rollout_map``?
                        agent = agents.pop(agent_id)
                        dead_agents.add(agent)

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
            if all(dones.values()):
                if config.print_repr:
                    print("All agents have died.")
                env_done = True

        value_losses: Dict[int, float] = {}
        action_losses: Dict[int, float] = {}
        dist_entropies: Dict[int, float] = {}
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
                    config.use_gae,
                    config.gamma,
                    config.gae_lambda,
                    config.use_proper_time_limits,
                )

                # This is a tuple of floats.
                value_loss, action_loss, dist_entropy = agent.update(rollouts)
                value_losses[agent_id] = value_loss
                action_losses[agent_id] = action_loss
                dist_entropies[agent_id] = dist_entropy
                rollouts.after_update()

        metrics = update_losses(
            env=env,
            config=config,
            losses=(value_losses, action_losses, dist_entropies),
            metrics=metrics,
        )

        # save for every interval-th episode or for the last epoch
        if (
            j % config.save_interval == 0 or j == num_updates - 1
        ) and args.save_root != "":

            # Save trainer state objects
            trainer_state = {
                "agents": agents,
                "rollout_map": rollout_map,
                "episode_rewards": episode_rewards,
                "minted_agents": minted_agents,
                "metrics": metrics,
                "dead_critics": dead_critics,
                "dead_agents": dead_agents,
                "state_dicts": state_dicts,
                "optim_state_dicts": optim_state_dicts,
            }
            trainer_state_path = os.path.join(save_dir, "%s_trainer.pkl" % codename)
            with open(trainer_state_path, "wb") as trainer_file:
                pickle.dump(trainer_state, trainer_file)

            # Save out environment state.
            state_path = os.path.join(save_dir, "%s_env.pkl" % codename)
            env.save(state_path)

            # Save out settings, removing log files (not paths) from object.
            settings_path = os.path.join(save_dir, "%s_settings.json" % codename)
            with open(settings_path, "w") as settings_file:
                json.dump(config.settings, settings_file)

        if env_done:
            break

    logging.getLogger().info(
        "Steps completed during episode out of total: %d / %d",
        env.iteration,
        config.time_steps,
    )

    return metrics.policy_score


def get_agent(
    config: Config, obs_space: gym.Space, act_space: gym.Space, device: torch.device,
) -> Tuple[Algo, Policy, RolloutStorage]:
    """
    Spins up a new agent/policy.

    Parameters
    ----------
    config : ``Config``.
        Config object parsed from settings file.
    obs_space : ``gym.Space``.
        Observation space from the environment.
    act_space : ``gym.Space``.
        Action space from the environment.
    device : ``torch.device``.
        The GPU/TPU/CPU.

    Returns
    -------
    agent : ``Algo``.
        Agent object from a2c-ppo-acktr.
    actor_critic : ``Policy``.
        The policy object.
    rollouts : ``RolloutStorage``.
        The rollout object.
    """

    actor_critic = Policy(
        obs_space.shape, act_space, base_kwargs={"recurrent": config.recurrent_policy}
    )
    actor_critic.to(device)
    agent: Algo

    if config.algo == "a2c":
        agent = algo.A2C_ACKTR(
            actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            alpha=config.alpha,
            max_grad_norm=config.max_grad_norm,
        )
    elif config.algo == "ppo":
        agent = algo.PPO(
            actor_critic,
            config.clip_param,
            config.ppo_epoch,
            config.num_mini_batch,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
        )
    elif config.algo == "acktr":
        agent = algo.A2C_ACKTR(
            actor_critic, config.value_loss_coef, config.entropy_coef, acktr=True
        )

    rollouts = RolloutStorage(
        config.num_steps,
        config.num_processes,
        obs_space.shape,
        act_space,
        actor_critic.recurrent_hidden_state_size,
    )
    return agent, actor_critic, rollouts
