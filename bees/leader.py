""" PyTorch environment trainer. """
import os
import json
import copy
import random
import pickle
import logging
import argparse
from collections import OrderedDict
from typing import Dict, Tuple, Set, List, Any, TextIO
from multiprocessing.connection import Connection

import gym
import numpy as np

import torch
import torch.multiprocessing as mp

from bees.rl import algo, utils
from bees.rl.model import Policy, CNNBase, MLPBase
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.config import Config
from bees.worker import single_agent_loop
from bees.analysis import (
    update_policy_score,
    update_losses,
    update_food_scores,
    Metrics,
)
from bees.initialization import Setup

# pylint: disable=bad-continuation, too-many-branches, duplicate-code
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

    setup = Setup(args)
    config: Config = setup.config
    save_dir: str = setup.save_dir
    codename: str = setup.codename
    env_log: TextIO = setup.env_log
    visual_log: TextIO = setup.visual_log
    metrics_log: TextIO = setup.metrics_log
    env_state_path: str = setup.env_state_path
    trainer_state: Dict[str, Any] = setup.trainer_state

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
    minted_agents: Set[int] = set()
    metrics = Metrics()
    agent_action_dists: Dict[int, torch.Tensor] = {}

    # Save dead objects to make creation faster.
    dead_critics: Set[Policy] = set()
    dead_agents: Set[Algo] = set()
    state_dicts: List[OrderedDict] = []
    optim_state_dicts: List[OrderedDict] = []

    # Multiprocessing maps.
    workers: Dict[int, mp.Process] = {}
    devices: Dict[int, torch.device] = {}

    env_spouts: Dict[int, Connection] = {}
    action_spouts: Dict[int, Connection] = {}
    action_dist_spouts: Dict[int, Connection] = {}
    loss_spouts: Dict[int, Connection] = {}

    env_funnels: Dict[int, Connection] = {}
    action_funnels: Dict[int, Connection] = {}
    action_dist_funnels: Dict[int, Connection] = {}
    loss_funnels: Dict[int, Connection] = {}

    # Set spawn start method for compatibility with torch.
    mp.set_start_method("spawn")

    if args.load_from:
        raise NotImplementedError
    else:
        obs = env.reset()

    # Initialize first policies.
    env_done = False
    for agent_id, agent_obs in obs.items():

        # TODO: Implement device assignment.
        devices[agent_id] = device

        if agent_id not in agents:
            agent, actor_critic, rollouts = get_agent(
                config, env.observation_space, env.action_space, devices[agent_id]
            )
            agents[agent_id] = agent
            actor_critics[agent_id] = actor_critic
            rollout_map[agent_id] = rollouts

            # If turned on, saves a copy of each state dict for reuse in dead policies.
            if config.reuse_state_dicts:
                state_dicts.append(copy.deepcopy(actor_critic.state_dict()))
                optim_state_dicts.append(copy.deepcopy(agent.optimizer.state_dict()))

        # Add pipes to pipe maps.
        env_spouts[agent_id], env_funnels[agent_id] = mp.Pipe()
        action_spouts[agent_id], action_funnels[agent_id] = mp.Pipe()
        action_dist_spouts[agent_id], action_dist_funnels[agent_id] = mp.Pipe()
        loss_spouts[agent_id], loss_funnels[agent_id] = mp.Pipe()

        # Create worker processes.
        # TODO: Replace ``num_updates`` argument.
        # TODO: Will torch even let us serialize the model and rollouts objects?
        workers[agent_id] = mp.Process(
            target=single_agent_loop,
            kwargs={
                "device": devices[agent_id],
                "agent_id": agent_id,
                "agent": agents[agent_id],
                "rollouts": rollout_map[agent_id],
                "config": config,
                "initial_step": 0,  # TODO: Fix to reflect ``env.iteration``.
                "initial_ob": obs[agent_id],
                "env_spout": env_spouts[agent_id],
                "action_funnel": action_funnels[agent_id],
                "action_dist_funnel": action_dist_funnels[agent_id],
                "loss_funnel": loss_funnels[agent_id],
            },
        )

    for _, worker in workers.items():
        worker.start()

    # TODO: Could we stagger these?
    backward_pass = False

    num_updates = (
        int(config.time_steps - env.iteration)
        // config.num_steps
        // config.num_processes
    )
    iterations = int(config.time_steps - env.iteration) // config.num_processes
    num_updates = iterations // config.num_steps
    for step in range(iterations):

        # Should these all be defined up above with other maps?
        minted_agents = set()
        action_dict: Dict[int, int] = {}
        timestep_scores: Dict[int, float] = {}

        # Get actions.
        for agent_id in action_spouts:
            action_dict[agent_id] = action_spouts[agent_id].recv()

        # Execute environment step.
        obs, rewards, dones, infos = env.step(action_dict)

        # TODO: Change this condition so there's only one loop.
        if step == 0:
            backward_pass = True
        for agent_id in obs:
            env_funnels[agent_id].send(
                (
                    step,
                    obs[agent_id],
                    rewards[agent_id],
                    dones[agent_id],
                    infos[agent_id],
                    backward_pass,
                )
            )

        # Write env state and metrics to log.
        env.log_state(env_log, visual_log)
        metrics_log.write(str(metrics.get_summary()) + "\n")

        # Get policy scores.
        # TODO: Fix race condition.
        """
        for agent_id in infos:
            timestep_scores[agent_id] = action_dist_spouts[agent_id].recv()
        """

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
                if args.trial.should_prune() or metrics.policy_score == float("inf"):
                    print("\nEnding training because ``policy_score_loss`` diverged.")
                    return metrics.policy_score

        # Update food scores if any agents were born/died this step, and on the
        # first iteration.
        if env.iteration == 1 or set(obs.keys()) != set(agents.keys()):
            metrics = update_food_scores(env, metrics)

        # Print debug output.
        end = "\n" if config.print_repr else "\r"
        print("Iteration: %d| " % env.iteration, end="")
        print("Num agents: %d| " % len(agents), end="")
        print("Policy score loss: %.6f" % metrics.policy_score, end="")
        print("/%.6f| " % metrics.initial_policy_score, end="")
        print("Food score: %.6f" % metrics.food_score, end="")
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

                    # TODO: Grab pipes from dead pipes.

                    # TODO: Is this expensive?
                    if config.reuse_state_dicts:
                        state_dict = copy.deepcopy(random.choice(state_dicts))
                        optim_state_dict = copy.deepcopy(
                            random.choice(optim_state_dicts)
                        )

                        # TODO: We may need to send these in a pipe.
                        # TODO: This is bound to be expensive.
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
                                actor_critic.base.main, actor_critic.base.critic_linear,
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
                    # TODO: Is this any slower than clearing data from the old one?
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

                    # TODO: Implement device assignment.
                    devices[agent_id] = device

                    agent, actor_critic, rollouts = get_agent(
                        config,
                        env.observation_space,
                        env.action_space,
                        devices[agent_id],
                    )

                    # If turned on, saves a copy of each state dict for reuse in dead policies.
                    if config.reuse_state_dicts:
                        state_dicts.append(copy.deepcopy(actor_critic.state_dict()))
                        optim_state_dicts.append(
                            copy.deepcopy(agent.optimizer.state_dict())
                        )

                    # Add pipes to pipe maps.
                    env_spouts[agent_id], env_funnels[agent_id] = mp.Pipe()
                    action_spouts[agent_id], action_funnels[agent_id] = mp.Pipe()
                    (
                        action_dist_spouts[agent_id],
                        action_dist_funnels[agent_id],
                    ) = mp.Pipe()
                    loss_spouts[agent_id], loss_funnels[agent_id] = mp.Pipe()

                    # Create worker processes.
                    # TODO: Replace ``num_updates`` argument.
                    # TODO: Will torch even let us serialize the model and rollouts objects?
                    workers[agent_id] = mp.Process(
                        target=single_agent_loop,
                        kwargs={
                            "device": devices[agent_id],
                            "agent_id": agent_id,
                            "agent": agents[agent_id],
                            "rollouts": rollout_map[agent_id],
                            "config": config,
                            "initial_step": step,  # TODO: Off by 1?
                            "initial_ob": obs[agent_id],
                            "env_spout": env_spouts[agent_id],
                            "action_funnel": action_funnels[agent_id],
                            "action_dist_funnel": action_dist_funnels[agent_id],
                            "loss_funnel": loss_funnels[agent_id],
                        },
                    )

                # Update dicts.
                agents[agent_id] = agent
                actor_critics[agent_id] = actor_critic
                rollout_map[agent_id] = rollouts

                # Copy first observations to rollouts, and send to device.
                rollouts = rollout_map[agent_id]
                obs_tensor = torch.FloatTensor([agent_obs])
                rollouts.obs[0].copy_(obs_tensor)
                rollouts.to(device)

            else:

                # If done then remove from environment.
                if agent_done:
                    actor_critic = actor_critics.pop(agent_id)
                    dead_critics.add(actor_critic)
                    # TODO: should we remove from ``rollout_map``?
                    # Yes, because otherwise we are relying on garbage collection
                    # because we reassign keys, which is bad practice.
                    rollout_map.pop(agent_id)
                    agent = agents.pop(agent_id)
                    dead_agents.add(agent)

        # Print out environment state.
        if all(dones.values()):
            if config.print_repr:
                print("All agents have died.")
            env_done = True

        # Only update losses and save on backward passes.
        if step % config.num_steps == 0:

            value_losses: Dict[int, float] = {}
            action_losses: Dict[int, float] = {}
            dist_entropies: Dict[int, float] = {}

            # Should we iterate over a different object?
            for agent_id in agents:
                if agent_id not in minted_agents:
                    value_loss, action_loss, dist_entropy = loss_spouts[agent_id].recv()
                    value_losses[agent_id] = value_loss
                    action_losses[agent_id] = action_loss
                    dist_entropies[agent_id] = dist_entropy

            metrics = update_losses(
                env=env,
                config=config,
                losses=(value_losses, action_losses, dist_entropies),
                metrics=metrics,
            )

            # save for every interval-th episode or for the last epoch
            if (
                step % config.save_interval == 0
                or step // config.num_steps == num_updates - 1
            ) and args.save_root != "":

                # Save trainer state objects
                trainer_state = {
                    "agents": agents,
                    "rollout_map": rollout_map,
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
