#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" PyTorch environment trainer. """
import os
import copy
import json
import time
import pickle
import random
import argparse
import collections
from typing import Any, Set, Dict, List, Tuple, TextIO

import numpy as np
import torch
import torch.multiprocessing as mp
from asta import dims, shapes

from bees.rl import utils
from bees.env import Env
from bees.pipe import Pipe
from bees.timer import Timer
from bees.config import Config
from bees.worker import act, get_masks, get_policy_score
from bees.analysis import Metrics, update_losses
from bees.creation import get_agent
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo
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

    # Create metrics and timer.
    metrics = Metrics()
    timer = Timer()

    # TIMER
    timer.start_interval("initialization")

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

    # Set asta dimensions and shapes.
    dims.N_ACTS = env.action_space.n
    shapes.OB = env.observation_space.shape

    # Set random seed for all packages.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # GPU setup.
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(config.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # Create multiagent maps.
    agents: Dict[int, Algo] = {}
    rollout_map: Dict[int, RolloutStorage] = {}
    minted_agents: Set[int] = set()

    # Save dead objects to make creation faster.
    dead_agents: Set[Algo] = set()
    dead_pipes: Dict[int, Pipe] = {}
    state_dicts: List[collections.OrderedDict] = []
    optim_state_dicts: List[collections.OrderedDict] = []

    # Multiprocessing maps.
    workers: Dict[int, mp.Process] = {}
    devices: Dict[int, torch.device] = {}
    pipes: Dict[int, Pipe] = {}

    # Set spawn start method for compatibility with torch.
    try:
        mp.set_start_method("spawn")
    except RuntimeError as err:
        print("Warning: multiprocessing:", err)

    # TODO: Implement this.
    if args.load_from:

        # Load the environment state from file.
        env.load(env_state_path)

        # Load in multiagent maps.
        agents = trainer_state["agents"]
        rollout_map = trainer_state["rollout_map"]
        minted_agents = trainer_state["minted_agents"]
        metrics = trainer_state["metrics"]

        # Load in dead objects.
        dead_agents = trainer_state["dead_agents"]
        state_dicts = trainer_state["state_dicts"]
        optim_state_dicts = trainer_state["optim_state_dicts"]

        # Don't reset environment if we are resuming a previous run.
        obs = {agent_id: agent.observation for agent_id, agent in env.agents.items()}

    else:
        obs = env.reset()

    # Initialize first policies.
    env_done = False
    step_ema = 1.0
    rew_ema = 0.0
    last_time = time.time()
    for agent_id, ob in obs.items():
        agent, rollouts, worker, device, pipe = get_agent(
            agent_id,
            env.iteration,
            env.agents[agent_id].age,
            ob,
            config,
            env.observation_space,
            env.action_space,
            agents,
            rollout_map,
            dead_agents,
            dead_pipes,
            state_dicts,
            optim_state_dicts,
        )

        # Optionally save a copy of each state dict for reuse in dead policies.
        if config.reuse_state_dicts and agent_id not in agents:
            state_dict = agent.actor_critic.state_dict()
            optim_state_dict = agent.optimizer.state_dict()
            state_dicts.append(copy.deepcopy(state_dict))
            optim_state_dicts.append(copy.deepcopy(optim_state_dict))

        # Copy first observations to rollouts, and send to device.
        if not config.mp:
            initial_observation = torch.FloatTensor([ob])
            rollouts.obs[0].copy_(initial_observation)
            rollouts.to(device)

        agents[agent_id] = agent
        workers[agent_id] = worker
        devices[agent_id] = device
        pipes[agent_id] = pipe
        rollout_map[agent_id] = rollouts

    # Whether or not we make a weight update on this iteration.
    backward_pass: bool = False
    while env.iteration < config.time_steps:

        # Should these all be defined up above with other maps?
        minted_agents = set()
        action_dict: Dict[int, int] = {}
        act_map: Dict[
            int,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        timestep_scores: Dict[int, float] = {}

        # Get actions.
        if config.mp:
            for agent_id in pipes:
                action_dict[agent_id] = pipes[agent_id].action_spout.recv()
        else:
            decay = config.use_linear_lr_decay and backward_pass
            for agent_id in agents:
                act_map[agent_id] = act(
                    env.iteration,
                    decay,
                    agents[agent_id],
                    rollout_map[agent_id],
                    config,
                    env.agents[agent_id].age,
                    None,
                )
                action_dict[agent_id] = int(act_map[agent_id][1][0])

        # Execute environment step.
        obs, rewards, dones, infos = env.step(action_dict)
        backward_pass = env.iteration % config.num_steps == 0 and env.iteration > 0

        # TODO: Check for keyerror: (make sure agent_id is in environment returns/obs).
        if config.mp:
            for agent_id in pipes:
                pipes[agent_id].env_funnel.send(
                    (
                        env.iteration,
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

        # Update EMA of time to execute a single step.
        step_ema = (config.ema_alpha * step_ema) + (
            (1 - config.ema_alpha) * (time.time() - last_time)
        )
        if env.iteration == 0:
            rew_ema = sum(rewards.values())
        rew_ema = config.ema_alpha * rew_ema + (1 - config.ema_alpha) * sum(
            rewards.values()
        )
        last_time = time.time()

        # Print debug output.
        end = "\n" if config.print_repr else "\r"

        print("Iteration: %d| " % env.iteration, end="")
        print("Num agents: %d| " % len(agents), end="")
        print("Total reward: %f| " % rew_ema, end="")
        print("Step EMA: %.6f" % step_ema, end="")
        print("||||||", end=end)

        # Agent creation and termination, rollout stacking.
        for agent_id in obs:
            ob = obs[agent_id]
            reward = rewards[agent_id]
            done = dones[agent_id]
            info = infos[agent_id]

            # Initialize new policies.
            if agent_id not in agents:
                agent, rollouts, worker, device, pipe = get_agent(
                    agent_id,
                    env.iteration,
                    env.agents[agent_id].age,
                    ob,
                    config,
                    env.observation_space,
                    env.action_space,
                    agents,
                    rollout_map,
                    dead_agents,
                    dead_pipes,
                    state_dicts,
                    optim_state_dicts,
                )

                # Copy first observations to rollouts, and send to device.
                if not config.mp:
                    initial_observation = torch.FloatTensor([ob])
                    rollouts.obs[0].copy_(initial_observation)
                    rollouts.to(device)

                agents[agent_id] = agent
                workers[agent_id] = worker
                devices[agent_id] = device
                pipes[agent_id] = pipe
                rollout_map[agent_id] = rollouts
                minted_agents.add(agent_id)

                # Optionally save a copy of each state dict for reuse in dead policies.
                if config.reuse_state_dicts:
                    state_dict = agent.actor_critic.state_dict()
                    optim_state_dict = agent.optimizer.state_dict()
                    state_dicts.append(copy.deepcopy(state_dict))
                    optim_state_dicts.append(copy.deepcopy(optim_state_dict))

            else:

                # If done then remove from environment.
                if done:
                    agent = agents.pop(agent_id)
                    rollout_map.pop(agent_id)
                    pipes.pop(agent_id)
                    dead_agents.add(agent)

                elif not config.mp:
                    rollouts = rollout_map[agent_id]
                    fwds = act_map[agent_id]
                    stack_rollouts(rollouts, ob, reward, done, info, fwds)

        # Print out environment state.
        if all(dones.values()):
            if config.print_repr:
                print("All agents have died.")
            env_done = True

        # Only update losses and save on backward passes.
        if env.iteration % config.num_steps == 0 and env.iteration > 0:

            value_losses: Dict[int, float] = {}
            action_losses: Dict[int, float] = {}
            dist_entropies: Dict[int, float] = {}

            # Should we iterate over a different object?
            for agent_id, agent in agents.items():
                if agent_id not in minted_agents:
                    if config.mp:
                        losses = pipes[agent_id].loss_spout.recv()
                    else:
                        rollouts = rollout_map[agent_id]
                        losses = update(agent, rollouts, config)
                    value_losses[agent_id] = losses[0]
                    action_losses[agent_id] = losses[1]
                    dist_entropies[agent_id] = losses[2]

            metrics = update_losses(
                env=env,
                agents=agents,
                config=config,
                losses=(value_losses, action_losses, dist_entropies),
                metrics=metrics,
                minted_agents=minted_agents,
            )

        # Save for every ``config.save_interval``-th step or on the last update.
        # TODO: Ensure that we aren't saving out an empty state on the last interaction.
        save_state: bool = env.iteration % config.save_interval == 0
        if save_state or env.iteration == config.time_steps - 1:

            # Update ``agents`` and ``rollouts`` from worker processes.
            if config.mp:
                for agent_id, agent in agents.items():
                    agent, rollouts = pipes[agent_id].save_spout.recv()
                    agents[agent_id] = agent
                    rollout_map[agent_id] = rollouts

            # Save trainer state objects
            trainer_state = {
                "agents": agents,
                "rollout_map": rollout_map,
                "minted_agents": minted_agents,
                "metrics": metrics,
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

        env.iteration += 1

    # Prints a single line to reset carriage.
    print("")

    return metrics.policy_score


# TODO: Consider calling these functions in ``worker.py`` as well.
def stack_rollouts(
    rollouts: RolloutStorage,
    ob: torch.Tensor,
    reward: float,
    done: bool,
    info: Dict[str, Any],
    fwds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,],
) -> None:
    # Shape correction and casting.
    # TODO: Change names so everything is statically-typed.
    observation = torch.FloatTensor([ob])
    reward = torch.FloatTensor([reward])
    masks, bad_masks = get_masks(done, info)

    # These are all CUDA tensors (on device).
    value: torch.Tensor = fwds[0]
    action: torch.Tensor = fwds[1]
    action_log_prob: torch.Tensor = fwds[2]
    recurrent_hidden_states: torch.Tensor = fwds[3]

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


def update(
    agent: Algo, rollouts: RolloutStorage, config: Config
) -> Tuple[float, float, float]:
    with torch.no_grad():
        next_value = agent.actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1],
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

    return value_loss, action_loss, dist_entropy
