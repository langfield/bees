""" PyTorch environment trainer. """
import os
import time
import json
import copy
import random
import pickle
import argparse
import collections
from typing import Dict, Set, List, Any, TextIO

import numpy as np

import torch
import torch.multiprocessing as mp

from bees.rl import utils
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.pipe import Pipe
from bees.config import Config
from bees.creation import get_agent
from bees.analysis import (
    update_policy_score_multiprocessed,
    update_losses,
    update_food_scores,
    Metrics,
)
from bees.initialization import Setup

# pylint: disable=bad-continuation, too-many-branches, duplicate-code
# pylint: disable=too-many-statements, too-many-locals

ALPHA = 0.99
DEBUG = False


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

    # Make sure environment allows us to handle incrementing ``env.iteration`` here.
    assert not config.increment

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
    metrics = Metrics()

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
    mp.set_start_method("spawn")

    # TODO: Implement this.
    if args.load_from:
        raise NotImplementedError
    else:
        obs = env.reset()

    # Initialize first policies.
    env_done = False
    step_ema = 1.0
    last_time = time.time()
    for agent_id, ob in obs.items():
        agent, rollouts, worker, device, pipe = get_agent(
            agent_id,
            env.iteration,
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

        agents[agent_id] = agent
        workers[agent_id] = worker
        devices[agent_id] = device
        pipes[agent_id] = pipe
        rollout_map[agent_id] = rollouts

    # Whether or not we make a weight update on this iteration.
    backward_pass: bool = False
    iterations = config.time_steps // config.num_processes
    num_updates = iterations // config.num_steps
    while env.iteration < iterations:

        # Should these all be defined up above with other maps?
        minted_agents = set()
        action_dict: Dict[int, int] = {}
        timestep_scores: Dict[int, float] = {}

        # Get actions.
        for agent_id in pipes:
            action_dict[agent_id] = pipes[agent_id].action_spout.recv()

        # Execute environment step.
        obs, rewards, dones, infos = env.step(action_dict)
        backward_pass = env.iteration % config.num_steps == 0 and env.iteration > 0
        # TODO: Check for keyerror: for agent_id in obs:
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

        # Update the policy score.
        if (env.iteration + 1) % config.policy_score_frequency == 0:
            # TODO: Check for keyerror: for agent_id in infos:
            for agent_id in pipes:
                timestep_scores[agent_id] = pipes[agent_id].action_dist_spout.recv()

            metrics = update_policy_score_multiprocessed(
                env=env,
                config=config,
                infos=infos,
                timestep_scores=timestep_scores,
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

        step_ema = (ALPHA * step_ema) + ((1 - ALPHA) * (time.time() - last_time))
        last_time = time.time()

        # Print debug output.
        end = "\n" if config.print_repr else "\r"

        if not DEBUG:
            print("Iteration: %d| " % env.iteration, end="")
            print("Num agents: %d| " % len(agents), end="")
            print("Policy score loss: %.6f" % metrics.policy_score, end="")
            print("/%.6f| " % metrics.initial_policy_score, end="")
            print("Food score: %.6f|" % metrics.food_score, end="")
            print("Step EMA: %.6f" % step_ema, end="")
            print("||||||", end=end)

        # Agent creation and termination, rollout stacking.
        for agent_id in obs:
            ob = obs[agent_id]
            done = dones[agent_id]

            # Initialize new policies.
            if agent_id not in agents:
                agent, rollouts, worker, device, pipe = get_agent(
                    agent_id,
                    env.iteration,
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
                    # TODO: Should we save a reference to dead ``rollouts``?
                    # TODO: Does garbage collection get these? Use ``del``?
                    rollout_map.pop(agent_id)
                    pipes.pop(agent_id)
                    dead_agents.add(agent)

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
            for agent_id in agents:
                if agent_id not in minted_agents:
                    losses = pipes[agent_id].loss_spout.recv()
                    value_losses[agent_id] = losses[0]
                    action_losses[agent_id] = losses[1]
                    dist_entropies[agent_id] = losses[2]

            metrics = update_losses(
                env=env,
                config=config,
                losses=(value_losses, action_losses, dist_entropies),
                metrics=metrics,
            )

            # Save for every ``config.save_interval``-th step or on the last update.
            save_state: bool = env.iteration % config.save_interval == 0
            update_index: int = env.iteration // config.num_steps
            if (save_state or update_index == num_updates - 1) and args.save_root:

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

    return metrics.policy_score
