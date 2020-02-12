""" PyTorch environment trainer. """
import os
import time
import json
import copy
import random
import logging
import pickle
import argparse
import collections
from collections import deque, OrderedDict
from typing import Dict, Tuple, Set, List, Any, TextIO
from pprint import pformat

import gym
import torch
import torch.multiprocessing as mp
import numpy as np

from bees.rl import algo, utils
from bees.rl.model import Policy, CNNBase, MLPBase
from bees.rl.storage import RolloutStorage
from bees.rl.algo.algo import Algo

from bees.env import Env
from bees.timer import Timer
from bees.pipe import Pipe
from bees.config import Config
from bees.creation import get_agent, get_policy
from bees.analysis import (
    update_policy_score_multiprocessed,
    update_losses,
    update_food_scores,
    Metrics,
)
from bees.worker import get_policy_score
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

    # This is for compatibility with the parallel trainer.
    assert config.increment

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
    agents: Dict[int, Algo] = {}
    rollout_map: Dict[int, RolloutStorage] = {}
    episode_rewards: Dict[int, collections.deque] = {}
    minted_agents: Set[int] = set()
    agent_action_dists: Dict[int, torch.Tensor] = {}

    # Save dead objects to make creation faster.
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
        dead_agents = trainer_state["dead_agents"]
        state_dicts = trainer_state["state_dicts"]
        optim_state_dicts = trainer_state["optim_state_dicts"]

        # Don't reset environment if we are resuming a previous run.
        obs = {agent_id: agent.observation for agent_id, agent in env.agents.items()}

    else:

        obs = env.reset()

    # Initialize first policies.
    env_done = False
    for agent_id, agent_obs in obs.items():
        if agent_id not in agents:
            agent, rollouts = get_policy(
                config, env.observation_space, env.action_space, device
            )
            agents[agent_id] = agent
            rollout_map[agent_id] = rollouts
            episode_rewards[agent_id] = deque(maxlen=10)

            if config.reuse_state_dicts:
                # Save a copy of the state dict.
                state_dicts.append(copy.deepcopy(agent.actor_critic.state_dict()))
                optim_state_dicts.append(copy.deepcopy(agent.optimizer.state_dict()))

        # Copy first observations to rollouts, and send to device.
        rollouts = rollout_map[agent_id]
        obs_tensor = torch.FloatTensor([agent_obs])
        rollouts.obs[0].copy_(obs_tensor)
        rollouts.to(device)

    timer.end_interval("initialization")
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

            timer.start_interval("act")

            minted_agents = set()
            value_dict: Dict[int, torch.Tensor] = {}
            action_dict: Dict[int, int] = {}
            action_tensor_dict: Dict[int, torch.Tensor] = {}
            action_log_prob_dict: Dict[int, torch.Tensor] = {}
            recurrent_hidden_states_dict: Dict[int, torch.Tensor] = {}
            timestep_scores: Dict[int, float] = {}

            # Sample actions.
            with torch.no_grad():
                for agent_id, agent in agents.items():
                    rollouts = rollout_map[agent_id]
                    ac_tuple = agent.actor_critic.act(
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
            timer.end_interval("act")

            # Execute environment step.
            timer.start_interval("step")
            obs, rewards, dones, infos = env.step(action_dict)
            timer.end_interval("step")

            # Write env state and metrics to log.
            timer.start_interval("logging")
            env.log_state(env_log, visual_log)
            metrics_log.write(str(metrics.get_summary()) + "\n")
            timer.end_interval("logging")

            # Update the policy score.
            timer.start_interval("metrics")
            if env.iteration % config.policy_score_frequency == 0:
                """
                metrics = update_policy_score(
                    env=env,
                    config=config,
                    infos=infos,
                    agent_action_dists=agent_action_dists,
                    metrics=metrics,
                )
                """
                for agent_id, action_dist in agent_action_dists.items():
                    timestep_scores[agent_id] = get_policy_score(
                        action_dist, infos[agent_id]
                    )

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
                    if args.trial.should_prune() or metrics.policy_score == float(
                        "inf"
                    ):
                        print(
                            "\nEnding training because ``policy_score_loss`` diverged."
                        )
                        return metrics.policy_score

            # Update food scores if any agents were born/died this step, and on the
            # first iteration.
            if env.iteration == 1 or set(obs.keys()) != set(agents.keys()):
                metrics = update_food_scores(env, metrics)

            timer.end_interval("metrics")

            # Print debug output.
            timer.start_interval("printing")
            end = "\n" if config.print_repr else "\r"
            print("Iteration: %d| " % env.iteration, end="")
            print("Num agents: %d| " % len(agents), end="")
            print("Policy score loss: %.6f" % metrics.policy_score, end="")
            print("/%.6f| " % metrics.initial_policy_score, end="")
            print("Food score: %.6f" % metrics.food_score, end="")
            print("||||||", end=end)
            timer.end_interval("printing")

            # Agent creation and termination, rollout stacking.
            timer.start_interval("agent creation/removal")
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
                    if len(dead_agents) > 0:

                        agent = dead_agents.pop()

                        if config.reuse_state_dicts:
                            state_dict = copy.deepcopy(random.choice(state_dicts))
                            optim_state_dict = copy.deepcopy(
                                random.choice(optim_state_dicts)
                            )

                            # Load initialized state dicts.
                            agent.actor_critic.load_state_dict(state_dict)
                            agent.optimizer.load_state_dict(optim_state_dict)
                        else:

                            # Reinitialize the policy of ``agent.actor_critic``.
                            if isinstance(agent.actor_critic.base, CNNBase):
                                (
                                    agent.actor_critic.base.main,
                                    agent.actor_critic.base.critic_linear,
                                ) = CNNBase.init_weights(
                                    agent.actor_critic.base.main,
                                    agent.actor_critic.base.critic_linear,
                                )
                            elif isinstance(agent.actor_critic.base, MLPBase):
                                (
                                    agent.actor_critic.base.actor,
                                    agent.actor_critic.base.critic,
                                    agent.actor_critic.base.critic_linear,
                                ) = MLPBase.init_weights(
                                    agent.actor_critic.base.actor,
                                    agent.actor_critic.base.critic,
                                    agent.actor_critic.base.critic_linear,
                                )
                            else:
                                raise NotImplementedError

                        # Create new RolloutStorage object.
                        rollouts = RolloutStorage(
                            config.num_steps,
                            config.num_processes,
                            env.observation_space.shape,
                            env.action_space,
                            agent.actor_critic.recurrent_hidden_state_size,
                        )

                    else:
                        agent, rollouts = get_policy(
                            config, env.observation_space, env.action_space, device
                        )

                        # Save a copy of the state dict.
                        if config.reuse_state_dicts:
                            state_dicts.append(
                                copy.deepcopy(agent.actor_critic.state_dict())
                            )
                            optim_state_dicts.append(
                                copy.deepcopy(agent.optimizer.state_dict())
                            )

                    # Update dicts.
                    agents[agent_id] = agent
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

            timer.end_interval("agent creation/removal")

        timer.start_interval("training")
        value_losses: Dict[int, float] = {}
        action_losses: Dict[int, float] = {}
        dist_entropies: Dict[int, float] = {}
        for agent_id, agent in agents.items():
            if agent_id not in minted_agents:

                rollouts = rollout_map[agent_id]

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
                value_losses[agent_id] = value_loss
                action_losses[agent_id] = action_loss
                dist_entropies[agent_id] = dist_entropy
                rollouts.after_update()

        timer.end_interval("training")
        """
        metrics = update_losses(
            env=env,
            config=config,
            losses=(value_losses, action_losses, dist_entropies),
            metrics=metrics,
        )
        """

        timer.start_interval("saving")

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

        timer.end_interval("saving")
        if env_done:
            break

    summary = timer.get_summary()
    percentages = {interval: summary[interval]["percentage"] for interval in summary}
    print("Time summary:")
    print(pformat(percentages))
    logging.getLogger().info(
        "Steps completed during episode out of total: %d / %d",
        env.iteration,
        config.time_steps,
    )

    return metrics.policy_score
