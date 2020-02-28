#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions for agent instantiation. """
import copy
import random
import collections
from typing import Set, List, Dict, Tuple, Optional, Any

import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from bees.rl.algo import Algo, PPO, A2C_ACKTR
from bees.rl.model import Policy, CNNBase, MLPBase
from bees.rl.storage import RolloutStorage

from bees.pipe import Pipe
from bees.config import Config
from bees.worker import worker_loop


def get_agent(
    agent_id: int,
    iteration: int,
    age: int,
    ob: np.ndarray,
    config: Config,
    obs_space: gym.Space,
    act_space: gym.Space,
    agents: Dict[int, Algo],
    rollout_map: Dict[int, RolloutStorage],
    dead_agents: Set[Algo],
    dead_pipes: Dict[int, Pipe],
    state_dicts: List[collections.OrderedDict],
    optim_state_dicts: List[collections.OrderedDict],
) -> Tuple[Algo, RolloutStorage, Optional[mp.Process], Optional[Pipe], torch.device]:
    """ Create an ``Algo`` object, and return pipes and workers if multiprocessed. """

    # REMOVE
    device = torch.device("cuda:0" if config.cuda else "cpu")
    pipe: Optional[Pipe]

    if config.mp:
        pipe = Pipe()
    else:
        worker = None
        pipe = None

    # Test whether we can reuse previously instantiated policy
    # objects, or if we need to create new ones.
    if len(dead_agents) > 0:

        # TODO: Pass in dead workers.
        agent = dead_agents.pop()

        # TODO: Consider reusing dead pipes.

        # TODO: Is this expensive?
        if config.reuse_state_dicts:
            state_dict = copy.deepcopy(random.choice(state_dicts))
            optim_state_dict = copy.deepcopy(random.choice(optim_state_dicts))

            # TODO: We may need to send these in a pipe.
            # TODO: This is bound to be expensive.
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
                    agent.actor_critic.base.main, agent.actor_critic.base.critic_linear,
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
        # TODO: Is this any slower than clearing data from the old one?
        rollouts = RolloutStorage(
            config.num_steps,
            config.num_processes,
            obs_space.shape,
            act_space,
            agent.actor_critic.recurrent_hidden_state_size,
        )

    else:
        # TODO: Do device assignment here.
        if agent_id not in agents:
            agent, rollouts = get_policy(
                config=config,
                obs_space=obs_space,
                act_space=act_space,
                base_kwargs=config.base_kwargs,
            )
        else:
            agent = agents[agent_id]
            rollouts = rollout_map[agent_id]

    # Create worker processes (need to implement worker reuse).
    # TODO: Consider calling ``get_policy`` in ``worker_loop()``.
    if config.mp and pipe:
        worker = mp.Process(
            target=worker_loop,
            kwargs={
                "device": device,
                "agent": agent,
                "rollouts": rollouts,
                "config": config,
                "initial_age": age,
                "initial_iteration": iteration,
                "initial_ob": ob,
                "env_spout": pipe.env_spout,
                "action_funnel": pipe.action_funnel,
                "action_dist_funnel": pipe.action_dist_funnel,
                "loss_funnel": pipe.loss_funnel,
                "save_funnel": pipe.save_funnel,
            },
        )
        worker.start()

    return agent, rollouts, worker, device, pipe


def get_policy(
    config: Config,
    obs_space: gym.Space,
    act_space: gym.Space,
    device: torch.device,
    base_kwargs: Dict[str, Any],
) -> Tuple[Algo, RolloutStorage]:
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
    base_kwargs : ``Dict[str, Any]
        Arguments to ``self.base`` (either CNNBase or MLPBase).

    Returns
    -------
    agent : ``Algo``.
        Agent object from a2c-ppo-acktr.
    rollouts : ``RolloutStorage``.
        The rollout object.
    """

    actor_critic = Policy(obs_space.shape, act_space, base_kwargs=base_kwargs)
    actor_critic.to(device)
    agent: Algo

    if config.algo == "a2c":
        agent = A2C_ACKTR(
            actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            alpha=config.alpha,
            max_grad_norm=config.max_grad_norm,
        )
    elif config.algo == "ppo":
        agent = PPO(
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
        agent = A2C_ACKTR(
            actor_critic, config.value_loss_coef, config.entropy_coef, acktr=True
        )

    rollouts = RolloutStorage(
        config.num_steps,
        config.num_processes,
        obs_space.shape,
        act_space,
        actor_critic.recurrent_hidden_state_size,
    )
    return agent, rollouts
