""" PyTorch environment trainer. """
import os
import json
import time
import copy
import random
import shutil
import logging
import datetime
import argparse
import collections
from collections import deque, OrderedDict
from typing import Dict, Tuple, Set, List, Any
import pickle

import gym
import torch
import numpy as np

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy, CNNBase, MLPBase
from a2c_ppo_acktr.storage import RolloutStorage

# from evaluation import evaluate

from main import create_env
from utils import get_token

# pylint: disable=bad-continuation


def train(settings: Dict[str, Any]) -> None:
    """
    Runs the environment.

    Parameters
    ----------
    settings : ``Dict[str, Any]``.
        Global settings file.
    """
    args = get_args()

    # TOMORROW: A whole bunch of stuff from here until environment creation needs to be
    # reworked. env_filename, trainer_filename, etc. need to be set in the else case,
    # and we should rename those variables in the first case to old_env_filename, etc,
    # so that we can set the new ones. We also should change args.save_dir to
    # args.save_root in a2c/arguments.py, and we need to get rid of all references to
    # settings["logging"]. We may need to do more, but do all of these first and then
    # run it and see.

    """
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

    """

    # TODO: Convert everything to abspaths.
    # Resume from previous run.
    if args.load_from:

        # TODO: assert ``args.load_from`` exists.

        # Construct new codename.
        # NOTE: we were going to have the basename be just the token, but this seems
        # ill-advised since you'd have to go into each folder to determine which is
        # newest.
        codename = os.path.basename(os.path.abspath(args.load_from))
        token = codename.split("_")[0]

        # Construct paths.
        env_filename = codename + "_env.pkl"
        trainer_filename = codename + "_trainer.pkl"
        settings_filename = codename + "_settings.json"
        env_state_path = os.path.join(args.load_from, env_filename)
        trainer_state_path = os.path.join(args.load_from, trainer_filename)
        settings_path = os.path.join(args.load_from, settings_filename)

        # Load trainer state.
        with open(trainer_state_path, "rb") as trainer_file:
            trainer_state = pickle.load(trainer_file)

    # New training run.
    else:
        token = get_token(args.save_root)
        date = str(datetime.datetime.now())
        date = date.replace(" ", "_")
        codename = "%s_%s" % (token, date)

    # TODO: Make args.settings optional; load from old settings file if "".
    if args.settings:
        settings_path = args.settings
    elif args.load_from == "":
        raise ValueError("You must pass a value for ``--settings``.")

    # Load settings dict.
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)

    # Construct a new ``save_dir`` in either case.
    save_dir = os.path.join(args.save_root, codename)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Construct log paths.
    env_log_filename = codename + "_env_log.txt"
    visual_log_filename = codename + "_visual_log.txt"
    env_log_path = os.path.join(save_dir, env_log_filename)
    visual_log_path = os.path.join(save_dir, visual_log_filename)

    # If ``save_dir`` is not the same as ``load_from`` we must copy the existing logs
    # into the new save directory, then contine to append to them.
    if args.load_from and os.path.abspath(save_dir) not in os.path.abspath(
        env_log_path
    ):
        new_env_log_filename = codename + "_env_log.txt"
        new_visual_log_filename = codename + "_visual_log.txt"
        new_env_log_path = os.path.join(save_dir, new_env_log_filename)
        new_visual_log_path = os.path.join(save_dir, new_visual_log_filename)
        shutil.copyfile(env_log_path, new_env_log_path)
        shutil.copyfile(visual_log_path, new_visual_log_path)
        env_log_path = new_env_log_path
        visual_log_path = new_visual_log_path

    # Open logs.
    env_log = open(env_log_path, "a+")
    visual_log = open(visual_log_path, "a+")

    # Create environment.
    args.num_env_steps = settings["trainer"]["time_steps"]
    print("Arguments:", str(args))
    env = create_env(settings)

    if not settings["trainer"]["reuse_state_dicts"]:
        print("Warning: this is slow.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(2)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.load_from:

        # Load the environment state from file.
        env.load(env_state_path)

        # Load in multiagent maps.
        agents = trainer_state["agents"]
        rollout_map = trainer_state["rollout_map"]
        episode_rewards = trainer_state["episode_rewards"]
        minted_agents = trainer_state["minted_agents"]
        value_losses = trainer_state["value_losses"]
        action_losses = trainer_state["action_losses"]
        dist_entropies = trainer_state["dist_entropies"]

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

        # Create multiagent maps.
        actor_critics: Dict[int, Policy] = {}
        agents: Dict[int, "AgentAlgo"] = {}
        rollout_map: Dict[int, RolloutStorage] = {}
        episode_rewards: Dict[int, collections.deque] = {}
        minted_agents: Set[int] = set()
        value_losses: Dict[int, float] = {}
        action_losses: Dict[int, float] = {}
        dist_entropies: Dict[int, float] = {}

        # Save dead objects to make creation faster.
        dead_critics: Set[Policy] = set()
        dead_agents: Set["AgentAlgo"] = set()
        state_dicts: List[OrderedDict] = []
        optim_state_dicts: List[OrderedDict] = []

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

            if settings["trainer"]["reuse_state_dicts"]:
                # Save a copy of the state dict.
                state_dicts.append(copy.deepcopy(actor_critic.state_dict()))
                optim_state_dicts.append(copy.deepcopy(agent.optimizer.state_dict()))

        # Copy first observations to rollouts, and send to device.
        rollouts = rollout_map[agent_id]
        obs_tensor = torch.FloatTensor([agent_obs])
        rollouts.obs[0].copy_(obs_tensor)
        rollouts.to(device)

    start = time.time()
    # MOD
    num_updates = (
        int(args.num_env_steps - env.iteration) // args.num_steps // args.num_processes
    )
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

            t_actions = time.time()
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
                    action_dict[agent_id] = tuple(ac_tuple[1][0].tolist())
                    action_tensor_dict[agent_id] = ac_tuple[1]
                    action_log_prob_dict[agent_id] = ac_tuple[2]
                    recurrent_hidden_states_dict[agent_id] = ac_tuple[3]
            print("Sample actions: %ss" % str(time.time() - t_actions))
            # time.sleep(1)

            t_step = time.time()
            # Observe reward and next obs.
            obs, rewards, dones, infos = env.step(action_dict)
            env.log_state(env_log, visual_log)
            print("Env step: %ss" % str(time.time() - t_step))
            # time.sleep(1)

            # NOTE: we assume ``args.num_processes`` is ``1``.

            t_creation = time.time()
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

                        # DEBUG
                        print(
                            "!!!!!!!!!############ REUSING DEAD POLICY ############!!!!!!!!!!!"
                        )

                        actor_critic = dead_critics.pop()
                        agent = dead_agents.pop()

                        # TOMORROW
                        if settings["trainer"]["reuse_state_dicts"]:
                            state_dict = copy.deepcopy(random.choice(state_dicts))
                            optim_state_dict = copy.deepcopy(
                                random.choice(optim_state_dicts)
                            )

                            # Load initialized state dicts.
                            actor_critic.load_state_dict(state_dict)
                            agent.optimizer.load_state_dict(optim_state_dict)
                        else:

                            # Reinitialize the policy of actor_critic
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
                            args.num_steps,
                            args.num_processes,
                            env.observation_space.shape,
                            env.action_space,
                            actor_critic.recurrent_hidden_state_size,
                        )

                    else:

                        # DEBUG
                        print(
                            "-------------------- CREATING NEW POLICY ----------------------"
                        )

                        agent, actor_critic, rollouts = get_agent(
                            args, env.observation_space, env.action_space, device
                        )

                        # Save a copy of the state dict.
                        if settings["trainer"]["reuse_state_dicts"]:
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
                        # TODO: should we remove from ``rollout_map`` and ``agents``?
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
            print("Creation: %ss" % str(time.time() - t_creation))
            # time.sleep(1)

            # Print out environment state.
            if all(dones.values()):
                if settings["env"]["print"]:
                    print("All agents have died.")
                env_done = True

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

        print("Get value and compute returns:", np.sum(t0_list))
        print("Updates:", np.sum(t1_list))
        # time.sleep(1)

        # save for every interval-th episode or for the last epoch
        if (
            j % args.save_interval == 0 or j == num_updates - 1
        ) and args.save_root != "":

            # Save trainer state objects
            trainer_state = {
                "agents": agents,
                "rollout_map": rollout_map,
                "episode_rewards": episode_rewards,
                "minted_agents": minted_agents,
                "value_losses": value_losses,
                "action_losses": action_losses,
                "dist_entropies": dist_entropies,
                "dead_critics": dead_critics,
                "dead_agents": dead_agents,
                "state_dicts": state_dicts,
                "optim_state_dicts": optim_state_dicts,
            }
            trainer_state_path = os.path.join(save_dir, "%s_trainer.pkl" % codename)
            with open(trainer_state_path, "wb") as trainer_file:
                pickle.dump(trainer_state, trainer_file)

            # Commented out because all of the information in actor_critics is
            # saved in "agents", so there is no need to double save.
            """
            for agent_id, agent in agents.items():
                if agent_id not in minted_agents:
                    actor_critic = actor_critics[agent_id]
                    save_path = os.path.join(save_dir, str(agent_id))
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    # TODO: implement ``ob_rms`` from ``VecNormalize`` in baselines in our env.
                    torch.save(
                        [actor_critic, getattr(env, "ob_rms", None)],
                        os.path.join(save_path, args.env_name + ".pt"),
                    )
            """

            # Save out environment state.
            state_path = os.path.join(save_dir, "%s_env.pkl" % codename)
            env.save(state_path)

            # Save out settings, removing log files (not paths) from object.
            settings_path = os.path.join(save_dir, "%s_settings.json" % codename)
            with open(settings_path, "w") as settings_file:
                json.dump(settings, settings_file)

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

        if env_done:
            break

    logging.getLogger().info(
        "Steps completed during episode: %d / %d" % (env.iteration, args.num_env_steps)
    )


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


if __name__ == "__main__":
    # Get settings and create environment.
    # HARDCODE
    SETTINGS_FILE = "settings/settings.json"
    with open(SETTINGS_FILE, "r") as f:
        SETTINGS = json.load(f)
    train(SETTINGS)
