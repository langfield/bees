#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Custom hypothesis strategies for bees. """
import json
from typing import Dict, Tuple, Callable, Any, Optional

import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy

from bees.env import Env
from bees.utils import flat_action_to_tuple
from bees.config import Config
from bees.analysis import Metrics

# pylint: disable=no-value-for-parameter, protected-access


@st.composite
def bees_settings(draw: Callable[[SearchStrategy], Any]) -> Dict[str, Any]:
    """ A strategy for valid bees settings files. """
    sample: Dict[str, Any] = {}

    # Environment.
    sample["width"] = draw(st.integers(min_value=1, max_value=9))
    sample["height"] = draw(st.integers(min_value=1, max_value=9))
    sample["sight_len"] = draw(st.integers(min_value=1, max_value=4))
    num_squares = sample["width"] * sample["height"]
    sample["num_agents"] = draw(st.integers(min_value=1, max_value=num_squares))
    sample["initial_food_density"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["initial_food_regen_prob"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["adaptive_food_type"] = draw(st.sampled_from(["agent", "", "food"]))
    sample["target_density"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["food_size_mean"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["food_size_stddev"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["food_plant_retries"] = draw(st.integers(min_value=0, max_value=5))
    sample["aging_rate"] = draw(st.floats(min_value=1e-6, max_value=1.0))
    sample["aging_type"] = draw(st.sampled_from(["linear", "quadratic"]))
    sample["mating_cooldown_len"] = draw(st.integers(min_value=0, max_value=100))
    sample["target_agent_density"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["print_repr"] = draw(st.booleans())

    # Trainer.
    sample["time_steps"] = draw(st.integers(min_value=0, max_value=1000))
    sample["reuse_state_dicts"] = draw(st.booleans())
    sample["policy_score_frequency"] = draw(st.integers(min_value=1, max_value=1000))
    sample["ema_alpha"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["mp"] = draw(st.booleans())

    # Reward.
    sample["n_layers"] = draw(st.integers(min_value=1, max_value=3))
    sample["hidden_dim"] = draw(st.integers(min_value=1, max_value=64))
    sample["reward_weight_mean"] = draw(st.floats(min_value=-2.0, max_value=2.0))
    sample["reward_weight_stddev"] = draw(st.floats(min_value=0.0, max_value=1.0))
    reward_inputs = st.sampled_from(["actions", "obs", "health"])
    sample["reward_inputs"] = list(draw(st.frozensets(reward_inputs, min_size=1)))
    sample["tabular"] = draw(st.booleans())

    # Genetics.
    sample["mut_sigma"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["mut_p"] = draw(st.floats(min_value=0.0, max_value=1.0))

    # Policy.
    sample["algo"] = draw(st.sampled_from(["ppo", "a2c"]))
    sample["lr"] = draw(st.floats(min_value=1e-6, max_value=0.2))
    sample["min_lr"] = draw(st.floats(min_value=1e-6, max_value=0.2))
    sample["eps"] = draw(st.floats(min_value=0.0, max_value=1e-2))
    sample["alpha"] = draw(st.floats(min_value=0.0, max_value=1e-2))
    sample["gamma"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["use_gae"] = draw(st.booleans())
    sample["gae_lambda"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["value_loss_coef"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["max_grad_norm"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["seed"] = draw(st.integers(min_value=0, max_value=10))
    sample["cuda_deterministic"] = draw(st.booleans())
    sample["num_processes"] = draw(st.integers(min_value=1, max_value=10))
    sample["num_steps"] = draw(st.integers(min_value=1, max_value=1000))
    sample["ppo_epoch"] = draw(st.integers(min_value=1, max_value=8))
    sample["num_mini_batch"] = draw(st.integers(min_value=1, max_value=8))
    sample["clip_param"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["log_interval"] = draw(st.integers(min_value=1, max_value=100))
    sample["save_interval"] = draw(st.integers(min_value=1, max_value=100))
    sample["cuda"] = draw(st.booleans())
    sample["use_proper_time_limits"] = draw(st.booleans())
    sample["recurrent_policy"] = draw(st.booleans())
    sample["use_linear_lr_decay"] = draw(st.booleans())

    # Read settings file for defaults.
    settings_path = "settings/settings.json"
    with open(settings_path, "r") as settings_file:
        settings: Dict[str, Any] = json.load(settings_file)

    # Fill settings with values from arguments.
    for key, value in sample.items():
        settings[key] = value

    return settings


@st.composite
def envs(draw: Callable[[SearchStrategy], Any]) -> Env:
    """ A hypothesis strategy for generating ``Env`` objects. """
    settings = draw(bees_settings())
    config = Config(settings)
    env = Env(config)

    return env


@st.composite
def grid_positions_and_moves(
    draw: Callable[[SearchStrategy], Any]
) -> Tuple[Env, Tuple[int, int], int]:
    """ Strategy for ``Env`` instances and valid grid positions. """
    env = draw(envs())
    pos: Tuple[int, int] = draw(
        st.tuples(
            st.integers(min_value=0, max_value=env.width - 1),
            st.integers(min_value=0, max_value=env.height - 1),
        )
    )
    valid_moves = [
        env.config.STAY,
        env.config.LEFT,
        env.config.RIGHT,
        env.config.UP,
        env.config.DOWN,
    ]
    move = draw(st.sampled_from(valid_moves))
    return env, pos, move


@st.composite
def env_and_metrics(draw: Callable[[SearchStrategy], Any]) -> Tuple[Env, Metrics]:
    """ Strategy for ``Env`` instances and ``Metric`` objects. """
    metrics = Metrics()
    env = draw(envs())

    return env, metrics


@st.composite
def grid_positions(
    draw: Callable[[SearchStrategy], Any]
) -> Tuple[Env, Tuple[int, int]]:
    """ Strategy for ``Env`` instances and valid grid positions. """
    env = draw(envs())
    pos: Tuple[int, int] = draw(
        st.tuples(
            st.integers(min_value=0, max_value=env.width - 1),
            st.integers(min_value=0, max_value=env.height - 1),
        )
    )
    return env, pos


@st.composite
def recursive_extension_dicts(
    draw: Callable[[SearchStrategy], Any], values: SearchStrategy
) -> Dict[str, Any]:
    """ Returns a strategy for dictionaries to be used in ``st.recursive()``. """
    dictionary: Dict[str, Any] = draw(
        st.dictionaries(keys=st.from_regex(r"[a-zA-Z_-]+"), values=values, max_size=100)
    )
    return dictionary


@st.composite
def settings_dicts(draw: Callable[[SearchStrategy], Any]) -> Dict[str, Any]:
    """ Strategy for settings dicts. """
    settings: Dict[str, Any] = draw(
        st.dictionaries(
            keys=st.from_regex(r"[a-zA-Z_-]+"),
            values=st.recursive(
                base=st.one_of(
                    st.floats(),
                    st.integers(),
                    st.text(st.characters()),
                    st.booleans(),
                    st.lists(st.integers(), max_size=10),
                    st.lists(st.floats(), max_size=10),
                    st.lists(st.text(st.characters()), max_size=10),
                ),
                extend=recursive_extension_dicts,
                max_leaves=3,
            ),
            max_size=100,
        )
    )
    return settings


@st.composite
def empty_positions(
    _draw: Callable[[SearchStrategy], Any], env: Env, obj_type_id: int
) -> Optional[Tuple[int, int]]:
    """ Strategy for grid positions with any objects of type ``obj_type_id``. """
    for x in range(env.width):
        for y in range(env.height):
            if not env._obj_exists(obj_type_id, (x, y)):
                return (x, y)
    return None


@st.composite
def positions(draw: Callable[[SearchStrategy], Any], env: Env) -> Tuple[int, int]:
    pos: Tuple[int, int] = draw(
        st.tuples(
            st.integers(min_value=0, max_value=env.width - 1),
            st.integers(min_value=0, max_value=env.height - 1),
        )
    )
    return pos


@st.composite
def obj_type_ids(draw: Callable[[SearchStrategy], Any], env: Env) -> int:
    obj_type_id: int = draw(st.sampled_from(list(env.obj_type_ids.values())))
    return obj_type_id


@st.composite
def moves(draw: Callable[[SearchStrategy], Any], env: Env) -> int:
    valid_moves = [
        env.config.STAY,
        env.config.LEFT,
        env.config.RIGHT,
        env.config.UP,
        env.config.DOWN,
    ]
    move: int = draw(st.sampled_from(valid_moves))
    return move


@st.composite
def consumptions(draw: Callable[[SearchStrategy], Any], env: Env) -> int:
    valid_consumptions = [
        env.config.EAT,
        env.config.NO_EAT,
    ]
    consumption: int = draw(st.sampled_from(valid_consumptions))
    return consumption


@st.composite
def action_dicts(draw: Callable[[SearchStrategy], Any], env: Env) -> Dict[int, int]:
    """ Strategy for flat action dictionaries. """
    action_dict: Dict[int, int] = {}
    max_action: int = env.action_space.n - 1
    for agent_id in env.agents:
        action_dict[agent_id] = draw(st.integers(min_value=0, max_value=max_action))
    return action_dict


@st.composite
def tuple_action_dicts(
    draw: Callable[[SearchStrategy], Any], env: Env
) -> Dict[int, Tuple[int, int, int]]:
    """ Strategy for tuple action dictionaries. """
    action_dict: Dict[str, int] = draw(action_dicts(env=env))

    # Convert flat action_dict to tuple.
    tuple_action_dict: Dict[int, Tuple[int, int, int]] = {
        agent_id: flat_action_to_tuple(action, env.subaction_sizes)  # type: ignore
        for agent_id, action in action_dict.items()
    }
    return tuple_action_dict
