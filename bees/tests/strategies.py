""" Custom hypothesis strategies for bees. """
import json
from typing import Dict, Tuple, Callable, Any

import hypothesis
import hypothesis.strategies as st

from bees.env import Env
from bees.config import Config
from bees.analysis import Metrics

# pylint: disable=no-value-for-parameter


@st.composite
def envs(draw: Callable[[st.SearchStrategy], Any]) -> Env:
    """ A hypothesis strategy for generating ``Env`` objects. """

    sample: Dict[str, Any] = {}

    sample["width"] = draw(st.integers(min_value=1, max_value=9))
    sample["height"] = draw(st.integers(min_value=1, max_value=9))
    sample["sight_len"] = draw(st.integers(min_value=1, max_value=4))
    sample["num_agents"] = draw(
        st.integers(min_value=1, max_value=sample["width"] * sample["height"])
    )
    sample["food_density"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["food_size_mean"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["food_size_stddev"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["food_plant_retries"] = draw(st.integers(min_value=0, max_value=5))
    sample["aging_rate"] = draw(st.floats(min_value=1e-6, max_value=1.0))
    sample["mating_cooldown_len"] = draw(st.integers(min_value=0, max_value=1000000))
    sample["target_agent_density"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["print_repr"] = draw(st.booleans())
    sample["time_steps"] = draw(st.integers(min_value=0, max_value=int(1e9)))
    sample["reuse_state_dicts"] = draw(st.booleans())
    sample["policy_score_frequency"] = draw(
        st.integers(min_value=1, max_value=int(1e9))
    )
    sample["ema_alpha"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["n_layers"] = draw(st.integers(min_value=1, max_value=3))
    sample["hidden_dim"] = draw(st.integers(min_value=1, max_value=64))
    sample["reward_weight_mean"] = draw(st.floats(min_value=-2.0, max_value=2.0))
    sample["reward_weight_stddev"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["reward_inputs"] = draw(st.sampled_from(["actions", "obs", "health"]))
    sample["mut_sigma"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["mut_p"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["algo"] = draw(
        st.lists(st.from_regex(r"ppo|a2c|acktr", fullmatch=True), unique=True)
    )
    sample["lr"] = draw(st.floats(min_value=1e-7, max_value=1.0))
    sample["min_lr"] = draw(st.floats(min_value=1e-7, max_value=1.0))
    sample["eps"] = draw(st.floats(min_value=0.0, max_value=1e-2))
    sample["alpha"] = draw(st.floats(min_value=0.0, max_value=1e-2))
    sample["gamma"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["use_gae"] = draw(st.booleans())
    sample["gae_lambda"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["value_loss_coef"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["max_grad_norm"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["seed"] = draw(st.integers(min_value=0, max_value=int(1e6)))
    sample["cuda_deterministic"] = draw(st.booleans())
    sample["num_processes"] = draw(st.integers(min_value=1, max_value=10))
    sample["num_steps"] = draw(st.integers(min_value=1, max_value=int(1e9)))
    sample["ppo_epoch"] = draw(st.integers(min_value=1, max_value=8))
    sample["num_mini_batch"] = draw(st.integers(min_value=1, max_value=8))
    sample["clip_param"] = draw(st.floats(min_value=0.0, max_value=1.0))
    sample["log_interval"] = draw(st.integers(min_value=1, max_value=int(1e9)))
    sample["save_interval"] = draw(st.integers(min_value=1, max_value=int(1e9)))
    sample["eval_interval"] = draw(st.integers(min_value=1, max_value=int(1e9)))
    sample["cuda"] = draw(st.booleans())
    sample["use_proper_time_limits"] = draw(st.booleans())
    sample["recurrent_policy"] = draw(st.booleans())
    sample["use_linear_lr_decay"] = draw(st.booleans())

    # Get variable names. It is important that the call to locals() stays at the top
    # of this function, before any other local variables are made.
    # arg_names = list(locals())

    # Read settings file for defaults.
    settings_path = "bees/settings/settings.json"
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)

    # Fill settings with values from arguments.
    for key, value in sample.items():
        settings[key] = value

    config = Config(settings)
    env = Env(config)

    return env


@st.composite
def grid_positions_and_moves(
    draw: Callable[[st.SearchStrategy], Any]
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
def env_and_metrics(draw: Callable[[st.SearchStrategy], Any]) -> Tuple[Env, Metrics]:
    """ Strategy for ``Env`` instances and ``Metric`` objects. """
    metrics = Metrics()
    env = draw(envs())

    return env, metrics


@st.composite
def grid_positions(
    draw: Callable[[st.SearchStrategy], Any]
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
    draw: Callable[[st.SearchStrategy], Any], values: st.SearchStrategy
) -> Dict[str, Any]:
    """ Returns a strategy for dictionaries to be used in ``st.recursive()``. """
    dictionary: Dict[str, Any] = draw(
        st.dictionaries(keys=st.from_regex(r"[a-zA-Z_-]+"), values=values)
    )
    return dictionary


@st.composite
def settings_dicts(draw: Callable[[st.SearchStrategy], Any]) -> Dict[str, Any]:
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
                    st.lists(st.integers()),
                    st.lists(st.floats()),
                    st.lists(st.text(st.characters())),
                ),
                extend=recursive_extension_dicts,
                max_leaves=4,
            ),
        )
    )
    return settings
