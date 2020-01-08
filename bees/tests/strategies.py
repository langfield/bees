import json
import pytest
from pprint import pprint
from typing import Dict, List, Any

import hypothesis.strategies as st
from hypothesis import given

from bees.env import Env
from bees.utils import DEBUG
from bees.config import Config

DEBUG = False


@st.composite
def envs(draw) -> Dict[str, Any]:
    """ A hypothesis strategy for generating ``Env`` objects. """

    width = draw(st.integers(min_value=1, max_value=9))
    height = draw(st.integers(min_value=1, max_value=9))
    sight_len = draw(st.integers(min_value=1, max_value=4))
    num_agents = draw(st.integers(min_value=1, max_value=width * height))
    food_density = draw(st.floats(min_value=0.0, max_value=1.0))
    food_size_mean = draw(st.floats(min_value=0.0, max_value=1.0))
    food_size_stddev = draw(st.floats(min_value=0.0, max_value=1.0))
    food_plant_retries = draw(st.integers(min_value=0, max_value=5))
    aging_rate = draw(st.floats(min_value=1e-6, max_value=1.0))
    mating_cooldown_len = draw(st.integers(min_value=0))
    min_mating_health = draw(st.floats(min_value=0.0, max_value=1.0))
    target_agent_density = draw(st.floats(min_value=0.0, max_value=1.0))
    print_repr = draw(st.booleans())
    time_steps = draw(st.integers(min_value=0, max_value=1e9))
    reuse_state_dicts = draw(st.booleans())
    policy_score_frequency = draw(st.integers(min_value=1, max_value=1e9))
    ema_alpha = draw(st.floats(min_value=0.0, max_value=1.0))
    n_layers = draw(st.integers(min_value=1, max_value=3))
    hidden_dim = draw(st.integers(min_value=1, max_value=512))
    reward_weight_mean = draw(st.floats())
    reward_weight_stddev = draw(st.floats(min_value=0.0, max_value=1.0))
    reward_inputs = draw(
        st.lists(st.from_regex(r"actions|obs|health", fullmatch=True), unique=True)
    )
    mut_sigma = draw(st.floats(min_value=0.0, max_value=1.0))
    mut_p = draw(st.floats(min_value=0.0, max_value=1.0))
    algo = draw(st.lists(st.from_regex(r"ppo|a2c|acktr", fullmatch=True), unique=True))
    lr = draw(st.floats(min_value=1e-7, max_value=1.0))
    min_lr = draw(st.floats(min_value=1e-7, max_value=1.0))
    eps = draw(st.floats(min_value=0.0, max_value=1e-2))
    alpha = draw(st.floats(min_value=0.0, max_value=1e-2))
    gamma = draw(st.floats(min_value=0.0, max_value=1.0))
    use_gae = draw(st.booleans())
    gae_lambda = draw(st.floats(min_value=0.0, max_value=1.0))
    value_loss_coef = draw(st.floats(min_value=0.0, max_value=1.0))
    max_grad_norm = draw(st.floats(min_value=0.0, max_value=1.0))
    seed = draw(st.integers(min_value=0, max_value=1e6))
    cuda_deterministic = draw(st.booleans())
    num_processes = draw(st.integers(min_value=1, max_value=10))
    num_steps = draw(st.integers(min_value=1, max_value=1e9))
    ppo_epoch = draw(st.integers(min_value=1, max_value=8))
    num_mini_batch = draw(st.integers(min_value=1, max_value=8))
    clip_param = draw(st.floats(min_value=0.0, max_value=1.0))
    log_interval = draw(st.integers(min_value=1, max_value=1e9))
    save_interval = draw(st.integers(min_value=1, max_value=1e9))
    eval_interval = draw(st.integers(min_value=1, max_value=1e9))
    num_env_steps = draw(st.integers(min_value=1, max_value=1e9))
    cuda = draw(st.booleans())
    use_proper_time_limits = draw(st.booleans())
    recurrent_policy = draw(st.booleans())
    use_linear_lr_decay = draw(st.booleans())

    # Get variable names. It is important that the call to locals() stays at the top
    # of this function, before any other local variables are made.
    arg_names = list(locals())

    # Read settings file for defaults.
    settings_path = "bees/settings/settings.json"
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)

    # Fill settings with values from arguments.
    for arg in arg_names:
        if arg == "draw":
            continue
        settings[arg] = eval(arg)

    if DEBUG:
        print("Settings:")
        pprint(settings)

    config = Config(settings)
    env = Env(config)

    return env
