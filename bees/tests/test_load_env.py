import json
import pytest
from typing import Dict, List, Any

import hypothesis.strategies as st
from hypothesis import given

from bees.utils import DEBUG
from bees.config import Config


@given(
    st.integers(min_value=1),  # width
    st.integers(min_value=1),  # height
    st.integers(min_value=1),  # sight_len
    st.integers(min_value=1),  # num_agents
    st.floats(min_value=0.0, max_value=1.0),  # food_density
    st.floats(min_value=0.0, max_value=1.0),  # food_size_mean
    st.floats(min_value=0.0, max_value=1.0),  # food_size_stddev
    st.integers(min_value=0),  # food_plant_retries
    st.floats(min_value=1e-6, max_value=1.0),  # aging_rate
    st.integers(min_value=0),  # mating_cooldown_len
    st.floats(min_value=0.0, max_value=1.0),  # min_mating_health
    st.floats(min_value=0.0, max_value=1.0),  # target_agent_density
    st.booleans(),  # print_repr
    st.integers(min_value=0), # time_steps
    st.booleans(), # reuse_state_dicts
    st.integers(min_value=1), # policy_score_frequency
    st.floats(min_value=0.0, max_value=1.0), # ema_alpha
    st.integers(min_value=1), # n_layers
    st.integers(min_value=1), # hidden_dim
    st.floats(), # reward_weight_mean
    st.floats(min_value=0.0), # reward_weight_stddev
    st.lists(["actions", "obs", "health"], unique=True), # reward_inputs
    st.floats(min_value=0.0), # mut_sigma
    st.floats(min_value=0.0, max_value=1.0), # mut_p
    st.one_of(st.just("ppo"), st.just("a2c"), st.just("acktr")), # algo
    st.floats(min_value=1e-7, max_value=1.0), # lr
    st.floats(min_value=1e-7, max_value=1.0), # min_lr
    st.floats(min_value=0.0, max_value=1e-2), # eps
    st.floats(min_value=0.0, max_value=1e-2), # alpha
    st.floats(min_value=0.0, max_value=1.0), # gamma
    st.booleans(), # use_gae
    st.floats(min_value=0.0, max_value=1.0), # gae_lambda
    st.floats(min_value=0.0, max_value=1.0), # value_loss_coef
    st.floats(min_value=0.0, max_value=1.0), # max_grad_norm
    st.integers(min_value=0), # seed
    st.booleans(), # cuda_deterministic
    st.integers(min_value=1), # num_processes
    st.integers(min_value=1), # num_steps
    st.integers(min_value=1), # ppo_epoch
    st.integers(min_value=1), # num_mini_batch
    st.floats(min_value=0.0, max_value=1.0), # clip_param
    st.integers(min_value=1), # log_interval
    st.integers(min_value=1), # save_interval
    st.integers(min_value=1), # eval_interval
    st.integers(min_value=1), # num_env_steps
    st.booleans(), # cuda
    st.booleans(), # use_proper_time_limits
    st.booleans(), # recurrent_policy
    st.booleans() # use_linear_lr_decay
)
def test_generate_config(
    width: int,
    height: int,
    sight_len: int,
    num_agents: int,
    food_density: float,
    food_size_mean: float,
    food_size_stddev: float,
    food_plant_retries: int,
    aging_rate: float,
    mating_cooldown_len: int,
    min_mating_health: float,
    target_agent_density: float,
    print_repr: bool,
    time_steps: int,
    reuse_state_dicts: bool,
    policy_score_frequency: int,
    ema_alpha: float,
    n_layers: int,
    hidden_dim: int,
    reward_weight_mean: float,
    reward_weight_stddev: float,
    reward_inputs: List[str],
    mut_sigma: float,
    mut_p: float,
    algo: str,
    lr: float,
    min_lr: float,
    eps: float,
    alpha: float,
    gamma: float,
    use_gae: bool,
    gae_lambda: float,
    value_loss_coef: float,
    max_grad_norm: float,
    seed: int,
    cuda_deterministic: bool,
    num_processes: int,
    num_steps: int,
    ppo_epoch: int,
    num_mini_batch: int,
    clip_param: float,
    log_interval: int,
    save_interval: int,
    eval_interval: int,
    num_env_steps: int,
    cuda: bool,
    use_proper_time_limits: bool,
    recurrent_policy: bool,
    use_linear_lr_decay: bool
) -> Dict[str, Any]:
    """ Generates a config object. """

    # Get argument names. It is important that the call to locals() stays at the top
    # of this function, before any other local variables are made.
    arg_names = list(locals())

    # Read settings file for defaults.
    settings_path = "bees/settings/settings.json"
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)

    # Fill settings with values from arguments.
    for arg in arg_names:
        settings[arg] = eval(arg)

    print("Settings:")
    print(settings)

    raise NotImplementedError


def test_env_loads_correctly():
    pass
