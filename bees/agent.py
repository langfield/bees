#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Agent object for instantiating agents in the environment. """
import copy
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

from asta import Array, dims, shapes, typechecked

from bees.config import Config
from bees.utils import one_hot, get_observation_features

# pylint: disable=bad-continuation, too-many-arguments, too-many-instance-attributes

N_ACTS = dims.N_ACTS


class Agent(Config):
    """
    An agent with position and health attributes. Note that all of the parameters are
    contained in the ``config`` argument.

    Parameters
    ----------
    sight_len : ``int``.
        How far an agent can see in each cardinal direction.
    num_obj_types : ``int``.
        The number of distinct entity classes in the environment. Note that
        we currently have only two (agents, food).
    consts : ``Dict[str, Any]``.
        Dictionary of various constants.
    n_layers : ``int``.
        Number of layers in the reward network.
    hidden_dim : ``int``.
        Hidden dimension of the reward network.
    num_actions : ``int``.
        The number of actions with which the input dimension of the reward
        network is computed.
    pos : ``Tuple[int, int]``, optional.
        Current grid position of the agent.
    initial_health : ``float``, optional.
        Agent's health bar value at creation time.
    reward_weights : ``List[np.ndarray]``, optional.
        Weights of the agent's reward network.
    reward_biases : ``List[np.ndarray]``, optional.
        Biases of the agent's reward network.
    reward_weight_mean : ``float``, optional.
        Mean for weight initialization distribution.
    reward_weight_stddev : ``float``, optional.
        Standard deviation for weight initialization distribution.
    mating_cooldown_len : ``int``, optional.
        How long agent must wait in between mate actions.
    """

    def __init__(
        self,
        config: Config,
        num_actions: int,
        pos: Tuple[int, int],
        initial_health: float,
        reward_weights: Optional[List[np.ndarray]] = None,
        reward_biases: Optional[List[np.ndarray]] = None,
    ) -> None:
        """ __init__ function for Agent class. """

        # Initialize ``__dict__``.
        super().__init__(config.settings, mutable=True)

        # Agent state.
        self.num_actions = num_actions
        self.pos = pos
        self.initial_health = initial_health
        self.mating_cooldown = self.mating_cooldown_len
        self.health = self.initial_health
        self.prev_health = self.initial_health
        self.tabular = config.tabular

        # Set initial agent observation.
        self.obs_width = 2 * config.sight_len + 1
        self.obs_shape = (self.num_obj_types, self.obs_width, self.obs_width)
        self.observation: np.ndarray = np.zeros(self.obs_shape)

        if config.tabular:
            self.input_dim = self.set_reward_network_tabular_input_dim()
        else:
            self.input_dim = self.set_reward_network_input_dim()

        # Initialize/set reward weights and biases.
        if reward_weights is None:
            self.initialize_reward_weights()
        else:
            self.reward_weights = copy.deepcopy(reward_weights)
        if reward_biases is None:
            self.initialize_reward_biases()
        else:
            self.reward_biases = copy.deepcopy(reward_biases)

        # Miscellaneous agent state.
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.policy_score_ema = float("inf")
        self.food_score = float("inf")
        self.age = 0
        self.num_children = 0
        self.is_mature = False
        self.last_action = None

    def set_reward_network_input_dim(self) -> int:
        """ This computes input dim for raw observation input. """
        # Calculate input dimension of reward network.
        # The ``+ 2`` is for the dimensions for current health and previous health.
        input_dim = 0
        if "obs" in self.reward_inputs:
            input_dim += (self.obs_width ** 2) * self.num_obj_types
        if "actions" in self.reward_inputs:
            input_dim += self.num_actions
        if "health" in self.reward_inputs:
            input_dim += 2
        return input_dim

    def set_reward_network_tabular_input_dim(self) -> int:
        """ This computes input dim for tabular feature input. """
        input_dim = 0
        if "obs" in self.reward_inputs:
            input_dim += 2
        if "health" in self.reward_inputs:
            input_dim += 1
        if "actions" in self.reward_inputs:
            input_dim += self.num_actions
        return input_dim

    def initialize_reward_weights(self) -> None:
        """ Initializes the weights of the reward function. """

        self.reward_weights = []
        input_dim = self.input_dim
        output_dim = self.hidden_dim
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                output_dim = 1
            self.reward_weights.append(
                np.random.normal(
                    self.reward_weight_mean,
                    self.reward_weight_stddev,
                    size=(input_dim, output_dim),
                )
            )
            input_dim = output_dim

    def initialize_reward_biases(self) -> None:
        """ Initializes the biases of the reward function. """

        self.reward_biases = []
        output_dim = self.hidden_dim
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                output_dim = 1
            self.reward_biases.append(np.zeros((output_dim,)))

    def compute_reward(self, action: int) -> float:
        """
        Computes agent reward given members of ``self.reward_inputs``.

        Parameters
        ----------
        action : ``int``.
            The current chosen action, represented in integer form (not a tuple).

        Updates
        -------
        self.total_reward : ``float``.
            The total reward summed over the agent's lifetime.

        Returns
        -------
        scalar_reward : ``float``.
            The reward computed via the reward network from the previous
            health, action, and observation of the agent.
        """

        input_arrays: List[np.ndarray] = []

        remaining_inputs = list(self.reward_inputs)
        if "obs" in remaining_inputs:
            if self.tabular:
                ob_features = get_observation_features(self.observation)
            else:
                ob_features = np.array(self.observation).flatten()
            input_arrays.append(ob_features)
            remaining_inputs.remove("obs")
        if "actions" in remaining_inputs:
            flat_action = one_hot(action, self.num_actions)
            input_arrays.append(flat_action)
            remaining_inputs.remove("actions")
        # TODO: Does it even make sense to include previous health??
        if "health" in remaining_inputs:
            flat_healths = np.array([self.prev_health, self.health])
            input_arrays.append(flat_healths)
            remaining_inputs.remove("health")
        if len(remaining_inputs) > 0:
            raise ValueError(
                "Unrecognized inputs to reward network: %s" % str(remaining_inputs)
            )

        inputs = np.concatenate(input_arrays)
        reward = np.copy(inputs)

        for i in range(self.n_layers):
            reward = np.matmul(reward, self.reward_weights[i]) + self.reward_biases[i]
            # ReLU.
            if i < self.n_layers - 1:
                reward = np.maximum(reward, 0)

        scalar_reward: float = float(reward)
        self.total_reward += scalar_reward
        self.last_reward = scalar_reward
        return scalar_reward

    @typechecked
    def get_flat_action(self, action: Tuple[int, int, int]) -> Array[float, N_ACTS + 1]:
        """
        Computes a binary vector (three concatentated one-hot vectors) which
        represents the action.

        Parameters
        ----------
        action : ``Tuple[int, int, int]``.
            The action as a tuple of integers representing the move, eat, and
            mating subactions, respectively.

        Returns
        -------
        action_array : ``np.ndarray``.
            The action as an array of shape ``(num_actions,)`` with three
            nonzero values (k-hot where k is the number of action types).
        """
        action_vec = [0.0 for _ in range(self.num_actions)]
        action_vec[action[0]] = 1.0
        # HARDCODE
        # The 5 here represents the length of the move action space.
        action_vec[5 + action[1]] = 1.0
        # The 2 here represents the length of the eat action space.
        action_vec[5 + 2 + action[2]] = 1.0
        action_array = np.array(action_vec)
        return action_array

    @typechecked
    def reset(self) -> Array[float, shapes.OB]:
        """
        Reset the agent.

        Returns
        -------
        self.observation : ``np.ndarray``.
            The current agent observation.
            Shape: ``(obs_len, obs_len, num_obj_types)``.
        """
        self.health = self.initial_health
        self.prev_health = self.initial_health
        self.total_reward = 0.0
        return self.observation

    def __repr__(self) -> str:
        """
        Returns one line of statistics for the agent.

        Returns
        -------
        output : ``str``.
            Health and total reward of the agent at current timestep.
        """
        output = "Health: %f| " % self.health
        output += "Policy score EMA: %f| " % self.policy_score_ema
        output += "Food score: %f| " % self.food_score
        output += "Total reward: %f\n" % self.total_reward
        return output

    def agent_state(self) -> Dict[str, Any]:
        """
        Returns a state of the agent as a json-style dictionary.
        """

        state = {}
        state_attributes = [
            "pos",
            "initial_health",
            "health",
            "last_reward",
            "age",
            "num_children",
            "last_action",
        ]
        for state_attribute in state_attributes:
            state[state_attribute] = getattr(self, state_attribute)

        return state
