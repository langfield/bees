""" Agent object for instantiating agents in the environment. """

import copy
from typing import Tuple, Dict, Any, List

import numpy as np

from policy import Policy
from utils import convert_obs_to_tuple

# pylint: disable=bad-continuation

class Agent:
    """
    An agent with position and health attributes.

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

    # TODO: Remove default values from ``__init__`` parameters.

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        sight_len: int,
        num_obj_types: int,
        consts: Dict[str, Any],
        n_layers: int,
        hidden_dim: int,
        num_actions: int,
        pos: Tuple[int, int] = None,
        initial_health: float = 1,
        reward_weights: List[np.ndarray] = None,
        reward_biases: List[np.ndarray] = None,
        reward_weight_mean: float = 0.0,
        reward_weight_stddev: float = 0.4,
        mating_cooldown_len: int = 10,
    ) -> None:
        """ ``health`` ranges in ``[0, 1]``. """
        self.sight_len = sight_len
        self.num_obj_types = num_obj_types
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.pos = pos
        self.initial_health = initial_health
        self.health = initial_health
        self.num_actions = num_actions

        self.reward_weight_mean = reward_weight_mean
        self.reward_weight_stddev = reward_weight_stddev

        # How many timesteps until they can mate again.
        self.mating_cooldown = mating_cooldown_len
        self.mating_cooldown_len = mating_cooldown_len

        # pylint: disable=invalid-name
        # Get constants.
        self.consts = consts

        self.policy = Policy(consts)
        self.obs_width = 2 * sight_len + 1
        self.obs_shape = (self.obs_width, self.obs_width, num_obj_types)
        self.observation = convert_obs_to_tuple(np.zeros(self.obs_shape))

        self.age = 0

        # The ``+ 2`` is for the dimensions for current health and previous health.
        self.input_dim = (self.obs_width ** 2) * num_obj_types + self.num_actions + 2
        self.total_reward = 0.0
        if reward_weights is None:
            self.initialize_reward_weights()
        else:
            self.reward_weights = copy.deepcopy(reward_weights)
        if reward_biases is None:
            self.initialize_reward_biases()
        else:
            self.reward_biases = copy.deepcopy(reward_biases)

    def get_action(self) -> Tuple[int, int, int]:
        """
        Uses the policy to choose an action based on the observation.

        Returns
        -------
        action : ``Tuple[int, int, int]``.
            Randomly generated action for dummy training runs.
        """
        return self.policy.get_action(self.observation, self.health)

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

    def compute_reward(self, prev_health: float, action: Tuple[int, int, int]) -> float:
        """
        Computes agent reward given health value before consumption.

        Parameters
        ----------
        prev_health : ``float``.
            Health on previous timestep.
        action : ``Tuple[int, int, int]``.
            The current chosen action.

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

        flat_obs = np.array(self.observation).flatten()
        flat_action = self.get_flat_action(action)
        flat_healths = np.array([prev_health, self.health])
        inputs = np.concatenate((flat_obs, flat_action, flat_healths))
        reward = np.copy(inputs)

        for i in range(self.n_layers):
            reward = np.matmul(reward, self.reward_weights[i]) + self.reward_biases[i]
            # ReLU.
            if i < self.n_layers - 1:
                reward = np.maximum(reward, 0)

        scalar_reward = np.asscalar(reward)
        self.total_reward += scalar_reward
        return scalar_reward

    def get_flat_action(self, action: Tuple[int, int, int]) -> np.ndarray:
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

    def reset(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        """
        Reset the agent.

        Returns
        -------
        self.observation : ``Tuple[Tuple[Tuple[int, ...], ...], ...]``.
            The current agent observation.
            Shape: ``(obs_len, obs_len, num_obj_types)``.
        """
        self.health = self.initial_health
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
        output = "Health: %f, " % self.health
        output += "Total reward: %f\n" % self.total_reward
        return output
