#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Utilities for evolution of reward functions. """
from typing import List, Tuple
import numpy as np
from deap.tools import cxOnePoint, mutGaussian
from bees.agent import Agent

# pylint: disable=invalid-name, bad-continuation
def reward_to_DNA(
    reward_weights: List[np.ndarray], reward_biases: List[np.ndarray]
) -> np.ndarray:
    """
    Takes as input reward weights and biases and flattens
    then into a DNA sequence.

    Parameters
    ----------
    reward_weights : ``List[np.ndarray]``.
        Weights of an agent's reward network.
    reward_biases : ``List[np.ndarray]``.
        Biases of an agent's reward network.

    Returns
    -------
    dna : ``np.ndarray``.
        Reward vector.
        Shape: (?,).
    """
    # TODO: Add shapes to docstrings.
    dna_segments = []
    for weight_array, bias_vector in zip(reward_weights, reward_biases):

        # Make bias vector 2-dimensional for concatenation.
        bias_array = np.reshape(bias_vector, (1, -1))
        reward_weights_and_biases = np.concatenate((weight_array, bias_array))

        # Flatten in column major order to keep weights of individual neurons together.
        dna_segment = reward_weights_and_biases.flatten(order="F")
        dna_segments.append(dna_segment)
    dna = np.concatenate(dna_segments)
    return dna


def DNA_to_reward(
    dna: np.ndarray, n_layers: int, input_dim: int, hidden_dim: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Takes as input a flat dna sequence, the input dimension of reward network
    of the agent, the hidden dimension of the reward network, and the number of
    layers in the reward network, and outputs a tuple of the reward weights and
    reward biases.

    Parameters
    ----------
    dna : ``np.ndarray``.
        Reward vector.
    n_layers : ``int``.
        Number of layers in the reward network.
    hidden_dim : ``int``.
        Hidden dimension of the reward network.
    hidden_dim : ``int``.
        Input dimension of the reward network, where the input is a concatenated
        vector of a flattened observation, actions, and prev/current health values.

    Returns
    -------
    reward_weights : ``List[np.ndarray]``.
        Weights of an agent's reward network.
    reward_biases : ``List[np.ndarray]``.
        Biases of an agent's reward network.
    """
    reward_weights = []
    reward_biases = []
    start = 0
    for i in range(n_layers):

        # Get number of rows and columns in each layer.
        if i == 0 == n_layers - 1:
            num_rows = input_dim
            num_cols = 1
        elif i == n_layers - 1:
            num_rows = hidden_dim
            num_cols = 1
        elif i == 0:
            num_rows = input_dim
            num_cols = hidden_dim
        else:
            num_rows = hidden_dim
            num_cols = hidden_dim

        # Extract single segment from ``dna`` sequence.
        seg_len = (num_rows + 1) * num_cols
        segment = dna[start : start + seg_len]
        segment_array = np.reshape(segment, (num_rows + 1, num_cols))

        # Construct weights and biases from segment.
        weight_array = segment_array[:-1]
        bias_vector = segment_array[-1]

        reward_weights.append(weight_array)
        reward_biases.append(bias_vector)
        start += seg_len
    return reward_weights, reward_biases


def get_child_reward_network(
    mom: Agent, dad: Agent, mut_sigma: float = 0.3, mut_p: float = 0.05
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Takes as input a pair of agents, and constructs the child's reward network.

    Parameters
    ----------
    mom : ``Agent``.
        First parent agent.
    dad : ``Agent``.
        First parent agent.
    mut_sigma : ``float``, optional.
        Standard deviation of mutation operation on reward vectors.
    mut_p : ``float``, optional.
        Probability of mutation on reward vectors.

    Returns
    -------
    childs_reward_weights : ``List[np.ndarray]``.
        Weights of the new agent's reward network.
    childs_reward_biases : ``List[np.ndarray]``.
        Biases of the new agent's reward network.

    """
    # Get parents' DNA.
    moms_dna = reward_to_DNA(mom.reward_weights, mom.reward_biases)
    dads_dna = reward_to_DNA(dad.reward_weights, dad.reward_biases)

    # Crossover.
    childs_dna, _ = cxOnePoint(moms_dna, dads_dna)

    # Mutation.
    # HARDCODE: ``mu`` and grabbing first and only tuple element.
    childs_dna = mutGaussian(childs_dna, 0.0, mut_sigma, mut_p)[0]

    # HARDCODE: using ``mom``'s reward hyperparams for child.
    childs_reward_weights, childs_reward_biases = DNA_to_reward(
        childs_dna, mom.n_layers, mom.input_dim, mom.hidden_dim
    )

    return childs_reward_weights, childs_reward_biases
