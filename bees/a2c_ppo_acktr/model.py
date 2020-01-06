from typing import Dict, Tuple, Any

import gym
import torch
import torch.nn as nn

from bees.utils import DEBUG
from bees.a2c_ppo_acktr.distributions import (
    Bernoulli,
    Categorical,
    DiagGaussian,
    CategoricalProduct,
)
from bees.a2c_ppo_acktr.base import NNBase, MLPBase, CNNBase


class Policy(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int],
        action_space: gym.spaces.space.Space,
        base: NNBase = None,
        base_kwargs: Dict[str, Any] = None,
    ):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
                obs_shape = (obs_shape[0],)
            else:
                raise NotImplementedError

        self.base = base(obs_shape, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Tuple":
            # Only support ``Tuple`` of ``Discrete`` spaces now.
            for subspace in action_space:
                if subspace.__class__.__name__ != "Discrete":
                    raise NotImplementedError

            subspace_num_outputs = [subspace.n for subspace in action_space]
            self.dist = CategoricalProduct(self.base.output_size, subspace_num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, x):
        pass

    def act(
        self,
        inputs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes a forward pass by passing observation inputs and policy hidden state
        (in the case that the policy is recurrent) to the policy, which subsequently
        returns an action and some other things, which are returned along with the
        log likelihood of that action.

        Note that ``hidden_dim`` is 1 when num_layers is 1.

        Parameters
        ----------
        inputs : ``torch.Tensor``.
            Shape : ``(num_processes,) + obs.shape``.
        rnn_hxs : ``torch.Tensor``.
            Shape : ``(num_processes, hidden_dim)``.
        masks : ``torch.Tensor``.
            Masks for GRU forward pass.
            Shape : ``(num_processes, hidden_dim)``.
        deterministic : ``bool``.
            Whether to sample from or just take the mode of the action distribution.
        """

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        probs = dist.probs()

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # ``dist_entropy`` isn't used at all here. This is identical to the original.
        action_log_probs = dist.log_probs(action)
        _dist_entropy = dist.entropy().mean()

        # Return entire action distribution.

        return value, action, action_log_probs, rnn_hxs, probs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
