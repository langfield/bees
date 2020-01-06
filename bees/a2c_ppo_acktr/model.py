from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bees.utils import DEBUG
from bees.a2c_ppo_acktr.distributions import (
    Bernoulli,
    Categorical,
    DiagGaussian,
    CategoricalProduct,
)
from bees.a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
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

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

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
        dist_entropy = dist.entropy().mean()

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


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        # ``input_shape`` is the shape of the input in CWH format.
        # ``inputs``, one of the params of forward call.
        kernel_size = 3
        channels = 32
        input_channels, input_width, input_height = input_shape
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 2 * channels, kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * channels, 4 * channels, kernel_size, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(input_width * input_height * 4 * channels, hidden_size),
            nn.ReLU(),
        )

        # Output dimension is ``1`` because it's computing discounted future reward
        # (i.e. value function).
        self.critic_linear = nn.Linear(hidden_size, 1)

        self.main, self.critic_linear = CNNBase.init_weights(
            self.main, self.critic_linear
        )

        self.train()

    @staticmethod
    def init_weights(
        main: nn.Sequential, critic_linear: nn.Linear
    ) -> Tuple[nn.Sequential, nn.Linear]:
        """ Runs initializers on arguments. """

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        layers: List[nn.Module] = []
        for module in main.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append(init_(module))
            elif not isinstance(module, nn.Sequential):
                layers.append(module)
        new_main = nn.Sequential(*layers)

        init_critic = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        new_critic_linear = init_critic(critic_linear)

        return new_main, new_critic_linear

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=64):
        num_inputs = input_shape[0]
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.critic_linear = nn.Linear(hidden_size, 1)

        self.actor, self.critic, self.critic_linear = MLPBase.init_weights(
            self.actor, self.critic, self.critic_linear
        )

        self.train()

    @staticmethod
    def init_weights(
        actor: nn.Sequential, critic: nn.Sequential, critic_linear: nn.Linear
    ) -> Tuple[nn.Sequential, nn.Sequential, nn.Linear]:
        """ Runs initializers on arguments. """

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        layers: List[nn.Module] = []
        for module in actor.modules():
            if isinstance(module, nn.Linear):
                layers.append(init_(module))
            elif not isinstance(module, nn.Sequential):
                layers.append(module)
        new_actor = nn.Sequential(*layers)

        layers: List[nn.Module] = []
        for module in critic.modules():
            if isinstance(module, nn.Linear):
                layers.append(init_(module))
            elif not isinstance(module, nn.Sequential):
                layers.append(module)
        new_critic = nn.Sequential(*layers)

        new_critic_linear = init_(critic_linear)

        return new_actor, new_critic, new_critic_linear

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
