#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
from bees.rl.utils import init


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class NNBase(nn.Module):
    def __init__(self, recurrent: bool, recurrent_input_size: int, hidden_size: int):
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
    def is_recurrent(self) -> bool:
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self) -> int:
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self) -> int:
        return self._hidden_size

    def _forward_gru(
        self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        recurrent: bool = False,
        hidden_size: int = 512,
    ):
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

    def forward(
        self, inputs: torch.Tensor, rnn_hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        recurrent: bool = False,
        hidden_size: int = 64,
    ):
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

        layers = []
        for module in critic.modules():
            if isinstance(module, nn.Linear):
                layers.append(init_(module))
            elif not isinstance(module, nn.Sequential):
                layers.append(module)
        new_critic = nn.Sequential(*layers)

        new_critic_linear = init_(critic_linear)

        return new_actor, new_critic, new_critic_linear

    def forward(
        self, inputs: torch.Tensor, rnn_hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
