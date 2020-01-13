import os
import glob
from typing import Any, Callable

import torch
import torch.nn as nn


class VecNormalize:
    """ Dummy class as a placeholder for Ikostrikov's ``envs.py``. """

    pass


# Get a render function
# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_num_epochs: int,
    initial_lr: float,
    min_lr: float,
) -> float:
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    lr = max(lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def init(
    module: nn.Module,
    weight_init: Callable[[torch.Tensor, int], None],
    bias_init: Callable[[torch.Tensor], Any],
    gain: float = 1,
) -> nn.Module:
    weight_init(module.weight.data, gain=gain)  # type: ignore
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir: str) -> None:
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)
