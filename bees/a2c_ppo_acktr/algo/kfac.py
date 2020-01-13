import math
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bees.a2c_ppo_acktr.model import Policy
from bees.a2c_ppo_acktr.utils import AddBias

# pylint: disable=too-many-arguments

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


def _extract_patches(
    x: torch.Tensor,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
) -> torch.Tensor:
    if padding[0] + padding[1] > 0:
        x = F.pad(
            x, (padding[1], padding[1], padding[0], padding[0])
        ).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(
    a: torch.Tensor,
    classname: str,
    layer_info: Optional[Tuple[Tuple[int, ...], ...]],
    fast_cnn: bool,
) -> torch.Tensor:
    batch_size = a.size(0)

    if classname == "Conv2d":
        if fast_cnn:
            a = _extract_patches(a, *layer_info)  # type: ignore
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)  # type: ignore
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == "AddBias":
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(
    g: torch.Tensor,
    classname: str,
    _layer_info: Optional[Tuple[Tuple[int, ...], ...]],
    fast_cnn: bool,
) -> torch.Tensor:
    batch_size = g.size(0)

    if classname == "Conv2d":
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == "AddBias":
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(
    aa: torch.Tensor, m_aa: torch.Tensor, momentum: float
) -> torch.Tensor:
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= 1 - momentum


class SplitBias(nn.Module):
    def __init__(self, module: nn.Module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, bias_input: torch.Tensor) -> torch.Tensor:
        x = self.module(bias_input)
        x = self.add_bias(x)
        return x


class KFACOptimizer(optim.Optimizer):
    def __init__(
        self,
        model: Policy,
        lr: float = 0.25,
        momentum: float = 0.9,
        stat_decay: float = 0.99,
        kl_clip: float = 0.001,
        damping: float = 1e-2,
        weight_decay: float = 0,
        fast_cnn: bool = False,
        Ts: int = 1,
        Tf: int = 10,
    ):
        defaults: Dict[str, Any] = dict()

        def split_bias(module: nn.Module) -> None:
            for mname, child in module.named_children():
                if hasattr(child, "bias") and child.bias is not None:

                    # pylint: disable=protected-access
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {"Linear", "Conv2d", "AddBias"}

        self.modules: List[nn.Module] = []
        self.grad_outputs: Dict[Any, Any] = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa: Dict[nn.Module, torch.Tensor] = {}
        self.m_gg: Dict[nn.Module, torch.Tensor] = {}
        self.Q_a: Dict[nn.Module, torch.Tensor] = {}
        self.Q_g: Dict[nn.Module, torch.Tensor] = {}
        self.d_a: Dict[nn.Module, torch.Tensor] = {}
        self.d_g: Dict[nn.Module, torch.Tensor] = {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = optim.SGD(
            model.parameters(), lr=self.lr * (1 - self.momentum), momentum=self.momentum
        )

    def _save_input(self, module: nn.Module, kfac_input: torch.Tensor) -> None:
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == "Conv2d":
                layer_info = (module.kernel_size, module.stride, module.padding)

            aa = compute_cov_a(kfac_input[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(
        self, module: nn.Module, _grad_input: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == "Conv2d":
                layer_info = (module.kernel_size, module.stride, module.padding)

            gg = compute_cov_g(
                grad_output[0].data, classname, layer_info, self.fast_cnn
            )

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self) -> None:
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not (
                    (classname in ["Linear", "Conv2d"]) and module.bias is not None
                ), "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self) -> None:
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for _, m in enumerate(self.modules):
            assert (
                len(list(m.parameters())) == 1
            ), "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                self.d_a[m], self.Q_a[m] = torch.symeig(self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == "Conv2d":
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1
