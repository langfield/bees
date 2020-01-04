""" Modifies standard torch distributions so they are compatible with this library. """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bees.utils import DEBUG
from bees.a2c_ppo_acktr.utils import AddBias, init

# pylint: disable=bad-continuation, abstract-method


#
# Standardize distribution interfaces
#


class FixedCategorical(torch.distributions.Categorical):
    r"""
    Fixed parameter categorical distribution.
    CONSTRAINT: Either ``probs`` or ``logits`` must be passed.
    

    Parameters
    ----------
    probs : ``Optional[torch.Tensorn[n]]``.
        The probabilities for each category, where ``n`` is the number of categories.
    logits : ``Optional[torch.Tensor[n]]``.
        The logits ($\log\frac{p}{1 - p}$ where $p$ is the probability) for each
        category, where ``n`` is the number of categories.
    validate_args : ``Optional[bool]``.
        Whether to validate arguments. Does not validate if not passed (functionally
        equivalent to a default of ``False``).
    """

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        """ Samples and unsqueezes. """

        # ===DEBUG===
        unperturbed_sample = super().sample()
        DEBUG(unperturbed_sample)
        unsqueezed_sample = unperturbed_sample.unsqueeze(-1)
        DEBUG(unsqueezed_sample)
        # ===DEBUG===

        return super().sample(sample_shape=sample_shape).unsqueeze(-1)  # ORIGINAL

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """
    Fixed parameter normal distribution.

    Parameters
    ----------
    loc : ``Union[float, torch.Tensor]``.
        The mean.
    scale : ``Union[float, torch.Tensor]``.
        The standard deviation.
    validate_args : ``Optional[bool]``.
        Whether to validate arguments. Does not validate if not passed (functionally
        equivalent to a default of ``False``).
    """

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    # pylint: disable=no-self-use
    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


class FixedBernoulli(torch.distributions.Bernoulli):
    """
    Fixed parameter Bernoulli distribution.
    CONSTRAINT: Either ``probs`` or ``logits`` must be passed.

    Parameters
    ----------
    probs : ``Optional[Union[int, float, torch.Tensor]]``.
        The scalar probability of sampling ``1``.
    logits : ``Optional[Union[int, float, torch.Tensor]]``.
        The scalar log-odds of sampling ``1``.
    validate_args : ``Optional[bool]``.
        Whether to validate arguments. Does not validate if not passed (functionally
        equivalent to a default of ``False``).
    """

    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    # pylint: disable=no-self-use
    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class FixedCategoricalProduct:
    def __init__(self, logits_list=None):
        self.logits_list = logits_list
        self.fixedCategoricals = [
            FixedCategorical(logits=logits) for logits in logits_list
        ]

    def mode(self):
        return torch.stack(
            [categorical.mode().view((1,)) for categorical in self.fixedCategoricals],
            dim=1,
        )

    def sample(self):
        return torch.stack(
            [categorical.sample().view((1,)) for categorical in self.fixedCategoricals],
            dim=1,
        )

    def log_probs(self, actions):
        """
        Parameters
        ----------
        actions : ``torch.Tensor``.
            The actions from ``action_tensor_dict``.
            Shape: ``(num_processes, num_subactions)``.
        """
        num_subactions = actions.shape[1]
        subaction_log_probs_list = []
        for categorical, subaction_index in zip(
            self.fixedCategoricals, range(num_subactions)
        ):
            action = actions[:, subaction_index]
            subaction_log_probs_list.append(categorical.log_probs(action))
        subaction_log_probs = torch.cat(subaction_log_probs_list, dim=-1)

        # Shape: ``(num_subactions,)``.
        log_probabilities = torch.sum(subaction_log_probs, dim=-1)
        return log_probabilities

    def entropy(self):
        entropies = torch.stack(
            [categorical.entropy() for categorical in self.fixedCategoricals]
        )
        ent = torch.sum(entropies)
        return ent


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class CategoricalProduct(nn.Module):
    def __init__(self, num_inputs, num_outputs_list):

        super(CategoricalProduct, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs_list = num_outputs_list
        self.distributions = [
            Categorical(num_inputs, outputs) for outputs in num_outputs_list
        ]

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linears = nn.ModuleList(
            [init_(nn.Linear(num_inputs, outputs)) for outputs in num_outputs_list]
        )

    def forward(self, x):
        logits_list = [linear(x) for linear in self.linears]
        return FixedCategoricalProduct(logits_list=logits_list)
