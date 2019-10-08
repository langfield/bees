import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = (
    lambda self, actions: log_prob_cat(self, actions.squeeze(-1))
    .view(actions.size(0), -1)
    .sum(-1)
    .unsqueeze(-1)
)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(
    -1, keepdim=True
)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = (
    lambda self, actions: log_prob_bernoulli(self, actions)
    .view(actions.size(0), -1)
    .sum(-1)
    .unsqueeze(-1)
)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()


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

        self.linears = nn.ModuleList([
            init_(nn.Linear(num_inputs, outputs)) for outputs in num_outputs_list
        ])

    def forward(self, x):
        logits_list = [linear(x) for linear in self.linears]
        return FixedCategoricalProduct(logits_list=logits_list)


class FixedCategoricalProduct:
    def __init__(self, logits_list=None):
        self.logits_list = logits_list
        self.fixedCategoricals = [
            FixedCategorical(logits=logits) for logits in logits_list
        ]

    def mode(self):
        return torch.cat([categorical.mode() for categorical in self.fixedCategoricals])

    def sample(self):
        return torch.cat(
            [categorical.sample() for categorical in self.fixedCategoricals]
        )

    def log_probs(self, actions):
        return tuple(
            [categorical.log_probs(actions) for categorical in self.fixedCategoricals]
        )

    def entropy(self):
        return sum([categorical.entropy() for categorical in self.fixedCategoricals])
