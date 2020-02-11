""" Modifies standard torch distributions so they are compatible with this library. """
from typing import List
from itertools import product

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from bees.rl.utils import AddBias, init
from bees.utils import DEBUG

# pylint: disable=bad-continuation, abstract-method, no-member


class FixedCategorical(torch.distributions.Categorical):
    r"""
    Fixed parameter categorical distribution.
    CONSTRAINT: Either ``probs`` or ``logits`` must be passed.

    Parameters
    ----------
    probs : ``Optional[torch.Tensor]``.
        The probabilities for each category, where ``n`` is the number of categories.
        Shape: ``(num_processes, n)``.
    logits : ``Optional[torch.Tensor]``.
        The logits ($\log\frac{p}{1 - p}$ where $p$ is the probability) for each
        category, where ``n`` is the number of categories.
        Shape: ``(num_processes, n)``.
    validate_args : ``Optional[bool]``.
        Whether to validate arguments. Does not validate if not passed (functionally
        equivalent to a default of ``False``).
    """

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        """
        Samples and unsqueezes. We add a dimension to the Tensor at the end of the shape
        to represent the process dimension.

        Parameters
        ----------
        sample_shape : ``torch.Size``, optional.
            The shape in which to request the sample.

        Returns
        -------
        action : ``torch.Tensor``.
            Shape : ``(num_processes, 1)``.
        """
        action = super().sample(sample_shape=sample_shape).unsqueeze(-1)
        return action

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes log probabilities for actions by first getting rid of the process
        dimension, calling the log_prob() function from the parent class, and
        summing over.

        Note that ``super().log_prob()`` returns a tensor of the same shape as its
        argument.

        Parameters
        ----------
        actions : ``torch.Tensor``.
            A tensor of (possibly many) actions.
            Shape : ``(num_processes, 1)`` for forward pass.
                    ``(num_processes * num_steps, 1)`` for backward pass.

        Returns
        -------
        <log_prob> : ``torch.Tensor``.
            The log probabilties of ``actions``.
            Shape : ``(num_processes, 1)`` for forward pass.
                    ``(num_processes * num_steps, 1)`` for backward pass.
        """
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self) -> torch.Tensor:
        """
        Returns the index of the category with highest probability. This is the
        deterministic version of ``self.sample()``.

        Returns
        -------
        action : ``torch.Tensor``.
            Shape : ``(num_processes, 1)``.
        """
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

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of ``actions``.
        Note that ``super().log_prob()`` returns a tensor of the same shape as its
        argument.

        Parameters
        ----------
        actions : ``torch.Tensor``.
            A single agent action tensor.
            Shape: ``(num_processes, 1)``.

        Returns
        -------
        log_prob : ``torch.Tensor``.
            The log probabilties of ``actions``.
            Shape: ``(num_processes, 1)``.
        """
        return super().log_prob(actions).sum(-1, keepdim=True)

    # pylint: disable=no-self-use
    def entrop(self) -> torch.Tensor:
        """
        Computes the entropy of the distribution. It's called ``entrop`` so we don't
        override ``super().entropy()``.

        Returns
        -------
        <entropy> : ``torch.FloatTensor``.
            Shape: ``(,)``.
        """
        return super().entropy().sum(-1)

    def mode(self) -> torch.Tensor:
        """ Returns the mean as a scalar tensor. """
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

    # pylint: disable=no-self-use
    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """ Returns the log probabilities of ``actions``. """
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    # pylint: disable=no-self-use
    def entropy(self) -> torch.Tensor:
        """ Returns distribution entropy. """
        return super().entropy().sum(-1)

    def mode(self) -> torch.Tensor:
        """ Returns the mode of the distribution. """
        return torch.gt(self.probs, 0.5).float()


class FixedCategoricalProduct(Distribution):
    """
    A cartesian product of ``FixedProduct`` distributions.
    ``m`` : number of distributions.

    Parameters
    ----------
    logits_list : ``List[torch.Tensor]``.
        List of logits, one tensor of logits per distribution.
    """

    def __init__(self, logits_list: List[torch.Tensor]):
        self.logits_list = logits_list
        self.fixed_categoricals = [
            FixedCategorical(logits=logits) for logits in logits_list
        ]

        # If the logits are changed, this object will break!
        self.probs = self.compute_probs()

    def mode(self) -> torch.Tensor:
        """
        Returns a tensor of the means of each distribution.

        Returns
        -------
        <means> : ``torch.Tensor``.
            Shape: ``(m,)``.
        """
        return torch.stack(
            [categorical.mode().view((1,)) for categorical in self.fixed_categoricals],
            dim=1,
        )

    def sample(self) -> torch.Tensor:
        """
        Returns a tensor of samples from each distribution.

        Returns
        -------
        <means> : ``torch.Tensor``.
            Shape: ``(m,)``.
        """
        return torch.stack(
            [
                categorical.sample().view((1,))
                for categorical in self.fixed_categoricals
            ],
            dim=1,
        )

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes the log likelihood of each subaction.

        Parameters
        ----------
        actions : ``torch.Tensor``.
            The actions from ``action_tensor_dict``.
            Shape: ``(num_processes, num_subactions)``.

        Returns
        -------
        log_probabilities : ``torch.Tensor``.
            The log probabilities of each action.
            Shape: ``(num_subactions,)``.
        """
        num_subactions = actions.shape[1]
        subaction_log_probs_list = []
        for categorical, subaction_index in zip(
            self.fixed_categoricals, range(num_subactions)
        ):
            action = actions[:, subaction_index]
            subaction_log_probs_list.append(categorical.log_probs(action))
        subaction_log_probs = torch.cat(subaction_log_probs_list, dim=-1)

        # Shape: ``(num_subactions,)``.
        log_probabilities = torch.sum(subaction_log_probs, dim=-1)
        return log_probabilities

    def compute_probs(self) -> torch.Tensor:
        """
        Computes the probability of each action, that is, each tuple of subactions.

        Returns
        -------
        probs : ``torch.Tensor``.
            The probability of each each.
            Shape: Same as the shape of the action space.
            Example: If we have:
            ```
            action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(5), gym.spaces.Discrete(2), gym.spaces.Discrete(2))
            )
            ```
            Then the shape of ``probs`` will be (5, 2, 2), and ``probs[3][1][0]`` will
            be the probability of taking action (3, 1, 0).
        """

        # Compute the shape of ``probs`` and initialize ``probs``.
        num_processes = self.fixed_categoricals[0].probs.shape[0]
        subaction_sizes = [dist.probs.shape[-1] for dist in self.fixed_categoricals]
        probs_shape = [num_processes] + subaction_sizes

        # HARDCODE
        device = torch.device("cuda:0")
        probs = torch.zeros(probs_shape).to(device)

        # Iterate over all actions by taking the product of subaction spaces.
        for action in product(
            *[list(range(subaction_size)) for subaction_size in subaction_sizes]
        ):
            for process in range(num_processes):

                # The probability of an action is the product of probabilities of its
                # subactions.
                probs_index = (process,) + action
                probs[probs_index] = 1.0
                for i, categorical in enumerate(self.fixed_categoricals):
                    probs[probs_index] *= categorical.probs[process][action[i]]

        return probs

    def entropy(self) -> torch.Tensor:
        """
        Computes the summed entropy of all distributions.

        Returns
        -------
        ent : ``torch.Tensor``.
            Shape: ``(,)``.
        """
        entropies = torch.stack(
            [categorical.entropy() for categorical in self.fixed_categoricals]
        )
        ent = torch.sum(entropies)
        return ent


class Categorical(nn.Module):
    """
    Categorical distribution generator module for when ``action_space`` is ``Discrete``.

    Parameters
    ----------
    num_inputs : ``int``.
        Base network output dimension.
    num_outputs : ``int``.
        Action space size (``action_space.n``).
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor) -> Distribution:
        """ Returns an action distribution given a tensor input from base network. """
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """
    A diagonal Gaussian distribution generator module. The forward function takes
    tensors of dimension ``num_inputs`` and returns fixed ``num_outputs``-variate
    diagonal Gaussian distributions with mean vector given by the output of
    ``self.fc_mean`` and std deviations of ``1``.

    Parameters
    ----------
    num_inputs : ``int``.
        The dimension of the input tensors.
    num_outputs : ``int``.
        The dimensionality of the continuous action space.
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x: torch.Tensor) -> Distribution:
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor) -> Distribution:
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class CategoricalProduct(nn.Module):
    def __init__(self, num_inputs: int, num_outputs_list: List[int]):

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

    def forward(self, x: torch.Tensor) -> Distribution:
        logits_list = [linear(x) for linear in self.linears]
        return FixedCategoricalProduct(logits_list=logits_list)
