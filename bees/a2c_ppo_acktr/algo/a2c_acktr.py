""" An implementation of A2C and ACTKR in one class. """
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from bees.a2c_ppo_acktr.model import Policy
from bees.a2c_ppo_acktr.storage import RolloutStorage
from bees.a2c_ppo_acktr.algo.algo import Algo
from bees.a2c_ppo_acktr.algo.kfac import KFACOptimizer


# pylint: disable=invalid-name, too-few-public-methods
class A2C_ACKTR(Algo):
    """
    Policy class for A2C and ACKTR.

    Parameters
    ----------
    actor_critic : ``Policy``.
        Policy object.
    """

    # TODO: Finish docstring.
    def __init__(
        self,
        actor_critic: Policy,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        alpha: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        acktr: Optional[bool] = False,
    ):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha
            )

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        """
        Performs weight update.

        Parameters
        ----------
        rollouts : ``RolloutStorage``.
            The rollout object containing experience.

        Returns
        -------
        value_loss.item() : ``float``.
            The value loss scalar torch.Tensor casted to a float.
        action_loss.item() : ``float``.
            The action loss scalar torch.Tensor casted to a float.
        dist_entropy.item() : ``float``.
            The distribution entropy scalar torch.Tensor casted to a float.
        """
        # TODO: Write detailed explanation of function in docstring and comment better.
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size
            ),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (
            value_loss * self.value_loss_coef
            + action_loss
            - dist_entropy * self.entropy_coef
        ).backward()

        if not self.acktr:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
