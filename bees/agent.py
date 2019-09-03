""" Agent object for instantiating agents in the environment. """

from typing import Tuple, Dict, Any
from policy import Policy


class Agent:
    """ An agent with position and health attributes. """

    def __init__(
        self,
        sight_len: int,
        obj_types: int,
        consts: Dict[str, Any],
        pos: Tuple[int, int] = None,
        initial_health: float = 1,
    ) -> None:
        """ ``health`` ranges in ``[0, 1]``. """
        self.sight_len = sight_len
        self.obj_types = obj_types

        self.pos = pos
        self.initial_health = initial_health
        self.health = initial_health

        self.policy = Policy(consts)
        self.obs_width = 2 * sight_len + 1
        self.obs_shape = (self.obs_width, self.obs_width, obj_types)
        self.observation = None
        self.total_reward = 0.0

    def get_action(self):
        """ Uses the policy to choose an action based on the observation. """
        return self.policy.get_action(self.observation, self.health)

    def compute_reward(self, prev_health: int) -> int:
        """ Computes agent reward given health value before consumption. """
        agent_rew = self.health - prev_health
        self.total_reward += agent_rew
        return agent_rew

    def reset(self):
        """ Reset the agent. """
        self.health = self.initial_health
        self.total_reward = 0.0
        return self.observation

    def __repr__(self):
        output = "Health: %f, " % self.health
        output += "Total reward: %f\n" % self.total_reward
        return output
