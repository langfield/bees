""" Agent object for instantiating agents in the environment. """

from typing import Tuple
from policy import Policy


class Agent:
    """ An agent with position and health attributes. """

    def __init__(self, pos: Tuple[int] = None, initial_health: float = 1) -> None:
        """ ``health`` ranges in ``[0, 1]``. """
        self.pos = pos
        self.initial_health = initial_health
        self.health = self.initial_health

        self.policy = Policy()
        self.observation = None
        self.avg_reward = 0.0
        self.age = 0

    def get_action(self):
        """ Uses the policy to choose an action based on the observation. """
        return self.policy.get_action(self.observation, self.health)

    def update_average_reward(self, new_reward):
        self.age += 1
        self.avg_reward = (self.avg_reward * (self.age - 1) + new_reward) / self.age

    def reset(self):
        """ Reset the agent. """
        self.health = self.initial_health
        self.avg_reward = 0.0
        return self.observation

    def __repr__(self):
        output = "Health: %f, " % self.health
        output += "Average reward: %f\n" % self.avg_reward
        return output
        
