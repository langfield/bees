""" Agent object for instantiating agents in the environment. """

from typing import Tuple
from policy import Policy


class Agent:
    """ An agent with position and health attributes. """

    def __init__(
        self, config: dict, pos: Tuple[int, int] = None, initial_health: float = 1
    ) -> None:
        """ ``health`` ranges in ``[0, 1]``. """
        self.sight_len = config["sight_len"]
        self.obj_types = config["obj_types"]
        
        self.pos = pos
        self.initial_health = initial_health
        self.health = self.initial_health

        self.policy = Policy(config)
        self.obs_width = 2 * self.sight_len + 1
        self.obs_shape = (self.obs_width, self.obs_width, self.obj_types)
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
