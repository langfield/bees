import random

from constants import *


class Policy:
    def get_action(self, obs, agent_health):
        move = random.choice(LEFT, RIGHT, UP, DOWN, STAY)
        consume = random.choice(EAT, NO_EAT)

        return (move, consume)
