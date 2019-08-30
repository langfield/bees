""" A dummy policy for bees agents. """
import random

# pylint: disable=too-few-public-methods
class Policy:
    """ Policy class defining random actions. """

    def __init__(self, config: dict) -> None:
        self.consts = config["constants"]

    def get_action(self, _obs, _agent_health):
        """ Returns a random action. """
        move = random.choice(
            [
                self.consts["LEFT"],
                self.consts["RIGHT"],
                self.consts["UP"],
                self.consts["DOWN"],
                self.consts["STAY"],
            ]
        )
        consume = random.choice([self.consts["EAT"], self.consts["NO_EAT"]])

        return (move, consume)
