import random

class Policy:

    def get_action(self, observation):
        move = random.choice(["left", "right", "up", "down", "stay"])
        consume = random.choice(["eat", "noeat"])

        return {"move": move, "consume": consume}
