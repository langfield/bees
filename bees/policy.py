import random

class Policy:

    def get_action(self, obs, agent_health):
        move = random.choice(["left", "right", "up", "down", "stay"])
        consume = random.choice(["eat", "noeat"])

        return {"move": move, "consume": consume}
