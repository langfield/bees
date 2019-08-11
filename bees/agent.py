from typing import Tuple

class Agent:

    def __init__(self, pos: Tuple[int] = None, health: float = 1) -> None:
        """``health`` ranges in ``[0, 1]``."""
        self.pos = pos
        self.health = health
    
    def reset(self):
        pass 
