from torch import nn

class RewardNetwork(nn.Module):
    
    def __init__(self, n_layers: int = 3):
        self.n_layers = n_layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear())
