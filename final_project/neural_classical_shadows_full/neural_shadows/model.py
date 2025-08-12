
import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, width: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
