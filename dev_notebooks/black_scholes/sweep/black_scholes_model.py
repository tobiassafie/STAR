import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class blackScholesPINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=4, neurons_per_layer=64, activation_fn=nn.Tanh()):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(activation_fn)

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation_fn)

        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def initialize_weights(model, method='xavier'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# --- Global Black-Scholes parameters and grids ---
r = 0.05          # Risk-free rate
sigma = 0.2       # Volatility
K = 100           # Strike price
T = 1.0           # Time to maturity (in years)
S_max = 250       # Max stock price in spatial domain
S_min = 0         # Min stock price
N = 500           # Number of spatial grid points

# Create spatial grid
S = torch.linspace(S_min, S_max, N).view(-1, 1).requires_grad_()
t = torch.linspace(0, T, N).view(-1, 1).requires_grad_()