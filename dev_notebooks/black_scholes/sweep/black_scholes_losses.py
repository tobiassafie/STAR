import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def pde_loss(model, S, t):
    S.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.cat((S, t), dim=1)
    V = model(X)

    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]

    residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V
    return torch.mean(residual.pow(2))


def boundary_loss(model, t):
    S0 = torch.zeros_like(t)
    S_high = torch.full_like(t, S_max)

    bc_low = model(torch.cat((S0, t), dim=1))
    bc_high = model(torch.cat((S_high, t), dim=1))
    expected_high = S_max - K * torch.exp(-r * (T - t))

    return torch.mean(bc_low.pow(2)) + torch.mean((bc_high - expected_high).pow(2))


def initial_loss(model, S):
    t0 = torch.zeros_like(S)
    X0 = torch.cat((S, t0), dim=1)

    V_pred = model(X0)
    V_true = torch.clamp(S - K, min=0.0)

    return torch.mean((V_pred - V_true).pow(2))

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