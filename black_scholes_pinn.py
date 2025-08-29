# pinn_black_scholes.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

def pde_loss(model, S, t, sigma, r):
    """
    Computes the residual of the Black-Scholes PDE.
    """
    S.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.cat((S, t), dim=1)
    V = model(X)

    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]

    residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V
    return torch.mean(residual.pow(2))

def boundary_loss(model, t, S_max, K, r, T):
    """
    Computes the loss at the spatial boundaries S=0 and S=S_max.
    """
    S0 = torch.zeros_like(t)
    S_high = torch.full_like(t, S_max)

    # V(0, t) = 0
    bc_low = model(torch.cat((S0, t), dim=1))
    
    # V(S_max, t) = S_max - K * exp(-r * (T - t))
    bc_high = model(torch.cat((S_high, t), dim=1))
    expected_high = S_max - K * torch.exp(-r * (T - t))

    return torch.mean(bc_low.pow(2)) + torch.mean((bc_high - expected_high).pow(2))

def initial_loss(model, S, K):
    """
    Computes the loss at the initial condition (payoff at t=0).
    Note: In finance, this is the terminal condition at t=T, but for a backward PDE
    solver, it serves as the initial condition for the time-reversed problem.
    This implementation sets it at t=0 for simplicity in the forward pass.
    """
    t0 = torch.zeros_like(S)
    X0 = torch.cat((S, t0), dim=1)

    V_pred = model(X0)
    V_true = torch.clamp(S - K, min=0.0) # Payoff for a European Call

    return torch.mean((V_pred - V_true).pow(2))

class BlackScholesPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Black-Scholes equation.
    """
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=4, neurons_per_layer=64, activation_fn=nn.Tanh()):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(activation_fn)

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation_fn)

        # Output layer
        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def main():
    """
    Main function to set up and run the PINN training.
    """
    # Manually set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Black-Scholes parameters
    r = 0.05          # Risk-free rate
    sigma = 0.2       # Volatility
    K = 100.0         # Strike price
    T = 1.0           # Time to maturity (in years)
    S_max = 250.0     # Max stock price in spatial domain
    S_min = 0.0       # Min stock price
    N = 500           # Number of grid points

    # Create spatial and temporal grids
    S = torch.linspace(S_min, S_max, N).view(-1, 1).requires_grad_()
    t = torch.linspace(0, T, N).view(-1, 1).requires_grad_()
    
    # Initialize model, optimizer, and training parameters
    model = BlackScholesPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 10000
    loss_history = []

    # Weights for each loss component
    pde_weight = 3.547
    bc_weight = 1.854
    ic_weight = 0.168

    print(f'Weights - PDE: {pde_weight}, BC: {bc_weight}, IC: {ic_weight}')

    # Adaptive learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=400, min_lr=1e-7, verbose=False)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Compute losses
        pde_l = pde_loss(model, S, t, sigma, r)
        bc_l = boundary_loss(model, t, S_max, K, r, T)
        ic_l = initial_loss(model, S, K)
        total_loss = (pde_weight * pde_l) + (bc_weight * bc_l) + (ic_weight * ic_l)

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.3f}, PDE: {pde_l.item():.3f}, BC: {bc_l.item():.3f}, IC: {ic_l.item():.3f}, LR: {current_lr:.6f}')

    print("Training finished.")

    # Plotting the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss for Black-Scholes PINN')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('black_scholes_loss.png')
    print("Loss plot saved to 'black_scholes_loss.png'")

if __name__ == "__main__":
    main()