# heston_pinn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import time

def pde_loss(model, S, v, t, r, kappa, theta, sigma_v, rho):
    """
    Calculates the residual of the Heston PDE.
    """
    C = model(S, v, t)
    
    # First derivatives
    C_t = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_S = torch.autograd.grad(C, S, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_v = torch.autograd.grad(C, v, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    
    # Second derivatives
    C_SS = torch.autograd.grad(C_S, S, grad_outputs=torch.ones_like(C_S), create_graph=True)[0]
    C_vv = torch.autograd.grad(C_v, v, grad_outputs=torch.ones_like(C_v), create_graph=True)[0]
    C_Sv = torch.autograd.grad(C_S, v, grad_outputs=torch.ones_like(C_S), create_graph=True)[0]
    
    # Heston PDE residual
    pde_residual = (
        C_t
        + r * S * C_S
        + kappa * (theta - v) * C_v
        + 0.5 * v * S**2 * C_SS
        + 0.5 * sigma_v**2 * v * C_vv
        + rho * sigma_v * v * S * C_Sv
        - r * C
    )
    
    return torch.mean(pde_residual**2)

def boundary_loss(model, v_boundary, t_boundary, S_max, K, r, T):
    """
    Calculates the loss at the spatial boundaries (S=0 and S=S_max).
    """
    # Loss at S=0 (option is worthless)
    S_zero = torch.zeros_like(t_boundary).requires_grad_()
    C_at_S_zero = model(S_zero, v_boundary, t_boundary)
    loss_S_zero = torch.mean(C_at_S_zero**2)
    
    # Loss at S=S_max (option behaves like S - K*exp(-r(T-t)))
    S_at_max = (torch.ones_like(t_boundary) * S_max).requires_grad_()
    C_at_S_max_pred = model(S_at_max, v_boundary, t_boundary)
    C_at_S_max_true = S_at_max - K * torch.exp(-r * (T - t_boundary))
    loss_S_max = torch.mean((C_at_S_max_pred - C_at_S_max_true)**2)
    
    return loss_S_zero + loss_S_max

def terminal_loss(model, S_terminal, v_terminal, K, T):
    """
    Calculates the loss at the terminal condition (t=T), i.e., the payoff.
    """
    t_terminal = (torch.ones_like(S_terminal) * T).requires_grad_()
    C_pred = model(S_terminal, v_terminal, t_terminal)
    
    # Payoff for a European Call option: max(S - K, 0)
    C_true = torch.clamp(S_terminal - K, min=0)
    
    return torch.mean((C_pred - C_true)**2)

class HestonPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Heston model.
    """
    def __init__(self, input_dim=3, output_dim=1, hidden_dim=4, neurons_per_layer=64, activation_fn=nn.Tanh()):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(activation_fn)

        for _ in range(hidden_dim - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation_fn)

        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, S, v, t):
        """Concatenates inputs and performs a forward pass."""
        x = torch.cat([S, v, t], dim=1)
        return self.net(x)

def main():
    """
    Main function to set up and run the Heston PINN training.
    """
    # Common Parameters
    r = 0.05          # Risk-free rate
    sigma = 0.2       # Black-Scholes volatility (used for anchoring)
    K = 100.0         # Strike price
    T = 1.0           # Time to maturity (in years)

    # Heston specific Parameters
    theta = sigma**2  # Long-term variance (0.04)
    kappa = 2.0       # Rate of mean reversion for variance
    sigma_v = 0.3     # Volatility of variance ("vol of vol")
    rho = -0.7        # Correlation between asset and variance

    # Domain Setup
    S_min, S_max = 0.0, 250.0
    v_min, v_max = 0.0, 1.0
    t_min, t_max = 0.0, T

    # Hyperparameters
    hidden_dim = 2
    num_neurons = 256
    num_epochs = 7000
    learning_rate = 0.0024

    # Number of points to sample for each loss component
    N_pde = 2500
    N_boundary = 500
    N_terminal = 500

    # Loss weights
    pde_weight = 9.188
    boundary_weight = 0.092
    terminal_weight = 0.05

    # Model, Optimizer, Scheduler
    model = HestonPINN(
        hidden_dim=hidden_dim,
        neurons_per_layer=num_neurons,
        activation_fn=nn.Tanh()
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # Training Loop
    print("Starting training for Heston PINN...")
    start_time = time.time()

    loss_history = {'total': [], 'pde': [], 'boundary': [], 'terminal': []}

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 1. PDE Loss (Interior Points)
        S_pde = (torch.rand(N_pde, 1) * (S_max - S_min) + S_min).requires_grad_()
        v_pde = (torch.rand(N_pde, 1) * (v_max - v_min) + v_min).requires_grad_()
        t_pde = (torch.rand(N_pde, 1) * (t_max - t_min) + t_min).requires_grad_()
        loss_pde = pde_loss(model, S_pde, v_pde, t_pde, r, kappa, theta, sigma_v, rho)

        # 2. Boundary Loss (S=0 and S=S_max)
        v_bc = (torch.rand(N_boundary, 1) * (v_max - v_min) + v_min).requires_grad_()
        t_bc = (torch.rand(N_boundary, 1) * (t_max - t_min) + t_min).requires_grad_()
        loss_bc = boundary_loss(model, v_bc, t_bc, S_max, K, r, T)

        # 3. Terminal Loss (t=T)
        S_tc = (torch.rand(N_terminal, 1) * (S_max - S_min) + S_min).requires_grad_()
        v_tc = (torch.rand(N_terminal, 1) * (v_max - v_min) + v_min).requires_grad_()
        loss_tc = terminal_loss(model, S_tc, v_tc, K, T)
        
        # Combine losses
        total_loss = (pde_weight * loss_pde) + (boundary_weight * loss_bc) + (terminal_weight * loss_tc)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(loss_pde.item())
        loss_history['boundary'].append(loss_bc.item())
        loss_history['terminal'].append(loss_tc.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4e}, "
                  f"PDE: {loss_pde.item():.4e}, BC: {loss_bc.item():.4e}, TC: {loss_tc.item():.4e}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Plot Losses
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, num_epochs + 1)
    
    plt.plot(epochs_range, loss_history['total'], label='Total Loss')
    plt.plot(epochs_range, loss_history['pde'], label='PDE Loss', linestyle='--')
    plt.plot(epochs_range, loss_history['boundary'], label='Boundary Loss', linestyle='--')
    plt.plot(epochs_range, loss_history['terminal'], label='Terminal Loss', linestyle='--')
    
    plt.title('Heston PINN Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('heston_loss.png')
    print("Loss plot saved to 'heston_loss.png'")

if __name__ == "__main__":
    main()