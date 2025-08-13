import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
r = 0.05          # Risk-free rate
sigma = 0.2       # Volatility
K = 100           # Strike price
T = 1.0           # Time to maturity (in years)
S_max = 250       # Max stock price in spatial domain
S_min = 0         # Min stock price
N = 500           # Number of spatial grid points

# Discretize stock price (space)
S = np.linspace(S_min, S_max, N)
dS = S[1] - S[0]

# Initial condition at maturity for European call option
V_T = np.maximum(S - K, 0)

# Define rhs of PDE
def black_scholes_rhs(t, V):
    dVdt = np.zeros_like(V)

    for i in range(1, N-1):
        delta = (V[i+1] - V[i-1]) / (2*dS)
        gamma = (V[i+1] - 2*V[i] + V[i-1]) / (dS**2)
        
        dVdt[i] = -0.5 * sigma**2 * S[i]**2 * gamma \
                  - r * S[i] * delta + r * V[i]

    # Boundary conditions
    dVdt[0] = 0  # V=0 at S=0
    dVdt[-1] = r * (S[-1] - K) - r * V[-1]  # linear payoff behavior at high S

    return dVdt

# Reverse time interval: solve backward from maturity
sol = solve_ivp(
    fun=black_scholes_rhs,
    t_span=[T, 0],            # backward integration
    y0=V_T,                   # initial value at maturity
    t_eval=np.linspace(T, 0, 200),  # time points to evaluate
    method='RK45'
)




'''
S_grid, T_grid = np.meshgrid(S, sol.t)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, T_grid, sol.y.T, cmap='plasma', alpha=0.8, edgecolor='none')

# Find strike price index
strike_idx = np.argmin(np.abs(S - K))
V_strike = sol.y[strike_idx, :]  # V(S=K, t) across time

# Plot the at-the-money slice
ax.plot(
    [S[strike_idx]] * len(sol.t),  # S is constant
    sol.t,                         # time axis
    V_strike,                      # value at S = K
    color='black',
    linewidth=2,
    linestyle='--',
    label=f'At-the-money (S = K = {K})'
)

# Add annotation label
ax.text(
    S[strike_idx],                # S location
    0.5,                          # t = 0.5 (middle)
    V_strike[len(sol.t)//2],     # value at mid-time
    'Strike Price (S = K)', 
    color='black', fontsize=10
)

# Labels and view angle
ax.set_xlabel('Stock Price S')
ax.set_ylabel('Time t')
ax.set_zlabel('Option Value V(S,t)')
ax.set_title('Black-Scholes Surface: RK4 Solution')
ax.view_init(elev=20, azim=210)

plt.show()
'''