import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulation parameters
# -------------------------------
dt = 0.01
total_time = 240  # hours
t = np.arange(0, total_time + dt, dt)

# -------------------------------
# Model parameters (TUNED FOR SUSTAINED OSCILLATION)
# -------------------------------
alpha1 = 4.0    # Much higher production rate
alpha2 = 2.0    # Faster intermediate dynamics
alpha3 = 2.0
alpha4 = 0.6
alpha5 = 0.6
alpha6 = 0.6

beta1 = 1.0     # Balanced degradation
beta2 = 1.0
beta3 = 1.0
beta4 = 0.20
beta5 = 0.20
beta6 = 0.10

k = 1.5         # Higher threshold for repression
n = 10          # Sharper repression curve

gamma = 0.0     # NO LIGHT

# -------------------------------
# Initial conditions
# -------------------------------
x0  = 0.1
y10 = 0.1
z10 = 2.0       # Start with high repressor
y20 = 0.2
z20 = 0.2
m0  = 0.2

# -------------------------------
# No light
# -------------------------------
def L(t):
    return 0.0

# -------------------------------
# Derivatives
# -------------------------------
def deriv(state, t):
    x, y1, z1, y2, z2, m = state
    Lt = L(t)

    dxdt  = alpha1 * (k**n / (k**n + z1**n)) - beta1 * x + gamma * Lt
    dy1dt = alpha2 * x - beta2 * y1
    dz1dt = alpha3 * y1 - beta3 * z1
    dy2dt = alpha4 * x - beta4 * y2
    dz2dt = alpha5 * y2 - beta5 * z2
    dmdt  = alpha6 * z2 - beta6 * m

    return np.array([dxdt, dy1dt, dz1dt, dy2dt, dz2dt, dmdt])

# -------------------------------
# Forward Euler solver
# -------------------------------
state = np.zeros((len(t), 6))
state[0] = np.array([x0, y10, z10, y20, z20, m0])

for i in range(len(t) - 1):
    state[i+1] = state[i] + dt * deriv(state[i], t[i])
    state[i+1] = np.maximum(state[i+1], 0.0)

# Unpack variables
x, y1, z1, y2, z2, m = state.T
L_arr = np.zeros_like(t)

# -------------------------------
# Plots
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t, x, label="Clock protein x", linewidth=2)
plt.plot(t, z1, label="Clock repressor z1", linewidth=2)
plt.plot(t, m, label="Mitochondria m", linewidth=1.5)
plt.xlabel("Time (h)", fontsize=12)
plt.ylabel("Concentration", fontsize=12)
plt.title("Sustained Autonomous Oscillations (No Light)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Phase portrait of the core clock
plt.figure(figsize=(7, 7))
plt.plot(z1, x, linewidth=2)
plt.plot(z1[0], x[0], 'go', markersize=10, label='Start')
plt.plot(z1[-1], x[-1], 'ro', markersize=10, label='End')
plt.xlabel("z1 (Repressor)", fontsize=12)
plt.ylabel("x (Clock protein)", fontsize=12)
plt.title("Core Clock Phase Portrait (No Light)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


