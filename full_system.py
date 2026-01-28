import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulation parameters
# -------------------------------
dt = 0.01
total_time = 72  # hours
t = np.arange(0, total_time + dt, dt)

# -------------------------------
# Model parameters
# -------------------------------
alpha1 = 4.0
alpha2 = 2.0
alpha3 = 2.0
alpha4 = 0.6
alpha5 = 0.6
alpha6 = 0.01      # MUCH LOWER to balance slow degradation

beta1 = 1.0
beta2 = 1.0
beta3 = 1.0
beta4 = 0.20
beta5 = np.log(2) / 1.5    # z2 half-life = 1.5 hours ≈ 0.462 hr⁻¹
beta6 = -np.log(0.90) / 24  # 10% degradation per day ≈ 0.00439 hr⁻¹

k = 1.5
n = 10

gamma = 1.0  # NO LIGHT

# -------------------------------
# Initial conditions
# -------------------------------
x0  = 0.255
y10 = 0.9479
z10 = 2.21
y20 = 1.5
z20 = 1.94
m0  = 4.32

print("=== Degradation Parameters ===")
print(f"β₅ (z2) = {beta5:.5f} hr⁻¹  →  half-life = {np.log(2)/beta5:.2f} hours")
print(f"β₆ (m)  = {beta6:.5f} hr⁻¹  →  half-life = {np.log(2)/beta6:.2f} hours ({np.log(2)/beta6/24:.2f} days)")
print(f"α₆/β₆ ratio = {alpha6/beta6:.2f} (determines equilibrium m when z2=1)")
print()

# -------------------------------
# No light
# -------------------------------
def L(t):
    return 0.0 if t%24 < 12 else 1.0

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

# -------------------------------
# Plots
# -------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Core clock oscillators
ax1.plot(t, x, label="Clock protein x", linewidth=2)
ax1.plot(t, z1, label="Clock repressor z1", linewidth=2)
ax1.set_xlabel("Time (h)", fontsize=12)
ax1.set_ylabel("Concentration", fontsize=12)
ax1.set_title("Core Clock Oscillations (No Light)", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: Mitochondrial pathway
ax2.plot(t, y2, label="Intermediate y2", linewidth=2, alpha=0.7)
ax2.plot(t, z2, label="Fast repressor z2 (t₁/₂ = 1.5h)", linewidth=2)
ax2.plot(t, m, label="Mitochondria m (10%/day)", linewidth=2.5)
ax2.set_xlabel("Time (h)", fontsize=12)
ax2.set_ylabel("Concentration", fontsize=12)
ax2.set_title("Mitochondrial Pathway - Now Oscillating!", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Phase portrait
plt.figure(figsize=(7, 7))
plt.plot(z1, x, linewidth=2)
plt.plot(z1[0], x[0], 'go', markersize=10, label='Start')
plt.plot(z1[-1], x[-1], 'ro', markersize=10, label='End')
plt.xlabel("z1 (Repressor)", fontsize=12)
plt.ylabel("x (Clock protein)", fontsize=12)
plt.title("Core Clock Phase Portrait", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Mitochondria vs z2 phase plot
plt.figure(figsize=(7, 7))
plt.plot(z2, m, linewidth=2, alpha=0.7)
plt.plot(z2[0], m[0], 'go', markersize=10, label='Start')
plt.plot(z2[-1], m[-1], 'ro', markersize=10, label='End')
plt.xlabel("z2", fontsize=12)
plt.ylabel("Mitochondria m", fontsize=12)
plt.title("Mitochondria vs z2 Phase Portrait", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# Print final concentrations
print(f"\n{'='*50}")
print(f"FINAL CONCENTRATIONS at t = {total_time} hours")
print(f"{'='*50}")
print(f"x  (Clock protein):     {x[-1]:.6f}")
print(f"y1 (Intermediate 1):    {y1[-1]:.6f}")
print(f"z1 (Clock repressor):   {z1[-1]:.6f}")
print(f"y2 (Intermediate 2):    {y2[-1]:.6f}")
print(f"z2 (Fast repressor):    {z2[-1]:.6f}")
print(f"m  (Mitochondria):      {m[-1]:.6f}")
print(f"{'='*50}\n")

plt.show()