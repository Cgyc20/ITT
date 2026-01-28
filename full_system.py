import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulation parameters
# -------------------------------
dt = 0.01
total_time = 1000  # hours
t = np.arange(0, total_time + dt, dt)

# -------------------------------
# Base rate (sets timescale)
# -------------------------------
alpha1 = 1.5   # BASE RATE

# -------------------------------
# Production rates (PROPORTIONAL)
# -------------------------------
alpha2 = 5.0 * alpha1
alpha3 = 1.0 * alpha1
alpha4 = 5.0 * alpha1
alpha5 = 1.0 * alpha1
alpha6 = 5.0 * alpha1   # mitochondrial production driven by z2

# -------------------------------
# Degradation rates (PROPORTIONAL)
# -------------------------------
beta1 = 0.1 * alpha1
beta2 = 0.1 * alpha1
beta3 = 0.1 * alpha1
beta4 = 0.1 * alpha1
beta5 = 0.1 * alpha1
beta6 = 0.01 * alpha1   # VERY slow mitochondrial degradation

# -------------------------------
# Hill repression parameters
# -------------------------------
k = 1.5
n = 8

gamma = 0.02  # light forcing

# -------------------------------
# Initial conditions
# -------------------------------
x0  = 0.008257
y10 = 0.434943
z10 = 3.932847
y20 = 0.434943
z20 = 3.932847
m0  = 1853.872742

print("=== PROPORTIONAL PARAMETER CHECK ===")
print(f"alpha1 = {alpha1}")
print(f"alpha2 = {alpha2}  (5×alpha1)")
print(f"alpha3 = {alpha3}  (1×alpha1)")
print(f"alpha4 = {alpha4}  (5×alpha1)")
print(f"alpha5 = {alpha5}  (1×alpha1)")
print(f"alpha6 = {alpha6}  (1×alpha1)")
print()
print(f"beta1..beta5 = {beta1}  (0.1×alpha1)")
print(f"beta6 = {beta6}  (0.001×alpha1)")
print(f"Mito half-life = {np.log(2)/beta6:.2f} h = {np.log(2)/beta6/24:.2f} days")
print()

# -------------------------------
# No light
# -------------------------------
def L(t):
    return 1.0 if t%24 <12 else 0.0

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

# Unpack
x, y1, z1, y2, z2, m = state.T

# -------------------------------
# Plots
# -------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(t, x, label="Clock protein x", linewidth=2)
ax1.plot(t, z1, label="Clock repressor z1", linewidth=2)
ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Concentration")
ax1.set_title("Core Clock Oscillations (Proportional rates)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(t, y2, label="Intermediate y2", linewidth=2, alpha=0.7)
ax2.plot(t, z2, label="z2", linewidth=2)
ax2.plot(t, m, label="Mitochondria m", linewidth=2.5)
ax2.set_xlabel("Time (h)")
ax2.set_ylabel("Concentration")
ax2.set_title("Mitochondrial Pathway")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Phase portrait
plt.figure(figsize=(7, 7))
plt.plot(z1, x, linewidth=2)
plt.plot(z1[0], x[0], 'go', label="Start")
plt.plot(z1[-1], x[-1], 'ro', label="End")
plt.xlabel("z1")
plt.ylabel("x")
plt.title("Core Clock Phase Portrait")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Final values
print(f"\nFINAL STATE at t = {total_time} h")
print(f"x  = {x[-1]:.6f}")
print(f"y1 = {y1[-1]:.6f}")
print(f"z1 = {z1[-1]:.6f}")
print(f"y2 = {y2[-1]:.6f}")
print(f"z2 = {z2[-1]:.6f}")
print(f"m  = {m[-1]:.6f}")
