import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
total_time = 400  # hours

# --- Core oscillator parameters (Goodwin model) ---
alpha1 = 1.0      # max transcription rate of X
K = 0.3
n = 8

gamma1 = 0.20     # decay of X
gamma2 = 0.20     # decay of Y
gamma3 = 0.20     # decay of Z

alpha2 = 0.6      # X -> Y
alpha3 = 0.6      # Y -> Z

# --- Light parameters ---
gammaL = 0.4      # light strength
use_light_on_X = True
use_light_on_Z = False

# Initial conditions
X0, Y0, Z0 = 0.2, 0.2, 0.2

# ---------- Light schedule ----------
t_light_on = 180.0   # DD first, then LD
# ----------------------------------

def light(t):
    if t < t_light_on:
        return 0.0
    return 1.0 if (t % 24) < 12 else 0.0

def deriv(X, Y, Z, t):
    L = light(t)

    repression = alpha1 / (1.0 + (Z / K)**n)

    # Light coupling
    if use_light_on_X:
        repression += gammaL * L

    extra_Z_decay = gammaL * L if use_light_on_Z else 0.0

    dXdt = repression - gamma1 * X
    dYdt = alpha2 * X - gamma2 * Y
    dZdt = alpha3 * Y - (gamma3 + extra_Z_decay) * Z

    return dXdt, dYdt, dZdt

def fwd_euler(X0, Y0, Z0, dt, total_time):
    t = np.arange(0, total_time + dt, dt)
    X = np.zeros_like(t)
    Y = np.zeros_like(t)
    Z = np.zeros_like(t)

    X[0], Y[0], Z[0] = X0, Y0, Z0

    for i in range(len(t) - 1):
        dXdt, dYdt, dZdt = deriv(X[i], Y[i], Z[i], t[i])
        X[i+1] = max(X[i] + dt * dXdt, 0.0)
        Y[i+1] = max(Y[i] + dt * dYdt, 0.0)
        Z[i+1] = max(Z[i] + dt * dZdt, 0.0)

    L = np.array([light(tt) for tt in t])
    return t, X, Y, Z, L

t, X, Y, Z, L = fwd_euler(X0, Y0, Z0, dt, total_time)

plt.figure()
plt.plot(t, X, label="X(t)")
plt.plot(t, Y, label="Y(t)")
plt.plot(t, Z, label="Z(t)")
plt.plot(t, 0.5 * L, label="Light (scaled)")
plt.axvline(t_light_on, linestyle="--", label="Light ON")
plt.xlabel("Time (h)")
plt.legend()
plt.show()

# Phase portrait (limit cycle)
plt.figure()
plt.plot(Z, X)
plt.xlabel("Z")
plt.ylabel("X")
plt.title("Phase portrait (X vs Z)")
plt.show()
