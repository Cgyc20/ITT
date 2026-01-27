import numpy as np
import matplotlib.pyplot as plt

dt = 0.05
total_time = 240  # 10 days so you can see entrainment

v_0 = 1.0
p_crit = 0.3
alpha_p = 0.6

beta_c = 0.03   # ~23 h half-life (slow activator-like)
beta_p = 0.40   # ~1.7 h half-life (PER2-like order)

gamma = 0.4     # light strength into C
n = 4           # stronger nonlinearity helps (n=2 often too soft)

p_0 = 0.2
c_0 = 0.2


def light(t):
    # 1 for first 12 hours of each 24h cycle, else 0
    return 1.0 if (t % 24) < 12 else 0.0

def deriv(p, c, t):
    L = light(t)
    dcdt = v_0 / (1.0 + (p / p_crit)**n) - beta_c * c
    dpdt = alpha_p * c - beta_p * p - gamma * L
    return dcdt, dpdt

def fwd_euler(p0, c0, dt, total_time):
    t = np.arange(0, total_time + dt, dt)
    p = np.zeros_like(t, dtype=float)
    c = np.zeros_like(t, dtype=float)

    p[0] = p0
    c[0] = c0

    for i in range(len(t) - 1):
        dcdt, dpdt = deriv(p[i], c[i], t[i])
        c[i+1] = c[i] + dt * dcdt
        p[i+1] = p[i] + dt * dpdt

    L = np.array([light(tt) for tt in t])
    return t, p, c, L

t, p, c, L = fwd_euler(p_0, c_0, dt, total_time)

plt.figure()
plt.plot(t, c, label="C(t)")
plt.plot(t, p, label="P(t)")

plt.xlabel("Time (h)")
plt.legend()
plt.show()
