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
alpha1 = 1.5

# Production rates (PROPORTIONAL)
alpha2 = 5.0 * alpha1
alpha3 = 1.0 * alpha1
alpha4 = 5.0 * alpha1
alpha5 = 1.0 * alpha1
alpha6 = 5.0 * alpha1

# Degradation rates (PROPORTIONAL)
beta1 = 0.1 * alpha1
beta2 = 0.1 * alpha1
beta3 = 0.1 * alpha1
beta4 = 0.1 * alpha1
beta5 = 0.1 * alpha1
beta6 = 0.01 * alpha1  # slow-ish mito degradation

# Hill repression
k = 1.5
n = 8

# Light coupling strength
gamma = 0.05  # increase if pulses look too weak

# -------------------------------
# Healthy LD + pulses schedule
# -------------------------------
day_length = 12.0  # 12:12 LD

# Let system entrain first, THEN start pulses
pulse_start_time = 600.0   # hours: pulses begin after entrainment
pulse_zt = 18.0            # ZT18 = mid-night pulse (night is ZT12-24)
pulse_width = 1.0          # hours
pulse_amp = 1.0            # 1=full light, 0.2=dim

# -------------------------------
# Irregular night pulse generator
# -------------------------------
pulse_start_time = 600.0   # pulses begin after stable entrainment (hours)
night_start = 12.0
night_end = 24.0

pulse_prob_per_night = 0.3     # 30% of nights have a pulse
pulse_min_width = 0.3          # hours
pulse_max_width = 1.5          # hours
pulse_amp = 1.0

rng = np.random.default_rng(seed=2)  # reproducible

pulse_windows = []

# loop over nights AFTER pulse_start_time
first_night = int(pulse_start_time // 24)
last_night = int(total_time // 24)

for night in range(first_night, last_night):
    if rng.random() < pulse_prob_per_night:
        # random pulse time within the night
        pulse_start_zt = rng.uniform(night_start, night_end - pulse_min_width)
        pulse_width = rng.uniform(pulse_min_width, pulse_max_width)

        t_start = 24 * night + pulse_start_zt
        t_end = t_start + pulse_width

        pulse_windows.append((t_start, t_end))



def LD(t):
    """Healthy 12:12 LD."""
    return 1.0 if (t % 24.0) < 12.0 else 0.0

def irregular_pulse(t):
    """Return 1 if t is inside any irregular pulse window."""
    for t0, t1 in pulse_windows:
        if t0 <= t <= t1:
            return pulse_amp
    return 0.0

def L(t):
    """Total light: regular LD + irregular night pulses."""
    return min(1.0, LD(t) + irregular_pulse(t))


# -------------------------------
# Model derivatives
# -------------------------------
def deriv(state, tt):
    x, y1, z1, y2, z2, m = state
    Lt = L(tt)

    dxdt  = alpha1 * (k**n / (k**n + z1**n)) - beta1 * x + gamma * Lt
    dy1dt = alpha2 * x - beta2 * y1
    dz1dt = alpha3 * y1 - beta3 * z1
    dy2dt = alpha4 * x - beta4 * y2
    dz2dt = alpha5 * y2 - beta5 * z2
    dmdt  = alpha6 * z2 - beta6 * m

    return np.array([dxdt, dy1dt, dz1dt, dy2dt, dz2dt, dmdt])

# -------------------------------
# Initial conditions (yours)
# -------------------------------
x0  = 0.008257
y10 = 0.434943
z10 = 3.932847
y20 = 0.434943
z20 = 3.932847
m0  = 1853.872742

state = np.zeros((len(t), 6))
state[0] = np.array([x0, y10, z10, y20, z20, m0])

# -------------------------------
# Forward Euler
# -------------------------------
for i in range(len(t) - 1):
    state[i+1] = state[i] + dt * deriv(state[i], t[i])
    state[i+1] = np.maximum(state[i+1], 0.0)

x, y1, z1, y2, z2, m = state.T
L_arr = np.array([L(tt) for tt in t])


# -------------------------------
# Time series plots
# -------------------------------
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax[0].plot(t, x, label="x (clock)")
ax[0].plot(t, z1, label="z1 (repressor)")
ax[0].plot(t, 0.5*L_arr, "--", label="Light (scaled)", alpha=0.7)
ax[0].axvline(pulse_start_time, linestyle=":", label="pulses begin")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(t, z2, label="z2 (mito regulator)")
ax[1].plot(t, y2, label="y2", alpha=0.7)
ax[1].axvline(pulse_start_time, linestyle=":")
ax[1].legend()
ax[1].grid(alpha=0.3)

ax[2].plot(t, m, label="m (mitochondria)")
ax[2].axvline(pulse_start_time, linestyle=":")
ax[2].legend()
ax[2].grid(alpha=0.3)

ax[2].set_xlabel("Time (h)")
plt.tight_layout()
plt.show()

# -------------------------------
# Phase plane comparison: before vs after pulses
# -------------------------------
# Choose windows that are well after transients in each regime
pre_window = (t > (pulse_start_time - 200)) & (t < (pulse_start_time - 50))
post_window = (t > (pulse_start_time + 50)) & (t < (pulse_start_time + 200))

plt.figure(figsize=(8, 7))
plt.plot(z1[pre_window], x[pre_window], label="Stable LD (pre-pulse)", linewidth=2)
plt.plot(z1[post_window], x[post_window], label="With night pulses (post)", linewidth=2)
plt.xlabel("z1")
plt.ylabel("x")
plt.title("Phase plane: z1 vs x (before vs after pulses)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
