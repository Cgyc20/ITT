import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
dt = 0.01               # hours
total_time = 100 * 24   # 100 days in hours
t = np.arange(0, total_time + dt, dt)

# =========================================================
# CORE CIRCADIAN CLOCK PARAMETERS
# =========================================================
alpha1 = 2
alpha2 = 0.8
alpha3 = 0.8

beta1 = 0.3
beta2 = 0.5
beta3 = 0.3

k = 1.0
n = 10

# =========================================================
# MITOCHONDRIAL PATHWAY PARAMETERS
# =========================================================
alpha4 = 0.3
alpha5 = 0.6
alpha6 = 0.8

beta4 = 0.4
beta5 = 0.5
beta6 = 0.03

# =========================================================
# TIMESCALE TUNING
# =========================================================
s = 0.43  # slow core clock down

alpha1 *= s
alpha2 *= s
alpha3 *= s
beta1  *= s
beta2  *= s
beta3  *= s

# =========================================================
# LIGHT COUPLING
# =========================================================
# =========================================================
# LIGHT: normal day/night + random night pulses
# =========================================================
gamma = 0.005

def L_baseline(tt):
    # Normal: day=1, night=0
    return 1.0 if (tt % 24) < 12 else 0.0

# Random pulses (only at night)
rng = np.random.default_rng(seed=42)
min_pulse = 10 / 60   # 10 minutes
max_pulse = 2.0       # 1 hour
pulse_amp = 2.0

pulse_windows = []
num_days = int(total_time / 24)

for day in range(num_days):
    night_start = day * 24 + 12
    night_end   = day * 24 + 24

    pulse_duration = rng.uniform(min_pulse, max_pulse)
    pulse_start = rng.uniform(night_start, night_end - pulse_duration)
    pulse_end = pulse_start + pulse_duration
    pulse_windows.append((pulse_start, pulse_end))

def L(tt):
    base = L_baseline(tt)

    # if it's night, allow a pulse
    if base == 0.0:
        for t0, t1 in pulse_windows:
            if t0 <= tt <= t1:
                return pulse_amp
    return base




# =========================================================
# ODE SYSTEM
# =========================================================
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

# =========================================================
# INITIAL CONDITIONS
# =========================================================
state = np.zeros((len(t), 6))
state[0] = np.array([
    0.29131664,
    0.44173577,
    1.15648731,
    0.20619798,
    0.24328239,
    6.61339159
])

# =========================================================
# FORWARD EULER INTEGRATION
# =========================================================
for i in range(len(t) - 1):
    state[i+1] = state[i] + dt * deriv(state[i], t[i])
    state[i+1] = np.maximum(state[i+1], 0.0)

x, y1, z1, y2, z2, m = state.T
t_days = t / 24

# =========================================================
# NICE PLOTS + PHASE PORTRAIT (X vs M)
# =========================================================
last_days = 10
mask_last = t_days >= (t_days[-1] - last_days)

L_arr = np.array([L(tt) for tt in t])

# Time series
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(t_days[mask_last], x[mask_last], linewidth=2.2, label="X (clock)")
ax[0].plot(t_days[mask_last], z1[mask_last], linewidth=2.0, alpha=0.85, label="Z1 (repressor)")
ax[0].set_ylabel("Concentration")
ax[0].set_title(f"LD + Random Night Pulses (last {last_days} days)", fontsize=14)
ax[0].legend(frameon=False)
ax[0].grid(alpha=0.25)

ax[1].plot(t_days[mask_last], m[mask_last], linewidth=2.4, label="M (mitochondria)")
ax[1].set_ylabel("Concentration")
ax[1].legend(frameon=False)
ax[1].grid(alpha=0.25)

ax[2].plot(t_days[mask_last], L_arr[mask_last], linewidth=2.0, label="Light L(t)")
ax[2].set_xlabel("Time (days)")
ax[2].set_ylabel("Light")
ax[2].legend(frameon=False)
ax[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("Final_plots/figures/timeseries_light_pulse.png", dpi=300, bbox_inches='tight')
plt.show()

# Phase portrait: X vs M
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.plot(x[mask_last], m[mask_last], linewidth=2.2)

idx0 = np.where(mask_last)[0][0]
idx1 = np.where(mask_last)[0][-1]
ax.scatter(x[idx0], m[idx0], s=80, marker="o", label="Start (window)")
ax.scatter(x[idx1], m[idx1], s=80, marker="s", label="End (window)")

ax.set_xlabel("X (clock)")
ax.set_ylabel("M (mitochondria)")
ax.set_title(f"Phase portrait: X vs M (last {last_days} days)", fontsize=14)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("Final_plots/figures/phase_light_pulse.png", dpi=300, bbox_inches='tight')
plt.show()
