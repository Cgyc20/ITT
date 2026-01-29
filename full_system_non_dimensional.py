import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PHYSICALLY REALISTIC SIMULATION PARAMETERS
# =========================================================
# TIME SCALING: 1 dimensionless unit = 1 hour
# This gives natural circadian period of ~24 units
dt = 0.01              # 0.01 hour = 0.6 minutes
total_time = 30 * 24   # 30 days in hours
t = np.arange(0, total_time + dt, dt)

# =========================================================
# PHYSICALLY REALISTIC PARAMETERS
# =========================================================

# =========================================================
# CORE CIRCADIAN CLOCK (targeting ~24h period)
# =========================================================
alpha1 = 0.5    # x transcription rate (1/hr)
alpha2 = 0.8    # y1 translation rate (1/hr)  
alpha3 = 0.8    # z1 entry to nucleus (1/hr)

beta1 = 0.3     # x degradation (~2.3h half-life)
beta2 = 0.5     # y1 degradation (~1.4h half-life)
beta3 = 0.3     # z1 degradation (~2.3h half-life)

k = 1.0         # repression threshold
n = 4           # Hill coefficient (cooperativity)

# =========================================================
# MITOCHONDRIAL PATHWAY
# =========================================================
alpha4 = 0.3    # y2 production rate (1/hr)
alpha5 = 0.6    # z2 production from y2 (1/hr)
alpha6 = 0.8    # Mitochondrial biogenesis signal (1/hr)

beta4 = 0.4     # y2 degradation (~1.7h half-life)
beta5 = 0.5     # z2 degradation (~1.4h half-life)
beta6 = 0.03    # Mitochondrial turnover (~23h half-life, realistic mitophagy rate)

# =========================================================
# LIGHT COUPLING
# =========================================================
gamma = 0.15    # Light sensitivity (moderate entrainment)

# =========================================================
# LIGHT SCHEDULE (IN HOURS)
# =========================================================
day_length = 12        # 12 hours light per day

# Irregular pulse parameters
pulse_start_time = 15 * 24  # Start pulses after 15 days (360 hours)
pulse_prob_per_night = 0.3
pulse_min_width = 0.5       # 30 minutes
pulse_max_width = 2.0       # 2 hours
pulse_amp = 0.6             # 60% of full daylight intensity

# Generate random pulse windows
rng = np.random.default_rng(seed=42)
pulse_windows = []

first_night = int(pulse_start_time / 24)
last_night = int(total_time / 24)

for night in range(first_night, last_night):
    if rng.random() < pulse_prob_per_night:
        night_start = night * 24 + day_length
        night_end = (night + 1) * 24
        # Ensure pulse fits within night
        max_start = night_end - pulse_max_width
        pulse_start = night_start + rng.uniform(0, max_start - night_start)
        pulse_width = rng.uniform(pulse_min_width, pulse_max_width)
        pulse_windows.append((pulse_start, pulse_start + pulse_width))

print(f"Generated {len(pulse_windows)} irregular light pulses")

def LD(t):
    """Healthy LD cycle: 12h light (0-12), 12h dark (12-24)."""
    t_mod = t % 24
    return 1.0 if t_mod < day_length else 0.0

def irregular_pulse(t):
    """Return pulse amplitude if t is within any pulse window."""
    for t0, t1 in pulse_windows:
        if t0 <= t <= t1:
            return pulse_amp
    return 0.0

def L(t):
    """Total light input: regular LD + irregular pulses."""
    return min(1.0, LD(t) + irregular_pulse(t))

# =========================================================
# ODE SYSTEM
# =========================================================
def deriv(state, tt):
    """
    System of ODEs for circadian-mitochondrial coupling:
    x: clock gene mRNA
    y1, z1: clock protein (cytoplasm, nucleus) - z1 represses x
    y2, z2: mitochondrial pathway intermediates
    m: mitochondrial content/activity
    """
    x, y1, z1, y2, z2, m = state
    Lt = L(tt)

    # Core circadian clock with Hill repression
    dxdt  = alpha1 * (k**n / (k**n + z1**n)) - beta1 * x + gamma * Lt
    dy1dt = alpha2 * x - beta2 * y1
    dz1dt = alpha3 * y1 - beta3 * z1
    
    # Mitochondrial pathway driven by clock
    dy2dt = alpha4 * x - beta4 * y2
    dz2dt = alpha5 * y2 - beta5 * z2
    dmdt  = alpha6 * z2 - beta6 * m

    return np.array([dxdt, dy1dt, dz1dt, dy2dt, dz2dt, dmdt])

# =========================================================
# INITIAL CONDITIONS
# =========================================================
state = np.zeros((len(t), 6))
# Start with reasonable initial values
state[0] = np.array([0.5, 0.5, 1.0, 0.3, 0.5, 10.0])

# =========================================================
# FORWARD EULER INTEGRATION
# =========================================================
print("Running simulation...")
for i in range(len(t) - 1):
    state[i+1] = state[i] + dt * deriv(state[i], t[i])
    # Enforce non-negativity
    state[i+1] = np.maximum(state[i+1], 0.0)
    
    if (i + 1) % 100000 == 0:
        print(f"  Progress: {100 * (i+1) / len(t):.1f}%")

print("Simulation complete!")

# Extract variables
x, y1, z1, y2, z2, m = state.T
L_arr = np.array([L(tt) for tt in t])

# Convert time to days for plotting
t_days = t / 24

# =========================================================
# FIGURE 1: TIME SERIES
# =========================================================
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Clock variables
ax[0].plot(t_days, x, label="x (clock mRNA)", linewidth=1.5, color='#1f77b4')
ax[0].plot(t_days, z1, label="z₁ (nuclear repressor)", linewidth=1.5, color='#ff7f0e')
ax[0].plot(t_days, 0.8 * L_arr, "--", label="Light (scaled)", linewidth=1, color='gold', alpha=0.7)
ax[0].axvline(pulse_start_time / 24, linestyle=":", color='red', 
              label="Irregular pulses begin", linewidth=2)
ax[0].set_ylabel("Concentration (a.u.)", fontsize=11)
ax[0].set_title("Core Circadian Clock Dynamics", fontsize=13, fontweight='bold')
ax[0].legend(loc='upper right', fontsize=10)
ax[0].grid(alpha=0.3)

# Mitochondrial pathway intermediates
ax[1].plot(t_days, y2, label="y₂ (cytoplasmic)", linewidth=1.5, color='#2ca02c')
ax[1].plot(t_days, z2, label="z₂ (signaling)", linewidth=1.5, color='#d62728')
ax[1].axvline(pulse_start_time / 24, linestyle=":", color='red', linewidth=2)
ax[1].set_ylabel("Concentration (a.u.)", fontsize=11)
ax[1].set_title("Mitochondrial Pathway Intermediates", fontsize=13, fontweight='bold')
ax[1].legend(loc='upper right', fontsize=10)
ax[1].grid(alpha=0.3)

# Mitochondrial content
ax[2].plot(t_days, m, label="m (mitochondrial content)", linewidth=2, color='#9467bd')
ax[2].axvline(pulse_start_time / 24, linestyle=":", color='red', 
              label="Irregular pulses begin", linewidth=2)
ax[2].set_ylabel("Mitochondrial Content", fontsize=11)
ax[2].set_xlabel("Time (days)", fontsize=12)
ax[2].set_title("Mitochondrial Dynamics", fontsize=13, fontweight='bold')
ax[2].legend(loc='upper right', fontsize=10)
ax[2].grid(alpha=0.3)

plt.tight_layout()

plt.show()

# =========================================================
# FIGURE 2: PHASE PLANE ANALYSIS
# =========================================================
# Define time windows (in hours, convert to indices)
pre_start = pulse_start_time - 8 * 24
pre_end = pulse_start_time - 2 * 24
post_start = pulse_start_time + 2 * 24
post_end = pulse_start_time + 10 * 24

pre_window = (t >= pre_start) & (t <= pre_end)
post_window = (t >= post_start) & (t <= post_end)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Phase plane: z1 vs x
ax1.plot(z1[pre_window], x[pre_window], label="Stable LD (before pulses)", 
         linewidth=2, color='#1f77b4', alpha=0.8)
ax1.plot(z1[post_window], x[post_window], label="With irregular pulses", 
         linewidth=2, color='#ff7f0e', alpha=0.8)
ax1.set_xlabel(r"$z_1$ (nuclear repressor)", fontsize=12)
ax1.set_ylabel(r"$x$ (clock mRNA)", fontsize=12)
ax1.set_title("Clock Phase Plane Distortion", fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Phase plane: z2 vs m
ax2.plot(z2[pre_window], m[pre_window], label="Stable LD (before pulses)", 
         linewidth=2, color='#2ca02c', alpha=0.8)
ax2.plot(z2[post_window], m[post_window], label="With irregular pulses", 
         linewidth=2, color='#d62728', alpha=0.8)
ax2.set_xlabel(r"$z_2$ (signaling)", fontsize=12)
ax2.set_ylabel(r"$m$ (mitochondrial content)", fontsize=12)
ax2.set_title("Mitochondrial Phase Plane", fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# =========================================================
# FIGURE 3: DETAILED VIEW AROUND PULSE ONSET
# =========================================================
# Zoom in on transition period
zoom_start = pulse_start_time - 3 * 24
zoom_end = pulse_start_time + 7 * 24
zoom_window = (t >= zoom_start) & (t <= zoom_end)

fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

# Clock mRNA
ax[0].plot(t_days[zoom_window], x[zoom_window], linewidth=2, color='#1f77b4')
ax[0].fill_between(t_days[zoom_window], 0, L_arr[zoom_window], 
                    alpha=0.2, color='gold', label='Light')
ax[0].axvline(pulse_start_time / 24, linestyle="--", color='red', linewidth=2)
ax[0].set_ylabel("x (mRNA)", fontsize=11)
ax[0].set_title("Transition to Irregular Light Schedule", fontsize=13, fontweight='bold')
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.3)

# Repressor
ax[1].plot(t_days[zoom_window], z1[zoom_window], linewidth=2, color='#ff7f0e')
ax[1].axvline(pulse_start_time / 24, linestyle="--", color='red', linewidth=2)
ax[1].set_ylabel("z₁ (repressor)", fontsize=11)
ax[1].grid(alpha=0.3)

# Mitochondrial signal
ax[2].plot(t_days[zoom_window], z2[zoom_window], linewidth=2, color='#d62728')
ax[2].axvline(pulse_start_time / 24, linestyle="--", color='red', linewidth=2)
ax[2].set_ylabel("z₂ (signal)", fontsize=11)
ax[2].grid(alpha=0.3)

# Mitochondrial content
ax[3].plot(t_days[zoom_window], m[zoom_window], linewidth=2, color='#9467bd')
ax[3].axvline(pulse_start_time / 24, linestyle="--", color='red', 
              linewidth=2, label='Pulses begin')
ax[3].set_ylabel("m (mito)", fontsize=11)
ax[3].set_xlabel("Time (days)", fontsize=12)
ax[3].legend(fontsize=9)
ax[3].grid(alpha=0.3)

plt.tight_layout()

plt.show()

# =========================================================
# STATISTICS
# =========================================================
print("\n" + "="*60)
print("SIMULATION STATISTICS")
print("="*60)

# Calculate periods using autocorrelation (before pulses)
pre_pulse_window = t < pulse_start_time - 24
x_pre = x[pre_pulse_window]
t_pre = t[pre_pulse_window]

# # Simple peak detection for period
# from scipy.signal import find_peaks
# peaks, _ = find_peaks(x_pre, distance=int(20/dt))  # at least 20 hours apart
# if len(peaks) > 1:
#     periods = np.diff(t_pre[peaks])
#     mean_period = np.mean(periods)
#     print(f"Mean circadian period (before pulses): {mean_period:.2f} hours ({mean_period/24:.3f} days)")
# else:
#     print("Could not determine period (insufficient peaks)")

# # Mitochondrial content change
# m_before = np.mean(m[pre_pulse_window][-1000:])
# m_after = np.mean(m[(t > pulse_start_time + 5*24) & (t < pulse_start_time + 10*24)])
# m_change = ((m_after - m_before) / m_before) * 100

# print(f"\nMitochondrial content:")
# print(f"  Before irregular pulses: {m_before:.3f}")
# print(f"  After irregular pulses:  {m_after:.3f}")
# print(f"  Change: {m_change:+.2f}%")

# # Amplitude changes
# x_amp_before = np.max(x[pre_pulse_window][-1000:]) - np.min(x[pre_pulse_window][-1000:])
# post_pulse_window = (t > pulse_start_time + 5*24) & (t < pulse_start_time + 10*24)
# x_amp_after = np.max(x[post_pulse_window]) - np.min(x[post_pulse_window])
# amp_change = ((x_amp_after - x_amp_before) / x_amp_before) * 100

# print(f"\nClock amplitude (x):")
# print(f"  Before irregular pulses: {x_amp_before:.3f}")
# print(f"  After irregular pulses:  {x_amp_after:.3f}")
# print(f"  Change: {amp_change:+.2f}%")

# print("\n" + "="*60)
# print(f"Total pulses delivered: {len(pulse_windows)}")
# print("="*60)