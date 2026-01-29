import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
dt = 0.01              # hours
total_time = 60 * 24   # 40 days in hours
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
n = 8

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
# TIMESCALE TUNING (make oscillator ~24h)
# =========================================================
s = 0.43  # <--- slow everything down; try 0.5 if it's ~2x too fast

alpha1 *= s
alpha2 *= s
alpha3 *= s
beta1  *= s
beta2  *= s
beta3  *= s



# =========================================================
# LIGHT COUPLING
# =========================================================
gamma = 0.0  # keep 0 if you truly want "no light effect at all"

def L_darkness(tt):
    return 0.0

# =========================================================
# ODE SYSTEM
# =========================================================
def deriv(state, tt):
    x, y1, z1, y2, z2, m = state
    Lt = L_darkness(tt)

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

# =========================================================
# OPTIONAL: QUICK PLOT (just x and z1)
# =========================================================
# =========================================================
# NICE PLOTS + PHASE PORTRAIT (X vs M)
# =========================================================
t_days = t / 24

# Plot only the last window to show steady behaviour nicely
last_days = 10
mask_last = t_days >= (t_days[-1] - last_days)



transient_days = 20
mask = t_days > transient_days

t_seg = t[mask]
x_seg = x[mask]

# 2) Compute discrete derivative
dx = np.diff(x_seg)

# 3) Peak candidates: derivative changes from + to -
peak_indices = np.where((dx[:-1] > 0) & (dx[1:] < 0))[0] + 1

# 4) Amplitude threshold (reject tiny wiggles)
amp_thresh = np.mean(x_seg) + 0.5 * np.std(x_seg)
peak_indices = peak_indices[x_seg[peak_indices] > amp_thresh]

# 5) Enforce minimum separation (in hours)
min_sep_hours = 16     # circadian-safe
min_sep_steps = int(min_sep_hours / dt)

filtered_peaks = [peak_indices[0]]
for idx in peak_indices[1:]:
    if idx - filtered_peaks[-1] > min_sep_steps:
        filtered_peaks.append(idx)

filtered_peaks = np.array(filtered_peaks)

# 6) Compute periods
if len(filtered_peaks) > 1:
    peak_times = t_seg[filtered_peaks]
    periods = np.diff(peak_times)

    print("\n" + "="*60)
    print("CIRCADIAN PERIOD ESTIMATE (from X peaks)")
    print("="*60)
    print(f"Mean period = {np.mean(periods):.2f} h")
    print(f"Std dev     = {np.std(periods):.2f} h")
    print(f"Peaks used  = {len(periods)}")
    print("="*60)
else:
    print("Not enough peaks detected to estimate period.")




# 1) Time series: X and M (plus Z1 if you want)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax[0].plot(t_days[mask_last], x[mask_last], linewidth=2.2, label="X (clock)")
ax[0].plot(t_days[mask_last], z1[mask_last], linewidth=2.0, alpha=0.85, label="Z1 (repressor)")
ax[0].set_ylabel("Concentration")
ax[0].set_title(f"Constant darkness ($L(t)=0$): last {last_days} days", fontsize=14)
ax[0].legend(frameon=False)
ax[0].grid(alpha=0.25)

ax[1].plot(t_days[mask_last], m[mask_last], linewidth=2.4, label="M (mitochondria)")
ax[1].set_xlabel("Time (days)")
ax[1].set_ylabel("Concentration")
ax[1].legend(frameon=False)
ax[1].grid(alpha=0.25)

plt.tight_layout()
#plt.savefig("Final_plots/figures/timeseries_no_light.png", dpi=300, bbox_inches='tight')
plt.show()


# 2) Phase portrait: X vs M (show last N days)
fig, ax = plt.subplots(figsize=(7.5, 6.5))

ax.plot(x[mask_last], m[mask_last], linewidth=2.2)

# Mark start/end of the shown window
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
plt.show()


# 3) OPTIONAL: Phase portrait coloured by time (last N days)
# (Nice for showing direction along the curve)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sc = ax.scatter(x[mask_last], m[mask_last], c=t_days[mask_last],
                s=6, alpha=0.8, cmap="viridis")
ax.set_xlabel("X (clock)")
ax.set_ylabel("M (mitochondria)")
ax.set_title(f"Phase portrait (time-coloured): X vs M (last {last_days} days)", fontsize=14)
ax.grid(alpha=0.25)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Time (days)")
plt.tight_layout()
#plt.savefig("Final_plots/figures/phase_portrait_no_light.png", dpi=300, bbox_inches='tight')
plt.show()


# =========================================================
# FINAL STATE (COPY–PASTE AS INITIAL CONDITIONS)
# =========================================================
# final_state = np.array([x[-1], y1[-1], z1[-1], y2[-1], z2[-1], m[-1]])

# print("\n" + "="*60)
# print("FINAL STATE (CONSTANT DARKNESS) — COPY AS INITIAL CONDITIONS")
# print(f"Time = {total_time/24:.1f} days")
# print("="*60)

# print(f"x0  = {final_state[0]:.8f}")
# print(f"y10 = {final_state[1]:.8f}")
# print(f"z10 = {final_state[2]:.8f}")
# print(f"y20 = {final_state[3]:.8f}")
# print(f"z20 = {final_state[4]:.8f}")
# print(f"m0  = {final_state[5]:.8f}")

# print("\nstate0 = np.array([")
# print(f"    {final_state[0]:.8f},")
# print(f"    {final_state[1]:.8f},")
# print(f"    {final_state[2]:.8f},")
# print(f"    {final_state[3]:.8f},")
# print(f"    {final_state[4]:.8f},")
# print(f"    {final_state[5]:.8f}")
# print("])")
# print("="*60)


# =========================================================
# PERIOD ESTIMATION FROM PEAK-TO-PEAK DISTANCE
# =========================================================

# 1) Ignore early transients
transient_days = 20
mask = t_days > transient_days

t_seg = t[mask]
x_seg = x[mask]

# 2) Compute discrete derivative
dx = np.diff(x_seg)

# 3) Peak candidates: derivative changes from + to -
peak_indices = np.where((dx[:-1] > 0) & (dx[1:] < 0))[0] + 1

# 4) Amplitude threshold (reject tiny wiggles)
amp_thresh = np.mean(x_seg) + 0.5 * np.std(x_seg)
peak_indices = peak_indices[x_seg[peak_indices] > amp_thresh]

# 5) Enforce minimum separation (in hours)
min_sep_hours = 16     # circadian-safe
min_sep_steps = int(min_sep_hours / dt)

filtered_peaks = [peak_indices[0]]
for idx in peak_indices[1:]:
    if idx - filtered_peaks[-1] > min_sep_steps:
        filtered_peaks.append(idx)

filtered_peaks = np.array(filtered_peaks)

# 6) Compute periods
if len(filtered_peaks) > 1:
    peak_times = t_seg[filtered_peaks]
    periods = np.diff(peak_times)

    print("\n" + "="*60)
    print("CIRCADIAN PERIOD ESTIMATE (from X peaks)")
    print("="*60)
    print(f"Mean period = {np.mean(periods):.2f} h")
    print(f"Std dev     = {np.std(periods):.2f} h")
    print(f"Peaks used  = {len(periods)}")
    print("="*60)
else:
    print("Not enough peaks detected to estimate period.")

