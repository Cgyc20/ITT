import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
dt = 0.01               # hours
total_days = 100        # Increased to allow for burn-in
total_time = total_days * 24
t = np.arange(0, total_time + dt, dt)
t_days = t / 24.0

# =========================================================
# PARAMETERS
# =========================================================
alpha1 = 2
alpha2 = 0.8
alpha3 = 0.8
beta1 = 0.3
beta2 = 0.5
beta3 = 0.3
k = 1.0
n = 8
alpha4 = 0.3
alpha5 = 0.6
alpha6 = 0.8
beta4 = 0.4
beta5 = 0.5
beta6 = 0.03

s = 0.43
alpha1 *= s; alpha2 *= s; alpha3 *= s
beta1  *= s; beta2  *= s; beta3  *= s

gamma = 0.0005
K_L = 1
K_z = 1.0
p_gate = 4

# =========================================================
# TIMELINE WITH BURN-IN PERIOD
# =========================================================
burn_in_days = 20                      # Allow 20 days to reach steady state
steady_state_start = burn_in_days      # Day when system should be stable

usa_end_day = steady_state_start + 30  # USA period after burn-in
travel_day_length = 1.0
uk_start_day = usa_end_day + travel_day_length
shift_hours = +8.0
travel_day_mode = "uk"

# =========================================================
# LIGHT SCHEDULE FUNCTIONS
# =========================================================
def LD_12_12(local_time_hours):
    lt = local_time_hours % 24.0
    return 1.0 if lt < 12.0 else 0.0

def L_usa(tt):
    return LD_12_12(tt)

def L_uk(tt):
    return LD_12_12(tt + shift_hours)

def L(tt):
    day = tt / 24.0
    if day < usa_end_day:
        return L_usa(tt)
    elif day < uk_start_day:
        if travel_day_mode == "usa":
            return L_usa(tt)
        if travel_day_mode == "uk":
            return L_uk(tt)
        if travel_day_mode == "dark":
            return 0.0
        if travel_day_mode == "light":
            return 1.0
        return L_uk(tt)
    else:
        return L_uk(tt)

# =========================================================
# ODE SYSTEM
# =========================================================
def deriv(state, tt):
    x, y1, z1, y2, z2, m = state
    Lt = L(tt)
    Lsat = Lt / (K_L + Lt)
    G = (z1**p_gate) / (K_z**p_gate + z1**p_gate)
    dxdt  = alpha1 * (k**n / (k**n + z1**n)) - beta1 * x + gamma * Lsat * G
    dy1dt = alpha2 * x - beta2 * y1
    dz1dt = alpha3 * y1 - beta3 * z1
    dy2dt = alpha4 * x - beta4 * y2
    dz2dt = alpha5 * y2 - beta5 * z2
    dmdt  = alpha6 * z2 - beta6 * m
    return np.array([dxdt, dy1dt, dz1dt, dy2dt, dz2dt, dmdt])

# =========================================================
# INTEGRATION
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

for i in range(len(t) - 1):
    state[i+1] = state[i] + dt * deriv(state[i], t[i])
    state[i+1] = np.maximum(state[i+1], 0.0)

x, y1, z1, y2, z2, m = state.T
L_arr = np.array([L(tt) for tt in t])

# =========================================================
# CLEAN, SIMPLE PLOT (after burn-in)
# =========================================================
# Start plotting after burn-in period
plot_start_day = burn_in_days
plot_end_day = uk_start_day + 20  # Show 20 days after UK starts

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Circadian rhythm (X) - only after burn-in
mask = (t_days >= plot_start_day) & (t_days <= plot_end_day)
ax1.plot(t_days[mask], x[mask], 'b-', linewidth=2.5, label='Circadian Rhythm (X)')
ax1.set_ylabel('Circadian Protein X', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')

# Add shaded regions (excluding burn-in)
ax1.axvspan(burn_in_days, usa_end_day, alpha=0.1, color='blue', label=f'USA (Days {burn_in_days}-{usa_end_day:.0f})')
ax1.axvspan(usa_end_day, uk_start_day, alpha=0.1, color='red', label='Travel Day')
ax1.axvspan(uk_start_day, plot_end_day, alpha=0.1, color='green', label=f'UK (Days {uk_start_day:.0f}+)')

# Plot 2: Light schedule
ax2.plot(t_days[mask], L_arr[mask], 'r-', linewidth=1.5, label='Light Schedule')
ax2.set_ylabel('Light (0=Dark, 1=Light)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.1, 1.1)
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3, linestyle='--')

# Add vertical lines for transitions (after burn-in)
for ax in [ax1, ax2]:
    ax.axvline(usa_end_day, color='k', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(uk_start_day, color='k', linestyle='--', linewidth=2.5, alpha=0.8)
    # Also mark burn-in end
    ax.axvline(burn_in_days, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

ax1.legend(loc='upper right', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

plt.suptitle(f'Circadian Rhythm Resynchronization (After {burn_in_days}-Day Burn-in)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# ZOOMED VIEW: Just the transition period
# =========================================================
fig2, ax2 = plt.subplots(figsize=(14, 6))

# Zoom to show 5 days before travel and 15 days after
start_zoom = usa_end_day - 5
end_zoom = uk_start_day + 15
mask_zoom = (t_days >= start_zoom) & (t_days <= end_zoom)

# Plot with thicker lines for clarity
ax2.plot(t_days[mask_zoom], x[mask_zoom], 'b-', linewidth=3, label='Circadian Rhythm (X)')
ax2.plot(t_days[mask_zoom], L_arr[mask_zoom], 'r-', linewidth=2, alpha=0.7, label='Light Schedule')

# Mark key transition points
ax2.axvline(usa_end_day, color='k', linestyle='--', linewidth=3, 
           label=f'End USA (Day {usa_end_day:.0f})')
ax2.axvline(uk_start_day, color='k', linestyle=':', linewidth=3,
           label=f'Start UK (Day {uk_start_day:.0f})')

# Add shaded regions
ax2.axvspan(start_zoom, usa_end_day, alpha=0.1, color='blue')
ax2.axvspan(usa_end_day, uk_start_day, alpha=0.1, color='red')
ax2.axvspan(uk_start_day, end_zoom, alpha=0.1, color='green')

ax2.set_xlabel('Time (Days)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Amplitude / Light', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_title('Zoom: Clear View of Circadian Rhythm Shift After Time Zone Change', 
              fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# DAILY ALIGNMENT PLOT: Show synchronization progress
# =========================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

# Select specific days to show alignment
days_to_plot = [
    usa_end_day - 2,     # USA (synchronized)
    uk_start_day,        # UK Day 1 (jetlagged)
    uk_start_day + 3,    # UK Day 3 (adjusting)
    uk_start_day + 10    # UK Day 10 (resynchronized)
]

titles = ['USA: Fully Synchronized', 'UK Day 1: Jetlagged', 
          'UK Day 3: Adjusting', 'UK Day 10: Resynchronized']

for idx, (day, title) in enumerate(zip(days_to_plot, titles)):
    ax = axes3.flatten()[idx]
    
    # Get exactly one day of data
    mask_day = (t_days >= day) & (t_days < day + 1)
    
    # Extract hours for x-axis
    hours = t[mask_day] % 24
    
    # Plot circadian rhythm (normalized for comparison)
    x_day = x[mask_day]
    x_norm = (x_day - np.min(x_day)) / (np.max(x_day) - np.min(x_day))
    ax.plot(hours, x_norm, 'b-', linewidth=3, label='Circadian Rhythm (norm)')
    
    # Plot light schedule
    L_day = L_arr[mask_day]
    ax.plot(hours, L_day, 'r-', linewidth=2, alpha=0.7, label='Light Schedule')
    
    # Add expected peak markers
    if day < usa_end_day:
        ax.axvline(x=12, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Expected USA Peak')
    else:
        ax.axvline(x=4, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Expected UK Peak')
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    
    # Add legend to first plot only (to avoid clutter)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Daily Alignment: How Circadian Rhythm Synchronizes with Local Light Schedule', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# PHASE DIFFERENCE PLOT: Quantitative measure of sync
# =========================================================
fig4, ax4 = plt.subplots(figsize=(12, 6))

# Calculate daily phase (time of peak) for X
phase_data = []
day_list = []

# Only look at days after burn-in
for day in np.arange(burn_in_days, plot_end_day, 1):
    mask_day = (t_days >= day) & (t_days < day + 1)
    if np.sum(mask_day) == 0:
        continue
    
    # Find time of maximum X in this day
    day_hours = t[mask_day] % 24
    day_x = x[mask_day]
    peak_hour = day_hours[np.argmax(day_x)]
    
    phase_data.append(peak_hour)
    day_list.append(day)

# Calculate phase difference from expected schedule
phase_diff = []
for day, phase in zip(day_list, phase_data):
    if day < usa_end_day:
        # During USA: expect peak at hour 12
        diff = phase - 12
    else:
        # During UK: expect peak at hour 4 (8 hours earlier)
        diff = phase - 4
    
    # Handle wrap-around
    if diff > 12:
        diff -= 24
    elif diff < -12:
        diff += 24
    
    phase_diff.append(diff)

# Plot phase difference
ax4.plot(day_list, phase_diff, 's-', linewidth=2.5, markersize=8,
         color='purple', markerfacecolor='white', markeredgewidth=2,
         label='Phase Difference')

# Add zero line (perfect sync)
ax4.axhline(y=0, color='green', linestyle='-', linewidth=2, alpha=0.7,
           label='Perfect Synchronization')

# Add shaded regions
ax4.axvspan(burn_in_days, usa_end_day, alpha=0.1, color='blue', label='USA')
ax4.axvspan(usa_end_day, uk_start_day, alpha=0.1, color='red', label='Travel')
ax4.axvspan(uk_start_day, plot_end_day, alpha=0.1, color='green', label='UK')

# Add transition lines
ax4.axvline(usa_end_day, color='k', linestyle='--', linewidth=2)
ax4.axvline(uk_start_day, color='k', linestyle='--', linewidth=2)

ax4.set_xlabel('Day', fontsize=12, fontweight='bold')
ax4.set_ylabel('Phase Difference (Hours)', fontsize=12, fontweight='bold')
ax4.set_title('Quantifying Resynchronization: Phase Difference from Expected Schedule', 
              fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper right', fontsize=10)

# Add annotation for resynchronization time
# Find when phase difference first enters ±1 hour zone
uk_days = [d for d in day_list if d >= uk_start_day]
uk_diffs = [phase_diff[i] for i, d in enumerate(day_list) if d >= uk_start_day]

if len(uk_days) > 0:
    # Find first day within ±1 hour
    in_sync_indices = [i for i, diff in enumerate(uk_diffs) if abs(diff) <= 1]
    if in_sync_indices:
        first_sync_idx = in_sync_indices[0]
        days_to_sync = uk_days[first_sync_idx] - uk_start_day
        
        ax4.annotate(f'Resynchronized after\n{days_to_sync:.1f} days',
                    xy=(uk_days[first_sync_idx], 0),
                    xytext=(uk_days[first_sync_idx] + 5, 2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()