import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PHYSICALLY REALISTIC SIMULATION PARAMETERS
# =========================================================
dt = 0.01              # 0.01 hour = 0.6 minutes
total_time = 40 * 24   # 40 days in hours
t = np.arange(0, total_time + dt, dt)

# =========================================================
# CORE CIRCADIAN CLOCK PARAMETERS
# =========================================================
alpha1 = 0.5    # x transcription rate (1/hr)
alpha2 = 0.8    # y1 translation rate (1/hr)  
alpha3 = 0.8    # z1 entry to nucleus (1/hr)

beta1 = 0.3     # x degradation (~2.3h half-life)
beta2 = 0.5     # y1 degradation (~1.4h half-life)
beta3 = 0.3     # z1 degradation (~2.3h half-life)

k = 1.0         # repression threshold
n = 10           # Hill coefficient

# =========================================================
# MITOCHONDRIAL PATHWAY PARAMETERS
# =========================================================
alpha4 = 0.3    # y2 production rate (1/hr)
alpha5 = 0.6    # z2 production from y2 (1/hr)
alpha6 = 0.8    # Mitochondrial biogenesis signal (1/hr)

beta4 = 0.4     # y2 degradation (~1.7h half-life)
beta5 = 0.5     # z2 degradation (~1.4h half-life)
beta6 = 0.03    # Mitochondrial turnover (~23h half-life)

# =========================================================
# LIGHT COUPLING
# =========================================================
gamma = 0.15    # Light sensitivity

# =========================================================
# CONDITION 1: NORMAL 12:12 LD CYCLE
# =========================================================
def L_normal(t):
    """Regular 12h light, 12h dark cycle"""
    t_mod = t % 24
    return 1.0 if t_mod < 12 else 0.0

# =========================================================
# CONDITION 2: IRREGULAR NIGHT PULSES
# =========================================================
pulse_start_time = 15 * 24
pulse_prob_per_night = 0.3
pulse_min_width = 0.5
pulse_max_width = 2.0
pulse_amp = 0.6

rng_pulses = np.random.default_rng(seed=42)
pulse_windows = []

first_night = int(pulse_start_time / 24)
last_night = int(total_time / 24)

for night in range(first_night, last_night):
    if rng_pulses.random() < pulse_prob_per_night:
        night_start = night * 24 + 12
        night_end = (night + 1) * 24
        max_start = night_end - pulse_max_width
        pulse_start = night_start + rng_pulses.uniform(0, max_start - night_start)
        pulse_width = rng_pulses.uniform(pulse_min_width, pulse_max_width)
        pulse_windows.append((pulse_start, pulse_start + pulse_width))

def L_irregular_pulses(t):
    """12:12 LD with irregular night pulses after day 15"""
    base_light = L_normal(t)
    if t < pulse_start_time:
        return base_light
    # Check for pulses
    for t0, t1 in pulse_windows:
        if t0 <= t <= t1:
            return min(1.0, base_light + pulse_amp)
    return base_light

# =========================================================
# CONDITION 3: CONSTANT DARKNESS (NO SUNLIGHT)
# =========================================================
def L_darkness(t):
    """Complete darkness - free-running conditions"""
    return 0.0

# =========================================================
# CONDITION 4: IRREGULAR SLEEP PATTERNS
# =========================================================
# Generate irregular sleep schedule
rng_sleep = np.random.default_rng(seed=123)
sleep_schedule = []

# Start irregular sleep after 15 days
irregular_start = 15

for day in range(irregular_start, int(total_time / 24)):
    # Sleep duration: uniform between 6-9 hours
    sleep_duration = rng_sleep.uniform(6, 9)
    
    # Sleep start time: uniform between 20:00 and 02:00 (next day)
    # This represents going to bed anywhere from 8pm to 2am
    sleep_start_hour = rng_sleep.uniform(20, 26)  # 26 = 2am next day
    if sleep_start_hour >= 24:
        sleep_start_hour -= 24
    
    sleep_start = day * 24 + sleep_start_hour
    sleep_end = sleep_start + sleep_duration
    
    sleep_schedule.append((sleep_start, sleep_end))

def L_irregular_sleep(t):
    """
    Irregular sleep schedule:
    - Before day 15: normal 12:12 LD
    - After day 15: light on when awake, off when asleep
    - Sleep duration: 6-9 hours (uniform)
    - Sleep start: 20:00-02:00 (uniform)
    """
    if t < irregular_start * 24:
        return L_normal(t)
    
    # Check if currently in sleep window
    for sleep_start, sleep_end in sleep_schedule:
        if sleep_start <= t <= sleep_end:
            return 0.0  # Darkness during sleep
    
    return 1.0  # Light when awake

# =========================================================
# ODE SYSTEM
# =========================================================
def deriv(state, tt, light_func):
    """System of ODEs with specified light function"""
    x, y1, z1, y2, z2, m = state
    Lt = light_func(tt)

    dxdt  = alpha1 * (k**n / (k**n + z1**n)) - beta1 * x + gamma * Lt
    dy1dt = alpha2 * x - beta2 * y1
    dz1dt = alpha3 * y1 - beta3 * z1
    
    dy2dt = alpha4 * x - beta4 * y2
    dz2dt = alpha5 * y2 - beta5 * z2
    dmdt  = alpha6 * z2 - beta6 * m

    return np.array([dxdt, dy1dt, dz1dt, dy2dt, dz2dt, dmdt])

# =========================================================
# SIMULATION FUNCTION
# =========================================================
def simulate_condition(light_func, condition_name):
    """Run simulation for a given light condition"""
    print(f"\nSimulating: {condition_name}")
    
    state = np.zeros((len(t), 6))
    state[0] = np.array([0.5, 0.5, 1.0, 0.3, 0.5, 10.0])
    
    for i in range(len(t) - 1):
        state[i+1] = state[i] + dt * deriv(state[i], t[i], light_func)
        state[i+1] = np.maximum(state[i+1], 0.0)
        
        if (i + 1) % 100000 == 0:
            print(f"  Progress: {100 * (i+1) / len(t):.1f}%")
    
    x, y1, z1, y2, z2, m = state.T
    L_arr = np.array([light_func(tt) for tt in t])
    
    return {'x': x, 'y1': y1, 'z1': z1, 'y2': y2, 'z2': z2, 'm': m, 'L': L_arr}

# =========================================================
# RUN ALL CONDITIONS
# =========================================================
print("="*60)
print("RUNNING ALL CONDITIONS")
print("="*60)

conditions = {
    'Normal LD 12:12': simulate_condition(L_normal, 'Normal LD 12:12'),
    'Irregular Night Pulses': simulate_condition(L_irregular_pulses, 'Irregular Night Pulses'),
    'Constant Darkness': simulate_condition(L_darkness, 'Constant Darkness'),
    'Irregular Sleep': simulate_condition(L_irregular_sleep, 'Irregular Sleep')
}

print("\n" + "="*60)
print("SIMULATIONS COMPLETE")
print("="*60)

# =========================================================
# ANALYSIS FUNCTION
# =========================================================
def analyze_condition(data, name, analysis_window_start=25*24, analysis_window_end=35*24):
    """Analyze circadian and mitochondrial metrics"""
    x = data['x']
    m = data['m']
    
    # Analysis window
    analysis_mask = (t >= analysis_window_start) & (t <= analysis_window_end)
    x_analysis = x[analysis_mask]
    t_analysis = t[analysis_mask]
    
    # Period calculation
    try:
        peaks, _ = find_peaks(x_analysis, distance=int(20/dt))
        if len(peaks) > 1:
            periods = np.diff(t_analysis[peaks])
            mean_period = np.mean(periods)
            std_period = np.std(periods)
        else:
            mean_period = np.nan
            std_period = np.nan
    except:
        mean_period = np.nan
        std_period = np.nan
    
    # Amplitude
    amplitude = np.max(x_analysis) - np.min(x_analysis)
    
    # Mitochondrial content
    m_mean = np.mean(m[analysis_mask])
    m_std = np.std(m[analysis_mask])
    
    # Rhythmicity (coefficient of variation)
    x_mean = np.mean(x_analysis)
    x_cv = np.std(x_analysis) / x_mean if x_mean > 0 else 0
    
    return {
        'name': name,
        'period_mean': mean_period,
        'period_std': std_period,
        'amplitude': amplitude,
        'm_mean': m_mean,
        'm_std': m_std,
        'rhythmicity': x_cv
    }

# =========================================================
# ANALYZE ALL CONDITIONS
# =========================================================
print("\n" + "="*60)
print("ANALYSIS RESULTS (Days 25-35)")
print("="*60)

results = {}
for name, data in conditions.items():
    results[name] = analyze_condition(data, name)
    r = results[name]
    
    print(f"\n{name}:")
    print(f"  Period: {r['period_mean']:.2f} ± {r['period_std']:.2f} hours")
    print(f"  Amplitude: {r['amplitude']:.3f}")
    print(f"  Rhythmicity (CV): {r['rhythmicity']:.3f}")
    print(f"  Mitochondria: {r['m_mean']:.3f} ± {r['m_std']:.3f}")

# =========================================================
# FIGURE 1: COMPREHENSIVE TIME SERIES COMPARISON
# =========================================================
t_days = t / 24
colors = {
    'Normal LD 12:12': '#1f77b4',
    'Irregular Night Pulses': '#ff7f0e',
    'Constant Darkness': '#2ca02c',
    'Irregular Sleep': '#d62728'
}

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

# Clock mRNA (x)
ax = axes[0]
for name, data in conditions.items():
    ax.plot(t_days, data['x'], label=name, color=colors[name], linewidth=1.5, alpha=0.8)
ax.axvline(15, linestyle=':', color='black', linewidth=2, label='Disruption begins')
ax.set_ylabel('x (clock mRNA)', fontsize=11)
ax.set_title('Core Clock Dynamics Across Conditions', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Nuclear repressor (z1)
ax = axes[1]
for name, data in conditions.items():
    ax.plot(t_days, data['z1'], label=name, color=colors[name], linewidth=1.5, alpha=0.8)
ax.axvline(15, linestyle=':', color='black', linewidth=2)
ax.set_ylabel('z₁ (repressor)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Mitochondrial signal (z2)
ax = axes[2]
for name, data in conditions.items():
    ax.plot(t_days, data['z2'], label=name, color=colors[name], linewidth=1.5, alpha=0.8)
ax.axvline(15, linestyle=':', color='black', linewidth=2)
ax.set_ylabel('z₂ (mito signal)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Mitochondrial content (m)
ax = axes[3]
for name, data in conditions.items():
    ax.plot(t_days, data['m'], label=name, color=colors[name], linewidth=2, alpha=0.8)
ax.axvline(15, linestyle=':', color='black', linewidth=2)
ax.set_ylabel('m (mitochondria)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Light schedules
ax = axes[4]
for name, data in conditions.items():
    if name != 'Constant Darkness':  # Skip plotting darkness (always 0)
        ax.plot(t_days, data['L'], label=name, color=colors[name], linewidth=1, alpha=0.7)
ax.axvline(15, linestyle=':', color='black', linewidth=2)
ax.set_ylabel('Light Input', fontsize=11)
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_title('Light Schedules', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# FIGURE 2: PHASE PLANE COMPARISON
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

analysis_start = 25 * 24
analysis_end = 35 * 24
analysis_mask = (t >= analysis_start) & (t <= analysis_end)

for idx, (name, data) in enumerate(conditions.items()):
    ax = axes.flatten()[idx]
    
    x_plot = data['x'][analysis_mask]
    z1_plot = data['z1'][analysis_mask]
    
    # Color by time to show progression
    scatter = ax.scatter(z1_plot, x_plot, c=t_days[analysis_mask], 
                        cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('z₁ (nuclear repressor)', fontsize=11)
    ax.set_ylabel('x (clock mRNA)', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (days)', fontsize=9)

plt.tight_layout()

plt.show()
# =========================================================
# FIGURE 3: DETAILED METRICS COMPARISON
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

condition_names = list(results.keys())
x_pos = np.arange(len(condition_names))

# Period
ax = axes[0, 0]
periods = [results[name]['period_mean'] for name in condition_names]
period_stds = [results[name]['period_std'] for name in condition_names]
bars = ax.bar(x_pos, periods, yerr=period_stds, 
              color=[colors[name] for name in condition_names], alpha=0.7, capsize=5)
ax.axhline(24, linestyle='--', color='red', linewidth=2, label='24-hour reference')
ax.set_ylabel('Period (hours)', fontsize=11)
ax.set_title('Circadian Period', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(condition_names, rotation=15, ha='right', fontsize=9)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Amplitude
ax = axes[0, 1]
amplitudes = [results[name]['amplitude'] for name in condition_names]
ax.bar(x_pos, amplitudes, color=[colors[name] for name in condition_names], alpha=0.7)
ax.set_ylabel('Amplitude', fontsize=11)
ax.set_title('Clock Amplitude', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(condition_names, rotation=15, ha='right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Rhythmicity
ax = axes[1, 0]
rhythmicity = [results[name]['rhythmicity'] for name in condition_names]
ax.bar(x_pos, rhythmicity, color=[colors[name] for name in condition_names], alpha=0.7)
ax.set_ylabel('Coefficient of Variation', fontsize=11)
ax.set_title('Rhythmicity (higher = more variable)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(condition_names, rotation=15, ha='right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Mitochondrial content
ax = axes[1, 1]
mito_means = [results[name]['m_mean'] for name in condition_names]
mito_stds = [results[name]['m_std'] for name in condition_names]
ax.bar(x_pos, mito_means, yerr=mito_stds, 
       color=[colors[name] for name in condition_names], alpha=0.7, capsize=5)
ax.set_ylabel('Mitochondrial Content', fontsize=11)
ax.set_title('Mean Mitochondrial Content', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(condition_names, rotation=15, ha='right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# =========================================================
# FIGURE 4: ZOOMED COMPARISON (DAYS 14-20)
# =========================================================
zoom_start = 14 * 24
zoom_end = 20 * 24
zoom_mask = (t >= zoom_start) & (t <= zoom_end)

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)

# Clock
ax = axes[0]
for name, data in conditions.items():
    ax.plot(t_days[zoom_mask], data['x'][zoom_mask], 
            label=name, color=colors[name], linewidth=2, alpha=0.8)
ax.axvline(15, linestyle='--', color='red', linewidth=2, label='Disruption begins')
ax.set_ylabel('x (clock mRNA)', fontsize=11)
ax.set_title('Transition Period Detail (Days 14-20)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Mitochondrial signal
ax = axes[1]
for name, data in conditions.items():
    ax.plot(t_days[zoom_mask], data['z2'][zoom_mask], 
            label=name, color=colors[name], linewidth=2, alpha=0.8)
ax.axvline(15, linestyle='--', color='red', linewidth=2)
ax.set_ylabel('z₂ (mito signal)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Mitochondrial content
ax = axes[2]
for name, data in conditions.items():
    ax.plot(t_days[zoom_mask], data['m'][zoom_mask], 
            label=name, color=colors[name], linewidth=2.5, alpha=0.8)
ax.axvline(15, linestyle='--', color='red', linewidth=2)
ax.set_ylabel('m (mitochondria)', fontsize=11)
ax.set_xlabel('Time (days)', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# FIGURE 5: SLEEP SCHEDULE VISUALIZATION
# =========================================================
fig, ax = plt.subplots(figsize=(16, 8))

# Plot sleep windows as rectangles
for i, (sleep_start, sleep_end) in enumerate(sleep_schedule[:20]):  # First 20 days
    day_num = int(sleep_start / 24)
    start_hour = (sleep_start % 24)
    duration = sleep_end - sleep_start
    
    # Rectangle: day on y-axis, time on x-axis
    rect = plt.Rectangle((start_hour, day_num), duration, 0.8, 
                         facecolor='navy', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)

# Reference line for normal sleep (22:00 - 06:00)
ax.axvline(22, linestyle='--', color='green', linewidth=2, alpha=0.5, label='Normal sleep start (22:00)')
ax.axvline(6, linestyle='--', color='orange', linewidth=2, alpha=0.5, label='Normal sleep end (06:00)')

ax.set_xlim(0, 24)
ax.set_ylim(15, 35)
ax.set_xlabel('Time of Day (hours)', fontsize=12)
ax.set_ylabel('Day Number', fontsize=12)
ax.set_title('Irregular Sleep Schedule (6-9h duration, 20:00-02:00 start time)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(np.arange(0, 25, 2))
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
# =========================================================
# SUMMARY STATISTICS TABLE
# =========================================================
print("\n" + "="*80)
print("COMPARATIVE SUMMARY TABLE")
print("="*80)
print(f"{'Condition':<25} {'Period (h)':<15} {'Amplitude':<12} {'Rhythmicity':<12} {'Mitochondria':<15}")
print("-"*80)

for name in condition_names:
    r = results[name]
    period_str = f"{r['period_mean']:.2f}±{r['period_std']:.2f}" if not np.isnan(r['period_mean']) else "N/A"
    print(f"{name:<25} {period_str:<15} {r['amplitude']:<12.3f} {r['rhythmicity']:<12.3f} {r['m_mean']:<15.3f}")

print("="*80)

# =========================================================
# RELATIVE CHANGES FROM BASELINE
# =========================================================
baseline = results['Normal LD 12:12']
print("\n" + "="*80)
print("RELATIVE CHANGES FROM NORMAL LD 12:12")
print("="*80)

for name in condition_names:
    if name == 'Normal LD 12:12':
        continue
    
    r = results[name]
    
    # Period change
    if not np.isnan(r['period_mean']) and not np.isnan(baseline['period_mean']):
        period_change = ((r['period_mean'] - baseline['period_mean']) / baseline['period_mean']) * 100
    else:
        period_change = np.nan
    
    # Amplitude change
    amp_change = ((r['amplitude'] - baseline['amplitude']) / baseline['amplitude']) * 100
    
    # Mitochondria change
    mito_change = ((r['m_mean'] - baseline['m_mean']) / baseline['m_mean']) * 100
    
    print(f"\n{name}:")
    if not np.isnan(period_change):
        print(f"  Period: {period_change:+.2f}%")
    else:
        print(f"  Period: Unable to determine")
    print(f"  Amplitude: {amp_change:+.2f}%")
    print(f"  Mitochondria: {mito_change:+.2f}%")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETE!")
print("="*80)