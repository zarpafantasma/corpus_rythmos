#!/usr/bin/env python3
"""
S3: Conscious Access Threshold Model
=====================================

RTM-Neuro's central hypothesis:
Conscious access requires α crossing a threshold (α_c ≈ 2.0)

This simulation models:
1. α dynamics during state transitions (wake→anesthesia→recovery)
2. Threshold crossing detection
3. Binding episodes (transient α peaks during working memory)
4. Clinical fingerprints (pathological α patterns)

The model tests whether α-threshold mechanics can reproduce
observed phenomenology of consciousness transitions.

THEORETICAL MODEL - REQUIRES EMPIRICAL VALIDATION
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# PARAMETERS
# =============================================================================

# Consciousness threshold
ALPHA_THRESHOLD = 2.0

# State-specific α parameters
STATE_PARAMS = {
    'awake': {'alpha_mean': 2.15, 'alpha_std': 0.12, 'tau_fluct': 5.0},
    'light_sedation': {'alpha_mean': 1.85, 'alpha_std': 0.15, 'tau_fluct': 8.0},
    'deep_anesthesia': {'alpha_mean': 1.45, 'alpha_std': 0.20, 'tau_fluct': 15.0},
    'rem_sleep': {'alpha_mean': 2.05, 'alpha_std': 0.15, 'tau_fluct': 6.0},
    'nrem_n2': {'alpha_mean': 1.70, 'alpha_std': 0.18, 'tau_fluct': 10.0},
    'nrem_n3': {'alpha_mean': 1.50, 'alpha_std': 0.22, 'tau_fluct': 12.0}
}


# =============================================================================
# α DYNAMICS MODELS
# =============================================================================

def generate_alpha_timeseries(state, duration_sec=300, dt=0.1):
    """
    Generate α time series for a given state using OU process.
    
    dα = θ(μ - α)dt + σdW
    
    where μ = state mean, θ = 1/τ_fluct, σ scales with state variability.
    """
    params = STATE_PARAMS[state]
    
    n_steps = int(duration_sec / dt)
    t = np.arange(n_steps) * dt
    
    alpha = np.zeros(n_steps)
    alpha[0] = params['alpha_mean']
    
    theta = 1.0 / params['tau_fluct']
    sigma = params['alpha_std'] * np.sqrt(2 * theta)
    
    for i in range(1, n_steps):
        dW = np.random.randn() * np.sqrt(dt)
        alpha[i] = alpha[i-1] + theta * (params['alpha_mean'] - alpha[i-1]) * dt + sigma * dW
    
    # Clip to physical range
    alpha = np.clip(alpha, 0.5, 3.5)
    
    return t, alpha


def generate_state_transition(from_state, to_state, duration_sec=120, 
                               transition_time=60, transition_width=20, dt=0.1):
    """
    Generate α during a state transition (e.g., anesthesia induction).
    
    Uses sigmoid transition between state means.
    """
    n_steps = int(duration_sec / dt)
    t = np.arange(n_steps) * dt
    
    p_from = STATE_PARAMS[from_state]
    p_to = STATE_PARAMS[to_state]
    
    # Sigmoid transition
    sigmoid = 1 / (1 + np.exp(-(t - transition_time) / (transition_width / 4)))
    
    # Interpolated parameters
    mean = p_from['alpha_mean'] * (1 - sigmoid) + p_to['alpha_mean'] * sigmoid
    std = p_from['alpha_std'] * (1 - sigmoid) + p_to['alpha_std'] * sigmoid
    tau = p_from['tau_fluct'] * (1 - sigmoid) + p_to['tau_fluct'] * sigmoid
    
    # Generate fluctuating α
    alpha = np.zeros(n_steps)
    alpha[0] = p_from['alpha_mean']
    
    for i in range(1, n_steps):
        theta = 1.0 / tau[i]
        sigma = std[i] * np.sqrt(2 * theta)
        dW = np.random.randn() * np.sqrt(dt)
        alpha[i] = alpha[i-1] + theta * (mean[i] - alpha[i-1]) * dt + sigma * dW
    
    alpha = np.clip(alpha, 0.5, 3.5)
    
    return t, alpha, mean


def generate_binding_episode(baseline_alpha=2.05, episode_duration=2.0,
                              peak_alpha=2.4, total_duration=20.0, dt=0.1):
    """
    Generate α during a binding/working memory episode.
    
    RTM predicts transient α increase during binding, then return to baseline.
    """
    n_steps = int(total_duration / dt)
    t = np.arange(n_steps) * dt
    
    alpha = np.ones(n_steps) * baseline_alpha
    
    # Add episode (gaussian bump)
    episode_center = total_duration / 2
    episode_sigma = episode_duration / 3
    bump = (peak_alpha - baseline_alpha) * np.exp(-0.5 * ((t - episode_center) / episode_sigma)**2)
    alpha += bump
    
    # Add fluctuations
    noise = 0.08 * np.random.randn(n_steps)
    noise = gaussian_filter1d(noise, sigma=int(0.5/dt))
    alpha += noise
    
    alpha = np.clip(alpha, 0.5, 3.5)
    
    return t, alpha


def generate_pathological_pattern(pathology='fragmented', duration_sec=120, dt=0.1):
    """
    Generate pathological α patterns.
    
    Types:
    - 'fragmented': Low mean, high variance (disorders of consciousness)
    - 'rigid': Normal mean but very low variance (certain depression)
    - 'unstable': Rapid oscillations around threshold
    """
    n_steps = int(duration_sec / dt)
    t = np.arange(n_steps) * dt
    
    if pathology == 'fragmented':
        # Low α with bursts
        base = 1.4 + 0.25 * np.random.randn(n_steps)
        bursts = np.random.rand(n_steps) < 0.01  # Rare bursts
        base[bursts] += 0.6
        alpha = gaussian_filter1d(base, sigma=int(2.0/dt))
        
    elif pathology == 'rigid':
        # Stuck at slightly elevated α
        alpha = 2.1 + 0.03 * np.random.randn(n_steps)
        alpha = gaussian_filter1d(alpha, sigma=int(5.0/dt))
        
    elif pathology == 'unstable':
        # Oscillations around threshold
        oscillation = 0.3 * np.sin(2 * np.pi * t / 15)  # ~15s period
        noise = 0.15 * np.random.randn(n_steps)
        noise = gaussian_filter1d(noise, sigma=int(1.0/dt))
        alpha = 2.0 + oscillation + noise
    
    else:
        raise ValueError(f"Unknown pathology: {pathology}")
    
    alpha = np.clip(alpha, 0.5, 3.5)
    
    return t, alpha


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_threshold_metrics(t, alpha, threshold=ALPHA_THRESHOLD):
    """
    Compute metrics related to threshold crossing.
    
    Returns:
    - time_above: fraction of time α > threshold
    - n_crossings: number of threshold crossings
    - mean_duration_above: mean duration of above-threshold episodes
    """
    above = alpha > threshold
    time_above = np.mean(above)
    
    # Count crossings (rising edges)
    crossings = np.diff(above.astype(int))
    n_crossings = np.sum(crossings == 1)
    
    # Duration of above-threshold episodes
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    
    if time_above > 0 and time_above < 1:
        # Find episode durations
        in_episode = False
        episode_starts = []
        episode_ends = []
        
        for i, a in enumerate(above):
            if a and not in_episode:
                episode_starts.append(i)
                in_episode = True
            elif not a and in_episode:
                episode_ends.append(i)
                in_episode = False
        
        if in_episode:
            episode_ends.append(len(above))
        
        durations = [(episode_ends[j] - episode_starts[j]) * dt 
                     for j in range(min(len(episode_starts), len(episode_ends)))]
        mean_duration = np.mean(durations) if durations else 0
    else:
        mean_duration = t[-1] if time_above == 1 else 0
    
    return {
        'time_above_threshold': time_above,
        'n_crossings': n_crossings,
        'mean_episode_duration': mean_duration,
        'mean_alpha': np.mean(alpha),
        'std_alpha': np.std(alpha)
    }


def simulate_consciousness_detection(alpha_series, threshold=ALPHA_THRESHOLD,
                                      min_duration=1.0, dt=0.1):
    """
    Simulate consciousness detection based on α threshold.
    
    Conscious = α > threshold for at least min_duration seconds.
    """
    above = alpha_series > threshold
    
    # Apply minimum duration criterion
    min_samples = int(min_duration / dt)
    conscious = np.zeros_like(above, dtype=bool)
    
    # Sliding window
    for i in range(len(above) - min_samples):
        if np.all(above[i:i+min_samples]):
            conscious[i:i+min_samples] = True
    
    return conscious


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3: Conscious Access Threshold Model")
    print("=" * 70)
    
    output_dir = "/home/claude/010-Rhythmic_Neuroscience/S3_conscious_threshold/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    results = []
    
    # =========================================================================
    # 1. State-specific α patterns
    # =========================================================================
    print("\n1. Generating state-specific α patterns...")
    
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10))
    
    for idx, (state, params) in enumerate(STATE_PARAMS.items()):
        ax = axes1[idx // 2, idx % 2]
        
        t, alpha = generate_alpha_timeseries(state, duration_sec=120)
        metrics = compute_threshold_metrics(t, alpha)
        
        ax.plot(t, alpha, linewidth=0.8, alpha=0.8)
        ax.axhline(y=ALPHA_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold (α={ALPHA_THRESHOLD})')
        ax.fill_between(t, ALPHA_THRESHOLD, alpha, 
                        where=alpha > ALPHA_THRESHOLD, alpha=0.3, color='green',
                        label='Conscious')
        ax.fill_between(t, alpha, ALPHA_THRESHOLD,
                        where=alpha < ALPHA_THRESHOLD, alpha=0.3, color='gray',
                        label='Unconscious')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('α')
        ax.set_title(f'{state.replace("_", " ").title()}\n'
                     f'Time above threshold: {100*metrics["time_above_threshold"]:.1f}%')
        ax.set_ylim(1.0, 2.8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        
        results.append({
            'condition': state,
            'type': 'steady_state',
            **metrics
        })
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_state_patterns.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_state_patterns.pdf'))
    plt.close()
    
    # =========================================================================
    # 2. State transitions
    # =========================================================================
    print("2. Simulating state transitions...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    transitions = [
        ('awake', 'deep_anesthesia', 'Anesthesia Induction'),
        ('deep_anesthesia', 'awake', 'Emergence from Anesthesia'),
        ('awake', 'nrem_n3', 'Sleep Onset'),
        ('nrem_n3', 'rem_sleep', 'NREM→REM Transition')
    ]
    
    for idx, (from_s, to_s, title) in enumerate(transitions):
        ax = axes2[idx // 2, idx % 2]
        
        t, alpha, mean_traj = generate_state_transition(from_s, to_s, 
                                                         duration_sec=180,
                                                         transition_time=90)
        
        ax.plot(t, alpha, linewidth=0.8, alpha=0.7, label='α trajectory')
        ax.plot(t, mean_traj, 'k--', linewidth=2, alpha=0.8, label='Mean')
        ax.axhline(y=ALPHA_THRESHOLD, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=90, color='orange', linestyle=':', linewidth=1.5, 
                   label='Transition onset')
        
        # Mark crossing point
        if mean_traj[0] > ALPHA_THRESHOLD and mean_traj[-1] < ALPHA_THRESHOLD:
            cross_idx = np.where(mean_traj < ALPHA_THRESHOLD)[0][0]
            ax.axvline(x=t[cross_idx], color='purple', linestyle=':', 
                       label=f'LOC at t={t[cross_idx]:.0f}s')
        elif mean_traj[0] < ALPHA_THRESHOLD and mean_traj[-1] > ALPHA_THRESHOLD:
            cross_idx = np.where(mean_traj > ALPHA_THRESHOLD)[0][0]
            ax.axvline(x=t[cross_idx], color='green', linestyle=':',
                       label=f'ROC at t={t[cross_idx]:.0f}s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('α')
        ax.set_title(title)
        ax.set_ylim(1.0, 2.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_transitions.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_transitions.pdf'))
    plt.close()
    
    # =========================================================================
    # 3. Binding episodes
    # =========================================================================
    print("3. Simulating binding episodes...")
    
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    
    t, alpha = generate_binding_episode(total_duration=30.0)
    
    ax3.plot(t, alpha, 'b-', linewidth=1.5)
    ax3.axhline(y=ALPHA_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (α={ALPHA_THRESHOLD})')
    ax3.fill_between(t, ALPHA_THRESHOLD, alpha,
                     where=alpha > ALPHA_THRESHOLD, alpha=0.3, color='green')
    
    ax3.axvline(x=15, color='orange', linestyle=':', linewidth=2)
    ax3.annotate('Binding\nepisode', xy=(15, 2.35), fontsize=12, ha='center')
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('α', fontsize=11)
    ax3.set_title('Working Memory Binding Episode\nTransient α Increase During Integration',
                  fontsize=12)
    ax3.set_ylim(1.8, 2.6)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_binding.png'), dpi=150)
    plt.close()
    
    # =========================================================================
    # 4. Pathological patterns
    # =========================================================================
    print("4. Simulating pathological patterns...")
    
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
    
    pathologies = [
        ('fragmented', 'Disorders of Consciousness\n(Fragmented α)'),
        ('rigid', 'Pathological Rigidity\n(Overregulated α)'),
        ('unstable', 'Threshold Instability\n(Oscillating α)')
    ]
    
    for idx, (path_type, title) in enumerate(pathologies):
        ax = axes4[idx]
        
        t, alpha = generate_pathological_pattern(path_type, duration_sec=120)
        metrics = compute_threshold_metrics(t, alpha)
        
        ax.plot(t, alpha, linewidth=0.8)
        ax.axhline(y=ALPHA_THRESHOLD, color='red', linestyle='--', linewidth=2)
        ax.fill_between(t, ALPHA_THRESHOLD, alpha,
                        where=alpha > ALPHA_THRESHOLD, alpha=0.3, color='green')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('α')
        ax.set_title(f'{title}\nAbove threshold: {100*metrics["time_above_threshold"]:.0f}%')
        ax.set_ylim(0.8, 2.8)
        ax.grid(True, alpha=0.3)
        
        results.append({
            'condition': f'pathology_{path_type}',
            'type': 'pathological',
            **metrics
        })
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_pathologies.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_pathologies.pdf'))
    plt.close()
    
    # =========================================================================
    # 5. Summary statistics
    # =========================================================================
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'S3_threshold_metrics.csv'), index=False)
    
    # Summary
    summary = f"""S3: Conscious Access Threshold Model
=====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-NEURO HYPOTHESIS
--------------------
Conscious access requires α ≥ {ALPHA_THRESHOLD} (threshold)
Higher α = more integrated multiscale dynamics
Below threshold = fragmented processing, no global access

STATE-SPECIFIC PREDICTIONS
--------------------------
State               Mean α   Time Above Threshold
{'-'*55}
"""
    
    for _, row in df_results[df_results['type'] == 'steady_state'].iterrows():
        summary += f"{row['condition']:<20}{row['mean_alpha']:<9.2f}{100*row['time_above_threshold']:.1f}%\n"
    
    summary += f"""
KEY PREDICTIONS
---------------
1. ANESTHESIA INDUCTION
   - α drops below threshold → Loss of Consciousness (LOC)
   - Crossing time predicts behavioral LOC

2. EMERGENCE
   - α rises above threshold → Return of Consciousness (ROC)
   - Individual variability in crossing dynamics

3. BINDING EPISODES
   - Transient α peaks during working memory maintenance
   - Integration requires temporary elevation above baseline

4. PATHOLOGICAL PATTERNS
   Fragmented (DoC): Low mean α, rarely crosses threshold
   Rigid: Normal mean but pathologically low variance
   Unstable: Oscillations around threshold, intermittent access

CLINICAL IMPLICATIONS
---------------------
- α monitoring could predict LOC/ROC in anesthesia
- α variance may distinguish vegetative vs minimally conscious
- Real-time α tracking for closed-loop depth-of-anesthesia

VALIDATION REQUIREMENTS
-----------------------
1. EEG/MEG recordings during anesthesia induction/emergence
2. Behavioral markers of LOC/ROC timing
3. Correlation between α-threshold crossing and behavioral transition
4. Comparison with BIS, PCI, spectral entropy

IMPORTANT DISCLAIMER
--------------------
This is a THEORETICAL MODEL demonstrating how α-threshold
mechanics could explain consciousness phenomenology.

Empirical validation requires:
- Human subject recordings across states
- Ground-truth consciousness markers
- Prospective prediction testing
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\nState-specific time above threshold:")
    for _, row in df_results[df_results['type'] == 'steady_state'].iterrows():
        print(f"  {row['condition']:<20}: {100*row['time_above_threshold']:.1f}%")
    
    print(f"\nOutputs saved to: {output_dir}/")
    
    return df_results


if __name__ == "__main__":
    main()
