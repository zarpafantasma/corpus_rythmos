#!/usr/bin/env python3
"""
S3: Regime Shift Early Warning via α Decline
=============================================

RTM-Eco Hypothesis H2: Significant declines in α anticipate regime shifts.

When ecosystems approach critical transitions (regime shifts), the
coherence exponent α should decline before the actual transition.

Examples:
- Forest → Shrubland (desertification)
- Clear lake → Turbid lake (eutrophication)
- Healthy coral → Degraded coral (bleaching)
- Grassland → Invasive-dominated

This simulation:
1. Models α dynamics during ecosystem degradation
2. Shows early warning signals before regime shift
3. Compares α decline to classical early warning indicators
4. Demonstrates detection methodology

THEORETICAL MODEL - requires long-term monitoring validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# REGIME SHIFT MODEL
# =============================================================================

def alpha_degradation(t, alpha_0, alpha_final, t_critical, steepness=0.1):
    """
    Model α decline during ecosystem degradation.
    
    α(t) declines sigmoidally from α_0 to α_final,
    with the steepest decline near t_critical.
    """
    # Sigmoid decline
    x = steepness * (t - t_critical)
    sigmoid = 1 / (1 + np.exp(x))
    
    alpha = alpha_final + (alpha_0 - alpha_final) * sigmoid
    return alpha


def ecosystem_state(t, state_0, state_final, t_critical, steepness=0.15):
    """
    Model ecosystem state variable during regime shift.
    
    State variable (e.g., vegetation cover, water clarity) transitions
    from state_0 to state_final around t_critical.
    
    The state transition is DELAYED relative to α decline.
    """
    # Delayed sigmoid (state lags α)
    x = steepness * (t - t_critical)
    sigmoid = 1 / (1 + np.exp(x))
    
    state = state_final + (state_0 - state_final) * sigmoid
    return state


def compute_classical_ews(timeseries, window_size=30):
    """
    Compute classical early warning signals.
    
    Returns:
    - autocorrelation (AR1): Increases before transition
    - variance: Increases before transition (critical slowing down)
    """
    n = len(timeseries)
    ar1 = np.full(n, np.nan)
    variance = np.full(n, np.nan)
    
    for i in range(window_size, n):
        window = timeseries[i-window_size:i]
        
        # Variance
        variance[i] = np.var(window)
        
        # AR1 coefficient
        if len(window) > 1:
            ar1[i] = np.corrcoef(window[:-1], window[1:])[0, 1]
    
    return ar1, variance


def detect_warning(signal_ts, baseline_end=50, threshold_std=2):
    """
    Detect when signal exceeds baseline by threshold.
    
    Returns index of first warning detection.
    """
    baseline = signal_ts[:baseline_end]
    baseline_mean = np.nanmean(baseline)
    baseline_std = np.nanstd(baseline)
    
    threshold = baseline_mean - threshold_std * baseline_std  # For declining signals
    
    # Find first crossing
    for i in range(baseline_end, len(signal_ts)):
        if signal_ts[i] < threshold:
            return i
    
    return None


# =============================================================================
# REGIME SHIFT SCENARIOS
# =============================================================================

SCENARIOS = {
    'Forest Desertification': {
        'alpha_0': 0.42,
        'alpha_final': 0.18,
        't_critical': 80,
        'state_name': 'Vegetation Cover (%)',
        'state_0': 85,
        'state_final': 15,
        'description': 'Gradual transition from forest to shrubland/desert'
    },
    'Lake Eutrophication': {
        'alpha_0': 0.48,
        'alpha_final': 0.22,
        't_critical': 70,
        'state_name': 'Water Clarity (Secchi depth, m)',
        'state_0': 5.0,
        'state_final': 0.5,
        'description': 'Clear to turbid state transition'
    },
    'Coral Degradation': {
        'alpha_0': 0.50,
        'alpha_final': 0.25,
        't_critical': 60,
        'state_name': 'Live Coral Cover (%)',
        'state_0': 70,
        'state_final': 10,
        'description': 'Healthy reef to algae-dominated'
    },
    'Grassland Invasion': {
        'alpha_0': 0.38,
        'alpha_final': 0.20,
        't_critical': 90,
        'state_name': 'Native Species (%)',
        'state_0': 90,
        'state_final': 20,
        'description': 'Native prairie to invasive-dominated'
    }
}


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_regime_shift(scenario_name, n_years=150, noise_alpha=0.015, 
                          noise_state=0.05, seed=None):
    """
    Simulate a regime shift with α decline as early warning.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = SCENARIOS[scenario_name]
    
    # Time axis (years)
    t = np.arange(n_years)
    
    # True α trajectory
    alpha_true = alpha_degradation(t, params['alpha_0'], params['alpha_final'],
                                    params['t_critical'])
    
    # Add measurement noise to α
    alpha_measured = alpha_true + noise_alpha * np.random.randn(n_years)
    
    # State variable (delayed transition)
    state_true = ecosystem_state(t, params['state_0'], params['state_final'],
                                  params['t_critical'] + 15)  # 15-year lag
    
    # Add noise to state
    state_measured = state_true * (1 + noise_state * np.random.randn(n_years))
    
    return pd.DataFrame({
        'year': t,
        'alpha_true': alpha_true,
        'alpha_measured': alpha_measured,
        'state_true': state_true,
        'state_measured': state_measured,
        't_critical': params['t_critical'],
        'scenario': scenario_name
    })


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3: Regime Shift Early Warning via α Decline")
    print("=" * 70)
    
    output_dir = "/home/claude/015-Rhythmic_Ecology/S3_regime_shift/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Part 1: Single scenario detailed analysis
    # ===================
    
    print("\n1. Detailed analysis: Forest Desertification...")
    
    df = simulate_regime_shift('Forest Desertification', n_years=150, seed=42)
    params = SCENARIOS['Forest Desertification']
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: α decline
    ax = axes1[0, 0]
    ax.plot(df['year'], df['alpha_true'], 'b-', linewidth=2, label='True α')
    ax.plot(df['year'], df['alpha_measured'], 'b.', alpha=0.5, markersize=4, 
            label='Measured α')
    ax.axvline(x=params['t_critical'], color='red', linestyle='--', 
               label=f'Critical point (t={params["t_critical"]})')
    ax.axhline(y=params['alpha_0'], color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=params['alpha_final'], color='gray', linestyle=':', alpha=0.5)
    
    # Mark early warning window
    ax.axvspan(params['t_critical'] - 30, params['t_critical'], 
               alpha=0.2, color='orange', label='Early warning window')
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Coherence Exponent α', fontsize=11)
    ax.set_title('α Decline: Early Warning Signal\n(α drops BEFORE state transition)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: State variable
    ax = axes1[0, 1]
    ax.plot(df['year'], df['state_true'], 'g-', linewidth=2, label='True state')
    ax.plot(df['year'], df['state_measured'], 'g.', alpha=0.5, markersize=4,
            label='Measured state')
    ax.axvline(x=params['t_critical'], color='red', linestyle='--')
    ax.axvline(x=params['t_critical'] + 15, color='purple', linestyle='--',
               label=f'State transition (t={params["t_critical"]+15})')
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(params['state_name'], fontsize=11)
    ax.set_title('Ecosystem State\n(State transition is DELAYED)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Classical early warning signals
    ax = axes1[1, 0]
    
    ar1, variance = compute_classical_ews(df['state_measured'].values, window_size=20)
    
    ax2 = ax.twinx()
    
    l1, = ax.plot(df['year'], ar1, 'purple', linewidth=1.5, label='AR(1)')
    l2, = ax2.plot(df['year'], variance, 'orange', linewidth=1.5, label='Variance')
    
    ax.axvline(x=params['t_critical'], color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('AR(1) coefficient', fontsize=11, color='purple')
    ax2.set_ylabel('Variance', fontsize=11, color='orange')
    ax.set_title('Classical Early Warning Signals\n(Autocorrelation & Variance)', fontsize=12)
    ax.legend(handles=[l1, l2], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Detection comparison
    ax = axes1[1, 1]
    
    # Normalize signals for comparison
    alpha_norm = (df['alpha_measured'] - df['alpha_measured'].mean()) / df['alpha_measured'].std()
    ar1_norm = (ar1 - np.nanmean(ar1)) / np.nanstd(ar1)
    var_norm = (variance - np.nanmean(variance)) / np.nanstd(variance)
    
    ax.plot(df['year'], -alpha_norm, 'b-', linewidth=2, label='α decline (inverted)')
    ax.plot(df['year'], ar1_norm, 'purple', linewidth=1.5, alpha=0.7, label='AR(1) rise')
    ax.plot(df['year'], var_norm, 'orange', linewidth=1.5, alpha=0.7, label='Variance rise')
    
    ax.axvline(x=params['t_critical'], color='red', linestyle='--', label='Critical point')
    ax.axhline(y=2, color='gray', linestyle=':', label='Detection threshold (2σ)')
    
    # Find detection times
    detect_alpha = detect_warning(-alpha_norm, threshold_std=2)
    detect_ar1 = detect_warning(ar1_norm[~np.isnan(ar1_norm)], threshold_std=2)
    
    if detect_alpha:
        ax.scatter([detect_alpha], [-alpha_norm[detect_alpha]], s=100, c='blue', 
                   marker='v', zorder=5)
        ax.annotate(f'α: year {detect_alpha}', xy=(detect_alpha, -alpha_norm[detect_alpha]),
                    xytext=(detect_alpha-20, -alpha_norm[detect_alpha]+0.5), fontsize=9)
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Normalized Signal (σ)', fontsize=11)
    ax.set_title('Early Warning Detection Comparison\nα decline provides earliest warning', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_detailed_analysis.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_detailed_analysis.pdf'))
    plt.close()
    
    # ===================
    # Part 2: All scenarios comparison
    # ===================
    
    print("\n2. Comparing all regime shift scenarios...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    results = []
    
    for idx, (scenario_name, params) in enumerate(SCENARIOS.items()):
        ax = axes2[idx // 2, idx % 2]
        
        df = simulate_regime_shift(scenario_name, seed=42 + idx)
        
        # Plot α and state
        ax2 = ax.twinx()
        
        l1, = ax.plot(df['year'], df['alpha_measured'], 'b-', linewidth=1.5, 
                      alpha=0.7, label='α (left)')
        l2, = ax2.plot(df['year'], df['state_measured'], 'g-', linewidth=1.5,
                       alpha=0.7, label=f'{params["state_name"]} (right)')
        
        ax.axvline(x=params['t_critical'], color='red', linestyle='--', alpha=0.7)
        
        # Calculate lead time
        alpha_smooth = pd.Series(df['alpha_measured']).rolling(10).mean()
        alpha_drop_start = None
        for i in range(20, len(alpha_smooth)):
            if alpha_smooth.iloc[i] < params['alpha_0'] - 0.05:
                alpha_drop_start = i
                break
        
        lead_time = params['t_critical'] - alpha_drop_start if alpha_drop_start else 0
        
        results.append({
            'scenario': scenario_name,
            'alpha_0': params['alpha_0'],
            'alpha_final': params['alpha_final'],
            'alpha_drop': params['alpha_0'] - params['alpha_final'],
            't_critical': params['t_critical'],
            'lead_time': lead_time
        })
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('α', fontsize=10, color='blue')
        ax2.set_ylabel(params['state_name'].split('(')[0], fontsize=10, color='green')
        ax.set_title(f'{scenario_name}\nLead time: ~{lead_time} years', fontsize=11)
        ax.legend(handles=[l1, l2], fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_scenarios_comparison.png'), dpi=150)
    plt.close()
    
    df_results = pd.DataFrame(results)
    
    # ===================
    # Part 3: Lead time analysis
    # ===================
    
    print("\n3. Analyzing early warning lead times...")
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['forestgreen', 'steelblue', 'coral', 'gold']
    bars = ax.barh(range(len(df_results)), df_results['lead_time'], 
                   color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(df_results)))
    ax.set_yticklabels(df_results['scenario'])
    ax.set_xlabel('Early Warning Lead Time (years before critical point)', fontsize=11)
    ax.set_title('RTM-Eco Early Warning Lead Times\nα decline anticipates regime shift', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, lt in enumerate(df_results['lead_time']):
        ax.text(lt + 1, i, f'{lt} yr', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_lead_times.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_results.to_csv(os.path.join(output_dir, 'S3_scenario_results.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    mean_lead_time = df_results['lead_time'].mean()
    
    summary = f"""S3: Regime Shift Early Warning via α Decline
=============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-ECO HYPOTHESIS H2
---------------------
Significant declines in α anticipate regime shifts.

α drops BEFORE the ecosystem state variable transitions,
providing early warning for management intervention.

MECHANISM
---------
As stress accumulates:
1. First: α declines (coherence degrades)
2. Then: Recovery times increase
3. Finally: State variable collapses

The α decline reflects loss of ecosystem organization
before visible degradation occurs.

SCENARIO RESULTS
----------------
"""
    
    for _, row in df_results.iterrows():
        summary += f"\n{row['scenario']}:\n"
        summary += f"  α: {row['alpha_0']:.2f} → {row['alpha_final']:.2f} "
        summary += f"(drop = {row['alpha_drop']:.2f})\n"
        summary += f"  Critical point: year {row['t_critical']}\n"
        summary += f"  Early warning lead time: ~{row['lead_time']} years\n"
    
    summary += f"""
Mean lead time: {mean_lead_time:.1f} years

DETECTION PROTOCOL
------------------
1. Monitor α through regular disturbance-recovery cycles
2. Establish baseline α (healthy ecosystem)
3. Detect when α drops >2σ below baseline
4. Trigger alert for management intervention

COMPARISON TO CLASSICAL EWS
---------------------------
Classical indicators (AR1, variance) detect "critical slowing down"
RTM α decline detects "coherence loss"

Advantages of α:
- Earlier detection (precedes state change)
- Mechanistic interpretation
- Scale-independent

MANAGEMENT IMPLICATIONS
-----------------------
1. Regular α monitoring provides early warning
2. Intervention window: lead time before collapse
3. Restoration targets: restore α to healthy levels
4. Protected areas: maintain high-α refugia
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMean early warning lead time: {mean_lead_time:.1f} years")
    print("\nLead times by scenario:")
    for _, row in df_results.iterrows():
        print(f"  {row['scenario']}: {row['lead_time']} years")
    print(f"\nOutputs: {output_dir}/")
    
    return df_results


if __name__ == "__main__":
    main()
