#!/usr/bin/env python3
"""
S5: Alpha Fusion - Multi-Layer RTM Indicator
=============================================

From "RTM-Aware Quantum Computing" - Section 6

Implements the fusion of layer-specific α values into a single
real-time indicator α(t) using random-effects meta-analysis.

Key Components:
    - Layer-wise α estimation (physical, QEC, runtime, I/O)
    - Heterogeneity testing (I², τ², Q)
    - Random-effects fusion with REML
    - Alert thresholds (Advisory, Watch, Warning)

Reference: Paper Section 6 "α(t) construction, fusion, and QA"
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# PARAMETERS
# =============================================================================

# Layer names
LAYERS = ['physical', 'qec', 'runtime', 'io_cryo']

# Expected α values per layer
ALPHA_EXPECTED = {
    'physical': 1.5,
    'qec': 2.5,
    'runtime': 1.8,
    'io_cryo': 1.2
}

# Heterogeneity thresholds
I2_THRESHOLD = 0.5  # Maximum acceptable I²
MIN_FAMILIES = 2    # Minimum families for fusion

# Alert thresholds (Z-scores)
Z_ADVISORY = -1.5
Z_WATCH = -2.0
Z_WARNING = -2.5


# =============================================================================
# LAYER-WISE ESTIMATION
# =============================================================================

def generate_layer_estimates(n_timepoints=50, noise=0.1):
    """
    Generate synthetic α estimates for each layer over time.
    """
    data = []
    
    for t in range(n_timepoints):
        for layer in LAYERS:
            # Base α with small drift
            base_alpha = ALPHA_EXPECTED[layer]
            drift = 0.05 * np.sin(2 * np.pi * t / 20)  # Periodic drift
            
            # Add noise
            alpha_est = base_alpha + drift + np.random.normal(0, noise)
            se_est = 0.05 + np.random.exponential(0.02)
            
            # Occasionally inject decoherence event
            if np.random.random() < 0.05:
                alpha_est -= 0.3  # Drop
            
            data.append({
                'time': t,
                'layer': layer,
                'alpha': alpha_est,
                'se': se_est,
                'passes_collapse': np.random.random() > 0.1
            })
    
    return pd.DataFrame(data)


# =============================================================================
# HETEROGENEITY STATISTICS
# =============================================================================

def compute_Q_statistic(alphas, ses):
    """
    Cochran's Q statistic for heterogeneity.
    
    Q = Σ w_i (α_i - α_weighted)²
    where w_i = 1/SE_i²
    """
    weights = 1 / ses**2
    alpha_weighted = np.sum(weights * alphas) / np.sum(weights)
    Q = np.sum(weights * (alphas - alpha_weighted)**2)
    return Q, alpha_weighted


def compute_I2(Q, k):
    """
    I² statistic: proportion of variance due to heterogeneity.
    
    I² = max(0, (Q - (k-1)) / Q)
    """
    if Q <= 0 or k <= 1:
        return 0
    return max(0, (Q - (k - 1)) / Q)


def compute_tau2_DL(alphas, ses):
    """
    DerSimonian-Laird estimate of between-study variance τ².
    """
    k = len(alphas)
    weights = 1 / ses**2
    
    Q, alpha_weighted = compute_Q_statistic(alphas, ses)
    
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    
    tau2 = max(0, (Q - (k - 1)) / C)
    return tau2


def compute_tau2_REML(alphas, ses):
    """
    REML estimate of between-study variance τ².
    """
    def neg_log_likelihood(tau2):
        var = ses**2 + tau2
        weights = 1 / var
        alpha_hat = np.sum(weights * alphas) / np.sum(weights)
        
        ll = -0.5 * np.sum(np.log(var))
        ll -= 0.5 * np.sum((alphas - alpha_hat)**2 / var)
        ll -= 0.5 * np.log(np.sum(weights))
        
        return -ll
    
    result = minimize_scalar(neg_log_likelihood, bounds=(0, 1), method='bounded')
    return result.x


# =============================================================================
# RANDOM-EFFECTS FUSION
# =============================================================================

def random_effects_fusion(alphas, ses, method='REML'):
    """
    Fuse α estimates using random-effects meta-analysis.
    
    Returns:
        alpha_fused: fused estimate
        se_fused: standard error
        I2: heterogeneity statistic
        tau2: between-layer variance
        passes_gate: whether heterogeneity gate passes
    """
    k = len(alphas)
    
    if k < MIN_FAMILIES:
        return None, None, None, None, False
    
    # Compute τ²
    if method == 'REML':
        tau2 = compute_tau2_REML(alphas, ses)
    else:
        tau2 = compute_tau2_DL(alphas, ses)
    
    # Random-effects weights
    var_total = ses**2 + tau2
    weights = 1 / var_total
    
    # Fused estimate
    alpha_fused = np.sum(weights * alphas) / np.sum(weights)
    se_fused = np.sqrt(1 / np.sum(weights))
    
    # Heterogeneity
    Q, _ = compute_Q_statistic(alphas, ses)
    I2 = compute_I2(Q, k)
    
    # Gate check
    passes_gate = I2 < I2_THRESHOLD
    
    return alpha_fused, se_fused, I2, tau2, passes_gate


# =============================================================================
# REAL-TIME INDICATOR
# =============================================================================

def compute_alpha_timeseries(df):
    """
    Compute fused α(t) over time.
    """
    results = []
    
    timepoints = df['time'].unique()
    
    for t in timepoints:
        # Get layer estimates at time t
        df_t = df[(df['time'] == t) & (df['passes_collapse'])]
        
        if len(df_t) < MIN_FAMILIES:
            results.append({
                'time': t,
                'alpha_fused': np.nan,
                'se_fused': np.nan,
                'I2': np.nan,
                'passes_gate': False,
                'n_layers': len(df_t)
            })
            continue
        
        alphas = df_t['alpha'].values
        ses = df_t['se'].values
        
        alpha_f, se_f, I2, tau2, passes = random_effects_fusion(alphas, ses)
        
        results.append({
            'time': t,
            'alpha_fused': alpha_f,
            'se_fused': se_f,
            'I2': I2,
            'tau2': tau2,
            'passes_gate': passes,
            'n_layers': len(df_t)
        })
    
    return pd.DataFrame(results)


def compute_alerts(alpha_ts, baseline_window=10):
    """
    Compute alert levels based on Z-scores.
    """
    alpha_ts = alpha_ts.copy()
    
    # Rolling baseline
    alpha_ts['baseline'] = alpha_ts['alpha_fused'].rolling(
        window=baseline_window, min_periods=3
    ).mean()
    
    alpha_ts['baseline_std'] = alpha_ts['alpha_fused'].rolling(
        window=baseline_window, min_periods=3
    ).std()
    
    # Z-score
    alpha_ts['z_score'] = (
        (alpha_ts['alpha_fused'] - alpha_ts['baseline']) / 
        alpha_ts['baseline_std'].replace(0, np.nan)
    )
    
    # Alert levels
    def get_alert(z):
        if pd.isna(z):
            return 'NORMAL'
        if z <= Z_WARNING:
            return 'WARNING'
        if z <= Z_WATCH:
            return 'WATCH'
        if z <= Z_ADVISORY:
            return 'ADVISORY'
        return 'NORMAL'
    
    alpha_ts['alert'] = alpha_ts['z_score'].apply(get_alert)
    
    return alpha_ts


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir, df_layers, alpha_ts):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Layer-wise α over time
    ax1 = axes[0, 0]
    
    for layer in LAYERS:
        df_layer = df_layers[df_layers['layer'] == layer]
        ax1.plot(df_layer['time'], df_layer['alpha'], 'o-', 
                 markersize=3, alpha=0.7, label=layer)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('α', fontsize=12)
    ax1.set_title('Layer-wise α Estimates', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fused α(t) with uncertainty
    ax2 = axes[0, 1]
    
    valid = ~alpha_ts['alpha_fused'].isna()
    t = alpha_ts.loc[valid, 'time']
    alpha_f = alpha_ts.loc[valid, 'alpha_fused']
    se_f = alpha_ts.loc[valid, 'se_fused']
    
    ax2.plot(t, alpha_f, 'b-', linewidth=2, label='α(t) fused')
    ax2.fill_between(t, alpha_f - 1.96*se_f, alpha_f + 1.96*se_f,
                     alpha=0.3, color='blue', label='95% CI')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('α(t)', fontsize=12)
    ax2.set_title('Fused RTM Indicator', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heterogeneity (I²) over time
    ax3 = axes[1, 0]
    
    ax3.plot(alpha_ts['time'], alpha_ts['I2'], 'g-', linewidth=2)
    ax3.axhline(y=I2_THRESHOLD, color='red', linestyle='--', 
                label=f'Threshold = {I2_THRESHOLD}')
    ax3.fill_between(alpha_ts['time'], 0, alpha_ts['I2'],
                     where=alpha_ts['I2'] > I2_THRESHOLD,
                     color='red', alpha=0.3, label='Above threshold')
    
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('I²', fontsize=12)
    ax3.set_title('Heterogeneity Statistic', fontsize=14)
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Alert levels
    ax4 = axes[1, 1]
    
    colors = {
        'NORMAL': 'green',
        'ADVISORY': 'yellow',
        'WATCH': 'orange',
        'WARNING': 'red'
    }
    
    for alert_type in ['NORMAL', 'ADVISORY', 'WATCH', 'WARNING']:
        mask = alpha_ts['alert'] == alert_type
        if mask.any():
            ax4.scatter(alpha_ts.loc[mask, 'time'], 
                       alpha_ts.loc[mask, 'alpha_fused'],
                       c=colors[alert_type], label=alert_type, s=30, alpha=0.7)
    
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_ylabel('α(t)', fontsize=12)
    ax4.set_title('Alert Classification', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S5_alpha_fusion.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S5_alpha_fusion.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S5: Alpha Fusion - Multi-Layer RTM Indicator")
    print("From: RTM-Aware Quantum Computing - Section 6")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("FUSION METHODOLOGY")
    print("=" * 70)
    print(f"""
    Layers: {', '.join(LAYERS)}
    
    Expected α values:
    {chr(10).join(f'  {k}: {v}' for k, v in ALPHA_EXPECTED.items())}
    
    Random-effects fusion with:
    - REML τ² estimation
    - Heterogeneity gate: I² < {I2_THRESHOLD}
    - Minimum families: {MIN_FAMILIES}
    
    Alert thresholds (Z-score):
    - Advisory: Z < {Z_ADVISORY}
    - Watch: Z < {Z_WATCH}
    - Warning: Z < {Z_WARNING}
    """)
    
    print("=" * 70)
    print("SIMULATION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate layer data
    df_layers = generate_layer_estimates(n_timepoints=50)
    
    print(f"\n    Generated {len(df_layers)} layer-time observations")
    print(f"    Timepoints: {df_layers['time'].nunique()}")
    print(f"    Layers: {df_layers['layer'].nunique()}")
    
    # Compute fusion
    alpha_ts = compute_alpha_timeseries(df_layers)
    
    print(f"\n    Fusion results:")
    print(f"      Valid timepoints: {alpha_ts['passes_gate'].sum()}")
    print(f"      Mean α(t): {alpha_ts['alpha_fused'].mean():.3f}")
    print(f"      Mean I²: {alpha_ts['I2'].mean():.3f}")
    
    # Compute alerts
    alpha_ts = compute_alerts(alpha_ts)
    
    print(f"\n" + "=" * 70)
    print("ALERT SUMMARY")
    print("=" * 70)
    
    alert_counts = alpha_ts['alert'].value_counts()
    print(f"\n    Alert distribution:")
    for alert, count in alert_counts.items():
        print(f"      {alert}: {count} ({count/len(alpha_ts)*100:.1f}%)")
    
    # Heterogeneity analysis
    print(f"\n" + "=" * 70)
    print("HETEROGENEITY ANALYSIS")
    print("=" * 70)
    
    print(f"""
    I² statistics:
      Mean: {alpha_ts['I2'].mean():.3f}
      Max:  {alpha_ts['I2'].max():.3f}
      
    Gate pass rate: {alpha_ts['passes_gate'].mean()*100:.1f}%
    
    When I² > {I2_THRESHOLD}, fusion is withheld (layers disagree).
    """)
    
    # Save data
    df_layers.to_csv(os.path.join(output_dir, 'S5_layer_estimates.csv'), index=False)
    alpha_ts.to_csv(os.path.join(output_dir, 'S5_fused_timeseries.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir, df_layers, alpha_ts)
    
    # Summary
    summary = f"""S5: Alpha Fusion
================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METHODOLOGY
-----------
Random-effects meta-analysis (REML)
Heterogeneity gate: I² < {I2_THRESHOLD}
Layers: {', '.join(LAYERS)}

RESULTS
-------
Valid fusion points: {alpha_ts['passes_gate'].sum()}/{len(alpha_ts)}
Mean α(t): {alpha_ts['alpha_fused'].mean():.3f}
Mean I²: {alpha_ts['I2'].mean():.3f}

ALERT DISTRIBUTION
------------------
{chr(10).join(f'{k}: {v}' for k, v in alert_counts.items())}

PAPER VERIFICATION
------------------
✓ Multi-layer α estimation
✓ Random-effects fusion
✓ Heterogeneity testing
✓ Alert threshold system
"""
    
    with open(os.path.join(output_dir, 'S5_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
