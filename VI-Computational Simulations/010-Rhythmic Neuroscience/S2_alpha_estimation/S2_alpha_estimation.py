#!/usr/bin/env python3
"""
S2: RTM α Estimation from Autocorrelation Scaling
==================================================

Validates methodology for estimating RTM coherence exponent α
from neural τ(L) scaling data.

Tests:
1. Noise robustness
2. Sample size requirements  
3. State discrimination capability
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from datetime import datetime


def generate_tau_L_data(alpha_true, n_scales=7, L_range=(10, 110), 
                         tau_0=0.01, noise_level=0.1):
    """Generate synthetic τ(L) data with noise."""
    L = np.geomspace(L_range[0], L_range[1], n_scales)
    tau_true = tau_0 * L ** alpha_true
    
    if noise_level > 0:
        noise = np.exp(noise_level * np.random.randn(n_scales))
        tau = tau_true * noise
    else:
        tau = tau_true.copy()
    
    return L, tau, tau_true


def estimate_alpha_ols(L, tau):
    """OLS estimation on log-log data."""
    log_L = np.log(L)
    log_tau = np.log(tau)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_tau)
    return {'alpha': slope, 'log_tau0': intercept, 'r_squared': r_value**2,
            'p_value': p_value, 'std_err': std_err}


def estimate_alpha_robust(L, tau):
    """Theil-Sen robust estimation."""
    log_L = np.log(L)
    log_tau = np.log(tau)
    
    n = len(L)
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            if log_L[j] != log_L[i]:
                slopes.append((log_tau[j] - log_tau[i]) / (log_L[j] - log_L[i]))
    
    alpha = np.median(slopes)
    intercept = np.median(log_tau - alpha * log_L)
    
    predicted = alpha * log_L + intercept
    ss_res = np.sum((log_tau - predicted)**2)
    ss_tot = np.sum((log_tau - np.mean(log_tau))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    slope_mad = 1.4826 * np.median(np.abs(np.array(slopes) - alpha))
    std_err = slope_mad / np.sqrt(len(slopes))
    
    return {'alpha': alpha, 'log_tau0': intercept, 'r_squared': r2, 'std_err': std_err}


def estimate_alpha_bootstrap(L, tau, n_bootstrap=1000):
    """Bootstrap confidence intervals."""
    n = len(L)
    alphas = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        result = estimate_alpha_ols(L[idx], tau[idx])
        alphas.append(result['alpha'])
    alphas = np.array(alphas)
    return {'alpha': np.median(alphas), 'ci_low': np.percentile(alphas, 2.5),
            'ci_high': np.percentile(alphas, 97.5), 'std': np.std(alphas)}


def test_noise_robustness(alpha_true=2.0, n_trials=100):
    """Test noise effects on estimation."""
    noise_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = []
    
    for noise in noise_levels:
        errors_ols, errors_robust = [], []
        for _ in range(n_trials):
            L, tau, _ = generate_tau_L_data(alpha_true, noise_level=noise)
            errors_ols.append(estimate_alpha_ols(L, tau)['alpha'] - alpha_true)
            errors_robust.append(estimate_alpha_robust(L, tau)['alpha'] - alpha_true)
        
        results.append({
            'noise_level': noise,
            'ols_mae': np.mean(np.abs(errors_ols)),
            'ols_std': np.std(errors_ols),
            'robust_mae': np.mean(np.abs(errors_robust)),
            'robust_std': np.std(errors_robust)
        })
    return pd.DataFrame(results)


def test_sample_size(alpha_true=2.0, noise=0.15, n_trials=100):
    """Test sample size effects."""
    sample_sizes = [3, 4, 5, 7, 10, 15, 20, 30]
    results = []
    
    for n_scales in sample_sizes:
        errors, r2s = [], []
        for _ in range(n_trials):
            L, tau, _ = generate_tau_L_data(alpha_true, n_scales=n_scales, noise_level=noise)
            res = estimate_alpha_robust(L, tau)
            errors.append(res['alpha'] - alpha_true)
            r2s.append(res['r_squared'])
        
        results.append({
            'n_scales': n_scales, 'mae': np.mean(np.abs(errors)),
            'std_error': np.std(errors), 'mean_r2': np.mean(r2s)
        })
    return pd.DataFrame(results)


def test_state_discrimination(n_trials=200):
    """Test if α discriminates consciousness states."""
    states = {
        'awake': {'alpha_mean': 2.1, 'alpha_std': 0.15},
        'light_anesthesia': {'alpha_mean': 1.8, 'alpha_std': 0.15},
        'deep_anesthesia': {'alpha_mean': 1.5, 'alpha_std': 0.2},
        'rem_sleep': {'alpha_mean': 2.0, 'alpha_std': 0.15},
        'nrem_sleep': {'alpha_mean': 1.65, 'alpha_std': 0.2}
    }
    
    results = []
    for state_name, params in states.items():
        for _ in range(n_trials):
            alpha_true = np.clip(params['alpha_mean'] + params['alpha_std'] * np.random.randn(), 1.0, 3.0)
            L, tau, _ = generate_tau_L_data(alpha_true, noise_level=0.15)
            res = estimate_alpha_robust(L, tau)
            results.append({'state': state_name, 'alpha_true': alpha_true,
                          'alpha_estimated': res['alpha'], 'r_squared': res['r_squared']})
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("S2: RTM α Estimation Methodology Validation")
    print("=" * 70)
    
    output_dir = "/home/claude/010-Rhythmic_Neuroscience/S2_alpha_estimation/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # Run tests
    print("\n1. Testing noise robustness...")
    df_noise = test_noise_robustness()
    df_noise.to_csv(os.path.join(output_dir, 'S2_noise_robustness.csv'), index=False)
    
    print("2. Testing sample size effects...")
    df_samples = test_sample_size()
    df_samples.to_csv(os.path.join(output_dir, 'S2_sample_size.csv'), index=False)
    
    print("3. Testing state discrimination...")
    df_states = test_state_discrimination()
    df_states.to_csv(os.path.join(output_dir, 'S2_state_discrimination.csv'), index=False)
    
    # Visualization
    print("\nCreating figures...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Noise robustness
    ax1 = axes[0, 0]
    ax1.errorbar(df_noise['noise_level'], df_noise['ols_mae'], yerr=df_noise['ols_std'],
                 marker='o', label='OLS', capsize=3)
    ax1.errorbar(df_noise['noise_level'], df_noise['robust_mae'], yerr=df_noise['robust_std'],
                 marker='s', label='Theil-Sen', capsize=3)
    ax1.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable error')
    ax1.set_xlabel('Noise Level (log-normal σ)', fontsize=11)
    ax1.set_ylabel('Mean Absolute Error in α', fontsize=11)
    ax1.set_title('Noise Robustness', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample size
    ax2 = axes[0, 1]
    ax2.errorbar(df_samples['n_scales'], df_samples['mae'], yerr=df_samples['std_error'],
                 marker='o', capsize=3, color='purple')
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Spatial Scales', fontsize=11)
    ax2.set_ylabel('Mean Absolute Error in α', fontsize=11)
    ax2.set_title('Sample Size vs Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: State histograms
    ax3 = axes[1, 0]
    colors = {'awake': 'green', 'light_anesthesia': 'yellow', 'deep_anesthesia': 'red',
              'rem_sleep': 'blue', 'nrem_sleep': 'purple'}
    for state in df_states['state'].unique():
        data = df_states[df_states['state'] == state]['alpha_estimated']
        ax3.hist(data, bins=20, alpha=0.5, label=state.replace('_', ' ').title(),
                 color=colors.get(state, 'gray'))
    ax3.set_xlabel('Estimated α', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('α Distribution by State', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot
    ax4 = axes[1, 1]
    state_order = ['awake', 'rem_sleep', 'light_anesthesia', 'nrem_sleep', 'deep_anesthesia']
    data_box = [df_states[df_states['state'] == s]['alpha_estimated'].values for s in state_order]
    bp = ax4.boxplot(data_box, labels=[s.replace('_', '\n') for s in state_order], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['green', 'blue', 'yellow', 'purple', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax4.set_ylabel('Estimated α', fontsize=11)
    ax4.set_title('α by Consciousness State', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_estimation_validation.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_estimation_validation.pdf'))
    plt.close()
    
    # Statistics
    awake = df_states[df_states['state'] == 'awake']['alpha_estimated']
    deep = df_states[df_states['state'] == 'deep_anesthesia']['alpha_estimated']
    t_stat, p_value = stats.ttest_ind(awake, deep)
    cohens_d = (awake.mean() - deep.mean()) / np.sqrt((awake.std()**2 + deep.std()**2) / 2)
    
    min_scales = df_samples[df_samples['mae'] < 0.2]['n_scales'].min()
    max_noise = df_noise[df_noise['robust_mae'] < 0.2]['noise_level'].max()
    
    # Summary
    summary = f"""S2: RTM α Estimation Methodology Validation
============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY FINDINGS
------------
1. Noise robustness: MAE < 0.2 for σ ≤ {max_noise:.1f}
2. Sample size: Need ≥{min_scales} (L, τ) pairs for MAE < 0.2
3. State discrimination (awake vs deep anesthesia):
   - t-statistic: {t_stat:.2f}
   - p-value: {p_value:.2e}
   - Cohen's d: {cohens_d:.2f} (large effect)

STATE α VALUES (Estimated)
--------------------------
"""
    for state in state_order:
        data = df_states[df_states['state'] == state]['alpha_estimated']
        summary += f"{state:<20}: {data.mean():.3f} ± {data.std():.3f}\n"
    
    summary += f"""
INTERPRETATION
--------------
RTM predicts: Higher α = more integrated dynamics
Awake state shows highest α (2.1), deep anesthesia lowest (1.5)
This pattern is recoverable from τ(L) scaling data.

IMPORTANT DISCLAIMER
--------------------
This validates METHODOLOGY, not physical hypothesis.
Real validation requires EEG/MEG with ground-truth states.
"""
    
    with open(os.path.join(output_dir, 'S2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nNoise robustness: MAE < 0.2 for σ ≤ {max_noise:.1f}")
    print(f"Sample size: Need ≥{min_scales} scales")
    print(f"State discrimination: Cohen's d = {cohens_d:.2f}")
    print(f"\nOutputs: {output_dir}/")
    
    return df_noise, df_samples, df_states


if __name__ == "__main__":
    main()
