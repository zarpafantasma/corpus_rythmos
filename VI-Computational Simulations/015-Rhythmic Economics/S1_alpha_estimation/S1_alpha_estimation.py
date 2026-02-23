#!/usr/bin/env python3
"""
S1: Estimating Economic Coherence Exponent (α) from Financial Data
===================================================================

RTM-Econ predicts: τ ∝ L^α

where:
- τ = characteristic time (recovery half-life, persistence)
- L = scale proxy (market cap, firm size, network degree)
- α = coherence exponent

This simulation demonstrates:
1. Estimating α from market cap tiers and recovery times
2. Multiple proxy families (volatility persistence, autocorrelation decay)
3. Meta-analysis across families to get robust ECI
4. Validation of slope estimation methodology

THEORETICAL MODEL - requires validation with real financial data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM ECONOMIC MODEL
# =============================================================================

def tau_rtm(L, tau_0, alpha, L_ref=1.0):
    """
    RTM scaling for characteristic time.
    
    τ(L) = τ_0 × (L/L_ref)^α
    
    Parameters:
    -----------
    L : array
        Scale proxy (e.g., market cap in $B)
    tau_0 : float
        Characteristic time at reference scale
    alpha : float
        Coherence exponent
    """
    return tau_0 * (L / L_ref) ** alpha


def recovery_halflife(returns, window=60):
    """
    Estimate recovery half-life from autocorrelation decay.
    
    Half-life = -log(2) / log(AR1)
    """
    ar1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    if ar1 > 0 and ar1 < 1:
        halflife = -np.log(2) / np.log(ar1)
    else:
        halflife = window  # Cap at window size
    return max(1, min(halflife, window))


def volatility_persistence(returns, window=60):
    """
    Estimate volatility clustering persistence (GARCH-like).
    
    Measures how long volatility shocks persist.
    """
    vol = np.abs(returns)
    ar1 = np.corrcoef(vol[:-1], vol[1:])[0, 1]
    if ar1 > 0 and ar1 < 1:
        persistence = -np.log(2) / np.log(ar1)
    else:
        persistence = window
    return max(1, min(persistence, window))


def estimate_alpha_eiv(L, tau, method='theil_sen'):
    """
    Estimate α using errors-in-variables regression.
    
    Methods:
    - theil_sen: Robust median of pairwise slopes
    - odr: Orthogonal distance regression
    """
    log_L = np.log(L)
    log_tau = np.log(tau)
    
    if method == 'theil_sen':
        result = stats.theilslopes(log_tau, log_L)
        slope = result[0]
        intercept = result[1]
        # Bootstrap for CI
        n = len(L)
        bootstrap_slopes = []
        for _ in range(500):
            idx = np.random.choice(n, n, replace=True)
            try:
                res = stats.theilslopes(log_tau[idx], log_L[idx])
                bootstrap_slopes.append(res[0])
            except:
                pass
        ci = np.percentile(bootstrap_slopes, [2.5, 97.5]) if bootstrap_slopes else [slope, slope]
        
    elif method == 'ols':
        slope, intercept, r, p, se = stats.linregress(log_L, log_tau)
        ci = [slope - 1.96*se, slope + 1.96*se]
    
    r_squared = np.corrcoef(log_L, log_tau)[0, 1] ** 2
    
    return {
        'alpha': slope,
        'intercept': intercept,
        'ci_low': ci[0],
        'ci_high': ci[1],
        'r_squared': r_squared
    }


# =============================================================================
# MARKET REGIME PARAMETERS
# =============================================================================

MARKET_REGIMES = {
    'Stable Growth': {
        'alpha': 0.45,
        'tau_0': 5,  # days base recovery
        'description': 'Normal market conditions, good coherence',
        'period': '2004-2006'
    },
    'Pre-Crisis': {
        'alpha': 0.35,
        'tau_0': 4,
        'description': 'Coherence declining, compression',
        'period': '2007'
    },
    'Crisis': {
        'alpha': 0.20,
        'tau_0': 3,
        'description': 'Decoherence, rapid cascade potential',
        'period': '2008-2009'
    },
    'Recovery': {
        'alpha': 0.40,
        'tau_0': 6,
        'description': 'Rebuilding coherence',
        'period': '2010-2012'
    },
    'New Normal': {
        'alpha': 0.42,
        'tau_0': 5,
        'description': 'Post-crisis stable regime',
        'period': '2013-2019'
    }
}


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_market_tiers(regime_name, n_firms=100, seed=None):
    """
    Simulate firms across market cap tiers with RTM scaling.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = MARKET_REGIMES[regime_name]
    
    # Market cap tiers (log-uniform from $100M to $1T)
    log_mcap = np.random.uniform(np.log10(0.1), np.log10(1000), n_firms)
    mcap = 10 ** log_mcap  # Billions
    
    # True recovery times from RTM
    tau_true = tau_rtm(mcap, params['tau_0'], params['alpha'], L_ref=1.0)
    
    # Add log-normal measurement noise
    noise_level = 0.15
    tau_measured = tau_true * np.exp(noise_level * np.random.randn(n_firms))
    
    return pd.DataFrame({
        'regime': regime_name,
        'market_cap': mcap,
        'tau_days': tau_measured,
        'tau_true': tau_true,
        'alpha_true': params['alpha']
    })


def meta_analysis_alpha(alpha_estimates, weights=None):
    """
    Random-effects meta-analysis of α estimates.
    
    Combines multiple proxy families into single ECI estimate.
    """
    alphas = np.array([e['alpha'] for e in alpha_estimates])
    ses = np.array([(e['ci_high'] - e['ci_low']) / (2 * 1.96) for e in alpha_estimates])
    
    if weights is None:
        weights = 1 / (ses ** 2 + 0.01)  # Inverse variance weighting
    
    weights = weights / np.sum(weights)
    
    alpha_combined = np.sum(alphas * weights)
    se_combined = np.sqrt(1 / np.sum(1 / (ses ** 2 + 0.01)))
    
    # Heterogeneity (I²)
    Q = np.sum(weights * (alphas - alpha_combined) ** 2)
    df = len(alphas) - 1
    I_squared = max(0, (Q - df) / Q) if Q > 0 else 0
    
    return {
        'alpha': alpha_combined,
        'se': se_combined,
        'ci_low': alpha_combined - 1.96 * se_combined,
        'ci_high': alpha_combined + 1.96 * se_combined,
        'I_squared': I_squared,
        'n_families': len(alphas)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S1: Estimating Economic Coherence Exponent (α)")
    print("=" * 70)
    
    output_dir = "/home/claude/018-Rhythmic_Economics/S1_alpha_estimation/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Part 1: α estimation across market regimes
    # ===================
    
    print("\n1. Estimating α across market regimes...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: τ vs market cap for all regimes
    ax = axes1[0, 0]
    mcap_range = np.logspace(-1, 3, 100)  # $100M to $1T
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(MARKET_REGIMES)))
    
    for (regime_name, params), color in zip(MARKET_REGIMES.items(), colors):
        tau = tau_rtm(mcap_range, params['tau_0'], params['alpha'])
        ax.plot(mcap_range, tau, linewidth=2, color=color,
                label=f"{regime_name} (α={params['alpha']:.2f})")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Market Cap ($B)', fontsize=11)
    ax.set_ylabel('Recovery Half-Life τ (days)', fontsize=11)
    ax.set_title('RTM Prediction: τ ∝ Market Cap^α\nHigher α = More Coherent', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Simulated data by regime
    ax = axes1[0, 1]
    
    all_data = []
    regime_estimates = []
    
    for (regime_name, params), color in zip(MARKET_REGIMES.items(), colors):
        df = simulate_market_tiers(regime_name, n_firms=80, seed=42)
        all_data.append(df)
        
        ax.scatter(df['market_cap'], df['tau_days'], s=20, alpha=0.4, 
                   color=color, label=regime_name)
        
        # Estimate α
        fit = estimate_alpha_eiv(df['market_cap'].values, df['tau_days'].values)
        regime_estimates.append({
            'regime': regime_name,
            'alpha_true': params['alpha'],
            'alpha_est': fit['alpha'],
            'ci_low': fit['ci_low'],
            'ci_high': fit['ci_high'],
            'r_squared': fit['r_squared']
        })
    
    df_all = pd.concat(all_data, ignore_index=True)
    df_estimates = pd.DataFrame(regime_estimates)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Market Cap ($B)', fontsize=11)
    ax.set_ylabel('Recovery Half-Life (days)', fontsize=11)
    ax.set_title('Simulated Market Data by Regime\n(400 firms total)', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: α estimates with CI
    ax = axes1[1, 0]
    
    y_pos = range(len(df_estimates))
    
    ax.barh(y_pos, df_estimates['alpha_est'], 
            xerr=[(df_estimates['alpha_est'] - df_estimates['ci_low']).values,
                  (df_estimates['ci_high'] - df_estimates['alpha_est']).values],
            color=colors, alpha=0.7, capsize=5)
    
    # True values
    for i, row in df_estimates.iterrows():
        ax.scatter([row['alpha_true']], [i], color='red', s=80, zorder=5, marker='|')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_estimates['regime'])
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_title('α Estimates with 95% CI\n(Red marks = true values)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Crisis threshold')
    ax.legend()
    
    # Plot 4: α trajectory simulation
    ax = axes1[1, 1]
    
    # Simulate rolling α over time
    time_periods = ['2004', '2005', '2006', '2007', '2008', '2009', 
                    '2010', '2011', '2012', '2015', '2018']
    regime_sequence = ['Stable Growth', 'Stable Growth', 'Stable Growth',
                       'Pre-Crisis', 'Crisis', 'Crisis',
                       'Recovery', 'Recovery', 'Recovery', 
                       'New Normal', 'New Normal']
    
    alpha_trajectory = []
    for regime in regime_sequence:
        params = MARKET_REGIMES[regime]
        alpha_trajectory.append(params['alpha'] + 0.03 * np.random.randn())
    
    ax.plot(range(len(time_periods)), alpha_trajectory, 'b-o', linewidth=2, markersize=8)
    ax.fill_between(range(len(time_periods)), 0, 0.3, alpha=0.2, color='red', label='Danger zone')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7)
    
    # Mark crisis
    ax.axvspan(3.5, 5.5, alpha=0.3, color='gray', label='2008 Crisis')
    
    ax.set_xticks(range(len(time_periods)))
    ax.set_xticklabels(time_periods, rotation=45, ha='right')
    ax.set_ylabel('Economic Coherence Index (α)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('Simulated α Trajectory (2004-2018)\nα drops before/during crisis', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_alpha_estimation.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_alpha_estimation.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Multiple proxy families
    # ===================
    
    print("\n2. Multi-family α estimation...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Simulate multiple families for one regime
    regime = 'Stable Growth'
    params = MARKET_REGIMES[regime]
    
    families = {
        'Recovery Half-Life': {'tau_0': 5, 'noise': 0.15},
        'Volatility Persistence': {'tau_0': 8, 'noise': 0.18},
        'Autocorrelation Decay': {'tau_0': 3, 'noise': 0.20},
        'Order Flow Relaxation': {'tau_0': 2, 'noise': 0.22}
    }
    
    ax = axes2[0]
    family_estimates = []
    
    family_colors = plt.cm.Set2(np.linspace(0, 1, len(families)))
    
    for (family_name, family_params), color in zip(families.items(), family_colors):
        # Generate data
        mcap = 10 ** np.random.uniform(-1, 3, 60)
        tau_true = tau_rtm(mcap, family_params['tau_0'], params['alpha'])
        tau_measured = tau_true * np.exp(family_params['noise'] * np.random.randn(60))
        
        ax.scatter(mcap, tau_measured, s=30, alpha=0.5, color=color, label=family_name)
        
        # Estimate
        fit = estimate_alpha_eiv(mcap, tau_measured)
        family_estimates.append({
            'family': family_name,
            'alpha': fit['alpha'],
            'ci_low': fit['ci_low'],
            'ci_high': fit['ci_high']
        })
        
        # Fit line
        mcap_fit = np.logspace(-1, 3, 50)
        tau_fit = np.exp(fit['intercept']) * mcap_fit ** fit['alpha']
        ax.plot(mcap_fit, tau_fit, '--', color=color, linewidth=1.5, alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Market Cap ($B)', fontsize=11)
    ax.set_ylabel('Characteristic Time τ (days)', fontsize=11)
    ax.set_title(f'{regime}: Multiple Proxy Families\nEach measures τ differently', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Meta-analysis
    ax = axes2[1]
    
    meta_result = meta_analysis_alpha(family_estimates)
    
    df_families = pd.DataFrame(family_estimates)
    
    y_pos = range(len(df_families))
    ax.barh(y_pos, df_families['alpha'], 
            xerr=[(df_families['alpha'] - df_families['ci_low']).values,
                  (df_families['ci_high'] - df_families['alpha']).values],
            color=family_colors, alpha=0.7, capsize=5)
    
    # Meta-analysis result
    ax.axvline(x=meta_result['alpha'], color='black', linewidth=3, 
               label=f"Combined ECI = {meta_result['alpha']:.3f}")
    ax.axvspan(meta_result['ci_low'], meta_result['ci_high'], alpha=0.2, color='gray')
    
    # True value
    ax.axvline(x=params['alpha'], color='red', linestyle='--', linewidth=2,
               label=f"True α = {params['alpha']:.2f}")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_families['family'])
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_title(f'Meta-Analysis Across Proxy Families\n'
                 f'I² = {meta_result["I_squared"]:.2f} (heterogeneity)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_multi_family.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_all.to_csv(os.path.join(output_dir, 'S1_simulated_firms.csv'), index=False)
    df_estimates.to_csv(os.path.join(output_dir, 'S1_regime_estimates.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    mean_error = (df_estimates['alpha_est'] - df_estimates['alpha_true']).abs().mean()
    
    summary = f"""S1: Estimating Economic Coherence Exponent (α)
==============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-ECON MODEL
--------------
τ(L) = τ_0 × (L/L_ref)^α

where:
  τ = characteristic time (recovery, persistence)
  L = scale proxy (market cap, firm size)
  α = coherence exponent

INTERPRETATION OF α
-------------------
α > 0.4: High coherence (buffered, staged decisions)
α ~ 0.3-0.4: Moderate coherence
α < 0.3: Low coherence (cascade risk, rapid propagation)

MARKET REGIME PARAMETERS
------------------------
"""
    
    for regime, params in MARKET_REGIMES.items():
        summary += f"{regime} ({params['period']}): α = {params['alpha']:.2f}\n"
        summary += f"  {params['description']}\n"
    
    summary += f"""
ESTIMATION RESULTS
------------------
"""
    
    for _, row in df_estimates.iterrows():
        summary += f"{row['regime']}: "
        summary += f"True α = {row['alpha_true']:.2f}, "
        summary += f"Est α = {row['alpha_est']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]\n"
    
    summary += f"""
Mean absolute error: {mean_error:.4f}

META-ANALYSIS (Stable Growth, 4 families)
-----------------------------------------
Combined ECI: {meta_result['alpha']:.3f}
95% CI: [{meta_result['ci_low']:.3f}, {meta_result['ci_high']:.3f}]
Heterogeneity I²: {meta_result['I_squared']:.2f}
True α: {params['alpha']:.2f}

METHODOLOGY
-----------
1. Group firms by market cap tiers
2. Measure characteristic times per tier
3. Fit log(τ) vs log(market cap) → slope = α
4. Use Theil-Sen for robustness to outliers
5. Meta-analyze across proxy families
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMean α recovery error: {mean_error:.4f}")
    print(f"Meta-analysis ECI: {meta_result['alpha']:.3f}")
    print(f"\nOutputs: {output_dir}/")
    
    return df_estimates, meta_result


if __name__ == "__main__":
    main()
