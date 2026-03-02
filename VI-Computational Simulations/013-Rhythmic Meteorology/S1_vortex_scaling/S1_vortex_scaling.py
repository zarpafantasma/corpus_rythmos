#!/usr/bin/env python3
"""
S1: Vortex Coherence Exponent (α) Scaling by Diameter
=====================================================

RTM-Atmo predicts: τ ∝ L^α

where:
- τ = persistence time (e-folding of autocorrelation, lifetime)
- L = feature scale (vortex diameter, spectral band)
- α = coherence exponent (transport class indicator)

This simulation demonstrates:
1. τ vs L scaling for different atmospheric regimes
2. How α varies by regime type (cyclone, blocking, convection)
3. Validation of slope estimation methodology
4. Connection to spectral cascade theory

THEORETICAL MODEL - requires validation with reanalysis data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM ATMOSPHERIC MODEL
# =============================================================================

def tau_rtm(L, tau_0, alpha, L_ref=100):
    """
    RTM scaling for persistence time.
    
    τ(L) = τ_0 × (L/L_ref)^α
    
    Parameters:
    -----------
    L : array
        Feature scale (km)
    tau_0 : float
        Persistence time at reference scale (hours)
    alpha : float
        Coherence exponent
    L_ref : float
        Reference length (km)
    """
    return tau_0 * (L / L_ref) ** alpha


def spectrum_to_alpha(beta):
    """
    Convert spectral slope β to RTM α.
    
    For E(k) ∝ k^(-β), eddy turnover τ ∝ L^α where:
    α = (β - 1) / 2
    
    Examples:
    - 3D turbulence β=5/3: α ≈ 0.33
    - 2D inverse cascade β=5/3: α ≈ 0.33
    - Geostrophic turbulence β=3: α ≈ 1.0
    """
    return (beta - 1) / 2


# =============================================================================
# ATMOSPHERIC REGIMES
# =============================================================================

REGIMES = {
    'Mature Tropical Cyclone': {
        'alpha': 2.4,
        'tau_0': 12,  # hours at 100 km
        'L_range': (50, 500),  # km
        'description': 'Strong vortex, high coherence',
        'color': 'darkred',
        'beta': 5.8  # Steep spectrum
    },
    'Blocking High': {
        'alpha': 2.6,
        'tau_0': 24,
        'L_range': (500, 3000),
        'description': 'Persistent, quasi-stationary',
        'color': 'purple',
        'beta': 6.2
    },
    'Baroclinic Wave': {
        'alpha': 1.8,
        'tau_0': 8,
        'L_range': (200, 2000),
        'description': 'Growing instability, moderate coherence',
        'color': 'blue',
        'beta': 4.6
    },
    'Mesoscale Convective System': {
        'alpha': 1.5,
        'tau_0': 4,
        'L_range': (20, 300),
        'description': 'Organized convection, hierarchical',
        'color': 'orange',
        'beta': 4.0
    },
    'Tropical Disturbance': {
        'alpha': 1.2,
        'tau_0': 3,
        'L_range': (100, 400),
        'description': 'Pre-genesis, fragmented',
        'color': 'lightcoral',
        'beta': 3.4
    },
    'Frontal Zone': {
        'alpha': 1.6,
        'tau_0': 6,
        'L_range': (50, 500),
        'description': 'Active frontogenesis',
        'color': 'green',
        'beta': 4.2
    }
}


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_vortex_data(regime_name, n_samples=50, noise_level=0.12, seed=None):
    """
    Simulate vortex persistence data for a regime.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = REGIMES[regime_name]
    L_min, L_max = params['L_range']
    
    # Generate log-uniform scales
    log_L = np.random.uniform(np.log10(L_min), np.log10(L_max), n_samples)
    L = 10 ** log_L
    
    # True persistence times
    tau_true = tau_rtm(L, params['tau_0'], params['alpha'])
    
    # Add measurement noise
    tau_measured = tau_true * np.exp(noise_level * np.random.randn(n_samples))
    L_measured = L * np.exp(0.08 * np.random.randn(n_samples))
    
    return pd.DataFrame({
        'regime': regime_name,
        'diameter_km': L_measured,
        'tau_hours': tau_measured,
        'tau_true': tau_true,
        'alpha_true': params['alpha']
    })


def estimate_alpha(L, tau, method='theil_sen'):
    """
    Estimate α from log-log regression.
    """
    log_L = np.log(L)
    log_tau = np.log(tau)
    
    if method == 'theil_sen':
        result = stats.theilslopes(log_tau, log_L)
        slope = result[0]
        intercept = result[1]
    else:
        slope, intercept, _, _, _ = stats.linregress(log_L, log_tau)
    
    r_squared = np.corrcoef(log_L, log_tau)[0, 1] ** 2
    
    return {
        'alpha': slope,
        'intercept': intercept,
        'r_squared': r_squared
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S1: Vortex Coherence Exponent (α) Scaling")
    print("=" * 70)
    
    output_dir = "/home/claude/020-Rhythmic_Meteorology/S1_vortex_scaling/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Part 1: τ vs L for all regimes
    # ===================
    
    print("\n1. Demonstrating τ vs L scaling across regimes...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: All regimes
    ax = axes1[0, 0]
    
    for regime_name, params in REGIMES.items():
        L_min, L_max = params['L_range']
        L_range = np.logspace(np.log10(L_min), np.log10(L_max), 50)
        tau = tau_rtm(L_range, params['tau_0'], params['alpha'])
        
        ax.plot(L_range, tau, linewidth=2, color=params['color'],
                label=f"{regime_name} (α={params['alpha']:.1f})")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Feature Scale L (km)', fontsize=11)
    ax.set_ylabel('Persistence Time τ (hours)', fontsize=11)
    ax.set_title('RTM-Atmo: τ ∝ L^α\nHigher α = More Coherent/Organized', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Simulated data
    ax = axes1[0, 1]
    
    all_data = []
    regime_estimates = []
    
    for regime_name, params in REGIMES.items():
        df = simulate_vortex_data(regime_name, n_samples=40, seed=42)
        all_data.append(df)
        
        ax.scatter(df['diameter_km'], df['tau_hours'], s=25, alpha=0.5,
                   color=params['color'], label=regime_name)
        
        # Estimate α
        fit = estimate_alpha(df['diameter_km'].values, df['tau_hours'].values)
        regime_estimates.append({
            'regime': regime_name,
            'alpha_true': params['alpha'],
            'alpha_est': fit['alpha'],
            'r_squared': fit['r_squared'],
            'error': abs(fit['alpha'] - params['alpha'])
        })
    
    df_all = pd.concat(all_data, ignore_index=True)
    df_estimates = pd.DataFrame(regime_estimates)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Vortex Diameter (km)', fontsize=11)
    ax.set_ylabel('Persistence Time (hours)', fontsize=11)
    ax.set_title('Simulated Atmospheric Features\n(240 samples across 6 regimes)', fontsize=12)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: α estimates by regime
    ax = axes1[1, 0]
    
    df_sorted = df_estimates.sort_values('alpha_true')
    y_pos = range(len(df_sorted))
    colors = [REGIMES[r]['color'] for r in df_sorted['regime']]
    
    ax.barh(y_pos, df_sorted['alpha_est'], color=colors, alpha=0.7)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.scatter([row['alpha_true']], [i], color='red', s=80, zorder=5, marker='|')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.split()[0] + '...' for r in df_sorted['regime']], fontsize=9)
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_title('α Estimation by Regime\n(Red marks = true values)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add transport class annotations
    ax.axvspan(1.0, 1.5, alpha=0.1, color='yellow', label='Advective')
    ax.axvspan(1.5, 2.0, alpha=0.1, color='orange', label='Hierarchical')
    ax.axvspan(2.0, 2.8, alpha=0.1, color='green', label='Coherent')
    
    # Plot 4: α vs spectral slope
    ax = axes1[1, 1]
    
    betas = [REGIMES[r]['beta'] for r in df_estimates['regime']]
    alphas_true = df_estimates['alpha_true']
    alphas_est = df_estimates['alpha_est']
    colors = [REGIMES[r]['color'] for r in df_estimates['regime']]
    
    ax.scatter(betas, alphas_est, s=100, c=colors, zorder=3)
    
    for i, row in df_estimates.iterrows():
        ax.annotate(row['regime'].split()[0], 
                    xy=(REGIMES[row['regime']]['beta'], row['alpha_est']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Theoretical line: α = (β-1)/2
    beta_range = np.linspace(3, 7, 50)
    alpha_theory = (beta_range - 1) / 2
    ax.plot(beta_range, alpha_theory, 'k--', linewidth=2, 
            label='Theory: α = (β-1)/2')
    
    ax.set_xlabel('Spectral Slope β (E ∝ k^(-β))', fontsize=11)
    ax.set_ylabel('Coherence Exponent α', fontsize=11)
    ax.set_title('α vs Spectral Slope\n(Cascade connection)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_vortex_scaling.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_vortex_scaling.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Data collapse test
    # ===================
    
    print("\n2. Testing data collapse under rescaling...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Select one regime for detailed collapse
    regime = 'Mature Tropical Cyclone'
    params = REGIMES[regime]
    
    df = simulate_vortex_data(regime, n_samples=60, seed=42)
    
    ax = axes2[0]
    
    # Raw data
    ax.scatter(df['diameter_km'], df['tau_hours'], s=50, c='darkred', alpha=0.6,
               label='Raw data')
    
    # Fit
    fit = estimate_alpha(df['diameter_km'].values, df['tau_hours'].values)
    L_fit = np.logspace(np.log10(params['L_range'][0]), np.log10(params['L_range'][1]), 50)
    tau_fit = np.exp(fit['intercept']) * L_fit ** fit['alpha']
    ax.plot(L_fit, tau_fit, 'k-', linewidth=2, label=f"Fit: α = {fit['alpha']:.2f}")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Diameter L (km)', fontsize=11)
    ax.set_ylabel('Persistence τ (hours)', fontsize=11)
    ax.set_title(f'{regime}\nα_true = {params["alpha"]}, α_est = {fit["alpha"]:.2f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Collapse test
    ax = axes2[1]
    
    # Rescale: τ / L^α should be constant
    tau_rescaled = df['tau_hours'] / (df['diameter_km'] ** fit['alpha'])
    
    ax.scatter(df['diameter_km'], tau_rescaled, s=50, c='darkred', alpha=0.6)
    ax.axhline(y=tau_rescaled.mean(), color='green', linewidth=2, 
               label=f'Mean = {tau_rescaled.mean():.3f}')
    ax.axhline(y=tau_rescaled.mean() + tau_rescaled.std(), color='green', 
               linestyle='--', alpha=0.5)
    ax.axhline(y=tau_rescaled.mean() - tau_rescaled.std(), color='green',
               linestyle='--', alpha=0.5)
    
    cv = tau_rescaled.std() / tau_rescaled.mean()
    
    ax.set_xscale('log')
    ax.set_xlabel('Diameter L (km)', fontsize=11)
    ax.set_ylabel('τ / L^α (rescaled)', fontsize=11)
    ax.set_title(f'Data Collapse Test\nCV = {cv:.3f} ({"PASS" if cv < 0.3 else "FAIL"})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_collapse_test.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_all.to_csv(os.path.join(output_dir, 'S1_vortex_data.csv'), index=False)
    df_estimates.to_csv(os.path.join(output_dir, 'S1_alpha_estimates.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    mean_error = df_estimates['error'].mean()
    
    summary = f"""S1: Vortex Coherence Exponent (α) Scaling
==========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-ATMO MODEL
--------------
τ(L) = τ_0 × (L/L_ref)^α

where:
  τ = persistence time (hours)
  L = feature scale (km)
  α = coherence exponent

TRANSPORT CLASS INTERPRETATION
------------------------------
α ≈ 1.0-1.5: Advective/fragmented (tropical disturbances, convection)
α ≈ 1.5-2.0: Hierarchical (baroclinic waves, fronts, MCS)
α ≈ 2.0-2.5: Coherent/organized (mature cyclones, jets)
α ≈ 2.5-3.0: Strongly coherent (blocking, persistent vortices)

REGIME PARAMETERS
-----------------
"""
    
    for regime, params in REGIMES.items():
        summary += f"{regime}: α = {params['alpha']:.1f}, τ_0 = {params['tau_0']} h\n"
    
    summary += f"""
ESTIMATION RESULTS
------------------
"""
    
    for _, row in df_estimates.iterrows():
        summary += f"{row['regime']}: True α = {row['alpha_true']:.1f}, "
        summary += f"Est α = {row['alpha_est']:.2f}, Error = {row['error']:.3f}\n"
    
    summary += f"""
Mean absolute error: {mean_error:.4f}

DATA COLLAPSE TEST (Mature TC)
------------------------------
CV of τ/L^α: {cv:.3f}
Pass criterion: CV < 0.3
Result: {"PASS" if cv < 0.3 else "FAIL"}

PHYSICAL INTERPRETATION
-----------------------
Higher α indicates:
- Longer persistence relative to size
- More coherent, organized structure
- Steeper energy spectrum
- Lower cascade rate

Lower α indicates:
- Faster decorrelation
- Fragmented, turbulent structure
- Shallower spectrum
- Higher cascade rate
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMean α recovery error: {mean_error:.4f}")
    print(f"Collapse test CV: {cv:.3f}")
    print(f"\nOutputs: {output_dir}/")
    
    return df_estimates


if __name__ == "__main__":
    main()
