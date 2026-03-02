#!/usr/bin/env python3
"""
S1: NDVI Recovery Time vs Burned Patch Area
============================================

RTM-Eco predicts: τ ∝ L^α

where:
- τ = time to recover to X% of pre-fire NDVI
- L = burned patch area (ha)
- α = coherence exponent

This simulation:
1. Models post-fire NDVI recovery across patch sizes
2. Shows how α varies by ecosystem type
3. Demonstrates recovery time estimation methodology
4. Validates α extraction from simulated remote sensing data

Based on empirical patterns from fire ecology literature.

THEORETICAL MODEL - requires validation with real satellite data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import erf
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM RECOVERY MODEL
# =============================================================================

def recovery_time_rtm(L, tau_0, alpha, L_ref=100):
    """
    RTM scaling for recovery time.
    
    τ(L) = τ_0 × (L/L_ref)^α
    
    Parameters:
    -----------
    L : array
        Patch area (ha)
    tau_0 : float
        Recovery time at reference scale (days)
    alpha : float
        Coherence exponent
    L_ref : float
        Reference area (ha)
    """
    return tau_0 * (L / L_ref) ** alpha


def ndvi_recovery_curve(t, tau, k=3, ndvi_pre=0.6, ndvi_min=0.15):
    """
    Sigmoidal NDVI recovery curve.
    
    NDVI(t) = NDVI_min + (NDVI_pre - NDVI_min) × sigmoid(t/τ)
    
    Parameters:
    -----------
    t : array
        Time since fire (days)
    tau : float
        Characteristic recovery time
    k : float
        Steepness parameter
    ndvi_pre : float
        Pre-fire NDVI
    ndvi_min : float
        Minimum NDVI immediately post-fire
    """
    # Gompertz-like recovery
    x = (t / tau) - 1
    recovery_fraction = 1 - np.exp(-np.exp(k * x))
    
    ndvi = ndvi_min + (ndvi_pre - ndvi_min) * recovery_fraction
    return ndvi


def time_to_recovery(tau, target_fraction=0.8, k=3):
    """
    Compute time to reach target recovery fraction.
    
    Inverts the recovery curve.
    """
    # Solve: target = 1 - exp(-exp(k*(t/τ - 1)))
    # exp(k*(t/τ - 1)) = -ln(1 - target)
    # t/τ - 1 = ln(-ln(1 - target)) / k
    
    inner = -np.log(1 - target_fraction)
    t_tau = 1 + np.log(inner) / k
    return t_tau * tau


# =============================================================================
# ECOSYSTEM PARAMETERS
# =============================================================================

ECOSYSTEMS = {
    'Boreal Forest': {
        'alpha': 0.35,      # Slower recovery at larger scales
        'tau_0': 1500,      # ~4 years base recovery
        'ndvi_pre': 0.55,
        'ndvi_min': 0.12,
        'color': 'darkgreen'
    },
    'Mediterranean Shrubland': {
        'alpha': 0.28,      # Faster recovery, less scale-dependent
        'tau_0': 600,       # ~1.6 years
        'ndvi_pre': 0.45,
        'ndvi_min': 0.10,
        'color': 'orange'
    },
    'Temperate Grassland': {
        'alpha': 0.22,      # Quick recovery
        'tau_0': 180,       # ~6 months
        'ndvi_pre': 0.50,
        'ndvi_min': 0.15,
        'color': 'gold'
    },
    'Tropical Savanna': {
        'alpha': 0.30,      
        'tau_0': 90,        # ~3 months (wet season)
        'ndvi_pre': 0.60,
        'ndvi_min': 0.20,
        'color': 'yellowgreen'
    },
    'Temperate Forest': {
        'alpha': 0.32,
        'tau_0': 1000,      # ~2.7 years
        'ndvi_pre': 0.65,
        'ndvi_min': 0.15,
        'color': 'forestgreen'
    }
}


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_fire_recovery(ecosystem, n_fires=50, L_min=1, L_max=10000,
                           noise_level=0.15, seed=None):
    """
    Simulate fire recovery data for an ecosystem.
    
    Returns DataFrame with fire patches and recovery times.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = ECOSYSTEMS[ecosystem]
    
    # Generate log-uniform patch sizes
    log_L = np.random.uniform(np.log10(L_min), np.log10(L_max), n_fires)
    L = 10 ** log_L
    
    # True recovery times from RTM
    tau_true = recovery_time_rtm(L, params['tau_0'], params['alpha'])
    
    # Add log-normal noise to measured recovery times
    tau_measured = tau_true * np.exp(noise_level * np.random.randn(n_fires))
    
    # Add noise to area measurements (10% error)
    L_measured = L * np.exp(0.10 * np.random.randn(n_fires))
    
    return pd.DataFrame({
        'ecosystem': ecosystem,
        'area_ha': L_measured,
        'area_true': L,
        'tau_days': tau_measured,
        'tau_true': tau_true,
        'alpha_true': params['alpha'],
        'tau_0_true': params['tau_0']
    })


def estimate_alpha(L, tau, method='ols'):
    """
    Estimate α from log-log regression.
    
    log(τ) = log(τ_0) + α × log(L/L_ref)
    """
    log_L = np.log(L)
    log_tau = np.log(tau)
    
    if method == 'ols':
        slope, intercept, r, p, se = stats.linregress(log_L, log_tau)
    elif method == 'theil_sen':
        result = stats.theilslopes(log_tau, log_L)
        slope = result[0]
        intercept = result[1]
        r = np.corrcoef(log_L, log_tau)[0, 1]
        se = None
    
    return {
        'alpha': slope,
        'intercept': intercept,
        'r_squared': r**2,
        'se': se
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S1: NDVI Recovery Time vs Burned Patch Area")
    print("=" * 70)
    
    output_dir = "/home/claude/015-Rhythmic_Ecology/S1_ndvi_recovery/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    L_ref = 100  # ha reference
    
    # ===================
    # Part 1: RTM scaling across ecosystems
    # ===================
    
    print("\n1. Demonstrating RTM scaling across ecosystems...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: τ vs L for all ecosystems
    ax = axes1[0, 0]
    L_range = np.logspace(0, 4, 100)  # 1 to 10,000 ha
    
    for eco_name, params in ECOSYSTEMS.items():
        tau = recovery_time_rtm(L_range, params['tau_0'], params['alpha'], L_ref)
        ax.plot(L_range, tau / 365, linewidth=2, color=params['color'],
                label=f"{eco_name} (α={params['alpha']:.2f})")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Burned Patch Area (ha)', fontsize=11)
    ax.set_ylabel('Recovery Time τ (years)', fontsize=11)
    ax.set_title('RTM Prediction: τ ∝ L^α\nRecovery Time Scales with Patch Size', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Simulated recovery data
    ax = axes1[0, 1]
    
    all_data = []
    for eco_name in ['Boreal Forest', 'Temperate Grassland', 'Mediterranean Shrubland']:
        df = simulate_fire_recovery(eco_name, n_fires=30, seed=42)
        all_data.append(df)
        
        params = ECOSYSTEMS[eco_name]
        ax.scatter(df['area_ha'], df['tau_days'] / 365, s=40, alpha=0.6,
                   color=params['color'], label=eco_name)
    
    # Add trend lines
    for eco_name in ['Boreal Forest', 'Temperate Grassland', 'Mediterranean Shrubland']:
        params = ECOSYSTEMS[eco_name]
        tau = recovery_time_rtm(L_range, params['tau_0'], params['alpha'], L_ref)
        ax.plot(L_range, tau / 365, '--', color=params['color'], linewidth=1.5, alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Burned Patch Area (ha)', fontsize=11)
    ax.set_ylabel('Recovery Time (years)', fontsize=11)
    ax.set_title('Simulated Fire Recovery Data\n(Points = simulated observations)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Plot 3: NDVI recovery curves
    ax = axes1[1, 0]
    
    t = np.linspace(0, 2000, 200)  # days
    
    # Show recovery for different patch sizes in Boreal Forest
    params = ECOSYSTEMS['Boreal Forest']
    patch_sizes = [10, 100, 1000, 5000]
    
    for L in patch_sizes:
        tau = recovery_time_rtm(L, params['tau_0'], params['alpha'], L_ref)
        ndvi = ndvi_recovery_curve(t, tau, ndvi_pre=params['ndvi_pre'], 
                                    ndvi_min=params['ndvi_min'])
        ax.plot(t / 365, ndvi, linewidth=2, label=f'{L} ha (τ={tau/365:.1f} yr)')
    
    # Mark 80% recovery
    ax.axhline(y=params['ndvi_pre'] * 0.8 + params['ndvi_min'] * 0.2, 
               color='red', linestyle='--', label='80% recovery')
    ax.axhline(y=params['ndvi_pre'], color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time Since Fire (years)', fontsize=11)
    ax.set_ylabel('NDVI', fontsize=11)
    ax.set_title('Boreal Forest: NDVI Recovery Curves\nLarger patches recover slower', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    
    # Plot 4: α recovery validation
    ax = axes1[1, 1]
    
    results = []
    for eco_name in ECOSYSTEMS.keys():
        df = simulate_fire_recovery(eco_name, n_fires=50, seed=42)
        fit = estimate_alpha(df['area_ha'], df['tau_days'])
        
        results.append({
            'ecosystem': eco_name,
            'alpha_true': ECOSYSTEMS[eco_name]['alpha'],
            'alpha_est': fit['alpha'],
            'r_squared': fit['r_squared'],
            'error': abs(fit['alpha'] - ECOSYSTEMS[eco_name]['alpha'])
        })
    
    df_results = pd.DataFrame(results)
    
    x = range(len(df_results))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df_results['alpha_true'], width, 
           label='True α', color='blue', alpha=0.7)
    ax.bar([i + width/2 for i in x], df_results['alpha_est'], width,
           label='Estimated α', color='orange', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([e.split()[0] for e in df_results['ecosystem']], rotation=45, ha='right')
    ax.set_ylabel('Coherence Exponent α', fontsize=11)
    ax.set_title('α Recovery Across Ecosystems\n(Simulated data, n=50 fires each)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_recovery_scaling.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_recovery_scaling.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Detailed α estimation
    # ===================
    
    print("\n2. Validating α estimation methodology...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Single ecosystem detailed analysis
    eco_name = 'Temperate Forest'
    params = ECOSYSTEMS[eco_name]
    df = simulate_fire_recovery(eco_name, n_fires=80, seed=42)
    
    ax = axes2[0]
    ax.scatter(df['area_ha'], df['tau_days'], s=40, alpha=0.6, c='forestgreen')
    
    # Fit line
    fit = estimate_alpha(df['area_ha'], df['tau_days'])
    L_fit = np.logspace(0, 4, 50)
    tau_fit = np.exp(fit['intercept']) * L_fit ** fit['alpha']
    ax.plot(L_fit, tau_fit, 'r--', linewidth=2, 
            label=f"Fit: α = {fit['alpha']:.3f} (R² = {fit['r_squared']:.3f})")
    
    # True line
    tau_true = recovery_time_rtm(L_fit, params['tau_0'], params['alpha'], L_ref)
    ax.plot(L_fit, tau_true, 'k:', linewidth=2, label=f"True: α = {params['alpha']:.3f}")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Burned Patch Area (ha)', fontsize=11)
    ax.set_ylabel('Recovery Time τ (days)', fontsize=11)
    ax.set_title(f'{eco_name}: α Estimation\nError = {abs(fit["alpha"] - params["alpha"]):.4f}', 
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Collapse test (residuals)
    ax = axes2[1]
    
    log_L = np.log(df['area_ha'])
    log_tau = np.log(df['tau_days'])
    predicted = fit['intercept'] + fit['alpha'] * log_L
    residuals = log_tau - predicted
    
    ax.scatter(log_L, residuals, s=40, alpha=0.6, c='forestgreen')
    ax.axhline(y=0, color='red', linestyle='--')
    
    # Test for trend
    slope_resid, _, r_resid, p_resid, _ = stats.linregress(log_L, residuals)
    ax.plot(log_L, slope_resid * log_L, 'gray', linestyle=':', 
            label=f'Residual trend: {slope_resid:.4f} (p={p_resid:.3f})')
    
    ax.set_xlabel('log(Area)', fontsize=11)
    ax.set_ylabel('Residuals (log scale)', fontsize=11)
    ax.set_title(f'Collapse Test: Residuals vs Scale\n'
                 f'Pass if no trend (p > 0.05): {"PASS" if p_resid > 0.05 else "FAIL"}',
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_alpha_estimation.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_all.to_csv(os.path.join(output_dir, 'S1_simulated_fires.csv'), index=False)
    df_results.to_csv(os.path.join(output_dir, 'S1_alpha_recovery.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    mean_error = df_results['error'].mean()
    
    summary = f"""S1: NDVI Recovery Time vs Burned Patch Area
=============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-ECO PREDICTION
------------------
τ(L) = τ_0 × (L/L_ref)^α

where:
  τ = recovery time (days to 80% NDVI)
  L = burned patch area (ha)
  α = coherence exponent

ECOSYSTEM PARAMETERS
--------------------
"""
    
    for eco, params in ECOSYSTEMS.items():
        summary += f"{eco}: α = {params['alpha']:.2f}, τ_0 = {params['tau_0']} days\n"
    
    summary += f"""
α RECOVERY RESULTS
------------------
"""
    
    for _, row in df_results.iterrows():
        summary += f"{row['ecosystem']}: True α = {row['alpha_true']:.3f}, "
        summary += f"Est α = {row['alpha_est']:.3f}, Error = {row['error']:.4f}\n"
    
    summary += f"""
Mean absolute error: {mean_error:.4f}

INTERPRETATION
--------------
Higher α = more scale-dependent recovery
- Boreal forest (α~0.35): Large fires take much longer
- Grassland (α~0.22): Recovery less scale-dependent

RESILIENCE IMPLICATIONS
-----------------------
1. Large fires (>1000 ha) in boreal regions: 
   Recovery ~2-3× longer than 100 ha fires
   
2. Grasslands recover quickly at all scales:
   α~0.22 means 10× area only ~1.7× recovery time

EXPERIMENTAL VALIDATION
-----------------------
1. Collect fire perimeters from satellite data
2. Track NDVI recovery (Landsat/Sentinel)
3. Define τ as time to 80% pre-fire NDVI
4. Plot log(τ) vs log(L) by ecosystem
5. Slope = α
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMean α recovery error: {mean_error:.4f}")
    print("\nEcosystem α values:")
    for _, row in df_results.iterrows():
        print(f"  {row['ecosystem']}: {row['alpha_est']:.3f} (true: {row['alpha_true']:.3f})")
    print(f"\nOutputs: {output_dir}/")
    
    return df_results


if __name__ == "__main__":
    main()
