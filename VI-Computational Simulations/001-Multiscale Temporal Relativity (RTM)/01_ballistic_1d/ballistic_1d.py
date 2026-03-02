#!/usr/bin/env python3
"""
RTM Simulation: Ballistic 1-D Propagation
==========================================

This script simulates ballistic (straight-line) propagation in a 1-D system
to verify the RTM prediction that α ≈ 1 for ballistic transport.

In ballistic transport, a particle moves at constant velocity without scattering.
The time T to traverse a distance L scales as T ∝ L^α with α = 1.

This serves as the lower benchmark for RTM temporal-relativity tests.

Expected result: α ≈ 1.00-1.03

Author: RTM Corpus
License: CC BY 4.0
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# System sizes (in arbitrary length units)
SYSTEM_SIZES = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])

# Number of trials per system size (for statistical robustness)
N_TRIALS = 100

# Base velocity (arbitrary units, constant across all sizes)
BASE_VELOCITY = 1.0

# Velocity fluctuation (small noise to simulate real-world conditions)
VELOCITY_NOISE_STD = 0.01  # 1% standard deviation

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def ballistic_traversal_time(L, velocity):
    """
    Calculate the time for ballistic traversal of distance L.
    
    In pure ballistic transport: T = L / v
    
    Parameters
    ----------
    L : float
        Distance to traverse
    velocity : float
        Constant velocity of propagation
        
    Returns
    -------
    float
        Traversal time
    """
    return L / velocity


def simulate_ballistic_transport(L, n_trials, base_velocity, noise_std, rng):
    """
    Simulate ballistic transport with small velocity fluctuations.
    
    Parameters
    ----------
    L : float
        System size / distance
    n_trials : int
        Number of independent trials
    base_velocity : float
        Mean velocity
    noise_std : float
        Standard deviation of velocity noise
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    numpy.ndarray
        Array of traversal times for each trial
    """
    # Generate velocities with small Gaussian noise
    velocities = base_velocity + rng.normal(0, noise_std * base_velocity, n_trials)
    # Ensure velocities are positive
    velocities = np.maximum(velocities, base_velocity * 0.1)
    
    # Calculate traversal times
    times = L / velocities
    
    return times


def power_law(L, alpha, T0):
    """
    Power law function: T = T0 * L^alpha
    
    Parameters
    ----------
    L : float or array
        System size
    alpha : float
        Scaling exponent
    T0 : float
        Prefactor (intercept in log-log space)
        
    Returns
    -------
    float or array
        Characteristic time
    """
    return T0 * np.power(L, alpha)


def fit_power_law(L_values, T_values, T_errors=None):
    """
    Fit a power law T = T0 * L^alpha using linear regression in log-log space.
    
    Parameters
    ----------
    L_values : array
        System sizes
    T_values : array
        Mean traversal times
    T_errors : array, optional
        Standard errors of T values
        
    Returns
    -------
    dict
        Fitting results including alpha, T0, R^2, and confidence intervals
    """
    log_L = np.log10(L_values)
    log_T = np.log10(T_values)
    
    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
    
    # Calculate confidence intervals using bootstrap
    n_bootstrap = 1000
    rng = np.random.default_rng(RANDOM_SEED + 1)
    bootstrap_slopes = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(len(log_L), size=len(log_L), replace=True)
        boot_log_L = log_L[indices]
        boot_log_T = log_T[indices]
        boot_slope, _, _, _, _ = stats.linregress(boot_log_L, boot_log_T)
        bootstrap_slopes.append(boot_slope)
    
    bootstrap_slopes = np.array(bootstrap_slopes)
    ci_lower = np.percentile(bootstrap_slopes, 2.5)
    ci_upper = np.percentile(bootstrap_slopes, 97.5)
    
    return {
        'alpha': slope,
        'alpha_std_err': std_err,
        'alpha_ci_lower': ci_lower,
        'alpha_ci_upper': ci_upper,
        'T0': 10**intercept,
        'R_squared': r_value**2,
        'p_value': p_value
    }


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation():
    """
    Run the complete ballistic 1-D simulation.
    
    Returns
    -------
    pandas.DataFrame
        Results dataframe with L, T_mean, T_std, T_sem for each system size
    dict
        Fitting results
    """
    print("=" * 60)
    print("RTM SIMULATION: Ballistic 1-D Propagation")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    print(f"  - System sizes: {SYSTEM_SIZES[0]} to {SYSTEM_SIZES[-1]}")
    print(f"  - Number of sizes: {len(SYSTEM_SIZES)}")
    print(f"  - Trials per size: {N_TRIALS}")
    print(f"  - Base velocity: {BASE_VELOCITY}")
    print(f"  - Velocity noise: {VELOCITY_NOISE_STD * 100:.1f}%")
    print(f"  - Random seed: {RANDOM_SEED}")
    print()
    
    # Initialize random number generator
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Storage for results
    results = []
    
    print("Running simulations...")
    for i, L in enumerate(SYSTEM_SIZES):
        # Run trials
        times = simulate_ballistic_transport(L, N_TRIALS, BASE_VELOCITY, 
                                             VELOCITY_NOISE_STD, rng)
        
        # Calculate statistics
        T_mean = np.mean(times)
        T_std = np.std(times, ddof=1)
        T_sem = T_std / np.sqrt(N_TRIALS)
        
        results.append({
            'L': L,
            'T_mean': T_mean,
            'T_std': T_std,
            'T_sem': T_sem,
            'n_trials': N_TRIALS
        })
        
        print(f"  L = {L:6.0f}: T = {T_mean:10.4f} ± {T_sem:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Fit power law
    print("\nFitting power law T = T0 * L^α...")
    fit_results = fit_power_law(df['L'].values, df['T_mean'].values, 
                                df['T_sem'].values)
    
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Fitted exponent α = {fit_results['alpha']:.4f} ± {fit_results['alpha_std_err']:.4f}")
    print(f"  95% CI: [{fit_results['alpha_ci_lower']:.4f}, {fit_results['alpha_ci_upper']:.4f}]")
    print(f"  R² = {fit_results['R_squared']:.6f}")
    print(f"  p-value = {fit_results['p_value']:.2e}")
    print(f"\n  Expected (theoretical): α = 1.00")
    print(f"  RTM paper reported: α ≈ 1.03")
    
    # Check if result matches expectation
    expected_alpha = 1.0
    if abs(fit_results['alpha'] - expected_alpha) < 0.1:
        print(f"\n  ✓ CONFIRMED: Result matches RTM prediction for ballistic regime")
    else:
        print(f"\n  ⚠ WARNING: Result deviates from expected value")
    
    return df, fit_results


def create_plots(df, fit_results):
    """
    Create visualization plots.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results
    fit_results : dict
        Power law fitting results
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Log-log plot with fit
    ax1 = axes[0]
    ax1.errorbar(df['L'], df['T_mean'], yerr=df['T_sem'], 
                 fmt='o', markersize=8, capsize=3, label='Simulation data')
    
    # Fit line
    L_fit = np.logspace(np.log10(df['L'].min()), np.log10(df['L'].max()), 100)
    T_fit = power_law(L_fit, fit_results['alpha'], fit_results['T0'])
    ax1.plot(L_fit, T_fit, 'r-', linewidth=2, 
             label=f'Fit: α = {fit_results["alpha"]:.3f} ± {fit_results["alpha_std_err"]:.3f}')
    
    # Theoretical line (α = 1)
    T_theoretical = power_law(L_fit, 1.0, fit_results['T0'])
    ax1.plot(L_fit, T_theoretical, 'g--', linewidth=1.5, alpha=0.7,
             label='Theoretical: α = 1.00')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('System Size L', fontsize=12)
    ax1.set_ylabel('Traversal Time T', fontsize=12)
    ax1.set_title('Ballistic 1-D: T ∝ L^α Scaling', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    T_predicted = power_law(df['L'].values, fit_results['alpha'], fit_results['T0'])
    residuals = (df['T_mean'].values - T_predicted) / T_predicted * 100
    
    ax2.bar(range(len(df)), residuals, color='steelblue', alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f'{int(L)}' for L in df['L']], rotation=45)
    ax2.set_xlabel('System Size L', fontsize=12)
    ax2.set_ylabel('Residual (%)', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ballistic_1d_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ballistic_1d_results.pdf'))
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    
    plt.show()


def save_results(df, fit_results):
    """
    Save results to CSV files.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results
    fit_results : dict
        Power law fitting results
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_DIR, 'ballistic_1d_data.csv'), index=False)
    
    # Save fit results
    fit_df = pd.DataFrame([{
        'parameter': 'alpha',
        'value': fit_results['alpha'],
        'std_err': fit_results['alpha_std_err'],
        'ci_lower_95': fit_results['alpha_ci_lower'],
        'ci_upper_95': fit_results['alpha_ci_upper'],
        'R_squared': fit_results['R_squared'],
        'p_value': fit_results['p_value'],
        'expected_value': 1.0,
        'rtm_paper_value': 1.03,
        'status': 'CONFIRMED' if abs(fit_results['alpha'] - 1.0) < 0.1 else 'DEVIATION'
    }])
    fit_df.to_csv(os.path.join(OUTPUT_DIR, 'ballistic_1d_fit_results.csv'), index=False)
    
    # Save summary
    summary = f"""RTM Simulation: Ballistic 1-D Propagation
=========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
System sizes: {list(SYSTEM_SIZES)}
Trials per size: {N_TRIALS}
Base velocity: {BASE_VELOCITY}
Velocity noise: {VELOCITY_NOISE_STD * 100:.1f}%
Random seed: {RANDOM_SEED}

RESULTS
-------
Fitted exponent: α = {fit_results['alpha']:.4f} ± {fit_results['alpha_std_err']:.4f}
95% Confidence interval: [{fit_results['alpha_ci_lower']:.4f}, {fit_results['alpha_ci_upper']:.4f}]
R² = {fit_results['R_squared']:.6f}
p-value = {fit_results['p_value']:.2e}

COMPARISON
----------
Expected (theoretical): α = 1.00
RTM paper reported: α ≈ 1.03
This simulation: α = {fit_results['alpha']:.2f}

STATUS: {'CONFIRMED' if abs(fit_results['alpha'] - 1.0) < 0.1 else 'NEEDS REVIEW'}
"""
    
    with open(os.path.join(OUTPUT_DIR, 'ballistic_1d_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"Results saved to {OUTPUT_DIR}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run simulation
    df, fit_results = run_simulation()
    
    # Create plots
    create_plots(df, fit_results)
    
    # Save results
    save_results(df, fit_results)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
