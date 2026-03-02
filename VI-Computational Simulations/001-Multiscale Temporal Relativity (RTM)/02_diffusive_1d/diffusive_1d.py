#!/usr/bin/env python3
"""
RTM Simulation: Diffusive 1-D (Random Walk)
============================================

This script simulates classical diffusion via random walk in a 1-D system
to verify the RTM prediction that α ≈ 2 for diffusive transport.

In diffusive transport, a particle undergoes random walk (Brownian motion).
The mean first-passage time (MFPT) to traverse a distance L scales as T ∝ L^α 
with α = 2.

This is derived from the diffusion equation: ⟨x²⟩ = 2Dt
To travel distance L: T ∝ L²/D → α = 2

This serves as the diffusive benchmark for RTM temporal-relativity tests.

Expected result: α ≈ 2.00

Author: RTM Corpus
License: CC BY 4.0
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, List

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# System sizes (target distances in lattice units)
# Keep sizes moderate for reasonable computation time
SYSTEM_SIZES = np.array([5, 10, 20, 40, 80, 160, 320])

# Number of random walk trials per system size
N_TRIALS = 200

# Maximum number of steps (to prevent infinite loops)
# For L=320, theoretical MFPT = 320² = 102,400, so we need margin
MAX_STEPS = 2_000_000

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def random_walk_1d_mfpt(target_distance: int, max_steps: int, rng: np.random.Generator) -> int:
    """
    Perform a 1-D random walk and return the first-passage time to reach 
    the target distance from origin.
    
    OPTIMIZED VERSION using cumulative sum for massive speedup.
    
    Parameters
    ----------
    target_distance : int
        Distance from origin to reach (absorbing boundary at ±target_distance)
    max_steps : int
        Maximum number of steps before giving up
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    int
        Number of steps (first-passage time), or -1 if max_steps reached
    """
    # Process in chunks to balance memory and speed
    chunk_size = min(100_000, max_steps)
    position = 0
    total_steps = 0
    
    while total_steps < max_steps:
        remaining = max_steps - total_steps
        current_chunk = min(chunk_size, remaining)
        
        # Generate all steps at once
        steps = rng.choice(np.array([-1, 1], dtype=np.int32), size=current_chunk)
        
        # Compute cumulative positions
        cumulative = np.cumsum(steps) + position
        
        # Find first crossing of ±target_distance
        crossings = np.where(np.abs(cumulative) >= target_distance)[0]
        
        if len(crossings) > 0:
            # Found a crossing - return the step number
            first_crossing = crossings[0]
            return total_steps + first_crossing + 1
        
        # Update position and continue
        position = cumulative[-1]
        total_steps += current_chunk
    
    return -1


def simulate_diffusive_transport(L: int, n_trials: int, max_steps: int, 
                                  rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Run multiple random walk trials and collect first-passage times.
    
    Parameters
    ----------
    L : int
        Target distance (system size)
    n_trials : int
        Number of independent random walk trials
    max_steps : int
        Maximum steps per trial
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    times : numpy.ndarray
        Array of first-passage times for successful trials
    n_failed : int
        Number of trials that didn't reach target within max_steps
    """
    times = []
    n_failed = 0
    
    for _ in range(n_trials):
        fpt = random_walk_1d_mfpt(L, max_steps, rng)
        if fpt > 0:
            times.append(fpt)
        else:
            n_failed += 1
    
    return np.array(times), n_failed


def power_law(L: np.ndarray, alpha: float, T0: float) -> np.ndarray:
    """
    Power law function: T = T0 * L^alpha
    
    Parameters
    ----------
    L : array
        System sizes
    alpha : float
        Scaling exponent
    T0 : float
        Prefactor
        
    Returns
    -------
    array
        Characteristic times
    """
    return T0 * np.power(L.astype(float), alpha)


def fit_power_law(L_values: np.ndarray, T_values: np.ndarray, 
                  T_errors: np.ndarray = None) -> dict:
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
        Fitting results including alpha, T0, R², and confidence intervals
    """
    log_L = np.log10(L_values.astype(float))
    log_T = np.log10(T_values)
    
    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
    
    # Bootstrap for confidence intervals
    n_bootstrap = 1000
    rng = np.random.default_rng(RANDOM_SEED + 1)
    bootstrap_slopes = []
    
    for _ in range(n_bootstrap):
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
# THEORETICAL REFERENCE
# =============================================================================

def theoretical_mfpt_1d(L: int) -> float:
    """
    Theoretical mean first-passage time for 1-D symmetric random walk.
    
    For a random walk starting at origin with absorbing boundaries at ±L,
    the exact MFPT is L² (in units where step size = 1 and time step = 1).
    
    This is derived from solving the discrete Laplace equation with 
    boundary conditions.
    
    Parameters
    ----------
    L : int
        Target distance
        
    Returns
    -------
    float
        Theoretical MFPT = L²
    """
    return float(L * L)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation() -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete diffusive 1-D simulation.
    
    Returns
    -------
    pd.DataFrame
        Results dataframe with L, T_mean, T_std, T_sem, T_theoretical
    dict
        Fitting results
    """
    print("=" * 60)
    print("RTM SIMULATION: Diffusive 1-D (Random Walk)")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    print(f"  - System sizes (L): {list(SYSTEM_SIZES)}")
    print(f"  - Number of sizes: {len(SYSTEM_SIZES)}")
    print(f"  - Trials per size: {N_TRIALS}")
    print(f"  - Max steps per trial: {MAX_STEPS:,}")
    print(f"  - Random seed: {RANDOM_SEED}")
    print()
    
    # Initialize random number generator
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Storage for results
    results = []
    
    print("Running simulations...")
    print(f"{'L':>8} | {'T_mean':>12} | {'T_theory':>12} | {'Ratio':>8} | {'Failed':>6}")
    print("-" * 60)
    
    for L in SYSTEM_SIZES:
        # Run random walk trials
        times, n_failed = simulate_diffusive_transport(L, N_TRIALS, MAX_STEPS, rng)
        
        if len(times) == 0:
            print(f"  L = {L}: All trials failed!")
            continue
        
        # Calculate statistics
        T_mean = np.mean(times)
        T_std = np.std(times, ddof=1)
        T_sem = T_std / np.sqrt(len(times))
        T_median = np.median(times)
        T_theoretical = theoretical_mfpt_1d(L)
        ratio = T_mean / T_theoretical
        
        results.append({
            'L': L,
            'T_mean': T_mean,
            'T_std': T_std,
            'T_sem': T_sem,
            'T_median': T_median,
            'T_theoretical': T_theoretical,
            'ratio_to_theory': ratio,
            'n_trials': len(times),
            'n_failed': n_failed
        })
        
        print(f"{L:>8} | {T_mean:>12.2f} | {T_theoretical:>12.2f} | {ratio:>8.4f} | {n_failed:>6}")
    
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
    print(f"\n  Expected (theoretical): α = 2.00")
    print(f"  RTM paper reported: α ≈ 2.00")
    
    # Check if result matches expectation
    expected_alpha = 2.0
    tolerance = 0.1
    if abs(fit_results['alpha'] - expected_alpha) < tolerance:
        print(f"\n  ✓ CONFIRMED: Result matches RTM prediction for diffusive regime")
    else:
        print(f"\n  ⚠ WARNING: Result deviates from expected value by {abs(fit_results['alpha'] - expected_alpha):.4f}")
    
    return df, fit_results


def create_plots(df: pd.DataFrame, fit_results: dict) -> None:
    """
    Create visualization plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation results
    fit_results : dict
        Power law fitting results
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Log-log plot with fit
    ax1 = axes[0]
    ax1.errorbar(df['L'], df['T_mean'], yerr=df['T_sem'], 
                 fmt='o', markersize=8, capsize=3, color='blue', label='Simulation data')
    
    # Fit line
    L_fit = np.logspace(np.log10(df['L'].min()), np.log10(df['L'].max()), 100)
    T_fit = power_law(L_fit, fit_results['alpha'], fit_results['T0'])
    ax1.plot(L_fit, T_fit, 'r-', linewidth=2, 
             label=f'Fit: α = {fit_results["alpha"]:.3f} ± {fit_results["alpha_std_err"]:.3f}')
    
    # Theoretical line (α = 2)
    T_theoretical = power_law(L_fit, 2.0, fit_results['T0'])
    ax1.plot(L_fit, T_theoretical, 'g--', linewidth=1.5, alpha=0.7,
             label='Theoretical: α = 2.00')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('System Size L', fontsize=12)
    ax1.set_ylabel('Mean First-Passage Time T', fontsize=12)
    ax1.set_title('Diffusive 1-D: T ∝ L^α Scaling', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Comparison with exact theory
    ax2 = axes[1]
    ax2.scatter(df['T_theoretical'], df['T_mean'], s=80, c='blue', alpha=0.7, 
                label='Simulation vs Theory')
    
    # Perfect agreement line
    max_val = max(df['T_theoretical'].max(), df['T_mean'].max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect agreement')
    
    ax2.set_xlabel('Theoretical MFPT (L²)', fontsize=12)
    ax2.set_ylabel('Simulated MFPT', fontsize=12)
    ax2.set_title('Simulation vs Exact Theory', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Residuals (ratio to theory)
    ax3 = axes[2]
    ax3.bar(range(len(df)), (df['ratio_to_theory'] - 1) * 100, 
            color='steelblue', alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([f'{int(L)}' for L in df['L']], rotation=45)
    ax3.set_xlabel('System Size L', fontsize=12)
    ax3.set_ylabel('Deviation from Theory (%)', fontsize=12)
    ax3.set_title('(T_sim / T_theory - 1) × 100%', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'diffusive_1d_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'diffusive_1d_results.pdf'))
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    
    plt.show()


def save_results(df: pd.DataFrame, fit_results: dict) -> None:
    """
    Save results to CSV files.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation results
    fit_results : dict
        Power law fitting results
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_DIR, 'diffusive_1d_data.csv'), index=False)
    
    # Save fit results
    fit_df = pd.DataFrame([{
        'parameter': 'alpha',
        'value': fit_results['alpha'],
        'std_err': fit_results['alpha_std_err'],
        'ci_lower_95': fit_results['alpha_ci_lower'],
        'ci_upper_95': fit_results['alpha_ci_upper'],
        'R_squared': fit_results['R_squared'],
        'p_value': fit_results['p_value'],
        'expected_value': 2.0,
        'rtm_paper_value': 2.00,
        'status': 'CONFIRMED' if abs(fit_results['alpha'] - 2.0) < 0.1 else 'DEVIATION'
    }])
    fit_df.to_csv(os.path.join(OUTPUT_DIR, 'diffusive_1d_fit_results.csv'), index=False)
    
    # Save summary
    summary = f"""RTM Simulation: Diffusive 1-D (Random Walk)
============================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
System sizes (L): {list(SYSTEM_SIZES)}
Trials per size: {N_TRIALS}
Max steps per trial: {MAX_STEPS:,}
Random seed: {RANDOM_SEED}

RESULTS
-------
Fitted exponent: α = {fit_results['alpha']:.4f} ± {fit_results['alpha_std_err']:.4f}
95% Confidence interval: [{fit_results['alpha_ci_lower']:.4f}, {fit_results['alpha_ci_upper']:.4f}]
R² = {fit_results['R_squared']:.6f}
p-value = {fit_results['p_value']:.2e}

COMPARISON
----------
Expected (theoretical): α = 2.00
RTM paper reported: α ≈ 2.00
This simulation: α = {fit_results['alpha']:.4f}

THEORY
------
For 1-D symmetric random walk with absorbing boundaries at ±L,
the exact mean first-passage time is T = L².
This gives α = 2 exactly (diffusive scaling).

The simulation confirms this classic result, which serves as
the diffusive benchmark in the RTM framework.

STATUS: {'CONFIRMED' if abs(fit_results['alpha'] - 2.0) < 0.1 else 'NEEDS REVIEW'}
"""
    
    with open(os.path.join(OUTPUT_DIR, 'diffusive_1d_summary.txt'), 'w') as f:
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
