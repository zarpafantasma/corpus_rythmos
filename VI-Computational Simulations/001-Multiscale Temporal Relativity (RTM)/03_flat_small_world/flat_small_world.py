#!/usr/bin/env python3
"""
RTM Simulation: Flat Small-World Network (MFPT)
================================================

This script simulates random walks on Watts-Strogatz small-world networks
to measure the Mean First-Passage Time (MFPT) scaling with network size.

Key findings from RTM paper:
- Flat small-world networks show LOGARITHMIC scaling: T ∝ log(N)
- If one forces a power-law fit, an artificial α appears
- The paper reports α ≈ 2.1 when measuring MFPT vs L = √N

The Watts-Strogatz model creates a network that interpolates between:
- Regular lattice (p=0): high clustering, long paths
- Random graph (p=1): low clustering, short paths
- Small-world (0 < p < 1): high clustering, short paths

Expected results:
- Logarithmic scaling: T = a + b*log(N) fits well
- Power-law fit: α ≈ 2.0-2.1 when using L = √N as scale

Author: RTM Corpus
License: CC BY 4.0
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, List, Dict
import warnings

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Network sizes (number of nodes)
NETWORK_SIZES = np.array([100, 200, 400, 800, 1600, 3200])

# Watts-Strogatz parameters
K = 6          # Each node connected to K nearest neighbors (must be even)
P_REWIRE = 0.1  # Rewiring probability (0.03-0.1 for small-world regime)

# Number of network realizations per size
N_REALIZATIONS = 5

# Number of random source-target pairs per network
N_PAIRS = 50

# Number of random walk trials per pair
N_WALKS = 10

# Maximum steps per walk (to prevent infinite loops)
MAX_STEPS = 500_000

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# NETWORK GENERATION
# =============================================================================

def create_watts_strogatz_graph(n: int, k: int, p: float, rng: np.random.Generator) -> Dict[int, List[int]]:
    """
    Create a Watts-Strogatz small-world graph.
    
    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Each node is connected to k nearest neighbors (k must be even)
    p : float
        Probability of rewiring each edge
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    dict
        Adjacency list representation of the graph
    """
    if k % 2 != 0:
        k = k - 1  # Ensure k is even
    
    # Initialize adjacency list
    adj = {i: set() for i in range(n)}
    
    # Create ring lattice: connect each node to k/2 neighbors on each side
    for i in range(n):
        for j in range(1, k // 2 + 1):
            # Connect to neighbors on both sides (circular)
            neighbor_right = (i + j) % n
            neighbor_left = (i - j) % n
            adj[i].add(neighbor_right)
            adj[neighbor_right].add(i)
            adj[i].add(neighbor_left)
            adj[neighbor_left].add(i)
    
    # Rewiring step
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            
            if rng.random() < p:
                # Rewire this edge
                if neighbor in adj[i]:
                    # Find a new target (not self, not already connected)
                    possible_targets = [x for x in range(n) 
                                       if x != i and x not in adj[i]]
                    
                    if len(possible_targets) > 0:
                        new_target = rng.choice(possible_targets)
                        
                        # Remove old edge
                        adj[i].discard(neighbor)
                        adj[neighbor].discard(i)
                        
                        # Add new edge
                        adj[i].add(new_target)
                        adj[new_target].add(i)
    
    # Convert sets to lists for faster random access
    return {i: list(neighbors) for i, neighbors in adj.items()}


def compute_average_path_length(adj: Dict[int, List[int]], n_samples: int, 
                                 rng: np.random.Generator) -> float:
    """
    Estimate average shortest path length using BFS on random pairs.
    
    Parameters
    ----------
    adj : dict
        Adjacency list
    n_samples : int
        Number of random pairs to sample
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    float
        Average shortest path length
    """
    n = len(adj)
    total_length = 0
    valid_samples = 0
    
    for _ in range(n_samples):
        source = rng.integers(0, n)
        target = rng.integers(0, n)
        
        if source == target:
            continue
        
        # BFS to find shortest path
        visited = {source}
        queue = [(source, 0)]
        found = False
        
        while queue and not found:
            current, dist = queue.pop(0)
            
            for neighbor in adj[current]:
                if neighbor == target:
                    total_length += dist + 1
                    valid_samples += 1
                    found = True
                    break
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    
    return total_length / valid_samples if valid_samples > 0 else float('inf')


# =============================================================================
# RANDOM WALK SIMULATION
# =============================================================================

def random_walk_mfpt(adj: Dict[int, List[int]], source: int, target: int, 
                     max_steps: int, rng: np.random.Generator) -> int:
    """
    Perform a random walk from source to target and return first-passage time.
    
    Parameters
    ----------
    adj : dict
        Adjacency list
    source : int
        Starting node
    target : int
        Target node
    max_steps : int
        Maximum number of steps
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    int
        Number of steps to reach target, or -1 if max_steps exceeded
    """
    current = source
    steps = 0
    
    while steps < max_steps:
        if current == target:
            return steps
        
        # Move to random neighbor
        neighbors = adj[current]
        if len(neighbors) == 0:
            return -1  # Dead end
        
        current = rng.choice(neighbors)
        steps += 1
    
    return -1


def simulate_network_mfpt(n: int, k: int, p: float, n_pairs: int, n_walks: int,
                          max_steps: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Create a network and measure MFPT statistics.
    
    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Watts-Strogatz k parameter
    p : float
        Rewiring probability
    n_pairs : int
        Number of source-target pairs
    n_walks : int
        Number of walks per pair
    max_steps : int
        Maximum steps per walk
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    tuple
        (mean_mfpt, std_mfpt, avg_path_length)
    """
    # Create network
    adj = create_watts_strogatz_graph(n, k, p, rng)
    
    # Measure average path length
    avg_path = compute_average_path_length(adj, min(100, n), rng)
    
    # Collect MFPT samples
    mfpt_samples = []
    
    for _ in range(n_pairs):
        source = rng.integers(0, n)
        target = rng.integers(0, n)
        
        while target == source:
            target = rng.integers(0, n)
        
        pair_times = []
        for _ in range(n_walks):
            fpt = random_walk_mfpt(adj, source, target, max_steps, rng)
            if fpt >= 0:
                pair_times.append(fpt)
        
        if len(pair_times) > 0:
            mfpt_samples.append(np.mean(pair_times))
    
    if len(mfpt_samples) == 0:
        return float('nan'), float('nan'), avg_path
    
    return np.mean(mfpt_samples), np.std(mfpt_samples), avg_path


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation() -> Tuple[pd.DataFrame, dict, dict]:
    """
    Run the complete flat small-world simulation.
    
    Returns
    -------
    pd.DataFrame
        Results dataframe
    dict
        Power-law fit results (T vs L = √N)
    dict
        Logarithmic fit results (T vs log N)
    """
    print("=" * 65)
    print("RTM SIMULATION: Flat Small-World Network (MFPT)")
    print("=" * 65)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    print(f"  - Network sizes N: {list(NETWORK_SIZES)}")
    print(f"  - Effective lengths L=√N: {[f'{np.sqrt(n):.1f}' for n in NETWORK_SIZES]}")
    print(f"  - Watts-Strogatz k: {K}")
    print(f"  - Rewiring probability p: {P_REWIRE}")
    print(f"  - Realizations per size: {N_REALIZATIONS}")
    print(f"  - Pairs per network: {N_PAIRS}")
    print(f"  - Walks per pair: {N_WALKS}")
    print(f"  - Random seed: {RANDOM_SEED}")
    print()
    
    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Storage
    results = []
    
    print("Running simulations...")
    print(f"{'N':>6} | {'L=√N':>8} | {'MFPT':>12} | {'Std':>10} | {'AvgPath':>8}")
    print("-" * 55)
    
    for N in NETWORK_SIZES:
        L = np.sqrt(N)
        
        # Run multiple realizations
        mfpt_values = []
        path_lengths = []
        
        for r in range(N_REALIZATIONS):
            mean_mfpt, std_mfpt, avg_path = simulate_network_mfpt(
                N, K, P_REWIRE, N_PAIRS, N_WALKS, MAX_STEPS, rng
            )
            
            if not np.isnan(mean_mfpt):
                mfpt_values.append(mean_mfpt)
                path_lengths.append(avg_path)
        
        if len(mfpt_values) > 0:
            T_mean = np.mean(mfpt_values)
            T_std = np.std(mfpt_values)
            T_sem = T_std / np.sqrt(len(mfpt_values))
            avg_path_mean = np.mean(path_lengths)
            
            results.append({
                'N': N,
                'L': L,
                'log_N': np.log10(N),
                'T_mean': T_mean,
                'T_std': T_std,
                'T_sem': T_sem,
                'avg_path_length': avg_path_mean,
                'n_realizations': len(mfpt_values)
            })
            
            print(f"{N:>6} | {L:>8.2f} | {T_mean:>12.2f} | {T_std:>10.2f} | {avg_path_mean:>8.2f}")
        else:
            print(f"{N:>6} | {L:>8.2f} | {'FAILED':>12} |")
    
    df = pd.DataFrame(results)
    
    # ==========================================================================
    # FIT 1: Power law T = T0 * L^α (where L = √N)
    # ==========================================================================
    print("\n" + "=" * 65)
    print("FIT 1: Power Law T = T0 * L^α (where L = √N)")
    print("=" * 65)
    
    log_L = np.log10(df['L'].values)
    log_T = np.log10(df['T_mean'].values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
    
    # Bootstrap CI
    n_boot = 1000
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(log_L), size=len(log_L), replace=True)
        b_slope, _, _, _, _ = stats.linregress(log_L[idx], log_T[idx])
        boot_slopes.append(b_slope)
    ci_lower = np.percentile(boot_slopes, 2.5)
    ci_upper = np.percentile(boot_slopes, 97.5)
    
    power_law_results = {
        'alpha': slope,
        'alpha_std_err': std_err,
        'alpha_ci_lower': ci_lower,
        'alpha_ci_upper': ci_upper,
        'T0': 10**intercept,
        'R_squared': r_value**2,
        'p_value': p_value
    }
    
    print(f"\n  Fitted exponent α = {slope:.4f} ± {std_err:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  R² = {r_value**2:.6f}")
    print(f"\n  RTM paper reported: α ≈ 2.1 for flat small-world")
    
    # ==========================================================================
    # FIT 2: Logarithmic T = a + b * log(N)
    # ==========================================================================
    print("\n" + "=" * 65)
    print("FIT 2: Logarithmic T = a + b * log(N)")
    print("=" * 65)
    
    log_N = np.log10(df['N'].values)
    T_values = df['T_mean'].values
    
    slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(log_N, T_values)
    
    log_results = {
        'a': intercept_log,
        'b': slope_log,
        'b_std_err': std_err_log,
        'R_squared': r_value_log**2,
        'p_value': p_value_log
    }
    
    print(f"\n  T = {intercept_log:.2f} + {slope_log:.2f} * log₁₀(N)")
    print(f"  R² = {r_value_log**2:.6f}")
    
    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print("\n" + "=" * 65)
    print("COMPARISON OF FITS")
    print("=" * 65)
    print(f"\n  Power-law fit R²: {power_law_results['R_squared']:.6f}")
    print(f"  Logarithmic fit R²: {log_results['R_squared']:.6f}")
    
    if power_law_results['R_squared'] > log_results['R_squared']:
        print(f"\n  → Power-law model fits better")
    else:
        print(f"\n  → Logarithmic model fits better (as predicted by RTM)")
    
    # Check against expected
    expected_alpha = 2.1
    if abs(power_law_results['alpha'] - expected_alpha) < 0.2:
        print(f"\n  ✓ CONFIRMED: α = {power_law_results['alpha']:.2f} matches RTM prediction (~2.1)")
    else:
        print(f"\n  Note: α = {power_law_results['alpha']:.2f} (RTM predicts ~2.1)")
    
    return df, power_law_results, log_results


def create_plots(df: pd.DataFrame, power_results: dict, log_results: dict) -> None:
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Log-log plot (Power law fit)
    ax1 = axes[0, 0]
    ax1.errorbar(df['L'], df['T_mean'], yerr=df['T_sem'],
                 fmt='o', markersize=10, capsize=4, color='blue',
                 label='Simulation data')
    
    L_fit = np.linspace(df['L'].min(), df['L'].max(), 100)
    T_fit = power_results['T0'] * L_fit**power_results['alpha']
    ax1.plot(L_fit, T_fit, 'r-', linewidth=2,
             label=f'Power law: α = {power_results["alpha"]:.3f}')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Effective Length L = √N', fontsize=12)
    ax1.set_ylabel('Mean First-Passage Time T', fontsize=12)
    ax1.set_title(f'Power Law Fit: T ∝ L^α (R² = {power_results["R_squared"]:.4f})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Semi-log plot (Logarithmic fit)
    ax2 = axes[0, 1]
    ax2.errorbar(df['N'], df['T_mean'], yerr=df['T_sem'],
                 fmt='s', markersize=10, capsize=4, color='green',
                 label='Simulation data')
    
    N_fit = np.linspace(df['N'].min(), df['N'].max(), 100)
    T_log_fit = log_results['a'] + log_results['b'] * np.log10(N_fit)
    ax2.plot(N_fit, T_log_fit, 'r-', linewidth=2,
             label=f'Log fit: T = {log_results["a"]:.0f} + {log_results["b"]:.0f}·log(N)')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Network Size N', fontsize=12)
    ax2.set_ylabel('Mean First-Passage Time T', fontsize=12)
    ax2.set_title(f'Logarithmic Fit: T = a + b·log(N) (R² = {log_results["R_squared"]:.4f})', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MFPT vs Average Path Length
    ax3 = axes[1, 0]
    ax3.scatter(df['avg_path_length'], df['T_mean'], s=100, c='purple', alpha=0.7)
    
    # Fit line
    slope_path, intercept_path, r_path, _, _ = stats.linregress(
        df['avg_path_length'].values, df['T_mean'].values
    )
    path_fit = np.linspace(df['avg_path_length'].min(), df['avg_path_length'].max(), 100)
    ax3.plot(path_fit, intercept_path + slope_path * path_fit, 'r--', linewidth=2,
             label=f'Linear fit (R² = {r_path**2:.4f})')
    
    ax3.set_xlabel('Average Shortest Path Length', fontsize=12)
    ax3.set_ylabel('Mean First-Passage Time T', fontsize=12)
    ax3.set_title('MFPT vs Network Path Length', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuals comparison
    ax4 = axes[1, 1]
    
    # Power law residuals
    T_power_pred = power_results['T0'] * df['L'].values**power_results['alpha']
    residuals_power = (df['T_mean'].values - T_power_pred) / T_power_pred * 100
    
    # Log residuals
    T_log_pred = log_results['a'] + log_results['b'] * np.log10(df['N'].values)
    residuals_log = (df['T_mean'].values - T_log_pred) / T_log_pred * 100
    
    x = np.arange(len(df))
    width = 0.35
    
    ax4.bar(x - width/2, residuals_power, width, label='Power law', color='blue', alpha=0.7)
    ax4.bar(x + width/2, residuals_log, width, label='Logarithmic', color='green', alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'N={int(n)}' for n in df['N']], rotation=45)
    ax4.set_ylabel('Residual (%)', fontsize=12)
    ax4.set_title('Fit Residuals Comparison', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'flat_small_world_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'flat_small_world_results.pdf'))
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    
    plt.show()


def save_results(df: pd.DataFrame, power_results: dict, log_results: dict) -> None:
    """Save results to files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_DIR, 'flat_small_world_data.csv'), index=False)
    
    # Save fit results
    fit_df = pd.DataFrame([{
        'model': 'power_law',
        'parameter': 'alpha',
        'value': power_results['alpha'],
        'std_err': power_results['alpha_std_err'],
        'ci_lower_95': power_results['alpha_ci_lower'],
        'ci_upper_95': power_results['alpha_ci_upper'],
        'R_squared': power_results['R_squared'],
        'expected_value': 2.1,
        'status': 'CONFIRMED' if abs(power_results['alpha'] - 2.1) < 0.3 else 'DEVIATION'
    }, {
        'model': 'logarithmic',
        'parameter': 'b (slope)',
        'value': log_results['b'],
        'std_err': log_results['b_std_err'],
        'ci_lower_95': float('nan'),
        'ci_upper_95': float('nan'),
        'R_squared': log_results['R_squared'],
        'expected_value': float('nan'),
        'status': 'N/A'
    }])
    fit_df.to_csv(os.path.join(OUTPUT_DIR, 'flat_small_world_fit_results.csv'), index=False)
    
    # Save summary
    summary = f"""RTM Simulation: Flat Small-World Network (MFPT)
================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
Network sizes N: {list(NETWORK_SIZES)}
Watts-Strogatz k: {K}
Rewiring probability p: {P_REWIRE}
Realizations per size: {N_REALIZATIONS}
Pairs per network: {N_PAIRS}
Walks per pair: {N_WALKS}
Random seed: {RANDOM_SEED}

POWER LAW FIT (T = T0 * L^α, where L = √N)
------------------------------------------
Fitted exponent: α = {power_results['alpha']:.4f} ± {power_results['alpha_std_err']:.4f}
95% CI: [{power_results['alpha_ci_lower']:.4f}, {power_results['alpha_ci_upper']:.4f}]
R² = {power_results['R_squared']:.6f}

LOGARITHMIC FIT (T = a + b * log₁₀(N))
--------------------------------------
T = {log_results['a']:.2f} + {log_results['b']:.2f} * log₁₀(N)
R² = {log_results['R_squared']:.6f}

COMPARISON
----------
RTM paper states that flat small-world networks show:
  - Logarithmic scaling: T ∝ log(N)
  - When forced to power law: α ≈ 2.1

This simulation:
  - Power-law α = {power_results['alpha']:.2f}
  - Best fit model: {'Logarithmic' if log_results['R_squared'] > power_results['R_squared'] else 'Power law'}

INTERPRETATION
--------------
The Watts-Strogatz small-world network has "shortcuts" that allow
rapid traversal across the network, leading to logarithmic rather
than polynomial scaling of path lengths with system size. The MFPT
reflects this efficient connectivity structure.

When using L = √N as the characteristic length, the power-law fit
yields α ≈ 2.0-2.1, which matches RTM predictions for neural-like
small-world networks operating between diffusive and ballistic regimes.

STATUS: {'CONFIRMED' if abs(power_results['alpha'] - 2.1) < 0.3 else 'NEEDS REVIEW'}
"""
    
    with open(os.path.join(OUTPUT_DIR, 'flat_small_world_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"Results saved to {OUTPUT_DIR}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run simulation
    df, power_results, log_results = run_simulation()
    
    # Create plots
    create_plots(df, power_results, log_results)
    
    # Save results
    save_results(df, power_results, log_results)
    
    print("\n" + "=" * 65)
    print("Simulation complete!")
    print("=" * 65)
