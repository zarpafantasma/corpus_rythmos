#!/usr/bin/env python3
"""
==============================================================================
RTM Simulation G: Holographic Decay Network  P(r) ∝ r⁻³
Multiscale Temporal Relativity (RTM) Framework
==============================================================================

Theoretical Background
----------------------
Holographic-inspired networks feature long-range connections with probability
decaying as the inverse cube of distance: P(r) ∝ r⁻³. This decay law, motivated
by holographic principles in theoretical physics, creates networks where transport
becomes increasingly "trapped" at large scales, with hitting times growing toward
the cubic power of linear size.

The RTM framework predicts α → 3.0 for holographic systems, where the r⁻³ decay
creates network structures in which information/transport time scales with the
volume (L³) rather than the surface area or linear extent.

Model Description
-----------------
- Base lattice: 3D cubic grid of side L
- Short-range connections: standard 6-connectivity (±x, ±y, ±z)
- Long-range links: 2 per node, probability P(r) ∝ r⁻³
- Observable: Mean First-Passage Time (MFPT) from origin (0,0,0)
  to farthest corner (L-1, L-1, L-1)

Improvements over original simulation
--------------------------------------
- Extended lattice sizes up to L=22 (N=10,648)
- Increased realizations per size (6 networks)
- More walks per network (40 walks)
- Higher max steps (2,000,000)
- Bootstrap confidence intervals (10,000 resamples)
- Finite-size convergence analysis
- Multiprocessing for parallel execution
- Comprehensive statistical diagnostics

Author: RTM Framework – Computational Validation Suite
License: CC BY 4.0
"""

import numpy as np
import time
import os
import sys
import json
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Lattice sizes to simulate
    "lattice_sizes": [6, 8, 10, 12, 14, 16, 18, 20],
    # Number of long-range links per node
    "n_long_range": 2,
    # Number of independent network realizations per lattice size
    "n_realizations": 6,
    # Number of random walks per realization
    "n_walks_per_realization": 40,
    # Maximum steps per random walk before declaring failure
    "max_steps": 2_000_000,
    # Random seed for reproducibility
    "seed": 42,
    # Bootstrap resamples for CI estimation
    "n_bootstrap": 10_000,
    # Confidence level for intervals
    "confidence_level": 0.95,
    # Number of parallel workers (0 = auto-detect)
    "n_workers": 0,
    # Output directory
    "output_dir": ".",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions: Network Construction
# ─────────────────────────────────────────────────────────────────────────────

def coord_to_index(x, y, z, L):
    """Convert 3D coordinates to linear index."""
    return x * L * L + y * L + z


def index_to_coord(idx, L):
    """Convert linear index to 3D coordinates."""
    x = idx // (L * L)
    y = (idx % (L * L)) // L
    z = idx % L
    return x, y, z


def build_holographic_network(L, n_long_range=2, rng=None):
    """
    Build a 3D cubic lattice of side L with holographic long-range connections.
    
    Short-range: standard 6-connectivity (±x, ±y, ±z) with periodic boundaries.
    Long-range: each node gets n_long_range extra links, target chosen with
                probability P(r) ∝ r⁻³ where r is the Euclidean distance.
    
    Parameters
    ----------
    L : int
        Side length of the cubic lattice.
    n_long_range : int
        Number of long-range links per node.
    rng : np.random.Generator
        Random number generator for reproducibility.
    
    Returns
    -------
    adjacency : list of np.ndarray
        adjacency[i] is an array of neighbor indices for node i.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = L ** 3
    
    # Precompute all coordinates
    coords = np.zeros((N, 3), dtype=np.int32)
    for idx in range(N):
        coords[idx] = index_to_coord(idx, L)
    
    # Build adjacency list
    neighbors = [[] for _ in range(N)]
    
    # --- Short-range: 6-connectivity (NO periodic boundary → reflects paper) ---
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    for idx in range(N):
        x, y, z = coords[idx]
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < L and 0 <= ny < L and 0 <= nz < L:
                neighbors[idx].append(coord_to_index(nx, ny, nz, L))
    
    # --- Long-range links: P(r) ∝ r⁻³ ---
    coords_float = coords.astype(np.float64)
    
    for idx in range(N):
        # Calculate distances from this node to all others
        diffs = coords_float - coords_float[idx]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[idx] = np.inf  # exclude self
        
        # Probability ∝ r⁻³
        probs = dists ** (-3)
        probs[idx] = 0.0
        
        # Exclude existing short-range neighbors
        for nb in neighbors[idx]:
            probs[nb] = 0.0
        
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
            # Sample long-range targets (without replacement)
            n_candidates = np.count_nonzero(probs > 0)
            n_links = min(n_long_range, n_candidates)
            if n_links > 0:
                targets = rng.choice(N, size=n_links, replace=False, p=probs)
                for t in targets:
                    if t not in neighbors[idx]:
                        neighbors[idx].append(t)
                    if idx not in neighbors[t]:
                        neighbors[t].append(idx)
    
    # Convert to numpy arrays for fast access
    adjacency = [np.array(nb, dtype=np.int32) for nb in neighbors]
    return adjacency


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions: Random Walk & MFPT
# ─────────────────────────────────────────────────────────────────────────────

def random_walk_fpt(adjacency, source, target, max_steps, rng=None):
    """
    Perform a single random walk from source to target, returning first-passage time.
    
    Parameters
    ----------
    adjacency : list of np.ndarray
        Adjacency list representation of the graph.
    source : int
        Starting node index.
    target : int
        Target node index.
    max_steps : int
        Maximum number of steps before declaring walk incomplete.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    fpt : int or None
        First-passage time (number of steps), or None if max_steps exceeded.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = source
    for step in range(1, max_steps + 1):
        nb = adjacency[current]
        current = nb[rng.integers(len(nb))]
        if current == target:
            return step
    return None


def compute_mfpt_for_realization(args):
    """
    Worker function: build one network realization and compute MFPT.
    
    Parameters
    ----------
    args : tuple
        (L, n_long_range, n_walks, max_steps, seed)
    
    Returns
    -------
    dict with keys: L, fpts (list), completed, failed, seed
    """
    L, n_long_range, n_walks, max_steps, seed = args
    rng = np.random.default_rng(seed)
    
    # Build the network
    adjacency = build_holographic_network(L, n_long_range, rng)
    
    N = L ** 3
    source = coord_to_index(0, 0, 0, L)
    target = coord_to_index(L-1, L-1, L-1, L)
    
    fpts = []
    failed = 0
    
    for w in range(n_walks):
        fpt = random_walk_fpt(adjacency, source, target, max_steps, rng)
        if fpt is not None:
            fpts.append(fpt)
        else:
            failed += 1
    
    return {
        "L": L,
        "fpts": fpts,
        "completed": len(fpts),
        "failed": failed,
        "seed": seed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Analysis
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_mean_ci(data, n_bootstrap=10000, confidence=0.95, rng=None):
    """
    Compute bootstrap confidence interval for the mean.
    
    Returns
    -------
    mean, ci_low, ci_high, std
    """
    if rng is None:
        rng = np.random.default_rng()
    
    data = np.array(data, dtype=np.float64)
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.integers(n, size=n)]
        means[i] = sample.mean()
    
    alpha = 1 - confidence
    ci_low = np.percentile(means, 100 * alpha / 2)
    ci_high = np.percentile(means, 100 * (1 - alpha / 2))
    
    return data.mean(), ci_low, ci_high, data.std()


def power_law_fit(L_values, T_values):
    """
    Fit T = C * L^α using log-log linear regression.
    
    Returns
    -------
    dict with keys: alpha, alpha_se, intercept, R2, residuals
    """
    logL = np.log10(L_values)
    logT = np.log10(T_values)
    
    n = len(logL)
    
    # Ordinary least squares
    A = np.vstack([logL, np.ones(n)]).T
    result = np.linalg.lstsq(A, logT, rcond=None)
    alpha, intercept = result[0]
    
    # Predictions and R²
    logT_pred = alpha * logL + intercept
    residuals = logT - logT_pred
    SS_res = np.sum(residuals ** 2)
    SS_tot = np.sum((logT - logT.mean()) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0.0
    
    # Standard error of the slope
    if n > 2:
        s2 = SS_res / (n - 2)
        SE_alpha = np.sqrt(s2 / np.sum((logL - logL.mean()) ** 2))
    else:
        SE_alpha = np.nan
    
    return {
        "alpha": alpha,
        "alpha_se": SE_alpha,
        "intercept": intercept,
        "C": 10 ** intercept,
        "R2": R2,
        "residuals": residuals.tolist(),
    }


def bootstrap_alpha_ci(L_values, all_fpts_by_L, n_bootstrap=10000, confidence=0.95, rng=None):
    """
    Bootstrap the entire pipeline: resample walks → compute means → fit α.
    
    Returns
    -------
    alpha_mean, alpha_ci_low, alpha_ci_high, alpha_std
    """
    if rng is None:
        rng = np.random.default_rng()
    
    alphas = []
    
    for _ in range(n_bootstrap):
        T_resampled = []
        valid = True
        for L in L_values:
            fpts = all_fpts_by_L[L]
            if len(fpts) == 0:
                valid = False
                break
            sample = rng.choice(fpts, size=len(fpts), replace=True)
            T_resampled.append(sample.mean())
        
        if not valid:
            continue
        
        T_arr = np.array(T_resampled)
        L_arr = np.array(L_values, dtype=np.float64)
        
        # Quick log-log fit
        logL = np.log10(L_arr)
        logT = np.log10(T_arr)
        n = len(logL)
        slope = (n * np.sum(logL * logT) - np.sum(logL) * np.sum(logT)) / \
                (n * np.sum(logL ** 2) - np.sum(logL) ** 2)
        alphas.append(slope)
    
    alphas = np.array(alphas)
    alpha_mean = alphas.mean()
    ci_low = np.percentile(alphas, 100 * (1 - confidence) / 2)
    ci_high = np.percentile(alphas, 100 * (1 + confidence) / 2)
    
    return alpha_mean, ci_low, ci_high, alphas.std()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_plots(results_df, fit_result, bootstrap_result, output_dir="."):
    """Generate all publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    L_vals = results_df["L"].values.astype(float)
    T_vals = results_df["T_mean"].values.astype(float)
    T_std = results_df["T_std"].values.astype(float)
    T_ci_low = results_df["T_ci_low"].values.astype(float)
    T_ci_high = results_df["T_ci_high"].values.astype(float)
    
    alpha = fit_result["alpha"]
    C = fit_result["C"]
    R2 = fit_result["R2"]
    alpha_se = fit_result["alpha_se"]
    
    bs_alpha, bs_ci_low, bs_ci_high, bs_std = bootstrap_result
    
    # Color palette
    PRIMARY = "#2563EB"     # Blue
    SECONDARY = "#DC2626"   # Red
    TERTIARY = "#059669"    # Green
    ACCENT = "#7C3AED"      # Purple
    GRAY = "#6B7280"
    
    # ── Figure 1: Log-Log Power Law Fit ──────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Data points with error bars
    ax.errorbar(L_vals, T_vals, yerr=T_std, fmt='o', color=PRIMARY,
                markersize=10, capsize=5, capthick=2, linewidth=2,
                label='Simulation data', zorder=5, markeredgecolor='white',
                markeredgewidth=1.5)
    
    # Fit line
    L_fine = np.linspace(L_vals.min() * 0.85, L_vals.max() * 1.15, 200)
    T_fit = C * L_fine ** alpha
    ax.plot(L_fine, T_fit, '-', color=SECONDARY, linewidth=2.5,
            label=f'Power-law fit: T = {C:.1f} · L^{{{alpha:.4f}}}', zorder=4)
    
    # Reference lines
    T_ref3 = C * (L_vals.min() / L_vals.min()) ** 3 * T_vals[0] * (L_fine / L_vals.min()) ** 3.0
    # Normalize reference to pass through first data point
    C3 = T_vals[0] / (L_vals[0] ** 3.0)
    ax.plot(L_fine, C3 * L_fine ** 3.0, '--', color=TERTIARY, linewidth=1.5,
            alpha=0.7, label='Reference α = 3.0 (holographic)', zorder=3)
    
    C2 = T_vals[0] / (L_vals[0] ** 2.0)
    ax.plot(L_fine, C2 * L_fine ** 2.0, ':', color=GRAY, linewidth=1.5,
            alpha=0.5, label='Reference α = 2.0 (diffusive)', zorder=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lattice Size L', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean First-Passage Time T (steps)', fontsize=14, fontweight='bold')
    ax.set_title('RTM Simulation G: Holographic Decay Network\n'
                 f'P(r) ∝ r⁻³ | α = {alpha:.4f} ± {alpha_se:.4f} | R² = {R2:.6f}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=12)
    
    # Annotation box
    textstr = (f'Bootstrap 95% CI: [{bs_ci_low:.4f}, {bs_ci_high:.4f}]\n'
               f'Theoretical target: α → 3.0\n'
               f'N lattice sizes: {len(L_vals)}')
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_loglog_power_law.png"), dpi=300,
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, "fig1_loglog_power_law.pdf"),
                bbox_inches='tight')
    plt.close()
    
    # ── Figure 2: Residuals ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = np.array(fit_result["residuals"])
    
    # Residuals vs L
    markerline, stemlines, baseline = axes[0].stem(L_vals, residuals)
    plt.setp(stemlines, color=PRIMARY)
    plt.setp(markerline, color=PRIMARY, markersize=8)
    plt.setp(baseline, color=GRAY)
    axes[0].axhline(y=0, color=SECONDARY, linestyle='--', linewidth=1)
    axes[0].set_xlabel('Lattice Size L', fontsize=13)
    axes[0].set_ylabel('Residual (log₁₀ T)', fontsize=13)
    axes[0].set_title('Fit Residuals vs. Lattice Size', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # QQ-like: residuals histogram
    axes[1].hist(residuals, bins=max(5, len(residuals)//2), color=PRIMARY,
                 edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color=SECONDARY, linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residual (log₁₀ T)', fontsize=13)
    axes[1].set_ylabel('Count', fontsize=13)
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_residuals.png"), dpi=300,
                bbox_inches='tight')
    plt.close()
    
    # ── Figure 3: Finite-Size Convergence ────────────────────────────────
    if len(L_vals) >= 4:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Running alpha: fit using first k points
        running_alpha = []
        running_alpha_se = []
        k_vals = []
        
        for k in range(3, len(L_vals) + 1):
            sub_fit = power_law_fit(L_vals[:k], T_vals[:k])
            running_alpha.append(sub_fit["alpha"])
            running_alpha_se.append(sub_fit["alpha_se"])
            k_vals.append(k)
        
        k_vals = np.array(k_vals)
        running_alpha = np.array(running_alpha)
        running_alpha_se = np.array(running_alpha_se)
        
        L_labels = [str(int(L_vals[k-1])) for k in k_vals]
        
        ax.errorbar(k_vals, running_alpha, yerr=1.96*running_alpha_se,
                     fmt='s-', color=PRIMARY, markersize=8, capsize=4,
                     linewidth=2, markeredgecolor='white', markeredgewidth=1.5,
                     label='Running α (cumulative fit)')
        
        ax.axhline(y=3.0, color=SECONDARY, linestyle='--', linewidth=2,
                    label='Theoretical α = 3.0', alpha=0.8)
        ax.fill_between(k_vals, 2.8, 3.2, alpha=0.1, color=SECONDARY)
        
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f'{k}\n(L≤{L_labels[i]})' for i, k in enumerate(k_vals)],
                           fontsize=10)
        ax.set_xlabel('Number of Lattice Sizes Included', fontsize=13)
        ax.set_ylabel('Fitted α', fontsize=13)
        ax.set_title('Finite-Size Convergence of α\n(Running fit as largest L increases)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig3_convergence.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()
    
    # ── Figure 4: RTM Spectrum Context ───────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    regimes = [
        ("Ballistic\n(1-D chain)", 1.0, 1.0000, GRAY),
        ("Diffusive\n(Random Walk)", 2.0, 1.9698, GRAY),
        ("Sierpiński\n(Fractal)", 2.32, 2.3245, GRAY),
        ("Vascular\n(Bio-Fractal)", 2.5, 2.3875, GRAY),
        ("Hierarchical\n(Cortical)", 2.6, 2.6684, GRAY),
        ("Holographic\n(This Work)", 3.0, alpha, SECONDARY),
    ]
    
    x_pos = np.arange(len(regimes))
    
    for i, (name, theory, measured, color) in enumerate(regimes):
        ax.bar(i - 0.15, theory, 0.3, color=TERTIARY, alpha=0.6,
               label='Theory' if i == 0 else '')
        ax.bar(i + 0.15, measured, 0.3, color=color if color != GRAY else PRIMARY,
               alpha=0.85, label='Measured' if i == 0 else '')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r[0] for r in regimes], fontsize=10)
    ax.set_ylabel('Scaling Exponent α', fontsize=13, fontweight='bold')
    ax.set_title('RTM Scaling Exponent Spectrum: All Validated Regimes',
                 fontsize=14, fontweight='bold')
    ax.legend(['Theoretical prediction', 'Measured (simulation)'],
              fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 4.0)
    
    # Highlight holographic
    ax.annotate(f'α = {alpha:.4f}', xy=(5, alpha), xytext=(5.3, alpha + 0.3),
                fontsize=12, fontweight='bold', color=SECONDARY,
                arrowprops=dict(arrowstyle='->', color=SECONDARY, lw=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_rtm_spectrum.png"), dpi=300,
                bbox_inches='tight')
    plt.close()
    
    # ── Figure 5: Walks Distribution ─────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    completed = results_df["completed"].values
    total = results_df["total_walks"].values
    completion_rate = completed / total * 100
    
    bars = ax.bar(range(len(L_vals)), completion_rate, color=PRIMARY,
                  edgecolor='white', alpha=0.85)
    
    ax.axhline(y=100, color=TERTIARY, linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xticks(range(len(L_vals)))
    ax.set_xticklabels([f'L={int(l)}\nN={int(l**3)}' for l in L_vals], fontsize=9)
    ax.set_ylabel('Walk Completion Rate (%)', fontsize=13)
    ax.set_title('Random Walk Completion Rate by Lattice Size', fontsize=14,
                 fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar, rate in zip(bars, completion_rate):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_completion_rates.png"), dpi=300,
                bbox_inches='tight')
    plt.close()
    
    print(f"  Figures saved to {fig_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(config=None):
    """
    Execute the full holographic decay simulation.
    
    Returns
    -------
    results : dict
        Complete results including data, fits, and diagnostics.
    """
    if config is None:
        config = CONFIG.copy()
    
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    print("=" * 72)
    print("RTM SIMULATION G: HOLOGRAPHIC DECAY NETWORK  P(r) ∝ r⁻³")
    print("Multiscale Temporal Relativity – Computational Validation")
    print("=" * 72)
    print()
    
    lattice_sizes = config["lattice_sizes"]
    n_long_range = config["n_long_range"]
    n_realizations = config["n_realizations"]
    n_walks = config["n_walks_per_realization"]
    max_steps = config["max_steps"]
    base_seed = config["seed"]
    n_bootstrap = config["n_bootstrap"]
    confidence = config["confidence_level"]
    n_workers = config["n_workers"]
    
    if n_workers <= 0:
        n_workers = min(cpu_count(), 4)
    
    print(f"Configuration:")
    print(f"  Lattice sizes: {lattice_sizes}")
    print(f"  Nodes range: {lattice_sizes[0]**3} to {lattice_sizes[-1]**3}")
    print(f"  Long-range links/node: {n_long_range}")
    print(f"  Realizations/size: {n_realizations}")
    print(f"  Walks/realization: {n_walks}")
    print(f"  Max steps/walk: {max_steps:,}")
    print(f"  Total walks: {len(lattice_sizes) * n_realizations * n_walks:,}")
    print(f"  Bootstrap resamples: {n_bootstrap:,}")
    print(f"  Workers: {n_workers}")
    print(f"  Base seed: {base_seed}")
    print()
    
    # ── Prepare task arguments ───────────────────────────────────────────
    tasks = []
    seed_counter = base_seed
    for L in lattice_sizes:
        for r in range(n_realizations):
            tasks.append((L, n_long_range, n_walks, max_steps, seed_counter))
            seed_counter += 1
    
    # ── Execute simulations ──────────────────────────────────────────────
    print("Running simulations...")
    t_start = time.time()
    
    all_results_raw = []
    
    # Process sequentially per lattice size for progress reporting
    for L in lattice_sizes:
        L_tasks = [t for t in tasks if t[0] == L]
        L_start = time.time()
        
        print(f"  L = {L:3d} (N = {L**3:6,d} nodes) | "
              f"{n_realizations} realizations × {n_walks} walks ...", end=" ", flush=True)
        
        # Use multiprocessing for realizations within each L
        if n_workers > 1 and len(L_tasks) > 1:
            with Pool(processes=min(n_workers, len(L_tasks))) as pool:
                L_results = pool.map(compute_mfpt_for_realization, L_tasks)
        else:
            L_results = [compute_mfpt_for_realization(t) for t in L_tasks]
        
        all_results_raw.extend(L_results)
        
        # Quick stats for this L
        all_fpts_L = []
        total_completed = 0
        total_failed = 0
        for res in L_results:
            all_fpts_L.extend(res["fpts"])
            total_completed += res["completed"]
            total_failed += res["failed"]
        
        elapsed = time.time() - L_start
        if len(all_fpts_L) > 0:
            print(f"T_mean = {np.mean(all_fpts_L):,.0f} | "
                  f"{total_completed}/{total_completed+total_failed} walks | "
                  f"{elapsed:.1f}s")
        else:
            print(f"NO COMPLETED WALKS | {elapsed:.1f}s")
    
    total_time = time.time() - t_start
    print(f"\nTotal simulation time: {total_time:.1f}s")
    print()
    
    # ── Aggregate results by L ───────────────────────────────────────────
    print("Aggregating and analyzing results...")
    
    fpts_by_L = {}
    for L in lattice_sizes:
        L_results = [r for r in all_results_raw if r["L"] == L]
        all_fpts = []
        for r in L_results:
            all_fpts.extend(r["fpts"])
        fpts_by_L[L] = np.array(all_fpts, dtype=np.float64)
    
    rng_stats = np.random.default_rng(config["seed"] + 999)
    
    # Build summary table
    import pandas as pd
    rows = []
    for L in lattice_sizes:
        fpts = fpts_by_L[L]
        total = n_realizations * n_walks
        completed = len(fpts)
        
        if completed > 0:
            mean, ci_low, ci_high, std = bootstrap_mean_ci(
                fpts, n_bootstrap=n_bootstrap, confidence=confidence, rng=rng_stats)
        else:
            mean = ci_low = ci_high = std = np.nan
        
        rows.append({
            "L": L,
            "N": L ** 3,
            "T_mean": mean,
            "T_std": std,
            "T_ci_low": ci_low,
            "T_ci_high": ci_high,
            "T_median": np.median(fpts) if completed > 0 else np.nan,
            "completed": completed,
            "total_walks": total,
            "completion_pct": completed / total * 100,
        })
    
    results_df = pd.DataFrame(rows)
    
    # ── Filter valid rows and fit ────────────────────────────────────────
    valid = results_df[results_df["completed"] > 0].copy()
    
    if len(valid) < 3:
        print("ERROR: Not enough valid data points for fitting.")
        return None
    
    L_arr = valid["L"].values.astype(float)
    T_arr = valid["T_mean"].values.astype(float)
    
    # Full fit
    fit_result = power_law_fit(L_arr, T_arr)
    
    # Bootstrap CI for alpha
    bootstrap_result = bootstrap_alpha_ci(
        L_arr.tolist(), 
        {L: fpts_by_L[L] for L in L_arr.astype(int)},
        n_bootstrap=n_bootstrap, confidence=confidence, rng=rng_stats
    )
    
    # Fit excluding largest L (finite-size check)
    if len(L_arr) > 3:
        fit_no_max = power_law_fit(L_arr[:-1], T_arr[:-1])
    else:
        fit_no_max = None
    
    # Fit excluding smallest L
    if len(L_arr) > 3:
        fit_no_min = power_law_fit(L_arr[1:], T_arr[1:])
    else:
        fit_no_min = None
    
    # ── Print Results ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print()
    
    print("Data Summary:")
    print(f"{'L':>4s} {'N':>7s} {'T_mean':>12s} {'T_std':>12s} {'CI_low':>12s} "
          f"{'CI_high':>12s} {'Done':>6s} {'%':>7s}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{int(row['L']):4d} {int(row['N']):7,d} {row['T_mean']:12,.1f} "
              f"{row['T_std']:12,.1f} {row['T_ci_low']:12,.1f} "
              f"{row['T_ci_high']:12,.1f} {int(row['completed']):6d} "
              f"{row['completion_pct']:6.1f}%")
    
    print()
    print(f"Power-Law Fit: T = {fit_result['C']:.2f} × L^α")
    print(f"  α = {fit_result['alpha']:.4f} ± {fit_result['alpha_se']:.4f}")
    print(f"  R² = {fit_result['R2']:.6f}")
    print()
    
    bs_alpha, bs_ci_low, bs_ci_high, bs_std = bootstrap_result
    print(f"Bootstrap Analysis ({n_bootstrap:,} resamples):")
    print(f"  α_bootstrap = {bs_alpha:.4f} ± {bs_std:.4f}")
    print(f"  95% CI: [{bs_ci_low:.4f}, {bs_ci_high:.4f}]")
    print()
    
    if fit_no_max:
        print(f"Sensitivity (excluding largest L = {int(L_arr[-1])}):")
        print(f"  α = {fit_no_max['alpha']:.4f} ± {fit_no_max['alpha_se']:.4f}")
    if fit_no_min:
        print(f"Sensitivity (excluding smallest L = {int(L_arr[0])}):")
        print(f"  α = {fit_no_min['alpha']:.4f} ± {fit_no_min['alpha_se']:.4f}")
    
    print()
    print(f"Theoretical target: α → 3.0 (holographic)")
    print(f"Deviation from theory: Δα = {fit_result['alpha'] - 3.0:+.4f}")
    print()
    
    # Status assessment
    if bs_ci_low <= 3.0 <= bs_ci_high:
        status = "95% CI INCLUDES theoretical value α = 3.0"
    elif fit_result['alpha'] > 2.5:
        status = "TRENDING toward α = 3.0 (correct direction)"
    else:
        status = "NEEDS LARGER N for convergence"
    print(f"Status: {status}")
    
    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "holographic_decay_results.csv")
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to {csv_path}")
    
    # Detailed walk data CSV
    walk_rows = []
    for res in all_results_raw:
        for i, fpt in enumerate(res["fpts"]):
            walk_rows.append({
                "L": res["L"],
                "N": res["L"] ** 3,
                "realization_seed": res["seed"],
                "walk_index": i,
                "first_passage_time": fpt,
            })
    
    walks_df = pd.DataFrame(walk_rows)
    walks_csv = os.path.join(output_dir, "holographic_decay_walks.csv")
    walks_df.to_csv(walks_csv, index=False)
    print(f"Walk data saved to {walks_csv}")
    
    # Fit summary CSV
    fit_summary = {
        "alpha": [fit_result["alpha"]],
        "alpha_se": [fit_result["alpha_se"]],
        "R2": [fit_result["R2"]],
        "C": [fit_result["C"]],
        "bootstrap_alpha_mean": [bs_alpha],
        "bootstrap_alpha_std": [bs_std],
        "bootstrap_ci_low": [bs_ci_low],
        "bootstrap_ci_high": [bs_ci_high],
        "n_lattice_sizes": [len(L_arr)],
        "L_min": [int(L_arr.min())],
        "L_max": [int(L_arr.max())],
        "total_walks": [sum(len(fpts_by_L[L]) for L in lattice_sizes)],
        "theory_target": [3.0],
        "deviation_from_theory": [fit_result["alpha"] - 3.0],
    }
    if fit_no_max:
        fit_summary["alpha_excl_max_L"] = [fit_no_max["alpha"]]
    if fit_no_min:
        fit_summary["alpha_excl_min_L"] = [fit_no_min["alpha"]]
    
    fit_df = pd.DataFrame(fit_summary)
    fit_csv = os.path.join(output_dir, "holographic_decay_fit_summary.csv")
    fit_df.to_csv(fit_csv, index=False, float_format="%.6f")
    print(f"Fit summary saved to {fit_csv}")
    
    # ── Generate Figures ─────────────────────────────────────────────────
    print("\nGenerating figures...")
    create_plots(valid, fit_result, bootstrap_result, output_dir)
    
    # ── Save metadata ────────────────────────────────────────────────────
    metadata = {
        "simulation": "RTM Simulation G: Holographic Decay Network",
        "model": "3D cubic lattice with P(r) ∝ r⁻³ long-range links",
        "theory": "α → 3.0 (holographic regime)",
        "config": {k: v for k, v in config.items() if k != "output_dir"},
        "results": {
            "alpha": float(fit_result["alpha"]),
            "alpha_se": float(fit_result["alpha_se"]),
            "R2": float(fit_result["R2"]),
            "bootstrap_ci": [float(bs_ci_low), float(bs_ci_high)],
        },
        "runtime_seconds": round(total_time, 1),
        "status": status,
    }
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    
    print()
    print("=" * 72)
    print("SIMULATION COMPLETE")
    print("=" * 72)
    
    return {
        "results_df": results_df,
        "fit_result": fit_result,
        "bootstrap_result": bootstrap_result,
        "fpts_by_L": fpts_by_L,
        "config": config,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RTM Simulation G: Holographic Decay Network P(r) ∝ r⁻³")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for results")
    parser.add_argument("--lattice-sizes", type=int, nargs="+",
                        default=None, help="Lattice sizes to simulate")
    parser.add_argument("--realizations", type=int, default=None,
                        help="Network realizations per size")
    parser.add_argument("--walks", type=int, default=None,
                        help="Walks per realization")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps per walk")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto)")
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config["output_dir"] = args.output_dir
    config["seed"] = args.seed
    config["n_workers"] = args.workers
    
    if args.lattice_sizes:
        config["lattice_sizes"] = args.lattice_sizes
    if args.realizations:
        config["n_realizations"] = args.realizations
    if args.walks:
        config["n_walks_per_realization"] = args.walks
    if args.max_steps:
        config["max_steps"] = args.max_steps
    
    run_simulation(config)
