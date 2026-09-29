#!/usr/bin/env python3
"""
RTM Simulation: Hierarchical Small-World Network (Baseline Neural-Like)
========================================================================

This script simulates random walks on a hierarchical modular small-world 
network designed to mimic cortical-type neural organization.

Key features (from RTM paper Simulation F):
- Base module: complete graph of 8 nodes (K8)
- Branching factor: 3 child modules per parent hub
- Depth: 1-5 hierarchical levels
- Tree-like hub connections between modules
- Creates bottlenecks that slow down transport

The hierarchical structure amplifies temporal latency compared to flat
small-world networks, producing α ≈ 2.5-2.6 instead of α ≈ 2.1.

Expected results:
- RTM prediction: α ≈ 2.5 – 2.6
- Paper measured: α = 2.56

Author: RTM Corpus
License: CC BY 4.0
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import deque

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Module parameters
MODULE_SIZE = 8          # Complete graph K8 as base module
BRANCHING_FACTOR = 3     # Each hub spawns 3 child modules

# Hierarchy depths to test (depth d has approximately 8 * (1 + 3 + 9 + ...) nodes)
# Start from depth 2 because depth 1 is trivial (single module, T≈1)
HIERARCHY_DEPTHS = [2, 3, 4, 5, 6]

# Number of network realizations per depth
N_REALIZATIONS = 8

# Number of random walk trials per network
N_WALKS = 30

# Maximum steps per walk
MAX_STEPS = 1_000_000

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# NETWORK GENERATION
# =============================================================================

class HierarchicalNetwork:
    """
    Hierarchical modular network with complete-graph modules connected
    in a tree structure via hub nodes.
    """
    
    def __init__(self, depth: int, module_size: int = 8, branching: int = 3):
        """
        Build a hierarchical network.
        
        Parameters
        ----------
        depth : int
            Number of hierarchy levels (1 = single module)
        module_size : int
            Size of each complete-graph module
        branching : int
            Number of child modules per hub
        """
        self.depth = depth
        self.module_size = module_size
        self.branching = branching
        
        # Adjacency list
        self.adj: Dict[int, Set[int]] = {}
        
        # Track nodes by level for analysis
        self.nodes_by_level: Dict[int, List[int]] = {i: [] for i in range(depth)}
        
        # Track hub nodes
        self.hubs: Set[int] = set()
        
        # Root hub
        self.root_hub = 0
        
        # Build the network
        self._build_network()
        
        # Find farthest nodes from root
        self.farthest_nodes = self._find_farthest_nodes()
        
    def _build_network(self):
        """Recursively build the hierarchical network."""
        self.next_node_id = 0
        self._add_module_recursive(level=0, parent_hub=None)
        
    def _add_module_recursive(self, level: int, parent_hub: int = None) -> int:
        """
        Add a complete-graph module and recursively add children.
        
        Returns the hub node ID of this module.
        """
        if level >= self.depth:
            return None
        
        # Create module nodes
        module_nodes = list(range(self.next_node_id, self.next_node_id + self.module_size))
        self.next_node_id += self.module_size
        
        # First node is the hub
        hub = module_nodes[0]
        self.hubs.add(hub)
        
        # Track nodes by level
        self.nodes_by_level[level].extend(module_nodes)
        
        # Initialize adjacency for all nodes
        for node in module_nodes:
            self.adj[node] = set()
        
        # Create complete graph within module
        for i, node_i in enumerate(module_nodes):
            for node_j in module_nodes[i+1:]:
                self.adj[node_i].add(node_j)
                self.adj[node_j].add(node_i)
        
        # Connect to parent hub (if exists)
        if parent_hub is not None:
            self.adj[hub].add(parent_hub)
            self.adj[parent_hub].add(hub)
        
        # Recursively add child modules
        if level < self.depth - 1:
            for _ in range(self.branching):
                self._add_module_recursive(level + 1, parent_hub=hub)
        
        return hub
    
    def _find_farthest_nodes(self) -> List[int]:
        """Find nodes at maximum graph distance from root hub."""
        # BFS from root
        distances = {self.root_hub: 0}
        queue = deque([self.root_hub])
        
        while queue:
            node = queue.popleft()
            for neighbor in self.adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        max_dist = max(distances.values())
        return [node for node, dist in distances.items() if dist == max_dist]
    
    @property
    def n_nodes(self) -> int:
        return len(self.adj)
    
    @property
    def n_modules(self) -> int:
        return len(self.hubs)


def create_hierarchical_network(depth: int) -> HierarchicalNetwork:
    """Create a hierarchical network with given depth."""
    return HierarchicalNetwork(depth, MODULE_SIZE, BRANCHING_FACTOR)


# =============================================================================
# RANDOM WALK SIMULATION
# =============================================================================

def random_walk_hitting_time(adj: Dict[int, Set[int]], source: int, 
                              targets: Set[int], max_steps: int, 
                              rng: np.random.Generator) -> int:
    """
    Perform random walk from source until hitting any target node.
    
    Parameters
    ----------
    adj : dict
        Adjacency list (sets of neighbors)
    source : int
        Starting node
    targets : set
        Set of target nodes
    max_steps : int
        Maximum steps before giving up
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    int
        Number of steps to hit target, or -1 if max_steps exceeded
    """
    current = source
    steps = 0
    
    # Convert adj sets to lists for faster random choice
    adj_lists = {k: list(v) for k, v in adj.items()}
    
    while steps < max_steps:
        if current in targets:
            return steps
        
        neighbors = adj_lists[current]
        if len(neighbors) == 0:
            return -1
        
        current = rng.choice(neighbors)
        steps += 1
    
    return -1


def measure_network_hitting_time(network: HierarchicalNetwork, n_walks: int,
                                  max_steps: int, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Measure mean hitting time from root to a randomly selected farthest node.
    
    For each walk, we select one specific target from the farthest nodes,
    making the problem harder (must hit THAT specific node, not just any).
    
    Parameters
    ----------
    network : HierarchicalNetwork
        The network to measure
    n_walks : int
        Number of random walk trials
    max_steps : int
        Maximum steps per walk
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    tuple
        (mean_hitting_time, std_hitting_time)
    """
    source = network.root_hub
    farthest = network.farthest_nodes
    
    hitting_times = []
    
    for _ in range(n_walks):
        # Select ONE specific target for this walk
        target = rng.choice(farthest)
        targets = {target}  # Single target
        
        ht = random_walk_hitting_time(network.adj, source, targets, max_steps, rng)
        if ht >= 0:
            hitting_times.append(ht)
    
    if len(hitting_times) == 0:
        return float('nan'), float('nan')
    
    return np.mean(hitting_times), np.std(hitting_times)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation() -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete hierarchical small-world simulation.
    
    Returns
    -------
    pd.DataFrame
        Results dataframe
    dict
        Power-law fit results
    """
    print("=" * 70)
    print("RTM SIMULATION: Hierarchical Small-World Network (Baseline Neural-Like)")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    print(f"  - Module size: {MODULE_SIZE} (complete graph K{MODULE_SIZE})")
    print(f"  - Branching factor: {BRANCHING_FACTOR}")
    print(f"  - Hierarchy depths: {HIERARCHY_DEPTHS}")
    print(f"  - Realizations per depth: {N_REALIZATIONS}")
    print(f"  - Random walks per network: {N_WALKS}")
    print(f"  - Random seed: {RANDOM_SEED}")
    print()
    
    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Storage
    results = []
    
    print("Building networks and running simulations...")
    print(f"{'Depth':>6} | {'N_nodes':>8} | {'L=√N':>8} | {'N_modules':>9} | {'T_mean':>12} | {'T_std':>10}")
    print("-" * 70)
    
    for depth in HIERARCHY_DEPTHS:
        # Create sample network to get size
        sample_net = create_hierarchical_network(depth)
        N = sample_net.n_nodes
        L = np.sqrt(N)
        n_modules = sample_net.n_modules
        
        # Run multiple realizations
        hitting_times_all = []
        
        for r in range(N_REALIZATIONS):
            network = create_hierarchical_network(depth)
            mean_ht, std_ht = measure_network_hitting_time(network, N_WALKS, MAX_STEPS, rng)
            
            if not np.isnan(mean_ht):
                hitting_times_all.append(mean_ht)
        
        if len(hitting_times_all) > 0:
            T_mean = np.mean(hitting_times_all)
            T_std = np.std(hitting_times_all)
            T_sem = T_std / np.sqrt(len(hitting_times_all))
            
            results.append({
                'depth': depth,
                'N': N,
                'L': L,
                'n_modules': n_modules,
                'T_mean': T_mean,
                'T_std': T_std,
                'T_sem': T_sem,
                'n_realizations': len(hitting_times_all)
            })
            
            print(f"{depth:>6} | {N:>8} | {L:>8.2f} | {n_modules:>9} | {T_mean:>12.2f} | {T_std:>10.2f}")
        else:
            print(f"{depth:>6} | {N:>8} | {L:>8.2f} | {n_modules:>9} | {'FAILED':>12} |")
    
    df = pd.DataFrame(results)
    
    # Fit power law: T = T0 * L^α
    print("\n" + "=" * 70)
    print("POWER LAW FIT: T = T0 * L^α (where L = √N)")
    print("=" * 70)
    
    log_L = np.log10(df['L'].values)
    log_T = np.log10(df['T_mean'].values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
    
    # Bootstrap CI
    n_boot = 1000
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(log_L), size=len(log_L), replace=True)
        # Skip if all x values are identical
        if len(set(idx)) < 2:
            continue
        try:
            b_slope, _, _, _, _ = stats.linregress(log_L[idx], log_T[idx])
            boot_slopes.append(b_slope)
        except:
            continue
    
    if len(boot_slopes) > 0:
        ci_lower = np.percentile(boot_slopes, 2.5)
        ci_upper = np.percentile(boot_slopes, 97.5)
    else:
        ci_lower = slope - 2 * std_err
        ci_upper = slope + 2 * std_err
    
    fit_results = {
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
    print(f"\n  RTM prediction: α ≈ 2.5 – 2.6")
    print(f"  Paper reported: α = 2.56")
    
    # Check against expected
    expected_alpha = 2.56
    if abs(fit_results['alpha'] - expected_alpha) < 0.2:
        print(f"\n  ✓ CONFIRMED: α = {fit_results['alpha']:.2f} matches RTM prediction")
    else:
        print(f"\n  Note: α = {fit_results['alpha']:.2f} (RTM expects ~2.56)")
    
    return df, fit_results


def create_plots(df: pd.DataFrame, fit_results: dict) -> None:
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Log-log scaling
    ax1 = axes[0, 0]
    ax1.errorbar(df['L'], df['T_mean'], yerr=df['T_sem'],
                 fmt='o', markersize=12, capsize=5, color='darkgreen',
                 label='Simulation data', linewidth=2)
    
    L_fit = np.linspace(df['L'].min() * 0.9, df['L'].max() * 1.1, 100)
    T_fit = fit_results['T0'] * L_fit**fit_results['alpha']
    ax1.plot(L_fit, T_fit, 'r-', linewidth=2,
             label=f'Power law: α = {fit_results["alpha"]:.3f}')
    
    # Reference lines
    T_ref_2 = fit_results['T0'] * L_fit**2.0
    T_ref_3 = fit_results['T0'] * L_fit**3.0
    ax1.plot(L_fit, T_ref_2, 'b--', linewidth=1, alpha=0.5, label='α = 2.0 (diffusive)')
    ax1.plot(L_fit, T_ref_3, 'purple', linestyle='--', linewidth=1, alpha=0.5, label='α = 3.0')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Effective Length L = √N', fontsize=12)
    ax1.set_ylabel('Mean Hitting Time T', fontsize=12)
    ax1.set_title(f'Hierarchical Small-World: T ∝ L^α\n(R² = {fit_results["R_squared"]:.4f})', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: T vs Depth
    ax2 = axes[0, 1]
    ax2.errorbar(df['depth'], df['T_mean'], yerr=df['T_sem'],
                 fmt='s-', markersize=10, capsize=5, color='darkgreen',
                 linewidth=2, label='Mean hitting time')
    ax2.set_xlabel('Hierarchy Depth', fontsize=12)
    ax2.set_ylabel('Mean Hitting Time T', fontsize=12)
    ax2.set_title('Hitting Time vs Hierarchy Depth', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['depth'].values)
    
    # Plot 3: Network size growth
    ax3 = axes[1, 0]
    ax3.semilogy(df['depth'], df['N'], 'o-', markersize=10, color='blue', linewidth=2, label='Nodes (N)')
    ax3.semilogy(df['depth'], df['n_modules'], 's-', markersize=10, color='orange', linewidth=2, label='Modules')
    ax3.set_xlabel('Hierarchy Depth', fontsize=12)
    ax3.set_ylabel('Count (log scale)', fontsize=12)
    ax3.set_title('Network Size Growth with Depth', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['depth'].values)
    
    # Plot 4: Residuals
    ax4 = axes[1, 1]
    T_predicted = fit_results['T0'] * df['L'].values**fit_results['alpha']
    residuals = (df['T_mean'].values - T_predicted) / T_predicted * 100
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax4.bar(range(len(df)), residuals, color=colors, alpha=0.8)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([f'D={d}' for d in df['depth']])
    ax4.set_xlabel('Hierarchy Depth', fontsize=12)
    ax4.set_ylabel('Residual (%)', fontsize=12)
    ax4.set_title('Power Law Fit Residuals', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'hierarchical_small_world_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'hierarchical_small_world_results.pdf'))
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    
    plt.show()


def save_results(df: pd.DataFrame, fit_results: dict) -> None:
    """Save results to files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_DIR, 'hierarchical_small_world_data.csv'), index=False)
    
    # Save fit results
    fit_df = pd.DataFrame([{
        'parameter': 'alpha',
        'value': fit_results['alpha'],
        'std_err': fit_results['alpha_std_err'],
        'ci_lower_95': fit_results['alpha_ci_lower'],
        'ci_upper_95': fit_results['alpha_ci_upper'],
        'R_squared': fit_results['R_squared'],
        'p_value': fit_results['p_value'],
        'expected_value_low': 2.5,
        'expected_value_high': 2.6,
        'rtm_paper_value': 2.56,
        'status': 'CONFIRMED' if abs(fit_results['alpha'] - 2.56) < 0.2 else 'DEVIATION'
    }])
    fit_df.to_csv(os.path.join(OUTPUT_DIR, 'hierarchical_small_world_fit_results.csv'), index=False)
    
    # Save summary
    summary = f"""RTM Simulation: Hierarchical Small-World Network (Baseline Neural-Like)
======================================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
Module size: {MODULE_SIZE} (complete graph K{MODULE_SIZE})
Branching factor: {BRANCHING_FACTOR}
Hierarchy depths: {HIERARCHY_DEPTHS}
Realizations per depth: {N_REALIZATIONS}
Random walks per network: {N_WALKS}
Random seed: {RANDOM_SEED}

NETWORK STRUCTURE
-----------------
- Each module is a complete graph (all nodes connected within module)
- Modules are arranged in a tree hierarchy
- Hub nodes connect parent and child modules
- Creates "bottlenecks" that slow information flow

RESULTS
-------
Fitted exponent: α = {fit_results['alpha']:.4f} ± {fit_results['alpha_std_err']:.4f}
95% CI: [{fit_results['alpha_ci_lower']:.4f}, {fit_results['alpha_ci_upper']:.4f}]
R² = {fit_results['R_squared']:.6f}

COMPARISON
----------
RTM prediction: α ≈ 2.5 – 2.6
Paper reported: α = 2.56
This simulation: α = {fit_results['alpha']:.2f}

INTERPRETATION
--------------
The hierarchical modular structure creates bottlenecks between modules
that slow down transport compared to flat small-world networks (α ≈ 2.1).

This network mimics cortical organization where:
- Local modules represent cortical columns or areas
- Sparse inter-module connections create hierarchical pathways
- Information must traverse multiple bottlenecks

The higher α reflects the temporal cost of navigating through the
hierarchy, consistent with RTM predictions for cortical-type networks.

STATUS: {'CONFIRMED' if abs(fit_results['alpha'] - 2.56) < 0.2 else 'NEEDS REVIEW'}
"""
    
    with open(os.path.join(OUTPUT_DIR, 'hierarchical_small_world_summary.txt'), 'w') as f:
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
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
