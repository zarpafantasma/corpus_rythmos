#!/usr/bin/env python3
"""
S1: Multi-Well Potential V(β) for Branch Field
===============================================

From "Aetherion, The Jumper" - Chapter III, Section 3.2

Constructs the multi-well potential whose minima correspond to discrete
RTM branch indices (coherence layers / "local universes").

Key Equation (from paper Eq. 14):
    V(β) = λ × Σ_n [ (β - β_n)² × (β - β_{n+1})² ]
    
Where β_n corresponds to quantized α-values from RTM:
    α ∈ {2.00, 2.26, 2.47, 2.61, 2.72, 2.81, ...}

Reference: Paper Chapter III, Sections 2.1, 3.2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# RTM HIERARCHICAL α-VALUES
# =============================================================================

# Quantized α-values from RTM (Table 1 in paper)
# These correspond to different network topologies/coherence layers
RTM_ALPHA_VALUES = {
    0: 2.00,  # Diffusive baseline
    1: 2.26,  # Flat small-world
    2: 2.47,  # Hierarchical modular
    3: 2.61,  # Holographic decay
    4: 2.72,  # Deep fractal tree
    5: 2.81,  # Ultra-deep hierarchy
}

# Branch indices
BETA_VALUES = list(RTM_ALPHA_VALUES.keys())


# =============================================================================
# MULTI-WELL POTENTIAL
# =============================================================================

def V_beta(beta: np.ndarray, lambda_param: float = 1.0, 
           epsilon: float = 0.1, N_branches: int = 3) -> np.ndarray:
    """
    Multi-well potential V(β) with minima at integer branch indices.
    
    V(β) = λ × Σ_n [ (β - n)² × (β - (n+1))² ] + smoothing
    
    Parameters:
    -----------
    beta : array
        Branch field values
    lambda_param : float
        Barrier height parameter
    epsilon : float
        Smoothing parameter
    N_branches : int
        Number of branches (wells) to include
    
    Returns:
    --------
    V : array
        Potential values
    """
    V = np.zeros_like(beta)
    
    # Sum over branch pairs to create wells
    for n in range(N_branches):
        # Each term creates a well between n and n+1
        term = (beta - n)**2 * (beta - (n + 1))**2
        V += term
    
    # Scale by barrier height and add smoothing
    V = lambda_param * V / (1 + epsilon * beta**2)
    
    return V


def dV_dbeta(beta: np.ndarray, lambda_param: float = 1.0,
             epsilon: float = 0.1, N_branches: int = 3) -> np.ndarray:
    """
    Derivative of multi-well potential dV/dβ.
    
    Used in equations of motion.
    """
    # Numerical derivative for robustness
    db = 1e-6
    return (V_beta(beta + db, lambda_param, epsilon, N_branches) - 
            V_beta(beta - db, lambda_param, epsilon, N_branches)) / (2 * db)


def barrier_height(n: int, lambda_param: float = 1.0,
                   epsilon: float = 0.1, N_branches: int = 3) -> float:
    """
    Calculate barrier height between branch n and n+1.
    
    Barrier is at β = n + 0.5
    """
    beta_well = float(n)
    beta_barrier = n + 0.5
    
    V_well = V_beta(np.array([beta_well]), lambda_param, epsilon, N_branches)[0]
    V_barrier = V_beta(np.array([beta_barrier]), lambda_param, epsilon, N_branches)[0]
    
    return V_barrier - V_well


def alpha_from_beta(beta: float) -> float:
    """
    Map branch index β to physical RTM exponent α.
    
    Uses linear interpolation between quantized values.
    """
    beta_int = int(np.floor(beta))
    beta_frac = beta - beta_int
    
    if beta_int < 0:
        return RTM_ALPHA_VALUES[0]
    if beta_int >= len(RTM_ALPHA_VALUES) - 1:
        return RTM_ALPHA_VALUES[len(RTM_ALPHA_VALUES) - 1]
    
    alpha_low = RTM_ALPHA_VALUES[beta_int]
    alpha_high = RTM_ALPHA_VALUES[beta_int + 1]
    
    return alpha_low + beta_frac * (alpha_high - alpha_low)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir: str):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Multi-well potential V(β)
    ax1 = axes[0, 0]
    beta_range = np.linspace(-0.5, 4.5, 500)
    
    for lam, color, label in [(0.5, 'blue', 'λ=0.5'), 
                               (1.0, 'green', 'λ=1.0'),
                               (2.0, 'red', 'λ=2.0')]:
        V = V_beta(beta_range, lambda_param=lam, N_branches=4)
        ax1.plot(beta_range, V, color=color, linewidth=2, label=label)
    
    # Mark minima
    for n in range(5):
        ax1.axvline(x=n, color='gray', linestyle='--', alpha=0.3)
        ax1.text(n, -0.02, f'β={n}', ha='center', fontsize=10)
    
    ax1.set_xlabel('Branch index β', fontsize=12)
    ax1.set_ylabel('Potential V(β)', fontsize=12)
    ax1.set_title('Multi-Well Potential: Discrete Universe Branches', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 0.5)
    
    # Plot 2: β to α mapping
    ax2 = axes[0, 1]
    beta_vals = np.linspace(0, 5, 100)
    alpha_vals = [alpha_from_beta(b) for b in beta_vals]
    
    ax2.plot(beta_vals, alpha_vals, 'b-', linewidth=2)
    
    # Mark quantized values
    for n, alpha in RTM_ALPHA_VALUES.items():
        ax2.plot(n, alpha, 'ro', markersize=10)
        ax2.annotate(f'α={alpha}', (n, alpha), textcoords='offset points',
                     xytext=(10, 5), fontsize=9)
    
    ax2.set_xlabel('Branch index β', fontsize=12)
    ax2.set_ylabel('RTM exponent α', fontsize=12)
    ax2.set_title('Branch Index to Physical α Mapping', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Barrier heights
    ax3 = axes[1, 0]
    lambdas = np.linspace(0.1, 3.0, 30)
    
    for n, color in [(0, 'blue'), (1, 'green'), (2, 'red')]:
        barriers = [barrier_height(n, lam, N_branches=4) for lam in lambdas]
        ax3.plot(lambdas, barriers, color=color, linewidth=2, 
                 label=f'β: {n} → {n+1}')
    
    ax3.set_xlabel('Barrier parameter λ', fontsize=12)
    ax3.set_ylabel('Barrier height ΔV', fontsize=12)
    ax3.set_title('Barrier Height vs λ', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Force field -dV/dβ
    ax4 = axes[1, 1]
    beta_range = np.linspace(-0.5, 3.5, 500)
    
    V = V_beta(beta_range, lambda_param=1.0, N_branches=3)
    dV = dV_dbeta(beta_range, lambda_param=1.0, N_branches=3)
    
    ax4.plot(beta_range, -dV, 'purple', linewidth=2, label='-dV/dβ (force)')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark stable and unstable points
    for n in range(4):
        ax4.axvline(x=n, color='green', linestyle=':', alpha=0.5)
        if n < 3:
            ax4.axvline(x=n+0.5, color='red', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Branch index β', fontsize=12)
    ax4.set_ylabel('Force -dV/dβ', fontsize=12)
    ax4.set_title('Restoring Force (stable points at integers)', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_multiwell_potential.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_multiwell_potential.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("S1: Multi-Well Potential V(β) for Branch Field")
    print("From: Aetherion, The Jumper - Chapter III")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 66)
    print("THEORETICAL BACKGROUND")
    print("=" * 66)
    print("""
    RTM predicts quantized α-values from network topology:
    
    β = 0: α = 2.00 (diffusive baseline)
    β = 1: α = 2.26 (flat small-world)
    β = 2: α = 2.47 (hierarchical modular)
    β = 3: α = 2.61 (holographic decay)
    β = 4: α = 2.72 (deep fractal tree)
    β = 5: α = 2.81 (ultra-deep hierarchy)
    
    Each β corresponds to a "coherence layer" or "local universe"
    with its own temporal cadence.
    """)
    
    print("=" * 66)
    print("MULTI-WELL POTENTIAL")
    print("=" * 66)
    print("""
    Key Equation (Eq. 14):
    
        V(β) = λ × Σ_n [ (β - n)² × (β - (n+1))² ]
    
    Properties:
    - Minima at integer β (branch indices)
    - Barriers between adjacent branches
    - λ controls barrier height
    """)
    
    # Calculate barrier heights
    print("\n" + "=" * 66)
    print("BARRIER HEIGHTS (λ = 1.0)")
    print("=" * 66)
    
    records = []
    for n in range(4):
        h = barrier_height(n, lambda_param=1.0, N_branches=4)
        alpha_from = RTM_ALPHA_VALUES[n]
        alpha_to = RTM_ALPHA_VALUES[n + 1]
        print(f"  β: {n} → {n+1} (α: {alpha_from} → {alpha_to}): ΔV = {h:.4f}")
        records.append({
            'branch_from': n,
            'branch_to': n + 1,
            'alpha_from': alpha_from,
            'alpha_to': alpha_to,
            'barrier_height': h
        })
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'S1_barrier_heights.csv'), index=False)
    
    # Physical interpretation
    print("\n" + "=" * 66)
    print("PHYSICAL INTERPRETATION (from paper)")
    print("=" * 66)
    print("""
    Paper (Section 2.2) states:
    
    "Regions locked into a common β share the same temporal cadence
     and thus form a self-consistent 'mini-universe.'"
    
    "Adjacent layers are causally compatible (signals can cross the
     boundary) but perceive one another as running faster/slower."
    
    The multi-well potential:
    - Anchors discrete branch minima
    - Creates energy barriers between branches
    - Enables controlled transitions via α-gradients
    """)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Save potential data
    beta_range = np.linspace(-0.5, 5.5, 200)
    V_vals = V_beta(beta_range, lambda_param=1.0, N_branches=5)
    alpha_vals = [alpha_from_beta(b) for b in beta_range]
    
    df_potential = pd.DataFrame({
        'beta': beta_range,
        'V_beta': V_vals,
        'alpha': alpha_vals
    })
    df_potential.to_csv(os.path.join(output_dir, 'S1_potential_profile.csv'), index=False)
    
    # Summary
    summary = f"""S1: Multi-Well Potential V(β) for Branch Field
===============================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY EQUATION
------------
V(β) = λ × Σ_n [ (β - n)² × (β - (n+1))² ]

RTM QUANTIZED α-VALUES
----------------------
β = 0: α = 2.00 (diffusive)
β = 1: α = 2.26 (small-world)
β = 2: α = 2.47 (hierarchical)
β = 3: α = 2.61 (holographic)
β = 4: α = 2.72 (fractal)
β = 5: α = 2.81 (ultra-deep)

BARRIER HEIGHTS (λ = 1.0)
-------------------------
β: 0 → 1: ΔV = {barrier_height(0, 1.0, N_branches=4):.4f}
β: 1 → 2: ΔV = {barrier_height(1, 1.0, N_branches=4):.4f}
β: 2 → 3: ΔV = {barrier_height(2, 1.0, N_branches=4):.4f}

PHYSICAL INTERPRETATION
-----------------------
- Each β-minimum = distinct coherence layer
- Barrier crossing = multiverse transition
- Energy supplied by Aetherion α-pulse

FILES GENERATED
---------------
- S1_multiwell_potential.png/pdf
- S1_barrier_heights.csv
- S1_potential_profile.csv
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
