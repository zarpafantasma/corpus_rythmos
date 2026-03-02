#!/usr/bin/env python3
"""
S4: Jump Threshold Calculator
=============================

From "Aetherion, The Jumper" - Chapter III, Sections 5.2, 4.3

Calculates the minimum energy/gradient required to trigger a branch jump.

Key Equations (from paper):
    Barrier height: ΔV_β ≈ λ/16  (Eq. 21)
    
    Jump condition (Eq. 22):
        g_βα × (∇α)² × V_core ≥ ΔV_β
    
    For spherical core of radius R:
        (∇α)_min ≈ √(ΔV_β / (g_βα × (4π/3) × R³))

Reference: Paper Chapter III, Sections 4.3, 5.2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# BARRIER CALCULATIONS
# =============================================================================

def barrier_height(lambda_param: float) -> float:
    """
    Calculate barrier height between adjacent branches.
    
    ΔV_β ≈ λ/16 (from paper Eq. 21)
    
    For the potential V(β) = λ β² (β-1)²,
    the barrier is at β = 0.5 with height V(0.5) = λ/16
    """
    return lambda_param / 16


def min_gradient_for_jump(lambda_param: float, g_beta_alpha: float,
                          R_core: float) -> float:
    """
    Calculate minimum α-gradient to trigger a branch jump.
    
    From Eq. 22:
        g_βα × (∇α)² × V_core ≥ ΔV_β
    
    For spherical core:
        V_core = (4π/3) R³
    
    Solving for (∇α)_min:
        (∇α)_min = √(ΔV_β / (g_βα × V_core))
    
    Parameters:
    -----------
    lambda_param : float
        Barrier height parameter
    g_beta_alpha : float
        β-α coupling strength
    R_core : float
        Aetherion core radius (m)
    
    Returns:
    --------
    grad_alpha_min : float
        Minimum gradient (1/m)
    """
    delta_V = barrier_height(lambda_param)
    V_core = (4 * np.pi / 3) * R_core**3
    
    grad_alpha_min = np.sqrt(delta_V / (g_beta_alpha * V_core))
    return grad_alpha_min


def energy_for_jump(lambda_param: float, g_beta_alpha: float,
                    R_core: float, epsilon_vac: float = 1e-9) -> float:
    """
    Calculate energy required to trigger a branch jump.
    
    E_jump ≈ ΔV_β × V_core (in natural units)
    
    In SI: multiply by appropriate conversion factor.
    """
    delta_V = barrier_height(lambda_param)
    V_core = (4 * np.pi / 3) * R_core**3
    
    # Energy in natural units
    E_natural = delta_V * V_core
    
    # Convert to SI (approximate)
    # ε_vac gives the energy scale
    E_SI = E_natural * epsilon_vac * V_core
    
    return E_SI


def jump_time_estimate(lambda_param: float, g_beta_alpha: float,
                       grad_alpha: float) -> float:
    """
    Estimate time for β to cross the barrier.
    
    From dimensional analysis:
        τ_jump ~ 1 / √(g_βα × (∇α)² - λ/16)
    
    (Only valid if gradient exceeds threshold)
    """
    delta_V = barrier_height(lambda_param)
    drive = g_beta_alpha * grad_alpha**2
    
    if drive <= delta_V:
        return np.inf  # Below threshold
    
    return 1 / np.sqrt(drive - delta_V)


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

def classify_regime(lambda_param: float, g_beta_alpha: float,
                    grad_alpha: float) -> str:
    """
    Classify the transition regime.
    
    From paper Section 5.3:
    - Tunnelling: gradient < threshold (quantum fluctuations only)
    - Critical: gradient ≈ threshold (controlled single jump)
    - Over-driven: gradient >> threshold (multiple jumps possible)
    """
    delta_V = barrier_height(lambda_param)
    drive = g_beta_alpha * grad_alpha**2
    
    ratio = drive / delta_V
    
    if ratio < 0.9:
        return "TUNNELLING (sub-threshold)"
    elif ratio < 1.5:
        return "CRITICAL (controlled jump)"
    else:
        return "OVER-DRIVEN (multiple jumps)"


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir: str):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Minimum gradient vs core radius
    ax1 = axes[0, 0]
    R_range = np.logspace(-4, -1, 50)  # 0.1 mm to 10 cm
    
    for lam, color, label in [(0.5, 'blue', 'λ=0.5'),
                               (1.0, 'green', 'λ=1.0'),
                               (2.0, 'red', 'λ=2.0')]:
        grads = [min_gradient_for_jump(lam, 2.0, R) for R in R_range]
        ax1.loglog(R_range * 100, grads, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Core radius R (cm)', fontsize=12)
    ax1.set_ylabel('Minimum |∇α| (1/m)', fontsize=12)
    ax1.set_title('Jump Threshold vs Core Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Barrier height vs λ
    ax2 = axes[0, 1]
    lambda_range = np.linspace(0.1, 3.0, 50)
    barriers = [barrier_height(lam) for lam in lambda_range]
    
    ax2.plot(lambda_range, barriers, 'purple', linewidth=2)
    ax2.set_xlabel('Barrier parameter λ', fontsize=12)
    ax2.set_ylabel('Barrier height ΔV_β', fontsize=12)
    ax2.set_title('Barrier Height: ΔV_β = λ/16', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Mark practical range
    ax2.axvspan(0.8, 2.0, alpha=0.2, color='green', label='Practical range')
    ax2.legend()
    
    # Plot 3: Jump time vs gradient
    ax3 = axes[1, 0]
    grad_range = np.linspace(0.01, 1.0, 100)
    
    for lam, color in [(0.5, 'blue'), (1.0, 'green'), (2.0, 'red')]:
        times = [jump_time_estimate(lam, 2.0, g) for g in grad_range]
        times = np.array(times)
        times[times > 100] = np.nan  # Cap for visualization
        ax3.semilogy(grad_range, times, color=color, linewidth=2, label=f'λ={lam}')
    
    ax3.set_xlabel('|∇α|', fontsize=12)
    ax3.set_ylabel('Jump time τ', fontsize=12)
    ax3.set_title('Transition Time vs Gradient', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim(0.1, 100)
    
    # Plot 4: Regime diagram
    ax4 = axes[1, 1]
    
    lambda_range = np.linspace(0.2, 2.5, 50)
    grad_range = np.linspace(0.05, 0.8, 50)
    L, G = np.meshgrid(lambda_range, grad_range)
    
    # Compute regime
    g_beta_alpha = 2.0
    regime = np.zeros_like(L)
    for i in range(len(grad_range)):
        for j in range(len(lambda_range)):
            delta_V = barrier_height(L[i, j])
            drive = g_beta_alpha * G[i, j]**2
            ratio = drive / delta_V
            if ratio < 0.9:
                regime[i, j] = 0  # Tunnelling
            elif ratio < 1.5:
                regime[i, j] = 1  # Critical
            else:
                regime[i, j] = 2  # Over-driven
    
    im = ax4.contourf(L, G, regime, levels=[-0.5, 0.5, 1.5, 2.5],
                      colors=['lightblue', 'lightgreen', 'lightyellow'])
    ax4.contour(L, G, regime, levels=[0.5, 1.5], colors='black', linewidths=1)
    
    ax4.set_xlabel('Barrier parameter λ', fontsize=12)
    ax4.set_ylabel('|∇α|', fontsize=12)
    ax4.set_title('Transition Regime Map', fontsize=14)
    
    # Labels
    ax4.text(0.5, 0.2, 'TUNNELLING', fontsize=11, ha='center')
    ax4.text(1.2, 0.35, 'CRITICAL', fontsize=11, ha='center')
    ax4.text(1.8, 0.6, 'OVER-DRIVEN', fontsize=11, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S4_jump_threshold.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S4_jump_threshold.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("S4: Jump Threshold Calculator")
    print("From: Aetherion, The Jumper - Chapter III")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 66)
    print("KEY EQUATIONS")
    print("=" * 66)
    print("""
    Barrier height (Eq. 21):
        ΔV_β = λ/16
    
    Jump condition (Eq. 22):
        g_βα × (∇α)² × V_core ≥ ΔV_β
    
    Minimum gradient:
        (∇α)_min = √(ΔV_β / (g_βα × V_core))
    """)
    
    # Example calculations
    print("=" * 66)
    print("EXAMPLE: LAB-SCALE AETHERION CORE")
    print("=" * 66)
    
    # Parameters
    lambda_param = 1.0
    g_beta_alpha = 2.0
    R_core = 0.01  # 1 cm radius
    
    print(f"\nParameters:")
    print(f"  λ = {lambda_param}")
    print(f"  g_βα = {g_beta_alpha}")
    print(f"  R_core = {R_core*100:.1f} cm")
    
    # Calculations
    delta_V = barrier_height(lambda_param)
    grad_min = min_gradient_for_jump(lambda_param, g_beta_alpha, R_core)
    V_core = (4 * np.pi / 3) * R_core**3
    
    print(f"\nResults:")
    print(f"  Barrier height ΔV_β = {delta_V:.4f}")
    print(f"  Core volume = {V_core*1e6:.3f} cm³")
    print(f"  Minimum |∇α| = {grad_min:.3f} /m")
    
    # Regime classification
    print("\n" + "=" * 66)
    print("REGIME CLASSIFICATION")
    print("=" * 66)
    
    gradients = [0.05, 0.15, 0.25, 0.5]
    records = []
    
    print(f"\nFor λ = {lambda_param}, g_βα = {g_beta_alpha}:")
    print(f"\n{'|∇α|':>10} | {'Drive':>10} | {'Regime':<25}")
    print("-" * 50)
    
    for grad in gradients:
        regime = classify_regime(lambda_param, g_beta_alpha, grad)
        drive = g_beta_alpha * grad**2
        print(f"{grad:>10.2f} | {drive:>10.4f} | {regime}")
        records.append({
            'grad_alpha': grad,
            'drive': drive,
            'regime': regime
        })
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'S4_regime_classification.csv'), index=False)
    
    # Practical design guidance
    print("\n" + "=" * 66)
    print("DESIGN GUIDANCE (from paper Section 5)")
    print("=" * 66)
    print(f"""
    For a controlled single-jump (critical regime):
    
    Target: g_βα × (∇α)² ≈ 1.0 - 1.5 × ΔV_β
    
    With λ = 1.0, ΔV_β = {delta_V:.4f}:
      Drive target: {delta_V:.4f} - {1.5*delta_V:.4f}
      
    For R = 1 cm core with g_βα = 2.0:
      |∇α| target: {grad_min:.3f} - {grad_min * 1.2:.3f} /m
    
    Paper recommendations (Section 6.4):
      λ ≈ 1-2
      g_βα ≈ 2-4
      Δα ≈ 0.5-0.6
    """)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Save threshold data
    R_vals = np.logspace(-4, -1, 20)
    thresh_data = []
    for R in R_vals:
        thresh_data.append({
            'R_m': R,
            'R_cm': R * 100,
            'grad_min_lambda05': min_gradient_for_jump(0.5, 2.0, R),
            'grad_min_lambda10': min_gradient_for_jump(1.0, 2.0, R),
            'grad_min_lambda20': min_gradient_for_jump(2.0, 2.0, R)
        })
    df_thresh = pd.DataFrame(thresh_data)
    df_thresh.to_csv(os.path.join(output_dir, 'S4_threshold_vs_radius.csv'), index=False)
    
    # Summary
    summary = f"""S4: Jump Threshold Calculator
==============================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY EQUATIONS
-------------
Barrier height: ΔV_β = λ/16
Jump condition: g_βα × (∇α)² × V_core ≥ ΔV_β

EXAMPLE (1 cm core)
-------------------
λ = {lambda_param}
g_βα = {g_beta_alpha}
R = {R_core*100:.1f} cm

ΔV_β = {delta_V:.4f}
|∇α|_min = {grad_min:.3f} /m

REGIMES
-------
- TUNNELLING: drive < 0.9 × ΔV_β (sub-threshold)
- CRITICAL: drive ≈ 1.0-1.5 × ΔV_β (controlled jump)
- OVER-DRIVEN: drive > 1.5 × ΔV_β (multiple jumps)

DESIGN GUIDANCE
---------------
For controlled single-jump:
  λ ≈ 1-2
  g_βα ≈ 2-4
  |∇α| ≈ 0.2-0.4 /m (for cm-scale core)
"""
    
    with open(os.path.join(output_dir, 'S4_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
