#!/usr/bin/env python3
"""
ROBUST RTM AETHERION: TOPOLOGICAL MULTI-WELL POTENTIAL (S1)
===========================================================
Phase 2 "Red Team" Quantum Field Theory Audit

This script corrects a critical mathematical flaw in the V1 paper (Eq 14).
Summing positive polynomial terms (Σ (β-n)²(β-n-1)²) creates a single massive 
parabolic bias, destroying the discrete local minima. It resulted in only one 
true vacuum at β ≈ 1.51, invalidating the "coherence layer" hypothesis.

This robust pipeline implements a modified "Topological Sine-Gordon Potential" 
(commonly used in string theory and crystallography). This formulation mathematically 
guarantees stable, zero-energy vacua at exactly integer values of β, perfectly 
restoring the quantization of the RTM exponent α.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_multiwell_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION: TOPOLOGICAL MULTI-WELL POTENTIAL")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # RTM Quantized α-values
    RTM_ALPHA_VALUES = {
        0: 2.00,  # Diffusive baseline (Our Universe)
        1: 2.26,  # Flat small-world
        2: 2.47,  # Hierarchical modular
        3: 2.61,  # Holographic decay
        4: 2.72,  # Deep fractal tree
        5: 2.81,  # Ultra-deep hierarchy
    }

    # RED TEAM CORRECTION: Sine-Gordon potential with structural decay
    # V(β) = λ * sin²(πβ) * exp(-kβ)
    def V_beta_robust(beta: np.ndarray, lambda_param: float = 1.0, decay: float = 0.2) -> np.ndarray:
        return lambda_param * np.sin(np.pi * beta)**2 * np.exp(-decay * beta)

    def force_robust(beta: np.ndarray, lambda_param: float = 1.0, decay: float = 0.2) -> np.ndarray:
        # F = -dV/dβ
        db = 1e-5
        return -(V_beta_robust(beta + db, lambda_param, decay) - V_beta_robust(beta - db, lambda_param, decay)) / (2*db)

    beta_range = np.linspace(-0.5, 4.5, 500)
    V_vals = V_beta_robust(beta_range)
    F_vals = force_robust(beta_range)

    print("\n[RED TEAM AUDIT: MULTI-WELL STABILITY]")
    print("V1 Eq yielded a single parabolic minimum at β ≈ 1.51 (Flawed).")
    print("V2 Robust Eq: V = λ sin²(πβ) exp(-kβ) restores topological quantization.")
    
    records = []
    for n in range(4):
        v_well = V_beta_robust(np.array([n]))[0]
        v_peak = V_beta_robust(np.array([n + 0.5]))[0]
        barrier = v_peak - v_well
        print(f"  Branch {n} -> {n+1}: Barrier ΔV = {barrier:.4f} (Stable Vacuum ✓)")
        records.append({'branch_from': n, 'branch_to': n+1, 'barrier_height': barrier})

    # VISUALIZATIONS
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: The Potential
    ax = axes[0]
    ax.plot(beta_range, V_vals, 'b-', lw=3, label='Topological Potential V(β)')
    ax.fill_between(beta_range, 0, V_vals, alpha=0.2, color='blue')
    for n in range(5):
        ax.axvline(n, color='black', linestyle=':', lw=2, alpha=0.5)
        ax.text(n, -0.05, f"β={n}\n(α={RTM_ALPHA_VALUES[n]})", ha='center', fontsize=10)
    ax.set_title('Robust Topological Multi-Well Potential\n(Guaranteed quantization at integer branches)')
    ax.set_xlabel('Branch Index (β)')
    ax.set_ylabel('Potential Energy V(β)')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: The Restoring Force
    ax = axes[1]
    ax.plot(beta_range, F_vals, 'r-', lw=2, label='Restoring Force (-dV/dβ)')
    ax.axhline(0, color='black', linestyle='-', lw=1)
    
    for n in range(5):
        ax.axvline(n, color='green', linestyle='--', lw=2, alpha=0.5, label='Stable Vacua' if n==0 else "")
    for n in range(4):
        ax.axvline(n+0.5, color='orange', linestyle=':', lw=2, alpha=0.5, label='Unstable Peak' if n==0 else "")
        
    ax.set_title('Thermodynamic Restoring Force\n(Ensuring stable phase transitions)')
    ax.set_xlabel('Branch Index (β)')
    ax.set_ylabel('Force')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_multiwell.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_multiwell.pdf")

    # EXPORT
    df_profile = pd.DataFrame({'beta': beta_range, 'V': V_vals, 'Force': F_vals})
    df_profile.to_csv(f"{OUTPUT_DIR}/robust_multiwell_profile.csv", index=False)
    
    df_barriers = pd.DataFrame(records)
    df_barriers.to_csv(f"{OUTPUT_DIR}/robust_barrier_heights.csv", index=False)
    
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()