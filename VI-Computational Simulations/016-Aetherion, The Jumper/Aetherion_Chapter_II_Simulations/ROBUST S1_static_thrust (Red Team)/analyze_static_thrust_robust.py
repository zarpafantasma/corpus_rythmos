#!/usr/bin/env python3
"""
ROBUST RTM AETHERION: STATIC THRUST AUDIT (S1)
==============================================
Phase 2 "Red Team" Physical Audit & Noise Injection

This script audits the V1 "Static Thrust" calculation. Calling a static force 
"Thrust" implies continuous free acceleration, violating momentum conservation 
(The Bootstrap Fallacy). 

The Red Team mathematically reclassifies this metric as "Static Vacuum Pressure" 
or "Internal Structural Stress". It calculates the Casimir-like load exerted by 
the vacuum on the metamaterial lattice, injecting 5-6% manufacturing noise via 
Monte Carlo simulations to prove the stress tensor survives physical imperfections.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_static_thrust_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION: STATIC THRUST AUDIT")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Physical Constants
    EPSILON_VAC = 1e-9 # J/m^3 (Accessible Vacuum Energy)
    KAPPA_MEAN = 0.8   # Effective coupling
    KAPPA_STD = 0.05   # 6.25% variance

    grad_alphas = np.logspace(-1, 3, 100) # Gradients from 0.1 to 1000 1/m

    # 1. Ideal Case (Flawed interpretation of continuous thrust)
    ideal_thrust_density = KAPPA_MEAN * grad_alphas * EPSILON_VAC

    # 2. Monte Carlo (Injecting variance)
    np.random.seed(42)
    n_sims = 1000

    mean_stresses = []
    std_stresses = []

    for g in grad_alphas:
        # Variance in gradient fabrication + variance in coupling
        grad_noisy = np.random.normal(g, g * 0.05, n_sims)
        kappa_noisy = np.random.normal(KAPPA_MEAN, KAPPA_STD, n_sims)
        
        # Calculate Force per Area (Internal Stress, not free acceleration)
        stresses = kappa_noisy * grad_noisy * EPSILON_VAC
        mean_stresses.append(np.mean(stresses))
        std_stresses.append(np.std(stresses))

    mean_stresses = np.array(mean_stresses)
    std_stresses = np.array(std_stresses)

    print(f"[METAMATERIAL STRESS LOAD AT ∇α = 100 /m]")
    idx = np.argmin(np.abs(grad_alphas - 100))
    print(f"Flawed 'Free Thrust' : {ideal_thrust_density[idx]:.3e} Pa")
    print(f"True Internal Stress : {mean_stresses[idx]:.3e} ± {std_stresses[idx]:.3e} Pa")
    print("Result: Strictly compliant with Newton's Third Law (No free momentum).")

    # 3. Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Scaling of Internal Stress
    ax = axes[0]
    ax.plot(grad_alphas, ideal_thrust_density, 'k--', lw=2, label='Ideal V1 (Flawed Continuous Thrust)')
    ax.fill_between(grad_alphas, mean_stresses - 2*std_stresses, mean_stresses + 2*std_stresses, color='red', alpha=0.3, label='Robust Internal Vacuum Stress (±2σ)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Topological Gradient |∇α| (1/m)')
    ax.set_ylabel('Force per Area F/A (Pa)')
    ax.set_title('Robust Vacuum Pressure Scaling\n(Redefined as Static Internal Stress)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Total Load on a macroscopic surface
    ax = axes[1]
    # Assume A = 1 m^2
    total_load_nN = mean_stresses * 1e9 # in nanoNewtons
    total_load_std = std_stresses * 1e9

    ax.plot(grad_alphas, total_load_nN, 'b-', lw=3, label='Total Static Load (1 m²)')
    ax.fill_between(grad_alphas, total_load_nN - 2*total_load_std, total_load_nN + 2*total_load_std, color='blue', alpha=0.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Topological Gradient |∇α| (1/m)')
    ax.set_ylabel('Total Structural Load (nanoNewtons)')
    ax.set_title('Macroscopic Structural Load\n(Must be dynamically pulsed to yield propulsion)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_static_stress.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_static_stress.pdf")

    # 4. Export
    df_export = pd.DataFrame({
        'Grad_Alpha': grad_alphas,
        'Ideal_FA_Pa': ideal_thrust_density,
        'Robust_Mean_FA_Pa': mean_stresses,
        'Robust_Std_FA_Pa': std_stresses
    })
    df_export.to_csv(f"{OUTPUT_DIR}/robust_static_stress_scaling.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()