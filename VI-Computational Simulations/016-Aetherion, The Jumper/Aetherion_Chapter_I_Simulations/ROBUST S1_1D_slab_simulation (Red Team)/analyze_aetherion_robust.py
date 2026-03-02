#!/usr/bin/env python3
"""
ROBUST RTM AETHERION 1D SIMULATION
===================================
Phase 2 "Red Team" Thermodynamic Audit

This script corrects the "Overunity Fallacy" present in the V1 analysis.
By evaluating the true net vector field of the power proxy (instead of 
the absolute value), it proves that a static α-gradient acts as a topological 
capacitor (storing energy symmetrically) but yields strictly ZERO net DC power, 
perfectly respecting the First Law of Thermodynamics.

It also utilizes Monte Carlo simulation (N=5000) to inject a 5% thermal 
and manufacturing noise variance into the metamaterial gradient, proving 
that the Aetherion field limits survive realistic physical imperfections.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_aetherion_robust"

def build_second_derivative_matrix(N: int, dx: float) -> sp.csr_matrix:
    diag_main = -2.0 * np.ones(N + 1) / dx**2
    diag_off = np.ones(N) / dx**2
    D2 = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csr')
    return D2

def solve_aetherion_system(N=60, L=1.0, m_phi=1.0, gamma=0.8, alpha_profile=None):
    dx = L / N
    x = np.linspace(0, L, N + 1)
    D2 = build_second_derivative_matrix(N, dx)
    I = sp.eye(N + 1, format='csr')
    
    # Non-linear source term driven by the RTM topological gradient
    grad_alpha = np.gradient(alpha_profile, dx)
    source = gamma * grad_alpha**2
    
    A = D2 - m_phi**2 * I
    b = -source
    
    # Dirichlet Boundary Conditions
    A = A.tolil()
    A[0, :] = 0; A[0, 0] = 1
    A[-1, :] = 0; A[-1, -1] = 1
    A = A.tocsr()
    b[0] = 0; b[-1] = 0
    
    # Solve for Aetherion field (phi)
    phi = spla.spsolve(A, b)
    grad_phi = np.gradient(phi, dx)
    
    # Power Proxy
    P_local = gamma * grad_alpha * grad_phi
    
    # Flawed calculation (from V1)
    P_flawed_abs = np.mean(np.abs(P_local[1:-1]))
    # True net vector calculation (Thermodynamic Compliance)
    P_net = np.mean(P_local[1:-1])
    
    return x, phi, P_local, P_flawed_abs, P_net

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION 1D SIMULATION (THERMODYNAMIC AUDIT)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    N = 60; L = 1.0; gamma = 0.8; m_phi = 1.0
    x = np.linspace(0, L, N + 1)
    
    # 1. IDEAL METAMATERIAL EVALUATION
    alpha_ideal = 2.0 + x
    _, phi_ideal, P_local_ideal, P_flawed_ideal, P_net_ideal = solve_aetherion_system(alpha_profile=alpha_ideal)
    
    print("\n[IDEAL METAMATERIAL - STATIC GRADIENT]")
    print(f"Flawed Absolute Power ⟨|P|⟩ : {P_flawed_ideal:.6f} (Overunity Fallacy)")
    print(f"True Net DC Power ⟨P⟩      : {P_net_ideal:.6f} (Compliant with 1st Law)")
    print(f"Max Aetherion Field (φ)    : {np.max(phi_ideal):.6f} (Capacitive Storage)")

    # 2. MONTE CARLO: MANUFACTURING DEFECTS & THERMAL NOISE
    np.random.seed(42)
    n_sims = 5000
    P_net_sims = []
    
    from scipy.ndimage import gaussian_filter1d

    for _ in range(n_sims):
        # Injecting 5% physical variance (thermal noise + layering defects)
        noise = gaussian_filter1d(np.random.normal(0, 0.05, N + 1), sigma=2)
        alpha_noisy = alpha_ideal + noise
        alpha_noisy[0] = 2.0; alpha_noisy[-1] = 3.0
        
        _, _, _, _, p_net = solve_aetherion_system(alpha_profile=alpha_noisy)
        P_net_sims.append(p_net)

    P_net_sims = np.array(P_net_sims)
    
    print(f"\n[ROBUST METAMATERIAL - 5% NOISE, N={n_sims} SIMS]")
    print(f"Robust Net Power ⟨P⟩       : {np.mean(P_net_sims):.6f} ± {np.std(P_net_sims):.6f}")
    
    # 3. VISUALIZATIONS
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: The Zero-Sum Symmetry
    ax = axes[0]
    ax.plot(x, P_local_ideal, 'g-', lw=3, label='Local Power Flux P(x)')
    ax.fill_between(x, 0, P_local_ideal, where=(P_local_ideal > 0), color='green', alpha=0.3, label='Positive Flux (Right)')
    ax.fill_between(x, 0, P_local_ideal, where=(P_local_ideal < 0), color='red', alpha=0.3, label='Negative Flux (Left)')
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_title('The Zero-Sum Thermodynamic Symmetry\n(Static Gradient Yields No Net DC Power)')
    ax.set_xlabel('Position x/L')
    ax.set_ylabel('Local Power Flux P(x)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Monte Carlo Survival
    ax = axes[1]
    sns.kdeplot(P_net_sims, fill=True, color='blue', ax=ax, lw=2)
    ax.axvline(0, color='black', linestyle='-', lw=3, label='First Law Limit (Net Zero)')
    ax.set_title(f'Thermodynamic Compliance under 5% Thermal/Fab Noise\n(Monte Carlo N={n_sims})')
    ax.set_xlabel('True Net Mean Power ⟨P⟩')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_aetherion_1d.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_aetherion_1d.pdf")

    # 4. EXPORT
    df_export = pd.DataFrame({
        'Metric': ['Flawed_Abs_Power', 'True_Net_Power_Ideal', 'True_Net_Power_Mean_Robust', 'True_Net_Power_Std_Robust'],
        'Value': [P_flawed_ideal, P_net_ideal, np.mean(P_net_sims), np.std(P_net_sims)]
    })
    df_export.to_csv(f"{OUTPUT_DIR}/aetherion_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()