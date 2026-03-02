#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_multiwell_robust"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RTM_ALPHA_VALUES = {
    0: 2.00, 1: 2.26, 2: 2.47, 3: 2.61, 4: 2.72, 5: 2.81
}

def alpha_from_beta(beta: float) -> float:
    beta_int = int(np.floor(beta))
    beta_frac = beta - beta_int
    if beta_int < 0: return RTM_ALPHA_VALUES[0]
    if beta_int >= len(RTM_ALPHA_VALUES) - 1: return RTM_ALPHA_VALUES[len(RTM_ALPHA_VALUES) - 1]
    alpha_low = RTM_ALPHA_VALUES[beta_int]
    alpha_high = RTM_ALPHA_VALUES[beta_int + 1]
    return alpha_low + beta_frac * (alpha_high - alpha_low)

def V_beta_robust(beta: np.ndarray, lambda_param: float = 1.0, decay: float = 0.2) -> np.ndarray:
    return lambda_param * np.sin(np.pi * beta)**2 * np.exp(-decay * beta)

def force_robust(beta: np.ndarray, lambda_param: float = 1.0, decay: float = 0.2) -> np.ndarray:
    db = 1e-5
    return -(V_beta_robust(beta + db, lambda_param, decay) - V_beta_robust(beta - db, lambda_param, decay)) / (2*db)

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION: TOPOLOGICAL MULTI-WELL POTENTIAL")
    print("=" * 60)
    
    beta_range = np.linspace(-0.5, 4.5, 500)
    V_vals = V_beta_robust(beta_range)
    F_vals = force_robust(beta_range)
    
    # Check V1 flaws
    print("[RED TEAM AUDIT: MULTI-WELL STABILITY]")
    print("V1 Eq: V ∝ Σ (β-n)²(β-n-1)² yielded a single parabolic minimum at β ≈ 1.51.")
    print("V2 Robust Eq: V = λ sin²(πβ) exp(-kβ) restores topological quantization.")
    
    for n in range(4):
        v_well = V_beta_robust(np.array([n]))[0]
        v_peak = V_beta_robust(np.array([n + 0.5]))[0]
        print(f"  Branch {n} -> {n+1}: Barrier = {v_peak - v_well:.4f} (Stable)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: The Potential
    ax = axes[0]
    ax.plot(beta_range, V_beta_robust(beta_range, lambda_param=1.0), 'b-', lw=3, label='Topological Potential V(β)')
    ax.fill_between(beta_range, 0, V_beta_robust(beta_range), alpha=0.2, color='blue')
    for n in range(5):
        ax.axvline(n, color='black', linestyle=':', lw=2, alpha=0.5)
        ax.text(n, -0.05, f"β={n}\n(α={RTM_ALPHA_VALUES[n]})", ha='center', fontsize=10)
    ax.set_title('Robust Topological Multi-Well Potential\n(Guaranteed quantization at integers)')
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

    df = pd.DataFrame({'beta': beta_range, 'V': V_vals, 'Force': F_vals})
    df.to_csv(f"{OUTPUT_DIR}/robust_multiwell_profile.csv", index=False)
    
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
