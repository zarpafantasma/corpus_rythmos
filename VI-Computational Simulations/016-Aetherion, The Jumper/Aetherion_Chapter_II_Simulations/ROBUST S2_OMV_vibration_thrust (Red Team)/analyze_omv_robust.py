#!/usr/bin/env python3
"""
ROBUST RTM AETHERION: OMV VIBRATION AUDIT (PONDEROMOTIVE)
=========================================================
Phase 2 "Red Team" Non-Linear Dynamics Pipeline

This script corrects the linear absolute-value artifact from the V1 simulation.
Because topological vacuum stress scales quadratically with the gradient 
(F ∝ ∇α²), vibrating the metamaterial mathematically induces a Ponderomotive 
Force. The squared oscillation cos²(ωt) acts as a natural topological rectifier,
generating both an AC vibration and a unidirectional DC thrust.

A Monte Carlo simulation (N=100) injects a 5% acoustic/thermal jitter into the 
piezoelectric frequency and amplitude to prove the macroscopic drift survives 
real-world laboratory conditions.
"""
import numpy as np
import matplotlib.subplots as plt_subs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_omv_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION: OMV VIBRATION AUDIT (PONDEROMOTIVE)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Physical Constants
    EPSILON_VAC = 1e-9  # J/m³ 
    KAPPA_EFF = 0.8
    L = 0.1             # 10 cm slab
    mass = 0.01         # 10 g test mass
    freq = 10000        # 10 kHz
    alpha_1 = 0.1       # modulation amplitude
    
    omega = 2 * np.pi * freq
    k = np.pi / L       # Fundamental mode
    T = 1.0 / freq
    
    # 1. THE PONDEROMOTIVE CORRECTION
    t = np.linspace(0, T*10, 2000) # Simulate 10 cycles
    x = np.linspace(0, L, 100)
    X, T_mat = np.meshgrid(x, t)
    
    grad_alpha = alpha_1 * k * np.cos(k * X) * np.cos(omega * T_mat)
    
    # V1 Flawed Force (Artifactual Absolute Value)
    f_v1 = KAPPA_EFF * np.abs(grad_alpha) * EPSILON_VAC
    F_v1 = np.trapz(f_v1, x=x, axis=1)
    
    # True Non-Linear Ponderomotive Force (Quadratic Topological Stress)
    f_nl = KAPPA_EFF * (grad_alpha**2) * EPSILON_VAC
    F_nl = np.trapz(f_nl, x=x, axis=1)
    
    # Double integrate to get macroscopic displacement
    a_nl = F_nl / mass
    v_nl = cumulative_trapezoid(a_nl, t, initial=0)
    z_nl = cumulative_trapezoid(v_nl, t, initial=0)
    
    # 2. MONTE CARLO: ACOUSTIC JITTER & THERMAL NOISE
    np.random.seed(42)
    n_sims = 100
    z_drifts = []   # Total macroscopic drift over 10 cycles
    
    for _ in range(n_sims):
        # Injecting 5% frequency jitter (piezo instability) and amplitude noise
        f_jitter = freq * np.random.normal(1.0, 0.05) 
        a_jitter = alpha_1 * np.random.normal(1.0, 0.05) 
        
        w_j = 2 * np.pi * f_jitter
        grad_a_j = a_jitter * k * np.cos(k * X) * np.cos(w_j * T_mat)
        
        F_j = np.trapz(KAPPA_EFF * (grad_a_j**2) * EPSILON_VAC, x=x, axis=1)
        a_j = F_j / mass
        v_j = cumulative_trapezoid(a_j, t, initial=0)
        z_j = cumulative_trapezoid(v_j, t, initial=0)
        
        z_drifts.append(z_j[-1]) # Total displacement at t_end
        
    print(f"[PONDEROMOTIVE OMV DYNAMICS - 10 CYCLES]")
    print(f"Max Ponderomotive Force : {np.max(F_nl):.3e} N")
    print(f"Net DC Thrust (Steady)  : {np.mean(F_nl)*1e12:.2f} pN")
    print(f"Robust DC Drift (1ms)   : {np.mean(z_drifts)*1e12:.3f} ± {np.std(z_drifts)*1e12:.3f} pm")

    # 3. VISUALIZATIONS
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: The Ponderomotive Rectification
    ax = axes[0]
    ax.plot(t*1000, F_nl * 1e12, 'b-', lw=2, label='True Ponderomotive Force $F \\propto (\\nabla\\alpha)^2$')
    ax.plot(t*1000, F_v1 * 1e12, 'r--', lw=2, alpha=0.5, label='Flawed V1 Force $F \\propto |\\nabla\\alpha|$')
    ax.axhline(np.mean(F_nl)*1e12, color='k', linestyle=':', lw=2, label=f'Net DC Thrust: {np.mean(F_nl)*1e12:.2f} pN')
    ax.set_title('Ponderomotive Topological Rectification\n(Vibration naturally yields net unidirectional thrust)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Total Force (pN)')
    ax.set_xlim(0, 0.5) # Zoom in to first 5 cycles
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Macroscopic Drift vs Noise
    ax = axes[1]
    ax.plot(t*1000, z_nl * 1e12, 'k-', lw=3, label='Ideal Displacement (Drift + AC)')
    
    for i in range(5):
        f_jitter = freq * np.random.normal(1.0, 0.05)
        w_j = 2 * np.pi * f_jitter
        grad_a_j = alpha_1 * np.random.normal(1.0, 0.05) * k * np.cos(k * X) * np.cos(w_j * T_mat)
        F_j = np.trapz(KAPPA_EFF * (grad_a_j**2) * EPSILON_VAC, x=x, axis=1)
        z_j = cumulative_trapezoid(cumulative_trapezoid(F_j/mass, t, initial=0), t, initial=0)
        ax.plot(t*1000, z_j * 1e12, color='green', alpha=0.4, lw=1)
        
    ax.set_title('Robust Parabolic Drift\n(Surviving 5% Piezo-Acoustic Noise)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Displacement z(t) (picometers)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_omv_dynamics.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_omv_dynamics.pdf")
    
    # 4. EXPORT
    df_export = pd.DataFrame({'t_ms': t*1000, 'F_nl_pN': F_nl * 1e12, 'z_nl_pm': z_nl * 1e12})
    df_export.to_csv(f"{OUTPUT_DIR}/robust_omv_timeseries.csv", index=False)
    
    summary = pd.DataFrame({
        'Metric': ['Max_Ponderomotive_pN', 'Mean_DC_Thrust_pN', 'Mean_Drift_1ms_pm'],
        'Value': [np.max(F_nl)*1e12, np.mean(F_nl)*1e12, np.mean(z_drifts)*1e12]
    })
    summary.to_csv(f"{OUTPUT_DIR}/robust_omv_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()