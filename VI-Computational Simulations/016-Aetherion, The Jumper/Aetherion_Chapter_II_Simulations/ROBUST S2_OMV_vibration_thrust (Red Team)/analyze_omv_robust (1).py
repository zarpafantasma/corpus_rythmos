#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_omv_robust"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION: OMV VIBRATION AUDIT (PONDEROMOTIVE)")
    print("=" * 60)
    
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
    # V1 used a linear force with an artificial absolute value: F ~ |grad_alpha|
    # True physical rectification in a vibrating medium comes from the Ponderomotive Force: F ~ (grad_alpha)^2
    # Let's simulate the Non-Linear True Force
    
    t = np.linspace(0, T*10, 2000) # 10 cycles
    x = np.linspace(0, L, 100)
    X, T_mat = np.meshgrid(x, t)
    
    grad_alpha = alpha_1 * k * np.cos(k * X) * np.cos(omega * T_mat)
    
    # V1 Flawed Force
    f_v1 = KAPPA_EFF * np.abs(grad_alpha) * EPSILON_VAC
    F_v1 = np.trapz(f_v1, x=x, axis=1)
    
    # True Non-Linear Ponderomotive Force
    # Under RTM, topological stress energy density is u = kappa * (grad_alpha)^2 * eps_vac
    # Force is the gradient of this energy, or direct ponderomotive pressure.
    # We model the unidirectional Ponderomotive pressure push:
    f_nl = KAPPA_EFF * (grad_alpha**2) * EPSILON_VAC
    F_nl = np.trapz(f_nl, x=x, axis=1)
    
    # Double integrate to get displacement
    # Note: Ponderomotive force has a DC component (steady acceleration) + AC component (vibration)
    a_nl = F_nl / mass
    v_nl = cumulative_trapezoid(a_nl, t, initial=0)
    z_nl = cumulative_trapezoid(v_nl, t, initial=0)
    
    # 2. MONTE CARLO: ACOUSTIC JITTER & THERMAL NOISE
    # Does the vibration survive 5% frequency jitter (piezo instability) and amplitude noise?
    np.random.seed(42)
    n_sims = 100
    z_drifts = []   # DC drift over 10 cycles
    z_vibs = []     # AC peak-to-peak
    
    for _ in range(n_sims):
        f_jitter = freq * np.random.normal(1.0, 0.05) # 5% freq noise
        a_jitter = alpha_1 * np.random.normal(1.0, 0.05) # 5% amp noise
        
        w_j = 2 * np.pi * f_jitter
        grad_a_j = a_jitter * k * np.cos(k * X) * np.cos(w_j * T_mat)
        
        F_j = np.trapz(KAPPA_EFF * (grad_a_j**2) * EPSILON_VAC, x=x, axis=1)
        a_j = F_j / mass
        v_j = cumulative_trapezoid(a_j, t, initial=0)
        z_j = cumulative_trapezoid(v_j, t, initial=0)
        
        # Isolate DC drift vs AC vibration
        # Fit a polynomial to subtract drift
        coefs = np.polyfit(t, z_j, 2) 
        drift = np.polyval(coefs, t)
        ac_vib = z_j - drift
        
        z_drifts.append(z_j[-1]) # Total displacement at t_end
        z_vibs.append(np.max(ac_vib) - np.min(ac_vib))
        
    print(f"[PONDEROMOTIVE OMV DYNAMICS - 10 CYCLES]")
    print(f"Max Ponderomotive Force : {np.max(F_nl):.3e} N")
    print(f"Mean Net DC Drift (1ms) : {np.mean(z_drifts)*1e12:.3f} ± {np.std(z_drifts)*1e12:.3f} pm")
    print(f"Mean AC Vibration       : {np.mean(z_vibs)*1e15:.3f} ± {np.std(z_vibs)*1e15:.3f} fm")

    # 3. VISUALIZATIONS
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: The Ponderomotive Rectification
    ax = axes[0]
    ax.plot(t*1000, F_nl * 1e12, 'b-', lw=2, label='True Ponderomotive Force $F \propto (\nabla\alpha)^2$')
    ax.plot(t*1000, F_v1 * 1e12, 'r--', lw=2, alpha=0.5, label='Flawed V1 Force $F \propto |\nabla\alpha|$')
    ax.axhline(np.mean(F_nl)*1e12, color='k', linestyle=':', lw=2, label=f'Net DC Thrust: {np.mean(F_nl)*1e12:.2f} pN')
    ax.set_title('Ponderomotive Rectification\n(Vibration naturally yields net unidirectional thrust)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Total Force (pN)')
    ax.set_xlim(0, 0.5) # Zoom in to first 5 cycles
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Macroscopic Drift vs Noise
    ax = axes[1]
    ax.plot(t*1000, z_nl * 1e12, 'k-', lw=3, label='Ideal Displacement (Drift + AC)')
    
    # Plot a few noisy trajectories
    for i in range(5):
        f_jitter = freq * np.random.normal(1.0, 0.05)
        w_j = 2 * np.pi * f_jitter
        grad_a_j = alpha_1 * np.random.normal(1.0, 0.05) * k * np.cos(k * X) * np.cos(w_j * T_mat)
        F_j = np.trapz(KAPPA_EFF * (grad_a_j**2) * EPSILON_VAC, x=x, axis=1)
        z_j = cumulative_trapezoid(cumulative_trapezoid(F_j/mass, t, initial=0), t, initial=0)
        ax.plot(t*1000, z_j * 1e12, color='green', alpha=0.3)
        
    ax.set_title('Robust Parabolic Drift\n(5% Piezo-Acoustic Noise Injection)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Displacement z(t) (picometers)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_omv_dynamics.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_omv_dynamics.pdf")
    
    # 4. EXPORT
    df_export = pd.DataFrame({
        't_ms': t*1000,
        'F_nl_pN': F_nl * 1e12,
        'F_v1_pN': F_v1 * 1e12,
        'z_nl_pm': z_nl * 1e12
    })
    df_export.to_csv(f"{OUTPUT_DIR}/robust_omv_timeseries.csv", index=False)
    
    summary = pd.DataFrame({
        'Metric': ['Max_Ponderomotive_pN', 'Mean_DC_Thrust_pN', 'Mean_Drift_1ms_pm', 'Mean_AC_Vibration_fm'],
        'Value': [np.max(F_nl)*1e12, np.mean(F_nl)*1e12, np.mean(z_drifts)*1e12, np.mean(z_vibs)*1e15]
    })
    summary.to_csv(f"{OUTPUT_DIR}/robust_omv_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
