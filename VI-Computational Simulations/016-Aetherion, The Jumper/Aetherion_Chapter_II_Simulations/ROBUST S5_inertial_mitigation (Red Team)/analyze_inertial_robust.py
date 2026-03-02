#!/usr/bin/env python3
"""
ROBUST RTM AETHERION: INERTIAL MITIGATION AUDIT (S5)
====================================================
Phase 2 "Red Team" Dynamic Flicker & Jerk Pipeline

This script corrects the "Perfect Shield Fallacy" from V1. Assuming a perfectly 
stable α=50 field during a 100g maneuver ignores generator limits and thermal noise.
This pipeline injects realistic "Topological Flicker" (5% and 10% stochastic noise) 
into the α-field to evaluate not just the Absolute G-Force, but the potentially 
lethal 'Jerk' (rate of change of acceleration) experienced by the crew.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_inertial_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM AETHERION: INERTIAL MITIGATION AUDIT")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Physical Constants
    G_ACCEL = 9.81              # m/s²
    A_EXT_G = 100.0             # 100g external maneuver
    A_EXT = A_EXT_G * G_ACCEL   # 981 m/s²
    TARGET_ALPHA = 50.0         # Design target for cabin
    
    # Time domain for a 2-second extreme evasive maneuver
    dt = 1e-3
    t = np.arange(0, 2.0, dt)
    n_steps = len(t)

    # 1. IDEAL SCENARIO (V1 Baseline)
    alpha_ideal = np.full(n_steps, TARGET_ALPHA)
    a_felt_ideal = A_EXT / alpha_ideal
    g_felt_ideal = a_felt_ideal / G_ACCEL
    jerk_ideal = np.gradient(a_felt_ideal, dt) # Should be 0

    # 2. MONTE CARLO: TOPOLOGICAL FLICKER (Field Instability)
    np.random.seed(42)
    n_sims = 100
    
    max_g_5pct = []
    max_jerk_5pct = []
    max_g_10pct = []
    max_jerk_10pct = []

    for _ in range(n_sims):
        # 5% Field Flicker (Highly smoothed to represent physical capacitor limits)
        noise_5 = gaussian_filter1d(np.random.normal(0, TARGET_ALPHA * 0.05, n_steps), sigma=20)
        alpha_5 = np.clip(TARGET_ALPHA + noise_5, 1.0, None) # alpha must be >= 1
        
        a_felt_5 = A_EXT / alpha_5
        jerk_5 = np.gradient(a_felt_5, dt)
        
        max_g_5pct.append(np.max(a_felt_5) / G_ACCEL)
        max_jerk_5pct.append(np.max(np.abs(jerk_5)))

        # 10% Field Flicker
        noise_10 = gaussian_filter1d(np.random.normal(0, TARGET_ALPHA * 0.10, n_steps), sigma=20)
        alpha_10 = np.clip(TARGET_ALPHA + noise_10, 1.0, None)
        
        a_felt_10 = A_EXT / alpha_10
        jerk_10 = np.gradient(a_felt_10, dt)
        
        max_g_10pct.append(np.max(a_felt_10) / G_ACCEL)
        max_jerk_10pct.append(np.max(np.abs(jerk_10)))

    # Evaluate a sample trajectory for plotting (5% noise)
    sample_noise = gaussian_filter1d(np.random.normal(0, TARGET_ALPHA * 0.05, n_steps), sigma=20)
    sample_alpha = TARGET_ALPHA + sample_noise
    sample_g_felt = (A_EXT / sample_alpha) / G_ACCEL
    sample_jerk = np.gradient(A_EXT / sample_alpha, dt)

    print(f"\n[SURVIVABILITY METRICS UNDER 100G MANEUVER]")
    print(f"Ideal Design Target : {A_EXT_G / TARGET_ALPHA:.2f} G felt | 0.0 Jerk")
    print(f"5% Field Flicker    : Max {np.mean(max_g_5pct):.2f} G | Max Jerk: {np.mean(max_jerk_5pct):.1f} m/s³")
    print(f"10% Field Flicker   : Max {np.mean(max_g_10pct):.2f} G | Max Jerk: {np.mean(max_jerk_10pct):.1f} m/s³")
    print("Conclusion: G-Force is highly survivable. However, secondary mechanical dampeners are required to absorb the jerk from topological flicker.")

    # 3. VISUALIZATIONS
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Field Flicker vs Felt G-Force
    ax = axes[0]
    ax.plot(t, np.full_like(t, A_EXT_G / TARGET_ALPHA), 'k--', lw=2, label='Ideal 2.0G Baseline')
    ax.plot(t, sample_g_felt, 'r-', lw=2, alpha=0.8, label='Felt Gs (5% α-Flicker)')
    ax.axhline(9.0, color='red', linestyle=':', lw=2, label='Human Blackout Limit (9G)')
    ax.set_title('Inertial Shielding Stability\n(100G External Maneuver)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Perceived Acceleration (G)')
    ax.set_ylim(0, 5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Lethal Jerk Analysis
    ax = axes[1]
    sns.kdeplot(max_jerk_5pct, fill=True, color='orange', ax=ax, label='5% Field Noise')
    sns.kdeplot(max_jerk_10pct, fill=True, color='red', ax=ax, label='10% Field Noise')
    # Typical automotive comfort jerk is < 2 m/s^3. Crash limit is ~500 m/s^3.
    ax.set_title('Topological Jerk Distribution\n(Rate of change of perceived acceleration)')
    ax.set_xlabel('Max Jerk (m/s³)')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_inertial_dampening.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_inertial_dampening.pdf")

    # 4. EXPORT
    df_export = pd.DataFrame({
        't_sec': t,
        'alpha_fluctuating': sample_alpha,
        'felt_g': sample_g_felt,
        'jerk_m_s3': sample_jerk
    })
    df_export.to_csv(f"{OUTPUT_DIR}/robust_inertial_telemetry.csv", index=False)
    
    summary = pd.DataFrame({
        'Metric': ['Ideal_G', 'Mean_Max_G_5pct', 'Mean_Max_Jerk_5pct', 'Mean_Max_G_10pct', 'Mean_Max_Jerk_10pct'],
        'Value': [A_EXT_G / TARGET_ALPHA, np.mean(max_g_5pct), np.mean(max_jerk_5pct), np.mean(max_g_10pct), np.mean(max_jerk_10pct)]
    })
    summary.to_csv(f"{OUTPUT_DIR}/robust_inertial_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()