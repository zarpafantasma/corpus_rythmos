#!/usr/bin/env python3
"""
T1: RTM Coherence Exponent α(r) Profile Calculator
===================================================

From "Black Holes in the RTM Framework"

Calculates the environmental coherence exponent α as a function of
radius r (or confinement index ξ). Two profile types are supported:

1. Logistic (saturating): α(r) = α_∞ + (α_0 - α_∞) / [1 + exp(-(r - r_t)/w)]
2. Power-like (ramp): α(r) = α_∞ + (α_0 - α_∞) × (r/r_t)^(-q)

Key insight: α increases inward (as confinement/organization grows),
leading to steeper T ∝ L^α scaling at smaller radii.

Reference: Paper Sections 2.2, 8.1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# PROFILE FUNCTIONS
# =============================================================================

def alpha_logistic(r: np.ndarray, alpha_0: float = 1.0, alpha_inf: float = 3.0,
                   r_t: float = 6.0, w: float = 2.0) -> np.ndarray:
    """
    Logistic (saturating) profile for α(r).
    
    α(r) = α_∞ + (α_0 - α_∞) / [1 + exp(-(r - r_t)/w)]
    
    Parameters:
    -----------
    r : array-like
        Radial coordinate (in units of r_s = Schwarzschild radius)
    alpha_0 : float
        Asymptotic value at large r (far from hole)
    alpha_inf : float
        Saturation value at small r (near horizon)
    r_t : float
        Transition radius (midpoint of logistic)
    w : float
        Width of transition region
    
    Returns:
    --------
    α(r) : array-like
        Coherence exponent at each radius
    """
    return alpha_inf + (alpha_0 - alpha_inf) / (1 + np.exp(-(r - r_t) / w))


def alpha_ramp(r: np.ndarray, alpha_0: float = 1.0, alpha_inf: float = 3.0,
               r_t: float = 6.0, q: float = 1.5) -> np.ndarray:
    """
    Power-like (soft ramp) profile for α(r).
    
    α(r) = α_∞ + (α_0 - α_∞) × (r/r_t)^(-q)    for r > r_t
         = α_∞                                   for r ≤ r_t
    
    Parameters:
    -----------
    r : array-like
        Radial coordinate
    alpha_0 : float
        Asymptotic value at large r
    alpha_inf : float
        Saturation value at small r
    r_t : float
        Transition radius
    q : float
        Power-law exponent (steepness of ramp)
    """
    r = np.asarray(r)
    result = np.where(r > r_t,
                      alpha_inf + (alpha_0 - alpha_inf) * (r / r_t) ** (-q),
                      alpha_inf)
    return result


def alpha_confinement(xi: np.ndarray, alpha_0: float = 1.0, alpha_inf: float = 3.0,
                      xi_t: float = 0.5, w: float = 0.2) -> np.ndarray:
    """
    Logistic profile for α(ξ) where ξ is confinement index (0 to 1).
    
    Higher ξ = more confinement = higher α.
    """
    return alpha_0 + (alpha_inf - alpha_0) / (1 + np.exp(-(xi - xi_t) / w))


# =============================================================================
# GR REDSHIFT FACTOR
# =============================================================================

def Z_schwarzschild(r: np.ndarray, r_s: float = 2.0) -> np.ndarray:
    """
    Gravitational redshift factor for Schwarzschild metric.
    
    Z(r) = 1 / √(1 - r_s/r)
    
    Maps local proper time to asymptotic observer time.
    Z → ∞ as r → r_s (horizon).
    """
    r = np.asarray(r)
    # Avoid singularity at horizon
    r_safe = np.maximum(r, r_s * 1.001)
    return 1.0 / np.sqrt(1 - r_s / r_safe)


# =============================================================================
# PROCESS TIME MODEL
# =============================================================================

def tau_obs(L: np.ndarray, r: float, alpha_r: float, Z_r: float,
            L_0: float = 1.0, T_0: float = 1.0, noise_sigma: float = 0.0,
            rng: np.random.Generator = None) -> np.ndarray:
    """
    Observed process time as function of effective size L.
    
    τ_obs = Z(r) × (L/L_0)^α(r) × T_0 × ε
    
    where ε is lognormal noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    L = np.asarray(L)
    tau = Z_r * (L / L_0) ** alpha_r * T_0
    
    if noise_sigma > 0:
        epsilon = rng.lognormal(0, noise_sigma, size=L.shape)
        tau = tau * epsilon
    
    return tau


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir: str):
    """Create visualization of α profiles and their effects."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Logistic vs Ramp profiles
    ax1 = axes[0, 0]
    r = np.linspace(2.5, 20, 200)
    
    alpha_log = alpha_logistic(r, alpha_0=1.0, alpha_inf=3.0, r_t=6.0, w=2.0)
    alpha_rmp = alpha_ramp(r, alpha_0=1.0, alpha_inf=3.0, r_t=6.0, q=1.5)
    
    ax1.plot(r, alpha_log, 'b-', linewidth=2, label='Logistic')
    ax1.plot(r, alpha_rmp, 'r--', linewidth=2, label='Power-law ramp')
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='α₀ = 1 (far)')
    ax1.axhline(y=3.0, color='gray', linestyle=':', alpha=0.5, label='α_∞ = 3 (near)')
    ax1.axvline(x=6.0, color='green', linestyle='--', alpha=0.5, label='r_t = 6')
    ax1.axvline(x=2.0, color='black', linestyle='-', alpha=0.5, label='r_s (horizon)')
    
    ax1.set_xlabel('Radius r/r_s', fontsize=12)
    ax1.set_ylabel('Coherence exponent α(r)', fontsize=12)
    ax1.set_title('α(r) Profiles: Coherence Increases Inward', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 20)
    
    # Plot 2: α vs confinement index
    ax2 = axes[0, 1]
    xi = np.linspace(0, 1, 100)
    alpha_conf = alpha_confinement(xi, alpha_0=1.0, alpha_inf=3.0, xi_t=0.5, w=0.15)
    
    ax2.plot(xi, alpha_conf, 'b-', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=3.0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='ξ_t = 0.5')
    
    ax2.set_xlabel('Confinement index ξ', fontsize=12)
    ax2.set_ylabel('Coherence exponent α(ξ)', fontsize=12)
    ax2.set_title('α(ξ) for Analog Platforms (BEC/Fluids)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GR redshift factor Z(r)
    ax3 = axes[1, 0]
    r_z = np.linspace(2.1, 20, 200)
    Z = Z_schwarzschild(r_z, r_s=2.0)
    
    ax3.semilogy(r_z, Z, 'b-', linewidth=2)
    ax3.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Horizon (r_s)')
    ax3.axvline(x=3.0, color='orange', linestyle=':', alpha=0.7, label='ISCO (3r_s)')
    ax3.axvline(x=6.0, color='green', linestyle=':', alpha=0.7, label='r = 6r_s')
    
    ax3.set_xlabel('Radius r/r_s', fontsize=12)
    ax3.set_ylabel('Redshift factor Z(r)', fontsize=12)
    ax3.set_title('GR Time Dilation: Z = 1/√(1 - r_s/r)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2, 20)
    
    # Plot 4: Effect on τ_obs at different radii
    ax4 = axes[1, 1]
    L = np.geomspace(0.1, 10, 50)
    
    for r_val, color in [(15, 'blue'), (8, 'orange'), (4, 'red')]:
        alpha_r = alpha_logistic(r_val)
        Z_r = Z_schwarzschild(r_val)
        tau = tau_obs(L, r_val, alpha_r, Z_r)
        ax4.loglog(L, tau, color=color, linewidth=2, 
                   label=f'r = {r_val}: α = {alpha_r:.2f}, Z = {Z_r:.2f}')
    
    ax4.set_xlabel('Effective size L', fontsize=12)
    ax4.set_ylabel('Observed time τ_obs', fontsize=12)
    ax4.set_title('τ_obs = Z(r) × L^α(r): Slope = α, Level ∝ Z', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'T1_alpha_profile.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'T1_alpha_profile.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("T1: RTM Coherence Exponent α(r) Profile Calculator")
    print("From: Black Holes in the RTM Framework")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate profile data
    r_values = np.linspace(2.5, 20, 50)
    xi_values = np.linspace(0, 1, 50)
    
    records = []
    for r in r_values:
        records.append({
            'r': r,
            'alpha_logistic': alpha_logistic(r),
            'alpha_ramp': alpha_ramp(r),
            'Z_r': Z_schwarzschild(r)
        })
    
    df_radial = pd.DataFrame(records)
    df_radial.to_csv(os.path.join(output_dir, 'T1_radial_profiles.csv'), index=False)
    
    records_conf = []
    for xi in xi_values:
        records_conf.append({
            'xi': xi,
            'alpha': alpha_confinement(xi)
        })
    
    df_conf = pd.DataFrame(records_conf)
    df_conf.to_csv(os.path.join(output_dir, 'T1_confinement_profiles.csv'), index=False)
    
    # Print key results
    print("\nKEY α(r) VALUES (Logistic Profile)")
    print("-" * 50)
    for r in [3, 4, 6, 8, 10, 15, 20]:
        alpha = alpha_logistic(r)
        Z = Z_schwarzschild(r)
        print(f"  r = {r:2d}r_s:  α = {alpha:.3f},  Z = {Z:.3f}")
    
    print("\n" + "=" * 66)
    print("PROFILE EQUATIONS")
    print("=" * 66)
    print("""
Logistic:   α(r) = α_∞ + (α₀ - α_∞) / [1 + exp(-(r - r_t)/w)]
Power-ramp: α(r) = α_∞ + (α₀ - α_∞) × (r/r_t)^(-q)

GR redshift: Z(r) = 1 / √(1 - r_s/r)

Observed time: τ_obs = Z(r) × L^α(r) × T_0 × ε
    """)
    
    print("KEY INSIGHT:")
    print("-" * 50)
    print("  • Slope in log(τ) vs log(L) equals α(r)")
    print("  • Intercept depends on Z(r) (GR/kinematics)")
    print("  • RTM → slope evolution with radius")
    print("  • GR  → level (intercept) shift only")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Summary
    summary = f"""T1: RTM α(r) Profile Calculator
================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EQUATIONS
---------
Logistic profile:
  α(r) = α_∞ + (α₀ - α_∞) / [1 + exp(-(r - r_t)/w)]
  
Power-law ramp:
  α(r) = α_∞ + (α₀ - α_∞) × (r/r_t)^(-q)

GR redshift factor:
  Z(r) = 1 / √(1 - r_s/r)

DEFAULT PARAMETERS
------------------
α₀ = 1.0   (far from hole)
α_∞ = 3.0  (near horizon)
r_t = 6 r_s (transition radius)
w = 2 r_s  (transition width)

PHYSICAL INTERPRETATION
-----------------------
• α increases inward as confinement/organization grows
• Higher α → steeper T ∝ L^α scaling
• Z increases inward (GR time dilation)
• Competition: Z stretches times, high α shortens them

KEY VALUES (Logistic)
---------------------
r = 3 r_s:  α = {alpha_logistic(3):.3f}, Z = {Z_schwarzschild(3):.3f}
r = 6 r_s:  α = {alpha_logistic(6):.3f}, Z = {Z_schwarzschild(6):.3f}
r = 10 r_s: α = {alpha_logistic(10):.3f}, Z = {Z_schwarzschild(10):.3f}
r = 20 r_s: α = {alpha_logistic(20):.3f}, Z = {Z_schwarzschild(20):.3f}
"""
    
    with open(os.path.join(output_dir, 'T1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
