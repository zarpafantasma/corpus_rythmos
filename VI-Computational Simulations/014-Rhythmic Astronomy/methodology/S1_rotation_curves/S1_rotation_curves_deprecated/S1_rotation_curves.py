#!/usr/bin/env python3
"""
S1: Synthetic Rotation Curves with RTM α-Profiles
==================================================

From "Rhythmic Astronomy: An RTM Slope Law for Galaxy Rotation Curves"

PURPOSE
-------
This simulation ILLUSTRATES what the RTM model predicts for rotation curves
given different radial profiles of the coherence exponent α(r).

WHAT THIS DEMONSTRATES:
- How v ∝ r^(1-α/2) produces different curve shapes
- α = 2 gives flat rotation curves
- α < 2 gives rising curves
- α > 2 gives declining (Keplerian-like) curves

WHAT THIS DOES NOT PROVE:
- This does NOT prove RTM is correct
- This does NOT prove dark matter is unnecessary
- Real validation requires comparison with observed galaxy data
- This is a MODEL ILLUSTRATION, not an empirical test

The RTM hypothesis is that α encodes "baryonic coherence" and varies
with galactic structure. This is an ALTERNATIVE to dark matter halos,
not a proven replacement.

Reference: Paper Sections 3.1, 4.1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM ROTATION CURVE MODEL
# =============================================================================

def rtm_velocity(r, alpha, v_ref=200.0, r_ref=10.0):
    """
    RTM velocity law: v ∝ r^(1 - α/2)
    
    Derivation:
        RTM: T ∝ L^α
        Orbital period: T = 2πr/v
        Therefore: v ∝ r^(1 - α/2)
    
    Parameters
    ----------
    r : array
        Galactocentric radius (kpc)
    alpha : float or array
        RTM coherence exponent (can vary with r)
    v_ref : float
        Reference velocity (km/s) at r_ref
    r_ref : float
        Reference radius (kpc)
    
    Returns
    -------
    v : array
        Circular velocity (km/s)
    """
    # Handle scalar or array alpha
    if np.isscalar(alpha):
        alpha_arr = np.full_like(r, alpha)
    else:
        alpha_arr = alpha
    
    # Compute velocity with local alpha
    # v = v_ref * (r/r_ref)^(1 - alpha/2)
    exponent = 1 - alpha_arr / 2
    v = v_ref * (r / r_ref) ** exponent
    
    return v


def slope_from_alpha(alpha):
    """
    The log-log slope of v vs r is (1 - α/2).
    
    α = 2  → slope = 0 (flat curve)
    α = 1  → slope = 0.5 (rising)
    α = 3  → slope = -0.5 (declining)
    """
    return 1 - alpha / 2


# =============================================================================
# ALPHA PROFILE MODELS
# =============================================================================

def constant_alpha(r, alpha_value=2.0):
    """Constant α across all radii."""
    return np.full_like(r, alpha_value)


def radial_gradient_alpha(r, alpha_inner=2.5, alpha_outer=2.0, r_trans=5.0, width=2.0):
    """
    Smooth transition from high α (inner) to lower α (outer).
    
    This models: structured inner region → diffuse outer disk
    """
    return alpha_outer + (alpha_inner - alpha_outer) / (1 + np.exp((r - r_trans) / width))


def bar_bulge_alpha(r, alpha_bar=2.8, alpha_disk=2.0, r_bar=3.0):
    """
    High α in bar/bulge region, lower in disk.
    """
    return np.where(r < r_bar, alpha_bar, alpha_disk)


def clumpy_alpha(r, alpha_base=2.0, amplitude=0.3, n_clumps=5, seed=42):
    """
    Spatially varying α due to clumpy structure.
    """
    np.random.seed(seed)
    alpha = np.full_like(r, alpha_base)
    
    # Add clumps at random positions
    clump_positions = np.random.uniform(r.min(), r.max(), n_clumps)
    clump_widths = np.random.uniform(0.5, 2.0, n_clumps)
    
    for pos, width in zip(clump_positions, clump_widths):
        alpha += amplitude * np.exp(-((r - pos) / width) ** 2)
    
    return alpha


# =============================================================================
# COMPARISON WITH STANDARD MODELS
# =============================================================================

def nfw_rotation_curve(r, v200=200.0, c=10.0, r200=200.0):
    """
    NFW dark matter halo rotation curve for comparison.
    
    v²(r) = V200² × [ln(1+cx) - cx/(1+cx)] / [x × (ln(1+c) - c/(1+c))]
    where x = r/r200
    """
    x = r / r200
    cx = c * x
    
    f_c = np.log(1 + c) - c / (1 + c)
    f_cx = np.log(1 + cx) - cx / (1 + cx)
    
    v_squared = v200**2 * f_cx / (x * f_c)
    return np.sqrt(np.maximum(v_squared, 0))


def exponential_disk_velocity(r, v_max=200.0, r_d=3.0):
    """
    Freeman exponential disk (no halo).
    v(r) peaks at ~2.2 r_d then declines Keplerian-like.
    """
    y = r / (2 * r_d)
    # Approximation to the exact integral
    v = v_max * np.sqrt(y**2 * (3.2 * np.exp(-y) + 0.5 / (1 + y**1.5)))
    v = np.minimum(v, v_max * 1.1)  # Cap at reasonable max
    return v


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir):
    """Create comprehensive visualization."""
    
    r = np.linspace(0.5, 30, 200)  # kpc
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # ===================
    # Plot 1: Different constant α values
    # ===================
    ax1 = axes[0, 0]
    
    alpha_values = [1.5, 1.8, 2.0, 2.2, 2.5]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(alpha_values)))
    
    for alpha, color in zip(alpha_values, colors):
        v = rtm_velocity(r, alpha)
        slope = slope_from_alpha(alpha)
        ax1.plot(r, v, color=color, linewidth=2, 
                 label=f'α={alpha:.1f} (slope={slope:.2f})')
    
    ax1.set_xlabel('Radius (kpc)', fontsize=11)
    ax1.set_ylabel('v (km/s)', fontsize=11)
    ax1.set_title('RTM Prediction: Constant α', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(50, 350)
    
    # ===================
    # Plot 2: Log-log slopes
    # ===================
    ax2 = axes[0, 1]
    
    for alpha, color in zip(alpha_values, colors):
        v = rtm_velocity(r, alpha)
        ax2.loglog(r, v, color=color, linewidth=2, label=f'α={alpha:.1f}')
    
    ax2.set_xlabel('log(r)', fontsize=11)
    ax2.set_ylabel('log(v)', fontsize=11)
    ax2.set_title('Log-Log Slopes = (1 - α/2)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # ===================
    # Plot 3: Radial gradient α(r)
    # ===================
    ax3 = axes[0, 2]
    
    alpha_profile = radial_gradient_alpha(r, alpha_inner=2.4, alpha_outer=2.0)
    v_rtm = rtm_velocity(r, alpha_profile)
    
    # Compare with NFW
    v_nfw = nfw_rotation_curve(r)
    v_disk = exponential_disk_velocity(r)
    
    ax3.plot(r, v_rtm, 'b-', linewidth=2, label='RTM (varying α)')
    ax3.plot(r, v_nfw, 'r--', linewidth=2, label='NFW halo')
    ax3.plot(r, v_disk, 'g:', linewidth=2, label='Disk only')
    
    ax3.set_xlabel('Radius (kpc)', fontsize=11)
    ax3.set_ylabel('v (km/s)', fontsize=11)
    ax3.set_title('RTM vs Standard Models', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ===================
    # Plot 4: α(r) profile
    # ===================
    ax4 = axes[1, 0]
    
    ax4.plot(r, alpha_profile, 'b-', linewidth=2)
    ax4.axhline(y=2.0, color='gray', linestyle='--', label='α=2 (flat)')
    ax4.fill_between(r, 2.0, alpha_profile, alpha=0.3, 
                     where=alpha_profile > 2.0, color='red', label='α>2 (declining)')
    ax4.fill_between(r, 2.0, alpha_profile, alpha=0.3,
                     where=alpha_profile < 2.0, color='blue', label='α<2 (rising)')
    
    ax4.set_xlabel('Radius (kpc)', fontsize=11)
    ax4.set_ylabel('α(r)', fontsize=11)
    ax4.set_title('Coherence Exponent Profile', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(1.5, 2.8)
    
    # ===================
    # Plot 5: Bar/bulge case
    # ===================
    ax5 = axes[1, 1]
    
    alpha_bar = bar_bulge_alpha(r, alpha_bar=2.6, alpha_disk=2.0, r_bar=4.0)
    v_bar = rtm_velocity(r, alpha_bar)
    
    ax5.plot(r, v_bar, 'purple', linewidth=2, label='RTM (bar+disk)')
    ax5.axvline(x=4.0, color='orange', linestyle='--', label='Bar radius')
    
    ax5.set_xlabel('Radius (kpc)', fontsize=11)
    ax5.set_ylabel('v (km/s)', fontsize=11)
    ax5.set_title('Bar/Bulge Structure (α=2.6 → 2.0)', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ===================
    # Plot 6: Slope vs α relationship
    # ===================
    ax6 = axes[1, 2]
    
    alpha_range = np.linspace(1.0, 3.5, 100)
    slopes = slope_from_alpha(alpha_range)
    
    ax6.plot(alpha_range, slopes, 'k-', linewidth=2)
    ax6.axhline(y=0, color='gray', linestyle='--')
    ax6.axvline(x=2, color='red', linestyle='--', label='α=2 (flat)')
    
    # Mark key points
    ax6.scatter([1, 2, 3], [0.5, 0, -0.5], s=100, c=['blue', 'gray', 'green'], 
                zorder=5, edgecolors='black')
    ax6.annotate('Rising', (1.2, 0.45), fontsize=10)
    ax6.annotate('Flat', (2.1, 0.05), fontsize=10)
    ax6.annotate('Declining', (2.7, -0.45), fontsize=10)
    
    ax6.set_xlabel('α', fontsize=11)
    ax6.set_ylabel('Slope = 1 - α/2', fontsize=11)
    ax6.set_title('The RTM Slope Law', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0.8, 3.5)
    ax6.set_ylim(-0.8, 0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_rotation_curves.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_rotation_curves.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S1: Synthetic Rotation Curves with RTM α-Profiles")
    print("From: Rhythmic Astronomy - Sections 3.1, 4.1")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("IMPORTANT DISCLAIMERS")
    print("=" * 70)
    print("""
    This simulation ILLUSTRATES RTM predictions. It does NOT prove:
    - That RTM correctly describes galaxy dynamics
    - That dark matter is unnecessary
    - That α can be measured from baryonic structure
    
    Real validation requires comparison with observed rotation curves
    and independent estimation of α from structural proxies.
    
    RTM is a HYPOTHESIS, not an established theory.
    """)
    
    print("=" * 70)
    print("RTM ROTATION CURVE MODEL")
    print("=" * 70)
    print("""
    Core relation: T ∝ L^α  (RTM master law)
    
    For circular orbits:
        T_orb = 2πr/v
        Therefore: v ∝ r^(1 - α/2)
    
    Log-log slope: d(log v)/d(log r) = 1 - α/2
    
    Key predictions:
        α = 2.0  →  slope = 0    (FLAT curve)
        α = 1.5  →  slope = 0.25 (RISING curve)
        α = 2.5  →  slope = -0.25 (DECLINING curve)
        α = 3.0  →  slope = -0.5  (Keplerian-like)
    
    The hypothesis: α reflects baryonic coherence/structure
        - Diffuse outer disks → α ≈ 2 → flat rotation
        - Structured inner regions (bars/bulges) → α > 2
        - This is ALTERNATIVE to dark matter, not proven
    """)
    
    print("=" * 70)
    print("GENERATING SYNTHETIC CURVES")
    print("=" * 70)
    
    r = np.linspace(1, 25, 100)
    
    # Generate different scenarios
    scenarios = {
        'constant_flat': ('α=2 everywhere', constant_alpha(r, 2.0)),
        'constant_rising': ('α=1.8 everywhere', constant_alpha(r, 1.8)),
        'gradient': ('α: 2.4→2.0', radial_gradient_alpha(r)),
        'bar_disk': ('Bar (α=2.6) + Disk (α=2.0)', bar_bulge_alpha(r)),
    }
    
    results = []
    
    for name, (desc, alpha_profile) in scenarios.items():
        v = rtm_velocity(r, alpha_profile)
        
        # Compute local slopes
        log_r = np.log10(r)
        log_v = np.log10(v)
        
        # Numerical slope at each point
        slopes = np.gradient(log_v, log_r)
        
        mean_alpha = np.mean(alpha_profile)
        predicted_slope = 1 - mean_alpha / 2
        measured_slope = np.mean(slopes)
        
        print(f"\n    {desc}:")
        print(f"      Mean α = {mean_alpha:.2f}")
        print(f"      Predicted slope = {predicted_slope:.3f}")
        print(f"      Measured slope = {measured_slope:.3f}")
        
        results.append({
            'scenario': name,
            'description': desc,
            'mean_alpha': mean_alpha,
            'predicted_slope': predicted_slope,
            'measured_slope': measured_slope
        })
    
    # Save data
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'S1_scenarios.csv'), index=False)
    
    # Save sample curve data
    alpha_gradient = radial_gradient_alpha(r)
    v_gradient = rtm_velocity(r, alpha_gradient)
    
    df_curve = pd.DataFrame({
        'radius_kpc': r,
        'alpha': alpha_gradient,
        'velocity_km_s': v_gradient,
        'log_r': np.log10(r),
        'log_v': np.log10(v_gradient)
    })
    df_curve.to_csv(os.path.join(output_dir, 'S1_sample_curve.csv'), index=False)
    
    # Create plots
    print("\n\nCreating plots...")
    create_plots(output_dir)
    
    # Summary
    summary = f"""S1: Synthetic Rotation Curves with RTM
======================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMPORTANT: This is a MODEL ILLUSTRATION, not empirical validation.

RTM MODEL
---------
v ∝ r^(1 - α/2)

Slope = 1 - α/2:
  α = 2.0 → slope = 0 (flat)
  α < 2.0 → slope > 0 (rising)
  α > 2.0 → slope < 0 (declining)

SCENARIOS TESTED
----------------
{chr(10).join(f"  {r['description']}: α={r['mean_alpha']:.2f}, slope={r['measured_slope']:.3f}" for r in results)}

WHAT THIS SHOWS
---------------
✓ RTM velocity law produces expected curve shapes
✓ Varying α(r) creates realistic-looking rotation curves
✓ Flat outer curves require α ≈ 2

WHAT THIS DOES NOT SHOW
-----------------------
✗ That RTM is physically correct
✗ That α can be measured from structure
✗ That dark matter is unnecessary
✗ Comparison with real galaxy data

NEXT STEPS FOR VALIDATION
-------------------------
1. Measure α independently from baryonic structure proxies
2. Compare predicted slopes with observed rotation curves
3. Test collapse condition (residuals flat vs r)
4. Check bTFR residuals correlate with α-proxies
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
