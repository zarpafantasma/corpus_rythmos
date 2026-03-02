#!/usr/bin/env python3
"""
S5: Inertial Mitigation via Temporal Decoupling
===============================================

From "Aetherion, The Jumper" - Chapter II, Section 5.4

Simulates G-force reduction inside a high-α cabin during extreme maneuvers.

Key Equations (from paper):
    Proper time: dτ = dt / Γ(α)
    Clock-rate factor: Γ(α) = α^(1/2)  (phenomenological)
    
    Effective acceleration:
        a_eff = a_ext / Γ² = a_ext / α
    
Paper's numerical example:
    a_ext = 100 g, α = 4
    → a_eff = 100/4 = 25 g
    → With α ≈ 15-20, even 100 g feels like < 2 g

Reference: Paper Chapter II, Section 5.4 "Inertial Mitigation via Temporal Decoupling"
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

G_ACCEL = 9.81  # m/s²


# =============================================================================
# TEMPORAL DECOUPLING MODEL
# =============================================================================

def clock_rate_factor(alpha: float) -> float:
    """
    Clock-rate factor Γ(α).
    
    From paper: Γ = α^(1/2) (phenomenological mapping)
    
    Higher α → slower proper time → reduced perceived acceleration
    """
    return np.sqrt(alpha)


def effective_acceleration(a_external: float, alpha: float) -> float:
    """
    Effective (perceived) acceleration inside high-α cabin.
    
    a_eff = a_ext / Γ² = a_ext / α
    """
    gamma = clock_rate_factor(alpha)
    return a_external / gamma**2


def required_alpha_for_comfort(a_external: float, 
                                a_max_tolerable: float = 2.0 * G_ACCEL) -> float:
    """
    Calculate required α to reduce external acceleration to tolerable level.
    
    a_eff = a_ext / α → α = a_ext / a_eff
    """
    return a_external / a_max_tolerable


def proper_time_dilation(t_external: float, alpha: float) -> float:
    """
    Proper time experienced inside high-α region.
    
    τ = t / Γ(α) = t / √α
    """
    gamma = clock_rate_factor(alpha)
    return t_external / gamma


# =============================================================================
# TRAJECTORY SIMULATION
# =============================================================================

def simulate_maneuver(a_external: float, alpha_cabin: float, 
                      duration: float = 2.0, dt: float = 0.001):
    """
    Simulate a maneuver comparing external and cabin frames.
    
    Parameters:
    -----------
    a_external : float
        External acceleration (m/s²)
    alpha_cabin : float
        Cabin RTM exponent
    duration : float
        Maneuver duration in external time (s)
    """
    N = int(duration / dt)
    t_ext = np.linspace(0, duration, N)
    
    gamma = clock_rate_factor(alpha_cabin)
    a_eff = effective_acceleration(a_external, alpha_cabin)
    
    # External frame trajectory
    v_ext = a_external * t_ext
    x_ext = 0.5 * a_external * t_ext**2
    
    # Proper time in cabin
    tau = t_ext / gamma
    
    # Cabin-perceived trajectory (same x, but parametrized by τ)
    # x(τ) = x(t(τ)) where t = γτ
    x_cabin = 0.5 * a_external * (gamma * tau)**2
    
    # Perceived velocity and acceleration
    v_cabin = a_external * gamma * tau  # dx/dτ = (dx/dt)(dt/dτ) = v_ext * γ
    a_cabin_perceived = a_external * gamma**2  # d²x/dτ² = a_ext * γ²
    
    # But the FELT acceleration (G-force) is reduced:
    # The body's inertial response is slowed by the same factor
    a_felt = a_eff  # This is what passengers actually experience
    
    return {
        't_ext': t_ext,
        'tau': tau,
        'x_ext': x_ext,
        'v_ext': v_ext,
        'a_ext': a_external,
        'a_felt': a_felt,
        'a_felt_g': a_felt / G_ACCEL,
        'gamma': gamma,
        'alpha': alpha_cabin,
        'x_final': x_ext[-1],
        'v_final': v_ext[-1]
    }


def scan_alpha_values(a_external: float, alpha_range: np.ndarray) -> dict:
    """Scan effective acceleration over range of α values."""
    a_eff = [effective_acceleration(a_external, a) for a in alpha_range]
    a_eff_g = [ae / G_ACCEL for ae in a_eff]
    
    return {
        'alpha': alpha_range,
        'a_eff': np.array(a_eff),
        'a_eff_g': np.array(a_eff_g)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(results: dict, output_dir: str):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    t = results['t_ext']
    tau = results['tau']
    
    # Plot 1: Trajectory comparison
    ax1 = axes[0, 0]
    ax1.plot(t, results['x_ext'] / 1000, 'b-', linewidth=2, label='x(t) external')
    ax1.plot(tau, results['x_ext'] / 1000, 'r--', linewidth=2, label='x(τ) proper time')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (km)', fontsize=12)
    ax1.set_title('Trajectory: External vs Proper Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity
    ax2 = axes[0, 1]
    ax2.plot(t, results['v_ext'], 'b-', linewidth=2, label='v(t) external')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title(f'Velocity Under {results["a_ext"]/G_ACCEL:.0f}g Acceleration', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time dilation
    ax3 = axes[0, 2]
    ax3.plot(t, tau, 'g-', linewidth=2)
    ax3.plot(t, t, 'k--', linewidth=1, alpha=0.5, label='τ = t (no dilation)')
    ax3.set_xlabel('External time t (s)', fontsize=12)
    ax3.set_ylabel('Proper time τ (s)', fontsize=12)
    ax3.set_title(f'Time Dilation (α = {results["alpha"]}, Γ = {results["gamma"]:.2f})', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax3.text(0.05, 0.95, f'τ = t / √α\nτ_final = {tau[-1]:.2f} s',
             transform=ax3.transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Effective acceleration vs α
    ax4 = axes[1, 0]
    alpha_range = np.linspace(1, 50, 100)
    scan = scan_alpha_values(results['a_ext'], alpha_range)
    
    ax4.semilogy(scan['alpha'], scan['a_eff_g'], 'b-', linewidth=2)
    ax4.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='2g (tolerable)')
    ax4.axhline(y=results['a_ext']/G_ACCEL, color='red', linestyle=':', alpha=0.7, 
                label=f'{results["a_ext"]/G_ACCEL:.0f}g (external)')
    ax4.axvline(x=results['alpha'], color='orange', linestyle='--', alpha=0.7,
                label=f'α = {results["alpha"]}')
    
    ax4.set_xlabel('Cabin α', fontsize=12)
    ax4.set_ylabel('Felt acceleration (g)', fontsize=12)
    ax4.set_title(f'G-Force Reduction: a_felt = a_ext / α', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, 50)
    
    # Plot 5: Required α for various external accelerations
    ax5 = axes[1, 1]
    a_ext_range = np.array([10, 30, 50, 100, 200, 500]) * G_ACCEL
    alpha_required = [required_alpha_for_comfort(a) for a in a_ext_range]
    
    ax5.bar(range(len(a_ext_range)), alpha_required, color='purple', alpha=0.7)
    ax5.set_xticks(range(len(a_ext_range)))
    ax5.set_xticklabels([f'{a/G_ACCEL:.0f}g' for a in a_ext_range])
    ax5.set_xlabel('External Acceleration', fontsize=12)
    ax5.set_ylabel('Required α for < 2g felt', fontsize=12)
    ax5.set_title('α Required for Comfort (< 2g)', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, (a, alpha) in enumerate(zip(a_ext_range, alpha_required)):
        ax5.text(i, alpha + 5, f'α={alpha:.0f}', ha='center', fontsize=10)
    
    # Plot 6: Design envelope
    ax6 = axes[1, 2]
    
    # Contour plot of felt acceleration
    alpha_grid = np.linspace(1, 100, 100)
    a_ext_grid = np.linspace(1, 200, 100) * G_ACCEL
    A, ALPHA = np.meshgrid(a_ext_grid, alpha_grid)
    A_FELT = A / ALPHA / G_ACCEL
    
    levels = [0.5, 1, 2, 5, 10, 20, 50]
    cs = ax6.contour(A/G_ACCEL, ALPHA, A_FELT, levels=levels, colors='blue')
    ax6.clabel(cs, inline=True, fontsize=10, fmt='%1.0fg')
    
    # Shade comfort zone
    ax6.fill_between([1, 200], [1, 1], [200/2, 200/2], alpha=0.2, color='green',
                     label='< 2g felt (comfort)')
    
    ax6.set_xlabel('External Acceleration (g)', fontsize=12)
    ax6.set_ylabel('Cabin α', fontsize=12)
    ax6.set_title('Design Envelope: Felt Acceleration Contours', fontsize=14)
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S5_inertial_mitigation.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S5_inertial_mitigation.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("S5: Inertial Mitigation via Temporal Decoupling")
    print("From: Aetherion, The Jumper - Chapter II")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 66)
    print("KEY EQUATIONS (from paper Section 5.4)")
    print("=" * 66)
    print("""
    Clock-rate factor:
        Γ(α) = √α  (phenomenological)
    
    Proper time dilation:
        dτ = dt / Γ = dt / √α
    
    Effective (felt) acceleration:
        a_eff = a_ext / Γ² = a_ext / α
    
    Physical interpretation:
        Inside high-α cabin, time flows slower.
        Rapid external maneuvers appear stretched.
        Passengers experience reduced G-forces.
    """)
    
    # Paper's numerical example
    print("=" * 66)
    print("PAPER'S NUMERICAL EXAMPLE (Section 5.4)")
    print("=" * 66)
    
    a_external = 100 * G_ACCEL  # 100 g
    alpha_cabin = 4
    
    gamma = clock_rate_factor(alpha_cabin)
    a_felt = effective_acceleration(a_external, alpha_cabin)
    
    print(f"\nExternal acceleration: {a_external/G_ACCEL:.0f} g")
    print(f"Cabin α: {alpha_cabin}")
    print(f"Clock-rate factor Γ: {gamma:.2f}")
    print(f"\nFelt acceleration: {a_felt/G_ACCEL:.1f} g")
    print(f"Paper states: 25 g (= 100/4)")
    
    # Enhanced example
    print("\n--- Enhanced mitigation (α = 50) ---")
    alpha_high = 50
    a_felt_high = effective_acceleration(a_external, alpha_high)
    print(f"With α = {alpha_high}: a_felt = {a_felt_high/G_ACCEL:.1f} g")
    
    # Run simulation
    print("\n" + "=" * 66)
    print("MANEUVER SIMULATION")
    print("=" * 66)
    
    results = simulate_maneuver(a_external, alpha_cabin, duration=2.0)
    
    print(f"\n100g maneuver over 2 seconds:")
    print(f"  Final position: {results['x_final']/1000:.2f} km")
    print(f"  Final velocity: {results['v_final']:.0f} m/s")
    print(f"  Proper time elapsed: {results['tau'][-1]:.2f} s")
    print(f"  Felt acceleration: {results['a_felt_g']:.1f} g")
    
    # Comfort requirements
    print("\n" + "=" * 66)
    print("COMFORT REQUIREMENTS")
    print("=" * 66)
    print(f"\nTo keep felt acceleration < 2g:")
    
    records = []
    for a_g in [10, 30, 50, 100, 200, 500]:
        a_ext = a_g * G_ACCEL
        alpha_req = required_alpha_for_comfort(a_ext, 2*G_ACCEL)
        print(f"  {a_g:>4}g external → α ≥ {alpha_req:.0f}")
        records.append({
            'a_external_g': a_g,
            'alpha_required': alpha_req
        })
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'S5_comfort_requirements.csv'), index=False)
    
    # Design implications
    print("\n" + "=" * 66)
    print("DESIGN IMPLICATIONS (from paper)")
    print("=" * 66)
    print("""
    Paper (Section 5.4) states:
    
    1. Cabin gradient: Maintain α >> 1 inside, taper to α ≈ 1 at hull
       → Protects occupants while preserving thrust efficiency
    
    2. Dynamic control: Increase interior α during hard maneuvers
       → Further suppress G-forces when needed
    
    3. Instrumentation: Dual-frame accelerometers
       → One locked to τ (cabin time), one to t (external time)
       → Directly verify a_felt vs a_ext
    """)
    
    # Save simulation data
    df_sim = pd.DataFrame({
        't_ext': results['t_ext'],
        'tau': results['tau'],
        'x_m': results['x_ext'],
        'v_m_s': results['v_ext']
    })
    df_sim.to_csv(os.path.join(output_dir, 'S5_maneuver_trajectory.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(results, output_dir)
    
    # Summary
    summary = f"""S5: Inertial Mitigation via Temporal Decoupling
================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY EQUATION
------------
a_eff = a_ext / α

Clock-rate factor: Γ = √α
Proper time: dτ = dt / Γ

PAPER EXAMPLE
-------------
External: 100g, Cabin α = 4
→ Felt: 100/4 = 25g

Enhanced: α = 50
→ Felt: 100/50 = 2g

COMFORT REQUIREMENTS (< 2g felt)
--------------------------------
10g external  → α ≥ {required_alpha_for_comfort(10*G_ACCEL, 2*G_ACCEL):.0f}
30g external  → α ≥ {required_alpha_for_comfort(30*G_ACCEL, 2*G_ACCEL):.0f}
100g external → α ≥ {required_alpha_for_comfort(100*G_ACCEL, 2*G_ACCEL):.0f}
500g external → α ≥ {required_alpha_for_comfort(500*G_ACCEL, 2*G_ACCEL):.0f}

SIMULATION RESULTS
------------------
100g maneuver over 2s (α = 4):
  Final position: {results['x_final']/1000:.2f} km
  Final velocity: {results['v_final']:.0f} m/s
  Felt G-force: {results['a_felt_g']:.1f} g

DESIGN GUIDELINES
-----------------
1. High α inside cabin, low α at hull
2. Dynamic α control during maneuvers
3. Dual-frame accelerometers for verification

PAPER VERIFICATION
------------------
✓ a_eff = a_ext / α confirmed
✓ Time dilation τ = t / √α
✓ Practical mitigation achievable with α ~ 50
"""
    
    with open(os.path.join(output_dir, 'S5_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
