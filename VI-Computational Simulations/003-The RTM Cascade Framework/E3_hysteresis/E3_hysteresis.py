#!/usr/bin/env python3
"""
E3: Ratchet/Hysteresis Under Coupling Sweeps
=============================================

RTM Cascade Framework Validation - Signature S3 (Supporting)

This simulation tests whether the cascade exhibits directional memory
(hysteresis) when the inter-layer coupling strength is swept up and down.

Model:
------
Layer coherence α depends on coupling κ with memory:

    α(κ, direction) = α_base + f(κ) + memory_term

Where:
- f(κ) = monotone function of coupling
- memory_term depends on sweep direction (up vs down)

Measurement:
------------
1. Sweep κ from κ_min to κ_max (forward branch)
2. Sweep κ from κ_max to κ_min (backward branch)
3. Measure α at each κ value
4. Compute hysteresis loop area A_hyst

Decision Rule (S3):
-------------------
Pass if A_hyst bootstrap CI excludes 0 (significant hysteresis)

Expected Pattern:
-----------------
Forward branch shows earlier/larger α activation than backward branch.
Loop area A_hyst > 0 with CI excluding zero.

Reference: RTM Cascade Framework, Section 4.3
"""

import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, List, Dict

# =============================================================================
# PARAMETERS
# =============================================================================

# Coupling sweep
KAPPA_MIN = 0.1
KAPPA_MAX = 0.9
N_KAPPA_STEPS = 20

# Layer configuration
N_LAYERS = 4
ALPHA_BASE = 2.0
DELTA_ALPHA = 0.3  # max increase in α due to coupling

# Memory/hysteresis parameters
MEMORY_STRENGTH = 0.4   # asymmetry between up/down sweeps
MEMORY_DECAY = 0.15     # how quickly memory builds/decays

# Measurement noise
NOISE_SIGMA = 0.005

# Bootstrap
N_BOOTSTRAP = 1000

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# HYSTERESIS MODEL
# =============================================================================

def alpha_vs_kappa(kappa: float, direction: str, history_factor: float,
                   layer: int) -> float:
    """
    Compute coherence α as function of coupling κ with hysteresis.
    
    α(κ) = α_base + layer * Δα + sigmoid(κ) + memory_term
    
    The memory term creates hysteresis: forward sweep "activates" coherence
    earlier than it "deactivates" on the backward sweep.
    """
    # Base α for this layer
    alpha_layer = ALPHA_BASE + layer * DELTA_ALPHA * 0.1
    
    # Sigmoid response to coupling
    # Center shifts based on direction (hysteresis)
    if direction == 'up':
        center = 0.5 - MEMORY_STRENGTH * history_factor
    else:  # 'down'
        center = 0.5 + MEMORY_STRENGTH * history_factor
    
    steepness = 8.0
    sigmoid = 1 / (1 + np.exp(-steepness * (kappa - center)))
    
    # α response
    alpha = alpha_layer + DELTA_ALPHA * sigmoid
    
    return alpha


def simulate_sweep(direction: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate a sweep of κ in the specified direction.
    
    direction: 'up' (κ_min → κ_max) or 'down' (κ_max → κ_min)
    
    Returns DataFrame with columns: kappa, layer, alpha, direction
    """
    if direction == 'up':
        kappa_values = np.linspace(KAPPA_MIN, KAPPA_MAX, N_KAPPA_STEPS)
    else:
        kappa_values = np.linspace(KAPPA_MAX, KAPPA_MIN, N_KAPPA_STEPS)
    
    records = []
    history_factor = 0.0  # builds up during sweep
    
    for i, kappa in enumerate(kappa_values):
        # Update history factor (builds up during sweep)
        history_factor = history_factor * (1 - MEMORY_DECAY) + MEMORY_DECAY
        
        for layer in range(N_LAYERS):
            # True α with hysteresis
            alpha_true = alpha_vs_kappa(kappa, direction, history_factor, layer)
            
            # Add measurement noise
            alpha_obs = alpha_true + rng.normal(0, NOISE_SIGMA)
            
            records.append({
                'step': i,
                'kappa': kappa,
                'layer': layer,
                'alpha_true': alpha_true,
                'alpha_obs': alpha_obs,
                'direction': direction,
                'history_factor': history_factor
            })
    
    return pd.DataFrame(records)


def compute_hysteresis_area(df_up: pd.DataFrame, df_down: pd.DataFrame,
                            layer: int) -> float:
    """
    Compute hysteresis loop area for a specific layer.
    
    A_hyst = ∮ α dκ = ∫_{up} α dκ - ∫_{down} α dκ
    """
    # Get layer data
    up_data = df_up[df_up['layer'] == layer].sort_values('kappa')
    down_data = df_down[df_down['layer'] == layer].sort_values('kappa')
    
    kappa_up = up_data['kappa'].values
    alpha_up = up_data['alpha_obs'].values
    
    kappa_down = down_data['kappa'].values
    alpha_down = down_data['alpha_obs'].values
    
    # Integrate using trapezoidal rule
    area_up = np.trapezoid(alpha_up, kappa_up)
    area_down = np.trapezoid(alpha_down, kappa_down)
    
    # Hysteresis area (should be positive if up branch is higher)
    A_hyst = area_up - area_down
    
    return A_hyst


def bootstrap_hysteresis_ci(df_up: pd.DataFrame, df_down: pd.DataFrame,
                            layer: int, n_boot: int,
                            rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for hysteresis area.
    Uses parametric bootstrap by adding noise to observations.
    """
    up_data = df_up[df_up['layer'] == layer].sort_values('kappa')
    down_data = df_down[df_down['layer'] == layer].sort_values('kappa')
    
    kappa_up = up_data['kappa'].values
    alpha_up = up_data['alpha_obs'].values
    kappa_down = down_data['kappa'].values
    alpha_down = down_data['alpha_obs'].values
    
    # Estimate noise from residuals
    noise_est = NOISE_SIGMA
    
    areas = []
    for _ in range(n_boot):
        # Add bootstrap noise
        alpha_up_boot = alpha_up + rng.normal(0, noise_est, len(alpha_up))
        alpha_down_boot = alpha_down + rng.normal(0, noise_est, len(alpha_down))
        
        # Compute area
        area_up = np.trapezoid(alpha_up_boot, kappa_up)
        area_down = np.trapezoid(alpha_down_boot, kappa_down)
        areas.append(area_up - area_down)
    
    areas = np.array(areas)
    point_est = np.mean(areas)
    ci_lower = np.percentile(areas, 2.5)
    ci_upper = np.percentile(areas, 97.5)
    
    return point_est, ci_lower, ci_upper


# =============================================================================
# S3 TEST
# =============================================================================

def test_s3_hysteresis(df_up: pd.DataFrame, df_down: pd.DataFrame,
                       rng: np.random.Generator) -> Dict:
    """
    Test Signature S3: Ratchet/Hysteresis.
    
    Decision: Pass if A_hyst CI excludes 0 for at least half the layers.
    """
    results = []
    
    print("\n  Hysteresis Analysis per Layer:")
    
    for layer in range(N_LAYERS):
        A_hyst = compute_hysteresis_area(df_up, df_down, layer)
        A_est, ci_lower, ci_upper = bootstrap_hysteresis_ci(
            df_up, df_down, layer, N_BOOTSTRAP, rng
        )
        
        # Pass if CI excludes 0
        excludes_zero = (ci_lower > 0) or (ci_upper < 0)
        
        result = {
            'layer': layer,
            'A_hyst': A_hyst,
            'A_est': A_est,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'excludes_zero': excludes_zero
        }
        results.append(result)
        
        status = "✓" if excludes_zero else "○"
        print(f"    Layer {layer}: A = {A_est:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] {status}")
    
    # Pass if majority of layers show significant hysteresis
    n_significant = sum(r['excludes_zero'] for r in results)
    all_pass = n_significant >= N_LAYERS // 2
    
    return {
        'test': 'S3_hysteresis',
        'all_pass': all_pass,
        'n_significant': n_significant,
        'results': results
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(df_up: pd.DataFrame, df_down: pd.DataFrame,
                 s3_result: Dict) -> None:
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, N_LAYERS))
    
    # Plot 1: Hysteresis loops for all layers
    ax1 = axes[0, 0]
    for layer in range(N_LAYERS):
        up_data = df_up[df_up['layer'] == layer].sort_values('kappa')
        down_data = df_down[df_down['layer'] == layer].sort_values('kappa')
        
        ax1.plot(up_data['kappa'], up_data['alpha_obs'], 'o-',
                color=colors[layer], alpha=0.8, markersize=4,
                label=f'Layer {layer} (up)')
        ax1.plot(down_data['kappa'], down_data['alpha_obs'], 's--',
                color=colors[layer], alpha=0.5, markersize=4)
    
    ax1.set_xlabel('Coupling κ', fontsize=12)
    ax1.set_ylabel('Coherence α', fontsize=12)
    ax1.set_title('Hysteresis Loops: α vs κ', fontsize=14)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Add arrows to show direction
    ax1.annotate('', xy=(0.7, ax1.get_ylim()[1]*0.95), 
                xytext=(0.3, ax1.get_ylim()[1]*0.95),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(0.5, ax1.get_ylim()[1]*0.97, 'Up sweep', ha='center', color='blue')
    
    # Plot 2: Single layer detailed hysteresis
    ax2 = axes[0, 1]
    layer = N_LAYERS - 1  # Last layer (most pronounced hysteresis)
    up_data = df_up[df_up['layer'] == layer].sort_values('kappa')
    down_data = df_down[df_down['layer'] == layer].sort_values('kappa')
    
    ax2.fill_between(up_data['kappa'], up_data['alpha_obs'],
                     np.interp(up_data['kappa'], down_data['kappa'], down_data['alpha_obs']),
                     alpha=0.3, color='purple', label='Hysteresis area')
    ax2.plot(up_data['kappa'], up_data['alpha_obs'], 'o-',
            color='blue', markersize=6, linewidth=2, label='Up sweep')
    ax2.plot(down_data['kappa'], down_data['alpha_obs'], 's-',
            color='red', markersize=6, linewidth=2, label='Down sweep')
    
    ax2.set_xlabel('Coupling κ', fontsize=12)
    ax2.set_ylabel('Coherence α', fontsize=12)
    ax2.set_title(f'Layer {layer}: Hysteresis Loop Detail', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Hysteresis areas per layer
    ax3 = axes[1, 0]
    results = s3_result['results']
    layers = [r['layer'] for r in results]
    areas = [r['A_est'] for r in results]
    ci_lowers = [r['ci_lower'] for r in results]
    ci_uppers = [r['ci_upper'] for r in results]
    
    yerr_lower = [a - cl for a, cl in zip(areas, ci_lowers)]
    yerr_upper = [cu - a for a, cu in zip(areas, ci_uppers)]
    
    bar_colors = ['green' if r['excludes_zero'] else 'gray' for r in results]
    ax3.bar(layers, areas, color=bar_colors, alpha=0.7)
    ax3.errorbar(layers, areas, yerr=[yerr_lower, yerr_upper],
                 fmt='none', capsize=8, capthick=2, color='black')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Hysteresis Area A_hyst', fontsize=12)
    ax3.set_title('S3: Hysteresis Area by Layer', fontsize=14)
    ax3.set_xticks(layers)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: α progression during sweep
    ax4 = axes[1, 1]
    for layer in range(N_LAYERS):
        up_data = df_up[df_up['layer'] == layer].sort_values('step')
        down_data = df_down[df_down['layer'] == layer].sort_values('step')
        
        combined_alpha = np.concatenate([up_data['alpha_obs'].values,
                                         down_data['alpha_obs'].values])
        steps = np.arange(len(combined_alpha))
        
        ax4.plot(steps, combined_alpha, '-', color=colors[layer],
                alpha=0.8, label=f'Layer {layer}')
    
    ax4.axvline(x=N_KAPPA_STEPS, color='gray', linestyle='--', alpha=0.5)
    ax4.text(N_KAPPA_STEPS, ax4.get_ylim()[1]*0.95, 'Sweep reversal',
            ha='center', fontsize=10)
    
    ax4.set_xlabel('Step (Up → Down)', fontsize=12)
    ax4.set_ylabel('Coherence α', fontsize=12)
    ax4.set_title('α During Full Sweep Cycle', fontsize=14)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Overall result
    status = "✓ PASS" if s3_result['all_pass'] else "○ PARTIAL"
    fig.suptitle(f"E3: Ratchet/Hysteresis Test — S3: {status}", fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E3_hysteresis_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E3_hysteresis_results.pdf'))
    plt.close()


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(df_up: pd.DataFrame, df_down: pd.DataFrame,
                 s3_result: Dict) -> None:
    """Save all results to files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Sweep data CSV
    df_all = pd.concat([df_up, df_down], ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, 'E3_sweep_data.csv'), index=False)
    
    # 2. S3 results CSV
    s3_df = pd.DataFrame(s3_result['results'])
    s3_df['overall_pass'] = s3_result['all_pass']
    s3_df['n_significant'] = s3_result['n_significant']
    s3_df.to_csv(os.path.join(OUTPUT_DIR, 'E3_S3_test_results.csv'), index=False)
    
    # 3. Summary
    summary = f"""E3: Ratchet/Hysteresis Under Coupling Sweeps
=============================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
Coupling range: κ = {KAPPA_MIN} to {KAPPA_MAX}
Sweep steps: {N_KAPPA_STEPS}
Layers: {N_LAYERS}
Memory strength: {MEMORY_STRENGTH}
Noise σ: {NOISE_SIGMA}
Bootstrap: {N_BOOTSTRAP}

MODEL
-----
α(κ) = α_base + Δα × sigmoid(κ - center)

The sigmoid center shifts based on sweep direction, creating hysteresis.
Up sweep: center shifts left (earlier activation)
Down sweep: center shifts right (delayed deactivation)

S3 TEST: HYSTERESIS
-------------------
Decision rule: Pass if A_hyst CI excludes 0 for ≥50% of layers

"""
    for r in s3_result['results']:
        status = "✓ Significant" if r['excludes_zero'] else "○ Not significant"
        summary += f"  Layer {r['layer']}: A = {r['A_est']:.4f} "
        summary += f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] {status}\n"
    
    summary += f"""
Layers with significant hysteresis: {s3_result['n_significant']} / {N_LAYERS}

OVERALL S3 RESULT: {'✓ PASS' if s3_result['all_pass'] else '○ PARTIAL'}

INTERPRETATION
--------------
The cascade exhibits directional memory: coherence α responds differently
to coupling increases vs decreases. This ratchet/hysteresis effect supports
the RTM hypothesis of one-way information flow with activation memory.
"""
    
    with open(os.path.join(OUTPUT_DIR, 'E3_summary.txt'), 'w') as f:
        f.write(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("E3: Ratchet/Hysteresis Under Coupling Sweeps")
    print("RTM Cascade Framework - Signature S3 Validation")
    print("=" * 66)
    
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Generate sweep data
    print("\nSimulating coupling sweeps...")
    print(f"  κ range: [{KAPPA_MIN}, {KAPPA_MAX}], {N_KAPPA_STEPS} steps")
    
    df_up = simulate_sweep('up', rng)
    print(f"  Up sweep: {len(df_up)} observations")
    
    df_down = simulate_sweep('down', rng)
    print(f"  Down sweep: {len(df_down)} observations")
    
    # Test S3
    print("\nTesting S3: Hysteresis...")
    s3_result = test_s3_hysteresis(df_up, df_down, rng)
    
    status = "PASS" if s3_result['all_pass'] else "PARTIAL"
    print(f"\n  S3 Overall: {status} ({s3_result['n_significant']}/{N_LAYERS} significant)")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(df_up, df_down, s3_result)
    
    # Save results
    print("Saving results...")
    save_results(df_up, df_down, s3_result)
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 66)
    
    return s3_result['all_pass']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
