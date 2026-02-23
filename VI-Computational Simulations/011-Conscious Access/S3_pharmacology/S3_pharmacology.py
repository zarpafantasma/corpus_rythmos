#!/usr/bin/env python3
"""
S3: Pharmacological Effects on Consciousness Signatures
=======================================================

RTM-Consciousness predictions for pharmacological states:

PROPOFOL (GABAergic anesthesia):
- α decreases (coherence collapse)
- NDI decreases (directionality collapse)
- Both S1 and S2 fail → unconsciousness

PSYCHEDELICS (serotonergic):
- α may increase locally (enhanced coherence)
- NDI decreases or reverses (disrupted forward flow)
- S1 may pass, S2 fails → altered consciousness

This simulation demonstrates:
1. Dose-response curves for α and NDI
2. Propofol induction/emergence dynamics
3. Psychedelic state characterization
4. Dissociation between S1 and S2

THEORETICAL MODEL - requires validation with pharmacological studies
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# PHARMACOLOGICAL MODELS
# =============================================================================

def propofol_effect(dose, ec50=2.0, hill=3):
    """
    Propofol effect on neural coherence (Hill equation).
    
    Effect = 1 / (1 + (EC50/dose)^hill)
    
    Returns suppression factor (0 = no effect, 1 = full suppression)
    """
    if dose <= 0:
        return 0
    return 1 / (1 + (ec50 / dose) ** hill)


def psychedelic_effect(dose, ec50=0.15, hill=2):
    """
    Psychedelic effect (e.g., psilocybin, LSD).
    
    Enhances local coherence but disrupts hierarchical flow.
    """
    if dose <= 0:
        return 0
    return 1 / (1 + (ec50 / dose) ** hill)


def alpha_under_propofol(dose, alpha_baseline=0.72, alpha_min=0.25):
    """
    α as function of propofol dose.
    
    Propofol suppresses coherence → α decreases
    """
    suppression = propofol_effect(dose)
    return alpha_baseline - (alpha_baseline - alpha_min) * suppression


def ndi_under_propofol(dose, ndi_baseline=0.45, ndi_min=-0.1):
    """
    NDI as function of propofol dose.
    
    Propofol collapses directional flow → NDI approaches zero
    """
    suppression = propofol_effect(dose)
    return ndi_baseline - (ndi_baseline - ndi_min) * suppression


def alpha_under_psychedelic(dose, alpha_baseline=0.72, alpha_peak=0.85):
    """
    α as function of psychedelic dose.
    
    Psychedelics may increase local coherence
    """
    effect = psychedelic_effect(dose)
    return alpha_baseline + (alpha_peak - alpha_baseline) * effect


def ndi_under_psychedelic(dose, ndi_baseline=0.45, ndi_altered=-0.2):
    """
    NDI as function of psychedelic dose.
    
    Psychedelics disrupt hierarchical flow, may reverse direction
    """
    effect = psychedelic_effect(dose)
    return ndi_baseline - (ndi_baseline - ndi_altered) * effect


# =============================================================================
# TEMPORAL DYNAMICS
# =============================================================================

def propofol_induction_emergence(duration_min=60, induction_time=10, 
                                  maintenance_time=30, emergence_time=20):
    """
    Model α and NDI during propofol induction, maintenance, emergence.
    """
    t = np.linspace(0, duration_min, 300)
    
    alpha = np.zeros_like(t)
    ndi = np.zeros_like(t)
    
    alpha_awake = 0.72
    alpha_anesthesia = 0.28
    ndi_awake = 0.45
    ndi_anesthesia = 0.0
    
    tau_induction = 3  # minutes
    tau_emergence = 8  # minutes
    
    for i, ti in enumerate(t):
        if ti < induction_time:
            # Induction phase
            progress = 1 - np.exp(-ti / tau_induction)
            alpha[i] = alpha_awake - (alpha_awake - alpha_anesthesia) * progress
            ndi[i] = ndi_awake - (ndi_awake - ndi_anesthesia) * progress
        elif ti < induction_time + maintenance_time:
            # Maintenance (steady state)
            alpha[i] = alpha_anesthesia + 0.03 * np.random.randn()
            ndi[i] = ndi_anesthesia + 0.05 * np.random.randn()
        else:
            # Emergence
            time_in_emergence = ti - induction_time - maintenance_time
            progress = 1 - np.exp(-time_in_emergence / tau_emergence)
            alpha[i] = alpha_anesthesia + (alpha_awake - alpha_anesthesia) * progress
            ndi[i] = ndi_anesthesia + (ndi_awake - ndi_anesthesia) * progress
    
    # Add noise
    alpha += 0.02 * np.random.randn(len(t))
    ndi += 0.03 * np.random.randn(len(t))
    
    return t, alpha, ndi


def psychedelic_session(duration_min=180, onset_time=30, peak_time=60, 
                        offset_time=120):
    """
    Model α and NDI during psychedelic session.
    """
    t = np.linspace(0, duration_min, 360)
    
    alpha = np.zeros_like(t)
    ndi = np.zeros_like(t)
    
    alpha_baseline = 0.72
    alpha_peak = 0.82
    ndi_baseline = 0.45
    ndi_peak = -0.15
    
    for i, ti in enumerate(t):
        if ti < onset_time:
            # Pre-onset
            alpha[i] = alpha_baseline
            ndi[i] = ndi_baseline
        elif ti < peak_time:
            # Rising
            progress = (ti - onset_time) / (peak_time - onset_time)
            alpha[i] = alpha_baseline + (alpha_peak - alpha_baseline) * progress
            ndi[i] = ndi_baseline + (ndi_peak - ndi_baseline) * progress
        elif ti < offset_time:
            # Peak plateau
            alpha[i] = alpha_peak
            ndi[i] = ndi_peak
        else:
            # Coming down
            progress = (ti - offset_time) / (duration_min - offset_time)
            alpha[i] = alpha_peak - (alpha_peak - alpha_baseline) * progress
            ndi[i] = ndi_peak - (ndi_peak - ndi_baseline) * progress
    
    # Add characteristic fluctuations
    alpha += 0.04 * np.sin(2 * np.pi * 0.05 * t) + 0.02 * np.random.randn(len(t))
    ndi += 0.05 * np.random.randn(len(t))
    
    return t, alpha, ndi


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3: Pharmacological Effects on Consciousness Signatures")
    print("=" * 70)
    
    output_dir = "/home/claude/011-Conscious_Access/S3_pharmacology/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    ALPHA_CRIT = 0.50
    NDI_CRIT = 0.15
    
    # ===================
    # Part 1: Dose-response curves
    # ===================
    
    print("\n1. Computing dose-response curves...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Propofol dose-response
    doses_propofol = np.linspace(0, 5, 100)  # µg/mL
    
    ax = axes1[0, 0]
    
    alpha_prop = [alpha_under_propofol(d) for d in doses_propofol]
    ndi_prop = [ndi_under_propofol(d) for d in doses_propofol]
    
    ax.plot(doses_propofol, alpha_prop, 'b-', linewidth=2, label='α')
    ax.plot(doses_propofol, ndi_prop, 'r-', linewidth=2, label='NDI')
    ax.axhline(y=ALPHA_CRIT, color='blue', linestyle='--', alpha=0.5, label='α_crit')
    ax.axhline(y=NDI_CRIT, color='red', linestyle='--', alpha=0.5, label='NDI_crit')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Mark LOC threshold
    loc_dose = 2.0  # Approximate EC50 for LOC
    ax.axvline(x=loc_dose, color='green', linestyle=':', linewidth=2, label='LOC threshold')
    
    ax.set_xlabel('Propofol Dose (µg/mL)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Propofol: Dose-Response\nBoth α and NDI collapse', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 0.9)
    
    # Psychedelic dose-response
    doses_psy = np.linspace(0, 0.5, 100)  # mg/kg equivalent
    
    ax = axes1[0, 1]
    
    alpha_psy = [alpha_under_psychedelic(d) for d in doses_psy]
    ndi_psy = [ndi_under_psychedelic(d) for d in doses_psy]
    
    ax.plot(doses_psy, alpha_psy, 'b-', linewidth=2, label='α')
    ax.plot(doses_psy, ndi_psy, 'r-', linewidth=2, label='NDI')
    ax.axhline(y=ALPHA_CRIT, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=NDI_CRIT, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Psychedelic Dose (mg/kg)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Psychedelic: Dose-Response\nα increases, NDI decreases (S1↑, S2↓)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 0.9)
    
    # Propofol dynamics
    ax = axes1[1, 0]
    
    t_prop, alpha_dyn, ndi_dyn = propofol_induction_emergence()
    
    ax.plot(t_prop, alpha_dyn, 'b-', linewidth=2, label='α')
    ax.plot(t_prop, ndi_dyn, 'r-', linewidth=2, label='NDI')
    ax.axhline(y=ALPHA_CRIT, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=NDI_CRIT, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Mark phases
    ax.axvspan(0, 10, alpha=0.1, color='red', label='Induction')
    ax.axvspan(10, 40, alpha=0.1, color='gray', label='Maintenance')
    ax.axvspan(40, 60, alpha=0.1, color='green', label='Emergence')
    
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Propofol: Induction/Emergence\nα and NDI track consciousness level', fontsize=12)
    ax.legend(fontsize=8, loc='right')
    ax.grid(True, alpha=0.3)
    
    # Psychedelic dynamics
    ax = axes1[1, 1]
    
    t_psy, alpha_psy_dyn, ndi_psy_dyn = psychedelic_session()
    
    ax.plot(t_psy, alpha_psy_dyn, 'b-', linewidth=2, label='α')
    ax.plot(t_psy, ndi_psy_dyn, 'r-', linewidth=2, label='NDI')
    ax.axhline(y=ALPHA_CRIT, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Mark phases
    ax.axvspan(30, 60, alpha=0.1, color='purple', label='Onset')
    ax.axvspan(60, 120, alpha=0.1, color='magenta', label='Peak')
    
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Psychedelic: Session Dynamics\nα↑ but NDI↓ (dissociation)', fontsize=12)
    ax.legend(fontsize=8, loc='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_dose_response.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_dose_response.pdf'))
    plt.close()
    
    # ===================
    # Part 2: State comparison
    # ===================
    
    print("\n2. Comparing pharmacological states...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # α-NDI state space
    ax = axes2[0]
    
    states = {
        'Awake': (0.72, 0.45),
        'Light Sedation': (0.55, 0.25),
        'Deep Anesthesia': (0.28, 0.02),
        'REM': (0.65, 0.35),
        'NREM': (0.35, 0.05),
        'Psy Baseline': (0.72, 0.45),
        'Psy Peak': (0.82, -0.15),
        'Ketamine': (0.60, -0.05)
    }
    
    colors = {
        'Awake': 'green',
        'Light Sedation': 'orange',
        'Deep Anesthesia': 'red',
        'REM': 'cyan',
        'NREM': 'brown',
        'Psy Baseline': 'green',
        'Psy Peak': 'purple',
        'Ketamine': 'magenta'
    }
    
    for state, (alpha, ndi) in states.items():
        ax.scatter([alpha], [ndi], s=150, c=colors[state], label=state, zorder=3)
        ax.annotate(state, xy=(alpha, ndi), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # Draw threshold lines
    ax.axvline(x=ALPHA_CRIT, color='blue', linestyle='--', alpha=0.7, label='α_crit')
    ax.axhline(y=NDI_CRIT, color='red', linestyle='--', alpha=0.7, label='NDI_crit')
    
    # Shade conscious region
    ax.fill_between([ALPHA_CRIT, 1.0], NDI_CRIT, 0.6, alpha=0.1, color='green', 
                    label='Conscious region')
    
    # Draw trajectories
    # Propofol trajectory
    ax.annotate('', xy=(0.28, 0.02), xytext=(0.72, 0.45),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, 0.28, 'Propofol', fontsize=9, color='red', rotation=-30)
    
    # Psychedelic trajectory
    ax.annotate('', xy=(0.82, -0.15), xytext=(0.72, 0.45),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.text(0.78, 0.20, 'Psychedelic', fontsize=9, color='purple', rotation=-60)
    
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_ylabel('Net Directionality Index (NDI)', fontsize=11)
    ax.set_title('α-NDI State Space\nDifferent paths to altered consciousness', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(-0.3, 0.6)
    
    # Classification
    ax = axes2[1]
    
    # Simulate many trials
    n_trials = 200
    
    results = []
    for state, (alpha_base, ndi_base) in states.items():
        for _ in range(n_trials // len(states)):
            alpha = alpha_base + 0.05 * np.random.randn()
            ndi = ndi_base + 0.08 * np.random.randn()
            
            s1_pass = alpha > ALPHA_CRIT
            s2_pass = ndi > NDI_CRIT
            
            if s1_pass and s2_pass:
                prediction = 'Normal Conscious'
            elif s1_pass and not s2_pass:
                prediction = 'Altered Conscious'
            elif not s1_pass and not s2_pass:
                prediction = 'Unconscious'
            else:
                prediction = 'Ambiguous'
            
            results.append({
                'state': state,
                'alpha': alpha,
                'ndi': ndi,
                's1_pass': s1_pass,
                's2_pass': s2_pass,
                'prediction': prediction
            })
    
    df = pd.DataFrame(results)
    
    # Count predictions
    pred_counts = df.groupby(['state', 'prediction']).size().unstack(fill_value=0)
    
    pred_counts.plot(kind='bar', stacked=True, ax=ax, alpha=0.8,
                     color=['green', 'purple', 'gray', 'red'])
    
    ax.set_xlabel('State', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Classification by S1/S2 Criteria\nNormal, Altered, Unconscious', fontsize=12)
    ax.legend(title='Prediction', fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_state_comparison.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df.to_csv(os.path.join(output_dir, 'S3_classification_data.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S3: Pharmacological Effects on Consciousness Signatures
======================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-CONSCIOUSNESS PHARMACOLOGY
------------------------------
Different drugs affect α (S1) and NDI (S2) differently:

PROPOFOL (GABAergic anesthesia)
-------------------------------
- α: Decreases (0.72 → 0.28)
- NDI: Decreases (0.45 → 0.02)
- Both S1 and S2 fail
- Result: Unconsciousness (both criteria fail)
- Mechanism: Global inhibition, coherence collapse

PSYCHEDELICS (serotonergic)
---------------------------
- α: Increases (0.72 → 0.82)
- NDI: Decreases (0.45 → -0.15)
- S1 passes, S2 fails
- Result: Altered consciousness (dissociation)
- Mechanism: Enhanced local coherence, disrupted hierarchy

KETAMINE (NMDA antagonist)
--------------------------
- α: Moderate decrease (0.72 → 0.60)
- NDI: Near zero/negative (0.45 → -0.05)
- S1 borderline, S2 fails
- Result: Dissociative state

CLASSIFICATION CRITERIA
-----------------------
Normal Conscious: α > {ALPHA_CRIT} AND NDI > {NDI_CRIT}
Altered Conscious: α > {ALPHA_CRIT} AND NDI ≤ {NDI_CRIT}
Unconscious: α ≤ {ALPHA_CRIT} AND NDI ≤ {NDI_CRIT}

KEY PREDICTIONS
---------------
1. Propofol: Both α and NDI drop during induction
2. Emergence: α recovers before NDI
3. Psychedelics: α↑ and NDI↓ simultaneously
4. Different routes to altered states visible in α-NDI space

CLINICAL IMPLICATIONS
---------------------
1. Monitor α for depth of anesthesia
2. NDI may indicate recovery of directional processing
3. Psychedelic states distinct from anesthesia
4. Combined S1+S2 provides richer picture than either alone
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\nPropofol effects:")
    print(f"  α: 0.72 → 0.28 (collapse)")
    print(f"  NDI: 0.45 → 0.02 (collapse)")
    print("\nPsychedelic effects:")
    print(f"  α: 0.72 → 0.82 (increase)")
    print(f"  NDI: 0.45 → -0.15 (reversal)")
    print(f"\nOutputs: {output_dir}/")
    
    return df


if __name__ == "__main__":
    main()
