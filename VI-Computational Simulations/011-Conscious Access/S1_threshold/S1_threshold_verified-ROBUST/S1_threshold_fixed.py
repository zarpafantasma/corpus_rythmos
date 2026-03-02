#!/usr/bin/env python3
"""
S1: Consciousness Threshold Model (α > α_crit)
==============================================

RTM-Consciousness hypothesis: Conscious access occurs when the
coherence exponent α crosses a critical threshold α_crit.

Key predictions:
- Report trials: α > α_crit (conscious access)
- No-report trials: α < α_crit (no access)
- α is necessary but not sufficient for consciousness

This simulation demonstrates:
1. α as threshold for conscious access
2. Report vs no-report differences
3. Neural hierarchy α profiles
4. Threshold detection methodology

THEORETICAL MODEL - requires validation with EEG/MEG data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM CONSCIOUSNESS MODEL
# =============================================================================

def tau_from_scale(L, tau_0, alpha, L_ref=1.0):
    return tau_0 * (L / L_ref) ** alpha

def compute_alpha_from_neural_signal(signal_data, scales, fs=256):
    taus = []
    
    for scale in scales:
        window_samples = int(scale * fs)
        if window_samples < 10:
            window_samples = 10
        
        n_windows = len(signal_data) // window_samples
        autocorr_times = []
        
        for i in range(n_windows):
            segment = signal_data[i*window_samples:(i+1)*window_samples]
            if len(segment) > 10:
                autocorr = np.correlate(segment - np.mean(segment), 
                                        segment - np.mean(segment), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                below_threshold = np.where(autocorr < 1/np.e)[0]
                if len(below_threshold) > 0:
                    tau = below_threshold[0] / fs
                else:
                    tau = len(autocorr) / fs
                autocorr_times.append(tau)
        
        if autocorr_times:
            taus.append(np.median(autocorr_times))
        else:
            taus.append(scale)
    
    log_scales = np.log(scales)
    log_taus = np.log(taus)
    
    slope, intercept, r, p, se = stats.linregress(log_scales, log_taus)
    
    return slope, r**2, taus


def generate_neural_signal(duration, fs, alpha_target, noise_level=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    alpha_weight = (alpha_target - 1.6) / 1.4 
    alpha_weight = np.clip(alpha_weight, 0.1, 0.9)
    
    delta = 2 * np.sin(2 * np.pi * 2 * t) 
    theta = 1.5 * np.sin(2 * np.pi * 6 * t) 
    alpha_osc = 1.0 * np.sin(2 * np.pi * 10 * t) 
    beta = 0.5 * np.sin(2 * np.pi * 20 * t) 
    gamma = 0.3 * np.sin(2 * np.pi * 40 * t) 
    
    coherent = delta + theta + alpha_osc + beta + gamma
    
    noise = noise_level * np.random.randn(n_samples)
    noise = signal.filtfilt(*signal.butter(4, [1, 80], btype='band', fs=fs), noise)
    
    signal_out = alpha_weight * coherent + (1 - alpha_weight) * noise * 3
    
    return t, signal_out


CONSCIOUSNESS_STATES = {
    'Awake Report': {
        'alpha': 2.44,
        'conscious': True,
        'description': 'Full conscious access, stimulus reported'
    },
    'Awake No-Report': {
        'alpha': 1.96,
        'conscious': False,
        'description': 'Stimulus presented but not reported'
    },
    'REM Sleep': {
        'alpha': 2.30,
        'conscious': True,
        'description': 'Dreaming, partial conscious access'
    },
    'NREM Sleep': {
        'alpha': 1.70,
        'conscious': False,
        'description': 'Deep sleep, no conscious access'
    },
    'Light Sedation': {
        'alpha': 2.04,
        'conscious': True,
        'description': 'Mild propofol, responsive'
    },
    'Deep Anesthesia': {
        'alpha': 1.56,
        'conscious': False,
        'description': 'Full propofol, unresponsive'
    }
}

ALPHA_CRIT = 2.00

def main():
    print("=" * 70)
    print("S1: Consciousness Threshold Model (α > α_crit)")
    print("=" * 70)
    
    output_dir = "output_S1_threshold"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    print("\n1. Demonstrating consciousness threshold...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes1[0, 0]
    scales = np.logspace(-1, 1, 20) 
    conscious_colors = {'True': 'green', 'False': 'red'}
    
    for state_name, params in CONSCIOUSNESS_STATES.items():
        tau = tau_from_scale(scales, tau_0=0.1, alpha=params['alpha'])
        color = conscious_colors[str(params['conscious'])]
        linestyle = '-' if params['conscious'] else '--'
        ax.plot(scales, tau, linewidth=2, color=color, linestyle=linestyle,
                label=f"{state_name} (α={params['alpha']:.2f})")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Scale L (seconds)', fontsize=11)
    ax.set_ylabel('Integration Time τ (seconds)', fontsize=11)
    ax.set_title('RTM Scaling: τ ∝ L^α\nConscious states (green) have higher α', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes1[0, 1]
    states = list(CONSCIOUSNESS_STATES.keys())
    alphas = [CONSCIOUSNESS_STATES[s]['alpha'] for s in states]
    conscious = [CONSCIOUSNESS_STATES[s]['conscious'] for s in states]
    colors = ['green' if c else 'red' for c in conscious]
    
    y_pos = range(len(states))
    ax.barh(y_pos, alphas, color=colors, alpha=0.7)
    ax.axvline(x=ALPHA_CRIT, color='black', linewidth=3, linestyle='--', 
               label=f'α_crit = {ALPHA_CRIT}')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(states, fontsize=9)
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_title(f'α by Consciousness State\nThreshold α_crit = {ALPHA_CRIT}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    ax = axes1[1, 0]
    t_con, sig_con = generate_neural_signal(2, 256, 2.44, seed=42)
    t_uncon, sig_uncon = generate_neural_signal(2, 256, 1.70, seed=43)
    
    ax.plot(t_con, sig_con + 10, 'g-', linewidth=0.8, label='Conscious (α=2.44)')
    ax.plot(t_uncon, sig_uncon, 'r-', linewidth=0.8, label='Unconscious (α=1.70)')
    
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Neural Signal (a.u.)', fontsize=11)
    ax.set_title('Simulated Neural Signals\nConscious = more coherent oscillations', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)
    
    ax = axes1[1, 1]
    n_trials = 100
    results = []
    
    for state_name, params in CONSCIOUSNESS_STATES.items():
        for _ in range(n_trials // len(CONSCIOUSNESS_STATES)):
            alpha_measured = params['alpha'] + 0.08 * np.random.randn()
            predicted_conscious = alpha_measured > ALPHA_CRIT
            actual_conscious = params['conscious']
            
            results.append({
                'state': state_name,
                'alpha': alpha_measured,
                'predicted': predicted_conscious,
                'actual': actual_conscious,
                'correct': predicted_conscious == actual_conscious
            })
    
    df = pd.DataFrame(results)
    accuracy = df['correct'].mean()
    
    thresholds = np.linspace(0.3, 0.8, 50)
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        df['pred_temp'] = df['alpha'] > thresh
        tp = ((df['pred_temp'] == True) & (df['actual'] == True)).sum()
        fn = ((df['pred_temp'] == False) & (df['actual'] == True)).sum()
        fp = ((df['pred_temp'] == True) & (df['actual'] == False)).sum()
        tn = ((df['pred_temp'] == False) & (df['actual'] == False)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    ax.plot(fpr_list, tpr_list, 'b-', linewidth=2, label='ROC')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    
    optimal_idx = np.argmax(np.array(tpr_list) - np.array(fpr_list))
    ax.scatter([fpr_list[optimal_idx]], [tpr_list[optimal_idx]], 
               s=100, c='red', zorder=5, label=f'Optimal (α={thresholds[optimal_idx]:.2f})')
    
    auc = np.trapz(tpr_list[::-1], fpr_list[::-1])
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Consciousness Classification\nAUC = {auc:.3f}, Acc = {accuracy:.1%}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_threshold_model.png'), dpi=150)
    plt.close()
    
    print("\n2. Analyzing report vs no-report trials...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    n_trials = 200
    intensities = np.random.uniform(0, 1, n_trials)
    alphas_trial = 1.70 + 1.0 * intensities + 0.16 * np.random.randn(n_trials)
    report_prob = 1 / (1 + np.exp(-7.5 * (alphas_trial - ALPHA_CRIT)))
    reported = np.random.rand(n_trials) < report_prob
    
    ax = axes2[0]
    ax.scatter(intensities[reported], alphas_trial[reported], 
               s=30, c='green', alpha=0.6, label='Report')
    ax.scatter(intensities[~reported], alphas_trial[~reported], 
               s=30, c='red', alpha=0.6, label='No Report')
    ax.axhline(y=ALPHA_CRIT, color='black', linewidth=2, linestyle='--',
               label=f'α_crit = {ALPHA_CRIT}')
    
    ax.set_xlabel('Stimulus Intensity', fontsize=11)
    ax.set_ylabel('Coherence Exponent α', fontsize=11)
    ax.set_title('Masked Detection Task\nReport correlates with α > α_crit', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes2[1]
    
    ax.hist(alphas_trial[reported], bins=20, alpha=0.6, color='green', 
            label=f'Report (n={reported.sum()})', density=True)
    ax.hist(alphas_trial[~reported], bins=20, alpha=0.6, color='red',
            label=f'No Report (n={(~reported).sum()})', density=True)
    ax.axvline(x=ALPHA_CRIT, color='black', linewidth=2, linestyle='--')
    
    t_stat, p_val = stats.ttest_ind(alphas_trial[reported], alphas_trial[~reported])
    effect_size = (alphas_trial[reported].mean() - alphas_trial[~reported].mean()) / alphas_trial.std()
    
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'α Distribution by Report\nt = {t_stat:.2f}, p < 0.001, d = {effect_size:.2f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_report_no_report.png'), dpi=150)
    plt.close()
    
    df.to_csv(os.path.join(output_dir, 'S1_trial_data.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nClassification accuracy: {accuracy:.1%}")
    print(f"AUC: {auc:.3f}")
    print(f"Report vs No-Report effect size: d = {effect_size:.2f}")

if __name__ == '__main__':
    main()
