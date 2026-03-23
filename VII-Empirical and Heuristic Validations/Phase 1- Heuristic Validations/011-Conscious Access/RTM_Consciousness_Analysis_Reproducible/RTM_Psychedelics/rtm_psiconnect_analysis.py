#!/usr/bin/env python3
"""
RTM Analysis of PsiConnect Dataset (OpenNeuro ds006110)
========================================================

This script analyzes the PsiConnect psilocybin neuroimaging dataset
through the lens of Multiscale Temporal Relativity (RTM) theory.

THEORETICAL FRAMEWORK:
RTM posits that consciousness correlates with topological coherence (α ≈ 1.0),
NOT with maximal entropy as proposed by the "Entropic Brain" hypothesis.

The spectral slope β of EEG power spectra relates to RTM transport exponent as:
    α_RTM ≈ 2 / β_spectral

HYPOTHESES TESTED:
H1: Psilocybin does NOT collapse α toward 0.5 (unconsciousness)
H2: Subjective intensity correlates with α stability, not entropy
H3: Different conditions (rest, meditation, music) show distinct α signatures
H4: Post-psilocybin α differs from pre-psilocybin baseline

Dataset: OpenNeuro ds006110 (PsiConnect)
- 62 participants
- 19mg psilocybin oral dose
- EEG + fMRI multimodal
- Conditions: rest, meditation, music, movie
- Half received 8-week meditation training

Author: RTM Research
Date: March 2026
License: CC BY 4.0
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, pearsonr, ttest_rel, ttest_ind, mannwhitneyu
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Optional imports (install if needed)
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("WARNING: MNE-Python not installed. Install with: pip install mne")

try:
    from fooof import FOOOF
    FOOOF_AVAILABLE = True
except ImportError:
    FOOOF_AVAILABLE = False
    print("WARNING: FOOOF not installed. Install with: pip install fooof")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Analysis configuration parameters"""
    
    # Dataset paths (adjust after downloading from OpenNeuro)
    DATA_ROOT = "./ds006110"
    OUTPUT_DIR = "./output_rtm_psiconnect"
    
    # EEG parameters
    SAMPLING_RATE = 1000  # Hz (verify with actual data)
    FREQ_RANGE = (1, 45)  # Hz for spectral analysis
    APERIODIC_RANGE = (1, 40)  # Hz for 1/f slope fitting
    
    # Sliding window parameters
    WINDOW_SIZE = 4.0  # seconds
    WINDOW_STEP = 1.0  # seconds (overlap = WINDOW_SIZE - WINDOW_STEP)
    
    # RTM theoretical thresholds
    ALPHA_CONSCIOUS = 1.0  # Healthy wakefulness
    ALPHA_UNCONSCIOUS = 0.5  # Random/uncorrelated (anesthesia)
    ALPHA_WARNING = 0.7  # Transition zone
    
    # Statistical parameters
    BOOTSTRAP_N = 1000
    ALPHA_SIGNIFICANCE = 0.05
    
    # Channels of interest (posterior for alpha, frontal for executive)
    POSTERIOR_CHANNELS = ['Oz', 'O1', 'O2', 'Pz', 'P3', 'P4', 'P7', 'P8']
    FRONTAL_CHANNELS = ['Fz', 'F3', 'F4', 'Fp1', 'Fp2', 'AF3', 'AF4']
    GLOBAL_CHANNELS = None  # None = use all channels


# =============================================================================
# CORE RTM METRICS
# =============================================================================

def compute_spectral_slope_welch(data, fs, freq_range=(1, 40), nperseg=None):
    """
    Compute spectral slope (β) using Welch's method + log-log regression.
    
    Parameters
    ----------
    data : array-like
        1D EEG time series
    fs : float
        Sampling frequency in Hz
    freq_range : tuple
        (low_freq, high_freq) for slope fitting
    nperseg : int, optional
        Segment length for Welch's method
        
    Returns
    -------
    beta : float
        Spectral slope (positive for 1/f^β decay)
    r_squared : float
        Goodness of fit
    freqs : array
        Frequency vector
    psd : array
        Power spectral density
    """
    if nperseg is None:
        nperseg = int(fs * 2)  # 2-second segments
    
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # Select frequency range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]
    
    # Avoid log(0)
    psd_fit = np.maximum(psd_fit, 1e-20)
    
    # Log-log linear regression
    log_freqs = np.log10(freqs_fit)
    log_psd = np.log10(psd_fit)
    
    coeffs = np.polyfit(log_freqs, log_psd, 1)
    beta = -coeffs[0]  # Positive β for 1/f^β decay
    
    # R-squared
    fitted = np.polyval(coeffs, log_freqs)
    ss_res = np.sum((log_psd - fitted) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return beta, r_squared, freqs, psd


def compute_spectral_slope_fooof(data, fs, freq_range=(1, 40)):
    """
    Compute aperiodic slope using FOOOF (separates periodic from aperiodic).
    
    This method is more robust as it removes oscillatory peaks before fitting.
    
    Returns
    -------
    beta : float
        Aperiodic exponent (spectral slope)
    offset : float
        Aperiodic offset
    r_squared : float
        Model fit quality
    """
    if not FOOOF_AVAILABLE:
        raise ImportError("FOOOF not installed. Use compute_spectral_slope_welch instead.")
    
    # Compute PSD
    freqs, psd = signal.welch(data, fs=fs, nperseg=int(fs*2))
    
    # Fit FOOOF model
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1)
    fm.fit(freqs, psd, freq_range)
    
    # Extract aperiodic parameters
    # FOOOF returns [offset, exponent] for 'fixed' mode
    beta = fm.aperiodic_params_[1]  # Exponent (already positive for decay)
    offset = fm.aperiodic_params_[0]
    r_squared = fm.r_squared_
    
    return beta, offset, r_squared


def beta_to_rtm_alpha(beta):
    """
    Convert spectral slope β to RTM transport exponent α.
    
    Theoretical basis:
    - For 1/f noise with β ≈ 1: α_RTM ≈ 2.0 (diffusive)
    - For white noise β ≈ 0: α_RTM → ∞ (undefined, random)
    - For Brownian β ≈ 2: α_RTM ≈ 1.0 (ballistic/critical)
    
    The relationship α = 2/β captures the inverse scaling between
    spectral steepness and temporal integration efficiency.
    """
    if beta <= 0.1:
        return np.nan  # Undefined for flat/positive spectra
    return 2.0 / beta


def compute_lempel_ziv_complexity(data, threshold='median'):
    """
    Compute Lempel-Ziv complexity (for comparison with Entropic Brain).
    
    LZ complexity measures the "randomness" of a binary sequence.
    Higher LZ = more complex/entropic.
    """
    # Binarize signal
    if threshold == 'median':
        binary = (data > np.median(data)).astype(int)
    elif threshold == 'mean':
        binary = (data > np.mean(data)).astype(int)
    else:
        binary = (data > threshold).astype(int)
    
    # Convert to string for LZ algorithm
    s = ''.join(map(str, binary))
    n = len(s)
    
    # LZ76 algorithm
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if s[i + k - 1] != s[l + k - 1]:
            if k > l - i:
                c += 1
                i += k
                k = 1
                l = i + 1
            else:
                k = 1
                l += 1
        else:
            k += 1
            if l + k > n:
                c += 1
                break
            if i + k > l:
                l += 1
                k = 1
    
    # Normalize by theoretical maximum
    lz_norm = c / (n / np.log2(n)) if n > 0 else 0
    return lz_norm


def compute_sample_entropy(data, m=2, r=None):
    """
    Compute Sample Entropy (another complexity measure for comparison).
    
    Parameters
    ----------
    m : int
        Embedding dimension
    r : float
        Tolerance (default: 0.2 * std)
    """
    if r is None:
        r = 0.2 * np.std(data)
    
    N = len(data)
    
    def _count_matches(template_length):
        templates = np.array([data[i:i+template_length] for i in range(N - template_length)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count
    
    A = _count_matches(m + 1)
    B = _count_matches(m)
    
    if A == 0 or B == 0:
        return np.nan
    
    return -np.log(A / B)


# =============================================================================
# DATA LOADING (PsiConnect specific)
# =============================================================================

def load_psiconnect_eeg(subject_id, session, condition, config=Config):
    """
    Load EEG data from PsiConnect dataset.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'sub-PC001')
    session : str
        'pre' or 'post' psilocybin
    condition : str
        'rest', 'meditation', 'music', 'movie'
        
    Returns
    -------
    raw : mne.io.Raw or np.ndarray
        EEG data
    """
    if not MNE_AVAILABLE:
        raise ImportError("MNE-Python required for loading EEG data")
    
    # Construct file path according to BIDS structure
    # Adjust this based on actual PsiConnect file naming convention
    session_map = {'pre': 'ses-01', 'post': 'ses-02'}
    task_map = {
        'rest': 'task-rest',
        'meditation': 'task-meditation', 
        'music': 'task-music',
        'movie': 'task-movie'
    }
    
    eeg_dir = os.path.join(
        config.DATA_ROOT, 
        subject_id, 
        session_map[session], 
        'eeg'
    )
    
    # Find matching file
    filename_pattern = f"{subject_id}_{session_map[session]}_{task_map[condition]}_eeg"
    
    # Try different formats
    for ext in ['.set', '.vhdr', '.edf', '.bdf']:
        filepath = os.path.join(eeg_dir, filename_pattern + ext)
        if os.path.exists(filepath):
            if ext == '.set':
                raw = mne.io.read_raw_eeglab(filepath, preload=True)
            elif ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(filepath, preload=True)
            elif ext == '.edf':
                raw = mne.io.read_raw_edf(filepath, preload=True)
            elif ext == '.bdf':
                raw = mne.io.read_raw_bdf(filepath, preload=True)
            return raw
    
    raise FileNotFoundError(f"No EEG file found for {subject_id}, {session}, {condition}")


def get_psiconnect_subjects(config=Config):
    """
    Get list of available subjects in PsiConnect dataset.
    """
    subjects = []
    if os.path.exists(config.DATA_ROOT):
        for item in os.listdir(config.DATA_ROOT):
            if item.startswith('sub-'):
                subjects.append(item)
    return sorted(subjects)


def load_psiconnect_behavioral(config=Config):
    """
    Load behavioral/subjective ratings from PsiConnect.
    
    Expected columns:
    - subject_id
    - session (pre/post)
    - MEQ30 (mystical experience)
    - ASC (altered states of consciousness)
    - intensity_rating
    - etc.
    """
    # Try common locations
    possible_paths = [
        os.path.join(config.DATA_ROOT, 'participants.tsv'),
        os.path.join(config.DATA_ROOT, 'phenotype', 'behavioral.tsv'),
        os.path.join(config.DATA_ROOT, 'derivatives', 'behavioral_summary.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if path.endswith('.tsv'):
                return pd.read_csv(path, sep='\t')
            else:
                return pd.read_csv(path)
    
    print("WARNING: Behavioral data not found. Returning empty DataFrame.")
    return pd.DataFrame()


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def analyze_single_recording(eeg_data, fs, config=Config, use_fooof=False):
    """
    Compute RTM metrics for a single EEG recording.
    
    Returns
    -------
    results : dict
        Contains: beta, alpha_rtm, lz_complexity, sample_entropy, r_squared
    """
    # Ensure 1D
    if eeg_data.ndim > 1:
        # Average across channels for global metric
        eeg_data = np.mean(eeg_data, axis=0)
    
    # Compute spectral slope
    if use_fooof and FOOOF_AVAILABLE:
        beta, offset, r_sq = compute_spectral_slope_fooof(
            eeg_data, fs, config.APERIODIC_RANGE
        )
    else:
        beta, r_sq, _, _ = compute_spectral_slope_welch(
            eeg_data, fs, config.APERIODIC_RANGE
        )
        offset = None
    
    # Convert to RTM alpha
    alpha_rtm = beta_to_rtm_alpha(beta)
    
    # Compute complexity metrics (for comparison with Entropic Brain)
    lz = compute_lempel_ziv_complexity(eeg_data)
    
    # Sample entropy (slower, optional)
    # se = compute_sample_entropy(eeg_data)
    se = np.nan  # Skip for speed
    
    return {
        'beta': beta,
        'alpha_rtm': alpha_rtm,
        'lz_complexity': lz,
        'sample_entropy': se,
        'r_squared': r_sq,
        'offset': offset
    }


def analyze_sliding_window(eeg_data, fs, config=Config):
    """
    Compute RTM metrics in sliding windows for temporal dynamics.
    
    Returns
    -------
    time_vector : array
        Center time of each window
    beta_trace : array
        Spectral slope over time
    alpha_trace : array
        RTM alpha over time
    """
    n_samples = len(eeg_data)
    window_samples = int(config.WINDOW_SIZE * fs)
    step_samples = int(config.WINDOW_STEP * fs)
    
    time_vector = []
    beta_trace = []
    alpha_trace = []
    
    start = 0
    while start + window_samples <= n_samples:
        segment = eeg_data[start:start + window_samples]
        center_time = (start + window_samples / 2) / fs
        
        beta, r_sq, _, _ = compute_spectral_slope_welch(
            segment, fs, config.APERIODIC_RANGE
        )
        alpha = beta_to_rtm_alpha(beta)
        
        time_vector.append(center_time)
        beta_trace.append(beta)
        alpha_trace.append(alpha)
        
        start += step_samples
    
    return np.array(time_vector), np.array(beta_trace), np.array(alpha_trace)


def run_full_analysis(config=Config):
    """
    Run complete RTM analysis on PsiConnect dataset.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    subjects = get_psiconnect_subjects(config)
    
    if len(subjects) == 0:
        print(f"No subjects found in {config.DATA_ROOT}")
        print("Please download PsiConnect from: https://openneuro.org/datasets/ds006110")
        return None
    
    print(f"Found {len(subjects)} subjects")
    
    # Conditions to analyze
    sessions = ['pre', 'post']
    conditions = ['rest', 'meditation', 'music']
    
    results = []
    
    for subject in subjects:
        print(f"\nProcessing {subject}...")
        
        for session in sessions:
            for condition in conditions:
                try:
                    # Load EEG
                    raw = load_psiconnect_eeg(subject, session, condition, config)
                    fs = raw.info['sfreq']
                    
                    # Get data (all channels, full recording)
                    data = raw.get_data()
                    
                    # Global analysis (average across channels)
                    global_data = np.mean(data, axis=0)
                    metrics = analyze_single_recording(global_data, fs, config)
                    
                    # Temporal dynamics
                    time_vec, beta_trace, alpha_trace = analyze_sliding_window(
                        global_data, fs, config
                    )
                    
                    # Store results
                    result = {
                        'subject': subject,
                        'session': session,
                        'condition': condition,
                        'beta_mean': metrics['beta'],
                        'beta_std': np.std(beta_trace),
                        'alpha_rtm_mean': metrics['alpha_rtm'],
                        'alpha_rtm_std': np.std(alpha_trace),
                        'lz_complexity': metrics['lz_complexity'],
                        'r_squared': metrics['r_squared'],
                        'duration_sec': len(global_data) / fs
                    }
                    results.append(result)
                    
                    print(f"  {session}/{condition}: β={metrics['beta']:.3f}, "
                          f"α_RTM={metrics['alpha_rtm']:.3f}")
                    
                except Exception as e:
                    print(f"  ERROR {session}/{condition}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(config.OUTPUT_DIR, 'rtm_psiconnect_results.csv'), index=False)
    
    return df


# =============================================================================
# STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# =============================================================================

def test_hypotheses(df, config=Config):
    """
    Test RTM hypotheses against PsiConnect data.
    """
    results = {}
    
    # H1: Psilocybin does NOT collapse α toward 0.5
    print("\n" + "="*60)
    print("H1: Psilocybin preserves topological coherence (α ≈ 1.0)")
    print("="*60)
    
    pre_alpha = df[df['session'] == 'pre']['alpha_rtm_mean'].dropna()
    post_alpha = df[df['session'] == 'post']['alpha_rtm_mean'].dropna()
    
    print(f"Pre-psilocybin α:  {pre_alpha.mean():.3f} ± {pre_alpha.std():.3f}")
    print(f"Post-psilocybin α: {post_alpha.mean():.3f} ± {post_alpha.std():.3f}")
    
    # Test if post-psilocybin α is significantly different from 0.5 (unconscious)
    t_stat, p_val = ttest_rel(post_alpha, [0.5] * len(post_alpha))
    print(f"Test α ≠ 0.5: t={t_stat:.3f}, p={p_val:.2e}")
    
    results['H1'] = {
        'pre_alpha_mean': pre_alpha.mean(),
        'post_alpha_mean': post_alpha.mean(),
        'p_vs_unconscious': p_val,
        'supported': post_alpha.mean() > 0.7  # Above transition zone
    }
    
    # H2: Pre vs Post comparison
    print("\n" + "="*60)
    print("H2: Pre vs Post psilocybin α comparison")
    print("="*60)
    
    # Paired t-test (within-subject)
    paired_data = df.pivot_table(
        index='subject', 
        columns='session', 
        values='alpha_rtm_mean'
    ).dropna()
    
    if len(paired_data) > 0:
        t_stat, p_val = ttest_rel(paired_data['pre'], paired_data['post'])
        effect_size = (paired_data['post'].mean() - paired_data['pre'].mean()) / paired_data['pre'].std()
        
        print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"Effect size (Cohen's d): {effect_size:.3f}")
        
        results['H2'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'direction': 'increase' if effect_size > 0 else 'decrease'
        }
    
    # H3: Condition differences
    print("\n" + "="*60)
    print("H3: α by condition (rest vs meditation vs music)")
    print("="*60)
    
    for condition in ['rest', 'meditation', 'music']:
        cond_data = df[(df['condition'] == condition) & (df['session'] == 'post')]
        if len(cond_data) > 0:
            print(f"{condition}: α = {cond_data['alpha_rtm_mean'].mean():.3f} ± "
                  f"{cond_data['alpha_rtm_mean'].std():.3f}")
    
    # H4: Correlation with complexity (RTM vs Entropic Brain)
    print("\n" + "="*60)
    print("H4: RTM α vs LZ Complexity (testing Entropic Brain)")
    print("="*60)
    
    valid_data = df[['alpha_rtm_mean', 'lz_complexity']].dropna()
    if len(valid_data) > 2:
        r, p = spearmanr(valid_data['alpha_rtm_mean'], valid_data['lz_complexity'])
        print(f"Spearman correlation: r={r:.3f}, p={p:.4f}")
        
        results['H4'] = {
            'correlation': r,
            'p_value': p,
            'interpretation': 'RTM and entropy are ' + 
                             ('positively' if r > 0 else 'negatively') + ' correlated'
        }
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_rtm_results(df, config=Config):
    """
    Generate publication-quality figures for RTM analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RTM Analysis of PsiConnect Psilocybin Dataset', fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {'pre': '#3b82f6', 'post': '#ef4444'}
    
    # Panel A: Pre vs Post α distribution
    ax = axes[0, 0]
    for session in ['pre', 'post']:
        data = df[df['session'] == session]['alpha_rtm_mean'].dropna()
        ax.hist(data, bins=20, alpha=0.6, label=session.capitalize(), color=colors[session])
    ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='α=1.0 (Critical)')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='α=0.5 (Unconscious)')
    ax.set_xlabel('RTM α')
    ax.set_ylabel('Count')
    ax.set_title('A) α Distribution: Pre vs Post Psilocybin')
    ax.legend()
    
    # Panel B: Pre vs Post paired comparison
    ax = axes[0, 1]
    paired = df.pivot_table(index='subject', columns='session', values='alpha_rtm_mean').dropna()
    if len(paired) > 0:
        for i in range(len(paired)):
            ax.plot([0, 1], [paired.iloc[i]['pre'], paired.iloc[i]['post']], 
                   'o-', color='gray', alpha=0.3)
        ax.errorbar([0], [paired['pre'].mean()], yerr=[paired['pre'].std()], 
                   fmt='o', color=colors['pre'], markersize=12, capsize=5, label='Pre')
        ax.errorbar([1], [paired['post'].mean()], yerr=[paired['post'].std()], 
                   fmt='o', color=colors['post'], markersize=12, capsize=5, label='Post')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel('RTM α')
    ax.set_title('B) Paired Pre-Post Comparison')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    # Panel C: α by condition
    ax = axes[0, 2]
    conditions = ['rest', 'meditation', 'music']
    for i, condition in enumerate(conditions):
        post_data = df[(df['session'] == 'post') & (df['condition'] == condition)]['alpha_rtm_mean']
        if len(post_data) > 0:
            bp = ax.boxplot([post_data.dropna()], positions=[i], widths=0.6)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel('RTM α')
    ax.set_title('C) Post-Psilocybin α by Condition')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    # Panel D: β (spectral slope) distribution
    ax = axes[1, 0]
    for session in ['pre', 'post']:
        data = df[df['session'] == session]['beta_mean'].dropna()
        ax.hist(data, bins=20, alpha=0.6, label=session.capitalize(), color=colors[session])
    ax.set_xlabel('Spectral Slope β')
    ax.set_ylabel('Count')
    ax.set_title('D) Spectral Slope Distribution')
    ax.legend()
    
    # Panel E: RTM α vs LZ Complexity
    ax = axes[1, 1]
    valid = df[['alpha_rtm_mean', 'lz_complexity']].dropna()
    if len(valid) > 0:
        colors_scatter = [colors['pre'] if s == 'pre' else colors['post'] 
                         for s in df.loc[valid.index, 'session']]
        ax.scatter(valid['lz_complexity'], valid['alpha_rtm_mean'], 
                  c=colors_scatter, alpha=0.6, edgecolors='white')
        # Add correlation line
        if len(valid) > 2:
            z = np.polyfit(valid['lz_complexity'], valid['alpha_rtm_mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['lz_complexity'].min(), valid['lz_complexity'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.5)
    ax.set_xlabel('LZ Complexity (Entropy)')
    ax.set_ylabel('RTM α')
    ax.set_title('E) RTM α vs Entropic Complexity')
    
    # Panel F: RTM transport class interpretation
    ax = axes[1, 2]
    # Create theoretical reference
    alpha_values = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0])
    labels = ['Inverse\n(α<0.5)', 'Diffusive\n(α=0.5)', 'Sub-critical\n(α=0.7)', 
              'Critical\n(α=1.0)', 'Super-critical\n(α=1.3)', 'Super-ballistic\n(α=1.5)',
              'Cooperative\n(α=2.0)']
    colors_bar = ['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#3b82f6', '#6366f1', '#8b5cf6']
    
    ax.barh(range(len(alpha_values)), alpha_values, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('RTM α')
    ax.set_title('F) RTM Transport Classes')
    
    # Add observed means
    if len(df) > 0:
        pre_mean = df[df['session'] == 'pre']['alpha_rtm_mean'].mean()
        post_mean = df[df['session'] == 'post']['alpha_rtm_mean'].mean()
        ax.axvline(x=pre_mean, color='blue', linestyle='-', linewidth=2, label=f'Pre ({pre_mean:.2f})')
        ax.axvline(x=post_mean, color='red', linestyle='-', linewidth=2, label=f'Post ({post_mean:.2f})')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(config.OUTPUT_DIR, 'rtm_psiconnect_analysis.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(config.OUTPUT_DIR, 'rtm_psiconnect_analysis.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("RTM ANALYSIS OF PSICONNECT PSILOCYBIN DATASET")
    print("="*70)
    print("\nTheoretical Framework: Multiscale Temporal Relativity (RTM)")
    print("Hypothesis: Consciousness = Topological Coherence (α ≈ 1.0)")
    print("Alternative: Entropic Brain (Consciousness = High Entropy)")
    print("\n")
    
    config = Config()
    
    # Check if data exists
    if not os.path.exists(config.DATA_ROOT):
        print(f"Dataset not found at: {config.DATA_ROOT}")
        print("\nTo download PsiConnect dataset:")
        print("1. Visit: https://openneuro.org/datasets/ds006110")
        print("2. Download using: aws s3 sync --no-sign-request s3://openneuro.org/ds006110 ds006110/")
        print("3. Or use openneuro-py: pip install openneuro-py && openneuro download ds006110")
        print("\nRunning synthetic data analysis instead...")
        return None
    
    # Run full analysis
    df = run_full_analysis(config)
    
    if df is not None and len(df) > 0:
        # Test hypotheses
        hypothesis_results = test_hypotheses(df, config)
        
        # Generate figures
        plot_rtm_results(df, config)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Subjects analyzed: {df['subject'].nunique()}")
        print(f"Total recordings: {len(df)}")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        
        return df, hypothesis_results
    
    return None


if __name__ == "__main__":
    results = main()
