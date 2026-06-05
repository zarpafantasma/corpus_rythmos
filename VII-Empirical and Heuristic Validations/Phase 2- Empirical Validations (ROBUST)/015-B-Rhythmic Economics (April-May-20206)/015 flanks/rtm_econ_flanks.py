#!/usr/bin/env python3
"""
RTM ECONOMICS FLANKING CAMPAIGN
=================================
Five flanks using available data from Doc 015 package.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import welch
from numpy.linalg import lstsq
import json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Column names for Binance 1-min data
cols = ['timestamp','open','high','low','close','volume','close_time',
        'quote_vol','trades','taker_buy_base','taker_buy_quote','ignore']

print("=" * 70)
print("RTM ECONOMICS — FLANKING CAMPAIGN")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════
months = {}
for label, path in [
    ('2020-03_COVID', '/home/claude/015/RTM Forensic Report-The Liquidity Crisis of March 2020/BTCUSDT-1m-2020-03.csv'),
    ('2022-11_FTX', '/home/claude/015/RTM Forensic Report-The FTX Solvency Collapse November 2022/BTCUSDT-1m-2022-11.csv'),
    ('2025-10_Anomaly', '/home/claude/015/RTM Forensic Report-The Binance Glitch Anomaly October 2025/BTCUSDT-1m-2025-10.csv'),
    ('2023-09_Control', '/home/claude/015/Control Group Analysis September 2023/BTCUSDT-1m-2023-09.csv')
]:
    try:
        df = pd.read_csv(path, header=None, names=cols)
        for c in ['open','high','low','close','volume','trades']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['volatility'] = df['high'] - df['low']
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        months[label] = df
        print(f"  {label}: {len(df)} candles, ${df['close'].iloc[0]:.0f} -> ${df['close'].iloc[-1]:.0f}")
    except:
        print(f"  {label}: FAILED TO LOAD")

crash_data = pd.read_csv("/home/claude/015/ROBUST-RTM_Financial_Crash_Analysis_Reproducible/crash_alpha_analysis.csv")
print(f"  Crash events: {len(crash_data)}")

# ═══════════════════════════════════════════════════════
# HELPER: Rolling DFA-like α
# ═══════════════════════════════════════════════════════
def rolling_alpha(df, window=60):
    """Compute rolling volatility-volume α"""
    vol = df['volatility'].values
    volume = df['volume'].values
    alphas = []
    for i in range(window, len(df)):
        v = vol[i-window:i]
        q = volume[i-window:i]
        mask = (v > 0) & (q > 0)
        if mask.sum() < 20:
            alphas.append(np.nan)
            continue
        lv = np.log(v[mask])
        lq = np.log(q[mask])
        if np.std(lq) < 0.1:
            alphas.append(np.nan)
            continue
        s, _, r, p, _ = stats.linregress(lq, lv)
        alphas.append(s)
    return np.array(alphas)

# ═══════════════════════════════════════════════════════
# FLANK 1: OUT-OF-SAMPLE TEMPORAL PREDICTION
# Train threshold on pre-2022 crashes, test on FTX + 2025
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 1: OUT-OF-SAMPLE CRASH PREDICTION")
print("Train on pre-2022 crashes, test on FTX and 2025")
print(f"{'='*70}")

# Split crash events by date
crash_data['year'] = crash_data['Event'].apply(lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 2020)
# Fix: extract year more robustly
for idx, row in crash_data.iterrows():
    parts = row['Event'].split()
    for p in parts:
        if p.isdigit() and len(p) == 4:
            crash_data.loc[idx, 'year'] = int(p)

train = crash_data[crash_data['year'] < 2022]
test = crash_data[crash_data['year'] >= 2022]

print(f"\n  Training set (pre-2022): {len(train)} events")
print(f"  Test set (2022+): {len(test)} events")

# Train: optimal α-drop threshold
train_sig = train[train['Significant_Drop'] == True]
if len(train_sig) > 0:
    # Threshold = mean α-drop of significant events
    threshold_drop = train_sig['Alpha_Drop'].mean()
    threshold_alpha = train_sig['Immediate_Alpha'].mean()
    
    print(f"  Trained threshold: α-drop < {threshold_drop:.3f}")
    print(f"  Trained α level: α < {threshold_alpha:.3f}")
    
    # Test on test set
    print(f"\n  TEST SET PREDICTIONS:")
    for _, row in test.iterrows():
        predicted_crash = row['Alpha_Drop'] < threshold_drop
        actual_sig = row['Significant_Drop']
        correct = predicted_crash == actual_sig
        print(f"    {row['Event']:30s}: drop={row['Alpha_Drop']:+.3f}, "
              f"predicted={'CRASH' if predicted_crash else 'NORMAL':6s}, "
              f"actual={'CRASH' if actual_sig else 'NORMAL':6s}, "
              f"{'✓' if correct else '✗'}")
    
    # Overall accuracy
    test_preds = test['Alpha_Drop'] < threshold_drop
    test_actual = test['Significant_Drop']
    accuracy = (test_preds == test_actual).mean()
    print(f"\n  Out-of-sample accuracy: {accuracy*100:.0f}%")
    
    # Also test on the BTC months directly
    print(f"\n  DIRECT BTC MONTH PREDICTION:")
    for label, df in months.items():
        alpha = rolling_alpha(df, window=120)
        alpha_clean = alpha[~np.isnan(alpha)]
        mean_alpha = np.mean(alpha_clean) if len(alpha_clean) > 0 else np.nan
        min_alpha = np.min(alpha_clean) if len(alpha_clean) > 0 else np.nan
        is_crash = 'Control' not in label
        predicted = min_alpha < threshold_alpha if not np.isnan(min_alpha) else False
        print(f"    {label:20s}: mean α={mean_alpha:.3f}, min α={min_alpha:.3f}, "
              f"predicted={'CRASH' if predicted else 'NORMAL':6s}, "
              f"actual={'CRASH' if is_crash else 'NORMAL':6s}, "
              f"{'✓' if predicted == is_crash else '✗'}")

# ═══════════════════════════════════════════════════════
# FLANK 2: VOLUME-VOLATILITY SHAPE CONSPIRACY
# Analog of SPARC baryon-halo conspiracy
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 2: VOLUME-VOLATILITY SHAPE CONSPIRACY")
print("Does the SHAPE of volume track the SHAPE of volatility?")
print("Does this coupling break before crashes?")
print(f"{'='*70}")

for label, df in months.items():
    print(f"\n  --- {label} ---")
    
    # Hourly aggregation for cleaner signal
    df['hour'] = np.arange(len(df)) // 60
    hourly = df.groupby('hour').agg({
        'volume': 'sum', 'volatility': 'mean',
        'close': 'last', 'trades': 'sum'
    }).reset_index()
    
    vol_norm = hourly['volume'].values
    vola_norm = hourly['volatility'].values
    
    # Remove zeros
    mask = (vol_norm > 0) & (vola_norm > 0)
    vol_norm = vol_norm[mask]
    vola_norm = vola_norm[mask]
    
    if len(vol_norm) < 100: continue
    
    # Normalize
    vol_n = (vol_norm - vol_norm.mean()) / (vol_norm.std() + 1e-10)
    vola_n = (vola_norm - vola_norm.mean()) / (vola_norm.std() + 1e-10)
    
    # Global shape correlation
    r_global, p_global = stats.pearsonr(vol_n, vola_n)
    
    # Rolling conspiracy (24-hour window)
    window = 24
    conspiracies = []
    for i in range(window, len(vol_n)):
        v = vol_n[i-window:i]
        va = vola_n[i-window:i]
        r, _ = stats.pearsonr(v, va)
        conspiracies.append(r)
    conspiracies = np.array(conspiracies)
    
    print(f"    Global shape r = {r_global:+.3f} (p={p_global:.2e})")
    print(f"    Rolling conspiracy: mean = {np.nanmean(conspiracies):+.3f} ± {np.nanstd(conspiracies):.3f}")
    
    # If crash month: does conspiracy drop BEFORE crash date?
    if 'COVID' in label:
        # March 12
        crash_hour = 11 * 24
        pre = conspiracies[:max(1,crash_hour-24)]
        during = conspiracies[crash_hour-24:crash_hour+48] if crash_hour+48 < len(conspiracies) else conspiracies[crash_hour-24:]
        pre_clean = pre[~np.isnan(pre)]
        dur_clean = during[~np.isnan(during)]
        if len(pre_clean) > 5 and len(dur_clean) > 5:
            d_cons = (pre_clean.mean() - dur_clean.mean()) / np.sqrt((pre_clean.var(ddof=1) + dur_clean.var(ddof=1))/2)
            t_cons, p_cons = stats.ttest_ind(pre_clean, dur_clean)
            print(f"    Pre-crash vs during: d = {d_cons:+.3f}, p = {p_cons:.4f}")
    
    elif 'FTX' in label:
        crash_hour = 6 * 24  # Nov 7
        pre = conspiracies[:max(1,crash_hour-24)]
        during = conspiracies[crash_hour-24:crash_hour+120] if crash_hour+120 < len(conspiracies) else conspiracies[crash_hour-24:]
        pre_clean = pre[~np.isnan(pre)]
        dur_clean = during[~np.isnan(during)]
        if len(pre_clean) > 5 and len(dur_clean) > 5:
            d_cons = (pre_clean.mean() - dur_clean.mean()) / np.sqrt((pre_clean.var(ddof=1) + dur_clean.var(ddof=1))/2)
            t_cons, p_cons = stats.ttest_ind(pre_clean, dur_clean)
            print(f"    Pre-crash vs during: d = {d_cons:+.3f}, p = {p_cons:.4f}")

# Cross-month comparison
print(f"\n  CROSS-MONTH COMPARISON:")
month_conspiracies = {}
for label, df in months.items():
    df['hour'] = np.arange(len(df)) // 60
    hourly = df.groupby('hour').agg({'volume': 'sum', 'volatility': 'mean'}).reset_index()
    mask = (hourly['volume'] > 0) & (hourly['volatility'] > 0)
    h = hourly[mask]
    if len(h) < 50: continue
    r, p = stats.pearsonr(
        (h['volume'] - h['volume'].mean()) / (h['volume'].std() + 1e-10),
        (h['volatility'] - h['volatility'].mean()) / (h['volatility'].std() + 1e-10)
    )
    month_conspiracies[label] = r
    print(f"    {label:20s}: conspiracy r = {r:+.3f}")

# ═══════════════════════════════════════════════════════
# FLANK 3: MULTI-SCALE COHERENCE
# RTM predicts fractal coherence breaks before crashes
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 3: MULTI-SCALE COHERENCE")
print("Is α consistent across time scales? Does coherence break pre-crash?")
print(f"{'='*70}")

for label, df in months.items():
    print(f"\n  --- {label} ---")
    
    scales = [1, 5, 15, 60]  # minutes
    scale_alphas = {}
    
    for scale in scales:
        # Aggregate to this scale
        df['bin'] = np.arange(len(df)) // scale
        agg = df.groupby('bin').agg({
            'volume': 'sum', 'volatility': 'mean', 'close': 'last'
        }).reset_index()
        
        agg['vol_log'] = np.log(agg['volume'].clip(lower=1e-10))
        agg['vola_log'] = np.log(agg['volatility'].clip(lower=1e-10))
        
        mask = np.isfinite(agg['vol_log']) & np.isfinite(agg['vola_log']) & (agg['volume'] > 0) & (agg['volatility'] > 0)
        if mask.sum() < 50: continue
        
        s, _, r, p, _ = stats.linregress(agg['vol_log'][mask], agg['vola_log'][mask])
        scale_alphas[scale] = s
    
    if len(scale_alphas) >= 3:
        alpha_values = list(scale_alphas.values())
        alpha_std = np.std(alpha_values)
        alpha_range = max(alpha_values) - min(alpha_values)
        
        print(f"    Scale alphas: " + ", ".join([f"{s}min={a:.3f}" for s, a in scale_alphas.items()]))
        print(f"    Cross-scale σ = {alpha_std:.4f}, range = {alpha_range:.4f}")
        print(f"    {'COHERENT' if alpha_std < 0.15 else 'INCOHERENT'} (threshold σ < 0.15)")

# ═══════════════════════════════════════════════════════
# FLANK 4: CRASH-RECOVERY ASYMMETRY
# RTM predicts crash = fast phase transition, recovery = slow rebuilding
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 4: CRASH-RECOVERY ASYMMETRY")
print("Does α fall faster than it recovers?")
print(f"{'='*70}")

# Use the crash_alpha_analysis data
sig_events = crash_data[crash_data['Significant_Drop'] == True].copy()
print(f"\n  Significant events: {len(sig_events)}")

# For each event, compute fall rate and recovery rate
if 'Recovery_Hours' in sig_events.columns:
    print(f"  Using Recovery_Hours from data")
elif 'Lead_Time_Hours' in sig_events.columns:
    print(f"  Using Lead_Time_Hours as fall duration")

fall_rates = []
for _, row in sig_events.iterrows():
    drop = abs(row['Alpha_Drop'])
    lead = row['Lead_Time_Hours']
    if lead > 0:
        fall_rate = drop / lead  # α units per hour
        fall_rates.append({'event': row['Event'], 'fall_rate': fall_rate, 'drop': drop, 'lead': lead})

fdf = pd.DataFrame(fall_rates)
if len(fdf) > 0:
    print(f"\n  Fall rates (α-drop per hour):")
    for _, r in fdf.iterrows():
        print(f"    {r['event']:30s}: Δα={r['drop']:.3f} in {r['lead']:.0f}h → rate={r['fall_rate']:.5f}/h")
    print(f"\n  Mean fall rate: {fdf['fall_rate'].mean():.5f} ± {fdf['fall_rate'].std():.5f}")

# Now compute recovery from the BTC months directly
print(f"\n  DIRECT RECOVERY ANALYSIS FROM BTC DATA:")
for label, df in months.items():
    if 'Control' in label: continue
    
    alpha = rolling_alpha(df, window=120)
    alpha_clean = pd.Series(alpha).interpolate().values
    
    # Find the minimum
    if len(alpha_clean) < 100: continue
    min_idx = np.nanargmin(alpha_clean)
    min_val = alpha_clean[min_idx]
    
    # Pre-crash baseline (first 20%)
    baseline_end = len(alpha_clean) // 5
    baseline = np.nanmean(alpha_clean[:baseline_end])
    
    # Fall: baseline → min
    fall_duration = min_idx - baseline_end if min_idx > baseline_end else min_idx
    fall_magnitude = baseline - min_val
    
    # Recovery: min → back to baseline (or end of data)
    recovery_idx = len(alpha_clean) - 1
    for i in range(min_idx, len(alpha_clean)):
        if alpha_clean[i] >= baseline * 0.9:
            recovery_idx = i
            break
    recovery_duration = recovery_idx - min_idx
    recovery_magnitude = alpha_clean[recovery_idx] - min_val
    
    if fall_duration > 0 and recovery_duration > 0:
        fall_rate = fall_magnitude / fall_duration  # per candle
        recovery_rate = recovery_magnitude / recovery_duration
        asymmetry = fall_rate / (recovery_rate + 1e-10)
        
        print(f"    {label:20s}: fall={fall_duration} candles, recovery={recovery_duration} candles")
        print(f"      Fall rate: {fall_rate:.6f}/candle, Recovery rate: {recovery_rate:.6f}/candle")
        print(f"      Asymmetry ratio: {asymmetry:.2f}x {'(fall faster ✓)' if asymmetry > 1 else '(recovery faster)'}")

# ═══════════════════════════════════════════════════════
# FLANK 5: TRADE COUNT AS INDEPENDENT STRUCTURAL METRIC
# Volume can be gamed. Trade COUNT is harder to fake.
# Does trade count α track volume α? Or diverge before crashes?
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 5: TRADE COUNT — THE UNFAKEABLE METRIC")
print("Volume can be wash-traded. Trade count is harder to fake.")
print("Does trade-count structure diverge from volume before crashes?")
print(f"{'='*70}")

def rolling_alpha_trades(df, window=60):
    """α using trade count instead of volume"""
    vol = df['volatility'].values
    trades = df['trades'].values
    alphas = []
    for i in range(window, len(df)):
        v = vol[i-window:i]
        t = trades[i-window:i]
        mask = (v > 0) & (t > 0)
        if mask.sum() < 20:
            alphas.append(np.nan)
            continue
        lv = np.log(v[mask])
        lt = np.log(t[mask].astype(float))
        if np.std(lt) < 0.1:
            alphas.append(np.nan)
            continue
        s, _, r, p, _ = stats.linregress(lt, lv)
        alphas.append(s)
    return np.array(alphas)

for label, df in months.items():
    print(f"\n  --- {label} ---")
    
    alpha_vol = rolling_alpha(df, window=120)
    alpha_trade = rolling_alpha_trades(df, window=120)
    
    # Clean
    mask = (~np.isnan(alpha_vol)) & (~np.isnan(alpha_trade))
    av = alpha_vol[mask]
    at = alpha_trade[mask]
    
    if len(av) < 100: continue
    
    # Global correlation between volume-α and trade-α
    r_vt, p_vt = stats.pearsonr(av, at)
    
    # Divergence metric: rolling correlation
    window_d = min(1440, len(av) // 4)  # 1 day or quarter of data
    divergences = []
    for i in range(window_d, len(av)):
        r_local, _ = stats.pearsonr(av[i-window_d:i], at[i-window_d:i])
        divergences.append(r_local)
    divergences = np.array(divergences)
    
    print(f"    Global vol-α vs trade-α: r = {r_vt:+.3f}")
    print(f"    Mean rolling coherence: {np.nanmean(divergences):+.3f} ± {np.nanstd(divergences):.3f}")
    
    # For crash months: does divergence happen before crash?
    if 'COVID' in label:
        crash_candle = 11 * 24 * 60 - 120  # March 12, adjusted for window
        crash_div_idx = max(0, crash_candle - window_d - 120)
        if crash_div_idx < len(divergences):
            pre_div = divergences[:crash_div_idx]
            crash_div = divergences[crash_div_idx:min(crash_div_idx + 2*1440, len(divergences))]
            pre_clean = pre_div[~np.isnan(pre_div)]
            crash_clean = crash_div[~np.isnan(crash_div)]
            if len(pre_clean) > 10 and len(crash_clean) > 10:
                d_div = (pre_clean.mean() - crash_clean.mean()) / np.sqrt((pre_clean.var(ddof=1) + crash_clean.var(ddof=1))/2)
                print(f"    Pre-crash coherence: {pre_clean.mean():.3f}")
                print(f"    During-crash coherence: {crash_clean.mean():.3f}")
                print(f"    d = {d_div:+.3f} {'(coherence drops ✓)' if d_div > 0.2 else '(no change)'}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("ECONOMICS FLANKING CAMPAIGN — SUMMARY")
print(f"{'='*70}")

results = {
    "document": "015-Economics_Flanking",
    "months_analyzed": list(months.keys()),
    "crash_events": len(crash_data)
}

with open('/home/claude/rtm_econ_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved.")
