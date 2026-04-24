"""
RTM-SEISMO INSTABILITY INDEX
==============================
Second-order metrics that capture alpha VOLATILITY patterns
rather than alpha LEVEL.

The core insight: the signal before major earthquakes is not
"alpha is low" — it's "alpha is UNSTABLE." The Sanriku M7.4
showed alpha oscillating wildly (3.0 → 0.1 → 3.0 → 0.4)
for months, while the MEAN alpha remained high.

This script tests four second-order metrics:
  1. σ_α  — Rolling standard deviation of alpha
  2. TF   — Transition Frequency (regime crossings per unit time)
  3. MDD  — Maximum Drawdown rate (steepest alpha collapse)
  4. II   — Composite Instability Index (weighted combination)

Each is backtested against the same M7+ events and compared
to the raw alpha threshold approach.

Usage:
  pip install pandas numpy requests
  python instability_index.py

Runtime: ~30-60 minutes
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

RTM_WINDOW = 30
SMOOTHING = 3
LOOKBACK_DAYS = 180
RADIUS_KM = 300
MIN_MAG = 2.0
API_DELAY = 2

# Instability metric windows
VOLATILITY_WINDOW = 30        # Events for rolling std of alpha
TRANSITION_WINDOW = 50        # Events for transition counting
DRAWDOWN_WINDOW = 20          # Events for max drawdown rate
TRANSITION_THRESHOLD = 1.0    # Alpha level that defines a "regime crossing"

# Assessment windows (days before event)
WARNING_WINDOWS = [7, 14, 30, 60, 90]

# Control period offset
CONTROL_OFFSET_DAYS = 365

OUTPUT_CSV = "rtm_seismo_instability_results.csv"
OUTPUT_SUMMARY = "rtm_seismo_instability_report.txt"


# ==========================================
# TARGET EVENTS (same as backtest)
# ==========================================

TARGET_EVENTS = [
    ("Sanriku, Japan",             "2026-04-20", 39.5, 142.1, 7.4, "subduction"),
    ("Hokkaido, Japan",            "2025-12-14", 40.8, 142.8, 7.6, "subduction"),
    ("Chagos Archipelago",         "2025-07-13", -6.9,  72.1, 7.1, "intraplate"),
    ("Mariana Islands",            "2025-04-04", 17.9, 148.3, 7.0, "subduction"),
    ("Vanuatu",                    "2024-12-17", -17.7, 168.1, 7.3, "subduction"),
    ("Hualien, Taiwan",            "2024-04-02", 23.8, 121.6, 7.4, "subduction"),
    ("Noto, Japan",                "2024-01-01", 37.5, 137.3, 7.5, "crustal"),
    ("Cape Mendocino, US",         "2024-12-05", 40.4, -124.5, 7.0, "transform"),
    ("Mindanao, Philippines",      "2023-12-02", 9.1, 126.4, 7.6, "subduction"),
    ("Afghanistan",                "2023-10-07", 34.6,  62.0, 6.3, "crustal"),
    ("Marrakech, Morocco",         "2023-09-08", 31.1,  -8.4, 6.8, "crustal"),
    ("Kahramanmaras, Turkey I",    "2023-02-06", 37.2,  37.0, 7.8, "transform"),
    ("Kahramanmaras, Turkey II",   "2023-02-06", 38.0,  37.2, 7.5, "transform"),
    ("Mentawai, Indonesia",        "2022-11-18", -2.8, 100.0, 6.9, "subduction"),
    ("Luzon, Philippines",         "2022-07-27", 17.6, 120.7, 7.0, "subduction"),
    ("Mindanao II, Philippines",   "2022-03-08", 7.6, 127.0, 7.6, "subduction"),
    ("Fukushima, Japan",           "2022-03-16", 37.7, 141.6, 7.3, "subduction"),
    ("South Sandwich Islands",     "2021-08-12", -57.6, -25.1, 8.1, "subduction"),
    ("Chignik, Alaska",            "2021-07-29", 55.4, -157.8, 8.2, "subduction"),
    ("Haiti",                      "2021-08-14", 18.4, -73.5, 7.2, "transform"),
    ("Antofagasta, Chile",         "2021-06-17", -24.1, -68.4, 7.0, "subduction"),
    ("Sulawesi, Indonesia",        "2021-01-14", -2.2, 118.9, 6.2, "crustal"),
    ("Maduo, China",               "2021-05-21", 34.6,  98.3, 7.4, "crustal"),
    ("Samos, Greece/Turkey",       "2020-10-30", 37.9,  26.8, 7.0, "extensional"),
    ("Sand Point, Alaska",         "2020-07-22", 55.1, -158.6, 7.8, "subduction"),
    ("Oaxaca, Mexico",             "2020-06-23", 15.9, -96.0, 7.4, "subduction"),
    ("Molucca Sea, Indonesia",     "2020-01-06", 2.3, 126.9, 6.9, "subduction"),
    ("Petrinja, Croatia",          "2020-12-29", 45.4,  16.3, 6.4, "crustal"),
    ("Elazıg, Turkey",             "2020-01-24", 38.4,  39.1, 6.7, "transform"),
]


# ==========================================
# CORE ENGINE
# ==========================================

def fetch_usgs_data(lat, lon, radius_km, start_date, end_date, min_mag=MIN_MAG):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat, "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "orderby": "time", "limit": 10000,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        events = []
        for f in data.get("features", []):
            p = f["properties"]
            g = f["geometry"]["coordinates"]
            events.append({
                "time": pd.to_datetime(p["time"], unit="ms"),
                "mag": p.get("mag", 0),
                "depth": g[2] if len(g) > 2 else 0,
            })
        return pd.DataFrame(events).sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"    API Error: {e}")
        return None


def compute_alpha(df, window=RTM_WINDOW):
    """Core RTM alpha — identical to live app."""
    if df is None or len(df) < window + 5:
        return None
    df = df.copy()
    df['dt_hours'] = df['time'].diff().dt.total_seconds() / 3600.0
    df = df[(df['dt_hours'] > 0.001) & (df['mag'] > 0)].copy()
    if len(df) < window + 5:
        return None
    df['log_L'] = np.log(df['mag'])
    df['log_T'] = np.log(df['dt_hours'])
    cov = df['log_L'].rolling(window).cov(df['log_T'])
    var = df['log_L'].rolling(window).var()
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_alpha = np.abs(cov / var)
    raw_alpha = pd.Series(raw_alpha.values).replace([np.inf, -np.inf], np.nan)
    df['alpha'] = raw_alpha.rolling(SMOOTHING, min_periods=1).mean().values
    df = df.dropna(subset=['alpha']).reset_index(drop=True)
    df['alpha'] = df['alpha'].clip(0.01, 3.0)
    return df


# ==========================================
# INSTABILITY METRICS
# ==========================================

def compute_instability_metrics(alpha_df):
    """
    Compute four second-order metrics from the alpha time series:
    
    1. sigma_alpha: Rolling standard deviation of alpha
       High sigma = alpha is oscillating wildly = unstable
    
    2. transition_freq: Number of times alpha crosses the
       transition threshold per N events
       High TF = frequent regime changes = unstable
    
    3. max_drawdown_rate: Maximum drop in alpha over a short window
       (alpha_max - alpha_min) / alpha_max within the window
       High MDD = sudden collapses = unstable
    
    4. instability_index: Composite metric combining all three
       II = sigma_alpha_norm * TF_norm * (1 + MDD)
    """
    if alpha_df is None or len(alpha_df) < VOLATILITY_WINDOW + 10:
        return None

    df = alpha_df.copy()
    alphas = df['alpha'].values

    # --- 1. Rolling standard deviation ---
    df['sigma_alpha'] = df['alpha'].rolling(VOLATILITY_WINDOW, min_periods=10).std()

    # --- 2. Transition frequency ---
    # Count crossings of the transition threshold per window
    above = (alphas > TRANSITION_THRESHOLD).astype(int)
    crossings = np.abs(np.diff(above))
    crossings = np.concatenate([[0], crossings])  # pad to same length

    tf_series = pd.Series(crossings).rolling(TRANSITION_WINDOW, min_periods=10).sum()
    df['transition_freq'] = tf_series.values

    # --- 3. Maximum drawdown rate ---
    mdd_values = np.full(len(alphas), np.nan)
    for i in range(DRAWDOWN_WINDOW, len(alphas)):
        window_slice = alphas[i - DRAWDOWN_WINDOW:i + 1]
        running_max = np.maximum.accumulate(window_slice)
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdowns = (running_max - window_slice) / running_max
        mdd_values[i] = np.nanmax(drawdowns)
    df['max_drawdown'] = mdd_values

    # --- 4. Composite Instability Index ---
    # Normalize each component to 0-1 range using rolling percentile
    s = df['sigma_alpha']
    t = df['transition_freq']
    m = df['max_drawdown']

    # Use expanding max for normalization (avoids future leak)
    s_norm = s / s.expanding().max().replace(0, np.nan)
    t_norm = t / t.expanding().max().replace(0, np.nan)
    m_norm = m  # already 0-1

    # Composite: geometric-ish combination
    # High when ALL components are elevated
    df['instability_index'] = (
        0.4 * s_norm.fillna(0) +
        0.3 * t_norm.fillna(0) +
        0.3 * m_norm.fillna(0)
    )

    # Also compute a "spike detector" — did II exceed its own
    # rolling 90th percentile? This is the actionable signal.
    ii = df['instability_index']
    rolling_p90 = ii.expanding(min_periods=30).quantile(0.90)
    df['ii_spike'] = (ii > rolling_p90).astype(int)

    return df


# ==========================================
# ANALYSIS
# ==========================================

def analyze_period(df, event_date, is_control=False):
    """Extract instability metrics for a period."""
    if df is None or len(df) == 0:
        return None

    event_dt = pd.Timestamp(event_date)

    if is_control:
        # Use the whole period
        pre = df
    else:
        # Pre-event only
        pre = df[df['time'] < event_dt]

    if len(pre) < 20:
        return None

    results = {}
    results['n_readings'] = len(pre)

    # --- Raw alpha (for comparison) ---
    results['alpha_mean'] = float(pre['alpha'].mean())
    results['alpha_std'] = float(pre['alpha'].std())
    results['alpha_min'] = float(pre['alpha'].min())
    results['entered_critical_08'] = bool((pre['alpha'] < 0.8).any())
    results['entered_fracture_05'] = bool((pre['alpha'] < 0.5).any())
    results['frac_below_08'] = float((pre['alpha'] < 0.8).mean())

    # --- Sigma alpha (volatility) ---
    sa = pre['sigma_alpha'].dropna()
    if len(sa) > 5:
        results['sigma_alpha_mean'] = float(sa.mean())
        results['sigma_alpha_max'] = float(sa.max())
        results['sigma_alpha_p90'] = float(sa.quantile(0.9))
    else:
        results['sigma_alpha_mean'] = None
        results['sigma_alpha_max'] = None
        results['sigma_alpha_p90'] = None

    # --- Transition frequency ---
    tf = pre['transition_freq'].dropna()
    if len(tf) > 5:
        results['tf_mean'] = float(tf.mean())
        results['tf_max'] = float(tf.max())
        results['tf_p90'] = float(tf.quantile(0.9))
    else:
        results['tf_mean'] = None
        results['tf_max'] = None
        results['tf_p90'] = None

    # --- Max drawdown ---
    mdd = pre['max_drawdown'].dropna()
    if len(mdd) > 5:
        results['mdd_mean'] = float(mdd.mean())
        results['mdd_max'] = float(mdd.max())
        results['mdd_p90'] = float(mdd.quantile(0.9))
    else:
        results['mdd_mean'] = None
        results['mdd_max'] = None
        results['mdd_p90'] = None

    # --- Instability Index ---
    ii = pre['instability_index'].dropna()
    if len(ii) > 5:
        results['ii_mean'] = float(ii.mean())
        results['ii_max'] = float(ii.max())
        results['ii_p90'] = float(ii.quantile(0.9))

        # Spike detection — fraction of time II exceeded its own P90
        spikes = pre['ii_spike'].dropna()
        results['ii_spike_frac'] = float(spikes.mean()) if len(spikes) > 0 else None

        # Did II ever exceed 0.5? 0.7?
        results['ii_above_05'] = bool((ii > 0.5).any())
        results['ii_above_07'] = bool((ii > 0.7).any())
    else:
        results['ii_mean'] = None
        results['ii_max'] = None
        results['ii_p90'] = None
        results['ii_spike_frac'] = None
        results['ii_above_05'] = None
        results['ii_above_07'] = None

    # --- Warning window analysis (instability index) ---
    if not is_control:
        for w in WARNING_WINDOWS:
            cutoff = event_dt - timedelta(days=w)
            window_data = pre[pre['time'] >= cutoff]
            ii_w = window_data['instability_index'].dropna() if len(window_data) > 0 else pd.Series(dtype=float)
            sa_w = window_data['sigma_alpha'].dropna() if len(window_data) > 0 else pd.Series(dtype=float)

            results[f'ii_above_05_in_{w}d'] = bool((ii_w > 0.5).any()) if len(ii_w) > 0 else None
            results[f'ii_above_07_in_{w}d'] = bool((ii_w > 0.7).any()) if len(ii_w) > 0 else None
            results[f'ii_max_in_{w}d'] = float(ii_w.max()) if len(ii_w) > 0 else None
            results[f'sigma_max_in_{w}d'] = float(sa_w.max()) if len(sa_w) > 0 else None

    return results


# ==========================================
# MAIN LOOP
# ==========================================

def run_instability_backtest():
    print("=" * 70)
    print("RTM-SEISMO INSTABILITY INDEX BACKTEST")
    print("=" * 70)
    print(f"Events: {len(TARGET_EVENTS)}")
    print(f"Metrics: sigma_alpha, transition_freq, max_drawdown, instability_index")
    print(f"Lookback: {LOOKBACK_DAYS}d | Radius: {RADIUS_KM}km | Min: M{MIN_MAG}")
    print("=" * 70)
    print()

    all_results = []

    for i, (name, date_str, lat, lon, mag, tectonic) in enumerate(TARGET_EVENTS):
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        print(f"[{i+1:2d}/{len(TARGET_EVENTS)}] {name} -- M{mag} -- {date_str}")

        # --- EVENT PERIOD ---
        end_date = event_date + timedelta(days=3)
        start_date = event_date - timedelta(days=LOOKBACK_DAYS)

        print(f"    Fetching event period...")
        df = fetch_usgs_data(lat, lon, RADIUS_KM, start_date, end_date, MIN_MAG)
        time.sleep(API_DELAY)

        if df is None or len(df) < RTM_WINDOW + VOLATILITY_WINDOW + 10:
            print(f"    Insufficient data ({len(df) if df is not None else 0}). Skipping.")
            all_results.append({
                'event': name, 'date': date_str, 'mag': mag,
                'lat': lat, 'lon': lon, 'tectonic': tectonic,
                'status': 'INSUFFICIENT_DATA',
            })
            continue

        # Compute alpha + instability metrics
        alpha_df = compute_alpha(df)
        if alpha_df is None or len(alpha_df) < VOLATILITY_WINDOW + 10:
            print(f"    Alpha failed. Skipping.")
            all_results.append({
                'event': name, 'date': date_str, 'mag': mag,
                'lat': lat, 'lon': lon, 'tectonic': tectonic,
                'status': 'ALPHA_FAILED',
            })
            continue

        inst_df = compute_instability_metrics(alpha_df)
        if inst_df is None:
            print(f"    Instability metrics failed. Skipping.")
            all_results.append({
                'event': name, 'date': date_str, 'mag': mag,
                'lat': lat, 'lon': lon, 'tectonic': tectonic,
                'status': 'METRICS_FAILED',
            })
            continue

        metrics = analyze_period(inst_df, event_date, is_control=False)
        if metrics is None:
            print(f"    Analysis failed. Skipping.")
            all_results.append({
                'event': name, 'date': date_str, 'mag': mag,
                'lat': lat, 'lon': lon, 'tectonic': tectonic,
                'status': 'ANALYSIS_FAILED',
            })
            continue

        # --- CONTROL PERIOD ---
        ctrl_end = event_date - timedelta(days=CONTROL_OFFSET_DAYS)
        ctrl_start = ctrl_end - timedelta(days=LOOKBACK_DAYS)

        print(f"    Fetching control period...")
        ctrl_df_raw = fetch_usgs_data(lat, lon, RADIUS_KM, ctrl_start, ctrl_end, MIN_MAG)
        time.sleep(API_DELAY)

        ctrl_metrics = {}
        if ctrl_df_raw is not None and len(ctrl_df_raw) >= RTM_WINDOW + VOLATILITY_WINDOW + 10:
            ctrl_alpha = compute_alpha(ctrl_df_raw)
            if ctrl_alpha is not None and len(ctrl_alpha) >= VOLATILITY_WINDOW + 10:
                ctrl_inst = compute_instability_metrics(ctrl_alpha)
                if ctrl_inst is not None:
                    cm = analyze_period(ctrl_inst, ctrl_end, is_control=True)
                    if cm is not None:
                        ctrl_metrics = {f'ctrl_{k}': v for k, v in cm.items()}

        if not ctrl_metrics:
            ctrl_metrics = {'ctrl_status': 'INSUFFICIENT'}

        # Compile
        result = {
            'event': name, 'date': date_str, 'mag': mag,
            'lat': lat, 'lon': lon, 'tectonic': tectonic,
            'status': 'OK',
            **metrics,
            **ctrl_metrics,
        }
        all_results.append(result)

        # Print
        ii_flag = "YES" if metrics.get('ii_above_05') else "NO"
        ii_max_val = metrics.get('ii_max', 0) or 0
        sa_max_val = metrics.get('sigma_alpha_max', 0) or 0
        tf_max_val = metrics.get('tf_max', 0) or 0
        mdd_max_val = metrics.get('mdd_max', 0) or 0
        crit_flag = "Y" if metrics.get('entered_critical_08') else "N"

        print(f"    alpha: crit={crit_flag} | sigma_max={sa_max_val:.2f} | TF_max={tf_max_val:.0f} | MDD_max={mdd_max_val:.2f} | II_max={ii_max_val:.3f} | II>0.5={ii_flag}")

        if 'ctrl_ii_max' in ctrl_metrics:
            ctrl_ii = ctrl_metrics.get('ctrl_ii_max', 0) or 0
            print(f"    control: II_max={ctrl_ii:.3f}")
        print()

    # ==========================================
    # SAVE & REPORT
    # ==========================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    ok = results_df[results_df['status'] == 'OK']
    n_ok = len(ok)

    if n_ok == 0:
        print("No valid events.")
        return

    # --- Build report ---
    R = []
    R.append("=" * 70)
    R.append("RTM-SEISMO INSTABILITY INDEX REPORT")
    R.append("=" * 70)
    R.append(f"Valid events: {n_ok} / {len(TARGET_EVENTS)}")
    R.append("")

    # --- COMPARISON: Raw alpha vs Instability Index ---
    R.append("=" * 70)
    R.append("METRIC COMPARISON: RAW ALPHA vs INSTABILITY INDEX")
    R.append("=" * 70)
    R.append("")

    # Raw alpha detection (same as original backtest)
    n_crit = int(ok['entered_critical_08'].sum())
    n_frac = int(ok['entered_fracture_05'].sum())
    R.append(f"RAW ALPHA (original metric):")
    R.append(f"  alpha < 0.8 detected:  {n_crit}/{n_ok} ({100*n_crit/n_ok:.1f}%)")
    R.append(f"  alpha < 0.5 detected:  {n_frac}/{n_ok} ({100*n_frac/n_ok:.1f}%)")
    R.append("")

    # Instability Index detection
    n_ii_05 = int(ok['ii_above_05'].sum()) if 'ii_above_05' in ok else 0
    n_ii_07 = int(ok['ii_above_07'].sum()) if 'ii_above_07' in ok else 0
    R.append(f"INSTABILITY INDEX (new metric):")
    R.append(f"  II > 0.5 detected:     {n_ii_05}/{n_ok} ({100*n_ii_05/n_ok:.1f}%)")
    R.append(f"  II > 0.7 detected:     {n_ii_07}/{n_ok} ({100*n_ii_07/n_ok:.1f}%)")
    R.append("")

    # Sigma alpha detection
    sa_col = ok['sigma_alpha_max'].dropna()
    if len(sa_col) > 0:
        for thresh in [0.5, 0.7, 0.9, 1.0]:
            n_det = int((sa_col > thresh).sum())
            R.append(f"  sigma_alpha_max > {thresh}:  {n_det}/{len(sa_col)} ({100*n_det/len(sa_col):.1f}%)")
    R.append("")

    # --- BY WARNING WINDOW ---
    R.append("=" * 70)
    R.append("INSTABILITY INDEX BY ADVANCE WINDOW")
    R.append("=" * 70)
    R.append("")
    for w in WARNING_WINDOWS:
        col_05 = f'ii_above_05_in_{w}d'
        col_07 = f'ii_above_07_in_{w}d'
        if col_05 in ok.columns:
            valid = ok[ok[col_05].notna()]
            n05 = int(valid[col_05].sum())
            n07 = int(valid[col_07].sum()) if col_07 in valid else 0
            R.append(f"  Within {w:3d}d: II>0.5 {n05}/{len(valid)} ({100*n05/len(valid):.1f}%)  |  II>0.7 {n07}/{len(valid)} ({100*n07/len(valid):.1f}%)")
    R.append("")

    # --- CONTROL COMPARISON ---
    R.append("=" * 70)
    R.append("FALSE POSITIVE ANALYSIS (Event vs Control)")
    R.append("=" * 70)
    R.append("")

    ctrl_ok = ok[ok.get('ctrl_ii_max', pd.Series(dtype=float)).notna()]
    if len(ctrl_ok) > 0:
        # Event period
        evt_ii_05 = (ctrl_ok['ii_above_05'] == True).mean() if 'ii_above_05' in ctrl_ok else 0
        evt_ii_07 = (ctrl_ok['ii_above_07'] == True).mean() if 'ii_above_07' in ctrl_ok else 0
        evt_crit = (ctrl_ok['entered_critical_08'] == True).mean()

        # Control period
        ctrl_ii_05 = (ctrl_ok['ctrl_ii_above_05'] == True).mean() if 'ctrl_ii_above_05' in ctrl_ok else 0
        ctrl_ii_07 = (ctrl_ok['ctrl_ii_above_07'] == True).mean() if 'ctrl_ii_above_07' in ctrl_ok else 0
        ctrl_crit = (ctrl_ok['ctrl_entered_critical_08'] == True).mean() if 'ctrl_entered_critical_08' in ctrl_ok else 0

        R.append(f"  Events with control data: {len(ctrl_ok)}")
        R.append(f"")
        R.append(f"  {'Metric':<30s} {'Pre-Event':>10s} {'Control':>10s} {'Ratio':>8s}")
        R.append(f"  {'-'*60}")
        R.append(f"  {'alpha < 0.8 (original)':<30s} {evt_crit:>10.1%} {ctrl_crit:>10.1%} {(evt_crit/(ctrl_crit+0.001)):>7.1f}x")
        R.append(f"  {'II > 0.5 (new)':<30s} {evt_ii_05:>10.1%} {ctrl_ii_05:>10.1%} {(evt_ii_05/(ctrl_ii_05+0.001)):>7.1f}x")
        R.append(f"  {'II > 0.7 (new)':<30s} {evt_ii_07:>10.1%} {ctrl_ii_07:>10.1%} {(evt_ii_07/(ctrl_ii_07+0.001)):>7.1f}x")

        # Sigma comparison
        evt_sigma_mean = ctrl_ok['sigma_alpha_max'].mean() if 'sigma_alpha_max' in ctrl_ok else 0
        ctrl_sigma_mean = ctrl_ok['ctrl_sigma_alpha_max'].mean() if 'ctrl_sigma_alpha_max' in ctrl_ok else 0
        R.append(f"")
        R.append(f"  Mean sigma_alpha_max:")
        R.append(f"    Pre-event: {evt_sigma_mean:.3f}")
        R.append(f"    Control:   {ctrl_sigma_mean:.3f}")
        if ctrl_sigma_mean > 0:
            R.append(f"    Ratio:     {evt_sigma_mean/ctrl_sigma_mean:.2f}x")
    else:
        R.append("  No control data available.")

    R.append("")

    # --- EVENT DETAIL ---
    R.append("=" * 70)
    R.append("EVENT DETAIL TABLE")
    R.append("=" * 70)
    R.append("")
    R.append(f"{'Event':<28s} {'Mag':>4s} {'a_min':>6s} {'CRIT':>5s} {'s_max':>6s} {'TF':>4s} {'MDD':>5s} {'II_mx':>6s} {'II>.5':>5s} {'II>.7':>5s}")
    R.append("-" * 85)

    for _, row in ok.iterrows():
        crit = "  Y" if row.get('entered_critical_08') else "  N"
        sa = f"{row.get('sigma_alpha_max', 0) or 0:>6.2f}"
        tf = f"{row.get('tf_max', 0) or 0:>4.0f}"
        mdd = f"{row.get('mdd_max', 0) or 0:>5.2f}"
        ii = f"{row.get('ii_max', 0) or 0:>6.3f}"
        ii5 = "    Y" if row.get('ii_above_05') else "    N"
        ii7 = "    Y" if row.get('ii_above_07') else "    N"
        R.append(f"{row['event']:<28s} {row['mag']:>4.1f} {row.get('alpha_min', 0) or 0:>6.3f} {crit} {sa} {tf} {mdd} {ii} {ii5} {ii7}")

    R.append("")
    R.append("=" * 70)
    R.append("LEGEND")
    R.append("=" * 70)
    R.append("  a_min  = minimum raw alpha during pre-event period")
    R.append("  CRIT   = did alpha enter < 0.8 (original metric)")
    R.append("  s_max  = maximum rolling std of alpha (volatility)")
    R.append("  TF     = max transition frequency (regime crossings)")
    R.append("  MDD    = max drawdown rate of alpha")
    R.append("  II_mx  = maximum instability index (composite)")
    R.append("  II>.5  = did instability index exceed 0.5")
    R.append("  II>.7  = did instability index exceed 0.7")
    R.append("")
    R.append("=" * 70)
    R.append("END OF REPORT")
    R.append("=" * 70)

    report_text = "\n".join(R)

    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nResults: {OUTPUT_CSV}")
    print(f"Report:  {OUTPUT_SUMMARY}")
    print()
    print(report_text)


if __name__ == "__main__":
    run_instability_backtest()
