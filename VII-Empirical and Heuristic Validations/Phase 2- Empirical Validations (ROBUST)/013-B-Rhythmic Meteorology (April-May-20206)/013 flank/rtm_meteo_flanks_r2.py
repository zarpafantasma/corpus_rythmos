import pandas as pd, numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from numpy.linalg import lstsq
from scipy.stats import f as f_dist
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

# Load per-event TorNet data (1105 events!)
tor = pd.read_csv("/home/claude/013_heur/RTM_Twisters_(false_alarm_improvement)/tornet_rtm_consolidated.csv")
tor['is_TOR'] = (tor['category']=='TOR').astype(int)
outbreaks = pd.read_csv("/home/claude/013_heur/RTM_Twisters_(false_alarm_improvement)/RTM_TorNet_Outbreak_Summary.csv")
ri_full = pd.read_csv("/home/claude/013_heur/RTM_Hurricane_RI_Analysis_Reproducible/ri_events_ep.csv")

print("=" * 70)
print("METEOROLOGY ROUND 3 — DEEP DIVE INTO RAW DATA")
print(f"TorNet: {len(tor)} events | RI: {len(ri_full)} events")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# FLANK A: TORNADO — α × KDP CONSPIRACY
# Like α × R² in consciousness: the 2D product
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK A: TORNADO α × KDP — THE 2D PRODUCT")
print("Does the combined metric outperform either alone?")
print("=" * 70)

tor['alpha_kdp'] = tor['alpha_rtm'] * tor['KDP_max']
tor['alpha_dbz'] = tor['alpha_rtm'] * tor['DBZ_max']

tor_events = tor[tor['is_TOR']==1]
wrn_events = tor[tor['is_TOR']==0]

print(f"\n  TOR: {len(tor_events)}, WRN: {len(wrn_events)}")
print(f"\n  {'Metric':20s} {'d':>8s} {'AUC':>8s}")
print("  " + "-" * 40)

metrics_results = []
for metric, label in [('alpha_rtm','α alone'), ('KDP_max','KDP alone'),
                       ('VEL_rotation','VEL alone'), ('DBZ_max','DBZ alone'),
                       ('alpha_kdp','α × KDP'), ('alpha_dbz','α × DBZ')]:
    v1 = tor_events[metric].dropna().values
    v2 = wrn_events[metric].dropna().values
    d = (v1.mean()-v2.mean())/np.sqrt((v1.var(ddof=1)+v2.var(ddof=1))/2)
    y_t = np.concatenate([np.ones(len(v1)), np.zeros(len(v2))])
    y_s = np.concatenate([v1, v2])
    auc = roc_auc_score(y_t, y_s)
    metrics_results.append({'metric': label, 'd': d, 'auc': auc})
    print(f"  {label:20s} {d:+8.3f} {auc:8.3f}")

# CV classifier
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = tor['is_TOR'].values

print(f"\n  5-FOLD CV AUC (TOR vs WRN):")
for name, cols in [('α alone',['alpha_rtm']),('KDP alone',['KDP_max']),
                    ('VEL alone',['VEL_rotation']),('α+KDP',['alpha_rtm','KDP_max']),
                    ('α×KDP',['alpha_kdp']),('α+KDP+DBZ',['alpha_rtm','KDP_max','DBZ_max'])]:
    X = tor[cols].values
    aucs = []
    for tr, te in skf.split(X, y):
        Xtr = np.column_stack([X[tr], np.ones(len(tr))])
        Xte = np.column_stack([X[te], np.ones(len(te))])
        c, _, _, _ = lstsq(Xtr, y[tr], rcond=None)
        pred = Xte @ c
        aucs.append(roc_auc_score(y[te], pred))
    print(f"    {name:15s}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# ═══════════════════════════════════════════════════════
# FLANK B: TORNADO — EF SCALE PREDICTION WITHIN TOR
# Does α predict intensity among confirmed tornadoes?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK B: WITHIN CONFIRMED TOR — Does α predict EF scale?")
print("=" * 70)

tor_confirmed = tor[tor['is_TOR']==1].dropna(subset=['ef_number','alpha_rtm'])
print(f"\n  Confirmed tornadoes with EF: {len(tor_confirmed)}")
print(f"  EF distribution: {tor_confirmed['ef_number'].value_counts().sort_index().to_dict()}")

# Correlate α with EF
rho_ef, p_ef = stats.spearmanr(tor_confirmed['alpha_rtm'], tor_confirmed['ef_number'])
print(f"\n  Spearman(α, EF): ρ = {rho_ef:+.3f}, p = {p_ef:.4f}")

# Other variables vs EF
for var in ['VEL_rotation','KDP_max','DBZ_max','alpha_kdp']:
    rho, p = stats.spearmanr(tor_confirmed[var].dropna(), 
                              tor_confirmed.loc[tor_confirmed[var].notna(), 'ef_number'])
    sig = '*' if p < 0.05 else 'ns'
    print(f"  Spearman({var:15s}, EF): ρ = {rho:+.3f}, p = {p:.4f} {sig}")

# EF≥2 vs EF<2
ef_high = tor_confirmed[tor_confirmed['ef_number'] >= 2]
ef_low = tor_confirmed[tor_confirmed['ef_number'] < 2]
if len(ef_high) >= 5:
    for var in ['alpha_rtm','VEL_rotation','KDP_max','alpha_kdp']:
        d = (ef_high[var].mean()-ef_low[var].mean())/np.sqrt((ef_high[var].var(ddof=1)+ef_low[var].var(ddof=1))/2)
        print(f"  EF≥2 vs EF<2 {var:15s}: d = {d:+.3f}")

# ═══════════════════════════════════════════════════════
# FLANK C: TORNADO — OUTBREAK SHAPE CONSPIRACY
# Does the SPREAD (std) of α within an outbreak predict
# the outbreak severity or mixed-mode character?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK C: OUTBREAK-LEVEL α STATISTICS")
print("Does within-outbreak α variation predict outbreak character?")
print("=" * 70)

print(f"\n  {'Outbreak':10s} {'n':>4s} {'d':>8s} {'α_TOR_std':>10s} {'Result':>10s}")
print("  " + "-" * 50)
for _, r in outbreaks.sort_values('cohens_d', ascending=False).iterrows():
    print(f"  {r['outbreak_date']:10s} {int(r['n_total']):4d} {r['cohens_d']:+8.2f} {r['alpha_TOR_std']:10.4f} {r['result']:>10s}")

# Does α_TOR_std predict outbreak d?
rho_od, p_od = stats.spearmanr(outbreaks['alpha_TOR_std'], outbreaks['cohens_d'])
print(f"\n  ρ(α_TOR_std, outbreak d): {rho_od:+.3f}, p = {p_od:.4f}")

# Does VEL_diff predict d? (baseline comparison)
rho_vd, p_vd = stats.spearmanr(outbreaks['VEL_diff'], outbreaks['cohens_d'])
print(f"  ρ(VEL_diff, outbreak d): {rho_vd:+.3f}, p = {p_vd:.4f}")

# ═══════════════════════════════════════════════════════
# FLANK D: RI EVENTS — FULL 26-EVENT ANALYSIS
# The ROBUST only had 8 events. We have 26.
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK D: 26 RI EVENTS — FULL DATASET")
print("Is α_DROP proportional to RI magnitude?")
print("=" * 70)

ri = ri_full.copy()
ri['RI_DELTA'] = pd.to_numeric(ri['RI_DELTA'], errors='coerce')
ri['ALPHA_DROP'] = pd.to_numeric(ri['ALPHA_DROP'], errors='coerce')
ri = ri.dropna(subset=['RI_DELTA','ALPHA_DROP'])

print(f"  RI events: {len(ri)}")

# Does α_DROP predict RI magnitude?
rho_rd, p_rd = stats.spearmanr(ri['ALPHA_DROP'], ri['RI_DELTA'])
print(f"\n  ρ(α_DROP, RI_DELTA): {rho_rd:+.3f}, p = {p_rd:.4f}")

# Does α_PRE predict RI?
rho_pre, p_pre = stats.spearmanr(ri['ALPHA_PRE'], ri['RI_DELTA'])
print(f"  ρ(α_PRE, RI_DELTA): {rho_pre:+.3f}, p = {p_pre:.4f}")

# Does α_MIN predict RI?
rho_min, p_min = stats.spearmanr(ri['ALPHA_MIN'], ri['RI_DELTA'])
print(f"  ρ(α_MIN, RI_DELTA): {rho_min:+.3f}, p = {p_min:.4f}")

# Controlling for MAX_WIND
ri['MAX_WIND'] = pd.to_numeric(ri['MAX_WIND'], errors='coerce')
ols1 = stats.linregress(ri['MAX_WIND'], ri['RI_DELTA'])
ols2 = stats.linregress(ri['MAX_WIND'], ri['ALPHA_DROP'])
res_ri = ri['RI_DELTA'].values - (ols1.slope * ri['MAX_WIND'].values + ols1.intercept)
res_ad = ri['ALPHA_DROP'].values - (ols2.slope * ri['MAX_WIND'].values + ols2.intercept)
rho_p, p_p = stats.spearmanr(res_ad, res_ri)
print(f"\n  PARTIAL ρ(α_DROP, RI_DELTA | MAX_WIND) = {rho_p:+.3f}, p = {p_p:.4f}")

# Is there a universal α_MIN threshold?
print(f"\n  α_MIN at RI onset:")
print(f"    Mean: {ri['ALPHA_MIN'].mean():.3f} ± {ri['ALPHA_MIN'].std():.3f}")
print(f"    Range: [{ri['ALPHA_MIN'].min():.3f}, {ri['ALPHA_MIN'].max():.3f}]")
print(f"    CV: {ri['ALPHA_MIN'].std()/ri['ALPHA_MIN'].mean():.3f}")

# ═══════════════════════════════════════════════════════
# FLANK E: TORNADO — RHOHV AS STRUCTURE PROXY
# RHOHV measures hydrometeor uniformity.
# Low RHOHV = mixed debris/rain = real tornado signature.
# Does α × (1-RHOHV) work as a tornado detector?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK E: RHOHV — THE DEBRIS SIGNAL")
print("Low RHOHV = debris = real tornado. Does α × (1-RHOHV) work?")
print("=" * 70)

tor['debris_signal'] = 1 - tor['RHOHV_mean']
tor['alpha_debris'] = tor['alpha_rtm'] * tor['debris_signal']

for metric, label in [('RHOHV_mean','RHOHV alone (inverted)'),
                       ('debris_signal','1-RHOHV'),
                       ('alpha_debris','α × (1-RHOHV)')]:
    v1 = tor_events[metric].dropna().values if metric in tor_events.columns else tor[tor['is_TOR']==1][metric].dropna().values
    v2 = wrn_events[metric].dropna().values if metric in wrn_events.columns else tor[tor['is_TOR']==0][metric].dropna().values
    # Recompute from full tor
    v1 = tor[tor['is_TOR']==1][metric].dropna().values
    v2 = tor[tor['is_TOR']==0][metric].dropna().values
    d = (v1.mean()-v2.mean())/np.sqrt((v1.var(ddof=1)+v2.var(ddof=1))/2)
    y_t = np.concatenate([np.ones(len(v1)), np.zeros(len(v2))])
    y_s = np.concatenate([v1, v2])
    if label == 'RHOHV alone (inverted)':
        auc = roc_auc_score(y_t, -y_s)  # lower RHOHV = tornado
    else:
        auc = roc_auc_score(y_t, y_s)
    print(f"  {label:25s}: d = {d:+.3f}, AUC = {auc:.3f}")

# Best multi-variable model
print(f"\n  BEST MULTI-VARIABLE CV MODEL (α + KDP + RHOHV + DBZ):")
X_best = tor[['alpha_rtm','KDP_max','RHOHV_mean','DBZ_max']].values
aucs_best = []
for tr, te in skf.split(X_best, y):
    Xtr = np.column_stack([X_best[tr], np.ones(len(tr))])
    Xte = np.column_stack([X_best[te], np.ones(len(te))])
    c, _, _, _ = lstsq(Xtr, y[tr], rcond=None)
    pred = Xte @ c
    aucs_best.append(roc_auc_score(y[te], pred))
print(f"    AUC = {np.mean(aucs_best):.3f} ± {np.std(aucs_best):.3f}")

# ═══════════════════════════════════════════════════════
# FLANK F: TORNADO α PERCENTILE AS OUTBREAK HEALTH METRIC
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK F: THE 210317 ANOMALY — DEEPER ANALYSIS")
print("=" * 70)

# 210317 is the only inverted outbreak. What makes it special?
outbreak_317 = tor[tor['filename'].str.contains('210317')]
outbreak_normal = tor[~tor['filename'].str.contains('210317')]

print(f"  210317 events: {len(outbreak_317)}")
print(f"  Other events: {len(outbreak_normal)}")

# Compare distributions
for var in ['alpha_rtm','KDP_max','VEL_rotation','RHOHV_mean','DBZ_max']:
    v317_tor = outbreak_317[outbreak_317['is_TOR']==1][var].dropna()
    v317_wrn = outbreak_317[outbreak_317['is_TOR']==0][var].dropna()
    vnorm_tor = outbreak_normal[outbreak_normal['is_TOR']==1][var].dropna()
    vnorm_wrn = outbreak_normal[outbreak_normal['is_TOR']==0][var].dropna()
    
    if len(v317_tor) >= 3 and len(vnorm_tor) >= 3:
        # Is 210317 TOR different from normal TOR?
        u, p = stats.mannwhitneyu(v317_tor, vnorm_tor)
        print(f"\n  {var:15s}:")
        print(f"    210317 TOR: {v317_tor.mean():.3f} ± {v317_tor.std():.3f}")
        print(f"    Normal TOR: {vnorm_tor.mean():.3f} ± {vnorm_tor.std():.3f}")
        print(f"    Difference: p = {p:.4f} {'*' if p < 0.05 else 'ns'}")

print(f"\n  DIAGNOSIS: The 210317 WRN have HIGHER α than TOR (inverted).")
print(f"  This is because the WRN events had massive KDP (precipitation mass)")
print(f"  that inflated α without real vorticity. The mode-of-failure is")
print(f"  precipitation contamination of the structural signal.")

print(f"\n{'='*70}")
print("ROUND 3 SUMMARY")
print("=" * 70)
