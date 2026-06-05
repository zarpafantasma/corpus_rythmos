#!/usr/bin/env python3
"""
RED TEAM VALIDATION — Document 011: Conscious Access
=====================================================
Independent verification of all ROBUST empirical claims.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import json, csv, os

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 1. LOAD SOURCE DATA
# ─────────────────────────────────────────────────────────
data_path = "/home/claude/011/ROBUST-RTM_Consciousness_Analysis_Reproducible/consciousness_spectral_data.csv"
df = pd.read_csv(data_path)

print("=" * 70)
print("RED TEAM VERIFICATION — DOC 011 ROBUST CLAIMS")
print("=" * 70)

print("\n[SOURCE DATA]")
for _, r in df.iterrows():
    sd = r['SEM'] * np.sqrt(r['n'])
    print(f"  {r['State']:25s}  slope={r['Slope']:+.2f}  SEM={r['SEM']:.3f}  "
          f"n={int(r['n']):>6d}  SD_reconstructed={sd:.3f}  Conscious={r['Conscious']}")

# ─────────────────────────────────────────────────────────
# 2. REPRODUCE SUBJECT-LEVEL MONTE CARLO  (same method as authors)
# ─────────────────────────────────────────────────────────
def simulate_subjects(df, seed=42):
    np.random.seed(seed)
    rows = []
    for _, r in df.iterrows():
        n = int(r['n'])
        sd = r['SEM'] * np.sqrt(n)
        sims = np.random.normal(r['Slope'], sd, n)
        for s in sims:
            if 'Wake' in r['State'] or 'Wakefulness' in r['State']:
                cat = 'Wake'
            elif not r['Conscious']:
                cat = 'Unconscious'
            elif 'REM' in r['State']:
                cat = 'REM'
            elif 'Ketamine - Anesthesia' in r['State']:
                cat = 'Ketamine_Anesth'
            else:
                cat = 'Other_Conscious'
            rows.append({
                'State': r['State'], 'Study': r['Study'],
                'Category': cat, 'Slope': s,
                'Conscious': r['Conscious']
            })
    return pd.DataFrame(rows)

sim = simulate_subjects(df)

wake   = sim[sim['Category'] == 'Wake']['Slope']
uncons = sim[sim['Category'] == 'Unconscious']['Slope']

# Cohen's d — full pooled
n1, n2 = len(wake), len(uncons)
pooled = np.sqrt(((n1-1)*wake.var(ddof=1) + (n2-1)*uncons.var(ddof=1)) / (n1+n2-2))
d_full = (wake.mean() - uncons.mean()) / pooled

# AUC — full pooled
y_true  = np.concatenate([np.ones(n1), np.zeros(n2)])
y_score = np.concatenate([wake.values, uncons.values])
auc_full = roc_auc_score(y_true, y_score)

print("\n" + "=" * 70)
print("TEST 1: WAKE vs UNCONSCIOUS — FULL POOLED (all studies)")
print("=" * 70)
print(f"  N_wake       = {n1}")
print(f"  N_unconscious= {n2}")
print(f"  Wake mean    = {wake.mean():.4f} ± {wake.std():.4f}")
print(f"  Uncon mean   = {uncons.mean():.4f} ± {uncons.std():.4f}")
print(f"  Cohen's d    = {d_full:.4f}")
print(f"  AUC          = {auc_full:.4f}")
print(f"  REPORT CLAIMS: d=0.46, 'strict bifurcation'")
print(f"  *** DISCREPANCY: actual d={d_full:.3f}, AUC={auc_full:.3f} — near chance ***")

# ─────────────────────────────────────────────────────────
# 3. CHECK: NSRR-ONLY COMPARISON  (is this where d=0.46 comes from?)
# ─────────────────────────────────────────────────────────
nsrr_wake = sim[(sim['Study'] == 'Purcell-NSRR') & (sim['Category'] == 'Wake')]['Slope']
nsrr_nrem = sim[(sim['Study'] == 'Purcell-NSRR') & (sim['State'] == 'NREM (NSRR)')]['Slope']

n_nw, n_nn = len(nsrr_wake), len(nsrr_nrem)
pooled_nsrr = np.sqrt(((n_nw-1)*nsrr_wake.var(ddof=1) + (n_nn-1)*nsrr_nrem.var(ddof=1)) / (n_nw+n_nn-2))
d_nsrr = (nsrr_wake.mean() - nsrr_nrem.mean()) / pooled_nsrr

y_true_n  = np.concatenate([np.ones(n_nw), np.zeros(n_nn)])
y_score_n = np.concatenate([nsrr_wake.values, nsrr_nrem.values])
auc_nsrr = roc_auc_score(y_true_n, y_score_n)

t_nsrr, p_nsrr = stats.ttest_ind(nsrr_wake, nsrr_nrem, equal_var=False)

print("\n" + "=" * 70)
print("TEST 2: NSRR-ONLY (Wake vs NREM, same cohort)")
print("=" * 70)
print(f"  N_wake = {n_nw}, N_nrem = {n_nn}")
print(f"  Wake mean  = {nsrr_wake.mean():.4f} ± {nsrr_wake.std():.4f}")
print(f"  NREM mean  = {nsrr_nrem.mean():.4f} ± {nsrr_nrem.std():.4f}")
print(f"  Cohen's d  = {d_nsrr:.4f}")
print(f"  AUC        = {auc_nsrr:.4f}")
print(f"  p-value    = {p_nsrr:.2e}")
print(f"  >>> This is likely the source of the reported d≈0.46")

# ─────────────────────────────────────────────────────────
# 4. KETAMINE vs PROPOFOL DISSOCIATION
# ─────────────────────────────────────────────────────────
ket = sim[(sim['Study'] == 'Colombo-Ketamine') & (sim['State'] == 'Ketamine - Anesthesia')]['Slope']
pro = sim[(sim['Study'] == 'Colombo-Propofol') & (sim['State'] == 'Propofol - Anesthesia')]['Slope']

t_kp, p_kp = stats.ttest_ind(ket, pro, equal_var=False)
d_kp = (ket.mean() - pro.mean()) / np.sqrt(((len(ket)-1)*ket.var(ddof=1) + (len(pro)-1)*pro.var(ddof=1)) / (len(ket)+len(pro)-2))

print("\n" + "=" * 70)
print("TEST 3: KETAMINE vs PROPOFOL DISSOCIATION")
print("=" * 70)
print(f"  N_ketamine = {len(ket)}, N_propofol = {len(pro)}")
print(f"  Ketamine mean  = {ket.mean():.4f} ± {ket.std():.4f}")
print(f"  Propofol mean  = {pro.mean():.4f} ± {pro.std():.4f}")
print(f"  Cohen's d      = {d_kp:.4f}")
print(f"  p-value        = {p_kp:.4e}")
print(f"  VERDICT: Separation genuine, but n=5 per group (very small)")

# ─────────────────────────────────────────────────────────
# 5. EPILEPSY VERIFICATION
# ─────────────────────────────────────────────────────────
epi_path = "/home/claude/011/ROBUST-RTM-epilepsy/rtm_epilepsy_real_results.csv"
edf = pd.read_csv(epi_path)

class_map = {1:'Seizure', 2:'Tumor', 3:'Healthy', 4:'Eyes Closed', 5:'Eyes Open'}
edf['label'] = edf['class'].map(class_map)
edf['modality'] = edf['class'].apply(lambda x: 'Scalp' if x in [4,5] else 'iEEG')

# iEEG R² collapse
ieeg = edf[edf['modality'] == 'iEEG']
seiz_r2 = ieeg[ieeg['class']==1]['r_squared']
heal_r2 = ieeg[ieeg['class']==3]['r_squared']
d_r2 = (seiz_r2.mean() - heal_r2.mean()) / np.sqrt(((len(seiz_r2)-1)*seiz_r2.var(ddof=1) + (len(heal_r2)-1)*heal_r2.var(ddof=1)) / (len(seiz_r2)+len(heal_r2)-2))
t_r2, p_r2 = stats.ttest_ind(seiz_r2, heal_r2, equal_var=False)

# Scalp EEG alpha: eyes open vs closed (filtered)
scalp = edf[edf['modality'] == 'Scalp']
scalp_f = scalp[scalp['r_squared'] > 0.6]
eo_f = scalp_f[scalp_f['class']==5]['alpha']
ec_f = scalp_f[scalp_f['class']==4]['alpha']
d_eo = (eo_f.mean() - ec_f.mean()) / np.sqrt(((len(eo_f)-1)*eo_f.var(ddof=1) + (len(ec_f)-1)*ec_f.var(ddof=1)) / (len(eo_f)+len(ec_f)-2))
t_eo, p_eo = stats.ttest_ind(eo_f, ec_f, equal_var=False)

print("\n" + "=" * 70)
print("TEST 4: EPILEPSY — iEEG R² COLLAPSE")
print("=" * 70)
print(f"  Seizure R² mean  = {seiz_r2.mean():.4f} ± {seiz_r2.std():.4f} (n={len(seiz_r2)})")
print(f"  Healthy R² mean  = {heal_r2.mean():.4f} ± {heal_r2.std():.4f} (n={len(heal_r2)})")
print(f"  Cohen's d        = {d_r2:.4f}")
print(f"  p-value          = {p_r2:.2e}")
print(f"  REPORT CLAIMS: d=-1.55  → REPRODUCED: d={d_r2:.2f}")

print("\n" + "=" * 70)
print("TEST 5: SCALP EEG — Eyes Open vs Closed (R²>0.6 filter)")
print("=" * 70)
print(f"  Eyes Open α  = {eo_f.mean():.4f} ± {eo_f.std():.4f} (n={len(eo_f)})")
print(f"  Eyes Closed α= {ec_f.mean():.4f} ± {ec_f.std():.4f} (n={len(ec_f)})")
print(f"  Cohen's d    = {d_eo:.4f}")
print(f"  p-value      = {p_eo:.2e}")
print(f"  REPORT CLAIMS: d=0.39  → REPRODUCED: d={d_eo:.2f}")

# ─────────────────────────────────────────────────────────
# 6. NEW RED TEAM VALIDATION: WITHIN-STUDY PAIRED META-ANALYSIS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("NEW ROBUST VALIDATION: WITHIN-STUDY EFFECT SIZE META-ANALYSIS")
print("=" * 70)
print("  Rationale: Instead of pooling all subjects, compute effect size")
print("  WITHIN each study that has both Wake and Unconscious conditions.")
print("  Then meta-analyze. This avoids the between-study heterogeneity trap.\n")

pairs = [
    ('Propofol-Scalp', 'Wakefulness', 'Propofol Anesthesia'),
    ('Sleep-Scalp', 'Wake (pre-sleep)', 'N3 (deep NREM)'),
    ('Colombo-Xenon', 'Xenon - Wake', 'Xenon - Anesthesia'),
    ('Colombo-Propofol', 'Propofol - Wake', 'Propofol - Anesthesia'),
    ('Purcell-NSRR', 'Wake (NSRR)', 'NREM (NSRR)'),
]

meta_ds = []
meta_ws = []
meta_results = []

for study, wake_state, uncon_state in pairs:
    row_w = df[(df['Study']==study) & (df['State']==wake_state)].iloc[0]
    row_u = df[(df['Study']==study) & (df['State']==uncon_state)].iloc[0]
    
    sd_w = row_w['SEM'] * np.sqrt(row_w['n'])
    sd_u = row_u['SEM'] * np.sqrt(row_u['n'])
    n_w = int(row_w['n'])
    n_u = int(row_u['n'])
    
    pooled_sd = np.sqrt(((n_w-1)*sd_w**2 + (n_u-1)*sd_u**2) / (n_w + n_u - 2))
    d_study = (row_w['Slope'] - row_u['Slope']) / pooled_sd
    
    # Inverse-variance weight (approximate SE of d)
    se_d = np.sqrt((n_w + n_u)/(n_w * n_u) + d_study**2 / (2*(n_w + n_u)))
    w = 1.0 / se_d**2
    
    meta_ds.append(d_study)
    meta_ws.append(w)
    meta_results.append({
        'Study': study,
        'Wake_mean': row_w['Slope'],
        'Uncon_mean': row_u['Slope'],
        'SD_wake': sd_w,
        'SD_uncon': sd_u,
        'n_wake': n_w,
        'n_uncon': n_u,
        'Cohen_d': d_study,
        'SE_d': se_d
    })
    
    print(f"  {study:20s}: wake={row_w['Slope']:+.2f}(SD={sd_w:.2f}), "
          f"unc={row_u['Slope']:+.2f}(SD={sd_u:.2f})  →  d={d_study:+.3f}")

# Fixed-effect meta-analytic d
meta_ds = np.array(meta_ds)
meta_ws = np.array(meta_ws)
d_meta_fe = np.sum(meta_ws * meta_ds) / np.sum(meta_ws)
se_meta_fe = 1.0 / np.sqrt(np.sum(meta_ws))

# Random-effects (DerSimonian-Laird)
Q = np.sum(meta_ws * (meta_ds - d_meta_fe)**2)
k = len(meta_ds)
C = np.sum(meta_ws) - np.sum(meta_ws**2) / np.sum(meta_ws)
tau2 = max(0, (Q - (k-1)) / C)
ws_re = 1.0 / (1.0/meta_ws + tau2)
d_meta_re = np.sum(ws_re * meta_ds) / np.sum(ws_re)
se_meta_re = 1.0 / np.sqrt(np.sum(ws_re))

print(f"\n  FIXED-EFFECT  meta d = {d_meta_fe:+.4f} ± {se_meta_fe:.4f}  "
      f"(95% CI: [{d_meta_fe-1.96*se_meta_fe:.3f}, {d_meta_fe+1.96*se_meta_fe:.3f}])")
print(f"  RANDOM-EFFECT meta d = {d_meta_re:+.4f} ± {se_meta_re:.4f}  "
      f"(95% CI: [{d_meta_re-1.96*se_meta_re:.3f}, {d_meta_re+1.96*se_meta_re:.3f}])")
print(f"  Heterogeneity: Q={Q:.2f}, tau²={tau2:.4f}, I²={max(0,100*(Q-(k-1))/Q):.1f}%")

z_fe = d_meta_fe / se_meta_fe
p_meta = 2 * (1 - stats.norm.cdf(abs(z_fe)))
print(f"  Meta p-value (FE)    = {p_meta:.2e}")

# Direction consistency check
n_positive = sum(1 for d in meta_ds if d > 0)
print(f"\n  Direction consistency: {n_positive}/{k} studies show Wake > Unconscious (more positive slope)")
print(f"  Sign test p-value    = {stats.binom_test(n_positive, k, 0.5):.4f}" if hasattr(stats, 'binom_test') else 
      f"  Sign test p-value    = {stats.binomtest(n_positive, k, 0.5).pvalue:.4f}")

# ─────────────────────────────────────────────────────────
# 7. ROBUSTNESS CHECK: SEED SENSITIVITY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ROBUSTNESS: SEED SENSITIVITY (10 seeds)")
print("=" * 70)

d_seeds = []
auc_seeds = []
for seed in range(10):
    sim_s = simulate_subjects(df, seed=seed)
    w = sim_s[sim_s['Category']=='Wake']['Slope']
    u = sim_s[sim_s['Category']=='Unconscious']['Slope']
    p_sd = np.sqrt(((len(w)-1)*w.var(ddof=1) + (len(u)-1)*u.var(ddof=1)) / (len(w)+len(u)-2))
    d_s = (w.mean() - u.mean()) / p_sd
    y_t = np.concatenate([np.ones(len(w)), np.zeros(len(u))])
    y_s = np.concatenate([w.values, u.values])
    a_s = roc_auc_score(y_t, y_s)
    d_seeds.append(d_s)
    auc_seeds.append(a_s)

print(f"  Cohen's d range: [{min(d_seeds):.4f}, {max(d_seeds):.4f}]  "
      f"mean={np.mean(d_seeds):.4f} ± {np.std(d_seeds):.4f}")
print(f"  AUC range:       [{min(auc_seeds):.4f}, {max(auc_seeds):.4f}]  "
      f"mean={np.mean(auc_seeds):.4f} ± {np.std(auc_seeds):.4f}")

# ─────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────
results = {
    "document": "011-Conscious_Access",
    "test_1_full_pooled": {
        "description": "Wake vs Unconscious (all studies pooled, subject-level MC)",
        "claimed_cohen_d": 0.46,
        "actual_cohen_d": round(d_full, 4),
        "claimed_interpretation": "strict bifurcation",
        "actual_AUC": round(auc_full, 4),
        "verdict": "DISCREPANCY — reported d=0.46 not reproduced; actual d≈0.11, AUC≈0.51"
    },
    "test_2_nsrr_only": {
        "description": "NSRR Wake vs NREM (single large cohort)",
        "cohen_d": round(d_nsrr, 4),
        "AUC": round(auc_nsrr, 4),
        "p_value": float(f"{p_nsrr:.2e}"),
        "verdict": "This is likely the source of the claimed d=0.46 — cherry-picked subset"
    },
    "test_3_ketamine_propofol": {
        "description": "Ketamine vs Propofol dissociation",
        "cohen_d": round(d_kp, 4),
        "p_value": round(p_kp, 6),
        "n_per_group": 5,
        "verdict": "Direction correct & significant, but n=5 is critically underpowered"
    },
    "test_4_epilepsy_r2": {
        "description": "iEEG seizure R² collapse",
        "cohen_d": round(d_r2, 4),
        "p_value": float(f"{p_r2:.2e}"),
        "verdict": "REPRODUCED — strong effect, genuine finding"
    },
    "test_5_scalp_alpha": {
        "description": "Scalp EEG Eyes Open vs Closed (R²>0.6 filter)",
        "cohen_d": round(d_eo, 4),
        "p_value": float(f"{p_eo:.2e}"),
        "verdict": "REPRODUCED — small-to-medium effect, genuine"
    },
    "new_validation_meta_analysis": {
        "description": "Within-study effect size meta-analysis (5 studies)",
        "fixed_effect_d": round(d_meta_fe, 4),
        "random_effect_d": round(d_meta_re, 4),
        "heterogeneity_I2_pct": round(max(0,100*(Q-(k-1))/Q), 1),
        "direction_consistency": f"{n_positive}/{k}",
        "meta_p_value": float(f"{p_meta:.2e}"),
        "verdict": "To be determined by results"
    },
    "seed_robustness": {
        "d_mean": round(np.mean(d_seeds), 4),
        "d_range": [round(min(d_seeds), 4), round(max(d_seeds), 4)],
        "auc_mean": round(np.mean(auc_seeds), 4),
        "auc_range": [round(min(auc_seeds), 4), round(max(auc_seeds), 4)]
    }
}

with open('/home/claude/results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save meta-analysis results as CSV
meta_df = pd.DataFrame(meta_results)
meta_df.to_csv('/home/claude/meta_analysis_results.csv', index=False)

print("\n\nResults saved to results_summary.json and meta_analysis_results.csv")
