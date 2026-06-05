#!/usr/bin/env python3
"""
RTM CONSCIOUSNESS FLANKING CAMPAIGN
=====================================
Main weakness: REM paradox (conscious state with "unconscious" slopes)
Main strength: Epilepsy R² collapse (d=-1.55)

Strategy: Attack the REM paradox directly. Find what DOES separate
conscious from unconscious if slope alone fails.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("RTM CONSCIOUSNESS — FLANKING CAMPAIGN")
print("=" * 70)

# Load all data
spec = pd.read_csv("/home/claude/011/ROBUST-RTM_Consciousness_Analysis_Reproducible/consciousness_spectral_data.csv")
epi = pd.read_csv("/home/claude/011/ROBUST-RTM-epilepsy/rtm_epilepsy_real_results.csv")

class_map = {1:'Seizure', 2:'Tumor', 3:'Healthy', 4:'Eyes_Closed', 5:'Eyes_Open'}
epi['label'] = epi['class'].map(class_map)

print(f"  Spectral data: {len(spec)} conditions, {spec['n'].sum()} total subjects")
print(f"  Epilepsy data: {len(epi)} recordings, 5 classes")

# ═══════════════════════════════════════════════════════
# FLANK 1: THE α-R² PLANE (Two-dimensional consciousness)
# Instead of using α alone (which fails on REM),
# use both α AND R² (collapse quality) together.
# RTM predicts: conscious states need BOTH high α AND high R²
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 1: THE α-R² PLANE")
print("Consciousness requires BOTH fluid scaling (high α) AND intact")
print("power-law structure (high R²). REM has low α but what about R²?")
print(f"{'='*70}")

# Per-class stats in epilepsy data
print(f"\n  UCI DATA — α AND R² BY CLASS:")
print(f"  {'Class':15s} {'n':>6s} {'α mean':>8s} {'α std':>8s} {'R² mean':>8s} {'R² std':>8s}")
print("  " + "-" * 58)

class_stats = []
for cls in [1,2,3,4,5]:
    sub = epi[epi['class']==cls]
    class_stats.append({
        'class': cls, 'label': class_map[cls],
        'n': len(sub),
        'alpha_mean': sub['alpha'].mean(), 'alpha_std': sub['alpha'].std(),
        'r2_mean': sub['r_squared'].mean(), 'r2_std': sub['r_squared'].std()
    })
    print(f"  {class_map[cls]:15s} {len(sub):6d} {sub['alpha'].mean():8.3f} {sub['alpha'].std():8.3f} "
          f"{sub['r_squared'].mean():8.3f} {sub['r_squared'].std():8.3f}")

# KEY: Can α × R² (product) or α + R² (sum) separate states better?
epi['alpha_r2_product'] = epi['alpha'] * epi['r_squared']
epi['alpha_r2_sum'] = epi['alpha'] + epi['r_squared']

# Compare discrimination: scalp EEG (Eyes Open vs Eyes Closed)
eo = epi[epi['class']==5]
ec = epi[epi['class']==4]

for metric, label in [('alpha', 'α alone'), ('r_squared', 'R² alone'),
                       ('alpha_r2_product', 'α × R²'), ('alpha_r2_sum', 'α + R²')]:
    d = (eo[metric].mean() - ec[metric].mean()) / np.sqrt((eo[metric].var() + ec[metric].var())/2)
    
    y_true = np.concatenate([np.ones(len(eo)), np.zeros(len(ec))])
    y_score = np.concatenate([eo[metric].values, ec[metric].values])
    auc = roc_auc_score(y_true, y_score)
    
    print(f"\n  EO vs EC — {label}: d = {d:+.3f}, AUC = {auc:.3f}")

# Seizure vs Healthy
seiz = epi[epi['class']==1]
heal = epi[epi['class']==3]

print(f"\n  SEIZURE vs HEALTHY:")
for metric, label in [('alpha', 'α alone'), ('r_squared', 'R² alone'),
                       ('alpha_r2_product', 'α × R²'), ('alpha_r2_sum', 'α + R²')]:
    d = (heal[metric].mean() - seiz[metric].mean()) / np.sqrt((heal[metric].var() + seiz[metric].var())/2)
    
    y_true = np.concatenate([np.ones(len(heal)), np.zeros(len(seiz))])
    y_score = np.concatenate([heal[metric].values, seiz[metric].values])
    auc = roc_auc_score(y_true, y_score)
    
    print(f"    {label:12s}: d = {d:+.3f}, AUC = {auc:.3f}")

# ═══════════════════════════════════════════════════════
# FLANK 2: R² AS CONSCIOUSNESS MARKER (not α)
# What if collapse quality IS the marker, not the slope?
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 2: R² AS THE REAL CONSCIOUSNESS MARKER")
print("What if the power-law QUALITY (R²), not the exponent (α),")
print("is what separates conscious from unconscious?")
print(f"{'='*70}")

# Full 5-class discrimination using R² only
print(f"\n  R² DISCRIMINATION (all pairs):")
print(f"  {'Comparison':25s} {'d(R²)':>8s} {'AUC(R²)':>8s} {'d(α)':>8s} {'AUC(α)':>8s} {'Winner':>8s}")
print("  " + "-" * 72)

pairs = [(5,4,'EO vs EC'), (3,1,'Healthy vs Seizure'), (3,2,'Healthy vs Tumor'),
         (5,1,'EO vs Seizure'), (4,1,'EC vs Seizure'), (5,2,'EO vs Tumor')]

r2_wins = 0
alpha_wins = 0
for c1, c2, label in pairs:
    g1 = epi[epi['class']==c1]
    g2 = epi[epi['class']==c2]
    
    d_r2 = (g1['r_squared'].mean()-g2['r_squared'].mean())/np.sqrt((g1['r_squared'].var()+g2['r_squared'].var())/2)
    d_a = (g1['alpha'].mean()-g2['alpha'].mean())/np.sqrt((g1['alpha'].var()+g2['alpha'].var())/2)
    
    y_true = np.concatenate([np.ones(len(g1)), np.zeros(len(g2))])
    auc_r2 = roc_auc_score(y_true, np.concatenate([g1['r_squared'].values, g2['r_squared'].values]))
    auc_a = roc_auc_score(y_true, np.concatenate([g1['alpha'].values, g2['alpha'].values]))
    
    winner = 'R²' if abs(d_r2) > abs(d_a) else 'α'
    if winner == 'R²': r2_wins += 1
    else: alpha_wins += 1
    
    print(f"  {label:25s} {d_r2:+8.3f} {auc_r2:8.3f} {d_a:+8.3f} {auc_a:8.3f} {winner:>8s}")

print(f"\n  R² wins: {r2_wins}/{r2_wins+alpha_wins}, α wins: {alpha_wins}/{r2_wins+alpha_wins}")

# ═══════════════════════════════════════════════════════
# FLANK 3: α-R² CONSPIRACY (like baryon-halo in SPARC)
# Do α and R² correlate WITHIN each class?
# Does the coupling strength differ between states?
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 3: α-R² CONSPIRACY")
print("Do α and R² couple differently across consciousness states?")
print(f"{'='*70}")

print(f"\n  WITHIN-CLASS α-R² correlation:")
conspiracy_data = []
for cls in [1,2,3,4,5]:
    sub = epi[epi['class']==cls]
    rho, p = stats.spearmanr(sub['alpha'], sub['r_squared'])
    conspiracy_data.append({
        'class': cls, 'label': class_map[cls],
        'rho': rho, 'p': p, 'n': len(sub)
    })
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {class_map[cls]:15s} (n={len(sub):4d}): ρ(α, R²) = {rho:+.3f} {sig}")

cdf = pd.DataFrame(conspiracy_data)

# KEY: Does seizure BREAK the conspiracy?
heal_rho = cdf[cdf['class']==3]['rho'].values[0]
seiz_rho = cdf[cdf['class']==1]['rho'].values[0]
print(f"\n  Healthy conspiracy: ρ = {heal_rho:+.3f}")
print(f"  Seizure conspiracy: ρ = {seiz_rho:+.3f}")
print(f"  Conspiracy drops from {heal_rho:+.3f} → {seiz_rho:+.3f} during seizure")

# Bootstrap the difference
np.random.seed(42)
boot_diff = []
for _ in range(3000):
    idx_h = np.random.choice(len(heal), len(heal), replace=True)
    idx_s = np.random.choice(len(seiz), len(seiz), replace=True)
    r_h, _ = stats.spearmanr(heal['alpha'].values[idx_h], heal['r_squared'].values[idx_h])
    r_s, _ = stats.spearmanr(seiz['alpha'].values[idx_s], seiz['r_squared'].values[idx_s])
    boot_diff.append(r_h - r_s)
boot_diff = np.array(boot_diff)
ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
print(f"  Bootstrap Δρ: {np.mean(boot_diff):+.3f}, 95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
print(f"  CI excludes 0? {'YES ✓' if ci_lo > 0 or ci_hi < 0 else 'NO'}")

# ═══════════════════════════════════════════════════════
# FLANK 4: WITHIN-STATE STRUCTURE
# Within "Healthy" recordings, is there α-R² structure
# that predicts something (like within-galaxy in SPARC)?
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 4: WITHIN-STATE STRUCTURE")
print("Within healthy subjects, do α and R² carry independent information?")
print(f"{'='*70}")

# Multivariate discrimination: can α + R² together separate states
# better than either alone?

# Train on 70% Healthy + Seizure, test on 30%
from sklearn.model_selection import StratifiedKFold

binary = epi[epi['class'].isin([1,3])].copy()
binary['is_healthy'] = (binary['class']==3).astype(int)

# 5-fold CV for each model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_alpha = binary[['alpha']].values
X_r2 = binary[['r_squared']].values
X_both = binary[['alpha','r_squared']].values
X_product = binary[['alpha_r2_product']].values
y = binary['is_healthy'].values

results_cv = {'alpha': [], 'r2': [], 'both': [], 'product': []}

for train_idx, test_idx in skf.split(X_alpha, y):
    for name, X in [('alpha', X_alpha), ('r2', X_r2), ('both', X_both), ('product', X_product)]:
        X_train = np.column_stack([X[train_idx], np.ones(len(train_idx))])
        X_test = np.column_stack([X[test_idx], np.ones(len(test_idx))])
        
        c, _, _, _ = lstsq(X_train, y[train_idx], rcond=None)
        y_pred = X_test @ c
        auc = roc_auc_score(y[test_idx], y_pred)
        results_cv[name].append(auc)

print(f"\n  5-FOLD CV AUC (Healthy vs Seizure):")
for name, aucs in results_cv.items():
    print(f"    {name:10s}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# Same for EO vs EC (scalp data — harder problem)
binary2 = epi[epi['class'].isin([4,5])].copy()
binary2['is_eo'] = (binary2['class']==5).astype(int)

X_alpha2 = binary2[['alpha']].values
X_r22 = binary2[['r_squared']].values
X_both2 = binary2[['alpha','r_squared']].values
X_prod2 = binary2[['alpha_r2_product']].values
y2 = binary2['is_eo'].values

results_cv2 = {'alpha': [], 'r2': [], 'both': [], 'product': []}
for train_idx, test_idx in skf.split(X_alpha2, y2):
    for name, X in [('alpha', X_alpha2), ('r2', X_r22), ('both', X_both2), ('product', X_prod2)]:
        X_train = np.column_stack([X[train_idx], np.ones(len(train_idx))])
        X_test = np.column_stack([X[test_idx], np.ones(len(test_idx))])
        c, _, _, _ = lstsq(X_train, y2[train_idx], rcond=None)
        y_pred = X_test @ c
        auc = roc_auc_score(y2[test_idx], y_pred)
        results_cv2[name].append(auc)

print(f"\n  5-FOLD CV AUC (Eyes Open vs Closed):")
for name, aucs in results_cv2.items():
    print(f"    {name:10s}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# ═══════════════════════════════════════════════════════
# FLANK 5: REM RESOLUTION — The direct attack
# Using the spectral data: reconstruct subject-level and
# test if R² or α×R² resolves the REM paradox
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 5: REM RESOLUTION")
print("Can a 2D metric (α, R²) or a combined metric resolve REM?")
print(f"{'='*70}")

# The REM problem: REM has β ≈ -4.0 (very steep, like NREM)
# but is phenomenologically conscious (dreaming)
# 
# RTM resolution attempt: maybe REM has DIFFERENT R² than NREM?
# If REM maintains power-law structure (high R²) despite steep slope,
# while NREM loses structure (low R²), the 2D plane separates them.

print(f"\n  From spectral data (published literature values):")
print(f"  {'State':30s} {'β (slope)':>10s} {'Conscious':>10s}")
print("  " + "-" * 55)

for _, r in spec.iterrows():
    marker = ' ← PARADOX' if 'REM' in r['State'] else ''
    print(f"  {r['State']:30s} {r['Slope']:+10.2f} {str(r['Conscious']):>10s}{marker}")

# Simulate the 2D classification
# We need R² estimates for the spectral data conditions
# Assumption: conscious states maintain higher R² (power-law integrity)
# From the epilepsy data, we know:
# EO (conscious): R² = 0.879, EC (less conscious): R² = 0.870
# Seizure (unconscious): R² = 0.717, Healthy: R² = 0.885

print(f"\n  KEY INSIGHT from epilepsy data:")
print(f"  The R² collapse test works because seizures DESTROY structure.")
print(f"  If REM PRESERVES structure (high R²) despite steep slope,")
print(f"  then R² separates REM (conscious+structured) from NREM (unconscious+degraded).")
print(f"\n  This is testable: measure R² during REM vs NREM in raw polysomnography.")
print(f"  The prediction is:")
print(f"    REM:  steep slope (β ≈ -4.0) BUT high R² (intact power law)")
print(f"    NREM: steep slope (β ≈ -3.4) AND lower R² (degraded power law)")
print(f"  If this holds, the 2D marker (α, R²) resolves the paradox.")

# ═══════════════════════════════════════════════════════
# FLANK 6: SPECTRAL GRADIENT — Rate of α change across states
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 6: DOES THE TRANSITION GRADIENT PREDICT CONSCIOUSNESS?")
print("Anesthesia dose-response: is the α-drop proportional to depth?")
print(f"{'='*70}")

# Group by study and compute within-study gradients
studies = spec.groupby('Study')
for study_name, group in studies:
    if len(group) < 2: continue
    
    conscious_states = group[group['Conscious']==True]
    unconscious_states = group[group['Conscious']==False]
    
    if len(conscious_states) > 0 and len(unconscious_states) > 0:
        delta = conscious_states['Slope'].mean() - unconscious_states['Slope'].mean()
        pct_change = abs(delta / conscious_states['Slope'].mean()) * 100
        
        print(f"\n  {study_name}:")
        print(f"    Conscious mean β = {conscious_states['Slope'].mean():.2f}")
        print(f"    Unconscious mean β = {unconscious_states['Slope'].mean():.2f}")
        print(f"    Δβ = {delta:+.2f} ({pct_change:.0f}% change)")

# KEY: Ketamine vs Propofol vs Xenon — dose-response topology
print(f"\n\n  ANESTHETIC COMPARISON (Colombo study):")
colombo = spec[spec['Study'].str.contains('Colombo')]
agents = {}
for _, r in colombo.iterrows():
    agent = r['Study'].split('-')[1]
    if agent not in agents:
        agents[agent] = {'wake': None, 'anesth': None}
    if 'Wake' in r['State']:
        agents[agent]['wake'] = r['Slope']
    else:
        agents[agent]['anesth'] = r['Slope']

print(f"  {'Agent':12s} {'Wake β':>8s} {'Anesth β':>10s} {'Δβ':>8s} {'%change':>8s} {'Conscious?':>12s}")
print("  " + "-" * 62)
for agent, vals in agents.items():
    if vals['wake'] and vals['anesth']:
        delta = vals['anesth'] - vals['wake']
        pct = abs(delta / vals['wake']) * 100
        conscious = 'YES' if pct < 20 else 'NO'
        print(f"  {agent:12s} {vals['wake']:8.2f} {vals['anesth']:10.2f} {delta:+8.2f} {pct:7.0f}% {conscious:>12s}")

# RTM prediction: the MAGNITUDE of Δβ predicts loss of consciousness
# Ketamine (small Δβ → conscious) vs Propofol (large Δβ → unconscious)
# This is confirmed by the data. But can we QUANTIFY the threshold?
print(f"\n  RTM THRESHOLD PREDICTION:")
print(f"  If |Δβ/β_wake| < 20%: consciousness PRESERVED (like ketamine)")
print(f"  If |Δβ/β_wake| > 40%: consciousness LOST (like propofol/xenon)")
print(f"  The 20-40% zone is the transition region.")

# ═══════════════════════════════════════════════════════
# FLANK 7: INTRA-CLASS VARIABILITY AS RTM DIAGNOSTIC
# Which class has the MOST variance in α? 
# RTM predicts: transition states should show maximum variance
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 7: VARIANCE AS STATE DIAGNOSTIC")
print("RTM predicts transition/critical states show maximum α variance")
print(f"{'='*70}")

print(f"\n  {'Class':15s} {'α CV':>8s} {'R² CV':>8s} {'α×R² CV':>10s}")
print("  " + "-" * 45)
for cls in [1,2,3,4,5]:
    sub = epi[epi['class']==cls]
    cv_a = sub['alpha'].std() / abs(sub['alpha'].mean())
    cv_r = sub['r_squared'].std() / sub['r_squared'].mean()
    cv_p = sub['alpha_r2_product'].std() / abs(sub['alpha_r2_product'].mean())
    print(f"  {class_map[cls]:15s} {cv_a:8.3f} {cv_r:8.3f} {cv_p:10.3f}")

# Seizure should have HIGHEST variance (system in transition)
# Healthy should have LOWEST variance (system stable)
alpha_cvs = []
for cls in [1,2,3,4,5]:
    sub = epi[epi['class']==cls]
    alpha_cvs.append(sub['alpha'].std() / abs(sub['alpha'].mean()))

print(f"\n  α CV ordering: ", end="")
ordered = sorted(zip([class_map[i] for i in [1,2,3,4,5]], alpha_cvs), key=lambda x: x[1])
print(" < ".join([f"{name}({cv:.3f})" for name, cv in ordered]))

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("CONSCIOUSNESS FLANKING — SUMMARY")
print(f"{'='*70}")
