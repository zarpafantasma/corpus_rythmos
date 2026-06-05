#!/usr/bin/env python3
"""
RTM ECOLOGY FLANKING CAMPAIGN
===============================
Five flanks using available data from Doc 012 package.
"""
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
from scipy.stats import f as f_dist
from scipy.signal import welch
import json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("RTM ECOLOGY — FLANKING CAMPAIGN")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════
anage = pd.read_csv("/home/claude/012_h/AnAge_Longevity Database_Analysis/anage_data.txt",
                    sep='\t', encoding='latin-1')

# Clean AnAge
for col in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)',
            'Temperature (K)', 'Birth weight (g)', 'Weaning weight (g)',
            'Litter/Clutch size', 'Litters/Clutches per year',
            'Inter-litter/Interbirth interval']:
    if col in anage.columns:
        anage[col] = pd.to_numeric(anage[col], errors='coerce')

anage_clean = anage.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)'])
anage_clean = anage_clean[(anage_clean['Body mass (g)'] > 0) & (anage_clean['Maximum longevity (yrs)'] > 0)]
anage_clean['log_M'] = np.log10(anage_clean['Body mass (g)'])
anage_clean['log_L'] = np.log10(anage_clean['Maximum longevity (yrs)'])

# Add BMR where available
anage_bmr = anage_clean.dropna(subset=['Metabolic rate (W)'])
anage_bmr = anage_bmr[anage_bmr['Metabolic rate (W)'] > 0]
anage_bmr['log_BMR'] = np.log10(anage_bmr['Metabolic rate (W)'])

print(f"  AnAge total: {len(anage_clean)} species")
print(f"  AnAge with BMR: {len(anage_bmr)} species")
print(f"  Classes: {anage_clean['Class'].value_counts().to_dict()}")

# Isle Royale
ir = pd.read_csv("/home/claude/012_h/RTM_Ecology_Population_Dynamics/isle_royale_data.csv")
print(f"  Isle Royale: {len(ir)} years ({ir['year'].min()}-{ir['year'].max()})")

# GPDD
gpdd = pd.read_csv("/home/claude/012_h/RTM_Ecology_Population_Dynamics/gpdd_spectral.csv")
print(f"  GPDD: {len(gpdd)} taxon groups, {gpdd['n_series'].sum()} series")

# ═══════════════════════════════════════════════════════
# FLANK 1: TOPOLOGICAL COHERENCE — Kleiber RESIDUALS
# RTM predicts: at fixed mass, metabolic TOPOLOGY (not just rate)
# determines longevity. Different orders with different vascular
# networks should show different residual patterns.
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 1: KLEIBER RESIDUALS — Does Metabolic Topology Predict Longevity?")
print("At fixed mass, does BMR predict longevity DIFFERENTLY by order?")
print(f"{'='*70}")

# Global Kleiber: log_BMR ~ log_M
mammals_bmr = anage_bmr[anage_bmr['Class'] == 'Mammalia'].copy()
print(f"\n  Mammals with BMR: {len(mammals_bmr)}")

# Kleiber residuals
kleiber = stats.linregress(mammals_bmr['log_M'], mammals_bmr['log_BMR'])
mammals_bmr['kleiber_resid'] = mammals_bmr['log_BMR'] - (kleiber.slope * mammals_bmr['log_M'] + kleiber.intercept)
print(f"  Kleiber: BMR ~ M^{kleiber.slope:.3f}, R² = {kleiber.rvalue**2:.3f}")

# Longevity residuals (from mass)
long_mass = stats.linregress(mammals_bmr['log_M'], mammals_bmr['log_L'])
mammals_bmr['longevity_resid'] = mammals_bmr['log_L'] - (long_mass.slope * mammals_bmr['log_M'] + long_mass.intercept)

# KEY TEST: Do Kleiber residuals predict longevity residuals?
# This asks: at FIXED MASS, does metabolic efficiency predict lifespan?
rho_kl, p_kl = stats.spearmanr(mammals_bmr['kleiber_resid'], mammals_bmr['longevity_resid'])
print(f"\n  GLOBAL: Spearman(Kleiber_resid, Longevity_resid) = {rho_kl:+.3f}, p = {p_kl:.2e}")

# Does this vary by ORDER?
print(f"\n  BY ORDER (n >= 10):")
print(f"  {'Order':25s} {'n':>4s} {'ρ(BMR_resid, L_resid)':>24s} {'p':>10s}")
print("  " + "-" * 68)

order_results = []
for order in mammals_bmr['Order'].value_counts().index:
    sub = mammals_bmr[mammals_bmr['Order'] == order]
    if len(sub) < 10: continue
    
    rho_o, p_o = stats.spearmanr(sub['kleiber_resid'], sub['longevity_resid'])
    sig = '*' if p_o < 0.05 else ''
    print(f"  {order:25s} {len(sub):4d} {rho_o:+20.3f}{sig:>4s} {p_o:10.4f}")
    order_results.append({'order': order, 'n': len(sub), 'rho': rho_o, 'p': p_o})

odf = pd.DataFrame(order_results)
print(f"\n  Mean within-order ρ: {odf['rho'].mean():+.3f}")
print(f"  % negative: {100*np.mean(odf['rho'] < 0):.0f}%")
t_within, p_within = stats.ttest_1samp(odf['rho'], 0)
print(f"  t-test vs 0: t = {t_within:.2f}, p = {p_within:.4f}")

# RTM-specific: does the STRENGTH of BMR-longevity coupling vary 
# with order-level metabolic complexity?
# Proxy for vascular complexity: average mass (larger animals = more complex vasculature)
odf['mean_mass'] = [mammals_bmr[mammals_bmr['Order']==o]['log_M'].mean() for o in odf['order']]
rho_complexity, p_complexity = stats.spearmanr(odf['mean_mass'], odf['rho'])
print(f"\n  Does body size (complexity proxy) predict coupling strength?")
print(f"  Spearman(mean_mass, ρ_coupling) = {rho_complexity:+.3f}, p = {p_complexity:.4f}")

# ═══════════════════════════════════════════════════════
# FLANK 2: PREDATOR-PREY SHAPE CONSPIRACY (Isle Royale)
# Analog of the baryon-halo conspiracy in SPARC
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 2: PREDATOR-PREY SHAPE CONSPIRACY (Isle Royale)")
print("Does wolf population SHAPE track moose population SHAPE?")
print("Does this coupling break before ecosystem crashes?")
print(f"{'='*70}")

wolves = ir['wolves'].values.astype(float)
moose = ir['moose'].values.astype(float)
years = ir['year'].values

# Normalize to unit amplitude
wolves_norm = wolves / np.max(wolves)
moose_norm = moose / np.max(moose)

# Global shape correlation
r_shape, p_shape = stats.pearsonr(wolves_norm, moose_norm)
print(f"\n  Global shape correlation: r = {r_shape:+.3f}, p = {p_shape:.4f}")

# Rolling shape correlation (15-year window)
window = 15
rolling_conspiracy = []
for i in range(window, len(wolves)+1):
    w = wolves[i-window:i]
    m = moose[i-window:i]
    # Normalize within window
    w_n = (w - w.min()) / (w.max() - w.min() + 1e-10)
    m_n = (m - m.min()) / (m.max() - m.min() + 1e-10)
    r, p = stats.pearsonr(w_n, m_n)
    rolling_conspiracy.append({
        'year': years[i-1], 'r': r, 'p': p,
        'wolf_mean': np.mean(w), 'moose_mean': np.mean(m)
    })

rcdf = pd.DataFrame(rolling_conspiracy)

print(f"\n  Rolling shape conspiracy (15-yr window):")
print(f"  Mean r = {rcdf['r'].mean():+.3f} ± {rcdf['r'].std():.3f}")

# KEY: Does conspiracy CHANGE before crashes?
crashes = {
    'Wolf_1980': {'year': 1980, 'description': 'Parvovirus'},
    'Moose_1996': {'year': 1996, 'description': 'Vegetation crash'},
    'Wolf_2012': {'year': 2012, 'description': 'Inbreeding collapse'}
}

print(f"\n  CONSPIRACY BEFORE vs AFTER CRASHES:")
for name, info in crashes.items():
    cy = info['year']
    pre = rcdf[(rcdf['year'] >= cy-8) & (rcdf['year'] < cy)]
    post = rcdf[(rcdf['year'] >= cy) & (rcdf['year'] < cy+8)]
    baseline = rcdf[rcdf['year'] < cy-10]
    
    if len(pre) >= 3 and len(baseline) >= 3:
        d_pre = (pre['r'].mean() - baseline['r'].mean()) / np.sqrt((pre['r'].var(ddof=1) + baseline['r'].var(ddof=1))/2) if baseline['r'].std() > 0 else np.nan
        print(f"  {name} ({cy}):")
        print(f"    Baseline r = {baseline['r'].mean():+.3f} (n={len(baseline)})")
        print(f"    Pre-crash r = {pre['r'].mean():+.3f} (n={len(pre)})")
        if len(post) >= 3:
            print(f"    Post-crash r = {post['r'].mean():+.3f} (n={len(post)})")
        if not np.isnan(d_pre):
            t_pc, p_pc = stats.ttest_ind(pre['r'], baseline['r'])
            print(f"    d(pre vs baseline) = {d_pre:+.3f}, p = {p_pc:.4f}")
            if d_pre < -0.3 and p_pc < 0.1:
                print(f"    → Conspiracy WEAKENS before crash ✓ (RTM prediction)")
            elif d_pre > 0.3:
                print(f"    → Conspiracy STRENGTHENS before crash (opposite)")
            else:
                print(f"    → No significant change")

# Cross-correlation at multiple lags
print(f"\n  LAGGED SHAPE CORRELATION (who leads?):")
for lag in range(-5, 6):
    if lag == 0:
        r, p = stats.pearsonr(wolves_norm, moose_norm)
    elif lag > 0:
        r, p = stats.pearsonr(wolves_norm[lag:], moose_norm[:-lag])
    else:
        r, p = stats.pearsonr(wolves_norm[:lag], moose_norm[-lag:])
    sig = '*' if p < 0.05 else ''
    print(f"    Lag {lag:+2d} (wolf leads by {abs(lag)}yr): r = {r:+.3f} {sig}")

# ═══════════════════════════════════════════════════════
# FLANK 3: β FIRST-HALF PREDICTS VARIANCE SECOND-HALF
# Out-of-sample test using Isle Royale
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 3: OUT-OF-SAMPLE — Does β Predict Future Instability?")
print("Split time series in half. Does first-half β predict second-half variance?")
print(f"{'='*70}")

# For Isle Royale: split at different points
for species, series in [('Wolves', wolves), ('Moose', moose)]:
    print(f"\n  {species}:")
    
    # Try multiple split points
    splits = [0.4, 0.5, 0.6]
    for split_frac in splits:
        split_idx = int(len(series) * split_frac)
        first_half = series[:split_idx]
        second_half = series[split_idx:]
        
        # β of first half
        f1, psd1 = welch(first_half - first_half.mean(), fs=1.0, nperseg=min(8, len(first_half)//2))
        mask1 = (f1 > 0) & (psd1 > 0)
        if mask1.sum() >= 3:
            s1, _, _, _, _ = stats.linregress(np.log(f1[mask1]), np.log(psd1[mask1]))
            beta_first = -s1
        else:
            continue
        
        # Variance and CV of second half
        var_second = np.var(second_half)
        cv_second = np.std(second_half) / np.mean(second_half)
        
        # Also: did a crash happen in second half?
        max_drop = 0
        for i in range(1, len(second_half)):
            drop = (np.max(second_half[:i]) - second_half[i]) / np.max(second_half[:i])
            max_drop = max(max_drop, drop)
        
        yr_split = years[split_idx]
        print(f"    Split at {yr_split} ({split_frac:.0%}): β_first = {beta_first:.2f}, "
              f"CV_second = {cv_second:.2f}, max_drop = {max_drop:.0%}")

# Now do this systematically with GPDD data if available
# Use rolling β to predict rolling variance (shifted forward)
print(f"\n  ROLLING PREDICTION (wolves, 15yr windows, 5yr shift):")
betas = []
future_cvs = []
for i in range(15, len(wolves) - 5):
    chunk = wolves[i-15:i]
    future = wolves[i:i+5]
    
    f_w, psd_w = welch(chunk - chunk.mean(), fs=1.0, nperseg=min(8, len(chunk)//2))
    mask = (f_w > 0) & (psd_w > 0)
    if mask.sum() < 3: continue
    
    s, _, _, _, _ = stats.linregress(np.log(f_w[mask]), np.log(psd_w[mask]))
    beta = -s
    future_cv = np.std(future) / (np.mean(future) + 1e-10)
    
    betas.append(beta)
    future_cvs.append(future_cv)

betas = np.array(betas)
future_cvs = np.array(future_cvs)

rho_pred, p_pred = stats.spearmanr(betas, future_cvs)
print(f"  Spearman(β_past, CV_future): ρ = {rho_pred:+.3f}, p = {p_pred:.4f}")
if rho_pred > 0 and p_pred < 0.05:
    print(f"  → REDDER noise → MORE future variability ✓ (RTM prediction)")
elif rho_pred > 0:
    print(f"  → Correct direction, not significant")
else:
    print(f"  → Wrong direction")

# Same for moose
betas_m = []
future_cvs_m = []
for i in range(15, len(moose) - 5):
    chunk = moose[i-15:i]
    future = moose[i:i+5]
    
    f_m, psd_m = welch(chunk - chunk.mean(), fs=1.0, nperseg=min(8, len(chunk)//2))
    mask = (f_m > 0) & (psd_m > 0)
    if mask.sum() < 3: continue
    
    s, _, _, _, _ = stats.linregress(np.log(f_m[mask]), np.log(psd_m[mask]))
    beta = -s
    future_cv = np.std(future) / (np.mean(future) + 1e-10)
    
    betas_m.append(beta)
    future_cvs_m.append(future_cv)

betas_m = np.array(betas_m)
future_cvs_m = np.array(future_cvs_m)

rho_pred_m, p_pred_m = stats.spearmanr(betas_m, future_cvs_m)
print(f"\n  Moose: Spearman(β_past, CV_future): ρ = {rho_pred_m:+.3f}, p = {p_pred_m:.4f}")

# ═══════════════════════════════════════════════════════
# FLANK 4: AMPHIBIA — THE HONEST BOUNDARY TEST
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 4: AMPHIBIA — Does Vascular Simplicity Explain Low α?")
print(f"{'='*70}")

amphibia = anage_clean[anage_clean['Class'] == 'Amphibia'].copy()
print(f"\n  Total Amphibia: {len(amphibia)}")
print(f"  Orders: {amphibia['Order'].value_counts().to_dict()}")

# Anura (frogs/toads) vs Caudata (salamanders/newts)
# Anura have more developed lungs; Caudata rely more on cutaneous respiration
for order in ['Anura', 'Caudata']:
    sub = amphibia[amphibia['Order'] == order]
    if len(sub) < 5: continue
    ols = stats.linregress(sub['log_M'], sub['log_L'])
    print(f"  {order:15s} (n={len(sub):3d}): α = {ols.slope:.4f} ± {ols.stderr:.4f}, R² = {ols.rvalue**2:.3f}")

# Compare with reptiles (more complex vasculature)
reptilia = anage_clean[anage_clean['Class'] == 'Reptilia']
ols_rep = stats.linregress(reptilia['log_M'], reptilia['log_L'])
ols_amph = stats.linregress(amphibia['log_M'], amphibia['log_L'])
print(f"\n  Amphibia overall: α = {ols_amph.slope:.3f}, R² = {ols_amph.rvalue**2:.3f}")
print(f"  Reptilia overall: α = {ols_rep.slope:.3f}, R² = {ols_rep.rvalue**2:.3f}")

# Are they significantly different?
z_ar = (ols_rep.slope - ols_amph.slope) / np.sqrt(ols_rep.stderr**2 + ols_amph.stderr**2)
p_ar = 2 * (1 - stats.norm.cdf(abs(z_ar)))
print(f"  α difference: z = {z_ar:.2f}, p = {p_ar:.4f}")

# Taxonomic complexity ladder
print(f"\n  COMPLEXITY LADDER (RTM prediction: α increases with vascular complexity):")
for cls in ['Amphibia', 'Reptilia', 'Mammalia', 'Aves']:
    sub = anage_clean[anage_clean['Class'] == cls]
    if len(sub) < 5: continue
    ols = stats.linregress(sub['log_M'], sub['log_L'])
    print(f"    {cls:12s} (n={len(sub):3d}): α = {ols.slope:.4f}")

# Is the ladder monotonic with a known complexity ranking?
# Amphibia (3-chamber heart) < Reptilia (3.5-chamber) < Mammalia (4-chamber) < Aves (4-chamber + air sacs)
alphas = []
for cls in ['Amphibia', 'Reptilia', 'Mammalia', 'Aves']:
    sub = anage_clean[anage_clean['Class'] == cls]
    if len(sub) < 5: continue
    ols = stats.linregress(sub['log_M'], sub['log_L'])
    alphas.append(ols.slope)

complexity_rank = [1, 2, 3, 4]  # simple ordering
rho_ladder, p_ladder = stats.spearmanr(complexity_rank[:len(alphas)], alphas)
print(f"\n  Complexity rank vs α: ρ = {rho_ladder:+.3f}, p = {p_ladder:.4f}")
print(f"  (Expected: positive = more complex → higher α)")

# ═══════════════════════════════════════════════════════
# FLANK 5: CROSS-TAXON β vs BODY SIZE (GPDD)
# RTM predicts: larger organisms should have redder noise
# (higher β) because they have more topological layers
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK 5: DOES BODY SIZE PREDICT SPECTRAL COLOR? (GPDD)")
print("RTM: more topological layers → redder noise (higher β)")
print(f"{'='*70}")

# GPDD has taxon groups with beta_mean and n_series
# We need to link to typical body mass
# Approximate body mass for each taxon group
body_mass_approx = {
    'Insects': 0.01, 'Arachnids': 0.005, 'Crustaceans': 0.1,
    'Fish': 100, 'Amphibians': 20, 'Reptiles': 500,
    'Small mammals': 50, 'Large mammals': 50000,
    'Small birds': 30, 'Large birds': 3000,
    'Marine invertebrates': 10, 'Molluscs': 50
}

gpdd['log_mass_approx'] = gpdd['taxon_group'].map(
    lambda x: np.log10(body_mass_approx.get(x, np.nan)))

gpdd_mass = gpdd.dropna(subset=['log_mass_approx'])
print(f"\n  Taxon groups with mass estimate: {len(gpdd_mass)}")

if len(gpdd_mass) >= 5:
    rho_bm, p_bm = stats.spearmanr(gpdd_mass['log_mass_approx'], gpdd_mass['beta_mean'])
    print(f"  Spearman(log_mass, β): ρ = {rho_bm:+.3f}, p = {p_bm:.4f}")
    
    # Weighted by n_series
    weights = gpdd_mass['n_series'].values
    # Weighted Spearman approximation: use weighted rank
    print(f"\n  Per taxon group:")
    for _, r in gpdd_mass.sort_values('log_mass_approx').iterrows():
        print(f"    {r['taxon_group']:25s}: mass~{10**r['log_mass_approx']:.0f}g, β = {r['beta_mean']:.2f} (n={int(r['n_series'])})")

# Also check habitat effect
print(f"\n  HABITAT EFFECT:")
for hab in gpdd['habitat'].unique():
    sub = gpdd[gpdd['habitat'] == hab]
    if len(sub) >= 2:
        print(f"    {hab:20s}: mean β = {sub['beta_mean'].mean():.3f} (n={len(sub)} groups, {sub['n_series'].sum()} series)")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("ECOLOGY FLANKING CAMPAIGN — SUMMARY")
print(f"{'='*70}")

results = {
    "flank_1_kleiber_residuals": {
        "global_rho": round(rho_kl, 3),
        "global_p": float(f"{p_kl:.2e}"),
        "n_orders_tested": len(odf),
        "mean_within_order_rho": round(odf['rho'].mean(), 3),
        "complexity_correlation": round(rho_complexity, 3)
    },
    "flank_2_shape_conspiracy": {
        "global_shape_r": round(r_shape, 3),
        "rolling_mean_r": round(rcdf['r'].mean(), 3)
    },
    "flank_3_out_of_sample": {
        "wolves_rho_beta_cv": round(rho_pred, 3),
        "wolves_p": round(p_pred, 4),
        "moose_rho_beta_cv": round(rho_pred_m, 3),
        "moose_p": round(p_pred_m, 4)
    },
    "flank_4_amphibia": {
        "complexity_ladder_rho": round(rho_ladder, 3),
        "complexity_ladder_p": round(p_ladder, 4)
    }
}

with open('/home/claude/rtm_eco_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved.")
