# Red Team Validation Report: Document 011 — Conscious Access

**RTM Corpus — Independent Verification**  
**Date:** April 28, 2026  
**Scope:** All ROBUST empirical validations in Doc 011

---

## 1. What Document 011 Claims

Doc 011 proposes that conscious access occurs when a cortical network crosses a multiscale coherence threshold. Two signatures:

- **S1:** The RTM scaling slope (spectral slope β) separates conscious from unconscious states.
- **S2:** Net Directionality Index (NDI) — forward information flow — accompanies consciousness.

Three validation layers:

- **Appendix A (Computational):** Simulated classification using stipulated α and NDI values.
- **Appendix B (Empirical — ROBUST):** Monte Carlo subject-level reconstruction from published EEG spectral slope data across 14 conditions (N=30,873), including the "ketamine dissociation."
- **Epilepsy (ROBUST):** RTM analysis of the UCI Epileptic Seizure Recognition dataset (N=11,500 recordings).

---

## 2. Verification Methodology

I independently:

1. Read all source CSVs, Python scripts, and reports in the package.
2. Re-ran the Monte Carlo subject-level simulation with the same seed (42) and verified numerical reproduction.
3. Ran 10 additional seeds to test stability.
4. Performed a **new within-study meta-analysis** not present in the original package.
5. Verified the epilepsy dataset statistics independently.

---

## 3. Findings — Claim by Claim

### 3.1 Wake vs True Unconscious Separation (d=0.46)

| Metric | Claimed | Reproduced |
|--------|---------|------------|
| Cohen's d | 0.46 | **0.4614** |
| p-value | < 10⁻¹⁰ | **4.61 × 10⁻²²⁹** |
| AUC | not stated for this subset | **0.6284** |

**Verdict: NUMERICALLY REPRODUCED. ✓**

The d=0.46 is real and stable across 10 random seeds (range: 0.44–0.49). Statistical significance overwhelming because N ≈ 10,000 per group (dominated by NSRR).

**Critical context the report undersells:**

- AUC = 0.63 means classifying a random Wake vs Unconscious subject works only 63% of the time. Far from "strict bifurcation."
- Cohen's d = 0.46 is conventionally **small-to-medium**. "Triumphantly" is not warranted.
- The massive p-value comes from N=10,255 per group. Even trivial differences produce p < 10⁻²⁰⁰ at that N. Effect size matters, not p-value.

**For RTM: POSITIVE.** Direction correct and reproducible. Wake states have less steep spectral slopes than unconscious states. But distribution overlap is large.

### 3.2 Universal Classifier Including REM (d=0.11, AUC=0.51)

When REM sleep is included in the "conscious" category:

| Metric | Value |
|--------|-------|
| Cohen's d | 0.108 |
| AUC | 0.507 |

**Verdict: ESSENTIALLY RANDOM.** Classifier fails completely when REM included.

Authors acknowledge this ("aggregation fallacy"). REM is paradoxical: phenomenologically conscious but steep slopes like NREM. Their solution: exclude REM and analyze Wake vs True Unconscious.

**For RTM: PROBLEMATIC but not fatal.** REM breaks most EEG-based classifiers, not just RTM. But S1 becomes a **wakefulness** marker more than a **consciousness** marker. Theory needs to explain why REM has "unconscious-like" spectral topology.

### 3.3 Ketamine vs Propofol Dissociation

| Metric | Claimed | Reproduced |
|--------|---------|------------|
| Ketamine slope | −1.76 | **−1.762** |
| Propofol slope | −3.23 | **−3.232** |
| p-value | 0.002 (CSV) | **0.0032** |
| Cohen's d | not reported | **2.81** |

**Verdict: DIRECTION CORRECT, NUMERICALLY REPRODUCED. ✓**

Separation genuine and large (d=2.81). Ketamine preserves wake-like slopes; propofol collapses them. Aligns with RTM predictions and clinical phenomenology.

**Critical caveat: n=5 per group.** Single small study (Colombo). Effect enormous but CI extremely wide. "Definitively proves" is premature with n=5.

**For RTM: STRONGLY POSITIVE in direction.** Prediction correct, effect large. Language should be "strongly suggestive" not "definitively proves."

### 3.4 Epilepsy — iEEG R² Topological Collapse

| Metric | Claimed | Reproduced |
|--------|---------|------------|
| Cohen's d (Seizure vs Healthy R²) | −1.55 | **−1.5554** |
| Seizure R² mean | ~0.71 | **0.7165** |
| Healthy R² mean | ~0.88 | **0.8845** |
| p-value | < 0.0001 | **< 10⁻³⁰⁰** |

**Verdict: EXACTLY REPRODUCED. ✓**

**Strongest finding in the entire document.** Large effect (d=1.55), substantial sample (n=2,300 per group), conceptually clean: seizures destroy power-law EEG structure, measurable as R² collapse. Directly validates RTM's "collapse test" from Doc 002.

**For RTM: STRONGLY POSITIVE.** Genuine empirical validation with real data (UCI dataset), not simulation. The holonomic collapse interpretation maps cleanly onto Doc 002 framework.

### 3.5 Scalp EEG — Eyes Open vs Closed (Filtered)

| Metric | Claimed | Reproduced |
|--------|---------|------------|
| Cohen's d (unfiltered) | 0.33 | **0.3305** |
| Cohen's d (R²>0.6 filter) | 0.39 | **0.3901** |
| p-value (filtered) | < 10⁻²¹ | **7.9 × 10⁻²²** |

**Verdict: EXACTLY REPRODUCED. ✓**

R²-based quality filter improves the consciousness signal. Improvement from d=0.33 to d=0.39 is modest but systematic.

**For RTM: POSITIVE.** Small but real and reproducible.

---

## 4. New Red Team Validation: Within-Study Meta-Analysis

**Rationale:** Pooled d=0.46 is dominated by NSRR (n=10,255). To test direction consistency across independent studies, I computed Cohen's d within each study and meta-analyzed.

| Study | n_wake | n_uncon | Cohen's d |
|-------|--------|---------|-----------|
| Propofol-Scalp | 9 | 9 | +1.65 |
| Sleep-Scalp | 20 | 20 | +2.40 |
| Colombo-Xenon | 5 | 5 | +1.86 |
| Colombo-Propofol | 5 | 5 | +2.47 |
| Purcell-NSRR | 10,255 | 10,255 | +0.47 |

### Meta-analytic Results

| Metric | Value |
|--------|-------|
| **Direction consistency** | **5/5 studies (100%)** |
| Fixed-effect meta d | +0.47 (95% CI: [0.45, 0.50]) |
| Random-effect meta d | +1.69 (95% CI: [0.59, 2.79]) |
| Heterogeneity I² | **88.7%** (very high) |
| Meta p-value (FE) | < 10⁻¹⁰ |

**Interpretation:** All five independent studies show the same direction — wake slopes less steep than unconscious. Sign consistency (5/5) is a robust qualitative finding. I²=88.7% indicates heterogeneity in magnitude, expected across different EEG setups.

**For RTM: POSITIVE.** Direction universally consistent.

---

## 5. Methodological Issues

### 5.1 Simulated Data, Not Raw EEG (Acknowledged)

The "N=30,873" is Monte Carlo reconstruction from published means and SEMs — not 30,873 raw EEG recordings. Legitimate technique but assumes normal distribution. Authors are transparent about this.

### 5.2 REM Paradox — Unresolved

REM is phenomenologically conscious but has "unconscious-like" spectral slopes. Breaks universal classifier. Authors handle by exclusion — honest but leaves a theoretical gap.

### 5.3 Language Inflation

Numbers correct. Adjectives disproportionate:
- "Most triumphantly" for d=0.46
- "Definitively proves" with n=5
- "Strict mathematical precision" for AUC=0.63
- "Topology strictly bifurcates" with extensive overlap

### 5.4 Appendix A: Stipulated, Not Measured

Computational validation uses author-specified values. Consistency checks, not empirical evidence.

---

## 6. Seed Robustness

| Metric | Mean (10 seeds) | Range |
|--------|-----------------|-------|
| Cohen's d | 0.463 | [0.438, 0.491] |
| AUC | 0.629 | [0.621, 0.636] |

Stable. No seed dependence.

---

## 7. Overall Verdict

### POSITIVE for RTM

| Finding | Strength | Comment |
|---------|----------|---------|
| Epilepsy R² collapse (d=−1.55) | **STRONG** | Real data, large sample, large effect |
| Wake vs Unconscious direction (5/5) | **SOLID** | Universally consistent across studies |
| Ketamine dissociation (d=2.81) | **SOLID** | Correct prediction, caveat n=5 |
| Scalp EEG filtration (d=0.39) | **MODERATE** | Small but real |
| Reproducibility | **STRONG** | All numbers verified exactly |

### NEGATIVE or UNRESOLVED

| Issue | Severity | Comment |
|-------|----------|---------|
| REM paradox | **SIGNIFICANT GAP** | β fails when REM included |
| Modest discrimination (AUC=0.63) | **MODERATE** | Not a clinical classifier |
| Ketamine n=5 | **MODERATE** | Too small for definitive claims |
| Heterogeneity I²=88.7% | **MODERATE** | Magnitude unstable across studies |

### Bottom Line

**The ROBUST validations are methodologically honest and numerically correct.** Findings genuinely support RTM's qualitative predictions: spectral slope tracks structural coherence, seizures destroy scale-free topology, different anesthetics affect topology differently. The epilepsy finding is particularly strong and novel.

Main problems: rhetorical inflation and REM paradox. Neither invalidates the framework.

**The empirical findings are net POSITIVE for RTM.** Direction right, effects real, strongest finding (epilepsy R² collapse) genuinely novel. Framework needs refinement regarding REM, not rejection.

---

*Report generated independently. Computations reproducible via red_team_011.py and results_summary.json.*
