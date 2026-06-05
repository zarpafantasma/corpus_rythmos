# Red Team Validation — Doc 008
## Rhythmic Biochemistry: Protein Folding vs Enzyme Catalysis

**Independent Audit of RTM Biochemistry Empirical Validation**

---

## 1. Scope

Doc 008 tests RTM's discriminating power between two biological process types: **global topology-driven** (protein folding, predicted α >> 0) and **local chemistry-driven** (enzyme catalysis, predicted α ≈ 0). The original ROBUST validation reports folding α = 7.22 ± 0.62 and enzyme α = 0.26 ± 0.69.

---

## 2. Datasets

| Domain | n | Source | Scale proxy (L) | Time proxy (τ) |
|:---|:--|:--|:--|:--|
| Protein folding | 84 | Ivankov 2003, Maxwell 2005, etc. | Chain length (residues) | 1/k_f (folding time) |
| Enzyme kinetics | 69 | BRENDA database | Enzyme length (residues) | 1/k_cat (turnover time) |

---

## 3. Test Results

### Test 1 — Protein Folding

| Estimator | α | Notes |
|:--|:--|:--|
| OLS | 7.217 ± 0.617 | R² = 0.625, p = 3.6 × 10⁻¹⁹ |
| ODR (20% rate noise) | 7.218 ± 0.617 | |
| Theil-Sen | 7.551 [6.218, 8.921] | |
| Bootstrap (3000) | 7.255 ± 0.535 | CI [6.255, 8.379] |

By structure type:

| Type | n | α | R² |
|:--|:--|:--|:--|
| Alpha | 26 | 6.01 ± 0.89 | 0.653 |
| Beta | 29 | 7.53 ± 0.87 | 0.734 |
| Mixed | 29 | 7.16 ± 0.74 | 0.777 |

Leave-one-out range: [7.107, 7.580] — always positive, no single protein dominates.

The folding rate–chain length correlation is well-established in biophysics (Plaxco et al. 1998, Ivankov & Finkelstein 2004). RTM's contribution is not discovering this relationship but classifying it within a universal transport framework as "highly coherent/topology-driven."

**Original ROBUST:** α = 7.22 ± 0.62. **Reproduced exactly. ✓**

---

### Test 2 — Enzyme Kinetics

| Method | α | p (vs 0) |
|:--|:--|:--|
| Raw OLS (confounded) | 0.875 ± 0.588 | 0.142 |
| EC-normalized OLS | 0.256 ± 0.696 | 0.714 |
| EC-normalized ODR | 0.256 ± 0.690 | 0.712 |
| Bootstrap (EC-norm) | 0.218 ± 0.855 | CI [−1.515, 1.923] |

EC-class normalization is critical: it removes the confound that different enzyme classes (oxidoreductases vs hydrolases vs transferases) have inherently different k_cat ranges unrelated to enzyme size. After normalization, α is indistinguishable from zero (p = 0.71).

Per EC class, slopes vary wildly (from −4.2 for hydrolases to +5.2 for isomerases), confirming that k_cat is determined by chemistry, not enzyme size. This is exactly what RTM predicts for a "local" process.

**Original ROBUST:** α = 0.26 ± 0.69. **Reproduced exactly. ✓**

---

### Test 3 — Discriminating Power

| Metric | Value |
|:--|:--|
| Folding α (bootstrap) | 7.255 ± 0.535 |
| Enzyme α (bootstrap) | 0.218 ± 0.855 |
| Difference | 7.037 |
| Cohen's d | **6.98** |
| Bootstrap overlap | **0.0%** |

Cohen's d = 6.98 represents an extremely strong effect size. The two distributions do not overlap at all in 3,000 bootstrap iterations. RTM's α completely separates global processes from local ones.

**Verdict: STRONG DISCRIMINATION ✓**

---

### Test 4 — Literature Context

The protein folding rate–chain length relationship is canonical (Ivankov & Finkelstein 2004). The more precise predictor is absolute contact order (ACO), not just chain length. RTM's power-law τ ∝ L^α is a coarser proxy but correctly captures the directionality: longer chains fold slower, and the scaling is steep (α >> 1 reflects the topology-driven "folding funnel").

The enzyme result (α ≈ 0) is also well-established: catalytic rate depends on the active site microenvironment, not on how large the overall enzyme is. Carbonic anhydrase (260 residues, k_cat = 10⁶) and RNA polymerase (3300 residues, k_cat = 40) differ in k_cat by many orders of magnitude, but this reflects their chemistry, not their size.

---

### Test 5 — Sensitivity

| Domain | LOO α range | Stable? |
|:--|:--|:--|
| Folding | [7.107, 7.580] | YES — always > 7 ✓ |
| Enzyme (EC-norm) | [−0.162, 0.558] | Sign changes — expected for α ≈ 0 |

---

## 4. Assessment of Original ROBUST Validation

| Aspect | Assessment |
|:---|:---|
| Data sources | Ivankov 2003, BRENDA — canonical ✓ |
| EC-class normalization | Correct methodology ✓ |
| ODR with assay noise | Appropriate ✓ |
| Folding α = 7.22 ± 0.62 | Reproduced exactly ✓ |
| Enzyme α = 0.26 ± 0.69 | Reproduced exactly ✓ |
| "α as diagnostic" claim | Supported (Cohen's d = 6.98) ✓ |

**No methodological issues identified.**

---

## 5. Summary

| Test | Result | RTM-consistent? |
|:--|:--|:--|
| 1. Protein folding | α = 7.22 ± 0.62, CI [6.26, 8.38] | ✓ Topology-driven |
| 2. Enzyme kinetics | α = 0.26 ± 0.69, p = 0.71 | ✓ Local process |
| 3. Discrimination | Cohen's d = 6.98, 0% overlap | ✓ Clean separation |
| 4. Literature | Consistent with Ivankov, BRENDA | ✓ |
| 5. Sensitivity | Folding stable, enzyme ≈ 0 | ✓ |

---

## 6. Conclusion

This is one of the strongest validations in the RTM corpus. The framework makes a clear, testable prediction: global topology-driven processes (folding) should have high α, while local chemistry-driven processes (catalysis) should have α ≈ 0. The data confirm this with extremely strong effect size (Cohen's d = 6.98) and zero bootstrap overlap.

Both datasets come from canonical sources (Ivankov 2003 for folding, BRENDA for enzymes). The EC-class normalization correctly isolates the size effect from the chemistry effect. All numbers reproduce exactly.

**Overall assessment: POSITIVE for RTM. No issues found.**

---

*Report generated by independent red team audit. April 2026.*
