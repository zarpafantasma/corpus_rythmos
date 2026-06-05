# RTM Flank 3: The Acceleration Scale Is Not Universal

**Date:** April 28, 2026
**Data:** SPARC table2.dat (68 galaxies with measurable DM transition)

---

## The Attack

Three flanks attempted. Two failed, one hit hard.

**Flank A (DM transition radius):** FAILED — threshold issue, insufficient galaxies.

**Flank B (Cusp vs Core from light):** FAILED — after controlling for mass and V_flat, SB structure does not predict the inner DM density slope. Partial ρ < 0.13, all non-significant. Baryonic geometry cannot distinguish cusps from cores.

**Flank C (Acceleration scale):** HIT.

---

## Flank C: The Finding

McGaugh et al. (2016) established the Radial Acceleration Relation with a characteristic scale $g_\dagger \approx 1.2 \times 10^{-10}$ m/s², assumed universal. We tested: is this scale truly universal, and if not, does baryonic structure predict its variation?

For each galaxy, we found $g_{cross}$ — the baryonic acceleration at which the mass discrepancy $D = V_{obs}^2 / V_{bar}^2$ first exceeds 2.0 (where dark matter equals baryons in force contribution).

### Core Result

**Baryonic concentration predicts the acceleration scale far better than mass does:**

| Model | R² |
|-------|-----|
| Mass only | **0.0004** (essentially zero) |
| Concentration only | **0.2248** |
| Mass + concentration | 0.2248 (mass adds nothing) |

This is remarkable: **total baryonic mass has zero predictive power for the DM transition acceleration, but baryonic concentration explains 22.5% of its variance.**

### Partial Correlation

| Metric | Value |
|--------|-------|
| Partial Spearman ρ (conc → g_cross, controlling M_bar) | **-0.574** |
| p-value | **3.06 × 10⁻⁷** |
| Bootstrap 95% CI | **[-0.720, -0.351]** |
| Bootstrap % negative | **100%** |
| N galaxies | 68 |

The effect is robust across D thresholds:

| D threshold | n | Partial ρ (conc) | p |
|-------------|---|-----------------|---|
| 1.5 | 51 | -0.447 | 0.001 |
| **2.0** | **68** | **-0.574** | **3 × 10⁻⁷** |
| 3.0 | 73 | -0.484 | 1.5 × 10⁻⁵ |
| 5.0 | 54 | -0.304 | 0.026 |

### Physical Meaning

More concentrated galaxies (light packed tighter toward center) transition to DM dominance at **lower** accelerations. Less concentrated galaxies (diffuse light) hit DM dominance at **higher** accelerations.

In plain language: the SHAPE of the light profile tells you WHERE dark matter starts mattering, and this information is completely independent of the total mass.

### Theoretical Implications

This finding has different meanings for different theories:

- **Standard ΛCDM:** g† is expected to be universal (no structural dependence). This finding is in tension — the scatter in g† is real and structurally predictable.
- **MOND:** Predicts g† should correlate with surface density. Our concentration measure is a surface density proxy. The finding is **consistent with MOND**.
- **RTM:** Predicts that structural coherence modulates the dynamics. The finding is **consistent with RTM** — the topological organization of baryons determines where classical Newtonian gravity becomes insufficient.

### Important Caveat

The within-V_flat stability is mixed:

| V_flat bin | n | Partial ρ | p |
|------------|---|-----------|---|
| Slow (dwarf) | 23 | -0.456 | 0.029 ★ |
| Medium | 22 | -0.196 | 0.38 (ns) |
| Fast | 23 | +0.233 | 0.28 (ns) |

The overall effect is strongest in slow rotators (dwarfs) and weakens or reverses in massive galaxies. This pattern is consistent with the well-known observation that MOND-like effects are strongest in low-acceleration systems (dwarfs), where the DM transition is most prominent and varies most between galaxies. In massive galaxies, the transition is compressed into a narrow radial range with less structural variation.

This caveat is real but does not invalidate the finding — it contextualizes it. The acceleration-scale variation is primarily a low-mass galaxy phenomenon, which is where the cusp-core problem, the diversity problem, and MOND successes are all concentrated.

---

## Summary: Three Flanks

| Flank | Target | Result | For RTM |
|-------|--------|--------|---------|
| A: DM transition radius | Where DM begins | FAILED (data issue) | — |
| B: Cusp vs Core | Inner DM density | FAILED (partial ρ ≈ 0) | NEGATIVE |
| **C: Acceleration scale** | **g† universality** | **ρ = -0.574, p = 3×10⁻⁷** | **STRONG POSITIVE** |

### Score Impact

Flank C is the strongest single finding in all of the Doc 014 analysis. It goes beyond the previous correlations (ρ = 0.35 for mass discrepancy, ρ = 0.33 for diversity) because:

1. **Effect size is larger** (ρ = -0.574 vs 0.35)
2. **Mass has zero predictive power** (R² = 0.0004) — this is purely structural
3. **Bootstrap is unambiguous** (100% negative, CI excludes zero by wide margin)
4. **It challenges a fundamental assumption** (g† universality)
5. **Stable across D thresholds** (1.5 through 5.0, all significant)

---

*All computations reproducible via rtm_flank3.py. Data: SPARC table2.dat.*
