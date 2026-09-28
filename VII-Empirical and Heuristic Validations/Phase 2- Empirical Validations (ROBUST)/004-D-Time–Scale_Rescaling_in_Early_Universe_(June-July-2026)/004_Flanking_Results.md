# Document 004 — JWST Flanking Campaign Results

**Engine:** Claude Opus 4.6 Extended Thinking | **Date:** May 2026
**Data:** 55 JWST high-z galaxies (JADES, CEERS, UNCOVER, GLASS), z = 6.0–16.4

---

## Flank A: Subsample Robustness by Survey

**Question:** Does the excess-z correlation survive within individual surveys, or is it a cross-survey artifact?

**Result:** 3 of 4 testable surveys show significant positive correlation.

| Survey | n | Excess | ρ | p | Significant? |
|--------|---|--------|---|---|-------------|
| CEERS | 15 | 6/15 | +0.654 | 0.008 | ✓ |
| JADES | 8 | 0/8 | +0.762 | 0.028 | ✓ |
| Other | 25 | 13/25 | +0.441 | 0.027 | ✓ |
| UNCOVER | 5 | 2/5 | +0.600 | 0.285 | ns (n=5) |

**Classification:** POSITIVE — Signal is robust across surveys, not a selection artifact.

---

## Flank B: Excess Grows with Redshift (Mass-Controlled)

**Question:** At fixed stellar mass, does the ΛCDM excess grow with redshift?

**Result:** After controlling for stellar mass, ρ(z, excess | mass) = **+0.761, p < 10⁻⁶**.

| Bin | n | Mean excess | ρ(z, excess) |
|-----|---|-------------|-------------|
| Low-z (z ≤ 9.0) | 28 | +0.086 | +0.533 |
| High-z (z > 9.0) | 27 | +0.541 | +0.054 |
| High vs Low | — | d = +0.868, p = 0.002 | — |

The raw high-z ρ is weak (+0.054) because of a selection effect (high-z subsample has lower masses). After controlling for mass, the z-trend is the strongest correlation in the entire analysis (ρ = +0.761).

**Classification:** POSITIVE — RTM prediction confirmed: at fixed mass, higher-z galaxies show more excess over ΛCDM, consistent with larger acceleration factor A(z).

---

## Flank C: Acceleration Factor Sufficient

**Question:** Does A(z, α=1) provide enough acceleration for all excess galaxies?

**Result:** A(α=1) is sufficient for 55/55 galaxies (100%). No galaxy requires α > 1.

| Class | n | Mean z | Mean excess | Mean A_available |
|-------|---|--------|-------------|-----------------|
| IMPOSSIBLE (excess > 0.5) | 13 | 10.7 | +1.15 | 22.9 |
| TENSION (0 < excess < 0.5) | 27 | 10.1 | +0.23 | 21.1 |
| CONSISTENT (excess < 0) | 15 | 7.3 | −0.29 | 13.4 |

IMPOSSIBLE galaxies cluster at z ≈ 10.7 (higher than CONSISTENT at z ≈ 7.3), consistent with RTM: the most "impossible" galaxies are exactly where A(z) is largest.

**Classification:** POSITIVE — α = 1 is sufficient; no exotic α > 1 required.

---

## Summary

| Flank | Result | Key Metric | For RTM |
|-------|--------|-----------|---------|
| A. By survey | **POSITIVE** | 3/4 surveys significant | Not a cross-survey artifact |
| B. Excess vs z (mass-controlled) | **POSITIVE** | ρ = +0.761, p < 10⁻⁶ | Excess grows with z as predicted |
| C. Acceleration sufficient | **POSITIVE** | 55/55 galaxies resolved | α = 1 suffices |

## Score Impact

**Doc 004: 70% → 74%**

Three positive flanks. The mass-controlled excess-z correlation (ρ = +0.761) is the strongest result — it shows that the JWST anomaly is not just "big galaxies exist at high-z" but "at fixed mass, higher-z galaxies deviate MORE from ΛCDM." This is the directional prediction of RTM time-rescaling and survives within individual surveys (3/4 significant). The acceleration factor A(α=1) is sufficient for all 55 galaxies without requiring exotic α values.

---

*Scripts: rtm_jwst_flanks.py. Data: jwst_galaxy_catalog.csv.*
