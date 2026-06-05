# Red Team Validation Report: Document 015 — Rhythmic Economics

**RTM Corpus — Independent Verification**
**Date:** April 28, 2026

---

## 1. ROBUST Claims Tested

- **DFA α crash detection:** Baseline α ≈ 0.55 drops to ≈ 0.46 before crashes. d = -1.45, lead time ~10 days.
- **Inverse cubic law:** Return tail exponent α ≈ 2.97 across 16 markets.
- **Recovery scaling:** ODR slope ≈ 3.59 (deeper crashes → exponentially longer recovery).
- **BTC forensic reports:** Real 1-min Binance data across 4 crash events + 1 control.

---

## 2. Reproduction Results

### 2.1 DFA α Crash vs Baseline

| Metric | Reported | Reproduced |
|--------|----------|------------|
| Baseline α | 0.55 ± 0.05 | **0.551 ± 0.031** ✓ |
| Crash α | 0.46 ± 0.07 | **0.458 ± 0.063** ✓ |
| Cohen's d | -1.45 | **1.82** (sign convention differs) |
| Significant events | — | **9/13 (69%)** |
| Mean lead (sig) | ~10 days | **8.8 days** ✓ |

**Assessment:** Reproduced. The separation is real and large. However, 4/13 events fail to show significant α-drop (31% false negative rate: FTX, SP500 Q4 2018, SP500 COVID, Gold COVID). Lead times range from 92h to 489h — too variable for reliable operations.

DFA α declining before crashes is a known result in econophysics (Grech & Mazur 2004, Alvarez-Ramirez 2008). RTM's contribution is the framing as "topological phase transition" and the operational packaging.

**For RTM: POSITIVE. Real effect, but known technique.**

### 2.2 Return Tail Exponent (Inverse Cubic)

| Metric | Reported | Reproduced |
|--------|----------|------------|
| α mean | 2.966 ± 0.236 | **2.966 ± 0.187** ✓ |
| Test vs 3.0 | — | p = 0.473 (cannot reject) |

**Assessment:** Perfectly reproduced. But this IS the inverse cubic law (Gopikrishnan 1999, Gabaix 2003) — one of the most cited results in econophysics. The data is a curated literature table, not independent analysis. RTM's T~L^α framework doesn't directly predict return tail exponents; the connection is naming.

**For RTM: CONSISTENT but NOT NEW.**

### 2.3 Recovery Time Scaling

| Metric | Reported | Reproduced |
|--------|----------|------------|
| ODR slope | 3.59 ± 0.70 | **3.87 ± 0.78** ✓ (close) |
| N crashes | — | 16 |

**Assessment:** Close to reported value. The finding — deeper crashes take disproportionately longer to recover — is interesting and the power-law exponent (~3.6-3.9) is steep. But only 16 points spanning 1907-2025 with ambiguous recovery definitions.

**For RTM: MODERATELY POSITIVE.**

### 2.4 Real BTC Microstructure — MY INDEPENDENT ANALYSIS

I computed rolling volatility-volume α directly from raw Binance 1-minute data:

| Dataset | α mean ± std | Interpretation |
|---------|--------------|----------------|
| March 2020 (crash) | 0.535 ± 0.133 | Crash month |
| Sept 2023 (control) | 1.731 ± 0.683 | Quiet month |
| Nov 2022 (FTX) | 0.732 ± 0.117 | Crash month |

**Between-month separation:** d = 2.43 (crash vs control). Massive.

**HOWEVER — critical intra-month finding:**

Within March 2020, α *increases* during the crash (Mar 12-13: α = 0.65) compared to pre-crash (Mar 1-11: α = 0.51). This is the **opposite** of what the report claims.

For FTX (Nov 2022), α does drop slightly during the crash (0.79 → 0.76), consistent with RTM.

**Interpretation:** The between-month signal is strong — crash months have structurally different α than quiet months. But the *within-event* timing (α drops BEFORE the crash) is not consistently reproduced with my independent rolling α computation. This may reflect differences in the specific α definition used (my volatility-volume regression vs the report's DFA-based method). The metric definition matters enormously.

**For RTM: MIXED. Between-month signal strong. Intra-event precursor inconsistent.**

---

## 3. Key Strengths of Doc 015

1. **REAL DATA:** 4 months of raw Binance 1-minute OHLCV data (44,000+ candles each)
2. **CONTROL GROUP:** September 2023 as explicit null hypothesis test — good science
3. **Multiple markets:** BTC, S&P500, Gold (13 events total)
4. **Honest failures:** The FTX event and others are flagged as non-significant

---

## 4. Key Weaknesses

1. **DFA α as early warning is known:** Grech & Mazur (2004) published this 20+ years ago
2. **31% false negative rate:** 4/13 events show no significant α-drop
3. **Lead time too variable:** 4 to 20 days — not operationally useful
4. **Inverse cubic is literature compilation:** Not independent analysis
5. **My independent BTC analysis shows opposite intra-event direction for March 2020** — metric definition sensitivity

---

## 5. Overall Verdict

| Finding | Reproduced? | Novel? | Score |
|---------|-------------|--------|-------|
| DFA baseline vs crash (d=1.82) | ✓ | Known technique | SOLID |
| Inverse cubic (α=2.97) | ✓ | Known since 1999 | CONSISTENCY |
| Recovery scaling (ODR=3.87) | ✓ | Somewhat | MODERATE |
| BTC real data (between-month) | ✓ | Good design | SOLID |
| BTC real data (intra-event precursor) | Mixed | — | MIXED |
| Control group design | ✓ | Good practice | STRONG |

### Bottom Line

**Doc 015 is methodologically solid** — real data, control groups, multiple markets, honest about failures. The between-month crash vs control signal is massive and reproducible. The numbers check out.

**The main limitation is novelty:** DFA-based crash detection is established econophysics. RTM repackages it with "topological phase transition" language but doesn't add a fundamentally new tool. The operational utility is limited by a 31% false negative rate and wildly variable lead times.

**Better than Doc 012 and 014; comparable to Doc 011.**

**Score: Net POSITIVE for RTM. Good science, known tools, real data.**

---

*Report generated independently. Computations reproducible via red_team_015.py.*
