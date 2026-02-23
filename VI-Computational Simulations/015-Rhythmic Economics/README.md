# 015-Rhythmic Economics: RTM Economic Coherence Framework

## Paper: "Rhythmic Economics: Measuring Systemic Resilience with the RTM Coherence Exponent"

---

## Key Results

| Test | Result |
|------|--------|
| α estimation error | **0.56%** |
| Early warning lead time | **9 months** |
| α vs Crisis frequency | **r = -0.91** |
| α vs Drawdown | **r = -0.95** |

---

## Core RTM-Econ Prediction

**τ(L) = τ₀ × (L/L_ref)^α**

| Regime | α | Interpretation |
|--------|---|----------------|
| Stable Growth | 0.45 | Good coherence, buffered |
| Pre-Crisis | 0.35 | Coherence declining |
| Crisis | 0.20 | Decoherence, cascade risk |
| Recovery | 0.40 | Rebuilding coherence |

---

## Package Contents

```
015-Rhythmic_Economics/
│
├── S1_alpha_estimation/                   ← Estimate α from data
│   ├── S1_alpha_estimation.py
│   └── output/
│       ├── S1_alpha_estimation.png
│       └── S1_multi_family.png
│
├── S2_early_warning/                      ← Recession prediction
│   ├── S2_early_warning.py
│   └── output/
│       ├── S2_gfc_analysis.png
│       └── S2_all_recessions.png
│
├── S3_cross_country/                      ← Country comparison
│   ├── S3_cross_country.py
│   └── output/
│       ├── S3_country_comparison.png
│       └── S3_regional.png
│
└── requirements.txt
```

---

## Simulation Summaries

### S1: α Estimation from Financial Data

Demonstrates how to extract α from market cap tiers:
- **Method:** Theil-Sen robust regression on log(τ) vs log(L)
- **Error:** < 1% across market regimes
- **Meta-analysis:** Combines 4 proxy families

### S2: Early Warning Backtesting

Validates H2: α decline anticipates recessions:
- **2001 Dot-Com:** 9 month lead, Δα = 0.14
- **2008 GFC:** 15 month lead, Δα = 0.27
- **2020 COVID:** 3 month lead, Δα = 0.18

### S3: Cross-Country Comparison

Shows α predicts economic resilience:
- **Developed (α~0.52):** Few crises, small drawdowns
- **Emerging (α~0.28):** Frequent crises, large drawdowns
- **Correlation:** r = -0.91 with crisis frequency

---

## Running the Simulations

```bash
pip install numpy scipy pandas matplotlib

python S1_alpha_estimation/S1_alpha_estimation.py
python S2_early_warning/S2_early_warning.py
python S3_cross_country/S3_cross_country.py
```

---

## RTM-Econ Hypotheses

**H1 (Resilience):** Higher α → smaller drawdowns, faster recovery
**H2 (Anticipation):** α drops 6-18 months before recessions
**H3 (Cascade):** α non-decreasing from micro → macro

---

## Economic Coherence Index (ECI)

| ECI (α) | Interpretation |
|---------|----------------|
| > 0.45 | High coherence (stable, buffered) |
| 0.35-0.45 | Moderate coherence |
| 0.25-0.35 | Low coherence (vulnerable) |
| < 0.25 | Decoherence (crisis mode) |

---

## Early Warning Protocol

1. **Monitor** rolling ECI with 3-6 month window
2. **Baseline** α during economic expansion
3. **Alert** when α drops > 15% below baseline
4. **Confirm** with yield curve, credit spreads
5. **Lead time:** typically 6-18 months
