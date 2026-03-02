# 012-Rhythmic Ecology: RTM Ecosystem Resilience Framework

## Paper: "Rhythmic Ecology: A Slope-First Framework for Ecosystem Resilience"

---

## Key Results

| Test | Result |
|------|--------|
| NDVI recovery α error | **0.66%** |
| Watershed α error | **1.3%** |
| Regime shift lead time | **6.8 years** |
| ECI range | Wetland (0.86) → Urban (0.11) |

---

## Core RTM-Eco Prediction

**τ(L) = τ₀ × (L/L_ref)^α**

| Ecosystem Type | α | Meaning |
|----------------|---|---------|
| Boreal Forest | 0.35 | Large fires → much longer recovery |
| Grassland | 0.22 | Recovery less scale-dependent |
| Wetlands | 0.55 | High retention, slow release |
| Urban | 0.25 | Flashy, low resilience |

---

## Package Contents

```
012-Rhythmic_Ecology/
│
├── S1_ndvi_recovery/                      ← Fire Recovery Scaling
│   ├── S1_ndvi_recovery.py
│   └── output/
│       ├── S1_recovery_scaling.png
│       └── S1_alpha_estimation.png
│
├── S2_watershed_alpha/                    ← Watershed Resilience
│   ├── S2_watershed_alpha.py
│   └── output/
│       ├── S2_watershed_alpha.png
│       └── S2_resilience_comparison.png
│
├── S3_regime_shift/                       ← Early Warning
│   ├── S3_regime_shift.py
│   └── output/
│       ├── S3_detailed_analysis.png
│       ├── S3_scenarios_comparison.png
│       └── S3_lead_times.png
│
└── requirements.txt
```

---

## Simulation Summaries

### S1: NDVI Recovery vs Burned Area

Demonstrates τ ∝ L^α for post-fire vegetation recovery:
- **Boreal forest**: α ≈ 0.35 (large fires take much longer)
- **Grassland**: α ≈ 0.22 (quick recovery at all scales)
- **Recovery error**: < 1%

### S2: Watershed Coherence Exponent

Maps α to Ecosystem Coherence Index (ECI):
- **Wetlands**: ECI = 0.86 (very high resilience)
- **Forested**: ECI = 0.61 (high resilience)
- **Urban**: ECI = 0.11 (low resilience)

### S3: Regime Shift Early Warning

α decline anticipates ecosystem collapse:
- **Lead time**: 4-11 years before state transition
- **Detection**: α drops before visible degradation
- **Application**: Early warning for management

---

## Running the Simulations

```bash
pip install numpy scipy pandas matplotlib

python S1_ndvi_recovery/S1_ndvi_recovery.py
python S2_watershed_alpha/S2_watershed_alpha.py
python S3_regime_shift/S3_regime_shift.py
```

---

## Ecosystem Coherence Index (ECI)

**ECI = (α - 0.20) / (0.60 - 0.20)**

| ECI Range | Interpretation |
|-----------|----------------|
| 0.75 - 1.0 | Very high resilience |
| 0.50 - 0.75 | High resilience |
| 0.25 - 0.50 | Moderate resilience |
| 0.0 - 0.25 | Low resilience |

---

## RTM-Eco Hypotheses

**H1 (Resilience):** Higher α = more orderly recovery
**H2 (Decoherence):** α decline anticipates regime shifts
**H3 (Master curves):** τ vs L collapses across disturbance types

---

## Early Warning Protocol

1. **Monitor α** through regular disturbance-recovery observations
2. **Establish baseline** during healthy conditions
3. **Detect decline** when α drops > 2σ below baseline
4. **Intervene** within lead time window
