# 008-Rhythmic Biochemistry: RTM Enzyme Kinetics Validation

## Paper: "Rhythmic Biochemistry: Enzyme as a Coherence Instrument"

---

## Key Results

| Test | Result |
|------|--------|
| α recovery accuracy | **0.2% error** |
| Noise robustness | MAE < 0.15 for σ ≤ **0.15** |
| Class discrimination | Cohen's d = **3.12** |
| Collapse test | **11× worse** with wrong α |
| Selectivity tuning | **2-3× shift** across 10-100nm |

---

## Core RTM Prediction

**k_app(L) = k_app,0 × (L/L_ref)^(−α)**

| Transport Class | α | Example |
|-----------------|---|---------|
| Guided/ballistic | 1.5-1.8 | Proton wires |
| Diffusive | 2.0 | Random walk |
| Hierarchical | 2.1-2.5 | Fractal networks |

---

## Package Contents

```
008-Rhythmic_Biochemistry/
│
├── S1_michaelis_menten/                   ← RTM-Modified Kinetics
│   ├── S1_michaelis_menten.py
│   └── output/
│       ├── S1_michaelis_menten.png
│       └── S1_alpha_recovery.png
│
├── S2_confinement_scaling/                ← Methodology Validation
│   ├── S2_confinement_scaling.py
│   └── output/
│       ├── S2_confinement_scaling.png
│       └── S2_validation.png
│
├── S3_selectivity/                        ← Selectivity Prediction
│   ├── S3_selectivity.py
│   └── output/
│       ├── S3_selectivity.png
│       └── S3_rbci_and_tuning.png
│
└── requirements.txt
```

---

## Simulation Summaries

### S1: RTM-Modified Michaelis-Menten

Shows that k_cat ∝ L^(−α) produces measurable kinetic differences:
- Higher α = stronger confinement enhancement
- α recoverable from kinetic data with <1% error

### S2: Confinement Scaling Methodology

Validates the estimator α_enz = −d(log k_app)/d(log L):
- Robust to σ ≤ 0.15 measurement noise
- Needs ≥3 confinement scales
- Data collapse test distinguishes correct α (11× better CV)

### S3: Selectivity Prediction

Predicts confinement-tunable selectivity:
- If substrates have different α, selectivity changes with L
- Crossover points calculable from α values
- Applications: drug metabolism, enantioselectivity, allostery

---

## Running the Simulations

```bash
pip install numpy scipy pandas matplotlib

python S1_michaelis_menten/S1_michaelis_menten.py
python S2_confinement_scaling/S2_confinement_scaling.py
python S3_selectivity/S3_selectivity.py
```

---

## RBCI: Rhythmic Biochemistry Coherence Index

RBCI (0-1) aggregates coherence markers:

| Component | Weight | Measurement |
|-----------|--------|-------------|
| α_norm | 30% | Slope from scaling |
| CISS | 25% | Spin polarization |
| Vibrational | 25% | Raman/IR coherent modes |
| Variance reduction | 20% | Under on-resonance driving |

**RBCI > 0.6**: Strong RTM scaling expected
**RBCI < 0.3**: Deviations likely

---

## Experimental Validation

**Required:**
1. Enzyme with assayable k_cat
2. ≥5 confinement scales (10-200nm)
3. ≥3 replicates per scale

**Protocol:**
1. Measure k_app at each L
2. Plot log(k_app) vs log(L)
3. Slope = −α
4. Verify: k_app × L^α should be constant (collapse test)

**Confinement methods:**
- Nanoporous membranes (AAM, silica)
- Polymer crowding (PEG, dextran)
- Engineered cavities
