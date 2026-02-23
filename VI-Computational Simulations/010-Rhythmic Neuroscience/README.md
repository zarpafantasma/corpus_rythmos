# 010-Rhythmic Neuroscience: RTM-Neuro Computational Validation

## Paper: "Rhythmic Neuroscience: Conscious Access as Multiscale Coherence"

---

## Key Results

| Test | Result |
|------|--------|
| α recovery accuracy | **1.2% error** |
| Noise robustness | MAE < 0.2 for σ ≤ **0.3** |
| State discrimination | Cohen's d = **2.85** |
| Awake time above threshold | **94%** |
| Deep anesthesia time above threshold | **0%** |

---

## Package Contents

```
010-Rhythmic_Neuroscience/
│
│   ├── Revised Abstract
│   └── Appendix E: Computational Validation
│
├── S1_signal_generation/                  ← τ(L) Scaling Demo
│   ├── S1_signal_generation.py
│   ├── S1_signal_generation.ipynb
│   └── output/
│       ├── S1_scaling_law.png
│       ├── S1_band_predictions.png
│       ├── S1_state_predictions.png
│       └── S1_alpha_recovery.png
│
├── S2_alpha_estimation/                   ← Methodology Validation
│   ├── S2_alpha_estimation.py
│   ├── S2_alpha_estimation.ipynb
│   └── output/
│       ├── S2_estimation_validation.png
│       └── S2_*.csv
│
├── S3_conscious_threshold/                ← Threshold Model
│   ├── S3_conscious_threshold.py
│   ├── S3_conscious_threshold.ipynb
│   └── output/
│       ├── S3_state_patterns.png
│       ├── S3_transitions.png
│       ├── S3_binding.png
│       └── S3_pathologies.png
│
└── requirements.txt
```

---

## RTM-Neuro Core Hypothesis

**τ(L) = τ_0 × L^α**

- τ = characteristic time (autocorrelation e-folding)
- L = spatial scale (cortical distance)
- α = coherence exponent

**α > 2**: Integrated dynamics → conscious processing  
**α < 2**: Fragmented dynamics → unconscious processing

---

## Simulation Summaries

### S1: τ(L) Scaling Demonstration

**Purpose**: Show that τ(L) ∝ L^α produces distinct neural signatures

**Key Findings**:
- Band hierarchy: Delta (α=2.5) → Gamma (α=1.5)
- State ordering: Awake (α=2.15) → Deep anesthesia (α=1.45)
- α recovery from noisy data: 1.2% mean error

### S2: Estimation Methodology

**Purpose**: Validate α estimation from realistic neural data

**Key Findings**:
- Noise robust up to σ = 0.3
- Need ≥3 spatial scales
- State discrimination: Cohen's d = 2.85

### S3: Conscious Access Threshold

**Purpose**: Model α dynamics during consciousness transitions

**Key Findings**:
- Awake: 94% time above threshold
- Transition dynamics match LOC/ROC phenomenology
- Binding episodes show transient α peaks

---

## How to Update the Paper

### 1. Replace Abstract
Insert the revised abstract from `REVISED_ABSTRACT_AND_APPENDIX_E.md`

Key additions:
> "Computational validation. We implement and test the RTM-Neuro framework through three simulation suites..."

### 2. Add Appendix E
Insert "Appendix E — Computational Validation" after existing appendices

### 3. Reference Supplementary Materials
The output/ directories contain figures and data files that can be referenced or included in supplementary materials.

---

## Running the Simulations

```bash
# Install dependencies
pip install -r requirements.txt

# Run all simulations
python S1_signal_generation/S1_signal_generation.py
python S2_alpha_estimation/S2_alpha_estimation.py
python S3_conscious_threshold/S3_conscious_threshold.py

# Or use Docker
cd S1_signal_generation && docker build -t rtm-neuro-s1 . && docker run rtm-neuro-s1
```

---

## Critical Disclaimer

These simulations validate **methodology**, not the physical hypothesis.

Empirical validation requires:
- EEG/MEG recordings from human subjects
- Ground-truth consciousness state labels
- Comparison with PCI, BIS, spectral entropy
- Prospective LOC/ROC prediction testing
