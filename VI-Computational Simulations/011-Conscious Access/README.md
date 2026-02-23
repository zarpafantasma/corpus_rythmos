# 011-Conscious Access: RTM Consciousness Threshold Framework

## Paper: "Conscious Access as a Multiscale Coherence Threshold"

---

## Key Results

| Test | Result |
|------|--------|
| Classification AUC | **0.65** |
| Report vs No-Report | **d = 1.59** |
| Conscious vs Unconscious NDI | **t = 2.65** |
| Propofol α collapse | **-61%** |

---

## Core Concept: Two Signatures

**Conscious access requires BOTH:**

**S1: α > α_crit** (Coherence threshold)
- α_crit ≈ 0.50
- Higher α = more multiscale temporal coherence

**S2: NDI > 0** (Forward directionality)
- NDI = (TE_forward - TE_backward) / (TE_forward + TE_backward)
- Positive = forward cascade dominant

---

## Package Contents

```
011-Conscious_Access/
│
├── REVISED_ABSTRACT_AND_APPENDIX_L.md    ← INSERT INTO PAPER
│
├── S1_threshold/                          ← α Threshold Model
│   ├── S1_threshold.py
│   └── output/
│       ├── S1_threshold_model.png
│       └── S1_report_no_report.png
│
├── S2_directionality/                     ← Forward Cascade
│   ├── S2_directionality.py
│   └── output/
│       ├── S2_directionality.png
│       └── S2_signals.png
│
├── S3_pharmacology/                       ← Drug Effects
│   ├── S3_pharmacology.py
│   └── output/
│       ├── S3_dose_response.png
│       └── S3_state_comparison.png
│
└── requirements.txt
```

---

## Simulation Summaries

### S1: Consciousness Threshold

Demonstrates α as threshold for conscious access:
- **α > 0.50:** Conscious states (report, awake, REM)
- **α < 0.50:** Unconscious states (no-report, NREM, anesthesia)
- Classification accuracy: 85%

### S2: Forward Directionality

Validates cortical cascade direction:
- **Conscious:** NDI > 0 (forward dominant)
- **Unconscious:** NDI ≈ 0 (symmetric)
- Transfer entropy between V1→V2→V4→IT

### S3: Pharmacological Effects

Models drug effects on S1 and S2:
- **Propofol:** Both α and NDI collapse → unconscious
- **Psychedelics:** α↑ but NDI↓ → altered consciousness
- Different routes to altered states

---

## Running the Simulations

```bash
pip install numpy scipy pandas matplotlib

python S1_threshold/S1_threshold.py
python S2_directionality/S2_directionality.py
python S3_pharmacology/S3_pharmacology.py
```

---

## Consciousness Classification

| α (S1) | NDI (S2) | State |
|--------|----------|-------|
| > 0.50 | > 0.15 | **Normal Conscious** |
| > 0.50 | < 0 | **Altered Conscious** |
| < 0.50 | < 0.15 | **Unconscious** |

---

## State Profiles

| State | α | NDI | Classification |
|-------|---|-----|----------------|
| Awake Report | 0.72 | 0.45 | Normal Conscious |
| REM Sleep | 0.65 | 0.35 | Normal Conscious |
| Light Sedation | 0.52 | 0.25 | Normal Conscious |
| NREM Sleep | 0.35 | 0.05 | Unconscious |
| Deep Anesthesia | 0.28 | 0.02 | Unconscious |
| Psychedelic Peak | 0.82 | -0.15 | Altered Conscious |

---

## Pharmacological Predictions

### Propofol (GABAergic)
- α: 0.72 → 0.28 (collapse)
- NDI: 0.45 → 0.02 (collapse)
- Result: Unconsciousness

### Psychedelics (Serotonergic)  
- α: 0.72 → 0.82 (increase)
- NDI: 0.45 → -0.15 (reversal)
- Result: Altered consciousness

---

## The Hard Problem

This framework addresses **access consciousness** (reportability), not phenomenal consciousness (qualia). α and NDI are necessary but not sufficient for subjective experience.
