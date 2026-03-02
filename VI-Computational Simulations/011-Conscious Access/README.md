# 011-Conscious Access: RTM Consciousness Threshold Framework

## Paper: "Conscious Access as a Multiscale Coherence Threshold"

---

## Key Results

| Test | Result |
|------|--------|
| Classification Accuracy | **89.6%** |
| Report vs No-Report | **Cohen's d = 1.59** |
| Conscious vs Unconscious NDI | **t = 2.65** |
| Propofol α Collapse | **2.44 → 1.56** |

---

## Core Concept: The "Double-Lock" Signatures

**Conscious access is not determined by stimulus intensity alone, but requires BOTH of the following physical conditions to be met within the receiving network:**

**S1: α ≥ 2.0 (Spatial Coherence Threshold)**
- The macroscopic neural network must maintain a topological density exponent of α ≥ 2.0.
- Below this threshold, the global workspace fragments and sensory stimuli cannot propagate, leading to unconsciousness.
- *(Note: Local signal dynamics are rigorously mapped to the macroscopic RTM spatial topology via the fractional Brownian motion transformation: α_RTM = 2H + 1).*

**S2: NDI > 0 (Forward Directionality)**
- NDI = (TE_forward - TE_backward) / (TE_forward + TE_backward)
- A positive NDI indicates that the forward/feedforward predictive cascade is dominant, allowing information to penetrate higher-order association cortices.

---


## Simulation Summaries

### S1: Consciousness Threshold

Demonstrates α as the physical threshold for conscious access in Masked Detection Tasks:
- **α ≥ 2.0:** Conscious states (Awake, Report, REM). The network is highly integrated; the stimulus propagates.
- **α < 2.0:** Unconscious states (No-Report, NREM, Anesthesia). The network is topologically fragmented; the stimulus collapses.
- **Classification Accuracy:** 89.6%

### S2: Forward Directionality

Validates the cortical cascade direction (e.g., V1→V2→V4→IT) using Transfer Entropy (TE):
- **Conscious:** NDI > 0 (Forward-dominant flow reaches higher associative areas).
- **Unconscious:** NDI ≈ 0 (Symmetric or backward-dominant noise; no global broadcasting).

### S3: Pharmacological Effects

Models drug effects on the S1 and S2 signatures to map altered states:
- **Propofol:** Both α and NDI collapse → True unconsciousness.
- **Psychedelics:** α ↑ but NDI ↓ → Altered consciousness (hallucinatory dissociation).
- **Ketamine:** α borderline but NDI ↓ → Trance-like dissociation.

---

## Running the Simulations

```bash
pip install numpy scipy pandas matplotlib

python S1_threshold/S1_threshold_fixed.py
python S2_directionality/S2_directionality.py
python S3_pharmacology/S3_pharmacology_fixed.py
```

---

## Consciousness Classification ("Double-Lock" Criterion)

| Spatial Topology (S1: α) | Information Flow (S2: NDI) | Clinical State |
|--------|----------|-------|
| **≥ 2.0** | **> 0** | **Normal Conscious** (Reportable access) |
| **≥ 2.0** | **< 0** | **Altered Conscious** (Psychedelic/Dissociative) |
| **< 2.0** | **≤ 0** | **Unconscious** (Anesthesia/Deep Sleep) |

---

## State Profiles

| State | α (Coherence) | NDI (Directionality) | Classification |
|-------|---|-----|----------------|
| Awake Report | 2.44 | 0.45 | Normal Conscious |
| REM Sleep | 2.30 | 0.35 | Normal Conscious |
| Light Sedation | 2.04 | 0.25 | Normal Conscious |
| NREM Sleep | 1.70 | 0.05 | Unconscious |
| Deep Anesthesia | 1.56 | 0.02 | Unconscious |
| Psychedelic Peak | 2.64 | -0.15 | Altered Conscious |

---

## Pharmacological Predictions

### Propofol (GABAergic General Anesthesia)
- **α:** 2.44 → 1.56 (Topological fragmentation)
- **NDI:** 0.45 → 0.02 (Forward flow collapse)
- **Result:** Unconsciousness. The global workspace is physically dismantled.

### Psychedelics (Serotonergic Agonists)  
- **α:** 2.44 → 2.64 (Hyper-integration / dense spatial topology)
- **NDI:** 0.45 → -0.15 (Flow reversal / top-down predictive dominance)
- **Result:** Altered consciousness. The brain maintains a massive workspace but is decoupled from external sensory reality, trapping the network in internal predictive models.

### Ketamine (NMDA Antagonist)
- **α:** 2.44 → 2.20 (Borderline spatial stability)
- **NDI:** 0.45 → -0.05 (Flow reversal)
- **Result:** Altered consciousness. Matches clinical "trance-like" dissociative anesthesia.

---

## The Hard Problem

This framework addresses **access consciousness** (reportability, global broadcasting, and information routing), not phenomenal consciousness (qualia). The α and NDI parameters are mathematically necessary physical conditions for a biological system to process information globally, but they are not necessarily sufficient for subjective experience.