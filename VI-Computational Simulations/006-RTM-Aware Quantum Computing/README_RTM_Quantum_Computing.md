# RTM-Aware Quantum Computing - Computational Simulations

## Overview

This package contains five computational simulations (S1-S5) implementing the core concepts from "RTM-Aware Quantum Computing: A Multiscale, Slope-First Framework for Coherence, Scheduling, and Design".

---

## Theoretical Background

### RTM in Quantum Computing

RTM proposes that characteristic time T scales with size L as:

```
T ∝ L^α
```

where α (the coherence exponent) is invariant to clock/unit changes.

### Layer-Specific Definitions

| Layer | Scale Proxy (L) | Time Proxy (T) |
|-------|-----------------|----------------|
| Physical | Active qubits | Calibration interval |
| QEC | Code distance d | Cycles to failure |
| Runtime | Circuit width | Makespan |
| I/O-Cryo | Multiplexing degree | Readout latency |

---

## Simulations

### S1: Physical Layer Scaling
**Purpose:** Validate T ∝ L^α at physical layer

**Results:**
- α ≈ -1.5 for qubit count vs calibration time
- Collapse test passes
- R² > 0.99

---

### S2: QEC Layer Scaling
**Purpose:** Validate T ∝ d^α for error correction

**Results:**
- α ≈ 2.5 for code distance vs cycles to failure
- Syndrome jitter improves stability

---

### S3: Runtime/Compiler Scaling
**Purpose:** Validate T ∝ W^α for circuit execution

**Results:**
- α ≈ 1.8 for circuit width vs makespan
- Scheduling strategies reduce tail latencies

---

### S4: Collapse Test
**Purpose:** Implement RTM specification test

**Tests:**
1. Residual regression (R² < 0.05)
2. Clock placebo (α invariant)
3. LOESS trend (flat residuals)

**Results:**
- Valid RTM data: PASS
- Regime mixing: DETECTED
- Curvature: DETECTED

---

### S5: Alpha Fusion
**Purpose:** Multi-layer RTM indicator

**Components:**
- Random-effects meta-analysis (REML)
- Heterogeneity testing (I², τ², Q)
- Alert thresholds (Advisory/Watch/Warning)

---

## Key Concepts from Paper

### Collapse Test
After removing the power-law trend, residuals should be independent of scale.
If collapse fails, the bin should be split or rejected.

### Heterogeneity Gate
Fusion only when layers agree (I² < 0.5).
Otherwise, publish layer-wise α values.

### Alert System
| Level | Z-score | Action |
|-------|---------|--------|
| Advisory | < -1.5 | Monitor |
| Watch | < -2.0 | Prepare intervention |
| Warning | < -2.5 | Execute playbook |

---

## Usage

### Direct Execution
```bash
cd S1_physical_layer
pip install -r requirements.txt
python S1_physical_layer.py
```

### Docker
```bash
cd S1_physical_layer
docker build -t rtm_qc_s1 .
docker run -v $(pwd)/output:/app/output rtm_qc_s1
```

### Jupyter
```bash
cd S1_physical_layer
jupyter notebook S1_physical_layer.ipynb
```

---

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

---

## Reference

Paper: "RTM-Aware Quantum Computing"
Document: 006
Sections: 2-6 (Foundations, Estimation, Fusion, Design)

---

## License

CC BY 4.0
