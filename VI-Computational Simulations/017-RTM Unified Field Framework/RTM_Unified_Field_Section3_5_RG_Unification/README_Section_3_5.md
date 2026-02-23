# RTM Unified Field Framework - Section 3.5: RG Unification

## Overview

This package contains four computational tools (S1-S4) implementing the **Renormalization-Group Unification** analysis from Section 3.5 of "RTM Unified Field Framework".

The key result: RTM provides a pathway to **gauge coupling unification** at high scales through threshold corrections and the α-shift mechanism.

---

## Theoretical Background

### The Problem
Standard Model gauge couplings (g₁, g₂, g₃) do NOT unify:
- Running them to high energies shows they get close but never meet
- Minimum spread ~6 at ~10^14 GeV

### RTM Solution
1. **Threshold Corrections** (Section 3.5.2): New states modify beta coefficients
2. **α-Shift Mechanism** (Section 3.5.1): Scale-dependent correction Δα = η(μ/M_RTM)^ξ

### Paper Results
- **M_GUT ≈ 1.7 × 10^15 GeV** (unification scale)
- **M_RTM ≈ 3.2 × 10^11 GeV** (RTM threshold scale)
- **α_GUT⁻¹ ≈ 24.5** (unified coupling)
- All three couplings agree within 1σ

---

## Simulations

### S1: Gauge Coupling RGE Running
**Purpose:** Show SM couplings don't unify

**Method:** Two-loop RGE integration from M_Z to 10^17 GeV

**Result:** Minimum spread ~5-6, no unification

---

### S2: Threshold Matching
**Purpose:** Show effect of RTM state thresholds

**RTM States:**
| State | Mass (GeV) | Δb₁ | Δb₂ | Δb₃ |
|-------|------------|-----|-----|-----|
| RTM scalar | 3.2×10^11 | 0.1 | 0.17 | 0 |
| Heavy fermion F1 | 10^12 | 0.4 | 0 | 0.33 |
| Heavy fermion F2 | 5×10^12 | 0.2 | 0.5 | 0 |
| Heavy scalar S | 10^13 | 0.1 | 0.17 | 0 |
| RTM vector V | 5×10^13 | 0 | 1.33 | 2 |

**Result:** Spread reduced ~11%, but not enough alone

---

### S3: Unification Fit
**Purpose:** Find parameters achieving unification

**Method:** Parameter scan over (M_RTM, η)

**Result:** Best spread ~3.8 at M_GUT ~ 2×10^14 GeV

---

### S4: Alpha-Shift Effect
**Purpose:** Study the α-shift mechanism

**Equation:** Δα_shift = η × (μ/M_RTM)^ξ with ξ = 1

**Result:** Demonstrates how α-shift modifies running

---

## Key Findings

| Simulation | SM Only | With RTM | Improvement |
|------------|---------|----------|-------------|
| S1 | Spread ~5.5 | - | Baseline |
| S2 | Spread ~5.5 | Spread ~4.9 | 11% |
| S3 | - | Spread ~3.8 | Best fit |
| S4 | - | Variable | α-shift effect |

---

## Notes on Simplified Model

The simulations use simplified versions of the paper's model for computational efficiency:
- One-loop RGEs (vs two-loop in paper)
- Reduced threshold spectrum
- Capped α-shift for numerical stability

The full paper model achieves complete unification (spread → 0), while our simplified model shows significant convergence (~30-40% reduction in spread).

---

## Usage

### Direct Execution
```bash
cd S1_gauge_rge_running
pip install -r requirements.txt
python S1_gauge_rge_running.py
```

### Docker
```bash
cd S1_gauge_rge_running
docker build -t rtm_rg_s1 .
docker run -v $(pwd)/output:/app/output rtm_rg_s1
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

Paper: "RTM Unified Field Framework"
Document: 017
Section: 3.5 "Renormalization-Group Unification of the Three SM Gauge Couplings"

---

## License

CC BY 4.0
