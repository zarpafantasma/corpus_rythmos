# RTM Unified Field Framework - Section 3.1.3: Quantum Corrections

## Overview

This package contains four computational tools (S1-S4) implementing the **quantum corrections** analysis from Section 3.1.3 of "RTM Unified Field Framework".

The Coleman-Weinberg method is used to compute one-loop and two-loop corrections to the effective potential, demonstrating how quantum effects modify RTM predictions.

---

## Theoretical Background

### Coleman-Weinberg Effective Potential

The one-loop effective potential:

```
V_eff(ᾱ) = U_tree(ᾱ) + V_1-loop(ᾱ)

V_1-loop = (1/64π²) Σ_i m_i⁴(ᾱ) [ln(m_i²(ᾱ)/μ²) - 3/2]
```

Where the field-dependent masses are:
- `m²_α(ᾱ) = M² + U''(ᾱ)` (α-field fluctuations)
- `m²_φ(ᾱ) = m² + γ|∇ᾱ|²` (φ-field fluctuations)

### Key Results from Paper

1. Quantum corrections **shift the location of minima** (α-bands)
2. **Logarithmic terms** introduce scale dependence → β-functions
3. **RG running** of RTM parameters (λ, M, m, γ)
4. Perturbation theory converges: |V_2-loop| << |V_1-loop| << |V_tree|

---

## Simulations

### S1: Coleman-Weinberg Effective Potential
**Purpose:** Compute one-loop corrections to RTM potential

**Results:**
- Classical minima: α = 2.00, 2.50
- Quantum minima: α = 1.96, 2.54
- Shift: Δα ≈ ±0.04

---

### S2: Quantum-Corrected α-Band Structure
**Purpose:** Study how quantum corrections modify RTM bands

**Results:**
- All 5 RTM bands shift due to quantum corrections
- Band positions become μ-dependent (RG running)
- Barrier heights significantly modified

---

### S3: RG Flow of RTM Parameters
**Purpose:** Compute β-functions for RTM couplings

**β-functions:**
```
β_λ ~ (1/16π²)[λ² + γ²λ]
β_M² ~ (1/16π²)[λM² + γ²m²]
β_m² ~ (1/16π²)[γ²M²]
β_γ ~ (1/16π²)[γ³ + λγ]
```

**Results (μ/μ₀ ≈ 22000):**
- λ: +30.5%
- M²: +16.9%
- m²: +1.3%
- γ: +6.6%

---

### S4: Two-Loop Corrections
**Purpose:** Extend to two-loop order, test convergence

**Two-loop contributions:**
- Sunset diagrams: ~ λ² m⁴ ln²(m²/μ²)
- Double bubble: ~ m⁴ ln(m²) ln(m²)
- Vertex corrections: ~ γ² m⁴ ln(m²/μ²)

**Results:**
- |V_2-loop / V_1-loop| << 1 ✓
- Perturbation theory converges ✓

---

## Key Findings

| Prediction (Paper) | Simulation Result |
|-------------------|-------------------|
| Minima shift | ✓ Δα ≈ ±0.04 |
| μ-dependence | ✓ Bands run with scale |
| β-functions exist | ✓ All couplings run |
| Perturbative | ✓ |V_2| << |V_1| << |V_tree| |

---

## Physical Implications

1. **Precision predictions** must include quantum corrections
2. **RTM bands are not exact** - shifted by ~2-4% from classical values
3. **Scale dependence** requires specifying μ for measurements
4. **Perturbation theory valid** for RTM parameter ranges

---

## Usage

### Direct Execution
```bash
cd S1_coleman_weinberg
pip install -r requirements.txt
python S1_coleman_weinberg.py
```

### Docker
```bash
cd S1_coleman_weinberg
docker build -t rtm_quantum_s1 .
docker run -v $(pwd)/output:/app/output rtm_quantum_s1
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
Section: 3.1.3 "Canonical quantization and propagators"

---

## License

CC BY 4.0
