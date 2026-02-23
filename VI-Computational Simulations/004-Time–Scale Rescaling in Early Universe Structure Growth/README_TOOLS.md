# Computational Tools for "Time–Scale Rescaling in Early Universe Structure Growth"

## Overview

This package contains four computational tools (T1-T4) designed to reproduce, verify, and explore the calculations presented in the RTM paper on early universe structure growth. These tools validate the RTM acceleration factor and its implications for explaining "too-early/too-massive" galaxies observed by JWST.

---

## Purpose

The paper proposes that RTM (Relatividad Temporal Multiescala) provides a natural explanation for the rapid assembly of massive galaxies at high redshift (z > 10). The key mechanism is an **acceleration factor A** that effectively speeds up mesoscopic processes in the early universe.

These tools:
1. **Verify** the mathematical calculations in the paper
2. **Correct** numerical errors identified during validation
3. **Explore** the parameter space to understand feasibility
4. **Provide** reproducible computational evidence

---

## Tools Description

### T1: Acceleration Factor Calculator

**File:** `T1_acceleration_factor/T1_acceleration_factor.py`

**Purpose:** Calculate the RTM acceleration factor A as a function of redshift z and coherence exponent α.

**Key Equations:**
```
Einstein-de Sitter:  A = (1+z)^(3α/2)
ΛCDM:                A = [Ω_m(1+z)³ + Ω_Λ]^(α/2)
```

**What it does:**
- Computes A for both EdS and ΛCDM cosmologies
- Generates comparison tables across multiple redshifts
- Produces visualization plots
- Verifies paper predictions (and identifies corrections needed)

**Key Output:**
```
z = 10, α = 1:
  A_EdS  = 36.5  (paper said ~58 — CORRECTED)
  A_ΛCDM = 20.5  (paper said 30-40 — CORRECTED)
```

---

### T2: Galaxy Assembly Calculator

**File:** `T2_galaxy_assembly/T2_galaxy_assembly.py`

**Purpose:** Calculate the required acceleration factor A to reach a target stellar mass M_star at high redshift.

**Key Equation:**
```
M_star = f_b × M_halo × [1 - (1-ε)^(A×N_dyn)]

Required A:
A_required = ln[1 - M_star/(f_b×M_halo)] / [N_dyn × ln(1-ε)]
```

**Parameters:**
- `f_b`: Cosmic baryon fraction (0.157)
- `ε`: Star formation efficiency per dynamical time
- `N_dyn`: Number of dynamical times available
- `M_halo`: Halo mass

**What it does:**
- Calculates required A for various target stellar masses
- Tests paper's "Case A" (demanding) and "Case B" (moderate) scenarios
- Determines feasibility under EdS and ΛCDM cosmologies
- Maps the (ε, A) parameter space

**Key Output:**
```
Case A (M_star = 10¹¹ M☉, ε = 2%): A_required ≈ 10
Case B (M_star = 3×10¹⁰ M☉, ε = 2%): A_required ≈ 2

Both achievable with A_ΛCDM ≈ 20 at z=10 ✓
```

---

### T3: EdS vs ΛCDM Comparison

**File:** `T3_eds_vs_lcdm/T3_eds_vs_lcdm.py`

**Purpose:** Detailed comparison of Einstein-de Sitter and ΛCDM cosmological backgrounds for the RTM mechanism.

**What it does:**
- Compares H(z)/H₀ evolution in both models
- Computes cosmic age at each redshift
- Shows how A differs between EdS and ΛCDM
- Explains why the paper uses EdS for pedagogy but ΛCDM for realism

**Key Insight:**
```
EdS overestimates A by ~78% at high z because it assumes Ω_m = 1.
ΛCDM (Planck 2018: Ω_m = 0.315, Ω_Λ = 0.685) gives more realistic values.

The RTM mechanism works with either, but ΛCDM predictions are more accurate.
```

**Reference Table Generated:**
| z | Age (ΛCDM) | A_EdS | A_ΛCDM |
|---|------------|-------|--------|
| 5 | 1.17 Gyr | 14.7 | 8.3 |
| 10 | 0.47 Gyr | 36.5 | 20.5 |
| 15 | 0.27 Gyr | 64.0 | 35.9 |
| 20 | 0.18 Gyr | 96.2 | 54.0 |

---

### T4: Parameter Space Explorer

**File:** `T4_parameter_space/T4_parameter_space.py`

**Purpose:** Systematic exploration of the full parameter space (α, ε, z, N_dyn, M_halo, M_star) to identify which combinations can explain observed high-z galaxies.

**Parameters Scanned:**
- Redshift: z = 7, 10, 12, 15, 20
- RTM exponent: α = 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
- Star formation efficiency: ε = 1%, 2%, 5%, 10%
- Dynamical times: N_dyn = 3, 5, 7, 10
- Halo mass: M_halo = 10¹¹, 5×10¹¹, 10¹², 5×10¹² M☉
- Target stellar mass: M_star = 10¹⁰, 3×10¹⁰, 10¹¹ M☉

**What it does:**
- Tests 5,760 parameter combinations
- Identifies feasible vs infeasible configurations
- Calculates critical α needed for each scenario
- Generates feasibility heatmaps

**Key Results:**
```
Total configurations tested: 5,760
Feasible (A_available ≥ A_required): 72%

With α ~ 1 and ε ~ 2-5%, RTM explains:
- Moderate galaxies (M_star ~ 10¹⁰) at z > 15
- Massive galaxies (M_star ~ 10¹¹) at z ~ 10
```

---

## Outputs

Each tool generates:

| File | Description |
|------|-------------|
| `*.py` | Main Python script |
| `*.ipynb` | Jupyter notebook for interactive use |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container for reproducibility |
| `README.md` | Tool-specific documentation |
| `output/*.csv` | Numerical results |
| `output/*.png` | Visualization plots |
| `output/*.pdf` | Publication-quality figures |
| `output/*_summary.txt` | Human-readable summary |

---

## Usage

### Direct Execution
```bash
cd T1_acceleration_factor
pip install -r requirements.txt
python T1_acceleration_factor.py
```

### Docker
```bash
cd T1_acceleration_factor
docker build -t t1_rtm .
docker run -v $(pwd)/output:/app/output t1_rtm
```

### Jupyter
```bash
jupyter notebook T1_acceleration_factor.ipynb
```

## Scientific Context

### The Problem
JWST has observed galaxies at z > 10 that appear "too massive, too early" — their stellar masses suggest they formed faster than standard ΛCDM models predict.

### The RTM Solution
RTM proposes that mesoscopic timescales (cooling, collapse, star formation) scale as T ∝ L^α. In the early universe where L_env (Hubble scale) was smaller, these processes ran faster by a factor A.

### The Prediction
With α ~ 1:
- At z = 10: processes are ~20× faster (ΛCDM) or ~37× faster (EdS)
- At z = 15: processes are ~36× faster (ΛCDM) or ~64× faster (EdS)

This naturally explains rapid galaxy assembly without exotic physics.

### Falsifiability
The hypothesis is falsifiable:
1. Time-scale relations within the same z should show T ∝ L^α
2. Star formation efficiency should be higher at high z (due to the A factor)
3. No α effects should appear in BBN/CMB (homogeneous plasma era)

---

## Dependencies

- Python 3.8+
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Pandas ≥ 1.3.0
- Matplotlib ≥ 3.4.0

---

## License

CC BY 4.0

---

## Reference

Paper: "Time–Scale Rescaling in Early Universe Structure Growth"
RTM Corpus Document: 004

Tools created: February 2026
