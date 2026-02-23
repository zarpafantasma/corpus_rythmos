# Aetherion Chapter I: Vacuum-Energy Extraction Simulations

## Overview

This package contains three computational tools (S1-S3) that implement and validate the **Aetherion vacuum-energy extraction mechanism** described in Chapter I of "Aetherion, The Jumper".

The Aetherion hypothesis proposes that a quantum-confined scalar field φ can couple to spatial gradients in the RTM temporal-scaling exponent α to unlock zero-point energy.

---

## Theoretical Background

### The Aetherion Mechanism

**Key Equation (from paper Section 2.4):**
```
ℒ = ½(∂φ)² - ½m_φ²φ² + ½M²(∂α)² - V(α) + γ(∂α)·(∂φ)
```

Where:
- φ = Aetherion field (parameterizes local temporal coherence)
- α = RTM temporal-scaling exponent
- γ = coupling strength (dimension-4)
- m_φ = Aetherion mass parameter
- M = α-field stiffness

**Power Extraction:**
```
P = γ(∇α)·(∇φ)
```

The α-gradient acts as a "pump" that rectifies vacuum fluctuations, converting temporal latency into directed energy flow.

---

## Simulations

### S1: 1-D Slab Simulation

**File:** `S1_1D_slab_simulation/S1_1D_slab_simulation.py`

**Purpose:** Solve the coupled Poisson equations in a 1-D geometry.

**Setup:**
- 60 grid nodes on [0, 1]
- Linear α ramp: 2.0 → 3.0
- Dirichlet BCs: φ(0) = φ(L) = 0

**Results:**
- φ rises smoothly from boundaries to midpoint
- ⟨|P|⟩ = 0.142 (positive energy extraction)
- P = 0 when α is constant (control verified)

---

### S2: 2-D Simulation

**File:** `S2_2D_simulation/S2_2D_simulation.py`

**Purpose:** Extend to 2 dimensions with radial α profile.

**Setup:**
- 31×31 grid (as specified in paper)
- Radial α profile: min at edges, max at center
- Dirichlet BCs: φ = 0 on all boundaries

**Results:**
- φ_max = 0.109 at center where α is maximum
- ⟨P⟩ = 0.280 (higher than 1-D due to geometry)
- P = 0 at boundaries where ∇α = 0

---

### S3: Scaling Analysis

**File:** `S3_scaling_analysis/S3_scaling_analysis.py`

**Purpose:** Validate paper's scaling predictions.

**Tests:**
1. **P ∝ γ² scaling:** Fitted slope = 2.00 ± 0.00 ✓
2. **Mesh convergence:** < 1% change at N ≥ 120 ✓
3. **P vs ∇α:** Confirmed dependence on gradient magnitude ✓

---

## Key Findings

| Test | Paper Prediction | Result |
|------|------------------|--------|
| φ profile | Smooth, max at midpoint | ✓ Verified |
| P > 0 | When ∇α ≠ 0 | ✓ Verified |
| P = 0 | When α constant | ✓ Verified |
| P ∝ γ² | Slope = 2.0 | ✓ Slope = 2.00 |
| Convergence | < 1% at large N | ✓ N ≥ 120 |

**All paper predictions are validated by the simulations.**

---

## Usage

### Direct Execution
```bash
cd S1_1D_slab_simulation
pip install -r requirements.txt
python S1_1D_slab_simulation.py
```

### Docker
```bash
cd S1_1D_slab_simulation
docker build -t aetherion_s1 .
docker run -v $(pwd)/output:/app/output aetherion_s1
```

### Jupyter
```bash
jupyter notebook S1_1D_slab_simulation.ipynb
```

---

## Outputs

Each simulation generates:

| File | Description |
|------|-------------|
| `*.py` | Main Python script |
| `*.ipynb` | Jupyter notebook |
| `requirements.txt` | Dependencies |
| `Dockerfile` | Container definition |
| `README.md` | Documentation |
| `output/*.csv` | Numerical data |
| `output/*.png/pdf` | Plots |
| `output/*_summary.txt` | Results summary |

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

## Physical Interpretation

The simulations demonstrate that:

1. **An imposed α-gradient induces a non-trivial φ field** - The Aetherion field responds to spatial variations in the RTM exponent.

2. **The power proxy P is strictly positive** - This represents net energy extraction from the vacuum when ∇α ≠ 0.

3. **P vanishes without gradient** - The effect requires spatial variation in α, confirming it's not a numerical artifact.

4. **P scales as γ²** - Stronger coupling yields quadratically more power, as predicted by the Lagrangian structure.

---

## Experimental Implications

The paper (Section 5) proposes a prototype reactor:
- Cylindrical chamber with concentric metamaterial shells
- Each shell enforces a specific α value
- Combined effect creates radial α gradient
- Predicted power: ~1.8 µW (detectible with micro-calorimetry)

These simulations provide the theoretical foundation for such experiments.

---

## Reference

Paper: "Aetherion, The Jumper"  
Chapter: I - Vacuum-Energy Extraction via Temporal-Scaling Gradients  
RTM Corpus Document: 016

---

## License

CC BY 4.0
