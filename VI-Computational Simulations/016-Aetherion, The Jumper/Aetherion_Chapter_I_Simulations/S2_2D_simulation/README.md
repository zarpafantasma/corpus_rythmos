# S2: 2-D Aetherion Simulation

From "Aetherion, The Jumper" - Chapter I, Section 4.7

## Overview
Extends the 1-D simulation to 2 dimensions using a finite-difference sparse solver.

## Key Equations
```
∇²φ - m²φ = -γ|∇α|²
Power proxy: P = γ|∇α·∇φ|
```

## Parameters (from paper)
- 31×31 grid
- Radial α profile (min at edges, max at center)
- Dirichlet BCs: φ = 0 on boundaries

## Usage
```bash
python S2_2D_simulation.py
```

## Expected Results
- φ rises from zero at walls toward center
- φ_max where α is maximum
- P > 0 in interior
