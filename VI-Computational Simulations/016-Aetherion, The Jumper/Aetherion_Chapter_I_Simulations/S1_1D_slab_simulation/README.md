# S1: 1-D Slab Aetherion Simulation

From "Aetherion, The Jumper" - Chapter I, Section 4

## Overview
Solves the coupled Poisson-type equations for the Aetherion field φ and RTM exponent α in a 1-D slab geometry.

## Key Equations
```
∇²φ - m_φ²φ = -γ|∇α|²
Power proxy: P = γ(∇α)·(∇φ)
```

## Parameters (from paper)
- N = 60 nodes
- m_φ = 1.0, M = 0.5, γ = 0.8
- α: 2.0 → 3.0 (linear ramp)

## Usage
```bash
python S1_1D_slab_simulation.py
```

## Expected Results
- φ_max ≈ 0.09 at midpoint
- ⟨|P|⟩ > 0 (energy extraction)
- P = 0 when ∇α = 0 (control)
