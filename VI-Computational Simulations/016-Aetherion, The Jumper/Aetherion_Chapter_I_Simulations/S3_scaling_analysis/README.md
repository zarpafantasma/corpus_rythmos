# S3: Scaling and Convergence Analysis

From "Aetherion, The Jumper" - Chapter I, Sections 4.5-4.6, 6.1

## Overview
Tests key scaling predictions from the paper:
1. P ∝ γ² (coupling squared)
2. Mesh convergence (< 1% change)
3. P depends on ∇α magnitude

## Paper Predictions
- "P scales approximately with γ²"
- "P changes by less than 1% once N is sufficiently large"

## Usage
```bash
python S3_scaling_analysis.py
```

## Expected Results
- Slope ≈ 2.0 in log(P) vs log(γ)
- Convergence at N ≥ 120
- P increases with Δα
