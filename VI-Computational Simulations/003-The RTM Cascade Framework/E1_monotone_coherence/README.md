# E1: Four-Layer Cascade with Non-Decreasing Coherence

## RTM Cascade Framework - Signature S1 Validation

### Overview

This simulation validates **Signature S1** of the RTM Cascade Framework: that the coherence exponent α increases (or at least does not decrease) across nested layers in a cascade architecture.

### Model

For each layer n = 0, 1, 2, 3:

```
T_n(L) = c_n × L^α_n × ε
```

Where:
- **α_n = α_0 + n × Δα**: Monotone increasing coherence exponent
- **c_n**: Layer-level factor (affects intercept only, not slope)
- **ε ~ LogNormal(0, σ²)**: Multiplicative noise
- **L**: Effective size (geometric grid)

### Method

1. Generate synthetic data for 4 layers with known α values
2. For each layer, regress log(T) on log(L) via OLS
3. Compute 95% bootstrap confidence intervals for α
4. Test if all adjacent differences Δα have CI lower bound > -ε_tol

### Decision Rule (S1)

**Pass** if all Δα_{n,n+1} satisfy: CI_lower > -ε_tol (default ε_tol = 0.05)

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_LAYERS | 4 | Number of cascade layers |
| ALPHA_BASE | 2.0 | Base coherence exponent (α_0) |
| DELTA_ALPHA | 0.3 | Increase per layer |
| L_MIN, L_MAX | 10, 200 | Size range |
| N_SIZES | 10 | Sizes per layer |
| N_EVENTS | 50 | Events per (layer, size) |
| NOISE_SIGMA | 0.15 | Log-normal noise scale |
| N_BOOTSTRAP | 1000 | Bootstrap replicates |

### Expected Results

- **Layer 0**: α ≈ 2.0
- **Layer 1**: α ≈ 2.3
- **Layer 2**: α ≈ 2.6
- **Layer 3**: α ≈ 2.9

S1 Test: **PASS** (coherence increases monotonically)

### Files

- `E1_monotone_coherence.py`: Main simulation script
- `E1_monotone_coherence.ipynb`: Jupyter notebook
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container for reproducibility
- `output/`: Results directory
  - `E1_raw_data.csv`: All generated observations
  - `E1_layer_results.csv`: Per-layer α estimates and CIs
  - `E1_S1_test_results.csv`: S1 test outcomes
  - `E1_summary.txt`: Human-readable summary
  - `E1_monotone_coherence_results.png/pdf`: Visualizations

### Usage

```bash
# Direct execution
python E1_monotone_coherence.py

# With Docker
docker build -t e1_cascade .
docker run -v $(pwd)/output:/app/output e1_cascade
```

### Reference

RTM Cascade Framework, Section 4.1: "E1 — Four-layer cascade with non-decreasing coherence"

### License

CC BY 4.0
