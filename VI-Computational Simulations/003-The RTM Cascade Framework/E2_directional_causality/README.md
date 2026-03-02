# E2: Directional Causality in a Layered Chain

## RTM Cascade Framework - Signature S2 Validation

### Overview

This simulation validates **Signature S2** of the RTM Cascade Framework: that information flow is asymmetric (forward-only) between adjacent layers.

### Model

Forward-coupled time series:

```
Y_0(t) = ε_0(t)                           (base layer, pure noise)
Y_n(t) = κ × Y_{n-1}(t-1) + ε_n(t)        (forward coupling only)
```

Where:
- **κ**: Forward coupling strength (default 0.7)
- **ε_n(t) ~ N(0, σ²)**: Independent Gaussian noise
- No backward coupling (by construction)

### Method

For each adjacent pair (n-1, n):

1. **Transfer Entropy (TE)**
   - Estimate TE(n-1 → n) and TE(n → n-1)
   - Test significance using surrogate data (time-shuffle)
   
2. **Granger Causality**
   - F-test comparing restricted vs full VAR model
   - Test both directions

### Decision Rule (S2)

**Pass** if for all pairs:
- TE forward significant AND TE reverse not significant
- Granger forward significant AND Granger reverse not significant

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_LAYERS | 4 | Number of layers |
| N_SAMPLES | 2000 | Time series length |
| COUPLING_STRENGTH | 0.7 | κ (forward coupling) |
| NOISE_STD | 0.5 | σ (noise level) |
| MAX_LAG | 5 | Maximum lag for TE/Granger |
| N_SURROGATES | 500 | Surrogates for significance |
| ALPHA_SIG | 0.05 | Significance level |

### Expected Results

- **TE Forward >> TE Reverse** for all pairs
- **Granger F Forward >> F Reverse** for all pairs
- Forward direction significant (p < 0.05)
- Reverse direction not significant (p > 0.05)

### Files

- `E2_directional_causality.py`: Main simulation
- `E2_directional_causality.ipynb`: Jupyter notebook
- `requirements.txt`: Dependencies
- `Dockerfile`: Container
- `output/`: Results
  - `E2_time_series.csv`: Generated time series
  - `E2_S2_test_results.csv`: TE/Granger results
  - `E2_summary.txt`: Summary
  - `E2_*.png/pdf`: Plots

### Usage

```bash
python E2_directional_causality.py

# Docker
docker build -t e2_cascade .
docker run -v $(pwd)/output:/app/output e2_cascade
```

### Note on Statistical Variability

Due to the stochastic nature of the simulation, occasional false positives in the reverse direction may occur (p-values near the threshold). This is expected behavior and reflects real-world statistical testing challenges. FDR correction across multiple comparisons is recommended for formal analysis.

### Reference

RTM Cascade Framework, Section 4.2

### License

CC BY 4.0
