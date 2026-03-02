# E4: Null Controls (Flat Slopes and Symmetric Causality)

## RTM Cascade Framework - Negative Control Validation

### Overview

This simulation verifies that the S1 and S2 tests correctly return NULL when there is no cascade structure. This protects against false positives.

### Null Models

**S1 Null (Flat Slopes):**
```
T_n(L) = c_n × L^α × ε
```
- α is CONSTANT across all layers
- Intercepts c_n vary (different level factors)
- No coherence increase → Δα should ≈ 0

**S2 Null (Symmetric Causality):**
```
Y_n(t) = ε_n(t)   (independent noise)
```
- No coupling between layers
- Granger F-stats should be similar in both directions

### Expected Outcomes

| Test | Expected Result | Interpretation |
|------|-----------------|----------------|
| S1 | All Δα CIs include 0 | No false monotone trend |
| S2 | Granger symmetric | No false directionality |

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ALPHA_CONSTANT | 2.3 | Same α for all layers |
| N_LAYERS | 4 | Number of layers |
| N_SAMPLES | 2000 | Time series length |

### Decision Rules

- **S1 Null Confirmed**: All Δα confidence intervals include 0
- **S2 Null Confirmed**: Granger tests symmetric (neither direction significant, or both similar)

### Files

- `E4_null_controls.py`: Main simulation
- `E4_null_controls.ipynb`: Jupyter notebook
- `requirements.txt`: Dependencies
- `Dockerfile`: Container
- `output/`: Results

### Usage

```bash
python E4_null_controls.py

# Docker
docker build -t e4_cascade .
docker run -v $(pwd)/output:/app/output e4_cascade
```

### Interpretation

If null controls are CONFIRMED:
- The S1/S2 methodology is specific (low false positive rate)
- Positive results in E1-E3 are meaningful

If null controls FAIL:
- The methodology may detect spurious patterns
- Review test parameters and thresholds

### Reference

RTM Cascade Framework, Section 4.4

### License

CC BY 4.0
