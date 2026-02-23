# E3: Ratchet/Hysteresis Under Coupling Sweeps

## RTM Cascade Framework - Signature S3 Validation (Supporting)

### Overview

This simulation tests **Signature S3**: whether the cascade exhibits directional memory (hysteresis) when inter-layer coupling is swept up and down.

### Model

Coherence α responds to coupling κ with memory:

```
α(κ) = α_base + Δα × sigmoid(κ - center)
```

Where `center` shifts based on sweep direction:
- **Up sweep**: center shifts left (earlier activation)
- **Down sweep**: center shifts right (delayed deactivation)

This creates a hysteresis loop.

### Method

1. Sweep κ from κ_min to κ_max (forward branch)
2. Sweep κ from κ_max to κ_min (backward branch)
3. Measure α at each κ value
4. Compute hysteresis loop area: A_hyst = ∫_{up} α dκ - ∫_{down} α dκ
5. Bootstrap CI for A_hyst

### Decision Rule (S3)

**Pass** if A_hyst bootstrap CI excludes 0 for ≥50% of layers

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| KAPPA_MIN/MAX | 0.1, 0.9 | Coupling range |
| N_KAPPA_STEPS | 20 | Steps per sweep |
| MEMORY_STRENGTH | 0.4 | Hysteresis magnitude |
| N_LAYERS | 4 | Cascade layers |

### Expected Results

- Hysteresis loop area A > 0
- CI excludes zero
- S3: PASS

### Files

- `E3_hysteresis.py`: Main simulation
- `E3_hysteresis.ipynb`: Notebook
- `requirements.txt`: Dependencies
- `Dockerfile`: Container
- `output/`: Results

### Usage

```bash
python E3_hysteresis.py
```

### Reference

RTM Cascade Framework, Section 4.3

### License

CC BY 4.0
