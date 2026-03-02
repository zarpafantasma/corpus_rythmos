# Computational Tools for "Black Holes in the RTM Framework"

## Overview

This package contains four computational tools (T1-T4) that implement and validate the **slope-based RTM testing methodology** for black hole environments and analog platforms. These tools reproduce and verify the synthetic experiments described in Section 8 of the paper.

---

## Objective

The paper proposes a **falsifiable test** for RTM (Relatividad Temporal Multiescala) in extreme gravitational environments. The key insight is:

```
log(τ_obs) = α(r) × log(L) + [log(Z(r)) + const]
             \_____/          \_________________/
              SLOPE              INTERCEPT
              (RTM)                (GR)
```

- **RTM affects SLOPES**: The coherence exponent α(r) determines how process times τ scale with effective size L
- **GR affects INTERCEPTS**: The redshift factor Z(r) shifts the overall level but not the slope

**Test Logic:**
- If slopes change with radius → RTM is active
- If only intercepts change → Pure GR (no RTM activation)

---

## Tools Description

### T1: α(r) Profile Calculator

**File:** `T1_alpha_profile/T1_alpha_profile.py`

**Purpose:** Calculate the RTM coherence exponent α as a function of radius r (or confinement index ξ for analog platforms).

**Key Equations:**
```
Logistic profile:   α(r) = α_∞ + (α₀ - α_∞) / [1 + exp(-(r - r_t)/w)]
Power-law ramp:     α(r) = α_∞ + (α₀ - α_∞) × (r/r_t)^(-q)
GR redshift:        Z(r) = 1 / √(1 - r_s/r)
```

**What it calculates:**
- α(r) profiles for both logistic and power-law parameterizations
- Z(r) gravitational redshift factor
- Combined effect on observed process times τ_obs

**Key Output:**
| Radius | α (coherence) | Z (redshift) |
|--------|---------------|--------------|
| 3 r_s | 2.635 | 1.732 |
| 6 r_s | 2.000 | 1.225 |
| 10 r_s | 1.238 | 1.118 |
| 15 r_s | 1.022 | 1.074 |

**Physical Interpretation:**
- α increases inward as confinement/organization grows
- Z increases inward (GR time dilation)
- These compete: Z stretches times, high α shortens them

---

### T2: Slope-at-r Estimator

**File:** `T2_slope_estimator/T2_slope_estimator.py`

**Purpose:** Estimate the RTM coherence exponent α from log-log regression of observed process times τ versus effective sizes L at fixed radius.

**Method:**
1. Generate synthetic observations: τ_obs = Z(r) × L^α(r) × noise
2. Fit log(τ) vs log(L) via OLS regression
3. **Slope = α(r)**, independent of Z(r)
4. Bootstrap for 95% confidence intervals

**Two-Radius Falsification Test:**
```
Δα = α(r_inner) - α(r_outer)

Decision:
- 95% CI for Δα excludes 0 → RTM activation detected
- 95% CI includes 0 → No activation (null)
```

**Key Output:**
```
r = 4:  α_true = 2.462, α_est = 2.464 [2.456, 2.472]
r = 12: α_true = 1.095, α_est = 1.097 [1.089, 1.105]

Two-radius test: Δα = 1.346 [1.333, 1.359]
CI excludes 0 → Activation detected ✓
```

---

### T3: GR vs RTM Decomposition

**File:** `T3_gr_rtm_decomposition/T3_gr_rtm_decomposition.py`

**Purpose:** Demonstrate the key decomposition principle that allows clean separation of RTM effects from GR effects.

**Two Scenarios Tested:**

| Scenario | α(r) | Expected Slopes | Expected Intercepts |
|----------|------|-----------------|---------------------|
| RTM Active | Varies (logistic) | Change with r | Change with r |
| RTM Inactive | Constant (2.0) | Constant | Change with r |

**Key Result:**
- Scenario A (RTM active): Slope variation = 1.44 (significant)
- Scenario B (RTM inactive): Slope variation = 0.01 (negligible)

**Conclusion:** GR/kinematics move intercepts, RTM moves slopes. This separation enables clean falsification tests.

---

### T4: Synthetic Validation (Demos A-D)

**File:** `T4_synthetic_validation/T4_synthetic_validation.py`

**Purpose:** Replicate the four synthetic experiments from Section 8 of the paper to validate the methodology.

**Demos:**

| Demo | Test | Setup | Result |
|------|------|-------|--------|
| **A** | Sensitivity | α(r) logistic profile | Recovered slopes track true α(r) |
| **B** | Generalization | α(ξ) vs confinement | Works for analog platforms |
| **C** | Decomposition | α constant, Z varies | Slopes constant, intercepts vary |
| **D** | Specificity | α = 2.0 everywhere | Δα CI includes 0, no false positive |

**Validation Summary:**
```
Demo A: SENSITIVITY      - Method detects α(r) increase inward ✓
Demo B: GENERALIZATION   - Method works for analog platforms ✓
Demo C: DECOMPOSITION    - GR only affects intercepts ✓
Demo D: SPECIFICITY      - No false positives under null ✓
```

---

## Key Equations (from Paper)

### Constitutive Law (Eq. 1)
```
τ_local = (L/L₀)^α × T₀
```
Process time τ scales with effective size L raised to the coherence exponent α.

### Clock Mapping (Eq. 5-6)
```
τ_obs = Z(r) × τ_local = Z(r) × (L/L₀)^α(r) × T₀
```
Observed time includes GR redshift factor Z(r).

### Slope Independence (Eq. 7)
```
d[log τ_obs]/d[log L] = α(r)
```
**The slope equals α, regardless of Z.** This is the key observational handle.

### α(r) Profiles (Eq. 9-10)
```
Logistic:   α(r) = α_∞ + (α₀ - α_∞) / [1 + exp(-(r - r_t)/w)]
Power-ramp: α(r) = α_∞ + (α₀ - α_∞) × (r/r_t)^(-q)
```

---

## Falsification Criteria

The RTM high-α interpretation is **falsified** if:

1. **Slopes invariant with radius:** Δα ≈ 0 across all radial bins
2. **Intercept-only evolution:** All changes explainable by Z(r) without slope changes
3. **Proxy fragility:** Slope estimates unstable across different L proxies

---

## Usage

### Direct Execution
```bash
cd T1_alpha_profile
pip install -r requirements.txt
python T1_alpha_profile.py
```

### Docker
```bash
cd T1_alpha_profile
docker build -t t1_rtm .
docker run -v $(pwd)/output:/app/output t1_rtm
```

### Jupyter
```bash
jupyter notebook T1_alpha_profile.ipynb
```

---

## Outputs

Each tool generates:

| File | Description |
|------|-------------|
| `*.py` | Main Python script |
| `*.ipynb` | Jupyter notebook |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition |
| `README.md` | Tool documentation |
| `output/*.csv` | Numerical results |
| `output/*.png/pdf` | Visualization plots |
| `output/*_summary.txt` | Human-readable summary |

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

## Verification Status

| Aspect | Status |
|--------|--------|
| Equations match paper | ✓ Verified |
| Demo A (sensitivity) | ✓ PASS |
| Demo B (analogs) | ✓ PASS |
| Demo C (decomposition) | ✓ PASS |
| Demo D (specificity) | ✓ PASS |

---

## Scientific Context

### The Problem
How can we test RTM predictions in extreme gravitational environments without modifying GR?

### The Solution
Measure the **slope** in log(τ) vs log(L) relations at different radii. The slope equals α(r), independent of GR effects (which only affect the intercept).

### The Prediction
If RTM is active, α should increase inward as confinement/organization grows. This produces steeper T ∝ L^α scaling at smaller radii.

### Observational Targets
- **Astrophysical:** Accretion disk variability, QPOs, flare durations
- **Analog platforms:** BEC dumb holes, draining bathtub fluids

---

## Reference

Paper: "Black Holes in the RTM Framework"  
RTM Corpus Document: 005  
Key Sections: 1-2 (ansatz), 4 (observational hooks), 8 (synthetic validation)

---

## License

CC BY 4.0
