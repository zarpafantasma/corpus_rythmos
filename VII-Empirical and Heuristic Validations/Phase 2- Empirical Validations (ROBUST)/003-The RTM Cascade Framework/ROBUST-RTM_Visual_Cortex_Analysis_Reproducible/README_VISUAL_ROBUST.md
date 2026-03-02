# Robust RTM Empirical Validation: Visual Cortex 👁️

**Phase 2: "Red Team" ODR & Variance Correction**

## The Upgrade
The initial Phase 1 pipeline utilized Ordinary Least Squares (OLS) on 21 heavily aggregated data points. This introduced two fatal statistical vulnerabilities:
1. **Attenuation Bias:** OLS assumes $X$ (Receptive Field Size) is measured perfectly. In reality, fMRI/electrode measurements possess massive error bars. This error artificially forces the slope ($\\alpha$) toward zero.
2. **Aggregation Bias:** Compressing thousands of individual measurements into 21 dots artificially inflated the $R^2$ to $0.92$.

This **Phase 2 Pipeline** properly executes the RTM mathematical mandate (Errors-In-Variables). It utilizes Orthogonal Distance Regression (ODR) to absorb bidirectional variance and reconstructs the underlying neuronal/subject-level population to test the limits of the theory.

## Key Findings: The Sub-Diffusive Triumph

| Metric | Phase 1 (Flawed) | Phase 2 (Robust) |
|--------|------------------|------------------|
| **True RTM α** | 0.303 | **0.311 ± 0.021** (ODR) |
| **Population α** | N/A | **0.281** (Subject-level) |
| **R² (Natural Variance)**| 0.921 (Inflated) | **0.677** (Realistic) |

**Conclusion:** Even when properly punishing the model with extreme observational noise and un-aggregating the hierarchy, the $\alpha$ exponent remains strictly below $0.50$. 

This validates RTM physically: The brain's architecture operates in a **Sub-Diffusive Transport Class**. It leverages massive, parallel hierarchical topology to actively bypass the physical latency limits of standard thermal diffusion (where $\alpha$ would equal 0.5).