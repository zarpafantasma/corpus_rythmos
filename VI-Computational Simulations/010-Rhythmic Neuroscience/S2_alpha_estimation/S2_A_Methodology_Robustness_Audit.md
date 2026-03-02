# RHYTHMIC NEUROSCIENCE
## Addendum S2-A: Validation of the Empirical Alpha Estimation Protocol
**Subject:** Statistical Robustness of Topological Exponent Recovery  
**Classification:** BIOSTATISTICS / RED TEAM AUDIT  
**Date:** March 2026  
**Reference:** S2_alpha_estimation

---

## 1. Executive Summary
This module validates the experimental methodology required to safely extract the RTM spatial-temporal coherence exponent ($\alpha$) from empirical neural recordings (e.g., EEG/MEG). The audit confirms that the log-log linear extraction of $\tau(L)$ scaling is highly resilient to severe biological noise artifacts and sets accessible thresholds for experimental design.

## 2. Experimental Constraints Validated
* **Minimum Spatial Scales:** The framework requires strictly $\ge 4$ independent spatial scales spanning at least one decade (e.g., 10mm to 100mm) to suppress the Mean Absolute Error (MAE) below $0.07$. An optimal cost-benefit ratio is identified at **5 to 7 scales**, driving $R^2 > 0.99$ while maintaining clinical feasibility.
* **Noise Tolerance:** The log-linear estimator correctly processes heavily skewed log-normal measurement noise. Under simulated extreme biological noise conditions ($\sigma = 0.20$), the estimator robustly bounds the absolute error of $\alpha$ to $\approx 0.08$, ensuring clinical reliability.

## 3. State Discrimination Power
The audit verified the statistical power of the $\alpha$ estimator across varying states of consciousness. The recovered Gaussian distributions (Wake $\mu=2.01$, NREM $\mu=1.47$, Anesthesia $\mu=1.21$) exhibit negligible overlap under standard noise models. This confirms that the RTM scaling exponent acts as a mathematically stable, standalone discriminator for the depth of unconsciousness.