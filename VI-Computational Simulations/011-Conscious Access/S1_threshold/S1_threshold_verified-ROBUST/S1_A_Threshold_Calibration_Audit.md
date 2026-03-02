# CONSCIOUS ACCESS
## Addendum S1-A: Calibration of the Psychophysical Detection Threshold
**Subject:** Resolving the Hurst Exponent ($H$) to RTM Coherence ($\alpha$) Mapping  
**Classification:** PSYCHOPHYSICS / RED TEAM AUDIT  
**Date:** March 2026  
**Reference:** S1_threshold

---

## 1. Executive Summary
Simulation S1 models conscious perception (e.g., Masked Detection Tasks) using Signal Detection Theory (SDT) reframed through the RTM topology. The hypothesis states that sensory stimuli only breach conscious awareness ("Report") if the macroscopic neural network exceeds the coherence threshold ($\alpha_c \ge 2.0$) at the exact moment of stimulus processing.

## 2. Mathematical Calibration (The Hurst Mapping)
The initial audit revealed a trans-domain scaling discrepancy: the raw extraction algorithms output values spanning $0.3 - 0.8$, appearing to contradict the $\alpha > 2.0$ rule established in Domain 010. 

The Red Team audit identified that the raw output constitutes the **Hurst Exponent ($H$)** of the time-series correlation. When scaled to the RTM topological parameter using the rigorous fractional Brownian motion (fBm) translation:
$$\alpha_{RTM} = 2H + 1$$

The theoretical consistency is perfectly restored:
* **Awake (Report):** $H \