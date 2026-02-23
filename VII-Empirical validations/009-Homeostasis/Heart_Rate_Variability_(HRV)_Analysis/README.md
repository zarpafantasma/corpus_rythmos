# RTM-Homeostasis Empirical Validation: HRV Aging 💓

**Date:** February 17, 2026
**Subject:** Validation of Coherence Loss in Aging and Disease
**Dataset:** PhysioNet (Fantasia & CHF Databases)

## 1. Overview
This repository contains the analysis code validating the **Homeostasis** paper. We analyze the fractal scaling exponent ($\alpha$) of heart rate variability across different health states.

## 2. Findings
* **Young Healthy:** $\alpha \approx 1.05$ (Optimal).
* **Elderly Healthy:** $\alpha \approx 0.81$ (Degraded).
* **Heart Failure:** $\alpha \approx 0.55$ (Collapsed).
* **Implication:** Health is a state of high temporal coherence. Disease is a loss of this coherence.

## 3. How to Run
1.  Ensure `hrv_aging_data.txt.txt` is in this folder.
2.  Run: `python analyze_homeostasis_rtm.py`
3.  Result: `homeostasis_rtm_validation.png`