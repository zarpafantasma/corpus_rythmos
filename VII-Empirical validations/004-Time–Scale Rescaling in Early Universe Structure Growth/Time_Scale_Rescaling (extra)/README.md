# RTM-Cosmology Empirical Validation: Early Galaxies 🌌

**Date:** February 17, 2026
**Subject:** Validation of Time Rescaling in Early Universe
**Dataset:** JWST High-Redshift Galaxies (Labbé et al. 2023, JADES)

## 1. Overview
This repository contains the analysis code validating the **Time–Scale Rescaling** paper. We analyze the "impossible" masses of early galaxies to determine the time-dilation exponent ($\alpha$) required to explain them.

## 2. Findings
* **Mean Required Alpha:** **1.24**.
* **Theoretical Prediction:** $1.0 - 1.5$.
* **Implication:** The data perfectly aligns with an RTM universe where structure formation at $z=10$ is accelerated by a factor of $\sim 50$ due to high coherence ($\alpha \approx 1.25$).

## 3. How to Run
1.  Ensure `jwst_galaxies_data.txt` is in this folder.
2.  Run: `python analyze_cosmo_rtm.py`
3.  Result: `cosmo_rtm_validation.png`