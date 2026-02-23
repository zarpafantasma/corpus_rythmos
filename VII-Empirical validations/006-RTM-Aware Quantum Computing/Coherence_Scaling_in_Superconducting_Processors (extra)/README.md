# RTM-Quantum Empirical Validation: IBM Hardware ⚛️

**Date:** February 17, 2026
**Subject:** Validation of Coherence Scaling in Superconducting Processors
**Dataset:** IBM Quantum Historical Calibration (2017-2023)

## 1. Overview
This repository contains the analysis code validating the **RTM-Aware Quantum Computing** framework. We test whether the coherence time ($T$) scales with processor size ($N$) according to a power law.

## 2. Findings
* **Scaling Law:** $T_1 \propto N^{0.36}$.
* **Coherence Exponent ($\alpha$):** **0.36**.
* **Implication:** The positive exponent confirms that IBM's architecture is in a "Coherent Scaling" regime. Larger processors are, on average, more coherent than smaller predecessors, validating the RTM principle of engineering $\alpha > 0$.

## 3. How to Run
1. Ensure `ibm_quantum_data.txt` is in this folder.
2. Run: `python analyze_quantum_rtm.py`
3. Result: `quantum_rtm_validation.png`