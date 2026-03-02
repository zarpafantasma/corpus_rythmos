# Critical Update: Calibration of the Conscious Access Threshold (S1_threshold)

This document details the mathematical evolution and calibration of the `S1_threshold` simulation following the cross-domain integrity audit conducted by the *Red Team*.

## 1. The Problem: The Discrepancy in the Old README
In the initial version of the simulation, the model established the following parameters for a Masked Detection Task:
* **Reported Threshold (Old):** $\alpha_{crit} \approx 0.50$
* **Conscious Subjects:** $\alpha \approx 0.60 - 0.77$
* **Unconscious Subjects:** $\alpha \approx 0.30 - 0.32$

**Theoretical Alert:** These values broke the internal consistency of the RTM theoretical framework. In Domain 010 (Rhythmic Neuroscience), we rigorously demonstrated that the universal threshold for conscious processing and global integration requires a topological exponent of **$\alpha \ge 2.0$**. A threshold of `0.50` implied severe spatial fragmentation, which is incompatible with wakefulness.

## 2. The Solution: Mapping the Hurst Exponent ($H$)
The source code audit revealed that the simulation engine was correct, but the output variable was mislabeled. The extraction algorithm was calculating

You can find everything inside the folder S1_threshold_verified-ROBUST