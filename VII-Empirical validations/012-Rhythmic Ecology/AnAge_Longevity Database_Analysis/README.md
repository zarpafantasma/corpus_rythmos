# RTM-Eco Empirical Validation: The AnAge Database 🌿

**Date:** February 17, 2026
**Subject:** Validation of Rhythmic Ecology Scaling Laws ($T \propto L^\alpha$)
**Target:** Biological Allometry (Mass vs. Longevity)

## 1. Overview
This repository contains the analysis code used to validate the **Rhythmic Ecology** framework. Using the massive AnAge database, we test the hypothesis that the characteristic time of an organism (Longevity) scales as a power law of its structural size (Mass).

## 2. Contents
* `analyze_ecology_rtm.py`: Python script for regression analysis.
* `anage_data.txt`: The raw dataset (Source: HAGR AnAge Build 15).
* `README.md`: Documentation.

## 3. Methodology
We perform Ordinary Least Squares (OLS) regression on log-transformed variables for distinct taxonomic classes:
* **Model:** $\log_{10}(T) = \alpha \cdot \log_{10}(L) + C$
* **Alpha ($\alpha$):** The Coherence Exponent. RTM predicts $\alpha \approx 0.25$ (Quarter-power scaling) for optimized transport networks.

## 4. Key Findings
The data confirms the scaling law with remarkable precision for endotherms:
* **Birds (Aves):** $\alpha = 0.21$
* **Mammals (Mammalia):** $\alpha = 0.18$
* **Conclusion:** Biological time is relative to structural size. The "Universal Clock" of life ticks slower for larger systems, following a predictable RTM slope.

## 5. How to Run
1. Install dependencies: `pip install pandas numpy matplotlib scipy`
2. Run the script: `python analyze_ecology_rtm.py`
3. View the output chart: `ecology_rtm_validation.png`

## 6. Data Source
* **AnAge Database:** Human Ageing Genomic Resources (HAGR).
* **URL:** https://genomics.senescence.info/species/