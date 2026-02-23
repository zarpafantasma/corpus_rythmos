# RTM-Atmo Empirical Validation: Hurricane Otis (2023) 🌀

**Date:** February 17, 2026
**Subject:** Preliminary Empirical Validation of Rhythmic Meteorology (RTM-Atmo)
**Target Event:** Rapid Intensification of Hurricane Otis (October 2023)

## 1. Overview
This repository contains the data and code used to validate the **RTM Cascade Framework** ($T \propto L^\alpha$) in atmospheric systems. We analyze the structural evolution of Hurricane Otis, which devastated Acapulco in 2023, to test if the RTM "Coherence Exponent" ($\alpha$) could predict its historic rapid intensification (RI) before standard kinetic metrics.

## 2. Contents
* `analyze_otis_rtm.py`: The Python script that performs the forensic analysis.
* `ibtracs.last3years.list.v04r00.csv`: The official NOAA best-track dataset (Source: IBTrACS v4).
* Chart Description (Visual Evidence)

## 3. Methodology
The script calculates the **Wind-Pressure Coupling Slope ($k$)**, a proxy for the structural coherence of the vortex.
* **Formula:** $k = \frac{\partial \ln(V)}{\partial \ln(\Delta P)}$
* **Variables:**
    * $V$: Maximum sustained wind speed (kts).
    * $\Delta P$: Pressure deficit ($1010 \text{mb} - P_{min}$).
* **Hypothesis:** A drop in $k$ indicates a transition to a "Superfluid" state (high efficiency), which should precede kinetic intensification.

## 4. How to Run
1.  Ensure you have Python installed with the required libraries:
    ```bash
    pip install pandas numpy matplotlib scipy
    ```
2.  Place the `.csv` file and the `.py` script in the same folder.
3.  Run the script:
    ```bash
    python analyze_otis_rtm.py
    ```
4.  The script will generate an image file named `Otis_RTM_Validation.png`.

## 5. Key Findings
The analysis reveals a distinct **Phase Bifurcation**:
* **The Signal:** The coupling slope dropped to a critical low of **0.37** at **09:00 UTC on Oct 24**, 2023.
* **The Lead Time:** This structural collapse occurred approx. **12 hours before** the peak acceleration of wind speeds.
* **Conclusion:** The RTM signal successfully predicted the explosive potential of the storm while it was still technically a Tropical Storm, validating the theory that *structural coherence precedes kinetic energy*.

## 7. Chart Description (Visual Evidence)

The script generates a two-panel forensic chart (`Otis_RTM_Validation.png`) visualizing the phase transition:

**Panel 1: The Traditional View (Top)**
* **Red Line:** Maximum Sustained Wind Speed (knots).
* **Blue Dashed Line:** Central Barometric Pressure (millibars, inverted).
* *Observation:* This panel shows the "Lagging" nature of energy. Note how wind speeds remain relatively flat (Tropical Storm status) even as the pressure begins to deepen. A standard observer seeing this at 06:00 UTC might underestimate the impending explosion.

**Panel 2: The RTM View (Bottom)**
* **Purple Line:** The **RTM Coherence Slope ($k$)**. This measures the thermodynamic efficiency of the storm engine.
* **Yellow Zone ("Pre-Cognitive Drop"):** This is the critical finding.
    * At **09:00 UTC**, the purple line crashes to its lowest point ($\approx 0.37$), indicating the vortex has achieved a "Superfluid" structural state.
    * *Significance:* This structural signal occurs **12 hours before** the wind speed (red line) goes vertical.
    * *Conclusion:* The RTM metric successfully alerted that the storm was primed for rapid intensification while it was still seemingly weak on the surface.

## 7. Data Source
* **NOAA IBTrACS:** International Best Track Archive for Climate Stewardship.
* **Version:** v04r00 (Last 3 Years).
* **DOI:** 10.25921/82ty-9e16
