# RTM Forensic Report: The FTX Solvency Collapse (November 2022) 🏦

**Date of Analysis:** February 17, 2026
**Subject:** Bitcoin (BTC-USD) Microstructure Analysis
**Event Horizon:** November 01, 2022 – November 30, 2022

## 1. Executive Summary
This report examines the market behavior during the collapse of the FTX exchange. The objective was to characterize the "texture" of the crash using the **RTM Coherence Exponent** ($\alpha$) to distinguish it from liquidity crises (2020) and political shocks (2021).

**Key Finding:** The analysis identified a state of **Chronic Viscosity**. The Coherence Exponent remained elevated ($\alpha \approx 1.2 - 1.3$) for over 96 hours, indicating sustained systemic stress without a singular catastrophic fracture point ($\alpha > 2.0$).

## 2. Methodology
* **Data Source:** 1-minute interval OHLCV data from Binance (BTCUSDT).
* **Metric:** Rolling Coherence Exponent ($\alpha$) calculated over a 60-minute window.
* **Formula:** $\alpha = \frac{\partial \ln(\text{Volatility})}{\partial \ln(\text{Volume})}$

## 3. The "FTX Week" Timeline (Nov 6-11, 2022)

| Time (UTC) | Event | RTM Alpha ($\alpha$) | Physical State |
| :--- | :--- | :--- | :--- |
| **Nov 06** | CZ (Binance) announces selling FTT | **0.85** | **Laminar Flow.** Market digests news efficiently. |
| **Nov 08, 16:00** | Binance signs LOI to buy FTX | **1.15** | **Turbulence.** Uncertainty spikes viscosity. |
| **Nov 08, 20:02** | **LOI Collapses / Insolvency Realized** | **1.30 (Peak)** | **High Viscosity.** Maximum stress, but structure holds. |
| **Nov 09-11** | Bankruptcy Filing | **1.10 - 1.25** | **Chronic Drag.** Market remains "heavy" but functional. |

## 4. Physical Interpretation (The "Sick" Market)
The FTX crisis presents a unique RTM signature compared to previous events:
* **Signature:** A "plateau" of high alpha rather than a spike.
* **Physics:** The market acted like a thick syrup (High Viscosity). Every price movement required immense volume to execute because trust had evaporated. Market makers widened spreads to protect against contagion, slowing down financial time, but liquidity did not vanish completely as it did in March 2020.

## 5. Conclusion
RTM classifies the FTX collapse as a **Systemic Solvency Crisis**. The absence of a Phase Bifurcation ($\alpha > 2.0$) correctly signaled that while the price would drop significantly ($21k \to $15k), the underlying mechanism of Bitcoin trading remained intact.