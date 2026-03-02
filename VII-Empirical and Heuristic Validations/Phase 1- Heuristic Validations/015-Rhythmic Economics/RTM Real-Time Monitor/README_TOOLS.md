# RTM Real-Time Monitor: User Guide 📈

**Version:** 5.0 (Kraken Edition)
**Target Asset:** Bitcoin (BTC)

## 1. Overview
The RTM Monitor is a Python-based tool designed to measure the **Coherence Exponent ($\alpha$)** of the Bitcoin market in real-time. Unlike traditional indicators that track price direction, RTM tracks **structural integrity**.

It is based on the physical law derived in *Rhythmic Economics*:
$$T \propto L^\alpha$$
Where Time ($T$) scales relative to Liquidity Structure ($L$).

## 2. Included Files
This folder contains three scripts:

1.  **`rtm_diagnostic.py`**
    * **Purpose:** Connectivity Test.
    * **Use:** Run this first to check if your computer (or cloud environment) can connect to crypto exchanges (Kraken/Binance). It detects firewall issues or IP bans.

2.  **`rtm_monitor_colab.py`**
    * **Purpose:** Cloud Monitor.
    * **Use:** Copy and paste this code into **Google Colab**. It includes an auto-installer for dependencies and formats the output specifically for the browser.

3.  **`rtm_monitor_local.py`**
    * **Purpose:** Local Desktop Monitor.
    * **Use:** Run this on your personal computer (Windows/Mac/Linux) via terminal.
    * **Features:** Includes an audible alarm (beep) when the market enters a critical state (Windows only).

## 3. How to Interpret the Signal
The monitor calculates the Alpha ($\alpha$) every 60 seconds using a rolling 1-hour window.

| Alpha ($\alpha$) | Color | Status | Meaning |
| :--- | :--- | :--- | :--- |
| **0.00 - 0.79** | 🟢 Green | **LAMINAR** | **Healthy.** The market is efficient. Superfluid state. Safe to trade. |
| **0.80 - 1.49** | 🔵 Blue | **TURBULENT** | **Active.** Volatility is increasing, but the structure is holding. Standard trading conditions. |
| **1.50 - 1.99** | 🟡 Yellow | **VISCOUS** | **Warning.** Systemic stress. The market is struggling to process volume. Caution advised. |
| **> 2.00** | 🔴 Red | **BIFURCATION** | **CRITICAL FAILURE.** The structure has fractured. High probability of a Flash Crash or massive volatility event. **EXIT MARKETS.** |

## 4. Requirements (For Local Use)
To run `rtm_monitor_local.py`, you must have Python installed.
Install the required libraries by running:

```bash
pip install pandas numpy scipy ccxt colorama

Tools created: February 2026
