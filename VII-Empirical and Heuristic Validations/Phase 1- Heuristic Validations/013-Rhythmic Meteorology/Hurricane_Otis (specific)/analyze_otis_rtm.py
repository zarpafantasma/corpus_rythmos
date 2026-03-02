import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Data (User must download IBTrACS csv)
FILE_PATH = "ibtracs.last3years.list.v04r00.csv"

def analyze_hurricane_otis():
    print("Loading IBTrACS data...")
    # Skip unit row (row 1)
    df = pd.read_csv(FILE_PATH, header=0, skiprows=[1], low_memory=False)
    
    # 2. Filter for OTIS 2023
    otis = df[(df['NAME'] == 'OTIS') & (df['SEASON'] == 2023)].copy()
    
    # Clean Data
    otis['ISO_TIME'] = pd.to_datetime(otis['ISO_TIME'])
    otis['USA_WIND'] = pd.to_numeric(otis['USA_WIND'], errors='coerce')
    otis['USA_PRES'] = pd.to_numeric(otis['USA_PRES'], errors='coerce')
    otis = otis.dropna(subset=['USA_WIND', 'USA_PRES']).sort_values('ISO_TIME')
    
    # 3. RTM Calculation (Alpha Proxy)
    # Delta P = Ambient Pressure (1010 mb) - Central Pressure
    otis['Delta_P'] = 1010 - otis['USA_PRES']
    otis = otis[otis['Delta_P'] > 0]
    
    # Rolling Log-Log Slope (Window = 4 points / ~12 hours)
    def get_slope(series_x, series_y, window=4):
        slopes = [np.nan] * window
        lx = np.log(series_x.values)
        ly = np.log(series_y.values)
        for i in range(window, len(lx)):
            if np.std(lx[i-window:i]) > 0:
                res = stats.linregress(lx[i-window:i], ly[i-window:i])
                slopes.append(res.slope)
            else:
                slopes.append(np.nan)
        return slopes

    otis['RTM_Slope'] = get_slope(otis['Delta_P'], otis['USA_WIND'])
    
    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot Intensity
    ax1.plot(otis['ISO_TIME'], otis['USA_WIND'], 'r-o', label='Wind (kts)')
    ax1.set_ylabel('Wind Speed (kts)', color='r')
    ax1b = ax1.twinx()
    ax1b.plot(otis['ISO_TIME'], otis['USA_PRES'], 'b--', label='Pressure (mb)')
    ax1b.set_ylabel('Pressure (mb)', color='b')
    ax1b.invert_yaxis()
    ax1.set_title('Hurricane Otis (2023): Traditional View')
    
    # Plot RTM Signal
    ax2.plot(otis['ISO_TIME'], otis['RTM_Slope'], 'purple', linewidth=2, label='Coherence (Slope)')
    ax2.set_ylabel('Wind-Pressure Coupling (k)')
    ax2.axvspan(pd.to_datetime('2023-10-24 06:00'), pd.to_datetime('2023-10-24 12:00'), 
                color='yellow', alpha=0.3, label='Pre-Cognitive Drop')
    ax2.set_title('RTM View: Structural Efficiency')
    ax2.legend()
    
    plt.savefig('Otis_RTM_Validation.png')
    print("Analysis saved to Otis_RTM_Validation.png")

if __name__ == "__main__":
    analyze_hurricane_otis()