import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

# --- CONFIGURATION ---
FILE_NAME = "BTCUSDT-1m-2023-09.csv" 
WINDOW_SIZE = 60 # 1-hour rolling window
NOISE_FILTER_USD = 5.0 # Minimum volatility to effectively calculate Alpha

def load_and_process_data(file_path):
    print(f"Loading data from {file_path}...")
    
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
            'Taker buy quote asset volume', 'Ignore']
    
    try:
        df = pd.read_csv(file_path, names=cols)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
        return None

    # Handle Microsecond vs Millisecond Timestamps
    first_ts = df['Open time'].iloc[0]
    unit = 'us' if first_ts > 1e14 else 'ms'
    df['Date'] = pd.to_datetime(df['Open time'], unit=unit)
    
    # --- RTM TRANSFORMATION ---
    df['log_L'] = np.log(df['Volume'] + 1e-9)
    df['log_T'] = np.log(df['High'] - df['Low'] + 1e-9)
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def calculate_filtered_alpha(df, window, min_volatility):
    print(f"Calculating RTM Alpha (Filter > ${min_volatility})...")
    slopes = [np.nan] * window
    
    log_L = df['log_L'].values
    log_T = df['log_T'].values
    high = df['High'].values
    low = df['Low'].values
    
    for i in range(window, len(df)):
        # Extract window
        l_win = log_L[i-window:i]
        t_win = log_T[i-window:i]
        h_win = high[i-window:i]
        lo_win = low[i-window:i]
        
        # Apply Noise Filter
        mask = (h_win - lo_win) > min_volatility
        
        l_clean = l_win[mask]
        t_clean = t_win[mask]
        
        # Require at least 10 valid data points to calculate slope
        if len(l_clean) > 10 and np.std(l_clean) > 0 and np.std(t_clean) > 0:
            res = stats.linregress(l_clean, t_clean)
            slopes.append(res.slope)
        else:
            slopes.append(np.nan)
            
    return slopes

def plot_analysis(df):
    print("Generating plots...")
    
    # Plot 1: Full Month
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)', color=color)
    ax1.plot(df['Date'], df['Close'], color=color, linewidth=1, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('RTM Alpha (Healthy State)', color=color)
    ax2.plot(df['Date'], df['Filtered_Alpha'], color=color, linewidth=0.8, alpha=0.7, label='Alpha')
    
    # Thresholds
    ax2.axhline(y=1.5, color='orange', linestyle='--', label='Warning')
    ax2.axhline(y=2.0, color='darkred', linestyle='--', label='Critical')
    ax2.set_ylim(-0.5, 2.5) # Fixed scale for comparison
    
    plt.title('RTM Control Group: September 2023 (Laminar Baseline)')
    plt.savefig('RTM_Sept2023_Control.png')
    print("Saved: RTM_Sept2023_Control.png")
    plt.close()

    # Plot 2: Zoom Week
    mask = (df['Date'] >= '2023-09-10') & (df['Date'] <= '2023-09-17')
    zoom_df = df.loc[mask]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(zoom_df['Date'], zoom_df['Close'], color='tab:blue')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
    
    ax2 = ax1.twinx()
    ax2.plot(zoom_df['Date'], zoom_df['Filtered_Alpha'], color='tab:green')
    ax2.set_ylim(-0.5, 2.0)
    
    # Annotate Stats
    avg_alpha = zoom_df['Filtered_Alpha'].mean()
    plt.title(f'Forensic Zoom: Stable Laminar Flow (Avg Alpha = {avg_alpha:.2f})')
    plt.savefig('RTM_Sept2023_Zoom.png')
    print("Saved: RTM_Sept2023_Zoom.png")
    plt.close()

if __name__ == "__main__":
    df = load_and_process_data(FILE_NAME)
    if df is not None:
        df['Filtered_Alpha'] = calculate_filtered_alpha(df, WINDOW_SIZE, NOISE_FILTER_USD)
        plot_analysis(df)
        print("Done.")