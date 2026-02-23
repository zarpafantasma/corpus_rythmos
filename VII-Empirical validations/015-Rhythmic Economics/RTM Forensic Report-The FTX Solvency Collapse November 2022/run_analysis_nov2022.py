import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

# --- CONFIGURATION ---
FILE_NAME = "BTCUSDT-1m-2022-11.csv" # Ensure this file is in the same folder
WINDOW_SIZE = 60 # 1-hour rolling window

def load_and_process_data(file_path):
    print(f"Loading data from {file_path}...")
    
    # Define column names for Binance format
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
            'Taker buy quote asset volume', 'Ignore']
    
    try:
        df = pd.read_csv(file_path, names=cols)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found. Please place the CSV in this folder.")
        return None

    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
    
    # --- RTM TRANSFORMATION ---
    df['log_L'] = np.log(df['Volume'] + 1e-9)
    df['log_T'] = np.log(df['High'] - df['Low'] + 1e-9)
    
    # Clean infinite/NaN values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def calculate_rolling_alpha(df, window):
    print(f"Calculating RTM Alpha (Window: {window}m)...")
    
    alpha_values = [np.nan] * window
    log_L = df['log_L'].values
    log_T = df['log_T'].values
    
    for i in range(window, len(df)):
        x_win = log_L[i-window:i]
        y_win = log_T[i-window:i]
        
        if np.std(x_win) > 0 and np.std(y_win) > 0:
            res = stats.linregress(x_win, y_win)
            alpha_values.append(res.slope)
        else:
            alpha_values.append(np.nan)
            
    return alpha_values

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
    color = 'tab:red'
    ax2.set_ylabel('RTM Alpha', color=color)
    ax2.plot(df['Date'], df['Rolling_Alpha'], color=color, linewidth=0.8, alpha=0.6, label='Alpha')
    
    # Critical Thresholds
    ax2.axhline(y=1.5, color='orange', linestyle='--', label='Warning (1.5)')
    ax2.axhline(y=2.0, color='darkred', linestyle='--', label='Bifurcation (2.0)')
    
    plt.title('RTM Analysis: November 2022 (FTX Collapse)')
    plt.savefig('RTM_Nov2022_Full.png')
    print("Saved: RTM_Nov2022_Full.png")
    plt.close()

    # Plot 2: Zoom on Crash (Nov 7-11)
    mask = (df['Date'] >= '2022-11-07') & (df['Date'] <= '2022-11-11')
    zoom_df = df.loc[mask]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date (Nov 7-11)')
    ax1.set_ylabel('Price (USD)', color=color)
    ax1.plot(zoom_df['Date'], zoom_df['Close'], color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('RTM Alpha', color=color)
    ax2.plot(zoom_df['Date'], zoom_df['Rolling_Alpha'], color=color, linewidth=1.5)
    
    # Highlight Peak
    if not zoom_df.empty:
        peak_alpha = zoom_df['Rolling_Alpha'].max()
        peak_date = zoom_df.loc[zoom_df['Rolling_Alpha'].idxmax(), 'Date']
        print(f"Detected Peak Alpha: {peak_alpha:.4f} on {peak_date}")
        
        ax2.annotate(f'Peak: {peak_alpha:.2f}', 
                     xy=(peak_date, peak_alpha), 
                     xytext=(peak_date, peak_alpha + 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('Forensic Zoom: The Insolvency Event')
    plt.savefig('RTM_Nov2022_Zoom.png')
    print("Saved: RTM_Nov2022_Zoom.png")
    plt.close()

if __name__ == "__main__":
    df = load_and_process_data(FILE_NAME)
    if df is not None:
        df['Rolling_Alpha'] = calculate_rolling_alpha(df, WINDOW_SIZE)
        plot_analysis(df)
        print("Done.")