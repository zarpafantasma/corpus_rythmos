# --- STEP 1: AUTO-INSTALLER ---
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import ccxt
    import colorama
except ImportError:
    print("🛠️ Installing required libraries... please wait...")
    install("ccxt")
    install("colorama")

# --- STEP 2: RTM MONITOR ---
import ccxt
import pandas as pd
import numpy as np
from scipy import stats
import time
from datetime import datetime
from IPython.display import clear_output

# Configuration
SYMBOL_DISPLAY = 'BTC/USD' 
TIMEFRAME = '1m'     
WINDOW = 60          
EXCHANGE_ID = 'kraken' # Defaulting to Kraken for better connectivity

def get_exchange():
    try:
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        exchange = exchange_class({'enableRateLimit': True})
        return exchange
    except Exception as e:
        print(f"Error connecting to {EXCHANGE_ID}: {e}")
        return None

def fetch_data(exchange):
    try:
        # Handling Kraken's specific symbol naming
        symbol_to_fetch = 'BTC/USD'
        try:
            ohlcv = exchange.fetch_ohlcv(symbol_to_fetch, TIMEFRAME, limit=100)
        except:
            symbol_to_fetch = 'XBT/USD' 
            ohlcv = exchange.fetch_ohlcv(symbol_to_fetch, TIMEFRAME, limit=100)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception:
        return None

def calculate_alpha(df):
    try:
        recent = df.tail(WINDOW).copy()
        vol = recent['volume'].values
        high = recent['high'].values
        low = recent['low'].values
        
        # Noise Filter: If volume is near zero, data is invalid
        if np.sum(vol) < 1.0: return None 
        
        # RTM Physics: Log-Log Scale
        log_L = np.log(vol + 1e-9)
        log_T = np.log((high - low) + 1e-9)
        
        if np.std(log_L) == 0 or np.std(log_T) == 0: return None
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
        return slope
    except Exception:
        return None

def main():
    exchange = get_exchange()
    if not exchange: return

    print(f"🛰️ Connecting to {EXCHANGE_ID.upper()}... Initializing Feed...")
    
    history = [] 

    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        df = fetch_data(exchange)
        
        if df is not None and not df.empty:
            current_price = df['close'].iloc[-1]
            alpha = calculate_alpha(df)
            
            if alpha is not None:
                # RTM Semaphore Logic
                status = "LAMINAR"
                color_code = "🟢" 
                
                if alpha >= 0.8: 
                    status = "TURBULENT"
                    color_code = "🔵"
                if alpha >= 1.5: 
                    status = "VISCOUS"
                    color_code = "🟡"
                if alpha >= 2.0: 
                    status = "BIFURCATION"
                    color_code = "🔴"
                
                line = f"{timestamp} | BTC: ${current_price:,.0f} | Alpha: {alpha:.4f} | {color_code} {status}"
                history.append(line)
                if len(history) > 15: history.pop(0)
                
                # Update Display
                clear_output(wait=True)
                print(f"=== RTM LIVE MONITOR (Cloud Edition) ===")
                print(f"Source: {EXCHANGE_ID.upper()} | Window: {WINDOW} min")
                print("-" * 55)
                for log in history:
                    print(log)
                print("-" * 55)
                print(f"Last Update: {timestamp}. Next scanning in 60s...")
                
            else:
                print(".", end="") 
        else:
            print("x", end="")
        
        time.sleep(60)

if __name__ == "__main__":
    main()