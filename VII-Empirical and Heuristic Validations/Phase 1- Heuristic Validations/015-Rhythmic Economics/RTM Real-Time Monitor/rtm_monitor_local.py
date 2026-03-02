import ccxt
import pandas as pd
import numpy as np
from scipy import stats
import time
from datetime import datetime
import sys
import os

# Colors for Terminal
try:
    from colorama import init, Fore, Style
    init()
    C_HEADER = Fore.MAGENTA
    C_BLUE = Fore.BLUE
    C_GREEN = Fore.GREEN
    C_WARN = Fore.YELLOW
    C_FAIL = Fore.RED
    C_RESET = Style.RESET_ALL
except ImportError:
    C_HEADER = ""
    C_BLUE = ""
    C_GREEN = ""
    C_WARN = ""
    C_FAIL = ""
    C_RESET = ""

# --- CONFIGURATION ---
TIMEFRAME = '1m'     
WINDOW = 60          
EXCHANGE_ID = 'kraken' # Defaulting to Kraken

def get_exchange():
    try:
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        exchange = exchange_class({'enableRateLimit': True})
        return exchange
    except Exception as e:
        print(f"{C_FAIL}Error connecting to {EXCHANGE_ID}: {e}{C_RESET}")
        return None

def fetch_data(exchange):
    try:
        # Kraken symbol handling
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
        
        if np.sum(vol) < 1.0: return None 
        
        log_L = np.log(vol + 1e-9)
        log_T = np.log((high - low) + 1e-9)
        
        if np.std(log_L) == 0 or np.std(log_T) == 0: return None
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
        return slope
    except Exception:
        return None

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(C_HEADER + "=" * 55)
    print(f"   RTM LIVE MONITOR v5.0 | {EXCHANGE_ID.upper()}")
    print("   Detecting Phase Bifurcation in Real-Time")
    print("=" * 55 + C_RESET)
    print(f"{'TIME':<10} | {'PRICE':<10} | {'ALPHA':<8} | {'STATUS':<15}")
    print("-" * 55)

def main():
    print_banner()
    print("Initializing connection to global markets...")
    
    exchange = get_exchange()
    if not exchange: return

    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        df = fetch_data(exchange)
        
        if df is not None and not df.empty:
            current_price = df['close'].iloc[-1]
            alpha = calculate_alpha(df)
            
            if alpha is not None:
                # Status Logic
                status = ""
                color = C_GREEN
                
                if alpha < 0.8:
                    status = "LAMINAR"
                    color = C_GREEN
                elif 0.8 <= alpha < 1.5:
                    status = "TURBULENT"
                    color = C_BLUE
                elif 1.5 <= alpha < 2.0:
                    status = "VISCOUS ⚠️"
                    color = C_WARN
                else: 
                    status = "BIFURCATION 🚨"
                    color = C_FAIL
                
                print(f"{timestamp} | ${current_price:<9.0f} | {color}{alpha:.4f}{C_RESET}   | {color}{status}{C_RESET}")
                
                # Sound Alarm (Windows Only)
                if alpha >= 2.0 and os.name == 'nt':
                    try:
                        import winsound
                        winsound.Beep(1000, 500)
                    except: pass
            else:
                print(f"{timestamp} | CALIBRATING... (Insufficient Data)")
        else:
            print(f"{timestamp} | NETWORK ERROR (Retrying...)")
        
        # Wait 60 seconds with simple loading bar
        sys.stdout.write(C_BLUE + "   [Waiting 60s for next candle...]\r" + C_RESET)
        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()