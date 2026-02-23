import sys
import subprocess

# 1. Dependency Check
print("🛠️ Checking dependencies...")
try:
    import ccxt
    print("✅ CCXT is installed.")
except ImportError:
    print("⚠️ CCXT not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt
    print("✅ Installation complete.")

# 2. Connectivity Test
def test_connection():
    print("\n📡 STARTING CONNECTIVITY TEST...")
    
    # We test multiple exchanges to rule out geo-blocking
    exchanges_to_test = ['kraken', 'binance', 'coinbase']
    
    for exchange_id in exchanges_to_test:
        print(f"\n--- Pinging {exchange_id.upper()} ---")
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'timeout': 10000}) 
            
            # Fetch Ticker
            print(f"⏳ Requesting BTC price from {exchange_id}...")
            # Kraken sometimes uses XBT, others use BTC. CCXT usually handles this, 
            # but we force a common pair check.
            try:
                ticker = exchange.fetch_ticker('BTC/USDT')
            except:
                ticker = exchange.fetch_ticker('BTC/USD')
            
            price = ticker['last']
            print(f"✅ SUCCESS! {exchange_id} is online.")
            print(f"💰 Current BTC Price: ${price}")
            return True 
            
        except Exception as e:
            print(f"❌ FAILED {exchange_id}.")
            print(f"   Error: {e}")
            if "451" in str(e) or "Geo" in str(e):
                print("   (Likely Geo-blocking or IP restriction)")

    return False

if __name__ == "__main__":
    if test_connection():
        print("\n🎉 RESULT: Your environment is READY for the RTM Monitor.")
    else:
        print("\n💀 RESULT: No connection established. Check your firewall/internet.")