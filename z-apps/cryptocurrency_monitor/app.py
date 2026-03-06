import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import ccxt
import os
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & PREMIUM CSS
# ==========================================
st.set_page_config(
    page_title="RTM Economic Radar",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Fintech Dark Theme
st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background-color: #0B0E14 !important; height: 0px; }
    [data-testid="stSidebar"] { background-color: #0F1219 !important; border-right: 1px solid #1E232B; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #FFFFFF !important; 
    }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
        background-color: #1E232B !important; color: #00E5FF !important; border: 1px solid #00E5FF !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px; width: 100%; transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover {
        background-color: #00E5FF !important; color: #0B0E14 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #151923; border-color: #1E232B; color: white; }
    h1, h2, h3, h4 { font-weight: 300 !important; letter-spacing: 1px; color: #FFFFFF !important; text-transform: uppercase; }
    div[data-testid="stMetric"] { background-color: #151923; border: 1px solid #1E232B; padding: 20px; border-radius: 8px; }
    .rtm-info-card { background-color: #151923; border: 1px solid #1E232B; padding: 30px; border-radius: 8px; margin-top: 25px; line-height: 1.6; }
    .health-card { background-color: #11141D; border: 1px solid #1E232B; padding: 15px; border-radius: 6px; text-align: center; }
    .rtm-footer { text-align: center; padding: 40px 0 20px 0; color: #4A5568; font-size: 0.85em; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINES (Cached)
# ==========================================
def get_noise_filter(symbol):
    """Returns the absolute noise filter based on asset scale."""
    if 'BTC' in symbol: return 5.0
    elif 'ETH' in symbol: return 0.5
    elif 'SOL' in symbol: return 0.05
    elif 'XRP' in symbol: return 0.0001
    else: return 1.0

@st.cache_data
def load_and_process_data(file_path):
    """Loads CSV files and calculates the RTM Coherence Exponent (Alpha)."""
    if not os.path.exists(file_path):
        st.error(f"SYSTEM ERROR: File '{file_path}' not found in the repository.")
        return None
        
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
            'Taker buy quote asset volume', 'Ignore']
    try:
        df = pd.read_csv(file_path, names=cols)
        first_ts = df['Open time'].iloc[0]
        # Detect timestamp unit (ms vs us)
        unit = 'us' if first_ts > 1e14 else 'ms'
        df['Date'] = pd.to_datetime(df['Open time'], unit=unit)
        
        # Calculate log metrics for RTM Physics
        NOISE_FILTER_USD = 5.0
        df['log_L'] = np.log(df['Volume'] + 1e-9)
        spread = df['High'] - df['Low']
        spread_filtered = np.where(spread < NOISE_FILTER_USD, NOISE_FILTER_USD, spread)
        df['log_T'] = np.log(spread_filtered)
        
        window = 60
        cov = df['log_L'].rolling(window).cov(df['log_T'])
        var = df['log_L'].rolling(window).var()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_alpha = cov / var
            
        raw_alpha = pd.Series(raw_alpha).replace([np.inf, -np.inf], np.nan).fillna(0.45)
        df['Rolling_Alpha'] = raw_alpha.rolling(window=3, min_periods=1).mean()
        
        return df.dropna(subset=['Rolling_Alpha'])
    except Exception as e:
        st.error(f"DATA ERROR: Failed to process {file_path}. Details: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_live_rtm_data(symbol='BTC/USD'):
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=120)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        noise = get_noise_filter(symbol)
        df['log_L'] = np.log(df['Volume'] + 1e-9)
        spread = df['High'] - df['Low']
        df['log_T'] = np.log(np.where(spread < noise, noise, spread))
        
        window = 60
        cov = df['log_L'].rolling(window).cov(df['log_T'])
        var = df['log_L'].rolling(window).var()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = cov / (var + 1e-9)
        
        alpha = np.where(var < 1e-7, 0.45, alpha)
        df['Rolling_Alpha'] = pd.Series(alpha).fillna(0.45).rolling(window=3, min_periods=1).mean()
        return df.dropna(subset=['Rolling_Alpha'])
    except Exception as e:
        st.error(f"API ERROR: Failed to fetch live data. {e}")
        return None

@st.cache_data(ttl=120)
def fetch_systemic_health():
    assets = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD']
    health_data = []
    exchange = ccxt.kraken({'enableRateLimit': True})
    for sym in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1m', limit=60)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            noise = get_noise_filter(sym)
            df['log_L'] = np.log(df['Volume'] + 1e-9)
            spread = df['High'] - df['Low']
            df['log_T'] = np.log(np.where(spread < noise, noise, spread))
            var = df['log_L'].var()
            cov = df['log_L'].cov(df['log_T'])
            alpha = cov / (var + 1e-9) if var > 1e-7 else 0.45
            health_data.append({"asset": sym.split('/')[0], "alpha": alpha, "price": df['Close'].iloc[-1]})
        except:
            health_data.append({"asset": sym.split('/')[0], "alpha": None, "price": None})
    return health_data

# ==========================================
# 3. UI HELPERS
# ==========================================
def create_gauge_chart(alpha_value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = alpha_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "COHERENCE EXPONENT (α)", 'font': {'size': 14, 'color': '#A0AEC0'}},
        number = {'font': {'color': '#FFFFFF'}},
        gauge = {
            'axis': {'range': [None, 3.0], 'tickwidth': 1, 'tickcolor': "#2B323F"},
            'bar': {'color': "#FFFFFF", 'thickness': 0.1},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 0.79], 'color': "rgba(0, 230, 118, 0.15)"},
                {'range': [0.80, 1.19], 'color': "rgba(41, 121, 255, 0.15)"},
                {'range': [1.20, 1.99], 'color': "rgba(255, 234, 0, 0.15)"},
                {'range': [2.00, 3.0], 'color': "rgba(255, 23, 68, 0.25)"}
            ],
            'threshold': {'line': {'color': "#FF1744", 'width': 3}, 'value': 2.0}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def apply_premium_layout(fig, chart_height=750):
    fig.update_layout(
        template="plotly_dark", height=chart_height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        margin=dict(t=80, b=50, l=20, r=20), font=dict(family="Inter, sans-serif", color="#FFFFFF")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E232B')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E232B')
    return fig

# ==========================================
# 4. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.markdown("## RTM ECONOMIC MONITOR")
st.sidebar.markdown("---")

# Debugging tool in sidebar to verify files exist on GitHub/Server
if st.sidebar.checkbox("Show Debug File Tracker"):
    st.sidebar.write("### 🔎 Files found in Root:")
    st.sidebar.write(os.listdir("."))

menu = st.sidebar.radio("ANALYSIS MODULES", ("MICROSTRUCTURE (LIVE)", "MACRO EARLY WARNING", "FORENSIC LABORATORY", "MARKET PHYSICS"))

# ==========================================
# 5. MODULES
# ==========================================
if menu == "MICROSTRUCTURE (LIVE)":
    st.markdown("## LIVE MICROSTRUCTURE RADAR")
    asset = st.selectbox("SELECT ASSET", ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"])
    
    if st.button("REFRESH LIVE DATA"):
        st.cache_data.clear()
        st.rerun()

    live_df = fetch_live_rtm_data(asset)
    if live_df is not None:
        current_alpha = live_df['Rolling_Alpha'].iloc[-1]
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.plotly_chart(create_gauge_chart(current_alpha), use_container_width=True)
            st.metric("CURRENT PRICE", f"${live_df['Close'].iloc[-1]:,.2f}")
            
            if current_alpha < 0.8: status = ("LAMINAR", "#00E676")
            elif current_alpha < 1.2: status = ("TURBULENT", "#2979FF")
            elif current_alpha < 2.0: status = ("VISCOUS", "#FFEA00")
            else: status = ("BIFURCATION", "#FF1744")
            
            st.markdown(f"""<div style="border-left: 4px solid {status[1]}; background-color: #151923; padding: 15px;">
                <span style="color: {status[1]}; font-weight: 600;">STATUS: {status[0]}</span><br>
                <span style="color: #A0AEC0; font-size: 0.9em;">Systemic health is {status[0].lower()}.</span></div>""", unsafe_allow_html=True)
                
        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Close'], name="PRICE", line=dict(color='#00E5FF')), secondary_y=False)
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Rolling_Alpha'], name="ALPHA", line=dict(color='#FF0000')), secondary_y=True)
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", secondary_y=True)
            st.plotly_chart(apply_premium_layout(fig), use_container_width=True)

elif menu == "MACRO EARLY WARNING":
    st.markdown("## MACRO RADAR (EARLY WARNING)")
    if os.path.exists("crash_alpha_analysis.csv"):
        macro_df = pd.read_csv("crash_alpha_analysis.csv")
        fig = px.bar(macro_df, x='Event', y='Lead_Time_Hours', color='Drop_Pct', title="Warning Time before Systemic Failure (Hours)")
        st.plotly_chart(apply_premium_layout(fig, 500), use_container_width=True)
    else:
        st.error("Missing 'crash_alpha_analysis.csv' in your repository.")

elif menu == "FORENSIC LABORATORY":
    st.markdown("## RTM FORENSIC LABORATORY")
    event_dict = {
        "NOVEMBER 2022 (FTX COLLAPSE)": "BTCUSDT-1m-2022-11.csv",
        "MARCH 2020 (BLACK THURSDAY)": "BTCUSDT-1m-2020-03.csv",
        "MAY 2021 (CHINA BAN)": "BTCUSDT-1m-2021-05.csv",
        "SEPTEMBER 2023 (CONTROL GROUP)": "BTCUSDT-1m-2023-09.csv"
    }
    event_sel = st.selectbox("SELECT HISTORICAL EVENT:", list(event_dict.keys()))
    df_f = load_and_process_data(event_dict[event_sel])
    
    if df_f is not None:
        fig_f = make_subplots(specs=[[{"secondary_y": True}]])
        fig_f.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Close'], name="PRICE", line=dict(color='#00E5FF')), secondary_y=False)
        fig_f.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Rolling_Alpha'], name="ALPHA", line=dict(color='red')), secondary_y=True)
        st.plotly_chart(apply_premium_layout(fig_f), use_container_width=True)

elif menu == "MARKET PHYSICS":
    st.markdown("## UNIVERSAL MARKET LAWS")
    st.write("Visualizing Fat Tails and Recovery Scaling based on the RTM Physics Engine.")
    # (Existing probability calculator code remains here)
    sigma = st.slider("SIGMA (STD DEVIATIONS)", 2, 10, 5)
    r_prob = sigma ** -3
    st.success(f"RTM PHYSICS PROBABILITY: 1 IN {int(1/r_prob):,} DAYS")

st.markdown("""<div class="rtm-footer">Powered by RTM-Atmo Technology | Proof of Concept</div>""", unsafe_allow_html=True)
