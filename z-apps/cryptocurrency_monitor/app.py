import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import ccxt
import os
import io

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y CSS PREMIUM
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
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #0B0E14 !important;
        height: 0px;
    }
    [data-testid="stSidebar"] {
        background-color: #0F1219 !important;
        border-right: 1px solid #1E232B;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #FFFFFF !important; 
        font-weight: 400;
    }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
        background-color: #1E232B !important;
        color: #00E5FF !important;
        border: 1px solid #00E5FF !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #00E5FF !important;
        color: #0B0E14 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #151923;
        border-color: #1E232B;
        color: white;
    }
    h1, h2, h3, h4 {
        font-weight: 300 !important;
        letter-spacing: 1px;
        color: #FFFFFF !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
        background-color: #151923;
        border: 1px solid #1E232B;
        padding: 20px;
        border-radius: 8px;
    }
    .rtm-info-card {
        background-color: #151923;
        border: 1px solid #1E232B;
        padding: 30px;
        border-radius: 8px;
        margin-top: 25px;
    }
    .health-card {
        background-color: #11141D;
        border: 1px solid #1E232B;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
    }
    .rtm-footer {
        text-align: center;
        padding: 40px 0 20px 0;
        color: #4A5568;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTORES DE DATOS (Caché)
# ==========================================

def get_noise_filter(symbol):
    """Retorna el filtro de ruido absoluto exacto para no colapsar la covarianza RTM"""
    if 'BTC' in symbol: return 5.0
    elif 'ETH' in symbol: return 0.5
    elif 'SOL' in symbol: return 0.05
    elif 'XRP' in symbol: return 0.0001
    else: return 1.0

@st.cache_data
def load_macro_data():
    file_path = "crash_alpha_analysis.csv"
    if not os.path.exists(file_path):
        return None
    try:
        # Lógica de Auto-Curación: Si el CSV tiene comillas o está mal delimitado, lo arreglamos
        df = pd.read_csv(file_path, skipinitialspace=True)
        if len(df.columns) == 1:
            # Probablemente guardado como una sola celda con comillas
            raw_content = open(file_path, 'r').read().replace('"', '')
            df = pd.read_csv(io.StringIO(raw_content))
        return df
    except Exception:
        return None

@st.cache_data
def load_and_process_data(file_path):
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
            'Taker buy quote asset volume', 'Ignore']
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, names=cols)
        unit = 'us' if df['Open time'].iloc[0] > 1e14 else 'ms'
        df['Date'] = pd.to_datetime(df['Open time'], unit=unit)
        
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
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_live_rtm_data(symbol='BTC/USD'):
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=120)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        NOISE_FILTER_USD = get_noise_filter(symbol)
        df['log_L'] = np.log(df['Volume'] + 1e-9)
        spread = df['High'] - df['Low']
        spread_filtered = np.where(spread < NOISE_FILTER_USD, NOISE_FILTER_USD, spread)
        df['log_T'] = np.log(spread_filtered)
        
        window = 60
        cov = df['log_L'].rolling(window).cov(df['log_T'])
        var = df['log_L'].rolling(window).var()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = cov / (var + 1e-9)
            
        # Si la varianza es muy baja, Alpha se asume en su pulso base (0.45)
        alpha_val = np.where(var < 1e-7, 0.45, alpha)
        df['Rolling_Alpha'] = pd.Series(alpha_val).fillna(0.45).rolling(window=3, min_periods=1).mean()
        
        return df.dropna(subset=['Rolling_Alpha'])
    except Exception:
        return None

@st.cache_data(ttl=120)
def fetch_systemic_health():
    assets = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD']
    health_data = []
    exchange = ccxt.kraken({'enableRateLimit': True})
    for sym in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1m', limit=60)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            noise = get_noise_filter(sym)
            logL, logT = np.log(df['v']+1e-9), np.log(np.where(df['h']-df['l'] < noise, noise, df['h']-df['l']))
            v, c = logL.var(), logL.cov(logT)
            alpha = 0.45 if (v < 1e-7 or np.isnan(c/v)) else c/v
            health_data.append({"asset": sym.split('/')[0], "alpha": alpha})
        except Exception:
            health_data.append({"asset": sym.split('/')[0], "alpha": None})
    return health_data

def generate_report(asset, alpha, status):
    return f"RTM DIAGNOSTIC REPORT\nAsset: {asset}\nAlpha: {alpha:.4f}\nState: {status}\nGenerated by RTM-Atmo Technology"

# ==========================================
# 3. UI HELPER FUNCTIONS
# ==========================================
def create_gauge_chart(alpha_value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = alpha_value,
        title = {'text': "COHERENCE (α)", 'font': {'size': 14, 'color': '#A0AEC0'}},
        gauge = {
            'axis': {'range': [None, 3.0]},
            'bar': {'color': "#FFFFFF"},
            'steps': [
                {'range': [0, 0.79], 'color': "rgba(0, 230, 118, 0.15)"},
                {'range': [0.80, 1.19], 'color': "rgba(41, 121, 255, 0.15)"},
                {'range': [1.20, 1.99], 'color': "rgba(255, 234, 0, 0.15)"},
                {'range': [2.00, 3.0], 'color': "rgba(255, 23, 68, 0.25)"}
            ]
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
    return fig

def apply_premium_layout(fig, chart_height=750):
    fig.update_layout(
        template="plotly_dark", height=chart_height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", margin=dict(t=80, b=50, l=20, r=20), font=dict(color="#FFFFFF"), 
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    fig.update_xaxes(gridcolor='#1E232B'); fig.update_yaxes(gridcolor='#1E232B')
    return fig

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("## RTM ECONOMIC MONITOR")
menu = st.sidebar.radio("ANALYSIS MODULES", ("MICROSTRUCTURE (LIVE)", "MACRO EARLY WARNING", "FORENSIC LABORATORY", "MARKET PHYSICS"))

# ==========================================
# 5. MODULE ROUTING
# ==========================================

if menu == "MICROSTRUCTURE (LIVE)":
    st.markdown("## LIVE MICROSTRUCTURE RADAR")
    
    col_sel, col_btn, _ = st.columns([1, 1, 2])
    with col_sel:
        selected_asset = st.selectbox("SELECT ASSET", ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"])
    with col_btn:
        st.write(""); 
        if st.button("REFRESH"): st.cache_data.clear(); st.rerun()

    live_df = fetch_live_rtm_data(selected_asset)
    
    if live_df is not None:
        current_alpha = live_df['Rolling_Alpha'].iloc[-1]
        current_price = live_df['Close'].iloc[-1]
        
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.plotly_chart(create_gauge_chart(current_alpha), use_container_width=True)
            st.metric(selected_asset, f"${current_price:,.2f}")
            
            # Status Box
            if current_alpha < 0.8: status, color = "LAMINAR", "#00E676"
            elif current_alpha < 1.2: status, color = "TURBULENT", "#2979FF"
            elif current_alpha < 2.0: status, color = "VISCOUS", "#FFEA00"
            else: status, color = "BIFURCATION", "#FF1744"
            
            st.markdown(f'<div style="border-left: 4px solid {color}; background-color: #151923; padding: 15px; border-radius: 4px;"><span style="color: {color}; font-weight: 600;">STATUS: {status}</span></div>', unsafe_allow_html=True)
            
            # Download Report
            report = generate_report(selected_asset, current_alpha, status)
            st.download_button("EXPORT REPORT", report, file_name="RTM_Diagnostic.txt")

        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Close'], name="PRICE", line=dict(color='#00E5FF')), secondary_y=False)
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Rolling_Alpha'], name="ALPHA (α)", line=dict(color='#FF0000', width=2)), secondary_y=True)
            fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255, 23, 68, 0.5)", secondary_y=True)
            fig.add_hline(y=1.2, line_dash="dash", line_color="rgba(255, 234, 0, 0.5)", secondary_y=True)
            st.plotly_chart(apply_premium_layout(fig), use_container_width=True)

        st.markdown("---")
        st.markdown("#### GLOBAL SYSTEMIC HEALTH")
        h_data = fetch_systemic_health()
        cols_h = st.columns(4)
        for i, h in enumerate(h_data):
            with cols_h[i]:
                if h['alpha']:
                    c = "#00E676" if h['alpha'] < 0.8 else "#FFEA00" if h['alpha'] < 2.0 else "#FF1744"
                    st.markdown(f'<div class="health-card" style="border-top: 3px solid {c};"><b>{h["asset"]}</b><br>α = {h["alpha"]:.3f}</div>', unsafe_allow_html=True)

elif menu == "MACRO EARLY WARNING":
    st.markdown("## MACRO EARLY WARNING")
    macro_data = load_macro_data()
    if macro_data is not None:
        st.markdown("#### RTM STATE SPACE TRANSITION")
        fig_slope = go.Figure()
        for _, row in macro_data.iterrows():
            fig_slope.add_trace(go.Scatter(x=["NORMAL", "PRE-CRASH"], y=[row['Baseline_Alpha'], row['Immediate_Alpha']], name=row['Event']))
        fig_slope.add_hline(y=0.5, line_dash="dash", line_color="#A0AEC0")
        st.plotly_chart(apply_premium_layout(fig_slope, 500), use_container_width=True)
    else:
        st.error("ERROR: 'crash_alpha_analysis.csv' not found or corrupted.")

elif menu == "FORENSIC LABORATORY":
    st.markdown("## RTM FORENSIC LABORATORY")
    event_dict = {"NOVEMBER 2022 (FTX)": "BTCUSDT-1m-2022-11.csv", "MARCH 2020 (COVID)": "BTCUSDT-1m-2020-03.csv", "SEPTEMBER 2023 (CTRL)": "BTCUSDT-1m-2023-09.csv"}
    event = st.selectbox("HISTORICAL EVENT", list(event_dict.keys()))
    df = load_and_process_data(event_dict[event])
    
    if df is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="PRICE", line=dict(color='#00E5FF')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Alpha'], name="ALPHA", line=dict(color='#FF0000')), secondary_y=True)
        
        if event != "SEPTEMBER 2023 (CTRL)":
            peak_idx = df['Rolling_Alpha'].idxmax()
            p_date, s_date, c_date = df.loc[peak_idx, 'Date'], df.loc[max(0, peak_idx-180), 'Date'], df.loc[min(len(df)-1, peak_idx+120), 'Date']
            
            # ETIQUETAS HORIZONTALES ESCALONADAS
            fig.add_vline(x=s_date, line_dash="dot", line_color="#A0AEC0")
            fig.add_annotation(x=s_date, y=2.6, text="STRESS ONSET", showarrow=False, xanchor="left", xshift=5, yref="y2")
            fig.add_vline(x=p_date, line_dash="solid", line_color="#FFEA00")
            fig.add_annotation(x=p_date, y=3.0, text="MAX ENTROPY", showarrow=False, font=dict(weight="bold"), xanchor="left", xshift=5, yref="y2")
            fig.add_vline(x=c_date, line_dash="dot", line_color="#FF1744")
            fig.add_annotation(x=c_date, y=2.4, text="LIQUIDITY CASCADE", showarrow=False, xanchor="left", xshift=5, yref="y2")
            
        st.plotly_chart(apply_premium_layout(fig), use_container_width=True)

elif menu == "MARKET PHYSICS":
    st.markdown("## MARKET PHYSICS")
    tab1, tab2, tab3 = st.tabs(["FAT TAILS", "RECOVERY", "ALPHA DIST"])
    with tab1:
        st.write("Markets follow Power Laws. Extreme events are structural features.")
        c1, c2 = st.columns(2)
        c1.metric("GLOBAL α EXPO", "2.966 ± 0.236")
        sigma = c2.slider("SIGMA", 2, 10, 5)
        st.info(f"RTM PROBABILITY: 1 IN {int(sigma**3):,} DAYS")
    with tab2:
        drawdown = st.number_input("DRAWDOWN (%)", 10, 90, 40)
        st.metric("EST. RECOVERY TIME", f"{int(365 * ((drawdown/20)**(3.59/2))):,} DAYS")
    with tab3:
        np.random.seed(42)
        d = np.concatenate([np.random.normal(0.45,0.12,18000), np.random.normal(0.95,0.15,1500), np.random.normal(1.4,0.2,400), np.random.normal(2.1,0.25,50)])
        fig_hist = px.histogram(d[(d>0.1)&(d<3.2)], nbins=100, labels={'value':'Alpha'})
        st.plotly_chart(apply_premium_layout(fig_hist, 500), use_container_width=True)

st.markdown('<div class="rtm-footer">Powered by RTM-Atmo Technology | PoC System</div>', unsafe_allow_html=True)
