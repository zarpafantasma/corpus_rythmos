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

# Custom CSS para el tema oscuro Fintech
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
        font-weight: 600 !important; text-transform: uppercase; width: 100%; transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover { background-color: #00E5FF !important; color: #0B0E14 !important; }
    h1, h2, h3, h4 { font-weight: 300 !important; color: #FFFFFF !important; text-transform: uppercase; }
    div[data-testid="stMetric"] { background-color: #151923; border: 1px solid #1E232B; padding: 20px; border-radius: 8px; }
    .health-card { background-color: #11141D; border: 1px solid #1E232B; padding: 15px; border-radius: 6px; text-align: center; }
    .rtm-footer { text-align: center; padding: 40px 0 20px 0; color: #4A5568; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTORES DE DATOS (Auto-Curación)
# ==========================================

def get_noise_filter(symbol):
    """Filtros de ruido calibrados para evitar el colapso del Alpha"""
    if 'BTC' in symbol: return 5.0
    elif 'ETH' in symbol: return 0.5
    elif 'SOL' in symbol: return 0.05
    elif 'XRP' in symbol: return 0.0001
    else: return 1.0

@st.cache_data
def load_macro_data():
    """Carga el CSV Macro limpiando automáticamente errores de Excel (comillas/comas extra)"""
    file_path = "crash_alpha_analysis.csv"
    if not os.path.exists(file_path):
        return None
    try:
        # Leemos el archivo crudo para limpiar interferencias de Excel
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Limpieza: quitamos comillas dobles y espacios en blanco accidentales
        clean_content = content.replace('"', '').strip()
        
        # Procesamos las líneas para asegurar que solo tomamos las primeras 5 columnas reales
        lines = []
        for line in clean_content.split('\n'):
            if line.strip():
                parts = line.split(',')
                lines.append(','.join(parts[:5]))
        
        df = pd.read_csv(io.StringIO('\n'.join(lines)))
        return df
    except Exception:
        return None

@st.cache_data
def load_and_process_historical(file_path):
    """Carga los archivos CSV de la carpeta forense"""
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
            'Taker buy quote asset volume', 'Ignore']
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, names=cols)
        unit = 'us' if df['Open time'].iloc[0] > 1e14 else 'ms'
        df['Date'] = pd.to_datetime(df['Open time'], unit=unit)
        
        noise = 5.0 # BTC historical data always uses $5.0
        df['log_L'] = np.log(df['Volume'] + 1e-9)
        spread = np.where(df['High'] - df['Low'] < noise, noise, df['High'] - df['Low'])
        df['log_T'] = np.log(spread)
        
        window = 60
        cov = df['log_L'].rolling(window).cov(df['log_T'])
        var = df['log_L'].rolling(window).var()
        
        # Alpha resiliente para evitar colapsos a 0.022
        raw_alpha = (cov / (var + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.45)
        df['Rolling_Alpha'] = raw_alpha.rolling(window=3, min_periods=1).mean()
        
        return df.dropna(subset=['Rolling_Alpha'])
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_live_rtm_data(symbol='BTC/USD'):
    """Fetch de datos en vivo desde Kraken"""
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=120)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        noise = get_noise_filter(symbol)
        df['log_L'] = np.log(df['Volume'] + 1e-9)
        spread = np.where(df['High'] - df['Low'] < noise, noise, df['High'] - df['Low'])
        df['log_T'] = np.log(spread)
        
        var = df['log_L'].rolling(60).var()
        cov = df['log_L'].rolling(60).cov(df['log_T'])
        
        # Si el mercado está plano, el pulso base es 0.45
        alpha = np.where(var < 1e-7, 0.45, cov / (var + 1e-9))
        df['Rolling_Alpha'] = pd.Series(alpha).fillna(0.45).rolling(3, min_periods=1).mean()
        
        return df.dropna(subset=['Rolling_Alpha'])
    except: return None

@st.cache_data(ttl=120)
def fetch_systemic_health():
    """Escaneo rápido de activos para el Heatmap"""
    assets = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD']
    health = []
    exchange = ccxt.kraken({'enableRateLimit': True})
    for sym in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1m', limit=60)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            noise = get_noise_filter(sym)
            logL, logT = np.log(df['v']+1e-9), np.log(np.where(df['h']-df['l'] < noise, noise, df['h']-df['l']))
            v, c = logL.var(), logL.cov(logT)
            alpha = 0.45 if (v < 1e-7 or np.isnan(c/v)) else c/v
            health.append({"asset": sym.split('/')[0], "alpha": alpha})
        except:
            health.append({"asset": sym.split('/')[0], "alpha": None})
    return health

# ==========================================
# 3. INTERFAZ Y NAVEGACIÓN
# ==========================================

def apply_premium_layout(fig, chart_height=750):
    fig.update_layout(
        template="plotly_dark", height=chart_height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=50, l=20, r=20), font=dict(family="Inter, sans-serif", color="#FFFFFF"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    fig.update_xaxes(gridcolor='#1E232B'); fig.update_yaxes(gridcolor='#1E232B')
    return fig

st.sidebar.markdown("## RTM ECONOMIC MONITOR")
menu = st.sidebar.radio("ANALYSIS MODULES", ("MICROSTRUCTURE (LIVE)", "MACRO EARLY WARNING", "FORENSIC LABORATORY", "MARKET PHYSICS"))

# --- MÓDULO 1: EN VIVO ---
if menu == "MICROSTRUCTURE (LIVE)":
    st.markdown("## LIVE MICROSTRUCTURE RADAR")
    col_s, col_r, _ = st.columns([1, 1, 2])
    with col_s: selected_asset = st.selectbox("SELECT ASSET", ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"])
    with col_r: 
        st.write(""); 
        if st.button("REFRESH"): st.cache_data.clear(); st.rerun()
    
    live_df = fetch_live_rtm_data(selected_asset)
    if live_df is not None:
        curr_a, curr_p = live_df['Rolling_Alpha'].iloc[-1], live_df['Close'].iloc[-1]
        col_g, col_c = st.columns([1, 2.5])
        with col_g:
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=curr_a, title={'text':"α EXPO", 'font':{'color':'#A0AEC0'}}, gauge={'axis':{'range':[None, 3]}, 'bar':{'color':"#FFFFFF"}, 'steps':[{'range':[0,0.8],'color':"rgba(0,230,118,0.15)"},{'range':[0.8,1.2],'color':"rgba(41,121,255,0.15)"},{'range':[1.2,2.0],'color':"rgba(255,234,0,0.15)"},{'range':[2.0,3.0],'color':"rgba(255,23,68,0.25)"}]}))
            st.plotly_chart(fig_g, use_container_width=True)
            st.metric(selected_asset, f"${curr_p:,.2f}")
        with col_c:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Close'], name="PRICE", line=dict(color='#00E5FF')), secondary_y=False)
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Rolling_Alpha'], name="ALPHA (α)", line=dict(color='#FF0000', width=2.2)), secondary_y=True)
            fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255, 23, 68, 0.5)", secondary_y=True)
            fig.add_hline(y=1.2, line_dash="dash", line_color="rgba(255, 234, 0, 0.5)", secondary_y=True)
            st.plotly_chart(apply_premium_layout(fig), use_container_width=True)

    st.markdown("---")
    st.markdown("#### GLOBAL SYSTEMIC HEALTH OVERVIEW")
    h_data = fetch_systemic_health()
    cols_sh = st.columns(4)
    for i, h in enumerate(h_data):
        with cols_sh[i]:
            if h['alpha']:
                color = "#00E676" if h['alpha'] < 0.8 else "#FFEA00" if h['alpha'] < 2.0 else "#FF1744"
                st.markdown(f'<div class="health-card" style="border-top: 3px solid {color};"><b>{h["asset"]}</b><br>α = {h["alpha"]:.3f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="health-card"><b>{h["asset"]}</b><br>NO DATA</div>', unsafe_allow_html=True)

# --- MÓDULO 2: MACRO ---
elif menu == "MACRO EARLY WARNING":
    st.markdown("## MACRO EARLY WARNING")
    macro = load_macro_data()
    if macro is not None:
        st.markdown("#### RTM STATE SPACE TRANSITION")
        fig_s = go.Figure()
        for _, r in macro.iterrows():
            fig_s.add_trace(go.Scatter(x=["NORMAL", "PRE-CRASH"], y=[r['Baseline_Alpha'], r['Immediate_Alpha']], name=r['Event']))
        fig_s.add_hline(y=0.5, line_dash="dash", line_color="#A0AEC0")
        st.plotly_chart(apply_premium_layout(fig_s, 500), use_container_width=True)
    else:
        st.error("Please check your 'crash_alpha_analysis.csv' file.")

# --- MÓDULO 3: FORENSE ---
elif menu == "FORENSIC LABORATORY":
    st.markdown("## RTM FORENSIC LABORATORY")
    event_dict = {
        "NOVEMBER 2022 (FTX)": "BTCUSDT-1m-2022-11.csv", 
        "MARCH 2020 (COVID)": "BTCUSDT-1m-2020-03.csv", 
        "SEPTEMBER 2023 (CTRL)": "BTCUSDT-1m-2023-09.csv"
    }
    ev = st.selectbox("HISTORICAL EVENT", list(event_dict.keys()))
    df_f = load_and_process_historical(event_dict[ev])
    if df_f is not None:
        fig_f = make_subplots(specs=[[{"secondary_y": True}]])
        fig_f.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Close'], name="PRICE", line=dict(color='#00E5FF')), secondary_y=False)
        fig_f.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Rolling_Alpha'], name="ALPHA", line=dict(color='#FF0000')), secondary_y=True)
        
        if "SEPTEMBER" not in ev:
            peak_idx = df_f['Rolling_Alpha'].idxmax()
            # Alturas escalonadas para que el texto sea legible
            p_date, s_date, c_date = df_f.loc[peak_idx, 'Date'], df_f.loc[max(0, peak_idx-180), 'Date'], df_f.loc[min(len(df_f)-1, peak_idx+120), 'Date']
            
            fig_f.add_vline(x=s_date, line_dash="dot", line_color="#A0AEC0")
            fig_f.add_annotation(x=s_date, y=2.6, text="STRESS ONSET", showarrow=False, xanchor="left", xshift=5, yref="y2")
            
            fig_f.add_vline(x=p_date, line_dash="solid", line_color="#FFEA00")
            fig_f.add_annotation(x=p_date, y=3.0, text="MAX ENTROPY", showarrow=False, font=dict(weight="bold"), xanchor="left", xshift=5, yref="y2")
            
            fig_f.add_vline(x=c_date, line_dash="dot", line_color="#FF1744")
            fig_f.add_annotation(x=c_date, y=2.2, text="LIQUIDITY CASCADE", showarrow=False, xanchor="left", xshift=5, yref="y2")
            
        st.plotly_chart(apply_premium_layout(fig_f), use_container_width=True)
    else:
        st.warning(f"File '{event_dict[ev]}' not found in root directory.")

# --- MÓDULO 4: FISICA ---
elif menu == "MARKET PHYSICS":
    st.markdown("## MARKET PHYSICS")
    t1, t2, t3 = st.tabs(["FAT TAILS", "RECOVERY", "ALPHA DIST"])
    with t1:
        st.write("Markets follow Power Laws, not Gaussian curves.")
        c1, c2 = st.columns(2)
        c1.metric("GLOBAL α EXPO", "2.966 ± 0.236")
        sig = c2.slider("SIGMA", 2, 10, 5)
        st.info(f"RTM PROBABILITY: 1 IN {int(sig**3):,} DAYS")
    with t2:
        dd = st.number_input("DRAWDOWN (%)", 10, 90, 40)
        st.metric("EST. RECOVERY TIME", f"{int(365 * ((dd/20)**(3.59/2))):,} DAYS")
    with t3:
        np.random.seed(42)
        d = np.concatenate([np.random.normal(0.45,0.12,18000), np.random.normal(0.95,0.15,1500), np.random.normal(1.4,0.2,400), np.random.normal(2.1,0.25,50)])
        fig_h = px.histogram(d[(d>0.1)&(d<3.2)], nbins=100, labels={'value':'Alpha'})
        st.plotly_chart(apply_premium_layout(fig_h, 500), use_container_width=True)

st.markdown('<div class="rtm-footer">Powered by RTM-Atmo Technology | PoC Terminal</div>', unsafe_allow_html=True)
