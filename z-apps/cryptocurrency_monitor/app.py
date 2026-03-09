import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import ccxt
import io
import os

# Dynamically get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. PAGE CONFIGURATION AND PREMIUM CSS
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
    /* Main Backgrounds */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Remove white header strip */
    header[data-testid="stHeader"] {
        background-color: #0B0E14 !important;
        height: 0px;
    }
    
    /* Sidebar styling: Pure white text for high contrast */
    [data-testid="stSidebar"] {
        background-color: #0F1219 !important;
        border-right: 1px solid #1E232B;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important; 
        font-weight: 400;
    }
    
    /* Button Styling (Refresh & Download) */
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
        background-color: #1E232B !important;
        color: #00E5FF !important;
        border: 1px solid #00E5FF !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover {
        background-color: #00E5FF !important;
        color: #0B0E14 !important;
    }
    
    /* Selectbox Styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #151923;
        border-color: #1E232B;
        color: white;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-weight: 300 !important;
        letter-spacing: 1px;
        color: #FFFFFF !important;
        text-transform: uppercase;
    }
    
    hr {
        border-color: #1E232B;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #151923;
        border: 1px solid #1E232B;
        padding: 20px;
        border-radius: 8px;
    }
    
    /* Info Box Styling */
    .rtm-info-card {
        background-color: #151923;
        border: 1px solid #1E232B;
        padding: 30px;
        border-radius: 8px;
        margin-top: 25px;
        line-height: 1.6;
    }
    
    /* Systemic Health Cards */
    .health-card {
        background-color: #11141D;
        border: 1px solid #1E232B;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
    }
    
    /* Footer Styling */
    .rtm-footer {
        text-align: center;
        padding: 40px 0 20px 0;
        color: #4A5568;
        font-size: 0.85em;
        letter-spacing: 0.5px;
    }
    .rtm-footer a {
        color: #4A5568;
        text-decoration: none;
    }
    .rtm-footer a:hover {
        color: #00E5FF;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINES (Caching)
# ==========================================
def get_noise_filter(symbol):
    """Returns the exact absolute noise filter to prevent RTM covariance collapse"""
    if 'BTC' in symbol: return 5.0
    elif 'ETH' in symbol: return 0.5
    elif 'SOL' in symbol: return 0.05
    elif 'XRP' in symbol: return 0.0001
    else: return 1.0

@st.cache_data
def load_and_process_data(file_path):
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
            'Taker buy quote asset volume', 'Ignore']
    try:
        df = pd.read_csv(file_path, names=cols)
        first_ts = df['Open time'].iloc[0]
        unit = 'us' if first_ts > 1e14 else 'ms'
        df['Date'] = pd.to_datetime(df['Open time'], unit=unit)
        
        # Official RTM Noise Filter ($5.00 USD)
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
            
        raw_alpha = pd.Series(raw_alpha).replace([np.inf, -np.inf], np.nan)
        df['Rolling_Alpha'] = raw_alpha.rolling(window=3, min_periods=1).mean()
        
        return df.dropna(subset=['Rolling_Alpha'])
    except Exception as e:
        # Show exact error in UI if file reading fails
        st.error(f"Error processing file '{file_path}': {str(e)}")
        return None

@st.cache_data(ttl=60)
def fetch_live_rtm_data(symbol='BTC/USD'):
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=120)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Use strict noise filter per asset
        NOISE_FILTER_USD = get_noise_filter(symbol)
        
        df['log_L'] = np.log(df['Volume'] + 1e-9)
        spread = df['High'] - df['Low']
        spread_filtered = np.where(spread < NOISE_FILTER_USD, NOISE_FILTER_USD, spread)
        df['log_T'] = np.log(spread_filtered)
        
        window = 60
        cov = df['log_L'].rolling(window).cov(df['log_T'])
        var = df['log_L'].rolling(window).var()
        
        # INTACT MATH (No alterations causing 0.024)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_alpha = cov / var
            
        raw_alpha = pd.Series(raw_alpha).replace([np.inf, -np.inf], np.nan)
        df['Rolling_Alpha'] = raw_alpha.rolling(window=3, min_periods=1).mean()
        
        return df.dropna(subset=['Rolling_Alpha'])
    except Exception as e:
        st.error(f"Live API Fetch Error: {str(e)}")
        return None

@st.cache_data(ttl=120)
def fetch_systemic_health():
    assets = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD']
    health_data = []
    exchange = ccxt.kraken({'enableRateLimit': True})
    for sym in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1m', limit=120)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            noise_filter = get_noise_filter(sym)
            df['log_L'] = np.log(df['Volume'] + 1e-9)
            spread = df['High'] - df['Low']
            df['log_T'] = np.log(np.where(spread < noise_filter, noise_filter, spread))
            var = df['log_L'].rolling(60).var()
            cov = df['log_L'].rolling(60).cov(df['log_T'])
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_alpha = pd.Series(cov / var).replace([np.inf, -np.inf], np.nan)
            
            alpha = raw_alpha.rolling(3, min_periods=1).mean().iloc[-1]
            display_name = sym.split('/')[0]
            health_data.append({"asset": display_name, "alpha": alpha, "price": df['Close'].iloc[-1]})
        except Exception:
            display_name = sym.split('/')[0]
            health_data.append({"asset": display_name, "alpha": None, "price": None})
    return health_data

@st.cache_data
def load_macro_data():
    try:
        # Safely join the path to the CSV file
        file_path = os.path.join(BASE_DIR, "crash_alpha_analysis.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading macro data: {str(e)}")
        return None

def generate_report(asset, price, alpha, status, timestamp):
    report = f"""====================================================
RTM STRUCTURAL DIAGNOSTIC REPORT
====================================================
Generated: {timestamp}
Asset Monitored: {asset}
Current Price: ${price:,.2f}
----------------------------------------------------
METRICS
----------------------------------------------------
Coherence Exponent (Alpha): {alpha:.4f}
Systemic State: {status}

----------------------------------------------------
ANALYSIS
----------------------------------------------------
"""
    if alpha < 0.8:
        report += "The market exhibits Laminar Flow. The network structure is healthy, efficiently processing volume without generating anomalous volatility. Liquidity is deeply integrated."
    elif alpha < 1.2:
        report += "The market exhibits Turbulent Flow. Energy dissipation is increasing, but the order book topology remains intact. Monitor for further viscosity."
    elif alpha < 2.0:
        report += "WARNING: Viscous State Detected. The market is experiencing extreme internal friction. Volume is failing to move price efficiently. High probability of systemic stress and liquidity withdrawal."
    else:
        report += "CRITICAL: BIFURCATION. The structural topology of the network has fractured. Total loss of market memory. Terminal capitulation or extreme anomaly in progress."
        
    report += "\n\nPowered by RTM-Atmo Technology."
    return report

# ==========================================
# 3. UI HELPER FUNCTIONS
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
            'borderwidth': 0,
            'steps': [
                {'range': [0, 0.79], 'color': "rgba(0, 230, 118, 0.15)", 'name': 'LAMINAR'},
                {'range': [0.80, 1.19], 'color': "rgba(41, 121, 255, 0.15)", 'name': 'TURBULENT'},
                {'range': [1.20, 1.99], 'color': "rgba(255, 234, 0, 0.15)", 'name': 'VISCOUS'},
                {'range': [2.00, 3.0], 'color': "rgba(255, 23, 68, 0.25)", 'name': 'BIFURCATION'}
            ],
            'threshold': {'line': {'color': "#FF1744", 'width': 3}, 'thickness': 0.75, 'value': 2.0}
        }
    ))
    fig.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )
    return fig

def apply_premium_layout(fig, chart_height=750):
    """Applies institutional style with white legend text and stretched height"""
    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(t=80, b=50, l=20, r=20),
        font=dict(family="Inter, sans-serif", color="#FFFFFF"), 
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(size=12, color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E232B', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E232B', zeroline=False)
    return fig

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("## RTM ECONOMIC MONITOR")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "ANALYSIS MODULES",
    ("MICROSTRUCTURE (LIVE)", "MACRO EARLY WARNING", "FORENSIC LABORATORY", "MARKET PHYSICS")
)
st.sidebar.markdown("---")

# Disclaimer Block
st.sidebar.markdown("""
<div style="color: #A0AEC0; font-size: 0.78em; line-height: 1.4; border-left: 2px solid #4A5568; padding-left: 10px; margin-bottom: 20px;">
    <b>DISCLAIMER:</b> This platform is strictly a Proof of Concept (PoC) built upon the Rhythmic Time Measurement (RTM) theoretical framework. 
    It is not an infallible predictive source and does not constitute financial advice. 
    RTM-Atmo Technology and its developers assume no responsibility for financial losses or systemic risks arising from the use of this diagnostic tool.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="color: #FFFFFF; font-size: 0.85em; line-height: 1.5;">
    <b>RTM PHYSICS ENGINE</b><br>
    Financial time (T) is treated as an emergent property of market structure (L<sup>α</sup>).
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. MODULE ROUTING
# ==========================================

# ------------------------------------------
# MODULE 1: MICROSTRUCTURE (LIVE)
# ------------------------------------------
if menu == "MICROSTRUCTURE (LIVE)":
    st.markdown("## LIVE MICROSTRUCTURE RADAR")
    st.markdown("<p style='color: #A0AEC0;'>Real-time monitoring of multi-asset market friction via Kraken API.</p>", unsafe_allow_html=True)
    
    # MULTI-ASSET SELECTOR
    col_sel, col_btn, _ = st.columns([1, 1, 2])
    with col_sel:
        selected_asset = st.selectbox("SELECT ASSET", ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"])
    with col_btn:
        st.write("") # Spacer
        if st.button("REFRESH LIVE DATA (1M)"):
            st.cache_data.clear()
            st.rerun()

    live_df = fetch_live_rtm_data(selected_asset)
    
    if live_df is not None and not live_df.empty:
        current_alpha = live_df['Rolling_Alpha'].iloc[-1]
        current_price = live_df['Close'].iloc[-1]
        last_update = live_df['Date'].iloc[-1].strftime('%H:%M:%S UTC')
        
        status_text = ""
        
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.plotly_chart(create_gauge_chart(current_alpha), use_container_width=True)
            display_ticker = selected_asset.split('/')[0]
            st.metric(label=f"CURRENT PRICE ({display_ticker})", value=f"${current_price:,.2f}" if current_price > 1 else f"${current_price:.4f}", delta=f"UPDATED: {last_update}", delta_color="off")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if current_alpha < 0.8:
                status_text = "LAMINAR"
                st.markdown("""<div style="border-left: 4px solid #00E676; background-color: #151923; padding: 15px; border-radius: 4px;"><span style="color: #00E676; font-weight: 600; letter-spacing: 1px;">STATUS: LAMINAR</span><br><span style="color: #A0AEC0; font-size: 0.9em;">Healthy market. Superfluid state. Safe to trade.</span></div>""", unsafe_allow_html=True)
            elif current_alpha < 1.2:
                status_text = "TURBULENT"
                st.markdown("""<div style="border-left: 4px solid #2979FF; background-color: #151923; padding: 15px; border-radius: 4px;"><span style="color: #2979FF; font-weight: 600; letter-spacing: 1px;">STATUS: TURBULENT</span><br><span style="color: #A0AEC0; font-size: 0.9em;">Active conditions. Volatility increasing, structure holding.</span></div>""", unsafe_allow_html=True)
            elif current_alpha < 2.0:
                status_text = "VISCOUS"
                st.markdown("""<div style="border-left: 4px solid #FFEA00; background-color: #151923; padding: 15px; border-radius: 4px;"><span style="color: #FFEA00; font-weight: 600; letter-spacing: 1px;">STATUS: VISCOUS</span><br><span style="color: #A0AEC0; font-size: 0.9em;">Systemic stress detected. Caution advised.</span></div>""", unsafe_allow_html=True)
            else:
                status_text = "BIFURCATION"
                st.markdown("""<div style="border-left: 4px solid #FF1744; background-color: #231215; padding: 15px; border-radius: 4px;"><span style="color: #FF1744; font-weight: 600; letter-spacing: 1px;">STATUS: BIFURCATION</span><br><span style="color: #A0AEC0; font-size: 0.9em;">CRITICAL FAILURE. Structure fractured. EXIT MARKETS.</span></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # DOWNLOAD DIAGNOSTIC REPORT
            report_content = generate_report(selected_asset, current_price, current_alpha, status_text, last_update)
            st.download_button(
                label="EXPORT DIAGNOSTIC REPORT",
                data=report_content,
                file_name=f"RTM_Diagnostic_{selected_asset.replace('/','')}.txt",
                mime="text/plain"
            )
                
        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Close'], name=f"PRICE ({display_ticker})", line=dict(color='#00E5FF', width=2), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.05)'), secondary_y=False)
            
            # Alpha line updated to intense red, slightly thicker
            fig.add_trace(go.Scatter(x=live_df['Date'], y=live_df['Rolling_Alpha'], name="RTM ALPHA (α)", line=dict(color='#FF0000', width=2.2)), secondary_y=True)
            
            # Adding thresholds to legend
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(255, 23, 68, 0.6)', dash='dash'), name='FRACTURE (2.0)'), secondary_y=True)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(255, 234, 0, 0.6)', dash='dash'), name='VISCOSITY (1.2)'), secondary_y=True)
            
            # Draw actual lines (without overlapping labels)
            fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255, 23, 68, 0.5)", secondary_y=True)
            fig.add_hline(y=1.2, line_dash="dash", line_color="rgba(255, 234, 0, 0.5)", secondary_y=True)
            
            fig = apply_premium_layout(fig, chart_height=750) 
            fig.update_yaxes(title_text="", secondary_y=True, range=[-0.1, 3.1]) 
            st.plotly_chart(fig, use_container_width=True)

        # GLOBAL SYSTEMIC HEALTH OVERVIEW
        st.markdown("---")
        st.markdown("#### GLOBAL SYSTEMIC HEALTH OVERVIEW")
        health_data = fetch_systemic_health()
        cols = st.columns(4)
        for i, h_data in enumerate(health_data):
            with cols[i]:
                if h_data['alpha'] is not None and not np.isnan(h_data['alpha']):
                    color = "#00E676" if h_data['alpha'] < 0.8 else "#2979FF" if h_data['alpha'] < 1.2 else "#FFEA00" if h_data['alpha'] < 2.0 else "#FF1744"
                    state = "LAMINAR" if h_data['alpha'] < 0.8 else "TURBULENT" if h_data['alpha'] < 1.2 else "VISCOUS" if h_data['alpha'] < 2.0 else "FRACTURE"
                    st.markdown(f"""
                    <div class="health-card" style="border-top: 3px solid {color};">
                        <div style="color: #A0AEC0; font-size: 0.85em;">{h_data['asset']}</div>
                        <div style="color: #FFFFFF; font-size: 1.2em; font-weight: bold; margin: 5px 0;">α = {h_data['alpha']:.3f}</div>
                        <div style="color: {color}; font-size: 0.8em; font-weight: 600;">{state}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="health-card" style="border-top: 3px solid #1E232B;"><div style="color: #A0AEC0; font-size: 0.85em;">{h_data['asset']}</div><div style="color: #4A5568; margin: 5px 0;">NO DATA</div></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="rtm-info-card">
            <h3 style="color: #FFFFFF; margin-top: 0;">What does the RTM Economic Radar measure?</h3>
            <p style="color: #A0AEC0; margin-bottom: 20px;">
                Unlike traditional indicators that track price direction, <b>RTM (Rhythmic Time Measurement)</b> measures the <b>structural integrity</b> of the financial network. It uses fluid dynamics to analyze how information and capital flow through the order books.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                <div>
                    <b style="color: #FF1A00;">THE ALPHA EXPONENT (α)</b><br>
                    Acts as the "thermometer" of the network. It measures <b>Coherence</b>: how much energy (volume) is required to displace financial time (volatility). 
                    Low Alpha represents efficiency; high Alpha indicates structural decay.
                </div>
                <div>
                    <b style="color: #FFEA00;">VISCOSITY (VISCOUS STATE)</b><br>
                    Based on historical forensics (FTX, Black Thursday), <b>Alpha values above 1.20</b> indicate "internal friction." The market becomes heavy; volume enters massively but price moves inefficiently. 
                    This represents extreme systemic stress before a total fracture.
                </div>
            </div>
            <p style="color: #A0AEC0; margin-top: 20px; font-size: 0.9em; font-style: italic;">
                *When Alpha reaches the 2.0 threshold (Bifurcation), the financial network undergoes a topological break. This state signifies a total loss of structural memory, indicating the exchange engine can no longer dissipate incoming volumetric energy, leading to imminent terminal failure.*
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("CONNECTION ERROR: Could not fetch live data from Kraken API or data returned empty.")

# ------------------------------------------
# MODULE 2: MACRO EARLY WARNING
# ------------------------------------------
elif menu == "MACRO EARLY WARNING":
    st.markdown("## MACRO RADAR (EARLY WARNING SYSTEM)")
    st.markdown("<p style='color: #A0AEC0;'>DFA α analysis to detect structural memory loss days prior to global market collapses.</p>", unsafe_allow_html=True)
    
    macro_data = load_macro_data()
    
    if isinstance(macro_data, pd.DataFrame):
        macro_data['Lead_Time_Days'] = macro_data['Lead_Time_Hours'] / 24.0
        
        # Chart 1: State Space Transition
        st.markdown("#### RTM STATE SPACE TRANSITION")
        st.markdown("<p style='color: #A0AEC0; font-size: 0.95em;'>This mapping visualizes the physical decay of market coordination. It tracks the transition of the Coherence Exponent (α) from its institutional baseline toward the 'Random Walk' limit (0.5), signifying a total loss of structural memory prior to systemic collapses.</p>", unsafe_allow_html=True)
        fig_slope = go.Figure()
        for index, row in macro_data.iterrows():
            fig_slope.add_trace(go.Scatter(x=["NORMAL", "PRE-CRASH"], y=[row['Baseline_Alpha'], row['Immediate_Alpha']], mode='markers+lines', name=row['Event'], line=dict(width=1.5), marker=dict(size=6)))
        fig_slope.add_hline(y=0.5, line_dash="dash", line_color="#A0AEC0", annotation_text="RANDOM WALK", annotation_font_color="#FFFFFF")
        fig_slope = apply_premium_layout(fig_slope, chart_height=500)
        st.plotly_chart(fig_slope, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chart 2: Early Warning Lead Time
        st.markdown("#### EARLY WARNING LEAD TIME")
        fig_bar = px.bar(macro_data, x='Event', y='Lead_Time_Days', color='Drop_Pct', color_continuous_scale='Blues_r', labels={'Lead_Time_Days': 'DAYS OF WARNING'})
        fig_bar = apply_premium_layout(fig_bar, chart_height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown("""<div class="rtm-info-card"><h4 style="margin-top: 0;">OPERATIONAL PROTOCOL</h4><ol style="color: #A0AEC0;"><li>Calculate rolling 7-day DFA α for the underlying asset.</li><li>Trigger alarms if α diverges more than <strong>0.05</strong> from baseline equilibrium.</li><li>Risk managers have a statistical mean of <strong>9.75 days</strong> to reduce exposure before capitulation.</li></ol></div>""", unsafe_allow_html=True)
    else:
        st.warning("Ensure 'crash_alpha_analysis.csv' is present in the repository.")

# ------------------------------------------
# MODULE 3: FORENSIC LABORATORY
# ------------------------------------------
elif menu == "FORENSIC LABORATORY":
    st.markdown("## RTM FORENSIC LABORATORY")
    st.markdown("<p style='color: #A0AEC0;'>High-resolution reconstruction of historical phase transitions using 1-minute order book data.</p>", unsafe_allow_html=True)
    
    event_explanations = {
        "NOVEMBER 2022 (FTX COLLAPSE)": """<div style="color: #A0AEC0; font-size: 0.95em;"><h4 style="color: #FFFFFF;">FTX SOLVENCY CRISIS (CHRONIC VISCOSITY)</h4><b>EMPIRICAL EVIDENCE:</b> Sustained "plateau" of viscosity. α holds between 1.10 and 1.25 for over 96 hours.<br><br><b>PHYSICAL STATE:</b> Evaporation of trust. Market behaved like a high-viscosity fluid; all movements required immense volume.</div>""",
        "MARCH 2020 (BLACK THURSDAY)": """<div style="color: #A0AEC0; font-size: 0.95em;"><h4 style="color: #FFFFFF;">COVID LIQUIDITY SHOCK (SOLID STATE)</h4><b>EMPIRICAL EVIDENCE:</b> α climbs parabolically, crossing 1.20 at 10:15 UTC.<br><br><b>RTM LEAD TIME:</b> Definite warning <b>60 minutes before</b> the final plunge to $5,500.<br><br><b>PHYSICAL STATE:</b> Non-Newtonian fluid: extreme velocity forced the network to freeze into a solid state.</div>""",
        "MAY 2021 (CHINA BAN)": """<div style="color: #A0AEC0; font-size: 0.95em;"><h4 style="color: #FFFFFF;">CHINA BAN SHOCK (HIGH-ENERGY TURBULENCE)</h4><b>EMPIRICAL EVIDENCE:</b> α peaks at 1.33 but reverts swiftly, never touching the 2.0 fracture line.<br><br><b>PHYSICAL STATE:</b> Turbulent flow. Network remained liquid, processing repricing without structural failure.</div>""",
        "SEPTEMBER 2023 (CONTROL GROUP)": """<div style="color: #A0AEC0; font-size: 0.95em;"><h4 style="color: #FFFFFF;">THE CONTROL GROUP (RESTING HEART RATE)</h4><b>EMPIRICAL EVIDENCE:</b> Null Hypothesis test. Price action is minimal; α remains flat.<br><br><b>KEY FINDINGS:</b> Baseline average <b>α ≈ 0.45</b>. Zero false-positives confirmed.</div>"""
    }
    
    event_dict = {
        "NOVEMBER 2022 (FTX COLLAPSE)": "BTCUSDT-1m-2022-11.csv",
        "MARCH 2020 (BLACK THURSDAY)": "BTCUSDT-1m-2020-03.csv",
        "MAY 2021 (CHINA BAN)": "BTCUSDT-1m-2021-05.csv",
        "SEPTEMBER 2023 (CONTROL GROUP)": "BTCUSDT-1m-2023-09.csv"
    }
    
    event = st.selectbox("SELECT HISTORICAL EVENT:", list(event_dict.keys()))
    
    # Use os.path.join to find the file accurately based on where app.py is located
    file_name = event_dict[event]
    full_path = os.path.join(BASE_DIR, file_name)
    
    df = load_and_process_data(full_path)
    
    if df is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="PRICE (USD)", line=dict(color='#00E5FF', width=2), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.05)'), secondary_y=False)
        
        # Alpha line updated to intense red
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Alpha'], name="RTM ALPHA (α)", line=dict(color='#FF0000', width=2.2)), secondary_y=True)
        
        # Legend items for thresholds
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(255, 23, 68, 0.6)', dash='dash'), name='FRACTURE (2.0)'), secondary_y=True)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(255, 234, 0, 0.6)', dash='dash'), name='VISCOSITY (1.2)'), secondary_y=True)
        
        fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255, 23, 68, 0.5)", secondary_y=True)
        fig.add_hline(y=1.2, line_dash="dash", line_color="rgba(255, 234, 0, 0.5)", secondary_y=True)
        
        # DYNAMIC FORENSIC EVENT OVERLAYS
        if event != "SEPTEMBER 2023 (CONTROL GROUP)":
            peak_idx = df['Rolling_Alpha'].idxmax()
            
            # Calculate dynamic indices to ensure markers always appear within the data bounds
            start_idx = max(0, peak_idx - 180) # Approx 3 hours before peak
            cap_idx = min(len(df)-1, peak_idx + 120) # Approx 2 hours after peak
            
            peak_date = df.loc[peak_idx, 'Date']
            start_date = df.loc[start_idx, 'Date']
            cap_date = df.loc[cap_idx, 'Date']
            
            # Obtener el valor exacto de Alpha en esos momentos para alinear el texto
            start_alpha = df.loc[start_idx, 'Rolling_Alpha']
            peak_alpha = df.loc[peak_idx, 'Rolling_Alpha']
            cap_alpha = df.loc[cap_idx, 'Rolling_Alpha']
            
            # Adding vertical event lines
            fig.add_vline(x=start_date, line_dash="dot", line_color="#A0AEC0", opacity=0.6)
            fig.add_annotation(
                x=start_date, y=start_alpha + 0.15, text="SYSTEMIC STRESS ONSET", showarrow=False, 
                font=dict(color="#A0AEC0", size=10), yref="y2", bgcolor="rgba(11, 14, 20, 0.8)", bordercolor="#A0AEC0", borderwidth=1, borderpad=4,
                xanchor="left", xshift=10
            )
            
            fig.add_vline(x=peak_date, line_dash="solid", line_color="#FFEA00", opacity=0.8)
            fig.add_annotation(
                x=peak_date, y=peak_alpha + 0.15, text="MAX STRUCTURAL ENTROPY", showarrow=False, 
                font=dict(color="#FFEA00", size=10, weight="bold"), yref="y2", bgcolor="rgba(11, 14, 20, 0.8)", bordercolor="#FFEA00", borderwidth=1, borderpad=4,
                xanchor="left", xshift=10
            )
            
            fig.add_vline(x=cap_date, line_dash="dot", line_color="#FF1744", opacity=0.6)
            fig.add_annotation(
                x=cap_date, y=cap_alpha + 0.15, text="LIQUIDITY CASCADE", showarrow=False, 
                font=dict(color="#FF1744", size=10), yref="y2", bgcolor="rgba(11, 14, 20, 0.8)", bordercolor="#FF1744", borderwidth=1, borderpad=4,
                xanchor="left", xshift=10
            )
            
        fig = apply_premium_layout(fig, chart_height=750)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(event_explanations[event], unsafe_allow_html=True)
    else:
        st.warning(f"Unable to load data for {event}. Please ensure {file_name} is uploaded to the repository.")

# ------------------------------------------
# MODULE 4: MARKET PHYSICS
# ------------------------------------------
elif menu == "MARKET PHYSICS":
    st.markdown("## MARKET PHYSICS (UNIVERSAL LAWS)")
    # ALPHA DISTRIBUTION TAB
    tab1, tab2, tab3 = st.tabs(["FAT TAILS & POWER LAWS", "RECOVERY TIME SCALING", "ALPHA DISTRIBUTION"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        # Introductory text for Fat Tails
        st.markdown("""
        <div style="color: #E2E8F0; margin-bottom: 20px;">
            Traditional financial models assume that market returns follow a <b>Normal Distribution (Gaussian)</b>. 
            RTM Physics demonstrates that global markets are scale-free networks governed by <b>Power Laws</b>. 
            This fundamental shift reveals that extreme price movements are not random accidents, but predictable features of the market's internal geometry.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### INVERSE CUBIC LAW")
            st.markdown("<p style='color: #A0AEC0;'>Across global markets, the tail exponent resolves to <b>α ≈ 3</b>. Black Swans are topological features, not anomalies.</p>", unsafe_allow_html=True)
            st.metric(label="GLOBAL α EXPONENT", value="2.966 ± 0.236")
        with col2:
            st.markdown("#### PROBABILITY CALCULATOR")
            sigma = st.slider("SIGMA (STD DEVIATIONS)", 2, 10, 5)
            gauss_prob = np.exp(-sigma**2 / 2)
            rtm_prob = sigma ** -3
            st.write(f"Probability of a **{sigma}σ event**:")
            st.info(f"GAUSSIAN: 1 IN {int(1/gauss_prob):,} DAYS")
            st.success(f"RTM PHYSICS: 1 IN {int(1/rtm_prob):,} DAYS")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        # Introductory text for Recovery Scaling
        st.markdown("""
        <div style="color: #E2E8F0; margin-bottom: 20px;">
            When a financial network suffers a structural fracture (Bifurcation), it enters a state of trauma. 
            RTM analysis demonstrates that the <b>Recovery Time</b> follows a specific scaling law. 
            By measuring the depth of the collapse, we can mathematically estimate the duration required for the system to dissipate accumulated entropy and return to its baseline coherence.
        </div>
        """, unsafe_allow_html=True)
        
        drawdown = st.number_input("PEAK-TO-TROUGH DRAWDOWN (%)", 10, 90, 40)
        scaled_recovery = 365 * ((drawdown / 20.0) ** (3.59 / 2.0))
        st.metric(label="ESTIMATED RECOVERY TIME", value=f"{int(scaled_recovery):,} DAYS")

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #E2E8F0; margin-bottom: 20px;">
            <b>EMPIRICAL ALPHA DISTRIBUTION:</b> This histogram visualizes a simulated 10-year dataset based on RTM empirical findings. 
            Notice the extreme "Fat Tail" (Power Law) extending to the right. While the market spends 95% of its time in the Laminar zone (around 0.45), 
            the probability of crossing the 2.0 Fracture threshold, though rare, is significantly higher than traditional normal distributions predict.
        </div>
        """, unsafe_allow_html=True)
        
        # Generate synthetic realistic RTM Alpha distribution
        np.random.seed(42)
        laminar = np.random.normal(0.45, 0.12, 18000)
        turbulent = np.random.normal(0.95, 0.15, 1500)
        viscous = np.random.normal(1.4, 0.2, 400)
        fracture = np.random.normal(2.1, 0.25, 50)
        dist = np.concatenate([laminar, turbulent, viscous, fracture])
        dist = dist[(dist > 0.1) & (dist < 3.2)] # Clean boundaries
        
        fig_hist = px.histogram(dist, nbins=120, labels={'value': 'RTM Alpha (α)'})
        fig_hist.update_traces(marker_color='#00E5FF', marker_line_color='#0B0E14', marker_line_width=1)
        fig_hist.add_vline(x=1.2, line_dash="dash", line_color="#FFEA00", annotation_text="VISCOSITY", annotation_font_color="#FFEA00")
        fig_hist.add_vline(x=2.0, line_dash="dash", line_color="#FF1744", annotation_text="FRACTURE", annotation_font_color="#FF1744")
        
        fig_hist = apply_premium_layout(fig_hist, chart_height=500)
        fig_hist.update_layout(showlegend=False, yaxis_title="Frequency (Log Scale)", yaxis_type="log")
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("""
<div class="rtm-footer">
    Powered by RTM-Atmo Technology | <a href="https://github.com/zarpafantasma/corpus_rythmos" target="_blank">github.com/zarpafantasma/corpus_rythmos</a>
</div>
""", unsafe_allow_html=True)
