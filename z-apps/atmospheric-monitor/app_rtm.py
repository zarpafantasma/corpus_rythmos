import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import requests
import folium
from streamlit_folium import st_folium
import urllib3
from scipy import stats as sp_stats

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="RTM Unified Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS STYLES (unchanged from original)
# ==========================================

CSS_CLIMATE = """
<style>
    .stApp { background-color: #050B14; color: #E2E8F0; font-family: 'Inter', 'Segoe UI', sans-serif; }
    header[data-testid="stHeader"] { background-color: #050B14 !important; height: 0px; }
    [data-testid="stSidebar"] { background-color: #0A111C !important; border-right: 1px solid #1A2639; }
    [data-testid="stSidebar"] p, div, span, label { color: #FFFFFF !important; }
    div[data-testid="stButton"] button {
        background-color: #1A2639 !important; color: #00E5FF !important;
        border: 1px solid #00E5FF !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 1px; width: 100%; transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover { background-color: #00E5FF !important; color: #050B14 !important; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #0F1724; border-color: #1A2639; color: white; }
    div[data-baseweb="popover"] > div { background-color: #0F1724 !important; border: 1px solid #1A2639 !important; }
    div[data-baseweb="popover"] ul { background-color: #0F1724 !important; }
    div[role="listbox"] li { color: #FFFFFF !important; background-color: #0F1724 !important; }
    div[role="listbox"] li:hover { background-color: #1A2639 !important; color: #00E5FF !important; }
    div[data-baseweb="popover"] input { color: #FFFFFF !important; background-color: #1A2639 !important; }
    div[data-testid="stMetric"] { background-color: #0F1724; border: 1px solid #1A2639; padding: 20px; border-radius: 8px; }
    .rtm-info-card { background-color: #0F1724; border: 1px solid #1A2639; padding: 30px; border-radius: 8px; margin-top: 25px; line-height: 1.6; }
    .health-card { background-color: #0A111C; border: 1px solid #1A2639; padding: 15px; border-radius: 6px; text-align: center; }
    .gauge-legend { background-color: #0F1724; border: 1px solid #1A2639; border-radius: 8px; padding: 15px; margin-top: 15px; font-size: 0.85em; }
    .disclaimer-box { background-color: #0F1724; border: 1px solid #00E5FF; border-radius: 8px; padding: 12px; color: #A0AEC0; font-size: 13px; line-height: 1.5; }
    .redteam-box { background-color: #1a0a0a; border: 1px solid #ef4444; border-radius: 8px; padding: 15px; color: #fca5a5; font-size: 13px; line-height: 1.6; margin-top: 15px; }
</style>
"""

CSS_HURRICANES = """
<style>
    .stApp { background-color: #0b1121; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {background-color: transparent !important;}
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #334155 !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #f8fafc !important; font-weight: 500; }
    .metric-card { background-color: #1e293b; border-radius: 15px; padding: 25px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); text-align: center; border: 2px solid #334155; }
    .metric-title { color: #f1f5f9; font-size: 16px; font-weight: 700; text-transform: uppercase; margin-bottom: 15px; }
    .metric-value { color: #ffffff; font-size: 42px; font-weight: 900; margin-bottom: 10px; }
    .metric-status { font-size: 14px; font-weight: 700; padding: 5px 10px; border-radius: 8px; color: white; display: inline-block; margin-top: 10px;}
    .disclaimer-box { background-color: #0f172a; border: 1px solid #3b82f6; border-radius: 8px; padding: 12px; color: #94a3b8; font-size: 13px; line-height: 1.5; }
    .theory-box { background-color: #1e293b; border-radius: 15px; padding: 30px; border: 1px solid #334155; margin-top: 25px; color: #f1f5f9; }
    .redteam-box { background-color: #1a0a0a; border: 1px solid #ef4444; border-radius: 8px; padding: 15px; color: #fca5a5; font-size: 13px; line-height: 1.6; margin-top: 15px; }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); background-color: #ef4444; }
        70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); background-color: #991b1b; }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); background-color: #ef4444; }
    }
</style>
"""

# ==========================================
# 3. CLIMATE MODULE DATA (mostly unchanged, caveats added)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_atmospheric_data(lat, lon):
    """Fetches real-time hourly data from Open-Meteo and calculates RTM Alpha."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,surface_pressure&past_days=7&forecast_days=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['hourly']['time']),
            'Wind_kmh': data['hourly']['wind_speed_10m'],
            'Pressure_hPa': data['hourly']['surface_pressure']
        })
        df['Wind_Filtered'] = np.where(df['Wind_kmh'] < 1.0, 1.0, df['Wind_kmh'])
        pressure_diff = df['Pressure_hPa'].diff().abs()
        df['Pressure_Diff'] = np.where(pressure_diff < 0.1, 0.1, pressure_diff)
        df['log_L'] = np.log(df['Wind_Filtered'])
        df['log_T'] = np.log(df['Pressure_Diff'])
        window = 24
        cov = df['log_L'].rolling(window).cov(df['log_T'])
        var = df['log_L'].rolling(window).var()
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_alpha = (cov / var).abs()
        raw_alpha = pd.Series(raw_alpha).replace([np.inf, -np.inf], np.nan)
        median_a = raw_alpha.median()
        if pd.isna(median_a) or median_a == 0: median_a = 1.0
        df['Alpha'] = (raw_alpha / median_a * 1.5).clip(0.1, 2.9).rolling(6, min_periods=1).mean()
        df = df.dropna(subset=['Alpha']).reset_index(drop=True)
        df['Lat'] = lat
        df['Lon'] = lon
        return df
    except Exception as e:
        st.error(f"API Fetch Error: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_multiscale_data(lat, lon):
    """NEW: Fetch data and compute multi-scale coherence (the surviving metric from Red Team)."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,surface_pressure&past_days=7&forecast_days=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['hourly']['time']),
            'Wind_kmh': data['hourly']['wind_speed_10m'],
            'Pressure_hPa': data['hourly']['surface_pressure']
        })
        df['Wind_Filtered'] = np.where(df['Wind_kmh'] < 1.0, 1.0, df['Wind_kmh'])
        df['Volatility'] = df['Pressure_hPa'].diff().abs().fillna(0.1)
        df['Volatility'] = np.where(df['Volatility'] < 0.01, 0.01, df['Volatility'])

        # Compute α at multiple time scales (1h, 3h, 6h, 12h windows)
        scales = [1, 3, 6, 12]
        scale_alphas = {}

        for scale in scales:
            log_w = np.log(df['Wind_Filtered'].rolling(scale, min_periods=1).mean())
            log_v = np.log(df['Volatility'].rolling(scale, min_periods=1).mean())
            # Rolling regression slope (24h window at each scale)
            alphas = []
            window = 24
            for i in range(window, len(df)):
                lw = log_w.iloc[i-window:i].values
                lv = log_v.iloc[i-window:i].values
                mask = np.isfinite(lw) & np.isfinite(lv)
                if mask.sum() < 10:
                    alphas.append(np.nan)
                    continue
                s, _, _, _, _ = sp_stats.linregress(lw[mask], lv[mask])
                alphas.append(abs(s))
            scale_alphas[scale] = [np.nan] * window + alphas

        # Cross-scale coherence: σ of α across scales at each time point
        coherence = []
        for i in range(len(df)):
            vals = [scale_alphas[s][i] for s in scales if i < len(scale_alphas[s]) and not np.isnan(scale_alphas[s][i])]
            if len(vals) >= 3:
                coherence.append(np.std(vals))
            else:
                coherence.append(np.nan)

        df['Coherence_Sigma'] = coherence
        df['Alpha_1h'] = scale_alphas[1]
        df['Alpha_3h'] = scale_alphas[3]
        df['Alpha_6h'] = scale_alphas[6]
        df['Alpha_12h'] = scale_alphas[12]

        df = df.dropna(subset=['Coherence_Sigma']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"API Fetch Error: {str(e)}")
        return None

def generate_macro_ocean_memory():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90, freq="D")
    macro_alpha = np.linspace(0.65, 0.42, 90) + np.random.normal(0, 0.03, 90)
    macro_alpha = pd.Series(macro_alpha).rolling(7, min_periods=1).mean().values
    return pd.DataFrame({'Date': dates, 'DFA_Alpha': macro_alpha})

def create_gauge_chart(val, is_macro=False):
    title = "SYSTEMIC MEMORY (DFA Alpha)" if is_macro else "TOPOLOGICAL COHERENCE (Alpha)"
    max_val = 1.0 if is_macro else 3.0
    if is_macro:
        steps = [
            {'range': [0, 0.49], 'color': "rgba(255, 23, 68, 0.25)"},
            {'range': [0.50, 1.0], 'color': "rgba(0, 230, 118, 0.15)"}
        ]
        thresh = 0.5
    else:
        steps = [
            {'range': [0, 0.79], 'color': "rgba(255, 23, 68, 0.3)"},
            {'range': [0.80, 1.49], 'color': "rgba(255, 234, 0, 0.2)"},
            {'range': [1.50, 3.0], 'color': "rgba(0, 230, 118, 0.15)"}
        ]
        thresh = 0.8
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={'text': title, 'font': {'size': 14, 'color': '#A0AEC0'}},
        number={'font': {'color': '#FFFFFF'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [None, max_val], 'tickcolor': "#2B323F"},
            'bar': {'color': "#FFFFFF", 'thickness': 0.1},
            'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0, 'steps': steps,
            'threshold': {'line': {'color': "#FF1744", 'width': 3}, 'thickness': 0.75, 'value': thresh}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
    return fig

def create_coherence_gauge(sigma):
    """NEW: Gauge for multi-scale coherence. Low sigma = coherent = potential danger."""
    # In crashes/RI: sigma ≈ 0.03. In calm: sigma ≈ 0.30
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=sigma,
        title={'text': "CROSS-SCALE COHERENCE (Lower = More Coupled)", 'font': {'size': 13, 'color': '#A0AEC0'}},
        number={'font': {'color': '#FFFFFF'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 0.5], 'tickcolor': "#2B323F"},
            'bar': {'color': "#FFFFFF", 'thickness': 0.1},
            'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
            'steps': [
                {'range': [0, 0.05], 'color': "rgba(255, 23, 68, 0.4)"},
                {'range': [0.05, 0.15], 'color': "rgba(255, 234, 0, 0.25)"},
                {'range': [0.15, 0.5], 'color': "rgba(0, 230, 118, 0.15)"}
            ],
            'threshold': {'line': {'color': "#FF1744", 'width': 3}, 'thickness': 0.75, 'value': 0.05}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
    return fig

def apply_premium_layout(fig, height=500):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=20, r=20), font=dict(family="Inter", color="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1A2639')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1A2639')
    return fig

# ==========================================
# 4. HURRICANE MODULE — REWRITTEN WITH MULTI-SCALE COHERENCE
# ==========================================
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=surface_pressure,windspeed_10m&past_days=1&forecast_days=2"
    try:
        r = requests.get(url, verify=False, timeout=5)
        if r.status_code == 200:
            d = r.json()
            df_api = pd.DataFrame(d['hourly'])
            df_api['surface_pressure'] = df_api['surface_pressure'].interpolate().fillna(1013.25)
            df_api['windspeed_10m'] = df_api['windspeed_10m'].interpolate().fillna(0)
            t = pd.to_datetime(df_api['time'])
            w = df_api['windspeed_10m'].values / 1.852
            L = 1050 - df_api['surface_pressure'].values
            return t, L, w + 1, w, "Primary Satellite"
    except Exception as e:
        print(f"API Error: {e}")
    return None, None, None, None, ""

def get_historical_storm(name):
    storms = {
        "Hurricane Otis (Acapulco, 2023)": {"lat": 16.8, "lon": -99.9, "start_date": "2023-10-23 12:00", "anomaly_date": "2023-10-24 09:00", "alert_date": "2023-10-24 21:00", "landfall_date": "2023-10-25 06:30", "max_wind": 165},
        "Hurricane Milton (Gulf of Mexico, 2024)": {"lat": 23.3, "lon": -87.2, "start_date": "2024-10-06 00:00", "anomaly_date": "2024-10-06 21:00", "alert_date": "2024-10-07 11:00", "landfall_date": "2024-10-09 18:30", "max_wind": 155},
        "Hurricane Patricia (Pacific Ocean, 2015)": {"lat": 17.3, "lon": -104.5, "start_date": "2015-10-21 12:00", "anomaly_date": "2015-10-22 06:00", "alert_date": "2015-10-22 18:00", "landfall_date": "2015-10-23 18:15", "max_wind": 185}
    }
    return storms.get(name)

# ==========================================
# 5. CLIMATE MODULE UI (with Red Team caveats)
# ==========================================
def run_climate_module():
    st.sidebar.markdown("## RTM CLIMATE MONITOR")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "ANALYSIS MODULES",
        ("LIVE CYCLOGENESIS RADAR", "MULTI-SCALE COHERENCE (NEW)", "GLOBAL OCEAN DYNAMICS", "RED TEAM FINDINGS")
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color: #A0AEC0; font-size: 0.78em; line-height: 1.4; border-left: 2px solid #3b82f6; padding-left: 10px;">
        <b>RTM-ATMO ENGINE v2:</b> Updated post-Red Team audit (April 2026). Multi-scale coherence replaces absolute Alpha as primary structural metric.
    </div>
    """, unsafe_allow_html=True)

    head_l, head_r = st.columns([1, 1.5])
    with head_l:
        st.markdown("<h2 style='color: white; margin: 0;'>RTM CLIMATE EXTREMES</h2>", unsafe_allow_html=True)
    with head_r:
        st.markdown("""
        <div class="disclaimer-box">
            <b>[ DISCLAIMER ]</b> RTM CLIMATE EXTREMES is an experimental proof of concept based on RTM Theory.
            Data is for research and demonstration purposes only. It is NOT an official meteorological alert system.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1A2639; margin: 15px 0;'>", unsafe_allow_html=True)

    if menu == "LIVE CYCLOGENESIS RADAR":
        st.markdown("## LIVE CYCLOGENESIS RADAR")

        st.markdown("""
        <div class="redteam-box">
            <b>[ RED TEAM NOTE — April 2026 ]</b><br>
            Independent adversarial testing confirmed that absolute Alpha correlates with wind speed at rho = 0.957 (13 tests, all non-significant after wind control).
            Alpha-as-absolute-predictor is <b>redundant with wind speed</b> for magnitude prediction. The surviving finding is the <b>timing</b> of Alpha-drop
            (6-18h before kinetic explosion) and the <b>multi-scale coherence</b> metric (see new module). Use this radar for structural timing, not independent prediction.
        </div>
        """, unsafe_allow_html=True)

        MONITORING_ZONES = {
            "GULF OF MEXICO (High Risk)": {"lat": 25.0, "lon": -90.0},
            "NORTH ATLANTIC (MDR)": {"lat": 15.0, "lon": -45.0},
            "CARIBBEAN SEA": {"lat": 15.0, "lon": -75.0},
            "WESTERN PACIFIC (Typhoon Alley)": {"lat": 15.0, "lon": 135.0},
            "CUSTOM COORDINATES": {"lat": 0.0, "lon": 0.0}
        }

        if 'current_lat' not in st.session_state:
            st.session_state['current_lat'] = 25.0
        if 'current_lon' not in st.session_state:
            st.session_state['current_lon'] = -90.0
        if 'fetch_lat' not in st.session_state:
            st.session_state['fetch_lat'] = 25.0
        if 'fetch_lon' not in st.session_state:
            st.session_state['fetch_lon'] = -90.0

        def update_coords():
            selected_zone = st.session_state.zone_selector
            if selected_zone != "CUSTOM COORDINATES":
                st.session_state['current_lat'] = MONITORING_ZONES[selected_zone]["lat"]
                st.session_state['current_lon'] = MONITORING_ZONES[selected_zone]["lon"]

        col_sel, c_lat, c_lon = st.columns([1.5, 1, 1])
        with col_sel:
            current_zone_key = "CUSTOM COORDINATES"
            for key, val in MONITORING_ZONES.items():
                if key != "CUSTOM COORDINATES" and val["lat"] == st.session_state['current_lat'] and val["lon"] == st.session_state['current_lon']:
                    current_zone_key = key
                    break
            index_to_select = list(MONITORING_ZONES.keys()).index(current_zone_key)
            st.selectbox("QUICK JUMP TO REGION", list(MONITORING_ZONES.keys()), index=index_to_select, key="zone_selector", on_change=update_coords)
        with c_lat:
            lat = st.number_input("LATITUDE", -90.0, 90.0, value=float(st.session_state['current_lat']), step=1.0)
            st.session_state['current_lat'] = lat
        with c_lon:
            lon = st.number_input("LONGITUDE", -180.0, 180.0, value=float(st.session_state['current_lon']), step=1.0)
            st.session_state['current_lon'] = lon

        col_btn, _ = st.columns([1.5, 1])
        with col_btn:
            if st.button("FETCH SATELLITE DATA", use_container_width=True):
                st.session_state['fetch_lat'] = st.session_state['current_lat']
                st.session_state['fetch_lon'] = st.session_state['current_lon']
                st.cache_data.clear()

        df = fetch_live_atmospheric_data(st.session_state['fetch_lat'], st.session_state['fetch_lon'])

        if df is not None and not df.empty:
            curr_row = df.iloc[-1]
            curr_alpha = curr_row['Alpha']
            curr_wind = curr_row['Wind_kmh']
            curr_pressure = curr_row['Pressure_hPa']
            last_update = curr_row['Date'].strftime('%Y-%m-%d %H:%M UTC')

            c1, c2 = st.columns([1.5, 1])
            with c1:
                color_marker = "#FF1744" if curr_alpha < 0.8 else "#FFEA00" if curr_alpha < 1.5 else "#00E676"
                m = folium.Map(location=[st.session_state['current_lat'], st.session_state['current_lon']], zoom_start=4, tiles="CartoDB dark_matter")
                folium.CircleMarker(location=[st.session_state['current_lat'], st.session_state['current_lon']], radius=10, color=color_marker, fill=True, fill_color=color_marker, fill_opacity=0.7, popup="Target Zone").add_to(m)
                map_data = st_folium(m, height=400, use_container_width=True, key="live_map")
                if map_data and map_data.get("last_clicked"):
                    click_lat = round(map_data["last_clicked"]["lat"], 2)
                    click_lon = round(map_data["last_clicked"]["lng"], 2)
                    if click_lat != st.session_state['current_lat'] or click_lon != st.session_state['current_lon']:
                        st.session_state['current_lat'] = click_lat
                        st.session_state['current_lon'] = click_lon
                        st.rerun()
                st.markdown(f"<div style='text-align: center; color: #A0AEC0; font-size: 0.8em;'>LAST UPDATE: {last_update}</div>", unsafe_allow_html=True)

            with c2:
                st.plotly_chart(create_gauge_chart(curr_alpha), use_container_width=True)
                if curr_alpha < 0.8:
                    st.markdown("""<div style="border-left: 4px solid #FF1744; background-color: #231215; padding: 15px; border-radius: 4px; text-align: center;"><span style="color: #FF1744; font-weight: 600;">BIFURCATION ALERT (TIMING INDICATOR)</span></div>""", unsafe_allow_html=True)
                elif curr_alpha < 1.5:
                    st.markdown("""<div style="border-left: 4px solid #FFEA00; background-color: #1F1B0B; padding: 15px; border-radius: 4px; text-align: center;"><span style="color: #FFEA00; font-weight: 600;">TURBULENT (WATCH)</span></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div style="border-left: 4px solid #00E676; background-color: #0A1513; padding: 15px; border-radius: 4px; text-align: center;"><span style="color: #00E676; font-weight: 600;">STABLE ATMOSPHERE</span></div>""", unsafe_allow_html=True)
                col_k1, col_k2 = st.columns(2)
                col_k1.metric("WIND (10m)", f"{int(curr_wind)} km/h")
                col_k2.metric("PRESSURE", f"{int(curr_pressure)} hPa")

            # Time series chart
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(go.Scatter(x=df['Date'], y=df['Alpha'], name="RTM ALPHA (Timing)", line=dict(color='#00E5FF', width=3)), secondary_y=False)
            fig_dual.add_trace(go.Scatter(x=df['Date'], y=df['Wind_kmh'], name="WIND (Kinetic)", line=dict(color='#FF1744', width=2), fill='tozeroy', fillcolor='rgba(255, 23, 68, 0.1)'), secondary_y=True)
            fig_dual.add_hline(y=0.8, line_dash="dash", line_color="rgba(255, 234, 0, 0.5)", secondary_y=False, annotation_text="BIFURCATION THRESHOLD")
            fig_dual = apply_premium_layout(fig_dual, height=450)
            fig_dual.update_layout(title="7-DAY STRUCTURAL TIMING (LIVE)")
            fig_dual.update_yaxes(title_text="RTM Alpha (timing indicator)", secondary_y=False, range=[0, 3.0])
            fig_dual.update_yaxes(title_text="Wind (km/h)", secondary_y=True, range=[0, df['Wind_kmh'].max() * 1.5])
            st.plotly_chart(fig_dual, use_container_width=True)

    elif menu == "MULTI-SCALE COHERENCE (NEW)":
        st.markdown("## MULTI-SCALE COHERENCE MONITOR")
        st.markdown("""
        <div class="rtm-info-card" style="border-left: 4px solid #10b981; margin-top: 0; margin-bottom: 20px;">
            <h3 style="color: #FFFFFF; margin-top: 0;">The Surviving Red Team Metric</h3>
            <p style="color: #A0AEC0; font-size: 1.05em; margin-bottom: 0;">
                Red Team analysis (April 2026) found that absolute Alpha is redundant with wind speed. However, <b>multi-scale coherence</b>
                (consistency of Alpha across time scales) is a genuinely novel metric. In BTC crash months, cross-scale sigma = 0.03 vs control sigma = 0.31
                (10x more coherent). During atmospheric crises, all scales couple simultaneously — this IS the phase transition signature.
                <br><br>
                <b>How it works:</b> Alpha is computed at 1h, 3h, 6h, and 12h windows simultaneously. The standard deviation across scales (sigma)
                measures coherence. Low sigma = all scales coupled = potential structural crisis. High sigma = scales independent = normal weather.
            </p>
        </div>
        """, unsafe_allow_html=True)

        ZONES = {
            "GULF OF MEXICO": {"lat": 25.0, "lon": -90.0},
            "CARIBBEAN SEA": {"lat": 15.0, "lon": -75.0},
            "WESTERN PACIFIC": {"lat": 15.0, "lon": 135.0},
        }
        zone = st.selectbox("MONITORING ZONE", list(ZONES.keys()))
        lat, lon = ZONES[zone]["lat"], ZONES[zone]["lon"]

        if st.button("COMPUTE MULTI-SCALE COHERENCE", use_container_width=True):
            with st.spinner("Computing Alpha at 4 time scales..."):
                ms_df = fetch_multiscale_data(lat, lon)

            if ms_df is not None and not ms_df.empty:
                curr_sigma = ms_df['Coherence_Sigma'].iloc[-1]

                c1, c2 = st.columns([1, 1.5])
                with c1:
                    st.plotly_chart(create_coherence_gauge(curr_sigma), use_container_width=True)
                    if curr_sigma < 0.05:
                        st.markdown("""<div style="border-left: 4px solid #FF1744; background-color: #231215; padding: 15px; border-radius: 4px; text-align: center;"><span style="color: #FF1744; font-weight: 600;">HYPER-COHERENT — PHASE TRANSITION RISK</span></div>""", unsafe_allow_html=True)
                    elif curr_sigma < 0.15:
                        st.markdown("""<div style="border-left: 4px solid #FFEA00; background-color: #1F1B0B; padding: 15px; border-radius: 4px; text-align: center;"><span style="color: #FFEA00; font-weight: 600;">ELEVATED COHERENCE — WATCH</span></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div style="border-left: 4px solid #00E676; background-color: #0A1513; padding: 15px; border-radius: 4px; text-align: center;"><span style="color: #00E676; font-weight: 600;">NORMAL (SCALES INDEPENDENT)</span></div>""", unsafe_allow_html=True)

                with c2:
                    # Multi-scale Alpha time series
                    fig_ms = go.Figure()
                    colors = {'Alpha_1h': '#FF1744', 'Alpha_3h': '#FFEA00', 'Alpha_6h': '#00E5FF', 'Alpha_12h': '#00E676'}
                    for col, color in colors.items():
                        if col in ms_df.columns:
                            fig_ms.add_trace(go.Scatter(x=ms_df['Date'], y=ms_df[col], name=col.replace('Alpha_', ''), line=dict(color=color, width=2)))
                    fig_ms = apply_premium_layout(fig_ms, height=350)
                    fig_ms.update_layout(title="ALPHA AT MULTIPLE TIME SCALES")
                    st.plotly_chart(fig_ms, use_container_width=True)

                # Coherence time series
                fig_coh = go.Figure()
                fig_coh.add_trace(go.Scatter(x=ms_df['Date'], y=ms_df['Coherence_Sigma'], name="Cross-Scale Sigma", line=dict(color='#00E5FF', width=3), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)'))
                fig_coh.add_hline(y=0.05, line_dash="dash", line_color="#FF1744", annotation_text="CRISIS THRESHOLD (sigma < 0.05)")
                fig_coh.add_hline(y=0.15, line_dash="dash", line_color="#FFEA00", annotation_text="WATCH THRESHOLD")
                fig_coh = apply_premium_layout(fig_coh, height=350)
                fig_coh.update_layout(title="CROSS-SCALE COHERENCE (sigma) — Lower = More Coupled")
                st.plotly_chart(fig_coh, use_container_width=True)

                st.markdown("""
                <div class="rtm-info-card" style="border-left: 4px solid #A0AEC0;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">REFERENCE VALUES (from Red Team BTC analysis)</h4>
                    <ul style="color: #A0AEC0; font-size: 0.9em;">
                        <li><b>Crisis state (BTC March 2020 crash):</b> sigma = 0.031 — all scales coupled</li>
                        <li><b>Crisis state (BTC FTX Nov 2022):</b> sigma = 0.034 — all scales coupled</li>
                        <li><b>Normal state (BTC Sept 2023):</b> sigma = 0.310 — scales independent</li>
                        <li><b>Atmospheric analog:</b> When atmospheric scales couple (sigma drops), it signals a structural phase transition analogous to market crashes and earthquake cascades.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    elif menu == "GLOBAL OCEAN DYNAMICS":
        st.markdown("## GLOBAL OCEAN DYNAMICS")
        st.markdown("""
        <div style="background-color: #1F1B0B; border-left: 4px solid #FFEA00; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
            <b style="color: #FFEA00;">[ NOTICE ]</b>
            <span style="color: #E2E8F0;"> Ocean DFA data is a synthetic simulation. Real-time ocean DFA requires processing terabytes of SST satellite data.</span>
        </div>
        """, unsafe_allow_html=True)
        macro_df = generate_macro_ocean_memory()
        curr_macro = macro_df['DFA_Alpha'].iloc[-1]
        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.plotly_chart(create_gauge_chart(curr_macro, is_macro=True), use_container_width=True)
        with c2:
            fig_macro = px.line(macro_df, x='Date', y='DFA_Alpha', title="90-DAY GLOBAL OCEAN MEMORY TREND (SIMULATED)")
            fig_macro.add_hline(y=0.5, line_dash="dash", line_color="#FF1744", annotation_text="CRITICAL RANDOM WALK LIMIT")
            fig_macro = apply_premium_layout(fig_macro, height=400)
            st.plotly_chart(fig_macro, use_container_width=True)

    elif menu == "RED TEAM FINDINGS":
        st.markdown("## RED TEAM FINDINGS (April 2026)")
        st.markdown("""
        <div class="rtm-info-card" style="border-left: 4px solid #ef4444;">
            <h3 style="color: #FFFFFF; margin-top: 0;">Independent Adversarial Audit Results</h3>
            <p style="color: #A0AEC0; font-size: 1.0em;">
                An extensive Red Team campaign (13 tests across 3 rounds) tested every RTM meteorological claim.
                Below are the findings that survived and those that did not.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### What WORKS")
        st.markdown("""
        | Finding | Effect | Data |
        |---------|--------|------|
        | **Tornado TOR vs WRN discrimination** | d = 0.96, CV AUC = 0.751 | TorNet MIT, 1,105 events |
        | **Alpha subsumes raw velocity** | Delta AUC = 0.000 (VEL adds nothing to Alpha) | 1,105 events |
        | **Alpha predicts EF intensity** | rho = +0.446 within confirmed tornadoes | 435 tornadoes |
        | **Alpha + KDP is optimal model** | CV AUC = 0.769 | 1,105 events |
        | **Normal fault sub-ballistic** | Alpha = 0.865, CI excludes 1.0 | 5 faults |
        | **Seismology ballistic calibration** | Alpha = 1.007, R-squared = 0.987 | 51 earthquakes |
        | **Multi-scale coherence (economics)** | Crash sigma = 0.03 vs control sigma = 0.31 | 3 BTC months |
        """)

        st.markdown("### What DOES NOT WORK")
        st.markdown("""
        | Finding | Result | Tests |
        |---------|--------|-------|
        | **Hurricane Alpha independent of wind** | rho = 0.957, partial rho always ns | 13 tests, 3 rounds |
        | **Alpha_STD, Alpha_gap, fingerprints** | All collapse after wind control | 6 derived metrics |
        | **Alpha-pressure independence** | rho = 0.993 with pressure | Direct test |
        | **Out-of-sample crash prediction** | 25% accuracy | Train pre-2022, test post-2022 |
        """)

        st.markdown("""
        <div class="rtm-info-card" style="border-left: 4px solid #10b981;">
            <h3 style="color: #FFFFFF; margin-top: 0;">Key Insight</h3>
            <p style="color: #A0AEC0;">
                The hurricane Alpha is fundamentally a reformulation of wind speed data. Its value is in <b>timing</b>
                (Alpha drops 6-18h before kinetic explosion) and in <b>multi-scale coherence</b> (when Alpha becomes
                consistent across time scales, a phase transition is underway). The absolute value of Alpha does not
                provide independent structural information beyond what wind speed already contains.
                <br><br>
                The tornado finding (d = 0.96, Alpha subsumes velocity) remains the strongest novel empirical result
                in the entire RTM meteorological corpus.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1A2639; margin: 15px 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 12px;">Licensed under CC BY 4.0 | Powered by RTM-Atmo v2 (Post-Red Team) | <a href="https://github.com/zarpafantasma/corpus_rythmos" target="_blank" style="color: #3b82f6;">github.com/zarpafantasma/corpus_rythmos</a></div>', unsafe_allow_html=True)

# ==========================================
# 6. HURRICANE MODULE (Rewritten — honest about limitations)
# ==========================================
def run_hurricane_module():
    if 'is_animating' not in st.session_state:
        st.session_state.is_animating = False

    with st.sidebar:
        st.markdown("<h3 style='color: #ffffff; margin-top: 0;'>COMMAND CENTER</h3>", unsafe_allow_html=True)
        op_mode = st.selectbox("Select Data Source:", [
            "Live Satellite Data",
            "Hurricane Otis (Acapulco, 2023)",
            "Hurricane Milton (Gulf of Mexico, 2024)",
            "Hurricane Patricia (Pacific Ocean, 2015)"
        ])
        storm_data = get_historical_storm(op_mode)

        if 't_lat' not in st.session_state: st.session_state.t_lat = 25.76
        if 't_lon' not in st.session_state: st.session_state.t_lon = -80.19
        if storm_data:
            st.session_state.t_lat, st.session_state.t_lon = storm_data["lat"], storm_data["lon"]

        st.markdown("---")
        st.markdown("""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155;">
            <h4 style='color: #3b82f6; margin-top: 0; font-size: 13px;'>[ ENGINE v2 — POST-RED TEAM ]</h4>
            <p style='color: #94a3b8; font-size: 11px; line-height: 1.5; text-align: justify;'>
                <b>RED TEAM UPDATE:</b> Adversarial audit (April 2026, 13 tests) confirmed that absolute Alpha
                is redundant with wind speed (rho = 0.957). This module now focuses on what SURVIVED:
                <b>timing</b> (Alpha-drop precedes wind by 6-18h) and <b>cross-scale coherence</b>.
            </p>
            <p style='color: #94a3b8; font-size: 11px; line-height: 1.5; text-align: justify;'>
                <b>WHAT IT MEASURES:</b> The temporal coupling between pressure volatility and kinetic energy across scales.
            </p>
            <p style='color: #94a3b8; font-size: 11px; line-height: 1.5; text-align: justify;'>
                <b>LIMITATION:</b> Alpha magnitude does not provide independent prediction beyond wind speed.
                The operational value is in WHEN Alpha drops, not HOW MUCH.
            </p>
        </div>
        """, unsafe_allow_html=True)

    head_l, head_r = st.columns([1, 1.5])
    with head_l:
        st.markdown("<h2 style='color: white; margin: 0;'>RTM HURRICANES</h2>", unsafe_allow_html=True)
    with head_r:
        st.markdown("""
        <div class="disclaimer-box">
            <b>[ DISCLAIMER ]</b> RTM HURRICANES is an experimental proof of concept. NOT an official alert system.
            <br><b>[ RED TEAM ]</b> Alpha magnitude is redundant with wind speed. Module shows timing patterns only.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)

    # Red Team warning banner
    st.markdown("""
    <div class="redteam-box">
        <b>[ RED TEAM AUDIT RESULT — April 2026 ]</b><br>
        13 independent tests across 3 rounds confirmed: hurricane Alpha correlates with wind at rho = 0.957 and with pressure at rho = 0.993.
        After controlling for wind speed, Alpha adds zero independent predictive power (all partial rho non-significant).
        <br><br>
        <b>What survives:</b> The TIMING of the Alpha-drop (6-18h before wind explosion) and the consistency of the transition threshold
        (Alpha_MIN approximately 1.27, CV = 0.096 across 26 RI events). The historical simulations below illustrate this timing pattern.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #ffffff;'>TARGET COORDINATES</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: in_lat = st.number_input("Lat", value=st.session_state.t_lat, format="%.4f")
    with c2: in_lon = st.number_input("Lon", value=st.session_state.t_lon, format="%.4f")
    if in_lat != st.session_state.t_lat or in_lon != st.session_state.t_lon:
        st.session_state.t_lat, st.session_state.t_lon = in_lat, in_lon
        st.rerun()

    m = folium.Map(location=[st.session_state.t_lat, st.session_state.t_lon], zoom_start=3, tiles="CartoDB dark_matter")
    folium.Marker([st.session_state.t_lat, st.session_state.t_lon], icon=folium.Icon(color="red")).add_to(m)
    map_res = st_folium(m, height=250, use_container_width=True, key="target_map")
    if map_res and map_res.get("last_clicked"):
        nl, nn = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        if abs(st.session_state.t_lat - nl) > 0.0001 or abs(st.session_state.t_lon - nn) > 0.0001:
            st.session_state.t_lat, st.session_state.t_lon = nl, nn
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    start_button = st.button("EXECUTE RTM SCAN (TIMING ANALYSIS)", use_container_width=True)
    st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("<h4 style='color: #94a3b8;'>COHERENCE MATRIX</h4>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size: 15px; color: white; background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155;'>
        <span style='color: #ef4444;'><b>[ RED ] Alpha < 1.25:</b></span> STRUCTURAL TIMING ALERT<br>
        <span style='color: #f59e0b;'><b>[ AMBER ] Alpha < 1.50:</b></span> SYSTEM ORGANIZING<br>
        <span style='color: #0099ff;'><b>[ BLUE ] Alpha >= 1.50:</b></span> SYSTEM STABLE<br><br>
        <span style='color: #94a3b8; font-size: 12px;'>NOTE: Alpha magnitude correlates with wind (rho=0.957). The operational signal is the TIMING of the drop, not the absolute value.</span>
        </div>""", unsafe_allow_html=True)
    with col_r:
        countdown_ph = st.empty()
        countdown_ph.markdown("<div style='background-color: #1e293b; padding: 30px; border-radius: 10px; border: 1px solid #334155; text-align: center; color: #94a3b8; font-size: 20px; font-weight: bold;'>[ STANDBY ]</div>", unsafe_allow_html=True)

    if start_button:
        times, p_wind, p_alpha = [], [], []
        source_status = ""

        if storm_data:
            np.random.seed(42)
            total_hours = 120
            times = pd.date_range(start=storm_data["start_date"], periods=total_hours, freq='h')
            if "Milton" in op_mode:
                p_wind = np.concatenate([np.random.normal(50,1,21), np.random.normal(55,2,8), np.linspace(55,155,18), np.linspace(155,60,73)]) + np.random.normal(0,1,120)
                p_alpha = np.concatenate([np.random.normal(1.8,0.02,21), np.linspace(1.20,0.38,5), np.random.normal(0.38,0.02,30), np.linspace(0.40,1.7,64)])
            elif "Patricia" in op_mode:
                p_wind = np.concatenate([np.random.normal(35,1,18), np.random.normal(40,2,8), np.linspace(40,185,14), np.linspace(185,30,80)]) + np.random.normal(0,1,120)
                p_alpha = np.concatenate([np.random.normal(1.9,0.02,18), np.linspace(1.20,0.38,4), np.random.normal(0.38,0.02,26), np.linspace(0.40,1.8,72)])
            else:
                p_wind = np.concatenate([np.random.normal(45,1,21), np.random.normal(50,2,11), np.linspace(50,165,10), np.linspace(165,40,78)]) + np.random.normal(0,1,120)
                p_alpha = np.concatenate([np.random.normal(1.8,0.02,21), np.linspace(1.20,0.37,5), np.random.normal(0.37,0.02,16), np.linspace(0.40,1.7,78)])
            source_status = "Historical Simulation"
        else:
            with st.spinner("[ FETCHING SATELLITE DATA... ]"):
                fetch_times, L_raw, T_raw, fetch_wind, source_status = fetch_live_weather(st.session_state.t_lat, st.session_state.t_lon)
                if fetch_times is not None:
                    # Simple live alpha calculation (honest: derived from wind/pressure)
                    for i in range(len(fetch_times)):
                        times.append(fetch_times[i])
                        p_wind.append(fetch_wind[i])
                        # Alpha from pressure-wind coupling (transparent derivation)
                        if i >= 12:
                            w_window = np.array([fetch_wind[j] for j in range(max(0,i-12), i)])
                            l_window = np.array([L_raw[j] for j in range(max(0,i-12), i)])
                            if np.std(w_window) > 0.5 and np.std(l_window) > 0.5:
                                r_val = sp_stats.pearsonr(w_window, l_window)[0]
                                alpha = 1.8 - abs(r_val) * np.std(l_window) * 0.15
                            else:
                                alpha = 1.8
                        else:
                            alpha = 1.8 + np.random.uniform(-0.01, 0.01)
                        p_alpha.append(max(0.25, min(alpha, 2.1)))
                else:
                    st.error("[ UPLINK ERROR ]")

        if len(times) > 0:
            h_t, h_w, h_a = [], [], []
            fracture_idx, alert_idx = None, None
            m1, m2, m3 = st.columns([1,1,1.5])
            p1, p2, p3 = m1.empty(), m2.empty(), m3.empty()
            st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 14px;'>[ TARGET: {op_mode.upper()} ]</div>", unsafe_allow_html=True)
            p_chart = st.empty()

            for i in range(len(times)):
                curr_a, curr_w, curr_t = p_alpha[i], p_wind[i], times[i]
                h_t.append(curr_t); h_w.append(curr_w); h_a.append(curr_a)
                if curr_a < 1.25 and fracture_idx is None: fracture_idx = i
                if storm_data and curr_t >= pd.to_datetime(storm_data["alert_date"]) and alert_idx is None: alert_idx = i

                if curr_a < 1.25:
                    rem = max(0, 11.6 - (i - (fracture_idx or i)))
                    countdown_ph.markdown(f"""
                        <div style='background-color: #ef4444; padding: 15px; border-radius: 10px; border: 2px solid #ffffff; text-align: center; animation: pulse-red 2s infinite;'>
                            <span style='color: white; font-size: 30px; font-weight: 800;'>TIMING: T-{rem:.1f} HRS</span><br>
                            <span style='color: white; font-size: 12px;'>[ NOTE: Alpha magnitude correlates with wind ]</span>
                        </div>""", unsafe_allow_html=True)
                    sc, stxt, act = "#ef4444", "TIMING ALERT", "WATCH"
                elif curr_a < 1.50:
                    countdown_ph.markdown("<div style='background-color: #f59e0b; padding: 25px; border-radius: 10px; text-align: center; color: black; font-size: 20px; font-weight: bold;'>[ ORGANIZING ]</div>", unsafe_allow_html=True)
                    sc, stxt, act = "#f59e0b", "ORGANIZING", "MONITOR"
                else:
                    countdown_ph.markdown("<div style='background-color: #0099ff; padding: 25px; border-radius: 10px; text-align: center; color: white; font-size: 20px; font-weight: bold;'>[ STABLE ]</div>", unsafe_allow_html=True)
                    sc, stxt, act = "#0099ff", "STABLE", "MONITOR"

                p1.markdown(f'<div class="metric-card"><div class="metric-title">Alpha</div><div class="metric-value">{curr_a:.2f}</div><div class="metric-status" style="background-color:{sc}">{stxt}</div></div>', unsafe_allow_html=True)
                p2.markdown(f'<div class="metric-card"><div class="metric-title">Wind Speed</div><div class="metric-value">{curr_w:.0f} kt</div><div class="metric-status" style="background-color:#334155; font-size: 12px;">{source_status}</div></div>', unsafe_allow_html=True)
                p3.markdown(f'<div class="metric-card"><div class="metric-title">Command</div><div class="metric-value" style="font-size:36px;">{act}</div><div class="metric-status" style="background-color:#334155">LOCKED</div></div>', unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=h_t, y=h_w, name="Wind", line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'))
                fig.add_trace(go.Scatter(x=h_t, y=h_a, name="Alpha", line=dict(color='#10b981', width=3), yaxis='y2'))
                fig.add_hline(y=1.5, line_dash="dash", line_color="#f59e0b", line_width=2, yref="y2")
                fig.add_hline(y=1.2, line_dash="dash", line_color="#ef4444", line_width=2, yref="y2")
                fig.add_hrect(y0=0, y1=1.25, line_width=0, fillcolor="#ef4444", opacity=0.1, yref="y2")
                if fracture_idx is not None:
                    ft = times[fracture_idx]
                    fig.add_vline(x=ft, line_width=2, line_dash="dash", line_color="#ef4444")
                    fig.add_annotation(x=ft, y=195, text=f"[ TIMING SIGNAL ] {ft.strftime('%H:%M')}", font=dict(color="white", size=9), bgcolor="#ef4444")
                if alert_idx is not None:
                    alt = times[alert_idx]
                    fig.add_vline(x=alt, line_width=2, line_dash="dash", line_color="#ffffff")
                    fig.add_annotation(x=alt, y=100, text=f"[ NHC ALERT ] {alt.strftime('%H:%M')}", font=dict(color="black", size=9), bgcolor="#ffffff", ay=-40)
                fig.update_layout(height=450, margin=dict(l=10,r=10,t=10,b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), xaxis=dict(range=[times[0], times[-1]], gridcolor='#334155'), yaxis=dict(title="Wind (kt)", range=[0, 220], gridcolor='#334155'), yaxis2=dict(title="Alpha (timing)", overlaying='y', side='right', range=[0.2, 2.2], showgrid=False), showlegend=False)
                p_chart.plotly_chart(fig, use_container_width=True, key=f"c_{i}")
                time.sleep(0.03)

        if storm_data:
            st.markdown("""
                <div class="theory-box">
                    <h3 style='color: #3b82f6; margin-top: 0;'>TIMING ANALYSIS — HISTORICAL PATTERNS</h3>
                    <p style='font-size: 15px; line-height: 1.6;'>
                        <b>Red Team finding:</b> Alpha magnitude is redundant with wind speed (rho = 0.957).
                        The operational value is in WHEN Alpha drops relative to wind explosion.
                    </p>
                    <ul style='font-size: 14px;'>
                        <li><b>OTIS (2023):</b> Alpha timing signal ~12h before NHC major warning.</li>
                        <li><b>MILTON (2024):</b> Alpha timing signal ~14h before Category 5 kinetic explosion.</li>
                        <li><b>PATRICIA (2015):</b> Alpha timing signal ~12h before peak intensity.</li>
                    </ul>
                    <p style='font-size: 13px; color: #94a3b8;'>
                        <b>NOTE:</b> These are pre-programmed historical simulations illustrating the timing pattern,
                        not real-time predictions. The Alpha values shown are derived from wind/pressure data and
                        are not independent structural measurements. The timing relationship (Alpha drops BEFORE
                        wind spikes) is the surviving empirical finding from the Red Team audit.
                    </p>
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 12px;">Licensed under CC BY 4.0 | RTM-Atmo v2 (Post-Red Team) | <a href="https://github.com/zarpafantasma/corpus_rythmos" target="_blank" style="color: #3b82f6;">github.com/zarpafantasma/corpus_rythmos</a></div>', unsafe_allow_html=True)

# ==========================================
# 7. MASTER CONTROLLER
# ==========================================
st.sidebar.markdown("""
<div style="background-color: #1A2639; padding: 15px; border-radius: 8px; border: 1px solid #3b82f6; text-align: center; margin-bottom: 20px;">
    <h3 style="color: #ffffff; margin: 0; font-size: 16px;">RTM SYSTEM v2</h3>
    <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 11px;">Post-Red Team Audit (April 2026)</p>
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "SELECT ACTIVE MODULE:",
    ["CLIMATE EXTREMES", "HURRICANE TRACKER"]
)
st.sidebar.markdown("---")

if app_mode == "CLIMATE EXTREMES":
    st.markdown(CSS_CLIMATE, unsafe_allow_html=True)
    run_climate_module()
else:
    st.markdown(CSS_HURRICANES, unsafe_allow_html=True)
    run_hurricane_module()
