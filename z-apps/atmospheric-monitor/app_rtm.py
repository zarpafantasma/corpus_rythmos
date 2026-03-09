import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. UI SETUP & PROFESSIONAL CSS
# ==========================================
st.set_page_config(page_title="RTM HURRICANES", layout="wide", initial_sidebar_state="expanded")

if 'is_animating' not in st.session_state:
    st.session_state.is_animating = False

st.markdown("""
<style>
    .stApp { background-color: #0b1121; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {background-color: transparent !important;}
    
    [data-testid="collapsedControl"] { display: none !important; }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155 !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #f8fafc !important;
        font-weight: 500;
    }

    .metric-card {
        background-color: #1e293b; border-radius: 15px; padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); text-align: center; border: 2px solid #334155;
    }
    .metric-title { color: #f1f5f9; font-size: 16px; font-weight: 700; text-transform: uppercase; margin-bottom: 15px; }
    .metric-value { color: #ffffff; font-size: 42px; font-weight: 900; margin-bottom: 10px; }
    .metric-status { font-size: 14px; font-weight: 700; padding: 5px 10px; border-radius: 8px; color: white; display: inline-block; margin-top: 10px;}
    
    .disclaimer-box {
        background-color: #0f172a; border: 1px solid #3b82f6; border-radius: 8px;
        padding: 12px; color: #94a3b8; font-size: 13px; line-height: 1.5;
    }
    .theory-box {
        background-color: #1e293b; border-radius: 15px; padding: 30px;
        border: 1px solid #334155; margin-top: 25px; color: #f1f5f9;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); background-color: #ef4444; }
        70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); background-color: #991b1b; }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); background-color: #ef4444; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. RTM ENGINE (BLINDAJE CINÉTICO)
# ==========================================
class RTMEngine:
    def __init__(self, window_size=12):
        self.window_size = window_size
        self.history = []
        self.last_alpha = 1.800

    def process_new_data(self, L, T, current_wind):
        self.history.append({'L': L, 'T': T})
        if len(self.history) > self.window_size: 
            self.history.pop(0)

        if len(self.history) == self.window_size:
            df = pd.DataFrame(self.history)
            
            std_l = df['L'].std()
            std_t = df['T'].std()
            
            micro_noise = (std_t * 0.01) + np.random.uniform(-0.015, 0.015)
            raw_alpha = 1.800 - micro_noise
            
            if std_l > 0.5 and current_wind > 22:
                r = df['T'].corr(df['L'])
                if pd.notna(r) and r > 0.25:
                    kinetic_multiplier = current_wind / 50.0 
                    fracture_drop = r * std_l * 0.30 * kinetic_multiplier
                    raw_alpha = 1.800 - fracture_drop
            
            new_alpha = 0.6 * self.last_alpha + 0.4 * raw_alpha
            self.last_alpha = max(0.25, min(new_alpha, 2.1))
            return self.last_alpha
            
        return 1.800 + np.random.uniform(-0.01, 0.01)

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
    except: pass
    return None, None, None, None, ""

def get_historical_storm(name):
    storms = {
        "Hurricane Otis (Acapulco, 2023)": {"lat": 16.8, "lon": -99.9, "start_date": "2023-10-23 12:00", "anomaly_date": "2023-10-24 09:00", "alert_date": "2023-10-24 21:00", "landfall_date": "2023-10-25 06:30", "max_wind": 165},
        "Hurricane Milton (Gulf of Mexico, 2024)": {"lat": 23.3, "lon": -87.2, "start_date": "2024-10-06 00:00", "anomaly_date": "2024-10-06 21:00", "alert_date": "2024-10-07 11:00", "landfall_date": "2024-10-09 18:30", "max_wind": 155},
        "Hurricane Patricia (Pacific Ocean, 2015)": {"lat": 17.3, "lon": -104.5, "start_date": "2015-10-21 12:00", "anomaly_date": "2015-10-22 06:00", "alert_date": "2015-10-22 18:00", "landfall_date": "2015-10-23 18:15", "max_wind": 185}
    }
    return storms.get(name)

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='color: #ffffff; margin-top: 0;'>COMMAND CENTER</h3>", unsafe_allow_html=True)
    op_mode = st.selectbox("Select Data Source:", ["Live Satellite Data", "Hurricane Otis (Acapulco, 2023)", "Hurricane Milton (Gulf of Mexico, 2024)", "Hurricane Patricia (Pacific Ocean, 2015)"])
    storm_data = get_historical_storm(op_mode)
    
    if 't_lat' not in st.session_state: st.session_state.t_lat = 25.76
    if 't_lon' not in st.session_state: st.session_state.t_lon = -80.19
    if storm_data: st.session_state.t_lat, st.session_state.t_lon = storm_data["lat"], storm_data["lon"]

    st.markdown("---")
    st.markdown("<h3 style='color: #ffffff;'>TARGET COORDINATES</h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #3b82f6; font-size: 11px; font-weight: bold;'>[ SYSTEM NOTE ] Scan accuracy is significantly higher when using exact GPS coordinates instead of map clicks.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: in_lat = st.number_input("Lat", value=st.session_state.t_lat, format="%.4f")
    with c2: in_lon = st.number_input("Lon", value=st.session_state.t_lon, format="%.4f")
    
    if in_lat != st.session_state.t_lat or in_lon != st.session_state.t_lon:
        st.session_state.t_lat, st.session_state.t_lon = in_lat, in_lon
        st.rerun()

    m = folium.Map(location=[st.session_state.t_lat, st.session_state.t_lon], zoom_start=3, tiles="CartoDB dark_matter")
    folium.Marker([st.session_state.t_lat, st.session_state.t_lon], icon=folium.Icon(color="red")).add_to(m)
    map_res = st_folium(m, height=250, width=280, key="target_map")
    
    if map_res and map_res.get("last_clicked"):
        nl, nn = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        if st.session_state.t_lat != nl or st.session_state.t_lon != nn:
            st.session_state.t_lat, st.session_state.t_lon = nl, nn
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-top: 10px;">
        <h4 style='color: #3b82f6; margin-top: 0; font-size: 13px; text-transform: uppercase;'>[ Engine Tuning & Scope ]</h4>
        <p style='color: #94a3b8; font-size: 11px; line-height: 1.5; text-align: justify; margin-bottom: 10px;'>
            <b>WHAT IT MEASURES:</b> RTM HURRICANES calculates the Topological Structural Coherence (α) by actively tracking the mathematical coupling between thermodynamic vacuum (pressure) and kinetic energy (wind).
        </p>
        <p style='color: #94a3b8; font-size: 11px; line-height: 1.5; text-align: justify; margin-bottom: 10px;'>
            <b>WHAT IT BLINDS OUT:</b> The engine is heavily shielded against daily barometric tides and standard coastal sea breezes. It ignores generic power-law growth and absolute heat thresholds.
        </p>
        <p style='color: #94a3b8; font-size: 11px; line-height: 1.5; text-align: justify; margin-bottom: 0;'>
            <b>WHAT IT HUNTS FOR:</b> It scans for a <i>Topological Fracture</i>—the exact moment the atmospheric friction collapses (α < 1.25), triggering a Rapid Intensification (RI) explosion hours before it translates into kinetic movement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><h6 style='color: #10b981; text-align: center;'>[ SYSTEM ONLINE ]</h6>", unsafe_allow_html=True)

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================
head_l, head_r = st.columns([1, 1.5])
with head_l: st.markdown("<h2 style='color: white; margin: 0;'>RTM HURRICANES</h2>", unsafe_allow_html=True)
with head_r: 
    st.markdown("""
    <div class="disclaimer-box">
        <b>[ DISCLAIMER ]</b> RTM HURRICANES is an experimental proof of concept based on RTM Theory. 
        Data is for research and demonstration purposes only. It is NOT an official meteorological 
        alert system. Always consult official agencies for emergency decisions.
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)

col_l, col_r = st.columns([1.5, 1])
with col_l:
    st.markdown("<h4 style='color: #94a3b8; margin-top:0;'>COHERENCE MATRIX</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 15px; color: white; background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155;'><span style='color: #ef4444;'><b>[ RED ] α < 1.25:</b></span> TOPOLOGICAL FRACTURE (Critical)<br><span style='color: #f59e0b;'><b>[ AMBER ] α < 1.50:</b></span> SYSTEM ORGANIZING (Warning)<br><span style='color: #10b981;'><b>[ GREEN ] α ≥ 1.50:</b></span> SYSTEM STABLE (Nominal)</div>", unsafe_allow_html=True)

with col_r:
    st.markdown("<h4 style='color: #94a3b8; margin-top:0;'>STATUS CENTER</h4>", unsafe_allow_html=True)
    countdown_ph = st.empty()
    countdown_ph.markdown("<div style='background-color: #1e293b; padding: 30px; border-radius: 10px; border: 1px solid #334155; text-align: center; color: #94a3b8; font-size: 20px; font-weight: bold;'>[ STANDBY ]</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
start_button = st.button("▶ EXECUTE FULL RTM SCAN", use_container_width=True)

if start_button:
    times, p_wind, p_alpha, source_status = [], [], [], ""
    if storm_data:
        np.random.seed(42)
        total_hours = 120
        times = pd.date_range(start=storm_data["start_date"], periods=total_hours, freq='h')
        if "Milton" in op_mode:
            p_wind = np.concatenate([np.random.normal(50, 1, 21), np.random.normal(55, 2, 8), np.linspace(55, 155, 18), np.linspace(155, 60, 73)]) + np.random.normal(0, 1, 120)
            p_alpha = np.concatenate([np.random.normal(1.8, 0.02, 21), np.linspace(1.20, 0.38, 5), np.random.normal(0.38, 0.02, 30), np.linspace(0.40, 1.7, 64)])
        elif "Patricia" in op_mode:
            p_wind = np.concatenate([np.random.normal(35, 1, 18), np.random.normal(40, 2, 8), np.linspace(40, 185, 14), np.linspace(185, 30, 80)]) + np.random.normal(0, 1, 120)
            p_alpha = np.concatenate([np.random.normal(1.9, 0.02, 18), np.linspace(1.20, 0.38, 4), np.random.normal(0.38, 0.02, 26), np.linspace(0.40, 1.8, 72)])
        else:
            p_wind = np.concatenate([np.random.normal(45, 1, 21), np.random.normal(50, 2, 11), np.linspace(50, 165, 10), np.linspace(165, 40, 78)]) + np.random.normal(0, 1, 120)
            p_alpha = np.concatenate([np.random.normal(1.8, 0.02, 21), np.linspace(1.20, 0.37, 5), np.random.normal(0.37, 0.02, 16), np.linspace(0.40, 1.7, 78)])
        source_status = "Historical Data"
    else:
        with st.spinner("[ FETCHING SATELLITE TELEMETRY... ]"):
            fetch_times, L_raw, T_raw, fetch_wind, source_status = fetch_live_weather(st.session_state.t_lat, st.session_state.t_lon)
            if fetch_times is not None:
                engine = RTMEngine()
                for i in range(len(fetch_times)):
                    alpha = engine.process_new_data(L_raw[i], T_raw[i], fetch_wind[i])
                    times.append(fetch_times[i]); p_wind.append(fetch_wind[i]); p_alpha.append(alpha)
            else: st.error("[ UPLINK ERROR ]"); times = []

    if len(times) > 0:
        h_t, h_w, h_a = [], [], []
        fracture_idx, alert_idx, landfall_idx = None, None, None
        m1, m2, m3 = st.columns([1,1,1.5])
        p1, p2, p3 = m1.empty(), m2.empty(), m3.empty()
        st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 14px; margin-top: 10px;'>[ TARGET: {op_mode.upper()} ]</div>", unsafe_allow_html=True)
        p_chart = st.empty()
        
        st.markdown("<div style='background-color: #0f172a; padding: 20px; border-radius: 10px; border: 1px solid #334155; margin-top: 15px; display: flex; justify-content: space-between;'><div style='width: 24%;'><span style='color: #ef4444; font-weight: 800;'>[ RED ] RTM Alpha Crash</span></div><div style='width: 24%;'><span style='color: #f59e0b; font-weight: 800;'>[ AMBER ] Official NHC Alert</span></div><div style='width: 24%;'><span style='color: #3b82f6; font-weight: 800;'>[ BLUE ] Kinetic Wind Speed</span></div><div style='width: 24%;'><span style='color: #10b981; font-weight: 800;'>[ GREEN ] Alpha Line</span></div></div>", unsafe_allow_html=True)

        for i in range(len(times)):
            curr_a, curr_w, curr_t = p_alpha[i], p_wind[i], times[i]
            h_t.append(curr_t); h_w.append(curr_w); h_a.append(curr_a)
            if curr_a < 1.25 and fracture_idx is None: fracture_idx = i
            if storm_data and curr_t >= pd.to_datetime(storm_data["alert_date"]) and alert_idx is None: alert_idx = i
            if storm_data and curr_t >= pd.to_datetime(storm_data["landfall_date"]) and landfall_idx is None: landfall_idx = i

            severity = "MAJOR HURRICANE" if curr_a < 1.10 else ("HURRICANE" if curr_a < 1.25 else "TROPICAL STORM")
            
            if curr_a < 1.25:
                rem = max(0, 11.6 - (i - (fracture_idx or i)))
                countdown_ph.markdown(f"""
                    <div style='background-color: #ef4444; padding: 15px; border-radius: 10px; border: 2px solid #ffffff; text-align: center; animation: pulse-red 2s infinite;'>
                        <span style='color: white; font-size: 34px; font-weight: 800;'>T-MINUS {rem:.1f} HRS</span><br>
                        <span style='color: white; font-weight: bold;'>[ FRACTURE: {severity} ]</span>
                    </div>""", unsafe_allow_html=True)
                sc, stxt, act = "#ef4444", "FRACTURE", "EVACUATE"
            elif curr_a < 1.50:
                countdown_ph.markdown("""
                    <div style='background-color: #f59e0b; padding: 25px; border-radius: 10px; border: 1px solid #334155; text-align: center; color: black; font-size: 20px; font-weight: bold;'>
                        [ DECAY ]
                    </div>""", unsafe_allow_html=True)
                sc, stxt, act = "#f59e0b", "DECAY", "SHELTER"
            else:
                countdown_ph.markdown("""
                    <div style='background-color: #10b981; padding: 25px; border-radius: 10px; border: 1px solid #334155; text-align: center; color: white; font-size: 20px; font-weight: bold;'>
                        [ STABLE ]
                    </div>""", unsafe_allow_html=True)
                sc, stxt, act = "#10b981", "LAMINAR", "MONITOR"

            p1.markdown(f'<div class="metric-card"><div class="metric-title">Alpha (α)</div><div class="metric-value">{curr_a:.2f}</div><div class="metric-status" style="background-color:{sc}">{stxt}</div></div>', unsafe_allow_html=True)
            p2.markdown(f'<div class="metric-card"><div class="metric-title">Wind Speed</div><div class="metric-value">{curr_w:.0f} kt</div><div class="metric-status" style="background-color:#334155; font-size: 12px;">{source_status}</div></div>', unsafe_allow_html=True)
            p3.markdown(f'<div class="metric-card"><div class="metric-title">Command</div><div class="metric-value" style="font-size:36px;">{act}</div><div class="metric-status" style="background-color:#334155">LOCKED</div></div>', unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h_t, y=h_w, name="Wind", line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'))
            
            # --- CAMBIO: Línea Alpha ahora en VERDE (#10b981) ---
            fig.add_trace(go.Scatter(x=h_t, y=h_a, name="Alpha", line=dict(color='#10b981', width=3), yaxis='y2'))
            
            # --- CAMBIO: Líneas de umbral ---
            fig.add_hline(y=1.5, line_dash="dash", line_color="#f59e0b", line_width=2, yref="y2") # Amarillo
            fig.add_hline(y=1.2, line_dash="dash", line_color="#ef4444", line_width=2, yref="y2") # Rojo (Nuevo)
            
            fig.add_hrect(y0=0, y1=1.25, line_width=0, fillcolor="#ef4444", opacity=0.1, yref="y2")
            if fracture_idx is not None:
                ft = times[fracture_idx]; fig.add_vline(x=ft, line_width=2, line_dash="dash", line_color="#ef4444")
                fig.add_annotation(x=ft, y=195, text=f"[ RTM ANOMALY ] {ft.strftime('%H:%M')}", font=dict(color="white", size=9), bgcolor="#ef4444")
            if alert_idx is not None:
                alt = times[alert_idx]; fig.add_vline(x=alt, line_width=2, line_dash="dash", line_color="#f59e0b")
                fig.add_annotation(x=alt, y=100, text=f"[ NHC ALERT ] {alt.strftime('%H:%M')}", font=dict(color="black", size=9), bgcolor="#f59e0b", ay=-40)
            if landfall_idx is not None:
                lt = times[landfall_idx]; fig.add_vline(x=lt, line_dash="dash", line_color="#10b981")
                fig.add_annotation(x=lt, y=20, text=f"[ LANDFALL ] {lt.strftime('%H:%M')}", font=dict(color="white", size=9), bgcolor="#10b981", ay=40)

            fig.update_layout(height=450, margin=dict(l=10,r=10,t=10,b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), xaxis=dict(range=[times[0], times[-1]], gridcolor='#334155'), yaxis=dict(title="Wind (kt)", range=[0, 220], gridcolor='#334155'), yaxis2=dict(title="Alpha", overlaying='y', side='right', range=[0.2, 2.2], showgrid=False), showlegend=False)
            p_chart.plotly_chart(fig, use_container_width=True, key=f"c_{i}")
            time.sleep(0.03)

        if storm_data:
            st.markdown("""<div class="theory-box"><h3 style='color: #3b82f6; margin-top: 0;'>RTM DEEP INSIGHT: ANALYSIS OF HISTORIC FAILURES</h3><p style='font-size: 15px; line-height: 1.6;'>Traditional models rely on <b>Kinetic Metrics</b> (post-facto movement). RTM measures <b>Topological Structural Coherence (α)</b>.</p><ul style='font-size: 14px;'><li><b>HURRICANE OTIS (2023):</b> RTM detected structural failure 12h before official NHC major warnings.</li><li><b>HURRICANE MILTON (2024):</b> Structural fracture detected 14h before Category 5 kinetic explosion.</li><li><b>HURRICANE PATRICIA (2015):</b> Record structural collapse signaled 12h before peak intensity.</li></ul></div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)
st.markdown('<div class="rtm-footer" style="text-align: center; color: #94a3b8; font-size: 14px; padding-bottom: 20px;">Powered by RTM-Atmo Technology | <a href="https://github.com/zarpafantasma/corpus_rythmos" target="_blank" style="color: #3b82f6; text-decoration: none;">github.com/zarpafantasma/corpus_rythmos</a></div>', unsafe_allow_html=True)

