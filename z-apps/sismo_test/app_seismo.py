import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RTM-Seismo | Structural Coherence Monitor",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PREMIUM CSS (Dark Seismic Theme)
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #060A12;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    header[data-testid="stHeader"] { background-color: #060A12 !important; height: 0px; }
    
    [data-testid="stSidebar"] {
        background-color: #0A1122 !important;
        border-right: 1px solid #1E3A5F;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    div[data-testid="stButton"] button {
        background-color: #0A1628 !important;
        color: #38BDF8 !important;
        border: 1px solid #38BDF8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #38BDF8 !important;
        color: #060A12 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div { background-color: #0F1724; border-color: #1E293B; color: white; }
    div[data-baseweb="popover"] > div { background-color: #0F1724 !important; border: 1px solid #1E293B !important; }
    div[role="listbox"] li { color: #FFFFFF !important; background-color: #0F1724 !important; }
    div[role="listbox"] li:hover { background-color: #1E293B !important; color: #38BDF8 !important; }
    
    div[data-testid="stMetric"] {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        padding: 20px;
        border-radius: 8px;
    }
    
    .rtm-info-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        padding: 30px;
        border-radius: 8px;
        margin-top: 25px;
        line-height: 1.6;
    }
    
    .disclaimer-box {
        background-color: #0A1628;
        border: 1px solid #38BDF8;
        border-radius: 8px;
        padding: 12px;
        color: #94A3B8;
        font-size: 13px;
        line-height: 1.5;
    }
    
    .gauge-legend {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        font-size: 0.85em;
    }
    .legend-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 8px;
    }
    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        margin-top: 3px;
        flex-shrink: 0;
    }
    
    .health-card {
        background-color: #0A1122;
        border: 1px solid #1E293B;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MONITORING REGIONS
# ==========================================
REGIONS = {
    "San Andreas Fault Zone":      {"lat": 36.0,  "lon": -120.0, "radius": 500},
    "Western Pacific (Ring of Fire)": {"lat": 35.0,  "lon": 140.0,  "radius": 1000},
    "Chile Subduction Zone":       {"lat": -33.0, "lon": -72.0,  "radius": 800},
    "Anatolian Fault (Turkey)":    {"lat": 39.0,  "lon": 36.0,   "radius": 600},
    "Indonesia / Sunda Arc":       {"lat": -5.0,  "lon": 115.0,  "radius": 1000},
    "Italy (Apennines)":           {"lat": 42.5,  "lon": 13.0,   "radius": 400},
    "Mexico Pacific Coast":        {"lat": 16.5,  "lon": -99.0,  "radius": 600},
    "Himalayan Front":             {"lat": 28.0,  "lon": 85.0,   "radius": 800},
    "Central US (New Madrid)":     {"lat": 36.5,  "lon": -89.5,  "radius": 400},
    "Caribbean Plate":             {"lat": 18.0,  "lon": -70.0,  "radius": 700},
}

RTM_WINDOW = 30  # Rolling window for alpha estimation

# ==========================================
# 4. DATA ENGINE
# ==========================================
@st.cache_data(ttl=600)
def fetch_usgs_data(lat, lon, radius_km, days, min_mag):
    """Fetches earthquake catalog from USGS FDSN API."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "orderby": "time",
        "limit": 5000,
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        events = []
        for f in data.get("features", []):
            p = f["properties"]
            g = f["geometry"]["coordinates"]
            events.append({
                "time": pd.to_datetime(p["time"], unit="ms"),
                "mag": p.get("mag", 0),
                "place": p.get("place", ""),
                "depth": g[2] if len(g) > 2 else 0,
                "lat": g[1],
                "lon": g[0],
                "sig": p.get("sig", 0),
            })
        
        df = pd.DataFrame(events).sort_values("time").reset_index(drop=True)
        return df
    
    except Exception as e:
        st.error(f"USGS API Error: {str(e)}")
        return None


def compute_rtm_alpha(df, window=RTM_WINDOW):
    """
    Core RTM Engine: Computes rolling α from log-log covariance.
    
    L proxy = magnitude (energy release scale)
    T proxy = inter-event time in hours (temporal structure)
    
    α = |cov(log_L, log_T)| / var(log_L)
    
    This is the EXACT same formula used in:
    - app_rtm.py (hurricanes): L=Wind, T=ΔPressure
    - app.py (crypto): L=Volume, T=Spread
    """
    if df is None or len(df) < window + 5:
        return None
    
    # Compute inter-event times (hours)
    df = df.copy()
    df['dt_hours'] = df['time'].diff().dt.total_seconds() / 3600.0
    
    # Filter: remove zero/negative dt and zero magnitudes
    df = df[(df['dt_hours'] > 0.001) & (df['mag'] > 0)].copy()
    
    if len(df) < window + 5:
        return None
    
    df['log_L'] = np.log(df['mag'])
    df['log_T'] = np.log(df['dt_hours'])
    
    # Rolling covariance calculation
    cov = df['log_L'].rolling(window).cov(df['log_T'])
    var = df['log_L'].rolling(window).var()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_alpha = np.abs(cov / var)
    
    raw_alpha = pd.Series(raw_alpha).replace([np.inf, -np.inf], np.nan)
    
    # Smooth with 3-point MA
    df['Alpha'] = raw_alpha.rolling(3, min_periods=1).mean()
    df = df.dropna(subset=['Alpha']).reset_index(drop=True)
    
    # Clip to reasonable range
    df['Alpha'] = df['Alpha'].clip(0.01, 3.0)
    
    return df


def compute_b_value(df, min_mag=1.0):
    """Gutenberg-Richter b-value (maximum likelihood estimator)."""
    mags = df[df['mag'] >= min_mag]['mag'].values
    if len(mags) < 20:
        return None
    m_min = min_mag
    m_mean = np.mean(mags)
    if m_mean <= m_min:
        return None
    return 1.0 / (np.log(10) * (m_mean - m_min))


# ==========================================
# 5. UI HELPERS
# ==========================================
def create_gauge_chart(alpha_value):
    """Seismic coherence gauge — inverted logic vs crypto (low α = danger)."""
    steps = [
        {'range': [0, 0.49], 'color': "rgba(239, 68, 68, 0.30)", 'name': 'FRACTURE'},
        {'range': [0.50, 0.79], 'color': "rgba(249, 115, 22, 0.20)", 'name': 'CRITICAL'},
        {'range': [0.80, 1.19], 'color': "rgba(234, 179, 8, 0.15)", 'name': 'STRESSED'},
        {'range': [1.20, 3.0], 'color': "rgba(34, 197, 94, 0.12)", 'name': 'STABLE'},
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=alpha_value,
        title={'text': "STRUCTURAL COHERENCE (α)", 'font': {'size': 14, 'color': '#94A3B8'}},
        number={'font': {'color': '#FFFFFF'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 2.5], 'tickwidth': 1, 'tickcolor': "#1E293B"},
            'bar': {'color': "#FFFFFF", 'thickness': 0.1},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': steps,
            'threshold': {'line': {'color': "#EF4444", 'width': 3}, 'thickness': 0.75, 'value': 0.5},
        }
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )
    return fig


def apply_premium_layout(fig, height=500):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=20, r=20),
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B')
    return fig


def get_status(alpha):
    if alpha is None or np.isnan(alpha):
        return "NO DATA", "#6B7280"
    if alpha < 0.5:
        return "FRACTURE", "#EF4444"
    if alpha < 0.8:
        return "CRITICAL", "#F97316"
    if alpha < 1.2:
        return "STRESSED", "#EAB308"
    return "STABLE", "#22C55E"


# ==========================================
# 6. SIDEBAR
# ==========================================
st.sidebar.markdown("""
<div style="background-color: #0F172A; padding: 15px; border-radius: 8px; border: 1px solid #1E293B; text-align: center; margin-bottom: 20px;">
    <h3 style="color: #FFFFFF; margin: 0; font-size: 16px; letter-spacing: 2px;">RTM-SEISMO</h3>
    <p style="color: #94A3B8; margin: 5px 0 0; font-size: 11px;">STRUCTURAL COHERENCE MONITOR</p>
</div>
""", unsafe_allow_html=True)

selected_region = st.sidebar.selectbox("MONITORING REGION", list(REGIONS.keys()))
days_window = st.sidebar.selectbox("TIME WINDOW", [7, 14, 30, 60, 90, 180], index=2, format_func=lambda x: f"{x} days")
min_magnitude = st.sidebar.selectbox("MIN MAGNITUDE", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], index=1, format_func=lambda x: f"M ≥ {x}")

st.sidebar.markdown("---")

if st.sidebar.button("FETCH USGS DATA"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color: #94A3B8; font-size: 0.78em; line-height: 1.4; border-left: 2px solid #38BDF8; padding-left: 10px;">
    <b>RTM-SEISMO ENGINE:</b> Treating seismic networks as multiscale topological structures. 
    Major ruptures are structural fractures — not statistical anomalies. 
    α measures the temporal coherence of the magnitude-timing coupling.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="color: #94A3B8; font-size: 0.78em; line-height: 1.4; border-left: 2px solid #F97316; padding-left: 10px; margin-top: 10px;">
    <b>DISCLAIMER:</b> This is an experimental proof of concept based on RTM Theory. 
    It is NOT an official seismological alert system. Do not use for life-safety decisions. 
    Data sourced from USGS Earthquake Hazards Program.
</div>
""", unsafe_allow_html=True)


# ==========================================
# 7. MAIN INTERFACE
# ==========================================
head_l, head_r = st.columns([1, 1.5])
with head_l:
    st.markdown("<h2 style='color: white; margin: 0;'>RTM-SEISMO</h2>", unsafe_allow_html=True)
with head_r:
    st.markdown("""
    <div class="disclaimer-box">
        <b>[ DISCLAIMER ]</b> RTM-SEISMO is an experimental proof of concept based on RTM Theory 
        (Multiscale Temporal Relativity). It is NOT an official seismological alert system. 
        Data from USGS Earthquake Hazards Program API.
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color: #1E293B; margin: 15px 0;'>", unsafe_allow_html=True)

# Info card
st.markdown("""
<div class="rtm-info-card" style="border-left: 4px solid #38BDF8; margin-top: 0; margin-bottom: 20px;">
    <h3 style="color: #FFFFFF; margin-top: 0;">Seismic Structural Early Warning</h3>
    <p style="color: #94A3B8; font-size: 1.05em; margin-bottom: 0;">
        Traditional seismic monitoring tracks <b>kinetic metrics</b> (magnitude, ground acceleration). 
        RTM-Seismo monitors the <b>topological structure</b> of the seismic network by measuring how 
        magnitude and inter-event timing are coupled across scales. When this coupling breaks down 
        (α collapses), the fault network is losing coherence — analogous to how RTM-Atmo detects 
        hurricane intensification 12–18 hours before wind speed spikes, and RTM-Econ detects market 
        crashes before price collapses.
    </p>
</div>
""", unsafe_allow_html=True)

# Fetch data
reg = REGIONS[selected_region]
raw_df = fetch_usgs_data(reg["lat"], reg["lon"], reg["radius"], days_window, min_magnitude)

if raw_df is not None and len(raw_df) > 0:
    # Compute RTM Alpha
    alpha_df = compute_rtm_alpha(raw_df, window=RTM_WINDOW)
    b_val = compute_b_value(raw_df, min_mag=min_magnitude)
    
    # Current values
    current_alpha = alpha_df['Alpha'].iloc[-1] if alpha_df is not None and len(alpha_df) > 0 else None
    status_text, status_color = get_status(current_alpha)
    total_events = len(raw_df)
    max_mag = raw_df['mag'].max()
    last_update = raw_df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')
    
    # ---- GAUGE + METRICS ROW ----
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        if current_alpha is not None:
            st.plotly_chart(create_gauge_chart(current_alpha), use_container_width=True)
        
        # Gauge legend
        st.markdown(f"""
        <div class="gauge-legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(34, 197, 94, 0.5); border: 1px solid #22C55E;"></div>
                <div><b style="color: #22C55E;">STABLE (1.2+):</b> Normal seismic background. Energy distributed across the Gutenberg-Richter spectrum.</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(234, 179, 8, 0.5); border: 1px solid #EAB308;"></div>
                <div><b style="color: #EAB308;">STRESSED (0.8–1.2):</b> Temporal correlations stiffening. The network is accumulating strain.</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(249, 115, 22, 0.5); border: 1px solid #F97316;"></div>
                <div><b style="color: #F97316;">CRITICAL (0.5–0.8):</b> Magnitude-timing coupling breaking down. Structural fracture developing.</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(239, 68, 68, 0.5); border: 1px solid #EF4444;"></div>
                <div><b style="color: #EF4444;">FRACTURE (&lt; 0.5):</b> Total loss of multiscale coherence. Analogous to RTM-Atmo cyclogenesis (α &lt; 1.25).</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status card
        st.markdown(f"""
        <div style="border-left: 4px solid {status_color}; background-color: #0F172A; padding: 15px; border-radius: 4px; margin-top: 15px;">
            <span style="color: {status_color}; font-weight: 600; letter-spacing: 1px;">STATE: {status_text}</span><br>
            <span style="color: #94A3B8; font-size: 0.9em;">Region: {selected_region}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.metric("TOTAL EVENTS", f"{total_events:,}", delta=f"Last {days_window} days", delta_color="off")
        st.metric("MAX MAGNITUDE", f"M {max_mag:.1f}", delta=f"Updated: {last_update}", delta_color="off")
        if b_val is not None:
            st.metric("b-VALUE (Gutenberg-Richter)", f"{b_val:.3f}")
    
    with col2:
        if alpha_df is not None and len(alpha_df) > 5:
            # ---- MAIN CHART: Alpha + Magnitude ----
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Magnitude scatter (secondary y)
            fig.add_trace(go.Scatter(
                x=alpha_df['time'], y=alpha_df['mag'],
                name="MAGNITUDE",
                mode='markers',
                marker=dict(color='#38BDF8', size=4, opacity=0.4),
            ), secondary_y=False)
            
            # Alpha line (primary display)
            fig.add_trace(go.Scatter(
                x=alpha_df['time'], y=alpha_df['Alpha'],
                name="RTM ALPHA (α)",
                line=dict(color='#22C55E', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(34, 197, 94, 0.05)',
            ), secondary_y=True)
            
            # Threshold lines
            fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=2, secondary_y=True)
            fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(249, 115, 22, 0.5)", line_width=1.5, secondary_y=True)
            fig.add_hline(y=1.2, line_dash="dash", line_color="rgba(234, 179, 8, 0.4)", line_width=1, secondary_y=True)
            
            # Danger zone
            fig.add_hrect(y0=0, y1=0.5, line_width=0, fillcolor="rgba(239, 68, 68, 0.08)", secondary_y=True)
            fig.add_hrect(y0=0.5, y1=0.8, line_width=0, fillcolor="rgba(249, 115, 22, 0.04)", secondary_y=True)
            
            # Dummy traces for legend
            fig.add_trace(go.Scatter(x=[None], y=[None], name="FRACTURE (0.5)", line=dict(color="#EF4444", width=2, dash="dash")), secondary_y=True)
            fig.add_trace(go.Scatter(x=[None], y=[None], name="CRITICAL (0.8)", line=dict(color="#F97316", width=1.5, dash="dash")), secondary_y=True)
            
            fig = apply_premium_layout(fig, height=700)
            fig.update_yaxes(title_text="Magnitude", secondary_y=False, range=[0, max_mag + 1])
            fig.update_yaxes(title_text="Alpha (α)", secondary_y=True, range=[0, 2.5])
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough events to compute rolling α. Try increasing the time window or lowering the minimum magnitude.")
    
    # ---- CHART EXPLANATION ----
    st.markdown("""
    <div style="background-color: #0F172A; padding: 20px; border-radius: 10px; border: 1px solid #1E293B; margin-top: 15px;">
        <h4 style="color: #F1F5F9; margin-top: 0; font-size: 15px; text-transform: uppercase;">How to Read the Chart</h4>
        <ul style="color: #94A3B8; font-size: 13px; line-height: 1.8;">
            <li><b style="color: #22C55E;">Green Line (Alpha):</b> Read on the right axis. This is the RTM structural coherence exponent. It measures how well magnitude and inter-event timing are coupled. High α = stable coupling. Low α = structural fracture.</li>
            <li><b style="color: #38BDF8;">Blue Dots (Magnitude):</b> Read on the left axis. Each dot is an individual earthquake. Larger dots = bigger quakes.</li>
            <li><b style="color: #EF4444;">Red Dashed Line (0.5 — Fracture):</b> If the green line drops below this, the seismic network has lost multiscale coherence. This is the seismic equivalent of RTM-Atmo's cyclogenesis threshold (α &lt; 1.25).</li>
            <li><b style="color: #F97316;">Orange Dashed Line (0.8 — Critical):</b> Warning threshold. The temporal structure is degrading.</li>
            <li><b style="color: #EAB308;">Yellow Dashed Line (1.2 — Stress):</b> Transition from stable to stressed. Strain accumulation may be occurring.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---- SIGNIFICANT EVENTS TABLE ----
    st.markdown("#### SIGNIFICANT EVENTS IN WINDOW")
    major_events = raw_df[raw_df['mag'] >= 4.0].sort_values('time', ascending=False).head(15)
    
    if len(major_events) > 0:
        display_df = major_events[['time', 'mag', 'depth', 'place']].copy()
        display_df.columns = ['Time (UTC)', 'Magnitude', 'Depth (km)', 'Location']
        display_df['Time (UTC)'] = display_df['Time (UTC)'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Magnitude'] = display_df['Magnitude'].apply(lambda x: f"{x:.1f}")
        display_df['Depth (km)'] = display_df['Depth (km)'].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No events M ≥ 4.0 in the selected window. This is normal for lower-seismicity regions.")
    
    st.markdown("---")
    
    # ---- RTM THEORY BOX ----
    st.markdown("""
    <div class="rtm-info-card">
        <h3 style="color: #38BDF8; margin-top: 0;">RTM DEEP INSIGHT: STRUCTURAL vs KINETIC SEISMOLOGY</h3>
        <p style="font-size: 15px; line-height: 1.6; color: #94A3B8;">
            Traditional seismology relies on <b>kinetic metrics</b>: magnitude, peak ground acceleration, spectral response. 
            These measure the <em>consequence</em> of a rupture. RTM-Seismo measures <b>structural coherence</b> — the 
            temporal coupling between magnitude and inter-event timing across the network.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 15px;">
            <div>
                <b style="color: #22C55E;">RTM α (STRUCTURAL)</b><br>
                <span style="color: #94A3B8;">Measures how the seismic network distributes energy across scales. 
                When α drops, it means large and small events are decoupling — the fractal 
                structure of the Gutenberg-Richter distribution is breaking down temporally.</span>
            </div>
            <div>
                <b style="color: #A78BFA;">b-VALUE (FREQUENCY-MAGNITUDE)</b><br>
                <span style="color: #94A3B8;">Measures the ratio of small to large events (spatial distribution). 
                Both b-value and α may drop before major events, but α captures <em>temporal coherence loss</em> 
                that b-value misses. They are complementary indicators.</span>
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background-color: #0A1628; border-radius: 6px;">
            <b style="color: #F97316;">CROSS-DOMAIN VALIDATION</b><br>
            <span style="color: #94A3B8; font-size: 13px;">
                The same α-collapse pattern has been validated in:<br>
                • <b>Hurricanes (RTM-Atmo):</b> α &lt; 1.25 precedes Rapid Intensification by 12–18h (Doc 013)<br>
                • <b>Crypto Markets (RTM-Econ):</b> α &gt; 2.0 precedes crashes by minutes to hours (Doc 015)<br>
                • <b>Seismology (RTM-Seismo):</b> α collapse hypothesized to precede major ruptures — <em>to be validated</em>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("No earthquake data returned for this region and time window. Try a larger radius, longer time window, or lower minimum magnitude.")

# ---- FOOTER ----
st.markdown("<hr style='border-color: #1E293B; margin: 15px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 12px; margin-bottom: 5px;">
    This application is licensed under a 
    <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" style="color: #38BDF8; text-decoration: none;">
        Creative Commons Attribution 4.0 International License (CC BY 4.0)
    </a>.
</div>
<div style="text-align: center; color: #94A3B8; font-size: 14px; padding-bottom: 20px;">
    Powered by RTM-Seismo Technology · Data: USGS Earthquake Hazards Program · 
    <a href="https://github.com/zarpafantasma/corpus_rythmos" target="_blank" style="color: #38BDF8; text-decoration: none;">
        github.com/zarpafantasma/corpus_rythmos
    </a>
</div>
""", unsafe_allow_html=True)
