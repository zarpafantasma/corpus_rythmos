"""
RTM ATMOSPHERIC STRUCTURAL RADAR v3
====================================
Built from scratch post-Red Team audit.
Shows ONLY what survived adversarial testing.

Headline: Tornado (d=0.96, α subsumes VEL)
Novel:    Multi-Scale Coherence (σ crash=0.03 vs control=0.31)
Calib:    Seismology (α=1.007, normal faults α=0.865)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, re, math, json
import folium
from streamlit_folium import st_folium

def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx = np.sum((x - mx)**2)
    return np.sum((x - mx)*(y - my))/vx if vx > 0 else 0.0

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════
st.set_page_config(page_title="RTM Atmospheric Radar", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
.stApp{background:#06090f;color:#c9d1d9;font-family:'Inter',sans-serif}
header[data-testid="stHeader"]{background:#06090f!important;height:0}
[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #21262d}
[data-testid="stSidebar"] *{color:#c9d1d9!important}
div[data-testid="stButton"] button{background:#161b22!important;color:#58a6ff!important;border:1px solid #30363d!important;
    font-family:'JetBrains Mono',monospace!important;font-weight:600!important;letter-spacing:.5px;transition:.2s}
div[data-testid="stButton"] button:hover{background:#58a6ff!important;color:#06090f!important}
div[data-testid="stMetric"]{background:#0d1117;border:1px solid #21262d;padding:16px;border-radius:6px}
div[data-testid="stMetric"] label{color:#8b949e!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:1px!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f0f6fc!important;font-family:'JetBrains Mono',monospace!important}
h1,h2,h3{font-family:'JetBrains Mono',monospace!important;font-weight:600!important;color:#f0f6fc!important;letter-spacing:.5px}
hr{border-color:#21262d}
.stSelectbox div[data-baseweb="select"]>div{background:#161b22;border-color:#30363d;color:#c9d1d9}
div[data-baseweb="popover"]>div{background:#161b22!important;border:1px solid #30363d!important}
div[role="listbox"] li{color:#c9d1d9!important;background:#161b22!important}
div[role="listbox"] li:hover{background:#21262d!important;color:#58a6ff!important}
.card{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:20px;margin:8px 0}
.card-accent{border-left:3px solid #58a6ff}
.card-warn{border-left:3px solid #d29922}
.card-danger{border-left:3px solid #f85149}
.card-success{border-left:3px solid #3fb950}
.mono{font-family:'JetBrains Mono',monospace;font-size:13px}
.tag{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;letter-spacing:.5px}
.tag-novel{background:#1f3a1f;color:#3fb950;border:1px solid #238636}
.tag-convergent{background:#1a2332;color:#58a6ff;border:1px solid #1f6feb}
.tag-failed{background:#3d1214;color:#f85149;border:1px solid #da3633}
.tag-timing{background:#2d2000;color:#d29922;border:1px solid #9e6a03}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════
# DATA: TORNADO (NWS API)
# ══════════════════════════════════════
@st.cache_data(ttl=120)
def fetch_tornado_warnings():
    headers = {'User-Agent':'RTM-Atmospheric-Radar/3.0 (academic; github.com/zarpafantasma/corpus_rythmos)','Accept':'application/geo+json'}
    warnings = []
    try:
        r = requests.get("https://api.weather.gov/alerts/active?event=Tornado%20Warning&status=actual", headers=headers, timeout=10)
        if r.status_code == 200:
            for feat in r.json().get('features', []):
                p = feat.get('properties', {})
                g = feat.get('geometry', {})
                rot = _extract_rotation(p.get('description','') + ' ' + p.get('headline',''))
                cell_km = _cell_size(g)
                alpha = _rtm_alpha(rot, cell_km)
                sev = _classify(alpha)
                warnings.append({'headline':p.get('headline',''),'area':p.get('areaDesc','')[:60],
                    'onset':p.get('onset',''),'expires':p.get('expires',''),
                    'geometry':g,'rot_kt':rot,'cell_km':cell_km,'alpha':alpha,
                    'is_tor':alpha>0.74,'sev_label':sev[0],'sev_color':sev[1],'sev_text':sev[2]})
    except: pass
    watches = []
    try:
        r2 = requests.get("https://api.weather.gov/alerts/active?event=Tornado%20Watch&status=actual", headers=headers, timeout=10)
        if r2.status_code == 200:
            for feat in r2.json().get('features', []):
                p = feat.get('properties', {})
                watches.append({'area':p.get('areaDesc','')[:80],'expires':p.get('expires','')})
    except: pass
    return warnings, watches

def _extract_rotation(text):
    t = text.upper()
    for p in [r'(\d+)\s*(?:KT|KNOTS?)',r'ROTATION\s*(?:OF\s*)?(\d+)',r'(\d+)\s*MPH']:
        m = re.search(p, t)
        if m:
            v = float(m.group(1))
            return v * 0.869 if 'MPH' in p else v
    return 45.0

def _cell_size(geo):
    if not geo or geo.get('type') not in ['Polygon','MultiPolygon']: return 20.0
    try:
        coords = geo['coordinates'][0] if geo['type']=='Polygon' else geo['coordinates'][0][0]
        if len(coords)<3: return 20.0
        lons, lats = [c[0] for c in coords], [c[1] for c in coords]
        klon = 111.32 * math.cos(math.radians((max(lats)+min(lats))/2))
        return max(5, min(200, math.sqrt(((max(lons)-min(lons))*klon)**2 + ((max(lats)-min(lats))*111.32)**2)))
    except: return 20.0

def _rtm_alpha(rot, cell):
    try: return round(max(0.1,min(3.0,math.log10(max(rot,1))/math.log10(max(cell,2)))),3)
    except: return 0.5

def _classify(a):
    if a > 1.2: return ("SIGNIFICANT","#f85149","High structural coherence — strong tornado signature")
    if a > 0.74: return ("CONFIRMED","#d29922","Above TorNet threshold (AUC=0.751, d=0.96)")
    if a > 0.45: return ("MARGINAL","#8b949e","Below threshold — weak rotation")
    return ("LOW","#3fb950","Low α — unlikely tornado")

# ══════════════════════════════════════
# DATA: MULTI-SCALE COHERENCE (Open-Meteo)
# ══════════════════════════════════════
@st.cache_data(ttl=3600)
def fetch_coherence(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,surface_pressure&past_days=7&forecast_days=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return None
        d = r.json()
        df = pd.DataFrame({'Date':pd.to_datetime(d['hourly']['time']),
            'Wind':d['hourly']['wind_speed_10m'],'Pressure':d['hourly']['surface_pressure']})
        df['Wind'] = np.where(df['Wind']<1,1,df['Wind'])
        df['PressVol'] = df['Pressure'].diff().abs().fillna(0.1)
        df['PressVol'] = np.where(df['PressVol']<0.01,0.01,df['PressVol'])
        scales = [1,3,6,12]
        scale_a = {}
        for s in scales:
            lw = np.log(df['Wind'].rolling(s,min_periods=1).mean())
            lv = np.log(df['PressVol'].rolling(s,min_periods=1).mean())
            alphas = []
            for i in range(24, len(df)):
                w, v = lw.iloc[i-24:i].values, lv.iloc[i-24:i].values
                mask = np.isfinite(w) & np.isfinite(v)
                alphas.append(abs(_slope(w[mask],v[mask])) if mask.sum()>10 else np.nan)
            scale_a[s] = [np.nan]*24 + alphas
        sigma = []
        for i in range(len(df)):
            vals = [scale_a[s][i] for s in scales if i<len(scale_a[s]) and not np.isnan(scale_a[s][i])]
            sigma.append(np.std(vals) if len(vals)>=3 else np.nan)
        df['Sigma'] = sigma
        for s in scales: df[f'A_{s}h'] = scale_a[s]
        return df.dropna(subset=['Sigma'])
    except: return None

# ══════════════════════════════════════
# CHARTS
# ══════════════════════════════════════
def dark_layout(fig, h=400):
    fig.update_layout(template="plotly_dark",height=h,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=40,b=30,l=20,r=20),
        font=dict(family="JetBrains Mono, monospace",color="#c9d1d9",size=11),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
    fig.update_xaxes(showgrid=True,gridwidth=1,gridcolor='#161b22')
    fig.update_yaxes(showgrid=True,gridwidth=1,gridcolor='#161b22')
    return fig

def sigma_gauge(val):
    fig = go.Figure(go.Indicator(mode="gauge+number+delta",value=val,
        delta={'reference':0.15,'increasing':{'color':'#3fb950'},'decreasing':{'color':'#f85149'}},
        number={'font':{'color':'#f0f6fc','family':'JetBrains Mono'},'valueformat':'.3f'},
        title={'text':"CROSS-SCALE σ",'font':{'size':11,'color':'#8b949e','family':'JetBrains Mono'}},
        gauge={'axis':{'range':[0,0.4],'tickcolor':'#30363d','tickfont':{'size':9}},
            'bar':{'color':'#f0f6fc','thickness':0.08},'bgcolor':'rgba(0,0,0,0)','borderwidth':0,
            'steps':[{'range':[0,0.05],'color':'rgba(248,81,73,0.2)'},
                     {'range':[0.05,0.15],'color':'rgba(210,153,34,0.15)'},
                     {'range':[0.15,0.4],'color':'rgba(63,185,80,0.1)'}],
            'threshold':{'line':{'color':'#f85149','width':2},'thickness':0.75,'value':0.05}}))
    fig.update_layout(height=260,margin=dict(l=20,r=20,t=40,b=10),paper_bgcolor="rgba(0,0,0,0)",
        font={'family':'JetBrains Mono'})
    return fig

def tornado_map(warnings, watches):
    m = folium.Map(location=[38,-97],zoom_start=4,tiles="CartoDB dark_matter")
    for w in warnings:
        g = w['geometry']
        if g and g.get('type')=='Polygon':
            coords = [[c[1],c[0]] for c in g['coordinates'][0]]
            folium.Polygon(locations=coords,color=w['sev_color'],fill=True,fill_color=w['sev_color'],
                fill_opacity=0.3,weight=2,
                popup=f"<b>{w['sev_label']}</b><br>α={w['alpha']:.3f}<br>{w['area']}<br>Rot:{w['rot_kt']:.0f}kt | Cell:{w['cell_km']:.0f}km").add_to(m)
            cx = sum(c[1] for c in g['coordinates'][0])/len(g['coordinates'][0])
            cy = sum(c[0] for c in g['coordinates'][0])/len(g['coordinates'][0])
            folium.CircleMarker([cx,cy],radius=7 if w['is_tor'] else 4,color=w['sev_color'],
                fill=True,fill_color=w['sev_color'],fill_opacity=0.9).add_to(m)
    return m

# ══════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════
st.sidebar.markdown("""<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;text-align:center;margin-bottom:16px">
<span style="font-family:'JetBrains Mono';font-size:14px;font-weight:700;color:#f0f6fc">RTM ATMOSPHERIC RADAR</span><br>
<span style="font-family:'JetBrains Mono';font-size:10px;color:#8b949e">v3.0 — POST-RED TEAM BUILD</span></div>""",unsafe_allow_html=True)

module = st.sidebar.radio("MODULE",[
    "TORNADO VORTEX RADAR",
    "MULTI-SCALE COHERENCE",
    "SEISMOLOGY REFERENCE",
    "RED TEAM FINDINGS"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="color:#8b949e;font-size:11px;line-height:1.5;font-family:'JetBrains Mono',monospace;border-left:2px solid #30363d;padding-left:10px">
<b>DATA SOURCES</b><br>
Tornado: api.weather.gov (NWS)<br>
Coherence: api.open-meteo.com<br>
Seismology: Published catalogs<br><br>
<b>AUDIT</b><br>
Red Team: Claude Opus 4.6<br>
Date: April-May 2026<br>
Corpus score: 68%</div>""",unsafe_allow_html=True)

# ══════════════════════════════════════
# MODULE 1: TORNADO
# ══════════════════════════════════════
if module == "TORNADO VORTEX RADAR":
    st.markdown("## TORNADO VORTEX RADAR")

    st.markdown("""<div class="card card-success"><span class="tag tag-novel">CROWN JEWEL</span>
    <span style="color:#c9d1d9;font-size:13px;margin-left:8px">Strongest finding in the entire RTM corpus</span>
    <p class="mono" style="color:#8b949e;margin:10px 0 0 0">
    TorNet MIT (1,105 events): d = 0.96 · CV AUC = 0.751 · α subsumes VEL (ΔAUC = 0.000)<br>
    α predicts EF intensity within confirmed TOR (ρ = +0.446, n=435) · Optimal: α + KDP (AUC = 0.769)<br>
    Circularity 91% broken: structural-only radar (KDP+DBZ+RHOHV) achieves AUC = 0.698 without velocity
    </p></div>""",unsafe_allow_html=True)

    st.markdown("""<div class="card card-warn"><span class="tag tag-timing">DATA NOTE</span>
    <span class="mono" style="color:#8b949e;margin-left:8px">
    α_proxy = log₁₀(VEL_est) / log₁₀(L_polygon). Approximated from NWS alert text, NOT direct dual-pol radar.
    Treat as indicative. TorNet threshold: α > 0.74 → TOR class.</span></div>""",unsafe_allow_html=True)

    col_ref, col_btn = st.columns([2,1])
    with col_ref:
        st.markdown("""<div class="card"><table style="width:100%;font-family:'JetBrains Mono';font-size:12px;color:#c9d1d9">
        <tr style="color:#8b949e"><td>CLASS</td><td>α RANGE</td><td>MEANING</td></tr>
        <tr><td><span style="color:#f85149">■</span> SIGNIFICANT</td><td>> 1.2</td><td>Strong tornado signature</td></tr>
        <tr><td><span style="color:#d29922">■</span> CONFIRMED</td><td>0.74 – 1.2</td><td>Above TorNet threshold</td></tr>
        <tr><td><span style="color:#8b949e">■</span> MARGINAL</td><td>0.45 – 0.74</td><td>Weak rotation</td></tr>
        <tr><td><span style="color:#3fb950">■</span> LOW</td><td>< 0.45</td><td>Unlikely tornado</td></tr>
        </table></div>""",unsafe_allow_html=True)
    with col_btn:
        st.write(""); st.write("")
        if st.button("REFRESH NWS DATA",use_container_width=True):
            st.cache_data.clear(); st.rerun()

    with st.spinner("Fetching NWS active alerts..."):
        warnings, watches = fetch_tornado_warnings()

    # Metrics
    m1,m2,m3,m4 = st.columns(4)
    n_tor = sum(1 for w in warnings if w['is_tor'])
    mean_a = np.mean([w['alpha'] for w in warnings]) if warnings else 0
    max_a = max([w['alpha'] for w in warnings]) if warnings else 0
    m1.metric("ACTIVE WARNINGS",len(warnings),f"{len(watches)} watches")
    m2.metric("ABOVE THRESHOLD",n_tor,"α > 0.74")
    m3.metric("MEAN α PROXY",f"{mean_a:.3f}")
    m4.metric("MAX α PROXY",f"{max_a:.3f}")

    c_map, c_table = st.columns([1.6,1])
    with c_map:
        m = tornado_map(warnings, watches)
        st_folium(m, height=480, use_container_width=True, key="tor_map")
    with c_table:
        if warnings:
            rows = [{'Area':w['area'],'α':w['alpha'],'Rot(kt)':f"{w['rot_kt']:.0f}",
                     'L(km)':f"{w['cell_km']:.0f}",'Class':w['sev_label']}
                    for w in sorted(warnings,key=lambda x:x['alpha'],reverse=True)]
            st.dataframe(pd.DataFrame(rows),use_container_width=True,height=440)
        else:
            st.markdown("""<div class="card card-success" style="text-align:center;padding:80px 20px">
            <span style="font-size:28px;color:#3fb950;font-family:'JetBrains Mono'">ALL CLEAR</span><br>
            <span class="mono" style="color:#8b949e">No active tornado warnings in CONUS</span></div>""",unsafe_allow_html=True)

    # Distribution if enough events
    if len(warnings) >= 3:
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=[w['alpha'] for w in warnings],nbinsx=15,marker_color='#58a6ff',opacity=0.8))
        fig_d.add_vline(x=0.74,line_dash="dash",line_color="#f85149",annotation_text="TOR threshold")
        fig_d = dark_layout(fig_d, 280)
        fig_d.update_layout(title="α DISTRIBUTION — ACTIVE EVENTS vs TorNet THRESHOLD")
        st.plotly_chart(fig_d, use_container_width=True)

# ══════════════════════════════════════
# MODULE 2: MULTI-SCALE COHERENCE
# ══════════════════════════════════════
elif module == "MULTI-SCALE COHERENCE":
    st.markdown("## MULTI-SCALE COHERENCE")

    st.markdown("""<div class="card card-success"><span class="tag tag-novel">RED TEAM SURVIVOR</span>
    <span style="color:#c9d1d9;font-size:13px;margin-left:8px">The only genuinely novel atmospheric metric</span>
    <p class="mono" style="color:#8b949e;margin:10px 0 0 0">
    Cross-domain finding: during phase transitions (BTC crashes, atmospheric crises), all temporal scales<br>
    lock simultaneously (σ → 0.03). In normal conditions, scales operate independently (σ → 0.31).<br>
    This measures something no standard meteorological tool measures: scale-invariance of α.
    </p></div>""",unsafe_allow_html=True)

    ZONES = {"Gulf of Mexico":(25,-90),"Caribbean Sea":(15,-75),"North Atlantic MDR":(15,-45),
             "Western Pacific":(15,135),"Bay of Bengal":(15,88),"Eastern Pacific":(15,-110)}
    zone = st.selectbox("MONITORING ZONE",list(ZONES.keys()))
    lat, lon = ZONES[zone]

    c1, c2 = st.columns([1,3])
    with c1:
        st.write(""); st.write("")
        go_btn = st.button("COMPUTE σ",use_container_width=True)
    with c2:
        st.markdown(f"""<div class="card mono" style="padding:12px">
        <span style="color:#8b949e">TARGET:</span> {zone} ({lat}°N, {lon}°E) ·
        <span style="color:#8b949e">SCALES:</span> 1h, 3h, 6h, 12h ·
        <span style="color:#8b949e">SOURCE:</span> Open-Meteo (7-day lookback)</div>""",unsafe_allow_html=True)

    if go_btn:
        with st.spinner("Computing α at 4 time scales..."):
            df = fetch_coherence(lat, lon)
        if df is not None and len(df)>10:
            curr = np.nanmedian(df['Sigma'].iloc[-24:])

            gc, tc = st.columns([1,1.8])
            with gc:
                st.plotly_chart(sigma_gauge(curr),use_container_width=True)
                if curr < 0.05:
                    st.markdown("""<div class="card card-danger" style="text-align:center"><span style="color:#f85149;font-weight:700;font-family:'JetBrains Mono'">HYPER-COHERENT</span><br><span class="mono" style="color:#8b949e">All scales locked — phase transition signature</span></div>""",unsafe_allow_html=True)
                elif curr < 0.15:
                    st.markdown("""<div class="card card-warn" style="text-align:center"><span style="color:#d29922;font-weight:700;font-family:'JetBrains Mono'">ELEVATED</span><br><span class="mono" style="color:#8b949e">Scales coupling — monitor</span></div>""",unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="card card-success" style="text-align:center"><span style="color:#3fb950;font-weight:700;font-family:'JetBrains Mono'">NORMAL</span><br><span class="mono" style="color:#8b949e">Scales independent — no phase transition signal</span></div>""",unsafe_allow_html=True)

                st.markdown(f"""<div class="card mono" style="font-size:11px">
                <b style="color:#f0f6fc">This dataset:</b> σ = {curr:.4f}<br>
                <span style="color:#f85149">■</span> Crisis ref (BTC crash): 0.031<br>
                <span style="color:#d29922">■</span> Watch threshold: 0.05<br>
                <span style="color:#3fb950">■</span> Normal ref (BTC control): 0.310</div>""",unsafe_allow_html=True)

            with tc:
                colors = {'A_1h':'#f85149','A_3h':'#d29922','A_6h':'#58a6ff','A_12h':'#3fb950'}
                fig_ms = go.Figure()
                for col, color in colors.items():
                    if col in df.columns:
                        fig_ms.add_trace(go.Scatter(x=df['Date'],y=df[col],name=col.replace('A_',''),
                            line=dict(color=color,width=1.5)))
                fig_ms = dark_layout(fig_ms, 280)
                fig_ms.update_layout(title="α AT 4 TIME SCALES")
                st.plotly_chart(fig_ms, use_container_width=True)

            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=df['Date'],y=df['Sigma'],name="σ",line=dict(color='#58a6ff',width=2.5),
                fill='tozeroy',fillcolor='rgba(88,166,255,0.08)'))
            fig_s.add_hline(y=0.05,line_dash="dash",line_color="#f85149",annotation_text="CRISIS",
                annotation_font=dict(color="#f85149",size=10))
            fig_s.add_hline(y=0.15,line_dash="dash",line_color="#d29922",annotation_text="WATCH",
                annotation_font=dict(color="#d29922",size=10))
            fig_s = dark_layout(fig_s, 320)
            fig_s.update_layout(title="CROSS-SCALE COHERENCE (σ) — LOWER = MORE COUPLED")
            st.plotly_chart(fig_s, use_container_width=True)

# ══════════════════════════════════════
# MODULE 3: SEISMOLOGY REFERENCE
# ══════════════════════════════════════
elif module == "SEISMOLOGY REFERENCE":
    st.markdown("## SEISMOLOGY — RTM CALIBRATION ANCHOR")

    st.markdown("""<div class="card card-accent"><span class="tag tag-convergent">CONVERGENT</span>
    <span style="color:#c9d1d9;font-size:13px;margin-left:8px">α = 1.007 — recovers ballistic mechanics from RTM framework</span>
    <p class="mono" style="color:#8b949e;margin:10px 0 0 0">
    Seismic rupture: 51 earthquakes, ODR, R² = 0.987. The cleanest calibration in the corpus.<br>
    If RTM gets seismology wrong, the entire framework is suspect. It gets it right.
    </p></div>""",unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Fault-Type Classification")
        st.markdown("""<div class="card">
        <table style="width:100%;font-family:'JetBrains Mono';font-size:13px;color:#c9d1d9">
        <tr style="color:#8b949e;border-bottom:1px solid #21262d"><td>FAULT TYPE</td><td>n</td><td>α (ODR)</td><td>95% CI</td><td>INCLUDES 1.0?</td></tr>
        <tr><td>Strike-slip</td><td>27</td><td>1.040</td><td>[0.989, 1.091]</td><td><span style="color:#3fb950">YES</span></td></tr>
        <tr><td>Reverse</td><td>19</td><td>0.987</td><td>[0.942, 1.032]</td><td><span style="color:#3fb950">YES</span></td></tr>
        <tr style="border-top:1px solid #21262d"><td><b>Normal</b></td><td>5</td><td><b>0.865</b></td><td>[0.755, 0.975]</td><td><span style="color:#f85149"><b>NO</b></span></td></tr>
        </table></div>""",unsafe_allow_html=True)

        st.markdown("""<div class="card card-success"><span class="tag tag-novel">NOVEL</span>
        <span class="mono" style="color:#8b949e;margin-left:8px">Normal faults propagate sub-ballistically. CI excludes 1.0.</span>
        <p class="mono" style="color:#8b949e;margin:8px 0 0 0">
        Extensional ruptures have more complex geometry (hanging wall collapse, gravitational effects)
        → slower propagation. RTM predicts this from topological complexity. n=5 is small; replication needed.
        </p></div>""",unsafe_allow_html=True)

    with c2:
        st.markdown("### RG Fixed Points (from Chapter 11)")
        st.markdown("""<div class="card">
        <table style="width:100%;font-family:'JetBrains Mono';font-size:13px;color:#c9d1d9">
        <tr style="color:#8b949e"><td>FIXED POINT</td><td>α</td><td>OPERATOR</td><td>EXAMPLES</td></tr>
        <tr><td style="color:#3fb950">Frozen</td><td>0</td><td>Instantaneous</td><td>Enzymes (α≈0)</td></tr>
        <tr><td style="color:#d29922">White Noise</td><td>0.5</td><td>Max entropy</td><td>CHF terminal (0.53)</td></tr>
        <tr><td style="color:#58a6ff"><b>Ballistic</b></td><td><b>1.0</b></td><td><b>Wave eq.</b></td><td><b>Seismic (1.007), LIGO (1.024)</b></td></tr>
        <tr><td style="color:#bc8cff">Diffusive</td><td>2.0</td><td>Laplacian</td><td>Random walk</td></tr>
        </table>
        <p class="mono" style="color:#8b949e;margin:10px 0 0 0;font-size:11px">
        Of 10 validated systems with |α| < 3: 9 cluster within 0.2 of a fixed point,
        5 cluster within 0.05. The band structure is a consequence of RG flow,
        not empirical pattern-matching.</p></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════
# MODULE 4: RED TEAM FINDINGS
# ══════════════════════════════════════
elif module == "RED TEAM FINDINGS":
    st.markdown("## RED TEAM FINDINGS — METEOROLOGY")
    st.markdown("""<div class="card card-accent" style="margin-bottom:20px">
    <span class="mono" style="color:#8b949e">Independent adversarial audit: Claude Opus 4.6 Extended Thinking · April-May 2026 · 13 flanks across 3 rounds</span></div>""",unsafe_allow_html=True)

    st.markdown("### What WORKS")
    st.markdown("""
| Finding | Evidence | Tag |
|---------|----------|-----|
| **Tornado TOR vs WRN** (d=0.96, AUC=0.751) | TorNet MIT, 1,105 events | NOVEL |
| **α subsumes VEL** (ΔAUC = 0.000) | VEL adds zero to α | NOVEL |
| **α predicts EF intensity** (ρ=+0.446) | 435 confirmed tornadoes | NOVEL |
| **Structural-only radar** (AUC=0.698) | KDP+DBZ+RHOHV, no velocity | NOVEL |
| **Normal fault sub-ballistic** (α=0.865) | CI excludes 1.0, n=5 | NOVEL |
| **Seismology calibration** (α=1.007) | R²=0.987, 51 earthquakes | CONVERGENT |
| **Multi-scale coherence** (σ=0.03 vs 0.31) | Cross-domain finding | NOVEL |
| **RI timing lead** (11.6h mean) | 26 RI events, CV=0.096 | TIMING |
    """)

    st.markdown("### What DOES NOT WORK")
    st.markdown("""
| Claim | Test result | Tag |
|-------|------------|-----|
| **Hurricane α independent of wind** | ρ=0.957, 13 tests, all ns after control | FAILED |
| **α_STD, α_gap, fingerprints** | All collapse after wind control | FAILED |
| **α-pressure independence** | ρ=0.993 with pressure | FAILED |
| **Outbreak-level α variance** | Does not predict outbreak quality | FAILED |
    """)

    st.markdown("### Score: 68%")
    st.markdown("""<div class="card"><p class="mono" style="color:#8b949e">
    Tornado is the crown jewel (untouched by any flank). Hurricane α is definitively circular.
    The surviving hurricane finding is timing only (11.6h lead). Normal fault deviation is novel but n=5.
    Multi-scale coherence is the cross-domain novel metric. Seismology is the cleanest calibration anchor.
    </p></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown("---")
st.markdown("""<div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:11px;color:#484f58;padding:10px 0">
RTM ATMOSPHERIC RADAR v3.0 · Post-Red Team Build · CC BY 4.0 ·
<a href="https://github.com/zarpafantasma/corpus_rythmos" style="color:#58a6ff">github.com/zarpafantasma/corpus_rythmos</a></div>""",unsafe_allow_html=True)
