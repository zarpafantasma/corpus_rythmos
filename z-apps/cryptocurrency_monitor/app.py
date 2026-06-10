"""
RTM ECONOMIC STRUCTURAL RADAR v3
==================================
Built from scratch post-Red Team audit.
Shows ONLY what survived adversarial testing.

Headline: Multi-Scale Coherence (σ crash=0.031 vs control=0.310)
Forensic: Historical crash anatomy (post-hoc, labeled honestly)
Physics:  Inverse cubic law (convergent with Gabaix 2003)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx = np.sum((x - mx)**2)
    return np.sum((x - mx)*(y - my))/vx if vx > 0 else 0.0

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════
st.set_page_config(page_title="RTM Economic Radar", layout="wide", initial_sidebar_state="expanded")

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
.tag-forensic{background:#2d2000;color:#d29922;border:1px solid #9e6a03}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════
# DATA FUNCTIONS
# ══════════════════════════════════════
OHLCV_COLS = ['Open time','Open','High','Low','Close','Volume','Close time',
              'Quote vol','Trades','Taker buy base','Taker buy quote','Ignore']

def noise_filter(sym):
    if 'BTC' in sym: return 5.0
    if 'ETH' in sym: return 0.5
    if 'SOL' in sym: return 0.05
    return 0.001

@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, names=OHLCV_COLS)
        ts = df['Open time'].iloc[0]
        df['Date'] = pd.to_datetime(df['Open time'], unit='us' if ts>1e14 else 'ms')
        for c in ['Open','High','Low','Close','Volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except: return None

@st.cache_data(ttl=60)
def fetch_live(symbol='BTC/USD'):
    try:
        ex = ccxt.kraken({'enableRateLimit':True})
        ohlcv = ex.fetch_ohlcv(symbol, '1m', limit=120)
        df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return None

@st.cache_data(ttl=300)
def fetch_live_hourly(symbol='BTC/USD'):
    """Fetch hourly candles (14 days) for multi-scale coherence."""
    try:
        ex = ccxt.kraken({'enableRateLimit':True})
        ohlcv = ex.fetch_ohlcv(symbol, '1h', limit=336)
        df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return None

@st.cache_data(ttl=120)
def fetch_health():
    assets = ['BTC/USD','ETH/USD','SOL/USD','XRP/USD']
    ex = ccxt.kraken({'enableRateLimit':True})
    out = []
    for sym in assets:
        try:
            ohlcv = ex.fetch_ohlcv(sym, '1m', limit=120)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            nf = noise_filter(sym)
            spread = df['High'] - df['Low']
            lv = np.log(df['Volume']+1e-9)
            ls = np.log(np.where(spread<nf, nf, spread))
            cov = pd.Series(lv).rolling(60).cov(pd.Series(ls))
            var = pd.Series(lv).rolling(60).var()
            raw = pd.Series(cov/var).replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=1).mean()
            a = raw.iloc[-1]
            out.append({'asset':sym.split('/')[0],'alpha':a,'price':df['Close'].iloc[-1]})
        except:
            out.append({'asset':sym.split('/')[0],'alpha':None,'price':None})
    return out

def compute_alpha_series(df, nf=5.0):
    spread = df['High'] - df['Low']
    lv = np.log(df['Volume']+1e-9)
    ls = np.log(np.where(spread<nf, nf, spread))
    cov = pd.Series(lv).rolling(60).cov(pd.Series(ls))
    var = pd.Series(lv).rolling(60).var()
    return pd.Series(cov/var).replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=1).mean()

def compute_multiscale(df, scales=[1,5,15,60]):
    vol = df['Volume'].values
    vola = (df['High'] - df['Low']).values
    vola = np.where(vola<0.01, 0.01, vola)
    scale_a = {}
    for s in scales:
        bins = len(df)//s
        if bins < 30: continue
        va = np.array([vol[i*s:(i+1)*s].sum() for i in range(bins)])
        vo = np.array([vola[i*s:(i+1)*s].mean() for i in range(bins)])
        alphas = []
        w = min(60, bins//4)
        for i in range(w, bins):
            lva, lvo = np.log(va[i-w:i]+1e-9), np.log(vo[i-w:i]+1e-9)
            mask = np.isfinite(lva) & np.isfinite(lvo)
            alphas.append(abs(_slope(lva[mask],lvo[mask])) if mask.sum()>15 else np.nan)
        scale_a[s] = [np.nan]*w + alphas
    if len(scale_a)<2: return None, None
    ml = min(len(v) for v in scale_a.values())
    aligned = np.array([v[-ml:] for v in scale_a.values()])
    sigma = [np.std(aligned[:,i][~np.isnan(aligned[:,i])]) if np.sum(~np.isnan(aligned[:,i]))>=2 else np.nan for i in range(ml)]
    return np.array(sigma), scale_a

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
                     {'range':[0.05,0.12],'color':'rgba(210,153,34,0.15)'},
                     {'range':[0.12,0.4],'color':'rgba(63,185,80,0.1)'}],
            'threshold':{'line':{'color':'#f85149','width':2},'thickness':0.75,'value':0.05}}))
    fig.update_layout(height=260,margin=dict(l=20,r=20,t=40,b=10),paper_bgcolor="rgba(0,0,0,0)",font={'family':'JetBrains Mono'})
    return fig

def alpha_gauge(val):
    fig = go.Figure(go.Indicator(mode="gauge+number",value=val,
        number={'font':{'color':'#f0f6fc','family':'JetBrains Mono'},'valueformat':'.3f'},
        title={'text':"MICROSTRUCTURE α",'font':{'size':11,'color':'#8b949e','family':'JetBrains Mono'}},
        gauge={'axis':{'range':[0,3],'tickcolor':'#30363d'},
            'bar':{'color':'#f0f6fc','thickness':0.08},'bgcolor':'rgba(0,0,0,0)','borderwidth':0,
            'steps':[{'range':[0,0.8],'color':'rgba(63,185,80,0.12)'},
                     {'range':[0.8,1.2],'color':'rgba(88,166,255,0.12)'},
                     {'range':[1.2,2.0],'color':'rgba(210,153,34,0.12)'},
                     {'range':[2.0,3.0],'color':'rgba(248,81,73,0.15)'}]}))
    fig.update_layout(height=260,margin=dict(l=20,r=20,t=40,b=10),paper_bgcolor="rgba(0,0,0,0)",font={'family':'JetBrains Mono'})
    return fig

# ══════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════
st.sidebar.markdown("""<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;text-align:center;margin-bottom:16px">
<span style="font-family:'JetBrains Mono';font-size:14px;font-weight:700;color:#f0f6fc">RTM ECONOMIC RADAR</span><br>
<span style="font-family:'JetBrains Mono';font-size:10px;color:#8b949e">v3.0 — POST-RED TEAM BUILD</span></div>""",unsafe_allow_html=True)

module = st.sidebar.radio("MODULE",[
    "MULTI-SCALE COHERENCE",
    "LIVE MICROSTRUCTURE",
    "FORENSIC LABORATORY",
    "MARKET PHYSICS",
    "RED TEAM FINDINGS"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="color:#8b949e;font-size:11px;line-height:1.5;font-family:'JetBrains Mono',monospace;border-left:2px solid #30363d;padding-left:10px">
<b>DATA SOURCES</b><br>
Live: Kraken API (ccxt)<br>
Historical: Binance 1-min OHLCV<br>
Events: crash_alpha_analysis.csv<br><br>
<b>AUDIT</b><br>
Red Team: Claude Opus 4.6<br>
Out-of-sample: 25% accuracy<br>
Novel metric: Multi-Scale σ<br>
Score: 68%</div>""",unsafe_allow_html=True)

# ══════════════════════════════════════
# MODULE 1: MULTI-SCALE COHERENCE (headline)
# ══════════════════════════════════════
if module == "MULTI-SCALE COHERENCE":
    st.markdown("## MULTI-SCALE COHERENCE")

    st.markdown("""<div class="card card-success"><span class="tag tag-novel">RED TEAM SURVIVOR</span>
    <span style="color:#c9d1d9;font-size:13px;margin-left:8px">The only genuinely novel RTM economic metric</span>
    <p class="mono" style="color:#8b949e;margin:10px 0 0 0">
    During BTC crash months: σ = 0.031-0.034 (all scales locked — phase transition).<br>
    During control month: σ = 0.310 (scales independent — normal market).<br>
    10x separation. No standard financial indicator measures cross-scale α coherence.
    </p></div>""",unsafe_allow_html=True)

    tab_live, tab_hist = st.tabs(["LIVE (Kraken)","HISTORICAL (Binance)"])

    with tab_live:
        c_sel, c_btn = st.columns([1,1])
        with c_sel:
            sym = st.selectbox("ASSET",["BTC/USD","ETH/USD","SOL/USD","XRP/USD"])
        with c_btn:
            st.write("")
            if st.button("COMPUTE LIVE σ",use_container_width=True):
                st.cache_data.clear(); st.rerun()

        with st.spinner("Fetching 14 days of hourly data from Kraken..."):
            df_live = fetch_live_hourly(sym)

        if df_live is not None and len(df_live)>60:
            sigma, sa = compute_multiscale(df_live, scales=[1,3,6,12])
            if sigma is not None and len(sigma)>5:
                curr = np.nanmedian(sigma[-24:])
                gc, tc = st.columns([1,1.8])
                with gc:
                    st.plotly_chart(sigma_gauge(curr),use_container_width=True)
                    if curr<0.05:
                        st.markdown("""<div class="card card-danger" style="text-align:center"><span style="color:#f85149;font-weight:700;font-family:'JetBrains Mono'">HYPER-COHERENT</span><br><span class="mono" style="color:#8b949e">All scales locked</span></div>""",unsafe_allow_html=True)
                    elif curr<0.15:
                        st.markdown("""<div class="card card-warn" style="text-align:center"><span style="color:#d29922;font-weight:700;font-family:'JetBrains Mono'">ELEVATED</span><br><span class="mono" style="color:#8b949e">Scales coupling</span></div>""",unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="card card-success" style="text-align:center"><span style="color:#3fb950;font-weight:700;font-family:'JetBrains Mono'">NORMAL</span><br><span class="mono" style="color:#8b949e">Scales independent</span></div>""",unsafe_allow_html=True)
                with tc:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=sigma,name="σ",line=dict(color='#58a6ff',width=2.5),fill='tozeroy',fillcolor='rgba(88,166,255,0.08)'))
                    fig.add_hline(y=0.05,line_dash="dash",line_color="#f85149",annotation_text="CRISIS",annotation_font=dict(color="#f85149",size=10))
                    fig.add_hline(y=0.15,line_dash="dash",line_color="#d29922",annotation_text="WATCH",annotation_font=dict(color="#d29922",size=10))
                    fig = dark_layout(fig,300)
                    fig.update_layout(title=f"LIVE σ — {sym} (Kraken hourly, 14 days)")
                    st.plotly_chart(fig,use_container_width=True)
            else:
                st.info("Insufficient data for multi-scale computation (need >60 candles per scale)")
        else:
            st.error("Could not fetch live data from Kraken.")

    with tab_hist:
        events = {"March 2020 — COVID Crash":"BTCUSDT-1m-2020-03.csv",
                  "November 2022 — FTX Collapse":"BTCUSDT-1m-2022-11.csv",
                  "September 2023 — Control Group":"BTCUSDT-1m-2023-09.csv",
                  "October 2025 — Anomaly":"BTCUSDT-1m-2025-10.csv"}
        ev = st.selectbox("SELECT HISTORICAL DATASET",list(events.keys()))

        if st.button("COMPUTE HISTORICAL σ",use_container_width=True):
            path = os.path.join(BASE_DIR, events[ev])
            with st.spinner("Computing α at 4 time scales..."):
                df_h = load_csv(path)
            if df_h is not None:
                sigma_h, sa_h = compute_multiscale(df_h)
                if sigma_h is not None:
                    med = np.nanmedian(sigma_h)
                    gc2, tc2 = st.columns([1,1.8])
                    with gc2:
                        st.plotly_chart(sigma_gauge(med),use_container_width=True)
                        st.markdown(f"""<div class="card mono" style="font-size:11px">
                        <b style="color:#f0f6fc">{ev}:</b> σ = {med:.4f}<br>
                        <span style="color:#f85149">■</span> Crisis (COVID/FTX): 0.031-0.034<br>
                        <span style="color:#d29922">■</span> Watch: < 0.05<br>
                        <span style="color:#3fb950">■</span> Normal (Control): 0.310</div>""",unsafe_allow_html=True)
                    with tc2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(y=sigma_h,name="σ",line=dict(color='#58a6ff',width=2.5),fill='tozeroy',fillcolor='rgba(88,166,255,0.08)'))
                        fig2.add_hline(y=0.05,line_dash="dash",line_color="#f85149",annotation_text="CRISIS")
                        fig2.add_hline(y=0.15,line_dash="dash",line_color="#d29922",annotation_text="WATCH")
                        fig2 = dark_layout(fig2,300)
                        fig2.update_layout(title=f"σ — {ev}")
                        st.plotly_chart(fig2,use_container_width=True)

                    # Multi-scale α traces
                    colors = {1:'#f85149',5:'#d29922',15:'#58a6ff',60:'#3fb950'}
                    fig3 = go.Figure()
                    for s,c in colors.items():
                        if s in sa_h:
                            fig3.add_trace(go.Scatter(y=sa_h[s],name=f"{s}min",line=dict(color=c,width=1.5)))
                    fig3 = dark_layout(fig3,280)
                    fig3.update_layout(title="α AT 4 TIME SCALES")
                    st.plotly_chart(fig3,use_container_width=True)

# ══════════════════════════════════════
# MODULE 2: LIVE MICROSTRUCTURE
# ══════════════════════════════════════
elif module == "LIVE MICROSTRUCTURE":
    st.markdown("## LIVE MICROSTRUCTURE MONITOR")

    st.markdown("""<div class="card card-warn"><span class="tag tag-forensic">STRUCTURAL DESCRIPTOR</span>
    <span class="mono" style="color:#8b949e;margin-left:8px">
    α measures volume-volatility coupling. NOT a directional signal. Out-of-sample crash prediction: 25%.
    </span></div>""",unsafe_allow_html=True)

    c_sel, c_btn = st.columns([1,1])
    with c_sel: sym2 = st.selectbox("ASSET",["BTC/USD","ETH/USD","SOL/USD","XRP/USD"],key="live_sym")
    with c_btn:
        st.write("")
        if st.button("REFRESH",use_container_width=True): st.cache_data.clear(); st.rerun()

    df_l = fetch_live(sym2)
    if df_l is not None and len(df_l)>60:
        nf = noise_filter(sym2)
        df_l['Alpha'] = compute_alpha_series(df_l, nf)
        df_l = df_l.dropna(subset=['Alpha'])
        curr_a = df_l['Alpha'].iloc[-1]
        curr_p = df_l['Close'].iloc[-1]

        gc3, tc3 = st.columns([1,2])
        with gc3:
            st.plotly_chart(alpha_gauge(curr_a),use_container_width=True)
            states = [(0.8,"LAMINAR","#3fb950","Efficient coupling"),
                      (1.2,"TURBULENT","#58a6ff","Elevated friction"),
                      (2.0,"VISCOUS","#d29922","High friction — not a directional signal"),
                      (99,"EXTREME","#f85149","Extreme reading — 25% out-of-sample accuracy")]
            for th,label,color,desc in states:
                if curr_a < th:
                    st.markdown(f"""<div class="card" style="border-left:3px solid {color};text-align:center">
                    <span style="color:{color};font-weight:700;font-family:'JetBrains Mono'">{label}</span><br>
                    <span class="mono" style="color:#8b949e">{desc}</span></div>""",unsafe_allow_html=True)
                    break
            tk = sym2.split('/')[0]
            fmt = f"${curr_p:,.2f}" if curr_p > 1 else f"${curr_p:.4f}"
            st.metric(f"PRICE ({tk})", fmt)

        with tc3:
            fig4 = make_subplots(specs=[[{"secondary_y":True}]])
            fig4.add_trace(go.Scatter(x=df_l['Date'],y=df_l['Close'],name="PRICE",line=dict(color='#58a6ff',width=2),fill='tozeroy',fillcolor='rgba(88,166,255,0.05)'),secondary_y=False)
            fig4.add_trace(go.Scatter(x=df_l['Date'],y=df_l['Alpha'],name="α",line=dict(color='#f85149',width=2)),secondary_y=True)
            fig4.add_hline(y=2.0,line_dash="dash",line_color="rgba(248,81,73,0.4)",secondary_y=True)
            fig4.add_hline(y=1.2,line_dash="dash",line_color="rgba(210,153,34,0.4)",secondary_y=True)
            fig4 = dark_layout(fig4,380)
            fig4.update_yaxes(title_text="",secondary_y=True,range=[-0.1,3.1])
            fig4.update_layout(title=f"LIVE — {sym2}")
            st.plotly_chart(fig4,use_container_width=True)

        # Systemic health
        st.markdown("### SYSTEMIC HEALTH")
        health = fetch_health()
        cols = st.columns(4)
        for i,h in enumerate(health):
            with cols[i]:
                if h['alpha'] is not None and not np.isnan(h['alpha']):
                    color = '#3fb950' if h['alpha']<0.8 else '#58a6ff' if h['alpha']<1.2 else '#d29922' if h['alpha']<2 else '#f85149'
                    label = 'LAMINAR' if h['alpha']<0.8 else 'TURBULENT' if h['alpha']<1.2 else 'VISCOUS' if h['alpha']<2 else 'EXTREME'
                    st.markdown(f"""<div class="card" style="text-align:center;border-top:3px solid {color}">
                    <span class="mono" style="color:#8b949e">{h['asset']}</span><br>
                    <span style="color:#f0f6fc;font-size:20px;font-family:'JetBrains Mono';font-weight:700">α = {h['alpha']:.3f}</span><br>
                    <span class="mono" style="color:{color}">{label}</span></div>""",unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="card" style="text-align:center"><span class="mono" style="color:#484f58">{h['asset']}<br>NO DATA</span></div>""",unsafe_allow_html=True)
    else:
        st.error("Could not fetch live data from Kraken.")

# ══════════════════════════════════════
# MODULE 3: FORENSIC LABORATORY
# ══════════════════════════════════════
elif module == "FORENSIC LABORATORY":
    st.markdown("## FORENSIC LABORATORY")

    st.markdown("""<div class="card card-warn"><span class="tag tag-forensic">POST-HOC ANALYSIS</span>
    <span class="mono" style="color:#8b949e;margin-left:8px">
    Historical reconstruction. These patterns were identified AFTER crashes occurred.
    Out-of-sample prediction: 25% (1/4 events). This is forensic anatomy, not prospective prediction.
    </span></div>""",unsafe_allow_html=True)

    events = {"November 2022 — FTX Collapse":"BTCUSDT-1m-2022-11.csv",
              "March 2020 — Black Thursday":"BTCUSDT-1m-2020-03.csv",
              "May 2021 — China Ban":"BTCUSDT-1m-2021-05.csv",
              "September 2023 — Control Group":"BTCUSDT-1m-2023-09.csv",
              "October 2025 — Anomaly (Binance glitch)":"BTCUSDT-1m-2025-10.csv"}

    labels = {"November 2022 — FTX Collapse":("Chronic Viscosity","#d29922","α sustained at 1.10-1.25 for 4 days before capitulation. Solvency crisis, not flash crash."),
              "March 2020 — Black Thursday":("Sudden Bifurcation","#f85149","α peaked at 1.76. Classic liquidity shock. Fastest crash in dataset."),
              "May 2021 — China Ban":("Turbulence — No Fracture","#3fb950","α peaked at 1.33, reverted quickly. Market structure held. Regulatory stress, not structural failure."),
              "September 2023 — Control Group":("Laminar Baseline","#8b949e","α ≈ 0.45. Textbook healthy coupling. Zero false alarms across entire month."),
              "October 2025 — Anomaly (Binance glitch)":("Technical Anomaly","#d29922","Elevated α from Binance technical glitch. Not a fundamental crash. Single in-sample case.")}

    ev2 = st.selectbox("SELECT EVENT",list(events.keys()))
    path2 = os.path.join(BASE_DIR, events[ev2])
    df_f = load_csv(path2)

    if df_f is not None:
        lb = labels[ev2]
        st.markdown(f"""<div class="card" style="border-left:3px solid {lb[1]}">
        <span style="color:{lb[1]};font-weight:700;font-family:'JetBrains Mono'">{lb[0]}</span>
        <p class="mono" style="color:#8b949e;margin:8px 0 0 0">{lb[2]}</p></div>""",unsafe_allow_html=True)

        df_f['Alpha'] = compute_alpha_series(df_f)
        df_f = df_f.dropna(subset=['Alpha'])

        if "Control" not in ev2:
            peak_idx = df_f['Alpha'].idxmax()
            peak_date = df_f.loc[peak_idx,'Date']
            df_show = df_f[(df_f['Date']>=peak_date-pd.Timedelta(hours=4))&(df_f['Date']<=peak_date+pd.Timedelta(hours=4))]
        else:
            df_show = df_f

        fig5 = make_subplots(specs=[[{"secondary_y":True}]])
        fig5.add_trace(go.Scatter(x=df_show['Date'],y=df_show['Close'],name="PRICE",line=dict(color='#58a6ff',width=2),fill='tozeroy',fillcolor='rgba(88,166,255,0.05)'),secondary_y=False)
        fig5.add_trace(go.Scatter(x=df_show['Date'],y=df_show['Alpha'],name="α",line=dict(color='#f85149',width=2)),secondary_y=True)
        fig5.add_hline(y=2.0,line_dash="dash",line_color="rgba(248,81,73,0.3)",secondary_y=True)
        fig5.add_hline(y=1.2,line_dash="dash",line_color="rgba(210,153,34,0.3)",secondary_y=True)
        if "Control" not in ev2:
            fig5.add_vline(x=peak_date,line_dash="solid",line_color="#d29922",line_width=1,opacity=0.5)
            fig5.add_annotation(x=peak_date,y=df_show['Alpha'].max()+0.15,text="PEAK α",
                font=dict(color="#d29922",size=10,family="JetBrains Mono"),yref="y2",bgcolor="rgba(6,9,15,0.8)",
                bordercolor="#d29922",borderwidth=1,xanchor="left",xshift=10)
        fig5 = dark_layout(fig5,450)
        fig5.update_yaxes(title_text="",secondary_y=True,range=[-0.1,3.1])
        fig5.update_layout(title=f"FORENSIC — {ev2}")
        st.plotly_chart(fig5,use_container_width=True)

        # Stats
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("MAX α",f"{df_f['Alpha'].max():.3f}")
        m2.metric("MEAN α",f"{df_f['Alpha'].mean():.3f}")
        m3.metric("PRICE RANGE",f"${df_f['Close'].min():,.0f} — ${df_f['Close'].max():,.0f}")
        m4.metric("CANDLES",f"{len(df_f):,}")

# ══════════════════════════════════════
# MODULE 4: MARKET PHYSICS
# ══════════════════════════════════════
elif module == "MARKET PHYSICS":
    st.markdown("## MARKET PHYSICS")

    st.markdown("""<div class="card card-accent"><span class="tag tag-convergent">CONVERGENT</span>
    <span class="mono" style="color:#8b949e;margin-left:8px">
    Consistent with Gabaix et al. 2003. RTM reframes known scaling as topological transport class.
    </span></div>""",unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Inverse Cubic Law")
        st.markdown("""<div class="card"><p class="mono" style="color:#8b949e">
        Return distribution tail exponent across 16 global markets:<br><br>
        <span style="color:#f0f6fc;font-size:24px;font-weight:700">α = 2.966 ± 0.236</span><br><br>
        Consistent with the inverse cubic law (Gabaix 2003).<br>
        A 5σ event occurs 1 in 125 days (power law) vs 1 in 3.5 million days (Gaussian).<br>
        Fat tails are systematic structural features, not anomalies.</p></div>""",unsafe_allow_html=True)

    with c2:
        st.markdown("### Recovery Scaling")
        st.markdown("""<div class="card"><p class="mono" style="color:#8b949e">
        Recovery time scales nonlinearly with drawdown:<br><br>
        <span style="color:#f0f6fc;font-size:24px;font-weight:700">τ ∝ D<sup>3.59 ± 0.70</sup></span><br><br>
        A 20% drawdown recovers in ~1 year.<br>
        A 50% drawdown recovers in ~4 years.<br>
        An 80% drawdown recovers in ~15 years.<br><br>
        <span style="color:#8b949e;font-size:11px">Based on historical BTC drawdowns. Not a validated prediction model.</span></p></div>""",unsafe_allow_html=True)

    st.markdown("### Probability Calculator")
    sigma = st.slider("EVENT SIZE (σ)",2,10,5)
    gauss = np.exp(-sigma**2/2)
    power = sigma**-3
    c3,c4 = st.columns(2)
    c3.metric("GAUSSIAN PROBABILITY",f"1 in {int(1/gauss):,} days")
    c4.metric("POWER LAW (REAL)",f"1 in {int(1/power):,} days")

# ══════════════════════════════════════
# MODULE 5: RED TEAM FINDINGS
# ══════════════════════════════════════
elif module == "RED TEAM FINDINGS":
    st.markdown("## RED TEAM FINDINGS — ECONOMICS")
    st.markdown("""<div class="card card-accent" style="margin-bottom:20px">
    <span class="mono" style="color:#8b949e">Claude Opus 4.6 Extended Thinking · April-May 2026 · 5 flanks</span></div>""",unsafe_allow_html=True)

    st.markdown("### What WORKS")
    st.markdown("""
| Finding | Evidence | Tag |
|---------|----------|-----|
| **Multi-scale coherence** σ=0.031 vs 0.310 | 3 BTC months, 10x separation | NOVEL |
| **Volume-volatility coupling is real** | r > 0.88 all months | CONFIRMED |
| **Control group: zero false alarms** | Sept 2023, full month laminar | CONFIRMED |
| **Inverse cubic law** α=2.966±0.236 | 16 global markets | CONVERGENT |
| **Recovery scaling** τ ∝ D^3.59 | Historical BTC drawdowns | CONVERGENT |
| **Forensic pattern** α spikes in crashes | In-sample, 4 events | FORENSIC |
    """)

    st.markdown("### What DOES NOT WORK")
    st.markdown("""
| Claim | Test | Result |
|-------|------|--------|
| **"Crash early warning"** | Out-of-sample: 25% (1/4) | FAILED |
| **"96-hour FTX warning"** | Post-hoc observation | NOT VALIDATED |
| **"EXIT MARKETS" command** | No prospective basis | REMOVED |
| **October 2025 "15-hour warning"** | Binance technical glitch | NOT VALIDATED |
| **α-drop threshold generalizes** | Does not transfer post-2022 | FAILED |
    """)

    st.markdown("""<div class="card card-success"><span class="tag tag-novel">THE SURVIVING FINDING</span>
    <p class="mono" style="color:#8b949e;margin:10px 0 0 0">
    Multi-Scale Coherence: during phase transitions, all temporal scales lock simultaneously (σ → 0.03).<br>
    In normal markets, each scale operates independently (σ → 0.31).<br>
    This is RTM-native, cross-domain (also observed in atmosphere, ecology, brain), and measures<br>
    something no standard financial indicator measures. The forensic DFA patterns are real (d=−1.45<br>
    in-sample); the prospective prediction is not yet validated.
    </p></div>""",unsafe_allow_html=True)

    st.markdown("### Score: 68%")

# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown("---")
st.markdown("""<div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:11px;color:#484f58;padding:10px 0">
RTM ECONOMIC RADAR v3.0 · Post-Red Team Build · NOT financial advice · Out-of-sample: 25% · CC BY 4.0 ·
<a href="https://github.com/zarpafantasma/corpus_rythmos" style="color:#58a6ff">github.com/zarpafantasma/corpus_rythmos</a></div>""",unsafe_allow_html=True)
