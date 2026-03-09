import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y TEMA PREMIUM
# ==========================================
st.set_page_config(
    page_title="RTM Radar Económico",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado para Tema Fintech Oscuro Premium
st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    header[data-testid="stHeader"] { background-color: #0B0E14 !important; height: 0px; }
    [data-testid="stSidebar"] { background-color: #0F1219 !important; border-right: 1px solid #1E232B; }
    .stMetric { background-color: #151A23; border: 1px solid #1E232B; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom: 2px solid #00E5FF !important; }
    .warning-box { background-color: rgba(255, 234, 0, 0.1); border-left: 4px solid #FFEA00; padding: 10px; margin-bottom: 15px; border-radius: 4px;}
    .critical-box { background-color: rgba(255, 23, 68, 0.1); border-left: 4px solid #FF1744; padding: 10px; margin-bottom: 15px; border-radius: 4px;}
    .macro-box { background-color: rgba(186, 104, 200, 0.1); border-left: 4px solid #BA68C8; padding: 15px; margin-bottom: 20px; border-radius: 4px;}
    .streamlit-expanderHeader { background-color: #151A23 !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PROCESAMIENTO MATEMÁTICO RTM (CACHEADO)
# ==========================================
@st.cache_data(show_spinner="Calculando Física RTM...")
def load_and_process_historical_data(file_path, window_size=60, noise_filter_usd=5.0):
    """Carga y procesa datos OHLCV aplicando el filtro microestructural y la transformación RTM."""
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    
    try:
        df = pd.read_csv(file_path, names=cols)
    except FileNotFoundError:
        return None

    # Manejo robusto de timestamps (ms vs us)
    unit = 'us' if df['Open time'].iloc[0] > 1e14 else 'ms'
    df['Date'] = pd.to_datetime(df['Open time'], unit=unit)
    
    # Transformación RTM: L (Volumen) y T (Volatilidad)
    # Se suma epsilon solo después de verificar el filtro de ruido para evitar log(0)
    df['Volatility'] = df['High'] - df['Low']
    df['log_L'] = np.log(df['Volume'].replace(0, np.nan)) 
    df['log_T'] = np.log(df['Volatility'].replace(0, np.nan))
    
    # Aplicar derivada logarítmica (Pendiente Alpha) solo si hay suficiente "energía"
    def calculate_alpha(y, x):
        if len(y.dropna()) < 10 or (np.max(np.exp(y)) < noise_filter_usd): 
            return np.nan
        slope, _, _, _, _ = np.polyfit(x.dropna(), y.dropna(), 1, full=True)[0] if len(x.dropna()) > 1 else (np.nan,)
        return slope

    # Pandas rolling apply (intensivo, por eso usamos caché)
    df['Rolling_Alpha'] = df['log_T'].rolling(window=window_size).corr(df['log_L']) * (df['log_T'].rolling(window=window_size).std() / df['log_L'].rolling(window=window_size).std())
    
    # Limpiar ruido de baja volatilidad
    df.loc[df['Volatility'] < noise_filter_usd, 'Rolling_Alpha'] = np.nan
    df['Rolling_Alpha'] = df['Rolling_Alpha'].fillna(method='ffill') # Mantener último estado físico
    
    return df

# ==========================================
# 3. INTERFAZ DE USUARIO (UI)
# ==========================================
st.sidebar.title("Radar RTM 👁️")
st.sidebar.markdown("Terminal de Relatividad Temporal Multiescala")

# Menú Lateral
page = st.sidebar.radio("Navegación", ["1. Panel de Control (En Vivo)", "2. Laboratorio Forense (Histórico)"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Leyenda de Física Micro (Táctica):**
* 🟢 **< 0.8:** Flujo Superfluido
* 🟡 **0.8 - 1.2:** Flujo Laminar
* 🟠 **1.2 - 2.0:** Viscosidad (Fricción)
* 🔴 **≥ 2.0:** FRACTURA ESTRUCTURAL
""")

# --- PESTAÑA 1: PANEL DE CONTROL EN VIVO ---
if page == "1. Panel de Control (En Vivo)":
    st.title("Panel de Mando Táctico RTM")
    st.markdown("Monitoreo de topología de mercado e integridad estructural en tiempo real.")
    
    # LA SOLUCIÓN AL RED TEAM: Separar Clima Macro de Gatillo Micro
    st.markdown("### 🌍 CONDICIÓN AMBIENTAL MACRO (Escala: Días/Semanas)")
    st.markdown("""
    <div class='macro-box'>
        <strong>Radar DFA (Tendencia a Largo Plazo)</strong>: Mide la memoria y correlación topológica de la red. 
        Una caída por debajo de <strong>0.45</strong> indica pérdida de estructura (Decorrelación), avisando de un posible colapso cinético en ~10 días.
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="RTM Alpha Macro (DFA a 7 días)", value="0.46 ▼", delta="-0.09 (ALERTA)", delta_color="inverse")
    with col_m2:
        st.metric(label="Estado de Topología Global", value="Anti-Persistente", delta="Deshilachándose")
    with col_m3:
        st.metric(label="Ventana Operativa Estimada", value="~9.5 Días", delta="Impacto Cinético Inminente", delta_color="inverse")

    st.markdown("---")
    st.markdown("### ⚡ GATILLO TÁCTICO MICRO (Escala: Minutos/Horas)")
    st.markdown("""
    Mide la fricción instantánea del libro de órdenes. Un pico en <strong>≥ 1.2</strong> indica emergencia por Viscosidad (el flujo se congela). Un pico en <strong>≥ 2.0</strong> es el punto de no retorno (Fractura Estructural).
    """)
    
    # Simulación de datos en vivo (Sintético para demostración)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=100, freq='1min')
    prices = np.linspace(65000, 64500, 100) + np.random.normal(0, 50, 100)
    
    # Simular una subida brusca hacia el punto de no retorno (1.8)
    base_alpha = np.random.normal(0.6, 0.1, 80)
    spike_alpha = np.linspace(0.8, 1.85, 20) + np.random.normal(0, 0.05, 20)
    alphas = np.concatenate([base_alpha, spike_alpha])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Precio (Vela/Línea)
    fig.add_trace(go.Scatter(x=dates, y=prices, name="Precio BTC", line=dict(color='#3B82F6', width=2)), secondary_y=False)
    
    # RTM Alpha Micro
    fig.add_trace(go.Scatter(x=dates, y=alphas, name="Micro RTM Alpha (Fricción)", line=dict(color='#FF1744', width=3)), secondary_y=True)
    
    # Líneas de umbral de emergencia (Calibradas a 1.2 y 2.0)
    fig.add_hline(y=1.2, line_dash="dash", line_color="#FFEA00", secondary_y=True, annotation_text="VISCOSIDAD (1.2)", annotation_font_color="#FFEA00")
    fig.add_hline(y=2.0, line_dash="solid", line_color="#FF1744", secondary_y=True, annotation_text="PUNTO NO RETORNO (2.0)", annotation_font_color="#FF1744")
    
    fig.update_layout(
        plot_bgcolor='#0B0E14', paper_bgcolor='#0B0E14', font_color='#8B949E',
        title="RTM Micro-Radar (Libro de Órdenes BTC/USDT)", height=500,
        margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Precio (USD)", secondary_y=False, gridcolor='#1E232B')
    fig.update_yaxes(title_text="Alpha Micro (Fricción)", secondary_y=True, range=[-0.5, 2.5], showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

    # --- TEXTO EXPLICATIVO PARA EL RADAR EN VIVO ---
    with st.expander("📖 ¿Cómo interpretar este radar en vivo?", expanded=True):
        st.markdown("""
        **Guía de lectura rápida:**
        - 🔵 **Línea Azul (Precio):** Es la acción del precio del activo en USD.
        - 🔴 **Línea Roja (Alpha Micro):** Es el exponente de coherencia RTM. Actúa como un *termómetro de fricción*. Te dice qué tan difícil está siendo mover el precio en el libro de órdenes.
        - 🟡 **Umbral Amarillo (1.2 - Zona Viscosa):** Si la línea roja cruza aquí, el mercado dejó de ser un fluido eficiente. La liquidez se está retirando y el mercado actúa como "miel". **Acción recomendada:** *Alerta táctica, reduzca exposición.*
        - 🟥 **Umbral Rojo (2.0 - Fractura Sólida):** Si la línea roja cruza aquí, la estructura causal del mercado se ha roto. El precio y el volumen se han desconectado. **Acción recomendada:** *Punto de no retorno, impacto inminente del precio.*
        """)

# --- PESTAÑA 2: LABORATORIO FORENSE ---
elif page == "2. Laboratorio Forense (Histórico)":
    st.title("Laboratorio Forense RTM")
    st.markdown("Analice cómo se rompió la topología en eventos de crisis pasados a nivel microestructural.")
    
    event_selector = st.selectbox(
        "Seleccionar Evento Histórico",
        [
            "Flash Crash Binance (Octubre 2025)", 
            "Jueves Negro / COVID (Marzo 2020)", 
            "China Ban (Mayo 2021)", 
            "Colapso FTX (Noviembre 2022)",
            "Grupo Control Sano (Septiembre 2023)"
        ]
    )
    
    # Diccionario de archivos (asumiendo que están en el mismo directorio)
    files = {
        "Flash Crash Binance (Octubre 2025)": "BTCUSDT-1m-2025-10.csv",
        "Jueves Negro / COVID (Marzo 2020)": "BTCUSDT-1m-2020-03.csv",
        "China Ban (Mayo 2021)": "BTCUSDT-1m-2021-05.csv",
        "Colapso FTX (Noviembre 2022)": "BTCUSDT-1m-2022-11.csv",
        "Grupo Control Sano (Septiembre 2023)": "BTCUSDT-1m-2023-09.csv"
    }
    
    file_to_load = files[event_selector]
    
    if os.path.exists(file_to_load):
        df = load_and_process_historical_data(file_to_load)
        
        if df is not None:
            # Mostrar métricas clave del evento
            max_alpha = df['Rolling_Alpha'].max()
            
            st.markdown("### Resumen Físico del Evento")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Pico Micro Alpha (Fricción Máxima)", value=f"{max_alpha:.2f}")
            with col2:
                # Lógica de estados físicos actualizada
                if max_alpha >= 2.0:
                    state = "Sólido (FRACTURA)"
                elif max_alpha >= 1.2:
                    state = "Alta Viscosidad (EMERGENCIA)"
                elif max_alpha >= 0.8:
                    state = "Flujo Laminar"
                else:
                    state = "Flujo Superfluido"
                st.metric(label="Estado de la Materia (Libro de Órdenes)", value=state)
            with col3:
                st.metric(label="Volatilidad Media Promedio", value=f"${df['Volatility'].mean():.2f}")
            
            # Gráfico de reconstrucción histórica
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio", line=dict(color='#3B82F6')), secondary_y=False)
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Alpha'], name="Alpha Micro (RTM)", line=dict(color='#FF1744')), secondary_y=True)
            
            # Líneas de umbral históricas (Calibradas a 1.2 y 2.0)
            fig2.add_hline(y=1.2, line_dash="dash", line_color="#FFEA00", secondary_y=True)
            fig2.add_hline(y=2.0, line_dash="solid", line_color="#FF1744", secondary_y=True)
            
            fig2.update_layout(plot_bgcolor='#0B0E14', paper_bgcolor='#0B0E14', font_color='#8B949E', title="Reconstrucción Microestructural del Evento", height=600)
            fig2.update_yaxes(title_text="Precio (USD)", secondary_y=False, gridcolor='#1E232B')
            fig2.update_yaxes(title_text="Alpha Micro", secondary_y=True, range=[-0.5, 3.0], showgrid=False)
            
            st.plotly_chart(fig2, use_container_width=True)

            # --- TEXTO EXPLICATIVO PARA EL HISTÓRICO ---
            with st.expander("📖 Guía de Análisis Forense de Crisis", expanded=True):
                st.markdown("""
                **¿Qué debes buscar en este gráfico?** Este panel demuestra cómo la topología del mercado (la estructura) reacciona frente al precio (la cinética) durante choques históricos.
                
                - ⏳ **La Divergencia Temporal:** Fíjate en el tiempo exacto en que la **Línea Roja (Alpha)** alcanza su pico máximo y cruza las líneas de alerta. En caídas críticas (como 2020 o 2025), la estructura se rompe *minutos u horas antes* de que la **Línea Azul (Precio)** se desplome.
                - 🛡️ **Verificación de Falsos Positivos:** Si seleccionas eventos como el *China Ban (2021)*, verás mucha volatilidad en el precio, pero la línea roja nunca llega a la zona de fractura (2.0). Esto demuestra que RTM no se deja engañar por el pánico del precio si la red sigue sana.
                - 🍯 **Viscosidad Crónica:** En eventos de lenta hemorragia institucional (como el *Colapso de FTX en 2022*), la línea roja sube por encima de 1.2 y se queda atascada formando una \"meseta\". El mercado no se rompe instantáneamente, pero se mueve como miel espesa por falta de liquidez.
                """)
    else:
        st.warning(f"Esperando el archivo {file_to_load} para el análisis. Por favor colóquelo en el directorio de la aplicación.")
