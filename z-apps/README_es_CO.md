# z-apps — Aplicaciones RTM Funcionales

Esta carpeta contiene **aplicaciones Streamlit operativas** que demuestran los principios RTM en escenarios de monitoreo en tiempo real. Estas no son simulaciones ni validaciones — son herramientas operacionales diseñadas para detectar transiciones de fase en sistemas atmosféricos y financieros.

---

## Aplicaciones

### 1. Monitor Atmosférico (`atmospheric-monitor/`)

Un sistema de inteligencia climática de módulo dual.

**Módulos:**

| Módulo | Función | Fuente de Datos |
|--------|---------|-----------------|
| **EXTREMOS CLIMÁTICOS** | Monitoreo de eventos meteorológicos extremos en tiempo real | APIs meteorológicas (Open-Meteo, etc.) |
| **RASTREADOR DE HURACANES** | Predicción de Intensificación Rápida mediante α | IBTrACS + feeds simulados en tiempo real |

**Características del Rastreador de Huracanes:**
- **Cálculo de α en Vivo:** Calcula el exponente de acoplamiento viento-presión en tiempo real
- **Detección de Fase:** LAMINAR (α > 1.5) → DECAIMIENTO (1.25 < α < 1.5) → FRACTURA (α < 1.25)
- **Contador Regresivo:** Horas T-MINUS hasta la explosión de intensidad predicha
- **Repeticiones Históricas:** Otis (2023), Milton (2024), Patricia (2015)

**Idea Clave RTM:** La caída de α (Fractura Topológica) precede a las alertas oficiales del NHC por 12-14 horas. La aplicación demuestra este tiempo de anticipación predictivo con datos históricos anotados.

**Ejecución:**
```bash
cd atmospheric-monitor
pip install -r requirements.txt
streamlit run app_rtm.py
```

---

### 2. Monitor de Criptomonedas (`cryptocurrency_monitor/`)

Un radar de mercados financieros que utiliza la física de coherencia RTM.

**Módulos:**

| Módulo | Función |
|--------|---------|
| **RADAR EN VIVO** | Monitoreo en tiempo real de BTC/ETH mediante la API de Binance |
| **SALUD SISTÉMICA** | Panel de salud del mercado con diagnósticos basados en α |
| **ANÁLISIS FORENSE** | Repeticiones históricas de caídas con anotaciones RTM |
| **FÍSICA DE MERCADO** | Colas gruesas, leyes de potencia, calculadora de escala de recuperación |

**Eventos Forenses Incluidos:**

| Evento | Fecha | Hallazgo RTM |
|--------|-------|--------------|
| **Colapso de FTX** | Nov 2022 | Viscosidad Crónica (α ≈ 1.2) durante 4 días — advertencia de 100h |
| **Jueves Negro** | Mar 2020 | Bifurcación de Fase (α = 1.76) — advertencia de 60 min |
| **Prohibición de China** | May 2021 | Turbulencia de Alta Energía (α = 1.33) — recuperación instantánea predicha |
| **Grupo de Control** | Sep 2023 | Flujo Laminar (α ≈ 0.45) — tasa de falsas alarmas del 0% |

**Características de Física de Mercado:**
- **Ley Cúbica Inversa:** α global ≈ 2.97 (las colas gruesas son estructurales, no anomalías)
- **Calculadora de Recuperación:** Utiliza la pendiente ODR robusta (3.59) para estimar el tiempo de recuperación desde la caída
- **Distribución de α:** Histograma simulado de 10 años que muestra la probabilidad de fractura

**Ejecución:**
```bash
cd cryptocurrency_monitor
pip install -r requirements.txt
streamlit run app.py
```

---

## El Exponente de Coherencia RTM (α) en Ambas Aplicaciones

Ambas aplicaciones utilizan la misma métrica fundamental: el **Exponente de Coherencia RTM (α)**, que mide con qué eficiencia un sistema transporta información/energía a través de las escalas.

| Rango de α | Estado | Significado Atmosférico | Significado Financiero |
|------------|--------|-------------------------|------------------------|
| α > 1.5 | **LAMINAR** | Atmósfera estable | Flujo de mercado saludable |
| 1.2 < α < 1.5 | **DECAIMIENTO** | Debilitamiento estructural | Advertencia de viscosidad |
| α < 1.2 | **FRACTURA** | Intensificación rápida inminente | Caída/bifurcación inminente |
| α > 2.0 | **BIFURCACIÓN** | (No aplica — los huracanes se intensifican) | Estructura de mercado en ruptura |

**La idea clave:** α mide la *coherencia topológica*, no la actividad cinética. Los vientos de un huracán pueden estar en calma mientras α colapsa (prediciendo una explosión futura). El precio de un mercado puede estar estable mientras α sube (prediciendo una caída futura). La geometría estructural se rompe *antes* de que aparezcan los síntomas observables.

---

## Pila Tecnológica

| Componente | Biblioteca |
|------------|------------|
| Marco de UI | Streamlit |
| Visualización | Plotly, Folium (mapas) |
| Procesamiento de Datos | Pandas, NumPy |
| Datos Financieros | ccxt (API de Binance) |
| Datos Meteorológicos | Open-Meteo, APIs personalizadas |

**Requisitos:**
- Python 3.8+
- Conexión a internet (para datos en vivo)
- ~500 MB de RAM por aplicación

---

## Archivos de Datos Incluidos

**Monitor de Criptomonedas:**
- `BTCUSDT-1m-2020-03.csv` — Jueves Negro (caída COVID)
- `BTCUSDT-1m-2021-05.csv` — Shock por prohibición de China
- `BTCUSDT-1m-2022-11.csv` — Colapso de FTX
- `BTCUSDT-1m-2023-09.csv` — Grupo de control (período estable)
- `BTCUSDT-1m-2025-10.csv` — Anomalía de falla de Binance
- `crash_alpha_analysis.csv` — Valores α precalculados para eventos forenses

**Monitor Atmosférico:**
- `RTM CLIMATE-Global-Architecture-Vision.pdf` — Documentación de arquitectura del sistema

---

## Avisos Legales

### Rastreador de Huracanes
```
⚠️ HERRAMIENTA EXPERIMENTAL — NO APTA PARA DECISIONES DE EMERGENCIA
Esta aplicación es una demostración de investigación de la física atmosférica RTM.
Para decisiones que involucren la seguridad de vidas, siga siempre las indicaciones
oficiales del NHC y de las autoridades locales de emergencia.
```

### Monitor de Criptomonedas
```
⚠️ NO ES ASESORAMIENTO FINANCIERO
Esta aplicación demuestra la física de mercados RTM con fines educativos.
El rendimiento pasado no garantiza resultados futuros.
No tome decisiones de inversión basándose únicamente en esta herramienta.
```

---

## Relación con el Corpus RTM

Estas aplicaciones operacionalizan hallazgos de:

| Aplicación | Artículos Fuente |
|------------|-----------------|
| Rastreador de Huracanes | Artículo 013 (RTM-Atmo), Apéndice F (FAR de Tornados) |
| Monitor de Criptomonedas | Artículo 015 (Economía Rítmica), validaciones de Fase 2 |

Las aplicaciones no introducen teoría nueva — empaquetan métricas RTM validadas en interfaces de tiempo real utilizables.

---

## Citación

Si utilizas este trabajo, por favor cita:

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

---

## Licencia

© 2026 Álvaro José Quiceno Rendón  
Distribuido bajo [Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
Nota: **Utiliza el identificador DOI de Zenodo más reciente.**
