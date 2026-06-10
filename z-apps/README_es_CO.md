# z-apps — Herramientas de Monitoreo Estructural RTM

Esta carpeta contiene **aplicaciones Streamlit funcionales** que demuestran los principios RTM en monitoreo estructural en tiempo real. Ambas aplicaciones fueron reconstruidas desde cero (v3, mayo de 2026) tras una auditoría adversarial independiente (Red Team, Claude Opus 4.6 Extended Thinking) que invalidó varias afirmaciones de versiones anteriores. Estas aplicaciones muestran **únicamente lo que sobrevivió**.

---

## Aplicaciones

### 1. Radar Estructural Atmosférico (`atmospheric-monitor/`)

> **AVISO:** Prueba de concepto estrictamente académica. No es un sistema oficial de alertas meteorológicas. No abusa, envía spam ni realiza scraping masivo de APIs de datos comerciales.

**Módulos:**

| Módulo | Función | Fuente de Datos |
|--------|---------|-----------------|
| **TORNADO VORTEX RADAR** | Clasificación TOR en vivo mediante proxy RTM α | API del NWS (api.weather.gov) |
| **COHERENCIA MULTI-ESCALA** | Monitoreo de σ inter-escala | Open-Meteo (horario, 7 días) |
| **REFERENCIA SISMOLÓGICA** | Anclaje de calibración + tabla por tipo de falla | Catálogos publicados |
| **HALLAZGOS DEL RED TEAM** | Transparencia total de la auditoría | Red Team, abril de 2026 |

**Tornado Vortex Radar (destacado — hallazgo más sólido en todo el corpus RTM):**
- Obtiene alertas activas de tornado del NWS en tiempo real
- Calcula el proxy α = log₁₀(VEL_estimada) / log₁₀(L_polígono) a partir del texto de la alerta
- Clasifica usando el umbral calibrado por TorNet: α > 0.74 → clase de tornado confirmado
- Mapa oscuro con polígonos de advertencia codificados por color y tabla de clasificación
- Referencia: d = 0.96, CV AUC = 0.751, α subsume VEL (ΔAUC = 0.000), circularidad 91% eliminada

**Coherencia Multi-Escala (métrica novedosa — sobreviviente del Red Team):**
- Calcula α en 4 escalas temporales (1h, 3h, 6h, 12h) simultáneamente
- Rastrea σ inter-escala: estados de crisis → 0.03 (todas las escalas acopladas), normal → 0.31 (independientes)
- La misma separación de 10x observada en colapsos financieros (hallazgo inter-dominio)

**Lo que se eliminó de versiones anteriores:** Rastreador de huracanes (α circular con viento, ρ = 0.957, 13 pruebas), simulaciones históricas de huracanes (curvas preprogramadas, no predicciones), dinámica oceánica (datos sintéticos), lenguaje de tipo "EVACUAR".

**Ejecutar:**
```bash
cd atmospheric-monitor
pip install -r requirements.txt
streamlit run app_rtm.py
```

---

### 2. Radar Estructural Económico (`cryptocurrency_monitor/`)

> **DESCARGO DE RESPONSABILIDAD:** Herramienta de análisis topológico académica y de solo lectura. NO ejecuta operaciones, NO mina criptomonedas y NO constituye asesoramiento financiero. Precisión de predicción de colapsos fuera de muestra: 25%.

**Módulos:**

| Módulo | Función | Fuente de Datos |
|--------|---------|-----------------|
| **COHERENCIA MULTI-ESCALA** | σ inter-escala (en vivo + histórico) | API de Kraken (horario) + CSVs de Binance |
| **MICROESTRUCTURA EN VIVO** | Monitoreo de α en tiempo real (4 activos) | API de Kraken (1 min) |
| **LABORATORIO FORENSE** | Anatomía de colapsos históricos (post-hoc) | CSVs de Binance (1 min) |
| **FÍSICA DE MERCADOS** | Colas pesadas + escalamiento de recuperación | Resultados convergentes |
| **HALLAZGOS DEL RED TEAM** | Transparencia total de la auditoría | Red Team, abril de 2026 |

**Coherencia Multi-Escala (destacado — la única métrica económica RTM genuinamente novedosa):**
- En vivo: obtiene 14 días de datos horarios de Kraken, calcula α en escalas de 1h/3h/6h/12h
- Histórico: carga CSVs de Binance (1 min), calcula α en escalas de 1/5/15/60 min
- Rastrea σ inter-escala con indicador, serie temporal y valores de referencia
- Meses de colapso σ = 0.031-0.034 vs control σ = 0.310 (separación de 10x)

**Eventos forenses (etiquetados como post-hoc, no como predicción prospectiva):**

| Evento | Fecha | Hallazgo | Estado |
|--------|-------|----------|--------|
| **Colapso de FTX** | Nov 2022 | Viscosidad crónica (α ≈ 1.2, 4 días) | Forense |
| **Jueves Negro** | Mar 2020 | Bifurcación súbita (α = 1.76) | Forense |
| **Prohibición China** | May 2021 | Turbulencia, sin fractura (α = 1.33) | Forense |
| **Grupo de Control** | Sep 2023 | Laminar (α ≈ 0.45, cero falsos positivos) | Confirmado |
| **Fallo de Binance** | Oct 2025 | Anomalía técnica, no colapso fundamental | No validado |

**Lo que se eliminó de versiones anteriores:** Comando "SALIR DE LOS MERCADOS", encuadre como "sistema de alerta temprana de colapsos", afirmación de "advertencia de FTX con 96 horas de anticipación", afirmación de "predicción de octubre con 15 horas de anticipación", todo lenguaje que implicara señales operativas de trading.

**Ejecutar:**
```bash
cd cryptocurrency_monitor
pip install -r requirements.txt
streamlit run app_crypto.py
```

---

## La Métrica de Coherencia Multi-Escala (σ) — Compartida entre Ambas Aplicaciones

Ambas aplicaciones comparten el mismo hallazgo novedoso como módulo destacado: **Coherencia Multi-Escala**.

El concepto: calcular α en múltiples escalas temporales simultáneamente. Si todas las escalas muestran el mismo α (σ bajo), el sistema se encuentra en un estado coherente y acoplado — una transición de fase está en curso. Si cada escala muestra un α diferente (σ alto), el sistema opera con normalidad — no hay transición de fase.

| Rango de σ | Estado | Significado Atmosférico | Significado Financiero |
|------------|--------|-------------------------|------------------------|
| σ < 0.05 | **HIPER-COHERENTE** | Todas las escalas acopladas — crisis estructural | Todas las escalas acopladas — firma de colapso |
| 0.05 < σ < 0.15 | **ELEVADO** | Escalas acoplándose — monitorear | Escalas acoplándose — vigilar |
| σ > 0.15 | **NORMAL** | Escalas independientes | Escalas independientes |

**Valores de referencia (del análisis BTC del Red Team):**
- Crisis (COVID, marzo de 2020): σ = 0.031
- Crisis (FTX, noviembre de 2022): σ = 0.034
- Control (septiembre de 2023): σ = 0.310

Esta métrica es nativa de RTM: no es medida por ningún indicador meteorológico o financiero estándar. La consistencia inter-dominio (atmósfera + mercados) es la contribución empírica central del marco de monitoreo RTM.

---

## Stack Tecnológico

| Componente | Biblioteca |
|------------|------------|
| Framework de UI | Streamlit |
| Visualización | Plotly, Folium (mapas) |
| Procesamiento de datos | Pandas, NumPy |
| Datos financieros | ccxt (API de Kraken) |
| Datos meteorológicos | API de Open-Meteo |
| Alertas de tornado | API del NWS (api.weather.gov) |
| Tipografía | JetBrains Mono, Inter |
| Sistema de diseño | Paleta GitHub-dark |

**Requisitos:**
- Python 3.8+
- Conexión a internet (para feeds de datos en vivo)
- Sin dependencias pesadas como scipy

---

## Relación con el Corpus RTM

| Aplicación | Documento Fuente | Puntuación | Hallazgo Clave Sobreviviente |
|------------|------------------|------------|------------------------------|
| Radar Atmosférico | Doc 013 (Meteorología Rítmica) | 68% | Tornado d = 0.96, α subsume VEL |
| Radar Económico | Doc 015 (Economía Rítmica) | 68% | σ Multi-Escala: 10x colapso vs control |

Estas aplicaciones no introducen teoría nueva — empaquetan métricas RTM validadas en interfaces de monitoreo en tiempo real. Las versiones anteriores contenían afirmaciones que fueron invalidadas por la auditoría del Red Team. Las aplicaciones v3 contienen únicamente lo que sobrevivió.

---

## Cita

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

## Licencia

© 2026 Álvaro José Quiceno Rendón
Distribuido bajo [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Nota: **Utilice el identificador DOI de Zenodo más reciente.**
