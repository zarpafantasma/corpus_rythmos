## Aplicaciones y Herramientas Predictivas RTM (z.apps)

Esta sección aloja herramientas operativas desarrolladas bajo el marco RTM. Estas aplicaciones traducen las matemáticas de escalamiento en interfaces de monitoreo estructural.

### Objetivos Principales

- **Monitoreo Estructural:** Seguimiento del coeficiente α y la coherencia inter-escala (σ) como descriptores del estado del sistema.
- **Análisis Forense:** Reconstrucción post-hoc de patrones estructurales durante eventos de crisis conocidos.
- **Transparencia:** Cada módulo divulga los resultados de la auditoría del Red Team, incluyendo lo que falló.

---

### 1. RTM CLIMATE (Radar Estructural Atmosférico)

La primera implementación operativa de la Teoría RTM aplicada a la dinámica atmosférica. Reconstruida desde cero (v3, mayo de 2026) para mostrar únicamente los hallazgos que sobrevivieron a la auditoría adversarial independiente.

**Módulos:**

**(a) Radar de Vórtice de Tornado (destacado).** Obtiene alertas activas de tornado de la API del NWS (api.weather.gov) en tiempo real. Extrae la velocidad de rotación estimada del texto de la alerta, estima el tamaño de la celda a partir del polígono de advertencia y calcula un proxy RTM α = log₁₀(VEL) / log₁₀(L). Clasifica eventos utilizando el umbral calibrado por TorNet (α > 0.74 → clase de tornado confirmado). Este es el hallazgo empírico más sólido en todo el corpus RTM: d = 0.96, CV AUC = 0.751, α subsume completamente la velocidad bruta (ΔAUC = 0.000), y el 91% de la señal discriminativa es alcanzable sin entrada de velocidad (circularidad parcialmente eliminada). **Nota sobre los datos:** El proxy α aquí se aproxima a partir del texto de alertas del NWS, no de radar de doble polarización directo — tratar como indicativo.

**(b) Monitor de Coherencia Multi-Escala (novedoso).** Calcula α simultáneamente en ventanas de 1h, 3h, 6h y 12h utilizando datos atmosféricos de Open-Meteo. La desviación estándar inter-escala (σ) es la métrica nativa de RTM que sobrevivió: durante crisis estructurales, σ → 0.03 (todas las escalas se acoplan simultáneamente); durante condiciones normales, σ → 0.31 (las escalas operan de forma independiente). Ninguna herramienta meteorológica estándar mide esto. Este es un hallazgo inter-dominio — la misma separación de 10x aparece en colapsos financieros (BTC).

**(c) Referencia Sismológica.** Módulo de referencia estático que muestra el anclaje de calibración RTM: ruptura sísmica α = 1.007 ± 0.016 (balístico, R² = 0.987, 51 terremotos). Incluye el hallazgo novedoso de propagación sub-balística en fallas normales (extensionales) (α = 0.865, IC 95% excluye 1.0). Muestra la estructura de puntos fijos del grupo de renormalización (α = 0, 0.5, 1.0, 2.0).

**(d) Hallazgos del Red Team.** Transparencia total: 8 hallazgos que funcionan, 4 que fallaron. La circularidad del α de huracanes (ρ = 0.957, 13 pruebas) documentada explícitamente.

**Puntuación de la auditoría del Red Team: 68%** — Porcentaje de afirmaciones empíricas en el Doc 013 que sobrevivieron a la auditoría adversarial independiente (Red Team, abril de 2026). Tornado: intacto. α de huracanes: circular.

> **AVISO:** Prueba de concepto estrictamente académica. No es un sistema oficial de alertas meteorológicas. No abusa, envía spam ni realiza scraping masivo de APIs de datos comerciales.

- **Estado:** Operativo / Prueba de Concepto (v3 — Reconstrucción Post-Red Team)
- **Dominio:** Termodinámica Atmosférica y Física Climática Multi-escala
- **Abrir aplicación:** [Consola RTM CLIMATE MONITOR](https://corpusrythmos-atmospheric-monitor.streamlit.app/)

---

### 2. RTM ECONOMIC MONITOR (Radar Estructural Cripto)

La segunda implementación operativa de la Teoría RTM, que aplica principios topológicos a la microestructura de mercados financieros. Reconstruida desde cero (v3, mayo de 2026) para dar protagonismo a la única métrica que sobrevivió a las pruebas adversariales como genuinamente novedosa.

**Módulos:**

**(a) Monitor de Coherencia Multi-Escala (destacado — sobreviviente del Red Team).** La única métrica económica RTM que pasó las pruebas adversariales como genuinamente novedosa. Calcula α en agregaciones de 1h, 3h, 6h y 12h (en vivo desde datos horarios de Kraken) o en agregaciones de 1-min, 5-min, 15-min y 60-min (históricos desde CSVs de Binance). Rastrea la desviación estándar inter-escala (σ). Durante los meses de colapso de BTC (COVID marzo de 2020, FTX noviembre de 2022), σ = 0.031-0.034; durante el mes de control (septiembre de 2023), σ = 0.310 — 10x más coherente durante los colapsos. Todas las escalas se sincronizan simultáneamente durante las transiciones de fase. Ningún indicador financiero estándar mide esto.

**(b) Monitor de Microestructura en Vivo.** Rastrea el coeficiente de acoplamiento volumen-volatilidad (α) en tiempo real desde Kraken para BTC, ETH, SOL, XRP. Incluye panel de salud sistémica (comparación de α entre 4 activos). Enmarcado como un **descriptor estructural, no como una señal de predicción de colapso.** Precisión fuera de muestra (25%) divulgada en todos los elementos relevantes de la interfaz.

**(c) Laboratorio Forense.** Reconstrucción post-hoc de 5 eventos históricos: FTX (viscosidad crónica), COVID (bifurcación súbita), Prohibición China (turbulencia, sin fractura), Grupo de Control (cero falsos positivos), Octubre 2025 (fallo técnico de Binance, no colapso fundamental). Cada evento etiquetado como **forense (post-hoc)**, no como predicción prospectiva.

**(d) Física de Mercados.** Resultados convergentes: ley cúbica inversa (α = 2.966 ± 0.236, consistente con Gabaix et al. 2003), escalamiento del tiempo de recuperación (τ ∝ D^3.59), calculadora de probabilidades (gaussiana vs. ley de potencias).

**(e) Hallazgos del Red Team.** Transparencia total: 6 hallazgos que funcionan, 5 que fallaron. Precisión de predicción de colapsos fuera de muestra (25%) documentada explícitamente.

**Puntuación de la auditoría del Red Team: 68%** — Porcentaje de afirmaciones empíricas en el Doc 015 que sobrevivieron a la auditoría adversarial independiente (Red Team, abril de 2026). 32% invalidado o degradado.

> **DESCARGO DE RESPONSABILIDAD:** Herramienta de análisis topológico académica y de solo lectura. NO ejecuta operaciones, NO mina criptomonedas y NO constituye asesoramiento financiero. La precisión de predicción de colapsos fuera de muestra es del 25%. Esto no es una señal operativa de trading.

- **Estado:** Operativo / Prueba de Concepto (v3 — Reconstrucción Post-Red Team)
- **Dominio:** Finanzas Cuantitativas y Topología de Mercados
- **Abrir aplicación:** [Consola RTM ECONOMIC MONITOR](https://corpusrythmos-cryptomonitor.streamlit.app/)
