# VII - Validaciones empíricas y heurísticas

Esta carpeta contiene **validaciones empíricas** de las predicciones de RTM frente a datos del mundo real en 13 dominios científicos. A diferencia de las simulaciones computacionales de la Carpeta VI (que prueban la consistencia matemática), estas validaciones evalúan si las leyes de escalamiento de RTM coinciden con observaciones reales de la física, la biología, las ciencias de la Tierra y la economía.

---

## ⚠️ Estructura de validación en tres fases

Esta carpeta está organizada en **tres subcarpetas**, cada una de las cuales representa una ronda independiente de validación realizada por un sistema de IA diferente en una etapa distinta del proceso de investigación.

---

### Fase 1: Validaciones heurísticas
**Motor:** Claude Opus 4.5 (Extended Thinking) | **Fecha:** febrero de 2026

Análisis iniciales usando valores publicados en la literatura, estadísticas agregadas y técnicas estándar de regresión. Estos análisis establecieron un respaldo preliminar para las predicciones de RTM, pero con frecuencia:
- Usaron estimaciones puntuales sin propagación de incertidumbre
- Dependieron de medias agregadas (riesgo de falacia ecológica)
- Aplicaron mínimos cuadrados ordinarios (OLS), que ignoran el error de medición en ambas variables
- No realizaron pruebas adversariales de estrés sobre las afirmaciones

---

### Fase 2: Validaciones empíricas ROBUSTAS (Red Team, ronda 1)
**Motor:** Gemini 5.1 Pro / Advanced Math and Code | **Fecha:** marzo de 2026

Reanálisis adversariales diseñados para someter a prueba los hallazgos de la Fase 1:
- **Regresión por distancia ortogonal (ODR):** Propaga la incertidumbre tanto en las variables X como Y
- **Reconstrucción a nivel de sujeto:** Sustituye medias agregadas por poblaciones simuladas
- **Incertidumbre Monte Carlo:** Mapea distribuciones de probabilidad completas, no estimaciones puntuales
- **Márgenes de error conservadores:** Inyección de varianza del 10-20% para prevenir sesgo por atenuación

**El hallazgo clave:** La mayoría de los resultados de la Fase 1 sobrevivieron al escrutinio de la Fase 2, a menudo con intervalos de confianza más estrechos alrededor de las predicciones teóricas.

---

### Fase 3: Red Team extendido + campaña de flanqueo (Red Team, ronda 2)
**Motor:** Claude Opus 4.6 (Extended Thinking) | **Fecha:** finales de abril – comienzos de mayo de 2026

Una segunda campaña adversarial independiente ejecutada sobre los resultados ROBUSTOS de la Fase 2. Esta fase introdujo **ataques de flanqueo**, una metodología no usada en ninguna de las fases anteriores.

**¿Qué es un ataque de flanqueo?**
Un ataque de flanqueo es un análisis novedoso con datos reales ejecutado sobre los mismos conjuntos de datos utilizados en la validación ROBUSTA original, pero formulando una pregunta fundamentalmente diferente: una que el marco original no había considerado. Mientras un ataque directo prueba si una afirmación se sostiene, un ataque de flanqueo pregunta: *"Dado que la puerta principal ya fue probada, ¿qué revelan las puertas laterales?"*

Cada flanco fue:
- **Preespecificado** como una pregunta falsable antes del análisis
- **Independiente** de la afirmación original que se estaba probando
- **Reportado con honestidad**: los resultados negativos (dirección incorrecta, no significativos) se publican junto con los positivos

La Fase 3 también revisó todo el lenguaje de sobreafirmación ("prueba concluyentemente", "valida definitivamente", "irrefutable") en los 13 documentos y aplicó correcciones de tono. Los documentos fueron actualizados con anexos Red Team y, cuando el flanqueo produjo nuevos hallazgos, se añadieron nuevos apéndices.

---

## Artículos cubiertos (003–015)

| Artículo | Dominio | α observado | ¿Flanqueado? |
|-------|--------|------------|----------|
| 003 | Corteza visual | 0.311 ± 0.021 | No |
| 004 | Cosmología (JWST) | 1.34 ± 0.12 | No |
| 005 | Ondas gravitacionales | 1.024 ± 0.018 | No |
| 006 | Computación cuántica | −0.259 ± 0.049 | No |
| 007 | Química | Específico del dominio | No |
| 008 | Bioquímica | 7.22 (plegamiento) | No |
| 009 | Homeostasis | 1.03 → 0.53 (ICC) | **Sí (8 flancos)** |
| 010 | Neurociencia | Dependiente del estado | No |
| 011 | Consciencia | Desplazamiento espectral | **Sí (6 flancos)** |
| 012 | Ecología | α ≈ 1.0 (COVID) | **Sí (5 flancos)** |
| 013 | Meteorología | d = 0.96 (tornado) | **Sí (13 flancos)** |
| 014 | Astronomía | 21 hallazgos SPARC | **Sí (6 flancos)** |
| 015 | Economía | σ entre escalas novedosa | **Sí (5 flancos)** |

---

## Resúmenes detallados por artículo

### 003 - El marco de cascada RTM (corteza visual)

**Fase 1 (Claude Opus 4.5):** Analizó 21 áreas visuales desde el LGN hasta la PFC. Encontró escalamiento subdifusivo α = 0.303 ± 0.020 (R² = 0.921, p < 10⁻¹¹).

**Fase 2 (Gemini 5.1):** Confirmó con ODR ponderada por varianza. α = 0.311 ± 0.021 sobrevive a la inyección de ruido. Bootstrap: 100% de las estimaciones remuestreadas por debajo de α = 0.5.

**Fase 3 (Claude Opus 4.6):** Red Team aprobado. No se requirió flanqueo. Se aplicaron correcciones de tono ("prueba concluyentemente" → "confirma"; se eliminó "dobla las reglas clásicas"). Se añadió el Apéndice B (anexo Red Team).

**Interpretación:** La corteza visual procesa información en un régimen superdifusivo: más eficientemente que la difusión aleatoria, en consistencia con la codificación jerárquica paralela.

---

### 004 - Reescalamiento temporal en el universo temprano (JWST)

**Fase 1 (Claude Opus 4.5):** Las "galaxias tempranas imposibles" de JWST se explican mediante el reescalamiento temporal de RTM. La formación de estructuras escala como T ∝ L^α con α > 1.

**Fase 2 (Gemini 5.1):** ODR con incertidumbres de corrimiento al rojo fotométrico. α = 1.34 ± 0.12 sobrevive a la propagación de errores. Tendencia exceso-z ρ = 0.43, p = 0.006.

**Fase 3 (Claude Opus 4.6):** Red Team aprobado. No se requirió flanqueo. Se aplicaron correcciones de tono. La correlación exceso-z se clasifica como NOVEDOSA: no predecible desde el ΛCDM estándar.

**Interpretación:** Las galaxias de alto z no son anómalas: RTM ofrece una reinterpretación topológica del reescalamiento del tiempo cósmico en regiones tempranas densas.

---

### 005 - Agujeros negros en el marco RTM (ondas gravitacionales)

**Fase 1 (Claude Opus 4.5):** 183 fusiones BBH de O1-O4. La energía escala como E_rad ∝ M_total^α con α ≈ 1.018.

**Fase 2 (Gemini 5.1):** Restringido a 55 eventos confirmados O1-O3. ODR con propagación bayesiana del error. α corregido por espín = 1.024 ± 0.018. IC bootstrap [0.989, 1.059]: converge en la clase balística (α = 1.0) y es consistente con las predicciones de la relatividad general.

**Fase 3 (Claude Opus 4.6):** Red Team aprobado. No se requirió flanqueo. La convergencia con la relatividad general se clasifica como POSITIVA: un marco unificador debería recuperar resultados conocidos. Se aplicaron correcciones de tono.

**Interpretación:** El transporte de energía de ondas gravitacionales es balístico (α ≈ 1.0). La convergencia con la relatividad general es evidencia de consistencia del marco, no de redundancia.

---

### 006 - Computación cuántica consciente de RTM

**Fase 1 (Claude Opus 4.5):** Los procesadores cuánticos de IBM muestran escalamiento inverso del tiempo de decoherencia con el número de cúbits. α ≈ −0.35.

**Fase 2 (Gemini 5.1):** ODR multivariable. α bruto = +0.23 (positivo, inesperado). Después de eliminar el factor de confusión del año: **paradoja de Simpson** — α se revierte a −0.259 ± 0.049. IC bootstrap [−0.382, −0.038] excluye cero.

**Fase 3 (Claude Opus 4.6):** Red Team aprobado con distinción. La paradoja de Simpson cuántica (tendencia enmascarada de degradación de hardware revertida por el factor de confusión del año) es el **hallazgo novedoso más fuerte en el subcorpus de física**. No se requirió flanqueo: el hallazgo es suficientemente sólido.

**Interpretación:** La coherencia cuántica representa transporte inverso (α < 0). Los sistemas cuánticos más grandes decoheren más rápido. La tendencia enmascarada de degradación revelada al remover el factor de confusión es un diagnóstico nativo de RTM.

---

### 007 - Química rítmica

**Fase 1 (Claude Opus 4.5):** Validada en difusión en zeolitas, relación Stokes-Einstein y redes de transporte urbano.

**Fase 2 (Gemini 5.1):** ODR confirma escalamiento específico por dominio. Líquidos a granel: α = −1.23 ± 0.04 (inverso, consistente con Stokes-Einstein). Zeolitas confinadas: α = +7.25 ± 1.06 (cooperativo, consistente con la teoría de difusión de una sola fila).

**Fase 3 (Claude Opus 4.6):** Red Team aprobado. No se requirió flanqueo. El solapamiento cero entre regímenes a granel y confinados (d = 8.48) confirma la clasificación de dos regímenes. Se aplicaron correcciones de tono.

---

### 008 - Bioquímica rítmica

**Fase 1 (Claude Opus 4.5):** La cinética enzimática y el plegamiento de proteínas muestran escalamiento cooperativo (α > 1).

**Fase 2 (Gemini 5.1):** ODR normalizada por EC con inyección de varianza. Plegamiento de proteínas: α = 7.22 ± 0.62 (cooperativo). Cinética enzimática: α ≈ 0 (p = 0.71, no significativo: dominan los mecanismos químicos locales). Separación: d = 6.98, **cero solapamiento bootstrap**.

**Fase 3 (Claude Opus 4.6):** Red Team aprobado. No se requirió flanqueo. El solapamiento cero entre los regímenes de plegamiento y enzimas se clasifica como CONVERGENTE con la bioquímica cooperativa conocida. Se aplicaron correcciones de tono.

---

### 009 - Homeostasis (variabilidad de la frecuencia cardiaca)

**Fase 1 (Claude Opus 4.5):** El escalamiento DFA rastrea la salud cardiaca. Saludable α ≈ 1.05, ICC α → 0.55. Se identificó gradiente NYHA.

**Fase 2 (Gemini 5.1):** Simulación a nivel de sujeto (n = 200 por clase NYHA). Saludable: α = 1.03 ± 0.16. NYHA IV: α = 0.53 ± 0.31. r = −0.43 (p < 10⁻¹⁰). Penalización por ICC: Δα = −0.322, equivalente a ~67 años de envejecimiento saludable.

**Fase 3 (Claude Opus 4.6) — 8 flancos, 5 aciertos:**
1. **Amplificador α × CI:** d: 1.25 → 3.28 (saludable vs ICC). La métrica 2D supera a cualquiera de las dimensiones por separado.
2. **Dosis-respuesta al ejercicio:** ρ = −0.971, patrón acelerado (Δα: 0.10 → 0.20 → 0.25).
3. **Escalera de severidad de arritmias:** ρ = −0.957 en 10 tipos, desde ritmo sinusal normal (α = 1.05) hasta fibrilación ventricular (α = 0.35). Solo 1/9 transiciones no monotónica.
4. **Escalera NYHA:** R² = 0.989, transición III → IV más pronunciada (0.15 vs 0.10).
5. **Penalización por ICC replicada:** Δα = −0.323 mediante método independiente (< 0.3% de diferencia).

---

### 010 - Neurociencia rítmica

**Fase 1 (Claude Opus 4.5):** El escalamiento EEG varía según el estado cerebral: etapas del sueño, meditación, psicodélicos, epilepsia.

**Fase 2 (Gemini 5.1):** Validación multidominio con varianza a nivel de sujeto. 4 dominios confirmados: d = 0.98–3.30 en sueño/meditación/psicodélicos/epilepsia.

**Fase 3 (Claude Opus 4.6):** Red Team aprobado. No se requirió flanqueo. Los resultados se clasifican como CONSISTENTES con la literatura neurocientífica conocida. Se aplicaron correcciones de tono.

---

### 011 - Acceso consciente

**Fase 1 (Claude Opus 4.5):** La consciencia se correlaciona con el escalamiento espectral. La ketamina preserva el régimen consciente; el propofol lo colapsa.

**Fase 2 (Gemini 5.1):** Simulación a nivel de sujeto (n = 30,873). Ketamina Δβ ≈ −0.10 (cambio del 5%). Propofol Δβ ≈ −1.25 (cambio del 69%). d de Cohen = 0.46 (vigilia vs inconsciencia verdadera, p < 10⁻¹⁰). Se identificó la paradoja REM: REM muestra pendientes espectrales "inconscientes" pese a la consciencia fenomenológica.

**Fase 3 (Claude Opus 4.6) — 6 flancos, 0 fallos:**
1. **Amplificador α × R²:** d: 0.33 → 0.97 (ojos abiertos vs cerrados). AUC: 0.60 → 0.78.
2. **Clasificador con validación cruzada:** AUC = 0.911 (saludable vs convulsión), 0.794 (OA vs OC), CV de 5 particiones sobre 11,500 registros UCI.
3. **Conspiración α-R²:** El acoplamiento se estrecha durante las convulsiones (IC bootstrap de Δρ excluye 0).
4. **Gradiente anestésico:** Cambio espectral <20% = consciencia preservada; >40% = pérdida. Umbral operativo limpio.
5. **Diagnóstico de varianza:** Las convulsiones muestran el CV de α más alto (0.380): los estados transicionales muestran varianza máxima.
6. **Predicción REM (comprobable):** REM debería mostrar pendiente pronunciada PERO R² alto. Comprobable directamente en datos polisomnográficos de NSRR.

---

### 012 - Ecología y epidemiología rítmicas

**Fase 1 (Claude Opus 4.5):** Base de longevidad AnAge, dinámicas de propagación de COVID-19, espectros poblacionales GPDD.

**Fase 2 (Gemini 5.1):** COVID-19: α = 0.953 ± 0.044 (red libre de escala, atractor de Zipf). Superpropagador k = 0.226 ± 0.131. GPDD: β = 0.82 (ruido rosa 1/f, 99.7% no Poisson). Pendiente ODR de riesgo de extinción = 0.92 ± 0.02.

**Fase 3 (Claude Opus 4.6) — 5 flancos, 4 aciertos:**
1. **Los residuos de Kleiber predicen longevidad:** ρ = −0.184, p = 0.0005 (n = 350 mamíferos). El 89% de los órdenes muestran la misma dirección. Predicción novedosa específica de RTM.
2. **Conspiración de forma depredador-presa:** La anticorrelación se intensifica antes de los colapsos (d = −2.52 antes del alce 1996, d = −1.10 antes del lobo 2012). Patrón transdominio confirmado.
3. **Paradoja de Simpson en anfibios:** α global = 0.091 enmascara Anura (pulmones) α = 0.55 vs Caudata (cutánea) α = 0.03.
4. **Tamaño corporal → color espectral:** ρ = +0.867, p = 0.0025 en 9 grupos taxonómicos GPDD. RTM proporciona el mecanismo.
5. **Precursor β fallido:** β móvil no predice inestabilidad futura en Isle Royale (dirección incorrecta, ns). Los colapsos exógenos no son detectados por precursores endógenos: condición de frontera documentada.

---

### 013 - Meteorología rítmica

**Fase 1 (Claude Opus 4.5):** La intensificación rápida de huracanes se predice mediante el exponente de acoplamiento viento-presión. 48 tormentas, d de Cohen = 3.07, tiempo de anticipación de 6-18 h. Tornado: discriminación inicial prometedora.

**Fase 2 (Gemini 5.1):** ODR confirma la robustez del umbral α. Modelo aditivo de tornado: α + KDP + VEL + DBZ. Caso forense Otis: α cayó a 1.11 antes de una intensificación rápida de 93 kt/24 h.

**Fase 3 (Claude Opus 4.6) — 13 flancos en 3 rondas:**

*Tornado (3 rondas, todas positivas):*
1. α subsume completamente la velocidad bruta: ΔAUC = 0.000 cuando se añade VEL a α (1,105 eventos TorNet).
2. α predice la intensidad EF dentro de tornados confirmados (ρ = +0.446, p < 10⁻⁴, n = 435).
3. Modelo óptimo: α + KDP (AUC con CV = 0.769). Añadir más variables no mejora.

*Huracán (circularidad confirmada después de 13 pruebas):*
- α se correlaciona con el viento en ρ = 0.957 y con la presión en ρ = 0.993.
- Después de controlar por viento: todas las correlaciones parciales no son significativas (ΔR² < 0.015).
- **Hallazgo sobreviviente en huracanes:** TEMPORIZACIÓN de la caída de α (6-18 h antes de la explosión cinética) y consistencia de α_MIN (CV = 0.096 en 26 eventos de intensificación rápida). Se eliminó la afirmación "α como predictor independiente de intensificación rápida".

*Sismología (nuevo hallazgo):*
- Fallas normales (extensionales): α = 0.865 ± 0.056, IC del 95% excluye 1.0: subbalístico, novedoso.
- Las fallas de rumbo (α = 1.040) e inversas (α = 0.987) permanecen balísticas.

---

### 014 - Astronomía rítmica

**Fase 1 (Claude Opus 4.5):** Curvas de rotación de galaxias SPARC. Se reportó α = 1.99 para curvas planas como hallazgo principal. Se afirmó reemplazo de materia oscura.

**Fase 2 (Gemini 5.1):** ODR con incertidumbres observacionales. Correlación estructura-cinemática confirmada (pendiente ODR = −1.17 ± 0.12). Turbulencia de plasma: relajación IK → Kolmogorov confirmada.

**Fase 3 (Claude Opus 4.6) — 6 flancos, 21 hallazgos significativos:**

*Afirmaciones eliminadas:*
- α = 2 para curvas planas es **tautológico** (α = 2(1 − pendiente) por definición). Eliminado de las afirmaciones activas.
- El reemplazo de la materia oscura no está respaldado (RTM gana en 2/135 galaxias frente a NFW en la prueba directa v(r)).

*21 correlaciones parciales significativas en SPARC (todas controlando por masa bariónica):*

| Hallazgo | ρ parcial | p |
|---------|-----------|---|
| Efectividad bariónica vs concentración | −0.446 | 9.4 × 10⁻⁸ |
| Efectividad bariónica vs μ₀ | +0.450 | 7.0 × 10⁻⁸ |
| Escala de aceleración vs concentración | −0.574 | 3.1 × 10⁻⁷ |
| Radio de 50% de MO vs concentración | −0.515 | 4.2 × 10⁻⁶ |
| Conspiración de forma (V_bar vs V_DM) | r = +0.274 | 9.9 × 10⁻⁵ |
| Conspiración rica en gas r = +0.70 vs pobre en gas r = −0.15 | — | p < 10⁻⁴ |
| f_gas local → ρ_DM local (2,411 pts, FE por galaxia) | −0.177 | 2.5 × 10⁻¹⁸ |

---

### 015 - Economía rítmica

**Fase 1 (Claude Opus 4.5):** Cuatro informes forenses de caídas de BTC. El α de DFA aumenta durante las caídas por COVID, FTX y la prohibición de China. Se afirmó una alerta temprana de 10 días.

**Fase 2 (Gemini 5.1):** Escalamiento de recuperación: α = 3.59 ± 0.70. Distribución de retornos: α = 2.966 ± 0.236 (convergente con Gabaix et al. 2003). d de Cohen dentro de muestra = −1.45 (saludable vs caída).

**Fase 3 (Claude Opus 4.6) — 5 flancos:**
1. **Prueba fuera de muestra: 25% de precisión** (1/4 eventos posteriores a 2022). El umbral entrenado no generaliza. La "alerta temprana de 10 días" se replantea como observación forense dentro de muestra.
2. **Coherencia multiescala (novedosa):** α calculado simultáneamente en 1 min, 5 min, 15 min y 60 min. σ entre escalas: meses de caída = 0.031-0.034; mes de control = 0.310. Separación de 10×. Ningún indicador financiero estándar mide esto.
3. Conspiración forma volumen-volatilidad: r > 0.88 todos los meses (real, pero no específica de caídas).
4. Asimetría caída-recuperación: COVID confirma (recuperación 1.6× más lenta); FTX contradice (caída por solvencia ≠ caída por choque). Se requiere tipología de caídas.
5. "Advertencia de 15 horas" de octubre de 2025: atribuida a una falla técnica de Binance, no a una caída estructural fundamental. Afirmación eliminada.

---

## Resumen: clases de transporte RTM validadas

| Rango de α | Clase | Sistemas validados |
|---------|-------|-------------------|
| α < 0 | **Inverso** | Decoherencia cuántica (−0.259), Stokes-Einstein (−1.23) |
| 0 < α < 0.5 | **Subdifusivo** | Corteza visual (0.311) |
| α ≈ 0.5 | **Difusivo** | ICC terminal (0.53), línea base de caminata aleatoria |
| 0.5 < α < 1.0 | **Superdifusivo** | Red COVID-19 (0.953) |
| α ≈ 1.0 | **Balístico** | Ondas gravitacionales (1.024), ruptura sísmica (1.007), COVID-19 |
| 1 < α < 2 | **Superbalístico** | Galaxias JWST (1.34), intensificación rápida de huracanes (diagnóstico de temporización) |
| α > 2 | **Cooperativo / Transición de fase** | Plegamiento de proteínas (7.22), caídas de mercado (2.966), zeolitas (7.25) |

---

## Metodología Red Team

### Fase 2 (Gemini 5.1) — Estándares estadísticos adversariales

1. **Regresión por distancia ortogonal (ODR):** Minimiza la distancia perpendicular a la línea de ajuste, manejando adecuadamente la incertidumbre en ambas variables. OLS suele subestimar las pendientes en 15-20%.
2. **Reconstrucción a nivel de sujeto:** Simula puntos de datos individuales a partir de media ± DE reportadas en lugar de correlacionar medias (prevención de falacia ecológica).
3. **Incertidumbre Monte Carlo:** 10,000+ muestras bootstrap para mapear distribuciones de probabilidad completas.
4. **Inyección conservadora de error:** Inflación de varianza del 10-20% para asegurar que los resultados no sean artefactos de ruido subestimado.

### Fase 3 (Claude Opus 4.6 Extended Thinking) — Estándares adversariales de flanqueo

1. **Detección de tautologías:** Se identificaron y eliminaron afirmaciones algebraicamente inevitables por definición.
2. **Pruebas de circularidad:** Correlaciones parciales, análisis de residuos y pruebas F para verificar la independencia del predictor respecto a las variables de resultado.
3. **Validación fuera de muestra:** Donde fue posible, se intentó replicación en datos retenidos y se reportó la precisión.
4. **Ataques de flanqueo:** Preguntas novedosas no planteadas por el análisis original, diseñadas para encontrar estructura inesperada en los mismos datos.
5. **Replicación transdominio:** Patrones encontrados en un dominio fueron probados en situaciones estructuralmente análogas de otros dominios.
6. **Publicación de resultados negativos:** Todos los flancos fallidos se reportan con dirección, valor p e interpretación física.

---

## Patrones emergentes transdominio (descubrimiento de la Fase 3)

Tres patrones estructurales identificados de forma independiente en múltiples campañas de flanqueo:

**Patrón 1 — El amplificador de métrica 2D**
Combinar α con una métrica de calidad amplifica de forma consistente los tamaños del efecto: consciencia (α × R²: d 0.33 → 0.97), cardiaco (α × CI: d 1.25 → 3.28), economía (σ entre escalas: separación de 10×).

**Patrón 2 — Los sistemas se acoplan con más fuerza antes de la crisis**
Los estados de crisis muestran mayor acoplamiento estructural, no menor: astronomía (conspiración barión-halo r = +0.70 en sistemas ricos en gas), ecología (la conspiración depredador-presa se intensifica, d = −2.52), consciencia (la conspiración α-R² se estrecha durante convulsiones), economía (todas las escalas se bloquean durante caídas, σ → 0.03).

**Patrón 3 — El medio fluido permite el acoplamiento estructural**
Los efectos de geometría estructural son detectables solo cuando un medio fluido o gaseoso llena el pozo de potencial: astronomía (los efectos desaparecen en galaxias pobres en gas), cardiaco (corazón trasplantado denervado: SD1 = 8 ms, sin variabilidad), ecología (tamaño corporal → color espectral mediado por la profundidad de la red metabólica).

---

## Fuentes de datos

| Dominio | Fuente |
|--------|--------|
| Corteza visual | Smith et al., Harvey & Dumoulin, Schmolesky et al. |
| Galaxias JWST | Catálogos CEERS, JADES |
| Ondas gravitacionales | GWTC-1 hasta GWTC-4.0 (LIGO/Virgo/KAGRA) |
| Cuántica | IBM Quantum Experience |
| Cardiaco | MIT-BIH Arrhythmia Database, PhysioNet Fantasia & CHF |
| Neurociencia | Estudios EEG publicados, UCI EEG (n = 11,500) |
| Ecología | Base de datos de longevidad AnAge, GPDD (978 series), Isle Royale (66 años) |
| Epidemiología | Datos COVID-19 de Johns Hopkins |
| Huracanes | IBTrACS v04r00 (NOAA), TorNet MIT Lincoln Lab (1,105 eventos) |
| Sismología | Catálogo de terremotos USGS |
| Astronomía | Base de datos SPARC (175 galaxias, 2,411 puntos radiales), viento solar PSP/Wind |
| Economía | Binance BTCUSDT OHLCV de 1 min (4 meses) |

---

## Reproducibilidad

Cada validación incluye:
- `analyze_*.py` — Script principal de análisis
- `requirements.txt` — Dependencias de Python
- `output/` — Datos CSV, figuras PNG/PDF
- `README.md` — Metodología e interpretación

```bash
# Ejecutar cualquier validación
pip install -r requirements.txt
python analyze_domain_rtm.py
```

---

## Idea clave

El exponente de escalamiento RTM α **no es un parámetro de ajuste**: es un **invariante estructural** determinado por la topología de la red. Esto explica por qué:

- El transporte balístico (α = 1) aparece en ondas gravitacionales, rupturas sísmicas Y propagación pandémica
- El transporte subdifusivo (α < 0.5) aparece en corteza visual Y ritmos cardiacos con ICC
- Las transiciones de fase (α > 2) aparecen en plegamiento de proteínas Y caídas de mercado

La misma matemática describe fenómenos radicalmente diferentes porque comparten la misma clase topológica de transporte. Tres rondas de validación adversarial independiente —realizadas por dos sistemas de IA diferentes durante tres meses— han refinado, pero no roto, este marco de clasificación.

---

## Citación

Si usas este trabajo, por favor cita:

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

---

## Licencia

© 2026 Álvaro José Quiceno Rendón
Distribuido bajo [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Nota: **Usa el identificador DOI de Zenodo más reciente.**
