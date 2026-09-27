<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# Economía Rítmica
**Midiendo la resiliencia sistémica con el exponente de coherencia RTM**  
  
Álvaro Quiceno

</div>

**Resumen**  
Proponemos una visión rítmica de la dinámica económica fundamentada en el principio RTM (Relatividad del Tiempo en sistemas Multiescala) de que los tiempos característicos escalan con el tamaño del sistema como τ ∝ L^α. Traduciendo esto a la economía, definimos un Exponente de Coherencia Económica α que captura cuán rápido los procesos a diferentes escalas —hogares, empresas, sectores, mercados— se estabilizan, propagan o recuperan. A partir de α construimos un Índice de Coherencia Económica (ICE) en tiempo real: una meta-estimación de pendiente primero con errores en variables, obtenida de múltiples proxies independientes de "longitud" económica L (tamaño de red, nivel de capitalización) y "tiempo" económico τ (vidas medias de recuperación, tiempos de relajación). Debido a que los cambios de reloj o niveles afectan los interceptos y no las pendientes, α está diseñado para ser robusto a cambios de unidades y confusores a nivel de régimen.

**Validación computacional.** Implementamos y probamos el marco ICE mediante tres suites de simulación. S1 demuestra la estimación de α a partir de niveles de capitalización de mercado y tiempos de recuperación a través de cinco regímenes de mercado (crecimiento estable α≈0.45, pre-crisis α≈0.35, crisis α≈0.20), recuperando el exponente verdadero con un error del 0.6%. El metaanálisis a través de cuatro familias de proxies (vida media de recuperación, persistencia de volatilidad, decaimiento de autocorrelación, relajación de flujo de órdenes) arroja estimaciones combinadas del ICE con heterogeneidad cuantificada (I²). S2 valida la Hipótesis H2 —que la caída de α anticipa recesiones— mediante pruebas retrospectivas en tres episodios (Punto-Com 2001, Crisis Financiera Global 2008, COVID 2020), encontrando tiempos de anticipación medios de 9 meses con mayores reducciones de α (Δα = 0.14-0.27) precediendo crisis más severas. S3 demuestra la variación entre países: α se correlaciona fuertemente con la frecuencia de crisis (r = -0.91) y la reducción promedio (r = -0.95), con economías desarrolladas (α ≈ 0.48-0.55) mostrando mayor resiliencia que los mercados emergentes (α ≈ 0.25-0.35).

Articulamos hipótesis falsificables: H1 (Resiliencia), un α base más alto predice menores reducciones; H2 (Anticipación), caídas pronunciadas de α preceden recesiones por 6-18 meses; H3 (Cascada), α es no decreciente a través de capas de agregación. El marco ofrece una señal de alerta temprana complementaria, distinta de las métricas de volatilidad o apalancamiento, con implicaciones para pruebas de estrés conscientes de la coherencia y política macroprudencial.

**Validación empírica**$`\mathbf{\rightarrow}`$ **(Capítulos 11 y 12).** Más allá de la simulación, sometemos el marco a una prueba de estrés forense utilizando la microestructura del mercado de Bitcoin de alta frecuencia y caídas históricas del S&P 500 y el Oro. Una versión anterior de este trabajo informó una gran separación dentro de muestra entre los valores DFA-α de mercados saludables y de mercados en colapso (d = −1.45), con una anticipación de ~10 días, derivada de una tabla de 13 eventos entre distintos mercados. Posteriormente se descubrió que esa tabla (`crash_alpha_analysis.csv`) contenía **valores α generados sintéticamente** (reproducibles a partir de una semilla aleatoria fija), por lo que sus estadísticas se retiran como afirmaciones empíricas. La validación independiente sobre datos reales de mercado (5,104 días de rendimientos del S&P 500 que abarcan las crisis de las punto com, la GFC y la COVID) encuentra que la separación real de DFA-α entre crisis y calma es pequeña y no significativa (Cohen's d ≈ −0.2, p ≈ 0.22), y que α **no** disminuye de forma fiable antes de las crisis. La contribución empírica que se mantiene del marco es la métrica de **Coherencia Multiescala** (véase más abajo).

> **Corrección de procedencia de datos (julio de 2026).** Se descubrió que la tabla α de 13 eventos que sustentaba el resultado dentro de muestra d = −1.45 había sido generada sintéticamente; esas cifras se retiran (véase el Capítulo 12.6). Los hallazgos que aparecen a continuación se calculan a partir de datos brutos de mercado y no se ven afectados. **Advertencia del Equipo Rojo (abril de 2026).** La validación adversarial independiente (5 flancos analíticos) encontró que la precisión de predicción de caídas fuera de muestra es del **25%** (1 de 4 eventos post-2022 correctamente clasificados cuando el umbral se entrena con caídas pre-2022). Los patrones de caída de $`\alpha`$ en los Capítulos 11 y 12 son observaciones forenses dentro de la muestra, no predicciones prospectivas validadas. El hallazgo novedoso superviviente es la métrica de **Coherencia Multiescala**: durante los meses de caída de BTC (COVID marzo 2020, FTX noviembre 2022), la desviación estándar transversal entre escalas de $`\alpha`$ a través de ventanas temporales es $`\sigma = 0.031\text{-}0.034`$; durante el mes de control (septiembre 2023), $`\sigma = 0.310`$, una diferencia de 10x. Este efecto se replica en la misma dirección, fuera de muestra, en datos diarios del S&P 500 a lo largo de tres crisis (σ entre escalas durante crisis por debajo de σ en calma; Mann-Whitney p = 0.039), aunque la separación en acciones es mucho menor (~1.3×) que la separación intradía en BTC (~10×). La métrica es real, pero su magnitud depende del activo y de la escala temporal. Esta métrica es nativa de RTM y no es medida por indicadores financieros estándar. Detalles: Capítulo 12.7.

> Las simulaciones Monte Carlo a través de 16 mercados globales arrojan un exponente de cola de $`\alpha = 2.966 \pm 0.236`$, consistente con la "Ley Cúbica Inversa" de RTM y con la ley cúbica inversa empíricamente establecida de las fluctuaciones de precios (Gabaix et al. 2003). Este es un resultado convergente, correctamente enmarcado como confirmación de escalamientos financieros conocidos, no como rechazo categórico de los modelos gaussianos.

**1. Introducción**

**1.1 Motivación**

Los indicadores macrofinancieros estándar (crecimiento del PIB, inflación, desempleo, índices de volatilidad) resumen niveles o dispersión pero raramente miden cómo el **tempo** y la **organización a través de escalas** cambian a medida que los sistemas evolucionan. Sin embargo, las crisis, las interrupciones de cadenas de suministro y las cascadas súbitas de sentimiento son fenómenos multiescala: el *tiempo que toma* para que una perturbación se propague o disipe depende del *tamaño* y la *conectividad* de las estructuras que atraviesa. La RTM, la observación empírica de que los tiempos característicos escalan con el tamaño mediante una ley de potencia, ofrece una forma compacta de modelar esa dependencia. Llevar la RTM a la economía sugiere una cantidad única e interpretable, el **Exponente de Coherencia Económica** $`\alpha_{\text{econ}}`$, que cuantifica cuán "rápido" o "estructurado" es un sistema económico en un momento dado, a través de las capas.

**1.2 De RTM a** $`\mathbf{\alpha}_{\text{econ}}`$

La afirmación central de RTM es una simetría de escalamiento: si dos subsistemas son geométricamente similares pero difieren en escala, sus tiempos característicos se relacionan como $`T \propto L^{\alpha}`$. Estimar $`\alpha`$ se basa en **pendientes** en el espacio log-log dentro de ambientes fijos; los cambios en relojes, unidades o líneas base alteran los **interceptos** pero no las **pendientes**. En economía, proyectamos "longitud" $`L`$ como un proxy de escala —tamaño de empresa, capitalización, longitud de ruta de cadena de suministro, grado de red o alcance jurisdiccional— y "tiempo" $`T`$ como una métrica de persistencia o relajación —vidas medias de recuperación, ventanas de resiliencia del libro de órdenes, decaimiento de tiempo de entrega, relajación de sentimiento. El **Índice de Coherencia Económica (ICE)** es una meta-estimación rodante con errores en variables de $`\alpha_{\text{econ}}`$ que combina múltiples familias $`(L,T)`$ con validación cruzada y cuantificación de incertidumbre.

**1.3 Qué significa** $`\mathbf{\alpha}_{\text{econ}}`$ **(intuición)**

Un $`\alpha_{\text{econ}}`$ más alto implica que las estructuras más grandes se desaceleran *más que proporcionalmente*, lo que a menudo refleja **mayor organización y controlabilidad**: la información se filtra, existen amortiguadores y los flujos están orquestados. Un $`\alpha_{\text{econ}}`$ más bajo implica un gradiente tiempo-escala más plano: los choques atraviesan las capas rápidamente, a veces beneficioso para el rendimiento pero peligroso para la estabilidad. Así, $`\alpha_{\text{econ}}`$ reenmarca la compensación clásica entre velocidad bruta y resiliencia sistémica como un parámetro de **coherencia** ajustable.

**1.4 Cómo el ICE difiere de las métricas familiares**

La volatilidad (por ejemplo, VIX) mide dispersión a una escala dada; el apalancamiento mide la sensibilidad del balance; la liquidez mide el costo de transacción/profundidad de mercado. **El ICE mide la *pendiente* del tiempo vs. escala**: una propiedad estructural que complementa esas señales. Dado que el ICE se construye sobre pendientes, es comparativamente robusto a elecciones de unidades, derivas nominales y muchos cambios de nivel por políticas.

**1.5 Programa empírico**

Vamos a (i) construir pares $`(L,T)`$ a través de dominios independientes —microestructura de mercado, logística, renovación crediticia, decaimiento de información—, (ii) estimar $`\alpha_{\text{econ}}`$ dentro de contenedores de ambiente fijo mediante regresión robusta de errores en variables, (iii) validar el **colapso** interno (las curvas coinciden al reescalar por $`L^{\alpha}`$ dentro de un contenedor), (iv) construir un ICE(t) rodante con incertidumbre, y (v) probar H1–H3 en episodios retrospectivos y, prospectivamente, en pilotos en vivo. Los fallos de separación de pendiente o colapso se registran como **resultados negativos** que delimitan el alcance del ICE.

**1.6 Validación empírica sistemática: transiciones de fase y microestructura de mercado (Capítulos 11 y 12)**

Dentro del paradigma analítico de la RTM, una caída de mercado no es un evento puramente exógeno o de pánico aleatorio, sino el resultado final de una transición de fase topológica cuantificable. Para cerrar la brecha entre la termodinámica teórica y la gestión de riesgos aplicada, utilizamos caídas históricas y el mercado de Bitcoin de alta frecuencia como un túnel de viento computacional.

Un análisis inicial de 13 grandes colapsos históricos pareció mostrar fuertes correlaciones entre la descorrelación de DFA-α y la severidad del colapso; posteriormente se descubrió que ese análisis se basaba en valores α generados sintéticamente y se retira (Capítulo 12.6). En datos reales de mercado, la señal DFA-α de crisis es débil y no anticipa los colapsos (Capítulo 12.7). Para abordar las falacias ecológicas de estimaciones puntuales estáticas, desplegamos modelado continuo de Errores en Variables (ODR) e inyección de ruido Monte Carlo a través de miles de horas de negociación.

El cambio informado anteriormente desde una coherencia laminar de referencia ($`\alpha \approx 0.55`$) hacia la antipersistencia durante las caídas ($`\alpha \approx 0.46`$, d de Cohen $`= -1.45`$, ~10 días de anticipación dentro de la muestra) derivaba de la tabla α sintética y se retira (Capítulo 12.6). El escalamiento del tiempo de recuperación sigue una pendiente no lineal castigadora ($`3.59 \pm 0.70`$), consistente con la asimetría conocida entre la velocidad de la caída y la duración de la recuperación. Las distribuciones de retornos a través de 16 mercados globales convergen a $`\alpha = 2.966 \pm 0.236`$, consistente con la ley cúbica inversa empíricamente establecida (Gabaix et al. 2003).

**Limitación importante:** La validación fuera de muestra (Equipo Rojo, abril de 2026) encontró un 25% de precisión (1/4 eventos post-2022) al aplicar el umbral entrenado de caída de $`\alpha`$ a nuevos datos. Los patrones forenses (COVID, FTX, Crisis Financiera Global) son resultados dentro de la muestra. Si la señal estructural se generaliza prospectivamente requiere validación adicional en datos reservados. La métrica de Coherencia Multiescala ($`\sigma`$ de $`\alpha`$ a través de escalas temporales) muestra más promesa como indicador novedoso — ver Capítulo 12.7.

**Validación empírica de bifurcación de fase en mercados de alta frecuencia (Capítulo 11)**

Este capítulo somete a prueba de estrés al Monitor en Tiempo Real de RTM contra la varianza extrema de la microestructura de Bitcoin. Al abandonar los cierres diarios estáticos e inyectar el perfil completo de ruido continuo de datos OHLCV minuto a minuto, rastreamos el momento exacto de la fractura estructural. El análisis continuo aísla el umbral de Bifurcación de Fase ($`\alpha < \ 0.5`$), distinguiendo las fallas mecánicas de liquidez (por ejemplo, marzo 2020) de los eventos de estrés político de alta viscosidad ($`\alpha > \ 0.6`$, por ejemplo, mayo 2021). El evento de octubre 2025 mostró una lectura elevada de $`\alpha`$ consistente con una anomalía técnica (posteriormente atribuida a una falla de la plataforma Binance) en lugar de una caída fundamental. La observación de anticipación de 15 horas es un caso único dentro de la muestra y no debe generalizarse como predicción prospectiva validada hasta ser replicado en eventos fuera de muestra.

**Análisis empírico: el colapso de** $`\mathbf{\alpha}`$ **como señal predictiva (Capítulo 12)**

Este capítulo se aparta de los supuestos gaussianos tradicionales de las recuperaciones de mercado y las predicciones de caídas. Los modelos OLS ingenuos iniciales que predecían recuperaciones de caídas sufrían de un sesgo de atenuación masivo debido a las fronteras ambiguas de "recuperación del mercado". Al aplicar la Regresión de Distancia Ortogonal (ODR) para absorber un margen de ruido de medición del 20%, revelamos que el escalamiento del tiempo de recuperación es sustancialmente más castigador (pendiente = $`3.59\  \pm 0.70`$) de lo que se había modelado previamente.

Una versión anterior informó que un análisis Monte Carlo de 13 grandes colapsos mostraba que la descorrelación estructural α precedía los mínimos de precio en ~9.75 días (d = −1.45). Posteriormente se descubrió que la tabla subyacente de 13 eventos contenía valores α generados sintéticamente (Capítulo 12.6), por lo que estas cifras se **retiran**. Un análisis independiente sobre rendimientos diarios reales del S&P 500 (Capítulo 12.7) no encuentra una separación α significativa entre crisis y calma (d ≈ −0.2, p ≈ 0.22) ni una disminución de α previa al colapso. El resultado de escalamiento del tiempo de recuperación (véase más arriba) y la métrica de coherencia multiescala (Capítulo 12.7) se calculan a partir de datos independientes y se conservan.

**2. Introducción a RTM para economistas**

Esta sección destila la Relatividad Temporal Multiescala (RTM) en herramientas que se pueden usar con datos económicos. Explicamos el escalamiento maestro, por qué las **pendientes** (no los niveles) son la señal robusta, cómo estimar el exponente de coherencia y qué falsificaría el enfoque.

**2.1 La ley maestra y por qué una ley de potencia**

**Afirmación (RTM).** En sistemas multiescala, los tiempos característicos $`T`$ escalan con el tamaño del sistema $`L`$ mediante una ley de potencia:

``` math
T = \kappa\text{ }L^{\alpha},
```

donde $`\kappa > 0`$ es un factor de escala determinado por el ambiente (unidades, fricciones de referencia, "reloj"), y $`\alpha`$ es el **exponente de coherencia**. Tomando logaritmos:

``` math
\log T = \alpha\ \log L + \log\kappa.
```

**¿Por qué una ley de potencia?** Si (i) reescalar el sistema por un factor $`b > 0`$ simplemente reescala su tiempo característico por una función determinista $`f(b)`$, y (ii) reescalamientos independientes se componen ($`f(b_{1}b_{2}) = f(b_{1})f(b_{2})`$), entonces $`f(b) = b^{\alpha}`$. Esta ecuación funcional tipo Cauchy es la ruta estándar de la *simetría de escala* a las *leyes de potencia*. En economía, el "sistema" puede ser una empresa, una cadena de suministro o un subgrafo de mercado; el "tiempo" puede ser un tiempo de relajación, renovación o recuperación.

**2.2 Pendiente vs. intercepto: la perspectiva de "invariancia del reloj"**

Los cambios de unidades o relojes de medición uniformes típicamente **multiplican** las duraciones por una constante $`c`$ (es decir, $`T' = cT`$), lo que añade una constante a $`\log T`$ y desplaza el intercepto sin cambiar la pendiente $`\alpha`$, siempre que el ambiente se mantenga fijo.\
Sin embargo, los cambios de régimen que añaden un retardo aproximadamente constante $`b`$ a las duraciones observadas ($`T_{\text{obs}} = T + b`$) **no** añaden una constante a $`\log T`$ y pueden sesgar las estimaciones de pendiente a menos que se modelen o corrijan (por ejemplo, ajustar $`\log(T_{\text{obs}} - b)`$ después de validar $`b`$, o restringir a $`T \gg b`$). Por lo tanto:\
**Intercepto**: efectos de nivel (cambios de unidad, relojes multiplicativos, reescalamientos de línea base; compensaciones aditivas solo después de corrección).\
**Pendiente**: escalamiento estructural (cómo el tiempo crece con la escala dentro de un ambiente fijo).\
Implicación práctica: para obtener robustez, estime pendientes en contenedores de ambiente fijo **y** audite/ajuste las compensaciones aditivas antes de tomar logaritmos.

**Implicación práctica.** Si desea algo robusto a cambios de nivel y muchas rebasaciones de política, estime **pendientes** en **contenedores de ambiente fijo** (por ejemplo, país × régimen de política × trimestre).

**2.3 Qué es y qué no es** $`\mathbf{\alpha}`$

- $`\alpha`$ **no** es una temperatura, volatilidad ni un "dial de velocidad". Es un **gradiente de tempo a través de la escala**.

- $`\alpha`$ **no** es el exponente dinámico $`z`$ de los fenómenos críticos; aquí usamos una pendiente operacional directa entre $`L`$ y $`T`$ empíricos.

- Heurísticamente, un $`\alpha`$ **mayor** implica **mayor organización/persistencia**: las estructuras más grandes se desaceleran más que proporcionalmente, sugiriendo amortiguamiento, toma de decisiones escalonada y flujo de información filtrado. Un $`\alpha`$ **menor** implica una sincronización más plana entre escalas, propagación rápida y exposición potencial a cascadas.

**2.4 Mapeando la economía a** $`\mathbf{L}`$ **y** $`\mathbf{T}`$

RTM requiere pares $`(L,T)`$ extraídos de un **ambiente fijo**:

- **Proxies candidatos de** $`L`$ **(tamaño/escala)**

  - Micro: empleados, tamaño del balance, nivel de capitalización, conteo de proveedores.

  - Meso: longitud de ruta de cadena de suministro, grado/centralidad de red, tamaño del mercado regional.

  - Macro: amplitud de red comercial, tamaño de red interbancaria, alcance jurisdiccional.

- **Proxies candidatos de** $`T`$ **(tiempo/persistencia)**

  - Micro: ciclo de orden a cobro, tiempo para cubrir vacante, vida media de reversión de microprecio.

  - Meso: persistencia de tiempo de entrega de envío, vida media de reposición de inventario.

  - Macro: vida media de recuperación de brecha de producto después de un choque, tiempo de renovación de crédito rotativo, tiempo de decaimiento de sentimiento/noticias.

**Regla de compatibilidad.** Dentro de un contenedor, $`L`$ debe ser monótono con la escala y $`T`$ un **tiempo característico** vinculado a *la misma capa de proceso*. Mezclar capas en un contenedor (por ejemplo, $`L`$ a nivel de empresa con $`T`$ a nivel sectorial) viola la comparabilidad.

**2.5 Estimación de** $`\mathbf{\alpha}`$ **: pendiente primero con error de medición**

Los $L$ y $T$ económicos reales son ruidosos. Una regresión OLS simple de $\log T$ sobre $\log L$ está sesgada cuando $L$ tiene error. Use **errores en variables (EIV)** o alternativas robustas:

- **Mínimos cuadrados totales / regresión de distancia ortogonal** para ruido simétrico.

- Pendiente de **Theil–Sen** para robustez ante valores atípicos (mediana de pendientes por pares).

- **SIMEX** (simulación-extrapolación) si puede aproximar el nivel de ruido de $`L`$.

**Pipeline contenedor por contenedor (esquema).**

1.  Fijar un ambiente (por ejemplo, manufactura de EE.UU., 2012–2019, política estable).

2.  Particionar en niveles de tamaño (o ventanas deslizantes) tal que el *reloj ambiental* sea aproximadamente constante.

3.  En cada contenedor, ajustar $`\log\ T = \alpha\ \log\ L + c`$ con EIV; reportar $`\widehat{\alpha}`$ e IC.

4.  **Prueba de colapso:** reescalar cada curva por $`L^{\widehat{\alpha}}`$ y verificar que la estructura residual desaparezca dentro de ese contenedor (las curvas "colapsan"). Fallo del colapso ⇒ el contenedor mezcla regímenes incompatibles o $`\alpha`$ no está bien definido allí.

5.  Combinar múltiples familias independientes de $`(L,T)`$ mediante **metaanálisis de efectos aleatorios** para obtener $`{\widehat{\alpha}}_{\text{econ}}`$ e incertidumbre.

**2.6 Interpretando niveles vs. cambios en** $`\mathbf{\alpha}_{\text{econ}}`$

- **Nivel** $`{\bar{\alpha}}_{\text{econ}}`$ : coherencia/resiliencia de fondo de un sistema durante un período.

- **Cambio** $`\Delta\alpha_{\text{econ}}`$ : señal de alerta temprana; las caídas repentinas indican **decoherencia** (las sincronizaciones entre escalas se vuelven demasiado similares → los choques atraviesan rápidamente). Los aumentos repentinos pueden indicar reorganización, a veces a costa del rendimiento bruto.

**Compensación de diseño.** Un $`\alpha`$ más alto a menudo significa un tempo bruto más lento en las escalas más grandes, pero mejor control y estabilidad (menos cascadas catastróficas). Un $`\alpha`$ más bajo aumenta el rendimiento pero puede elevar el riesgo sistémico.

**2.7 Bandas de universalidad (orientación heurística)**

Aunque RTM no *fija* un $`\alpha`$ universal, las bandas empíricas ayudan a interpretar rangos:

- **Aplanado/co-moviente** ($`\alpha \approx 1`$): los tiempos escalan ~linealmente con el tamaño, propagación rápida, amortiguamiento mínimo.

- **Difusivo/mediado** ($`\alpha \approx 2`$): las estructuras más grandes se desaceleran más; las capas de coordinación son visibles.

- **Jerárquicamente amortiguado** ($`\alpha > 2`$): escalonamiento profundo, ciclos de planificación largos, holgura sustancial.

Estas son **heurísticas interpretativas**, no umbrales duros; los datos y las pruebas de colapso arbitran.

**2.8 Falsificabilidad: dónde debería fallar RTM en economía**

RTM hace afirmaciones lo suficientemente fuertes como para ser **incorrectas**:

- **Sin separación de pendiente:** Si, dentro de un ambiente fijo, $`\partial\ logT/\partial\ logL`$ es indistinguible de cero (o totalmente inestable) a través de múltiples familias independientes de $`(L,T)`$, RTM no es informativo para ese dominio.

- **Sin colapso:** Si el reescalamiento por $`L^{\widehat{\alpha}}`$ no logra colapsar las curvas dentro de un contenedor, $`\alpha`$ no está bien definido allí.

- **Cascada inversa:** Si las capas de agregación muestran $`\alpha`$ **decreciente** (macro más rápido a través de la escala que micro) de manera sistemática y robusta, la firma de cascada de RTM falla.

- **Simetría de direccionalidad:** Si la transferencia de información (por ejemplo, entropía de transferencia) es simétrica o dominante en reversa a través de las capas en estado estacionario, la afirmación de cascada falla.

Estos criterios actúan como **barandillas** — delimitan dónde el ICE es válido y dónde los modelos clásicos pueden ser suficientes.

**2.9 Microejemplo desarrollado (experimento mental)**

Suponga que estudiamos empresas manufactureras dentro de un solo país y un período de política estable. Sea:

- $`L =`$ nivel de conteo de empleados en log;

- $`T =`$ mediana del ciclo de **orden a cobro** por nivel.

Ajustamos $`\log T = \alpha\ \log\ L + c`$ usando Theil–Sen dentro de cada año. Hallazgos:

- 2014–2018: $`\widehat{\alpha} \in \lbrack 1.8,2.2\rbrack`$ y colapsos limpios → **régimen coherente y amortiguado**.

- 2019T4–2020T2: $`\widehat{\alpha}`$ cae a $`1.2`$ con colapso deficiente → **evento de decoherencia** (transmisión de choque), consistente con tensión en la cadena de suministro.

- 2021–2022: $`\widehat{\alpha}`$ rebota parcialmente a $`1.6`$ a medida que aumentan la relocalización y los amortiguadores de inventario.

Incluso sin magnitudes de PIB o inflación, la **pendiente** narra la estructura: si los diferenciales de escala en la sincronización están presentes (amortiguados) o aplanados (expuestos).

**2.10 Notas de implementación (para reutilización en la Sección 5)**

- **Contenedores.** Preferir múltiplos pequeños de contenedores de ambiente fijo (país × sector × régimen × trimestre). Usar detección de puntos de cambio para mantener los regímenes estables dentro de los contenedores.

- **Incertidumbre.** Bootstrap de empresas/aristas/envíos; reportar IC percentiles sobre $`\alpha`$. Rastrear la deriva en la cobertura (disponibilidad de datos) como métrica de calidad.

- **Placebos.** Reescalar relojes (por ejemplo, convertir días↔semanas) para verificar la invariancia de pendiente. Aleatorizar $`L`$ dentro de contenedores para estimar el sesgo que obtendría por azar.

- **Libro de registro.** Mantener un "libro de pendiente vs. intercepto": cada estimación de $`\alpha`$ debe ir acompañada del intercepto $`c`$ y una nota de los cambios de nivel conocidos (política, unidad, rebasaciones por inflación). Esto documenta que la robustez genuinamente reside en las pendientes.

**Conclusiones para los economistas**

1.  RTM proporciona un **único parámetro estructural**, la pendiente $`\alpha`$, para resumir cómo la sincronización se estira con la escala.

2.  Estimar $`\alpha`$ **contenedor por contenedor** y probar el **colapso** lo protege de muchos confusores que afectan a los indicadores basados en niveles.

3.  $`\alpha_{\text{econ}}`$ es **complementario**, no sustitutivo: añade una lente de coherencia a las métricas de volatilidad, liquidez y apalancamiento.

4.  RTM es **falsificable** en este dominio; modos de falla claros previenen la extralimitación.

**3. Definición de** $`\mathbf{\alpha}_{\text{econ}}`$ **y construcción del Índice de Coherencia Económica (ICE)**

Este capítulo formaliza el **Exponente de Coherencia Económica** $`\alpha_{\text{econ}}`$ y especifica el **Índice de Coherencia Económica (ICE)**, una estimación rodante y consciente de la incertidumbre derivada de múltiples familias de proxies $`(L,T)`$. Presentamos (i) definiciones de medición, (ii) un pipeline de estimación de pendiente primero con errores en variables, (iii) una prueba de colapso para validar el escalamiento por contenedor, (iv) un meta-estimador de efectos aleatorios que fusiona proxies, y (v) nowcasting en tiempo real y aseguramiento de calidad.

**3.1 Objetos y ambientes**

Sea $`\mathcal{U}`$ un **ambiente fijo** (por ejemplo, país × régimen de política × sector × trimestre). Dentro de $`\mathcal{U}`$, observamos $`N`$ unidades $`u = 1,\ldots,N`$ (empresas, aristas, productos, puertos, tickers…) y construimos **mediciones pareadas**

``` math
(L_{u},T_{u})\ \ \ \ \text{con    }{\ L}_{u} > 0,\ T_{u} > 0.
```

- $`L`$ es un **proxy de escala** (tamaño, longitud de ruta, nivel de capitalización, grado de red, alcance geográfico).

- $`T`$ es un **tiempo característico** de una *capa de proceso compatible* (relajación, renovación, recuperación, persistencia).

**Escalamiento RTM dentro de** $`\mathcal{U}`$ **:**

``` math
T_{u} = \kappa_{\mathcal{U}}\text{ }L_{u}^{\alpha_{\mathcal{U}}}\text{ }\varepsilon_{u},\mathbb{E}\lbrack\log\varepsilon_{u}\rbrack = 0.
```

Tomando logaritmos:

``` math
y_{u} = \log T_{u} = \alpha_{\mathcal{U}}x_{u} + c_{\mathcal{U}} + \eta_{u},{\ \ \ \ \ \ \ \ \ \ \ \ x}_{u} = \log L_{u},\ c_{\mathcal{U}} = \log\kappa_{\mathcal{U}}.
```

Permitimos **error de medición** en ambos $`x`$ y $`y`$ :

``` math
x_{u}^{\text{obs}} = x_{u} + \xi_{u},{\ \ \ \ \ \ \ \ \ \ \ \ y}_{u}^{\text{obs}} = y_{u} + \zeta_{u},
```

con $`\xi_{u},{\ \zeta}_{u}`$ de media cero, posiblemente heterocedásticos.

**Objetivo.** Estimar $`\alpha_{\mathcal{U}}`$ robustamente (pendiente primero), validar que un solo $`\alpha`$ explica el contenedor mediante **colapso**, y combinar a través de familias de proxies independientes para obtener $`{\widehat{\alpha}}_{\text{econ}}(\mathcal{U})`$.

**3.2 Familias de proxies para** $`\mathbf{L\ }`$ **y** $`\mathbf{T}`$

Recomendamos usar **al menos dos** familias independientes por ambiente; ejemplos:

**A. Microestructura de mercado**

- $`L`$ : nivel de capitalización; nivel de tamaño mediano de transacción; grado en una red de impacto cruzado.

- $`T`$ : vida media de reversión de microprecio; tiempo de resiliencia del libro de órdenes (recuperación de profundidad); persistencia de estabilidad de cotización.

**B. Logística y cadenas de suministro**

- $`L`$ : longitud de ruta (etapas), tamaño de ruta multimodal, nivel de capacidad portuaria.

- $`T`$ : persistencia de tiempo de entrega de envío; decaimiento de tiempo de permanencia; vida media de reposición de inventario.

**C. Crédito y financiamiento**

- $`L`$ : extensión de la escalera de vencimiento; grado de red interbancaria; nivel de tamaño del libro.

- $`T`$ : tiempo de renovación de crédito rotativo; vida media de reversión a la media del diferencial después de choques de financiamiento.

**D. Flujo de información**

- $`L`$ : nivel de audiencia/alcance; centralidad de red del medio; alcance jurisdiccional.

- $`T`$ : decaimiento de choque de sentimiento/noticias; vida media de dispersión de desacuerdo.

**Regla de compatibilidad.** Dentro de una familia, $`L`$ y $`T`$ deben describir la **misma capa de proceso**; no mezcle $`L`$ micro con $`T`$ macro dentro de una regresión.

**3.3 Estimación de pendiente por contenedor (EIV / robusta)**

Dados $`\mathcal{U}`$ y una familia de proxies $`f`$, ajustar

``` math
y_{u}^{\text{obs}} = \alpha_{\mathcal{U},f}\text{ }x_{u}^{\text{obs}} + c_{\mathcal{U},f} + \epsilon_{u}
```

con **errores en variables** para corregir la atenuación:

- **Regresión de Distancia Ortogonal (ODR)** o **Mínimos Cuadrados Totales (TLS)** cuando $`Var(\xi) \approx Var(\zeta)`$.

- **SIMEX** (simulación-extrapolación) si podemos aproximar $`\sigma_{\xi}^{2}`$ a partir de mediciones repetidas o precisión instrumental.

- Pendiente mediana de **Theil–Sen** (robusta ante valores atípicos) como verificación de sensibilidad.

- **Bootstrap** (agrupado por unidad/entidad) para ICs y corrección de sesgo.

**Entregables por contenedor y familia:** $`{\widehat{\alpha}}_{\mathcal{U},f}`$, IC 95%, intercepto $`{\widehat{c}}_{\mathcal{U},f}`$, diagnósticos de ajuste y diagnósticos de cobertura (cuán representativa es la muestra dentro de $`\mathcal{U}`$).

**3.4 Validación de colapso (por contenedor)**

Después de estimar $`\widehat{\alpha}`$, probamos si un **único escalamiento** explica el contenedor:

1.  **Reescalar** los datos: $`{\widetilde{y}}_{u} = y_{u}^{\text{obs}} - \widehat{\alpha}\text{ }x_{u}^{\text{obs}}`$.

2.  **Expectativa nula:** dentro de $`\mathcal{U}`$, $`{\widetilde{y}}_{u} \approx c_{\mathcal{U}} + \text{ruido}`$ **independiente de** $`x`$.

3.  **Cuantificar el colapso** con un estadístico tipo ANOVA:

``` math
\Delta_{\text{collapse}} = R^{2}\text{ }(\widetilde{y} \sim x^{\text{obs}}).
```

**Pasamos** el colapso si $`\Delta_{\text{collapse}}`$ está por debajo de un umbral pequeño (por ejemplo, \< 0.05) *y* los diagnósticos residuales no muestran tendencia sistemática vs. $`x`$. El fallo indica regímenes mixtos o una relación no potencial en ese contenedor.

**3.5 Fusión multiproxy: meta-estimación de efectos aleatorios**

Cuando al menos dos familias pasan el **colapso** y el CC, fusionamos sus pendientes por contenedor $`\{{\widehat{\alpha}}_{f}\}_{f = 1}^{F}`$ en una estimación del **Índice de Coherencia Económica** para esa ventana. Usamos un modelo **metaanalítico de efectos aleatorios** que reconoce la heterogeneidad entre familias.

**Estimador.** Sea $`{\widehat{\sigma}}_{f}^{2}`$ la varianza (agrupada/bootstrap) de $`{\widehat{\alpha}}_{f}`$. Estimar la varianza entre familias $`{\widehat{\tau}}^{2}`$ mediante **REML** (preferido; DerSimonian–Laird reportado como sensibilidad). Definir pesos de efectos aleatorios

``` math
w_{f}\  = \ \frac{1}{{\widehat{\sigma}}_{f}^{\text{ }2} + {\widehat{\tau}}^{\text{ }2}},
```

y calcular la pendiente fusionada y su varianza como

``` math
{\widehat{\alpha}}_{\text{econ}} = \frac{\sum_{f = 1}^{F}{w_{f}\text{ }{\widehat{\alpha}}_{f}}}{\sum_{f = 1}^{F}w_{f}},\ \ Var\text{ }({\widehat{\alpha}}_{\text{econ}}) = \frac{1}{\sum_{f = 1}^{F}w_{f}}.
```

Reportar intervalos del 50/95% a partir de la aproximación normal (o bootstrap de la fusión para robustez).

**Diagnósticos de heterogeneidad.** Publicamos tanto el resumen de efecto fijo como las estadísticas de heterogeneidad:

- **Q de Cochran** (usando pesos de *efecto fijo* $`w_{f}^{FE} = 1/{\widehat{\sigma}}_{f}^{\text{ }2}`$):

``` math
{\widehat{\alpha}}_{FE} = \frac{\sum_{f}^{}{w_{f}^{FE}\text{ }{\widehat{\alpha}}_{f}}}{\sum_{f}^{}w_{f}^{FE}},\ \ Q\  = \ \sum_{f = 1}^{F}{w_{f}^{FE}\text{ }({\widehat{\alpha}}_{f} - {\widehat{\alpha}}_{FE})^{2}}.
```

Bajo homogeneidad, $`Q \sim \chi_{F - 1}^{2}`$ aproximadamente.

- $`I^{2}`$ (proporción de la variación total debida a heterogeneidad):

``` math
I^{2}\  = \ \max\{ 0,\ \frac{Q - (F - 1)}{Q}\} \times 100\%.
```

**Puertas y umbrales (prerregistrados).**

- Proceder con un solo número fusionado solo si:

  - al menos **2 familias** pasan CC y colapso,

  - $`I^{2} < 50\%`$ *(heterogeneidad moderada o menor)*, y

  - REML converge con $`{\widehat{\tau}}^{2}`$ finito y $`{\widehat{\tau}}^{2}`$ por debajo de un tope histórico (por ejemplo, **≤ percentil 90** de ventanas limpias pasadas).

- Si $`I^{2} \geq 50\%`$ o la prueba $`Q`$ rechaza la homogeneidad con $`p < 0.05`$, **no publicamos un ICE único**. En su lugar:

  - reportamos los $`{\widehat{\alpha}}_{f}`$ **por familia** con ICs,

  - incluimos diagnósticos de influencia **dejando una familia fuera**, y

  - anotamos FAMILY_DIVERGENCE en CC.

**Panel de sensibilidad.** Junto con REML reportamos:

- estimación **DL** de $`\tau^{2}`$,

- el resumen de efecto fijo $`{\widehat{\alpha}}_{FE}`$,

- y un gráfico de bosque (por familia $`{\widehat{\alpha}}_{f}`$, peso $`w_{f}`$, IC), más $`Q`$, $`I^{2}`$, $`{\widehat{\tau}}^{2}`$.

**Justificación.** Los efectos aleatorios dan menor peso a las familias con gran incertidumbre interna ($`{\widehat{\sigma}}_{f}^{2}`$) **y** a las ventanas donde las familias discrepan (gran $`{\widehat{\tau}}^{2}`$). La puerta $`I^{2}`$ previene un número único engañoso cuando los proxies cuentan historias materialmente diferentes.

**3.6 De contenedores a un índice en tiempo real: ICE(t)**

Para producir una serie temporal, rodamos $`\mathcal{U}`$ a través de ventanas superpuestas (por ejemplo, mensuales con paso de 1 semana; trimestrales con paso de 1 mes).

**Algoritmo (alto nivel).**

1.  Definir ambientes rodantes $`\mathcal{U}_{t}`$ por ventana temporal y filtros de régimen (detección de puntos de cambio para mantener regímenes estables dentro de las ventanas).

2.  Para cada $`\mathcal{U}_{t}`$ y familia $`f`$, calcular $`{\widehat{\alpha}}_{\mathcal{U}_{t},f}`$ + prueba de colapso.

3.  Combinar familias mediante efectos aleatorios → $`{\widehat{\alpha}}_{\text{econ}}(t)`$.

4.  Aplicar **puertas de CC**: tamaño mínimo de muestra, cobertura de proxies, umbral de tasa de aprobación de colapso (por ejemplo, ≥ 2 familias pasan).

5.  Suavizar con un **filtro causal** (por ejemplo, EWMA con vida media de 2–3 ventanas) para estabilizar el ruido preservando los puntos de inflexión.

6.  Publicar **ICE(t)** como $`{\widehat{\alpha}}_{\text{econ}}(t)`$ con una banda de incertidumbre y banderas de CC.

**Banderas de CC (ejemplos).**\
LOW_COVERAGE, FAMILY_DIVERGENCE (alta heterogeneidad), NO_COLLAPSE, REGIME_MIX (punto de cambio dentro de la ventana), CLOCK_SHIFT (rebasación de unidad detectada).

**3.7 Eventos de decoherencia y señales anticipatorias**

Definir **eventos de decoherencia** como movimientos descendentes grandes y significativos:

``` math
\Delta\alpha_{\text{econ}}^{-}(t) = {\widehat{\alpha}}_{\text{econ}}(t) - {\widehat{\alpha}}_{\text{econ}}(t - h) \leq - \theta,
```

con horizonte $`h`$ (por ejemplo, 3 meses) y umbral $`\theta`$ elegido por percentil prerregistrado (por ejemplo, percentil 10 de cambios históricos) o por un múltiplo del error estándar rodante. Etiquetar eventos solo cuando las banderas de CC son verdes (sin mezcla de régimen; ≥2 familias pasan el colapso). Estos eventos sirven como **candidatos de alerta temprana** para H2 (anticipación).

**3.8 Estándares de reporte (nivel de contenedor e índice)**

**Por contenedor (**$`\mathcal{U}`$ **, familia** $`f`$ **):**

- $`{\widehat{\alpha}}_{\mathcal{U,}f}`$, IC 95%; $`{\widehat{c}}_{\mathcal{U,}f}`$.

- Estadístico de colapso $`\Delta_{\text{collapse}}`$ y aprobación/fallo.

- Tamaño de muestra, cobertura, puntos de influencia, esquema de bootstrap.

- Cambios de nivel conocidos (unidades, rebasación de política).

**Por tiempo** $`t`$ **:**

- $`{\widehat{\alpha}}_{\text{econ}}(t)`$, bandas del 50/95%; heterogeneidad $`{\widehat{\tau}}^{2}(t)`$.

- Contribuciones por familia e influencia dejando una fuera.

- Banderas de CC y notas sobre estabilidad de régimen.

**3.9 Robustez y ablaciones**

- **Sensibilidad de errores en variables.** Comparar pendientes ODR/TLS, Theil–Sen y corregidas por SIMEX.

- **Elecciones alternativas de** $`L,T`$ **.** Intercambiar proxies dentro de familias (por ejemplo, grado vs. longitud de ruta) y verificar estabilidad.

- **Placebos de reloj.** Cambiar unidades de tiempo (días ↔ semanas) para verificar invariancia de pendiente.

- **Pruebas de aleatorización.** Permutar aleatoriamente $`L`$ dentro de contenedores para estimar pendientes por azar; reportado como referencia nula.

- **Estabilidad de submuestras.** Jackknife de entidades/sectores/regiones.

- **Alternativa no potencial.** Ajustar $`\log T = g(\log L)`$ con splines; una curvatura fuerte y consistente entre contenedores falsifica la suposición de ley de potencia para ese dominio.

**3.10 Pseudocódigo mínimo**

for t in rolling_windows:

U_t = define_environment(t) \# ventana estable en régimen

family_estimates = \[\]

for f in proxy_families:

data = load_pairs(U_t, f) \# (L,T) con metadatos

xobs, yobs = log(L), log(T)

alpha_hat, c_hat, se = EIV_fit(xobs, yobs) \# ODR/SIMEX/Theil–Sen

collapse = R2_of_residual_vs_x(yobs - alpha_hat\*xobs, xobs)

if collapse \< threshold and coverage_ok(data):

family_estimates.append((alpha_hat, se))

if len(family_estimates) \>= 2:

alpha_RE, se_RE, tau2 = random_effects(family_estimates)

if QA_ok(family_estimates, tau2):

ECI\[t\] = (alpha_RE, se_RE, flags=None)

else:

ECI\[t\] = (alpha_RE, se_RE, flags=QA_flags)

else:

ECI\[t\] = (nan, nan, flags={'LOW_COVERAGE'})

**3.11 Guía de interpretación (práctica)**

- **ICE alto (mayor** $`\alpha_{\text{econ}}`$ **)**\
  Esperar un escalonamiento más profundo y una propagación entre escalas más lenta: mejor absorción de choques, potencialmente menor rendimiento bruto en las escalas más grandes; a menudo preferible durante períodos frágiles.

- **ICE bajo (menor** $`\alpha_{\text{econ}}`$ **)**\
  Gradientes de tiempo más planos: propagación más rápida, eficiente en tiempos tranquilos pero expone al sistema a fallas sincronizadas.

- **ICE en aumento** puede indicar reorganización post-choque (amortiguadores reconstruyéndose, gobernanza mejorando).

- **ICE en caída**, especialmente con CC limpio, amerita vigilancia: decoherencia que puede preceder episodios de estrés.

**3.12 Limitaciones específicas del ICE**

- **Fragilidad de proxy.** Algunos pares $`(L,T)`$ son cíclicos o sensibles a políticas; rotar familias y documentar la cobertura es esencial.

- **Detección de régimen.** Ambientes mal especificados mezclan relojes y sesgan pendientes; la detección de puntos de cambio no es infalible.

- **Endogeneidad.** La coherencia puede ser el resultado de acciones de política en lugar de una causa exógena; la interpretación causal requiere diseños adicionales (instrumentos, diferencias en diferencias).

- **Latencia de datos.** Algunos proxies de $`T`$ se actualizan lentamente; el índice debe divulgar la latencia y usar nowcasting con prudencia.

**4. Hipótesis falsificables y diseño de estudio**

Este capítulo convierte el constructo en pruebas que pueden *pasar o fallar*. Nosotros (i) enunciamos hipótesis, (ii) prerregistramos las elecciones de identificación, (iii) definimos métricas de resultado y ventanas de evaluación, (iv) especificamos modelos estadísticos, (v) detallamos la lógica de validación (pruebas de colapso, CC), y (vi) enumeramos los modos de falla que falsificarían el enfoque.

**4.1 Hipótesis**

**H1 — Resiliencia (transversal y variable en el tiempo).**\
Dentro de ambientes comparables, una coherencia de referencia $`{\bar{\alpha}}_{\text{econ}}`$ más alta está asociada con (a) menores reducciones de pico a valle durante choques y (b) recuperaciones más rápidas (vidas medias más cortas de retorno a la tendencia).

**H2 — Anticipación (indicador anticipado).**\
Movimientos negativos grandes y con CC limpio en la coherencia, **eventos de decoherencia** $`\Delta\alpha_{\text{econ}}^{-}(t)`$, predicen estrés macrofinanciero subsecuente (recesiones, indicadores de crisis, estrechamientos de liquidez) con horizontes de 6–18 meses, fuera de muestra.

**H3 — Firma de cascada (multicapa).**\
A través de las capas de agregación (micro → meso → macro) dentro de un régimen fijo, (a) $`\alpha`$ es **no decreciente** con el índice de capa, y (b) el flujo de información dirigido tiene sesgo hacia adelante (micro→meso→macro) según lo evaluado por entropía de transferencia/causalidad de Granger.

**4.2 Prerregistro: estimandos, ventanas y puertas de CC**

Prerregistraremos lo siguiente antes de cualquier inspección de resultados:

- **Estimandos.**

  - $`\alpha_{\mathcal{U},f}`$ : pendientes por contenedor por familia de proxies $`f`$.

  - $`\alpha_{\text{econ}}(t)`$ : fusión de efectos aleatorios (Sección 3) con incertidumbre.

  - **Evento de decoherencia**: $`\Delta\alpha_{\text{econ}}(t) \leq - \theta_{h}`$ sobre horizonte $`h \in \{ 1,3,6\}`$ meses, con $`\theta_{h}`$ establecido en el percentil 10 histórico de cambios o $`k`$ veces el error estándar rodante (prerregistrado $`k`$).

- **Ventanas y muestreo.**

  - Ventanas trimestrales rodantes (primario), paso mensual; ventanas semestrales (sensibilidad).

  - Paneles país-sector para sección transversal; familias de mercado/crédito/logística/información para redundancia multiproxy.

- **Puertas de CC.**

  - Mínimo de **dos** familias de proxies pasando **colapso** dentro de un contenedor.

  - Umbrales de cobertura (mínimo $`N`$ por contenedor, participación mínima del panel presente).

  - Estabilidad de régimen dentro de una ventana (pruebas de punto de cambio).

  - Verificaciones de invariancia de reloj (reescalamiento placebo).

Las observaciones que fallen las puertas de CC serán marcadas y excluidas de las pruebas de hipótesis (mantenidas solo para gráficos descriptivos).

**4.3 Resultados y etiquetas de verdad fundamental**

- **Reducción por choque**: porcentaje de caída de pico a valle en la variable objetivo $`Y`$ (por ejemplo, producción industrial, ventas reales, índice de mercado) durante una ventana de choque identificada por cronología externa o umbrales basados en reglas.

- **Vida media de recuperación**: tiempo para recuperar el 50% de la reducción (o para regresar dentro de una banda de la tendencia pre-choque).

- **Etiquetas de estrés** (para H2): marcadores binarios semanales/mensuales para recesiones (datación oficial del ciclo económico), índices de estrés financiero, crisis de liquidez o estrés de mercado basado en reglas (por ejemplo, decil superior de reducción o explosión de diferenciales).

- **Definiciones de capas** (para H3):

  - **Micro**: nivel de empresa/puerto/ticker;

  - **Meso**: agregados sectoriales/ruta/cluster;

  - **Macro**: agregados de país/sistema de mercado.

**4.4 Identificación: conjuntos de condicionamiento y controles**

Para reducir la confusión:

- **Efectos fijos**: EF de ambiente (país × sector × régimen), EF temporales (trimestre calendario), y donde sea apropiado EF de entidad (empresa/puerto/ticker).

- **Controles** (que **no** colisionen con el escalamiento temporal): nivel de volatilidad, proxies de apalancamiento, profundidad de liquidez, diferenciales de crédito y factores globales (índices de materias primas, cambios de tasa de política). Estos entran como *covariables*, mientras que $`\alpha`$ permanece como el **estimando de pendiente** derivado aguas arriba; los controles no pueden alterar la construcción de pendiente.

**4.5 Pruebas estadísticas**

**H1 — Resiliencia**

**(a) Tamaño de la reducción (sección transversal, panel).**

``` math
\text{Reducción}_{i,s} = \beta_{0} + \beta_{1}\text{ }{\bar{\alpha}}_{\text{econ},i,s}^{(\text{pre})} + \gamma'X_{i,s} + \text{EF} + \varepsilon_{i,s}
```

donde $`i`$ indexa país (o sector), $`s`$ el episodio de choque, $`X`$ controles, y $`{\bar{\alpha}}^{(\text{pre})}`$ es el ICE promedio en la línea base pre-choque. **Predicción:** $`\beta_{1} < 0`$. EE agrupados por $`i`$ y $`s`$.

**(b) Vida media de recuperación (supervivencia AFT/paramétrica).**\
Modelo de tiempo de falla acelerado:

``` math
\log(\text{VidaMedia}_{i,s}) = \delta_{0} + \delta_{1}\text{ }{\bar{\alpha}}_{\text{econ},i,s}^{(\text{pre})} + \phi'X_{i,s} + \text{EF} + \eta_{i,s}.
```

**Predicción:** $`\delta_{1} < 0`$. Robustez: modelo de Cox con ICE como covariable y fragilidad compartida.

**H2 — Anticipación**

**Estudio de eventos y clasificación.**

1.  **Predicción binaria.**\
    Logit/Probit:

``` math
\Pr(\text{Estrés}_{t + h} = 1) = \sigma\text{ }(\theta_{0} + \theta_{1}\text{ }\Delta^{-}\alpha_{\text{econ}}(t) + \psi'Z_{t} + \text{EF}),
```

con horizontes $`h \in \{ 6,12,18\}`$ meses y $`Z_{t}`$ indicadores anticipados estándar. **Predicción:** $`\theta_{1} > 0`$.

2.  **Reglas de puntuación.**\
    Pruebas retrospectivas fuera de muestra con origen rodante; evaluar **AUC**, **puntaje de Brier**, **PR-AUC**; comparar con indicadores canónicos (volatilidad, diferencial de plazo, diferenciales de crédito). Requerir mejora estadísticamente significativa (prueba de DeLong para AUC; Diebold–Mariano para puntajes), controlando por horizontes múltiples.

3.  **Alineación de puntos de cambio.**\
    Gráficos de Kaplan–Meier de tiempo-al-estrés después de eventos de decoherencia vs. ventanas placebo coincidentes; pruebas de log-rank.

**H3 — Firma de cascada**

**(a) Monotonicidad a través de capas.**\
Dentro de ventanas estables en régimen, calcular $`{\widehat{\alpha}}_{\mathcal{l}}`$ para $`\mathcal{l} \in \{\text{micro},\text{meso},\text{macro}\}`$ usando pares $`(L,T)`$ compatibles con la capa. Probar:

``` math
H_{0}:\alpha_{\text{micro}} \geq \alpha_{\text{meso}}\text{ o }\alpha_{\text{meso}} \geq \alpha_{\text{macro}}\text{ vs }H_{A}:\alpha_{\text{micro}} \leq \alpha_{\text{meso}} \leq \alpha_{\text{macro}},
```

usando comparaciones pareadas con ICs bootstrap y control de pruebas múltiples a través de ventanas.

**(b) Direccionalidad (ET/Granger).**\
Calcular **entropía de transferencia** $`TE_{\mathcal{l} \rightarrow \mathcal{l}'}`$ y pruebas de **causalidad de Granger** entre señales de capas compatibles con ICE (por ejemplo, actividad meso vs. agregados macro). **Predicción:** $`TE_{\text{micro} \rightarrow \text{meso}} > TE_{\text{meso} \rightarrow \text{micro}}`$ y similarmente $`\text{meso} \rightarrow \text{macro}`$. Prerregistrar dimensiones de inmersión, órdenes de retardo y pruebas de surrogados para significancia.

**4.6 Lógica de validación y falsificación**

Una hipótesis se **cuenta como aprobada** solo si:

- Las pruebas de **colapso** por contenedor pasan para las familias de proxies contribuyentes;

- Las banderas de CC están limpias;

- Los signos de los efectos coinciden con las predicciones con niveles de significancia prerregistrados;

- El rendimiento fuera de muestra supera las líneas base por márgenes prerregistrados.

El enfoque es **falsificado** para un dominio si, repetidamente a través de regímenes y conjuntos de datos:

- **Sin separación de pendiente** es detectada (α indistinguible de 0 o inestable) en contenedores bien formados;

- **Sin colapso** ocurre después del reescalamiento;

- **Cascada inversa** (α decrece con la agregación) es sistemática;

- **Simetría de direccionalidad** (sin sesgo hacia adelante) persiste después de verificaciones de robustez;

- El contenido predictivo de H2 desaparece relativo a líneas base fuertes bajo evaluación adecuada fuera de muestra.

Los resultados negativos serán documentados y publicados como fronteras de alcance.

**4.7 Comparaciones múltiples, incertidumbre y robustez**

- **Multiplicidad.** Controlar FDR a través de horizontes y capas (Benjamini–Hochberg).

- **Propagación de incertidumbre.** Llevar $`\text{SE}(\widehat{\alpha})`$ de los ajustes por contenedor a los modelos aguas abajo mediante bootstrap paramétrico o variantes jerárquicas bayesianas.

- **Construcciones alternativas.** Reemplazar ODR con Theil–Sen/SIMEX; intercambiar proxies $`L,T`$; variar ventanas; probar alternativas no potenciales (spline $`g\ (\log L)`$).

- **Placebos.** Reescalamientos de reloj, $`L`$ aleatorizado dentro de contenedores, pseudo-eventos coincidentes por controles.

- **Estabilidad.** ICE rodante dejando una familia fuera; umbrales de heterogeneidad ($`{\widehat{\tau}}^{2}`$) para aceptación.

**4.8 Gobernanza de datos y reproducibilidad**

- **Prerregistro** de hipótesis, ventanas, umbrales y clases de modelo.

- **Artefactos versionados**: entradas crudas (donde sean licenciables), constructores de características, definiciones de contenedores, estimaciones de α, series de ICE, banderas de CC y cuadernos de análisis.

- **Auditorías**: recómputo independiente de α en un subconjunto reservado por un equipo separado; protocolos de equipo rojo para sondear fugas o circularidad (por ejemplo, uso de variables de resultado en la construcción de proxies).

**4.9 Consideraciones de potencia (estimación gruesa)**

Dados los tamaños típicos de panel (cientos a miles por contenedor) y ruido de medición moderado, la estimación de α por contenedor mediante EIV arroja EE en el rango de 0.05–0.15. Detectar cambios de $`\Delta\alpha`$ de 0.2–0.3 (un cambio prácticamente significativo en la coherencia) al 5% de significancia con \>80% de potencia es factible con ventanas mensuales/trimestrales sobre períodos multianuales, siempre que al menos dos familias de proxies pasen el colapso.

**4.10 Qué significan el éxito y el fracaso**

- **Éxito**: $`\alpha_{\text{econ}}`$ añade información distinta y robusta sobre la *estructura de la sincronización a través de la escala*, mejorando la inferencia de resiliencia y las alertas tempranas más allá de la volatilidad/liquidez/apalancamiento.

- **Fracaso**: α se comporta como un artefacto inestable de relojes, unidades o regímenes; el colapso rara vez pasa; el valor predictivo es nulo fuera de muestra. En tales casos, la lente RTM *no* es informativa para ese dominio económico, y recomendamos ceñirse a las herramientas clásicas.

**5. Datos y métodos**

Esta sección especifica cómo transformamos flujos económicos crudos en pendientes por contenedor $`\alpha`$ y un Índice de Coherencia Económica en tiempo real **ICE(t)**. Detallamos (i) conjuntos de datos, (ii) construcción de características para pares $`(L,T)`$, (iii) ambientes/contenedores y control de régimen, (iv) algoritmos de estimación (EIV/TLS/SIMEX/robustos), (v) pruebas de colapso, (vi) fusión multiproxy, (vii) nowcasting con CC/manejo de latencia, y (viii) reproducibilidad.

**5.1 Conjuntos de datos (familias y cadencia)**

Organizamos las entradas en **cuatro familias de proxies**. Cada familia es opcional en cualquier ventana dada; el ICE requiere ≥2 familias pasando CC.

**A. Microestructura de mercado (intradiario a diario)**\
Libro de órdenes, transacciones/cotizaciones (L1–L3), cinta consolidada, componentes de índice, acciones corporativas.

**B. Logística y cadenas de suministro (diario a mensual)**\
Escalas portuarias y tiempos de permanencia, reservas de carga, tiempos de entrega de envío, niveles de inventario, metadatos de enrutamiento.

**C. Crédito y financiamiento (diario a mensual)**\
Tasas/volúmenes interbancarios, diferenciales de financiamiento, actividad de renovación/crédito rotativo, escaleras de vencimiento.

**D. Flujo de información (horario a diario)**\
Marcas de tiempo de agencias de noticias, grafos de artículos/enlaces, niveles de audiencia/alcance, embeddings de texto/sentimiento, señales sociales.

**Gobernanza de datos.** Para cada flujo mantenemos una ficha de datos: procedencia, cadencia, cobertura (entidades×tiempo), política de revisiones y restricciones legales/de licencia. Todas las marcas de tiempo se normalizan a UTC; los efectos de calendario de negociación se rastrean.

**5.2 Construcción de características: mapeo a** $`\mathbf{(L,T)}`$

Cada familia produce mediciones pareadas dentro de un **ambiente fijo** $`\mathcal{U}`$ (Sec. 5.3). Calculamos:

**A. Microestructura de mercado**

- **Escala** $`L`$ : nivel de capitalización (cuantiles), nivel de tamaño mediano de transacción, grado/centralidad en redes de impacto cruzado o de correlación.

- **Tiempo** $`T`$ :

  - *Vida media de reversión de microprecio*: ajustar ARMA/ECM a las desviaciones de la cotización media; reportar $`t_{1/2}`$.

  - *Resiliencia del libro de órdenes*: tiempo para reponer la profundidad después de un choque estandarizado.

  - *Persistencia de estabilidad de cotización*: tiempo esperado por encima de un umbral de diferencial.

**B. Logística y cadenas de suministro**

- **Escala** $`L`$ : longitud de ruta (etapas en la lista de materiales), tamaño de ruta (bandas de TEU), nivel de capacidad portuaria.

- **Tiempo** $`T`$ :

  - *Persistencia de tiempo de entrega*: constante de decaimiento de retrasos de envío post-choque.

  - *Decaimiento de tiempo de permanencia*: ajuste de cola exponencial para duraciones de patio/fondeo.

  - *Vida media de reposición de inventario*: tiempo para regresar a las bandas de stock objetivo.

**C. Crédito y financiamiento**

- **Escala** $`L`$ : extensión de la escalera de vencimiento, grado de red (exposiciones interbancarias), nivel de tamaño del libro.

- **Tiempo** $`T`$ :

  - *Renovación de crédito rotativo*: tiempo mediano para refinanciar un tramo vencido.

  - *Reversión a la media del diferencial*: $`t_{1/2}`$ de los choques del diferencial de financiamiento.

  - *Persistencia de cola*: tiempo para que el rezago de emisión primaria se despeje.

**D. Flujo de información**

- **Escala** $`L`$ : nivel de audiencia del medio, centralidad del grafo, alcance jurisdiccional.

- **Tiempo** $`T`$ :

  - *Decaimiento de sentimiento*: tiempo de relajación de la polaridad del tema después de un choque.

  - *Dispersión de desacuerdo*: vida media de la varianza entre fuentes.

  - *Vida media de atención*: tiempo para que las impresiones de un artículo caigan al 50%.

**Notas de medición.**

1.  Calculamos $`L`$ como niveles monótonos o magnitudes en escala logarítmica; $`T`$ es siempre un **tiempo característico** (vida media, constante de decaimiento, retorno al objetivo, resiliencia).

2.  Cada par $`(L,T)`$ lleva metadatos: marca de tiempo, ID de entidad, receta del método, EE para $`T`$ y banderas de calidad (R² de ajuste, diagnósticos residuales).

**5.3 Ambientes y contenedores (estabilidad de régimen)**

Un **contenedor** $`\mathcal{U}`$ se define por: *(país \| moneda) × sector (o mercado) × régimen de política × ventana temporal*.

- **Ventanas temporales.** Primario: ventanas **trimestrales** rodantes con **paso mensual**; sensibilidad: ventanas semestrales.

- **Estabilidad de régimen.** Aplicamos **detección de puntos de cambio** univariada y multivariada (por ejemplo, PELT, Bai–Perron) para garantizar que la política macro, los estándares de reporte o la microestructura de mercado *no* cambien dentro de $`\mathcal{U}`$. Si lo hacen, la ventana se divide y se marca REGIME_MIX.

- **Umbrales de cobertura.** Mínimo de entidades por familia por contenedor (por ejemplo, ≥50 para microestructura; ≥20 rutas/puertos; ≥10 bancos) y mínimo de marcas de tiempo por entidad para estimar $`T`$.

**5.4 Estimación de pendientes por contenedor** $`\mathbf{\alpha}`$ **(errores en variables)**

Para cada $`\mathcal{U}`$ y familia $`f`$, ajustamos:

``` math
\log T_{u}^{\text{obs}} = \alpha_{\mathcal{U},f}\text{ }\log L_{u}^{\text{obs}} + c_{\mathcal{U},f} + \epsilon_{u},
```

permitiendo ruido en ambos ejes.

**Estimadores.**

- **ODR/TLS (predeterminado):** minimiza residuos ortogonales; bueno cuando los errores son comparables.

- **SIMEX (corrección de atenuación):** si podemos estimar $`Var(\xi)`$ de $`\log L`$ a partir de réplicas/instrumentos.

- **Theil–Sen (verificación robusta):** mediana de pendientes por pares; resiste valores atípicos/colas pesadas.

**Incertidumbre.** Bootstrap por bloques/entidades (remuestreo agrupado); ICs percentiles del 95%. Reportamos los **interceptos** $`c_{\mathcal{U},f}`$ en un "libro de pendiente–intercepto" para documentar los cambios de nivel que *no* deberían afectar a $`\alpha`$.

**5.5 Prueba de colapso (validación de escalamiento)**

Después de obtener $`{\widehat{\alpha}}_{\mathcal{U},f}`$, calculamos los resultados residualizados:

``` math
{\widetilde{y}}_{u} = \log\ T_{u}^{\text{obs}} - {\widehat{\alpha}}_{\mathcal{U},f}\text{ }\log\ L_{u}^{\text{obs}}.
```

Probamos la **independencia** de $`\widetilde{y}`$ respecto a $`\log L`$ dentro de $`\mathcal{U}`$ :

- **Estadístico:** $`\Delta_{\text{collapse}} = R^{2}(\widetilde{y} \sim \log\ L)`$.

- **Regla de aprobación:** $`\Delta_{\text{collapse}} < 0.05`$ y sin tendencia visible en gráficos residuales (suavizado no paramétrico \< ancho de banda prerregistrado).

- **Acciones de fallo:** marcar el contenedor-familia como NO_COLLAPSE; excluir de la fusión; anotar en CC.

**5.6 Fusión multiproxy (efectos aleatorios)**

Dadas las estimaciones por familia $`\{{\widehat{\alpha}}_{\mathcal{U},f}\}`$ que pasan el colapso:

``` math
{\widehat{\alpha}}_{\text{econ}}\mathcal{(U) =}\frac{\sum_{f}^{}{w_{f}{\widehat{\alpha}}_{\mathcal{U,}f}}}{\sum_{f}^{}w_{f}},\ \ w_{f} = \frac{1}{{\widehat{\sigma}}_{\mathcal{U,}f}^{2} + {\widehat{\tau}}^{2}},
```

con $`{\widehat{\sigma}}_{\mathcal{U},f}^{2}`$ la varianza bootstrap y $`{\widehat{\tau}}^{2}`$ la heterogeneidad entre familias (REML predeterminado; DerSimonian–Laird como sensibilidad). Publicamos $`{\widehat{\alpha}}_{\text{econ}}`$, bandas del 50/95%, **estadístico Q**, $`{\widehat{\tau}}^{2}`$, y un análisis de influencia **dejando una familia fuera**.

**5.7 De contenedores a ICE(t): pipeline de nowcasting**

**Construcción rodante.**

1.  **Definir ventanas** $`\mathcal{U}_{t}`$ (trimestrales, paso mensual), ejecutar verificaciones de régimen.

2.  **Por familia**, calcular $`{\widehat{\alpha}}_{\mathcal{U}_{t},f}`$, ICs, pruebas de colapso, estadísticas de cobertura.

3.  **Fusionar** en $`{\widehat{\alpha}}_{\text{econ}}(t)`$ con efectos aleatorios.

4.  **Puertas de CC:** requerir ≥2 familias pasando colapso; limitar heterogeneidad ($`{\widehat{\tau}}^{2}`$ por debajo de un umbral prerregistrado); aplicar mínimos de cobertura.

5.  **Suavizado:** aplicar un **EWMA causal** (vida media de 2–3 ventanas) para estabilizar ruido; el suavizado *nunca* se usa en pruebas de hipótesis, solo para publicar la serie titular del ICE.

6.  **Banderas:** adjuntar LOW_COVERAGE, FAMILY_DIVERGENCE, NO_COLLAPSE, REGIME_MIX, CLOCK_SHIFT según corresponda.

**Manejo de latencia.** Cada observación lleva **fecha de corte** y **vintage**. Mantenemos un archivo en tiempo real y recalculamos ICE(t) por vintages para evaluar la sensibilidad a revisiones (gráficos de confiabilidad).

**5.8 Eventos de decoherencia (definición de señal)**

Definimos un **evento de decoherencia** cuando todo se cumple simultáneamente:

- $`{\widehat{\alpha}}_{\text{econ}}(t) - {\widehat{\alpha}}_{\text{econ}}(t - h) \leq - \theta_{h}`$, con $`h \in \{ 1,3,6\}`$ meses y $`\theta_{h}`$ prerregistrado (percentil o $`k \cdot SE`$).

- Las puertas de CC pasan en $`t`$ y en la ventana retrospectiva; sin REGIME_MIX.

- La caída es **confirmable** por al menos **dos** familias individualmente (consistentes en signo, aunque las magnitudes difieran).

Los eventos se marcan temporalmente y luego se alinean con los resultados de estrés en H2.

**5.9 Controles y placebos**

- **Placebos de reloj.** Convertir unidades de tiempo (días↔semanas↔meses) dentro de un contenedor; las pendientes deben permanecer invariantes mientras los interceptos cambian.

- **Placebos de aleatorización.** Permutar etiquetas de $`L`$ dentro de contenedores para estimar distribuciones de pendiente nula (líneas base de atenuación).

- **Formas alternativas.** Ajustar $`\log T = g(\log L)`$ con splines cúbicas; curvatura sistemática entre contenedores falsifica la forma de ley de potencia para ese dominio.

**5.10 Suite de robustez**

- **Intercambio de estimadores.** ODR ↔ Theil–Sen ↔ SIMEX; comparar deltas de $`\widehat{\alpha}`$ y superposición de IC.

- **Intercambio de proxies.** Reemplazar grado por longitud de ruta, tamaño mediano de transacción por nivel de capitalización, etc.; recalcular ICE.

- **Sensibilidad de ventana.** Ventanas semestrales; pasos alternativos; superpuestas vs. disjuntas.

- **Estrés de cobertura.** Submuestrear entidades; verificar degradación de anchos de IC y tasas de colapso.

- **Umbrales de heterogeneidad.** Variar el $`{\widehat{\tau}}^{2}`$ aceptable y remarcar CC.

**5.11 Software, computación y artefactos**

- **Pila tecnológica.** Python/R para ingeniería de datos; ODR (scipy), SIMEX (paquete personalizado o de R), Theil–Sen (statsmodels), puntos de cambio (ruptures/strucchange), metaanálisis (metafor/py-meta).

- **Pipelines.** DAGs reproducibles (por ejemplo, make, dvc, prefect) con semillas deterministas.

- **Artefactos.** Tablas versionadas en parquet/feather para: características, definiciones de contenedores, $`{\widehat{\alpha}}_{\mathcal{U},f}`$, estadísticas de colapso, salidas de fusión, ICE(t) con banderas de CC y todas las figuras.

- **Documentación.** Especificación YAML para cada proxy: fórmulas, filtros, convenciones de unidades, política de datos faltantes.

- **Pruebas.** Verificaciones de IC para (i) invariancia ante cambios de unidad (placebos de reloj), (ii) reproducibilidad de $`\widehat{\alpha}`$ con tolerancia de 1e-6 en una muestra fija, (iii) cotas de estadísticas de colapso.

**5.12 Privacidad y ética**

- **Agregación.** Publicar solo salidas a nivel de contenedor e índice; suprimir identificadores micro a menos que se consientan explícitamente y se anonimicen.

- **Sesgo.** Monitorear la participación por familia por país/sector para evitar que el ICE refleje riqueza de datos en lugar de estructura; incluir una bandera LOW_COVERAGE y abstenerse de inferencia cuando esté marcada.

- **Ciencia abierta.** Prerregistrar hipótesis y umbrales; liberar código y réplicas sintéticas donde las licencias impidan compartir datos crudos.

**5.13 Resumen**

Transformamos flujos económicos heterogéneos en pares $`(L,T)`$ coherentes, estimamos **pendientes por contenedor** con corrección de error de medición, **validamos el escalamiento** mediante pruebas de colapso, **fusionamos** estimaciones multiproxy bajo efectos aleatorios, y publicamos un **ICE(t)** con CC, incertidumbre y rastreo de latencia. El pipeline está expresamente diseñado para ser **falsificable** (modos de falla claros), **auditable** (artefactos versionados), y **complementario** a los indicadores clásicos de volatilidad/liquidez/apalancamiento.

**6. Resultados — Pruebas retrospectivas**

*Esta sección muestra cómo se comporta el Índice de Coherencia Económica (ICE) en datos históricos. Dado que este es un artículo de métodos, enfatizamos plantillas transparentes, criterios de aprobación/fallo, incertidumbre y hallazgos negativos. Los números concretos son marcadores de posición que ilustran el formato de reporte; el paquete de replicación prerregistrado los reemplazará con estimaciones reales.*

**6.1 Configuración y protocolo de evaluación**

**Ventanas.** Contenedores trimestrales rodantes (paso mensual) para cada país × sector × régimen.\
**Familias.** Al menos dos de: *Microestructura de mercado, Logística, Crédito, Información*.\
**Puertas de aceptación.** En cada contenedor, las familias deben pasar **colapso** y umbrales de cobertura; heterogeneidad $`({\widehat{\tau}}^{2})`$ dentro de límites.\
**Incertidumbre.** Bandas del 50/95% para $`{\widehat{\alpha}}_{\text{econ}}(t)`$; ICs agrupados/bootstrap fluyen a las pruebas aguas abajo.\
**Líneas base.** Índices de volatilidad, diferencial de plazo, diferenciales de crédito, índices compuestos de estrés financiero (IEF).\
**Métricas prerregistradas.** AUC/PR-AUC, puntaje de Brier, coeficientes Cox/AFT, pruebas Diebold–Mariano vs. líneas base.

**Figura 1 (plantilla).** *ICE(t) con bandas del 50/95% y banderas de CC,* junto con líneas base (escaladas).\
**Tabla 1 (plantilla).** *Tasas de aprobación de colapso,* por familia, por régimen.

**6.2 H1 — Resiliencia (reducciones y recuperaciones)**

**6.2.1 Reducciones transversales**

Regresamos las reducciones de pico a valle durante cada episodio de choque sobre el $`{\bar{\alpha}}_{\text{econ}}`$ **pre-choque** con controles y efectos fijos.

**Tabla 2 (plantilla).** Regresiones de reducción

- $`\beta_{1}`$ sobre $`{\bar{\alpha}}_{\text{econ}}`$ (se espera **negativo**)

- Controles: volatilidad, apalancamiento, liquidez; EF: país×episodio

- EE agrupados; $`R^{2}`$; $`N`$ contenedores

*Formato ilustrativo:*\
$`\beta_{1} = - 0.28\lbrack - 0.41, - 0.15\rbrack`$, $`p < 0.001`$. Interpretación: un aumento de +0.5 en la coherencia de referencia se asocia con ~14% menores reducciones, ceteris paribus.

**6.2.2 Vidas medias de recuperación**

Modelo de tiempo de falla acelerado (AFT) para el tiempo de recuperación al 50%.

**Tabla 3 (plantilla).** Estimaciones AFT\
$`\delta_{1}`$ sobre $`{\bar{\alpha}}_{\text{econ}}`$ (se espera **negativo**); términos de fragilidad; concordancia.

*Formato ilustrativo:*\
$`\delta_{1} = - 0.35\lbrack - 0.52, - 0.18\rbrack`$. Interpretación: mayor coherencia predice recuperaciones más rápidas (vidas medias más cortas), más allá de los efectos de volatilidad/liquidez.

**Figura 2 (plantilla).** Curvas de Kaplan–Meier estratificadas por terciles de $`{\bar{\alpha}}_{\text{econ}}`$.

**Robustez.** Resultados estables bajo: (i) ventanas semestrales, (ii) ICE dejando una familia fuera, (iii) definiciones alternativas de resultado (retorno a tendencia vs. a banda pre-choque).

**6.3 H2 — Anticipación (señal anticipada de estrés)**

Definimos **eventos de decoherencia** como caídas con CC limpio en el ICE que exceden umbrales prerregistrados sobre $`h \in \{ 1,3,6\}`$ meses.

**6.3.1 Rendimiento de clasificación**

Logit/Probit prediciendo estrés a horizontes $`h = 6,12,18`$ meses.

**Tabla 4 (plantilla).** Rendimiento fuera de muestra

- AUC, PR-AUC, puntaje de Brier para: *(i)* solo ICE, *(ii)* Líneas base, *(iii)* ICE + Líneas base

- Pruebas de DeLong y Diebold–Mariano; control de multiplicidad (FDR).

*Formato ilustrativo:*\
A $`h = 12`$ m, AUC solo-ICE 0.72 (0.68–0.76); Líneas base 0.66 (0.62–0.70); **Combinado** 0.77 (0.73–0.80), el ICE añade valor incremental significativo ($`p < 0.01`$).

**6.3.2 Alineación de eventos**

Análisis de supervivencia desde eventos de decoherencia hasta la primera señal de estrés.

**Figura 3 (plantilla).** Curvas de tiempo-al-estrés para **eventos de ICE** vs. ventanas placebo coincidentes; valores $`p`$ de log-rank.

**Distribución del tiempo de anticipación.** Mediana de anticipación 9–14 meses (ilustrativo), con rango intercuartil reportado por régimen.

**6.3.3 Falsos positivos y negativos**

- **Falsos positivos** (caídas del ICE sin estrés subsecuente): catalogados con notas de CC (por ejemplo, REGIME_MIX cercano a la ventana, o choques sectoriales idiosincráticos que se revierten).

- **Omisiones** (estrés sin caída previa del ICE): analizados por problemas de **cobertura** (pocas familias) o **fragilidad de proxy**.

**6.4 H3 — Firma de cascada (monotonicidad de capas y dirección)**

**6.4.1 Monotonicidad de capas**

Calcular $`{\widehat{\alpha}}_{\text{micro}},{\widehat{\alpha}}_{\text{meso}},{\widehat{\alpha}}_{\text{macro}}`$ dentro de ventanas estables en régimen. Probar el orden no decreciente con ICs bootstrap.

**Tabla 5 (plantilla).** α por capas con diferencias pareadas

- Proporción de ventanas donde $`\alpha_{\text{micro}} \leq \alpha_{\text{meso}} \leq \alpha_{\text{macro}}`$ (se espera alta).

- Violaciones documentadas por régimen.

**6.4.2 Pruebas de direccionalidad**

Entropía de transferencia (ET) y causalidad de Granger entre agregados de capas compatibles con la construcción del ICE.

**Figura 4 (plantilla).** Flechas de ET (micro→meso→macro) con bandas de confianza; pruebas de surrogados para significancia.

*Plantilla de interpretación:* El sesgo hacia adelante se mantiene en X% de las ventanas (controlado por FDR), consistente con cascada; las excepciones coinciden con REGIME_MIX o rupturas estructurales.

**6.5 Robustez, ablaciones y diagnósticos**

**6.5.1 Intercambios de estimador**

ICE recalculado con correcciones Theil–Sen y SIMEX.

**Tabla 6 (plantilla).** $`\Delta\widehat{\alpha}`$ vs. ODR; tasas de superposición de IC; cambios de heterogeneidad $`({\widehat{\tau}}^{2})`$.\
Formato del resultado: cambio absoluto mediano ≤ 0.06; sin efecto material en las decisiones de H1–H3.

**6.5.2 Intercambios de proxy**

Dentro de las familias, proxies alternativos de $`L,T`$ (por ejemplo, grado↔longitud de ruta, capitalización↔tamaño de transacción).

**Figura 5 (plantilla).** Diagrama de araña de contribuciones de familias al ICE bajo intercambios de proxy; carriles de estabilidad.

**6.5.3 Sensibilidad de ventana/cobertura**

- Ventanas semestrales vs. trimestrales; diferentes pasos.

- Submuestreo de entidades para estresar la cobertura.

- Tasas de banderas de CC (LOW_COVERAGE, FAMILY_DIVERGENCE, NO_COLLAPSE) reportadas.

**Tabla 7 (plantilla).** Incidencia de banderas de CC e impacto en las pruebas de hipótesis.

**6.5.4 Placebos y diagnósticos nulos**

- **Placebos de reloj:** reescalamientos de unidades cambian interceptos pero preservan pendientes (tasas de aprobación reportadas).

- **Nulos de aleatorización:** distribuciones de pendiente bajo etiquetas de $`L`$ permutadas (deben centrarse cerca de 0–línea base de atenuación).

- **Alternativa no potencial:** pruebas de curvatura con spline $`g(\log L)`$, fracción de contenedores donde se rechaza la ley de potencia.

**6.6 Resultados negativos y condiciones de alcance**

Documentamos los dominios donde el ICE **falla** (por diseño):

- **Sin separación de pendiente:** α inestable o indistinguible de 0 a pesar de buena cobertura → RTM no informativo en esa capa (registrado).

- **Sin colapso:** tendencias residuales persistentes después del reescalamiento → regímenes mixtos o forma funcional incorrecta (excluir).

- **Cascada inversa:** $`\alpha_{\text{macro}} < \alpha_{\text{micro}}`$ sistemático en estado estacionario → considerar arquitecturas alternativas; RTM puede no aplicar.

- **Simetría de direccionalidad:** ET no muestra sesgo hacia adelante después de surrogados → la afirmación de cascada falla para ese régimen.

**Tabla 8 (plantilla).** Registro de resultados negativos

- Dominio, régimen, razón, diagnósticos, acción (excluir / revisar proxies / modelo alternativo).

**6.7 Lo que implican los resultados (síntesis)**

1.  **Estructura, no niveles.** Donde el colapso pasa, $`\alpha`$ captura un **gradiente de tiempo–escala** de la economía más allá de la volatilidad/liquidez.

2.  **Lente de resiliencia.** Mayor coherencia pre-choque se alinea con reducciones más superficiales y recuperaciones más rápidas.

3.  **Alertas tempranas.** Los eventos de decoherencia a menudo **anticipan** el estrés por trimestres, y añaden valor más allá de las líneas base familiares.

4.  **Las cascadas son estratificadas.** El flujo de información con sesgo hacia adelante y el $`\alpha`$ no decreciente a través de capas aparecen en regímenes estables, precisamente donde el diseño de políticas puede influir en amortiguadores y transparencia.

**6.8 Lista de verificación de replicación (lo que un lector debería poder rehacer)**

- Recalcular $`\widehat{\alpha}`$ por contenedor para cada familia aceptada con el código publicado y los vintages de datos.

- Verificar que los resultados de colapso aprobación/fallo y las banderas de CC coincidan.

- Recrear ICE(t), bandas de incertidumbre y marcas de tiempo de eventos de decoherencia.

- Reejecutar las pruebas H1–H3 con nuestras semillas para reproducir tablas/figuras dentro de la tolerancia.

- Intercambiar estimadores/proxies y ver envolventes de estabilidad similares a las nuestras.

**7. Discusión**

Esta sección interpreta qué mide el **ICE**, cómo difiere de indicadores familiares, dónde es más informativo, cuándo *no* debería usarse y cómo interpretar éxitos y fracasos. También consideramos explicaciones alternativas, límites de identificación causal e implicaciones para el diseño y la política (ampliadas en la Sección 8).

**7.1 Qué mide realmente el ICE**

**El ICE es una *pendiente* estructural**: el gradiente del **tiempo** característico respecto a la **escala** dentro de un ambiente fijo. Donde el colapso pasa, $`\alpha_{\text{econ}}`$ resume *cuán rápido se estira la sincronización al moverse hacia arriba en la escalera de tamaño/agregación*. No es un índice de volatilidad ni un velocímetro de toda la economía; es un estadístico de *geometría del tempo*:

- $`\alpha`$ **alto** → la sincronización aumenta pronunciadamente con la escala: las unidades grandes son más lentas respecto a las pequeñas. Esto usualmente indica *estratificación, amortiguadores y flujo de información filtrado*, rasgos correlacionados con resiliencia pero que potencialmente reducen el rendimiento bruto en las escalas más grandes.

- $`\alpha`$ **bajo** → la sincronización aumenta débilmente con la escala: la propagación es rápida entre capas. Esto impulsa el rendimiento a corto plazo pero aumenta la posibilidad de fallas sincronizadas.

Dado que los **cambios de reloj/nivel** residen en el intercepto, el ICE es comparativamente robusto a rebasaciones, cambios de unidades y algunos cambios de nivel a nivel de régimen, *siempre que* el ambiente se mantenga correctamente fijo.

**7.2 Cómo el ICE complementa las señales familiares**

- **Volatilidad (por ejemplo, VIX):** dispersión a una escala dada; puede ser alta tanto en regímenes coherentes como incoherentes. El ICE captura la *estructura de sincronización entre escalas* que la volatilidad no puede ver.

- **Profundidad de liquidez/diferenciales:** fricciones transaccionales; pueden mejorar a medida que $`\alpha`$ sube (flujo escalonado) o caer si el amortiguamiento obstruye la ejecución. Sin relación de signo fija.

- **Apalancamiento/diferenciales de crédito:** presión del balance; puede co-moverse con el ICE pero es conceptualmente distinto. Un sistema altamente apalancado puede permanecer coherente, hasta que el apalancamiento fuerza la des-estratificación y $`\alpha`$ cae.

- **Indicadores del ciclo económico (PMIs, desempleo):** dinámica de niveles; el ICE puede anticipar o rezagarse dependiendo de si la coherencia se reorganiza *antes* de que los niveles se muevan.

**Resultado neto:** Tratar el ICE como un *tercer eje* —estructura del tiempo a través de la escala— ortogonal al nivel y la dispersión.

**7.3 Mecanismos: por qué la coherencia tiende a ayudar a la resiliencia**

Tres canales genéricos explican los patrones H1/H2 que observamos cuando los contenedores pasan el colapso:

1.  **Amortiguamiento y escalonamiento.** Las unidades más grandes mantienen inventarios, amortiguadores de capital y puntos de control de decisión. A medida que $`\alpha`$ sube, las perturbaciones se disipan en cada etapa, alargando la sincronización macro pero reduciendo el estrés pico.

2.  **Filtrado de información.** Los sistemas coherentes desaceleran las cascadas de rumores y los bucles de reflejo algorítmico, reduciendo el sobrepaso por retroalimentación.

3.  **Relojes heterogéneos.** Cuando las capas funcionan a tempos diferenciados (grande lento, pequeño rápido), la sincronización entre capas es más difícil; los choques luchan por fijar todas las escalas en la misma fase.

Estos mecanismos pueden ser diseñados (gobernanza, divulgación, interruptores de circuito, redundancia) y, crucialmente, *medidos* con proxies $`(L,T)`$. También clarifican la compensación: un $`\alpha`$ alto puede reducir el rendimiento bruto o la "velocidad" titular, lo cual a veces se lee erróneamente como ineficiencia.

**7.4 Explicaciones alternativas (y cómo nos protegemos contra ellas)**

1.  **Regímenes de volatilidad disfrazados de coherencia.**\
    Si la alta volatilidad alarga el $`T`$ observado uniformemente, las pendientes podrían empinarse mecánicamente. Mitigamos (i) estimando **dentro** de contenedores de ambiente fijo, (ii) incluyendo la volatilidad como **control** en los modelos H1/H2, y (iii) requiriendo que el **colapso** pase (los cambios de nivel uniformes solos no pasarán).

2.  **Relojes de medición y artefactos de unidades.**\
    Ejecutamos **placebos de reloj** (días↔semanas↔meses) para asegurar que las pendientes son invariantes mientras los interceptos se mueven; mantenemos un **libro de pendiente–intercepto**. Los fallos aquí invalidan contenedores.

3.  **Endogeneidad/selección.**\
    La coherencia puede ser *elegida* en anticipación de choques (causalidad inversa). Por lo tanto (i) prerregistramos ventanas/umbrales, (ii) usamos evaluación fuera de muestra para H2, y (iii) en extensiones, aprovechamos instrumentos o diferencias en diferencias donde las políticas crean cambios exógenos de coherencia (por ejemplo, mandatos de divulgación, reglas de interruptores de circuito).

4.  **Capas confundidas.**\
    Si $`L`$ y $`T`$ no pertenecen a la misma capa de proceso, surgen pendientes espurias. Nuestra **regla de compatibilidad** y el **colapso** a nivel de contenedor están diseñados para fallar en ese caso — por diseño, un fallo informativo.

**7.5 Condiciones de alcance: dónde el ICE es (y no es) útil**

**Funciona mejor cuando:**

- La estructura es cuasiestacionaria dentro de contenedores (política/microestructura estable).

- Múltiples familias independientes de $`(L,T)`$ están disponibles (≥2) con cobertura aceptable.

- Los procesos de sincronización son *generados internamente* (renovación/relajación) en lugar de completamente regulados por políticas.

**Debe evitarse o marcarse cuando:**

- REGIME_MIX: rupturas estructurales rápidas dentro de ventanas.

- LOW_COVERAGE: muy pocas entidades por familia; pendientes inestables.

- **ICE de familia única**: sin redundancia, reportar pero abstenerse de inferencia.

- **Forma no potencial**: las pruebas de spline revelan curvatura consistente (forma RTM rechazada).

**7.6 Interpretando niveles y cambios entre regímenes**

- **Comparación entre países.** Comparar solo cuando las *definiciones de contenedor coincidan* (por ejemplo, estándares de reporte similares y microestructura de mercado). El ICE no es un ranking universal; es *contextual*.

- **Sector vs. mercado.** Los sectores con escalonamiento diseñado (servicios públicos, farmacéuticas) a menudo exhiben $`\alpha`$ más alto que los sectores hipercompetitivos de justo a tiempo. Las políticas que fuerzan transparencia y amortiguamiento pueden desplazar $`\alpha`$ hacia arriba.

- **Rupturas de tendencia.** Un aumento *persistente* del ICE después de un choque a menudo refleja reorganización deliberada (estrategias de inventario, redundancia, gobernanza). Un pico transitorio con alta heterogeneidad puede ser ruido o artefactos de medición.

**7.7 Causalidad: lo que podemos y no podemos afirmar**

El ICE es observacional y **estructural-descriptivo**. H1/H2/H3 proporcionan evidencia *predictiva* y *asociativa*. Para argumentar **causalidad**, necesitamos:

- Choques exógenos o cuasiexógenos a la coherencia (experimentos naturales de política).

- Variables instrumentales que desplacen $`\alpha`$ pero no los resultados excepto a través de $`\alpha`$.

- Intervenciones piloto aleatorizadas (por ejemplo, cadencia de divulgación obligatoria, diseños de interruptores de circuito) con medición de α pre/post.

Hasta que dichos diseños se ejecuten, recomendamos formular las afirmaciones causales con cautela ("asociado con", "predictivo de").

**7.8 Riesgo de modelo y sobreajuste**

- **Proliferación de proxies.** Más proxies aumentan la cobertura pero elevan el riesgo de pruebas múltiples; lo contenemos prerregistrando familias, usando fusión de efectos aleatorios y publicando **resultados negativos**.

- **Mala especificación de EIV.** Si los errores de $`L`$ se estiman incorrectamente, las correcciones SIMEX pueden sesgar pendientes; por lo tanto publicamos resultados de **intercambio de estimador** (ODR vs. Theil–Sen vs. SIMEX).

- **Sesgo prospectivo.** Todas las series ICE(t) se calculan por **vintage**, y las pruebas de hipótesis usan solo información disponible a esa fecha.

**7.9 Relación con el corpus más amplio de RTM**

La economía hereda la misma disciplina de **pendiente primero** vista en los dominios físicos y biológicos de RTM: **escalamiento por contenedor**, **validación de colapso** y **firmas de cascada**. Conceptualmente, $`\alpha_{\text{econ}}`$ juega el papel de un *exponente de coherencia* análogo a la cinética de ambiente controlado de la química o los gradientes de persistencia de la meteorología. Los fallos (sin pendiente, sin colapso) no son errores; son **fronteras de alcance** — señales de que, en ese dominio o régimen, el escalamiento simple de RTM no describe la sincronización.

**7.10 Guía práctica de lectura para profesionales**

- **Si el ICE está cayendo** con CC limpio: prepararse para propagación *más rápida* entre escalas, ajustar amortiguadores de liquidez, ensayar planes de contingencia y verificar exposiciones correlacionadas.

- **Si el ICE está subiendo** de manera sostenida: explorar compensaciones de rendimiento — ¿pueden algunos amortiguadores ser racionalizados sin erosionar la resiliencia?

- **Si las familias divergen** (alto $`{\widehat{\tau}}^{2}`$): investigar rupturas de medición o idiosincrasias sectoriales antes de actuar.

- **Si las banderas de CC se activan**: tratar el ICE como *contexto* informativo, no como disparador de decisión.

**7.11 Consideraciones éticas y de equidad**

La coherencia puede ser *diseñada* de maneras que inadvertidamente desventajen a entidades más pequeñas (por ejemplo, cargas de divulgación). Cualquier uso de política del ICE debería:

- Publicar la metodología y el CC de manera transparente.

- Incluir **evaluaciones de impacto de equidad** (¿se penaliza sistemáticamente a las pequeñas empresas o regiones de bajos ingresos?).

- Preferir **incentivos** (estándares, herramientas) sobre **penalizaciones** que afiancen la incumbencia.

- Respetar la privacidad de datos y las licencias; liberar réplicas sintéticas cuando el intercambio de datos crudos esté restringido.

**7.12 Conclusiones clave**

1.  **El ICE es una lente estructural**: mide la *forma* de la sincronización a través de las escalas, no niveles ni ruido instantáneo.

2.  **Resiliencia ↔ coherencia**: un $`\alpha`$ más alto a menudo se alinea con menores reducciones y recuperaciones más rápidas, pero con una compensación de rendimiento.

3.  **Alerta temprana**: las caídas limpias del ICE frecuentemente preceden al estrés — valiosas junto a, no en lugar de, indicadores clásicos.

4.  **Aplicabilidad limitada**: donde el colapso falla o los regímenes se mezclan, no forzar RTM — registrar el resultado negativo y recurrir a herramientas específicas del dominio.

5.  **La accionabilidad** proviene de una **interpretación consciente del CC**, redundancia entre familias y, eventualmente, diseños causales que pasen de la predicción a la política.

**8. Implicaciones de política y diseño**

Esta sección convierte el **ICE** de un diagnóstico en **orientación de diseño**. Delineamos (i) pruebas de estrés conscientes de la coherencia, (ii) estándares de divulgación que hacen $`\alpha_{\text{econ}}`$ medible, (iii) patrones de diseño de estructura de mercado y cadena de suministro, (iv) usos macroprudenciales, (v) manuales operativos para instituciones públicas y privadas, y (vi) barandillas de gobernanza. El tema es simple: **diseñar el tempo a través de las escalas** para que los choques se disipen en lugar de amplificarse, *sin* congelar el flujo productivo.

**8.1 Pruebas de estrés conscientes de la coherencia**

**Objetivo.** Ir más allá de los choques de nivel (PIB, ratios de capital) hacia **choques de sincronización entre escalas**: "¿qué pasa si el gradiente de tiempo se aplana (ICE↓) o se empina (ICE↑)?"

**Bloque de prueba A — Choques de pendiente.**

- **Escenario A1 (Decoherencia):** imponer $`\Delta\alpha_{\text{econ}} = - 0.3`$ durante 2–3 ventanas con condiciones de CC limpias; propagar a través de la sincronización sectorial insumo-producto: vidas medias de inventario más cortas, cascadas de rumores más rápidas, resiliencia reducida del libro de órdenes.

- **Escenario A2 (Sobre-estratificación):** imponer $`\Delta\alpha_{\text{econ}} = + 0.3`$; propagar tiempos de liquidación y reposición más largos; evaluar pérdida de rendimiento vs. mitigación de reducciones.

**Métricas.** Resultados de pico a valle, vida media de recuperación, índices de sincronización (acoplamiento de fase entre capas) y multiplicadores de contagio.

**Criterios de aprobación.** (i) Los servicios críticos permanecen por encima de umbrales prerregistrados de continuidad; (ii) sin acoplamiento de fase a través de \>2 capas en A1; (iii) en A2, pérdida de rendimiento ≤ tolerancia de política.

**8.2 Estándares de divulgación que hacen visible a** $`\mathbf{\alpha}`$

**Problema.** Muchas jurisdicciones recopilan niveles y datos de balance pero no **tiempos característicos**.

**Divulgación mínima viable (por sector).**

- **Logística:** distribuciones de tiempo de entrega, colas de tiempo de permanencia, cadencia de reorden (anonimizado).

- **Crédito:** escaleras de vencimiento, ventanas de renovación, tasas de renovación por plazo.

- **Mercados:** métricas de resiliencia del libro de órdenes, vidas medias estandarizadas de reversión de microprecio.

- **Información:** latencia de corrección/errata, cadencia editorial, marcas de tiempo de API.

**Estándar.** Publicar **cuantiles de tiempos característicos** y las **definiciones de contenedor** (metadatos del ambiente). Esto permite a terceros calcular $`\alpha`$ sin exponer identificadores micro.

**8.3 Patrones de estructura de mercado que elevan la coherencia (sin matar el rendimiento)**

**M1 — Interruptores de circuito escalonados (conscientes del tiempo).**\
Pausas escalonadas vinculadas a condiciones *entre escalas* (por ejemplo, resiliencia de microestructura fallando a través de niveles de capitalización), en lugar de paradas de umbral único. **Efecto:** aumenta $`\alpha`$ transitoriamente para prevenir el acoplamiento de fase.

**M2 — Subastas de reposición de profundidad.**\
Micro-subastas activadas cuando la profundidad del libro de órdenes cae por debajo de umbrales escalonados; restauran el escalonamiento sin pausas prolongadas.

**M3 — Desincronización de relojes.**\
Micro-desfases aleatorios en subastas por lotes o reportes — pequeños pero suficientes para prevenir el comportamiento de manada algorítmico.

**M4 — Transparencia sobre sincronización en lugar de volumen bruto.**\
Obligar a publicar métricas de resiliencia/vida media junto con estadísticas de liquidez; los mercados compiten en calidad de *velocidad de recuperación*, no solo en diferencial.

**8.4 Patrones de cadena de suministro y operaciones**

**S1 — Fijación de amortiguadores por** $`\alpha`$ **.**\
Vincular stocks de seguridad y puntos de reorden al $`\widehat{\alpha}`$ sectorial: cuando el ICE cae, ampliar automáticamente amortiguadores para insumos críticos; cuando el ICE sube, permitir normalización escalonada.

**S2 — Enrutamiento multiruta por banderas de decoherencia.**\
Cuando se activa ICE_EVENT, cambiar a conjuntos de rutas que reduzcan la **varianza de longitud de ruta** (no necesariamente la más corta), estabilizando $`T`$.

**S3 — Compras cadenciadas.**\
Evitar mega-órdenes sincronizadas; aplicar **desfases de fase** entre proveedores para mantener la heterogeneidad de sincronización.

**S4 — Simulacros de contención.**\
Tratar la decoherencia como un incidente cibernético: manuales para desacelerar la propagación entre capas (cuotas temporales, horarios escalonados, depósitos alternativos).

**8.5 Usos macroprudenciales**

**P1 — Amortiguador ICE contracíclico.**\
Análogo al CCyB: cuando el ICE cae por debajo de un umbral percentil (CC limpio), elevar los amortiguadores contracíclicos de capital/liquidez; relajar a medida que el ICE se normaliza.

**P2 — Armónicos de vencimiento.**\
Desincentivar el agrupamiento excesivo de vencimientos corporativos o soberanos (reducir el acoplamiento de fase); ofrecer incentivos para **escaleras escalonadas**.

**P3 — Gobernanza de cadencia de divulgación.**\
Estabilizar $`\alpha`$ estableciendo **ventanas de anuncio predecibles** (impresiones macro, actualizaciones de política) para evitar sorpresas en cascada.

**P4 — Contingencia interbancaria por sincronización.**\
Probar estrés en *tiempos de renovación* en lugar de solo niveles; pre-arreglar facilidades vinculadas a vidas medias de renovación, no solo a diferenciales.

**8.6 Operacionalización del sector público (hoja de ruta)**

**Fase 0 — Línea base.** Construir un **laboratorio ICE**: calcular ICE retrospectivo con datos públicos; publicar metodología, tasas de aprobación de colapso, CC.

**Fase 1 — Piloto.**

- Seleccionar 2–3 sectores con buena cobertura; ejecutar **ICE(t) en vivo** durante 12 meses.

- Integrar con paneles de estrés existentes; definir manuales de respuesta a **eventos ICE**.

**Fase 2 — Estandarización.**

- Emitir **plantillas de divulgación de sincronización**; incorporar entidades reguladas.

- Convocar un **grupo de trabajo ICE** (oficinas de estadística, banco central, operadores de mercado, agencias de cadena de suministro).

**Fase 3 — Integración de política.**

- Vincular **herramientas contracíclicas** (amortiguadores, facilidades) a disparadores del ICE;

- Publicar **informes de transparencia** (frecuencia de uso de banderas del ICE; resultados).

**8.7 Operacionalización del sector privado**

**Empresas y fondos.**

- Añadir ICE a los paneles de riesgo; realizar **escenarios de equipo rojo** de decoherencia.

- Incorporar **reglas condicionadas al ICE**: por ejemplo, topes de apalancamiento, escalamiento de VaR, amortiguadores de inventario.

- Compras y tesorería coordinan **desfases de fase de vencimiento/pedidos**.

- Relaciones con inversores publican **KPIs de sincronización** (vida media de recuperación, vida media de reposición).

**8.8 Gobernanza, equidad y riesgos de uso indebido**

**Barandillas.**

- **Sin tiranía de número único.** Publicar **banderas de CC** y **bandas de incertidumbre**; nunca obligar acciones basándose solo en el ICE.

- **Acceso igualitario.** Las divulgaciones de sincronización deben ser **públicas** (o licenciadas simétricamente) para evitar ventajas de información privilegiada.

- **Sesgo contra PyMEs/ME.** Proporcionar herramientas/apoyo para que las pequeñas empresas y los mercados emergentes puedan cumplir las divulgaciones de sincronización sin carga indebida.

- **Privacidad.** Publicar **cuantiles agregados** y réplicas sintéticas; auditorías independientes para riesgo de reidentificación.

- **Auditabilidad.** Mantener un **libro de pendiente–intercepto** y archivos de vintage; permitir la recómputo por terceros.

**8.9 Plantillas de implementación (listas para usar)**

**Plantilla A — Disparador de política ICE (público).**

- **Disparador:** $`ICE(t)\  - \ ICE(t - 3m)\  \leq \  - \theta`$, CC limpio, dos familias confirman.

- **Acciones:** elevar CCyB en X pb; activar facilidades de liquidez vinculadas a *vidas medias de renovación*; instruir a operadores de mercado para habilitar **M1/M2**.

- **Caducidad:** revisión automática a +6 meses; revertir si el ICE se normaliza y el estrés está ausente.

**Plantilla B — Manual corporativo (privado).**

- **Disparador:** $`ICE\_ EVENT`$ en sector/región.

- **Acciones:** ampliar stocks de seguridad en y%; aplicar cadencia **S3**; diversificar vencimientos; ajustar aceleradores algorítmicos; enviar avisos de desfase de fase a proveedores.

- **KPIs:** contención de reducción, vida media hasta recuperación del nivel de servicio, exposición al acoplamiento de fase (proporción de proveedores sincronizados).

**8.10 Límites y consecuencias no deseadas**

- **Riesgo de sobre-estratificación.** "Elevar $`\alpha`$ " ciegamente puede inflar la burocracia; aplicar **topes de rendimiento** y *cláusulas de caducidad*.

- **Gaming.** Las entidades podrían escalonar cosméticamente los reportes; requerir validación de colapso **ex post** y auditorías aleatorias.

- **Fallas de coordinación.** Si solo parte de una red cambia la cadencia, las fricciones temporales pueden aumentar; usar **corredores piloto** antes de despliegues nacionales.

**8.11 Resumen**

La política y el diseño pueden **moldear el tempo a través de las escalas**. El ICE proporciona un **manejo medible y falsificable** de esa estructura, habilitando pruebas de estrés, divulgaciones e intervenciones **conscientes de la coherencia**. El principio rector son los **relojes diferenciados**: suficiente escalonamiento para prevenir cascadas, no tanto que se estrangule el rendimiento. Con CC transparente, salvaguardas de equidad y replicación abierta, el ICE puede sentarse junto a la volatilidad, la liquidez y el apalancamiento como un **tercer eje** para economías resilientes.

**9. Limitaciones**

Esta sección hace explícito dónde **RTM–ICE** puede engañar, fallar o ser superado por enfoques clásicos. Agrupamos las limitaciones en **datos**, **medición**, **identificación**, **forma del modelo**, **operacionalización** y **validez externa**, y declaramos qué evidencia cambiaría nuestra opinión.

**9.1 Limitaciones de datos**

- **Heterogeneidad de cobertura.** Algunas familias de proxies son ricas para mercados desarrollados (microestructura) pero escasas para logística/crédito en economías más pequeñas. **Riesgo:** el ICE refleja *dónde existen datos*, no coherencia. **Mitigación:** banderas LOW_COVERAGE, puertas de cobertura mínima, publicar mapas de participación, abstenerse de inferencia cuando esté marcado.

- **Latencia y revisiones.** Las series de logística/crédito pueden llegar tarde o ser revisadas. **Riesgo:** oscilaciones espurias del ICE y sesgo retrospectivo. **Mitigación:** contabilidad de vintage, divulgación de latencia, pruebas retrospectivas en tiempo real sobre vintages congelados.

- **Rupturas en estándares de reporte.** Los cambios regulatorios o de proveedor pueden desplazar el $`T`$ medido sin cambio estructural. **Riesgo:** desplazamientos escalonados en el intercepto que filtran a la pendiente si los contenedores mezclan regímenes. **Mitigación:** filtros de puntos de cambio; libro de pendiente–intercepto; excluir ventanas REGIME_MIX.

**9.2 Limitaciones de medición**

- **Fragilidad de proxy.** Algunos pares $`(L,T)`$ dependen de elecciones de modelado (por ejemplo, cómo se ajusta la "vida media"). **Riesgo:** pendientes inducidas por el estimador. **Mitigación:** fichas de receta, intercambios de estimador (ODR/Theil–Sen/SIMEX), envolventes de robustez.

- **Incompatibilidad de capas.** $`L`$ y $`T`$ desalineados (micro vs. macro) crean relaciones espurias. **Mitigación:** regla de compatibilidad; prueba de colapso diseñada para **fallar** tales contenedores.

- **Mala especificación de errores en variables.** Si subestimamos/sobreestimamos el ruido en $`L`$, las correcciones SIMEX/ODR pueden sesgar $`\widehat{\alpha}`$. **Mitigación:** mediciones replicadas donde sea posible; cotas de sensibilidad; reportar distribuciones nulas (aleatorización).

**9.3 Límites de identificación y causalidad**

- **Naturaleza asociativa.** El ICE es estructural-descriptivo; H1–H2 dan contenido *predictivo*, no prueba causal. **Riesgo:** extralimitación de política a partir de correlación. **Mitigación:** reservar las afirmaciones causales para entornos con instrumentos, experimentos naturales o intervenciones aleatorizadas de cadencia.

- **Confusión por relojes de política.** Los cambios uniformes en la sincronización (por ejemplo, moratorias de liquidación obligatorias) pueden alterar interceptos y a veces pendientes si se adoptan heterogéneamente. **Mitigación:** contener por régimen; probar invariancia de pendiente pre/post; documentar en el libro de registro.

- **Elección inversa de sincronización.** Los agentes pueden *aumentar la estratificación* en anticipación de choques, haciendo que el ICE parezca presciente. **Mitigación:** prerregistro, evaluación fuera de muestra, diferencias en diferencias donde las políticas de cadencia varían exógenamente.

**9.4 Limitaciones de forma del modelo**

- **Mala especificación de ley de potencia.** Algunos dominios pueden seguir $`\log T = g(\log L)`$ con curvatura. **Riesgo:** α sesgado, colapsos falsos. **Mitigación:** alternativas de spline; declarar contenedores *no potenciales*; tratar como resultados negativos (frontera de alcance).

- **Supuesto de α único dentro de contenedores.** Sub-regímenes heterogéneos pueden requerir modelos de mezcla. **Riesgo:** un α promedio oculta estructuras opuestas. **Mitigación:** estratificar; ajustes de mezcla finita; elevar umbrales de heterogeneidad ($`{\widehat{\tau}}^{2}`$) para fusión.

- **No estacionariedad temporal dentro de ventanas.** Las rupturas estructurales rápidas violan la premisa de "ambiente fijo". **Mitigación:** acortar ventanas; aumentar sensibilidad de puntos de cambio; descartar ventanas con REGIME_MIX.

**9.5 Limitaciones operacionales y de gobernanza**

- **Extralimitación de número único.** Tratar el ICE como un control maestro arriesga **sobre-estratificación burocrática** (α↑ con pérdida de rendimiento). **Mitigación:** paneles multi-métrica; cláusulas de caducidad; topes de rendimiento; nunca activar política solo con ICE.

- **Gaming y ley de Goodhart.** Si las métricas de sincronización se convierten en objetivos, los agentes pueden escalonar cosméticamente los reportes. **Mitigación:** auditorías aleatorias; validación de colapso ex-post; verificaciones de consistencia entre familias.

- **Equidad y acceso.** Las divulgaciones de sincronización pueden cargar a PyMEs/ME. **Mitigación:** publicar plantillas, subsidiar herramientas, permitir cuantiles agregados, monitorear impactos de equidad.

**9.6 Validez externa y transferibilidad**

- **Comparabilidad entre países.** El ICE es contextual a las definiciones de contenedor y convenciones de datos. **Riesgo:** pseudo-rankings entre regímenes incomparables. **Mitigación:** armonizar contenedores antes de comparar; reportar puntajes de comparabilidad.

- **Heterogeneidad sectorial.** Los sectores de alto α (servicios públicos, farmacéuticas) y de bajo α (retail rápido) difieren estructuralmente; las recetas de política uniformes son inapropiadas. **Mitigación:** manuales específicos por sector; evitar objetivos universales.

- **Tipología de choques.** Algunos choques son **exógenos al reloj** (moratorias de política) o puramente choques de nivel; el ICE puede añadir poco. **Mitigación:** declarar clases de choque de "bajo rendimiento" a priori; usar herramientas clásicas en su lugar.

**9.7 Qué cambiaría nuestra opinión**

Consideraríamos RTM–ICE como **no útil** para un dominio si, a través de múltiples conjuntos de datos y regímenes:

1.  **Sin separación de pendiente** es detectable bajo ajustes EIV robustos;

2.  **El colapso falla rutinariamente** en contenedores bien especificados;

3.  **Cascada inversa** (α decreciente con la agregación) aparece persistentemente en estado estacionario;

4.  **H2 no añade valor predictivo** fuera de muestra más allá de líneas base fuertes;

5.  Los resultados **se invierten** bajo intercambios razonables de proxy/estimador.

Publicar tales resultados es parte del programa: definen la **frontera de alcance** del método.

**9.8 Hoja de ruta para reducir limitaciones**

- **Datos:** expandir divulgaciones de sincronización; estandarizar fichas de receta; construir réplicas sintéticas abiertas.

- **Medición:** invertir en mediciones repetidas para calibrar EIV; ampliar familias de proxies.

- **Identificación:** buscar experimentos naturales (mandatos de cadencia, reformas de interruptores de circuito); pilotar desfases de cadencia aleatorizados.

- **Forma del modelo:** añadir diagnósticos de mezcla/spline; automatizar banderas de no-potencial.

- **Gobernanza:** codificar puertas de CC, bandas de incertidumbre, auditorías de equidad; mantener archivos públicos de vintage.

**10. Ética y gobernanza**

Este capítulo establece barandillas para **usar, publicar y actuar sobre el ICE**. Debido a que el ICE puede influir en la asignación de capital, la regulación y las narrativas públicas, el objetivo de gobernanza es doble: (i) **prevenir el uso indebido** (tiranía de número único, gaming, acceso desigual), y (ii) **institucionalizar las buenas prácticas** (transparencia, equidad, reproducibilidad). Estructuramos la orientación a través de (A) transparencia y rendición de cuentas, (B) equidad y acceso, (C) privacidad, (D) protocolos de decisión, (E) auditorías y equipo rojo, y (F) administración y ciencia abierta.

**10.1 Transparencia y rendición de cuentas**

**10.1.1 Fichas de método públicas.**\
Toda serie ICE publicada debe ir acompañada de una "ficha de método" que declare: definiciones de contenedor, familias de proxies, elecciones de estimador (ODR/Theil–Sen/SIMEX), tasas de aprobación de colapso, heterogeneidad $`{\widehat{\tau}}^{2}`$, banderas de CC y política de vintage. Proporcionar un resumen legible por humanos y una especificación legible por máquina (YAML/JSON).

**10.1.2 Libro de pendiente–intercepto.**\
Mantener un libro de registro de **cambios de nivel/reloj** conocidos (unidades, rebasaciones de política) junto a $`\alpha`$. Esto clarifica por qué se movieron los interceptos y defiende la robustez de la pendiente.

**10.1.3 Resultados negativos.**\
Registrar y publicar los contenedores donde el **colapso falla** o **α es inestable**. La no publicación de fallos sesga los incentivos e invita a la ley de Goodhart.

**10.2 Equidad y acceso**

**10.2.1 Divulgación igualitaria de sincronización.**\
Las métricas de sincronización (distribuciones de tiempo de entrega, vidas medias de resiliencia) deben ser **públicas o licenciadas simétricamente**, no bajo muro de pago para actores selectos. Si los reguladores dependen del ICE, deben asegurar acceso igualitario a los insumos.

**10.2.2 Carga para PyMEs/ME.**\
Las pequeñas empresas y los mercados emergentes no pueden soportar cargas pesadas de reporte. Proporcionar **plantillas, herramientas de código abierto y subvenciones** para que las divulgaciones de sincronización no afiancen la incumbencia.

**10.2.3 Evaluaciones de impacto.**\
Antes de políticas guiadas por ICE (por ejemplo, amortiguadores vinculados al ICE), ejecutar una **evaluación de impacto de equidad**: ¿quién asume los costos/beneficios por tamaño, sector y región? Publicar mitigaciones (cronogramas de introducción gradual, exenciones).

**10.3 Privacidad y confidencialidad**

**10.3.1 Agregación por diseño.**\
Publicar **cuantiles** de variables de sincronización e ICE a nivel de contenedor; evitar micro-identificadores. Cuando los datos micro sean necesarios para investigación, usar **enclaves seguros** y acceso auditado.

**10.3.2 Auditorías de reidentificación.**\
Ejecutar **pruebas periódicas de vinculación** contra registros externos para evaluar el riesgo de reidentificación; rotar o englobar el contenedor si el riesgo aumenta.

**10.3.3 Réplicas sintéticas.**\
Liberar **conjuntos de datos sintéticos** que preserven las propiedades distribucionales y el comportamiento de colapso, permitiendo verificación independiente sin exponer datos micro crudos.

**10.4 Protocolos de decisión (cómo actuar sobre el ICE)**

**10.4.1 Sin disparadores de número único.**\
El ICE **no** debería ser una regla de decisión solitaria. Combinar con métricas de volatilidad/liquidez/apalancamiento e inteligencia cualitativa. Documentar cuándo el ICE informó pero no decidió.

**10.4.2 Uso condicionado al CC.**\
Las acciones vinculadas al ICE requieren estado **CC limpio**: ≥2 familias de proxies, colapso aprobado, heterogeneidad por debajo del umbral, sin REGIME_MIX. Si el CC falla, el ICE puede informar el *monitoreo*, no la *acción*.

**10.4.3 Cláusulas de caducidad y topes de rendimiento.**\
Las políticas que "elevan $`\alpha`$ " (más escalonamiento) deben incluir **caducidades** y **topes de rendimiento** para evitar sobre-estratificación burocrática.

**10.4.4 Escaleras de escalamiento.**\
Vincular **respuestas graduadas** a la *magnitud y persistencia* de los movimientos del ICE (por ejemplo, asesoría → amortiguadores focalizados → medidas de todo el sistema), con desescalamiento automático cuando el ICE se normaliza.

**10.5 Auditorías, equipo rojo y riesgo de modelo**

**10.5.1 Recómputo independiente.**\
Al menos anualmente, un tercero recalcula $`\widehat{\alpha}`$ por contenedor, estadísticas de colapso e ICE(t) a partir de los artefactos publicados. Las diferencias más allá de la tolerancia activan un postmortem público.

**10.5.2 Escenarios de equipo rojo.**\
Comisionar revisiones adversariales que sondeen: fuga de proxy (resultados alimentando insumos), fragilidad de estimador, "hackeos de reloj" y exclusividad de proveedores de datos. Publicar hallazgos y correcciones.

**10.5.3 Pruebas de estrés de supuestos.**\
Ejecutar alternativas **no potenciales** (spline $`g(\log L)`$), modelos de mezcla y simulaciones de mezcla de régimen. Donde la ley de potencia falla persistentemente, marcar el ICE como **no aplicable**.

**10.5.4 Sobrepesos de gobernanza.**\
Si una sola familia de proxies domina repetidamente los pesos de fusión, requerir un **plan de diversificación** (añadir proxies complementarios o limitar pesos) para reducir el riesgo de monocultivo de modelo.

**10.6 Ética de la comunicación**

**10.6.1 Evitar lenguaje determinista.**\
Usar "asociado con", "predictivo de", no afirmaciones causales, a menos que estén respaldadas por diseños explícitos (instrumentos, experimentos naturales, ECAs).

**10.6.2 Contextualizar la incertidumbre.**\
Siempre mostrar **bandas y banderas**. Proporcionar explicaciones en lenguaje claro de lo que significan los modos de falla ("no pudimos validar el escalamiento este trimestre").

**10.6.3 Responsabilidad histórica.**\
Cuando el ICE informe políticas que afecten el sustento de las personas, publicar **informes post-acción**: qué señales vimos, elecciones realizadas y resultados (incluyendo errores).

**10.7 Administración institucional y ciencia abierta**

**10.7.1 Registros públicos.**\
Albergar un **registro de vintages de ICE**, especificaciones de contenedores, registros de CC y prerregistros de hipótesis. Marcar temporalmente todo.

**10.7.2 Grupos de trabajo.**\
Crear **grupos de trabajo de ICE** interinstitucionales (oficinas de estadística, bancos centrales, bolsas, puertos, academia) para armonizar contenedores y compartir resultados negativos.

**10.7.3 Educación.**\
Publicar cartillas para profesionales y lectores cívicos que expliquen pendientes vs. niveles, pruebas de colapso y por qué los **hallazgos negativos** son éxitos para la ciencia.

**10.8 Líneas rojas éticas**

- **Sin ampliación de vigilancia.** Las divulgaciones de sincronización no deben transformarse en monitoreo conductual a nivel individual.

- **Sin uso punitivo sin debido proceso.** Las banderas del ICE no son fundamento para sanciones en ausencia de marcos estatutarios y derechos de apelación.

- **Sin licencias excluyentes.** Si las entidades públicas actúan sobre el ICE, los insumos y métodos centrales deben ser accesibles para los afectados.

**Resumen.** El ICE se vuelve éticamente usable cuando las instituciones **comparten métodos e incertidumbre**, **protegen contra la inequidad y el gaming**, **evitan la elaboración de reglas de número único** e **invitan a la recómputo independiente**. La gobernanza debe hacer *fácil* hacer lo correcto (transparente, condicionado al CC, auditado) y *difícil* hacer lo incorrecto (opaco, exclusivo, sobreconfiado).

**Capítulo 11: Validación empírica de bifurcación de fase en mercados de alta frecuencia**

Este capítulo somete a prueba de estrés al Monitor en Tiempo Real de RTM contra la varianza extrema de la microestructura de Bitcoin. Al abandonar los cierres diarios estáticos e inyectar el perfil completo de ruido continuo de datos OHLCV minuto a minuto, rastreamos el momento exacto de la fractura estructural. El análisis continuo aísla el umbral de Bifurcación de Fase ($`\alpha < \ 0.5`$), distinguiendo las fallas mecánicas de liquidez (por ejemplo, marzo 2020) de los eventos de estrés político de alta viscosidad ($`\alpha > \ 0.6`$, por ejemplo, mayo 2021). Durante el evento de octubre de 2025, la métrica corregida por ruido mostró una lectura α elevada aproximadamente 15 horas antes del movimiento del precio. Se trata de una **única observación dentro de muestra**, y posteriormente el evento se atribuyó a un fallo técnico de la plataforma Binance en lugar de a un colapso estructural fundamental. Se presenta únicamente como estudio de caso y no debe interpretarse como una predicción prospectiva validada; sería necesaria una replicación en eventos reservados fuera de muestra para realizar cualquier afirmación anticipatoria.

**Capítulo 12: Análisis empírico: el colapso de** $`\mathbf{\alpha}`$ **como señal predictiva**

Este capítulo se aparta de los supuestos gaussianos tradicionales de las recuperaciones de mercado y las predicciones de caídas. Los modelos OLS ingenuos iniciales que predecían recuperaciones de caídas sufrían de un sesgo de atenuación masivo debido a las fronteras ambiguas de "recuperación del mercado". Al aplicar la Regresión de Distancia Ortogonal (ODR) para absorber un margen de ruido de medición del 20%, revelamos que el escalamiento del tiempo de recuperación es sustancialmente más castigador (pendiente = $`3.59\  \pm 0.70`$) de lo que se había modelado previamente. (Este resultado deriva del registro histórico de magnitud/duración de los colapsos, no de la tabla α retirada; fue reproducido de forma independiente con una pendiente ≈ 3.9. Se basa en 16 eventos con definiciones ambiguas de las fechas de recuperación y debe interpretarse como sugestivo.)

Una versión anterior informó que un análisis Monte Carlo de 13 grandes colapsos mostraba que la descorrelación estructural α precedía los mínimos de precio en ~9.75 días (d = −1.45). Posteriormente se descubrió que la tabla subyacente de 13 eventos contenía valores α generados sintéticamente (Capítulo 12.6), por lo que estas cifras se **retiran**. Un análisis independiente sobre rendimientos diarios reales del S&P 500 (Capítulo 12.7) no encuentra una separación α significativa entre crisis y calma (d ≈ −0.2, p ≈ 0.22) ni una disminución de α previa al colapso. El resultado de escalamiento del tiempo de recuperación (véase más arriba) y la métrica de coherencia multiescala (Capítulo 12.7) se calculan a partir de datos independientes y se conservan.

**12. Conclusión**

### 12.1 Lo que este artículo se propuso hacer — y lo que encontró

Este artículo propuso que el tiempo económico no es una variable de fondo absoluta, sino una dimensión dinámica que escala con la organización estructural multiescala del sistema. Mediante la derivación del Índice de Coherencia Económica (ECI) y su marco de cascada subyacente (Capítulos 1–10), construimos un aparato teórico y operativo completo para esta idea: familias de proxies, pruebas de colapso, metaanálisis, lógica de alertas y protocolos de gobernanza. La arquitectura teórica se mantiene como un marco propuesto.

Los resultados empíricos son más modestos de lo que afirmaban las primeras versiones de este trabajo, y la honestidad intelectual exige exponer claramente el registro revisado.

Un análisis de 13 eventos entre distintos mercados informó originalmente una gran separación dentro de muestra entre exponentes DFA de estados saludables y estados de colapso ($`d = -1.45`$, ~10 días de anticipación, correlación con severidad $`r = 0.97`$). Ese análisis se basaba en valores $`\alpha`$ generados sintéticamente (Capítulo 12.6) y queda **retirado**. La validación independiente sobre rendimientos diarios reales del S&P 500 (5,104 días que abarcan las crisis de las punto com, la GFC y la COVID) encuentra que la separación real DFA-$`\alpha`$ entre crisis y calma es pequeña y no significativa ($`d \approx -0.2`$, $`p = 0.22`$). La hipótesis de que $`\alpha`$ disminuye antes de las crisis (H2) queda contradicha: $`\alpha`$ *aumentó* antes de ambos inicios comprobables (GFC y COVID), y no muestra una depresión previa al colapso a resolución mensual a lo largo de 155 años de datos.

Lo que se mantiene es más limitado, pero real. La **métrica de coherencia multiescala** —la varianza entre escalas de $`\alpha`$— es menor durante ventanas de crisis que durante ventanas de calma tanto en datos intradía de BTC ($`\sigma = 0.03`$ vs $`0.31`$) como en el S&P 500 diario ($`p = 0.039`$, AUC = 0.60). La dirección se replica; la magnitud depende del activo y de la escala temporal (~10× en cripto, ~1.3× en acciones). La **cola cúbica inversa** ($`\alpha \approx 3`$) se reproduce de forma independiente a partir de rendimientos diarios brutos (estimador de Hill de cola extrema $`\alpha = 2.9\text{–}3.3`$), confirmando un resultado conocido (Gabaix et al. 2003) en lugar de aportar uno nuevo. La observación de diferenciación entre crisis —que los vacíos de liquidez (COVID 2020, $`\alpha < 0.5`$) son estructuralmente distintos de los choques políticos (China Ban 2021, $`\alpha > 0.6`$)— sigue siendo una caracterización descriptiva legítima derivada de datos brutos de microestructura de BTC.

La precisión fuera de muestra del Red Team, del 25% (1 de 4 eventos posteriores a 2022), es la conclusión correcta en materia de predicción.

### 12.2 Por qué la economía es diferente: la condición de frontera

En el corpus RTM más amplio, el marco demuestra una tracción empírica genuina en física (decoherencia de procesadores cuánticos), geociencia (escalamiento Gutenberg-Richter, decaimiento de réplicas de Omori, discriminación de tornados), biología (alometría masa-longevidad, bonificación de longevidad asociada al vuelo) y dinámica de fluidos (dispersión de Richardson). En cada uno de estos dominios, el exponente de escalamiento $`\alpha`$ se comporta como exige el Doc 002: un invariante estructural, estable dentro de las ventanas de medición, determinado por la topología de la red de transporte del sistema y no por el observador.

La economía es el dominio donde esto falla — y el fallo no es accidental. Revela una **condición de frontera** del marco RTM que, formulada con precisión, fortalece la teoría en lugar de debilitarla al delimitar dónde se aplica y dónde no.

Cuatro propiedades distinguen conjuntamente a los mercados financieros de cualquier otro sistema del corpus:

**Reflexividad.** Un terremoto no lee su propio valor $`b`$ de Gutenberg-Richter y ajusta su magnitud en respuesta. Un murciélago no observa su cociente de longevidad y modifica su tasa metabólica. Pero un participante del mercado que descubra que una caída de $`\alpha`$ predice colapsos *actuará sobre esa señal* — y al hacerlo, la eliminará o la transformará. Los mercados son el único sistema del corpus en el que los agentes dentro del sistema observan, interpretan y reconfiguran el sistema en tiempo real. Toda regularidad de escalamiento que se vuelve conocida y accionable tiende a ser arbitrada hasta desaparecer. En física y biología no existe un mecanismo equivalente: las leyes de escala no están sujetas a manipulación estratégica por parte de las entidades que gobiernan.

**Topología endógena.** RTM funciona cuando existe una red de transporte cuya estructura es estable durante la escala temporal de medición. La red vascular de un mamífero, la retícula de qubits de un chip superconductor, un sistema de fallas en límites de placas — estas topologías cambian en escalas temporales (evolución, fabricación, geología) mucho mayores que los fenómenos que mide RTM. En los mercados financieros, la red —quién negocia con quién, qué algoritmos están activos, qué correlaciones entre activos se mantienen— se reconfigura endógenamente en milisegundos. La topología es en sí misma una variable rápida, no un parámetro lento. Cuando la red se reconecta más rápido de lo que el exponente puede equilibrarse, $`\alpha`$ no puede funcionar como un invariante estructural. Se convierte, en el mejor de los casos, en una instantánea de un estado transitorio.

**No conservación de la cantidad transportada.** La derivación por semigrupo (Doc 001, Apéndice J) modela el transporte de una cantidad conservada a través de una estructura fija. En física, lo que fluye se conserva: energía, masa, carga, momento. En los mercados, las cantidades relevantes —crédito, información, sentimiento, liquidez— no se conservan en el sentido físico. El crédito se crea ex nihilo mediante préstamos. La información se amplifica, distorsiona y destruye. El pánico se autorrefuerza sin aporte externo. Cuando la sustancia que fluye por la red puede crearse y destruirse endógenamente, las leyes de transporte que suponen conservación se cumplen con menor rigidez. Aun así, pueden aplicarse de forma local y transitoria —durante un colapso, cuando el mercado se comporta brevemente como una cascada física—, pero no como invariantes estructurales estables.

**No estacionariedad estructural.** En los sistemas naturales, las leyes de escalamiento son aproximadamente estacionarias durante las ventanas de observación. El valor $`b`$ de Gutenberg-Richter se mantiene durante décadas. La ley de Kleiber se mantiene entre órdenes taxonómicos. El $`\alpha`$ de tornados se replica en 7 de 9 brotes independientes dentro de un mismo año. Pero el comportamiento de escalamiento de un mercado cambia con la regulación, la política monetaria, la tecnología y el aprendizaje colectivo de sus participantes. El patrón encontrado dentro de muestra en 2018–2022 puede no existir en 2023–2025 — no porque la medición fuera incorrecta, sino porque el sistema realmente cambió. Por eso la precisión fuera de muestra del 25% no significa simplemente "el efecto es más débil"; refleja la posibilidad de que la relación de escalamiento específica descubierta dentro de muestra *dejara de existir* durante el periodo de prueba.

En conjunto, estas cuatro propiedades explican por qué el exponente RTM funciona como un invariante estructural en física, biología y geociencia, pero se degrada hasta convertirse en un descriptor transitorio en economía. La maquinaria matemática del marco no es incorrecta cuando se aplica a los mercados; se está aplicando a un sistema que viola sus supuestos fundamentales — topología estable, transporte conservado, agentes no reflexivos y estacionariedad aproximada.

### 12.3 El colapso como ventana de comportamiento físico

Existe un corolario revelador. El único régimen económico en el que el escalamiento tipo RTM *sí* se replica parcialmente —la firma de coherencia multiescala durante los colapsos— es precisamente el régimen en el que las cuatro violaciones de frontera se relajan temporalmente:

- Durante un pánico, **la reflexividad se desactiva parcialmente**: el cálculo estratégico da paso a la liquidación mecánica. Los agentes dejan de optimizar y comienzan a vender indiscriminadamente. El mercado actúa brevemente más como una cascada física que como un juego estratégico.

- Durante una crisis de liquidez, **la topología se rigidiza**: las contrapartes se retiran, cesa la creación de mercado y la red de conexiones activas se congela en una estructura más simple y rígida. La topología de variable rápida se convierte temporalmente en un parámetro lento.

- Durante un colapso, **los flujos se aproximan más a la conservación**: la dinámica dominante es la redistribución de un fondo fijo (y decreciente) de liquidez, no la creación de nuevo crédito o información.

- Durante un colapso, la **no estacionariedad estructural se reduce**: todos los participantes enfrentan la misma restricción (sobrevivir), y la diversidad de estrategias colapsa hacia un "vender todo" casi uniforme. El sistema se vuelve brevemente estacionario en su desesperación.

Por eso la métrica de coherencia multiescala —que mide si $`\alpha`$ se vuelve invariante respecto de la escala— muestra su señal más fuerte durante los colapsos: el mercado se comporta temporalmente como un sistema físico, y los supuestos de RTM se satisfacen de forma transitoria. En cuanto comienza la recuperación y regresa la diversidad estratégica, la señal se debilita. La separación de 1.3× en acciones (frente a 10× en BTC intradía) probablemente refleja el grado en que las acciones conservan estructura institucional incluso durante el pánico, mientras que los mercados cripto —menos regulados, más impulsados por inversores minoristas y con menor profundidad— se acercan más al límite de "cascada física pura".

### 12.4 Diagnóstico, no predicción

Estas consideraciones reformulan a qué puede y a qué no puede aspirar el Índice de Coherencia Económica:

**Lo que no puede hacer.** Predecir de forma fiable colapsos fuera de muestra con un tiempo de anticipación operativo. La precisión del 25% no es un punto de partida que deba mejorarse con más datos; es una consecuencia de las propiedades estructurales descritas anteriormente. Un mejor predictor de colapsos basado en RTM requeriría un mercado cuya topología fuera estable, cuyos participantes no aprendieran y cuyas cantidades transportadas fueran conservadas — un mercado que, por definición, ya no funcionaría como mercado. Se retiran las afirmaciones anteriores del documento sobre una "autoridad matemática para predecir fallos estructurales".

**Lo que sí puede hacer — y esto no es poca cosa.** Caracterizar el *tipo* de estrés bajo el que se encuentra un mercado casi en tiempo real. La distinción entre un vacío de liquidez ($`\alpha < 0.5`$, antipersistente, sin memoria) y un choque político viscoso ($`\alpha > 0.6`$, persistente, estructuralmente intacto) es un diagnóstico estructural genuino que el precio y la volatilidad por sí solos no capturan. Si la topología está *actualmente* coherente entre escalas ($`\sigma`$ cercano a cero) o fragmentada ($`\sigma`$ elevado) es una pregunta que RTM puede responder — de forma descriptiva, no predictiva. Esto posiciona al ECI como un **clasificador de regímenes**, no como un sistema de alerta temprana: indica *en qué tipo de régimen se encuentra el mercado ahora*, no *qué hará a continuación*.

Para responsables de políticas públicas y gestores de riesgo, esto sigue siendo útil. Saber que un mercado se encuentra en un estado de vacío de liquidez (que requiere una provisión de emergencia de creación de mercado) frente a un estado de estrés viscoso (que requiere una resolución paciente de la incertidumbre de solvencia) es un diagnóstico que orienta la *respuesta*, aunque no pueda orientar la *anticipación*. La analogía es médica: un termómetro indica que el paciente tiene fiebre y qué tipo (alta, baja, cíclica), pero no predice cuándo desaparecerá. Sigue siendo un instrumento útil.

### 12.5 La posición honesta

RTM-Economics ocupa una posición única e instructiva dentro del corpus más amplio. Es el dominio donde las predicciones del marco son más débiles — y el dominio que, precisamente por su resistencia, delimita con mayor precisión las condiciones de frontera del marco. Las cuatro propiedades que hacen que los mercados sean resistentes al análisis de escalamiento (reflexividad, topología endógena, no conservación, no estacionariedad) son exactamente las propiedades que *definen* lo que es un mercado, en contraste con un sistema físico o biológico. RTM no falla en economía porque sus matemáticas sean incorrectas; falla porque los mercados son el único sistema del corpus cuyos participantes pueden observar el exponente y modificarlo.

Esta condición de frontera — **RTM funciona donde la topología es un parámetro lento y el transporte obedece a conservación; se degrada donde la topología es endógena y los agentes son reflexivos** — es, creemos, una contribución genuina a la comprensión de los límites del análisis de escalamiento en sistemas complejos. Explica por qué el mismo marco matemático que separa claramente tormentas tornádicas de no tornádicas ($`d = 0.96`$), cuantifica el costo de coherencia del escalamiento cuántico monolítico ($`\alpha = -0.26`$) y recupera la bonificación de longevidad asociada al vuelo en 2,400 endotermos ($`d = 0.85`$) no puede producir un predictor fiable de colapsos en los mercados financieros — y por qué esperar que pudiera hacerlo siempre fue formular la pregunta equivocada.

La pregunta correcta nunca fue "¿puede RTM predecir colapsos?" Fue: "¿qué significa que los supuestos de escalamiento de RTM se violen en los mercados, pero no en la física?" La respuesta —que los mercados son reflexivos, topológicamente inestables y no conservativos de formas que ningún sistema natural lo es— es en sí misma un hallazgo, y pertenece a la conclusión en lugar de quedar oculta detrás de una afirmación inflada de poder predictivo.

**12.6 Corrección de procedencia de datos (julio de 2026)**
Una auditoría forense estableció que `crash_alpha_analysis.csv` —la fuente de todas las estadísticas α de 13 eventos de este capítulo— fue generado sintéticamente en lugar de medirse a partir de datos de mercado. Reproducir el generador aleatorio de `numpy` con semilla 42 y consumir cinco extracciones uniformes por evento reproduce las columnas del archivo con precisión de máquina:

> | Columna | Generador | Coincidencias exactas |
> |---|---|---|
> | `Baseline_Alpha` | 0.5 + 0.1·U(0,1) | 13/13 |
> | `Pre_Alpha` | Baseline + (−0.03 + 0.06·U) | 13/13 |
> | `Lead_Time_Hours` | int(72 + 432·U) → U(3,21) days | 13/13 |
> | `Immediate_Alpha` | función de Baseline y `Drop_Pct` | severidad codificada por construcción |

Consecuencias: (i) la separación d = −1.45 es una propiedad del diseño de simulación; (ii) la anticipación de ~9.75 días es la media de extracciones uniformes de 3–21 días en el subconjunto significativo; (iii) la correlación r = 0.97 con la severidad estaba incorporada en el generador (`Immediate_Alpha` depende de `Drop_Pct`). Las tres se retiran como hallazgos empíricos. La canalización ODR/Monte Carlo posterior (`analyze_financial_robust.py`) no es en sí misma responsable: procesó fielmente una entrada sintética que había ingresado a la canalización en una etapa anterior y posteriormente fue tratada como real. El Red Team de abril de 2026 reprodujo las *estadísticas* de la tabla, pero no examinó la *procedencia*; la verificación de procedencia ahora forma parte del estándar de auditoría. Reproducir con `forensic_audit_crash_alpha.py`.

**12.7 Replicación con datos reales (qué se mantiene)**
Para sustituir la tabla retirada, las afirmaciones centrales se volvieron a poner a prueba con datos públicos y verificables: 5,104 días de OHLCV del S&P 500 (2000–2020, que abarcan las crisis de las punto com, la GFC y la COVID), VIX diario (1990–2026) y S&P mensual de Shiller (1871–2026). La implementación DFA se calibró sobre procesos conocidos (ruido blanco α = 0.51; movimiento browniano α = 1.50).

> | Prueba | Afirmación previa (sintética) | Resultado con datos reales | Estado |
> |---|---|---|---|
> | DFA-α en crisis vs calma | d = −1.45 | d = −0.22 (drawdown, p = 0.22); d = +0.03 (VIX, p = 0.93) | No respaldado |
> | H2: α disminuye antes de las crisis | 6–18 mo lead | α **aumentó** antes de la GFC (+0.05) y la COVID (+0.07); sin señal a escala mensual 1871–2026 | Contradicho |
> | Coherencia multiescala σ | 10× (BTC, 3 months) | La dirección se replica en acciones: crisis σ < calma σ, p = 0.039, AUC 0.60, ratio 1.31× | Respaldado parcialmente |
> | Cola cúbica inversa α ≈ 3 | Cita de la literatura | Estimador de Hill sobre rendimientos brutos: α de cola extrema = 2.9–3.3 | Reproducido independientemente |

La métrica de coherencia multiescala es la única señal nativa de RTM que sobrevive a la transferencia fuera de muestra hacia una clase de activo independiente; es real pero débil en acciones, y su magnitud depende de la escala temporal. La hipótesis H2 (anticipación mediante disminución de α) queda ahora contradicha por tres líneas de evidencia independientes (estudio de eventos con datos reales, análisis mensual a escala de un siglo y la precisión fuera de muestra del 25% del Red Team) y debe retirarse o reformularse. Canalización completa: `rtm_econ_new_validations.py`.


**Apéndices**

Estos apéndices dan a los implementadores todo lo necesario para reproducir, auditar y extender el **Índice de Coherencia Económica (ICE)**: las matemáticas detrás de la ley de escalamiento y los estimadores, una especificación completa (esquemas de datos, puertas de CC, valores predeterminados), métricas de evaluación, y listas de verificación de robustez/ablación.

**Apéndice A — Notas matemáticas**

**A.1 De la simetría de escala a una ley de potencia**

Suponga un tiempo característico $`T(L)`$ que depende de un proxy de tamaño/escala $`L > 0`$ y satisface **simetría de escala**:

- Para cualquier $`b > 0`$, reescalar $`L \mapsto bL`$ reescala el tiempo por un factor $`f(b)`$ : $`T(bL) = f(b)\text{ }T(L)`$.

- La composición de reescalamientos es multiplicativa: $`f(b_{1}b_{2}) = f(b_{1})f(b_{2})`$.

Entonces $`f`$ resuelve la ecuación exponencial de Cauchy en $`\mathbb{R}_{> 0}`$, produciendo $`f(b) = b^{\alpha}`$ para algún $`\alpha`$ real. Fijando cualquier $`L_{0} > 0`$ da

``` math
T(L) = T(L_{0})\text{ }(\frac{L}{L_{0}})^{\alpha} = \kappa\text{ }L^{\alpha},\ \ \kappa = T(L_{0})L_{0}^{- \alpha}.
```

Tomando logaritmos: $`\log\ T = \alpha\ \log\ L + \log\kappa`$.

**Implicación.** Cualquier cambio uniforme de "reloj" o nivel multiplica $`\kappa`$ (intercepto), no $`\alpha`$ (pendiente).

**A.2 Estimadores de pendiente con errores en variables (EIV)**

Sea $`x = \log\ L`$, $`y = \log\ T`$, con observaciones ruidosas $`x^{obs} = x + \xi`$, $`y^{obs} = y + \zeta`$, $`\mathbb{E}\lbrack\xi\rbrack = \mathbb{E}\lbrack\zeta\rbrack = 0`$.

**Regresión de Distancia Ortogonal (ODR / TLS).**\
Estimar $`(\widehat{\alpha},\widehat{c})`$ minimizando la suma de **residuos cuadrados ortogonales** a la línea $`y = \alpha x + c`$ :

``` math
\underset{\alpha,c}{\min}\sum_{u}^{}{\frac{(y_{u}^{obs} - \alpha x_{u}^{obs} - c)^{2}}{1 + \alpha^{2}}.}
```

Existen formas cerradas (vía SVD del diseño centrado); la mayoría de las bibliotecas lo implementan iterativamente.

**SIMEX (simulación-extrapolación).**\
Si $`\sigma_{\xi}^{2}`$ (o una cota) es conocida/estimable:

1.  Simular ruido añadido: $`x^{(\lambda)} = x^{obs} + \sqrt{\lambda}\text{ }\widetilde{\xi}`$, $`\lambda \in \Lambda \subset \mathbb{R}_{\geq 0}`$.

2.  Ajustar pendientes $`\widehat{\alpha}(\lambda)`$ para cada $`\lambda`$.

3.  Extrapolar $`\lambda \rightarrow - 1`$ (cero error de medición) con un polinomio de bajo orden para obtener $`{\widehat{\alpha}}_{\text{SIMEX}}`$.

**Theil–Sen (robusto).**\
Mediana de pendientes por pares $`\{(y_{j} - y_{i})/(x_{j} - x_{i})\}`$ sobre todos los $`i < j`$. Resistente a valores atípicos; usar como verificación de sensibilidad.

**A.3 Estadístico de prueba de colapso**

Dado $`\widehat{\alpha}`$ para un contenedor, definir resultados residualizados $`{\widetilde{y}}_{u} = y_{u}^{obs} - \widehat{\alpha}x_{u}^{obs}`$. En un contenedor de ley de potencia válido, $`\widetilde{y}`$ debería ser **independiente** de $`x`$ (excepto ruido). Usar:

``` math
\Delta_{\text{collapse}}: = R^{2}(\widetilde{y} \sim x^{obs}).
```

**Regla de aprobación (predeterminada):** $`\Delta_{\text{collapse}} < 0.05`$ y un suavizado no paramétrico (por ejemplo, LOESS con ancho de banda prerregistrado) no muestra tendencia visible.

**A.4 Fusión de efectos aleatorios (DerSimonian–Laird / REML)**

Con estimaciones específicas por familia $`{\widehat{\alpha}}_{f}`$ y varianzas $`{\widehat{\sigma}}_{f}^{2}`$,

``` math
{\widehat{\tau}}^{2} = \max\{\frac{Q - (F - 1)}{\sum w_{f} - \sum w_{f}^{2}/\sum w_{f}},\text{ }0\},w_{f} = \frac{1}{{\widehat{\sigma}}_{f}^{2}},Q = \sum w_{f}({\widehat{\alpha}}_{f} - {\overset{ˉ}{\alpha}}_{w})^{2},
```

$`{\overset{ˉ}{\alpha}}_{w} = \sum w_{f}{\widehat{\alpha}}_{f}/\sum w_{f}`$. La estimación fusionada:

``` math
{{\widehat{\alpha}}_{\text{econ}} = \frac{\sum_{f}^{}{w_{f}\text{ }{\widehat{\alpha}}_{f}}}{\sum_{f}^{}w_{f}},{\ \ \ \ \ \ \ w}_{f} = \frac{1}{{\widehat{\sigma}}_{f}^{\text{ }2} + {\widehat{\tau}}^{\text{ }2}}.
}
```

REML puede reemplazar a DL para $`{\widehat{\tau}}^{2}`$ cuando $`F`$ es pequeño.

**A.5 Pruebas de direccionalidad (ET/Granger) para H3**

**Causalidad de Granger.** $`X`$ "causa-Granger" a $`Y`$ si el $`X`$ rezagado mejora la predicción de $`Y`$ más allá del $`Y`$ rezagado. Usar VAR con retardo prerregistrado $`p`$; pruebas de Wald con control de multiplicidad.

**Entropía de transferencia (ET).** Asimetría teórico-informacional:

``` math
TE_{X \rightarrow Y} = \sum p(y_{t + 1},y_{t}^{(p)},x_{t}^{(p)})\text{ }\log\frac{p(y_{t + 1} \mid y_{t}^{(p)},x_{t}^{(p)})}{p(y_{t + 1} \mid y_{t}^{(p)})}.
```

Estimar vía kNN o surrogados basados en modelos; significancia con surrogados de aleatorización por bloques.

**Apéndice B — Especificación del ICE (datos, CC, valores predeterminados)**

**B.1 Esquema de datos (características a nivel de entidad → tabla de contenedor)**

**Tabla de entidad (por familia):**

- entity_id

- timestamp (UTC)

- L_value (proxy de escala, positivo)

- T_value (tiempo característico, positivo)

- L_unit, T_unit (cadenas)

- L_method, T_method (IDs de receta)

- fit_r2, fit_se_T (opcional)

- env_keys (país, sector, régimen de política, window_id)

- quality_flags (campo de bits)

**Tabla de contenedor (por familia × ambiente):**

- env_keys

- n_entities, coverage_share

- alpha_hat, alpha_ci_low, alpha_ci_high

- intercept_hat

- collapse_R2, collapse_pass (bool)

- clock_placebo_pass (bool)

- qa_flags (enum: LOW_COVERAGE, NO_COLLAPSE, REGIME_MIX, CLOCK_SHIFT)

**Tabla del ICE (fusionado por ambiente/tiempo):**

- window_id, env_keys

- alpha_econ_hat, band50_low/high, band95_low/high

- tau2 (heterogeneidad), Q_stat, families_used

- qa_flags (como arriba)

- vintage_asof

**B.2 Definición de ambiente y control de régimen**

- **Ventanas:** trimestrales (primario), paso mensual.

- **Verificaciones de régimen:** pruebas de puntos de cambio univariadas/multivariadas en series representativas de nivel y proxies de sincronización; dividir ventana en puntos de ruptura detectados; marcar REGIME_MIX si no es divisible.

**B.3 Valores predeterminados de estimación**

- **Estimador:** ODR/TLS (predeterminado), Theil–Sen (verificación robusta), SIMEX (cuando $`\sigma_{\xi}^{2}`$ es conocido).

- **Bootstrap:** agrupado por entidad (≥1,000 réplicas); reportar ICs percentiles.

- **Umbral de colapso:** $`\Delta_{\text{collapse}} < 0.05`$ y tendencia de suavizado visualmente plana.

- **Fusión:** efectos aleatorios (REML), requerir ≥2 familias con collapse_pass==True.

- **Tope de heterogeneidad:** rechazar fusión si $`{\widehat{\tau}}^{2}`$ excede el percentil prerregistrado (por ejemplo, percentil 90 del histórico de $`\tau^{2}`$); marcar FAMILY_DIVERGENCE.

**B.4 Puertas de CC y aceptación**

Un contenedor contribuye al ICE solo si:

1.  coverage_share ≥ mínimo específico del sector,

2.  collapse_pass == True,

3.  clock_placebo_pass == True,

4.  sin REGIME_MIX.

El ICE(t) fusionado en una ventana se **publica** solo si ≥2 contenedores (familias) cumplen estas puertas; de lo contrario, poblar qa_flags y retener el uso decisional.

**B.5 Definición de evento de decoherencia (predeterminados)**

- Horizontes $`h \in \{ 1,3,6\}`$ meses.

- Umbral $`\theta_{h}`$ : percentil 10 histórico de $`\Delta\alpha_{\text{econ}}`$ al horizonte $`h`$ **o** $`k \cdot SE_{t}`$ con $`k`$ prerregistrado (por ejemplo, 1.64).

- Confirmación: al menos dos familias muestran caídas consistentes en signo; CC limpio en $`t`$ y durante $`\lbrack t - h,t\rbrack`$.

**B.6 Política de publicación y vintage**

- Publicar archivos ICE **con marca de vintage** (asof), con código para reconstruir cualquier curva histórica.

- Mantener un **libro de pendiente–intercepto** registrando cambios de unidades, rebasaciones de política, cambios de proveedor.

**Apéndice C — Métricas de pronóstico e inferencia**

**C.1 Clasificación (H2)**

- **AUC / PR-AUC** con intervalos de DeLong y bootstrap para PR-AUC.

- **Puntaje de Brier**, **pérdida logarítmica**; pruebas de Diebold–Mariano para diferencias de puntaje de pronóstico.

- **Calibración**: diagramas de confiabilidad; Error de Calibración Esperado (ECE).

- **Confusión**: tasas de acierto/falsa alarma a umbrales relevantes para la política (prerregistrados).

**C.2 Supervivencia / duración (H1b, H2)**

- Coeficientes del **modelo AFT** con EE robustos; índice de concordancia.

- **Modelo de Cox** (sensibilidad): razones de riesgo para terciles del ICE; pruebas de Schoenfeld para el supuesto de riesgos proporcionales.

**C.3 Regresión transversal (H1a)**

- Efectos fijos; EE robustos agrupados.

- Análisis de **Shapley / dominancia** para poder explicativo incremental vs. volatilidad/liquidez/apalancamiento.

**C.4 Cascada multicapa (H3)**

- **Pruebas de diferencias pareadas** para $`\alpha_{\text{micro}} \leq \alpha_{\text{meso}} \leq \alpha_{\text{macro}}`$ a través de ventanas; control de FDR.

- Estadísticos de asimetría **ET/Granger** con valores $`p`$ basados en surrogados.

**C.5 Plantillas de presentación**

- **Figura A (obligatoria):** ICE(t) + bandas del 50/95% + banderas de CC.

- **Tabla A:** tasas de aprobación de colapso por familia y régimen.

- **Tabla B:** pesos de fusión, $`{\widehat{\tau}}^{2}`$, influencia (dejando una familia fuera).

- **Figura B:** gráficos alineados a eventos (caídas de ICE vs. inicios de estrés).

**Apéndice D — Robustez y ablaciones**

**D.1 Ablaciones de estimador y proxy**

- **Intercambio de estimador:** ODR ↔ Theil–Sen ↔ SIMEX; reportar $`\Delta\widehat{\alpha}`$, superposición de IC, estabilidad de decisión para H1–H3.

- **Intercambio de proxy:** dentro de cada familia, alternar $`L`$ (grado ↔ longitud de ruta, capitalización ↔ nivel de tamaño de transacción) y $`T`$ (variante de vida media, ajuste alternativo).

- **Intercambio de resultado:** etiquetas de estrés alternativas (por ejemplo, cronologías de crisis locales).

**D.2 Ventanas y cobertura**

- Ventanas: trimestrales vs. semestrales; paso: mensual vs. quincenal (donde sea factible).

- Submuestrear entidades; rastrear ampliación de IC y tasas de fallo de colapso.

**D.3 Placebos y nulos**

- **Placebos de reloj:** reescalar unidades de tiempo; confirmar invariancia de pendiente, cambio de intercepto.

- **Nulo de aleatorización:** permutar etiquetas de $`L`$ dentro de contenedores; almacenar distribución de pendiente nula y asegurar que $`\widehat{\alpha}`$ observado excede el nulo por márgenes prerregistrados.

- **Alternativa no potencial:** ajustes de spline $`g(\log L)`$; si la curvatura persiste entre contenedores, marcar dominio **no potencial** y excluir.

**D.4 Gestión de heterogeneidad**

- Limitar fusión cuando $`{\widehat{\tau}}^{2}`$ es excesivo; publicar $`{\widehat{\alpha}}_{f}`$ por familia en lugar de un ICE único.

- Requerir un plan de diversificación si una familia domina repetidamente los pesos.

**Apéndice E — Reproducibilidad y empaquetado**

- **Estructura del repositorio.**\
  data_raw/, data_processed/, features/, bins/, alpha_estimates/, collapse/, fusion/, eci/, qa/, figures/.

- **Pipelines.** Grafo acíclico dirigido con semillas deterministas; pruebas de IC: (i) invariancia de unidades, (ii) reproducción exacta dentro de tolerancia, (iii) cotas de colapso.

- **Documentación.** Especificaciones YAML para recetas de proxy; CHANGELOG para versiones de estimadores; archivos de ambiente reproducible.

**Apéndice F — Glosario (mínimo)**

- $`L`$ : proxy de escala (tamaño/ruta/grado/nivel de capitalización).

- $`T`$ : tiempo característico (vida media, decaimiento, reposición, resiliencia).

- $`\alpha`$ : pendiente en $`\log T = \alpha\log L + c`$ dentro de un ambiente fijo; exponente de coherencia.

- **ICE(t)**: serie temporal fusionada, con puertas de CC, de $`{\widehat{\alpha}}_{\text{econ}}`$.

- **Colapso**: independencia residual de $`\widetilde{y} = \log T - \widehat{\alpha}\ logL`$ respecto a $`\log L`$.

- **Evento de decoherencia**: caída significativa del ICE con CC limpio sobre un horizonte prerregistrado.

- **Banderas de CC**: LOW_COVERAGE, NO_COLLAPSE, FAMILY_DIVERGENCE, REGIME_MIX, CLOCK_SHIFT.

**APÉNDICE G — Validación computacional del marco RTM-Econ**

**G.1 Descripción general**

Este apéndice presenta la validación computacional del marco de Economía Rítmica (RTM-Econ). Tres suites de simulación demuestran:

1. α puede estimarse de manera confiable a partir de datos financieros transversales (S1)

2. La caída de α proporciona alerta temprana de recesiones (S2)

3. α varía sistemáticamente entre economías y predice la resiliencia (S3)

**G.2 S1: Estimación de α a partir de datos financieros**

**G.2.1 Modelo**

**Escalamiento RTM-Econ:**

τ(L) = τ₀ × (L/L_ref)^α

donde:

- τ = tiempo característico (recuperación, persistencia)

- L = proxy de escala (capitalización de mercado, tamaño de empresa)

- α = exponente de coherencia

**G.2.2 Parámetros de régimen de mercado**

| Régimen | Período | α | Interpretación |
|--------|--------|---|----------------|
| Crecimiento estable | 2004-2006 | 0.45 | Buena coherencia |
| Pre-crisis | 2007 | 0.35 | Coherencia en declive |
| Crisis | 2008-2009 | 0.20 | Decoherencia, riesgo de cascada |
| Recuperación | 2010-2012 | 0.40 | Reconstruyendo coherencia |
| Nueva normalidad | 2013-2019 | 0.42 | Post-crisis estable |

**G.2.3 Resultados de estimación**

| Régimen | α real | α estimado | Error |
|--------|--------|-------------|-------|
| Crecimiento estable | 0.45 | 0.447 | 0.003 |
| Pre-crisis | 0.35 | 0.346 | 0.004 |
| Crisis | 0.20 | 0.192 | 0.008 |
| Recuperación | 0.40 | 0.396 | 0.004 |
| Nueva normalidad | 0.42 | 0.416 | 0.004 |

**Error absoluto medio: 0.0056 (1.3%)**

**G.2.4 Metaanálisis multifamilia**

Combinando cuatro familias de proxies para el régimen de Crecimiento Estable:

| Familia | α estimado | IC 95% |
|--------|-------------|--------|
| Vida media de recuperación | 0.44 | \[0.38, 0.50\] |
| Persistencia de volatilidad | 0.46 | \[0.39, 0.53\] |
| Decaimiento de autocorrelación | 0.43 | \[0.35, 0.51\] |
| Relajación de flujo de órdenes | 0.47 | \[0.38, 0.56\] |

**ICE combinado: 0.447** (Real: 0.45)

**Heterogeneidad I²: 0.12** (baja, las familias coinciden)

**G.3 S2: Pruebas retrospectivas de alerta temprana**

**G.3.1 Hipótesis H2**

**Afirmación:** Caídas pronunciadas de α preceden recesiones por 6-18 meses.

**G.3.2 Análisis de recesiones**

| Recesión | α_pre → α_mínimo | Δα | Anticipación |
|-----------|------------------|-----|-----------|
| 2001 Punto-Com | 0.42 → 0.28 | 0.14 | 9 meses |
| 2008 CFG | 0.45 → 0.18 | 0.27 | 15 meses |
| 2020 COVID | 0.40 → 0.22 | 0.18 | 3 meses |

**Anticipación media: 9 meses**

**Reducción media de α: 0.20**

**G.3.3 Comparación con otros indicadores**

| Indicador | Tipo | Anticipación típica |
|-----------|------|-------------------|
| ICE (α) | Anticipado (estructural) | 6-15 meses |
| Curva de rendimiento | Anticipado (financiero) | 8-12 meses |
| VIX | Concurrente | 0-1 meses |
| Crecimiento del PIB | Rezagado | Negativo |

**G.3.4 Protocolo de detección**

1. Monitorear ICE rodante con ventana de 3-6 meses

2. Establecer α de referencia durante la expansión

3. Alertar cuando α caiga \>15% por debajo de la referencia

4. Confirmar con otros indicadores anticipados

5. Anticipación esperada: 6-18 meses

**G.4 S3: Comparación entre países**

**G.4.1 Clasificación de países**

| Tipo | Países | α medio | Resiliencia |
|------|-----------|--------|------------|
| Desarrollado | Alemania, Japón, Suiza | 0.52 | Muy alta |
| Centro financiero | EE.UU., RU, Singapur | 0.42 | Moderada |
| Transición | China, India, Corea del Sur | 0.39 | Variable |
| Emergente | Brasil, Turquía, Argentina | 0.28 | Baja |

**G.4.2 Resultados de correlación**

| Relación | Correlación | valor p |
|--------------|-------------|---------|
| α vs Frecuencia de crisis | r = -0.91 | \< 0.001 |
| α vs Reducción promedio | r = -0.95 | \< 0.001 |
| α vs PIB per cápita | r = +0.68 | \< 0.05 |

**G.4.3 Economías más resilientes**

| Rango | País | α | Puntaje de resiliencia |
|------|---------|---|------------------|
| 1 | Suiza | 0.55 | 0.87 |
| 2 | Japón | 0.52 | 0.80 |
| 3 | Alemania | 0.48 | 0.74 |

**G.5 Resumen de la validación computacional**

| Prueba | Métrica | Resultado |
|------|--------|--------|
| Estimación de α | Error medio | 0.56% |
| Metaanálisis | Heterogeneidad I² | 0.12 |
| Alerta temprana | Anticipación media | 9 meses |
| Entre países | Correlación α-crisis | r = -0.91 |

**G.6 Predicciones falsificables**

RTM-Econ falla si:

1. **Sin escalamiento:** τ vs L no muestra ley de potencia dentro de regímenes de mercado

2. **Sin anticipación:** α no disminuye antes de recesiones

3. **Sin patrón entre países:** Las economías de alto α tienen tasas de crisis iguales

4. **Alta heterogeneidad:** Las familias de proxies discrepan (I² \> 0.75)

**G.7 Implicaciones de política**

1. **Pruebas de estrés:** Incluir el monitoreo de α en la vigilancia macroprudencial

2. **Alerta temprana:** La caída de α señala fragilidad en construcción

3. **Diseño de política:** Las intervenciones que aumentan α (amortiguadores, escalonamiento) mejoran la resiliencia

4. **Entre países:** Las economías de bajo α necesitan amortiguadores institucionales más fuertes

**G.8 Nota metodológica sobre escalamiento de recuperación y distribuciones no gaussianas**

Las regresiones estándar de Mínimos Cuadrados Ordinarios (OLS) subestiman severamente la dificultad de la recuperación del mercado debido a un sesgo de atenuación. Definir el día exacto en que un mercado "se recupera" involucra un ruido inmenso (por ejemplo, fronteras de inflación, lógica de reinversión de dividendos). Al desplegar la Regresión de Distancia Ortogonal (ODR) para absorber estos masivos errores de frontera (10% de varianza en la profundidad de la caída, 20% en el tiempo de recuperación), la pendiente de escalamiento de recuperación se empina dramáticamente de un defectuoso 2.49 a un robusto $`3.59\  \pm 0.70`$. Esto sugiere que la recuperación económica es sustancialmente más castigadora y no lineal de lo que suponen los modelos clásicos.

Además, para mapear rigurosamente la forma de las distribuciones financieras globales, utilizamos una simulación Monte Carlo ($`n = 16,000`$) a través de 16 mercados globales. El exponente de cola medio universal converge en $`\alpha = \ 2.966\  \pm 0.236`$. Esto es consistente con el límite teórico de la "Ley Cúbica Inversa" de RTM ($`\alpha \approx 3.0`$) y se aparta de los supuestos gaussianos. Es un resultado convergente, en línea con la ley cúbica inversa establecida (Gabaix et al. 2003), en la que los movimientos extremos de precios son rasgos estadísticos recurrentes de los mercados y no anomalías aisladas.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
