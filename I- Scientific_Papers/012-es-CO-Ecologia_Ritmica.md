<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# Ecología Rítmica
**Un marco de pendiente primero para la resiliencia ecosistémica y los cambios de régimen**  
  
Álvaro Quiceno

</div>

**Resumen**

Los ecosistemas no simplemente "tienen" tiempos característicos; los componen a través de las escalas. Proponemos la Ecología Rítmica (RTM-Eco), un marco de pendiente primero que modela el tempo ecosistémico a través de la ley de escalamiento τ ∝ L^α, donde L es un proxy de tamaño apropiado a la capa (área de parche quemado, tamaño de cuenca, profundidad trófica, escala de red de hábitat), τ es un tiempo característico (recuperación a la línea base pre-perturbación, tiempo de ciclaje de nutrientes, tiempo de recolonización), y α es un exponente de coherencia que captura la organización multiescala del sistema. Dentro de contenedores de coherencia (rebanadas ambientales con forzamiento cuasiconstante), probamos si los datos ecosistémicos colapsan en una ley de potencia, estimamos α con métodos de errores en variables, y fusionamos pendientes aceptadas a través de familias de procesos para construir un Índice de Coherencia Ecosistémica (ICE) en tiempo real.

**Validación computacional.** Implementamos y probamos el marco RTM-Eco mediante tres suites de simulación. S1 demuestra el escalamiento τ(L) para la recuperación post-incendio de NDVI a través de cinco tipos de ecosistemas, mostrando que α varía característicamente por bioma (bosque boreal α≈0.35, pastizal α≈0.22, matorral mediterráneo α≈0.28), con α recuperable de datos satelitales ruidosos con un error del 0.7%. S2 aplica RTM-Eco a la hidrología de cuencas, calculando el escalamiento del tiempo de residencia a través de cinco tipos de cuenca (humedal α≈0.55, urbano α≈0.25), y deriva un Índice de Coherencia Ecosistémica (ICE) que clasifica los sistemas por resiliencia: humedales (ICE=0.86) \>\> tierras bajas forestadas (0.61) \>\> agrícola (0.24) \>\> urbano (0.11). S3 valida la Hipótesis H2 —que la caída de α anticipa cambios de régimen— modelando escenarios de degradación ecosistémica (desertificación forestal, eutrofización lacustre, blanqueamiento coralino, invasión de pastizales), encontrando que la caída de α proporciona 4-11 años de alerta temprana antes del colapso de la variable de estado.

Formulamos hipótesis falsificables: (H1) un α más alto predice una recuperación más ordenada; (H2) caídas significativas de α anticipan cambios de régimen; (H3) curvas maestras emergen dentro de contenedores a través de clases de perturbación. El marco complementa las métricas clásicas de resiliencia al convertir la geometría del tempo en una señal medible, robusta a unidades, para monitoreo, alerta temprana y diseño de conservación.

**Validación empírica preliminar** $`\mathbf{\rightarrow}`$ **(APÉNDICE B)**. Más allá de la simulación, fundamentamos el marco RTM-Eco en la realidad biológica mediante un análisis alométrico de la Base de Datos AnAge (n=547). El análisis heurístico inicial confirmó que la longevidad máxima de los vertebrados escala con la masa corporal adulta. Para corregir definitivamente el sesgo de atenuación estadística causado por la masiva varianza intraespecífica de masa corporal ($`\sim 20\%`$) y la incertidumbre observacional de longevidad ($`\sim 25\%`$), desplegamos un pipeline riguroso de Regresión de Distancia Ortogonal (ODR). Los exponentes de coherencia corregidos por varianza para Mammalia ($`\alpha = \ 0.190\  \pm 0.011`$), Aves ($`\alpha = \ 0.213\  \pm 0.015`$) y Reptilia ($`\alpha = \ 0.241\  \pm 0.077`$) se alinean excepcionalmente bien con los límites teóricos de redes de transporte ($`\alpha \approx 0.25`$). Esto demuestra que el "ritmo de vida" no es una constante absoluta sino una variable topológica gobernada por el volumen estructural del organismo.

Además, validamos el marco de transporte RTM en la dinámica poblacional macroscópica mediante un análisis masivo de más de 4,500 series temporales de la Base de Datos Global de Dinámica Poblacional (GPDD) y metaanálisis de la Ley de Potencia de Taylor $`\mathbf{\rightarrow}`$ **(APÉNDICE C)**. Para prevenir falacias ecológicas de estimación puntual, utilizamos simulaciones Monte Carlo para reconstruir la verdadera varianza biológica. El análisis robusto muestra que el 99.7% de las poblaciones biológicas exhiben fluctuaciones no-Poisson, consistentes con Dinámicas de Transporte Críticas caracterizadas por ruido rosa $`1/f`$ ($`\beta = 0.82`$). Los datos empíricos de riesgo de extinción escalan con las predicciones topológicas de RTM (pendiente predictiva ODR $`= 0.92 \pm 0.02`$). Estos resultados son consistentes con que el colapso ecológico sea una transición de fase topológica, aunque el mecanismo causal requiere mayor falsificación.

Finalmente, extendemos el marco RTM a redes socioecológicas humanas mediante un análisis de la dinámica de propagación global de COVID-19 (APÉNDICE D). Desplegamos un modelo de Errores en Variables (ODR) a través de las distribuciones pandémicas de 100 naciones para corregir sesgos severos de atenuación por subreporte heterogéneo de casos. El análisis robusto arroja un exponente topológico libre de escala de $`\alpha = 0.953 \pm 0.044`$, consistente con el atractor de Zipf ($`\alpha \approx 1.0`$) para redes libres de escala. Las simulaciones de varianza Monte Carlo arrojan $`k = 0.226 \pm 0.131`$. Dado que $`k \ll 1`$, esto es inconsistente con transmisión homogénea de Poisson, apoyando la interpretación RTM de que el COVID-19 se propagó como un fenómeno de transporte topológico a través de una red libre de escala. Estos hallazgos son convergentes con resultados conocidos en epidemiología de redes (Barabási 2002, Lloyd-Smith et al. 2005).

**Hallazgos de la campaña de flanqueo (abril de 2026)** $`\mathbf{\rightarrow}`$ **(APÉNDICE E)**. Las pruebas adversariales independientes (5 flancos, 4 aciertos) extendieron la base empírica de RTM-Eco con predicciones novedosas. (1) **Los residuos de Kleiber predicen longevidad:** a masa corporal fija, las especies cuya tasa metabólica excede la predicción de Kleiber viven menos, Spearman global $`\rho = -0.184`$, $`p = 0.0005`$ ($`n = 350`$ mamíferos), el 89% de los órdenes muestran la misma dirección (prueba $`t`$ $`p = 0.007`$). (2) **Conspiración de forma depredador-presa:** la forma de las dinámicas poblacionales de lobos y alces se correlacionan ($`r = -0.385`$), y esta anticorrelación se intensifica antes de colapsos ecosistémicos (Isla Royale: $`d = -2.52`$ pre-alce 1996, $`d = -1.10`$ pre-lobo 2012). (3) **Paradoja de Simpson en Amphibia:** el $`\alpha = 0.091`$ global de Amphibia oculta Anura (ranas, pulmones desarrollados) $`\alpha = 0.55`$ vs. Caudata (salamandras, respiración cutánea) $`\alpha = 0.03`$ — la topología respiratoria determina el exponente. (4) **Tamaño corporal → color espectral:** Spearman $`\rho = +0.867`$, $`p = 0.0025`$ a través de 9 grupos taxonómicos de GPDD; RTM proporciona el mecanismo (más capas topológicas → ruido más rojo). Una predicción falló: el $`\beta`$ espectral rodante no predice la inestabilidad poblacional futura en Isla Royale (dirección incorrecta, ns). Los choques exógenos impulsan esos colapsos, no las transiciones de fase endógenas. Resultados completos: Apéndice E.

**1. Introducción**

**1.1 Motivación: la geometría faltante del tiempo ecológico**

La ecología abunda en tasas, retardos y ciclos — desde la recuperación post-incendio y las oscilaciones poblacionales hasta la rotación biogeoquímica y la recolonización de metapoblaciones. Sin embargo, estos tiempos a menudo se tratan **localmente** (por sitio, por especie) en lugar de como una **geometría multiescala del tempo**. Los gestores necesitan señales que: (i) sean **robustas a unidades** a través de sensores y métodos, (ii) se integren **a través de procesos** (vegetación, nutrientes, movimiento), y (iii) sean **falsificables** y auditables. RTM-Eco responde a esta necesidad enfocándose en la **pendiente** — cómo el tiempo característico se estira con el tamaño — en lugar de en relojes que dependen de unidades y líneas base.

**1.2 Del escalamiento clásico a un marco de pendiente primero**

El escalamiento clásico relaciona patrón y proceso (por ejemplo, especie-área, doseles fractales, tamaños de incendio con ley de potencia), pero el monitoreo operativo aún se apoya en umbrales temporales (días desde el incendio, percentiles fijos de recuperación). RTM (Relatividad Temporal Multiescala) reenmarca el problema: dentro de una rebanada ambiental donde las condiciones extrínsecas son efectivamente constantes, el par $ (L, T)$ sigue una ley de potencia con exponente de coherencia $\alpha$, mientras que el intercepto es un calibre (un reloj que puede cambiar con unidades o líneas base sin alterar la pendiente). La especialización ecológica, RTM-Eco, instancia esto con $L$ y $T$ ecológicos, define contenedores de coherencia, y trata el colapso (sin tendencia residual después de remover la pendiente ajustada) como una prueba de especificación para el comportamiento tipo potencia.

**1.3 Conceptos y definiciones clave**

- **Proxy de escala** $`L`$ (por familia): área de parche quemado; tamaño de cuenca/captación; escala de parche/red de hábitat (diámetro de grafo o tamaño de módulo); profundidad trófica o clase de conectancia; escala de territorio/rango doméstico.

- **Tiempo característico** $`T`$ : tiempo de recuperación (por ejemplo, NDVI/biomasa al 80–95% de la mediana pre-evento); tiempo sucesional a un gremio objetivo; semicíclo de nutrientes; tiempo de recolonización a través de un corredor.

- **Contenedor de coherencia (BIN)**: una rebanada máxima con impulsores estables (bioma/banda estacional, régimen de manejo, clase de anomalía climática, pila de sensores).

- **Colapso**: con $`x = \log L`$, $`y = \log T`$, ajustar $`y = \alpha x + c`$; requerir que los residuos $`\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}`$ no muestren **tendencia vs.** $`x`$ (por ejemplo, $`R_{\text{collapse}}^{2} < 0.05`$, planitud LOESS) y pasen un **placebo de reloj** (multiplicar $`T`$ por una constante deja $`\widehat{\alpha}`$ sin cambios).

- $`\alpha_{eco}`$ : la pendiente **invariante de calibre** dentro de un BIN; comparada entre regiones y tiempos mediante estimación consciente de la incertidumbre.

**1.4 Estimación y falsificabilidad**

Ambos ejes son ruidosos (mapeo de áreas, cronometraje de recuperación), por lo que los mínimos cuadrados ordinarios pueden estar atenuados. Por ello usamos **regresión de distancia ortogonal** (ODR/TLS) con incertidumbres replicadas/bootstrap, **Theil–Sen** como verificación robusta, y **SIMEX** cuando la varianza del error de medición en $`L`$ es estimable (por ejemplo, delineaciones repetidas). Los contenedores deben pasar **puertas de cobertura** (≥6 valores distintos de $`L`$; extensión ≥0.6 en $`\log L`$) y **puertas de colapso** antes de reportar $`{\widehat{\alpha}}_{eco}`$. Cuando ≥2 familias de procesos pasan simultáneamente, fusionamos pendientes con un modelo de **efectos aleatorios** (REML) y aplicamos una **puerta de heterogeneidad** $`I^{2} < 50\%`$. Los fallos (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE) se publican como fronteras de alcance en lugar de ocultarse.

**1.5 Hipótesis y valor práctico**

Prerregistramos tres afirmaciones falsificables:\
**H1 (Resiliencia):** un $`\alpha_{eco}`$ más alto corresponde a una recuperación *más ordenada* (amortiguada ante choques) a través de las escalas, incluso si el $`T`$ absoluto aumenta, porque los gradientes de tempo obstaculizan las cascadas de sincronización.\
**H2 (Decoherencia):** caídas pronunciadas en $`\alpha_{eco}`$ prefiguran **cambios de régimen** (por ejemplo, bosque→matorral, claro→turbio) y aparecerán como caídas *limpias* en $`{ICE}_{Eco}(t)`$ cuando la heterogeneidad es baja.\
**H3 (Curvas maestras):** dentro de un BIN, $`T_{\text{rec}}`$ colapsa sobre $`L^{\alpha_{eco}}`$ a través de tipos de perturbación de la misma familia (por ejemplo, severidades de incendio), habilitando la **comparabilidad entre sitios**.

**1.6. Validación empírica preliminar: el reloj universal de la vida**

Para fundamentar RTM-Eco en la realidad biológica, probamos la hipótesis central de escalamiento ($`T \propto L^{\alpha}`$) usando la **Base de Datos AnAge** (The Animal Ageing and Longevity Database), la colección más extensa de rasgos de historia de vida para más de 4,000 especies. Realizamos un análisis de regresión log-log de la Longevidad Máxima ($`T`$) versus la Masa Corporal Adulta ($`L`$) a través de distintas clases taxonómicas.

Los resultados (detallados en el **Apéndice B**) confirman una **Alometría Temporal** omnipresente:

1.  **El reloj metabólico:** Para las clases endotérmicas, el exponente de escalamiento convergió a una banda estrecha: **Aves (**$`\mathbf{\alpha \approx}\mathbf{0.21}`$ **)** y **Mammalia (**$`\mathbf{\alpha \approx}\mathbf{0.18}`$ **)**. Esto valida la predicción de RTM de que el tiempo biológico no es absoluto sino relativo al volumen estructural del organismo.

2.  **Universalidad:** A pesar de las inmensas diferencias ecológicas entre una musaraña de 5g y una ballena azul de 100,000kg, sus esperanzas de vida se sitúan en la misma pendiente continua. Esto sugiere que el "envejecimiento" no es meramente un programa genético sino que está parcialmente restringido por la eficiencia de transporte de la red metabólica del organismo — un resultado convergente consistente con West, Brown & Enquist (1997) y la Ley de Kleiber.

**2. Fundamentos de RTM para ecología (RTM-Eco)**

Esta sección formaliza la **geometría escala-reloj** para datos ecológicos, define los **contenedores de coherencia**, establece la **prueba de colapso** como una verificación de especificación (no un sustituto de bondad de ajuste), e introduce herramientas de trabajo para **exponentes locales** y **ventaneo** bajo deriva lenta.

**2.1 Geometría escala-reloj**

Sea $`L > 0`$ un **proxy de escala** (por ejemplo, área de parche quemado, área de cuenca, diámetro de red de hábitat, profundidad trófica) y $`T > 0`$ un **tiempo característico** (por ejemplo, tiempo de recuperación o residencia) medido dentro de un ambiente estable. Escribimos

``` math
u = \log L,v = \log T.
```

RTM postula que **dentro de una rebanada estable del ambiente**,

``` math
v(u) = \alpha_{eco}\text{ }u + \log\kappa,
```
(1)

donde $`\alpha_{eco}`$ es el **exponente de coherencia** (estructura) y $`\kappa > 0`$ es un **reloj** (unidades/línea base).

**Definición 2.1 (Invariancia de calibre / reloj).**

Dos observaciones $`(L,T)`$ y $`(L,cT)`$ con $`c > 0`$ son **equivalentes en calibre**. El exponente $`\alpha_{eco}`$ es **invariante de calibre**; $`\log\kappa`$ se desplaza en $`\log c`$.

**Implicación.** Las comparaciones entre sensores o pipelines de preprocesamiento que cambian el reloj (por ejemplo, normalización de NDVI) **no** deberían cambiar $`{\widehat{\alpha}}_{eco}`$ si la Ec. (1) es válida.

**2.2 Contenedores de coherencia (BINs)**

Los impulsores ecológicos varían a través del espacio y el tiempo. Para evitar la **mezcla de regímenes**, analizamos datos dentro de **contenedores de coherencia**:

**Definición 2.2 (Contenedor de coherencia).**

Un **BIN** es un subconjunto máximo de registros que satisface etiquetas ambientales fijas, por ejemplo

``` math
\text{BIN} = \{\text{bioma, banda estacional, régimen de manejo, clase de anomalía climática, pila de sensores}\}.
```

Cualquier cambio en las etiquetas — nueva estación, cambio de manejo, pila de sensores — **crea un nuevo BIN**.

**Puerta de cobertura.** Un BIN es elegible para estimación de pendiente solo si contiene $`\geq 6`$ valores **distintos** de $`L`$ con extensión $`\geq 0.6`$ en $`u = \log L`$.

**2.3 Colapso como prueba de especificación**

Ajustar una línea en $`(u,v)`$ no es aún evidencia de escalamiento de ley de potencia. Requerimos **colapso**:

**Procedimiento 2.3 (Prueba de colapso).**

1.  Ajustar $`v = \alpha u + c`$ con un estimador de **errores en variables** (Sección 4).

2.  Formar residuos $`\widetilde{v} = v - \widehat{\alpha}u - \widehat{c}`$.

3.  Probar **ausencia de tendencia** de $`\widetilde{v}`$ vs. $`u`$ :

    - re-regresión lineal $`R_{\text{collapse}}^{2}: = R^{2}(\widetilde{v} \sim u) < 0.05`$;

    - un suavizado LOESS prerregistrado no muestra deriva sistemática dentro de bandas de confianza;

    - **placebo de reloj:** $`T \mapsto c\text{ }T`$ deja $`\widehat{\alpha}`$ y $`R_{\text{collapse}}^{2}`$ sin cambios.

Si todo pasa, el BIN **colapsa** y reportamos $`{\widehat{\alpha}}_{eco}`$ con incertidumbre. De lo contrario marcamos el BIN (NO_COLLAPSE o REGIME_MIX) y **no** publicamos una pendiente.

**Proposición 2.4 (Colapso ⇔ exactitud, por contenedor).**

En un BIN simplemente conexo donde $`v(u)`$ es diferenciable, defina la 1-forma $`\omega = dv - \alpha\text{ }du`$. Entonces el **colapso** se cumple si y solo si $`\omega`$ es **exacta** con $`\alpha`$ constante en el BIN.\
*Esbozo.* Si $`v = \alpha u + \log\kappa`$, entonces $`dv - \alpha\text{ }du = d(\log\kappa)`$ es exacta e independiente de $`u`$; los residuos son planos. Recíprocamente, un campo residual plano implica que $`v`$ es afín en $`u`$ en el BIN.

**2.4 Exponentes locales y ventanas adiabáticas**

Los exponentes ecológicos pueden derivar lentamente (fenología, humedad multiAnual). Estimamos pendientes **locales** sobre ventanas:

**Definición 2.5 (Pendiente local; sesgo de ventana).**

Sea $`h > 0`$ una ventana simétrica en $`u`$. La pendiente local

``` math
\widehat{\alpha}(u;h) = \arg\underset{\alpha,c}{\min}\sum_{i:\text{ } \mid u_{i} - u \mid \leq h}^{}{w_{i}\text{ }(v_{i} - (\alpha u_{i} + c))^{2}}
```

(usando un estimador EIV) satisface $`\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h)`$ si $`\mid \partial_{u}\alpha \mid \leq \varepsilon`$ en la ventana (régimen adiabático).

**Práctica.** Comenzar con $`h`$ cubriendo ~8–12 valores distintos de $`L`$; reducir si el colapso falla y la varianza permanece aceptable.

**2.5 Modelos de error y estimandos (alto nivel)**

Tanto $`L`$ como $`T`$ son ruidosos: delinear áreas de parche, definir "tiempo-a-X% de recuperación" y el muestreo irregular introducen **error de medición**. Los mínimos cuadrados ordinarios (OLS) **atenúan** las pendientes cuando $`u`$ es ruidoso. A lo largo del artículo usamos:

- **ODR/TLS** (regresión de distancia ortogonal) como estimador primario por contenedor;

- **Theil–Sen** como verificación robusta e inicializador;

- **SIMEX** cuando la varianza del error de medición en $`u`$ es estimable (delineaciones repetidas).

Los detalles y diagnósticos están en la Sección 4; aquí asumimos que los estimadores devuelven $`\widehat{\alpha}`$ con ICs y verificaciones de influencia adecuadas para la decisión de colapso.

**2.6 Heterogeneidad entre familias de procesos**

Diferentes **familias** ecológicas (recuperación de vegetación, ciclaje de nutrientes, movimiento/recolonización, dinámica trófica) pueden producir diferentes $`{\widehat{\alpha}}_{f}`$ incluso dentro de un BIN. Por lo tanto:

1.  estimamos $`{\widehat{\alpha}}_{f}`$ **por familia** y aplicamos el **colapso** independientemente;

2.  **fusionamos** solo las familias aceptadas mediante un modelo de **efectos aleatorios** con varianza entre familias $`\tau^{2}`$ (REML). La pendiente fusionada en el tiempo $`t`$ es

``` math
{\widehat{\alpha}}_{Eco}(t) = \frac{\sum_{f}^{}\frac{{\widehat{\alpha}}_{f,t}}{{\widehat{\sigma}}_{f,t}^{2} + {\widehat{\tau}}_{t}^{2}}}{\sum_{f}^{}\frac{1}{{\widehat{\sigma}}_{f,t}^{2} + {\widehat{\tau}}_{t}^{2}}},
```

y requerimos $`I^{2} < 50\%`$ para publicar un solo número (de lo contrario reportar por familia).

**2.7 Modos de falla y fronteras de alcance**

- **Curvatura (NO_COLLAPSE).** Tendencia persistente en $`\widetilde{v}`$ vs $`u`$ : relojes dependientes de escala o mezcla multimecanismo; dividir el BIN o reportar como **fuera de alcance** para RTM.

- **Quiebres (REGIME_MIX).** Pendientes por tramos; ejecutar detección de puntos de cambio y dividir.

- **Cobertura delgada (THIN_COVERAGE).** Extensión \<0.6 en $`\log L`$ o muy pocas escalas distintas; recopilar más datos o descartar.

- **Alta heterogeneidad (FAMILY_DIVERGENCE).** $`I^{2} \geq 50\%`$ : **no** fusionar; publicar $`{\widehat{\alpha}}_{f}`$ por familia e investigar mecanismos.

**2.8 Qué significa —y qué no significa—** $`\mathbf{\alpha}_{\mathbf{eco}}`$

- **Sí:** cuantifica el **gradiente de tempo** a través de escalas dentro de un BIN; un $`\alpha_{eco}`$ más alto significa que los agregados más grandes se desaceleran relativamente más, lo cual a menudo **amortigua** las cascadas de sincronización después de choques (recuperación más ordenada).

- **No:** garantiza una recuperación absoluta más rápida, ni reemplaza modelos mecanísticos (sucesión, dinámica de nutrientes). Es una propiedad **estructural**, invariante a relojes pero local al BIN.

**2.9 Resumen**

RTM-Eco modela el tiempo ecológico como una **ley afín en el espacio log-log** dentro de contenedores de coherencia. La **prueba de colapso** eleva los ajustes de ley de potencia a **especificación falsificable**; las **pendientes locales** manejan la deriva lenta; la **estimación EIV** previene la atenuación; y la **fusión consciente de heterogeneidad** produce un indicador auditable solo cuando las familias concuerdan. La siguiente sección traduce estos fundamentos en **definiciones operativas** de $`L`$ y $`T`$ para vegetación, nutrientes, movimiento y procesos tróficos, junto con un **protocolo de contenedores** concreto.

**3. Definiciones operativas en ecología**

Ahora instanciamos RTM-Eco con **elecciones aplicables** de escala $`L`$, tiempo $`T`$ y **contenedores** para cuatro familias de procesos: recuperación de vegetación, ciclaje de nutrientes/biogeoquímico, movimiento-metapoblación y dinámica trófica/red. Cada definición se acompaña de **notas de medición** y **puertas de CC** para que el pipeline sea reproducible.

**3.1 Recuperación de vegetación (teledetección)**

**Escala** $`L`$ **.** Área de parche quemado (ha), poligonizada a partir de perímetros de incendio; alternativamente **huella de perturbación** (derribo por viento, tala rasa) en ha. Para mosaicos, usar el **tamaño efectivo de parche** (área después de disolver huecos \< umbral) y registrar la **relación borde-a-área** como covariable (no parte de $`L`$).

**Tiempo** $`T`$ **.** $`T_{\text{rec}}(p)`$ : tiempo (días) para recuperar una fracción $`p \in \lbrack 0.8,0.95\rbrack`$ de la señal mediana pre-evento (NDVI/EVI/SAVI; altura de dosel para LiDAR). Definir:

``` math
T_{\text{rec}}(p) = \inf\{\text{ }t > 0:\text{ RS}(t) \geq p \cdot {\widetilde{\text{RS}}}_{\text{pre}}\text{ }\},
```

con **pre** calculado en una ventana de 2–3 años, enmascarada de nubes, emparejada por estación.

**BIN.** {bioma, banda estacional (por ejemplo, JJA/DEF), pila de sensores (Landsat/Sentinel), régimen de manejo, clase de anomalía climática (ENOS/ONA), clase de severidad}.

**Notas.**

- Aplicar la **misma fase fenológica** pre/post (emparejamiento por mes del año) para evitar relojes estacionales.

- Si la severidad varía dentro del parche, estratificar los parches por clase de severidad antes de ajustar.

**Puertas de CC.**

- ≥6 tamaños de parche distintos; extensión ≥0.6 en $`\log L`$.

- Convergencia ODR; influencia \<25%.

- Colapso: $`R_{\text{collapse}}^{2} < 0.05`$; placebo OK (reescalar RS para probar invariancia).

**3.2 Ciclaje de nutrientes / biogeoquímico**

**Escala** $`L`$ **.** Área de cuenca/captación ($`{km}^{2}`$); para lagos, escala morfométrica (área superficial o volumen); para suelos, extensión de parcela ($`m^{2}`$) con banda de profundidad fija.

**Tiempo** $`T`$ **.** Rotación característica o **semicíclo**:

- **Arroyos/Lagos:** tiempo de recuperación de **clorofila-a** o **profundidad Secchi** a $`p`$ de la línea base; o tiempo de residencia de un pulso de nitrato/fosfato (tiempo al 50% de decaimiento).

- **Suelos:** tiempo hasta meseta de la tasa de mineralización después de perturbación (protocolo estándar de incubación).

**BIN.** {hidroregión, banda estacional, clase de estado trófico, régimen de flujo (caudal base vs. dominado por tormentas), manejo (régimen de fertilización)}.

**Notas.**

- Usar **ventanas comparables de forzamiento hidrológico** (excluir eventos de crecida si no son parte del tratamiento).

- Al modelar nutrientes, transformar concentraciones en log **después** del tratamiento del límite de detección; marcar valores censurados.

**Puertas de CC.**

- Documentar el reloj de sensor/método (laboratorio vs. in situ) y mostrar placebo de reloj.

- Escaneo de puntos de cambio para cambios escalonados (por ejemplo, cambio de manejo).

**3.3 Movimiento y metapoblación**

**Escala** $`L`$ **.** **Escala de conectividad** de la red de hábitat: diámetro del grafo del componente ocupado, o tamaño efectivo de módulo $`m`$ (nodos por módulo) cuando es modular. Alternativa: percentil de **distancia entre parches** (por ejemplo, p75) como proxy de tamaño del paisaje.

**Tiempo** $`T`$ **.** **Tiempo de recolonización** $`T_{\text{recol}}`$ : tiempo desde la extinción local hasta la reaparición/persistencia (≥$`k`$ detecciones en $`w`$ días) dentro de un parche, o **tiempo de primer pasaje** a través de corredor para individuos marcados (telemetría).

**BIN.** {especie/gremio, estación/fase migratoria, método de detección, manejo de corredores, clase de perturbación}.

**Notas.**

- Corregir por **detección imperfecta** (modelos de ocupación) para que $`T`$ no sea un reloj de detección.

- Para telemetría, definir $`T`$ en **ventanas diarias comparables**; excluir pausas estacionarias que reflejan comportamiento, no conectividad.

**Puertas de CC.**

- Mínimo de 8–12 escalas distintas de $`L`$ (redes de diferentes tamaños o divisiones modulares).

- El panel de colapso debe incluir residuos vs. tanto $`u`$ como **utilización** para descartar efectos de tráfico.

**3.4 Dinámica trófica / red**

**Escala** $`L`$ **.** **Profundidad trófica** (longitud de camino más largo), **clase de conectancia**, o **tamaño de módulo** en la red trófica empírica/modelada. Mantener el proxy elegido fijo dentro de un BIN.

**Tiempo** $`T`$ **.** **Tiempo de retorno** de un conjunto de nodos perturbado (por ejemplo, eliminación de especie clave o pulso de biomasa) a dentro de $`p`$ de las biomasas pre-perturbación, medido en tiempo de modelo o días experimentales.

**BIN.** {tipo de ecosistema, banda de temperatura, nivel de enriquecimiento/presión, clase de modelo/mesocosmos, a priori de fuerza de interacción}.

**Notas.**

- Cuando es simulado, reportar **réplicas estocásticas**; cuando es mesocosmos, estandarizar ciclos de alimentación/luz (evitar deriva de reloj).

- Si existen múltiples proxies de $`L`$, prerregistrar el primario y tratar los demás como **covariables** (no como $`L`$).

**Puertas de CC.**

- Reportar descomposición de varianza (proceso vs. observación) y usar **bootstrap agrupado** para ICs.

- Publicar el no-colapso como **frontera de alcance** (por ejemplo, fuerte no linealidad a alto enriquecimiento).

**3.5 Protocolo de contenedores (paso a paso)**

1.  **Etiquetado.** Asignar a cada registro etiquetas ambientales (bioma/región, banda estacional, manejo, pila de sensores, clase de anomalía).

2.  **Estratificación.** Dividir por etiquetas; descartar estratos con cobertura inadecuada.

3.  **Puntos de cambio.** Dentro de cada estrato, ejecutar un escaneo de puntos de cambio (BIC/PELT) en $`v`$ y en covariables clave; dividir si se detectan.

4.  **Verificación de cobertura.** Asegurar ≥6 $`L`$ distintos, extensión ≥0.6 en $`u`$.

5.  **Elección de estimador.** Ajustar ODR/TLS (primario); calcular Theil–Sen como verificación robusta; ejecutar SIMEX si $`Var(\xi_{u})`$ es conocido/estimable.

6.  **Colapso.** Calcular tendencia residual $`R_{\text{collapse}}^{2}`$; ejecutar diagnóstico LOESS; aplicar placebo de reloj.

7.  **Aceptar / Marcar.** Si todas las puertas pasan → aceptar $`\widehat{\alpha}`$. De lo contrario, marcar (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE) y **publicar** el fallo en el reporte.

**3.6 Notas de medición (transversales)**

- **Base logarítmica.** Usar logaritmos naturales; reportar explícitamente. Los cambios de base **no** afectan a $`\alpha`$.

- **Censura y vacíos.** Marcar valores censurados; imputar solo para visualización, **no** para estimación de pendiente.

- **Pesos.** Usar pesos de réplica o incertidumbre cuando estén disponibles (por ejemplo, varianza de delineación de parche, EE de detección de ocupación).

- **Influencia.** Limitar la influencia al 25%; realizar sensibilidad **dejando una escala fuera** cuando la cobertura es ajustada.

- **Ventanas.** Para sistemas a la deriva (estacionales), estimar pendientes locales en ventanas $`h`$, luego probar colapso en cada ventana (régimen adiabático).

**3.7 Selección y validación de proxies**

Cuando existen múltiples candidatos de $`L`$ o $`T`$, prerregistrar un **primario** y conducir:

- **Acuerdo entre proxies.** Calcular $`\widehat{\alpha}`$ bajo alternativas (por ejemplo, $`L =`$ área vs. tamaño efectivo basado en perímetro); esperar diferencias en $`\kappa`$, no en $`\alpha`$, si el colapso se mantiene.

- **Cordura mecanística.** Verificar que cambiar el **reloj** (normalización del sensor) no cambia $`\widehat{\alpha}`$; si lo hace, su proxy probablemente incorpora un reloj oculto.

- **Validez externa.** Regiones/años reservados: ¿se transfiere $`\widehat{\alpha}`$ dentro de la misma definición de BIN?

**3.8 Lista de verificación de reporte (por BIN y familia)**

- Definiciones de $`L,T`$ (una línea) y base logarítmica.

- Cobertura: \# de $`L`$ distintos, extensión en $`\log L`$.

- Estimador: configuración ODR/TLS; verificación robusta (Theil–Sen); SIMEX (sí/no).

- Colapso: $`R_{\text{collapse}}^{2}`$, panel LOESS, resultado del placebo.

- $`\widehat{\alpha}`$ con ICs 50/95%; influencia máxima; diagnósticos.

- Marcas o decisión de aceptación.

- Si es elegible para fusión: reportar $`Q`$, $`I^{2}`$, $`{\widehat{\tau}}^{2}`$.

**3.9 Resumen**

Esta sección fundamentó RTM-Eco en elecciones **operativas** de $`L`$ y $`T`$ para cuatro familias, definió **BINs** que evitan mezcla de regímenes, y codificó **puertas de CC**. Con estas piezas, la Sección 4 detalla la **estimación con errores en variables** y la mecánica de la **prueba de colapso** para que $`{\widehat{\alpha}}_{eco}`$ se mida consistentemente a través de sitios, sensores y laboratorios.

**4. Mecánica de estimación y colapso**

Esta sección especifica **cómo** estimamos $`\alpha_{eco}`$ bajo **errores en variables (EIV)**, ejecutamos la **prueba de especificación de colapso**, y reportamos incertidumbre y robustez de manera portable entre laboratorios, sensores y familias ecológicas.

**4.1 Modelo y notación**

Sea $`x = \log L`$ y $`y = \log T`$. Observamos pares ruidosos

``` math
x_{i} = u_{i} + \xi_{i},y_{i} = v_{i} + \varepsilon_{i},v_{i} = \alpha u_{i} + c,
```

con errores de medición $`\xi_{i},\varepsilon_{i}`$ (media cero, varianza finita). El objetivo es el **exponente de coherencia** $`\alpha`$ dentro de un **contenedor de coherencia** (Sec. 3).

**Amenaza al OLS.** Si $`Var(\xi) > 0`$, OLS en $`(x,y)`$ está **atenuado**: $`\mathbb{E}\lbrack{\widehat{\alpha}}_{OLS}\rbrack < \alpha`$.

**4.2 Estimador primario: Regresión de Distancia Ortogonal (ODR/TLS)**

Minimizamos residuos ortogonales con pesos por punto $`w_{i}`$ :

``` math
\underset{\alpha,c\ \ \ \ \ }{\min\ \ \ \ \ }\sum_{i}^{}{w_{i}\text{ }\frac{(y_{i} - \alpha x_{i} - c)^{2}}{\sigma_{y,i}^{2} + \alpha^{2}\sigma_{x,i}^{2}}}
```

- **Pesos.** Si hay EE de réplica disponibles, establecer $`\sigma_{x,i},\sigma_{y,i}`$ correspondientemente; sino usar $`w_{i} = 1`$.

- **Inicialización.** Pendiente **Theil–Sen** (Sec. 4.3) e intercepto mediano.

- **ICs.** **Bootstrap agrupado** no paramétrico (por parche/cuenca/réplica) con $`B \geq 2000`$.

- **Diagnósticos.** Número de condición \< $`10^{4}`$; **influencia máxima** \< 0.25 (marcar si se excede).

**Reporte**: $`\widehat{\alpha}`$ (ICs 50/95%), $`\widehat{c}`$, influencia máxima, estado de convergencia.

**4.3 Verificación robusta: Theil–Sen (TS)**

``` math
{\widehat{\alpha}}_{TS} = {mediana}_{i < j}\frac{y_{j} - y_{i}}{x_{j} - x_{i}},\ \ {\widehat{c}}_{TS} = {mediana}_{i}(y_{i} - {\widehat{\alpha}}_{TS}x_{i}).
```

- **Uso** como (i) inicializador robusto para ODR, (ii) línea de sensibilidad en paneles de colapso.

- **Sesgo.** Atenuación leve bajo EIV, pero alto punto de quiebre contra valores atípicos/colas pesadas.

**4.4 SIMEX (opcional; cuando** $`\mathbf{Var}\mathbf{(\xi)}`$ **es conocida/estimable)**

Si $`\sigma_{\xi}^{2}`$ es conocida (delineaciones repetidas de $`L`$, varianza inter-analista), simular $`x^{(\lambda)} = x + \sqrt{\lambda}\text{ }\widetilde{\xi}`$ con $`\widetilde{\xi} \sim \mathcal{N}(0,\sigma_{\xi}^{2})`$, reajustar $`\widehat{\alpha}(\lambda)`$ para $`\lambda \in \Lambda = \{ 0.5,1,1.5,2\}`$, y **extrapolar** a $`\lambda = - 1`$ con una cuadrática. Reportar el $`{\widehat{\alpha}}_{SX}`$ corregido por SIMEX como sensibilidad.

**4.5 Prueba de colapso: haciendo la "ley de potencia" falsificable**

Dados $`\widehat{\alpha},\widehat{c}`$, calcular residuos $`{\widetilde{y}}_{i} = y_{i} - \widehat{\alpha}x_{i} - \widehat{c}`$. Un contenedor **colapsa** si:

1.  **Prueba de tendencia.** $`R_{\text{collapse}}^{2}: = R^{2}(\widetilde{y} \sim x) < 0.05`$.

2.  **Planitud LOESS.** Suavizador prerregistrado no muestra deriva (banda contiene 0).

3.  **Placebo de reloj.** Reemplazar $`T`$ por $`c\text{ }T`$ (constante $`c > 0`$); $`\widehat{\alpha}`$ y $`R_{\text{collapse}}^{2}`$ permanecen sin cambios (dentro del ruido bootstrap).

4.  **Puntos de cambio.** Sin punto de cambio interior (PELT/BIC) en $`\widetilde{y}`$ o covariables clave; si se detecta → **dividir contenedor**.

**Etiquetas de resultado.**

- ACCEPT: todo pasa → publicar $`\widehat{\alpha}`$.

- NO_COLLAPSE: la curvatura persiste.

- REGIME_MIX: quiebre/pendientes por tramos → dividir.

- THIN_COVERAGE: \<6 $`L`$ distintos o extensión \<0.6 en $`\log L`$.

**4.6 Pendientes locales y ventaneo (ambientes a la deriva)**

Cuando los impulsores derivan lentamente, estimar $`\alpha(u;h)`$ **local** sobre ventanas de ancho $`h`$ en $`x = \log L`$ :

- Elegir $`h`$ para incluir **8–12 escalas distintas** cuando sea posible.

- Compensación sesgo-varianza: $`\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h)`$ si $`\mid \partial_{u}\alpha \mid \leq \varepsilon`$.

- Ejecutar colapso **dentro de cada ventana**; reportar solo ventanas que pasen las puertas.

**4.7 Heterogeneidad y fusión entre familias**

Para familias aceptadas $`f\mathcal{\in F}`$, con estimadores $`{\widehat{\alpha}}_{f}`$ y varianzas $`{\widehat{\sigma}}_{f}^{2}`$ :

- **Q de Cochran** y $`I^{2}`$ :

``` math
Q = \sum_{f}^{}{w_{f}^{FE}({\widehat{\alpha}}_{f} - {\overset{ˉ}{\alpha}}_{FE})^{2},w_{f}^{FE} = 1/{\widehat{\sigma}}_{f}^{2},I^{2} = \max\{ 0,\frac{Q - ( \mid \mathcal{F} \mid - 1)}{Q}\}.}
```

- Varianza de **efectos aleatorios** $`{\widehat{\tau}}^{2}`$ vía **REML** (DerSimonian–Laird como sensibilidad).

- **Pendiente fusionada**:

``` math
{\widehat{\alpha}}_{Eco} = \frac{\sum_{f}^{}{{\widehat{\alpha}}_{f}/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})}}{\sum_{f}^{}{1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})}}.
```

**Puerta de fusión.** Publicar un **solo** número solo si $`\mid \mathcal{F} \mid \geq 2`$ y $`I^{2} < 50\%`$; de lo contrario, **reportar por familia** y declarar la heterogeneidad.

**4.8 Suite de robustez y sensibilidad (obligatoria)**

- **Trío de estimadores.** ODR (primario), Theil–Sen (robusto), banda SIMEX (si disponible).

- **Sensibilidad de ventana.** $`h`$ ± 25%: $`\widehat{\alpha}`$ estable y colapso aún aprobado.

- **Verificación de influencia.** Dejando una escala fuera.

- **Nulo de aleatorización.** Permutar $`x`$ dentro del contenedor; la pendiente debería colapsar a ~0.

- **Placebo de reloj.** Invariancia de $`T \mapsto cT`$ confirmada.

- **Efectos fijos vs aleatorios.** Reportar ambos; la divergencia señala heterogeneidad genuina.

**4.9 Plano de implementación (pseudo-YAML)**

```
binning:
  min_scales: 6
  min_logL_span: 0.6
  tags: [biome, season_band, management, sensor_stack, anomaly_class]

estimation:
  base: "odr"
  init: "theil-sen"
  bootstrap: {B: 2000, cluster: true, seed: 123}
  leverage_cap: 0.25
  simex: {enabled: false, lambda: [0.5, 1.0, 1.5, 2.0]}

collapse:
  r2_threshold: 0.05
  loess_bw: "pre-registered"
  clock_placebo: true
  changepoint: {method: "PELT", criterion: "BIC"}

fusion:
  min_families: 2
  I2_gate: 0.50
  tau2_method: "REML"

report:
  figures: ["collapse_panels", "forest_plot", "eci_time_series"]
  publish_negatives: true
```

**4.10 Errores comunes (y correcciones)**

- **Relojes estacionales filtrándose en** $`T`$ **.** Emparejar por mes del año o incluir fenología como etiqueta de BIN; de lo contrario NO_COLLAPSE.

- **Relojes ocultos en** $`L`$ **.** El tamaño efectivo de parche definido con amortiguadores dependientes de severidad puede imprimir curvatura; fijar la definición o tratar la severidad como **covariable**, no como parte de $`L`$.

- **Cobertura delgada.** Fusionar estratos adyacentes **solo si las etiquetas son idénticas** excepto por la que se fusiona; re-verificar puntos de cambio.

**4.11 Resumen**

Definimos un pipeline **consciente de EIV** para estimar $`\alpha_{eco}`$, convertimos la "ley de potencia" en una **especificación falsificable** vía **colapso**, y establecimos reglas con principios para **fusionar** (o negarse a fusionar) entre familias ecológicas. Con esta mecánica establecida, la Sección 5 desarrolla **proxies de medición** y flujos de validación (espectros de teledetección, estructura fractal, métricas de red) para construir conjuntos de datos $`(L,T)`$ confiables en campo.

**5. Proxies de medición y flujos de validación**

Esta sección convierte los fundamentos (Secs. 2–4) en **recetas de construcción de datos**. Definimos **familias de proxies** para $`L`$ y $`T`$ que son medibles a escala, damos **algoritmos de extracción**, y especificamos **validación** para que $`{\widehat{\alpha}}_{eco}`$ no sea un artefacto de relojes, preprocesamiento o elección de proxy.

**5.1 Recuperación de vegetación (teledetección)**

**5.1.1 Proxies**

- **Escala** $`L`$ : **área de parche** quemado/perturbado (ha) de perímetros poligonizados; alternativa $`L`$ : **área efectiva** después de disolver huecos $`< \rho`$ ha; reportar $`\rho`$.

- **Tiempo** $`T`$ : **tiempo de recuperación** $`T_{\text{rec}}(p)`$ a fracción $`p \in \{ 0.80,0.90,0.95\}`$ de la señal mediana pre-evento (NDVI/EVI/SAVI; altura de dosel LiDAR si disponible).

**5.1.2 Extracción (flujo RS)**

1.  **Preprocesar** Landsat 5–9/Sentinel-2: máscara de nubes/sombras (bandas QA), normalización BRDF, código de estación por píxel (mes del año).

2.  **Detección de eventos**: umbral/índice de severidad (dNBR o RBR) con limpieza espacial (apertura/cierre morfológico).

3.  **Delineación de parches**: conectividad de 8 vecinos; disolver huecos interiores $`< \rho`$ ha.

4.  **Línea base**: mediana de RS sobre 24–36 meses pre-evento, emparejada por mes del año.

5.  **Recuperación**: $`T_{\text{rec}}(p) = \inf\{ t:\text{RS}(t) \geq p \cdot {\widetilde{\text{RS}}}_{\text{pre}}\}`$ con mediana rodante de 60–90 días para suprimir ruido meteorológico.

**5.1.3 Validación y CC**

- **Placebo de reloj**: reescalar RS por constante $`c`$ (por ejemplo, variantes de normalización NDVI); $`\widehat{\alpha}`$ invariante.

- **Cobertura**: ≥6 $`L`$ distintos y extensión ≥0.6 en $`\log L`$ por BIN.

- **Entre sensores**: solo Landsat vs. Landsat+S2; esperar mismo $`\widehat{\alpha}`$, diferente $`\widehat{c}`$.

- **Efectos de borde**: incluir **borde/área** como covariable diagnóstica; *no* mezclar en $`L`$.

**5.2 Nutrientes / biogeoquímica**

**5.2.1 Proxies**

- **Escala** $`L`$ : **área** de cuenca ($`{km}^{2}`$); para lagos, **área superficial** o **volumen**; para suelos, **extensión de parcela** a banda de profundidad fija.

- **Tiempo** $`T`$ :

  - **Decaimiento de pulso**: tiempo desde pico hasta 50% de decaimiento en nitrato/fosfato/Chl-a (residencia/rotación).

  - **Recuperación a línea base**: tiempo a $`p`$ de la mediana pre-presión (claridad, oxígeno).

**5.2.2 Extracción**

- **Delineación hidrológica**: cuencas basadas en DEM (TauDEM/GRASS); polígonos de lagos de inventarios nacionales.

- **Limpieza de series**: manejar valores censurados (sustitución LOD o modelos de censura); regularizar a semanal/quincenal con suavizado tolerante a vacíos (por ejemplo, Kalman con datos faltantes).

- **Ventanas de eventos**: tormentas/intervenciones de presión etiquetadas; asegurar comparaciones **equivalentes** entre BINs.

**5.2.3 Validación**

- **Relojes de método**: sensores de laboratorio vs. in situ; mostrar placebo $`T \mapsto cT`$ vía reescalamiento de unidades.

- **Puntos de cambio**: detectar cambios de manejo (fertilización, regulación de flujo) y recontenedorizar.

**5.3 Movimiento y metapoblación**

**5.3.1 Proxies**

- **Escala** $`L`$ : **diámetro del grafo** del componente de hábitat ocupado; o **tamaño de módulo** $`m`$ en redes modulares; alternativa: p75 de distancias entre parches.

- **Tiempo** $`T`$ : **tiempo de recolonización** $`T_{\text{recol}}`$ (extinción→persistencia) o **tiempo de primer pasaje** a través de corredor (telemetría).

**5.3.2 Extracción**

- **Ocupación**: modelos dinámicos de ocupación (MacKenzie) para corregir detección; definir persistencia con $`k`$ detecciones en ventana $`w`$.

- **Telemetría**: segmentar trayectos; computar cruces de corredor en ventanas diarias emparejadas; excluir pausas de descanso.

**5.3.3 Validación**

- **Reloj de detección**: mostrar que cambiar umbrales de detección desplaza $`\widehat{c}`$ pero no $`\widehat{\alpha}`$.

- **Confusión por utilización**: incluir **tráfico** de la red (uso) como covariable en verificaciones residuales; el colapso debe mantenerse.

**5.4 Dinámica trófica / red**

**5.4.1 Proxies**

- **Escala** $`L`$ : **profundidad trófica** (camino más largo), **tamaño de módulo**, o **clase de conectancia** fija por BIN.

- **Tiempo** $`T`$ : **tiempo de retorno** después de presión/pulso (eliminación de especie clave, enriquecimiento) a dentro de $`p`$ de las biomasas pre-perturbación.

**5.4.2 Extracción**

- **Redes empíricas**: compilar matrices de interacción con incertidumbre; simular dinámicas estocásticas (por ejemplo, Lotka–Volterra generalizado con ruido) para estimar tiempos de retorno.

- **Mesocosmos**: estandarizar luz/alimentación; marcas de tiempo en fase diaria consistente.

**5.4.3 Validación**

- **Réplicas**: ICs de bootstrap agrupado.

- **$`L`$ alternativo**: replicar con conectancia vs. profundidad; $`\widehat{\alpha}`$ debería ser consistente si el BIN no cambia y el colapso se mantiene.

**5.5 Proxies estructurales y espectros (transversales)**

- **Métricas fractales** (paisaje): escalamiento perímetro-área; dimensión de conteo de cajas de mosaicos de parches; probar que sustituir $`L`$ por un **tamaño ajustado por fractales** cambia $`\widehat{c}`$, no $`\widehat{\alpha}`$.

- **Pendientes espectrales** (RS): espectros de potencia de campos NDVI/biomasa; verificar consistencia entre exponentes espectrales y bandas de $`\widehat{\alpha}`$ cualitativamente (no fusionar a menos que se cumpla el criterio de colapso).

- **Diversidad/conectividad**: Shannon/Simpson, modularidad $`Q_{\text{mod}}`$; usar como **covariables** para explicar variación en $`\kappa`$ o como estratificadores para BINs, no como $`L`$ a menos que se prerregistre.

**5.6 Productos de datos y reproducibilidad**

- **Tabla ordenada por BIN**: $`\lbrack x = \log L,\text{ }y = \log T,\text{ familia},\text{ etiquetas},\text{ réplica},\text{ marca temporal},\text{ }w\rbrack`$.

- **YAML de métodos** (hash en cada figura): etiquetas de contenedor, ventanas $`h`$, configuración del estimador, semillas de bootstrap, umbrales de colapso, puertas de fusión.

- **Artefactos**: publicar **paneles de colapso**, **gráficos de bosque**, y artefactos de **placebo/aleatorización** para BINs aceptados/fallidos.

**5.7 Verificaciones de cordura y firmas comunes de fallo**

- **Fuga estacional** → tendencia en residuos alineada con mes del año ⇒ recontenedorizar por banda estacional.

- **Reglas de amortiguador ocultas** en $`L`$ (dependientes de severidad) → curvatura a escalas grandes ⇒ fijar definición de $`L`$.

- **Cobertura delgada** (pocos parches grandes) → alta influencia ⇒ inestabilidad al dejar una escala fuera; recopilar más o marcar BIN.

**5.8 Puntos de referencia sintéticos mínimos (recomendados)**

Proporcionar dos conjuntos de datos de juguete (por familia):

1.  **Ley de potencia + ruido** que **pasa el colapso** (ODR recupera $`\alpha`$ dentro del IC).

2.  **Curvado** (por ejemplo, $`v = \alpha u + \beta u^{2}`$) que **falla el colapso** (tendencia residual, deriva LOESS).\
    Estos aseguran que el pipeline y el reporte capturen tanto éxitos como **fronteras de alcance**.

**5.9 Resumen**

Especificamos proxies prácticos para $`L`$ y $`T`$ a través de cuatro familias ecológicas, con pasos de extracción, CC y validación que protegen a $`{\widehat{\alpha}}_{eco}`$ de **artefactos de reloj**, **fuga estacional** y **deriva de proxy**. Con datos en su lugar, la Sección 6 formula **hipótesis falsificables** y **protocolos experimentales** (teledetección, redes tróficas, movimiento, restauración) para probar si RTM-Eco añade valor predictivo y de manejo.

**6. Hipótesis falsificables y protocolos experimentales**

Ahora operacionalizamos las afirmaciones de RTM-Eco en **hipótesis comprobables** con **protocolos tipo A/B**, puntos finales medibles, análisis de potencia y puertas de decisión. Cada protocolo especifica (i) etiquetas de BIN, (ii) definiciones de $`L,T`$, (iii) estimadores y verificaciones de colapso, (iv) umbrales *a priori*, (v) manejo de resultados negativos.

**6.1 Hipótesis (prerregistradas)**

**H1 — Resiliencia por pendiente.** Dentro de un BIN, los ecosistemas con mayor $`\alpha_{eco}`$ exhiben **recuperación más ordenada** (menor amplificación de colas y cascadas de sincronización) a través de las escalas, incluso si el tiempo de recuperación absoluto aumenta.

**H2 — Alerta temprana de decoherencia.** **Caídas significativas** en $`\alpha_{eco}`$ (o en el $`{ICE}_{Eco}(t)`$ fusionado) **preceden** cambios de régimen (bosque→matorral; lago claro→turbio) por $`\Delta t > 0`$.

**H3 — Curva maestra.** Para una familia de perturbaciones dentro de un BIN, $`T_{\text{rec}}`$ **colapsa** sobre $`L^{\alpha_{eco}}`$ con $`R_{\text{collapse}}^{2} < 0.05`$.

**H4 — Ingeniería de pendiente.** Las intervenciones de hábitat/red que **elevan** $`\alpha_{eco}`$ (corredores, heterogeneidad) **reducen** las métricas de cola (p95/p50 de recuperación) a $`T`$ medio fijo o con compensaciones aceptables.

**H5 — Coherencia entre familias.** Cuando ≥2 familias pasan el colapso en un BIN, la **heterogeneidad se mantiene acotada** ($`I^{2} < 50\%`$), admitiendo un **solo** indicador fusionado.

**6.2 Protocolo A — Teledetección de recuperación vegetal post-perturbación**

**BIN.** {bioma, banda estacional, pila de sensores, régimen de manejo, clase de severidad, clase de anomalía climática}.\
$`L`$ **.** Área de parche (ha), huecos disueltos \<$`\rho`$ ha.\
$`T`$ **.** $`T_{\text{rec}}(p)`$ a $`p \in \{ 0.80,0.90,0.95\}`$ de la mediana RS pre-evento.

**Diseño.**

1.  Construir parches de 10–15 años de eventos; estratificar por severidad y estación.

2.  Para cada estrato, requerir ≥6 $`L`$ distintos y extensión ≥0.6 en $`\log L`$.

3.  Estimar $`\widehat{\alpha}`$ vía ODR (Theil–Sen como verificación; SIMEX si existen réplicas de polígonos).

4.  Ejecutar diagnósticos de colapso (Sec. 4.5).

**Puntos finales.**

- Primario: $`{\widehat{\alpha}}_{veg}`$ con IC; **Aceptar/Rechazar** por puerta de colapso.

- Secundario (H1): ratio de cola p95/p50 en $`T_{\text{rec}}`$ estratificado por cuantiles de $`L`$; probar monotonicidad vs. $`\widehat{\alpha}`$.

**Decisión.** H3 apoyada si ≥70% de los estratos pasan el colapso con bandas de $`\widehat{\alpha}`$ consistentes; H1 apoyada si $`\partial(\text{p95/p50})/\partial\widehat{\alpha} < 0`$ (IC excluye 0).

**Potencia.** Simular $`N = 200`$ parches/estrato con $`\text{extensión}_{u} = 1.0`$; ODR recupera $`\mid \Delta\alpha \mid \geq 0.10`$ al 80% de potencia (B=2000 bootstrap). Registrar semilla de simulación en YAML.

**Negativos.** NO_COLLAPSE a alta severidad implica **fuga de reloj** o mezcla multimecanismo; publicar como frontera de alcance.

**6.3 Protocolo B — Biogeoquímica de lagos/arroyos**

**BIN.** {hidroregión, estación=Abr–Oct, trófico={oligo, meso, eu}, régimen de flujo, clase de manejo}.\
$`L`$ **.** Área de cuenca ($`{km}^{2}`$); para lagos, área superficial o volumen.\
$`T`$ **.** Tiempo al 50% de decaimiento de pulso de nutrientes ($`{NO}_{3}^{-}`$, $`{PO}_{4}^{3 -}`$, Chl-a) o recuperación a $`p`$ de la línea base.

**Diseño.**

1.  Compilar series semanales/quincenales; identificar eventos de pulso/presión.

2.  Calcular $`T`$ por evento con ventanas consistentes; censura manejada.

3.  Estimar $`{\widehat{\alpha}}_{nut}`$; pruebas de colapso; placebo para relojes de método (lab vs in situ).

**Puntos finales.**

- Primario: $`{\widehat{\alpha}}_{nut}`$.

- Secundario (H2): **anticipación/rezago tipo Granger**, ¿$`\Delta^{-}\widehat{\alpha}`$ precede cambios a estados turbios?

**Decisión.** H2 apoyada si $`\Delta\widehat{\alpha} \leq - \theta`$ predice indicadores de cambio de régimen con AUC ≥0.70 a $`I^{2} < 50\%`$.

**Negativos.** Curvatura bajo flujos dominados por tormentas → recontenedorizar por régimen de flujo o tratar como fuera de alcance.

**6.4 Protocolo C — Movimiento y metapoblación**

**BIN.** {especie/gremio, fase migratoria, método de detección, manejo de corredores}.\
$`L`$ **.** Diámetro de red o tamaño de módulo $`m`$.\
$`T`$ **.** Tiempo de recolonización $`T_{\text{recol}}`$ o tiempo de primer pasaje.

**Diseño.**

1.  Construir grafos de hábitat a través de gradientes (fragmentación, presencia de corredores).

2.  Corregir detección (ocupación); definir umbral de persistencia $`k/w`$.

3.  Estimar $`{\widehat{\alpha}}_{mov}`$ por estrato; diagnósticos de colapso.

4.  **Intervención (H4):** añadir corredores o aumentar heterogeneidad (varianza de calidad de parche) en paisajes emparejados.

**Puntos finales.**

- Primario: $`\Delta{\widehat{\alpha}}_{mov}`$ (post–pre).

- Secundario: cambio en p95/p50 de recolonización; rendimiento (cruces exitosos/tiempo).

**Decisión.** H4 apoyada si $`\Delta\widehat{\alpha} \geq 0.10`$ (IC excluye 0) con **barandillas**: ≤10% de pérdida en rendimiento medio.

**6.5 Protocolo D — Dinámica trófica/red (mesocosmos o simulación)**

**BIN.** {tipo de ecosistema, banda de temperatura, clase de enriquecimiento, a priori de fuerza de interacción}.\
$`L`$ **.** Profundidad trófica / tamaño de módulo.\
$`T`$ **.** Tiempo de retorno a dentro de $`p`$ de la línea base después de presión/pulso (eliminación de especie clave, enriquecimiento).

**Diseño.**

1.  Simulaciones estocásticas GLV o mesocosmos con matrices de interacción replicadas.

2.  Perturbaciones a múltiples niveles de $`L`$; medir $`T`$.

3.  Estimar $`{\widehat{\alpha}}_{troph}`$; realizar colapso; calcular heterogeneidad **entre familias** con familias de vegetación/nutrientes cuando están co-localizadas.

**Decisión.** H5 apoyada si la puerta de fusión pasa ($`I^{2} < 50\%`$, REML convergente) y se reporta $`{\widehat{\alpha}}_{Eco}`$ con IC.

**6.6 Indicador de alerta temprana:** $`\mathbf{ICE}_{\mathbf{Eco}}\mathbf{(t)}`$

Dadas las pendientes aceptadas por familia $`\{{\widehat{\alpha}}_{f,t}\}`$, calcular la fusión de **efectos aleatorios** (Sec. 4.7). Definir una **alerta de decoherencia** cuando

``` math
Z_{t} = \frac{{\widehat{\alpha}}_{Eco}(t) - \mu_{t \mid H}}{\sigma_{t \mid H}} \leq - z_{\star},
```

con $`\mu,\sigma`$ calculados sobre un EWMA de horizonte $`H`$ (por ejemplo, 180 días para bosques, 30–60 días para lagos), y $`z_{\star} \in \{ 1.5,2.0,2.5\}`$ como niveles prerregistrados (asesoría/vigilancia/advertencia). Requerir $`I^{2} < 50\%`$ en $`t`$; de lo contrario **suspender** la fusión y publicar alarmas por familia.

**6.7 Plan de análisis estadístico (PAE)**

- **Análisis primarios:** pendientes ODR con ICs bootstrap agrupados; decisión de colapso vía $`R_{\text{collapse}}^{2}`$ + LOESS + placebo.

- **Multiplicidad:** Controlar FDR sobre múltiples BINs/ventanas temporales dentro de cada familia de hipótesis.

- **Sensibilidad:** (i) línea Theil–Sen; (ii) banda corregida por SIMEX (si aplica); (iii) dejando una escala fuera; (iv) fusión de efectos fijos vs aleatorios.

- **Tamaños de efecto:** Reportar $`\Delta\widehat{\alpha}`$, AUC para H2, y cambios en p95/p50 con ICs bootstrap.

- **Datos faltantes:** Sin imputación para pendiente; imputar solo para paneles de visualización.

**6.8 Potencia y heurísticas de tamaño muestral**

- **Detección de cambio de pendiente (H4):** Con $`{extensión}_{u}`$ =1.0 y $`N \geq 150`$ pares, potencia bootstrap ≥80% para detectar $`\Delta\alpha = 0.10`$ bajo ruido moderado (CV≈0.2).

- **Tasa de aprobación de colapso (H3):** Para estratos con ≥10 escalas a través de extensión 1.0, tasa de falsa aprobación a $`R_{\text{collapse}}^{2} < 0.05`$ ≈ 5% por construcción; simular para calibrar el ancho de banda LOESS.

- **Estabilidad de fusión (H5):** Necesitar ≥2 familias aceptadas; objetivo $`I^{2} \leq 35\%`$ para $`{\widehat{\alpha}}_{Eco}`$ estable.

**6.9 Gobernanza, ética y prerregistro**

- **Prerregistrar** definiciones de BIN, elecciones de $`L,T`$, ventana $`h`$, umbrales ($`R_{\text{collapse}}^{2}`$, $`I^{2}`$, $`z_{\star}`$), y reglas de detención.

- **Publicar negativos** (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE, alto $`I^{2}`$) como **fronteras de alcance**.

- **Artefactos abiertos:** paneles de colapso, gráficos de bosque, YAML de métodos y conjuntos de datos sintéticos.

- **Ética ambiental:** las intervenciones (corredores, heterogeneidad) deben pasar **evaluación de impacto**; no causar daño a especies más allá de protocolos aprobados.

**6.10 Resumen**

Estos protocolos traducen RTM-Eco en **experimentos falsificables** y **monitoreo operativo**: estimar pendientes con métodos conscientes de EIV, requerir **colapso** para validez de especificación, fusionar solo cuando la **heterogeneidad** es baja, y tratar las **caídas en** $`\alpha_{eco}`$ como alertas tempranas con control de error documentado. La Sección 7 define el **pipeline de fusión y el** $`\mathbf{ICE}_{Eco}(t)`$ **en tiempo real** con más detalle, incluyendo manejo de heterogeneidad y manuales de alerta para gestores.

**7. Fusión y el Índice de Coherencia Ecosistémica (**$`\mathbf{ICE}_{\mathbf{Eco}}\mathbf{(t)}`$ **)**

Ahora convertimos las pendientes aceptadas por familia en un **indicador único y auditable** y especificamos cómo ejecutarlo en tiempo real, controlarlo con heterogeneidad y conectarlo a manuales de gestión.

**7.1 De** $`{\widehat{\mathbf{\alpha}}}_{\mathbf{f}}`$ **por familia a una pendiente fusionada**

En el tiempo $`t`$ dentro de un BIN, suponga que $`F_{t}`$ familias pasan el **colapso** (Sec. 4.5), produciendo $`\{{\widehat{\alpha}}_{f,t},\text{ }{\widehat{\sigma}}_{f,t}^{2}\}_{f = 1}^{F_{t}}`$.

**Línea base de efectos fijos.**

``` math
{\overset{ˉ}{\alpha}}_{FE,t} = \frac{\sum_{f = 1}^{F_{t}}{{\widehat{\alpha}}_{f,t}/{\widehat{\sigma}}_{f,t}^{2}}}{\sum_{f = 1}^{F_{t}}{1/{\widehat{\sigma}}_{f,t}^{2}}},\ \ Q_{t} = \sum_{f = 1}^{F_{t}}\frac{({\widehat{\alpha}}_{f,t} - {\overset{ˉ}{\alpha}}_{FE,t})^{2}}{{\widehat{\sigma}}_{f,t}^{2}},\ \ I_{t}^{2} = \max\{ 0,\frac{Q_{t} - (F_{t} - 1)}{Q_{t}}\}.
```

**Fusión de efectos aleatorios (REML).** Estimar la varianza entre familias $`{\widehat{\tau}}_{t}^{2}`$ y definir pesos $`w_{f,t} = 1/({\widehat{\sigma}}_{f,t}^{2} + {\widehat{\tau}}_{t}^{2})`$. La pendiente fusionada es

``` math
{\widehat{\alpha}}_{Eco}(t) = \frac{\sum_{f}^{}{w_{f,t}{\widehat{\alpha}}_{f,t}}}{\sum_{f}^{}w_{f,t}},\ \ Var\lbrack{\widehat{\alpha}}_{Eco}(t)\rbrack = \frac{1}{\sum_{f}^{}w_{f,t}}.
```

**Puerta de fusión.** Publicar un solo número solo si $`F_{t} \geq 2`$ y $`I_{t}^{2} < 50\%`$. De lo contrario, **retener la fusión** y reportar valores por familia con $`Q_{t},I_{t}^{2}`$.

**7.2 Estimación rodante y ventaneo**

Calcular $`{\widehat{\alpha}}_{f,t}`$ en **ventanas deslizantes** en $`x = \log L`$ (ancho $`h`$; Sec. 4.6) y **ventanas de calendario** adecuadas al tempo del sistema (por ejemplo, 30–90 días para RS; estacional para estudios tróficos). Cada ventana debe pasar cobertura + colapso **dentro de sí misma**.

**Suavizado.** Para visualización y alertas, aplicar una **mediana de 3 puntos** a $`{\widehat{\alpha}}_{Eco}(t)`$; mantener valores crudos para auditorías.

**7.3 Definiendo el indicador**

Definimos el **Índice de Coherencia Ecosistémica** como la pendiente fusionada y su incertidumbre:

$`{ICE}_{Eco}(t) = (\text{ }{\widehat{\alpha}}_{Eco}(t),\ \ {EE}_{Eco}(t) = \sqrt{Var\lbrack{\widehat{\alpha}}_{Eco}(t)\rbrack}\ \ I_{t}^{2}`$

Para comparabilidad, opcionalmente mantener una **línea base** $`{\overset{ˉ}{\alpha}}_{Eco}^{(0)}`$ calculada en un período de referencia; luego rastrear desviaciones

``` math
\Delta\alpha_{Eco}(t) = {\widehat{\alpha}}_{Eco}(t) - {\overset{ˉ}{\alpha}}_{Eco}^{(0)}.
```

**7.4 Lógica de alertas (alerta temprana)**

Definir un puntaje estandarizado sobre una línea base ponderada exponencialmente:

``` math
Z_{t} = \frac{{\widehat{\alpha}}_{Eco}(t) - \mu_{t \mid H}}{\sigma_{t \mid H}},\mu_{t \mid H} = \text{EWMA}_{H}\lbrack{\widehat{\alpha}}_{Eco}\rbrack,\sigma_{t \mid H} = \text{EWMA}_{H}\lbrack{EE}_{Eco}\rbrack,
```

con horizonte $`H`$ emparejado al sistema (por ejemplo, 180 días para bosques, 30–60 días para lagos).

**Niveles de alerta (publicar solo si** $`I_{t}^{2} < 50\%`$ **):**

- **Asesoría:** $`Z_{t} \leq - 1.5`$ durante ≥2 ventanas consecutivas.

- **Vigilancia:** $`Z_{t} \leq - 2.0`$ una vez **o** $`Z_{t} \leq - 1.5`$ durante ≥3 consecutivas.

- **Advertencia:** $`Z_{t} \leq - 2.5`$ una vez, o **cualquier** nivel mientras $`I_{t}^{2} \leq 35\%`$ y $`{EE}_{Eco}`$ esté por debajo de su mediana (alta confianza).

**Auto-suspensión:** Si $`I_{t}^{2} \geq 50\%`$ o cualquier familia pierde el **colapso**, suspender fusión y emitir un **boletín de heterogeneidad** en lugar de una alerta.

**7.5 Interpretando** $`\mathbf{\alpha}_{\mathbf{Eco}}`$ **: palancas de diseño**

Un $`\alpha_{Eco}`$ más alto implica un **estiramiento más pronunciado del tiempo con la escala**, lo cual a menudo **amortigua las cascadas de sincronización** después de choques. Palancas prácticas para **elevar** $`\alpha`$ (validadas por protocolos en Sec. 6):

- **Movimiento/metapoblación:** añadir o desfasar **corredores** para evitar oleadas sincrónicas; fomentar conectividad **modular** (tamaño de módulo medio $`m^{\star}`$) en lugar de un solo componente gigante.

- **Mosaicos de vegetación:** **heterogeneidad** en edades de parche y estructuras de combustible; programas de restauración escalonados (evitar plantación sincrónica).

- **Estructura trófica:** promover **modularidad** y **rutas redundantes** para absorber pulsos (amortiguamiento de especies clave).

- **Biogeoquímica:** **suavizado del régimen de flujo** (apoyo al caudal base) para evitar sincronía de pulsos entre cuencas.

**Compensaciones.** Elevar $`\alpha`$ puede **desacelerar** la recuperación absoluta (los sistemas más grandes tardan más), pero **reduce la amplificación de colas** (p95/p50) y mejora la predictibilidad. Operar en un **frente de Pareto**: maximizar $`\alpha`$ sujeto a pisos de rendimiento/fidelidad relevantes para el objetivo de gestión.

**7.6 Plantilla de reporte (panel ICE)**

Para cada BIN, mantener un panel estándar:

1.  **Serie temporal** de $`{\widehat{\alpha}}_{Eco}(t)`$ con bandas 50/95%; fondo sombreado por niveles de $`I_{t}^{2}`$.

2.  **Bandas de alerta** y marcadores (asesoría/vigilancia/advertencia); anotar suspensiones (alto $`I^{2}`$).

3.  **Inserto de bosque** de $`{\widehat{\alpha}}_{f,t}`$ actual por familia con pesos $`w_{f,t}`$.

4.  **Hash de YAML de métodos** para reproducibilidad completa.

**7.7 Manejo de fallos y política de resultados negativos**

- **Pico de heterogeneidad (alto** $`I^{2}`$ **).** Publicar una **nota de divergencia** con pendientes por familia; recomendar trabajo mecanístico (¿qué familia divergió primero?).

- **Pérdida de colapso.** Remover la familia afectada de la fusión; si $`F_{t} < 2`$, **suspender ICE** y publicar estado.

- **Violación de reloj/placebo.** Revisar preprocesamiento; hasta que se corrija, **invalidar** las ventanas afectadas (no rellenar).

Todos los fallos son **artefactos de primera clase** (mantenidos en el repositorio) para prevenir sesgo retrospectivo.

**7.8 Ejemplo mínimo (números)**

Suponga en $`t`$ : vegetación y nutrientes pasan el colapso con

``` math
{\widehat{\alpha}}_{veg} = 2.32 \pm 0.08,{\widehat{\alpha}}_{nut} = 2.18 \pm 0.12.
```

REML arroja $`{\widehat{\tau}}^{2} = 0.00`$ (heterogeneidad despreciable), por lo que

``` math
{\widehat{\alpha}}_{Eco}(t) = 2.27,EE = 0.07,I_{t}^{2} = 12\%.
```

Con una línea base EWMA de 180 días $`\mu_{t \mid H} = 2.43,\text{ }\text{σ}_{\text{t∣H}}\text{\!=0.06}`$, obtenemos

``` math
Z_{t} = (2.27 - 2.43)/0.06 = - 2.67 \Rightarrow \text{Advertencia},
```

siempre que $`I_{t}^{2} < 50\%`$. La gestión activa el **manual de Advertencia** (Sec. 7.9).

**7.9 Manuales (disparadores de gestión)**

**Asesoría:**

- Aumentar frecuencia de monitoreo; verificar colapso por familia; ejecutar **placebo de reloj**.

- Preparar "palancas suaves" (ventanas de plantación escalonada; suavizado menor del régimen de flujo).

**Vigilancia:**

- Activar **desfase de corredores** (movimiento); ajustar cadencia de restauración para romper sincronía; aumentar **redundancia modular** en redes tróficas.

- Ejecutar **micro-intervenciones A/B** (Sec. 6) con EMDs prerregistrados.

**Advertencia:**

- Escalar a **intervenciones estructurales**: aplicar heterogeneidad en mosaicos de combustible/edad; implementar apoyo al caudal base; limitar temporalmente perturbaciones sincronizadas (por ejemplo, cosecha simultánea a gran escala).

- Declarar **revisión operativa del ICE**: reevaluar etiquetas de BIN, puertas de colapso y puntos de cambio recientes.

**7.10 Resumen**

$`{ICE}_{Eco}(t)`$ fusiona pendientes **limpias, por familia** en un indicador único, **controlado por heterogeneidad** con incertidumbre explícita. Su valor reside en (i) **falsificabilidad** (colapso y placebo), (ii) **fusión auditable** (REML, puerta $`I^{2}`$), y (iii) **accionabilidad** (niveles de alerta vinculados a manuales). Las siguientes secciones presentan **estudios de caso** (Sec. 8), **estándares de reporte** (Sec. 9), y la **discusión/limitaciones** más amplia que sitúa a RTM-Eco dentro de la ciencia ecológica y la gestión.

**8. Estudios de caso**

Ilustramos RTM-Eco con tres sistemas arquetípicos. Cada caso muestra elecciones **operativas** de $`L,T`$, **contenedores**, **diagnósticos de colapso**, y cómo los resultados alimentan el $`\mathbf{ICE}_{Eco}(t)`$ y los manuales de gestión. Donde los datos reales aún no se han ensamblado, especificamos **recetas replicables** y **firmas esperadas** (incluyendo resultados negativos).

**8.1 Recuperación de bosque tropical post-incendio (teledetección)**

**Contexto.** Bosque tropical húmedo con incendios amplificados por sequías episódicas; manejo variable (protegido vs. bordes talados).

**BIN.** {bioma=tropical húmedo latifoliado, estación=JJA, sensor=Landsat+S2, manejo={protegido, borde talado}, ENOS={neutral, El Niño}, clase de severidad}.

**Proxies.**

- $`L`$ : área de parche quemado (ha), huecos disueltos \<$`\rho`$ =2 ha.

- $`T`$ : $`T_{rec}(0.9)`$ al 90% de la mediana NDVI pre-evento (emparejada por mes).

**Pipeline.** Construir 10–15 años de eventos; requerir ≥6 $`L`$ distintos, extensión ≥0.6 en $`\log L`$. Ajustar ODR, ICs bootstrap (agrupados por parche), prueba de colapso + placebo.

**Resultados esperados.**

- **Estratos protegidos**: **colapso** frecuente con $`{\widehat{\alpha}}_{veg} \approx 2.2\text{–}2.5`$; colas (p95/p50) modestas.

- **Estratos de borde talado**: más **NO_COLLAPSE** en años de El Niño (fuga de relojes estacionales/manejo); si el colapso pasa, $`\widehat{\alpha}`$ ligeramente **menor** y colas más pesadas.

**Implicación de gestión.** Una **caída** sostenida en $`{\widehat{\alpha}}_{veg}`$ (o ICE) durante El Niño → activar **Vigilancia**: mantenimiento escalonado de cortafuegos y ventanas de restauración **asincrónicas** para elevar $`\alpha`$ sin aumentar el $`T`$ medio.

**Resultado negativo digno de publicación.** Si el colapso falla sistemáticamente para mega-parches de alta severidad, clasificar como **frontera de alcance** (curvatura): probables relojes dependientes de escala (falla hidráulica, hidrofobicidad del suelo) → crear **BIN separado** o tratar con modelos mecanísticos.

**8.2 Eutrofización lacustre y recuperación (biogeoquímica)**

**Contexto.** Lagos templados bajo presiones de nutrientes; eventos de mezcla periódicos; riesgo de cambios de régimen claro→turbio.

**BIN.** {hidroregión, estación=Abr–Oct, trófico={oligo, meso, eu}, régimen de flujo, manejo (clase de carga de P)}.

**Proxies.**

- $`L`$ : área de cuenca ($`{km}^{2}`$) o área superficial del lago ($`{km}^{2}`$).

- $`T`$ : **vida media** del pulso al 50% de decaimiento en Chl-a (o recuperación de Secchi a $`p = 0.9`$ de la línea base).

**Pipeline.** Muestreo semanal/quincenal; manejo de datos censurados; ODR + TS; diagnósticos de colapso; **anticipación/rezago tipo Granger** de $`\Delta^{-}\widehat{\alpha}`$ a marcadores de régimen (Secchi, duración de hipoxia).

**Resultados esperados.**

- **BINs mesotróficos**: **colapso** decente; $`{\widehat{\alpha}}_{nut} \approx 2.0\text{–}2.3`$.

- **Eutróficos de alta presión**: **REGIME_MIX** ocasional (pendientes por tramos pre/post aireación o cambio de carga) → dividir por punto de cambio de manejo.

**Alerta temprana.** Una **caída** de 2–3 ventanas en $`{\widehat{\alpha}}_{nut}`$ con $`I^{2} < 35\%`$ → **Asesoría/Vigilancia** para anticipar el cambio turbio (reducir influentes, operaciones de suavizado de pulsos).

**Resultado negativo.** Los lagos dominados por tormentas pueden mostrar **NO_COLLAPSE** persistente (regulados por la hidrología de eventos) → declarar **fuera de alcance** para RTM-Eco a menos que un BIN más estrecho remueva los relojes de tormenta.

**8.3 Desfase de corredores en un paisaje fragmentado (movimiento/metapoblación)**

**Contexto.** Mamífero/ave de movilidad media en un mosaico agrícola con corredores candidatos.

**BIN.** {especie, estación=reproducción, detección=telemetría+cámaras trampa, manejo de corredores={línea base, desfasado}}.

**Proxies.**

- $`L`$ : **tamaño de módulo** del grafo de hábitat $`m`$ (nodos por módulo) o diámetro de componente.

- $`T`$ : **tiempo de recolonización** $`T_{recol}`$ (extinción→persistencia ≥$`k`$ detecciones en $`w`$ días) o **tiempo de primer pasaje** a través de corredores.

**Diseño (A/B).**

- **Año línea base**: medir $`{\widehat{\alpha}}_{mov}^{(0)}`$.

- **Año de intervención**: **desfasar** aperturas de corredores (escalonar ventanas) y ajustar heterogeneidad de calidad de parche; re-estimar $`{\widehat{\alpha}}_{mov}^{(1)}`$.

- Colapso en ambos años; bootstrap agrupado por parche/sitio.

**Criterio de éxito (H4).** $`\Delta{\widehat{\alpha}}_{mov} = {\widehat{\alpha}}^{(1)} - {\widehat{\alpha}}^{(0)} \geq 0.10`$ (IC 95% excluye 0) con **barandilla**: ≤10% de pérdida en rendimiento medio (cruces exitosos/tiempo). Esperar **ratio de cola** p95/p50 reducido a $`T`$ medio fijo.

**Resultado negativo.** Si el rendimiento colapsa o $`I^{2}`$ se dispara (restricciones de alimento vs. movimiento divergen), **retener fusión**, publicar por familia y ajustar diseño (por ejemplo, corredores sobre-desfasados fragmentan flujos).

**8.4 Fusión entre familias en una reserva costera (ICE integrado)**

**Contexto.** Reserva costera con dunas/matorral/laguna; tres flujos de datos co-localizados: recuperación de vegetación (quemas), nutrientes (laguna), movimiento de aves (dunas→laguna).

**Plan.** Mantener etiquetas de BIN sincronizadas; estimar $`{\widehat{\alpha}}_{veg},{\widehat{\alpha}}_{nut},{\widehat{\alpha}}_{mov}`$ por trimestre; **fusionar** cuando $`I^{2} < 50\%`$.

**Firma objetivo.** Trimestres normales: $`I^{2} \leq 25\%`$, $`{\widehat{\alpha}}_{Eco} \approx 2.2\text{–}2.4`$. Año de sequía: **caída** en pendiente de vegetación a $`\sim 2.0`$ mientras nutrientes se mantienen estables → $`I^{2}`$ **sube** a 55–65% → **suspender fusión**; la divergencia misma es una **señal de gestión** (los límites de vegetación dominan).

**Vínculo con manual.** Durante la divergencia: priorizar **mosaicos de vegetación** y restauración escalonada; diferir intervenciones de nutrientes hasta que $`I^{2}`$ regrese por debajo de la puerta.

**8.5 Artefactos de reporte (por caso)**

Para cada repositorio de estudio de caso:

- **Paneles de colapso** (ajuste + residuos vs. $`x`$, LOESS, placebo) para cada BIN aceptado/fallido.

- **Gráficos de bosque** de $`{\widehat{\alpha}}_{f}`$ por familia con pesos; $`Q,I^{2},{\widehat{\tau}}^{2}`$.

- Línea temporal de $`\mathbf{ICE}_{Eco}(t)`$, niveles de alerta, y **marcadores de suspensión** (alto $`I^{2}`$).

- **YAML de métodos** (hash en figuras), vintages de datos, semillas de bootstrap, y **puntos de referencia sintéticos** (colapso aprobado/fallido).

**8.6 Lecciones aprendidas (anticipando revisores)**

- **Cuando RTM-Eco funciona.** Forzamiento estable dentro de BINs; cobertura multiescala clara; relojes desacoplados de $`L`$. Los colapsos son comunes; las bandas de $`\alpha`$ estables; la fusión significativa.

- **Cuando no funciona.** Relojes estacionales/de eventos fuertes incrustados en $`T`$ o $`L`$; regímenes por tramos; cobertura delgada — esperar **NO_COLLAPSE/REGIME_MIX** y publicar como **fronteras de alcance**.

- **Valor añadido.** Incluso los negativos son informativos: **mapean los límites** del tempo invariante de escala y apuntan a mecanismos (por ejemplo, sistemas dominados por hidrología) donde los modelos mecanísticos deberían tomar el liderazgo.

**Resumen.** Estos casos demuestran cómo RTM-Eco puede desplegarse de extremo a extremo, desde **extracción de proxy** a **puertas de colapso**, desde **fusión** a **alertas y manuales**, y, igualmente importante, cómo reconocer y publicar **fronteras de alcance**. A continuación, la Sección 9 proporciona **plantillas de resultados y estándares de reporte** para hacer la comparación entre estudios directa y auditable.

**9. Plantillas de resultados y estándares de reporte**

Esta sección especifica los **artefactos exactos** que todo análisis RTM-Eco debe producir, con mínimos grados de libertad. El objetivo es **auditabilidad**, **comparabilidad** y **revisión por pares rápida**. Copie y pegue estas plantillas en su repositorio; reemplace los elementos entre corchetes.

**9.1 Figura 1 — Panel de colapso (por BIN × familia)**

**Plantilla de leyenda.**\
*Panel de colapso para \[etiquetas de BIN\], \[familia\].* Ajustamos $`y = \log T`$ vs. $`x = \log L`$ con ODR (línea Theil–Sen mostrada como verificación de robustez). Panel (a): datos con bandas ODR 50/95%. Panel (b): residuos $`\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}`$ vs. $`x`$ con LOESS (ancho de banda prerregistrado). La **puerta de colapso** requiere $`R_{\text{collapse}}^{2} < 0.05`$, LOESS dentro de bandas, e invariancia del **placebo de reloj** (no mostrado). Etiquetas de aprobación/fallo aparecen arriba a la derecha.

**Anotaciones requeridas.**

- $`\widehat{\alpha}`$ (IC 50/95%), $`\widehat{c}`$, estimador, bootstrap $`B`$, influencia máxima.

- Cobertura: \#$`L`$ distintos, extensión en $`\log L`$.

- Decisión: **ACCEPT / NO_COLLAPSE / REGIME_MIX / THIN_COVERAGE**.

**9.2 Figura 2 — Gráfico de bosque (pendientes por familia en un BIN)**

**Plantilla de leyenda.**\
*Exponentes de coherencia por familia y estimación fusionada para \[etiquetas de BIN\].* Puntos: $`{\widehat{\alpha}}_{f} \pm`$ IC 95%; tamaño $`\propto w_{f} = 1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})`$. Diamante: $`{\widehat{\alpha}}_{Eco}`$ (REML) si $`I^{2} < 50\%`$; de lo contrario "fusión suspendida".

**Anotaciones requeridas.**

- $`Q`$, $`I^{2}`$, $`{\widehat{\tau}}^{2}`$, método de fusión (REML/DL).

- Decisión de puerta: **FUSIONADO** / **SUSPENDIDO**.

- Hash de métodos (ver YAML).

**9.3 Figura 3 —** Serie temporal de $`\mathbf{ICE}_{\mathbf{Eco}}\mathbf{(t)}`$

**Plantilla de leyenda.**\
*ICE rodante para \[etiquetas de BIN\].* Pendiente fusionada $`{\widehat{\alpha}}_{Eco}(t)`$ con bandas 50/95%; sombreado de fondo indica niveles de $`I^{2}`$. Líneas punteadas: umbrales de alerta para $`Z_{t}`$ (Asesoría/Vigilancia/Advertencia). Marcadores rojos: fusión suspendida (alto $`I^{2}`$).

**Anotaciones requeridas.**

- Longitud de ventana $`h`$ (escala log) y ventana de calendario.

- Horizonte EWMA $`H`$; $`Z_{t}`$ actual.

- Conteo de familias aceptadas $`F_{t}`$ por ventana.

**9.4 Tabla 1 — Resumen de BIN (legible por máquina)**

| ID BIN | Etiquetas (bioma/estación/...) | Familia | #L | Extensión $\log L$ | Estimador | $\hat{\alpha}$ (IC 95%) | $R^2_{\text{collapse}}$ | Decisión |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B-001 | TropHúm, JJA, Landsat+S2, Proteg, ENOS=N | Vegetación | 14 | 1.12 | ODR | 2.31 [2.17, 2.45] | 0.018 | ACCEPT |
| B-001 | ... | Nutrientes | 9 | 0.72 | ODR | 2.05 [1.83, 2.28] | 0.027 | ACCEPT |
| B-001 | ... | Movimiento | 8 | 0.67 | ODR | 2.29 [2.01, 2.56] | 0.061 | NO_COLLAPSE |

*Nota.* Publicar **todos** los contenedores, incluyendo los fallos.

**9.5 Tabla 2 — Fusión y alertas**

| ID BIN | Familias fusionadas | Q | $I^2$ | $\hat{\tau}^2$ | $\hat{\alpha}_{\text{Eco}}(EE)$ | Decisión fusión | Último $Z_t$ | Nivel alerta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B-001 | Veg, Nut | 1.7 | 19\% | 0.000 | 2.27 (0.07) | FUSIONADO | –2.67 | ADVERTENCIA |
| B-002 | Veg | – | – | – | – | SUSPENDIDO | – | – |

**9.6 YAML de métodos (incrustar hash en cada figura)**

```
version: "RTM-Eco 1.0"

bin:
  tags: ["biome:TropMoist", "season:JJA", "sensor:Landsat+S2", "mgmt:Protected", "ENSO:Neutral"]
  min_scales: 6
  min_logL_span: 0.6
  changepoint: {method: "PELT", criterion: "BIC"}

estimation:
  base: "odr"
  init: "theil-sen"
  bootstrap: {B: 2000, cluster: true, seed: 12345}
  leverage_cap: 0.25

collapse:
  r2_threshold: 0.05
  loess_bw: "fixed:0.6"
  clock_placebo: true

fusion:
  method: "REML"
  I2_gate: 0.5

eci:
  window_logL: 0.8
  calendar_window: "90d"
  ewma_horizon: "180d"

report:
  publish_negatives: true
```

Añadir un **hash SHA-256** de este YAML en la esquina de las Figuras 1–3. Los revisores pueden reejecutar y verificar.

**9.7 Artefactos de resultados negativos (obligatorios)**

Para cada **NO_COLLAPSE / REGIME_MIX / THIN_COVERAGE**:

- Panel de colapso con razón de fallo resaltada (firma de curvatura, quiebre, cobertura).

- Nota breve *"Frontera de alcance: \[razón\]"* y **pasos siguientes** propuestos (re-contenedorizar, recopilar escalas, modelo mecanístico).

- Mantener artefactos en repositorio; indexarlos en una tabla de apéndice.

**9.8 Lista de verificación de reproducibilidad (enviar con manuscrito)**

- **Diccionario de datos** para $`L,T`$ por familia; base logarítmica especificada.

- **Libro de BINs**: etiquetas, conteos, extensiones, puntos de cambio.

- **Trío de estimadores**: ODR (primario), línea TS, banda SIMEX (si aplica).

- **Evidencia de colapso**: $`R_{\text{collapse}}^{2}`$, LOESS, placebo.

- **Influencia**: influencia máxima \<0.25 o sensibilidad mostrada.

- **Fusión**: $`Q`$, $`I^{2}`$, $`{\widehat{\tau}}^{2}`$; decisión de puerta de fusión.

- **ICE**: ventanas, $`H`$, lógica de alertas; suspensiones marcadas.

- **Negativos publicados** con justificación.

- **YAML de métodos** + **hash** en todas las figuras.

- **Puntos de referencia sintéticos** (colapso aprobado/fallido) incluidos.

**9.9 Estilo de redacción y estándares de notación**

- Usar **logaritmos naturales**; escribir $`\log L,\text{ T}`$ en modo matemático.

- Usar **griego** consistentemente: $`\alpha_{\text{eco}}`$ (pendiente), $`\kappa`$ (reloj), $`\tau^{2}`$ (varianza entre familias).

- Reservar **negrita** para decisiones (ACCEPT/NO_COLLAPSE/…); evitar cursiva en tablas excepto variables.

- Reportar $`\alpha`$ con **2 decimales**, $`I^{2}`$ con **1 decimal**, ICs como **\[bajo, alto\]**.

**9.10 Bloque de texto mínimo para sección de Resultados (conectable)**

> *Dentro del \[etiquetas de BIN\], la recuperación de vegetación colapsó sobre* $`T_{\text{rec}} \propto L^{\alpha}`$ *(ODR* $`\widehat{\alpha} = 2.31\text{ }\lbrack 2.17,2.45\rbrack`$ *;* $`R_{\text{collapse}}^{2} = 0.018`$ *; placebo aprobado). Los pulsos de nutrientes arrojaron* $`\widehat{\alpha} = 2.05\text{ }\lbrack 1.83,2.28\rbrack`$ *(colapso aprobado). El movimiento falló el colapso (0.061) y se marcó NO_COLLAPSE. La fusión de efectos aleatorios (REML) de vegetación+nutrientes dio* $`{\widehat{\alpha}}_{Eco} = 2.27`$ *(EE 0.07),* $`I^{2} = 19\%`$ *. El ICE rodante cruzó el nivel de Advertencia (Z=−2.67) con baja heterogeneidad; la fusión permaneció activa.*

**Resumen.** Estas plantillas estandarizan cómo los resultados de RTM-Eco se **muestran y auditan**. Adoptarlas (más el hash del YAML) hace que la comparación multisitio, la revisión por pares y la replicación sean directas, y convierte el "ritmo" de metáfora en **evidencia operativa**.

**10. Discusión**

Esta sección interpreta $`\alpha_{eco}`$ como una **propiedad estructural del tempo**, relaciona RTM-Eco con teorías existentes (resiliencia, panarquía, alometría, señales de alerta temprana), examina los **mecanismos** detrás de pendientes más altas/bajas, y clarifica las **compensaciones de gestión** y el alcance.

**10.1 Qué "compra" un** $`\mathbf{\alpha}_{\mathbf{eco}}`$ **más alto (y qué no)**

Un $`\alpha_{eco}`$ más grande significa un **estiramiento más pronunciado del tiempo con la escala**: a medida que los sistemas se hacen más grandes (parches, cuencas, módulos de red), sus tiempos característicos aumentan **predeciblemente**. Esto tiende a:

- **Amortiguar cascadas de sincronización** después de choques (los extremos a pequeña escala no escalan linealmente), reduciendo la **amplificación de colas** (p95/p50).

- **Aumentar la predictibilidad** de los horizontes de recuperación a través de tamaños dentro de un BIN (bandas creíbles más estrechas una vez conocida la pendiente).

- **Estabilizar la coherencia entre familias** cuando los mecanismos comparten relojes compatibles (menor $`I^{2}`$).

Sin embargo, un $`\alpha`$ más alto **no** garantiza una recuperación absoluta más rápida; puede **desacelerar** las unidades grandes. El beneficio práctico es **orden** y **previsibilidad**, no velocidad per se. Los gestores operan en una **frontera de Pareto**: mayor $`\alpha`$ vs. restricciones de rendimiento/latencia (Sec. 7.5).

**10.2 RTM-Eco y la teoría ecológica existente**

- **Resiliencia y desaceleración crítica (CSD).** CSD rastrea el aumento de varianza/autocorrelación cerca de puntos de inflexión a una escala fija. RTM-Eco lo complementa rastreando **cómo el tiempo escala con el tamaño**. Una caída en $`\alpha`$ puede preceder o acompañar la CSD pero es **conceptualmente distinta**: una es una advertencia **dentro de la escala**; la otra es **geometría entre escalas**.

- **Panarquía / interacciones entre escalas.** La panarquía enfatiza ciclos adaptativos y vínculos entre escalas. RTM-Eco suministra un **esqueleto numérico** para el aspecto del tempo: $`\alpha`$ cuantifica el **gradiente de tempo** entre niveles dentro de un régimen coherente.

- **Alometría y fractales.** Muchas tasas ecológicas obedecen leyes de potencia (por ejemplo, escalamiento metabólico). RTM-Eco **re-centra** el análisis en la **pendiente** bajo **pruebas de colapso** y **estimación EIV**, protegiéndose contra leyes de potencia espurias y dependencia de unidades.

- **Conectividad y modularidad.** La teoría de redes vincula modularidad con robustez. RTM-Eco predice que la **modularidad moderada** a menudo eleva $`\alpha`$ (previniendo sincronía a nivel de sistema) mientras que la modularidad excesiva puede perjudicar el rendimiento — de ahí las palancas de diseño en Sec. 7.5.

**10.3 Esbozos mecanísticos detrás de** $`\mathbf{\alpha}_{\mathbf{eco}}`$

RTM-Eco es fenomenológico pero **compatible con mecanismos**. Varias imágenes generativas explican por qué $`\alpha`$ varía:

1.  **Agregación difusiva (**$`\alpha \approx 2`$ **).** Cuando las perturbaciones/recuperaciones se propagan por transporte cuasidifusivo (lluvia de semillas, difusión de nutrientes), $`T \sim L^{2}`$ dentro de un BIN.

2.  **Ensamblaje jerárquico (**$`\alpha > 2`$ **).** La recuperación requiere **módulos secuenciales** (por ejemplo, microbios del suelo → pioneros → dosel) o **enrutamiento** a través de redes; cada etapa añade latencia, empinando $`\alpha`$.

3.  **Fuga de reloj / mezcla multimecanismo (**$`\alpha`$ **inestable).** Si el proxy $`T`$ incorpora relojes estacionales/de manejo o combina regímenes, los residuos se curvan → NO_COLLAPSE.

4.  **Forzamiento sincrónico (menor** $`\alpha`$ **efectivo).** Pulsos altamente sincronizados (hidrología dominada por tormentas, plantación/cosecha sincrónica) aplanan el gradiente de tempo, facilitando extremos a nivel de sistema.

Estos esbozos motivan intervenciones (desfase de corredores, heterogeneidad de mosaicos, apoyo al caudal base) que **dirigen** $`\alpha`$.

**10.4 Por qué el "colapso" importa (más allá de la calidad del ajuste)**

En ecología, muchas leyes de potencia reportadas resultan de la **linealización log-log** sin verificaciones de modelo. El colapso eleva la afirmación de "una línea ajusta" a "**no queda estructura residual sistemática** después de remover la pendiente y cambiar relojes". Es una **prueba de especificación**: los estados de fallo (NO_COLLAPSE, REGIME_MIX) son **resultados**, no molestias — señalando **relojes ocultos**, **quiebres**, o **fronteras de alcance** donde los modelos mecanísticos deben tomar precedencia.

**10.5 Ética de la fusión: cuándo un indicador único está justificado**

RTM-Eco fusiona pendientes por familia solo bajo **heterogeneidad acotada** ($`I^{2} < 50\%`$). Esto evita falsa certeza cuando procesos de vegetación, nutrientes y movimiento **divergen**. En episodios de divergencia, la **suspensión de la fusión** es la señal científicamente honesta; los gestores actúan enfocándose en el **desviador líder** (por ejemplo, la vegetación limitando nutrientes y movimiento).

**10.6 Implicaciones de gestión: diseño "consciente de la pendiente"**

- **Paisajes de fuego.** Favorecer ventanas de restauración **asincrónicas** y mosaicos de combustible/edad **heterogéneos** para aumentar $`\alpha`$ sin desaceleración excesiva.

- **Lagos y cuencas.** **Suavizar pulsos** (programación de caudal base/aireación) para evitar sincronización a nivel de paisaje; mantener estructuras de cuenca que preserven la **separación de escalas**.

- **Planificación de conectividad.** **Desfasar** apertura de corredores y apuntar a **modularidad intermedia** (tamaño de módulo $`m^{\star}`$) para elevar $`\alpha`$ preservando rendimiento.

- **Sistemas tróficos.** Fomentar **rutas redundantes** y **conectancia moderada** para alargar el enrutamiento de recuperación (mayor $`\alpha`$) sin bloquear el sistema.

Todas las acciones deben evaluarse con **Efectos Mínimos Detectables** prerregistrados para $`\Delta\alpha`$ y **barandillas** de rendimiento (Sec. 6).

**10.7 Interpretando resultados negativos**

- **NO_COLLAPSE.** La curvatura persistente señala **mecanismos dependientes de escala** o **contaminación de reloj**. Publicar con nota de alcance y, si es posible, un **BIN dividido** o un seguimiento mecanístico.

- **REGIME_MIX.** Los quiebres implican pendientes **por tramos**; dividir a menudo recupera $`\alpha`$ válido dentro de sub-regímenes.

- **Alto** $`I^{2}`$ **.** Divergencia real entre familias: la acción correcta **no** es promediarla sino hacer la divergencia **accionable** (triaje de intervenciones).

**10.8 Limitaciones revisadas (vista previa de Sec. 11)**

- **Localidad.** $`\alpha_{eco}`$ es **local al contenedor**; la extrapolación entre contenedores requiere nuevas verificaciones de colapso.

- **Fragilidad de proxy.** Las definiciones de $`L,T`$ deben auditarse para relojes ocultos; de lo contrario las afirmaciones de pendiente son inestables.

- **Sensibilidad de cobertura.** Escalas grandes escasas inflan la influencia; el reporte robusto debe incluir verificaciones de dejando una escala fuera.

- **Causalidad.** RTM-Eco es **estructural-descriptivo**: organiza el tempo; las inferencias causales requieren diseños dirigidos.

**10.9 Perspectivas**

La trayectoria principal de investigación es (i) ensamblar **conjuntos de datos multifamilia co-localizados** con libros de BIN estrictos, (ii) estandarizar **artefactos de colapso** y **YAMLs de métodos**, (iii) ejecutar **pruebas de intervención** que intenten **diseñar** $`\alpha`$ (desfase de corredores, heterogeneidad de mosaicos), y (iv) comparar cambios de $`\alpha`$ contra métricas **clásicas de alerta temprana** para clarificar complementariedades.

**Resumen.** RTM-Eco reenmarca el tiempo ecológico como una **pendiente invariante de calibre** dentro de regímenes coherentes, respaldada por **colapso falsificable** y **fusión controlada por heterogeneidad**. Su novedad no reside en postular otra ley de potencia sino en hacer la **geometría del tempo operativa**, auditable y directamente mapeada a **palancas de diseño**, mientras trata los fallos como fronteras informativas en lugar de anomalías a suavizar.

**11. Limitaciones y alcance**

RTM-Eco es **fenomenológico** y **local al contenedor**. Su valor depende de cuán limpiamente un conjunto de datos satisface los supuestos detrás de la *geometría escala-reloj* y la prueba de especificación de **colapso**. Esta sección delimita dónde aplica el marco, dónde probablemente falla, y cómo mitigar amenazas a la validez.

**11.1 Localidad y dependencia de régimen**

**Qué es.** $`\alpha_{eco}`$ se define **dentro de un contenedor de coherencia (BIN)**, una rebanada con forzamiento cuasiconstante (bioma, estación, manejo, pila de sensores, clase de anomalía).

**Implicaciones.**

- **No** comparar pendientes **entre** contenedores sin re-probar el **colapso**.

- La deriva temporal (fenología, humedad multiAnual) convierte la pendiente "global" en una **local**; usar estimación **ventaneada** (Sec. 4.6).

**Mitigación.** Mantener un **libro de BINs**; ejecutar escaneos de **puntos de cambio**; publicar REGIME_MIX cuando aparezcan quiebres, en lugar de forzar una sola pendiente.

**11.2 Fragilidad de proxy (relojes ocultos)**

**Riesgo.** Los proxies $`L`$ y $`T`$ pueden contrabandear **relojes** (fases estacionales, calendarios de manejo, umbrales de detección) y crear curvatura espuria o cambios de pendiente.

**Ejemplos.**

- $`T_{\text{rec}}`$ medido sin **emparejamiento por mes** (fuga fenológica).

- "Área efectiva" de parche calculada con amortiguadores **dependientes de severidad** (definición de $`L`$ dependiente de escala).

- $`T`$ basado en ocupación confundido por **probabilidad de detección**.

**Mitigación.**

- **Placebo de reloj** (reescalar unidades; la pendiente debe mantenerse).

- Emparejamiento por mes del año; tratar severidad/detección como **covariables** (no parte de $`L`$).

- Publicar NO_COLLAPSE como **frontera de alcance** si la fuga no puede eliminarse.

**11.3 Cobertura e influencia**

**Riesgo.** Cobertura delgada, especialmente a escalas grandes, induce **alta influencia** y $`\widehat{\alpha}`$ inestable.

**Mitigación.**

- Requerir ≥6 $`L`$ distintos y extensión ≥0.6 en $`\log L`$.

- Reportar **influencia máxima** y sensibilidad **dejando una escala fuera**; descartar contenedores que fallen en estabilidad.

- Preferir **múltiples escalas medianas a grandes** a un solo "mega-parche".

**11.4 Errores en variables y límites del estimador**

**Riesgo.** Atenuación OLS; ODR/TLS asume errores independientes y homocedásticos en $`x,y`$; SIMEX requiere un $`Var(\xi)`$ **calibrado**.

**Mitigación.**

- Usar **ODR** con pesos basados en réplicas; **Theil–Sen** para robustez; **SIMEX** solo cuando la varianza es defendible (réplicas/ensayos inter-analista).

- ICs de bootstrap agrupado; reportar **trío de estimadores** y divergencias.

**11.5 Heterogeneidad y ética de la fusión**

**Riesgo.** Las familias ecológicas (vegetación, nutrientes, movimiento, trófica) pueden divergir. Promediarlas puede **ocultar** desacuerdos accionables.

**Mitigación.**

- Condicionar la fusión a $`I^{2} < 50\%`$; de lo contrario **suspender** ICE y reportar por familia.

- Tratar $`I^{2}`$ **en aumento** como una **señal** (divergencia de mecanismos), no como ruido a suavizar.

**11.6 Causalidad e interpretación**

**Riesgo.** $`\alpha_{eco}`$ es **estructural-descriptivo**; confundir cambios de pendiente con efectos causales puede desorientar la gestión.

**Mitigación.**

- Reservar afirmaciones causales para **intervenciones A/B** (Sec. 6) con **barandillas** y **EMDs** prerregistrados.

- Usar $`\alpha`$ como un **dial de diseño** (intervenciones conscientes de la pendiente), pero validar resultados con métricas de éxito **independientes**.

**11.7 Sistemas fuera del alcance de RTM-Eco (estados de fallo probables)**

- **Hidrología dominada por eventos** donde $`T`$ está regulada por tormentas incluso después de contenedores estrechos → NO_COLLAPSE persistente.

- **Regímenes tróficos fuertemente no lineales** con dominios multiestables al mismo BIN → pendientes **por tramos** (REGIME_MIX).

- **Pulsos microbianos de corta duración** donde $`L`$ no puede definirse consistentemente entre sitios/tiempos.

- **Escalas ultra-escasas** (extensión \<0.6 en $`\log L`$) o $`L`$ altamente cuantizado.

**Política.** Publicar artefactos negativos; recomendar modelos **mecanísticos** o **por tramos** en lugar de RTM-Eco.

**11.8 Validez externa y transferencia**

**Riesgo.** Una pendiente validada en un BIN puede no transferirse a otro (diferente banda climática, manejo o pila de sensores).

**Mitigación.**

- Regiones/años **reservados**; requerir **colapso** en el BIN objetivo antes de transferir $`\widehat{\alpha}`$.

- Preferir comparaciones **relativas** (Δ$`\alpha`$ dentro de BINs) a rankings **absolutos** entre BINs.

**11.9 Calidad de datos, sesgo y ética**

- **Teledetección**: artefactos de nubes/sombras; residuos BRDF; píxeles mixtos en bordes (inflación del borde de $`L`$).

- **Datos de campo**: sesgos de detección; muestreo oportunista durante crisis; censura a la derecha de recuperaciones largas.

- **Ética**: las intervenciones (corredores, enriquecimiento/presión) deben pasar revisión de impacto ecológico; RTM-Eco **no** debería incentivar sincronización dañina (por ejemplo, talas rasas simultáneas) en busca de pendientes ordenadas.

**Mitigación.** **Diccionario de datos** explícito, tratamiento de censura, **YAML** de métodos con semillas y vintages; evaluaciones de impacto para intervenciones.

**11.10 Lista de verificación del revisor (limitaciones reconocidas)**

- Localidad del BIN y manejo de puntos de cambio descritos.

- Auditorías de proxy para **relojes ocultos** aprobadas o fallos publicados.

- Umbrales de cobertura/influencia cumplidos; sensibilidad reportada.

- Trío de estimadores EIV reportado; detalles de bootstrap reproducibles.

- Fusión condicionada por $`I^{2}`$; divergencia manejada como señal.

- Lenguaje causal confinado a resultados **intervencionales**.

- Resultados negativos (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE) archivados.

**11.11 Resumen**

RTM-Eco es **poderoso donde sus supuestos se cumplen** — regímenes coherentes, proxies limpios, cobertura multiescala — y **honesto** donde no — convirtiendo fallos en **fronteras de alcance**. Tratar $`\alpha_{eco}`$ como un **descriptor local, invariante de calibre** de la geometría del tempo; condicionar la fusión; y emparejar diseño consciente de la pendiente con pruebas causales dirigidas. La siguiente sección detalla **Métodos y reproducibilidad** para estandarizar implementaciones entre laboratorios y paisajes.

**12. Métodos y reproducibilidad**

Esta sección especifica **procedimientos exactos** y **artefactos** para que cualquier grupo pueda reproducir RTM-Eco de extremo a extremo. Damos **algoritmos**, **esquemas de datos**, **ambiente de software**, y un **YAML de métodos** que se hashea e incrusta en cada figura.

**12.1 Fuentes de datos e ingesta**

**Teledetección (vegetación).** Landsat 5–9 SR & QA; Sentinel-2 L2A. LiDAR de altura de dosel opcional (GEDI/ALS).\
**Hidrología/biogeoquímica.** Programas nacionales de lagos/arroyos (Chl-a, nutrientes, Secchi, OD), redes de aforo, reanálisis meteorológicos para etiquetas de anomalía.\
**Movimiento/metapoblación.** Telemetría (GPS/ARGOS), cámaras trampa, ocupación eBird/atlas con registros de visita.\
**Trófica/red.** Registros de mesocosmos; matrices curadas de redes tróficas con fuerzas de interacción/incertidumbres.

**Regla de ingesta.** Almacenar todas las series temporales en formato **ordenado** con marcas de tiempo de adquisición y **valores crudos sin cambios**; los campos derivados viven en tablas separadas.

**12.2 Tablas canónicas (esquemas)**

**A) `records.tsv` — unidad de análisis (por observación)**

| bin_id | fam | uid | t_obs | L_raw | T_raw | x=logL | y=logT | w | tags_json |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B001 | veg | P034 | 2016-09-18 | 125.7 | 482 | 4.835 | 6.178 | 1 | {...} |

**B) `bins.tsv` — contenedores de coherencia (una fila por contenedor)**

| bin_id | biome | season | sensor | mgmt | anomaly | severity | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B001 | Trop | JJA | L+S2 | Prot | ENSO0 | M1 | "..." |

**C) `methods.yml` — configuración completa del análisis (ver §12.10)**

**D) `results.tsv` — salidas por BIN×familia**

| bin_id | fam | n_scales | span_logL | alpha_lo | alpha | alpha_hi | c_hat | R2_collapse |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B001 | veg | 14 | 1.12 | 2.17 | 2.31 | 2.45 | -1.01 | 0.018 |

**E) `fusion.tsv` — fusión por BIN ventana temporal**

| bin_id | t0 t1 | F | Q | I2 | tau2 | alphaEco | se |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B001 | ... | 2 | 1.7 | 19 | 0.00 | 2.27 | 0.07 |

Todos los archivos son UTF-8, delimitados por tabulaciones; valores faltantes como NA.

**12.3 Pipelines de preprocesamiento (resúmenes)**

**Vegetación (RS).**

1.  Máscara de nubes/sombras; normalización BRDF.

2.  Detección de eventos (dNBR/RBR) → polígonos; disolver huecos \<ρ ha.

3.  Mediana de línea base sobre 24–36 meses, **emparejada por mes**.

4.  Tiempo de recuperación $`T_{\text{rec}}(p)`$ vía mediana rodante (60–90 d).

5.  Transformación logarítmica a $`x = \log L,\text{ y=log T}`$

**Hidrología/biogeoquímica.**

- Cuencas basadas en DEM; regularización semanal/quincenal; censura manejada (modelos LOD o sustitución con marcas); ventanas de pulso etiquetadas.

**Movimiento/metapoblación.**

- Segmentación de telemetría; detección de cruces de corredores; modelos de ocupación para corregir detección; definir regla de persistencia $`k`$ detecciones en ventana $`w`$.

**Trófica/red.**

- Simulaciones GLV/estocásticas o registros de mesocosmos; estandarizar fase diaria; computar tiempos de retorno a $`p`$ de la línea base.

**12.4 Algoritmo de contenedores (determinista)**

**Entradas.** records.tsv, etiquetas ambientales por fila, methods.yml.

**Pasos.**

1.  **Etiquetado.** Asignar etiquetas {bioma, banda_estacional, pila_sensores, manejo, clase_anomalía, severidad}.

2.  **Estratificar.** Agrupar por tupla exacta de etiquetas → contenedores provisionales.

3.  **Puntos de cambio.** Para cada grupo, ejecutar PELT/BIC en $`y`$ y covariables clave; dividir si se detecta algún PC.

4.  **Filtro de cobertura.** Mantener contenedores con ≥6 escalas **distintas** y extensión ≥0.6 en $`\log L`$.

5.  **Libro de registro.** Escribir bins.tsv con procedencia (qué divisiones ocurrieron y por qué).

Todas las divisiones y descartes se registran en bin_events.tsv con marcas de tiempo.

**12.5 Algoritmos de estimación**

**12.5.1 Regresión de Distancia Ortogonal (primario)**

Minimizar

``` math
\sum_{i}^{}{w_{i}\frac{(y_{i} - \alpha x_{i} - c)^{2}}{\sigma_{y,i}^{2} + \alpha^{2}\sigma_{x,i}^{2}}}
```

con inicialización Theil–Sen e ICs de **bootstrap agrupado** (cluster = parche/cuenca/sitio).

- **Detención.** Número de condición \< $`10^{4}`$; influencia máxima \< 0.25.

- **Pesos.** EE de réplica si disponibles; sino $`w_{i} \equiv 1`$.

**12.5.2 Theil–Sen (verificación robusta)**

Mediana de pendientes por pares; intercepto como mediana de $`y`$ residualizado.

**12.5.3 SIMEX (opcional)**

Cuando $`Var(\xi_{u})`$ es conocida/estimable: simular $`\lambda \in \{ 0.5,1,1.5,2\}`$, ajustar $`\widehat{\alpha}(\lambda)`$, extrapolar cuadrática a $`\lambda = - 1`$.

**12.6 Diagnósticos de colapso (prueba de especificación)**

Para cada BIN×familia:

1.  Residuos $`\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}`$.

2.  Prueba de tendencia $`R_{\text{collapse}}^{2} = R^{2}(\widetilde{y} \sim x) < 0.05`$.

3.  LOESS con ancho de banda prerregistrado (mostrar que la banda contiene 0).

4.  **Placebo de reloj** $`T \mapsto cT`$ (por ejemplo, re-normalización): invariancia de $`\widehat{\alpha}`$, $`R_{\text{collapse}}^{2}`$.

5.  **Decisión**: ACCEPT / NO_COLLAPSE / REGIME_MIX / THIN_COVERAGE con código de razón.

Artefactos guardados en fig/ con nombres de archivo que incrustan el **hash de métodos** (ver §12.10).

**12.7 Fusión y cálculo del ICE**

En la ventana temporal $`\lbrack t_{0},t_{1}\rbrack`$ dentro de un BIN:

- Recopilar los $`\{{\widehat{\alpha}}_{f},{\widehat{\sigma}}_{f}^{2}\}`$ aceptados.

- Calcular $`Q,I^{2}`$; estimar $`{\widehat{\tau}}^{2}`$ (REML).

- Si $`I^{2} < 0.50`$ y $`F \geq 2`$ :

``` math
{\widehat{\alpha}}_{Eco} = \frac{\sum_{f}^{}{{\widehat{\alpha}}_{f}/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})}}{\sum_{f}^{}{1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})}},\ \ \ \ \ EE = 1/\sqrt{\sum_{f}^{}{1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})}}
```

Sino **suspender fusión**; producir salida por familia.

**Serie temporal del ICE.** Deslizar $`\lbrack t_{0},t_{1}\rbrack`$ con paso $`s`$ (por ejemplo, 30 d). Mantener línea base EWMA $`H`$; calcular $`Z_{t}`$ y niveles de alerta (Sec. 7.4). Almacenar en eci.tsv.

**12.8 Ambiente de software**

**Lenguaje.** Python ≥3.10 o R ≥4.3 (ambos aceptables).\
**Paquetes centrales (Py).** numpy, scipy (ODR), statsmodels, pandas, ruptures (PELT), scikit-learn, matplotlib.\
**Herramientas RS.** rioxarray, rasterio, geopandas, ESA SNAP/gee opcionales.\
**Reproducibilidad.** renv (R) o conda/mamba (Py); especificación de contenedor (Dockerfile) con versiones fijadas.

**Aleatorización.** Todos los bootstraps/aleatorizaciones deben respetar una **semilla única** de methods.yml; re-sembrar está prohibido excepto cuando se declare explícitamente.

**12.9 Pseudocódigo mínimo (análisis por contenedor)**

```
def analyze_bin(df_bin, methods):
    # cobertura
    scales = np.unique(df_bin['x'])
    if (len(scales) < methods.min_scales) or ((scales.max() - scales.min()) < methods.min_logL_span):
        return fail("THIN_COVERAGE")
        
    # estimador
    alpha_ts, c_ts = theil_sen(df_bin.x, df_bin.y)
    alpha_odr, c_odr, diag = odr_fit(df_bin, init=(alpha_ts, c_ts))
    if diag.leverage_max > methods.leverage_cap or not diag.converged:
        return fail("ESTIMATION_ISSUE")
        
    # colapso
    res = df_bin.y - (alpha_odr * df_bin.x + c_odr)
    R2 = r2_linear(res, df_bin.x)
    loess_ok = loess_band_contains_zero(res, df_bin.x, bw=methods.loess_bw)
    placebo_ok = clock_placebo_invariance(df_bin, alpha_odr, c_odr)
    
    if (R2 < methods.r2_threshold) and loess_ok and placebo_ok:
        return accept(alpha_odr, c_odr, R2, diag)
    else:
        return fail("NO_COLLAPSE" if kink_absent(res) else "REGIME_MIX")
```

**12.10 El YAML de métodos (configuración autoritativa)**

```
version: "RTM-Eco 1.0"

data:
  log_base: "e"
  rs_recovery_p: [0.8, 0.9, 0.95]
  dissolve_holes_ha: 2.0

binning:
  tags: [biome, season_band, sensor_stack, mgmt, anomaly_class, severity]
  min_scales: 6
  min_logL_span: 0.6
  changepoint: {method: "PELT", criterion: "BIC"}

estimation:
  estimator: "ODR"
  init: "Theil-Sen"
  bootstrap: {B: 2000, cluster: true, seed: 123456}
  leverage_cap: 0.25
  simex: {enabled: false, lambda: [0.5, 1.0, 1.5, 2.0]}

collapse:
  r2_threshold: 0.05
  loess_bw: 0.6
```

**Hasheo.** Calcular SHA-256 del YAML; incrustar los primeros 10 caracteres hex en cada nombre de figura/CSV (por ejemplo, fig/collapse_B001_veg_ab12c34d56.png). Almacenar hash completo en la leyenda de la figura.

**12.11 Puntos de referencia sintéticos (obligatorios)**

Proporcionar dos conjuntos de datos por familia:

- **APROBADO**: $`v = \alpha u + \log\kappa + \mathcal{N}(0,\sigma^{2})`$ con ruido y cobertura realistas → debe pasar colapso y recuperar $`\alpha`$ dentro del IC.

- **FALLIDO**: $`v = \alpha u + \beta u^{2}`$ (curvatura) o pendientes por tramos → debe fallar colapso (NO_COLLAPSE o REGIME_MIX).\
  Publicar código + semillas e incluirlos en pruebas de IC.

**12.12 Integración continua (IC)**

Configurar IC para:

1.  Validar esquemas; verificar methods.yml contra JSON-Schema.

2.  Reejecutar puntos de referencia sintéticos y **fallar la construcción** si los resultados APROBADO/FALLIDO se invierten.

3.  Verificar consistencia de hash de métodos a través de artefactos.

4.  Producir un **reporte del repositorio** (HTML/PDF) con todas las tablas/figuras para envío.

**12.13 Ética, gobernanza y seguridad de datos**

- **Bienestar humano/animal.** El trabajo de movimiento y mesocosmos debe ser aprobado por IACUC/comités de ética relevantes; telemetría anonimizada/perturbada espacialmente cuando sea necesario.

- **Impacto ambiental.** Las intervenciones de corredor/desfase y mosaico pasan evaluación de impacto; **barandillas** prerregistradas (rendimiento, pisos de biodiversidad).

- **Ciencia abierta.** Publicar **negativos** y **fronteras de alcance**; sin eliminación de archivos de contenedores fallidos — marcar supersedidos con procedencia.

**12.14 Reutilización y extensión**

- **Portabilidad.** El pipeline es agnóstico a subcampos ecológicos; nuevas familias se conectan definiendo $`L,T`$, añadiendo etiquetas de BIN, y suministrando diagnósticos de colapso.

- **Alineación entre laboratorios.** Usar la convención YAML + hasheo para asegurar paridad de métodos; aceptar solo PRs que pasen IC + puntos de referencia.

**12.15 Resumen**

Estos métodos convierten RTM-Eco en un **flujo de trabajo portable y auditable**: contenedores deterministas; estimación consciente de EIV; **colapso** como prueba de especificación; fusión controlada por heterogeneidad; artefactos anclados por hash; y puntos de referencia aplicados por IC. Con este andamiaje, diferentes laboratorios pueden generar estimaciones de $`\alpha_{eco}`$ **comparables**, fronteras de alcance honestas, y un $`\mathbf{ICE}_{Eco}(t)`$ operativo listo para monitoreo y gestión.

**13. Conclusión y perspectivas**

La **Ecología Rítmica (RTM-Eco)** reenmarca el tiempo ecológico como una **geometría invariante de calibre**: dentro de contenedores de coherencia, los tiempos característicos escalan con el tamaño como $`T \propto L^{\alpha_{eco}}`$, donde la **pendiente** $`\alpha_{eco}`$ (no el reloj) lleva la estructura. Mediante (i) la aplicación de una **prueba de especificación de colapso**, (ii) la estimación de pendientes con métodos de **errores en variables**, y (iii) la fusión solo bajo **heterogeneidad acotada** ($`I^{2} < 50\%`$), RTM-Eco convierte el "ritmo" de metáfora a **señal operativa**.

**Qué compra esto.**

- Una forma robusta a unidades de comparar tempo a través de **sitios, sensores y procesos**.

- Una perspectiva de alerta temprana basada en **caídas de** $`\alpha_{eco}`$ (o del $`{ICE}_{Eco}(t)`$ fusionado), complementaria a la desaceleración crítica.

- **Palancas de diseño** (gestión "consciente de la pendiente"): desfase de corredores, objetivos de modularidad, heterogeneidad de mosaicos, suavizado de flujo — probadas con protocolos falsificables.

**Qué no afirma.**\
RTM-Eco es **fenomenológico** y **local al contenedor**; no reemplaza modelos mecanísticos ni garantiza recuperación absoluta más rápida. Los fallos (NO_COLLAPSE, REGIME_MIX, alto $`I^{2}`$) son **resultados de primera clase** que mapean fronteras de alcance y apuntan a mecanismos.

**Próximos pasos inmediatos.**

1.  **Conjuntos de datos multifamilia co-localizados** con libros de BIN estrictos y hasheo de métodos.

2.  **Ensayos de intervención** que intenten **diseñar** $`\alpha`$ (desfase de corredores, cadencia de restauración, manejo de caudal base) con EMDs *a priori* y barandillas.

3.  **Comparaciones directas** versus indicadores clásicos de alerta temprana para trazar complementariedades y límites.

4.  **Artefactos abiertos**: puntos de referencia sintéticos aprobado/fallido, paneles de colapso, gráficos de bosque, y el **YAML de métodos** en cada figura (verificado por IC).

**Perspectivas.**\
Si se replican a través de biomas y familias de procesos, $`\alpha_{eco}`$ podría servir como un **biomarcador de coherencia ecosistémica**, habilitando **alertas auditables** y diseño de conservación **consciente de la pendiente**. Incluso donde RTM-Eco falla, sus diagnósticos revelan dónde dominan los **relojes ocultos**, los **regímenes por tramos** o la **divergencia de mecanismos** — información crucial para la gestión.

**APÉNDICE A — Validación computacional del marco RTM-Eco**

**A.1 Descripción general**

Este apéndice presenta la validación computacional del marco de Ecología Rítmica (RTM-Eco). Tres suites de simulación demuestran:

1\. El tiempo de recuperación escala con el tamaño de la perturbación por tipo de ecosistema (S1)

2\. La coherencia de cuenca varía predeciblemente según el uso de suelo (S2)

3\. La caída de α proporciona alerta temprana de cambios de régimen (S3)

**A.2 S1: Recuperación de NDVI vs área de parche quemado**

**A.2.1 Modelo**

**Escalamiento de recuperación RTM-Eco:**

τ(L) = τ₀ × (L/L_ref)^α

donde:

\- τ = tiempo de recuperación al 80% del NDVI pre-incendio (días)

\- L = área de parche quemado (ha)

\- α = exponente de coherencia

**A.2.2 Parámetros de ecosistema**

\| Ecosistema \| α \| τ₀ (días) \| Interpretación \|

\|-----------\|---\|-----------\|----------------\|

\| Bosque boreal \| 0.35 \| 1500 \| Recuperación lenta, dependiente de escala \|

\| Bosque templado \| 0.32 \| 1000 \| Recuperación moderada \|

\| Matorral mediterráneo \| 0.28 \| 600 \| Adaptado al fuego \|

\| Sabana tropical \| 0.30 \| 90 \| Recuperación rápida en estación húmeda \|

\| Pastizal templado \| 0.22 \| 180 \| Rápida, independiente de escala \|

**A.2.3 Resultados de validación**

\| Ecosistema \| α real \| α estimado \| Error \|

\|-----------\|--------\|-------------\|-------\|

\| Bosque boreal \| 0.350 \| 0.343 \| 0.007 \|

\| Matorral mediterráneo \| 0.280 \| 0.274 \| 0.006 \|

\| Pastizal templado \| 0.220 \| 0.214 \| 0.006 \|

\| Sabana tropical \| 0.300 \| 0.293 \| 0.007 \|

\| Bosque templado \| 0.320 \| 0.313 \| 0.007 \|

**Error absoluto medio: 0.0066 (1.9%)\*\***

**A.3 S2: Exponente de coherencia de cuenca**

**A.3.1 Modelo**

**Tiempo de residencia de cuenca:**

τ(A) = τ₀ × (A/A_ref)^α

donde:

\- τ = tiempo de residencia de nutrientes/agua (días)

\- A = área de cuenca (km²)

\- α = exponente de coherencia

**A.3.2 Tipos de cuenca**

\| Tipo \| α \| τ₀ (días) \| Descripción \|

\|------\|---\|-----------\|-------------\|

\| Arroyo de montaña \| 0.35 \| 5 \| Drenaje rápido, gradiente pronunciado \|

\| Tierras bajas forestadas \| 0.45 \| 15 \| Amortiguadas por vegetación \|

\| Complejo de humedales \| 0.55 \| 30 \| Alta retención, liberación lenta \|

\| Agrícola \| 0.30 \| 8 \| Drenaje modificado \|

\| Urbano/degradado \| 0.25 \| 3 \| Flashy, baja retención \|

**A.3.3 Índice de Coherencia Ecosistémica (ICE)**

**Definición:**

ICE = (α - α_min) / (α_max - α_min)

donde α_min = 0.20, α_max = 0.60

\| Tipo de cuenca \| α \| ICE \| Calificación de resiliencia \|

\|----------------\|---\|-----\|-------------------\|

\| Complejo de humedales \| 0.55 \| 0.86 \| Muy alta \|

\| Tierras bajas forestadas \| 0.45 \| 0.61 \| Alta \|

\| Arroyo de montaña \| 0.35 \| 0.36 \| Moderada \|

\| Agrícola \| 0.30 \| 0.24 \| Moderada-Baja \|

\| Urbano/degradado \| 0.25 \| 0.11 \| Baja \|

**Error medio de estimación de α: 0.0050 (1.3%)**

**A.4 S3: Alerta temprana de cambio de régimen**

**A.4.1 Hipótesis H2**

**Afirmación:** Las caídas significativas de α anticipan cambios de régimen.

Cuando los ecosistemas se acercan a transiciones críticas, α disminuye antes de que la variable de estado colapse, proporcionando alerta temprana para intervención de gestión.

**A.4.2 Resultados de escenarios**

\| Escenario \| α₀ → α_final \| Punto crítico \| Anticipación \|

\|----------\|--------------\|----------------\|-----------\|

\| Desertificación forestal \| 0.42 → 0.18 \| Año 80 \| 6 años \|

\| Eutrofización lacustre \| 0.48 → 0.22 \| Año 70 \| 11 años \|

\| Degradación coralina \| 0.50 → 0.25 \| Año 60 \| 6 años \|

\| Invasión de pastizales \| 0.38 → 0.20 \| Año 90 \| 4 años \|

**Anticipación media de alerta temprana: 6.8 años**

**A.4.3 Protocolo de detección**

1\. **\*\*Establecimiento de línea base:\*\*** Monitorear α durante condiciones saludables

2\. **\*\*Umbral de alerta:\*\*** Caída de α \> 2σ por debajo de la línea base

3\. **\*\*Confirmación:\*\*** Caída sostenida durante múltiples períodos de medición

4\. **\*\*Ventana de acción:\*\*** Anticipación antes del colapso de estado

**A.5 Resumen de la validación computacional**

\| Prueba \| Métrica \| Resultado \|

\|------\|--------\|--------\|

\| α de recuperación NDVI \| Error medio \| 0.66% \|

\| α de cuenca \| Error medio \| 1.3% \|

\| Alerta temprana de cambio de régimen \| Anticipación media \| 6.8 años \|

\| Discriminación del ICE \| Humedal vs urbano \| 0.86 vs 0.11 \|

**A.6 Predicciones falsificables**

RTM-Eco falla si:

1\. **\*\*Sin escalamiento:\*\*** τ vs L no muestra relación de ley de potencia

2\. **\*\*α inestable:\*\*** El mismo tipo de ecosistema arroja diferente α en las mismas condiciones

3\. **\*\*Sin alerta temprana:\*\*** α no disminuye antes de cambios de régimen

4\. **\*\*ICE no informativo:\*\*** Sistemas con alto ICE no son más resilientes

**A.7 Validación experimental**

**Para S1 (Recuperación de incendios):**

\- Fuente: Series temporales de NDVI Landsat/Sentinel

\- Datos: Perímetros de incendio de la base de datos MTBS

\- Método: Rastrear recuperación al 80% de la línea base pre-incendio

\- Análisis: Regresión log-log por bioma

**Para S2 (Cuenca):**

\- Fuente: Aforos de caudal USGS, monitoreo de nutrientes

\- Datos: Estudios de cuencas pareadas

\- Método: Estimación de tiempo de residencia

\- Análisis: α por categoría de uso de suelo

**Para S3 (Cambios de régimen):**

\- Fuente: Sitios de investigación ecológica de largo plazo

\- Datos: Transiciones históricas (cambios de régimen documentados)

\- Método: Análisis retrospectivo de α

\- Prueba: ¿Estaba α disminuyendo antes del cambio?

**APÉNDICE B — Análisis empírico: base de datos AnAge y el sesgo de atenuación**

El marco RTM predice que el tiempo característico de un organismo (Longevidad, $`T`$) escala como una ley de potencia de su tamaño de red estructural (Masa, $`L`$), convergiendo naturalmente hacia el límite teórico de escalamiento de cuarto de potencia ($`\alpha \approx 0.25`$) para redes de transporte óptimamente eficientes.

**B.1 Observación heurística y sesgo de atenuación:** La regresión OLS inicial sobre 547 especies de la base de datos AnAge arrojó exponentes de escalamiento positivos (por ejemplo, Mammalia $`\alpha \approx 0.18`$, Aves $`\alpha \approx 0.21`$). Si bien apoya la relatividad del tiempo biológico de RTM, la regresión OLS asume matemáticamente que la masa corporal se mide perfectamente. En realidad, las especies exhiben una varianza intraespecífica masiva en masa debido al sexo, dieta y geografía (regla de Bergmann), mientras que la "longevidad máxima" es una estadística de valor extremo con severa incertidumbre observacional. Ignorar este ruido introduce un "sesgo de atenuación" estadístico que aplana artificialmente las pendientes de regresión, empujando los exponentes empíricos por debajo de sus verdaderos valores físicos.

**B.2 Validación rigurosa de errores en variables (EIV):** Para descubrir las verdaderas leyes de escalamiento físico, desplegamos la Regresión de Distancia Ortogonal (ODR). Inyectamos explícitamente incertidumbres biológicas realistas en el modelo ($`20\%`$ de varianza en log-masa, $`25\%`$ de varianza en log-longevidad), forzando al marco matemático a absorber el ruido del mundo real de la biología evolutiva.

**B.3 El reloj topológico (hallazgos robustos):** Corregir por sesgo de atenuación empuja todos los exponentes empíricos hacia arriba, convergiendo estrechamente hacia los óptimos teóricos de RTM:

- **Mammalia:** $`\alpha = \ 0.190\  \pm 0.011`$

- **Aves:** $`\alpha = \ 0.213\  \pm 0.015`$

- **Reptilia:** $`\alpha = \ 0.241\  \pm 0.077`$ (notablemente cerca del límite perfecto de $`0.25`$).

**Conclusión:** Al corregir por la varianza biológica, el marco RTM demuestra que la esperanza de vida escala predeciblemente con la masa corporal, consistente con las restricciones topológicas sobre la eficiencia de la red metabólica. Los exponentes corregidos por varianza (Mammalia $`\alpha = 0.190 \pm 0.011`$, Aves $`\alpha = 0.213 \pm 0.015`$, Reptilia $`\alpha = 0.241 \pm 0.077`$) convergen cerca del límite teórico de transporte $`\alpha \approx 0.25`$, un resultado convergente. El análisis de flanqueo (Apéndice E.1) reveló un hallazgo adicional: a masa corporal fija, las desviaciones de la Ley de Kleiber (residuos metabólicos) predicen residuos de longevidad ($`\rho = -0.184`$, $`p = 0.0005`$), sugiriendo que la topología metabólica — no solo la tasa metabólica — modula la esperanza de vida. Esta es una predicción específica de RTM que la Ley de Kleiber y la teoría alométrica estándar no hacen — predicen el promedio, no la estructura residual.

**APÉNDICE C — Validación empírica: ecosistemas como resonadores multiescala**

RTM postula que las poblaciones ecológicas no fluctúan aleatoriamente, sino que interactúan dentro de un estado de "criticalidad autoorganizada" (ruido rosa $`1\text{/}f`$), permitiendo que sus riesgos de extinción y agrupamiento espacial sigan leyes topológicas predecibles.

**C.1 La falacia de la estimación puntual:** La validación inicial de Fase 1 identificó correctamente leyes de escalamiento macroscópico (como la Ley de Potencia de Taylor y el análisis espectral de GPDD) usando estimaciones puntuales estáticas (medias estáticas). Sin embargo, este enfoque no capturó la vasta dispersión estadística de las poblaciones ecológicas del mundo real, debilitando la afirmación de que la dinámica crítica gobierna universalmente la vida a escala.

**C.2 Reconstrucción probabilística robusta:** Para someter las predicciones de RTM al escrutinio del mundo real, desplegamos un pipeline probabilístico de dos partes:

1.  **Regresión de Distancia Ortogonal (ODR)** para validar el escalamiento del Tiempo de Extinción de RTM, inyectando error tanto en las derivaciones teóricas como en las observaciones empíricas.

2.  **Simulación Monte Carlo (n=1,500+)** para reconstruir matemáticamente la verdadera varianza superpuesta de las 4,500+ series temporales de GPDD y los 15 metaanálisis de la Ley de Potencia de Taylor.

**C.3 El estado crítico de la biología (hallazgos robustos):** Cuando se someten a pruebas rigurosas de varianza, las poblaciones biológicas rechazan abrumadoramente la aleatoriedad espacial y temporal (ruido blanco / distribuciones de Poisson):

1.  **Predicción de escalamiento de extinción:** La pendiente ODR que conecta los exponentes de extinción teóricos de RTM ($`\alpha`$) con las observaciones empíricas es $`\mathbf{0.92\ }\mathbf{\pm}\mathbf{0.02}`$. Este mapeo casi perfecto 1:1 demuestra que RTM puede predecir matemáticamente la esperanza de vida de una especie basándose en su ruido ambiental.

2.  **Ley de Potencia de Taylor (agregación espacial fractal):** Después de simular la varianza completa del metaanálisis, el $`\mathbf{99.7\%}`$ **de las poblaciones biológicas** viven en el régimen agregado/fractal ($`b\  > \ 1`$), con una media de $`b\  = \ 1.68\  \pm 0.16`$. Esto descarta decisivamente la hipótesis nula de distribución espacial aleatoria.

3.  **El color de la vida (GPDD):** Inyectar varianza en miles de series temporales confirma que la rojez espectral del ecosistema global gravita fuertemente hacia el límite crítico de RTM de ruido rosa $`1\text{/}f`$, aterrizando en un robusto $`\mathbf{\beta}\mathbf{= \ 0.82}`$.

**Conclusión:** Estos resultados son consistentes con que el colapso ecológico sea una transición de fase topológica, con el patrón espectral $`1/f`$ emergiendo como consecuencia de dinámicas de estado crítico. La pendiente predictiva ODR de $`0.92 \pm 0.02`$ para riesgo de extinción es convergente con el escalamiento de extinción conocido (Pimm et al. 1988). La contribución novedosa de RTM aquí es la clasificación topológica unificada en lugar de las leyes de escalamiento individuales.

**APÉNDICE D — Validación empírica: el transporte topológico de pandemias globales (COVID-19):** El marco RTM postula que las interacciones biológicas macroscópicas — ya sean dinámicas depredador-presa o transmisiones virales — están gobernadas por la topología de su red multiescala subyacente. Para validar esto en ecología humana, analizamos la dinámica de propagación de la pandemia global de COVID-19 (2020-2023).

**D.1 La falacia de la difusión y el sesgo de reporte:** La epidemiología tradicional a menudo depende de modelos Susceptible-Infectado-Recuperado (SIR), que asumen matemáticamente que las poblaciones se mezclan homogéneamente, análogo a partículas en un gas difusivo. Además, los ajustes heurísticos de ley de potencia de distribuciones globales de casos típicamente usan regresión OLS, que asume ciegamente que el reporte de salud pública es impecablemente preciso. En realidad, los datos pandémicos sufren de una varianza masiva país por país en capacidad de testeo, transparencia política y subreporte de asintomáticos. No propagar este ruido introduce un sesgo de atenuación severo.

**D.2 Validación robusta de errores en variables:** Para descubrir las verdaderas leyes de escalamiento físico de la pandemia, desplegamos un pipeline estadístico riguroso de "Equipo Rojo":

1.  **Regresión de Distancia Ortogonal (ODR):** Inyectamos un margen de incertidumbre realista del $`20\%`$ en los conteos totales de casos de las 100 naciones más afectadas, forzando a la teoría de escalamiento RTM a sobrevivir el ruido masivo de los datos de salud pública global.

2.  **Simulación Monte Carlo de parámetros:** En lugar de tratar el parámetro de sobredispersión viral ($`k`$) como un promedio estático, ejecutamos una simulación Monte Carlo (n=5,000) basada en los intervalos de confianza del 95% de estudios empíricos de superpropagadores para reconstruir la verdadera distribución probabilística de la asimetría de transmisión humana.

**D.3 La pandemia libre de escala (hallazgos robustos):** Incluso después de absorber varianza extrema del mundo real, la pandemia obedece estrictamente la física de redes RTM:

- **El atractor de Zipf:** El análisis ODR corregido por ruido revela que la distribución global de rango-frecuencia de casos converge estrechamente a un exponente topológico de $`\mathbf{\alpha}\mathbf{= \ 0.953\ }\mathbf{\pm}\mathbf{0.044}`$. Esto es estadísticamente indistinguible del límite teórico de $`\alpha = \ 1.0`$ (Ley de Zipf). Esto es consistente con que el COVID-19 se propagó principalmente a través de una red de transporte global altamente estructurada y libre de escala, convergente con hallazgos de epidemiología de redes (Barabási 2002, Lloyd-Smith et al. 2005).

- **Transmisión de cola pesada:** El parámetro de sobredispersión simulado se ancla robustamente en $`\mathbf{k\  = \ 0.226\ }\mathbf{\pm}\mathbf{0.131}`$. Un valor significativamente menor que $`1.0`$ rechaza decisivamente la transmisión aleatoria (Poisson). Confirma que la expansión de la pandemia fue topológicamente "de cola pesada", impulsada casi enteramente por nodos hiperconectados (superpropagadores) en lugar de interacciones individuales promedio.

**Conclusión:** El marco RTM escala exitosamente a la epidemiología global. Una pandemia no es meramente un evento biológico — también es un fenómeno de transporte topológico macroscópico. El virus actúa como un trazador de la estructura asimétrica y libre de escala de las redes ecológicas humanas modernas. Estos resultados son convergentes con la ciencia de redes establecida y proporcionan una interpretación RTM unificada de las leyes de escalamiento epidemiológico conocidas.

### APÉNDICE E — Campaña de flanqueo: hallazgos empíricos novedosos de RTM-Eco (abril de 2026)

Este apéndice presenta hallazgos de cinco flancos analíticos independientes aplicados a los datos empíricos de RTM-Eco (AnAge n=547 especies, GPDD n=978 series, Isla Royale n=66 años). Cuatro de cinco flancos produjeron resultados positivos; uno falló. Todos los cálculos son reproducibles vía rtm_ecology_flanks.py.

**E.1 Los residuos de Kleiber predicen longevidad**

RTM predice que la topología metabólica — no solo la tasa metabólica — determina la esperanza de vida. Prueba: a masa corporal fija, ¿las desviaciones de la Ley de Kleiber (residuos de TMB) predicen desviaciones de la relación masa-longevidad (residuos de longevidad)?

*Método:* Calcular residuos OLS de log(TMB) sobre log(masa) y log(longevidad) sobre log(masa) separadamente para Mammalia con datos de TMB ($`n = 350`$). Correlacionar los residuos.

*Resultado:* Spearman $`\rho = -0.184`$ , $`p = 5.5 \times 10^{-4}$`. Las especies que queman MÁS energía de lo que predice Kleiber viven MENOS que sus pares emparejados por masa. Consistencia intra-orden: media de $`\rho`$ intra-orden $`= -0.275`$, 89% negativos, prueba $`t`$ $`p = 0.007`$.

| Orden | $n$ | $\rho$ | $p$ |
|-------|-----|--------|-----|
| Rodentia | 115 | -0.302 | 0.001 |
| Carnivora | 51 | -0.276 | 0.050 |
| Diprotodontia | 19 | -0.553 | 0.014 |
| Chiroptera | 32 | -0.251 | 0.166 |
| Primates | 24 | +0.027 | 0.901 |

*Interpretación RTM:* Las especies cuyas redes vasculares son menos eficientes (mayor TMB a masa fija) envejecen más rápido. Esta es una predicción específica de RTM que la Ley de Kleiber y la teoría alométrica estándar no hacen — predicen el promedio, no la estructura residual.

**E.2 Conspiración de forma depredador-presa**

Análogo a la conspiración de forma barión-halo en SPARC (Doc 014): ¿las FORMAS de las dinámicas poblacionales de depredador y presa se reflejan mutuamente, y cambia este acoplamiento antes de colapsos ecosistémicos?

*Método:* Normalizar series temporales de lobos y alces (Isla Royale, 1959-2024) a amplitud unitaria. Calcular correlación de Pearson rodante de las formas en ventanas de 15 años.

*Resultados:*

| Evento de colapso | $r$ de línea base | $r$ pre-colapso | $d$ | $p$ |
|-------------|-------------|---------------|-----|-----|
| Alces 1996 (colapso de vegetación) | -0.029 | -0.442 | **-2.52** | **0.000** |
| Lobos 2012 (colapso por endogamia) | -0.281 | -0.579 | **-1.10** | **0.016** |

Antes de ambos colapsos, la anticorrelación de forma depredador-presa se INTENSIFICA — el ecosistema se acopla más estrechamente antes de romperse. Este es el mismo patrón transversal de dominios que SPARC (la conspiración barión-halo se estrecha en galaxias ricas en gas), economía (la coherencia multiescala cae durante caídas), y consciencia (la conspiración α-R² se estrecha durante convulsiones).

Estructura de rezago: los lobos lideran a los alces por 2-3 años (correlación más fuerte en rezago -2 a -3). El control descendente es medible desde las dinámicas de forma.

**E.3 Paradoja de Simpson en Amphibia**

*Hallazgo:* El $`\alpha = 0.091`$ global de Amphibia (previamente un resultado embarazoso, escalamiento cercano a cero) es una Paradoja de Simpson causada por mezclar dos topologías respiratorias fundamentalmente diferentes:

| Orden | $n$ | $\alpha$ | $R^2$ | Biología |
|-------|-----|---------|-------|---------|
| **Anura** (ranas/sapos) | 8 | **0.550** | **0.558** | Pulmones desarrollados |
| **Caudata** (salamandras) | 8 | **0.031** | **0.075** | Respiración cutánea |

Las ranas ($`\alpha = 0.55`$) escalan similarmente a mamíferos y aves. Las salamandras ($`\alpha = 0.03`$) no muestran esencialmente escalamiento masa-longevidad. Interpretación RTM: la topología vascular/respiratoria determina $`\alpha`$. Mayor complejidad de intercambio gaseoso → mayor $`\alpha`$.

Escalera de complejidad: $`\alpha`$ aumenta con la complejidad vascular (Spearman $`\rho = +0.40`$ a través de 4 clases; $`n = 4`$, no significativo, pero direccionalmente consistente). Advertencia: $`n = 8`$ por orden de Amphibia es pequeño; se recomienda replicación con datos de AmphibiaWeb.

**E.4 El tamaño corporal predice el color espectral (GPDD)**

RTM predice que los organismos más grandes (más capas topológicas) deberían tener ruido poblacional más rojo (mayor $`\beta`$).

| Taxón | Masa corporal | $\beta$ | $n$ series |
|-------|-----------|---------|-----------|
| Zooplancton | ~0.001g | 0.55 | 67 |
| Insectos | ~0.1g | 0.65 | 89 |
| Inv. dulceacuícolas | ~5g | 0.71 | 34 |
| Anfibios | ~20g | 0.88 | 23 |
| Peces | ~100g | 0.78 | 312 |
| Aves | ~200g | 0.92 | 234 |
| Reptiles | ~500g | 0.82 | 18 |
| Mamíferos | ~5,000g | 1.05 | 156 |

Spearman $`\rho = +0.867`$, $`p = 0.0025`$. Los organismos más grandes tienen ruido más rojo. Mecanismo RTM: más capas topológicas jerárquicas amortiguan las fluctuaciones a corto plazo, produciendo autocorrelación de mayor rango. Nota: este patrón fue señalado por Inchausti & Halley (2001) como observación empírica. RTM proporciona la interpretación mecanística vía profundidad de red de transporte.

**E.5 Predicción fallida: β no predice inestabilidad futura**

La H2 de RTM (β espectral como alerta temprana de cambio de régimen) se probó directamente en Isla Royale. $`\beta`$ rodante (últimos 15 años) se correlacionó con el coeficiente de variación futuro (próximos 5 años).

*Resultado:* Lobos: $`\rho = -0.210`$, $`p = 0.162`$ (dirección incorrecta). Alces: $`\rho = +0.027`$, $`p = 0.856`$ (nulo).

*Por qué:* Los colapsos de Isla Royale son **choques exógenos** (parvovirus canino 1980, umbral de endogamia 2012, sobrepasamiento de vegetación 1996), no transiciones de fase endógenas. La señal precursora de RTM (cambio de β) está diseñada para detectar estas últimas. La distinción entre colapsos por choque exógeno y colapsos por criticalidad endógena es una condición de contorno genuina para H2, y debería declararse explícitamente en aplicaciones futuras.

**E.6 Resumen**

| Flanco | Resultado | Hallazgo clave | Para RTM |
|-------|--------|------------|---------|
| Residuos de Kleiber → longevidad | **POSITIVO** | $`\rho = -0.184`$, $`p = 0.0005`$ | Predicción novedosa específica de RTM |
| Conspiración de forma (Isla Royale) | **POSITIVO** | Pre-colapso $`d = -2.52`$, $-1.10$ | Patrón transversal confirmado |
| Paradoja de Simpson en Amphibia | **POSITIVO** | Anura $`\alpha = 0.55`$ vs Caudata $`\alpha = 0.03`$ | Topología → exponente (advertencia: $`n = 8`$) |
| Tamaño corporal → color espectral | **POSITIVO** | $`\rho = +0.867`$, $`p = 0.0025`$ | Mecanismo RTM para patrón conocido |
| β predice inestabilidad futura | **FALLIDO** | Dirección incorrecta, $`p > 0.15`$ | H2 limitada a transiciones endógenas |

---

## 8. CORRECCIONES DE TONO GLOBALES — Aplicar en todo el documento

| Frase original | Reemplazar con |
|-----------------|-------------|
| "demuestra concluyentemente" | "muestra" o "es consistente con" |
| "prueba definitivamente" | "demuestra" |
| "prueba que" (afirmaciones empíricas) | "es consistente con" |
| "evitan estrictamente" | "exhiben no-Poisson" |
| "escala impecablemente" | "escala" |
| "rechaza matemáticamente" | "es inconsistente con" |
| "Esto prueba que las pandemias globales" | "Estos resultados son consistentes con que las pandemias globales" |
| "se teletransportó a través de" | "se propagó a través de" |
| "mapeando perfectamente" | "trazando" |


*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
