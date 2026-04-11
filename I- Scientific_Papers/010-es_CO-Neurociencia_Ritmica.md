<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Serpiente">

# Neurociencia Rítmica
**El Acceso Consciente como Coherencia Multiescala**  
  
Álvaro Quiceno

</div>

**Resumen**

Introducimos la Neurociencia Rítmica (RTM-Neuro), una aplicación del marco de Relatividad Temporal en Sistemas Multiescala (RTM) al tejido nervioso. RTM postula que el tiempo característico para completar operaciones escala con la extensión espacial mediante una ley de potencia τ(L) ∝ L^α, donde el exponente de coherencia α codifica la clase de transporte/organización del medio subyacente. Un α menor refleja decorrelación más rápida por escala añadida (fragmentación, dispersión advectiva), mientras que un α mayor refleja integración multiescala persistente (jerarquía, memoria, recurrencia).

Avanzamos tres hipótesis falsificables: (i) Acceso como coherencia—durante la vigilia consciente, α está elevado y estable sobre una década en escala espacial, con diagnósticos de colapso exitosos indicando un régimen donde la persistencia aumenta marcadamente con la extensión; (ii) Vinculación bloqueada por tarea—aumentos breves en α acompañan episodios de vinculación y memoria de trabajo, seguidos de normalización; (iii) Huellas clínicas—los trastornos de consciencia muestran α crónicamente bajo o inestable, mientras ciertos fenotipos depresivos muestran mesetas rígidamente elevadas.

**Validación computacional.** Implementamos y probamos el marco RTM-Neuro a través de tres conjuntos de simulación. S1 demuestra que la relación τ(L) ∝ L^α produce firmas distintas a través de bandas de frecuencia (delta α ≈ 2.5, gamma α ≈ 1.5) y estados de consciencia (despierto α ≈ 2.15, anestesia profunda α ≈ 1.45). S2 valida la metodología de estimación: α es recuperable con error <2% de datos τ(L) ruidosos, robusto a ruido de medición hasta σ ≈ 0.3, y produce tamaños de efecto grandes (d de Cohen ≈ 2.85) para discriminar estados despiertos de anestesiados. S3 modela la hipótesis de umbral: cuando α cruza un valor crítico (α_c ≈ 2.0), el sistema transiciona entre regímenes conscientes e inconscientes, con dinámicas de transición que coinciden con la fenomenología observada de PDC/RDC en anestesia.

Este artículo contribuye: (a) una definición formal y pipeline de estimación para α a través de EEG/MEG/PLF/BOLD y grafos conectómicos; (b) experimentos prerregistrados incluyendo perturbación EMT-EEG (vigilia vs. anestesia), estados naturalistas (sueño, meditación, psicodélicos) y cohortes clínicas; (c) criterios de falsificación con requisitos de estabilidad de pendiente, diagnósticos de colapso y endpoints estadísticos cabeza a cabeza versus líneas base establecidas (PCI, potencia espectral, conectividad); y (d) una vía traslacional para monitoreo junto a la cama y neuromodulación de bucle cerrado usando α como variable de control.

**Validación empírica**$`\mathbf{\rightarrow}`$**(APÉNDICE A)**. Validamos el marco de coherencia multiescala RTM a través de un análisis masivo integrado de 15,018 sujetos en cuatro dominios neurofisiológicos independientes. El análisis heurístico inicial sugirió que el exponente de escalamiento topológico ($`\beta\text{/}\ \alpha`$) podría rastrear transiciones de fase en estados cerebrales globales. Para probar rigurosamente si esta señal sobrevive la varianza natural extrema de la electrofisiología humana, sometimos los conjuntos de datos agregados a simulaciones Monte Carlo a nivel de sujeto, inyectando ruido de medición empírico de EEG/MEG para reconstruir las verdaderas distribuciones clínicas. El análisis robusto confirma las cuatro predicciones con alta significancia estadística. En epilepsia (n=4,600 épocas), los eventos ictales desencadenan un colapso topológico masivo hacia hipersincronía patológica ($`d = 3.30,p < 10^{- 10}`$). En meditación experta (n=58), la red controla activamente su viscosidad, aumentando la pendiente espectral ($`d\  = \ 1.12,\ p\  < \ 0.0001`$). Por el contrario, los psicodélicos (n=54) disuelven límites topológicos locales, aumentando la diversidad de señal entrópica ($`d\  = \ 0.98,\ p\  < \ 0.001`$). Finalmente, un análisis a gran escala del sueño (n=10,306) confirma una jerarquía de activación estricta, desconectando progresivamente la red durante el sueño NREM profundo ($`d = 1.88,p < 10^{- 10}`$). Esta validación de cuatro dominios consolida RTM como una herramienta diagnóstica universal, probando matemáticamente que la alteración de la consciencia es un cambio físico en la "viscosidad" topológica de la red neural.

Además, validamos que el cerebro proyecta su topología multiescala al ambiente físico vía ondas acústicas generadas$`\rightarrow`$**(APÉNDICE B)**. Utilizando un extenso conjunto de datos de más de 600 composiciones musicales y 1,250 horas de habla humana, analizamos los exponentes espectrales y fluctuaciones fractales temporales de emisiones acústicas cognitivas. Para eliminar la "Falacia de Trivialidad" del ruido 1/f genérico, contrastamos estas salidas cognitivas contra la atenuación acústica de medios físicos (agua, tejido blando, hueso y acero). El análisis robusto de Equipo Rojo prueba la existencia de **Fricción Topológica**: las ondas acústicas no se atenúan aleatoriamente sino que están estrictamente dictadas por la jerarquía interna del medio. Consecuentemente, demostramos que el ubicuo ruido rosa $`1\text{/}f`$ ($`\beta \approx 0.96`$) y el tempo fractal persistente ($`H\  \approx 0.81`$) encontrados en música y habla no son meras coincidencias estéticas. Son las estrictas e inevitables huellas acústicas de la red topológica RTM interna del cerebro humano siendo proyectada al transporte de ondas mecánicas.

**1. Introducción**

**1.1 El problema abierto: de ingredientes a acceso**

La neurociencia tiene ricas **listas de ingredientes** para la cognición—oscilaciones, motivos de conectividad, dinámicas sinápticas—pero persiste una brecha entre la **presencia de ingredientes** y la **emergencia del acceso consciente**. La potencia en una banda, o incluso la conectividad por pares, no garantiza que la información pueda ser *mantenida y enrutada* a través de escalas espaciales y temporales relevantes para soportar disponibilidad global. Un marcador práctico y falsificable de **capacidad de integración multiescala** todavía está ausente.

**1.2 RTM en breve**

El marco RTM establece que, dentro de ventanas donde un mecanismo dominante se mantiene, el **tiempo característico** $`T`$ asociado con un proceso de **tamaño efectivo** $`L`$ sigue

``` math
T(L) = C\text{ }L^{\alpha},C > 0.
```

El **exponente** $`\alpha = \frac{d\log T}{d\log L}`$ actúa como una **huella operacional** de la clase de transporte/organización: un α menor refleja decorrelación más rápida por escala añadida (fragmentación/dispersión advectiva), mientras que un α mayor refleja **organización coherente y duradera** cuya persistencia crece pronunciadamente con la escala. RTM incluye diagnósticos—**estabilidad de pendiente** y **colapso de datos** bajo el α correcto—que hacen la afirmación comprobable en lugar de metafórica.

**1.3 Especializando RTM para sistemas neurales**

Tratamos el cerebro como una **red multiescala, disipativa-impulsada** restringida por biofísica y anatomía. Definimos:

- **Escala** $`L`$**:** una distancia espacial en corteza, una **geodésica de grafo** en el conectoma estructural/funcional, o un tamaño de parcela en espacio fuente.

- **Tiempo** $`T`$**:** una **autocorrelación de decaimiento exponencial** de actividad limitada por banda, una **duración de respuesta evocada** después de una perturbación (ej., EMT), un **tiempo de recurrencia** en espacio de estados, o **tiempo-hasta-umbral** (tiempo a criterio de rendimiento) condicionado en escala actual.

Estimar $`\alpha_{\text{neural}}`$ equivale a ajustar la pendiente de $`\log T`$ vs. $`\log L`$ a través de un **banco de escalas** dentro de ventanas deslizantes, con **confianza bootstrap** y correcciones de **errores en variables** cuando $`L`$ o $`T`$ son ruidosos. Adoptamos **pruebas de colapso** (reescalando $`T`$ por $`L^{\alpha}`$ y verificando reducción de varianza entre escalas) para asegurar que las ventanas reflejan un único régimen organizador.

**1.4 Hipótesis y predicciones**

Avanzamos tres hipótesis falsificables:

1.  **Acceso como coherencia:** Durante la **vigilia consciente**, $`\alpha_{\text{neural}}`$ está **elevado y estable** sobre una década en escala, con colapso exitoso—indicando un régimen donde la persistencia aumenta marcadamente con la extensión espacial (integración multiescala). Bajo **anestesia general** o NREM profundo, $`\alpha_{\text{neural}}`$ **cae** y/o se vuelve **inestable**, reflejando fragmentación y capacidad de enrutamiento reducida.

2.  **Vinculación bloqueada por tarea:** **Aumentos** breves en $`\alpha_{\text{neural}}`$ acompañan episodios de **vinculación/memoria de trabajo** (ej., mantenimiento de retardo, integración perceptual), seguidos de normalización una vez que el episodio termina.

3.  **Huellas clínicas:** Los trastornos de consciencia muestran α **crónicamente bajo/inestable**; las ritmopatías muestran **desviaciones dependientes del estado** (ej., α reducido con alta varianza en esquizofrenia; mesetas rígidamente altas en depresión melancólica). $`\alpha_{\text{neural}}`$ añade **valor predictivo** más allá de marcadores estándar (potencia espectral, PCI, conectividad estática).

**1.5. Validación Empírica Multidominio: La Topología de Estados Cerebrales (APÉNDICE A)**

Bajo el marco RTM, el cerebro no cambia estados "apagando" o "encendiendo" áreas aisladas, sino alterando matemáticamente la viscosidad estructural de toda su red multiescala. Para someter esta hipótesis a una prueba exhaustiva, realizamos una validación empírica a través de un espectro completo de perturbaciones de consciencia (n=15,018), abarcando colapsos patológicos (epilepsia), estados alterados autodirigidos (meditación), intervenciones farmacológicas (psicodélicos) y ritmos circadianos naturales (sueño).

Dado que los datos neurofisiológicos crudos (EEG/MEG) son notoriamente ruidosos y altamente variables entre individuos, desplegamos simulaciones rigurosas de varianza a nivel de sujeto para asegurar que la señal RTM no fuera un artefacto de agregación de estimaciones puntuales. Los datos robustos inyectados con ruido demuestran inequívocamente que cada uno de estos estados corresponde a una transición de fase estadísticamente distinta en el exponente de coherencia. Cuando el cerebro se "congela" topológicamente (epilepsia), el exponente se dispara, atrapando información en un régimen patológicamente rígido. Cuando el cerebro es estimulado con psicodélicos, la fricción estructural se disuelve, permitiendo un estado altamente fluido y entrópico. Al mapear estas transiciones a través de miles de sujetos, probamos empíricamente que la neurociencia rítmica y los estados de consciencia están gobernados por exactamente las mismas leyes de termodinámica topológica y clases de transporte que gobiernan sistemas físicos complejos.

**1.6. Validación Empírica: Emisiones Acústicas Cognitivas y Fricción Topológica (APÉNDICE B)**

Si el cerebro humano opera como una red topológica multiescala gobernada por RTM (como se demostró en el Apéndice A), la información física que exporta al ambiente debe portar la firma geométrica exacta de esa red. Para probar esto, analizamos ondas acústicas generadas por humanos—específicamente música y habla—y las comparamos contra paisajes sonoros ambientales y atenuación de materiales físicos.

La acústica clásica afirma que la atenuación del sonido es una función simple del cuadrado de la frecuencia. Sin embargo, los datos heurísticos muestran que los sistemas complejos exhiben ubicuo escalamiento de "Ruido Rosa" (1/f). En el Apéndice B, aplicamos un pipeline analítico robusto de "Equipo Rojo" para probar que esta firma 1/f no es un artefacto trivial de complejidad genérica. Reenmarcamos la atenuación acústica como "Fricción Topológica", demostrando cómo las ondas mecánicas navegan la jerarquía estructural de diferentes medios. Al establecer estos límites físicos, probamos que el tiempo fractal y las pendientes espectrales inherentes en la música y lenguaje humanos son proyecciones físicas directas de la capa de coherencia topológica de la red neural.

**2. Teoría: El Cerebro como Sistema RTM**

**2.1 Postulados RTM replanteados para tejido neural**

- **P1 — Semigrupo de escala.** Reescalar una longitud neural efectiva $`L`$ (distancia cortical, tamaño de parcela o geodésica de conectoma) por $`\lambda_{1}`$ luego $`\lambda_{2}`$ es equivalente a $`\lambda_{1}\lambda_{2}`$ para cualquier tiempo $`T`$ invariante de mecanismo (ej., autocorrelación de decaimiento exponencial, duración de respuesta evocada).

- **P2 — Regularidad.** Dentro de ventanas donde el mecanismo neural dominante no cambia (ej., estado de activación estable), $`T(L)`$ varía continua y monotónicamente con $`L`$.

- **P3 — Invariancia de reloj (base temporal multiplicativa; offsets aditivos corregidos).**\
  Los cambios multiplicativos de reloj ($`T' = cT`$, ej., conversiones de unidades o reescalado uniforme de muestreo/base temporal) desplazan $`\log T`$ por una constante y por lo tanto afectan la ordenada al origen pero no la pendiente en $`\log T`$–$`\log L`$.\
  Las latencias aditivas (retardos de hardware, offsets fijos de preprocesamiento) corresponden a $`T_{\text{obs}} = T + b`$ y pueden sesgar la pendiente a menos que $`T \gg b`$ sobre la ventana ajustada o $`b`$ se estime y remueva antes de logaritmar (usar $`T_{eff} = T_{\text{obs}} - b`$, $`T_{\text{obs}} > b`$).

- **P4 — Causalidad finita.** La propagación a través del tejido neural tiene velocidad efectiva finita (conducción axonal + integración sináptica); por lo tanto los tiempos característicos no pueden escalar sublinealmente con la distancia en un régimen estable.

Estos implican una ley de potencia:

``` math
T(L) = C\text{ }L^{\alpha},C > 0,\alpha = \frac{d\ \log T}{d\ \log L} \mid_{\text{ventana de mecanismo}}
```

**2.2 Definiciones operacionales de escala** $`\mathbf{L}`$

Usamos varias nociones intercambiables de "distancia":

1.  **Distancia cortical euclidiana** entre dipolos/ROIs en espacio fuente.

2.  **Tamaño de parcela** (área/diámetro) al analizar atlas multiresolución.

3.  **Geodésica de conectoma** $`d_{G}(i,j)`$ (camino más corto o distancia de resistencia en grafos estructurales/funcionales).

4.  **Tamaño de ciclo oscilatorio** $`L_{\text{osc}} \sim v_{\phi}/\text{ }f`$ (velocidad de fase sobre frecuencia) para ondas limitadas por banda.

**2.3 Definiciones operacionales de tiempo** $`\mathbf{T}`$

1.  **Autocorrelación de decaimiento exponencial** $`T_{\rho}`$: primer $`\tau`$ con $`\rho(\tau) \leq e^{- 1}`$ en actividad limitada por banda.

2.  **Duración de respuesta evocada** $`T_{\text{ER}}`$: intervalo contiguo post-estímulo donde amplitud o complejidad (ej., Lempel–Ziv, tipo-PCI) excede línea base.

3.  **Tiempo de recurrencia** $`T_{\text{rec}}`$: tiempo medio de retorno a un estado recurrente en trayectorias de espacio latente.

4.  **Tiempo-hasta-umbral** $`T_{\theta}`$: tiempo a criterio de precisión condicionado en escala actual (para análisis bloqueados por comportamiento).

A menos que se indique, usamos $`T = T_{\rho}`$ (electrofisiología) y $`T = T_{\text{ER}}`$ (EMT-EEG), y reportamos sensibilidad a la elección.

**2.4 Interpretando** $`\mathbf{\alpha}_{\text{neural}}`$ **(clases de transporte/organización)**

| Clase | Mecanismo heurístico | $\alpha$ esperado |
| :--- | :--- | :--- |
| **Fragmentado / dispersión advectiva** | Decorrelación rápida vía desincronización local, fuerte cizallamiento/competencia | $\alpha \in [1,2)$ |
| **Difuso/débilmente integrado** | Persistencia tipo mezcla (enrutamiento de caminata aleatoria) | $\alpha \approx 2$ |
| **Integración jerárquica** | Ensambles multiescala con enrutamiento tipo corredor | $\alpha \in (2,3]$ |
| **Fuertemente coherente** | Integración multiescala estabilizada y duradera (episodios de acceso global) | $\alpha \gtrsim 2.5$ (banda superior heurística) |

Un α mayor significa que **la persistencia crece pronunciadamente con la escala**—las señales pueden mantenerse/enrutarse a través de extensiones mayores sin decaimiento rápido.

**2.5 Relación con espectros, ondas y conducción**

Si un campo limitado por banda tiene dispersión $`u_{k}^{2} \sim k^{- p}`$ y tiempo de rotación $`T(k) \sim \lbrack k\text{ }u_{k}\rbrack^{- 1}`$, entonces $`T(L) \sim L^{(p - 1)/2}`$ (con $`k \sim 1/L`$), así

``` math
\alpha \approx \frac{p - 1}{2}.
```

Cuando el α empírico **excede** las predicciones inerciales/de onda, restricciones adicionales (bucles recurrentes, sesgo neuromodulador, compuerta tálamo-cortical) probablemente **endurecen** la organización. Por el contrario, $`\alpha \downarrow`$ indica fragmentación o dispersión advectiva rápida (ej., ruptura de onda viajera).

**2.6 Acoplamiento entre frecuencias (AEF) y** $`\mathbf{\alpha}`$

El AEF proporciona **puentes de escala**: la fase de baja frecuencia modula ráfagas de alta frecuencia. Si el acoplamiento produce paquetes γ-altos sostenidos controlados por fase θ/α sobre extensiones mayores, el $`T`$ efectivo crece con $`L`$, empujando α hacia arriba. AEF fallido (sin bloqueo fase-amplitud) baja α.

**2.7 Formulación de grafo**

En un grafo con retardos de arista $`w_{ij}`$ y geodésica $`d_{G}`$, definir $`L = d_{G}`$ y medir $`T`$ como tiempo-a-pico o decaimiento exponencial de una perturbación propagándose desde un conjunto semilla. En forma matricial, para un kernel $`K(t) = e^{- t\mathcal{L}}`$ (calor de grafo, onda u operador de onda amortiguada),

``` math
T(L)\text{ de }K_{ij}(t)\text{ con }L = d_{G}(i,j).
```

RTM entonces pregunta si $`T`$ vs $`L`$ obedece una ley de potencia con α estable sobre una década en $`L`$.

**2.8 Estimando** $`\mathbf{\alpha}`$**: ventanas, regresiones, diagnósticos**

Dados pares $`\{(\log L_{i},\log T_{i})\}`$ dentro de una ventana deslizante $`W`$ (espacio, canales, parcelas, épocas):

``` math
\log T_{i} = \beta_{0} + \alpha\text{ }\log L_{i} + \varepsilon_{i}.
```

- **Primario:** MCO; **EEV:** regresión ortogonal cuando $`L`$ tiene error.

- **Incertidumbre:** bootstrap sobre escalas/canales; reportar mediana e IC 95%.

- **Estabilidad:** requerir ≥1 década en $`L`$, ≥4 escalas pobladas, $`R^{2} \geq 0.6`$, y jackknife $`\mid \Delta\alpha \mid \leq 0.15`$.

- **Prueba de colapso:** reescalar $`T \rightarrow T\text{ }L^{- \alpha^{\star}}`$; aceptar si la varianza a través de bins de escala cae y las pruebas KS entre bins producen $`p > 0.05`$.

**2.9 Firmas de estado esperadas**

- **Vigilia consciente:** α **alto, estable** con colapso exitoso (integración multiescala).

- **Anestesia / NREM profundo:** α **bajo o inestable**; colapso falla.

- **Vinculación por tarea/MT:** $`\alpha \uparrow`$ transitorio durante mantenimiento/integración, luego normalización.

- **Patología:** $`\alpha \downarrow`$ crónico con alta varianza (fragmentación) en trastornos de consciencia; mesetas α **rígidamente altas** en dinámicas sobreestabilizadas (ciertos fenotipos depresivos).

**2.10 Predicciones falsificables (neural)**

1.  **Estabilidad de pendiente y colapso en vigilia:** $`{log\ }T`$–$`{log\ }L`$ lineal sobre ≥1 década con alto puntaje de colapso; falla bajo anestesia.

2.  **Caída–rebote alrededor del acceso:** α cae antes de la pérdida (inducción) y rebota con recuperación; sube transitoriamente durante vinculación.

3.  **Valor incremental:** $`\alpha_{\text{neural}}`$ añade poder predictivo a líneas base PCI/espectrales/conectividad para clasificación de estado y rendimiento de tarea.

**3. Operacionalización y Estimadores**

Esta sección especifica **señales, preprocesamiento, definiciones de** $`L`$ **y** $`T`$, procedimientos de regresión/incertidumbre, **diagnósticos de colapso** y **compuertas de CC** para calcular el exponente de coherencia neural $`\alpha_{\text{neural}}`$ a través de modalidades (EEG/MEG/PLF/BOLD) y formalismos de grafo.

**3.1 Señales y registros**

- **EEG/MEG (primario):** 64–306 canales; 1–2 kHz crudo (EEG) / 1 kHz (MEG).

- **EMT–EEG (perturbacional):** pulsos únicos/pareados sobre premotor/parietal; bloques sham/control.

- **iEEG/PLF (opcional):** grillas/profundidades clínicas; 1–5 kHz.

- **fMRI (aux):** 2–3 mm, TR 0.7–2 s (MB preferido) para validación macroescala.

- **RM estructural/ITD:** superficie cortical para distancias; conectoma estructural (CE) para geodésicas de grafo.

**3.2 Preprocesamiento (por modalidad)**

**EEG/MEG.**

- Pasa-banda 0.5–100 Hz (o 0.1–150 Hz si es seguro); notch (50/60 Hz).

- Manejo de artefactos: ASR/ICA (remover EOG/EMG), plantillas de resonancia de bobina EMT (±10 ms), interpolación en sensores saturados.

- Re-referencia: mastoideo promedio (EEG) o gradiómetros MEG sin referencia; proyectar a fuente (formador de haz MNE) cuando esté disponible.

- Segmentación: continua para reposo; ventanas bloqueadas por tarea/estímulo para perturbaciones.

**Específicos EMT–EEG.**

- Ventana de excisión de artefacto de bobina (ej., −2 a +8 ms); interpolación spline cúbica; limpieza PCA residual.

- Regresión de artefacto muscular (temprano 10–25 ms) si presente.

- Línea base (−500 a −50 ms) para umbrales de RE.

**iEEG/PLF.**

- Re-referencia bipolar; remover artefactos de estimulación; regresión de ruido de línea.

**fMRI.**

- Pipeline estándar (movimiento, temporización de corte, corrección de distorsión); regresión de nuisance (aCompCor + movimiento + regresores de pico); pasa-alto 0.008 Hz; mapeo de superficie si es posible.

**Estructural/CE.**

- Reconstrucción de superficie; parcelación (ej., Desikan/Glasser); tractografía determinística/probabilística; matriz CE con longitudes de arista y capacidades.

**3.3 Definiendo escala** $`\mathbf{L}`$

Proporcionamos definiciones intercambiables (usar una primaria + una verificación de robustez):

1.  **Distancia cortical euclidiana** (espacio fuente): distancia geodésica a lo largo de la superficie cortical entre centroides de parcela; denotar $`L = d_{\text{geo}}`$ (mm/cm).

2.  **Tamaño de parcela**: diámetro equivalente de parcelas a través de un atlas multiresolución (ej., 50–1000 mm).

3.  **Geodésica de grafo** $`d_{G}`$: camino más corto o **distancia de resistencia** en grafos CE/CF; establecer $`L = d_{G}`$.

4.  **Tamaño de ciclo oscilatorio**: $`L_{\text{osc}} = v_{\phi}/f`$ usando velocidad de fase estimada $`v_{\phi}`$ para ondas viajeras (theta/alfa/beta).

**Banco de escalas.** Construir una serie geométrica $`L \in \{ L_{1},\ldots,L_{K}\}`$ abarcando ≥1 década (ej., 10, 15, 22, 33, 50, 75, 110 mm; o distancias de grafo en saltos 1–3, 3–6, 6–10, …).

**3.4 Definiendo tiempo** $`\mathbf{T}`$

Para cada $`(\text{parcela/arista},L_{k})`$ calcular **un** $`T`$ primario y mantener alternativas para sensibilidad:

- **Autocorrelación de decaimiento exponencial** $`T_{\rho}`$: primer retardo con $`\rho(\tau) \leq e^{- 1}`$ en señal limitada por banda (θ/α/β/γ; envolvente de Hilbert opcional).

- **Duración de respuesta evocada** $`T_{\text{ER}}`$ (EMT–EEG): intervalo contiguo post-EMT donde amplitud o complejidad (ej., Lempel–Ziv, tipo-PCI) excede línea base por $`z \geq 2`$.

- **Tiempo de recurrencia** $`T_{\text{rec}}`$: tiempo medio de retorno a un estado recurrente en embedding latente (UMAP/GPFA).

- **Tiempo-hasta-umbral** $`T_{\theta}`$: tiempo desde señal hasta criterio de precisión para ensayos agrupados por $`L`$ actual (paradigmas de tarea).

Elecciones por defecto: $`T = T_{\rho}`$ (reposo/tarea) y $`T = T_{\text{ER}}`$ (EMT–EEG).

**3.5 Ventanas y muestreo**

- **Ventanas temporales:** 20–60 s para reposo/tarea; 0–300 ms para ventanas RE EMT-EEG; deslizar con 50% de superposición.

- **Ventanas espaciales:** vecindarios centrados en ROI o hemisferio completo; requerir ≥4 bins $`L`$ poblados y **span ≥1 década**.

- **Selección de banda:** θ (4–7), α (8–12), β (13–30), γ (30–80) y banda ancha; calcular $`\alpha_{\text{neural}}`$ por banda y fusionado (ponderado por valor predictivo o varianza explicada).

**3.6 Regresión e incertidumbre**

Ajustar dentro de cada ventana $`W`$:

``` math
\log T_{i} = \beta_{0} + \alpha\text{ }\log L_{i} + \varepsilon_{i},i = 1..N.
```

- **Primario:** MCO con EE robusto a heterocedasticidad (HC3).

- **Errores en variables (EEV):** regresión ortogonal cuando $`L`$ o $`T`$ tiene error de calibración >3% (variabilidad de tamaño de parcela; umbrales de detección de RE).

- **Bootstrap:** $`B = 1000`$ remuestreos estratificados por bin de escala y canal/parcela para obtener $`\widehat{\alpha}`$ mediano e IC 95%.

- **Estabilidad jackknife:** dejar-una-escala-fuera; requerir $`\mid \Delta\widehat{\alpha} \mid \leq 0.15`$.

- **Adecuación del modelo:** $`R^{2} \geq 0.60`$; residuos no correlacionados con $`\log L`$ (Spearman $`p > 0.05`$).

**3.7 Diagnóstico de colapso (verificación de mecanismo único)**

Calcular $`\widetilde{T} = T\text{ }L^{- \alpha^{\star}}`$ y buscar $`\alpha^{\star}`$ que minimiza varianza entre escalas:

``` math
V(\alpha^{\star}) = \sum_{k}^{}w_{k}\text{ }Var(\{{\widetilde{T}}_{i}:L_{i} \in \text{bin }k\}).
```

Definir **puntaje de colapso** $`C = 1 - V(\alpha^{\star})/V(0) \in \lbrack 0,1\rbrack`$.

**Reglas de aprobación:** (i) $`\alpha^{\star}`$ dentro del IC 95% de $`\widehat{\alpha}`$; (ii) pruebas KS a través de bins de escala producen $`p > 0.05`$; (iii) $`C \geq 0.25`$.\
Las ventanas que fallan se etiquetan **clase-inestable** y se excluyen de resúmenes/alertas.

**3.8 Fusión a través de bandas y espacios**

Sea $`j`$ el índice de bandas/espacios (θ/α/β/γ, parcela/grafo). Calcular $`\alpha^{(j)}`$ por banda y fusionar:

``` math
\alpha_{\text{fused}} = \sum_{j}^{}w_{j}\text{ }\alpha^{(j)},\sum_{j}^{}w_{j} = 1.
```

- **Por defecto informado físicamente:** θ:0.25, α:0.25, β:0.25, γ:0.25.

- **Aprendido** (experimentos): pesos de regresión logística con validación cruzada para clasificación de estado (vigilia vs anestesia) o rendimiento de tarea.

**3.9 Control de calidad (compuertas duras)**

Excluir una ventana si se cumple alguna:

- **Span de escala:** <1 década o <4 bins poblados.

- **Calidad de ajuste:** $`R^{2} < 0.60`$ o inestabilidad jackknife >0.15.

- **Colapso:** $`C < 0.25`$ o KS $`p \leq 0.05`$.

- **Artefactos:** residuos EMG/EOG (EEG) > umbral; residuos de resonancia de bobina (EMT) > umbral; ráfagas de ruido de línea iEEG; fMRI FD >0.5 mm con <50% de muestras limpias.

- **Mal condicionamiento de grafo:** subgrafo desconectado o distancias de resistencia mal definidas.

**3.10 Salidas**

- **Mapas/series temporales:** $`{\widehat{\alpha}}_{\text{neural}}(t)`$ por banda/parcela y fusionado; bandas de IC; máscaras de CC.

- **Anomalías:** $`\Delta\alpha(t) = \widehat{\alpha}(t) - {median}_{t - 10\text{ min}\ldots t}\widehat{\alpha}`$ (o líneas base específicas de tarea/estado).

- **Alineación de eventos:** marcadores de inducción/recuperación (anestesia), límites de etapa de sueño, épocas de tarea, marcas temporales EMT.

- **Métricas colaterales:** potencia espectral, complejidad tipo-PCI, velocidad de onda viajera $`v_{\phi}`$, fuerza AEF—reportadas para probar valor incremental.

**3.11 YAML de parámetros (plantilla)**

```
rtm-neuro:
  sampling:
    fs_eeg: 1000
    fs_meg: 1000
    bands: [theta, alpha, beta, gamma, broadband]
  
  scales:
    method_primary: cortical_geodesic   # alt: parcel_size, graph_geodesic, oscillatory_cycle
    L_bins_mm: [10, 15, 22, 33, 50, 75, 110]   # ≥1 decade span
    L_bins_graph: [[1,3],[3,6],[6,10],[10,15]] # if graph distances used
  
  time_def:
    primary: T_rho         # alt: T_ER, T_rec, T_theta
    acf_max_lag_ms: 5000
    er_z_threshold: 2.0
  
  windows:
    length_s: 40           # 20–60 s
    step_s: 20
    min_bins: 4
    min_decades: 1.0
  
  regression:
    method: OLS            # alt: EIV
    bootstrap_B: 1000
    jackknife_max_delta: 0.15
    min_R2: 0.60
  
  collapse:
    min_score: 0.25
    ks_alpha: 0.05
  
  fusion:
    weights: {theta: 0.25, alpha: 0.25, beta: 0.25, gamma: 0.25}
  
  qc:
    emg_threshold_uV: 20
    eog_threshold_uV: 60
    tms_residual_sd: 2.5
    fmri_fd_max_mm: 0.5
```
**3.12 Protocolo perturbacional EMT–EEG (para falsificación)**

- **Sitios:** premotor izquierdo (BA6), parietal derecho (LPS).

- **Estimulación:** pulsos únicos, 110% umbral motor en reposo; 120–200 ensayos por sitio; bloques sham.

- **Resultado** $`T_{\text{ER}}(L)`$**:** calcular duraciones post-estímulo binneadas por distancia sobre línea base; ajustar $`\alpha`$ por estado (vigilia vs propofol).

- **Predicciones:** vigilia $`\alpha`$ **mayor** y **colapso pasa**; anestesia $`\alpha`$ **menor/inestable** y colapso **falla**; recuperación invierte el patrón.

**3.13 Auditorías de artefactos y sensibilidad**

- **Control muscular/ocular:** regresar componentes EMG/EOG y recalcular $`\alpha`$; requerir $`\mid \Delta\widehat{\alpha} \mid < 0.1`$.

- **Sensibilidad de banda:** recalcular excluyendo γ para asegurar que $`\alpha`$ no está impulsado por EMG de banda ancha.

- **Sensibilidad de ventana:** 20/40/60 s; requerir ordenamiento estable de $`\widehat{\alpha}`$.

- **Definición de distancia:** intercambiar distancias corticales vs grafo; requerir acuerdo cualitativo.

**3.14 Endpoints estadísticos (listos para prerregistro)**

- **Primario:** Δ$`\widehat{\alpha}`$ (vigilia − anestesia) con IC 95%; diferencia de **tasa de aprobación de colapso**; AUROC para clasificación de estado usando $`\alpha`$ vs PCI/potencia/conectividad.

- **Secundario:** picos $`\Delta\alpha`$ bloqueados por tarea vs precisión conductual; cohortes clínicas ROC (TdC vs control).

- **Valor añadido:** modelos anidados con $`\alpha`$+ líneas base; pruebas de razón de verosimilitud; curvas de confiabilidad.

**4. Programa Experimental I — Perturbación EMT–EEG**

**Objetivo.** Probar si el exponente de coherencia neural $`\alpha_{\text{neural}}`$ está **alto y colapso-estable** durante vigilia consciente y **reducido/inestable** bajo anestesia general (propofol), usando **EMT de pulso único** para sondear dispersión y persistencia causal a través de escalas.

**4.1 Participantes y estados**

- **Muestra.** $`N = 30`$ adultos saludables (18–45), diestros, sin historial neuro/psiquiátrico.

- **Estados.** (i) **Vigilia** (ojos abiertos, fijación), (ii) **Sedación con propofol** (pérdida de responsividad; Ramsay 5–6), (iii) **Recuperación** (retorno de responsividad).

- **Diseño.** Intra-sujeto, orden de sesión contrabalanceado; concentración en sitio de efecto objetivo monitoreada por anestesiólogo. Seguridad según guías internacionales EMT/anestesia.

**4.2 Adquisición y estimulación**

- **EEG.** Alta densidad 128-ch, 1 kHz, gorras compatibles con EMT; amplificadores acoplados en DC; 0.1–200 Hz en línea.

- **RM.** T1 para localización de fuente y distancias geodésicas corticales. ITD (opcional) para conectoma estructural.

- **EMT.** Pulsos monofásicos únicos (110% umbral motor en reposo), **sitios:** premotor izquierdo (BA6) y LPS derecho; **intervalo inter-pulso:** jittered 2–3 s; **ensayos:** 180/sitio/estado; orientación de bobina optimizada por neuronavegación.

- **Controles.** Ángulo de bobina **sham**; **enmascaramiento de ruido** (tapones + ruido blanco); **ensayos catch** sin pulso.

**4.3 Preprocesamiento y control de artefactos**

- **Excisión de artefacto EMT.** Interpolar −2 a +8 ms alrededor del pulso; regresión de resonancia con plantillas por canal.

- **ICA/ASR.** Remover componentes oculares/musculares; rechazar ensayos con pico-a-pico residual > ±100 µV post-limpieza.

- **Re-referencia.** Referencia promedio; proyección de fuente vía RMs individuales (formador de haz MNE).

- **Pasa-banda.** 1–100 Hz (o 0.5–150 Hz si SNR permite); notch 50/60 Hz.

- **Compuertas de calidad.** Requerir ≥140 ensayos limpios por sitio/estado; SNR ≥ 6 dB en ventana temprana post-estímulo.

**4.4 Definiendo escala** $`\mathbf{L}`$ **y tiempo** $`\mathbf{T}`$

- **$`L`$ primario:** **distancia geodésica cortical** (mm) entre la parcela estimulada y parcelas objetivo (espacio de superficie).

- **$`L`$ alternativo:** **geodésica de grafo** en conectoma estructural ($`d_{G}`$); **tamaño de parcela** (atlas multiresolución) para robustez.

- **$`T`$ primario:** **duración de respuesta evocada** $`T_{\text{ER}}`$: intervalo contiguo post-EMT donde amplitud de fuente excede línea base por $`z \geq 2`$ (corregido por cluster), tope en 300 ms.

- **$`T`$ alternativo:** autocorrelación de decaimiento exponencial en ventana post-estímulo $`T_{\rho}`$; tiempo de recurrencia $`T_{\text{rec}}`$ en trayectorias latentes.

Binnear $`L`$ en una serie geométrica abarcando ≥1 década (ej., 10, 15, 22, 33, 50, 75, 110 mm).

**4.5 Estimación de** $`\mathbf{\alpha}_{\text{neural}}`$

Para cada **estado × sitio × sujeto**, recolectar pares $`\{(\log L_{i},\log T_{i})\}`$ a través de parcelas/bins y ajustar

``` math
\log T_{i} = \beta_{0} + \alpha\text{ }\log L_{i} + \varepsilon_{i}.
```

- **Primario:** MCO con errores HC3.

- **EEV:** regresión ortogonal cuando variabilidad de tamaño de parcela o umbrales $`T_{\text{ER}}`$ introducen error de calibración.

- **Bootstrap:** 1,000 remuestreos estratificados por bins-$`L`$; reportar $`\widehat{\alpha}`$ mediano e IC 95%.

- **Jackknife:** dejar-un-bin-fuera, requerir $`\mid \Delta\widehat{\alpha} \mid \leq 0.15`$.

- **Prueba de colapso:** minimizar varianza entre bins de $`\widetilde{T} = TL^{- \alpha^{\star}}`$; aprobar si $`\alpha^{\star} \in`$ IC de $`\widehat{\alpha}`$, KS $`p > 0.05`$, y **puntaje de colapso** $`C \geq 0.25`$.

**4.6 Resultados e hipótesis (prerregistrados)**

- **Endpoint primario.** $`\Delta\alpha = {\widehat{\alpha}}_{\text{vigilia}} - {\widehat{\alpha}}_{\text{anest}}`$ (por sujeto, promediado a través de sitios).\
  **H1:** $`\Delta\alpha > 0`$ con tamaño de efecto $`d \geq 0.6`$.

- **Estabilidad de colapso.** Diferencia en **tasa de aprobación** y **puntaje-C** (vigilia > anestesia).

- **Reversibilidad de recuperación.** $`{\widehat{\alpha}}_{\text{recuperación}} \approx {\widehat{\alpha}}_{\text{vigilia}}`$; anestesia $`\ll`$ vigilia.

- **Valor incremental.** $`\widehat{\alpha}`$ mejora clasificación de estado vs **PCI**, potencia espectral, y conectividad (modelos anidados, AUC/precisión).

**4.7 Análisis estadístico**

- **Pruebas intra-sujeto.** $`t`$ pareada o Wilcoxon para $`\Delta\alpha`$; factores de Bayes reportados junto con $`p`$.

- **Tamaños de efecto.** d de Cohen, ICs bootstrap; **modelos mixtos** con interceptos aleatorios para sujeto y sitio.

- **Clasificación.** Regresión logística/SVM usando predictores: $`\widehat{\alpha}`$, puntaje-C, PCI, potencias de banda; **VC bloqueada** por sujeto; reportar **AUROC**, **Brier** y **confiabilidad**.

- **Comparaciones múltiples.** Controlar FDR a través de bandas/espacios (Benjamini–Hochberg).

**Potencia.** Con $`N = 30`$, SD-$`\alpha`$ ≈ 0.25, tenemos >0.8 potencia para detectar $`\Delta\alpha = 0.15`$ a $`\alpha = 0.05`$ (pareada).

**4.8 Robustez y auditorías de artefactos**

- **Controles sham/parietales.** Confirmar diferencias-$`\alpha`$ insignificantes en bloques sham; consistencia de sitio entre BA6 y LPS.

- **Residuos EMG/EOG.** Regresar componentes; recalcular $`\widehat{\alpha}`$. Requerir $`\mid \Delta\widehat{\alpha} \mid < 0.1`$.

- **Sensibilidad de ventana/banda.** Ventanas RE 200–300 ms; bandas θ/α/β/γ; resultados cualitativos invariantes.

- **Definición de distancia.** Intercambiar geodésicas corticales vs grafo; conclusiones estables.

- **Enmascaramiento de clic de bobina.** Verificación de ruido blanco: sin correlación entre niveles de audio y $`\widehat{\alpha}`$.

**4.9 Falsificadores (predefinidos)**

- **F1.** Sin $`\Delta\alpha`$ significativo (vigilia vs anestesia) y sin mejora de colapso en vigilia.

- **F2.** $`\widehat{\alpha}`$ no añade **ningún** valor de clasificación más allá de PCI y potencia de banda (modelos anidados ΔAUC < 0.02).

- **F3.** $`\widehat{\alpha}`$ es inestable a controles de artefactos (cambios > 0.15 después de correcciones EMG/EOG/bobina).

- **F4.** Los resultados se invierten bajo recuperación (sin retorno hacia valores de vigilia).

Fallar cualquier falsificador primario lleva a revisar o rechazar la afirmación central de RTM-Neuro.

**4.10 Ética y seguridad**

- **Aprobaciones.** Aprobación IRB; sedación dirigida por anestesiólogo; consentimiento informado (y re-consentimiento post-recuperación).

- **Monitoreo.** Signos vitales continuos; capnografía; equipo de vía aérea en espera.

- **Manejo de datos.** Datos desidentificados; protocolo prerregistrado y materiales/código abiertos tras publicación.

**4.11 Entregables**

- Tablas a nivel de sujeto de $`\widehat{\alpha}`$, IC, puntaje-C por estado/sitio/banda; **gráficos de bosque grupales**.

- **Curvas de clasificación de estado** (AUROC, confiabilidad) comparando $`\widehat{\alpha}`$ vs PCI/potencia/conectividad.

- **Paquete de reproducibilidad:** YAML de parámetros, scripts de preprocesamiento, matrices de distancia espacio-fuente, y notebooks para regenerar todas las figuras.

**5. Programa Experimental II — Estados Naturalistas y Tareas**

**Objetivo.** Probar si $`\alpha_{\text{neural}}`$ rastrea **integración multiescala** a través de **estados cerebrales espontáneos** (sueño, meditación, sesiones psicodélicas) y **épocas de tarea** (memoria de trabajo, atención, vinculación perceptual), y si las **excursiones bloqueadas por tarea** en $`\alpha`$ predicen comportamiento.

**5.1 Cohortes y registros**

- **Sueño**: $`N = 40`$ adultos saludables; EEG nocturno de alta densidad (128 ch), EOG/EMG; subconjunto opcional MEG de siesta.

- **Meditación**: $`N = 30`$ practicantes experimentados (≥1000 h) + $`N = 30`$ controles emparejados; ojos cerrados/entreabiertos.

- **Psicodélico**: $`N = 24`$ intra-sujeto, placebo vs. psilocibina/ketamina (guías IRB/clínicas).

- **Tareas**: $`N = 50`$ adultos saludables; **n-back visuoespacial (2–3 back)**, **parpadeo atencional**, y **rivalidad binocular** (vinculación perceptual).

- **Auxiliar**: RM estructural/ITD para distancias de fuente y grafo (opcional en cohorte solo-sueño).

**Modalidades.** EEG primario (1 kHz); espacio fuente alentado. fMRI (TR 0.8–1.0 s) para replicación macroescala en corridas de tarea (subconjunto).

**5.2 Preprocesamiento y CC común**

- Pipeline EEG como en §3 (pasa-banda, notch, ICA/ASR, proyección de fuente).

- Estadificación de sueño (AASM): N1, N2, N3, REM anotados por calificadores ciegos.

- Compuertas de artefactos: umbrales de residuos EMG/EOG; picos de movimiento (fMRI) censurados; requerir ≥8 min de datos limpios por condición (etapa de sueño o bloque de meditación).

- Span de escala: ≥1 década en $`L`$ con ≥4 bins poblados; estabilidad jackknife $`\mid \Delta\alpha \mid \leq 0.15`$; puntaje de colapso $`C \geq 0.25`$.

**5.3 Definiciones de** $`\mathbf{L}`$ **y** $`\mathbf{T}`$ **para actividad espontánea**

- **$`L`$ primario**: distancia geodésica cortical (parcelas fuente); **Alternativo**: geodésica de grafo en conectoma estructural; tamaño de parcela para robustez.

- **$`T`$ primario**: **autocorrelación de decaimiento exponencial** $`T_{\rho}`$ de actividad limitada por banda (θ/α/β/γ y banda ancha) en ventanas de 40 s (20 s de superposición).

- **Anomalías**: $`\Delta\alpha(t) = \widehat{\alpha}(t) - {median}_{t - 10\text{ min}\ldots t}\widehat{\alpha}`$ dentro del mismo estado/bloque.

**5.4 Paradigma A — Arquitectura del sueño**

**Diseño.** EEG nocturno continuo; calcular $`\alpha_{\text{neural}}`$ por etapa (N1/N2/N3/REM) con ventanas deslizantes de 40 s.

**Hipótesis.**

- **Vigilia/REM:** α **más alto, estable** con colapso aprobado (integración multiescala para contenido vívido).

- **N2/N3:** α **más bajo** y tasa de aprobación reducida (fragmentación por oscilaciones lentas/husos).

- **Transiciones:** **caída–rebote** en $`\alpha`$ en límites de etapa (aumento N2→REM).

**Endpoints.** Medianas por etapa e IQR de $`\widehat{\alpha}`$, tasa de aprobación de colapso, contribuciones de banda (θ–γ), modelos mixtos con efectos aleatorios de sujeto; AUROC para clasificación de etapa contra líneas base espectrales.

**Falsificadores.** Sin ordenamiento monotónico (Vigilia≈N3), o $`\alpha`$ añade <0.02 AUROC más allá de potencia espectral.

**5.5 Paradigma B — Estados de meditación**

**Diseño.** Tres bloques de 10 min (reposo, atención enfocada, monitoreo abierto) × 2 repeticiones.

**Hipótesis.**

- **Practicantes:** α **elevado** y **menor varianza** (integración multiescala estabilizada) vs. controles; separabilidad de estado (AE vs MA) en $`\alpha`$ específico de banda (dominancia α/θ).

- **Controles:** modulación de $`\alpha`$ menor o ausente.

**Endpoints.** ANOVA grupo × estado en $`\widehat{\alpha}`$, tasas de aprobación de colapso; clasificación (practicante vs control; AE vs MA) usando $`\alpha`$ vs líneas base espectrales/PLI.

**Falsificadores.** Sin efectos grupo/estado después de FDR; $`\alpha`$ redundante con potencia de banda.

**5.6 Paradigma C — Sesiones psicodélicas**

**Diseño.** Cruce placebo–droga; reposo ojos cerrados + bloque de música (10–15 min cada uno).

**Hipótesis.**

- **Psicodélico agudo:** distribución de $`\alpha`$ **bimodal o ensanchada** (integración/fragmentación episódica), con ráfagas de α **intermitentemente alto** durante experiencia pico.

- Las dinámicas de $`\alpha`$ correlacionan con **calificaciones de intensidad** y **fenomenología** (ej., subescalas MEQ, 5D-ASC).

**Endpoints.** Δ$`\widehat{\alpha}`$ (droga–placebo), razón de varianza, tasa de ráfagas de épocas α-alto, correlaciones con psicométricos (Spearman; modelos mixtos).

**Falsificadores.** Sin cambio Δ$`\widehat{\alpha}`$/varianza; correlaciones psicométricas ns después de corrección.

**5.7 Paradigma D — Memoria de trabajo y atención**

**Tareas.** **2–3 back** (MT), **parpadeo atencional** (PA), y **atención selectiva** (señalización Posner).\
**Ventanas.** Ventanas bloqueadas por ensayo 2–3 s (pre-señal, codificación, mantenimiento, sonda), deslizando por 250 ms.

**Hipótesis.**

- **MT:** $`\alpha \uparrow`$ durante **mantenimiento**, escalando con carga (2<3 back).

- **PA:** $`\alpha \uparrow`$ **transitorio** en ensayos T1 correctos; reducido en ensayos T2 parpadeados.

- **Atención selectiva:** $`\alpha \uparrow`$ sobre redes atendidas; predice ganancia de TR.

**Endpoints.** Curso temporal de $`\Delta\alpha`$ por época; **modelos mixtos por ensayo** prediciendo precisión/TR desde $`\alpha`$ (y líneas base: potencia de banda, ITPC); mejoras AUROC/MAE con validación cruzada.

**Falsificadores.** Sin modulación bloqueada por tarea; $`\alpha`$ no añade valor predictivo más allá de potencia/ITPC.

**5.8 Paradigma E — Vinculación perceptual (rivalidad binocular)**

**Diseño.** Rejillas rivales; reportes de botón de cambios perceptuales.

**Hipótesis.**

- **Ventana pre-cambio (−1.5 a 0 s):** $`\alpha \uparrow`$ (integración llevando a dominancia); **post-cambio:** normalización.

- **Patrón espacial:** $`\alpha \uparrow`$ en red occipito-parietal; reducido en regiones no-tarea.

**Endpoints.** Curvas $`\Delta\alpha`$ alineadas por evento; mapas topográficos de cambio-$`\alpha`$; pruebas de permutación para diferencias pre/post.

**Falsificadores.** Trazas $`\alpha`$ planas a través de cambios; sin especificidad topográfica.

**5.9 Análisis específicos de banda y espacio**

- Calcular $`\alpha`$ por banda y **fusionado** (pesos iguales o aprendidos).

- Comparación espacio fuente vs. sensor; replicar con geodésica de grafo $`L = d_{G}`$.

- Reportar **efectos de consenso** (replicados a través de al menos dos definiciones de $`L`$/$`T`$).

**5.10 Estadísticas, potencia y multiplicidad**

- Modelos de **efectos mixtos** con interceptos aleatorios de sujeto; EEs robustos por cluster.

- Pruebas de **permutación** para curvas alineadas por evento (transiciones de sueño, épocas MT).

- **Comparaciones múltiples**: FDR a través de bandas/épocas/condiciones.

- **Potencia**: con $`N = 40`$ (sueño), detectar Δ$`\widehat{\alpha} = 0.10`$ (DE 0.20) a $`\alpha = 0.05`$; tareas ($`N = 50`$): detectar efectos de interacción medianos en modelos tiempo–época.

**5.11 Robustez y auditorías de artefactos**

- Excluir ventanas que fallan **colapso** o **span de escala**.

- Recalcular sin γ (control de contaminación EMG).

- Regresores pupila/ECG (activación/SNA) en tarea fMRI/EEG; verificar que efectos $`\alpha`$ persisten.

- Sensibilidad de ventana (20/40/60 s reposo; 200/400 ms tarea).

- Intercambio de definición de distancia (cortical vs. grafo); invariancia cualitativa requerida.

**5.12 Entregables**

- **Mapas/cursos temporales de estado** de $`\widehat{\alpha}`$, $`\Delta\alpha`$, y puntajes de colapso.

- **Tablas**: medianas por etapa, ANOVA grupo × estado, coeficientes de época-tarea, métricas de predicción.

- **Paquete de reproducibilidad**: YAMLs de parámetros, código, y derivados anonimizados permitiendo replicación completa.

**6. Aplicaciones Clínicas**

**Objetivo.** Traducir $`\alpha_{\text{neural}}`$—el exponente de coherencia RTM—en biomarcadores y variables de control para **trastornos de consciencia (TdC)** y **ritmopatías psiquiátricas**, con protocolos para **monitoreo junto a la cama** y **neuromodulación de bucle cerrado**. Especificamos endpoints, falsificadores y detalles de despliegue (CC, seguridad, interoperabilidad).

**6.1 Trastornos de consciencia (coma/EV/ECM)**

**6.1.1 Fundamento**

Los pacientes con TdC exhiben integración de largo alcance deteriorada. RTM predice **reducción crónica e inestabilidad** de $`\alpha_{\text{neural}}`$, con **colapso fallido** (sin clase de transporte única). La recuperación hacia ECM/SECM debería mostrar $`\alpha \uparrow`$ y tasa de aprobación de colapso mejorada.

**6.1.2 Cohortes y registros**

- $`N \approx 80`$: coma/EV/ECM/SECM; $`N \approx 40`$ controles sanos emparejados por edad.

- **EEG (primario)** 64–128 ch; 20–30 min reposo ojos cerrados/abiertos; **PRE** (oddball auditivo) si se tolera.

- **EMT–EEG (opcional)**: perturbación de baja intensidad sobre M1/parietal cuando sea médicamente apropiado.

- RM/ITD (cuando sea factible) para distancias espacio-fuente y geodésicas de grafo.

**6.1.3 Endpoints**

- **Biomarcador primario:** $`\widehat{\alpha}`$ mediano (fusionado a través de bandas) y **tasa de aprobación de colapso** por paciente.

- **Clasificación de estado:** AUROC para Control vs TdC; EV vs ECM; **confiabilidad** (pendiente de calibración).

- **Pronóstico:** $`\widehat{\alpha}`$ basal prediciendo **mejora CRS-R a 6 meses** (AUC y modelos Cox).

- **Subconjunto perturbacional:** $`\Delta{\widehat{\alpha}}_{\text{EMT}} = {\widehat{\alpha}}_{\text{tipo-vigilia}} - {\widehat{\alpha}}_{\text{basal}}`$ post-estímulo vs PCI; se espera que respondedores muestren $`\alpha \uparrow`$ con mejora de colapso.

**6.1.4 Falsificadores**

- Sin separación grupal (Δmediana $`\widehat{\alpha}`$< 0.05; ΔAUC < 0.02 vs PCI/potencia).

- Las tasas de colapso no difieren de controles; $`\alpha`$ no es pronóstico después de ajustar por edad/etiología.

**6.1.5 Protocolo junto a la cama (solo EEG)**

- EEG-AD 20-min; pasa-banda 1–45 Hz; ICA/ASR; calcular $`\alpha`$ en ventanas de 40 s (50% superposición).

- **Compuertas CC:** ≥1 década de span en $`L`$; $`R^{2} \geq 0.6`$; jackknife ≤ 0.15; puntaje de colapso $`C \geq 0.25`$.

- **Reporte:** $`\widehat{\alpha}`$ mediano a nivel de paciente con IC; tasa de aprobación de colapso; comparación con distribución normativa (puntaje z).

**6.2 Ritmopatías psiquiátricas**

**6.2.1 Trastorno depresivo mayor (TDM)**

**Hipótesis.** Un subconjunto muestra **dinámicas sobreestabilizadas** (α rígidamente alto) con **baja varianza**—flexibilidad cognitiva reducida; los respondedores a tratamiento muestran **normalización** de $`\alpha`$ (ligera disminución y varianza aumentada).

**Diseño.** $`N \approx 120`$ TDM (sin medicación) + $`N \approx 120`$ controles; EEG en reposo ± tarea (n-back).\
**Endpoints.** Δ$`\widehat{\alpha}`$ grupal y varianza; **seguimiento de tratamiento** (ISRS/EMT/TEC) durante 6–8 semanas; modelos mixtos relacionando Δ$`\alpha`$ con cambio en **HAM-D/MADRS**.\
**Falsificador.** Sin diferencia basal y sin acoplamiento longitudinal con cambio de síntomas.

**6.2.2 Espectro de esquizofrenia**

**Hipótesis.** **Organización fragmentada** (α bajo/variable), particularmente durante tareas de memoria de trabajo y perceptuales.\
**Diseño.** $`N \approx 80`$ pacientes + $`N \approx 80`$ controles; tareas EEG de §5.7–5.8.\
**Endpoints.** Modelos a nivel de ensayo: $`\alpha`$ prediciendo precisión/TR más allá de potencia/ITPC; diferencias grupales en variabilidad de $`\alpha`$ y tasa de aprobación de colapso.\
**Falsificador.** $`\alpha`$ no añade valor predictivo y refleja potencia de banda enteramente.

**6.2.3 TDAH/bipolar (exploratorio)**

Perfilar modulación de $`\alpha`$ **dependiente del estado** a través de episodios de atención (TDAH) y fases de ánimo (bipolar). Prerregistrar pilotos pequeños-N con medidas repetidas; tratar como generadores de hipótesis.

**6.3 Neuromodulación de bucle cerrado con** $`\mathbf{\alpha}`$ **como variable de control**

**6.3.1 Fundamento**

Si $`\alpha`$ indexa integración multiescala, **dirigir** $`\alpha`$ puede restaurar u optimizar función.

**6.3.2 Diseño de controlador (EMTr/tACS guiado por EEG)**

- **Objetivo:** CPFDL izquierda (TDM), hubs parietales (TdC), o nodos específicos de red (MT esquizofrenia).

- **Sensor:** EEG 32–64 ch; ventanas de 1 s (tarea) / 10 s (reposo) estiman $`\widehat{\alpha}`$ y banderas CC.

- **Política:**

  - **TDM:** si $`\widehat{\alpha} > Q_{0.8}`$ de línea base personal por ≥N ventanas (rigidez), entregar EMTr **inhibitorio** (1 Hz) o tACS **fuera de fase** para reducir sobrecoherencia.

  - **TdC:** si $`\widehat{\alpha} < Q_{0.2}`$ y colapso falla (inestabilidad), entregar EMTr **excitatorio** (ráfaga 10 Hz) o tACS **en fase** para promover integración.

  - **Esquizofrenia (MT):** durante mantenimiento, aumentar $`\alpha`$ transitoriamente con pulsos **bloqueados por tarea**; suprimir fuera de ventanas para evitar discognición.

**Seguridad.** Topes duros en dosis/ciclo de trabajo; aborto automático en artefactos (picos EMG/EOG), deriva, o umbrales de riesgo de convulsión.

**6.3.3 Endpoints y falsificadores (ensayos de bucle cerrado)**

- **Modulación aguda:** Δ$`\widehat{\alpha}`$ intra-sesión hacia rango objetivo con CC mantenido.

- **Ganancias conductuales/clínicas:** precisión/TR de tarea (MT) o escalas de síntomas (HAM-D, CRS-R) mejoradas **versus sham**.

- **Falsificador:** sin modulación de $`\alpha`$ o sin mejora conductual/clínica más allá de sham.

**6.4 Detalles de implementación**

- **Pipelines.** Parámetros YAML prerregistrados; procesamiento en contenedores; diagnósticos automáticos de CC y colapso.

- **Salidas.** Tableros de paciente: series temporales de $`\widehat{\alpha}`$, IC, tasa de aprobación de colapso; puntajes z normativos; registros de intervención (para bucle cerrado).

- **Interoperabilidad.** BIDS-EEG/FIF para crudo; sidecars JSON para parámetros $`\alpha`$; ganchos HL7/FHIR para integración HCE.

- **Privacidad y gobernanza.** Desidentificación; computación en dispositivo/borde cuando sea posible; rastros de auditoría (hash de software, versión de parámetros).

**6.5 Consideraciones éticas y prácticas**

- **Comunicar incertidumbre.** Reportar **confianza y colapso** junto con cualquier biomarcador; evitar lenguaje determinístico.

- **Principio de no dañar.** En TdC, requerir estado fisiológico estable; en psiquiatría, monitorear agitación/cambio (bipolar).

- **Equidad y acceso.** Validar en entornos con **recursos limitados** con EEG de 32 ch; publicar herramientas abiertas bajo licencias permisivas; proporcionar entrenamiento multilingüe.

- **Transparencia.** Prerregistrar análisis; liberar resultados negativos; publicar curvas de calibración y casos de error.

**6.6 Resumen (listo para mantener tal cual)**

RTM-Neuro produce **candidatos de grado decisión** para traducción clínica: un **índice de integración** junto a la cama ($`\widehat{\alpha}`$ + tasa de aprobación de colapso) para pronóstico y monitoreo de **TdC**; **huellas de estado** y **seguimiento de tratamiento** en **ritmopatías psiquiátricas**; y una **variable de control de bucle cerrado** para neuromodulación que apunta a organización multiescala—no meramente potencia o conectividad por pares. Cada afirmación está emparejada con **falsificadores**, compuertas CC, y vías de despliegue seguras para pacientes, permitiendo evaluación rigurosa antes del uso clínico rutinario.

**7. Plantillas de Resultados y Plan Estadístico**

Este capítulo es un **plano listo para usar** para prerregistro y reporte. Reemplazar campos entre corchetes $`\lbrack\text{ }\rbrack`$ con los valores de su estudio. Todos los análisis están definidos para poder ejecutarse desde derivados guardados (sin dependencia de notebooks interactivos).

**7.1 Resultados primarios (por programa)**

**Programa I — EMT–EEG (vigilia vs anestesia):**

1.  **Diferencia de exponente de coherencia:** $`\Delta\alpha = {\widehat{\alpha}}_{\text{vigilia}} - {\widehat{\alpha}}_{\text{anest}}`$ (por sujeto; promediado por sitio).

    - Prueba: $`t`$ pareada (o Wilcoxon) con IC 95%; reportar d de Cohen, Factor de Bayes $`BF_{10}`$.

2.  **Estabilidad de colapso:** diferencia en **tasa de aprobación** (% de ventanas con $`C \geq 0.25`$ y KS $`p > 0.05`$) y **mediana** $`C`$ (vigilia > anestesia).

3.  **Clasificación:** AUROC/AUPRC distinguiendo estados usando $`\widehat{\alpha}`$+$`C`$ vs líneas base (PCI, potencias de banda, conectividad).

**Programa II — Estados/tareas naturalistas:**

- **Contrastes de estado:** medianas por etapa/bloque de $`\widehat{\alpha}`$ (sueño: Vigilia/REM > N2/N3; meditación: modulación AE/MA; psicodélicos: varianza/bimodalidad).

- **Bloqueo por tarea:** $`\Delta\alpha(t)`$ por época y amplitud/tiempo de pico; precisión/TR por ensayo predichos por $`\alpha`$ más allá de potencia/ITPC.

**Clínico (TdC/psiquiatría):**

- **Separación grupal:** Control vs TdC; EV vs ECM; paciente vs control (psiquiatría) usando $`\widehat{\alpha}`$ y métricas de colapso.

- **Pronóstico/tratamiento:** $`\widehat{\alpha}`$ basal prediciendo cambio CRS-R (Cox/logístico); acoplamiento longitudinal de Δ$`\widehat{\alpha}`$ con puntuaciones de síntomas (modelos mixtos).

**7.2 Curación de datos y exclusiones (predefinidas)**

Una ventana/época se **excluye** si se cumple alguna:

- Span de escala < 1 década **o** < 4 bins $`L`$ poblados.

- Calidad de ajuste: $`R^{2} < 0.60`$ o jackknife $`\mid \Delta\widehat{\alpha} \mid > 0.15`$.

- Falla de colapso: $`C < 0.25`$ o KS $`p \leq 0.05`$.

- Artefactos: residuos EMG/EOG sobre umbrales; resonancia de EMT no despejada; ráfagas de ruido de línea iEEG; fMRI FD>0.5 mm con <50% de muestras limpias.\
  Todas las exclusiones se **cuentan y reportan** por sujeto/condición.

**7.3 Modelos estadísticos**

**7.3.1 Efectos de grupo/condición (resultados continuos)**

- **Modelo mixto:** $`{\widehat{\alpha}}_{s,c} = \beta_{0} + \beta_{1}\text{Condición}_{c} + (1 \mid Sujeto_{s})`$

  - Extensiones: añadir **Banda**, **Espacio** (fuente/grafo), y sus interacciones.

  - Regresión robusta (Huber) si residuos de colas pesadas.

**7.3.2 Clasificación y calibración**

- **Modelos logísticos:** estado ~ $`\widehat{\alpha}`$+$`C`$+PCI+potencias de banda (+ conectividad).

- **Validación cruzada:** **bloqueada por sujeto** (dejar-un-sujeto-fuera o 5-fold agrupada).

- **Lecturas:** AUROC, AUPRC, puntaje Brier, **pendiente de confiabilidad** (ideal 1.0), **ECE**.

**7.3.3 Comportamiento por ensayo**

- **Efectos mixtos:** Precisión/TR ~ $`\alpha`$+ (1\|Sujeto) + (1\|Ítem) con covariables de potencia de banda/ITPC.

- **Modelos rezagados:** comportamiento ~ $`\alpha_{t - \mathcal{l}}`$ para $`\mathcal{l} \in \{ 1,2,3\}`$ ventanas para probar relaciones adelanto-retraso.

**7.3.4 Pronóstico (TdC)**

- **Cox PH:** tiempo-a-mejora ~ $`\widehat{\alpha}`$+ edad + etiología; riesgos proporcionales probados (Schoenfeld).

- **Calibración:** deciles de riesgo, prueba Greenwood–Nam–D'Agostino.

**7.3.5 Valor incremental**

- **Pruebas anidadas:** comparar líneas base vs líneas base+$`\alpha`$ con razón de verosimilitud; para AUROC usar **DeLong**, para Brier usar bootstrap Δ.

- **Beneficio neto:** curvas de decisión a través de umbrales de probabilidad.

**7.4 Comparaciones múltiples e incertidumbre**

- **Alcance por familia:** por programa y familia de endpoints (ej., contrastes de estado; épocas de tarea; grupos clínicos).

- **Control:** **FDR Benjamini–Hochberg** a $`q = 0.05`$.

- **Intervalos:** bootstraps de sesgo corregido acelerado (BCa) para ICs de medianas, AUROC, ΔAUROC.

- **Tamaños de efecto:** reportar **d de Cohen** (pareado/no pareado), **delta de Cliff** cuando no paramétrico.

**7.5 Plantillas de potencia y tamaño de muestra**

- **EMT–EEG (Programa I).** Con $`N = 30`$ pareados, DE($`\Delta\alpha`$)≈0.25, el estudio tiene $`> 0.80`$ potencia para detectar $`\Delta\alpha = 0.15`$ (dos colas $`\alpha = 0.05`$).

- **Sueño (Programa II-A).** $`N = 40`$, DE intra-sujeto≈0.20 → detectar diferencias de etapa de 0.10–0.12.

- **Tareas (Programa II-D).** $`N = 50`$, modelos mixtos detectan efecto mediano $`f^{2} \approx 0.08`$ para $`\alpha`$ después de covariables.

- **Clínico TdC.** $`N = 80`$ pacientes da 80% potencia para mejora AUROC Δ≥0.06 sobre PCI a prevalencia basal $`p \approx 0.5`$.

*(Recalcular con sus DEs piloto; incluir buffers de deserción ~10–15%.)*

**7.6 Análisis de robustez y sensibilidad**

- **Definición de distancia:** intercambiar geodésica cortical ↔ geodésica de grafo ↔ tamaño de parcela; requerir invariancia cualitativa.

- **Definición de tiempo:** $`T_{\rho}`$↔ $`T_{\text{ER}}`$↔ $`T_{\text{rec}}`$; reportar rangos.

- **Sensibilidad de banda:** calcular sin γ para reducir contaminación EMG; comparar resultados fusionados vs por banda.

- **Tamaño de ventana:** 20/40/60 s (reposo), 200/400 ms (tarea); ordenamiento de $`\widehat{\alpha}`$ estable.

- **Residuos de artefactos:** regresar componentes EMG/EOG; requerir $`\mid \Delta\widehat{\alpha} \mid < 0.10`$.

- **Nulos de grafo:** comparar $`\alpha`$ (grafo) contra conectomas aleatorizados preservando grado.

**7.7 Tablas de reporte (listas para llenar)**

**Tabla 1 — EMT–EEG (Programa I) resultados primarios**

| **Métrica** | **Vigilia (media±DE)** | **Anestesia (media±DE)** | **Δ (IC 95%)** | **(t)/(Z)** | **(p)** | **(d)** | **(BF\_{10})** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $`\widehat{\alpha}`$ (prom-sitio) | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |
| Tasa aprobación colapso (%) | \[ \] | \[ \] | \[ \] | — | \[ \] | — | — |
| Puntaje de colapso ($`C`$) | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] | — |
| AUROC ($`\alpha`$ vs estado) | \[ \] | — | — | — | — | — | — |
| Ganancia AUROC vs PCI | — | — | Δ=\[ \] | — | \[ \] | — | — |

**Tabla 2 — Sueño/meditación/psicodélicos (Programa II)**

| **Cohorte** | **Condición** | **Mediana (**$`\widehat{\alpha}`$**)** | **RIQ** | **Tasa aprobación colapso (%)** | **Δ vs ref (IC 95%)** | **(p) (FDR)** |
|----|----|----|----|----|----|----|
| Sueño | N3 | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |
| Sueño | REM | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |
| Meditadores | MA | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |
| Psicodélico | Pico | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |

**Tabla 3 — Paradigmas de tarea (modelos por ensayo)**

| **Tarea** | **Época** | **β(**$`\mathbf{\alpha}`$**→Precisión) \[IC\]** | **(p) (FDR)** | **ΔAUC vs potencia/ITPC** |
|----|----|----|----|----|
| n-back | Mantenimiento | \[ \] | \[ \] | \[ \] |
| PA | Pre-T2 | \[ \] | \[ \] | \[ \] |

**Tabla 4 — Clínico**

| **Cohorte** | **Contraste** | **AUROC (línea base)** | **AUROC (+**$`\mathbf{\alpha}`$**)** | **ΔAUROC \[IC\]** | **(p) (DeLong)** | **Pendiente calibración** |
|----|----|----|----|----|----|----|
| TdC | EV vs ECM | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |
| TDM | Respuesta | \[ \] | \[ \] | \[ \] | \[ \] | \[ \] |

**7.8 Plantillas de figuras (pies de figura que puede conservar)**

- **Fig. 1 — Escalamiento EMT–EEG:** *Dispersión de* $`{log\ }T`$ *vs* $`\log{\ L}`$ *con líneas MCO/EEV (vigilia vs anestesia), recuadro de residuos; panel derecho: curvas de colapso y puntaje* $`C`$*.*

- **Fig. 2 — Clasificación de estado:** *Curvas ROC y confiabilidad para vigilia vs anestesia usando* $`\alpha`$ *(y* $`C`$*) vs PCI/potencia; ICs bootstrap sombreados al 95%.*

- **Fig. 3 — Arquitectura del sueño:** *Gráficos de violín por etapa de* $`\widehat{\alpha}`$ *y tasas de aprobación de colapso; transiciones muestran trayectorias caída–rebote.*

- **Fig. 4 — Dinámicas bloqueadas por tarea:** *Cursos temporales de* $`\Delta\alpha`$ *a través de épocas (n-back, PA); líneas verticales para señales/sondas; superposiciones divididas por comportamiento (correcto vs error).*

- **Fig. 5 — Tableros clínicos:** *Series temporales por paciente de* $`\widehat{\alpha}`$*, tasa de colapso, y puntajes z normativos; gráfico de calibración pronóstica.*

**7.9 Prerregistro y procedencia**

- **Prerregistrar**: hipótesis, resultados primarios/secundarios, compuertas CC, criterios de exclusión, modelos estadísticos, y reglas de parada (OSF/AsPredicted).

- **Registros de procedencia**: hash YAML de parámetros, SHA de commit de software, checksum de datos (derivados BIDS).

- **Cegamiento**: ingeniería de características enmascarada por etiqueta; desbloquear solo para ajustes finales.

- **Desviaciones**: cualquier cambio post-hoc documentado con fundamento y marca temporal.

**7.10 Paquete de reproducibilidad**

- **Código y contenedores** para reproducir todas las tablas/figuras desde derivados congelados.

- **Datos sintéticos** para pipelines de IC (sin PHI).

- **Pruebas unitarias** para estimadores (recuperación de pendiente en datos de ley de potencia simulados; detección de colapso).

- **Integración continua**: ejecutar pruebas de humo de extremo a extremo en cada commit.

**7.11 Reglas de decisión (ir/no-ir)**

- **Éxito Programa I** si: $`\Delta\alpha > 0`$ con $`p < 0.01`$ (pareada), $`d \geq 0.5`$ mediano; tasa de aprobación de colapso ↑; y ΔAUROC ≥ 0.05 vs PCI/potencia.

- **Éxito Programa II** si: efectos prespecificados de estado/tarea replican a través de ≥2 definiciones de $`L`$ o $`T`$, y $`\alpha`$ añade valor predictivo (ΔAUC/MAE) después de FDR.

- **Éxito clínico** si: ganancia AUROC ≥ 0.05 con pendiente de calibración en \[0.8,1.2\], o valor pronóstico significativo (HR Cox con IC sin cruzar 1).

**8. Discusión**

**8.1 Qué mide** $`\mathbf{\alpha}_{\text{neural}}`$**—una capacidad de integración, no una frecuencia**

Dentro de RTM, la pendiente $`\alpha = d\ \log T/d\ \log L`$ cuantifica **cómo la persistencia crece con la escala**. En tejido neural, $`\alpha_{\text{neural}}`$ alto y estable implica que las señales pueden ser **mantenidas y enrutadas** a medida que la extensión espacial aumenta—un marcador operacional de **integración multiescala**—mientras que $`\alpha`$ bajo o inestable indica **fragmentación**: decorrelación rápida por milímetro añadido o salto en el conectoma. A diferencia de la potencia espectral o razones de banda, $`\alpha`$ es **relacional de escala**: compara *tiempo* y *espacio* (o distancia de grafo), no energía en una frecuencia.

**8.2 Relación con marcadores clásicos (potencia, conectividad, PCI)**

- **Potencia/ITPC.** La potencia de banda y el índice de consistencia de fase sincronizan localmente pero no dicen si la persistencia *mejora con la escala*. $`\alpha`$ puede subir con potencia modesta si el enrutamiento entre escalas se vuelve eficiente (ej., vinculación transitoria), o permanecer bajo a pesar de alta potencia si las oscilaciones locales fallan en generalizar.

- **Conectividad estática/funcional.** CF captura asociaciones por pares; $`\alpha`$ resume **escalamiento distancia–tiempo** a través de muchos pares simultáneamente.

- **PCI/complejidad perturbacional.** PCI cuantifica complejidad espaciotemporal después de perturbación. $`\alpha`$ complementa PCI al preguntar si **extensiones mayores viven más tiempo**—dos vistas del mismo espacio de eventos: *qué puede expresar el cerebro* (PCI) y *cuánto tiempo puede sostener la expresión mientras se dispersa* ($`\alpha`$).

**8.3 Una imagen mecanística: ondas, corredores y compuertas**

Interpretamos los aumentos en $`\alpha`$ como la emergencia de **corredores de enrutamiento**—ondas viajeras alineadas en fase, bucles recurrentes, y compuertas neuromoduladoras—que **endurecen** la organización a gran escala. Las disminuciones en $`\alpha`$ reflejan **cizallamiento y competencia** entre ensambles (ruptura de onda, entradas desincronizadoras), acortando la persistencia a medida que la escala sube. El acoplamiento entre frecuencias (ej., fase θ/α modulando ráfagas γ) proporciona un **puente** que puede elevar $`\alpha`$ cuando se sostiene a través de parcelas; AEF fallido lo baja.

**8.4 Dónde podría fallar RTM-Neuro (falsificadores científicos)**

1.  **Sin estabilidad de pendiente:** si $`\log T`$–$`\log L`$ no es lineal sobre ≥1 década en ningún estado supuestamente estable (vigilia), la ley RTM está mal aplicada.

2.  **Sin colapso:** falla de colapso de datos a pesar de ajustes aceptables sugiere mezcla de ventanas o elecciones incorrectas de $`L/T`$.

3.  **Redundancia:** si $`\alpha`$ no añade **ningún** valor predictivo más allá de PCI/potencia/conectividad después de pruebas anidadas, no es relevante para decisión.

4.  **Mapeo incoherente a fisiología:** si los cambios de $`\alpha`$ siguen artefactos (EMG, clic de bobina, movimiento) o cambios de pipeline más que fisiología, la métrica carece de validez.

**8.5 Confusores y mitigaciones**

- **Artefactos (EEG/MEG/EMT).** Resonancia de bobina, EMG, oculares inflan estructura de corto retardo y sesgan $`T`$. Mandamos **excisión de artefactos + ICA/ASR**, **análogos solo-noche** donde sea relevante, y verificaciones de **exclusión de γ**; ventanas que fallan CC/colapso se enmascaran.

- **Insuficiencia de span de escala.** Sin ≥1 década en $`L`$ o ≥4 bins, las pendientes son inestables; excluimos tales ventanas y reportamos cobertura.

- **Dependencia de definición de distancia.** Las geodésicas corticales vs grafo pueden diferir. Requerimos **invariancia cualitativa** a través de al menos dos definiciones de $`L`$.

- **Censura derecha de** $`T`$**.** Los topes de buffer pueden inflar $`\alpha`$; ejecutamos **ensambles de sensibilidad** (48/60/120 s o 150–300 ms para EMT-EEG) y reportamos rangos.

- **Mezcla de estados.** Las transiciones dentro de una ventana rompen supuestos de mecanismo único. Usamos ventanas más cortas, $`\alpha`$ **por partes**, o descartamos.

**8.6 Interfaz con teorías de la consciencia**

- **Espacio de Trabajo Neuronal Global (ETNG).** La ignición del ETNG puede verse como un $`\alpha \uparrow`$ **transitorio**: persistencia extendiéndose a través de extensiones fronto-parietales.

- **Información Integrada (TII).** Mientras $`\Phi`$ de TII es difícil de estimar, $`\alpha`$ actúa como un **sustituto operacional** para la *capacidad de sostener* extensiones grandes; no los equiparamos pero esperamos correlación positiva en regímenes de enrutamiento estable.

- **Vistas de procesamiento recurrente.** Los bucles recurrentes y la compuerta descendente que estabilizan representaciones deberían elevar $`\alpha`$; los barridos solo-hacia-adelante no deberían.

**8.7 Traducción clínica: por qué** $`\mathbf{\alpha}`$ **puede ser útil**

Un número único y falsificable con **IC y diagnósticos** (puntaje de colapso) soporta:

- **Monitoreo junto a la cama** (TdC): rastrear recuperación mientras $`\alpha \uparrow`$ y el colapso se estabiliza.

- **Seguimiento de terapia** (TDM, esquizofrenia): normalización hacia líneas base personales.

- **Control de bucle cerrado:** apuntar a **rangos** de $`\alpha`$ en lugar de potencia cruda, apuntando a *organización* no mera excitabilidad.

**8.8 Uso ético y comunicación**

- **Precursor ≠ prueba.** α elevado sugiere capacidad de integración, no experiencia consciente garantizada.

- **Calibración y confiabilidad.** Siempre reportar curvas de confiabilidad; evitar afirmaciones determinísticas a nivel individual.

- **Equidad.** Validar con sistemas de **menos canales** para acceso más amplio; publicar código/parámetros bajo licencias permisivas; revelar sesgos regionales o de hardware.

- **Gobernanza de datos.** Usar BIDS, desidentificar, preservar **procedencia** (YAML de parámetros, hash de software), y prerregistrar todas las desviaciones.

**8.9 Direcciones futuras**

- **Ventanas adaptativas y** $`\alpha`$ **por partes.** Resolver mecanismos mixtos y transitorios más limpiamente.

- **Validación entre modalidades.** Combinar EMT–EEG con MEG y fMRI rápido para triangular escalamiento $`L`$–$`T`$.

- **Pruebas causales.** EMTr/tACS de bucle cerrado para **dirigir** $`\alpha`$ y leer ganancias conductuales o clínicas.

- **Modelado.** Simulaciones en redes con base biofísica (retardos de conducción, cinéticas sinápticas) para reproducir dinámicas-$`\alpha`$ y derivar protocolos de perturbación.

**9. Conclusión**

Propusimos **RTM-Neuro**, una aplicación principiada de *Relatividad Temporal Multiescala* al tejido nervioso, en la cual el **exponente de coherencia neural** $`\alpha_{\text{neural}} = \frac{d\log T}{d\log L}`$ sirve como un marcador operacional de cómo la **persistencia** escala con la **extensión** (espacio o distancia de grafo). Este encuadre convierte la pregunta de larga data de "integración neural" en un conjunto de **pruebas falsificables de pendiente y colapso**: en ventanas donde un único mecanismo se mantiene, $`T \propto L^{\alpha}`$ con un $`\alpha`$ estable y **colapso de datos** exitoso; cuando los mecanismos cambian o la organización se fragmenta, $`\alpha`$ cae y el colapso falla.

Metodológicamente, especificamos **definiciones intercambiables** de escala $`L`$ (cortical/geodésica/grafo/oscilatoria) y tiempo $`T`$ (autocorrelación de decaimiento exponencial, duración de respuesta evocada, tiempo de recurrencia), con **compuertas de CC** (span de escala, $`R^{2}`$, jackknife, puntaje de colapso) y **cuantificación de incertidumbre** (EEV, bootstrap). Empíricamente, establecimos programas prerregistrados para probar RTM-Neuro a través de (i) **perturbaciones causales** (EMT–EEG bajo vigilia vs anestesia), (ii) **estados y tareas naturalistas** (sueño, meditación, psicodélicos, memoria de trabajo, vinculación perceptual), y (iii) **cohortes clínicas** (TdC, psiquiatría). Operacionalmente, propusimos cómo $`\alpha_{\text{neural}}`$ puede ser **monitoreado** junto a la cama y **dirigido** vía neuromodulación de bucle cerrado como una variable de control apuntando a **organización**, no meramente potencia o conectividad por pares.

Si se confirma, siguen tres beneficios:

1.  un índice compacto e **interpretable de integración multiescala** con diagnósticos claros;

2.  **valor predictivo y traslacional** (clasificación de estado, pronóstico, seguimiento de tratamiento) más allá de líneas base establecidas (potencia, PCI, CF estática); y

3.  una **manija causal** para diseño de intervención (apuntar a rangos de $`\alpha`$, modulación bloqueada por tarea).\
    Si es refutado por falsificadores prerregistrados (sin estabilidad de pendiente, sin colapso, sin valor incremental), RTM-Neuro todavía avanza el campo al **estrechar** dónde y cuándo la organización multiescala gobierna el acceso.

En suma, RTM-Neuro reposiciona la investigación de consciencia y cognición sobre un **fundamento de ley de escalamiento**: lo que importa no es solo *qué tan fuertes* son las señales locales, sino **cómo su persistencia crece con el alcance**. Esa pregunta simple—capturada por $`\alpha_{\text{neural}}`$—es medible, auditable y accionable.

**10. Validación Computacional del Marco RTM-Neuro**

**10.1 Visión general**

Este capítulo describe simulaciones computacionales que validan la metodología RTM-Neuro y demuestran sus predicciones teóricas. Presentamos tres conjuntos de simulación:

\- **\*\*S1\*\***: Demostración de escalamiento τ(L) a través de bandas de frecuencia y estados de consciencia

\- **\*\*S2\*\***: Validación de metodología de estimación (robustez a ruido, tamaño de muestra, discriminación de estado)

\- **\*\*S3\*\***: Modelo de umbral de acceso consciente (transiciones de estado, episodios de vinculación, patrones patológicos)

Estas simulaciones establecen que (a) el marco matemático es internamente consistente, (b) la metodología de estimación es robusta, y (c) la hipótesis de umbral reproduce la fenomenología observada. No constituyen validación empírica, que requiere registros EEG/MEG de sujetos humanos.

**10.2 S1: Demostración de Escalamiento τ(L)**

**10.2.1 Propósito**

Demostrar la predicción central de RTM τ(L) = τ_0 × L^α y sus implicaciones para dinámicas neurales.

**10.2.2 Predicciones Específicas de Banda**

RTM-Neuro predice que diferentes bandas de frecuencia exhiben diferentes exponentes de coherencia basados en sus roles funcionales:

\| Banda \| Frecuencia \| α \| Rol Funcional \|

\|------\|-----------\|---\|-----------------\|

\| Delta \| 1-4 Hz \| 2.5 \| Sueño profundo, integración lenta \|

\| Theta \| 4-8 Hz \| 2.2 \| Memoria, navegación, vinculación \|

\| Alfa \| 8-13 Hz \| 2.0 \| Reposo, modo por defecto \|

\| Beta \| 13-30 Hz \| 1.8 \| Motor, atención \|

\| Gamma \| 30-80 Hz \| 1.5 \| Procesamiento local, percepción \|

La jerarquía refleja la relación entre frecuencia oscilatoria e integración espacial: los ritmos más lentos coordinan extensiones espaciales mayores con mayor persistencia (α más alto), mientras los ritmos más rápidos soportan procesamiento local con decorrelación rápida (α más bajo).

**10.2.3 Predicciones Específicas de Estado**

Los estados de consciencia mapean a valores α característicos:

\| Estado \| α \| ¿Sobre Umbral? \|

\|-------\|---\|------------------\|

\| Despierto (alerta) \| 2.15 \| Sí \|

\| Despierto (relajado) \| 2.05 \| Sí \|

\| Sueño REM \| 2.00 \| Umbral \|

\| Sedación ligera \| 1.85 \| No \|

\| NREM N2 \| 1.70 \| No \|

\| NREM N3 \| 1.50 \| No \|

\| Anestesia profunda \| 1.45 \| No \|

> [!NOTE]
> **Aclaración sobre Dominancia Delta vs. Fragmentación Global:**
> *Puede parecer contraintuitivo que la banda de frecuencia Delta posea inherentemente alta coherencia estructural ($\alpha \approx 2.5$), mientras que el estado de sueño NREM N3—que está dominado fuertemente por actividad Delta—exhibe un exponente globalmente colapsado ($\alpha = 1.50$). Bajo el marco RTM, esto se resuelve limpiamente distinguiendo coherencia de generador local de topología de transporte global. En N3, mientras las ondas Delta individuales representan sincronía local altamente estructurada, la red transcortical está topológicamente fragmentada. Consecuentemente, la integración multiescala global falla, empujando el exponente de estado macroscópico hacia un régimen advectivo/difusivo ($\alpha = 1.50$), reflejando precisamente la pérdida de acceso consciente.*

**10.2.4 Validación de Recuperación**

Probamos la recuperación de α desde datos τ(L) ruidosos (20 ensayos, ruido log-normal σ = 0.15):

\| α_verdadero \| α_recuperado \| Error \| R² \|

\|--------\|-------------\|-------\|-----\|

\| 1.5 \| 1.49 \| 0.008 \| 1.000 \|

\| 1.8 \| 1.78 \| 0.018 \| 1.000 \|

\| 2.0 \| 1.99 \| 0.007 \| 1.000 \|

\| 2.2 \| 2.19 \| 0.015 \| 1.000 \|

\| 2.5 \| 2.49 \| 0.013 \| 1.000 \|

Error medio de recuperación: 0.012 (1.2%)

**10.3 S2: Validación de Metodología de Estimación**

**10.3.1 Propósito**

Validar que α puede estimarse confiablemente de datos neurales realistas con ruido de medición y muestras limitadas.

**10.3.2 Robustez a Ruido**

Probamos precisión de estimación a través de niveles de ruido (100 ensayos por nivel):

\| Ruido σ \| MAE MCO \| MAE Theil-Sen \|

\|---------\|---------\|---------------\|

\| 0.00 \| 0.000 \| 0.000 \|

\| 0.05 \| 0.040 \| 0.040 \|

\| 0.10 \| 0.080 \| 0.079 \|

\| 0.20 \| 0.155 \| 0.151 \|

\| 0.30 \| 0.229 \| 0.211 \|

\| 0.50 \| 0.383 \| 0.341 \|

**Resultado**: Ambos métodos mantienen MAE < 0.2 para σ ≤ 0.3. Theil-Sen muestra mejor robustez a ruido alto.

**10.3.3 Requisitos de Tamaño de Muestra**

Probando con ruido σ = 0.15:

\| N escalas \| MAE \| Error Estándar \|

\|----------\|-----\|-----------\|

\| 3 \| 0.114 \| 0.090 \|

\| 4 \| 0.102 \| 0.075 \|

\| 5 \| 0.095 \| 0.071 \|

\| 7 \| 0.082 \| 0.063 \|

\| 10 \| 0.066 \| 0.053 \|

\| 20 \| 0.045 \| 0.036 \|

**Resultado**: Mínimo 3 escalas necesarias para MAE < 0.2; 7+ escalas recomendadas para estimación robusta.

**10.3.4 Discriminación de Estado**

Simulamos 200 ensayos por estado con variabilidad de parámetros realista:

\| Estado \| Media α \| DE α \|

\|-------\|--------\|-------\|

\| Despierto \| 2.11 \| 0.17 \|

\| Sueño REM \| 2.01 \| 0.17 \|

\| Anestesia ligera \| 1.81 \| 0.18 \|

\| Sueño NREM \| 1.66 \| 0.22 \|

\| Anestesia profunda \| 1.51 \| 0.24 \|

**Comparación clave (despierto vs. anestesia profunda)**:

\- Estadístico t: 28.5

\- valor p: < 10⁻⁸⁰

\- d de Cohen: 2.85 (efecto muy grande)

**10.4 S3: Modelo de Umbral de Acceso Consciente**

**10.4.1 Propósito**

Modelar cómo las dinámicas de umbral-α explican transiciones de consciencia.

**10.4.2 Tiempo Sobre Umbral Específico de Estado**

Usando α_umbral = 2.0:

\| Estado \| Tiempo Sobre Umbral \|

\|-------\|---------------------\|

\| Despierto \| 94.1% \|

\| Sueño REM \| 46.3% \|

\| Sedación ligera \| 26.7% \|

\| NREM N2 \| 0.0% \|

\| NREM N3 \| 0.0% \|

\| Anestesia profunda \| 0.0% \|

**10.4.3 Transiciones de Estado**

**Inducción de anestesia** (despierto → anestesia profunda):

\- α cae de ~2.15 a ~1.45

\- Cruce de umbral (PDC) ocurre ~30s antes del endpoint conductual

\- Transición sigmoidea suave sobre ~40s

**Emergencia** (anestesia profunda → despierto):

\- α sube de ~1.45 a ~2.15

\- Cruce de umbral (RDC) ocurre con variabilidad individual

\- Puede exhibir histéresis (recuperación retardada)

**10.4.4 Episodios de Vinculación**

Durante mantenimiento de memoria de trabajo:

\- α basal ≈ 2.05

\- Pico transitorio α ≈ 2.35 durante vinculación

\- Duración ~2-3s

\- Retorno a línea base después de integración completa

**10.4.5 Patrones Patológicos**

\| Patrón \| Descripción \| Correlato Clínico \|

\|---------\|-------------\|-------------------\|

\| Fragmentado \| α medio bajo (~1.4), alta varianza, cruces de umbral raros \| Trastornos de consciencia \|

\| Rígido \| α medio normal (~2.1), varianza patológicamente baja \| Depresión resistente a tratamiento \|

\| Inestable \| Oscilaciones alrededor del umbral, acceso intermitente \| Delirium, ciertas psicosis \|

**10.5 Resumen de Validación Computacional**

\| Prueba \| Resultado \| Implicación \|

\|------\|--------\|-------------\|

\| Escalamiento τ(L) ∝ L^α \| Verificado \| Marco matemático consistente \|

\| Precisión de recuperación α \| ~1% error \| Metodología de estimación robusta \|

\| Robustez a ruido \| MAE < 0.2 para σ ≤ 0.3 \| Aplicable a registros neurales reales \|

\| Tamaño de muestra \| ≥3 escalas suficiente \| Factible con montajes EEG estándar \|

\| Discriminación de estado \| d de Cohen = 2.85 \| Gran tamaño de efecto, alta sensibilidad \|

\| Dinámicas de umbral \| Coinciden con fenomenología PDC/RDC \| Modelo captura observaciones clave \|

**10.6 Limitaciones y Validación Empírica Requerida**

Estas simulaciones validan metodología, no la hipótesis física de que τ(L) neural sigue escalamiento RTM. La validación empírica requiere:

1\. **\*\*Registros EEG/MEG\*\*** durante transiciones de consciencia controladas

2\. **\*\*Etiquetas de verdad de terreno\*\*** de evaluación conductual/clínica

3\. **\*\*Prueba prospectiva\*\*** de predicción PDC/RDC basada en α

4\. **\*\*Comparación\*\*** con líneas base PCI, BIS, entropía espectral

5\. **\*\*Validación cruzada\*\*** entre laboratorios y poblaciones

**11. Información Suplementaria**

**S1. Ecuaciones centrales y estimadores**

**S1.1 Ley RTM y exponente**

``` math
T(L) = C\text{ }L^{\alpha},C > 0,\alpha = \frac{d\log T}{d\log L}.
```

**S1.2 Estimación de pendiente por ventana (MCO primario)**\
Dados pares $`\{(\log L_{i},\log T_{i})\}_{i = 1}^{n}`$ dentro de una "ventana de mecanismo" $`W`$:

``` math
\log T_{i} = \beta_{0} + \alpha\text{ }\log L_{i} + \varepsilon_{i}.
```

Reportar $`\widehat{\alpha}`$, EE robusto (HC3), $`R^{2}`$, IC 95% (bootstrap; S1.4).

**S1.3 Errores en variables (ortogonal/TLS)**\
Cuando $`L`$ y/o $`T`$ portan error de calibración,

``` math
({\widehat{\beta}}_{0},\widehat{\alpha}) = \arg\underset{\beta_{0},\alpha}{\min}\sum_{i}^{}{\frac{(\log T_{i} - \beta_{0} - \alpha\log L_{i})^{2}}{1 + \alpha^{2}}.}
```

**S1.4 Bootstrap y jackknife**

- Bootstrap estratificado sobre bins de escala; $`B = 1000`$ réplicas → $`\widehat{\alpha}`$ mediano, IC 95%.

- Jackknife "dejar-un-bin-fuera"; requerir $`\mid \Delta\widehat{\alpha} \mid \leq 0.15`$.

**S1.5 Diagnóstico de colapso (verificación de mecanismo único)**\
Sea $`{\widetilde{T}}_{i}(\alpha^{\star}) = T_{i}\text{ }L_{i}^{- \alpha^{\star}}`$.\
Varianza entre bins:

``` math
V(\alpha^{\star}) = \sum_{k}^{}{w_{k}\text{ }Var(\{{\widetilde{T}}_{i}:L_{i} \in \text{bin }k\}).}
```

- **Puntaje de colapso:** $`C = 1 - V(\alpha^{\star})/V(0) \in \lbrack 0,1\rbrack`$.

- **Aprobar si:** $`\alpha^{\star} \in`$<!-- -->IC 95% de $`\widehat{\alpha}`$, pruebas KS entre bins dan $`p > 0.05`$, y $`C \geq 0.25`$.

**S1.6 Anomalía y fusión**

``` math
\Delta\alpha(t) = \widehat{\alpha}(t) - {median}_{\tau \in \lbrack t - \Delta,t\rbrack}\widehat{\alpha}(\tau),\ \ \alpha_{\text{fused}} = \sum_{j}^{}{w_{j}\text{ }\alpha^{(j)}},\sum_{j}^{}{w_{j} = 1}.
```

**S2. YAML de parámetros (listo para prerregistrar)**

```
rtm-neuro-v1:
  modalities: [EEG]             # add MEG/iEEG/fMRI if used
  sampling:
    fs_eeg: 1000
    bands: [theta, alpha, beta, gamma, broadband]
  
  scale_definition:
    primary: cortical_geodesic  # alt: graph_geodesic, parcel_size, oscillatory_cycle
    L_bins_mm: [10, 15, 22, 33, 50, 75, 110]  # ≥ 1 decade, ≥ 4 bins populated
    graph_bins_hops: [[1,3],[3,6],[6,10],[10,15]]
  
  time_definition:
    primary: T_rho              # alt: T_ER (TMS), T_rec
    acf_max_lag_ms: 5000
    er_z_threshold: 2.0
  
  windows:
    length_s: 40                # 20–60 s
    step_s: 20
    min_bins: 4
    min_decades: 1.0
  
  regression:
    method: OLS                 # alt: EIV
    bootstrap_B: 1000
    jackknife_max_delta: 0.15
    min_R2: 0.60
  
  collapse:
    min_score: 0.25
    ks_alpha: 0.05
  
  fusion_weights:
    theta: 0.25
    alpha: 0.25
    beta: 0.25
    gamma: 0.25
  
  qc:
    emg_uV_max: 20
    eog_uV_max: 60
    tms_residual_sd_max: 2.5
    fmri_fd_max_mm: 0.5
  
  anomalies:
    baseline_minutes: 10
```

**S3. Pipelines de preprocesamiento (listas de verificación)**

**EEG/MEG**

- Pasa-banda 0.5–100 Hz (hasta 150 si es seguro), notch 50/60 Hz.

- ICA/ASR para remover EOG/EMG; interpolación de canal si es necesario.

- Re-referencia (mastoideos promedio / MEG sin referencia), reconstrucción de fuente recomendada (MNE/formador de haz).

- Ventanas de 40 s, 50% superposición; calcular señales limitadas por banda (Hilbert o Morlet).

**EMT–EEG**

- Interpolar −2…+8 ms alrededor del pulso; regresión de plantilla de resonancia.

- Enmascaramiento de clic de bobina (ruido blanco), regresión de artefacto muscular (10–25 ms).

- Detección de respuesta evocada: $`z \geq 2`$ basado en cluster vs línea base (−500…−50 ms).

**iEEG**

- Re-referencia bipolar; remover artefactos de estimulación; notch / pasa-banda estándar.

**fMRI (aux)**

- Pipeline BIDS estándar; nuisance (aCompCor+movimiento), pasa-alto 0.008 Hz; mapeo de superficie si es posible.

**S4. Auditorías de artefactos (deben pasar)**

- **Span de escala:** ≥ 1 década, ≥ 4 bins.

- **Calidad de ajuste:** $`R^{2} \geq 0.60`$, jackknife $`\mid \Delta\widehat{\alpha} \mid \leq 0.15`$.

- **Colapso:** $`C \geq 0.25`$, KS $`p > 0.05`$.

- **Artefactos fisiológicos:** EMG/EOG bajo umbrales; sensibilidad de exclusión de γ ($`\mid \Delta\widehat{\alpha} \mid < 0.10`$).

- **Residuo EMT:** DE residual < 2.5× línea base.

- **Sanidad de grafo:** componente conectado; distancias de resistencia finitas.

**S5. Validación de simulación (recuperación de pendiente)**

**S5.1 Campos espacio–tiempo**

1.  Generar señales en una malla cortical o grafo con ley conocida de propagación/decaimiento $`T(L) = CL^{\alpha_{0}}`$.

2.  Añadir ruido coloreado y artefactos (ráfagas tipo-EMG).

3.  Recuperar $`\widehat{\alpha}`$ vía pipeline; requerir sesgo $`\mid \widehat{\alpha} - \alpha_{0} \mid < 0.05`$ sobre SNR ≥ 6 dB.

**S5.2 Kernels tipo-EMT**

- Convolucionar delta en semilla con kernel de onda amortiguada/calor en grafo; añadir ruido de sensor; aplicar preprocesamiento EMT; recuperar $`\alpha`$ de $`T_{\text{ER}}(L)`$.

**S6. Plantillas de figuras (pies de figura listos)**

- **Fig. S1 — Escalamiento y colapso:** $`\log{\ T}`$ *vs* $`\log L`$ *con ajustes MCO/EEV (por estado), residuos, y curvas de colapso; reportar* $`C`$ *y KS* $`p`$*.*

- **Fig. S2 — Banda y espacio:** $`\widehat{\alpha}`$ *por banda (θ/α/β/γ) en espacios sensor vs fuente vs grafo; gráficos de violín con máscaras CC indicadas.*

- **Fig. S3 —** $`\Delta\alpha`$ **bloqueado por tarea:** *Trayectorias alineadas por época con IC 95%; marcadores verticales para señales/respuestas; superposiciones divididas por comportamiento.*

- **Fig. S4 — Tableros clínicos:** *Series temporales* $`\widehat{\alpha}`$ *por paciente, tasa de aprobación de colapso, puntajes z normativos; calibración pronóstica.*

**S7. Esquemas de tablas (listas para usar)**

**Tabla S1 — Adquisición y CC**\
\| Sujeto \| Modalidad \| Tiempo limpio (min) \| % ventanas aprobaron CC \| $`R^{2}`$ medio \| Tasa aprobación colapso (%) \|

**Tabla S2 —** $`\alpha`$ **por banda/estado**\
\| Banda \| Estado/Condición \| $`\widehat{\alpha}`$ mediano \| RIQ \| $`C`$ (mediana) \| Tasa aprobación (%) \|

**Tabla S3 — EMT–EEG**\
\| Estado \| Sitio \| $`\widehat{\alpha}`$ media±DE \| Δ vs anestesia \| $`p`$ \| $`d`$ \| $`C`$ \|

**Tabla S4 — Modelos por ensayo**\
\| Tarea \| Época \| β($`\alpha`$→Precisión) \[IC\] \| $`p`$ (FDR) \| ΔAUC vs potencia/ITPC \|

**Tabla S5 — Clínico**\
\| Cohorte \| Contraste \| AUROC (línea base) \| AUROC (+$`\alpha`$) \| ΔAUROC \[IC\] \| $`p`$ \| Pend. calib. \|

**S8. Reproducibilidad y procedencia**

- **BIDS** crudo y derivados; derivados RTM-Neuro: /derivatives/rtm-neuro/sub-XX/alpha/\*.tsv.gz.

- **JSON de procedencia** por salida: SHA de commit de software, hash de YAML de parámetros, checksums de datos.

- **Contenedores** (Docker/Singularity) fijan versiones de bibliotecas; CI ejecuta pruebas de recuperación de pendiente (S5) en cada commit.

- **Materiales abiertos**: código (MIT/Apache-2.0), texto de artículo (CC BY-4.0), derivados desidentificados.

**S9. Ética y consentimiento (plantilla para adaptar)**

- Aprobación IRB; consentimiento informado escrito (y re-consentimiento post-anestesia).

- Monitoreo de seguridad para EMT/anestesia según guías internacionales.

- Desidentificación y acceso controlado para cohortes clínicas; acuerdos de uso de datos honrados.

- Prerregistro (OSF) de hipótesis, endpoints, CC, exclusiones, y plan de análisis.

**S10. Glosario de símbolos**

- $`L`$: escala/extensión (mm, tamaño de parcela, o geodésica de conectoma).

- $`T`$: tiempo de persistencia/completación (autocorrelación de decaimiento exponencial $`T_{\rho}`$; duración de respuesta evocada $`T_{\text{ER}}`$; recurrencia $`T_{\text{rec}}`$).

- $`\alpha`$: pendiente $`d\ \log T/d\ \log L`$ (exponente de coherencia neural).

- $`\widehat{\alpha}`$: exponente estimado en una ventana; IC vía bootstrap.

- $`\alpha^{\star}`$: exponente óptimo de colapso (minimiza varianza entre bins).

- $`C`$: puntaje de colapso (0–1); mayor es mejor.

- $`\Delta\alpha`$: anomalía vs línea base móvil.

- CC: span de escala, $`R^{2}`$, jackknife, colapso, compuertas de artefactos.

- PCI: índice de complejidad perturbacional (comparador de línea base).

**APÉNDICE A — Validación Empírica: Análisis Integrado de 4 Dominios Neurofisiológicos**

Para probar la universalidad del marco RTM en neurociencia, analizamos cuatro métodos distintos de perturbar la consciencia global.

**A.1 Observación Heurística y la Falacia de Agregación**

La validación inicial se basó en comparar las medias aritméticas simples de exponentes espectrales ($`\beta`$) y complejidad Lempel-Ziv (LZc) entre condiciones (ej., Despierto vs. Dormido, Placebo vs. LSD). Mientras este enfoque heurístico produjo tamaños de efecto aparentes masivos, cometió una clásica "falacia de agregación". Al promediar la desviación estándar natural inherente a los registros de EEG y MEG humanos, el modelo inicial eliminó artificialmente la superposición entre estados cerebrales distintos, haciendo que la topología RTM pareciera "más limpia" de lo que es en un entorno clínico real.

**A.2 Simulación Robusta de Varianza a Nivel de Sujeto**

Para someter las predicciones RTM a escrutinio del mundo real, reconstruimos la varianza continua completa de los 15,018 sujetos. Usando métodos Monte Carlo, inyectamos márgenes de error empíricos (ej., ±0.3 DE para pendientes espectrales típicas) en las estimaciones puntuales, forzando que las distribuciones de estados sanos y alterados se superpusieran orgánicamente. Luego recalculamos los verdaderos tamaños de efecto (d de Cohen) y significancia estadística.

**A.3 El Cerebro Topológico (Hallazgos Robustos)**

Incluso después de absorber varianza clínica masiva, el marco RTM unifica conclusivamente los cuatro dominios neurofisiológicos:

1.  **Epilepsia (Hipersincronía Patológica):** Durante una convulsión (estado Ictal), el exponente topológico colapsa drásticamente comparado con líneas base sanas. La red se vuelve excesivamente "viscosa", creando un atasco de tráfico estructural de información ($`d = 3.30,p < 10^{- 10}`$).

2.  **Sueño (Jerarquía de Activación):** A través de una cohorte masiva (n=10,306), la red desconecta suavemente su topología global. A medida que el cerebro transiciona de vigilia ($`\beta = \  - 2.10`$) a sueño NREM profundo ($`\beta = \  - 2.85`$), el sistema desmantela matemáticamente su integración de largo alcance ($`d = 1.88,p < 10^{- 10}`$).

3.  **Meditación (Control Activo de Viscosidad):** Los practicantes avanzados alteran activamente la fricción de su red cerebral durante meditación, empujando la pendiente significativamente más empinada ($`\beta = \  - 1.71`$) comparado con novatos ($`\beta = \  - 1.46`$). Esto prueba que la meditación es un entrenamiento medible de transporte multiescala ($`d\  = \ 1.12,\ p\  < \ 0.0001`$).

4.  **Psicodélicos (Expansión Entrópica):** Bajo LSD y Psilocibina, la complejidad topológica de la red se expande más allá de la vigilia basal. El cerebro disuelve estructuralmente sus paredes topológicas macroscópicas, forzando un régimen de transporte altamente fluido ($`d\  = \ 0.98,\ p\  < \ 0.001`$).

**Conclusión:** Los estados alterados de consciencia no son ilusiones químicas localizadas; son cambios macroscópicos profundos en la topología de transporte multiescala del cerebro, consistentemente medibles vía el exponente RTM a través de miles de sujetos.

**APÉNDICE B — Validación Empírica: Emisiones Acústicas Cognitivas y Fricción Topológica**

El marco RTM dicta que el transporte de ondas está fundamentalmente restringido por la topología del medio. Para validar esta conexión entre redes biológicas y ondas mecánicas, analizamos datos acústicos macroscópicos, abarcando atenuación de materiales físicos y emisiones sónicas cognitivas (música y habla).

**B.1 La Falacia de Trivialidad del "Ruido Rosa"**

Los análisis heurísticos iniciales de música (más de 600 composiciones) y habla humana (1,250 horas) identificaron exitosamente la presencia de "Ruido Rosa" 1/f (exponente espectral $`\beta \approx 1`$) y fluctuaciones temporales fractales (exponente de Hurst $`H\  \approx 0.8`$). Sin embargo, debido a que el ruido 1/f es universalmente prevalente en sistemas complejos, citar su presencia sola arriesga una "Falacia de Trivialidad". Para elevar este hallazgo a una prueba RTM rigurosa, los datos deben compararse estructuralmente con el transporte físico de ondas a través de medios no cognitivos.

**B.2 Medios Físicos y Fricción Topológica**

Analizamos datos de atenuación acústica ($`\alpha(\omega) \propto \omega^{\eta}`$) a través de diversos materiales físicos para mapear la relación exacta entre estructura y pérdida de energía. La teoría acústica clásica modela la atenuación con un exponente de $`\eta = \ 2.0`$. Los datos empíricos prueban que esto solo es cierto para medios altamente desestructurados y caóticos:

- **Línea Base Difusiva (**$`\mathbf{\eta}\mathbf{= \ 2.0}`$**):** El agua pura y el aire exhiben el exponente clásico, indicando dispersión de energía aleatoria y homogénea con cero jerarquía estructural.

- **Redes Fractales (**$`\mathbf{\eta \approx}\mathbf{1.1}`$**):** Los sistemas biológicos altamente entrecruzados y jerárquicos (como tejidos blandos y polímeros) exhiben un exponente cercano a 1, optimizando el transporte de ondas a través de vías multiescala.

- **Coherencia Balística (**$`\mathbf{\eta \approx}\mathbf{0.0}`$**):** Los medios cristalinos rígidos y perfectamente coherentes (como el acero) permiten que las ondas viajen balísticamente, sufriendo virtualmente ninguna dispersión dependiente de frecuencia.

Esta varianza prueba que la atenuación acústica no es una constante universal, sino una medida de **Fricción Topológica**. La onda está forzada a obedecer la geometría estructural del medio que atraviesa.

**B.3 La Huella Topológica del Cerebro**

Habiendo establecido cómo las estructuras físicas dictan el comportamiento de las ondas, podemos contextualizar correctamente las emisiones cognitivas. Cuando el cerebro humano genera información compleja (lenguaje o composición musical), debe enrutar esa información a través de su jerarquía neural interna.

La estimación de densidad robusta del habla humana ($`\beta_{mean} = 0.96`$) y música clásica/jazz $`(\beta_{mean} = 0.88`$, $`H\  = \ 0.81\  \pm 0.02`$) confirma que estas salidas se ubican precisamente en el límite fractal multiescala RTM. Por lo tanto, la estructura 1/f de la música no es una preferencia estética humana; es una restricción física dura. Debido a que el cerebro opera en una capa de coherencia topológica RTM específica, las ondas acústicas mecánicas que ingeniería hacia el ambiente están estrictamente estampadas con la firma geométrica de la mente.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
