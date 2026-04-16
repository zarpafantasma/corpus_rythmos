<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **El Marco de Cascadas RTM**  
**Dinámica Jerárquica: Estabilidad Dependiente de Escala y Bifurcación de Fase**  
  
Álvaro Quiceno
</div>

**Significancia y Operacionalización (del Concepto a la Prueba)**

**Por qué esto importa.** Muchos sistemas multiescala *parecen* organizar la información a medida que se mueve a través de capas anidadas, pero la mayoría de la evidencia está confundida por desplazamientos de nivel (ganancias, retardos, cinemática). Separamos la **coherencia** de los **efectos de nivel** tratando la **pendiente log–log** $`\alpha = \partial\ \log\ T/\partial\ \log\ L`$ como el marcador operacional de organización, y relegando los factores a nivel de capa a la ordenada al origen. Esto produce una pregunta limpia y falsificable: *¿aumenta la coherencia (o al menos no disminuye) a lo largo de la secuencia, y es el flujo de información unidireccional hacia adelante?*

**Del concepto a la prueba.** Convertimos la narrativa en dos firmas empíricas:

**S1 — Coherencia monótona a través de capas.** Dentro de cada capa, regresione $`\log T`$ sobre $`\log\ L`$; la hipótesis de cascada requiere $`\alpha_{n + 1} \geq \alpha_{n} - \varepsilon`$ (ICs bootstrap al 95%; $`\varepsilon`$ pre-registrado).

**S2 — Direccionalidad solo hacia adelante.** Estime la **entropía de transferencia** y la **causalidad de Granger** entre capas adyacentes; requiera significancia hacia adelante $`(n \rightarrow n + 1)`$ y no en reversa, con valores p basados en sustitutos y control FDR.

**Controles, pruebas de estrés y reproducibilidad.** Incluimos (i) controles **solo de ordenadas al origen** (α plano con grandes desplazamientos de nivel), (ii) un **nulo** con capas desacopladas (sin direccionalidad), y (iii) un barrido de **histéresis/trinquete** (evidencia de apoyo de memoria direccional).

**Resumen**

Mientras que los artículos anteriores establecieron la relación de escalamiento estática $`T\backslash proptoL^{\backslash}alpha`$, este documento explora la dinámica de tales sistemas a través de escalas jerárquicas. Analizamos cómo la energía y la información se propagan a través de la "Red RTM", proponiendo un modelo de **Cascadas Jerárquicas**. Demostramos que los sistemas con exponentes $`\backslash alpha`$ diferentes (ej., difusivo vs. balístico) no pueden acoplarse eficientemente sin una interfaz transicional, conduciendo a fenómenos de **Desajuste de Impedancia**. Además, formalizamos los límites superiores del tamaño estructural ("Inestabilidad Alométrica") y derivamos las condiciones bajo las cuales un sistema experimenta **Ruptura de Simetría** en su evolución temporal. Este marco proporciona un mecanismo para la formación de estructuras a macro-escala a partir de coherencia a micro-escala, sin invocar física exótica.

**Validación empírica**$`\mathbf{\rightarrow}`$**(APÉNDICE A)**. Validamos el marco de cascadas RTM en sistemas neuronales biológicos a través de un análisis sistemático expandido de 21 áreas dentro de la jerarquía de la corteza visual. El análisis heurístico inicial sugirió un régimen de escalamiento Super-Difusivo basado en campos receptivos espaciales ($`\Delta X`$) y latencias de procesamiento temporal ($`\Delta T`$) altamente agregados. Para corregir rigurosamente los sesgos de atenuación y agregación inherentes a las mediciones ruidosas de fMRI y electrodos, desplegamos una tubería de Errores en Variables (ODR) y reconstruimos la varianza poblacional subyacente a nivel de sujeto. El análisis robusto confirma que el sistema opera estrictamente en un régimen de escalamiento Super-Difusivo, produciendo un exponente corregido por varianza de $`\mathbf{\alpha}\mathbf{= \ 0.31\ }\mathbf{\pm}\mathbf{0.02}`$ (nivel poblacional $`\alpha = \ 0.28`$). Este exponente dicta que el cerebro integra información a través del espacio cortical matemáticamente más eficientemente que una red de difusión aleatoria clásica ($`\alpha = \ 0.5`$). Al combinar procesamiento paralelo con codificación jerárquica, la biología logra evadir exitosamente los límites físicos del transporte difusivo, validando $`\alpha`$ como una métrica fundamental para cuantificar la eficiencia arquitectónica en redes neuronales complejas.

**1. Introducción**

**1.1 Motivación y alcance**

RTM (Relatividad Temporal Multiescala) postula que el tiempo característico de un proceso escala con un tamaño efectivo según $`{T/T}_{0}{{= (L/L}_{0})}^{\alpha}`$, donde el exponente $`\alpha`$ operacionaliza la **coherencia mesoscópica**. Los ensayos conceptuales *Simulacrum* y *La Arquitectura del Eco* motivan una imagen en la cual la información es **recodificada** en estructuras cada vez más ordenadas y propagada **secuencialmente** a través de capas anidadas, una "arquitectura de armónicos resonantes" que avanza hacia adelante en lugar de hacia atrás. Nuestro objetivo aquí es traducir esa narrativa en **firmas testeables y falsificables** que puedan ser probadas con datos reales o análogos, sin comprometerse con afirmaciones metafísicas.

**1.2 Planteamiento del problema**

Dado un sistema descomponible en capas $`n = 1,\ldots,N`$ (envolturas espaciales, módulos funcionales, o etapas análogas controladas), ¿**aumenta** la **coherencia** (o al menos no disminuye) a lo largo de la secuencia, y es el **flujo de información** predominantemente **de** $`n`$ **a** $`n + 1`$? Si es así, RTM predice cambios en la **pendiente** $`\alpha_{n} = \partial\ \log\ T/\partial\ \log\ L`$ y **causalidad direccional** entre los observables de las capas. Si no, las pendientes deberían permanecer invariantes y las métricas causales deberían ser simétricas.

**1.3 Firmas testeables (lo que este artículo mide)**

Nos enfocamos en dos asas empíricas:

- **(S1) Coherencia basada en pendiente:** en cada capa, regresione $`\log\ T`$ sobre $`\log\ L`$ para estimar $`\alpha_{n}`$ con ICs bootstrap. La hipótesis de cascada requiere $`\alpha_{n + 1}{\geq \alpha}_{n} - \varepsilon`$ para tolerancia pequeña $`\varepsilon`$.

- **(S2) Causalidad direccional:** cuantifique la **entropía de transferencia** (TE) y/o la **causalidad de Granger** entre series temporales derivadas de capas adyacentes; espere significancia para $`n \rightarrow n + 1`$ y no a la inversa.

Las extensiones opcionales que exploraremos incluyen **trinquete/histéresis** bajo barridos de control del acoplamiento entre capas y un **compromiso coherencia–desorden** (aumento en $`\alpha`$ junto con una reducción de la entropía de actividad).

**1.4. Validación empírica sistemática: El régimen Super-Difusivo de la corteza visual (APÉNDICE A)**

Dentro del marco teórico de RTM, las arquitecturas de cascada jerárquica (como el cerebro humano) no solo procesan información; deben navegar restricciones topológicas fundamentales de espacio y tiempo. Para someter esta premisa a una prueba empírica, analizamos la relación de escalamiento entre la extensión espacial del campo receptivo ($`\Delta X`$) y la latencia de procesamiento temporal ($`\Delta T`$) a través de 21 áreas distintas de la jerarquía visual.

Debido a que las mediciones neurológicas espaciales y temporales poseen error observacional masivo, las regresiones iniciales de estimación puntual son altamente susceptibles a sesgos estadísticos de atenuación y agregación. Al aplicar Regresión de Distancia Ortogonal (ODR) robusta y desagregar los datos para simular la varianza poblacional natural, demostramos conclusivamente qué clase de transporte físico utiliza la biología. Los datos robustos revelan abrumadoramente que la corteza visual no opera bajo la ineficiencia de la difusión aleatoria ($`\alpha = \ 0.5`$), sino que ha evolucionado hacia un régimen Super-Difusivo altamente optimizado ($`\alpha \approx 0.31`$). Este hallazgo demuestra que la macroarquitectura del cerebro, impulsada por procesamiento masivamente paralelo en cada nivel jerárquico, logra "curvar" las reglas clásicas de la física estadística para alcanzar integración de información hiper-eficiente, efectivamente tendiendo un puente entre la cinética difusiva y balística.

**2. Formulación matemática esencial**

**2.1 Escalamiento RTM en sistemas estratificados**

Consideramos un sistema descompuesto en $`N`$ **capas** anidadas $`n \in \{ 1,\ldots,N\}`$. En cada capa, un tiempo de proceso mesoscópico $`T_{n}`$ asociado con un tamaño efectivo $`L`$ sigue la ley RTM

| (2.1) |
|-------|

``` math
\frac{T_{n}}{T_{0}} = \left( \frac{L}{L_{n}} \right)^{\alpha_{n}}\Xi_{n}
```

donde:

- $`\alpha_{n}`$ es el **exponente de coherencia** para la capa n (la cantidad que deseamos estimar);

- $`T_{0},`$ $`L_{n}`$ son escalas de referencia (fijas a través de las capas);

- $`\Xi_{n}`$ es un **factor a nivel de capa** que desplaza los niveles pero **no depende de** $`\mathbf{L}`$ (ej., corrimiento al rojo/cinemática en configuraciones astrofísicas, latencia instrumental, ganancia a nivel de capa).

Tomando logaritmos,

| (2.2) |
|-------|

``` math
\underset{y_{n}}{\overset{\log T_{n}}{︸}} = \underset{\text{pendiente}}{\overset{\alpha_{n}}{︸}} \cdot \underset{x}{\overset{\log L}{︸}} + \underset{\beta_{n}}{\overset{\log\left( T_{0}/L_{0}^{\alpha_{n}} \right) + \log\Xi_{n}}{︸}} +
```

así que a **capa fija** $`\mathbf{n}`$ la **pendiente log–log** es igual a $`\alpha_{n}`$​ y la **ordenada al origen** $`\beta_{n}`$ absorbe $`\Xi_{n}`$. Esta es la base para la Firma **(S1)** (coherencia basada en pendiente). Conceptualmente, $`\alpha`$ captura **organización/coherencia**, consistente con la narrativa de "recodificación" que motiva este trabajo.

**2.2 Objetivo de estimación y modelo de regresión**

Dadas las observaciones $`\left\{ \left( L_{ni},T_{ni} \right) \right\}_{i = 1}^{m_{n}}`$ en la capa n abarcando un rango de tamaños $`L`$, estimamos

``` math
\alpha_{n} = \left. \ \frac{\partial\log T_{n}}{\partial\log L} \right|_{n}
```

vía **mínimos cuadrados ordinarios (OLS)** en el modelo (2.2). Reportamos:

- estimación puntual $`{\widehat{\alpha}}_{n}`$​,

- **ICs bootstrap al 95%** (remuestreando eventos dentro de la capa $`n`$),

- diagnósticos de bondad de ajuste (residuos vs. $`\log\ L`$).

**Nota de diseño.** La identificabilidad de $`\alpha_{n}`$ requiere **dispersión** en $`L`$ dentro de la capa (≥6–8 tamaños distintos es una buena regla general).

**2.3 Flujo de información direccional entre capas**

Sea $`X_{n}(t)`$ una serie observable específica de capa (ej., tasa de eventos, energía de pulso, ancho). La Firma **(S2)** prueba si la **influencia causal** es **asimétrica** de $`n \rightarrow n + 1`$.

**(a) Entropía de transferencia (TE)**

Para retardos $`k`$ y $`l`$,

| (2.3) |
|-------|

$`\text{TE}_{n \rightarrow n + 1} = \sum_{}^{}{p\left( x_{n + 1}(t + 1),x_{n + 1}^{(k)}(t),x_{n}^{(l)}(t) \right)}\log\frac{p\left( x_{n + 1}(t + 1)\  \middle| \ x_{n + 1}^{(k)}(t),x_{n}^{(l)}(t) \right)}{p\left( x_{n + 1}(t + 1)\  \middle| {\ x}_{n + 1}^{(k)}(t) \right)}`$

Probamos $`\text{TE}_{n \rightarrow n + 1} > \text{TE}_{n + 1 \rightarrow n}`$ usando sustitutos de permutación/bootstrap para obtener valores p.

**(b) Causalidad de Granger (prueba G)**

En un VAR bivariado de orden $`p`$,

| (2.4) |
|-------|

``` math
X_{n + 1}(t) = \sum_{j = 1}^{p}{a_{j}X_{n + 1}(t - j)} + \sum_{j = 1}^{p}{b_{j}X_{n}(t - j)} + \eta(t)
```

Probamos $`H_{0}:b_{1} = \cdots = b_{p} = 0`$ (sin causalidad de Granger $`n \rightarrow n + 1`$). La direccionalidad requiere significancia para $`n \rightarrow n + 1`$ y **no** para $`n + 1 \rightarrow n`$.

**Interpretación.** Una "arquitectura de armónicos resonantes" hacia adelante implica **coherencia no decreciente** ($`\alpha_{n + 1} \geq \alpha_{n} - \varepsilon`$) junto con **flujo de información asimétrico** $`n \rightarrow n + 1`$.

**2.4 Hipótesis y criterios de falsificación**

**Pendiente (S1).**

- $`H_{0}:\alpha_{n + 1} < \alpha_{n} - \varepsilon`$ para algún par adyacente (disminución más allá de la tolerancia).

- $`H_{1}:\alpha_{n + 1} < \alpha_{n} - \varepsilon`$ para todo $`n`$.

> **Decisión:** Rechazar el apoyo si algún IC adyacente para $`{\widehat{\alpha}}_{n + 1} - {\widehat{\alpha}}_{n}`$ yace **enteramente por debajo** de $`- \varepsilon`$

**Direccionalidad (S2).**

- $`H_{0}:`$ simetría o influencia reversa, $`{TE}_{n \rightarrow n + 1}{\leq TE}_{n + 1 \rightarrow n}`$ (y similarmente para Granger).

- $`H_{1}:`$ asimetría hacia adelante, $`{TE}_{n \rightarrow n + 1}{> TE}_{n + 1 \rightarrow n}`$ (**y** prueba G significativa solo hacia adelante).\
  **Decisión:** Requerir que tanto $`TE`$ como Granger coincidan en asimetría hacia adelante (con control de comparaciones múltiples a través de $`n`$).

**2.5 Controles y confusores (ordenadas al origen vs pendientes)**

Los factores a nivel de capa $`\Xi_{n}`$ (ej., mapeo gravitacional/cinemático, ganancias globales) actúan **solo en la ordenada al origen** $`\beta_{n}`$ en (2.2). Por tanto:

- Los **cambios de ordenada al origen** a través de capas **no** implican cambio de coherencia.

- Los **cambios de pendiente** indican organización relevante para RTM.\
  Esto desentraña los efectos de "mapeo de reloj"/instrumentales de la coherencia, exactamente como en el estudio previo de RTM sobre entornos compactos.

**2.6 Ruido, robustez y proxies para L**

- **Modelo de ruido.** Asumimos fluctuaciones multiplicativas: $`\varepsilon`$ log-normal con escala $`\sigma_{\log} \in \lbrack 0.05,0.2\rbrack`$. Los ICs bootstrap mitigan la no gaussianidad.

- **Valores atípicos.** Si se sospechan colas pesadas, complementar OLS con verificaciones de sensibilidad **Theil–Sen** o regresión Huber.

- **Proxies para** $`\mathbf{L}`$**.** Cuando $`L`$ no se mide directamente, defina proxies **geométricos**, **cinemáticos** $`\mathbf{(}{\mathbf{L}\mathbf{\approx}\mathbf{vT}}_{\mathbf{subida}}\mathbf{)}`$, o **estadísticos** (longitud de correlación). Reporte pendientes para **múltiples proxies** y verifique la estabilidad.

**2.7 Plantillas paramétricas para simulaciones/análogos**

Para estudios sintéticos y análogos usamos un perfil monótono para la coherencia a través de capas:

| (2.5) |
|-------|

$`\alpha_{n} = \alpha_{\text{base}} + \Delta\alpha\frac{1}{1 + \exp\left( \frac{n - n_{c}}{w} \right)}\quad\left( \text{logística} \right),\quad\text{o}\quad\alpha_{n} = \alpha_{\text{base}} + \Delta\alpha\left( \frac{n_{c}}{\max\left( n,n_{c} \right)} \right)^{p}\quad\left( \text{rampa suave} \right)`$

El acoplamiento direccional se introduce solo de $`n`$ a $`n + 1`$ (para S2), con una intensidad sintonizable $`g`$ usada después para sondear el **trinquete/histéresis**.

**2.8 Potencia y guía de diseño**

- **Dispersión dentro de capa en** $`\mathbf{L}`$ domina la potencia para $`\alpha_{n}`$: apunte a ≥6–8 tamaños distintos por capa y ≥1 década de dispersión cuando sea posible.

- **Longitud de serie para TE/Granger:** al menos $`10^{2} - 10^{3}`$ muestras efectivas por par de capas, con validación cruzada en órdenes de retardo.

- **Pre-registro:** (i) tolerancia $`\varepsilon`$; (ii) órdenes de retardo para TE/Granger; (iii) control de comparaciones múltiples; (iv) criterios para aceptación nula.

**Procedencia y delimitación.** La imagen estratificada de "armónicos resonantes" viene de *La Arquitectura del Eco*; la intuición de **recodificación de información** de *Simulacrum*. Aquí restringimos ambas a **firmas operacionales** (pendientes, asimetría causal) que pueden ser confirmadas o refutadas con datos de sistemas reales o análogos.

**3. Predicciones testeables y reglas de decisión**

Esta sección convierte la formulación RTM estratificada (§2) en **predicciones concretas y falsificables** con pruebas explícitas, umbrales y reglas de parada. Las predicciones se agrupan como **centrales** (deben pasar) y **de apoyo** (fortalecen la afirmación pero no son requeridas). La procedencia conceptual, *Simulacrum* (recodificación) y *La Arquitectura del Eco* (armónicos resonantes secuenciales), se mantiene solo como **motivación**; las pruebas abajo se sostienen sobre bases operacionales.

**3.1 Firma central S1 — Coherencia monótona a través de capas (prueba de pendiente)**

**Predicción.** A lo largo del índice de capa $`n = 1,\ldots,N`$, el exponente de coherencia es **no decreciente** dentro de la tolerancia $`\varepsilon`$:

| (3.1) |
|-------|

``` math
\Delta\alpha_{n} \equiv \alpha_{n + 1} - \alpha_{n} \geq - \varepsilon\quad\text{para todo }n
```

**Estimador.** Para cada capa $`n`$, ajuste $`{\log\ T}_{\log} = \alpha_{n}\ \log\ L + \beta_{n} + \varepsilon`$ (Ec. 2.2), obtenga $`{\widehat{\alpha}}_{n}`$ y un **IC bootstrap al 95%** (remuestreando eventos dentro de la capa $`n`$, ≥1000 réplicas).

**Prueba por capa.** Para cada par adyacente,

| (3.2) |
|-------|

``` math
\widehat{\Delta}\alpha_{n} = {\widehat{\alpha}}_{n + 1} - {\widehat{\alpha}}_{n},\quad\text{con IC bootstrap }\left\lbrack \text{inf}_{n},\text{sup}_{n} \right\rbrack
```

**Pasa** si $`{inf}_{n} \geq - \varepsilon`$ para todo $`n`$. **Falla** (falsificación) si algún $`{sup}_{n} < - \varepsilon`$

**Prueba global (robustez opcional).** Ajuste una **regresión isotónica** ($`\alpha_{n}`$ no decreciente) y compare contra ajustes sin restricción vía un **bootstrap de razón de verosimilitud**; rechace la monotonicidad si el modelo restringido es significativamente peor (ej., p<0.05).

**Notas de diseño.** La potencia está dominada por la **dispersión en** $`\mathbf{L}`$ por capa (§2.8). Apunte a $`\geq 6 - 8`$ valores distintos de $`L`$ y ≳ una década de dispersión.

**3.2 Firma central S2 — Causalidad direccional (armónicos resonantes hacia adelante)**

Sea $`X_{n}(t)`$ una serie temporal específica de capa (tasa, energía de pulso, anchos, o una característica extraída consistentemente a través de capas).

**Predicción.** El flujo de información es **asimétrico hacia adelante**:

| (3.3) |
|-------|

$`\text{TE}_{n \rightarrow n + 1} > \text{TE}_{n + 1 \rightarrow n}\quad\text{y}\quad\text{Granger}(n \rightarrow n + 1)\text{ significativo, Granger}(n + 1 \rightarrow n)\text{ no.}`$

**Entropía de transferencia (TE).** Estime $`\text{TE}_{n \rightarrow n + 1}`$ y $`\text{TE}_{n + 1 \rightarrow n}`$ con embedding coincidente; obtenga **valores p** vía **sustitutos de permutación/mezcla de fase** (≥1000). Aplique **BH-FDR** a través de pares.

**Granger.** Ajuste un VAR bivariado con orden seleccionado por AIC/BIC. Pruebe $`H_{0\ }`$ (sin Granger) vía prueba F. Requiera significancia **solo** para la dirección hacia adelante.

**Regla de decisión.** Afirme **armónicos resonantes hacia adelante** solo si **ambos** TE y Granger coinciden en asimetría hacia adelante después del control de comparaciones múltiples. De lo contrario: **sin apoyo** para direccionalidad.

**Notas de diseño.** Use $`{\geq 10}^{2}{- 10}^{3}`$ muestras efectivas por par; valide cruzadamente los retardos; verifique estacionariedad (diferencie/elimine tendencia si es necesario).

**3.3 Firma de apoyo S3 — Trinquete/histéresis bajo barridos de acoplamiento**

Introduzca un acoplamiento entre capas controlable $`g`$ (plataforma análoga). Barra $`g`$ **hacia arriba** y luego **hacia abajo**, midiendo $`{\widehat{\alpha}}_{n + 1}(g).`$

**Predicción.** **Ciclo de histéresis**: las ramas hacia adelante y hacia atrás difieren (memoria de activación unidireccional).

**Cuantificación.** Defina el área del ciclo

| (3.4) |
|-------|

``` math
\mathcal{A}_{\mathcal{n} + 1} = \oint_{}^{}{\widehat{\alpha}}_{n + 1}(g)\, dg
```

(trapecios discretos). **Pasa** si el IC bootstrap de $`\mathcal{A}_{\mathcal{n} + 1}`$ difiere de cero; **falla** si es consistente con cero.

**3.4 Firma de apoyo S4 — Compromiso coherencia–desorden**

Defina una métrica de **desorden dinámico** en cada capa (elija una, pre-registre): (i) entropía de Shannon de intervalos entre eventos; (ii) entropía espectral; (iii) entropía de permutación.

**Predicción.** A través de capas,

| (3.5) |
|-------|

``` math
\text{corr}\left( {\Delta\widehat{\alpha}}_{n}, - {\Delta\widehat{S}}_{din,n} \right) > 0,
```

es decir, los aumentos en coherencia (pendiente) acompañan las reducciones en desorden dinámico bajo el mismo engrosamiento (narrativa operacional de "recodificación"). Pruebe con **Spearman** (robusto a no linealidad) y reporte ICs vía bootstrap.

**3.5 Lógica de decisión conjunta (pre-registrada)**

- **Apoyo:** S1 **y** S2 pasan.

- **Apoyo fortalecido:** S1 y S2 pasan **y** al menos uno de S3/S4 pasa.

- **Nulo / falsificación:** S1 falla (caída de pendiente significativa $`< - \varepsilon)`$ **o** S2 falla (sin asimetría hacia adelante). S3/S4 informativos pero no requeridos.

Establezca $`\varepsilon`$ por instrumento/diseño (ej., $`\varepsilon = 0.05\, - \, 0.1`$ en unidades de $`\alpha`$), pre-registre rangos de retardo y conteos de sustitutos para TE, y aplique BH-FDR a través de todas las pruebas por pares.

**3.6 Controles de robustez y confusores**

- **Ordenada al origen vs pendiente.** Las diferencias en factores a nivel de capa $`\Xi_{n}`$ (corrimiento al rojo/cinemática; ganancias globales) afectan **solo las ordenadas al origen** (§2.5). **No** interprete desplazamientos de ordenada al origen como cambios de coherencia.

- **Proxies de** $`\mathbf{L}`$**.** Reporte pendientes para **múltiples proxies de** $`\mathbf{L}`$ (geométrico/cinemático/estadístico); afirme S1 solo si las conclusiones son estables.

- **Pruebas de ventana.** Reajuste pendientes después de (i) descartar el $`L`$ más grande; (ii) usar solo los top-k tamaños; (iii) ajustes Huber/Theil–Sen para proteger contra valores atípicos.

- **Sensibilidad de causalidad.** Repita TE/Granger con (i) diferentes embeddings/retardos; (ii) tipos de sustitutos (mezcla temporal vs. fase aleatorizada); (iii) datos submuestreados para probar efectos de resolución temporal.

- **Controles negativos.** Incluya un **segmento nulo** con capas intencionalmente desacopladas; requiera que S2 sea nulo allí.

**3.7 Lista de verificación mínima de reporte (para métodos/resultados)**

1.  Proxies de $`L`$ (definiciones, incertidumbres) y dispersión dentro de capa.

2.  $`{\widehat{\alpha}}_{n}`$ con ICs bootstrap al 95%; $`\widehat{\Delta}\alpha_{n}`$ por pares con ICs.

3.  Configuraciones de TE/Granger (retardos, embeddings), conteos de sustitutos, valores p ajustados.

4.  Métricas S3/S4 (si se usan), incluyendo ICs bootstrap y tamaños de efecto.

5.  Resultados de robustez (cambios de proxy, pruebas de ventana, regresiones alternativas).

6.  ε pre-registrado, control de comparaciones múltiples, y **regla de falsificación**.

**Qué *no* contaría como apoyo.** Pendientes planas a través de capas con solo diferencias de ordenada al origen; simetría de TE/Granger; área de histéresis $`\mathcal{A}`$ consistente con cero; sin correlación coherencia–desorden. Cualquiera de estas niega la interpretación de coherencia secuencial (armónicos resonantes) **en ese sistema**, independientemente de narrativas motivacionales.

**4. Simulaciones y controles sintéticos (E1–E4)**

Esta sección valida las dos firmas centrales, **(S1)** coherencia monótona a través de capas (prueba de pendiente) y **(S2)** direccionalidad hacia adelante (TE/Granger), usando modelos sintéticos ligeros. Cada experimento especifica: **modelo**, **medición**, **regla de decisión**, y **patrones de resultado típicos**. También incluimos pruebas de estrés y un paquete mínimo de reproducibilidad.

**4.1 E1 — Cascada de cuatro capas con coherencia no decreciente (S1)**

**Modelo.** Capas $`n \in \{ 1,2,3,4\}`$ con

``` math
T_{n} = \Xi_{n}T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha_{n}}\varepsilon,\quad\alpha_{n} = \alpha_{\text{base}} + \Delta\alpha \cdot \frac{1}{1 + \exp\left( n - n_{c}/w \right)}
```

Aquí $`\Xi_{n}`$ es un factor a nivel de capa independiente de $`L`$ (solo nivel/"mapeo de reloj"); $`{log\varepsilon \sim N(0,\sigma}_{\log}^{2})`$. Elija $`L`$ en una cuadrícula geométrica (≥8–10 tamaños por capa; ≥1 década de dispersión).

**Medición.** Dentro de cada capa $`n`$, regresione $`\log T`$ sobre $`\log\ L`$ (OLS), reporte $`{\widehat{\alpha}}_{n}`$ e ICs bootstrap al 95% (remuestree eventos dentro de $`n`$, ≥1000 reps). Calcule diferencias adyacentes $`{\widehat{\Delta}\alpha}_{n} = {\widehat{\alpha}}_{n + 1} - {\widehat{\alpha}}_{n}`$ con ICs.

**Regla de decisión (S1).** **Pasa** si todo $`inf\left( {\widehat{\Delta}\alpha}_{n} \right) \geq - \varepsilon`$. Verificación global opcional: el ajuste isotónico (no decreciente) para $`\alpha_{n}`$ no es significativamente peor que el sin restricción (bootstrap LR).

Patrón típico. $`{\widehat{\alpha}}_{n}`$ sube (o se estabiliza) con $`n`$; los ICs no muestran caídas significativas; las ordenadas al origen difieren entre capas pero no afectan las pendientes.

**4.2 E2 — Causalidad direccional en una cadena estratificada (S2)**

**Modelo.** Los observables de capa $`X_{n}(t)`$ obedecen un proceso bivariado **acoplado hacia adelante** (por pares) entre vecinos:

``` math
X_{n + 1}(t) = \sum_{j = 1}^{p}{a_{j}X_{n + 1}(t - j)} + \sum_{j = 1}^{p}{b_{j}X_{n}(t - j)} + \eta_{n + 1}(t),
```

``` math
X_{n}(t) = \sum_{j = 1}^{p}{c_{j}X_{n}(t - j)} + \nu_{n}(t),
```

con $`b_{j} \neq 0`$ (hacia adelante), sin acoplamiento reverso en este experimento. Genere $`{\sim 10}^{3}`$ muestras/par de capa; coincida órdenes de retardo por AIC/BIC.

**Medición.**

- **TE:** estime $`{TE}_{n \rightarrow n + 1}`$ y $`{TE}_{n + 1 \rightarrow n}`$ con embeddings coincidentes; obtenga valores $`p`$ vía pruebas de sustitutos (permutación/mezcla de fase, $`\geq 1000`$).

- **Granger:** pruebas F sobre $`b_{j}`$ vs. $`0`$; verifique la dirección reversa por separado.

**Regla de decisión (S2).** Afirme direccionalidad hacia adelante si **ambos** TE y Granger son significativos para $`n \rightarrow n + 1`$ y **no** para $`n + 1 \rightarrow n`$ (ajustado por FDR).

Patrón típico. $`{TE}_{n \rightarrow n + 1} \gg {TE}_{n + 1 \rightarrow n}`$; Granger significativo solo hacia adelante. Cuando se reduce el acoplamiento hacia adelante, ambas métricas disminuyen suavemente hacia el nulo.

**4.3 E3 — Trinquete/histéresis bajo barridos de acoplamiento (apoyo S3)**

**Modelo.** Introduzca un acoplamiento controlable $`g \in \lbrack gmin,gmax\rbrack`$ que module ya sea $`\alpha_{n + 1}(g)`$ (a través de organización efectiva) o los coeficientes hacia adelante $`b_{j}(g)`$. Barra $`g`$ **hacia arriba** y luego **hacia abajo**, permitiendo que un estado interno lento produzca memoria.

**Medición.** Rastree $`{\widehat{\alpha}}_{n + 1}(g)`$ (pendiente por capa en cada $`g`$) y calcule el **área del ciclo** $`\mathcal{A}_{\mathcal{n} + 1} = \oint_{}^{}{\widehat{\alpha}}_{n + 1}(g)\, dg`$ usando integración trapezoidal.

**Regla de decisión (S3).** **Pasa** si el IC bootstrap de $`\mathcal{A}_{\mathcal{n} + 1}`$ excluye 0 (memoria direccional); de lo contrario **sin trinquete**.

**Patrón típico.** La rama hacia adelante muestra activación de $`\widehat{\alpha}`$ más temprana/mayor que la rama hacia atrás; área del ciclo $`> 0`$ dentro del IC.

**4.4 E4 — Controles nulos (pendientes planas y causalidad simétrica)**

**Modelo.** Mantenga $`\alpha_{n} \equiv \alpha_{\star}`$ constante para todo $`n`$ y establezca acoplamientos simétricos o cero. Mantenga los factores de capa $`\Xi_{n}`$ heterogéneos para asegurar que las diferencias de ordenada al origen permanezcan presentes.

**Medición y decisión.**

- **S1:** Los ICs de $`{\widehat{\Delta}\alpha}_{n}`$ adyacentes incluyen 0 (sin tendencia monótona).

- **S2:** TE y Granger son simétricos o no significativos después de FDR.

**Patrón típico.** $`{\widehat{\alpha}}_{n}`$ plano a través de capas con desplazamientos de ordenada al origen no nulos; TE/Granger no muestran una dirección favorecida, esto protege contra falsos positivos.

**4.5 Pruebas de estrés (robustez y modos de falla)**

- **Ruido de proxy para** $`\mathbf{L}`$**.** Reemplace $`L`$ verdadero por proxies con error multiplicativo; la **pendiente** permanece estable cuando los errores son i.i.d. dentro de una capa; el sesgo severo dependiente de capa puede imitar $`\Delta\alpha`$ (señale vía proxies alternativos y pruebas de ventana).

- **Dispersión en** $`\mathbf{L}`$**.** Reducir el rango de $`L`$ infla los ICs; la potencia cae abruptamente por debajo de $`\sim 6`$ tamaños distintos/capa o $`< 0.5`$ décadas de dispersión.

- **Ruido heteroscedástico/de colas pesadas.** Use ICs bootstrap; ejecute sensibilidad Huber/Theil–Sen, las afirmaciones deben persistir.

- **Agrupamiento incorrecto entre capas.** Mezclar $`\Xi_{n}`$ distintos dentro de una capa puede filtrar efectos de nivel en estimaciones de pendiente; mitigue con bins estrechos y definiciones de proxy consistentes.

- **Configuraciones de causalidad.** TE/Granger son sensibles a embedding/retardos; pre-registre rangos y verifique direccionalidad bajo múltiples elecciones razonables; use sustitutos rigurosamente.

**4.6 Paquete mínimo de reproducibilidad**

Liberamos (i) scripts para generar datos para E1–E4 con una semilla RNG fija, (ii) estimadores de pendiente OLS+bootstrap, (iii) rutinas TE/Granger con pruebas de sustitutos, y (iv) scripts de graficación. Las salidas incluyen CSVs por capa ($`{\widehat{\alpha}}_{n}`$​, ICs, métricas TE/Granger) y figuras PNG para cada experimento. Un breve **README** documenta entradas, parámetros, y la lógica de decisión (S1–S2, más S3/S4 cuando se usan).

**Resumen de resultados sintéticos**

A través de E1–E4 la tubería se comporta como se pretende: cuando una cascada hacia adelante está presente, las **pendientes son no decrecientes** y el **flujo causal es asimétrico**; cuando está ausente, las **pendientes son planas** y la **direccionalidad se desvanece**, a pesar de los desplazamientos de ordenada al origen. Estos controles muestran que el programa RTM estratificado produce firmas empíricas **sensibles** y **específicas**, preparando el escenario para análogos de laboratorio y análisis observacionales.

**5. Experimentos análogos (diseño y protocolos)**

Esta sección convierte la cascada RTM en **protocolos de laboratorio** que pueden producir las dos firmas centrales: **(S1)** pendiente no decreciente $`\alpha_{n}`$​ a través de capas y **(S2)** flujo de información solo hacia adelante (TE/Granger). Cada plataforma define: **capas**, un **proxy de tamaño efectivo** $`L`$, un **tiempo mesoscópico** $`T`$, un **acoplamiento direccional** $`n \rightarrow n + 1`$ con intensidad sintonizable $`g`$, y una tubería de medición que aísla **pendiente (coherencia)** de **ordenada al origen (nivel/mapeo de reloj)**.

**5.1 Plataforma A — Cadena direccional de resonadores acoplados (óptico / RF / mecánico)**

**Objetivo.** Realizar $`N`$ capas anidadas como una **serie de resonadores** con **acoplamiento unidireccional**. Ejemplos:

- **Óptico:** cavidades de anillo de fibra o micro-anillo enlazadas por **aisladores ópticos** o circuladores.

- **RF/microondas:** cavidades superconductoras o a temperatura ambiente con **circuladores** (no recíprocos).

- **Mecánico:** cantilevers débilmente acoplados/resonadores masa–resorte con **retroalimentación unidireccional** activa.

**Definición de capa y observables.**

- **Capa** $`\mathbf{n}`$**:** el $`n`$-ésimo resonador.

- **Proxy de tamaño** $`\mathbf{L}`$**:** **ancho de pulso** inyectado (temporal), o **ancho de banda espectral** (frecuencia) tratado como una "escala" efectiva. Use $`\geq 6 - 8`$ $`L`$ distintos por capa.

- **Tiempo mesoscópico** $`\mathbf{T}`$**:** **tiempo de decaimiento** de la cavidad, **tiempo de amortiguamiento**, o **tiempo de primer paso/escape** de la envolvente del pulso.

**Control y direccionalidad.**

- **Unidireccionalidad:** aislador/circulador entre $`n`$ y $`n + 1`$; bloquear $`n + 1 \rightarrow n`$.

- **Intensidad de acoplamiento** $`\mathbf{g}`$**:** establecida por transmisividad del acoplador / capacitancia de acoplamiento / ganancia de retroalimentación. Barra $`g`$ (arriba y abajo) para **histéresis** (S3).

**Adquisición.**

- Inyecte familias de pulsos en **cada capa** (o solo en la primera si propaga el mismo pulso en cascada).

- Para cada $`n`$, recolecte $`m_{n}`$ eventos por $`L`$ (objetivo $`m_{n} \geq 30`$) y muestree series temporales $`X_{n}(t)`$ (ej., envolvente o energía) a $`\geq 10 \times`$ la dinámica más rápida.

**Análisis.**

- **S1:** OLS de $`\log\ T`$ vs $`\log\ L`$ por capa $`\rightarrow {\widehat{\alpha}}_{n} +`$ **ICs bootstrap al 95%**. Verifique que los ICs de $`{\widehat{\Delta}\alpha}_{n} = {\widehat{\alpha}}_{n + 1} - {\widehat{\alpha}}_{n}`$ no sean $`\text{<} - \varepsilon`$.

- **S2:** Calcule **TE** $`(n \rightarrow n + 1)`$ vs $`(n + 1 \rightarrow n)`$ con valores p de sustitutos; ejecute **Granger** (VAR bivariado) con orden elegido por AIC/BIC. Requiera significancia solo hacia adelante (BH-FDR).

- **S3 (opcional):** grafique $`{\widehat{\alpha}}_{n + 1}`$ para barridos arriba/abajo; bootstrap del **área del ciclo** $`\mathcal{A}`$ y pruebe $`\mathcal{A} \neq 0`$.

**Control de confusores.**

- **Ordenada al origen vs pendiente**: las pérdidas y ganancias de trayecto cambian **ordenadas al origen; solo las pendientes** diagnostican coherencia.

- **Retardo de fase/grupo**: trate como un factor de nivel separado $`\Xi_{n}`$​; manténgalo fijo dentro de cada ajuste de pendiente.

- **Estacionariedad:** elimine tendencia de las series temporales antes de TE/Granger; valide con pruebas de raíz unitaria.

**Pasa/Falla.** **Pasa** si $`{\widehat{\alpha}}_{n}`$ es no decreciente dentro de $`\varepsilon`$ **y** la causalidad solo hacia adelante es significativa. **Falla** si alguna caída de pendiente adyacente $`< - \varepsilon`$ o la direccionalidad es simétrica.

**5.2 Plataforma B — Guías de onda fluídicas/fonónicas en cascada con confinamiento creciente**

**Objetivo.** Construir un **canal anidado** (ej., canal de agua con deflectores, guías de onda acústicas/fonónicas) donde el confinamiento **aumenta** corriente abajo.

**Capa y observables.**

- **Capa** $`\mathbf{n}`$**:** segmento entre deflectores (o la $`n`$-ésima celda de guía de onda).

- **Proxy de tamaño** $`\mathbf{L}`$**:** **diámetro de gota** inyectada (fluido), **ancho de paquete** espacial (acústico), o **escala de wavelet dominante** de imagen.

- **Tiempo mesoscópico** $`\mathbf{T}`$**:** tiempo de **tránsito / escape / decaimiento** medido por video de alta velocidad o sensores de presión/acústicos.

**Control.**

- **Índice de confinamiento** $`\mathbf{g}`$**:** ancho de boquilla, espaciado de deflectores, o fineza de cavidad → **monótono** a través de capas.

- **Direccionalidad:** flujo impuesto o elementos acústicos tipo diodo para suprimir la retropropagación.

**Adquisición y análisis.**

- Replique el protocolo de pendiente **S1** por capa con $`\geq 6 - 8`$ tamaños $`L`$.

- Para **S2**, calcule TE/Granger entre sensores corriente arriba–corriente abajo.

- Registre los números de Reynolds/Froude para documentar el régimen; manténgalos **constantes dentro de una corrida** (factor de ordenada al origen).

**Confusores.**

- **Turbulencia on/off:** reporte el régimen; si la turbulencia varía por capa, trátela como $`\Xi_{n}`$ (ordenada al origen) y verifique la estabilidad de la pendiente.

- **Sesgo de imagen:** calibre $`L`$ de gota/wavelet contra un objetivo; ejecute **sensibilidad de proxy** (geométrico vs. estadístico $`L`$).

**Pasa/Falla.** Como en 5.1.

**5.3 Plataforma C — Escalera electrónica (RLC/activa) con acoplamiento unidireccional**

**Objetivo.** Una realización de **banco de laboratorio accesible**: una cadena de celdas RLC (capas) con **enlaces activos no recíprocos** (buffers op-amp / giradores / redes de diodos) para emular acoplamiento unidireccional.

**Capa y observables.**

- **Capa n:** nodo de salida de la $`n`$-ésima celda.

- **Proxy de** $`\mathbf{L}`$**:** **ancho** de pulso de entrada o **ancho de banda del filtro** (establecido por RC).

- $`\mathbf{T}`$**:** tiempo de decaimiento (envolvente $`1/e`$), tiempo de subida/estabilización, o tiempo de primer paso de umbral.

**Control.**

- **Acoplamiento** $`\mathbf{g}`$**:** resistor/ganancia controlable en la trayectoria hacia adelante solamente. Incluya un barrido arriba/abajo para **histéresis**.

**Análisis y confusores.**

- Aplique la misma tubería **S1/S2**; caracterice el **piso de ruido** y el **muestreo ADC** como factores de nivel.

- Use robustez **Theil–Sen** si aparecen valores atípicos; confirme la estabilidad de la pendiente al descartar el $`L`$ más grande.

**5.4 Lista de verificación de medición (por plataforma)**

1.  **Dispersión de** $`\mathbf{L}`$ **dentro de capa:** $`\geq 6 - 8`$ tamaños distintos; apunte a $`\gtrsim 1`$ década.

2.  **Réplicas:** $`m_{n} \geq 30`$ eventos por $`L`$ por capa para ICs bootstrap confiables.

3.  **Longitud de serie temporal (S2):** $`\mathbf{10}^{\mathbf{2}}\mathbf{-}\mathbf{10}^{\mathbf{3}}`$ muestras efectivas por par adyacente; pre-registre rangos de retardo.

4.  **Hardware de direccionalidad:** aisladores/circuladores/diodos documentados; atenuación de trayecto reverso medida (dB).

5.  **Controles:** incluya un **segmento nulo** (recíproco o desacoplado) para verificar que S2 retorne simetría.

**5.5 Tubería de datos y análisis (pre-registrada)**

- **Pre-procesamiento:** elimine tendencia, limite banda si es necesario; marque eventos con tiempo; calcule $T$ de umbrales consistentes.

- **Ajustes de pendiente:** OLS en $`\log\ T - \log L`$, ICs; pruebas de ventana (descarte el $`L`$ más grande, tamaños top-$`k`$).

- **Causalidad:** TE con sustitutos de permutación/mezcla de fase; Granger con selección de orden AIC/BIC; corrección **BH-FDR**.

- **Decisión:** S1 pasa si todo inf $`({\widehat{\Delta}\alpha}_{n}) \geq - \varepsilon`$; S2 pasa si solo hacia adelante es significativo.

- **Registro de artefactos:** documente cualquier desplazamiento de $`\Xi_{n}`$ específico de capa (ganancias, retardos, cambios de régimen).

**5.6 Patrones esperados y modos de falla**

**Patrones de apoyo.** $`{\widehat{\alpha}}_{n}`$ monótono (sube/meseta) a través de capas; TE/Granger significativo solo hacia adelante; área de histéresis $`A > 0`$ al barrer $`g`$.

**Patrones sin apoyo.** $`{\widehat{\alpha}}_{n}`$ plano o **decreciente** más allá de $`- \varepsilon`$; TE/Granger simétrico; $`A \approx 0`$ después de barridos; conclusiones de pendiente frágiles a la elección de proxy de $`L`$.

**5.7 Consideraciones prácticas, seguridad y ética**

- **Seguridad:** aislamiento láser/óptico (gafas OD), precauciones de alto voltaje en RF/electrónica, seguridad de salpicaduras/impulsor para fluidos.

- **Materiales abiertos:** libere CAD/esquemáticos, BOM, firmware, scripts de adquisición, y notebooks de análisis (con semillas) para permitir replicación.

- **Registro:** pre-registre $`\varepsilon`$, rangos de retardo, y segmentos nulos; archive datos crudos y código.

**Conclusión.** Las tres plataformas arriba proporcionan **rutas independientes** para probar la hipótesis RTM de **coherencia secuencial** bajo condiciones controladas. Un resultado positivo requiere **tanto** monotonicidad de pendiente (S1) como causalidad solo hacia adelante (S2); un resultado nulo o mixto argumenta contra la interpretación de cascada de Armónicos Resonantes **en esa plataforma**, precisamente el estándar de falsificabilidad que queremos.

**6. Discusión**

**6.1 Qué significaría un resultado positivo**

Un **aumento (o no disminución)** consistente de $`{\widehat{\alpha}}_{n}`$​ a través de capas **y** una señal de TE/Granger **solo hacia adelante** indica que:

- La coherencia (como es capturada por la pendiente RTM) **se acumula** a lo largo de la secuencia; y

- La **influencia causal** se propaga **de** la capa $`n`$ **a** $`n + 1`$, no simétricamente.

En términos operacionales, el sistema está realizando una **recodificación secuencial** de la dinámica en comportamiento mesoscópico cada vez más organizado. Esta es exactamente la lectura empírica de la "arquitectura de armónicos resonantes" conceptual: la metáfora de "transmutación de información en un código más ordenado" se convierte en una firma concreta de **pendiente-y-direccionalidad**.

**Consecuencias.** Un resultado positivo justifica:

- Mapear $`\alpha_{n}`$ vs. $`n`$ como un nuevo **diagnóstico estructural** (comparable a través de plataformas).

- Estudiar cómo $`\alpha`$ depende de variables de control (confinamiento, acoplamiento, estratificación), para inferir **curvas de respuesta** y potenciales **puntos críticos**.

- Comenzar un programa microfísico para **derivar** $`\alpha`$ de interacciones efectivas (ej., acoplamiento jerárquico, bloqueo de fase, supresión de transporte), en lugar de tratarlo como un exponente puramente fenomenológico.

**6.2 Qué significaría un resultado nulo o mixto**

Si (i) las pendientes por capa permanecen **planas** (los ICs de $`\Delta\widehat{\alpha}`$ incluyen cero o son negativos más allá de la tolerancia) y/o (ii) TE/Granger es **simétrico**, la hipótesis de cascada de Armónicos Resonantes **no está apoyada** en ese sistema. Esto no es una falla del método: es la falsificabilidad deseada. Prácticamente:

- El enfoque cambia a **por qué** la coherencia no se acumula: acoplamiento insuficiente, contra-flujos, desajuste de proxy para $`L`$, o física intrínseca que simplemente carece de una cascada unidireccional.

- Los resultados negativos en análogos ayudan a **afinar** diseños para corridas subsecuentes (ej., enlaces no recíprocos más fuertes, mayor dispersión en $`L`$, series temporales más largas).

**6.3 Ordenadas al origen vs. pendientes y la separación del libro mayor**

A través de todos los análisis, las **ordenadas al origen** absorben factores de nivel (mapeo de reloj, ganancias globales, líneas base de régimen) y **no** son evidencia de cambio organizacional. Las **pendientes** son el libro mayor de coherencia. Esta separación es la razón principal por la que el enfoque permanece compatible con dinámica estándar (ej., GR en configuraciones astrofísicas): puede tener grandes desplazamientos de ordenada al origen sin tocar $`\alpha`$.

**6.4 Cómo leer** $`\mathbf{\alpha}`$ **microfísicamente**

Aunque $`\alpha`$ se mide estadísticamente, plausiblemente codifica:

- **Modo de transporte:** balístico $`(\alpha \approx 1)\  \rightarrow \ difusivo\ (\alpha \approx 2)\  \rightarrow \ `$ tiempos mesoscópicos **super-comprimidos** en $`\alpha`$ más grande.

- **Organización de fase:** un bloqueo de fase más fuerte o alineación de retroalimentación puede **elevar** α al reducir grados de libertad efectivos a una escala dada.

- **Confinamiento multiescala:** trampas/guías de onda/cavidades anidadas promueven acoplamiento **jerárquico** que empina la ley tiempo–tamaño.

La teoría futura debería conectar $`\alpha`$ con **ecuaciones de grano grueso** (ej., difusión generalizada con núcleos de memoria; redes de osciladores acoplados con enlaces dirigidos) para predecir cómo $`\alpha`$ cambia bajo modificaciones controladas del medio.

**6.5 Alcance entre dominios**

La misma tubería, estimación de pendiente estratificada + TE/Granger, aplica a:

- **Análogos de laboratorio:** cadenas de resonadores óptico/RF/mecánico; guías fluídicas/fonónicas con confinamiento creciente (como se diseñó en §5).

- **Sistemas observacionales:** cualquier entorno con **capas estratificadas** o **módulos** donde familias de procesos puedan medirse sobre un rango de tamaños efectivos $`L`$ (ej., envolturas espaciales, bandas de altitud, regiones anidadas).

El requisito crucial es una **dispersión dentro de capa en** $`\mathbf{L}`$ suficiente para ajustar una pendiente con incertidumbre útil, y series temporales suficientemente largas para estimar direccionalidad.

**6.6 Trampas y cómo evitarlas**

- **Dispersión estrecha en** $`\mathbf{L}`$ **/ muy pocos tamaños:** infla ICs y oculta tendencias. *Mitigación:* diseñe para ≥6–8 $`L`$ distintos por capa y ≥0.5–1 década de dispersión.

- **Deriva de proxy para** $`\mathbf{L}`$**:** diferentes proxies a través de capas pueden imitar $`\Delta\alpha`$. *Mitigación:* reporte **múltiples proxies** y requiera estabilidad de conclusiones.

- **Agrupamiento incorrecto de capa:** mezclar regímenes distintos dentro de una capa puede filtrar efectos de nivel en pendientes. *Mitigación:* bins más estrechos; documente indicadores de régimen como parte de $`\Xi_{n}`$.

- **Sobreajuste de causalidad:** TE/Granger son sensibles a embeddings/retardos. *Mitigación:* pre-registre rangos de retardo; use pruebas de sustitutos; aplique FDR a través de pares.

- **Sesgo de N pequeño en TE:** series temporales cortas inflan falsas asimetrías. *Mitigación:* pruebas de submuestreo, ICs de bootstrap por bloques, y segmentos de control negativo con simetría conocida.

**6.7 Relación con las narrativas motivacionales**

El lenguaje conceptual sobre **recodificación** y "simulacrum" permanece como **motivación**, no como una afirmación empírica. Las afirmaciones de este artículo se sostienen o caen sobre **dos observables**: (S1) **pendientes no decrecientes** a través de capas; (S2) **direccionalidad solo hacia adelante**. La evidencia positiva **motivaría** exploración más profunda de mecanismos de recodificación; la evidencia nula **acotaría** esas narrativas sin prejuicio.

**6.8 Qué permite esto a continuación**

- Una **suite de referencia**: publique pendientes $`{\widehat{\alpha}}_{n}`$​, ICs, y tablas TE/Granger para cada plataforma/fuente, permitiendo comparación directa entre laboratorios y conjuntos de datos.

- **Mapas de respuesta**: mida $`\alpha_{n}(g)`$ como función de acoplamiento/confinamiento para identificar **regiones de operación** donde las ganancias de coherencia son mayores.

- **Hacia derivaciones**: use mapas de $`\alpha`$ empíricos para restringir **modelos efectivos** candidatos (núcleos de memoria de transporte, grafos de acoplamiento dirigido, confinamiento multiescala), guiando derivaciones en lugar de postulados.

- **Ángulo de ingeniería**: si $`\alpha`$ monótono y TE hacia adelante son robustos, se puede apuntar a **diseñar** cascadas que deliberadamente **eleven** $`\alpha`$ capa por capa para tareas de control o procesamiento de información, claramente marcado como seguimiento de ingeniería, no parte de las afirmaciones presentes.

**Conclusión.** La cascada RTM es ahora una historia **testeable**: o las **pendientes suben (o se mantienen) hacia adelante** y la **causalidad apunta hacia adelante**, o no. Ambos resultados son científicamente valiosos, uno abre un programa microfísico y de ingeniería; el otro descarta limpiamente una narrativa seductora pero innecesaria para ese sistema.

7.  **Divergencia Estructural y Bifurcación del Espacio de Fase**

**7.1 De la Coherencia Local a la Topología Global**

En las secciones anteriores, establecimos la cascada como el principio organizador a través del cual la coherencia se propaga a través de escalas. Sin embargo, esta propagación está sujeta a restricciones de estabilidad. Cada nodo en la cascada representa un locus de **estabilidad dinámica** cuya persistencia depende del alineamiento de fase con sus dominios adyacentes.

Cuando estos alineamientos derivan más allá de un umbral crítico, la trayectoria del sistema en el espacio de fase ($`\backslash Gamma`$) pierde unicidad. La variedad de evoluciones posibles se refracta en múltiples atractores estables.

Esta sección formaliza este fenómeno no como una divergencia metafísica, sino como **Bifurcación del Espacio de Fase**: una separación estructurada de trayectorias impulsada por la dinámica interna del acoplamiento $`\backslash alpha`$.

**7.2 Mecanismos de Separación de Fase**

Dentro del formalismo RTM, la coherencia es una función dependiente de escala. Cuando el coeficiente de acoplamiento entre capas adyacentes cae por debajo de un valor crítico ($`C_{crit}`$), el sistema experimenta **Ruptura de Simetría Espontánea**.

Sea el vector de estado de una capa $`\backslash varphi_{n}(t)`$. La condición de continuidad es:

``` math
C_{n,n + 1}(t) = \backslash cos\left\lbrack \varphi_{n + 1}(t) - \varphi_{n}(t) \right\rbrack \geq C_{crit}
```

Cuando $`C_{n,n + 1} < C_{crit}`$, la correspondencia causal entre capas se degrada. La "divergencia" es esencialmente un **evento de decoherencia**: las capas se desacoplan y evolucionan a lo largo de trayectorias termodinámicas distintas.

Lo que a menudo se modela en cosmología como "burbujas distintas" puede describirse rigurosamente aquí como **armónicos de fase ortogonales** dentro de un solo espacio de estados de alta dimensión.

**7.3 Inestabilidad Estructural Alométrica (El Límite de Impedancia)**

Una restricción crítica sobre este acoplamiento es la **Inestabilidad Alométrica**.

Así como las estructuras biológicas obedecen leyes de cuadrado-cubo, las estructuras temporales obedecen **Límites de Latencia de Información**.

Si la diferencia de escalamiento entre dos dominios ($`\backslash Delta\ \backslash alpha`$) excede un umbral estructural ($`\backslash Delta\ \backslash alpha\  > \ 0.5`$), la razón de escalamiento métrico se vuelve incompatible:

``` math
\rho_{eff} \propto k^{- 4\Delta\backslash alpha}
```

Esto crea un **Desajuste de Impedancia**. Cualquier señal coherente que intente cruzar este gradiente experimenta distorsión asintótica.

Operacionalmente, esto proporciona un límite superior físico sobre el rango de interacción en sistemas multiescala: la coherencia no puede mantenerse a través de gradientes arbitrariamente pronunciados sin un mecanismo "transformador" intermedio (ej., escalamiento resonante).

**7.4 Cierre Informacional (El Límite de Retroalimentación)**

En el límite superior de la cascada, donde la desviación de fase se vuelve despreciable, el sistema entra en un régimen de **Cierre Informacional**.

En este estado, el sistema transiciona de ser un muestreador externo del campo a un nodo autoconsistente dentro de él. El ciclo de retroalimentación entre la estimación del estado del sistema y la dinámica ambiental se estabiliza ($`dI_{entrada}\text{/}dt \approx dI_{salida}\text{/}dt`$).

Esto es consistente con el **Principio de Energía Libre** en biología teórica: el sistema minimiza su energía libre variacional (sorpresa) maximizando la coherencia de su modelo interno con el entorno.

**8. Limitaciones, supuestos y modos de falla**

Este capítulo declara lo que nuestra prueba de cascada RTM **sí** y **no** establece, los supuestos bajo los cuales las estadísticas son válidas, y las circunstancias concretas que **invalidarían** la afirmación.

**8.1 Alcance y no-afirmaciones**

- $`\mathbf{\alpha}`$ **operacional, no entropía.** $`\alpha`$ es una pendiente en $`T\backslash log\ L`$; **no** es entropía termodinámica ni una constante microfísica.

- **Sin dinámica modificada.** Los factores de nivel (ganancias, retardos, GR/cinemática) se tratan como **ordenadas al origen**; la dinámica del medio es por lo demás estándar.

- **Narrativas motivacionales.** "Recodificación/simulacrum/armónicos_resonantes" sirven como **motivación**, no como afirmaciones empíricas a menos que estén apoyadas por S1–S2.

**8.2 Identificabilidad y requisitos de diseño**

- **Dispersión dentro de capa en** $`\mathbf{L}`$**.** Estimar $`\alpha_{n}`$ requiere $`\geq 6 - 8`$ tamaños efectivos distintos y preferiblemente $`\gtrsim 1`$ década de dispersión; de lo contrario los ICs se inflan y las tendencias se difuminan.

- **Proxy de** $`\mathbf{L}`$ **consistente por capa.** Mezclar diferentes definiciones de $`L`$ a través de capas puede imitar $`\Delta\alpha`$. Reporte **múltiples proxies** y requiera estabilidad de conclusión.

- **Réplicas.** Apunte a $`m_{n} \geq 30`$ eventos por $`L`$ por capa; para TE/Granger use $`10^{2} - 10^{3}`$ muestras efectivas.

- **Estacionariedad para S2.** Aplique eliminación de tendencia/diferenciación según sea necesario; valide con pruebas de raíz unitaria/residuales.

**8.3 Supuestos estadísticos (y cómo los relajamos)**

- **Modelo de ruido.** OLS asume residuos homoscedásticos en $`log\ T`$; nos protegemos con **ICs bootstrap** y verificaciones de robustez (Huber, Theil–Sen).

- **Errores en variables (EIV).** El ruido de medición en $`L`$ sesga las pendientes **hacia cero**; ejecute sensibilidad **SIMEX**/proxy instrumental y pruebas de ventana ("descarte el $`L`$ más grande", tamaños top-$`k`$).

- **Embeddings de causalidad.** TE/Granger dependen de la elección de retardo/embedding; pre-registramos rangos y usamos **sustitutos** + **FDR** a través de pares.

**8.4 Modos de falla concretos (qué invalidaría el apoyo)**

- Caída de pendiente significativa más allá de $`- \varepsilon`$ entre cualquier par de capas adyacentes.

- TE/Granger simétrico o reverso significativo después de FDR.

- Conclusiones de pendiente que cambian al alternar proxies de $`L`$ o al ejecutar pruebas de ventana.

- Segmento nulo que no retorna simetría (indica artefactos en la tubería de direccionalidad).

**8.5–8.7 [Secciones de robustez y verificación adicionales]**

Se aplican pruebas de sensibilidad adicionales como se describe en las secciones de métodos, incluyendo verificaciones de proxy alternativo, análisis de submuestreo y validación de control negativo.

**8.8 Ética, seguridad y apertura**

- **Seguridad:** precauciones de láser/RF/fluidos y documentación de hardware no recíproco.

- **Apertura:** libere semillas, scripts, CAD/esquemáticos, BOMs, y datos crudos para permitir replicación completa.

- **Atribución:** etiquete claramente el contenido motivacional vs. las afirmaciones empíricas.

**8.9 Conclusión**

La afirmación de cascada **se sostiene o cae** sobre dos observables: **(S1)** $`\alpha_{n}`$ no decreciente y **(S2)** direccionalidad solo hacia adelante. Si cualquiera falla bajo los controles anteriores, o si los resultados dependen de elecciones de proxy o desaparecen bajo verificaciones de robustez, la interpretación **no está apoyada** en ese sistema. Esa falsificabilidad es una característica, no un defecto.

**9. Conclusión y perspectiva**

**Qué hicimos.** Tradujimos la narrativa de "Arquitectura de Armónicos Resonantes" en un **programa RTM testeable** con dos firmas centrales, operacionales: **(S1)** **pendiente** log–log no decreciente $`\alpha_{n} = \partial\ \log\ T/\partial\ \log\ L`$ a través de capas anidadas, y **(S2)** **direccionalidad solo hacia adelante** (entropía de transferencia / Granger) de la capa $`n`$ a $`n + 1`$. Separamos **pendientes** (coherencia/organización) de **ordenadas al origen** (factores de nivel/mapeo de reloj), manteniendo intacta la dinámica estándar.

**Qué encontramos (sintético).** La suite E1–E4 muestra que el método es **sensible** (detecta $`\alpha`$ creciente cuando está presente), **específico** (no inventa activación bajo nulos), y **diagnóstico** (las ordenadas al origen pueden moverse fuertemente sin alterar las pendientes). Estos controles reducen el riesgo del análisis antes de moverse a plataformas de laboratorio y conjuntos de datos observacionales.

**Qué significa esto.** La cascada RTM se convierte en una afirmación falsificable: o $`\alpha`$ sube (o al menos no cae) a lo largo de la secuencia **y** la causalidad apunta hacia adelante, o no. Ambos resultados son informativos: un resultado positivo motiva modelado microfísico de cómo el acoplamiento dirigido empina la ley tiempo–tamaño; un resultado nulo acota limpiamente la narrativa en ese sistema.

**9.1 Próximos pasos prácticos**

1.  **Ejecute la tubería en cadenas análogas.**\
    Construya cualquiera de las plataformas de escalera (resonador, electrónica, fluido/fonónica). Pre-registre: $`\varepsilon`$ para S1, rangos de retardo/embedding y conteos de sustitutos para S2, y un segmento nulo para controles de direccionalidad. Apunte a $`\geq 6 - 8`$ valores distintos de $`L`$ por capa y $`10^{2} - 10^{3}`$ muestras efectivas para TE/Granger.

2.  **Reporte pendientes primero, luego causalidad.**

> Publique $`{\widehat{\alpha}}_{n}`$ con ICs bootstrap al 95% y diferencias adyacentes $`{\widehat{\Delta}\alpha}_{n}`$​; solo entonces agregue TE/Granger (ambas direcciones, ajustado por FDR). Haga explícita la **regla de falsificación**.

3.  **Robustez por diseño.**\
    Repita S1 con **proxies de** $`\mathbf{L}`$ **alternativos** y pruebas de ventana (descarte el $`L`$ más grande, tamaños top-$`k`$); repita S2 con múltiples elecciones de retardo/embedding y familias de sustitutos (mezcla temporal, fase aleatorizada).

4.  **Materiales abiertos.**\
    Libere semillas, scripts, CAD/esquemáticos (si hay hardware), datos crudos, y notebooks. Un breve README con "cómo reproducir en tres comandos" elimina ambigüedad.

**9.2 Beneficios científicos si las firmas se sostienen**

- **Un nuevo descriptor cuantitativo.** Mapas de $`\alpha_{n}`$​ a través de capas actúan como **diagnósticos estructurales** de organización, comparables a través de plataformas y laboratorios.

- **Curvas de control.** Medir $`\alpha_{n}(g)`$ contra acoplamiento/confinamiento traza funciones de respuesta y potenciales umbrales para activación de coherencia.

- **Puente a la teoría.** Los perfiles de $`\alpha`$ observados restringen modelos efectivos (transporte con núcleo de memoria, redes de osciladores dirigidos, confinamiento jerárquico), guiando derivaciones en lugar de postulados.

**9.3 Límites y lo que *no* afirmamos**

- $`\alpha`$ es una **pendiente operacional**, no una entropía termodinámica o nueva constante fundamental.

- Los factores de nivel/mapeo de reloj (ganancias, retardos, GR/cinemática) viven en el libro mayor de **ordenadas al origen**; **no** son evidencia de cambio de coherencia.

- La imaginería más amplia de "simulación/recodificación" permanece como **motivación**, no como un resultado empírico, a menos que S1–S2 tengan éxito bajo los controles pre-registrados.

**9.4 Si los resultados son nulos o mixtos**

Un perfil de pendiente plano (o decreciente más allá de la tolerancia) y direccionalidad simétrica **falsifican la cascada** en ese sistema. Esto es éxito del método: previene la sobre-interpretación y enfoca el trabajo futuro en por qué la coherencia **no** se acumula (acoplamiento unidireccional insuficiente, problemas de proxy, mezcla de régimen) o en observables alternativos mejor adaptados al medio.

**Conclusión.** El trabajo convierte una historia multiescala convincente en **asas empíricas limpias**. Mida pendientes; separe ordenadas al origen; pruebe direccionalidad. Si la cascada hacia adelante existe, debería aparecer en estos dos números. Si no, la respuesta es igualmente valiosa, e inequívoca.

**10. Evidencia de simulación integrada**

**10.1 Propósito y diseño**

Usamos cinco simulaciones ligeras para auditar las dos firmas RTM centrales bajo condiciones controladas: **(S1)** pendiente no decreciente $`\alpha`$ a través de capas y **(S2)** direccionalidad solo hacia adelante. Cada experimento retorna CSVs y figuras con semillas RNG fijas para replicación (paquete E1–E4).

**10.2 E1 — Cascada de cuatro capas (S1: monotonicidad de pendiente)**

**Configuración.** Cuatro capas, aumento logístico del exponente de coherencia verdadero $`\alpha_{n}`$​ con índice de capa; factores de capa $`\Xi_{n}`$ varían pero son independientes de $`L`$.

**Resultado.** Las pendientes estimadas $`{\widehat{\alpha}}_{n}`$ **suben** con $`n`$ (corrida típica: $`\approx 1.68,\ 1.80,\ 2.09,\ 2.20`$) y rastrean $`\alpha_{verdadero}(n)`$ dentro de los ICs bootstrap al 95%.\
**Conclusión.** La tubería de pendiente es **sensible** a coherencia monótona; los desplazamientos de ordenada al origen no se hacen pasar por cambios de pendiente.

**10.3 E1b — Control solo de ordenadas al origen (S1 nulo)**

**Configuración.** Misma geometría que E1 pero $`\alpha`$ mantenido **constante** a través de capas; $`\Xi_{n}`$ varía fuertemente con $`n`$.\
**Resultado.** $`{\widehat{\alpha}}_{n}`$ permanece **plano** a través de capas (≈2.10, 2.03, 2.05, 1.99; ICs mutuamente superpuestos) mientras las líneas en log *T –* log *L* se desplazan verticalmente.

**Conclusión.** El método es **específico**: grandes cambios de nivel (ganancias, retardos, "mapeo de reloj") alteran **ordenadas al origen**, no **pendientes**.

**10.4 E2 (condicional) — Direccionalidad con control corriente arriba (S2)**

**Configuración.** Cadena AR acoplada hacia adelante; probamos $`n \rightarrow n + 1`$ **y** la reversa, luego repetimos con pruebas **condicionales** que controlan la serie corriente arriba (para eliminar caminos indirectos).\
**Resultado.**

- **Entropía de Transferencia (condicional)**: hacia adelante p≈**0.002** para todos los pares; reversa no significativa, excepto un pequeño residual en 2↔3 (p≈0.03–0.04) muy por debajo de la señal hacia adelante.

- **Granger (condicional)**: hacia adelante p≈**0.002** para 1→2, 2→3\|X1, 3→4\|X2; reversa no significativa excepto un residual débil en 2↔3.\
  **Conclusión.** Después de condicionar en la capa corriente arriba, la **direccionalidad solo hacia adelante** permanece robusta; los efectos reversos aparentes son atribuibles a **rutas indirectas** (ej., 1→2→3).

**10.5 E3 — Trinquete/histéresis (apoyo)**

**Configuración.** Barra un parámetro de acoplamiento $`g`$ **hacia arriba** y **hacia abajo** con un estado interno lento; estime $`\widehat{\alpha}(g)`$ en cada paso.\
**Resultado.** Las curvas forman un **ciclo de histéresis** con área $`\mathcal{A} \approx - 2.24\ (IC\ 95\%\ \lbrack - 2.54, - 1.87\rbrack`$) el signo indica orientación del ciclo, la magnitud indica memoria.\
**Conclusión.** Evidencia de **memoria direccional** consistente con activación tipo trinquete. Esto fortalece (pero no se requiere para) la afirmación central.

**10.6 E4 — Control de direccionalidad nula**

**Configuración.** Cuatro procesos AR independientes (sin acoplamiento).\
**Resultado.** TE es pequeño y **simétrico**; Granger **no significativo** en ninguna dirección a través de pares.\
**Conclusión.** La tubería **no** inventa direccionalidad, la **especificidad** es alta bajo el nulo.

**10.7 Veredicto conjunto (regla de decisión S1/S2)**

- **S1:** Pasó, $`{\widehat{\alpha}}_{n}`$ es no decreciente en E1 e invariante en el control solo de ordenadas al origen E1b.

- **S2:** Pasó, la direccionalidad hacia adelante es significativa (E2), permanece después de condicionar corriente arriba (E2c), y se desvanece bajo acoplamiento nulo (E4).

- **Apoyo:** La histéresis (E3) proporciona evidencia convergente para memoria direccional.

**Conclusión.** Bajo condiciones sintéticas pero transparentes, el programa de cascada RTM **detecta** acumulación de coherencia y la **desentraña** de efectos de nivel, mientras **confirma** flujo de información solo hacia adelante.

**10.8 Implicaciones y repercusiones**

1.  **Claridad operacional.** La **separación pendiente–ordenada al origen** no es solo conceptual, sobrevive ruido, variabilidad de proxy, y grandes desplazamientos de nivel. Esto protege contra sobre-interpretar "mapeo de reloj" o retardos instrumentales como organización.

2.  **Preparación experimental.** Las mismas métricas (ICs de pendiente, TE/Granger con sustitutos, histéresis opcional) pueden portarse directamente a plataformas análogas (cadenas de resonadores, escaleras de guía de onda/cavidad, escaleras electrónicas), con $`\varepsilon`$ y cuadrículas de retardo/embedding pre-registradas.

3.  **Falsificabilidad.** El programa es **falsificable con dos números**: o (i) las pendientes suben (o se mantienen) **y** (ii) la direccionalidad es solo hacia adelante, o la interpretación de cascada **no está apoyada** en ese sistema.

4.  **Restricciones de modelo.** Resultados positivos de S1/S2 restringen modelos efectivos (ej., transporte con núcleo de memoria, redes de acoplamiento dirigido) que pueden **predecir** cómo responde $`\alpha`$ al acoplamiento/confinamiento, permitiendo diseño dirigido de cascadas.

5.  **Disciplina de alcance.** Los hallazgos permanecen agnósticos sobre interpretaciones metafísicas; las narrativas más amplias (ej., "recodificación" o simulación) son **compatibles** pero **no requeridas**. Las afirmaciones se sostienen sobre las **firmas operacionales** solas.

**10.9 Lista de verificación práctica para replicación**

- Dispersión dentro de capa en $`L:\  \geq 6 - 8`$ tamaños distintos (objetivo ≳1 década).

- Réplicas por tamaño: $`m_{n} \geq 30`$; ruido log-normal tolerado vía ICs bootstrap.

- Pruebas de direccionalidad: $`\geq 10² - 10³`$ muestras efectivas; sustitutos de permutación/fase; FDR a través de pares; variantes **condicionales** para eliminar caminos indirectos.

- Incluya al menos un **segmento nulo** y, si es factible, un **barrido** para sondear histéresis.

**Conclusión integrada.** La suite sintética demuestra **sensibilidad**, **especificidad**, y **valor diagnóstico** de las pruebas de cascada RTM. Con estos controles en su lugar, el artículo avanza de una narrativa motivada a un **programa empírico reproducible** que puede ser confirmado o refutado en sistemas reales.

**APÉNDICE A — Validación Empírica: Escalamiento Espaciotemporal en la Cascada de la Corteza Visual**

> [!NOTE]
> **Clarificación de convención.** En este apéndice, $\alpha$ denota la pendiente de $\log_{10}(\text{Latencia})$ vs. $\log_{10}(\text{Tamaño del Campo Receptivo})$, produciendo $T \propto L^\alpha$ con $\alpha \approx 0.31 \pm 0.02$ (ODR). Esta es la convención estándar de RTM.
> 
> **Distinción importante respecto al "límite difusivo":** La línea de referencia $\alpha = 0.5$ etiquetada como "Límite Difusivo" en las Figuras A.1–A.2 se refiere a una *referencia de integración jerárquica*, no a difusión física de caminata aleatoria. En física de transporte estándar, la difusión de caminata aleatoria produce $T \propto L^2$ ($\alpha = 2$ en notación RTM), y la propagación balística produce $T \propto L$ ($\alpha = 1$).
> 
> El hallazgo empírico $\alpha \approx 0.31$ por tanto indica que la jerarquía de la corteza visual logra integración de información *más rápido que el transporte balístico* en el sentido escala-tiempo. Esta eficiencia "super-balística" surge del procesamiento masivamente paralelo en cada nivel jerárquico, donde muchas neuronas contribuyen simultáneamente a campos receptivos más grandes sin acumulación proporcional de latencia.
> 
> Para evitar confusión con la terminología del Doc 001 Sec. 2.2: la corteza visual opera en un régimen de $\alpha < 1$ (integración más rápida que balística), que no tiene análogo directo en las clases de transporte físico (balístico/difusivo/subdifusivo) definidas para dinámica de partícula única. Este régimen es característico de arquitecturas jerárquicas paralelas y representa una clase de universalidad distinta única de sistemas de procesamiento distribuido.

El marco RTM dicta que la eficiencia de una red de procesamiento de información puede mapearse vía su escalamiento topológico espacial-temporal ($`\alpha`$). Evaluamos esto dentro de las 21 áreas jerárquicas de la corteza visual de primates.

**A.1 Observación Heurística y Sesgos Estadísticos**

La validación inicial utilizó regresión de Mínimos Cuadrados Ordinarios (OLS) en 21 puntos de datos altamente agregados, produciendo un exponente aparente de $`\alpha \approx 0.30`$. Mientras que esta observación heurística apoyó la predicción RTM Super-Difusiva, contenía dos vulnerabilidades estadísticas fatales:

1.  **Sesgo de Atenuación:** OLS matemáticamente asume que los campos receptivos espaciales se miden sin error. En realidad, las mediciones de fMRI y electrodos llevan barras de error masivas. Ignorar este ruido bidireccional aplana artificialmente las pendientes de regresión.

2.  **Sesgo de Agregación:** Comprimir miles de mediciones neuronales individuales en solo 21 coordenadas agregadas eliminó artificialmente la varianza biológica natural, inflando el coeficiente de determinación a un irreal $`R^{2} \approx 0.92`$.

**A.2 Validación EIV Rigurosa (ODR y Varianza a Nivel de Sujeto)**

Para probar que el régimen Super-Difusivo es un mecanismo biológico genuino y no un artefacto estadístico, desplegamos una tubería de validación "Equipo Rojo":

- **Regresión de Distancia Ortogonal (ODR):** Se utilizó un modelo de Errores en Variables para absorber explícitamente la varianza de medición bidireccional de tanto latencia como tamaños de campo receptivo.

- **Reconstrucción Poblacional:** Desagregamos la jerarquía, simulando la varianza neuronal cruda a nivel de sujeto para probar los límites de la teoría de transporte RTM bajo ruido biológico realista.

**A.3 El Cerebro Super-Difusivo (Hallazgos Robustos)**

Incluso cuando se penaliza fuertemente con ruido observacional extremo y jerarquía desagregada, la ley de escalamiento física permanece estrictamente intacta:

- **Exponente Topológico Robusto:** La pendiente de escalamiento corregida por varianza ODR se fija en $`\mathbf{\alpha}\mathbf{= \ 0.311\ }\mathbf{\pm}\mathbf{0.021}`$ (con la simulación cruda a nivel poblacional confirmando un $`\alpha = \ 0.281`$ subyacente).

- **Coherencia Biológica Realista:** La varianza natural reconstruida produce un realista $`R^{2} = 0.677`$, probando que la correlación permanece como un impulsor físico dominante de la arquitectura cortical sin caer en falacias de sobreajuste.

> [!NOTE]
> **Nota sobre Simetría Recíproca:** El exponente de transporte medido ($\alpha_t \approx 0.31$) representa la velocidad operacional de información a través de la jerarquía. Este es el recíproco matemático del exponente de coherencia estructural ($\alpha_s \approx 3.2$) definido en el marco RTM fundacional (Ver Doc 001). Esta simetría ($\alpha_t \approx 1/\alpha_s$) prueba que la arquitectura de alta viscosidad del cerebro es precisamente lo que permite su eficiencia de transporte super-difusiva. La estructura confina información para integrarla, permitiendo que la señal evada los límites térmicos estándar.

**Conclusión:** El marco RTM aísla exitosamente la física macroscópica del cerebro. La corteza visual opera estrictamente en una **Clase de Transporte Super-Difusivo (régimen Super-balístico)** ($`\alpha \ll 0.5`$). El cerebro aprovecha su topología jerárquica masiva y paralela para evadir activamente los límites de latencia física de la difusión térmica estándar, logrando integración sensorial óptima.

© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).
