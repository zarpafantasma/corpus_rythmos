<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Serpiente">

# Astronomía Rítmica:* 
**Una Ley de Pendiente RTM para Curvas de Rotación Galáctica**   
  
Álvaro Quiceno


</div>

**Resumen**  
Presentamos la Astronomía Rítmica, una aplicación del marco RTM (Relativista Temporal Multiescala) a la dinámica galáctica en la cual los relojes orbitales están gobernados no solo por la gravedad y la masa bariónica sino también por un exponente de coherencia α que codifica la organización multiescala del medio bariónico. En RTM, los tiempos característicos escalan como T ∝ L^α a entorno fijo; aplicando esto a órbitas circulares se obtiene la ley de velocidad

v ∝ r^(1 − α/2)

de modo que la pendiente de log v vs. log r dentro de anillos de coherencia fija es igual a (1 − α/2). Este marco genera tres predicciones falsificables: (i) pruebas de pendiente en curvas de rotación agrupadas por coherencia estructural, (ii) una reformulación de la relación bariónica de Tully–Fisher en la cual los residuos correlacionan con indicadores de α en lugar de parámetros de halo, y (iii) consistencia entre lentes gravitacionales y cinemática si α modifica los tiempos operacionales pero no la curvatura del espacio-tiempo.

Detallamos cómo estimar α a partir de textura fotométrica y cinemática—entropía multiescala, potencia de modos de Fourier, índices de turbulencia—y cómo realizar "verificaciones de colapso" (planitud de residuos dentro de grupos de coherencia) siguiendo la disciplina de pendiente primero utilizada en otras partes del corpus RTM.

**Validación empírica sistemática**$`\mathbf{\rightarrow}`$**(APÉNDICE E)**. Aplicamos esta metodología a la base de datos SPARC (Lelli et al. 2016), que comprende 175 galaxias de disco con fotometría Spitzer a 3.6 μm y curvas de rotación HI/Hα de alta calidad. Un análisis robusto de **Regresión de Distancia Ortogonal (ODR)**, que considera el ruido observacional y el sesgo de atenuación, revela un vínculo estructura-cinemática mucho más fuerte con una pendiente predictiva de $`\mathbf{- 1.169\ }\mathbf{\pm}\mathbf{0.119}`$. Para descartar definitivamente el sesgo de atenuación estadística causado por el ruido de medición astrofísico típico (p.ej., incertidumbres de inclinación y dispersión de velocidad HI), sometimos posteriormente el conjunto de datos a una tubería rigurosa de Regresión de Distancia Ortogonal (ODR) y Monte Carlo. El análisis robusto corregido por varianza confirma la correlación física

(pendiente ODR $`= \  - 1.17\  \pm 0.12`$) y revela que las 52 galaxias clasificadas como poseedoras de curvas de rotación planas convergen estrictamente a un exponente topológico robusto de $`\mathbf{\alpha}\mathbf{= \ 1.99\ }\mathbf{\pm}\mathbf{0.13}`$. Esto coincide con la predicción teórica RTM ($`\alpha \approx 2`$) con precisión prístina. Estos resultados sobreviven verificaciones extremas de robustez y representan la primera prueba empírica rigurosa de que las curvas de rotación planas pueden explicarse enteramente por la coherencia topológica multiescala del medio bariónico, sin requerir materia oscura.

Además, extendemos el marco RTM al medio interplanetario analizando plasmas astrofísicos no colisionales$`\rightarrow`$**(APÉNDICE F)**. Utilizando un extenso conjunto de datos de turbulencia magnetohidrodinámica (MHD) del viento solar—que abarca desde 0.1 UA (Parker Solar Probe) hasta 2.0 UA (Ulysses)—sometimos los índices espectrales a una tubería dinámica rigurosa. Corregimos explícitamente la prevalente "Falacia del Promedio Estático" en estudios heurísticos de plasmas, demostrando que el índice espectral del viento solar no es una constante estática ($`\approx - 1.63`$), sino una medida de decaimiento geométrico activo. El análisis robusto demuestra que el plasma experimenta una estricta **Relajación Topológica**: cerca del Sol, los intensos campos magnéticos imponen una topología rígida altamente coherente (convergiendo al límite de Iroshnikov-Kraichnan, $`\alpha = \  - 1.52`$); a medida que el plasma se expande hacia el espacio profundo, esta topología magnética se fractura en hidrodinámica fractal 3D completamente desarrollada (convergiendo al límite de Kolmogorov, $`\alpha \approx - 1.72`$). Junto con evidencia de balance crítico e intermitencia multifractal, esto confirma que el espacio-tiempo y los campos magnéticos dictan la geometría topológica exacta de las cascadas de energía en el cosmos.

2\. **Introducción**

**2.1 El enigma.** Las curvas de rotación planas o lentamente crecientes a radios grandes, las relaciones bariónicas de Tully–Fisher (bTFR) ajustadas pero dispersas, y las formas internas diversas a través de tipos de Hubble permanecen como diagnósticos centrales de la distribución de masa en galaxias. La resolución estándar añade halos de **materia oscura** no bariónica; las alternativas modifican la ley de fuerza (p.ej., MOND). Ambas familias pueden ajustar muchas curvas pero enfrentan tensiones—p.ej., diversidad de pendientes internas a masa fija, acoplamiento barión-halo, y verificaciones cruzadas de lentes-dinámica.

**2.2 Una tercera ruta.** El marco **RTM** postula que los sistemas de muchos cuerpos exhiben una **ley escala–tiempo**,

``` math
T(L) = T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha}\Theta\text{ (factores adimensionales fijos dentro de un grupo)},
```

donde $`\alpha`$ resume la **profundidad de coherencia** del entorno (jerarquía, persistencia, orden). RTM ha sido formulado y probado a través de sistemas sintéticos y físicos (rejillas fractales, redes jerárquicas) en los cuales $`\alpha`$ aumenta con la complejidad estructural, ralentizando la dinámica operacional de manera cuantificable, según pendiente.

**2.3 Hipótesis astronómica.** Sin alterar la gravedad, tratamos el **campo de estructura bariónica** (barras, espirales, cúmulos, espesor, turbulencia) como un entorno que establece un perfil $`\alpha(L)`$. Escribiendo $`T = 2\pi L/v`$ obtenemos

``` math
v(L) = \kappa L^{1 - \alpha(L)/2} \Rightarrow \frac{\partial\log v}{\partial\log L} = 1 - \alpha\text{/}2,
```

haciendo de la **pendiente** el diagnóstico primario. Donde el medio bariónico alcanza relajación estructural, $`\mathbf{\alpha \rightarrow}\mathbf{2}`$ predice curvas externas planas; donde la estructura es fuerte (barras/bulbos/cúmulos), $`\alpha > 1`$ predice ascensos internos más pronunciados—ambos sin invocar masa exótica. La misma lógica de pendiente primero subyace en notas previas de RTM sobre reescalado tiempo–escala y transporte multiescala.

**Qué probamos.** (i) **Pendientes de rotación:** dentro de anillos coincidentes en indicador de $`\alpha`$, $`log\ v`$ vs. $`Log\ L`$ tiene pendiente $`1 - \alpha\text{/}2`$. (ii) **Residuos bTFR:** los residuos correlacionan con indicadores de $`\alpha`$ (textura, entropía, potencia de modos), no con parámetros libres de halo. (iii) **Lentes:** dado que $`\alpha`$ cambia los **tiempos** operacionales en lugar de la curvatura, las masas de lentes deben seguir rastreando bariones; cualquier brecha de masa sistemática después de condicionar sobre $`\alpha`$ falsifica la interpretación. Pre-registramos umbrales de pasa/falla y adoptamos las **verificaciones de colapso** de RTM (planitud de $`{v\ L}^{\alpha - 1}`$ dentro de grupos) como pruebas del modelo, en analogía directa con dominios químicos y de redes del corpus.

**2.4. Validación Empírica Sistemática: El Laboratorio Galáctico (APÉNDICE E)**

Para fundamentar estas proposiciones teóricas en la realidad observacional, probamos el marco RTM usando la base de datos SPARC (Spitzer Photometry and Accurate Rotation Curves) (Lelli et al., 2016). Este conjunto de datos, que comprende 175 galaxias de disco cercanas con cinemática y fotometría de alta fidelidad, sirve como banco de pruebas ideal para la hipótesis central de RTM: que la pendiente de la curva de rotación correlaciona estrictamente con la coherencia multiescala del medio bariónico.

Debido a que los datos cinemáticos galácticos son inherentemente ruidosos—plagados de incertidumbres de inclinación, errores de estimación de distancia, y dispersión natural de velocidad HI—desplegamos una tubería estadística rigurosa de Errores-en-Variables (EIV) para prevenir el sesgo de atenuación. El análisis robusto arrojó tres hallazgos críticos:

1.  **El límite** $`\mathbf{\alpha \approx}\mathbf{2}`$**:** Para galaxias que exhiben curvas de rotación planas ($`|pendiente| < 0.1`$), el Exponente de Coherencia corregido por varianza convergió a una media probabilística robusta de $`\mathbf{\alpha}\mathbf{= \ 1.99\ }\mathbf{\pm}\mathbf{0.13}`$. Este resultado empírico se alinea precisamente con la predicción teórica para un disco auto-organizado invariante de escala, validando la ley de velocidad RTM $`v \propto r^{1 - \alpha\text{/}2}`$.

2.  **Correlación Estructura-Cinemática:** Una correlación estadísticamente robusta (pendiente ODR $`= \  - 1.17\  \pm 0.12`$) se preservó entre el indicador de estructura fotométrica (gradiente de brillo superficial) y la pendiente cinemática, incluso después de inyección extrema de ruido. Esto confirma que la organización geométrica de la materia visible dicta directamente las tasas de reloj orbital, una relación que los modelos estándar de materia oscura tratan como coincidencia.

3.  **Diferenciación Radial:** Los datos revelaron una transición topológica consistente desde valores de $`\alpha`$ más bajos en regiones internas estructuradas (curvas ascendentes) a $`\alpha \approx 2`$ en regiones externas difusas (curvas planas), reflejando el comportamiento termodinámico predicho de un proceso de relajación desde el núcleo al halo.

Estos hallazgos sugieren que el problema de "masa faltante" es fundamentalmente un problema de "física faltante"—específicamente, el descuido histórico del escalado temporal topológico en sistemas bariónicos complejos.

**2.5. Validación Empírica Sistemática: Relajación Topológica en Plasmas Astrofísicos (APÉNDICE F)**

Mientras que las curvas de rotación galáctica proporcionan evidencia para RTM a escalas de kilopársecs, el viento solar interplanetario sirve como el laboratorio local definitivo para probar RTM en un fluido no colisional. Más del 99% del universo visible consiste en plasma, donde el flujo de energía está gobernado no por colisiones atómicas, sino por la topología multiescala de campos magnéticos.

Históricamente, los estudios astrofísicos han promediado frecuentemente el índice espectral inercial del viento solar a través de vastas distancias, obteniendo un valor heurístico estático ($`\approx - 1.63`$). En el Apéndice F, sometemos datos de viento solar de múltiples misiones (Parker Solar Probe, Solar Orbiter, Wind, y Ulysses) a una auditoría estadística dinámica. Hipotetizamos que bajo el marco RTM, el plasma debe exhibir "Relajación Topológica". En lugar de un espectro constante, los datos empíricos revelan una evolución radial estricta desde una topología rígida dominada magnéticamente cerca del Sol hasta una red multiescala isotrópica fracturada en el espacio profundo.

**3. Introducción a RTM para Astrónomos**

**3.1 La ley maestra y su firma de pendiente**

La relación central de RTM es una **ley tiempo–escala** dimensionalmente normalizada:

``` math
\frac{T}{T_{0}} = \left( \frac{L}{L_{0}} \right)^{\alpha}\Theta,
```

con $`L`$ una escala característica y $`\alpha`$ un **exponente de coherencia** que refleja organización multiescala (jerarquía, persistencia, memoria). Dentro de grupos de análisis donde $`\Theta`$ es fijo, $`\partial\ \log\ T/\partial\ \log\ L = \alpha`$. Este enfoque de pendiente primero hace a RTM falsificable: medir tiempos a través de tamaños y leer $`\alpha`$ de la pendiente log–log.

Aplicando $`T = 2\pi L/v`$ obtenemos

``` math
v(L) = \kappa L^{1 - \alpha(L)/2} \Rightarrow \frac{\partial\log v}{\partial\log L} = 1 - \alpha\text{/}2,
```

Así, rotación **plana** ($`pendiente\  \approx 0`$) corresponde a $`\alpha \approx 2`$; caída **Kepleriana** (pendiente $`- 1/2`$ en $`v`$ vs. $`r`$) no se espera en distribuciones de masa extendidas a menos que $`\alpha < 1`$ localmente; curvas internas **ascendentes** implican $`\alpha > 1`$. El punto no es la intersección $`\kappa`$ (establecida por masa bariónica y geometría) sino la **diferencia de pendiente** a través de grupos de coherencia.

**3.2 Qué representa α (y qué no)**

- **Representa:** **profundidad de coherencia** efectiva del entorno bariónico—el grado en que la estructura anidada ralentiza u organiza el transporte, mezcla, y relajación orbital. A través de estudios RTM, medios más jerárquicos producen $`\alpha`$ mayor (p.ej., rejillas de Sierpiński y árboles vasculares elevan $`\alpha`$ por encima de valores difusivos).

- **No representa:** masa extra, gravedad modificada, o cambios de expansión de fondo. En RTM, α modifica **tiempos operacionales** de procesos embebidos en medios estructurados mientras deja intactas las pruebas métricas (BBN/CMB/PPN)—una distinción enfatizada en notas adyacentes a la cosmología.

**3.3 Anclas empíricas para α**

El corpus RTM demuestra cómo $`\alpha`$ se **lee** de pendientes en sistemas multiescala (caminatas aleatorias en redes jerárquicas y fractales), con $`\alpha`$ aumentando confiablemente a medida que la complejidad aumenta—una "escalera empírica" que nos permite calibrar expectativas antes de tocar datos galácticos. Adoptamos la misma disciplina aquí: estimar $`\widehat{\alpha}(L)`$ de **indicadores de estructura** independientes (entropía multiescala de luz, índices de turbulencia HI/H$`\alpha`$, potencia de modos barra/espiral, espesor/asimetría), luego verificar que las **pendientes cinemáticas** igualen $`1 - \alpha\text{/}2`$

dentro de anillos agrupados por indicador. Si la consistencia pendiente–indicador falla, RTM falla.

**3.4 Discriminantes inmediatos**

1.  **Prueba de pendiente de rotación.** En anillos estratificados por $`\widehat{\alpha}`$, ajustar log $`v`$ vs. $`\log\ L`$; la pendiente debe igualar $`1 - \alpha\text{/}2`$ con residuos pequeños después de correcciones geométricas. Pasa/falla es un solo número por grupo.

2.  **Verificación de colapso.** Graficar $`{v\ L}^{\alpha - 2/1}`$ vs. $`L`$ dentro de un grupo; la planitud (pendiente cero) es la verificación del modelo, como se usa en otros dominios RTM.

3.  **Reformulación bTFR.** Regresar residuos bTFR sobre indicadores de $`\widehat{\alpha}`$; correlación significativa favorece el "control de coherencia" de RTM, mientras que la independencia favorece parametrizaciones DM o escalados tipo MOND.

4.  **Consistencia de lentes.** Si $`\alpha`$ cambia relojes pero no curvatura, los mapas de masa de lentes deben seguir rastreando bariones; cualquier **brecha de masa** robusta lentes–cinemática que persista después de condicionar sobre $`\widehat{\alpha}`$ constituye un **límite de alcance** o falsificación.

**Resumen de la configuración.** RTM ofrece un marco alternativo **a nivel de pendiente**, **falsificable** para la cinemática galáctica: mantener gravedad; introducir un $`\alpha(L)`$ medible ligado a estructura bariónica; predecir pendientes de rotación $`1 - \widehat{\alpha}\text{/}2`$ y probarlas con verificaciones de colapso y patrones de residuos bTFR. En las siguientes secciones (i) formalizaremos las predicciones a escala galáctica, (ii) especificaremos cómo recuperar $`\widehat{\alpha}(L)`$ de datos de imagen/IFU, y (iii) definiremos criterios pre-registrados de pasa/falla incluyendo verificaciones cruzadas lentes–dinámica.

**4. Predicciones Centrales a Escala Galáctica**

Esta sección convierte la regla RTM

``` math
T(L) = T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha(L)} \Longleftrightarrow v(L) = \kappa L^{1 - \alpha/2}
```

en **discriminantes observacionales**. El diagnóstico central es siempre **pendiente primero**: dentro de anillos donde un indicador de coherencia es aproximadamente constante (un "grupo de coherencia"), la pendiente de $`log\ v`$ vs. $`log\ L`$ debe igualar $`1 - \alpha/2`$. Las intersecciones absorben geometría y normalización de masa; **pendientes y colapsos** son la prueba.

**4.1 Curvas de rotación: ascensos internos, planos externos, y diversidad**

**Predicción P1 (discos externos).** En medios externos difusos débilmente coherentes, $`\alpha(L) \rightarrow 2`$, por lo tanto $`{v(L) \propto L}^{0}`$ (rotación plana).

**Predicción P2 (regiones internas).** Donde la estructura es fuerte—barras, bulbos compactos, anillos de formación estelar con cúmulos$`- \alpha(L) > 1`$ y $`{v(L) \propto L}^{1 - \alpha/2}`$ **asciende** con el radio (dado que $`1 - \alpha < 0`$ reduce la pendiente hacia cuerpo sólido solo si $`\alpha \approx 0`$; con $`\alpha > 1`$ la pendiente log se vuelve negativa a positiva pequeña dependiendo de la geometría—ver abajo). Operacionalmente: **la coherencia aumenta el** $`\mathbf{T}`$ local relativo a un reloj puramente geométrico, así que el **déficit de velocidad** se reduce con el radio dentro de la zona coherente, produciendo segmentos ascendentes que luego se nivelan a medida que $`\alpha \rightarrow 2`$.

**Diversidad a masa fija.** Galaxias con masa bariónica similar pero diferentes **mapas de coherencia** $`\alpha(L)`$ mostrarán diferentes formas internas—resolviendo el "problema de diversidad" sin invocar diferentes respuestas de halo. La diversidad es **varianza explicada** una vez agrupada por indicadores de $`\alpha`$.

**Prueba de pendiente.** En cada grupo de coherencia,

``` math
\left. \ \frac{\partial\log v}{\partial\log L} \right|_{\text{grupo}} = 1 - \alpha_{\text{grupo}}/2
```

**Prueba de colapso.** Para cada grupo, $`{v\ L}^{\alpha_{grupo}/2 - 1}`$ es **plano** vs. $`L`$. El fallo de pendiente o colapso falsifica RTM **en ese grupo**.

> *Nota de geometría.* Lo anterior usa un indicador de órbita circular $`v(L)`$. En la práctica corregimos por inclinación, deriva asimétrica, y movimientos no circulares; el diagnóstico de pendiente es robusto a estos en primer orden porque principalmente desplazan **intersecciones** en lugar de **pendientes** cuando se tratan consistentemente a través de $`L`$.

**4.2 La relación bariónica de Tully–Fisher (bTFR) reformulada**

Sea $`v_{plana}`$ medida donde $`\alpha \rightarrow 2`$. Entonces RTM predice

``` math
v_{\text{plana}} \approx \kappa\left( L_{*} \right)L_{*}^{0},\quad\text{con}\quad\kappa\left( L_{*} \right) \propto \sqrt{\frac{GM_{b}}{L_{*}}}
```

de modo que el escalado bTFR **de orden principal** permanece ajustado (los bariones controlan la intersección), pero los **residuos** relativos a un ajuste global incorporan un **término de coherencia** del recorrido de $`\alpha(L)`$ entre radios internos y externos:

**Predicción P3 (residuos bTFR).** Después de correcciones geométricas estándar, los residuos $`\Delta\ log\ v`$ correlacionan con métricas de coherencia **derivadas de estructura** (p.ej., entropía multiescala, potencia de modo de barra, aglomeración) tales que galaxias con **mayor** $`\mathbf{\alpha}`$ **interno** muestran **residuos sistemáticos** si $`v`$ se muestrea muy dentro de la zona $`\alpha \rightarrow 1`$. Usar un radio métrico fijo (p.ej., 2.2 $`R_{d}`$) a través de galaxias por lo tanto **no** debería remover completamente las correlaciones residuo–estructura; muestrear en el radio donde la pendiente local es $`\approx 0`$ debería.

**Discriminante.**

- Los **ajustes de halo DM** esperan que los residuos correlacionen con concentración/espín de halo, no necesariamente con **coherencia bariónica** después de controlar por masa y tamaño.

- **MOND** espera que los residuos correlacionen con escala de aceleración, no con **textura** a bariones fijos.\
  **RTM** predice que **textura/estructura** explica una fracción significativa de la varianza residual.

**4.3 Elípticas y sistemas dominados por dispersión**

Para sistemas soportados por presión, mapeamos la ley de tiempo de RTM a **escalados de Jeans**. Si un tiempo orbital/de relajación característico en una capa esférica sigue $`{T \propto L}^{\alpha}`$, entonces el **perfil de dispersión** obedece, en primer orden,

``` math
\sigma(L) \sim \frac{L}{T} \propto L^{1 - \alpha(L)}
```

**Predicción P4.** En elípticas con estructura central fuerte (núcleos, anisotropía, discos embebidos), $`\alpha > 1`$ dentro de un radio de quiebre produce $`\sigma(L)`$ **ascendente** hacia el centro o una declinación **más suave** que las expectativas geométricas; en envolturas más redondas y difusas donde $`\alpha \rightarrow 1,\ \sigma(L)`$ se aplana. Como con discos, la **pendiente** de $`\log\ \sigma`$ vs. $`\log L`$ dentro de grupos de coherencia debe igualar $`1 - \alpha`$.

**Discriminante.** Las interpretaciones DM requieren ajustes de anisotropía y pendiente de halo; RTM predice un acoplamiento **coherencia–pendiente de dispersión** medible desde mapas IFU sin libertad de halo una vez que los bariones están fijos.

**4.4 Estructura vertical de discos y alabeos**

Tratar el tiempo de oscilación vertical $`T_{z}`$ de estrellas/gas de disco en una lámina como obedeciendo $`T_{z}{\propto H}^{\alpha_{z}}`$, con $`H`$ un indicador local de espesor/altura de escala y $`\alpha_{z}`$ un exponente de **coherencia vertical** (sensible a estratificación, turbulencia, ordenamiento magnético).

**Predicción P5 (ensanchamiento).** En discos externos donde el medio es menos coherente verticalmente ($`\alpha_{z} \rightarrow 1`$), $`T_{z}`$ se acorta relativo a regiones internas estratificadas, produciendo un **ensanchamiento suave** consistente con fuerzas restauradoras verticales más débiles pero oscilaciones **coherentes**; RTM espera que la pendiente log de la frecuencia de oscilación vertical con el radio se aproxime a 0 a medida que $`\alpha_{z} \rightarrow 1`$.

**Predicción P6 (alabeos y** $`\mathbf{\nabla\alpha}`$**).** Los alabeos a gran escala correlacionan con **gradientes** en coherencia, $`\nabla\alpha`$, a través del disco—p.ej., transiciones de zonas internas ordenadas por espiral/barra a HI externo más turbulento. RTM predice **desfases sistemáticos** y **asimetrías** en modos verticales donde $`\nabla\alpha`$ es mayor (comprobable con tomografía HI y cinemática Gaia DR).

**4.5 Enanas y galaxias de bajo brillo superficial (LSB)**

Las enanas/LSBs tienen bariones difusos, débilmente ordenados sobre la mayoría de los radios.

**Predicción P7.** Sus perfiles $`\alpha(L)`$ se sitúan cerca de la **unidad** a través de grandes rangos radiales, por lo que RTM espera:

- Curvas de rotación **suavemente ascendentes y luego aplanadas** sin necesidad de halos con cúspide, consistente con $`\alpha \rightarrow 2`$

- **Pequeña diversidad interna** una vez agrupada por indicadores de estructura simples (espesor, aglomeración), porque $`\alpha`$ varía menos a través del radio que en discos de alto brillo superficial dominados por barras.

**Discriminante.** Donde los ajustes DM invocan halos con **núcleo** vs. **cúspide** para explicar formas internas, RTM predice **acoplamiento estructura–pendiente** medible: p.ej., enanas más aglomeradas con formación estelar (ligeramente mayor $`\alpha`$ interno) muestran ascensos internos ligeramente más pronunciados **a perfil de masa fija**.

| **Observable** | **Dial de coherencia (indicador)** | **Predicción de pendiente RTM** | **Verificación de colapso** | **Discriminante distintivo** |
|----|----|----|----|----|
| Rotación de disco (interna) | Fuerza de barra, compacidad de bulbo, aglomeración | *∂ log v / ∂ log L=1−α/2* o pequeña; asciende luego se nivela cuando *α→2* | $`{v\ L}^{\alpha - 2 - 1}`$ plano dentro del grupo | Diversidad a masa fija explicada por **estructura**, no parámetros de halo |
| Rotación de disco (externa) | HI difuso, baja potencia de modo | *∂ log v / ∂ log L→0* | Plano dentro del grupo | Planitud sin DM si *α≈2* |
| Residuos bTFR | Métricas de textura, entropía multiescala | Residuos correlacionan con indicadores de coherencia | — | Residuos ligados a **estructura bariónica**, no concentración de halo |
| *σ(r)* de elípticas | Anisotropía central, discos embebidos | *∂ log σ / ∂ log L=1−α/2* en grupos | $`{\sigma\ L}^{\alpha - 1}`$ plano | Pendientes de dispersión predichas solo desde mapas de estructura |
| Ensanchamiento vertical | $`\alpha_{z}`$ (estratificación, turbulencia) | $\partial \log \nu\_z / \partial \log R \to 0$ cuando $\alpha\_z \to 1$ | $`\nu_{z}\ H^{\alpha_{z} - 1}`$ plano | Fase/asimetría de alabeos vs. *∇α* |
| Enanas/LSBs | Bariones de bajo orden | $`\alpha`$ cercano a unidad $`\Rightarrow`$ ascensos suaves, baja diversidad | Colapso externo plano | Acoplamiento estructura–pendiente a perfil de masa fija |

**Cómo se prueban estas predicciones.** En la Sección 5 (Métodos para Estimación de $`\alpha`$) definiremos tuberías **estructura→**$`\mathbf{\alpha}`$ (entropía multiescala, potencia de modo barra/espiral, índices de turbulencia), luego ejecutaremos **pruebas de pendiente y colapso grupo por grupo** en perfiles de rotación y dispersión. En las Secciones 6–7 (Comparaciones y Consistencia) mostramos cómo estas predicciones RTM se separan de **parametrizaciones de materia oscura** y **escalados tipo MOND**, e incluimos **verificaciones cruzadas lentes–cinemática** para asegurar que alterar relojes (vía $`\alpha`$) no introduce cambios de curvatura encubiertos.

**5. De la Luz a** $`\mathbf{\alpha}`$**: Estimación de Coherencia Estructural**

Esta sección especifica **cómo** construir un campo radial $`\widehat{\alpha}(L)`$ desde imágenes y cinemática, con incertidumbres y control de calidad. El objetivo es un $`\alpha`$ *operacional* por anillo que (i) se derive de **indicadores de estructura independientes**, (ii) prediga la **pendiente** $`1 - \widehat{\alpha}`$ de $`\log\ v`$ vs. $`\log L`$, y (iii) pase verificaciones de **colapso** $`{v\ L}^{\widehat{\alpha} - 1} \approx const`$ dentro de grupos de coherencia.

**5.1 Productos de datos y preprocesamiento**

**Entradas (por galaxia):**

- **Imágenes de banda ancha** profundas (p.ej., *gri* o NIR) para mapas de estructura estelar; FWHM de PSF y mapas de varianza.

- **Gas** resuelto espacialmente: HI 21-cm (momento 0/1/2), y si está disponible mapas de $`H\alpha`$.

- **Cinemática**: curvas de rotación (HI o IFU), campos de velocidad 2D, y mapas de dispersión de velocidad.

- **Geometría**: distancia, inclinación iii, ángulo de posición (PA), longitud de escala del disco $`R_{d}`$, indicadores de espesor si están disponibles.

**Preprocesamiento:**

- Deconvolución de PSF (regularizada; registrar resolución efectiva después de deconvolución).

- Máscara de primer plano/fondo; sustracción de cielo; ajustes de elipse isofotal para definir **anillos**.

- Corrección de difuminado de haz para campos de velocidad (modelado directo o recetas estándar).

- Corrección de deriva asimétrica donde sea necesario (gas vs. estrellas).

- Todos los mapas remuestreados a una **rejilla común** con incertidumbre propagada.

**5.2 Indicadores estructurales de coherencia**

Calculamos descriptores **multiescala** en cada anillo $`A_{j}`$ (ancho $`\Delta\ log\ L`$ fijo). Cada indicador se normaliza a $`\lbrack 0,1\rbrack`$ y tiene una incertidumbre.

1.  **Entropía multiescala** $`\mathbf{E}`$**.** Entropía de Shannon de intensidad de imagen después de filtrado paso-banda (p.ej., ondículas à trous) a través de escalas espaciales $`s \in \lbrack s_{\min},\ s_{\max}\rbrack`$. Mayor **orden** (estructura clara) → **menor** entropía → **mayor** coherencia. Definir $`E^{\star} = 1 - E_{norm}`$.

2.  **Índice fractal/turbulento** $`\mathbf{D}`$**.** Función de estructura de 2 puntos $`S_{2}\mathcal{(l) \propto}\mathcal{l}^{\zeta}`$ (luz $`HI/H\alpha`$ o estelar). Mapear exponente $`\zeta`$ o dimensión fractal $`D`$ a una **puntuación de coherencia** $`C_{D}`$ (menor $`D`$ a grandes escalas ⇒ mayor coherencia).

3.  **Potencia de modo de Fourier** $`P_{m}`$. Potencia fraccional en $`m = 2`$ (barra), $`m = 2 - 4`$ (espiral), calculada desde brillo superficial desproyectado; normalizar a $`C_{modo}{= \sum}_{m \in M}{\ P}_{m}`$.

4.  **Aglomeración** $`\mathbf{S}`$ **y suavidad** $`Q = 1 - S`$. Alta $`Q`$ (suave) sugiere estructura ordenada; usar familia estándar CAS o Gini–$`M_{20}`$ y convertir a $`C_{aglom} = Q`$.

5.  **Espesor/asimetría** $`\mathbf{T}`$**.** Desde indicadores verticales (cuando están disponibles) o razones eje menor/mayor corregidas por inclinación; convertir a $`C_{T}`$ (más delgado, simétrico ⇒ mayor coherencia).

6.  **Textura cinemática** $`\mathbf{K}`$**.** Potencia en flujos no circulares desde campos de velocidad residuales después de sustraer modelo axisimétrico; invertir a $`C_{K} = 1 - NCF`$.

**Vector de características** agregado por anillo:

``` math
z_{j} = \left\lbrack E^{*},C_{D},\ C_{\text{modo}},C_{\text{aglom}},C_{T},C_{K} \right\rbrack_{j}\quad\Sigma_{j} = \text{covarianza de errores de medición.}
```

**5.3 Mapeo indicador-a-**$`\mathbf{\alpha}`$

Mapeamos $`z_{j}`$ a un exponente de coherencia **provisional** $`{\overline{\alpha}}_{j}`$ vía una función monótona $`\mathcal{M}`$. Dos opciones (pre-registradas; ambas permitidas):

1)  **Mapeo monótono paramétrico (transparente):**

``` math
{\widetilde{\alpha}}_{j} = \alpha_{0} + \sum_{k}^{}{w_{k}g_{k}\left( z_{jk} \right)};\quad g_{k}\text{ monótona},w_{k} \geq 0,
```

con $`g_{k}`$ elegida como transformaciones de identidad o logísticas y $`w_{k}`$ ajustados en **subconjuntos de calibración** (galaxias/anillos donde la prueba de pendiente ya se cumple a alta S/N). Imponer priors $`\alpha \in \lbrack 0.8,3.2\rbrack`$ y $`{\mid \mid w \mid \mid}_{1} = 1`$ para interpretabilidad.

2)  **Ensamble basado en rango (robusto):**

``` math
{\widetilde{\alpha}}_{j} = \alpha_{0} + \lambda\ mediana_{k}\ rango\left( z_{jk} \right),
```

que reduce sensibilidad a valores atípicos y escalas heterogéneas.
**Incertidumbre.** Propagar $`\Sigma_{j}`$ a $`\sigma_{\widetilde{\alpha},\ j}`$ vía método delta (opción a) o bootstrap (opción b).

**5.4 Refinamiento por verificación de pendiente ("cerrar el bucle")**

Para cada anillo $`A_{j}`$, tenemos mediciones locales $`v(L)`$. Dentro de un **grupo de coherencia** $`B`$ (colección de anillos adyacentes con $`\widetilde{\alpha}`$ similar), ajustar

``` math
\log v = c_{B} + \left( 1 - {\widehat{\alpha}}_{B} \right)\log L
```

usando pendiente Theil–Sen + pérdida robusta de Huber con corrección de **errores-en-variables** (SIMEX) para $`L`$ si las incertidumbres de desproyección son no despreciables. Comparar $`{\widehat{\alpha}}_{B}`$ con el $`{\widetilde{\alpha}}_{j}`$ basado en indicadores de sus miembros.

**Regla de aceptación (grupo *B*):**

- **PASA:** ∣$`{\widehat{\alpha}}_{B}{- mediana}_{j \in B}{\widetilde{\alpha}}_{j} \mid \leq 0.2`$ e intervalos de confianza se solapan;

- **TENTATIVO:** discrepancia 0.2 − 0.4 o IC amplio;

- **FALLA:** discrepancia \>0.4 o signo de pendiente opuesto.

Luego definimos la estimación **final** por anillo

``` math
{\widetilde{\alpha}}_{j} = contraer({\widetilde{\alpha}}_{j},\ {\widehat{\alpha}}_{B})
```

vía una combinación convexa simple ponderada por incertidumbres.

**5.5 Verificación de colapso y diagnósticos de residuos**

Dentro de cada grupo de coherencia $`B`$, calcular

``` math
{y(L) = v(L)L}^{{\widehat{\alpha}}_{B} - 1}
```

**Predicción:** $`y(L)`$ es **plano** vs. $`L`$. Regresar $`log\ y`$ sobre $`log\ L`$; una pendiente residual con $`\mid m \mid > 0.1`$ (95% IC excluyendo 0) señala **mala especificación del modelo** (p.ej., $`\alpha`$ variable dentro del grupo, sistemáticos de geometría).

**Residuos secundarios:** Examinar $`y(L)`$ vs. (i) error de inclinación, (ii) métrica de difuminado de haz, (iii) corrección de deriva asimétrica. Correlaciones significativas indican que las tuberías de reducción necesitan ajuste.

**5.6 Estrategia de agrupamiento y tamaño de muestra**

- **Anillos:** espaciado logarítmico con $`\Delta\ \log\ L = 0.08 - 0.12`$, asegurando $`\geq 5`$ elementos de resolución a través del ancho.

- **Grupos de coherencia:** agrupar anillos adyacentes por $`\widetilde{\alpha}`$ usando agrupamiento Ward 1-D con restricción de **contigüidad en radio**; objetivo $`\geq 5`$ anillos por grupo.

- **Meta entre galaxias:** por tipo de grupo (coherencia baja/media/alta), agrupar estimaciones de pendiente entre galaxias usando meta-análisis de efectos aleatorios para reportar un valor poblacional de $`1 - \alpha`$.

**5.7 Incertidumbre, control de calidad, y exclusiones**

- **Incertidumbre de inclinación/PA:** propagar vía Monte Carlo (extraer $`i`$, PA de posteriores; reajustar pendientes).

- **Incertidumbre de distancia:** afecta intersecciones más que pendientes; aún se propaga en el MC.

- **Umbral de resolución:** excluir anillos con menos de 3 elementos de resolución a través del ancho radial o con FWHM de PSF $`> \ 0.5\,\Delta R`$.

- **Difuminado de haz:** requerir factor de corrección $`< 20\%`$ o marcar como TENTATIVO.

- **Deriva asimétrica:** aplicar solo cuando fracción de dispersión $`> 0.15`$; de lo contrario la rotación del gas se usa tal cual.

**Criterios de parada (por galaxia):** marcar galaxia **NO APTA** si $`< 2`$ grupos de coherencia pasan tanto verificaciones de pendiente como de colapso después de control de calidad.

**5.8 Pseudocódigo (contrato de análisis)**

```
para cada galaxia G:
    preprocesar_imagenes_y_cinematica(G)
    anillos = crear_anillos_log(G, dlogL=0.1)

    para cada anillo A_j en anillos:
        z_j, Sigma_j = calcular_caracteristicas_estructura(A_j)
        talpha_j, sigma_talpha_j = mapear_caracteristicas_a_alpha(z_j, Sigma_j) # Sec. 5.3

    # agrupamiento de coherencia con restricción de contigüidad
    grupos = agrupar_adyacentes_por_alpha(talpha_j, k_min="5 anillos")

    resultados = []

    para grupo B en grupos:
        # Ley de pendiente
        m, IC_m = pendiente_EIV_robusta(log v vs. log L en B)
        alpha_pendiente = 1 - m

        # Comparar con alpha del indicador
        alpha_indicador = mediana(talpha_j en B)
        estado = PASA si |alpha_pendiente - alpha_indicador| <= 0.2 y IC se solapan sino TENTATIVO/FALLA

        # Colapso
        y = v * L**(alpha_pendiente - 1)
        m_c, IC_c = pendiente(log y vs. log L)
        colapso_ok = (|m_c| <= 0.1 con IC incluyendo 0)

        resultados.agregar({alpha_pendiente, IC_m, alpha_indicador, estado, colapso_ok})

    # Alpha final por anillo por contracción a pendiente del grupo
    para j en anillos:
        alpha_final[j] = contraer(talpha_j, alpha_pendiente_del_grupo(j), sigmas)

    exportar(G, resultados, alpha_final, banderas_CC)
```

**5.9 Entregables por Galaxia**

- **Mapa:** $`\widehat{\alpha}(L)`$ con banda de $`1\sigma`$.

- **Gráfico:** $`log\ v`$ vs. $`log\ L`$ coloreado por grupos de coherencia; pendientes anotadas con $`1 - \alpha/2`$.

- **Panel de colapso:** $`{v\, L}^{\widehat{\alpha} - 1}`$ vs. $`L`$ por grupo.

- **Tabla:** para cada grupo$`- {\widehat{\alpha}}_{indicador}`$, $`{\widehat{\alpha}}_{pendiente}`$, ICs, veredicto de colapso, banderas de CC.

**5.10 Reglas de interpretación (por grupo)**

1.  **PASA (soporte fuerte):** pendiente $`= 1 - \widehat{\alpha}`$ (solapamiento IC) y colapso plano; sin banderas CC fuertes.

2.  **PARCIAL:** pendiente coincide pero colapso débil (sugiere deriva leve de α o residuos de geometría).

3.  **FALLA:** pendiente no coincide o colapso muestra tendencia significativa; verificar CC; si persiste, RTM no soportado en ese grupo.

**6. Comparación con Expectativas de Solo Gravedad**

Este capítulo convierte la ley de pendiente de RTM en **contrastes directos y falsificables** con dos líneas base:

- **GR + solo bariones (sin DM):** dinámica clásica con distribución de masa luminosa; las asíntotas de rotación dependen de la extensión bariónica.

- **GR + halos DM (práctica ΛCDM):** añadir un halo paramétrico (p.ej., NFW, Burkert) y ajustar parámetros libres por galaxia.

RTM mantiene la gravedad intacta pero añade un **campo de coherencia** $`\alpha(L)`$ que modifica **tiempos operacionales**. Los discriminantes a continuación se enmarcan como **pruebas de pendiente** y **verificaciones de colapso** que no dependen de normalización absoluta.

**6.1 Asíntotas de disco externo: planitud sin halos vs. caída Kepleriana**

**Expectativa de solo gravedad.** Para discos finitos, más allá de la mayoría de los bariones se espera $`{v(L) \propto L}^{- 1/2}`$ (aproximación a Kepleriano con correcciones geométricas). En la práctica, los modelos puramente bariónicos luchan por mantener $`v`$ **plano** a través de décadas en $`L`$ sin masa añadida.

**Predicción RTM (P1 redux).** Si el medio externo es **débilmente coherente** ($`\alpha \rightarrow 1`$), entonces

``` math
\frac{\partial\log v}{\partial\log L} = 1 - \alpha \rightarrow 0 \Rightarrow v(L) \approx \text{const.}
```

**Discriminante D1 (auditoría de pendiente).** En **anillos externos** seleccionados por indicadores de baja coherencia, ajustar $`log\ v`$ vs. $`log\ L`$.

- **RTM PASA:** pendiente $`m`$ se agrupa ajustadamente cerca de 0 **y** el colapso $`{v\ L}^{\alpha - 1}`$ es plano.

- **Solo bariones FALLA:** mismos datos, mismos anillos, requerirían $`m \approx - 1/2`$ a menos que se añada masa oculta.

- **Ambigüedad DM:** los halos pueden ajustar $`m \approx 0`$, pero los **mismos anillos** también deben pasar D2–D4 abajo para distinguir RTM.

**6.2 Diversidad de curva interna: coherencia vs. ajuste de halo**

**Hecho observado.** Galaxias con masa bariónica similar muestran **formas internas diversas** (ascensos pronunciados/lentos). Los ajustes DM acomodan esto con concentración de halo/perfiles contraídos; MOND invoca aceleración local; **ambos** requieren *ajuste* por galaxia.

**Mecanismo RTM.** Dentro de barras/bulbos/cúmulos, $`\alpha(L) >`$<!-- -->1 eleva los tiempos orbitales locales, produciendo

``` math
m = \frac{\partial\log v}{\partial\log L} = 1 - \alpha < 0\quad\text{(ascensos más pronunciados/declives más suaves dependen de la geometría)}.
```

El punto clave es la **covariación**: la **pendiente** interna debe rastrear el $`\mathbf{\alpha}`$ **derivado de estructura**, no un parámetro libre de halo.

**Discriminante D2 (acoplamiento estructura–pendiente).** Después de controlar por masa y geometría, regresar residuos de pendiente interna $`\Delta m`$ sobre indicadores de coherencia (potencia de barra $`P_{2}`$, entropía multiescala $`E^{\star}`$, aglomeración $`Q`$, etc.).

- **RTM PASA:** corr($`\Delta m`$, $`\widehat{\alpha}`$) es significativa y positiva en magnitud (más coherencia → $`m`$ más negativa o ascenso/aplanamiento más pronunciado, según geometría), y permanece después de parcializar tamaño y densidad superficial.

- **DM/MOND FALLA:** los residuos se alinean principalmente con parámetros de halo/aceleración, y **no** con estructura una vez que los bariones están fijos.

**6.3 La relación bariónica de Tully–Fisher (bTFR): anatomía de residuos**

**Comportamiento base.** bTFR es ajustada pero muestra **residuos**. En ajustes DM, los residuos correlacionan con **concentración/espín de halo**; en MOND, con matices de **función de interpolación/aceleración**.

**Reformulación RTM.** Si $`v`$ se muestrea donde $`\alpha \rightarrow 1`$, la bTFR **de orden principal** se cumple con residuos mínimos. Si se muestrea más adentro ($`\alpha`$ mayor), el $`v`$ medido está **sistemáticamente sesgado** relativo al valor asintótico.

**Discriminante D3 (vínculo residuo–coherencia).**

- Calcular residuos $`\Delta\ log\ v`$ de un ajuste bTFR a nivel de galaxia.

- Probar $`\Delta\ log\ v`$ vs. un **índice de desajuste de** $`\mathbf{\alpha}`$, p.ej., $`\delta_{\alpha} \equiv \widehat{\alpha}(R_{med}) - 1.`$

  - **RTM PASA:** $`\Delta\ log\ v`$ correlaciona con $`\delta_{\alpha}`$ (muestrear dentro de zona coherente deprime $`v`$, residuo negativo), y la correlación **desaparece** cuando se mide $`v`$ en el **radio de pendiente cero** en cada galaxia.

  - **DM/MOND FALLA:** correlación residuo–$`\delta_{\alpha}`$ es débil/ausente una vez que masa y tamaño están controlados.

**6.4 Colapso entre anillos vs. libertad paramétrica**

**Colapso RTM.** Dentro de cualquier grupo de coherencia $`B:\ y(L) = v(L)\, L^{{\widehat{\alpha}}_{B} - 1}`$ debe ser **plano**. Esta es una restricción **funcional** más fuerte que ajustar una intersección.

**Discriminante D4 (colapso por grupo).**

- **RTM PASA:** pendientes residuales $`\mid mB \mid \leq 0.1`$ (IC incluye 0) entre grupos y galaxias; meta-pendiente agrupada de efectos aleatorios consistente con 0.

- **DM/MOND FALLA (como prueba de mecanismo):** Aunque halos/leyes de aceleración pueden reproducir **una curva**, **no** predicen colapsos por grupo ligados a coherencia **medida independientemente**. El fallo de colapso después de condicionar sobre $`\widehat{\alpha}`$ cuenta contra RTM; el éxito cuenta como firma única.

**6.5 Elípticas y perfiles de dispersión: Jeans vs. coherencia**

**Línea base de Jeans.** Con anisotropía $`\beta(r)`$ y perfil de masa $`M(r),\ \sigma(r)`$ sigue de la ecuación de Jeans; DM añade masa a $`r`$ grande, pronunciando/aplanando perfiles por elección de halo y $`\beta`$.

**Regla de pendiente RTM para dispersiones.** En grupos donde $`\alpha(r)`$ es aproximadamente constante,

``` math
\frac{\partial\log\sigma}{\partial\log r} = 1 - \alpha\quad\left( \text{hasta correcciones de anisotropía} \right)
```

Discriminante D5 (pendiente de dispersión vs. estructura).

- **RTM PASA:** pendiente de $`\sigma`$ rastrea $`\widehat{\alpha}`$ desde textura fotométrica (núcleos/discos embebidos → mayor $`\alpha\  \rightarrow`$ pendiente más positiva/menos negativa), y $`{\sigma\ r}^{\widehat{\alpha} - 1\ }`$ por grupo colapsa.

- **DM/MOND FALLA:** Los cambios necesarios se absorben en $`M(r)`$ o $`\beta(r)`$ con poco/ningún vínculo a estructura **medida**.

**6.6 Donde las líneas base y RTM coinciden (verificaciones de sanidad)**

Hay regímenes donde **todos** los modelos predicen comportamiento similar; los usamos como **pruebas nulas**:

- **Controles Keplerianos:** binarias amplias, sistemas planetarios externos, globulares a $`r`$ grande. La coherencia es irrelevante; RTM debe reducirse a pendientes clásicas.

- **Núcleos de cuerpo sólido:** efectos puramente geométricos en regiones muy centrales pueden imitar $`m \approx + 1`$. RTM **no** reclama crédito ahí; las pruebas deben evitar radios sub-resolución.

- **HI externo ultradifuso:** si los indicadores de estructura confirman $`\alpha \approx 1`$, **todos** los modelos permiten $`m \approx 0`$. Los discriminantes entonces se desplazan a **anatomía de residuos bTFR** (Sec. 6.3) y **colapso** (Sec. 6.4).

**6.7 Matriz de decisión (por galaxia, por grupo)**

| **Prueba** | **Evidencia a favor de RTM** | **Evidencia contra RTM** | **Qué diría DM/MOND** |
|----|----|----|----|
| **D1:** pendiente externa | $m \approx 0$ en grupos de $\alpha$ baja **y** colapso | *m≈−1/2* o sin colapso | DM puede ajustar *m≈0* pero no predice colapso |
| **D2:** diversidad interna | $`\Delta m`$ correlaciona con $`\widehat{\alpha}`$ (estructura) | $`\Delta m`$ no correlaciona con estructura | DM: parámetros de halo; MOND: escala de aceleración |
| **D3:** residuos bTFR | $`\Delta\ log\ v\  \leftrightarrow \delta_{\alpha}`$ desaparece en radio de pendiente cero | Sin relación con $`\delta_{\alpha}`$ | DM: residuos $`\leftrightarrow`$ concentración/espín de halo |
| **D4:** colapso | $`{v\ L}^{\widehat{\alpha} - 1}`$ plano por grupo | Pendiente residual ( | m_B |
| **D5:** dispersiones | *∂ log σ / ∂ log r=1−*$`\widehat{\alpha}`$ y colapso | Sin vínculo pendiente–estructura | *M(r), β(r)* ajustables lo ajustan post-hoc |

**6.8 Modos de fallo pre-registrados**

Las afirmaciones galácticas de RTM son **falsificadas** si, después de control de calidad (Sec. 5):

1.  Los grupos externos de α baja muestran pendientes **no cero** inconsistentes con 0 (D1 falla) **y** no colapsan (D4 falla).

2.  Los residuos de pendiente interna **no** correlacionan con $`\widehat{\alpha}`$ derivado de estructura una vez que masa/tamaño están controlados (D2 falla).

3.  Los residuos bTFR son **independientes** de $`\delta_{\alpha}`$ y permanecen así incluso cuando se muestrea en el radio de pendiente cero (D3 falla).

4.  Las pendientes de dispersión en elípticas no muestran **ninguna** relación con $`\widehat{\alpha}`$ basado en textura (D5 falla).

> Cualquier dos fallos independientes bajo buen control de calidad marcan RTM **no soportado** en escalas galácticas; pasar D1–D4 a través de una muestra diversa constituye **soporte fuerte**.

**7. Lentes Gravitacionales y Cúmulos: Verificaciones de Consistencia**

RTM afirma alterar **tiempos operacionales** (relojes orbitales) vía el exponente de coherencia $`\alpha(L)`$, no la **curvatura** del espacio-tiempo. Si es verdad, las **lentes gravitacionales**—que dependen de curvatura generada por tensor energía-momento—deberían seguir rastreando la **distribución de masa bariónica** (más cualquier masa genuinamente no bariónica, si está presente) independientemente de $`\alpha`$. Este capítulo presenta pruebas que comparan **masa inferida de lentes** con **cinemática reinterpretada bajo RTM**, desde galaxias hasta cúmulos. Cualquier **brecha de masa persistente y coherente** después de condicionar sobre $`\widehat{\alpha}(L)`$ constituye un **límite de alcance** o **falsificación** directa en esas escalas.

**7.1 Relojes vs. curvatura: el principio guía**

- **Qué cambia RTM:** el mapeo $`{T \propto L}^{\alpha(L)}`$ que gobierna tiempos orbitales/de relajación. Los observables cinemáticos que dependen de períodos o deriva (velocidades de rotación, dispersiones, frecuencias epicíclicas/verticales) son modificados vía $`{v \propto L}^{1 - \alpha}`$ o $`{\sigma \propto L}^{1 - \alpha}`$ **dentro de grupos de coherencia**.

- **Qué no cambia RTM:** las ecuaciones de campo de Einstein y las geodésicas que establecen la deflexión de luz y lentes. Así, los **mapas de masa de lentes** deben ser consistentes con **bariones** (dentro de sistemáticos conocidos) a menos que exista masa real no vista o RTM falle en describir la dinámica.

**Prueba operacional.** Construir, para cada sistema, dos inferencias de masa:

1.  $`M_{lentes}(R)`$ desde lentes fuertes/débiles (o dinámica+rayos X en cúmulos).

2.  $`M_{cin}^{RTM}(R)`$ desde velocidades/dispersión observadas **después** de reinterpretar la cinemática con $`\widehat{\alpha}(R)`$.

La consistencia requiere $`M_{cin}^{RTM} \approx M_{lentes}`$ dentro de incertidumbres; un sesgo sistemático que **sobrevive** al condicionamiento sobre $`\alpha`$ señala un límite de RTM o masa extra genuina.

**7.2 Galaxias con lentes fuertes (anillos de Einstein y cuádruples)**

**Configuración.** Elegir lentes con anillos de Einstein/cuádruples de alta calidad ($`M_{lentes}(R_{E})`$ preciso). Obtener cinemática IFU para construir $`\widehat{\alpha}(R)`$ (Sec. 5).

Prueba de consistencia RTM SL-1 (masa encerrada en $`R_{E}`$​).

- Calcular $`M_{cin}^{RTM}(R_{E})`$ desde el soporte rotacional/de dispersión observado usando la **ley de velocidad RTM** dentro de grupos de coherencia que intersectan $`R_{E}`$.

- **Pasa:** $|M\_{\text{cin}}^{RTM} - M\_{\text{lentes}}| / M\_{\text{lentes}} \leq \varepsilon$ ($\varepsilon$ pre-registrado, p.ej., 15%)

- **Falla:** sobreestimaciones o subestimaciones sistemáticas a través de la muestra que no pueden trazarse a calibración de $`\alpha`$ o sistemáticos de anisotropía.

**Discriminante RTM SL-2 (colapso anular).\**
Dentro de un anillo alrededor de $`R_{E}`$ con $`\widehat{\alpha}`$ aproximadamente constante, la cantidad

``` math
{y(R) = v(R)R}^{\alpha - 2/1}
```

debe ser **plana** vs. $`R`$. El fallo de colapso mientras la masa de lentes está bien restringida argumenta que la reinterpretación cinemática de RTM es inadecuada **a la escala de lentes**.

**Añadido de retardo temporal SL-3.** Para cuásares con lentes con retardos temporales medidos, verificar que las inferencias cosmográficas (p.ej., $`H_{0}`$) permanezcan **sin cambios** al cambiar el modelo dinámico a RTM, ya que los retardos dependen de **curvatura + diferencias de potencial**, no de relojes orbitales. Cualquier cambio indica conteo doble (dejando incorrectamente que $`\alpha`$ se filtre a las lentes).

**7.3 Lentes débiles en galaxias de disco (halos apilados)**

**Configuración.** Usar perfiles de cizalla de lentes débiles apilados de grandes muestras de discos agrupados por **coherencia estructural** (p.ej., fuerza de barra, métricas de textura) para obtener $`M_{lentes}(R)`$ a decenas–cientos de kpc.

**Prueba de consistencia RTM WL-1 (grupos externos).**\
En **anillos externos de** $`\widehat{\mathbf{\alpha}}`$ **baja** (donde las curvas de rotación se aplanan), RTM predice **cinemática plana** sin curvatura extra. Por lo tanto, la señal de **lentes** a $`R`$ grande debería ser explicable por **bariones + gas conocido** solo si la curvatura realmente rastrea solo masa.

- **Pasa:** $`M_{lentes}(R)`$ apilada consistente con mapas de bariones y con $`M_{cin}^{RTM}(R)`$.

- **Alcance/Falla:** un exceso robusto en cizalla **después** del condicionamiento sobre $`\alpha`$ indica masa más allá de bariones—o el alcance de RTM termina aquí o se necesita masa oscura.

**Verificación cruzada interna WL-2 (división por estructura).**\
Dividir discos a masa estelar fija por coherencia (alta vs. baja barra/textura).

- RTM espera **halos de lentes débiles similares** (ya que las lentes ignoran $`\alpha`$) pero **diferentes pendientes cinemáticas internas**.

- Si los perfiles de lentes *también* se dividen sistemáticamente con coherencia a mapas de bariones fijos, eso sugiere una correlación entre **estructura** y **masa verdadera** (no un efecto solo de $`\alpha`$), ajustando el alcance.

**7.4 Cúmulos de galaxias: donde RTM puede (no) aplicarse**

**Verificación de realidad.** Los cúmulos ricos exhiben masas de lentes fuertes/débiles y masas hidrostáticas de rayos X que **exceden** los bariones. Si RTM solo re-temporiza **relojes orbitales** dentro de bariones estructurados, **no debería** borrar déficits de masa en cúmulos—incluso si $`\alpha`$ afecta alguna dinámica intra-cúmulo.

**Prueba de cúmulos CL-1 (presupuesto de masa).**

- Construir $`M_{lentes}(R)`$ y $`M_{X}(R)\ (rayos X).`$

- Medir campos $`\widehat{\alpha}(R)`$ desde textura ICM (fluctuaciones de presión/densidad, espectros de potencia) y subestructura de galaxias.

- Calcular $`M_{cin}^{RTM}(R)`$ desde dispersiones de galaxias usando $`{\sigma \propto R\,}^{1 - \widehat{\alpha}}`$ en **grupos de coherencia** (Jeans con relojes RTM).

- **Resultado esperado:** incluso con RTM, una **masa residual significativa** permanece en cúmulos—la señal clásica de DM.

- **Interpretación:** **condición de alcance** de RTM: es una re-temporización cinemática a **escala galáctica**, no un reemplazo para DM en escalas de cúmulos. Si, implausiblemente, RTM borrara la brecha de masa de cúmulos, la consistencia lentes–dinámica se rompería (contradiciendo masa basada en curvatura).

**Fusiones tipo Bullet CL-2.** En sistemas donde ocurren desplazamientos gas–galaxia, los picos de lentes siguen la masa sin colisiones. RTM predice **sin desplazamiento** de picos de lentes con $`\alpha`$; cualquier intento de usar $`\alpha`$ para imitar el desplazamiento dejaría incorrectamente que los relojes alteren la curvatura—**no permitido**.

**7.5 Algoritmo de reconciliación cinemática–lentes (por sistema)**

1.  Medir $`\widehat{\alpha}(R)`$**:** construir grupos de coherencia desde indicadores de estructura (Sec. 5).

2.  **Dinámica inferida por RTM:** dentro de cada grupo, ajustar pendientes $`m = 1 - \widehat{\alpha}`$, verificar colapso $`{v\ R}^{\widehat{\alpha} - 1}`$, y recuperar $`M_{cin}^{RTM}(R)`$ con correcciones EIV y priors de anisotropía (para dispersiones).

3.  **Masa de lentes:** obtener $`M_{lentes}(R)`$ (fuerte/débil) con covarianzas completas.

4.  **Comparar:** calcular $`{\Delta(R) = M}_{cin}^{RTM}(R) - M_{lentes}(R)`$ y su incertidumbre; reportar residuos **por grupo** en lugar de un solo número global.

5.  **Decisión:**

- **CONSISTENTE:** $`\mid \Delta \mid /M_{lentes} \leq \varepsilon`$ en la mayoría de los grupos y sin tendencia con $`\widehat{\alpha}`$.

-  **LÍMITE DE ALCANCE:** los residuos se concentran a **radios de escala de cúmulo** o en sistemas donde $`\widehat{\alpha}`$ no puede estimarse establemente.

-  **FALSIFICADO:** residuos coherentes y significativos a través de muchos grupos de **escala galáctica** donde el control de calidad pasa y $`\widehat{\alpha}`$ es estable.

**7.6 Retardos temporales y pruebas relativistas (sanidad)**

- **Retardos temporales de lentes fuertes:** dependen del **potencial de Fermat** (curvatura + geometría). RTM **no** debe alterar los retardos predichos cuando el mapa de masa está fijo. Por lo tanto re-ajustamos retardos bajo GR con la misma masa y mostramos **invarianza** al reemplazar dinámica Newtoniana con RTM para los movimientos **estelares/gas**.

- **Restricciones PPN/sistema solar:** en regímenes de baja coherencia relevantes para pruebas del sistema solar, $`\alpha`$ se reduce a su línea base clásica y las restricciones de lentes/deflexión permanecen **sin cambios**—una verificación de sanidad incorporada.

**7.7 Resultados pre-registrados (pasa/falla)**

- **PASA (escala galáctica):**

  1)  Los grupos externos de $`\widehat{\alpha}`$ baja muestran $`m \approx 0`$ **y** colapsan;

  2)  $`M_{cin}^{RTM}(R)`$ coincide con $`M_{lentes}(R)`$ en anillos/cuádruples dentro de $`\leq 15\%`$;

  3)  Las pilas de lentes débiles a bariones fijos **no** se dividen por coherencia, mientras que las pendientes cinemáticas **sí**

<!-- -->

- **ALCANCE (cúmulos):**\
  RTM **no** remueve la brecha de masa de cúmulos; $`M_{lentes}(R)`$ excede bariones + cinemática RTM. RTM está así limitado a **cinemática a escala galáctica** a menos que se introduzca física adicional.

- **FALLA (escala galáctica):**\
  Brechas de masa lentes–cinemática consistentes y significativas **después** del condicionamiento sobre $`\alpha`$, o no-colapsos por grupo acoplados con estimaciones estables de $`\widehat{\alpha}`$ y buen control de calidad, falsifican RTM como mecanismo explicativo para perfiles de rotación/dispersión galácticos.

**Conclusión.** Las lentes son el **guardacarril** de RTM: al separar **relojes** de **curvatura**, podemos decir cuándo la re-temporización por coherencia es suficiente (galaxias) y dónde no puede serlo (cúmulos). Pasar las verificaciones de consistencia de lentes hace de RTM una reinterpretación creíble y estrechamente delimitada de la cinemática galáctica; fallarlas traza un límite claro y preserva la gravedad estándar donde debe permanecer intocada.

**8. Crecimiento de Estructura Cósmica (Bosquejo)**

Este capítulo esboza cómo un **campo** $`\mathbf{\alpha}`$—un exponente de coherencia espacialmente variable ligado a la organización bariónica—podría modular **escalas de tiempo** durante el ensamblaje de galaxias y sus subestructuras sin alterar la gravedad. La postura permanece **pendiente primero**: RTM predice **qué tan rápido** se desarrollan los procesos a una escala dada, no **que** aparezcan nuevas fuerzas. La sección cierra con **observables** y **pruebas de fallo** que mantienen el programa falsificable.

**8.1 Relojes de colapso bajo RTM**

Sea $`t_{col}(L)`$ el tiempo característico para que un parche bariónico auto-gravitante de tamaño $`L`$ proceda desde crecimiento lineal a no linealidad (fragmentación/condensación). La teoría estándar suministra un tiempo dinámico $`t_{din} \sim 1/\sqrt{G\rho}`$ y retardos adicionales del transporte de momento angular, enfriamiento, turbulencia. RTM trata el **tiempo operacional** como

``` math
t_{\text{col}}(L) = t_{\text{din}}(L)\left( \frac{L}{L_{0}} \right)^{\alpha(L) - \alpha_{0}}\Theta
```

donde $`\alpha_{0}`$ es una banda base (débilmente coherente) y $`\Theta`$ agrega microfísica adimensional mantenida fija **dentro** de un grupo de coherencia. Consecuencias:

- Las regiones con **mayor coherencia** ($`\alpha > \alpha_{0}`$) **alargan** los relojes de colapso en esa *misma escala*, retrasando el crecimiento de barras/espirales o la condensación de cúmulos relativo a zonas difusas.

- Los **gradientes** $`\nabla\alpha`$ siembran **temporización diferencial** a través de radios, imprimiendo desfases entre barras, espirales, y alabeos.

**8.2 Transporte de momento angular y cronologías de barras**

La formación de barras requiere redistribución de momento angular. Sea $`t_{J}(L)`$ la escala de tiempo característica para transporte de $`J`$ en un anillo de ancho $`\sim L`$. Con RTM:

``` math
t_{J}(L) \propto L^{\alpha(L)}\quad \Rightarrow \quad\frac{\partial\log t_{J}}{\partial\log L} = \alpha(L).
```

**Predicciones.**

- **Secuenciación de adentro hacia afuera.** Si los discos internos son más coherentes ($`\alpha_{in} > \alpha_{out}`$), las barras/espirales internas **quedan atrás** del crecimiento de patrones externos; inversamente, si la retroalimentación fragmenta la coherencia interna ($`\alpha_{in} \rightarrow 1`$), las barras emergen **más temprano** de lo que los tiempos seculares estándar sugerirían.

- **Longitud de barra vs. gradiente de** $`\mathbf{\alpha}`$**.** Los semiejes mayores de barras anticorrelacionan con $`\nabla\alpha`$: **caídas** más fuertes hacia afuera en $`\alpha`$ (interno coherente → externo difuso) limitan el crecimiento de barras más temprano (el disco externo supera al interno en el desprendimiento de $`J`$).

**Observables.** A masa y fracción de gas fijas, **fracción de barras** y **longitud de barra** correlacionan con la **forma** de $`\widehat{\alpha}(R)`$: barras largas y fuertes prefieren perfiles de $`\alpha`$ **más planos**; barras cortas/débiles aparecen donde $`\alpha`$ cae rápidamente con el radio.

**8.3 Formación de cúmulos, migración, y discos gruesos**

Los cúmulos masivos de formación estelar en discos de alto $`z`$ migran hacia adentro en una escala de tiempo $`t_{mig}`$ establecida por torques y fricción dinámica.

**Modulación RTM.**

``` math
t_{\text{mig}}\left( L_{\text{cumulo}} \right) \sim t_{\text{din}}\left( \frac{L_{\text{cumulo}}}{L_{0}} \right)^{\alpha - 1}
```

así que a tamaño de cúmulo fijo, **mayor** $`\mathbf{\alpha}`$ **local ralentiza la migración**, permitiendo que los cúmulos **vivan más** y engrosen discos vía dispersión prolongada.

**Predicciones.**

- **Longevidad de cúmulos vs.** $`\mathbf{\alpha}`$**.** A densidad superficial fija, discos con mayor $`\widehat{\alpha}`$ sostienen mayores **tiempos de vida de cúmulos** y muestran capas estelares **más gruesas** más temprano.

- **Gradientes de edad.** Si $`\alpha`$ declina con el radio, los cúmulos internos (mayor $`\alpha`$) envejecen **más** in situ que los cúmulos externos (menor $`\alpha`$) para el mismo tiempo de retrospección—una tendencia edad–radio **invertida** relativa a las expectativas de pura fricción dinámica.

**8.4 Planos de satélites, alabeos, y desfases**

Los gradientes de coherencia pueden **bloquear en fase** ciertas familias orbitales.

**Predicciones.**

- **Planos de satélites.** Si el disco externo/CGM del huésped exhibe un campo $`\mathbf{\alpha}`$ **anisotrópico** (p.ej., a lo largo de filamentos), las órbitas de satélites **persisten** preferentemente en ese plano (períodos operacionales más largos para difusión fuera del plano), aumentando la probabilidad de **alineaciones planares aparentes** sin invocar anisotropías especiales de DM.

- **Fases de alabeos.** Las zonas radiales donde $`\nabla\alpha`$ es mayor deberían mostrar **desfases** entre alabeos de HI y flexiones estelares; el signo del desfase cambia con el signo de $`\nabla\alpha`$.

- **Asimetría.** Los modos $`m = 1`$ persistentes correlacionan con variaciones **azimutales** en $`\alpha`$ (barras + cúmulos de un lado), produciendo **asimetrías cinemáticas** que rastrean mapas de estructura.

**8.5 Historias de formación estelar (SFHs) y** $`\mathbf{\alpha}`$

Porque $`t_{col}`$ y $`t_{J}`$ se estiran con $`\alpha`$, las **SFHs** heredan **firmas de coherencia**:

- **De adentro hacia afuera vs. de afuera hacia adentro.** Discos con alto $`\alpha`$ interno y bajo $`\alpha`$ externo tienden **de afuera hacia adentro** en temporización de estallidos (anillos externos se encienden primero); la forma de $`\alpha`$ inversa invierte la tendencia.

- **Ráfagas.** Los parches de bajo $`\alpha`$ (difusos/turbulentos) tienen **ciclos más cortos**, aumentando las ráfagas e impulsando mayor potencia HI/H$`\alpha`$ a escalas pequeñas; los parches de alto $`\alpha`$ suavizan las SFHs.

- **Dispersiones de metalicidad.** La migración prolongada bajo alto $`\alpha`$ amplía las distribuciones de metalicidad a radio dado (tiempos de mezclado de fase más largos), comprobable con mapas de metalicidad IFU.

**8.6 Tendencias de alto corrimiento al rojo**

A $`z \gtrsim 1`$, los discos ricos en gas son aglomerados y turbulentos. Dos escenarios estilizados:

- **Escenario A (** $`\mathbf{\alpha}`$ **global bajo).** Si los discos tempranos son en gran parte **difusos** (la retroalimentación fragmenta la coherencia), $`\alpha \approx 1`$ sobre radios amplios $`\Rightarrow`$ crecimiento de patrón **rápido**, tiempos de vida de cúmulos **cortos**, aproximación más rápida a rotación plana más allá de bulbos compactos.

- **Escenario B (** $`\mathbf{\alpha}`$ **jerárquico).** Si las estructuras anidadas (cúmulos gigantes, cadenas) aumentan la coherencia ($`\alpha > 1`$) localmente, barras y cúmulos de larga vida deberían **coexistir** temprano; las pendientes de rotación exhiben fuerte **diversidad radial** que **se desvanece** a medida que $`\alpha \rightarrow 1`$ con el tiempo cósmico (asentamiento del disco).

**Palanca observable.** Comparar la **evolución** de la **distribución de pendientes** $`m(R) = \partial\ log\ v\ /\ \partial\ log\ R`$ a través del corrimiento al rojo después de condicionar sobre **indicadores de** $`\mathbf{\alpha}`$. RTM predice que la **dispersión** en mmm a masa fija se estrecha a medida que los campos de $`\alpha`$ **se aplanan** con el tiempo.

**8.7 Bosquejo de simulación (cómo probar lo anterior)**

**Integrador de órbitas consciente de alfa.** Tomar un código N-cuerpos+gas estándar o un banco de pruebas sin colisiones; en cada paso, reescalar **avances de tiempo** en una celda por $`dt' = {dt(L/L_{0})}^{{\alpha(x) - \alpha}_{0}}`$. Mantener fuerzas **sin cambios**. Alimentar α($`x`$) desde (i) perfiles analíticos (alto $`\alpha`$ centrado en barra), (ii) mapas de indicadores derivados de luz, o (iii) reglas auto-actualizables (la coherencia crece con densidad superficial sostenida). Leer:

- Pendientes de rotación y **colapso** $`{vR}^{\alpha - 1}`$ dentro de grupos;

- Tiempo de formación de barra vs. $`\nabla\alpha`$;

- Tiempos de vida de cúmulos y engrosamiento de disco vs. $`\alpha`$ local;

- Desfases de alabeos vs. $`\nabla\alpha`$.

**Falsificación dentro del sandbox.** Si mantener fuerzas fijas y solo **re-temporizar** no puede reproducir ninguna de las secuencias observadas (p.ej., patrones de emergencia de barras) cuando los campos de $`\alpha`$ están ajustados a estructura **medida**, la historia RTM a nivel de crecimiento se debilita.

**8.8 Resumen de observables y condiciones de fallo**

| **Fenómeno** | **Firma RTM** | **Cómo medir** | **Falla si…** |
|----|----|----|----|
| Emergencia de barra | Temporización rastrea ∇α; barras largas necesitan α(R) plano | Fracción/longitud de barra vs. forma de $`\widehat{\alpha}`$(R) | Sin correlación después de control de masa/tamaño |
| Longevidad de cúmulos | Mayor α local ⇒ más longevos, discos más gruesos | Edades de cúmulos, espesor vs. $`\alpha`$ | Tiempos de vida independientes de $`\widehat{\alpha}`$ |
| Alabeos | Desfases donde ∇α es grande | Flexiones HI vs. estelares vs.∇α | Sin vínculo sistemático desfase–gradiente |
| Planos de satélites | Alineación con α anisotrópico en CGM | Orientación de plano vs. anisotropía de α | Sin alineación a bariones fijos |
| Temporización SFH | De afuera hacia adentro o de adentro hacia afuera establecido por forma de ($`\alpha`$) | SFHs resueltas vs. forma de α | Las tendencias desaparecen al condicionar sobre $`\widehat{\alpha}`$ |

**8.9 Nota de alcance**

Estos bosquejos **no** afirman que RTM reemplaza la física bariónica detallada (enfriamiento, retroalimentación, turbulencia). Afirman que un **campo de exponente único** $`\alpha(x)`$ puede **organizar la temporización** de procesos por lo demás estándar. La recompensa es un portafolio de pruebas a **nivel de pendiente** y de **secuenciación**—cada una con un **modo de fallo** claro—que conectan historias de crecimiento con **mapas de estructura** medibles. Si esos vínculos no se materializan bajo buen control de calidad, el papel de RTM en el crecimiento cósmico está **limitado** o **falsificado** para los regímenes probados.

**9. Plan de Datos y Medición**

Este capítulo convierte las predicciones en un **contrato de análisis**: conjuntos de datos, selección, preprocesamiento, construcción de $`\widehat{\alpha}`$(L) (Sec. 5), pruebas de pendiente/colapso, anatomía de residuos bTFR, y reconciliación lentes–cinemática (Sec. 7). Todo lo siguiente está redactado para que otro grupo pueda reproducir la tubería de principio a fin.

**9.1 Muestras y criterios de inclusión**

**Galaxias de disco (foco en rotación):**

- Cinemática HI o Hα resuelta espacialmente con ≥10 puntos radiales independientes más allá de $`{2\ R}_{d}`$

- Imágenes profundas ópticas/NIR (FWHM de PSF ≤ 0.5 del ancho del anillo interno) para mapas de estructura.

- Distancia conocida, inclinación $`i \in \lbrack 30 \circ ,80 \circ \rbrack`$, ángulo de posición (PA), y mapas de masa estelar/gas.

- Objetivo de **tres cohortes** balanceadas en masa y morfología:\
  C1: barradas de alto brillo superficial; C2: espirales de gran diseño sin barra; C3: enanas/LSBs.

**Elípticas (foco en dispersión):**

- Espectroscopía IFU con perfiles radiales de $`\sigma(R)`$ hasta $`{\geq 1.5 - 2\ R}_{e}`$

- Imágenes de alta S/N (núcleos, discos embebidos discernibles)

**Galaxias con lentes fuertes:**

- Anillos de Einstein/cuádruples con cinemática IFU que intersecta $`R_{E}`$

- Modelos de lentes públicos con covarianza (para $`M_{lentes}(R))`$

**Pilas de lentes débiles:**

- Grandes muestras de discos con catálogos de cizalla y etiquetas estructurales (fuerza de barra, métricas de textura).

**9.2 Preprocesamiento y geometría**

- **Imágenes:** sustracción de cielo, enmascaramiento, caracterización de PSF; desproyección usando $`i,\ PA`$; re-rejillado a escala de píxel común.

- **Cinemática:** corrección de difuminado de haz (modelo directo preferido); deriva asimétrica aplicada donde fracción de dispersión estelar \> 0.15; gas asumido frío.

- **Anillos:** anillos logarítmicos con $`\Delta\ \log\ L = 0.1`$; requerir $`\geq 3`$ elementos de resolución por anillo.

Todos los pasos producen **incertidumbres por anillo** (covariantes donde sea relevante).

**9.3 Construcción de** $`\widehat{\mathbf{\alpha}}\mathbf{(L)}`$

Aplicar Sec. 5: calcular características estructurales por anillo (entropía multiescala, potencia de modo, aglomeración, índices fractales/turbulencia, espesor, textura cinemática). Mapear características → $`\widehat{\alpha}`$ provisional (monótono paramétrico o ensamble de rangos), agrupar anillos adyacentes en **grupos de coherencia contiguos**, ajustar pendiente $`m = 1 - {\widehat{\alpha}}_{B}`$ en cada grupo (EIV robusto), comparar con mediana del indicador, y **contraer** para obtener $`{\widehat{\alpha}}_{j}`$, CC: verificación de colapso $`{vL}^{{\widehat{\alpha}}_{B} - 1}`$ pendiente $`\mid mc \mid \leq 0.1`$ con IC incluyendo 0.

**9.4 Pruebas de hipótesis primarias (por galaxia)**

**H-RC (Pendiente de rotación):** En cada grupo de coherencia $`B`$:

- Estimar $`m_{B} = \partial\ log\ v/\partial\ log\ L`$

- Probar $`m_{B} = 1 - \alpha/2\ mediana({\widehat{\alpha}}_{j \in B})`$ (solapamiento IC ±0.2).

**H-CL (Colapso):** Regresar $`{\log\lbrack v\, L}^{{\widehat{\alpha}}_{B} - 1}\rbrack`$ vs. $`\log L`$; requerir $`\mid m_{c} \mid \leq 0.1`$, IC incluye 0.

**H-bTFR (Anatomía de residuos):**

- Ajuste global: $`{\log\ v}_{plana}{= a + b\ \log\ M}_{b}`$

- Residuos $`\Delta\ \log\ v`$ regresados sobre $`\delta_{\alpha} \equiv \widehat{\alpha}(R_{med}) - 1`$, controlando por tamaño y densidad superficial.

- Recalcular en **radio de pendiente cero**; la correlación debe desaparecer si RTM se cumple.

**H-Disp (Elípticas):** En grupos de coherencia, $`\partial\ \log\ \sigma/\partial\ \log\ r = 1 - \widehat{\alpha}`$ (EIV-robusto); colapso de $`{\sigma\ r}^{\widehat{\alpha} - 1}`$

**H-Lentes (Consistencia de lentes):**

- **Lente fuerte:** comparar $`M_{cin}^{RTM}(R_{E})`$ con $`M_{lentes}(R_{E})`$; tolerancia $`\leq 15\%.`$

- **Pilas de lentes débiles:** a bariones fijos, los perfiles de cizalla **no** deben dividirse por coherencia; las pendientes cinemáticas **sí**.

**9.5 Plan estadístico**

- **Pendientes:** estimador Theil–Sen con pérdida de Huber; SIMEX para errores de $`L`$; ICs bootstrap (B=2000).

- **Meta-análisis:** Efectos aleatorios combinan pendientes entre galaxias dentro del mismo tipo de grupo (coherencia baja/media/alta). Reportar $`m`$ agrupado, heterogeneidad $`I^{2}`$

- **Correlaciones parciales:** Para residuos bTFR, regresar $`\Delta\ \log\ v`$ sobre $`\delta_{\alpha}`$ mientras se controla por $`{\log\ R}_{d}`$, $`\Sigma_{\star}`$

- **Pruebas múltiples:** FDR de Benjamini–Hochberg al 5% entre grupos y pruebas.

- **Pre-registro:** Congelar mapas indicador-a-$`\alpha`$ y umbrales ($`{\mid m}_{c} \mid \leq 0.1`$; $`{\mid \widehat{\alpha}}_{pendiente} - {\widehat{\alpha}}_{indicador} \mid \leq 0.2`$) antes de ver objetivos científicos.

**9.6 Expectativas de potencia (orden de magnitud)**

- **Pendientes de rotación:** Con 6–8 anillos por grupo, $`\sigma_{\log\ v} \sim 0.04`$, pendiente corregida por EIV $`SE\  \sim 0.08`$ Diferencias de $`\Delta(1 - \alpha) = 0.3`$ entre grupos dan $`> 90\%`$ potencia a $`\alpha = 0.05`$.

- **Prueba de colapso:** Detectar $`{\mid m}_{c} \mid = 0.12`$ con $`\sim 80\%`$ potencia por grupo.

- **Residuo bTFR–**$`\delta_{\alpha}`$: Con $`N \sim 150`$ discos y dispersión residual 0.08 dex, correlación $`\mid r \mid \geq 0.25`$ es detectable a $`> 90\%`$ potencia.

- **Lentes (fuertes):** Diez anillos de alta calidad con errores de masa de lentes del $`10\%`$ bastan para detectar un sesgo sistemático del $`15\%`$ a $`> 80\%`$ potencia.

**9.7 Control de calidad, exclusiones, y verificaciones adversariales**

- **Umbral de resolución:** descartar anillos con FWHM de PSF $`> \ 0.5`$ del ancho del anillo.

- **Difuminado de haz:** marcar si corrección $`> 20\%`$; excluir si $`> 35\%`$.

- **Inclinación/PA:** Monte Carlo sobre posteriores de $i, PA$; grupos que fallan estabilidad (deriva de pendiente $>0.15$) son **TENTATIVO/FALLA**.

- **Robustez de indicadores:** recalcular $`\widehat{\alpha}`$ con (i) dejar-un-indicador-fuera, (ii) mapeo basado en rango; requerir estabilidad de clasificación.

- **Galaxias de control negativo:** sistemas con estructura extremadamente suave (S0 sin rasgos) deben producir $`\alpha \rightarrow 1`$ y $`m \rightarrow 0`$ externo; el fallo dispara auditoría de tubería.

**9.8 Entregables**

Para cada galaxia:

- **Mapas:** $`\widehat{\alpha}(L)`$ con incertidumbres; máscara de grupos de coherencia.

- **Paneles:** (i) $`log\ v`$ vs. $`log\ L`$ coloreado por grupo con pendientes ajustadas; (ii) gráficos de colapso $`{vL}^{\alpha/2 - 1}`$; (iii) diagnósticos de residuos.

- **Tablas:** por grupo$`{- \widehat{\alpha}}_{indicador}`$, $`{\widehat{\alpha}}_{pendiente}`$, IC, veredicto de colapso, banderas CC.

- **Reconciliación de lentes (donde esté disponible):** $`M_{cin}^{RTM}(R_{E})`$ vs. $`M_{lentes}(R_{E})`$ con residuos.

Para la muestra:

- **Meta-pendientes** (coherencia baja/media/alta), $`I^{2}`$, y conteos pasa/falla.

- **Regresiones de residuos bTFR** y remediciones en "radio de pendiente cero".

- **Divisiones de pilas de lentes débiles** (por coherencia) y su comparación nula.

**9.9 Libro de pasa/falla (pre-declarado)**

Una galaxia contribuye **soporte** si ≥2 grupos de coherencia **PASAN** tanto H-RC como H-CL, y (si aplica) H-Lentes pasa. Una contribución **parcial** requiere PASA en H-RC o H-CL con el otro TENTATIVO y sin banderas rojas de CC. **Falla** si todos los grupos fallan pendiente o colapso bajo buen CC.

**9.10 Reproducibilidad**

- Liberar **código de análisis** (extracción de indicadores, mapeo de $`\alpha`$, pendientes EIV, verificaciones de colapso) con entornos con versiones bloqueadas.

- Proporcionar **catálogos por anillo** (características, $`\widehat{\alpha}`$, cinemática, banderas CC).

- Publicar **pre-registro** (hipótesis, umbrales, reglas de exclusión) y mapas de indicadores **congelados** antes de tocar la muestra científica principal.

> **Resultado de este plan.** El contrato de datos asegura que las afirmaciones de RTM suben o caen sobre **pendientes y colapsos por grupo** ligados a **coherencia** medida independientemente. Siguiente (Sec. 10) especificamos la **suite de simulación** que prueba la tubería bajo estrés, explora sesgos, y genera puntos de referencia de observables simulados para dinámica consciente de $`\alpha`$.

**10. Simulaciones**

Este capítulo especifica una **suite de simulación consciente de** $`\mathbf{\alpha}`$ para (i) probar si las firmas de pendiente/colapso de RTM son recuperables cuando las fuerzas son estándar pero los relojes están re-temporizados; (ii) cuantificar sesgos y modos de fallo de la tubería en Sec. 5–9; y (iii) generar **sondeos simulados** con verdad conocida ($`\alpha_{verdadero}(x)`$, masa, geometría) para validación de principio a fin.

**10.1 Filosofía: mantener fuerzas, re-temporizar actualizaciones**

Preservamos fuerzas Newtonianas/GR (sin gravedad modificada, sin masa añadida). RTM entra **solo** a través de un **reescalado temporal** local:

``` math
dt'(x) = dt\left( \frac{L(x)}{L_{0}} \right)^{\alpha(x) - \alpha_{0}}
```

donde $`L(x)`$ es una escala estructural elegida (p.ej., escala de anillo radial, espesor local del disco, longitud de suavizado), $`\alpha_{0}`$ una banda base ($`\approx 1`$), y $`\alpha(x)`$ el campo de coherencia (fijo o evolucionando). Todos los integradores abajo simplemente usan $`dt'`$ para actualizaciones de estado mientras calculan aceleraciones del potencial **sin cambios**.

**10.2 Familias de simulación**

**S1. Banco de pruebas sin colisiones (órbitas en potenciales fijos).**

- Potenciales: discos Miyamoto–Nagai + bulbos Hernquist + halos NFW opcionales (para comparaciones base).

- Partículas: $`10^{6}`$ trazadores; integrador: leapfrog o simpléctico de 4to orden con $`dt'`$ **adaptativo**.

- $`\alpha(x)`$: perfiles analíticos (prominencia centrada en barra, unidad plana externa); o anisotropía azimutal para experimentos de alabeo.

**S2. N-cuerpos de disco delgado con respuesta viva de barra/espiral.**

- Auto-gravedad en rejilla polar 2D; suavizado elegido para resolver $`< 0.5`$ ancho de anillo.

- Gas opcional como partículas inelásticamente colisionantes (esquema pegajoso) para emular disipación.

- $`\alpha(x,t)`$: (i) fijo; (ii) **acoplado a estructura** (ver §10.5).

**S3. Cubos IFU simulados / mapas de momento HI.**

- Tomar instantáneas S1/S2; renderizar campos de velocidad **línea de visión** con haz, PSF, ruido, y resolución espectral coincidentes con sondeos reales.

- Generar curvas de rotación y perfiles de dispersión con la **misma tubería** que los datos (Sec. 5 y 9).

**S4. Análogos elípticos (partículas de Jeans).**

- Poblaciones trazadoras esféricas/axisimétricas con anisotropía $`\beta(r)`$; aplicar $`dt'`$ a movimientos radiales para emular moldeado de $`\sigma(r)`$ por $`\alpha(r)`$.

- Comparar $`\widehat{\alpha}`$ recuperado de pendientes de $`\sigma`$ con la verdad.

**10.3 Definiendo el campo de coherencia** $`\mathbf{\alpha(x)}`$

**Prescripciones estáticas (verdad conocida):**

- **Perfil escalonado:** $`\alpha = \alpha_{\text{in}} > 1\text{ para }R < R_{b},\alpha = 1\text{ para }R \geq R_{b}`$

- **Perfil gradiente:** $`\alpha(R) = 1 + \Delta\alpha\ exp\left\lbrack - \left( R\text{/}R_{g} \right)^{p} \right\rbrack`$

- **Anisotropía azimutal:** $`\alpha(R,\phi) = \alpha(R)\ \left\lbrack {1 + \epsilon\ cos}2\left( \phi - \phi_{b} \right) \right\rbrack`$ para patrones tipo barra.

- **Vertical:** $`\alpha_{z}(z) = 1 + \Delta\alpha_{z}\, e^{- |z|\text{/}H}`$

**Prescripciones evolucionantes (retroalimentación a estructura):**

- $`\alpha(x,t) = 1 + \lambda_{1}\mathcal{\ S}(x,t) + \lambda_{2}\mathcal{\ T}(x,t),`$

donde $`S`$ es densidad superficial suavizada (indicador de orden) y $`\mathcal{T}`$ una medida de turbulencia/varianza (orden inverso). Elegir $`\lambda_{1,\ 2}`$ para que $`\in \lbrack 0.8,3.0\rbrack`$.

**10.4 Numérica y estabilidad**

- **Verificaciones de conservación.** Con actualizaciones re-temporizadas, asegurar que las aproximaciones de simplecticidad se mantienen: monitorear derivas de energía y momento angular vs. $`dt`$ y el **gradiente espacial** de $`dt'`$.

- **Condición tipo Courant para re-temporización.** Imponer $`\mid \nabla\ \ln\ dt' \mid \lesssim 0.5`$ por celda para evitar cizalla en el paso temporal; de lo contrario subciclar.

- **Acoplamiento rejilla–partícula.** Cuando se usan rejillas (S2), calcular $`L(x)`$ desde tamaño de celda o un mapa estructural proporcionado por el usuario; suavizar $`\alpha`$ para evitar oscilaciones.

**10.5** $`\mathbf{\alpha}`$ **acoplado a estructura (auto-actualizable)**

Para emular retroalimentación entre orden y coherencia, actualizar $`\alpha`$ cada $`N`$ pasos:

``` math
\alpha^{(n + 1)} = (1 - \eta)\alpha^{(n)} + \eta\left\lbrack 1 + \lambda_{1}\widetilde{\Sigma} + \lambda_{2}\left( 1 - \widetilde{E} \right) \right\rbrack
```

donde $`\widetilde{\Sigma}`$ es densidad superficial normalizada y $`\widetilde{E}`$ un indicador local de entropía multiescala calculado desde la distribución de partículas; $`0 < \eta \leq 0.2`$ controla la suavidad de actualización. Esto permite que barras/cúmulos **eleven** $`\alpha`$ localmente mientras estallidos/turbulencia pueden **bajarlo**.

**10.6 Tubería de observación simulada**

Para cada instantánea:

1.  Proyectar al cielo con inclinación/PA, distancia; aplicar PSF y haz.

2.  Añadir ruido Gaussiano coincidiendo S/N del sondeo; incluir difuminado de haz y dispersión instrumental.

3.  Extraer perfiles de rotación/dispersión exactamente como en Sec. 5 (mismos anillos, mismas correcciones).

4.  Construir mapas de estructura (entropía, modos, aglomeración) y recuperar $`\widehat{\alpha}(L)`$ vía el **mismo** mapa de indicadores usado en datos reales.

5.  Ejecutar pruebas de pendiente y colapso; calcular residuos bTFR y diagnósticos irrelevantes para lentes.

Esto asegura comparabilidad **de principio a fin** y expone sesgos de medición, no solo física.

**10.7 Pruebas de recuperación de parámetros**

**Objetivo.** Verificar que la tubería recupera la **verdad** $`\alpha_{verdadero}`$, pendientes, colapsos) dentro de tolerancia.

- Métrica de recuperación: $`{\Delta\alpha(L) = \widehat{\alpha}(L) - \alpha}_{verdadero}(L)`$; reportar mediana y dispersión 68% por grupo.

- **Tolerancia:** mediana $`\mid \Delta\alpha \mid \leq 0.2`$ y residuos de pendiente $`{\mid m - (1 - \alpha}_{verdadero}) \mid \leq 0.1.`$

- **Curvas de sensibilidad:** variar FWHM de PSF, S/N, inclinación, haz, y ancho de anillo para mapear regiones donde la recuperación se vuelve **sesgada** o **inestable**.

- **Casos adversariales:** escalones bruscos de $`\alpha`$ dentro de un grupo; flujos no circulares fuertes; discos alabeados; $`\alpha(\phi)`$ anisotrópico. Registrar con qué frecuencia el colapso falla cuando $`\alpha`$ varía dentro de un grupo—esto establece **reglas de agrupamiento**.

**10.9 Discriminantes contra DM/MOND in silico**

- **Prueba de degeneración de halo.** Ajustar halos DM estándar a las **mismas** curvas simuladas; mostrar que muchos halos ajustan $`v(R)`$, pero **ninguno** reproduce **colapsos** por grupo ligados al campo $`\mathbf{\alpha}`$ **conocido** (firma única de RTM).

- **Clasificador MOND.** Generar simulaciones donde $`m = 0`$ externo pero pendientes **internas** siguen $`\alpha`$ impuesto; confirmar que una ley de aceleración simple tipo MOND no puede producir las **correlaciones estructura–pendiente** observadas a mapas de bariones fijos.

**10.10 Pruebas de estrés y casos límite**

- **Nulos Keplerianos.** Análogos de binarias amplias: establecer $`\alpha \rightarrow 1`$ y estructura despreciable; confirmar pendientes clásicas y que la estimación de $`\widehat{\alpha}`$ revierte a unidad.

- **Discos ultradifusos.** $`\alpha \simeq 1`$ global con turbulencia irregular; probar tasa de falsos positivos para $`\alpha > 1`$ espurio debido a ruido.

- **Trampas de alto** $`\mathbf{\alpha}`$**.** Bolsas de $`\alpha`$ muy grande (régimen sobre-restringido) pueden congelar evolución local; verificar que la tubería marca grupos que no colapsan (modo de fallo del modelo, no éxito).

**10.11 Entregables**

- **Código abierto**: integradores conscientes de $`\alpha`$ (S1–S4), módulos de actualización de α, herramientas de observación simulada, y cuadernos de análisis; contenedores con versiones fijadas.

- **Catálogos simulados**: verdad por anillo ($`\alpha_{verdadero},\ v,\sigma`$), valores observados (con ruido), $`\widehat{\alpha}`$ recuperado, pendientes, métricas de colapso, banderas CC.

- **Tablas de sesgo**: funciones para sesgos inducidos por haz/inclinación/indicador y umbrales de exclusión recomendados.

**10.12 Criterios de éxito (para la suite de simulación)**

- La tubería **recupera** $`\alpha`$ y pendientes dentro de tolerancia a través de regímenes realistas de S/N y resolución.

- Los **colapsos** $`{v\ R}^{\widehat{\alpha} - 1}`$ son planos en grupos donde $`\alpha`$ es verdaderamente constante; fallan donde $`\alpha`$ varía—diagnóstico, no error.

- Los discriminantes distintivos de RTM (colapso por grupo; acoplamiento estructura–pendiente) **sobreviven** efectos de observación simulada, mientras que las líneas base DM/MOND **no** pueden reproducirlos sin parámetros ad hoc ligados a estructura.

**Resultado.** Con estas simulaciones (i) validamos que **relojes re-temporizados** solos pueden reproducir la fenomenología de pendiente/colapso bajo campos de $`\alpha`$ controlados; (ii) cuantificamos dónde la tubería de datos es **confiable** o **sesgada**; y (iii) producimos **puntos de referencia simulados** públicos para que grupos independientes puedan intentar **recuperación ciega** de $`\alpha`$ y desafiar RTM en terreno neutral.

**11. Discriminantes vs. Materia Oscura y MOND**

Este capítulo enumera **pruebas decisivas y pre-registradas** que separan RTM de (i) **GR+bariones+halos DM** y (ii) **dinámica modificada tipo MOND**. Nos enfocamos en cantidades donde RTM hace afirmaciones a **nivel de pendiente** o **nivel de colapso** que las líneas base no predicen **sin ajuste ad hoc** ligado a estructura bariónica.

**11.1 Qué predice realmente cada marco**

- **RTM (este trabajo):** Dentro de anillos de coherencia fija,

``` math
\frac{\partial\log v}{\partial\log L} = 1 - \alpha,\quad vL^{\alpha - 1}\text{ es plano (colapso)}
```

> Los residuos de escalados globales (p.ej., bTFR) correlacionan con $`\alpha`$ **derivado de estructura**, no con parámetros de masa oculta.

- **GR + halos DM:** Reproduce casi cualquier **forma** de $`v(L)`$ ajustando concentración de halo/tamaño de núcleo y acoplamiento barión-halo. **No** predice genéricamente **colapsos** por anillo ligados a **textura** medida independientemente a menos que los parámetros de halo estén **forzados** a covariar con esas texturas.

- **MOND/leyes de aceleración:** Predice una relación específica entre **aceleración** y **velocidad** (p.ej., $`v^{4}{\propto GM}_{\alpha_{0}}`$ en el régimen profundo); puede ajustar planos externos y relaciones tipo Tully–Fisher. **No** predice acoplamiento **estructura–pendiente** a bariones fijos, ni colapsos por grupo condicionados sobre indicadores de coherencia.

**11.2 Clasificador de pendiente de rotación (por grupo)**

**Prueba D-R1 (Identidad de pendiente).** Para cada grupo de coherencia $`B`$,

``` math
m_{B}\frac{\partial\ log\ v}{\partial\ log\ L}? = {1 - \widehat{\alpha}}_{B}
```

- **RTM PASA:** la identidad se cumple dentro de $`\pm 0.2`$ y **colapso** pasa (∣$`m_{c}`$​∣≤0.1).

- **DM/MOND:** puede coincidir **ya sea** pendiente **o** colapso por grupo con ajuste, pero no puede predecir **ambos** a través de grupos **desde** $`\widehat{\mathbf{\alpha}}`$ **independiente** sin incorporar $`\widehat{\alpha}`$ en la ley de masa/aceleración.

**Regla de decisión:** Si $`\geq 70\%`$ de grupos a través de la muestra satisfacen pendiente+colapso **usando** $`\widehat{\mathbf{\alpha}}`$ **solo desde estructura**, clasificar **RTM-favorecido**.

**11.3 Acoplamiento estructura–pendiente vs. parámetros ocultos**

**Prueba D-R2 (Correlación parcial).** Regresar residuos de pendiente interna $`\Delta m`$ sobre:

- **(A)** indicadores de $`\widehat{\alpha}`$ (potencia de barra, entropía multiescala, aglomeración),

- **(B)** parámetros de halo DM (concentración $`c`$, tamaño de núcleo $`r_{c}`$),

- **(C)** indicadores MOND (aceleración en radio de muestreo, elección de función $`\mu`$).

**Predicción RTM:** $`r`$-parcial significativo para conjunto (A), pero **no** para (B) una vez que bariones están fijos; (C) débil/ausente después de controlar por $`\widehat{\alpha}`$.

**Clasificador:** Si $`{Adj\ R}_{A}^{2} - {Adj\ R}_{B,C}^{2} \geq 0.1`$ a través de la muestra, contar **victoria RTM**.

**11.4 Anatomía de residuos bTFR**

**Prueba D-TF1 (Vínculo residuo–coherencia).** Con $`v`$ medida a un radio fiducial fijo $`R_{f}`$:

- Regresar $`\Delta\ log\ v`$ sobre $`\delta_{\alpha} \equiv \widehat{\alpha}(R_{f}) - 1`$ controlando por tamaño/densidad superficial.

- Re-medir $`v`$ en el **radio de pendiente cero** $`R_{0}`$ por galaxia (donde $`m \simeq 0`$ en un grupo de bajo $`\alpha`$) y repetir.

**Predicción RTM:** Correlación fuerte en $`R_{f}`$, correlación **que desaparece** en $`R_{0}`$

**Predicción DM:** Los residuos correlacionan con $`c`$/espín de halo, no necesariamente con $`\delta_{\alpha}`$; la correlación **no** desaparece en $`R_{0}`$ a menos que los parámetros se reajusten.

**Predicción MOND:** Los residuos están ligados a muestreo de aceleración; ningún papel especial para $`\delta_{\alpha}`$ o $`R_{0}`$

**11.5 Colapso por grupo como restricción funcional**

**Prueba D-C1 (Colapso funcional).** En cada grupo, ajustar la pendiente residual de

``` math
{y(L) = v(L)\ L}^{{\widehat{\alpha}}_{B} - 1}
```

- **RTM:** meta-pendiente agrupada $`\overline{m}`$ entre grupos $`\approx 0`$, heterogeneidad $`I^{2}`$ pequeña.

- **DM/MOND:** Sin razón para $`\overline{m} \rightarrow 0`$ **condicionado** sobre $`\widehat{\alpha}`$ a menos que los parámetros ocultos estén ajustados para **rastrear** indicadores de estructura—una suposición añadida que probamos directamente (abajo).

**Verificación anti-trampa (D-C1b).** Forzar que los parámetros de halo sean funciones explícitas de los mismos indicadores usados para construir $`\widehat{\alpha}`$; medir si esta **imitación** también reproduce **D-R1** (identidad de pendiente) y **D-TF1** (desaparición de residuos en $`R_{0}`$) sin sobreajustar (validación cruzada entre galaxias). Si no, **RTM gana** por parsimonia.

**11.6 Elípticas y perfiles de dispersión**

**Prueba D-E1 (Identidad de pendiente de dispersión).** En grupos de coherencia de elípticas,

``` math
\frac{\partial\ log\ \sigma}{\partial\ log\ L}? = 1 - \widehat{\alpha}
```

- **RTM:** identidad + colapso de $`{\sigma r}^{\widehat{\alpha} - 1}`$

- **DM/MOND:** requieren ajustes de anisotropía y perfil de masa no relacionados con estructura medida; no predicen vínculo **directo** a mapas de $`\widehat{\alpha}`$.

**Clasificador:** Contar tasa de PASA por grupo; $`> 60\%`$ a través de la muestra elíptica marca **RTM-favorecido**.

**11.7 Verificaciones cruzadas lentes–cinemática (resumen como discriminantes)**

- **Anillos/cuádruples de lente fuerte (D-L1):** Después de reinterpretación RTM de cinemática estelar/gas con $`\widehat{\alpha}`$, la masa encerrada en $`R_{E}`$ debe coincidir con lentes dentro de $`\leq 15\%`$. Desplazamientos sistemáticos **después** del condicionamiento sobre $`\alpha`$ desfavorecen RTM en escalas galácticas.

- **Pilas de lentes débiles (D-L2):** A bariones fijos, los perfiles de cizalla **no** se dividen por clase de coherencia, pero las pendientes cinemáticas **sí**; si la cizalla se divide por coherencia, esto sugiere que masa real covaría con estructura → **límite de alcance** para RTM.

**11.8 Puntuador de tres vías y superficie de decisión**

Definimos un **triplete de puntuación** por galaxia (o por tipo de grupo):

- $`S_{RTM} \in \lbrack 0,1\rbrack`$: fracción de pruebas (D-R1, D-C1, D-TF1, D-E1, D-L1/L2 cuando estén disponibles) que **PASAN**.

- $`S_{DM} \in \lbrack 0,1\rbrack`$: fracción de pruebas mejor explicadas por ajustes de halo **sin** usar indicadores de estructura (o requiriéndolos solo post hoc).

- $`S_{MOND} \in \lbrack 0,1\rbrack`$: fracción explicada solo por escalados de aceleración.

**Superficie de decisión:**

- **RTM soportado** si $`S_{RTM} - \max(S_{DM},\ S_{MOND}) \geq 0.2`$ a través de la muestra (con IC bootstrap \> 0).

- **Indeterminado** si diferencias \< 0.2.

- **RTM desfavorecido** si $`S_{RTM} \leq \max(S_{DM},\ S_{MOND}) - 0.2`$

Reportamos estos con incertidumbres y realizamos sensibilidad **dejar-un-indicador-fuera** para asegurar que la ventaja de RTM no está impulsada por una sola característica frágil.

**11.9 Casos límite donde los discriminantes se difuminan**

- **S0/Sa muy suaves con textura mínima:** $`\widehat{\alpha}`$→1 globalmente; todos los modelos predicen pendientes externas casi planas. Los discriminantes se desplazan a **desaparición de residuos bTFR en** $`\mathbf{R}_{\mathbf{0}}`$ y verificaciones de **colapso**.

- **Discos altamente alabeados o fuertemente no axisimétricos:** el análisis sectorial reemplaza los anillos circulares; las predicciones RTM aún se cumplen **por sector**, pero los ajustes DM/MOND ganan margen de maniobra extra. Tratamos estos como **TENTATIVO** a menos que los colapsos sectoriales tengan éxito.

- **Regímenes dominados por cúmulos:** las lentes demandarán masa extra; RTM queda **fuera de alcance** (no intenta arreglar presupuestos de masa de cúmulos).

**11.10 Guía práctica para lectores y árbitros**

1.  **Buscar pendientes y colapsos, no solo ajustes.** Un modelo que ajusta una curva no es suficiente; RTM afirma **identidades** (pendiente $`= \ 1 - \alpha`$) y **planitud** después del reescalado.

2.  **Exigir independencia de** $`\widehat{\mathbf{\alpha}}`$**.** Si un modelo de comparación toma prestados los mismos indicadores de estructura para ajustar sus parámetros libres, requerir validación **retenida** entre galaxias.

3.  **Confiar en las lentes como guardacarril.** Si la cinemática RTM contradice lentes después del condicionamiento sobre $`\alpha`$, la contradicción es real—contar esto contra RTM, no contra curvatura.

**11.11 Conclusión**

RTM compite en **parsimonia** y **estructura predictiva**: una vez que $`\widehat{\alpha}(L)`$ se mide desde **luz/textura**, hace afirmaciones de **pendiente y colapso por grupo** sin **masa libre adicional**. DM y MOND pueden ajustar muchas formas pero carecen de estos **invariantes condicionados por estructura**. Si los datos pasan las pruebas de pendiente/colapso de RTM, muestran residuos bTFR que **desaparecen** en el radio de pendiente cero, y permanecen **consistentes con lentes**, RTM gana poder explicativo en **escalas galácticas**. Si no, los discriminantes aquí proporcionan un camino principiado y cuantitativo para decir **dónde termina RTM**—y por qué.

**12. Falsificación y Condiciones de Alcance**

Este capítulo declara—por adelantado—**cómo puede fallar RTM** en escalas galácticas y **dónde no debería aplicarse**. El objetivo es hacer el programa *decidible*: un lector debería poder ejecutar la tubería y concluir **soportado**, **limitado**, o **falsificado** sin margen de interpretación.

**12.1 Qué cuenta como falsificación (por galaxia, por grupo)**

Un grupo de coherencia $`B`$ (anillos adyacentes con $`\widehat{\alpha}`$ similar) produce **RTM FALLA** si **cualquiera** de los siguientes se cumple bajo buen control de calidad (Sec. 5 y 9):

1.  **La identidad de pendiente falla:** La pendiente EIV robusta $`m_{B} = \partial\ \log\ v/\partial l\ og\ L`$ **no** satisface

``` math
m_{B} = 1 - {\widehat{\alpha}}_{B}
```

dentro de ±0.2 **y** los ICs al 95% no se solapan.

2.  **El colapso falla:** Después de reescalar con el $`{\widehat{\alpha}}_{B}`$ derivado de pendiente

``` math
{y(L) = v(L)L}^{{\widehat{\alpha}}_{B} - 1}
```

tiene una pendiente residual log–log ∣ $`m_{c}`$∣\>0.1 con IC excluyendo cero.

3.  **Desacuerdo de indicadores:** El $`\widehat{\alpha}`$ basado en indicadores y el $`{\widehat{\alpha}}_{B}`$ derivado de pendiente difieren por $`> 0.4`$ sin evidencia de deriva de $`\alpha`$ interna al grupo (es decir, el desacuerdo no se explica por heterogeneidad del grupo).

Una galaxia es **RTM FALLA** si $`\geq 2`$ grupos fallan (o el único grupo usable falla) mientras el control de calidad pasa (verificaciones de resolución, difuminado de haz, inclinación, y deriva asimétrica).

**12.2 Qué cuenta como soporte (por galaxia, por muestra)**

**Por galaxia:** **RTM SOPORTADO** si ≥2 grupos **PASAN** tanto (i) identidad de pendiente (±0.2 con solapamiento IC) **y** (ii) planitud de colapso $`(|m_{c}|\  \leq \ 0.1`$ con IC incluyendo 0), sin banderas CC severas. Un soporte **PARCIAL** requiere al menos pendiente PASA con colapso **TENTATIVO**, o viceversa, y sin banderas rojas de CC.

**A través de la muestra:** RTM está **soportado** en escalas galácticas si:

- ≥70% de todos los grupos evaluados **PASAN** pendiente+colapso;

- **Acoplamiento estructura–pendiente** (Sec. 6, D2) es significativo después de controles de masa/tamaño;

- **Correlación residuo bTFR–**$`\mathbf{\delta}_{\mathbf{\alpha}}`$ está presente a radio fijo y desaparece en el radio de pendiente cero (Sec. 6, D3);

- **Verificaciones lentes–cinemática** pasan a tolerancia ≤15% donde apliquen (Sec. 7).

El fallo de cualquier **dos** de los cuatro criterios entre galaxias bajo buen control de calidad constituye **RTM DESFAVORECIDO** en escalas galácticas.

**12.3 Condiciones de alcance (dónde RTM debería/no debería usarse)**

**Régimen válido (alcance previsto):**

- Dinámica estelar/gas a **escala galáctica** donde una sola **longitud dominante** por anillo es definible e indicadores de **coherencia** estructural son medibles (barras, espirales, cúmulos, espesor, textura cinemática).

- **Pruebas de baja curvatura:** RTM solo re-temporiza **relojes orbitales/de relajación**; no altera curvatura del espacio-tiempo.

**Regímenes fuera de alcance o precaución:**

- **Escalas de cúmulos:** presupuestos de masa de lentes fuertes/débiles + rayos X que exceden bariones; RTM *no* se espera que remueva estas brechas.

- **Flujos relativistas/campos fuertes:** cerca de SMBHs o en jets donde la dilatación temporal GR domina; la re-temporización por $`\alpha`$ no es sustituto para GR.

- $`\mathbf{\alpha}`$ **rápidamente variable, no axisimétrico:** grupos con fuerte anisotropía azimutal o $`\nabla\alpha`$ pronunciado dentro del grupo (se requiere análisis sectorial; por defecto a **TENTATIVO**).

- **Datos de pobre resolución:** PSF/haz tan grande que los anillos tienen \<3 elementos de resolución, o incertidumbres de inclinación/PA dominan errores de pendiente.

**12.4 Taxonomía de fallo (qué significa un fallo y qué hacer)**

- **Tipo A — Desajuste de pendiente, buen colapso.**\
  *Interpretación:* indicadores de $`\widehat{\alpha}`$ están mal calibrados; el entorno es coherente, pero el mapeo estructura→α está mal.\
  *Acción:* Reajustar mapa de indicadores solo en **galaxias de calibración**; **no** reclamar RTM hasta que la identidad de pendiente se cumpla con mapas revisados.

- **Tipo B — Fallo de colapso, identidad de pendiente se cumple.**\
  *Interpretación:* $`\alpha`$ varía dentro del grupo o las correcciones de geometría están incompletas.\
  *Acción:* Estrechar grupos, adoptar análisis **sectorial**, o mejorar correcciones de haz/alabeo.

- **Tipo C — Tanto pendiente como colapso fallan.**\
  *Interpretación:* RTM no describe la dinámica en ese régimen (falsificación verdadera) o el control de calidad es inadecuado.\
  *Acción:* Si el CC pasa, registrar como **grupo falsificado**; reclasificar galaxia si múltiples grupos fallan.

- **Tipo D — Inconsistencia de lentes.**\
  *Interpretación:* La reinterpretación RTM de cinemática contradice masa basada en curvatura.\
  *Acción:* Contar contra RTM en **escala galáctica**; marcar cúmulos como **fuera de alcance** por diseño.

**12.5 Guardacarriles contra sobreajuste**

- **Mapas congelados.** Los mapeos indicador→$`\alpha`$ están **congelados** antes de analizar objetivos científicos; cualquier ajuste post hoc debe ser re-validado en galaxias **retenidas**.

- **Pruebas retenidas.** El acoplamiento estructura–pendiente y las verificaciones de colapso deben replicarse en un subconjunto retenido con umbrales idénticos.

- **Anti-fuga.** $`\widehat{\alpha}`$ **no** puede inferirse de la cinemática misma en el análisis principal (sin circularidad); debe venir de mapas de **luz/textura**.

**12.6 Controles negativos y expectativas nulas**

- **Regímenes tipo Kepleriano:** binarias amplias, planetas externos, afueras de globulares—RTM debe revertir a pendientes clásicas; cualquier desviación indica error de tubería.

- **Discos S0/Sa sin rasgos:** los indicadores deben producir $\hat{\alpha} \to 1$ globalmente; los grupos externos deben PASAR colapso con $m \approx 0$.

- **Nulos simulados:** conjuntos de datos simulados con $`\alpha \equiv 1`$ en todas partes deben devolver pendientes $`m \approx 0`$ y **ninguna** correlación espuria con métricas de textura.

**12.7 Contingencias si RTM está limitado, no falsificado**

Si RTM pasa pendiente/colapso **solo** para ciertas morfologías o rangos de masa, reportaremos **curvas de alcance**:

- **Alcance de morfología:** fracción de PASA de grupos vs. tipo de Hubble (barrada, sin barra, LSB, enana).

- **Alcance de densidad superficial:** fracción de PASA vs. $`\Sigma_{\star}`$ o fracción de gas.

- **Alcance de corrimiento al rojo:** fracción de PASA vs. tiempo de retrospección (donde existan datos IFU/HI).

Estas curvas son resultados legítimos; delimitan **dónde** importa la re-temporización por coherencia.

**12.8 Resumen de una figura (para árbitros)**

Incluiremos un resumen de una página por muestra:

1.  **Arriba-izquierda:** distribuciones de $`\widehat{\alpha}(R)`$ entre galaxias.

2.  **Arriba-derecha:** Gráfico de identidad de pendiente por grupo mmm vs. $`1 - \widehat{\alpha}`$ con línea 1:1 (color = estado de CC).

3.  **Abajo-izquierda:** Distribución de meta-pendiente de colapso (debe tener pico en 0).

4.  **Abajo-derecha:** Residuos lentes–cinemática (donde estén disponibles) y relación residuo bTFR–$`\delta_{\alpha}`$ en $`R_{f}`$ y en $`R_{0}`$

Un lector puede juzgar **de un vistazo** si RTM se cumple, está limitado, o falla.

**12.9 Conclusión**

RTM será declarado **soportado** solo si **identidades de pendiente** y **colapsos** se cumplen grupo por grupo con $`\widehat{\alpha}`$ medido **independientemente** de estructura, y si las **lentes** permanecen consistentes a escalas galácticas. Está **falsificado** si pendientes y colapsos fallan ampliamente bajo buen control de calidad o si las brechas lentes–cinemática persisten **después** del condicionamiento sobre $`\alpha`$. Está **limitado** si el éxito se localiza a morfologías o entornos específicos. Este capítulo hace esos resultados **pre-registrados e inequívocos**—para que la comunidad pueda decidir, no solo ajustar.

**13. Discusión**

Esta sección sintetiza qué significaría la **Astronomía Rítmica** si las pruebas pre-registradas **pasan**, cómo interpretar resultados **mixtos**, y qué nos enseña un **fallo**. Cerramos mapeando los pasos de decisión más importantes y clarificando límites conceptuales.

**13.1 Si el programa pendiente–colapso pasa**

Un hallazgo consistente de que, dentro de anillos de coherencia fija,

``` math
\frac{\partial\log v}{\partial\log L} = 1 - \widehat{\alpha}\quad\text{y}\quad vL^{\widehat{\alpha} - 1} \approx \text{const}
```

establecería que los **relojes cinemáticos** de una galaxia están co-gobernados por un **campo organizacional** $`\alpha(L)`$ medible solo desde *estructura bariónica*. Los beneficios prácticos son inmediatos:

- **Diversidad predictiva.** Las formas de curva interna a masa fija dejan de ser dispersión de molestia; se convierten en *varianza predicha* una vez que $`\widehat{\alpha}`$ se mapea desde barras, espirales, cúmulos, espesor, y textura cinemática.

- **Anatomía bTFR clarificada.** Los residuos a $`M_{b}`$ fijo heredan una geometría simple: medir en el radio de pendiente cero (donde $`\widehat{\alpha} \rightarrow 1`$) y la relación se ajusta; muestrear dentro de zonas coherentes y aparece un sesgo predecible.

- **Parsimonia vs. ajuste post-hoc.** Los halos de materia oscura (o interpolaciones MOND) pueden ajustar muchas formas, pero no **a priori** ligan *colapsos funcionales* por anillo a textura medida independientemente. RTM añadiría una restricción estructural faltante.

**13.2 Si vemos soporte parcial**

Un patrón común que anticipamos es **coincidencias de pendiente** con **colapsos imperfectos** en grupos donde $`\alpha`$ deriva a través del anillo o sistemáticos de geometría (haz, inclinación, alabeos) permanecen. Esto no es una trivialidad; es diagnóstico:

- **Qué ajustar.** Estrechar grupos, adoptar análisis sectorial, o mejorar correcciones de haz/alabeo. Reverificar con mapas de $`\widehat{\alpha}`$ dejar-un-indicador-fuera.

- **Qué reportar.** Llamar a estos **PARCIAL** por diseño (Sección 12), y publicar los modos de fallo. Un campo aprende más rápido de "casi" limpios que de victorias ambiguas.

**13.3 Si el programa falla limpiamente**

Si (i) las pendientes no igualan $`1 - \widehat{\alpha}`$, (ii) los colapsos muestran inclinación residual significativa, y (iii) los residuos bTFR ignoran $`\delta_{\alpha}`$ **después del control de calidad**, entonces **RTM no es la abstracción correcta para la cinemática galáctica**. Esto sigue siendo valioso:

- **Límite aprendido.** La re-temporización por coherencia puede ser poderosa en sistemas de laboratorio (química, redes), pero insuficiente para flujos auto-gravitantes una vez que la curvatura y la geometría tridimensional dominan.

- **Disciplina reutilizable.** Las verificaciones de pendiente primero + colapso, congelamiento de indicadores, y pre-registro permanecen como plantilla para otras hipótesis conscientes de estructura en astronomía.

**13.4 Clarificaciones conceptuales (qué es RTM—y qué no es)**

- **No una nueva fuerza ni masa oculta.** Las fuerzas y la curvatura permanecen GR; RTM re-temporiza procesos **operacionales** embebidos en medios estructurados.

- **Sin almuerzo gratis en cúmulos.** Donde las lentes demandan masa más allá de bariones (cúmulos ricos), RTM está fuera de alcance a menos que se acompañe de materia adicional genuina.

- **Sin circularidad.** $`\widehat{\alpha}`$ viene de **luz/textura**, no de cinemática; las pendientes/colapsos entonces se predicen, no se ajustan.

**13.5 Relación con dinámica clásica de disco**

RTM no reemplaza el análisis de Jeans; lo **aumenta** con una restricción sobre cómo las *escalas de tiempo* varían con escala cuando el medio está organizado jerárquicamente. En la práctica:

- Tratar $`\widehat{\alpha}`$ como un **campo de hiperparámetro** que regulariza modelos dinámicos: priors sobre comportamiento de pendiente permitido por anillo.

- Usar RTM para **elegir radios** para escalados globales (p.ej., donde $`\widehat{\alpha} \rightarrow 1`$ para bTFR), reduciendo sistemáticos entre muestras.

**13.6 Fuentes de falsos positivos y cómo nos protegimos contra ellos**

- **Difuminado de haz / errores de inclinación.** Estos aplanan pendientes pero no inducen genéricamente **colapsos por grupo** después del reescalado $`L^{\widehat{\alpha} - 1}`$; nuestras correcciones EIV y umbrales de CC abordan esto.

- **Flujos no circulares.** Barras y alabeos complican $`v(R)`$. Manejamos esto con análisis sectorial e incluyendo **textura cinemática** como indicador negativo en $`\widehat{\alpha}`$.

- **Fuga de indicadores.** Si los indicadores accidentalmente codifican cinemática (p.ej., usando campos de velocidad), aparece circularidad. Separamos estrictamente entradas de **estructura** de salidas de **dinámica** (Sec. 5, 9).

**13.7 Qué** ***significa* físicamente** **un** $`\mathbf{\alpha}`$ **medido**

A través del corpus RTM, mayor $`\alpha`$ refleja mayor **persistencia** y **jerarquía**: tiempos de permanencia más largos, menos caminos efectivos, mezcla más lenta. En discos, eso se traduce en:

- **Barras/bulbos/cúmulos internos:** $`\alpha`$ elevado → relojes orbitales locales más lentos → ascensos internos más pronunciados o aplanamiento retardado.

- **Afueras difusas:** $`\alpha`$ → 1 → asíntotas planas sin invocar masa extra *si* la curvatura no necesita aumentar (lentes consistentes es el guardacarril).

Esta es una imagen unificadora: **diseñar tiempo** diseñando **estructura**.

**13.8 Intersecciones con retroalimentación y turbulencia**

El enfriamiento, la retroalimentación, y la turbulencia ya dan forma a la estructura del disco. RTM postula que su **resultado organizacional neto**—no cada detalle microfísico—entra en la dinámica principalmente a través de $`\alpha`$:

- **Retroalimentación que fragmenta orden** impulsa $`\alpha`$ ↓ (discos externos se asientan más rápido, cúmulos mueren antes).

- **Características coherentes de larga vida** (barras, anillos) impulsan $`\alpha`$ ↑ (relojes internos se ralentizan, diversidad aumenta).\
  Esto proporciona una **estadística resumen** para modelos subrejilla en simulaciones: en lugar de ajustar muchas perillas, ajustar cómo **desplazan** $`\mathbf{\alpha}`$.

**13.9 ¿Qué convencería a un escéptico?**

Tres gráficos:

1.  **Identidad de pendiente:** puntos de mmm medida vs. $`1 - \widehat{\alpha}`$ abrazando la línea 1:1 a través de muchas galaxias.

2.  **Colapso funcional:** distribuciones de pendientes residuales por grupo centradas en 0 con ICs ajustados.

3.  **Armonía de lentes:** $`M_{cin}^{RTM}`$ coincidiendo con $`M_{lentes}`$ a escalas galácticas mientras las brechas de cúmulos permanecen.

Si estos se replican con indicadores congelados y muestras retenidas, RTM pasa el listón.

**13.10 Próximas decisiones (qué haríamos *después* de primeros resultados)**

- **Si PASA:** Expandir a sondeos ricos en IFU, publicar mapas abiertos de $`\widehat{\alpha}`$, y presionar sobre **evolución** (cómo los campos de $`\alpha`$ se aplanan con tiempo cósmico). Explorar predicciones condicionadas por simetría (p.ej., fase de barra vs. $`\nabla\alpha`$).

- **Si PARCIAL:** Enfocarse en sectores y estructura vertical; refinar definiciones de grupos; probar regímenes de enanas/LSB donde $`\alpha`$ está cerca de unidad para aislar asíntotas limpias.

- **Si FALLA:** Publicar el resultado negativo con el pre-registro completo, luego reutilizar la tubería como **arnés de consistencia** para cualquier propuesta futura consciente de estructura.

**13.11 Significado más amplio**

Independientemente del resultado, este trabajo trae una metodología de **grado de laboratorio**—inferencia de pendiente primero, verificaciones de colapso, umbrales pre-registrados—a la astronomía extragaláctica. La idea de que la **organización controla los relojes** es ya sea un unificador poderoso (si se soporta) o un callejón sin salida claramente circunscrito (si se falsifica). En ambos casos, el campo gana: ya sea un nuevo eje (coherencia) en sus relaciones de escalado o una comprensión más aguda de por qué **masa** y **curvatura** solas deben seguir llevando la carga.

**14. Conclusiones y Perspectivas**

La **Astronomía Rítmica** avanza un relato falsificable, de pendiente primero de la dinámica galáctica: una vez que un **campo de coherencia** $`\alpha(L)`$ se mide solo desde estructura bariónica, los relojes orbitales obedecen

``` math
v(L) = \kappa L^{1 - \alpha(L)}\quad \Rightarrow \quad\frac{\partial\log v}{\partial\log L} = 1 - \alpha/2,
```

y **colapsos por grupo** $`{v\, L}^{\alpha - 1} \approx const`$ deben aparecer cuando α es localmente constante. A diferencia de parametrizaciones de materia oscura o modificaciones de ley de aceleración, RTM predice **identidades funcionales por grupo** condicionadas sobre estructura medida independientemente, y mantiene **curvatura** (lentes) en GR estándar.

**14.1 Qué contaría como éxito**

- **Pendientes de rotación/dispersión** coinciden con $`1 - \widehat{\alpha}`$ a través de anillos agrupados por coherencia con ICs pequeños.

- Los **colapsos** son planos dentro de grupos después del reescalado $`L^{\widehat{\alpha} - 1}`$.

- Los **residuos bTFR** correlacionan con $`\delta_{\alpha}`$ a radio de muestreo fijo y **desaparecen** en el radio de pendiente cero.

- La **reconciliación lentes–cinemática** se cumple a ≤15% a escala galáctica, mientras que los cúmulos permanecen como límite de alcance.

Si estos se replican con **mapas de indicadores congelados**, muestras retenidas, y cuadernos abiertos, RTM gana un lugar junto al modelado de masa como una **ley de temporización condicionada por estructura** para galaxias.

**14.2 Qué aprendimos incluso si los resultados son mixtos**

- La disciplina de **pendiente/colapso** separa geometría/sistemáticos de regularidades dinámicas verdaderas.

- Resultados negativos o parciales **agudizan límites**: donde $`\alpha`$ no puede estimarse establemente, o donde lentes demandan masa independientemente de coherencia, RTM está **limitado**.

**14.3 Próximos pasos inmediatos (90–180 días)**

1.  **Conjunto de calibración** (∼20 galaxias): congelar mapas característica→$`\alpha`$; publicar pre-registro.

2.  **Muestra de prueba central** (∼150 discos + 40 elípticas): ejecutar pendiente/colapso por grupo; liberar catálogos por anillo y banderas de CC.

3.  **Verificaciones cruzadas de lentes**: 10–15 lentes fuertes con IFU; divisiones de lentes débiles apiladas por clase de coherencia.

4.  **Puntos de referencia de simulación**: simulaciones públicas conscientes de α con verdad para desafíos de recuperación ciega.

**14.4 Riesgos y mitigaciones**

- **Fragilidad de indicadores** → familias de mapas duales (paramétrico + ensamble de rangos), verificaciones de estabilidad dejar-un-indicador-fuera.

- **Sesgos de haz/inclinación** → correcciones EIV, umbrales de resolución, análisis sectorial para casos alabeados/no circulares.

- **P-hacking** → umbrales pre-registrados, replicación retenida, y código/datos públicos.

**14.5 Implicaciones más amplias**

- Si se soporta, $`\alpha`$ se convierte en un **nuevo eje** en relaciones de escalado—vinculando **textura** (barras, espirales, cúmulos, espesor) a **temporización** (pendientes, perfiles de dispersión), y proporcionando un objetivo compacto para modelos subrejilla en simulaciones ("**diseña el tiempo de la galaxia**").

- Si está limitado o falsificado, la comunidad gana una **plantilla transparente** para probar ideas conscientes de estructura sin confundir relojes y curvatura.

**Conclusión.** RTM no reemplaza la gravedad o el modelado de masa bariónica; añade un **reloj condicionado por coherencia** que puede probarse correcto o incorrecto con datos actuales. Las firmas decisivas son **pendientes** y **colapsos** ligados a **estructura medida independientemente**, con **lentes** como guardacarril. Cualquier resultado—soporte o fallo bien documentado—mueve la dinámica extragaláctica hacia adelante con palancas más claras, límites más claros, y un camino reproducible que otros pueden auditar.

**Apéndice A — Derivaciones e Identidades**

**A.1 De la ley de tiempo RTM a las leyes de rotación/dispersión**

RTM postula un **tiempo operacional** para procesos a escala $`L`$

``` math
T(L) = T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha(L)}\Theta
```

donde $`\alpha(L)`$ es el **exponente de coherencia** y $`\Theta`$ es adimensional y se trata como constante **dentro de un grupo de coherencia** (Sec. 5). Para órbitas casi circulares,

``` math
T = \frac{2\pi L}{v}\quad \Rightarrow \quad v(L) = \kappa L^{1 - \alpha(L)/2},\quad\kappa \equiv \frac{2\pi L_{0}}{T_{0}\Theta}
```

Tomando derivadas **dentro de un grupo** donde $`\alpha`$ es aproximadamente constante,
                                                           
 ``` math                                                     
 \frac{\partial\log v}{\partial\log L} = 1 - \alpha\text{/}2  
 ```
(A1)

que es la **ley de pendiente** usada a lo largo del documento.

Para sistemas soportados por dispersión (capa esférica de espesor $`\sim L`$), una velocidad aleatoria característica escala como $`L/T`$, dando

``` math
\left. \ \frac{\partial\log\sigma\ }{\partial\log L\ } \right|_{\text{grupo}} = 1 - \alpha
```
(A2)

$`{\sigma(L)\  \propto \ L}^{1 - \alpha(L)} \Rightarrow`$

**A.2 Verificación de colapso**

Definir la **variable colapsada**

``` math
{y(L) \equiv v(L)\ L}^{\alpha/2 - 1}
```

Si $`\alpha`$ es constante dentro del grupo, entonces $`y(L) = \kappa =`$ constante y


``` math
\left. \ \frac{\partial\log y\ }{\partial\log L\ } \right|_{\text{grupo}} = 0
```
(A3)

La misma forma se cumple para dispersiones con $`{y(L) = \sigma(L)\ L}^{\alpha - 1}`$

**A.3 Movimientos no circulares y sistemáticos geométricos (primer orden)**

Sea $`v_{\text{obs}}^{2} = v_{\phi}^{2} + \delta v_{\text{nc}}^{2}`$ donde $`{\delta v}_{nc}`$ codifica flujo de barra/espiral y correcciones de deriva asimétrica. Si $`{\delta v}_{nc}/v_{\phi}`$ varía lentamente con $`L`$ dentro de un grupo, la pendiente de $`\log_{vobs}`$ versus $`log\ L`$ está perturbada a $`\mathcal{O}\left( \frac{\partial\log\delta v_{\text{nc}}}{\partial\log L} \right)`$ es decir, principalmente un cambio de **intersección**.\
Esto justifica el enfoque de **pendiente primero** y el **refinamiento sectorial** cuando la no circularidad es fuerte.

**A.4 Casos axisimétricos vs. esféricos**

- **Discos delgados.** Usando geometría de anillo inclinado, la escala característica local es el radio del anillo $`L = R`$; los resultados (A1–A3) aplican por anillo.

- **Sistemas esféricos.** Con modelado de Jeans, reemplazar el tiempo dinámico $`t_{din}{\sim (G\rho)}^{- 1/2}`$ por el **operacional** $`{T \propto L}^{\alpha}`$ cambia solo la **tasa** a la que las órbitas mezclan fase; la identidad de pendiente medible (A2) permanece por grupo siempre que la anisotropía varíe lentamente a través del grupo.

**A.5 Cuando** $`\mathbf{\alpha}`$ **varía dentro de un grupo**

Sea $`(L) = \alpha_{B} + \delta\alpha(L)`$ con $`\mid \delta\alpha \mid \ll 1`$ a través del ancho $`\Delta\ \log\ L`$. Entonces

``` math
\frac{\partial\log y}{\partial\log L} = \underset{= 0}{\overset{\left( 1 - \alpha_{B} \right) + \left( \alpha_{B} - 1 \right)}{︸}} - \delta\alpha(L)
```

así que la pendiente residual del colapso es aproximadamente $`{- \langle\ \delta\alpha\rangle}_{B}`$. Este es el diagnóstico usado para ajustar grupos (o sectorizar) hasta que el residuo sea consistente con 0.

**Apéndice B — Construcción de** $`\widehat{\mathbf{\alpha}}`$ **desde Observables**

**Objetivo.** Mapear **indicadores de estructura** multiescala a un exponente de coherencia por anillo $`\widehat{\alpha}`$ con incertidumbre, usando solo **luz/textura** (sin cinemática), luego verificar con pendientes y colapsos.

**B.1 Conjunto de características**

Para cada anillo desproyectado $`A_{j}`$ (Sec. 5):

1.  **Entropía multiescala** $`\mathbf{E}`$**.** Calcular pirámide de ondículas à trous $`I_{s}`$ sobre escalas $`s`$, luego entropía $`H_{s}`$. Definir $`E^{\star} = 1 - zscore(\sum_{s}\ w_{s}\ H_{s})`$. Menor entropía → mayor orden → mayor $`\alpha`$.

2.  **Potencia de modo de Fourier** $`P_{m}`$**.** Desde brillo superficial desproyectado, medir potencia fraccional en modos $`m = 2`$ y $`m = 2 - 4`$ (espiral): $`C_{modo} = \sum_{m \in \{ 2,3,4\}}\ P_{m}`$

3.  **Aglomeración/Suavidad** $`Q`$. Usar CAS o Gini–$`M_{20}`$ para formar $`Q = 1 - S`$ (más suave → más coherente).

4.  **Índice fractal/turbulento** $`D`$ (gas). Pendiente de función de estructura $`\zeta`$ o dimensión fractal $`D`$; convertir a $`C_{D}`$ para que más orden a gran escala ⇒ mayor $`C_{D}`$

5.  **Espesor/Asimetría** $`T`$. Desde indicadores verticales o razones de ejes corregidas; definir $`C_{T}`$ (más delgado/simétrico → mayor $`C_{T}`$).

6.  **Textura cinemática** $`K`$ (indicador negativo). Potencia de flujo no circular desde campos de velocidad residuales; usar $`C_{K} = 1 - NCF`$ cuando esté disponible, u omitir para mapeo puramente fotométrico.

> $`z_{j} = \left\lbrack E^{*},C_{\text{modo}},Q,C_{D},C_{T},C_{K} \right\rbrack`$ con covarianza $`\Sigma_{j}`$

**B.2 Mapeo monótono a** $`\widehat{\mathbf{\alpha}}`$

Dos opciones intercambiables, pre-registradas:

- **Mapeo monótono paramétrico:**

> $`\widetilde{\alpha} = \alpha_{0} + \sum_{k}^{}{w_{k}g_{k}\left( z_{k} \right)},\quad w_{k} \geq 0,g_{k}`$ monótona (identidad/logística). Regularizar con $`\sum_{}^{}w_{k} = 1`$ y prior $`\alpha \in \lbrack 0.8,3.2\rbrack`$

- **Ensamble de rangos:**

$`\widetilde{\alpha} = \alpha_{0} + \lambda\backslash mediana_{k}\ rango\left( z_{k} \right),`$ robusto a valores atípicos y escala.

Las incertidumbres vienen del método delta (paramétrico) o bootstrap (rango).

**B.3 Agrupamiento de coherencia y contracción**

- **Restricción de contigüidad.** Agrupar anillos **adyacentes** por $`\widehat{\alpha}`$ (Ward 1-D), asegurando contigüidad radial.

- **Reconciliación de pendiente.** En cada grupo $`B`$, ajustar $`m_{B}`$ y establecer $`{\widehat{\alpha}}_{B}{= 1 - m}_{B}`$. Contraer $`{\widehat{\alpha}}_{j}`$ por anillo hacia $`{\widehat{\alpha}}_{B}`$ con pesos $`{\propto 1/SE}^{2}`$.

**B.4 Umbrales de control de calidad**

- Resolución: ≥3 elementos de resolución por anillo.

- Corrección de difuminado de haz \<20% (marcar TENTATIVO si 20–35%).

- Robustez de indicadores: desplazamiento dejar-un-indicador-fuera $`\leq 0.2`$ en $`\widehat{\alpha}`$.

- Estacionariedad: pendiente PSD o textura debe ser aproximadamente ley de potencia en banda (rechazar curvatura fuerte).

**Apéndice C — Algoritmos de Simulación Conscientes de** $`\mathbf{\alpha}`$

**C.1 Principio**

Mantener **fuerzas** estándar; aplicar **reescalado temporal** localmente:

``` math
dt'(x) = dt\left( \frac{L(x)}{L_{0}} \right)^{\alpha(x) - \alpha_{0}}
```

Los integradores avanzan estados con $`dt'`$ (re-temporización), no cambiando gravedad.

**C.2 Órbitas sin colisiones (S1)**

- Potencial: disco Miyamoto–Nagai + bulbo Hernquist (opcionalmente añadir NFW para comparaciones base).

- Partículas: $`{N \sim 10}^{6}`$ trazadores; paso leapfrog/simpléctico con $`dt'`$ adaptativo.

- Campos de $`\alpha`$: prominencias radiales analíticas, gradientes, o patrones azimutales $`m = 2`$.

- Salidas: curvas de rotación por sector; pendientes y colapsos por grupo.

**C.3 Disco delgado con respuesta viva (S2)**

- Auto-gravedad en rejilla 2D (solucionador FFT o Poisson en rejilla polar).

- Gas vía esquema de partículas pegajosas para disipación.

- $`\alpha(x,t)`$: fijo o **acoplado a estructura**

- $`\alpha^{n + 1} = (1 - \eta)\alpha^{n} + \eta\left\lbrack 1 + \lambda_{1}\widetilde{\Sigma} + \lambda_{2}\left( 1 - \widetilde{E} \right) \right\rbrack`$

- Diagnósticos: fuerza de barra vs. $`\nabla\alpha`$, tiempos de vida de cúmulos vs. $`\alpha`$ local.

**C.4 Cubos IFU/HI simulados (S3)**

- Proyectar instantáneas con inclinación/PA; construir mapas momento-0/1/2.

- Convolucionar con PSF/haz; añadir ruido; ejecutar la **misma** tubería de extracción de anillos y $`\widehat{\alpha}`$ que para datos reales.

**C.5 Estabilidad y guarda tipo CFL**

- Imponer $`\mid \nabla\ \ln dt' \mid \lesssim 0.5`$ por celda; subciclar de lo contrario.

- Monitorear deriva de energía y momento angular; ajustar dt para que la re-temporización no rompa comportamiento simpléctico.

**C.6 Pruebas de recuperación**

- Tolerancia: mediana $`\mid \widehat{\alpha} - \alpha_{verdadero} \mid \leq 0.2`$; residuo de pendiente $`\mid m - (1 - \alpha_{verdadero}) \mid \leq 0.1`$; meta-pendiente de colapso $`\mid \overline{m} \mid \leq 0.05`$

- Mapas de sesgo vs. PSF, S/N, inclinación, y ancho de grupo; registrar umbrales de exclusión.

**Apéndice D — Plantilla de Pre-registro y Recetas de Figuras**

**D.1 Pre-registro (a publicar antes del análisis)**

**Título:** Astronomía Rítmica: pruebas de pendiente/colapso con anillos condicionados por coherencia.

**Puntos finales primarios:**

- H-RC: En cada grupo, $`m = 1 - \widehat{\alpha}`$ dentro de ±0.2 (solapamiento IC 95%).

- H-CL: En cada grupo, pendiente residual de $`{y = vL}^{\widehat{\alpha} - 1}`$ es $`\mid m_{c} \mid \leq 0.1`$ con IC incluyendo 0.

- H-TF: Residuos bTFR $`\Delta\ log\ v`$ correlacionan con $`\delta_{\alpha}`$ a radio fiducial fijo y **desaparecen** en el radio de pendiente cero.

- H-Lentes (donde aplique): $`\left| M_{\text{cin}}^{\text{RTM}} - M_{\text{lentes}} \right|\text{/}M_{\text{lentes}} \leq 0.15.`$

**Exclusión/CC:**

- PSF/haz \< 0.5 del ancho del anillo; incertidumbre de inclinación \< 5°; corrección de haz \< 35%.

- El anillo debe tener ≥3 elementos de resolución y ≥30 píxeles independientes.

**Mapeo indicador→**$`\mathbf{\alpha}`$**:** fijar coeficientes (paramétrico) y parámetros de ensamble de rangos en el **conjunto de calibración** ($`N \approx 20`$), luego **congelar**.

**Plan estadístico:** Theil–Sen + SIMEX para pendientes; ICs bootstrap (B=2000); meta de efectos aleatorios para pendientes agrupadas; FDR 5%.

**Reglas de fallo:** Como en Sec. 12—dos fallos independientes entre galaxias bajo buen CC → RTM desfavorecido.

**D.2 Figuras canónicas (por galaxia)**

1.  **Mapa de estructura y** $`\widehat{\mathbf{\alpha}}`$**:** imagen desproyectada, paneles de indicadores, y $`\widehat{\alpha}(R)`$ radial con IC.

2.  **Gráfico de pendiente:** $`log\ v`$ vs. $`log\ R`$ coloreado por grupos de coherencia; anotar $`m`$ ajustada y $`1 - \widehat{\alpha}`$

3.  **Paneles de colapso:** $`{vR}^{\widehat{\alpha} - 1}`$ vs. $`R`$ por grupo, con pendiente residual e IC.

4.  **Posición bTFR:** galaxia en el bTFR global; residuo vs. $`\delta_{\alpha}`$

5.  **(Si lente):** $`M_{\text{cin}}^{\text{RTM}}(R)\text{ vs. }M_{\text{lentes}}(R)`$ con residuos.

**D.3 Figuras canónicas (nivel de muestra)**

1.  **Nube de identidad de pendiente:** $`m`$ de todos los grupos vs. $`1 - \widehat{\alpha}`$ con línea 1:1, sombreado de densidad.

2.  **Histograma de meta-pendiente de colapso:** distribución de pendientes residuales por grupo con 0 marcado.

3.  **Anatomía de residuos bTFR:** $`\Delta\ \log\ v`$ vs. $`\delta_{\alpha}`$ en $`R_{f}`$ y en $`R_{0}`$

4.  **Reconciliación de lentes:** dispersión de $`{\Delta M/M}_{lentes}`$ en $`R_{E}`$ (o bandas de perfil) con media ±IC.

5.  **Gráficos de alcance:** fracción de PASA vs. morfología, densidad superficial, corrimiento al rojo.

**APÉNDICE E — Análisis Empírico Robusto: La Base de Datos SPARC y Topología Bariónica**

El marco RTM propone que las curvas de rotación galáctica planas no son causadas por halos invisibles de materia oscura, sino por un cambio macroscópico en la coherencia topológica de la red bariónica ($`\alpha \approx 2`$). Para validar esto, analizamos galaxias de disco de la base de datos SPARC.

**E.1 Observación Heurística y Sesgo de Atenuación**

El análisis OLS inicial fue suprimido por **sesgo de atenuación**. Una vez corregido vía **Regresión de Distancia Ortogonal (ODR)** para absorber 15% de varianza de hardware y observacional, el vínculo verdadero estructura-cinemática se revela como una pendiente más pronunciada y definitiva de $`\mathbf{- 1.169\ }\mathbf{\pm}\mathbf{0.119}`$. Además, las 52 galaxias con curvas de rotación planas produjeron un exponente de coherencia derivado de $`\alpha = \ 1.99`$. Aunque este hallazgo heurístico estuvo notablemente cerca de la predicción teórica de $`\alpha = \ 2`$, depender de OLS de estimación puntual estándar en astrofísica es estadísticamente frágil.

OLS asume que las variables independientes se miden perfectamente. En realidad, los datos SPARC contienen incertidumbre significativa derivada de ángulos de inclinación galáctica, derivas asimétricas, y dispersión de velocidad HI. No propagar este ruido introduce un "sesgo de atenuación" que aplana artificialmente las pendientes de regresión y crea una falsa sensación de precisión en promedios estáticos.

**E.2 Validación Probabilística Rigurosa (ODR y Propagación de Errores)**

Para asegurar que la ley de velocidad RTM representa un mecanismo físico genuino y no una ilusión estadística, el conjunto de datos fue sometido a una tubería estadística de "Equipo Rojo":

1.  **Regresión de Distancia Ortogonal (ODR):** Reemplazamos OLS con un modelo robusto de Errores-en-Variables (EIV) para evaluar el vínculo estructura-cinemática. Inyectamos explícitamente incertidumbres observacionales en el modelo (una varianza del $`5\%`$ para gradientes fotométricos y los errores de velocidad observacionales documentados), forzando a las predicciones teóricas RTM a sobrevivir la ambigüedad de la observación telescópica del mundo real.

2.  **Distribución Monte Carlo:** Para las 52 galaxias de curva plana, simulamos 52,000 puntos de datos inyectando los márgenes de error de velocidad rotacional específicos de vuelta en las derivaciones de pendiente, mapeando la verdadera distribución probabilística del exponente topológico $`\alpha`$.

**E.3 La Curva de Rotación Topológica (Hallazgos Robustos)**

Incluso bajo penalización fuerte por varianza observacional, el marco RTM tiene éxito abrumadoramente:

- **El Atractor de Curva Plana:** La distribución Monte Carlo robusta para las galaxias de curva plana se ajusta a un hermoso atractor Gaussiano en $`\mathbf{\alpha}\mathbf{= \ 1.993\ }\mathbf{\pm}\mathbf{0.130}`$. Esto es estadísticamente indistinguible del límite teórico RTM de $`\alpha = \ 2.0`$. Prueba que a medida que el disco bariónico se difunde hacia afuera, naturalmente se relaja a un estado topológico invariante de escala, que matemáticamente impone un perfil de velocidad constante independiente de la masa.

- **El Vínculo Estructura-Cinemática:** El análisis ODR robusto prueba que el vínculo físico entre estructura bariónica visible y cinemática orbital es mucho más pronunciado y definitivo de lo que OLS sugirió (Pendiente ODR $`= \  - 1.169\  \pm 0.119`$).

**Conclusión:** Al tratar la galaxia como una red de transporte cohesiva y multiescala en lugar de una colección de masas puntuales Newtonianas independientes embebidas en un halo de materia oscura, el marco RTM explica exitosamente los datos cinemáticos. Las curvas de rotación planas "anómalas" son estrictamente la firma de un sistema bariónico operando en la clase de transporte topológico $`\alpha \approx 2`$.

**APÉNDICE F — Validación Empírica: Relajación Topológica y Turbulencia MHD en el Viento Solar**

El marco RTM dicta que la propagación de energía a través de cualquier medio está estrictamente gobernada por su coherencia topológica. Para validar esto a escalas astrofísicas, analizamos la turbulencia magnetohidrodinámica (MHD) del viento solar, un plasma no colisional donde los campos magnéticos actúan como la red estructural para el transporte de energía.

**F.1 La Falacia del Promedio Estático**

El análisis robusto de Fase 2 prueba que el índice del viento solar no es una constante estática, sino una medida de **Relajación Topológica**. El índice evoluciona radialmente desde $`\mathbf{- 1.52}`$ (Topología Rígida Cercana al Sol a 0.1 UA) a $`\mathbf{- 1.72}`$ (Fluido Fractal de Espacio Profundo a 2.0 UA).

Sin embargo, tratar el viento solar en expansión como un medio estático y homogéneo introduce un fallo analítico crítico. Promediar estas métricas destruye la física dinámica subyacente y oscurece la evolución geométrica del plasma.

**F.2 Relajación Topológica Radial**

Para probar robustamente el marco RTM, analizamos la evolución radial del índice espectral desde 0.1 UA (Parker Solar Probe) hasta 2.0 UA (Ulysses). La trayectoria corregida por varianza prueba inequívocamente que el plasma experimenta una **Relajación Topológica** macroscópica:

- **Topología Rígida Cercana al Sol (0.1 UA):** En la vecindad inmediata del Sol, los intensos campos magnéticos imponen una jerarquía rígida, altamente coherente tipo 1D. El índice espectral empírico aquí converge firmemente a $`- 1.52`$, coincidiendo perfectamente con el límite teórico de Iroshnikov-Kraichnan (IK) ($`- 3\text{/}2`$).

- **Fluido Fractal de Espacio Profundo (1.0 - 2.0 UA):** A medida que el plasma se expande y el campo magnético global se debilita, la restricción topológica rígida se rompe. El plasma "se relaja," fracturándose en un estado isotrópico 3D. El índice espectral cae a $`- 1.68`$ a $`- 1.72`$, alineándose con el límite de turbulencia fractal de Kolmogorov ($`- 5\text{/}3`$).

La regresión lineal de esta relajación (pendiente = $`- 0.18`$ por década UA, $`R^{2} = 0.98`$) prueba que el cambio espectral no es un error de medición, sino la firma matemática de decaimiento de coherencia multiescala.

**F.3 Balance Crítico y Fricción Topológica**

Evidencia adicional de geometría RTM se encuentra en la anisotropía espectral del plasma. Los datos empíricos demuestran que el espectro de energía cambia dependiendo del ángulo relativo al campo magnético local ($`\theta_{B}`$). La energía que atraviesa *a través de* las líneas de campo magnético encuentra "Fricción Topológica," forzando al sistema a un escalado fractal asimétrico conocido como Balance Crítico ($`k_{\parallel} \propto k_{\bot}^{2\text{/}3}`$). El plasma está geométricamente restringido por la red magnética.

**F.4 Intermitencia Multifractal**

Finalmente, un análisis de las funciones de estructura de orden superior ($`\zeta_{q}`$) de datos MMS (Magnetospheric Multiscale) revela desviaciones severas del escalado monofractal lineal. Esto confirma que la energía del plasma no se disipa en una rejilla perfectamente uniforme; más bien, la topología subyacente es un **multifractal**. Los vórtices de alta energía crean "hoyos" topológicos temporales o estructuras coherentes, reflejando perfectamente las concentraciones de energía discretas y heterogéneas predichas por RTM.

**Conclusión:** El viento solar no es un gas simple; es una red topológica que se relaja dinámicamente. El mapeo impecable de la evolución del plasma desde el límite de Iroshnikov-Kraichnan al límite de Kolmogorov proporciona prueba empírica definitiva de que la Teoría Rítmica de la Materia gobierna con precisión el transporte de energía no colisional en el cosmos.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
