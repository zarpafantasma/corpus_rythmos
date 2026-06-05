<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Serpiente">

# Astronomía Rítmica 
**Una ley de pendiente RTM para curvas de rotación galáctica**   
  
Álvaro Quiceno


</div>

**Resumen**
Presentamos la Astronomía Rítmica, una aplicación del marco RTM (Relativista Temporal Multiescala) a la dinámica galáctica, en la cual los relojes orbitales están gobernados no solo por la gravedad y la masa bariónica, sino también por un exponente de coherencia α que codifica la organización multiescala del medio bariónico. En RTM, los tiempos característicos escalan como T ∝ L\^α a entorno fijo; al mapear esto a órbitas circulares se obtiene la ley de velocidad

v ∝ r\^(1 − α/2)

de modo que la pendiente de log v vs. log r dentro de anillos con coherencia fija es igual a (1 − α/2). Este marco genera tres predicciones falsificables: (i) pruebas de pendiente en curvas de rotación agrupadas por coherencia estructural, (ii) una reformulación de la relación bariónica de Tully--Fisher en la cual los residuos se correlacionan con indicadores de α en lugar de parámetros de halo, y (iii) consistencia entre lentes gravitacionales y cinemática si α modifica los tiempos operacionales pero no la curvatura del espacio-tiempo.

Detallamos cómo estimar α a partir de textura fotométrica y cinemática---entropía multiescala, potencia de modos de Fourier, índices de turbulencia---y cómo realizar verificaciones de "colapso" (planitud de residuos dentro de bins de coherencia), reflejando la disciplina de pendiente primero utilizada en otras partes del corpus RTM.

**Validación empírica sistemática** $\mathbf{\rightarrow}$ **(APÉNDICE E)**. Aplicamos esta metodología a la base de datos SPARC (Lelli et al. 2016), que comprende 175 galaxias de disco con fotometría Spitzer a 3.6 μm y curvas de rotación HI/Hα de alta calidad. El análisis arroja tres hallazgos principales: (1) Una correlación robusta estructura-cinemática (pendiente ODR $= -1.17 \pm 0.12$) sobrevive a la inyección de ruido y la propagación de errores, confirmando que la organización geométrica de la materia visible covaría con la cinemática orbital a través de tipos de galaxias. (2) Un análisis de discrepancia de masa utilizando las componentes de velocidad bariónica (V\_gas, V\_disk, V\_bul) revela que la **concentración** bariónica predice la discrepancia de masa **más allá de lo que la masa bariónica total explica** (Spearman parcial $\rho = +0.346$, $p = 0.0001$, controlando por $M_{bar}$), con la estructura agregando un 8.5% de varianza explicada respecto a la masa sola (prueba $F$, $p = 0.002$). (3) Sin embargo, la relación bariónica de Tully-Fisher absorbe virtualmente toda la señal estructural para la predicción de velocidad ($\Delta R^2 < 0.1\%$), indicando que aunque la geometría bariónica contribuye información secundaria a la discrepancia de masa, no reemplaza la necesidad de masa adicional o dinámica modificada a la precisión actual.

**Nota sobre el límite de curva plana.** La ley de velocidad $v \propto r^{1-\alpha/2}$ implica $\alpha = 2(1 - \text{pendiente})$; para curvas de rotación planas (pendiente $\approx 0$), $\alpha \rightarrow 2$ se sigue por identidad algebraica. El contenido empírico no radica en este mapeo sino en si parámetros estructurales independientes predicen propiedades cinemáticas más allá de lo que la masa sola provee.

Además, extendemos el marco RTM al medio interplanetario analizando plasmas astrofísicos no colisionales $\rightarrow$ **(APÉNDICE F)**. Utilizando un extenso conjunto de datos de turbulencia magnetohidrodinámica (MHD) del viento solar---que abarca desde 0.1 UA (Parker Solar Probe) hasta 2.0 UA (Ulysses)---sometimos los índices espectrales a un riguroso pipeline dinámico. Corregimos explícitamente la prevalente "Falacia del Promedio Estático" en estudios heurísticos de plasmas, demostrando que el índice espectral del viento solar no es una constante estática ($\approx - 1.63$), sino una medida de decaimiento geométrico activo. El análisis robusto demuestra que el plasma experimenta una estricta **Relajación Topológica**: cerca del Sol, campos magnéticos intensos imponen una topología rígida y altamente coherente (convergiendo al límite de Iroshnikov-Kraichnan, $\alpha = \  - 1.52$); a medida que el plasma se expande hacia el espacio profundo, esta topología magnética se fractura en hidrodinámica fractal 3D completamente desarrollada (convergiendo al límite de Kolmogorov, $\alpha \approx - 1.72$). Junto con evidencia de balance crítico e intermitencia multifractal, esto confirma que el espacio-tiempo y los campos magnéticos dictan la geometría topológica exacta de las cascadas de energía en el cosmos.

> **Hallazgos Red Team (abril 2026).** Una validación adversarial independiente utilizando el modelo bariónico completo de SPARC (V\_gas, V\_disk, V\_bul) produjo tres resultados adicionales: (i) Un análisis de discrepancia de masa revela que la **concentración** bariónica predice la discrepancia de masa $D = V_{obs}^2 / V_{bar}^2$ más allá de lo que la masa bariónica total explica (Spearman parcial $\rho = +0.346$, $p = 0.0001$, controlando por $M_{bar}$), con la estructura añadiendo 8.5% de varianza explicada ($F$-test $p = 0.002$). (ii) El **problema de diversidad** — por qué galaxias con el mismo $V_{flat}$ muestran distintas formas internas de curva de rotación — se aborda parcialmente: la pendiente del perfil de brillo superficial predice la razón ascenso-a-planicie a masa fija (parcial $\rho = +0.329$, $p = 0.0001$), con el efecto replicándose dentro de los bins de rotadores medios y rápidos. (iii) Sin embargo, una prueba directa de predicción de velocidad RTM ($v_{RTM}(r) = \kappa \cdot r^{1-\alpha(r)/2}$ con un parámetro libre) fracasa decisivamente frente a ajustes de halo NFW (RTM gana 2/135 galaxias). La relación bariónica de Tully-Fisher absorbe prácticamente toda la señal estructural para predicción de velocidad ($\Delta R^2 < 0.1\%$). Estos resultados establecen que la geometría bariónica contiene información secundaria estadísticamente significativa sobre el patrón de discrepancia de masa, pero no reemplaza la necesidad de masa adicional o dinámica modificada con la precisión actual.
>
> **Nota sobre la identidad de curva plana.** La ley de velocidad $v \propto r^{1-\alpha/2}$ implica $\alpha = 2(1-\text{pendiente})$; para curvas de rotación planas (pendiente $\approx 0$), $\alpha \rightarrow 2$ se sigue por identidad algebraica. El contenido empírico del Apéndice E reside en las correlaciones estructurales, no en este mapeo.
> 

2\. **Introducción**

**2.1 El enigma.** Las curvas de rotación planas o lentamente crecientes a grandes radios, las relaciones bariónicas de Tully--Fisher (bTFR) estrechas pero con dispersión, y las diversas formas internas a través de los tipos de Hubble siguen siendo diagnósticos centrales de la distribución de masa en galaxias. La resolución estándar añade halos de **materia oscura** no bariónica; las alternativas modifican la ley de fuerza (p. ej., MOND). Ambas familias pueden ajustar muchas curvas pero enfrentan tensiones---p. ej., diversidad de pendientes internas a masa fija, acoplamiento barión--halo y verificaciones cruzadas lentes--dinámica.

**2.2 Una tercera vía.** El marco **RTM** postula que los sistemas de muchos cuerpos exhiben una **ley escala--tiempo**,

$$T(L) = T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha}\Theta\text{ (factores adimensionales fijos dentro de un bin)},$$

donde $\alpha$ resume la **profundidad de coherencia** del entorno (jerarquía, persistencia, orden). RTM ha sido formulado y probado en sistemas sintéticos y físicos (rejillas fractales, redes jerárquicas) en los cuales $\alpha$ aumenta con la complejidad estructural, ralentizando la dinámica operacional de manera cuantificable según la pendiente.

**2.3 Hipótesis astronómica.** Sin alterar la gravedad, tratar el **campo de estructura bariónica** (barras, espirales, grumos, espesor, turbulencia) como un entorno que establece un perfil $\alpha(L)$. Escribiendo $T = 2\pi L/v$ se obtiene

$$v(L) = \kappa L^{1 - \alpha(L)/2} \Rightarrow \frac{\partial\log v}{\partial\log L} = 1 - \alpha\text{/}2,$$

haciendo de la **pendiente** el diagnóstico principal. Donde el medio bariónico alcanza relajación estructural, $\mathbf{\alpha \rightarrow}\mathbf{2}$ predice curvas externas planas; donde la estructura es fuerte (barras/bulbos/grumos), $\alpha > 1$ predice ascensos internos más pronunciados---ambos sin invocar masa exótica. La misma lógica de pendiente primero subyace a notas previas de RTM sobre reescalamiento temporal y transporte multiescala.

**Qué probamos.** (i) **Pendientes de rotación:** dentro de anillos emparejados por indicador de $\alpha$, $log\ v$ vs. $Log\ L$ tiene pendiente $1 - \alpha\text{/}2$. (ii) **Residuos de la bTFR:** los residuos se correlacionan con indicadores de $\alpha$ (textura, entropía, potencia de modos), no con parámetros libres de halo. (iii) **Lentes gravitacionales:** dado que $\alpha$ cambia los **tiempos** operacionales y no la curvatura, las masas por lentes deberían seguir rastreando los bariones; cualquier brecha de masa sistemática después de condicionar por $\alpha$ falsifica la interpretación. Pre-registramos umbrales de aprobación/rechazo y adoptamos las **verificaciones de colapso** de RTM (planitud de ${v\ L}^{\alpha - 1}$ dentro de bins) como pruebas del modelo, en analogía directa con los dominios químicos y de redes del corpus.

**2.4. Validación empírica sistemática: El laboratorio galáctico (APÉNDICE E)**

Para fundamentar estas proposiciones teóricas en la realidad observacional, probamos el marco RTM usando la base de datos SPARC (Spitzer Photometry and Accurate Rotation Curves) (Lelli et al., 2016). Este conjunto de datos, que comprende 175 galaxias de disco cercanas con cinemática y fotometría de alta fidelidad, sirve como banco de pruebas ideal para la hipótesis central de RTM: que la pendiente de la curva de rotación se correlaciona con la coherencia multiescala del medio bariónico.

Dado que los datos cinemáticos galácticos son inherentemente ruidosos---plagados de incertidumbres de inclinación, errores de estimación de distancia y dispersión natural de velocidad HI---desplegamos un riguroso pipeline estadístico de Errores en Variables (EIV) para prevenir el sesgo de atenuación. El análisis arrojó tres hallazgos principales y una limitación importante:

1.  **Correlación estructura-cinemática:** Una correlación ODR robusta (pendiente $= -1.17 \pm 0.12$) vincula el indicador de estructura fotométrica (gradiente de brillo superficial) con la pendiente cinemática a través de tipos de galaxias. Esta correlación entre tipos es genuina: galaxias con diferentes perfiles estructurales tienen formas de curvas de rotación sistemáticamente diferentes. Sin embargo, el análisis intra-cuartil muestra que el efecto es impulsado principalmente por la clasificación morfológica (entre galaxias concentradas y difusas) en lugar de por variación estructural continua dentro de un tipo dado.

2.  **Discrepancia de masa y estructura bariónica (Hallazgo nuevo):** Usando las componentes de velocidad bariónica provistas por SPARC (V\_gas, V\_disk, V\_bul), calculamos la discrepancia de masa $D = V_{obs}^2 / V_{bar}^2$ en cada radio y probamos si los parámetros estructurales predicen esta discrepancia más allá de la masa bariónica total. A $M_{bar}$ fija, la **concentración** bariónica predice significativamente la discrepancia de masa (Spearman parcial $\rho = +0.346$, $p = 0.0001$), con los parámetros estructurales agregando un 8.5% de varianza explicada respecto a la masa sola (prueba $F$, $p = 0.002$). Esto demuestra que la organización geométrica de la materia bariónica contiene información sobre el problema de la "masa faltante" que la masa total sola no captura.

3.  **Limitación---Dominancia de la BTFR:** A pesar del hallazgo sobre la discrepancia de masa, la estructura bariónica no puede predecir la velocidad de rotación asintótica ($v_{flat}$) más allá de la Relación Bariónica de Tully-Fisher. La BTFR sola alcanza $R^2 = 0.91$; agregar parámetros estructurales produce $\Delta R^2 < 0.1\%$. Los residuos de la BTFR muestran correlación nula con los parámetros estructurales ($\rho \approx 0$, $p > 0.7$). Esto indica que aunque la estructura contribuye información secundaria a la discrepancia de masa, no reemplaza el rol primario de la masa bariónica total en la determinación de la cinemática.

4.  **Nota sobre la identidad de curva plana:** Las 52 galaxias con curvas de rotación planas producen $\alpha = 1.993 \pm 0.130$ bajo el mapeo RTM. Dado que $\alpha = 2(1 - \text{pendiente})$, este resultado se sigue algebraicamente de pendientes planas ($\approx 0$) y representa una verificación de consistencia interna más que un descubrimiento empírico independiente.

Estos hallazgos indican que la geometría bariónica provee poder predictivo secundario pero estadísticamente significativo para la discrepancia de masa, consistente con la intuición RTM de que la topología estructural importa. Sin embargo, aún no constituyen una alternativa independiente a los halos de materia oscura o la gravedad modificada, ya que la BTFR absorbe virtualmente toda la señal estructural para la predicción de velocidad. La cuestión de si refinamientos al indicador estructural (p. ej., entropía multiescala, potencia de modos de Fourier) pueden aumentar la contribución predictiva independiente permanece abierta y se aborda en la metodología propuesta (Secciones 5--9).

**2.5. Validación empírica sistemática: Relajación topológica en plasmas astrofísicos (APÉNDICE F)**

Mientras las curvas de rotación galáctica proveen evidencia para RTM a escalas de kilopársecs, el viento solar interplanetario sirve como el laboratorio local definitivo para probar RTM en un fluido no colisional. Más del 99% del universo visible consiste en plasma, donde el flujo de energía está gobernado no por colisiones atómicas, sino por la topología multiescala de los campos magnéticos.

Históricamente, los estudios astrofísicos han promediado frecuentemente el índice espectral inercial del viento solar a través de vastas distancias, produciendo un valor heurístico estático ($\approx - 1.63$). En el Apéndice F, sometemos datos de viento solar multimisión (Parker Solar Probe, Solar Orbiter, Wind y Ulysses) a una auditoría estadística dinámica. Hipotetizamos que bajo el marco RTM, el plasma debe exhibir "Relajación Topológica". En lugar de un espectro constante, los datos empíricos revelan una evolución radial estricta desde una topología rígida dominada magnéticamente cerca del Sol hasta una red multiescala fracturada e isotrópica en el espacio profundo.

**2.6. Actualización Empírica Red Team: Qué Puede y Qué No Puede Hacer la Estructura Bariónica**

El análisis original del Apéndice E estableció una correlación estructura-cinemática (pendiente ODR $= -1.17 \pm 0.12$) y reportó que las curvas de rotación planas convergen a $\alpha = 1.993 \pm 0.130$. Pruebas adversariales posteriores (Red Team, abril 2026) refinaron y extendieron estos hallazgos de tres maneras importantes.

**Lo que la estructura bariónica SÍ puede hacer.** Dos análisis de correlación parcial independientes demuestran que la organización geométrica de la materia bariónica contiene información sobre la dinámica galáctica más allá de lo que la masa total proporciona:

1. **Predicción de discrepancia de masa.** A masa bariónica total $M_{bar}$ fija, la concentración de brillo superficial de una galaxia predice significativamente su discrepancia de masa exterior $D = V_{obs}^2 / V_{bar}^2$ (Spearman parcial $\rho = +0.346$, $p = 0.0001$; $n = 120$ galaxias). Un modelo multivariable que incluye concentración y pendiente de SB añade 8.5% de varianza explicada más allá de la masa sola ($F$-test $p = 0.002$). Las galaxias con perfiles bariónicos más concentrados exhiben mayores discrepancias de masa a masa fija.

2. **Diversidad de curvas de rotación.** A $V_{flat}$ fijo, la pendiente del perfil de SB predice la razón interna de ascenso-a-planicie de la curva de rotación (parcial $\rho = +0.329$, $p = 0.0001$; $n = 131$). Esto aborda el conocido "problema de diversidad" (Oman et al. 2015): galaxias con la misma velocidad asintótica pero diferente concentración bariónica muestran formas internas de curva de rotación sistemáticamente distintas. El efecto se replica dentro de los terciles de rotadores medios ($\rho = +0.320$, $p = 0.037$) y rápidos ($\rho = +0.348$, $p = 0.021$), pero no en rotadores lentos (enanas) donde los perfiles de SB son demasiado difusos para proporcionar contraste estructural.

**Lo que la estructura bariónica NO puede hacer.** Tres pruebas establecen las limitaciones actuales:

1. **La predicción directa de velocidad fracasa.** Un modelo de velocidad RTM de física pura ($v_{RTM}(r) = \kappa \cdot r^{1-\alpha(r)/2}$, donde $\alpha(r) = |d\log I / d\log r|$, con un parámetro libre $\kappa$) fue probado cara a cara contra ajustes de halo NFW (2 parámetros) y solo bariones (0 parámetros) en 135 galaxias. RTM gana solo 2/135 frente a NFW (RMS mediano: RTM 80.5%, NFW 11.9%). El mapeo $\alpha(r)$ a partir del gradiente bruto de SB es demasiado crudo para capturar el campo de velocidad.

2. **Dominancia de la BTFR.** La Relación Bariónica de Tully-Fisher sola alcanza $R^2 = 0.91$ para predicción de $V_{flat}$. Agregar parámetros estructurales produce $\Delta R^2 < 0.1\%$. Los residuos de la BTFR muestran correlación nula con la concentración o pendiente de SB ($\rho \approx 0$, $p > 0.7$). La estructura no puede mejorar la predicción de velocidad asintótica más allá de la masa.

3. **El scatter de la RAR no se reduce.** El gradiente local de SB no reduce significativamente la dispersión de la Relación de Aceleración Radial (reducción $< 0.1\%$; McGaugh 2016 reporta $\sigma \approx 0.13$ dex). La RAR es tan ajustada que la estructura local prácticamente no añade nada.

Estos resultados enmarcan la contribución de RTM a la dinámica galáctica con precisión: la geometría bariónica proporciona información predictiva estadísticamente significativa pero secundaria ($\sim 4$–$8.5\%$ $\Delta R^2$) sobre el *patrón* de discrepancia de masa, pero no reemplaza el rol primario de la masa total en la determinación de velocidades. Si proxies estructurales refinados (entropía multiescala, descomposición de modos de Fourier, textura cinemática) pueden aumentar sustancialmente esta contribución permanece como una pregunta empírica abierta.


**3. Introducción a RTM para astrónomos**

**3.1 La ley maestra y su signatura de pendiente**

La relación central de RTM es una **ley tiempo--escala** normalizada dimensionalmente:

$$\frac{T}{T_{0}} = \left( \frac{L}{L_{0}} \right)^{\alpha}\Theta,$$

con $L$ una escala característica y $\alpha$ un **exponente de coherencia** que refleja la organización multiescala (jerarquía, persistencia, memoria). Dentro de bins de análisis donde $\Theta$ es fijo, $\partial\ \log\ T/\partial\ \log\ L = \alpha$. Este enfoque de pendiente primero hace a RTM falsificable: medir tiempos a través de tamaños y leer $\alpha$ de la pendiente log--log.

Mapeando $T = 2\pi L/v$ se obtiene

$$v(L) = \kappa L^{1 - \alpha(L)/2} \Rightarrow \frac{\partial\log v}{\partial\log L} = 1 - \alpha\text{/}2,$$

Así, la rotación **plana** ($pendiente\  \approx 0$) corresponde a $\alpha \approx 2$; la caída **kepleriana** (pendiente $- 1/2$ en $v$ vs. $r$) no se espera en distribuciones de masa extendidas a menos que $\alpha < 1$ localmente; las curvas internas **ascendentes** implican $\alpha > 1$. El punto no es la intersección $\kappa$ (fijada por la masa y geometría bariónicas) sino la **diferencia de pendiente** entre bins de coherencia.

**3.2 Qué representa α (y qué no)**

-   **Representa:** la **profundidad de coherencia** efectiva del entorno bariónico---el grado en que la estructura anidada ralentiza u organiza el transporte, la mezcla y la relajación orbital. A través de los estudios RTM, medios más jerárquicos producen mayores $\alpha$ (p. ej., rejillas de Sierpiński y árboles vasculares elevan $\alpha$ por encima de los valores difusivos).

-   **No representa:** masa extra, gravedad modificada ni cambios en la expansión de fondo. En RTM, α modifica los **tiempos operacionales** de procesos incrustados en medios estructurados mientras deja intactas las pruebas métricas (BBN/CMB/PPN)---una distinción enfatizada en notas adyacentes a la cosmología.

**3.3 Anclajes empíricos para α**

El corpus RTM demuestra cómo $\alpha$ se **lee** de las pendientes en sistemas multiescala (caminatas aleatorias en redes jerárquicas y fractales), con $\alpha$ aumentando de manera confiable a medida que la complejidad crece---una "escalera empírica" que nos permite calibrar expectativas antes de tocar datos galácticos. Adoptamos la misma disciplina aquí: estimar $\widehat{\alpha}(L)$ a partir de **indicadores de estructura** independientes (entropía multiescala de la luz, índices de turbulencia HI/H$\alpha$, potencia de modos de barra/espiral, espesor/asimetría), y luego verificar que las **pendientes cinemáticas** sean iguales a $1 - \alpha\text{/}2$

dentro de anillos agrupados por indicador. Si la consistencia pendiente--indicador falla, RTM falla.

**3.4 Discriminantes inmediatos**

1.  **Prueba de pendiente de rotación.** En anillos estratificados por $\widehat{\alpha}$, ajustar log $v$ vs. $\log\ L$; la pendiente debe ser igual a $1 - \alpha\text{/}2$ con residuos pequeños tras correcciones geométricas. Aprobación/rechazo es un solo número por bin.

2.  **Verificación de colapso.** Graficar ${v\ L}^{\alpha - 2/1}$ vs. $L$ dentro de un bin; la planitud (pendiente cero) es la verificación del modelo, como se usa en otros dominios de RTM.

3.  **Reformulación de la bTFR.** Regresar los residuos de la bTFR sobre indicadores de $\widehat{\alpha}$; una correlación significativa favorece el "control por coherencia" de RTM, mientras que la independencia favorece las parametrizaciones de materia oscura o el escalamiento tipo MOND.

4.  **Consistencia de lentes.** Si $\alpha$ cambia los relojes pero no la curvatura, los mapas de masa por lentes deberían seguir rastreando los bariones; cualquier **brecha de masa** robusta entre lentes y cinemática que persista después de condicionar por $\widehat{\alpha}$ constituye un **límite de alcance** o falsificación.

**Resumen de la configuración.** RTM ofrece un marco alternativo **falsificable** a **nivel de pendiente** para la cinemática galáctica: mantener la gravedad; introducir un $\alpha(L)$ medible ligado a la estructura bariónica; predecir pendientes de rotación $1 - \widehat{\alpha}\text{/}2$ y probarlas con verificaciones de colapso y patrones de residuos de la bTFR. En las siguientes secciones (i) formalizaremos las predicciones a escala galáctica, (ii) especificaremos cómo recuperar $\widehat{\alpha}(L)$ a partir de datos de imagen/IFU, y (iii) definiremos criterios pre-registrados de aprobación/rechazo incluyendo verificaciones cruzadas lentes--dinámica.

**4. Predicciones centrales a escala galáctica**

Esta sección convierte la regla RTM

$$T(L) = T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha(L)} \Longleftrightarrow v(L) = \kappa L^{1 - \alpha/2}$$

en **discriminantes observacionales**. El diagnóstico central es siempre **pendiente primero**: dentro de anillos donde un indicador de coherencia es aproximadamente constante (un "bin de coherencia"), la pendiente de $log\ v$ vs. $log\ L$ debe ser igual a $1 - \alpha/2$. Las intersecciones absorben la geometría y la normalización de masa; las **pendientes y colapsos** son la prueba.

**4.1 Curvas de rotación: ascensos internos, planos externos y diversidad**

**Predicción P1 (discos externos).** En medios externos difusos y débilmente coherentes, $\alpha(L) \rightarrow 2$, por lo tanto ${v(L) \propto L}^{0}$ (rotación plana).

**Predicción P2 (regiones internas).** Donde la estructura es fuerte---barras, bulbos compactos, anillos de formación estelar con grumos $- \alpha(L) > 1$ y ${v(L) \propto L}^{1 - \alpha/2}$ **asciende** con el radio (dado que $1 - \alpha < 0$ reduce la pendiente hacia cuerpo sólido solo si $\alpha \approx 0$; con $\alpha > 1$ la pendiente log se vuelve negativa a positiva pequeña dependiendo de la geometría---ver abajo). Operacionalmente: **la coherencia aumenta el** $\mathbf{T}$ **local** respecto a un reloj puramente geométrico, por lo que el **déficit de velocidad** se reduce con el radio dentro de la zona coherente, produciendo segmentos ascendentes que luego se nivelan cuando $\alpha \rightarrow 2$.

**Diversidad a masa fija.** Galaxias con masa bariónica similar pero diferentes **mapas de coherencia** $\alpha(L)$ mostrarán diferentes formas internas---resolviendo el "problema de diversidad" sin invocar diferentes respuestas de halo. La diversidad es **varianza explicada** una vez agrupada por indicadores de $\alpha$.

**Prueba de pendiente.** En cada bin de coherencia,

$$\left. \ \frac{\partial\log v}{\partial\log L} \right|_{\text{bin}} = 1 - \alpha_{\text{bin}}/2$$

**Prueba de colapso.** Para cada bin, ${v\ L}^{\alpha_{bin}/2 - 1}$ es **plano** vs. $L$. La falla de la pendiente o el colapso falsifica RTM **en ese bin**.

> *Nota geométrica.* Lo anterior usa un indicador de órbita circular $v(L)$. En la práctica corregimos por inclinación, deriva asimétrica y movimientos no circulares; el diagnóstico de pendiente es robusto ante estos a primer orden porque principalmente desplazan **intersecciones** en lugar de **pendientes** cuando se tratan consistentemente a través de $L$.

**4.2 La relación bariónica de Tully--Fisher (bTFR) reformulada**

Sea $v_{flat}$ medida donde $\alpha \rightarrow 2$. Entonces RTM predice

$$v_{\text{flat}} \approx \kappa\left( L_{*} \right)L_{*}^{0},\quad\text{con}\quad\kappa\left( L_{*} \right) \propto \sqrt{\frac{GM_{b}}{L_{*}}}$$

por lo que el escalamiento bTFR de **orden principal** permanece estrecho (los bariones controlan la intersección), pero los **residuos** respecto a un ajuste global recogen un **término de coherencia** de la variación de $\alpha(L)$ entre radios internos y externos:

**Predicción P3 (residuos de la bTFR).** Después de correcciones geométricas estándar, los residuos $\Delta\ log\ v$ se correlacionan con métricas de coherencia **derivadas de la estructura** (p. ej., entropía multiescala, potencia del modo de barra, grumosidad) tales que las galaxias con **mayor** $\mathbf{\alpha}$ **interno** muestran **residuos sistemáticos** si $v$ se muestrea demasiado dentro de la zona $\alpha \rightarrow 1$. Usar un radio métrico fijo (p. ej., 2.2 $R_{d}$) a través de galaxias no debería **eliminar completamente** las correlaciones residuo--estructura; muestrear en el radio donde la pendiente local es $\approx 0$ debería hacerlo.

**Discriminante.**

-   Los **ajustes de halo de materia oscura** esperan que los residuos se correlacionen con la concentración/espín del halo, no necesariamente con la **coherencia bariónica** después de controlar por masa y tamaño.

-   **MOND** espera que los residuos se correlacionen con la escala de aceleración, no con la **textura** a bariones fijos.\
    **RTM** predice que la **textura/estructura** explica una fracción significativa de la varianza residual.

**4.3 Elípticas y sistemas dominados por dispersión**

Para sistemas soportados por presión, mapeamos la ley temporal de RTM a **escalamientos de Jeans**. Si un tiempo orbital/de relajación característico en una capa esférica sigue ${T \propto L}^{\alpha}$, entonces el **perfil de dispersión** obedece, a primer orden,

$$\sigma(L) \sim \frac{L}{T} \propto L^{1 - \alpha(L)}$$

**Predicción P4.** En elípticas con estructura central fuerte (núcleos, anisotropía, discos embebidos), $\alpha > 1$ dentro de un radio de quiebre produce $\sigma(L)$ **ascendente** hacia el centro o una caída **más suave** de lo que las expectativas geométricas sugieren; en envolventes más redondas y difusas donde $\alpha \rightarrow 1,\ \sigma(L)$ se aplana. Como en los discos, la **pendiente** de $\log\ \sigma$ vs. $\log L$ dentro de bins de coherencia debería ser igual a $1 - \alpha$.

**Discriminante.** Las interpretaciones de materia oscura requieren ajustes de anisotropía y pendiente de halo; RTM predice un acoplamiento **coherencia--pendiente de dispersión** medible a partir de mapas IFU sin libertad de halo una vez que los bariones están fijados.

**4.4 Estructura vertical de discos y alabeos**

Tratar el tiempo de oscilación vertical $T_{z}$ de estrellas/gas del disco en una lámina como obedeciendo $T_{z}{\propto H}^{\alpha_{z}}$, con $H$ un indicador de espesor/altura de escala local y $\alpha_{z}$ un exponente de **coherencia vertical** (sensible a la estratificación, turbulencia, ordenamiento magnético).

**Predicción P5 (ensanchamiento).** En discos externos donde el medio es menos coherente verticalmente ($\alpha_{z} \rightarrow 1$), $T_{z}$ se acorta respecto a las regiones internas estratificadas, produciendo un **ensanchamiento suave** consistente con fuerzas restauradoras verticales más débiles pero oscilaciones **coherentes**; RTM espera que la pendiente log de la frecuencia de oscilación vertical con el radio se aproxime a 0 cuando $\alpha_{z} \rightarrow 1$.

**Predicción P6 (alabeos y** $\mathbf{\nabla\alpha}$).Los alabeos a gran escala se correlacionan con **gradientes** en la coherencia, $\nabla\alpha$, a través del disco---p. ej., transiciones de zonas internas ordenadas por espiral/barra a HI externo más turbulento. RTM predice **desfases de fase** sistemáticos y **asimetrías** en modos verticales donde $\nabla\alpha$ es mayor (comprobable con tomografía HI y cinemática de Gaia DR).

**4.5 Enanas y galaxias de bajo brillo superficial (LSB)**

Las enanas/LSBs tienen bariones difusos y débilmente ordenados en la mayoría de los radios.

**Predicción P7.** Sus perfiles $\alpha(L)$ se sitúan cerca de la **unidad** a través de amplios rangos radiales, por lo que RTM espera:

-   Curvas de rotación **suavemente ascendentes y luego aplanándose** sin necesidad de halos con cúspide, consistente con $\alpha \rightarrow 2$

-   **Poca diversidad interna** una vez agrupadas por indicadores simples de estructura (espesor, grumosidad), porque $\alpha$ varía menos a través del radio que en discos dominados por barras y de alto brillo superficial.

**Discriminante.** Donde los ajustes de materia oscura invocan halos con **núcleo** vs. **cúspide** para explicar las formas internas, RTM predice un **acoplamiento estructura--pendiente** medible: p. ej., enanas más grumosas en formación estelar (ligeramente mayor $\alpha$ interno) muestran ascensos internos ligeramente más pronunciados **a perfil de masa fijo**.

| **Observable** | **Control de coherencia (indicador)** | **Predicción de pendiente RTM** | **Verificación de colapso** | **Discriminante distintivo** |
|----|----|----|----|----|
| Rotación de disco (interna) | Fuerza de barra, compacidad del bulbo, grumosidad | *∂ log v / ∂ log L=1−α/2* o pequeña; asciende y luego se nivela cuando *α→2* | $`{v\ L}^{\alpha - 2 - 1}`$ plano dentro del bin | Diversidad a masa fija explicada por **estructura**, no por parámetros de halo |
| Rotación de disco (externa) | HI difuso, baja potencia de modos | *∂ log v / ∂ log L→0* | Plano dentro del bin | Planitud sin materia oscura si *α≈2* |
| Residuos de la bTFR | Métricas de textura, entropía multiescala | Los residuos se correlacionan con indicadores de coherencia | — | Residuos ligados a **estructura bariónica**, no a concentración de halo |
| *σ(r)* de elípticas | Anisotropía central, discos embebidos | *∂ log σ / ∂ log L=1−α/2* en bins | $`{\sigma\ L}^{\alpha - 1}`$ plano | Pendientes de dispersión predichas solo a partir de mapas de estructura |
| Ensanchamiento vertical | $`\alpha_{z}`$ (estratificación, turbulencia) | $\partial \log \nu\_z / \partial \log R \to 0$ cuando $\alpha\_z \to 1$ | $`\nu_{z}\ H^{\alpha_{z} - 1}`$ plano | Fase/asimetría de alabeos vs. *∇α* |
| Enanas/LSBs | Bariones de bajo orden | $`\alpha`$ cercano a la unidad $`\Rightarrow`$ ascensos suaves, baja diversidad | Colapso externo plano | Acoplamiento estructura--pendiente a perfil de masa fijo |

**Cómo se prueban estas predicciones.** En la Sección 5 (Métodos para estimación de $\alpha$) definiremos pipelines **estructura→**$\mathbf{\alpha}$ (entropía multiescala, potencia de modos de barra/espiral, índices de turbulencia), y luego ejecutaremos **pruebas de pendiente y colapso bin a bin** en perfiles de rotación y dispersión. En las Secciones 6--7 (Comparaciones y consistencia) mostraremos cómo estas predicciones RTM se separan de las **parametrizaciones de materia oscura** y los **escalamientos tipo MOND**, e incluiremos **verificaciones cruzadas lentes--cinemática** para asegurar que alterar los relojes (vía $\alpha$) no introduce cambios de curvatura encubiertos.

**5. De la luz a** $\mathbf{\alpha}$ **: Estimación de coherencia estructural**

Esta sección especifica **cómo** construir un campo radial $\widehat{\alpha}(L)$ a partir de imágenes y cinemática, con incertidumbres y control de calidad (QA). El objetivo es un $\alpha$ *operacional* por anillo que (i) se derive de **indicadores de estructura independientes**, (ii) prediga la **pendiente** $1 - \widehat{\alpha}$ de $\log\ v$ vs. $\log L$, y (iii) pase las **verificaciones de colapso** ${v\ L}^{\widehat{\alpha} - 1} \approx const$ dentro de bins de coherencia.

**5.1 Productos de datos y preprocesamiento**

**Entradas (por galaxia):**

-   **Imágenes** profundas de **banda ancha** (p. ej., *gri* o NIR) para estructura estelar; FWHM del PSF y mapas de varianza.

-   **Gas** espacialmente resuelto: HI 21 cm (momentos 0/1/2), y si están disponibles, mapas de $H\alpha$.

-   **Cinemática**: curvas de rotación (HI o IFU), campos de velocidad 2D y mapas de dispersión de velocidad.

-   **Geometría**: distancia, inclinación iii, ángulo de posición (PA), longitud de escala del disco $R_{d}$, indicadores de espesor si están disponibles.

**Preprocesamiento:**

-   Deconvolución del PSF (regularizada; registrar la resolución efectiva tras la deconvolución).

-   Máscara de primer plano/fondo; sustracción de cielo; ajustes de elipses isofotales para definir **anillos**.

-   Corrección de difuminado de haz para campos de velocidad (modelado directo o recetas estándar).

-   Corrección de deriva asimétrica donde sea necesario (gas vs. estrellas).

-   Todos los mapas remuestreados a una **grilla común** con incertidumbre propagada.

**5.2 Indicadores estructurales de coherencia**

Calculamos descriptores **multiescala** en cada anillo $A_{j}$ (ancho $\Delta\ log\ L$ fijo). Cada indicador se normaliza a $\lbrack 0,1\rbrack$ y tiene una incertidumbre.

1.  **Entropía multiescala** $\mathbf{E}$ **.** Entropía de Shannon de la intensidad de imagen después de filtrado pasa-banda (p. ej., wavelets à trous) a través de escalas espaciales $s \in \lbrack s_{\min},\ s_{\max}\rbrack$. Mayor **orden** (estructura clara) → **menor** entropía → **mayor** coherencia. Definir $E^{\star} = 1 - E_{norm}$.

2.  **Índice fractal/turbulento** $\mathbf{D}$ **.** Función de estructura de 2 puntos $S_{2}\mathcal{(l) \propto}\mathcal{l}^{\zeta}$ (luz $HI/H\alpha$ o estelar). Mapear el exponente $\zeta$ o la dimensión fractal $D$ a un **puntaje de coherencia** $C_{D}$ (menor $D$ a escalas grandes ⇒ mayor coherencia).

3.  **Potencia de modos de Fourier** $P_{m}$. Potencia fraccional en $m = 2$ (barra), $m = 2 - 4$ (espiral), calculada a partir del brillo superficial deproyectado; normalizar a $C_{mode}{= \sum}_{m \in M}{\ P}_{m}$.

4.  **Grumosidad** $\mathbf{S}$ **y suavidad** $Q = 1 - S$. Alto $Q$ (suave) sugiere estructura ordenada; usar la familia estándar CAS o Gini--$M_{20}$ y convertir a $C_{clump} = Q$.

5.  **Espesor/asimetría** $\mathbf{T}$ **.** A partir de indicadores verticales (cuando están disponibles) o relaciones de ejes menor/mayor corregidas por inclinación; convertir a $C_{T}$ (más delgado, simétrico ⇒ mayor coherencia).

6.  **Textura cinemática** $\mathbf{K}$ **.** Potencia en flujos no circulares de campos de velocidad residuales después de sustraer el modelo axisimétrico; invertir a $C_{K} = 1 - NCF$.

> **Vector de características** agregado por anillo:
>
> $$z_{j} = \left\lbrack E^{*},C_{D},\ C_{\text{mode}},C_{\text{clump}},C_{T},C_{K} \right\rbrack_{j}\quad\Sigma_{j} = \text{covarianza de errores de medición.}$$
>
> **5.3 Mapeo indicador-a-**$\mathbf{\alpha}$
>
> Mapeamos $z_{j}$ a un exponente de coherencia **provisional** ${\overline{\alpha}}_{j}$ mediante una función monótona $\mathcal{M}$. Dos opciones (pre-registradas; ambas permitidas):

(a) **Mapa monótono paramétrico (transparente):**

$${\widetilde{\alpha}}_{j} = \alpha_{0} + \sum_{k}^{}{w_{k}g_{k}\left( z_{jk} \right)};\quad g_{k}\text{ monótono},w_{k} \geq 0,$$

> con $g_{k}$ elegido como transformaciones de identidad o logísticas y $w_{k}$ ajustados en **subconjuntos de calibración** (galaxias/anillos donde la prueba de pendiente ya se cumple a alto S/N). Imponer priors $\alpha \in \lbrack 0.8,3.2\rbrack$ y ${\mid \mid w \mid \mid}_{1} = 1$ para interpretabilidad.

(b) **Ensamble basado en rangos (robusto):**

> $${\widetilde{\alpha}}_{j} = \alpha_{0} + \lambda\ median_{k}\ rank\left( z_{jk} \right),$$
>
> lo cual reduce la sensibilidad a valores atípicos y escalas heterogéneas.
>
> **Incertidumbre.** Propagar $\Sigma_{j}$ a $\sigma_{\widetilde{\alpha},\ j}$ vía método delta (opción a) o bootstrap (opción b).
>
> **5.4 Refinamiento por verificación de pendiente ("cerrar el ciclo")**
>
> Para cada anillo $A_{j}$, tenemos mediciones locales $v(L)$. Dentro de un **bin de coherencia** $B$ (colección de anillos adyacentes con $\widetilde{\alpha}$ similar), ajustar
>
> $$\log v = c_{B} + \left( 1 - {\widehat{\alpha}}_{B} \right)\log L$$
>
> usando pendiente de Theil--Sen + pérdida robusta de Huber con corrección de **errores en variables** (SIMEX) para $L$ si las incertidumbres de deproyección no son despreciables. Comparar ${\widehat{\alpha}}_{B}$ con el ${\widetilde{\alpha}}_{j}$ basado en indicadores de sus miembros.
>
> **Regla de aceptación (bin *B*):**

-   **APROBADO:** ∣${\widehat{\alpha}}_{B}{- median}_{j \in B}{\widetilde{\alpha}}_{j} \mid \leq 0.2$ y los IC se solapan;

-   **TENTATIVO:** discrepancia 0.2 − 0.4 o IC amplio;

-   **RECHAZADO:** \>0.4 de discrepancia o signo de pendiente opuesto.

Luego definimos la estimación **final** por anillo

$${\widetilde{\alpha}}_{j} = shrink({\widetilde{\alpha}}_{j},\ {\widehat{\alpha}}_{B})$$

vía una combinación convexa simple ponderada por incertidumbres.

**5.5 Verificación de colapso y diagnósticos residuales**

Dentro de cada bin de coherencia $B$, calcular

$${y(L) = v(L)L}^{{\widehat{\alpha}}_{B} - 1}$$

**Predicción:** $y(L)$ es **plano** vs. $L$. Regresar $log\ y$ sobre $log\ L$; una pendiente residual con $\mid m \mid > 0.1$ (IC al 95% excluyendo 0) señala **mala especificación del modelo** (p. ej., $\alpha$ variable dentro del bin, sistemáticas geométricas).

**Residuos secundarios:** Examinar $y(L)$ vs. (i) error de inclinación, (ii) métrica de difuminado de haz, (iii) corrección de deriva asimétrica. Correlaciones significativas indican que los pipelines de reducción necesitan ajuste.

**5.6 Estrategia de agrupamiento y tamaño de muestra**

-   **Anillos:** espaciamiento logarítmico con $\Delta\ \log\ L = 0.08 - 0.12$, asegurando $\geq 5$ elementos de resolución a través del ancho.

-   **Bins de coherencia:** agrupar anillos adyacentes por $\widetilde{\alpha}$ usando clustering Ward 1-D con restricción de **contigüidad en radio**; objetivo $\geq 5$ anillos por bin.

-   **Meta entre galaxias:** por tipo de bin (coherencia baja/media/alta), agrupar estimaciones de pendiente a través de galaxias usando metaanálisis de efectos aleatorios para reportar un valor poblacional de $1 - \alpha$.

**5.7 Incertidumbre, QA y exclusiones**

-   **Incertidumbre de inclinación/PA:** propagar vía Monte Carlo (muestrear $i$, PA de las posteriores; reajustar pendientes).

-   **Incertidumbre de distancia:** afecta intersecciones más que pendientes; aún así se propaga en el MC.

-   **Umbral de resolución:** excluir anillos con menos de 3 elementos de resolución a través del ancho radial o con FWHM del PSF $> \ 0.5\,\Delta R$.

-   **Difuminado de haz:** requerir factor de corrección $< 20\%$ o marcar como TENTATIVO.

-   **Deriva asimétrica:** aplicar solo cuando la fracción de dispersión $> 0.15$; de lo contrario, la rotación del gas se usa tal cual.

**Criterios de detención (por galaxia):** marcar galaxia como **NO APTA** si $< 2$ bins de coherencia pasan tanto las pruebas de pendiente como de colapso después del QA.

**5.8 Pseudocódigo (contrato de análisis)**

```
para cada galaxia G:
    preprocesar_imágenes_y_cinemática(G)
    anillos = crear_anillos_log(G, dlogL=0.1)

    para cada anillo A_j en anillos:
        z_j, Sigma_j = calcular_características_estructura(A_j)
        talpha_j, sigma_talpha_j = mapear_características_a_alpha(z_j, Sigma_j) # Sec. 5.3

    # agrupamiento por coherencia con restricción de contigüidad
    bins = agrupar_adyacentes_por_alpha(talpha_j, k_min="5 anillos")

    resultados = []

    para bin B en bins:
        # Ley de pendiente
        m, CI_m = pendiente_EIV_robusta(log v vs. log L en B)
        alpha_pendiente = 1 - m

        # Comparar con alpha del indicador
        alpha_indicador = mediana(talpha_j en B)
        estado = APROBADO si |alpha_pendiente - alpha_indicador| <= 0.2 y IC se solapan sino TENTATIVO/RECHAZADO

        # Colapso
        y = v * L**(alpha_pendiente - 1)
        m_c, CI_c = pendiente(log y vs. log L)
        colapso_ok = (|m_c| <= 0.1 con IC incluyendo 0)

        resultados.append({alpha_pendiente, CI_m, alpha_indicador, estado, colapso_ok})

    # Alpha final por anillo mediante contracción hacia pendiente del bin
    para j en anillos:
        alpha_final[j] = contraer(talpha_j, alpha_pendiente_del_bin(j), sigmas)

    exportar(G, resultados, alpha_final, banderas_QA)
```

**5.9 Entregables por galaxia**

-   **Mapa:** $\widehat{\alpha}(L)$ con banda de $1\sigma$.

-   **Gráfico:** $log\ v$ vs. $log\ L$ coloreado por bins de coherencia; pendientes anotadas con $1 - \alpha/2$.

-   **Panel de colapso:** ${v\, L}^{\widehat{\alpha} - 1}$ vs. $L$ por bin.

-   **Tabla:** para cada bin $-{\widehat{\alpha}}_{indicador}$, ${\widehat{\alpha}}_{pendiente}$, ICs, veredicto de colapso, banderas QA.

**5.10 Reglas de interpretación (por bin)**

1.  **APROBADO (apoyo fuerte):** pendiente $= 1 - \widehat{\alpha}$ (IC se solapan) y colapso plano; sin banderas QA severas.

2.  **PARCIAL:** pendiente coincide pero colapso débil (sugiere leve deriva de α o residuos geométricos).

3.  **RECHAZADO:** pendiente discrepa o colapso muestra tendencia significativa; verificar QA; si persiste, RTM no respaldado en ese bin.

**6. Comparación con expectativas de solo gravedad**

Este capítulo convierte la ley de pendiente de RTM en **contrastes directos y falsificables** con dos líneas base:

-   **RG + solo bariones (sin materia oscura):** dinámica clásica con distribución de masa luminosa; las asíntotas de rotación dependen de la extensión bariónica.

-   **RG + halos de materia oscura (práctica ΛCDM):** agregar un halo paramétrico (p. ej., NFW, Burkert) y ajustar parámetros libres por galaxia.

RTM mantiene la gravedad intacta pero agrega un **campo de coherencia** $\alpha(L)$ que modifica los **tiempos operacionales**. Los discriminantes a continuación se formulan como **pruebas de pendiente** y **verificaciones de colapso** que no dependen de la normalización absoluta.

**6.1 Asíntotas de disco externo: planitud sin halos vs. caída kepleriana**

**Expectativa de solo gravedad.** Para discos finitos, más allá de la mayor parte de los bariones se espera ${v(L) \propto L}^{- 1/2}$ (aproximación a kepleriano con correcciones geométricas). En la práctica, los modelos puramente bariónicos luchan por mantener $v$ **plano** sobre décadas en $L$ sin masa añadida.

**Predicción RTM (P1 revisada).** Si el medio externo es **débilmente coherente** ($\alpha \rightarrow 1$), entonces

$$\frac{\partial\log v}{\partial\log L} = 1 - \alpha \rightarrow 0 \Rightarrow v(L) \approx \text{const.}$$

**Discriminante D1 (auditoría de pendiente).** En **anillos externos** seleccionados por indicadores de baja coherencia, ajustar $log\ v$ vs. $log\ L$.

-   **RTM APROBADO:** la pendiente $m$ se agrupa estrechamente cerca de 0 **y** el colapso ${v\ L}^{\alpha - 1}$ es plano.

-   **Solo bariones RECHAZADO:** los mismos datos, los mismos anillos, requerirían $m \approx - 1/2$ a menos que se añada masa oculta.

-   **Ambigüedad de materia oscura:** los halos pueden ajustar $m \approx 0$, pero los **mismos anillos** también deben pasar D2--D4 abajo para distinguir RTM.

**6.2 Diversidad de curva interna: coherencia vs. ajuste de halo**

**Hecho observado.** Galaxias con masa bariónica similar muestran **formas internas diversas** (ascensos rápidos/lentos). Los ajustes de materia oscura acomodan esto con concentración de halo/perfiles contraídos; MOND invoca aceleración local; **ambos** requieren *ajuste* por galaxia.

**Mecanismo RTM.** Dentro de barras/bulbos/grumos, $\alpha(L) >$ `<!-- -->`{=html}1 eleva los tiempos orbitales locales, produciendo

$$m = \frac{\partial\log v}{\partial\log L} = 1 - \alpha < 0\quad\text{(ascensos más pronunciados}\text{/}\text{caídas más suaves dependen de la geometría)}.$$

El punto clave es la **covariación**: la **pendiente** interna debe rastrear el $\mathbf{\alpha}$ **derivado de la estructura**, no un parámetro libre de halo.

**Discriminante D2 (acoplamiento estructura--pendiente).** Después de controlar por masa y geometría, regresar los residuos de pendiente interna $\Delta m$ sobre indicadores de coherencia (potencia de barra $P_{2}$, entropía multiescala $E^{\star}$, grumosidad $Q$, etc.).

-   **RTM APROBADO:** corr($\Delta m$, $\widehat{\alpha}$) es significativa y positiva en magnitud (más coherencia → $m$ más negativo o ascenso/aplanamiento más pronunciado, según geometría), y permanece después de parcializar tamaño y densidad superficial.

-   **Materia oscura/MOND RECHAZADO:** los residuos se alinean principalmente con parámetros de halo/aceleración, y **no** con la estructura una vez que los bariones están fijados.

**6.3 La relación bariónica de Tully--Fisher (bTFR): anatomía residual**

**Comportamiento base.** La bTFR es estrecha pero muestra **residuos**. En ajustes de materia oscura, los residuos se correlacionan con **concentración/espín del halo**; en MOND, con matices de **función de interpolación/aceleración**.

**Reformulación RTM.** Si $v$ se muestrea donde $\alpha \rightarrow 1$, la bTFR de **orden principal** se cumple con residuos mínimos. Si se muestrea más adentro ($\alpha$ más alto), la $v$ medida está **sistemáticamente sesgada** respecto al valor asintótico.

**Discriminante D3 (vínculo residuo--coherencia).**

-   Calcular residuos $\Delta\ log\ v$ de un ajuste bTFR a nivel de toda la galaxia.

-   Probar $\Delta\ log\ v$ vs. un **índice de desajuste de** $\mathbf{\alpha}$, p. ej., $\delta_{\alpha} \equiv \widehat{\alpha}(R_{meas}) - 1.$

    -   **RTM APROBADO:** $\Delta\ log\ v$ se correlaciona con $\delta_{\alpha}$ (muestrear dentro de la zona coherente deprime $v$, residuo negativo), y la correlación **desaparece** al medir $v$ en el radio de **pendiente cero** de cada galaxia.

    -   **Materia oscura/MOND RECHAZADO:** la correlación residuo--$\delta_{\alpha}$ es débil/ausente una vez que masa y tamaño están controlados.

**6.4 Colapso entre anillos vs. libertad paramétrica**

**Colapso RTM.** Dentro de cualquier bin de coherencia $B:\ y(L) = v(L)\, L^{{\widehat{\alpha}}_{B} - 1}$ debe ser **plano**. Esta es una restricción **funcional** más fuerte que ajustar una intersección.

**Discriminante D4 (colapso por bin).**

-   **RTM APROBADO:** pendientes residuales $\mid mB \mid \leq 0.1$ (IC incluye 0) a través de bins y galaxias; meta-pendiente agrupada de efectos aleatorios consistente con 0.

-   **Materia oscura/MOND RECHAZADO (como prueba de mecanismo):** Aunque los halos/leyes de aceleración pueden reproducir **una curva**, **no** predicen colapsos por bin ligados a coherencia **medida independientemente**. La falla en colapsar después del condicionamiento por $\widehat{\alpha}$ cuenta contra RTM; el éxito cuenta como una signatura única.

**6.5 Elípticas y perfiles de dispersión: Jeans vs. coherencia**

**Línea base de Jeans.** Con anisotropía $\beta(r)$ y perfil de masa $M(r),\ \sigma(r)$ se sigue de la ecuación de Jeans; la materia oscura agrega masa a $r$ grande, empinando/aplanando perfiles por elección de halo y $\beta$.

**Regla de pendiente RTM para dispersiones.** En bins donde $\alpha(r)$ es aproximadamente constante,

$$\frac{\partial\log\sigma}{\partial\log r} = 1 - \alpha\quad\left( \text{salvo correcciones de anisotropía} \right)$$

Discriminante D5 (pendiente de dispersión vs. estructura).

-   **RTM APROBADO:** la pendiente de $\sigma$ rastrea $\widehat{\alpha}$ desde la textura fotométrica (núcleos/discos embebidos → mayor $\alpha\  \rightarrow$ pendiente más positiva/menos negativa), y los colapsos por bin ${\sigma\ r}^{\widehat{\alpha} - 1\ }$ se cumplen.

-   **Materia oscura/MOND RECHAZADO:** Los cambios necesarios se absorben en $M(r)$ o $\beta(r)$ con poco/ningún vínculo con la estructura **medida**.

**6.6 Donde las líneas base y RTM coinciden (verificaciones de cordura)**

Hay regímenes donde **todos** los modelos predicen comportamiento similar; los usamos como **pruebas nulas**:

-   **Controles keplerianos:** binarias amplias, sistemas planetarios externos, cúmulos globulares a $r$ grande. La coherencia es irrelevante; RTM debe reducirse a las pendientes clásicas.

-   **Núcleos de cuerpo sólido:** efectos puramente geométricos en regiones muy centrales pueden imitar $m \approx + 1$. RTM **no** reclama crédito allí; las pruebas deben evitar radios sub-resolución.

-   **HI externo ultra-difuso:** si los indicadores de estructura confirman $\alpha \approx 1$, **todos** los modelos permiten $m \approx 0$. Los discriminantes entonces se desplazan a la **anatomía residual de la bTFR** (Sec. 6.3) y el **colapso** (Sec. 6.4).

**6.7 Matriz de decisión (por galaxia, por bin)**

| **Prueba** | **Evidencia a favor de RTM** | **Evidencia contra RTM** | **Qué dirían materia oscura/MOND** |
|----|----|----|----|
| **D1:** pendiente externa | $m \approx 0$ en bins de bajo $alpha$ **y** colapso | *m≈−1/2* o sin colapso | La materia oscura puede ajustar *m≈0* pero no predice colapso |
| **D2:** diversidad interna | $`\Delta m`$ se correlaciona con $`\widehat{\alpha}`$ (estructura) | $`\Delta m`$ no correlacionado con estructura | Materia oscura: parámetros de halo; MOND: escala de aceleración |
| **D3:** residuos de bTFR | $`\Delta\ log\ v\  \leftrightarrow \delta_{\alpha}`$ desaparece en radio de pendiente cero | Sin relación con $`\delta_{\alpha}`$ | Materia oscura: residuos $`\leftrightarrow`$ concentración/espín de halo |
| **D4:** colapso | $`{v\ L}^{\widehat{\alpha} - 1}`$ plano por bin | Pendiente residual ( | m_B |
| **D5:** dispersiones | *∂ log σ / ∂ log r=1−*$`\widehat{\alpha}`$ y colapso | Sin vínculo pendiente--estructura | Ajustan *M(r), β(r)* post-hoc |

**6.8 Modos de falla pre-registrados**

Las afirmaciones galácticas de RTM están **falsificadas** si, después del QA (Sec. 5):

1.  Los bins externos de bajo α muestran pendientes **distintas de cero** inconsistentes con 0 (falla D1) **y** no colapsan (falla D4).

2.  Los residuos de pendiente interna **no** se correlacionan con $\widehat{\alpha}$ derivado de la estructura una vez controlados masa/tamaño (falla D2).

3.  Los residuos de la bTFR son **independientes** de $\delta_{\alpha}$ y permanecen así incluso al muestrear en el radio de pendiente cero (falla D3).

4.  Las pendientes de dispersión en elípticas no muestran **ninguna** relación con $\widehat{\alpha}$ basado en textura (falla D5).

> Cualesquiera dos fallas independientes bajo buen QA marcan RTM como **no respaldado** a escalas galácticas; pasar D1--D4 a través de una muestra diversa constituye **respaldo fuerte**.

**7. Lentes gravitacionales y cúmulos: verificaciones de consistencia**

RTM afirma alterar los **tiempos operacionales** (relojes orbitales) vía el exponente de coherencia $\alpha(L)$, no la **curvatura** del espacio-tiempo. Si es cierto, las **lentes gravitacionales**---que dependen de la curvatura generada por el tensor de energía-momento---deberían seguir rastreando la **distribución de masa bariónica** (más cualquier masa genuinamente no bariónica, si está presente) independientemente de $\alpha$. Este capítulo establece pruebas que comparan la **masa inferida por lentes** con la **cinemática reinterpretada bajo RTM**, desde galaxias hasta cúmulos. Cualquier **brecha de masa persistente y coherente** después de condicionar por $\widehat{\alpha}(L)$ constituye un **límite de alcance** o **falsificación** directa en esas escalas.

**7.1 Relojes vs. curvatura: el principio rector**

-   **Lo que RTM cambia:** el mapeo ${T \propto L}^{\alpha(L)}$ que gobierna los tiempos orbitales/de relajación. Los observables cinemáticos que dependen de períodos o deriva (velocidades de rotación, dispersiones, frecuencias epicíclicas/verticales) se modifican vía ${v \propto L}^{1 - \alpha}$ o ${\sigma \propto L}^{1 - \alpha}$ **dentro de bins de coherencia**.

-   **Lo que RTM no cambia:** las ecuaciones de campo de Einstein y las geodésicas que determinan la deflexión de la luz y las lentes. Así, los **mapas de masa por lentes** deberían ser consistentes con los **bariones** (dentro de las sistemáticas conocidas) a menos que exista masa real no vista o RTM falle en describir la dinámica.

**Prueba operacional.** Construir, para cada sistema, dos inferencias de masa:

1.  $M_{lens}(R)$ de lentes fuertes/débiles (o dinámica+rayos X en cúmulos).

2.  $M_{kin}^{RTM}(R)$ de velocidades/dispersión observadas **después** de reinterpretar la cinemática con $\widehat{\alpha}(R)$.

La consistencia requiere $M_{kin}^{RTM} \approx M_{lens}$ dentro de las incertidumbres; un sesgo sistemático que **sobreviva** al condicionamiento por $\alpha$ señala un límite de RTM o masa extra genuina.

**7.2 Galaxias con lentes fuertes (anillos de Einstein y cuádruples)**

**Configuración.** Elegir lentes con anillos de Einstein/cuádruples de alta calidad ($M_{lens}(R_{E})$ preciso). Obtener cinemática IFU para construir $\widehat{\alpha}(R)$ (Sec. 5).

Prueba de consistencia RTM SL-1 (masa encerrada en $R_{E}$).

-   Calcular $M_{kin}^{RTM}(R_{E})$ del soporte rotacional/de dispersión observado usando la **ley de velocidad RTM** dentro de bins de coherencia que intersectan $R_{E}$.

-   **Aprobado:** $M_{kin}^{RTM} - M_{lens} \mid /M_{lens} \leq \varepsilon\ $ (pre-registrado $\varepsilon$, p. ej., 15%)

-   **Rechazado:** sobreestimaciones o subestimaciones sistemáticas a través de la muestra que no pueden rastrearse a calibración de $\alpha$ o sistemáticas de anisotropía.

**Discriminante RTM SL-2 (colapso anular).\
**Dentro de un anillo alrededor de $R_{E}$ con $\widehat{\alpha}$ aproximadamente constante, la cantidad

$${y(R) = v(R)R}^{\alpha - 2/1}$$

debería ser **plana** vs. $R$. La falla en colapsar mientras la masa de la lente está bien restringida argumenta que la reinterpretación cinemática de RTM es inadecuada **a la escala de la lente**.

**Adición de retardo temporal SL-3.** Para cuásares con lentes gravitacionales con retardos temporales medidos, verificar que las inferencias cosmográficas (p. ej., $H_{0}$) permanezcan **sin cambio** al cambiar el modelo dinámico a RTM, ya que los retardos dependen de **curvatura + diferencias de potencial**, no de relojes orbitales. Cualquier cambio indica doble conteo (permitir incorrectamente que $\alpha$ se filtre en las lentes).

**7.3 Lentes débiles en galaxias de disco (halos apilados)**

**Configuración.** Usar perfiles de cizallamiento de lentes débiles apilados de grandes muestras de discos agrupados por **coherencia estructural** (p. ej., fuerza de barra, métricas de textura) para obtener $M_{lens}(R)$ a decenas--cientos de kpc.

**Prueba de consistencia RTM WL-1 (bins externos).**\
En **anillos externos de bajo** $\widehat{\mathbf{\alpha}}$ (donde las curvas de rotación se aplanan), RTM predice **cinemática plana** sin curvatura extra. Por lo tanto, la señal de **lentes** a $R$ grande debería ser explicable solo por **bariones + gas conocido** si la curvatura realmente solo rastrea masa.

-   **Aprobado:** $M_{lens}(R)$ apilado consistente con mapas bariónicos y con $M_{kin}^{RTM}(R)$.

-   **Alcance/Rechazado:** un exceso robusto en cizallamiento **después** del condicionamiento por $\alpha$ indica masa más allá de los bariones---o el alcance de RTM termina aquí o se necesita masa oscura.

**Verificación cruzada interna WL-2 (división por estructura).**\
Dividir discos a masa estelar fija por coherencia (alta vs. baja barra/textura).

-   RTM espera **halos de lentes débiles similares** (dado que las lentes ignoran $\alpha$) pero **pendientes cinemáticas internas diferentes**.

-   Si los perfiles de lentes *también* se dividen sistemáticamente con la coherencia a mapas bariónicos fijos, eso sugiere una correlación entre **estructura** y **masa verdadera** (no un efecto solo de $\alpha$), estrechando el alcance.

**7.4 Cúmulos de galaxias: donde RTM puede (o no) aplicarse**

**Verificación de realidad.** Los cúmulos ricos exhiben masas de lentes fuertes/débiles y masas hidrostáticas de rayos X que **exceden** los bariones. Si RTM solo retemporizara los **relojes orbitales** dentro de bariones estructurados, **no debería** borrar los déficits de masa en cúmulos---incluso si $\alpha$ afecta alguna dinámica intracúmulo.

**Prueba de cúmulos CL-1 (presupuesto de masa).**

-   Construir $M_{lens}(R)$ y $M_{X}(R)\ (rayos\ X).$

-   Medir campos $\widehat{\alpha}(R)$ de la textura del ICM (fluctuaciones de presión/densidad, espectros de potencia) y subestructura galáctica.

-   Calcular $M_{kin}^{RTM}(R)$ de las dispersiones galácticas usando ${\sigma \propto R\,}^{1 - \widehat{\alpha}}$ en **bins de coherencia** (Jeans con relojes RTM).

-   **Resultado esperado:** incluso con RTM, una **masa residual significativa** permanece en cúmulos---la señal clásica de materia oscura.

-   **Interpretación:** la **condición de alcance** de RTM: es una re-temporización cinemática a **escala galáctica**, no un reemplazo de la materia oscura a escalas de cúmulos. Si, improbablemente, RTM borrara la brecha de masa de los cúmulos, la consistencia lentes--dinámica se rompería (contradiciendo la masa basada en curvatura).

**Fusiones tipo Bullet CL-2.** En sistemas donde ocurren desplazamientos gas--galaxia, los picos de lentes siguen la masa no colisional. RTM predice **ningún desplazamiento** de picos de lentes con $\alpha$; cualquier intento de usar $\alpha$ para imitar el desplazamiento dejaría incorrectamente que los relojes alteren la curvatura---**no permitido**.

**7.5 Algoritmo de reconciliación cinemática--lentes (por sistema)**

1.  Medir $\widehat{\alpha}(R)$ **:** construir bins de coherencia a partir de indicadores de estructura (Sec. 5).

2.  **Dinámica inferida por RTM:** dentro de cada bin, ajustar pendientes $m = 1 - \widehat{\alpha}$, verificar colapso ${v\ R}^{\widehat{\alpha} - 1}$, y recuperar $M_{kin}^{RTM}(R)$ con correcciones EIV y priors de anisotropía (para dispersiones).

3.  **Masa por lentes:** obtener $M_{lens}(R)$ (fuerte/débil) con covarianzas completas.

4.  **Comparar:** calcular ${\Delta(R) = M}_{kin}^{RTM}(R) - M_{lens}(R)$ y su incertidumbre; reportar residuos **por bin** en lugar de un solo número global.

5.  **Decisión:**

-   **CONSISTENTE:** $\mid \Delta \mid /M_{lens} \leq \varepsilon$ en la mayoría de bins y sin tendencia con $\widehat{\alpha}$.

-    **LÍMITE DE ALCANCE:** los residuos se concentran a **radios de escala de cúmulo** o en sistemas donde $\widehat{\alpha}$ no puede estimarse de manera estable.

-    **FALSIFICADO:** residuos coherentes y significativos a través de muchos bins a **escala galáctica** donde el QA pasa y $\widehat{\alpha}$ es estable.

**7.6 Retardos temporales y pruebas relativistas (cordura)**

-   **Retardos temporales de lentes fuertes:** dependen del **potencial de Fermat** (curvatura + geometría). RTM **no debe** alterar los retardos predichos cuando el mapa de masa es fijo. Por lo tanto, reajustamos los retardos bajo RG con la misma masa y mostramos **invariancia** al reemplazar la dinámica newtoniana con RTM para los movimientos **estelares/gaseosos**.

-   **Restricciones PPN/sistema solar:** en regímenes de baja coherencia relevantes para pruebas del sistema solar, $\alpha$ se reduce a su línea base clásica y las restricciones de lentes/deflexión permanecen **sin cambio**---una verificación de cordura incorporada.

**7.7 Resultados pre-registrados (aprobación/rechazo)**

-   **APROBADO (escala galáctica):**

    i.  Los bins externos de bajo $\widehat{\alpha}$ muestran $m \approx 0$ **y** colapso;

    ii. $M_{kin}^{RTM}(R)$ coincide con $M_{lens}(R)$ en anillos/cuádruples dentro de $\leq 15\%$;

    iii. Los apilamientos de lentes débiles a bariones fijos **no** se dividen por coherencia, mientras que las pendientes cinemáticas **sí**

```{=html}
<!-- -->
```
-   **ALCANCE (cúmulos):**\
    RTM **no** elimina la brecha de masa de cúmulos; $M_{lens}(R)$ excede bariones + cinemática-RTM. RTM queda así acotado a cinemática **a escala galáctica** a menos que se introduzca física adicional.

-   **RECHAZADO (escala galáctica):**\
    Brechas de masa lentes--cinemática consistentes y significativas **después** del condicionamiento por $\alpha$, o no-colapsos por bin acoplados con estimaciones estables de $\widehat{\alpha}$ y buen QA, falsifican RTM como mecanismo explicativo para perfiles de rotación/dispersión galácticos.

**Conclusión.** Las lentes son el **guardarraíl** de RTM: al separar **relojes** de **curvatura**, podemos saber cuándo la re-temporización por coherencia es suficiente (galaxias) y dónde no (cúmulos). Pasar las verificaciones de consistencia de lentes hace de RTM una reinterpretación creíble y de alcance delimitado de la cinemática galáctica; fallarlas traza un límite claro y preserva la gravedad estándar donde debe permanecer intacta.

**8. Crecimiento de estructura cósmica (Esbozo)**

Este capítulo esboza cómo un **campo** $\mathbf{\alpha}$---un exponente de coherencia espacialmente variable ligado a la organización bariónica---podría modular **escalas temporales** durante el ensamblaje de galaxias y sus subestructuras sin alterar la gravedad. La postura sigue siendo **pendiente primero**: RTM predice **cuán rápido** se desarrollan los procesos a una escala dada, no **que** aparezcan nuevas fuerzas. La sección cierra con **observables** y **pruebas de falla** que mantienen el programa falsificable.

**8.1 Relojes de colapso bajo RTM**

Sea $t_{coll}(L)$ el tiempo característico para que un parche bariónico autogravitante de tamaño $L$ proceda del crecimiento lineal a la no linealidad (fragmentación/condensación). La teoría estándar provee un tiempo dinámico $t_{dyn} \sim 1/\sqrt{G\rho}$ y retrasos adicionales por transporte de momento angular, enfriamiento, turbulencia. RTM trata el **tiempo operacional** como

$$t_{\text{coll}}(L) = t_{\text{dyn}}(L)\left( \frac{L}{L_{0}} \right)^{\alpha(L) - \alpha_{0}}\Theta$$

donde $\alpha_{0}$ es una banda base (débilmente coherente) y $\Theta$ agrega microfísica adimensional mantenida fija **dentro** de un bin de coherencia. Consecuencias:

-   Regiones con **mayor coherencia** ($\alpha > \alpha_{0}$) **alargan** los relojes de colapso a esa *misma escala*, retrasando el crecimiento de barras/espirales o la condensación de grumos respecto a zonas difusas.

-   Los **gradientes** $\nabla\alpha$ siembran **temporización diferencial** a través de los radios, imprimiendo desfases de fase entre barras, espirales y alabeos.

**8.2 Transporte de momento angular y líneas temporales de barras**

La formación de barras requiere redistribución de momento angular. Sea $t_{J}(L)$ la escala temporal característica para el transporte de $J$ en un anillo de ancho $\sim L$. Con RTM:

$$t_{J}(L) \propto L^{\alpha(L)}\quad \Rightarrow \quad\frac{\partial\log t_{J}}{\partial\log L} = \alpha(L).$$

**Predicciones.**

-   **Secuenciación de adentro hacia afuera.** Si los discos internos son más coherentes ($\alpha_{in} > \alpha_{out}$), las barras/espirales internas **se retrasan** respecto al crecimiento de patrones externos; inversamente, si la retroalimentación destruye la coherencia interna ($\alpha_{in} \rightarrow 1$), las barras emergen **antes** de lo que los tiempos seculares estándar sugerirían.

-   **Longitud de barra vs. gradiente de** $\mathbf{\alpha}$. Los semiejes mayores de las barras se anticorrelacionan con $\nabla\alpha$: **caídas** más fuertes hacia afuera en $\alpha$ (interno coherente → externo difuso) limitan el crecimiento de la barra antes (el disco externo supera al interno en desprendimiento de $J$).

**Observables.** A masa y fracción de gas fijas, la **fracción de barras** y la **longitud de barra** se correlacionan con la **forma** de $\widehat{\alpha}(R)$: barras largas y fuertes prefieren perfiles de $\alpha$ **más planos**; barras cortas/débiles aparecen donde $\alpha$ cae rápidamente con el radio.

**8.3 Formación de grumos, migración y discos gruesos**

Los grumos masivos de formación estelar en discos de alto $z$ migran hacia adentro en una escala temporal $t_{mig}$ fijada por torques y fricción dinámica.

**Modulación RTM.**

$$t_{\text{mig}}\left( L_{\text{grumo}} \right) \sim t_{\text{dyn}}\left( \frac{L_{\text{grumo}}}{L_{0}} \right)^{\alpha - 1}$$

por lo que a tamaño de grumo fijo, **mayor** $\mathbf{\alpha}$ **local ralentiza la migración**, permitiendo que los grumos **vivan más** y engrosen los discos mediante dispersión prolongada.

**Predicciones.**

-   **Longevidad de grumos vs.** $\mathbf{\alpha}$ **.** A densidad superficial fija, los discos con mayor $\widehat{\alpha}$ sostienen mayores **tiempos de vida de grumos** y muestran capas estelares **más gruesas** antes.

-   **Gradientes de edad.** Si $\alpha$ disminuye con el radio, los grumos internos ($\alpha$ mayor) envejecen **más** in situ que los grumos externos ($\alpha$ menor) para el mismo tiempo de retrospección---una tendencia edad--radio **invertida** respecto a las expectativas de pura fricción dinámica.

**8.4 Planos de satélites, alabeos y desfases de fase**

Los gradientes de coherencia pueden **fijar la fase** de ciertas familias orbitales.

**Predicciones.**

-   **Planos de satélites.** Si el disco externo/CGM del huésped exhibe un campo $\mathbf{\alpha}$ **anisotrópico** (p. ej., a lo largo de filamentos), las órbitas de satélites **persisten** preferentemente en ese plano (períodos operacionales más largos para difusión fuera del plano), aumentando la probabilidad de **alineamientos planares aparentes** sin invocar anisotropías especiales de materia oscura.

-   **Fase de alabeos.** Las zonas radiales donde $\nabla\alpha$ es mayor deberían mostrar **desfases de fase** entre alabeos de HI y flexiones estelares; el signo del desfase se invierte con el signo de $\nabla\alpha$.

-   **Lopsidedness.** Los modos persistentes $m = 1$ se correlacionan con variaciones **azimutales** en $\alpha$ (barras + grumos en un lado), produciendo **asimetrías cinemáticas** que rastrean mapas de estructura.

**8.5 Historias de formación estelar (SFHs) y** $\mathbf{\alpha}$

Dado que $t_{coll}$ y $t_{J}$ se estiran con $\alpha$, las **SFHs** heredan **signaturas de coherencia**:

-   **De adentro hacia afuera vs. de afuera hacia adentro.** Discos con alto $\alpha$ interno y bajo $\alpha$ externo tienden **de afuera hacia adentro** en temporización de brotes (los anillos externos se encienden antes); la forma inversa de $\alpha$ invierte la tendencia.

-   **Intermitencia.** Parches de bajo $\alpha$ (difusos/turbulentos) tienen **ciclos más cortos**, aumentando la intermitencia e impulsando mayor potencia de HI/H$\alpha$ a escalas pequeñas; parches de alto $\alpha$ suavizan las SFHs.

-   **Dispersiones de metalicidad.** La migración prolongada bajo alto $\alpha$ amplía las distribuciones de metalicidad a un radio dado (tiempos de mezcla de fases más largos), comprobable con mapas de metalicidad IFU.

**8.6 Tendencias a alto corrimiento al rojo**

A $z \gtrsim 1$, los discos ricos en gas son grumosos y turbulentos. Dos escenarios estilizados:

-   **Escenario A (bajo** $\mathbf{\alpha}$ **global).** Si los discos tempranos son en gran parte **difusos** (la retroalimentación destruye la coherencia), $\alpha \approx 1$ sobre amplios radios $\Rightarrow$ crecimiento de patrones **rápido**, tiempos de vida de grumos **cortos**, aproximación más rápida a rotación plana más allá de bulbos compactos.

-   **Escenario B (** $\mathbf{\alpha}$ **jerárquico).** Si las estructuras anidadas (grumos gigantes, cadenas) aumentan la coherencia ($\alpha > 1$) localmente, las barras y los grumos de larga vida deberían **coexistir** tempranamente; las pendientes de rotación exhiben fuerte **diversidad radial** que **se desvanece** a medida que $\alpha \rightarrow 1$ con el tiempo cósmico (asentamiento del disco).

**Palanca observable.** Comparar la **evolución** de la **distribución de pendientes** $m(R) = \partial\ log\ v\ /\ \partial\ log\ R$ a través del corrimiento al rojo después de condicionar por **indicadores de** $\mathbf{\alpha}$. RTM predice que la **dispersión** en $m$ a masa fija se estrecha a medida que los campos de $\alpha$ **se aplanan** con el tiempo.

**8.7 Esbozo de simulación (cómo probar lo anterior)**

**Integrador orbital consciente de α.** Tomar un código estándar de N-cuerpos+gas o un banco de pruebas no colisional; en cada paso, reescalar los **avances temporales** en una celda por $dt' = {dt(L/L_{0})}^{{\alpha(x) - \alpha}_{0}}$. Mantener las fuerzas **sin cambio**. Alimentar α($x$) desde (i) perfiles analíticos ($\alpha$ alto centrado en la barra), (ii) mapas de indicadores derivados de la luz, o (iii) reglas de auto-actualización (la coherencia crece con la densidad superficial sostenida). Leer:

-   Pendientes de rotación y **colapso** ${vR}^{\alpha - 1}$ dentro de bins;

-   Tiempo de formación de barra vs. $\nabla\alpha$;

-   Tiempos de vida de grumos y engrosamiento del disco vs. $\alpha$ local;

-   Desfases de fase de alabeos vs. $\nabla\alpha$.

**Falsificación dentro del sandbox.** Si mantener las fuerzas fijas y solo **re-temporizar** no puede reproducir ninguna de las secuencias observadas (p. ej., patrones de emergencia de barras) cuando los campos de $\alpha$ se ajustan a estructura **medida**, la historia de RTM a nivel de crecimiento se debilita.

**8.8 Resumen observable y condiciones de falla**

| **Fenómeno** | **Signatura RTM** | **Cómo medir** | **Falla si…** |
|----|----|----|----|
| Emergencia de barra | La temporización rastrea ∇α; barras largas necesitan α(R) plano | Fracción/longitud de barra vs. forma de $`\widehat{\alpha}`$ (R) | Sin correlación tras controlar masa/tamaño |
| Longevidad de grumos | Mayor α local ⇒ grumos más longevos, discos más gruesos | Edades de grumos, espesor vs. $`\alpha`$ | Tiempos de vida independientes de $`\widehat{\alpha}`$ |
| Alabeos | Desfases de fase donde ∇α es grande | Flexiones HI vs. estelares vs. ∇α | Sin vínculo desfase--gradiente sistemático |
| Planos de satélites | Alineamiento con α anisotrópico en CGM | Orientación del plano vs. anisotropía de α | Sin alineamiento a bariones fijos |
| Temporización de SFH | De afuera hacia adentro o de adentro hacia afuera determinado por la forma de ($`\alpha`$) | SFHs resueltas vs. forma de α | Tendencias desaparecen al condicionar por $`\widehat{\alpha}`$ |

**8.9 Nota de alcance**

Estos esbozos **no** afirman que RTM reemplace la física bariónica detallada (enfriamiento, retroalimentación, turbulencia). Afirman que un **único campo exponente** $\alpha(x)$ puede **organizar la temporización** de procesos por lo demás estándar. La recompensa es un portafolio de pruebas a **nivel de pendiente** y **secuenciación**---cada una con un **modo de falla** claro---que conectan historias de crecimiento con **mapas de estructura** medibles. Si esos vínculos no se materializan bajo buen QA, el rol de RTM en el crecimiento cósmico está **acotado** o **falsificado** para los regímenes probados.

**9. Plan de datos y medición**

Este capítulo convierte las predicciones en un **contrato de análisis**: conjuntos de datos, selección, preprocesamiento, construcción de $\widehat{\alpha}$ (L) (Sec. 5), pruebas de pendiente/colapso, anatomía residual de la bTFR y reconciliación lentes--cinemática (Sec. 7). Todo lo siguiente está formulado para que otro grupo pueda reproducir el pipeline de extremo a extremo.

**9.1 Muestras y criterios de inclusión**

**Galaxias de disco (enfoque en rotación):**

-   Cinemática HI o Hα espacialmente resuelta con ≥10 puntos radiales independientes más allá de ${2\ R}_{d}$

-   Imágenes profundas ópticas/NIR (FWHM del PSF ≤ 0.5 del ancho del anillo interno) para mapas de estructura.

-   Distancia conocida, inclinación $i \in \lbrack 30 \circ ,80 \circ \rbrack$, ángulo de posición (PA) y mapas de masa estelar/gaseosa.

-   Aspirar a **tres cohortes** equilibradas en masa y morfología:\
    C1: alta densidad superficial con barra; C2: espirales de gran diseño sin barra; C3: enanas/LSBs.

**Elípticas (enfoque en dispersión):**

-   Espectroscopía IFU con perfiles radiales de $\sigma(R)$ hasta ${\geq 1.5 - 2\ R}_{e}$

-   Imágenes de alto S/N (núcleos, discos embebidos discernibles)

**Galaxias con lentes fuertes:**

-   Anillos de Einstein/cuádruples con cinemática IFU que intersecte $R_{E}$

-   Modelos de lentes públicos con covarianza (para $M_{lens}(R))$

**Apilamientos de lentes débiles:**

-   Grandes muestras de discos con catálogos de cizallamiento y etiquetas estructurales (fuerza de barra, métricas de textura).

**9.2 Preprocesamiento y geometría**

-   **Imágenes:** sustracción de cielo, enmascaramiento, caracterización del PSF; deproyección usando $i,\ PA$; regrilla a escala de píxel común.

-   **Cinemática:** corrección de difuminado de haz (modelado directo preferido); deriva asimétrica aplicada donde fracción de dispersión estelar > 0.15; gas asumido frío.

-   **Anillos:** anillos logarítmicos con $\Delta\ \log\ L = 0.1$; requerir $\geq 3$ elementos de resolución por anillo.

Todos los pasos producen **incertidumbres por anillo** (covariantes cuando sea relevante).

**9.3 Construcción de** $\widehat{\mathbf{\alpha}}\mathbf{(}\mathbf{L}\mathbf{)}$

Aplicar Sec. 5: calcular características estructurales por anillo (entropía multiescala, potencia de modos, grumosidad, índices fractales/de turbulencia, espesor, textura cinemática). Mapear características → $\widehat{\alpha}$ provisional (monótono paramétrico o ensamble de rangos), agrupar anillos adyacentes en **bins de coherencia contiguos**, ajustar pendiente $m = 1 - {\widehat{\alpha}}_{B}$ en cada bin (EIV robusto), comparar con la mediana del indicador, y **contraer** para obtener ${\widehat{\alpha}}_{j}$, QA: verificación de colapso ${vL}^{{\widehat{\alpha}}_{B} - 1}$ pendiente $\mid mc \mid \leq 0.1$ con IC incluyendo 0.

**9.4 Pruebas de hipótesis principales (por galaxia)**

**H-RC (Pendiente de rotación):** En cada bin de coherencia $B$:

-   Estimar $m_{B} = \partial\ log\ v/\partial\ log\ L$

-   Probar $m_{B} = 1 - \alpha/2\ median({\widehat{\alpha}}_{j \in B})$ (solapamiento de IC ±0.2).

**H-CL (Colapso):** Regresar ${\log\lbrack v\, L}^{{\widehat{\alpha}}_{B} - 1}\rbrack$ vs. $\log L$; requerir $\mid m_{c} \mid \leq 0.1$, IC incluye 0.

**H-bTFR (Anatomía residual):**

-   Ajuste global: ${\log\ v}_{flat}{= a + b\ \log\ M}_{b}$

-   Residuos $\Delta\ \log\ v$ regresados sobre $\delta_{\alpha} \equiv \widehat{\alpha}(R_{meas}) - 1$, controlando por tamaño y densidad superficial.

-   Recalcular en el **radio de pendiente cero**; la correlación debería desaparecer si RTM se cumple.

**H-Disp (Elípticas):** En bins de coherencia, $\partial\ \log\ \sigma/\partial\ \log\ r = 1 - \widehat{\alpha}$ (EIV robusto); colapso de ${\sigma\ r}^{\widehat{\alpha} - 1}$

**H-Lens (Consistencia de lentes):**

-   **Lente fuerte:** comparar $M_{kin}^{RTM}(R_{E})$ con $M_{lens}(R_{E})$; tolerancia $\leq 15\%.$

-   **Apilamientos de lentes débiles:** a bariones fijos, los perfiles de cizallamiento **no** deberían dividirse por coherencia; las pendientes cinemáticas **sí**.

**9.5 Plan estadístico**

-   **Pendientes:** estimador de Theil--Sen con pérdida de Huber; SIMEX para errores en $L$; ICs por bootstrap (B=2000).

-   **Metaanálisis:** Efectos aleatorios combinan pendientes a través de galaxias dentro del mismo tipo de bin (coherencia baja/media/alta). Reportar $m$ agrupado, heterogeneidad $I^{2}$

-   **Correlaciones parciales:** Para residuos de la bTFR, regresar $\Delta\ \log\ v$ sobre $\delta_{\alpha}$ controlando por ${\log\ R}_{d}$, $\Sigma_{\star}$

-   **Pruebas múltiples:** Benjamini--Hochberg FDR al 5% a través de bins y pruebas.

-   **Pre-registro:** Congelar mapas indicador-a-$\alpha$ y umbrales (${\mid m}_{c} \mid \leq 0.1$; ${\mid \widehat{\alpha}}_{pendiente} - {\widehat{\alpha}}_{indicador} \mid \leq 0.2$) antes de mirar los objetivos científicos.

**9.6 Expectativas de potencia (orden de magnitud)**

-   **Pendientes de rotación:** Con 6--8 anillos por bin, $\sigma_{\log\ v} \sim 0.04$, pendiente corregida por EIV $SE\  \sim 0.08$. Diferencias de $\Delta(1 - \alpha) = 0.3$ entre bins dan $> 90\%$ de potencia a $\alpha = 0.05$.

-   **Prueba de colapso:** Detectar ${\mid m}_{c} \mid = 0.12$ con $\sim 80\%$ de potencia por bin.

-   **Residuo de bTFR--**$\delta_{\alpha}$: Con $N \sim 150$ discos y dispersión residual de 0.08 dex, correlación $\mid r \mid \geq 0.25$ es detectable a $> 90\%$ de potencia.

-   **Lentes (fuerte):** Diez anillos de alta calidad con 10% de errores de masa por lentes bastan para detectar un sesgo sistemático del 15% a $> 80\%$ de potencia.

**9.7 QA, exclusiones y verificaciones adversariales**

-   **Umbral de resolución:** descartar anillos con FWHM del PSF $> \ 0.5$ del ancho del anillo.

-   **Difuminado de haz:** marcar si corrección $> 20\%$; excluir si $> 35\%$.

-   **Inclinación/PA:** Monte Carlo sobre posteriores de $i, PA$; bins que fallan en estabilidad (deriva de pendiente >0.15) son **TENTATIVO/RECHAZADO**.

-   **Robustez de indicadores:** recalcular $\widehat{\alpha}$ con (i) eliminación de un indicador, (ii) mapa basado en rangos; requerir estabilidad de clasificación.

-   **Galaxias de control negativo:** sistemas con estructura extremadamente suave (S0 sin rasgos) deben producir $\alpha \rightarrow 1$ y $m$ externo $\rightarrow 0$; la falla dispara auditoría del pipeline.

**9.8 Entregables**

Por galaxia:

-   **Mapas:** $\widehat{\alpha}(L)$ con incertidumbres; máscara de bins de coherencia.

-   **Paneles:** (i) $log\ v$ vs. $log\ L$ coloreado por bin con pendientes ajustadas; (ii) gráficos de colapso ${vL}^{\alpha/2 - 1}$; (iii) diagnósticos residuales.

-   **Tablas:** por bin ${-\widehat{\alpha}}_{indicador}$, ${\widehat{\alpha}}_{pendiente}$, IC, veredicto de colapso, banderas QA.

-   **Reconciliación de lentes (donde esté disponible):** $M_{kin}^{RTM}(R_{E})$ vs. $M_{lens}(R_{E})$ con residuos.

Para la muestra:

-   **Meta-pendientes** (coherencia baja/media/alta), $I^{2}$, y conteos de aprobado/rechazado.

-   **Regresiones de residuos de la bTFR** y remediciones en el "radio de pendiente cero".

-   **Divisiones de apilamientos de lentes débiles** (por coherencia) y su comparación nula.

**9.9 Registro de aprobación/rechazo (pre-declarado)**

Una galaxia contribuye **apoyo** si ≥2 bins de coherencia **APRUEBAN** tanto H-RC como H-CL, y (si aplica) H-Lens aprueba. Una contribución **parcial** requiere APROBADO en H-RC o H-CL con el otro TENTATIVO y sin banderas rojas de QA. **Rechazado** si todos los bins fallan en pendiente o colapso bajo buen QA.

**9.10 Reproducibilidad**

-   Publicar **código de análisis** (extracción de indicadores, mapeo de $\alpha$, pendientes EIV, verificaciones de colapso) con entornos de versión fija.

-   Proveer **catálogos por anillo** (características, $\widehat{\alpha}$, cinemática, banderas QA).

-   Publicar **pre-registro** (hipótesis, umbrales, reglas de exclusión) y mapas de indicadores **congelados** antes de tocar la muestra científica principal.

> **Resultado de este plan.** El contrato de datos asegura que las afirmaciones de RTM se sostienen o caen sobre **pendientes y colapsos por bin** ligados a **coherencia** medida independientemente. A continuación (Sec. 10) especificamos la **suite de simulación** que pone a prueba el pipeline, explora sesgos y genera benchmarks de observables simulados para dinámica consciente de $\alpha$.

**10. Simulaciones**

Este capítulo especifica una **suite de simulación consciente de** $\mathbf{\alpha}$ para (i) probar si las signaturas de pendiente/colapso de RTM son recuperables cuando las fuerzas son estándar pero los relojes son re-temporizados; (ii) cuantificar sesgos y modos de falla del pipeline de las Secciones 5--9; y (iii) generar **sondeos simulados** con verdad conocida ($\alpha_{true}(x)$, masa, geometría) para validación de extremo a extremo.

**10.1 Filosofía: mantener fuerzas, re-temporizar actualizaciones**

Preservamos las fuerzas newtonianas/RG (sin gravedad modificada, sin masa añadida). RTM entra **únicamente** a través de un **reescalamiento temporal** local:

$$dt^{'}(x) = dt\left( \frac{L(x)}{L_{0}} \right)^{\alpha(x) - \alpha_{0}}$$

donde $L(x)$ es una escala estructural elegida (p. ej., escala del anillo radial, espesor local del disco, longitud de suavizado), $\alpha_{0}$ una banda base ($\approx 1$), y $\alpha(x)$ el campo de coherencia (fijo o evolutivo). Todos los integradores a continuación simplemente usan $dt'$ para las actualizaciones de estado mientras calculan aceleraciones del potencial **sin cambio**.

**10.2 Familias de simulación**

**S1. Banco de pruebas no colisional (órbitas en potenciales fijos).**

-   Potenciales: discos de Miyamoto--Nagai + bulbos de Hernquist + halos NFW opcionales (para comparaciones base).

-   Partículas: $10^{6}$ trazadores; integrador: leapfrog o simpléctico de 4to orden con $dt'$ **adaptativo**.

-   $\alpha(x)$: perfiles analíticos (pico centrado en barra, unidad plana exterior); o anisotropía azimutal para experimentos de alabeo.

**S2. N-cuerpos de disco delgado con respuesta viva de barra/espiral.**

-   Autogravedad en una grilla polar 2D; suavizado elegido para resolver $< 0.5$ del ancho del anillo.

-   Gas opcional como partículas con colisiones inelásticas (esquema adhesivo) para emular disipación.

-   $\alpha(x,t)$: (i) fijo; (ii) **acoplado a estructura** (ver §10.5).

**S3. Cubos simulados IFU / mapas de momentos HI.**

-   Tomar instantáneas de S1/S2; renderizar campos de velocidad de **línea de visión** con haz, PSF, ruido y resolución espectral ajustados a sondeos reales.

-   Generar curvas de rotación y perfiles de dispersión con el **mismo pipeline** que los datos (Sec. 5 y 9).

**S4. Análogos de elípticas (partículas de Jeans).**

-   Poblaciones trazadoras esféricas/axisimétricas con anisotropía $\beta(r)$; aplicar $dt'$ a los movimientos radiales para emular la configuración de $\sigma(r)$ por $\alpha(r)$.

-   Comparar $\widehat{\alpha}$ recuperado de pendientes de $\sigma$ con la verdad.

**10.3 Definición del campo de coherencia** $\mathbf{\alpha}\mathbf{(x)}$

**Prescripciones estáticas (verdad conocida):**

-   **Perfil escalonado:** $\alpha = \alpha_{\text{in}} > 1\text{ para }R < R_{b},\alpha = 1\text{ para }R \geq R_{b}$

-   **Perfil de gradiente:** $\alpha(R) = 1 + \Delta\alpha\ exp\left\lbrack - \left( R\text{/}R_{g} \right)^{p} \right\rbrack$

-   **Anisotropía azimutal:** $\alpha(R,\phi) = \alpha(R)\ \left\lbrack {1 + \epsilon\ cos}2\left( \phi - \phi_{b} \right) \right\rbrack$ para patrones tipo barra.

-   **Vertical:** $\alpha_{z}(z) = 1 + \Delta\alpha_{z}\, e^{- |z|\text{/}H}$

**Prescripciones evolutivas (retroalimentación a la estructura):**

-   $\alpha(x,t) = 1 + \lambda_{1}\mathcal{\ S}(x,t) + \lambda_{2}\mathcal{\ T}(x,t),$

donde $S$ es la densidad superficial suavizada (indicador de orden) y $\mathcal{T}$ una medida de turbulencia/varianza (orden inverso). Elegir $\lambda_{1,\ 2}$ tal que $\in \lbrack 0.8,3.0\rbrack$.

**10.4 Numérica y estabilidad**

-   **Verificaciones de conservación.** Con actualizaciones re-temporizadas, asegurar que las aproximaciones simplécticas se cumplan: monitorear derivas de energía y momento angular vs. $dt$ y el **gradiente espacial** de $dt'$.

-   **Condición tipo Courant para re-temporización.** Imponer $\mid \nabla\ \ln\ dt' \mid \lesssim 0.5$ por celda para evitar cizallamiento en el paso temporal; de lo contrario, subciclar.

-   **Acoplamiento grilla--partícula.** Al usar grillas (S2), calcular $L(x)$ del tamaño de celda o de un mapa estructural provisto por el usuario; suavizar $\alpha$ para evitar oscilaciones.

**10.5** $\mathbf{\alpha}$ **acoplado a estructura (auto-actualización)**

Para emular la retroalimentación entre orden y coherencia, actualizar $\alpha$ cada $N$ pasos:

$$\alpha^{(n + 1)} = (1 - \eta)\alpha^{(n)} + \eta\left\lbrack 1 + \lambda_{1}\widetilde{\Sigma} + \lambda_{2}\left( 1 - \widetilde{E} \right) \right\rbrack$$

donde $\widetilde{\Sigma}$ es la densidad superficial normalizada y $\widetilde{E}$ un indicador local de entropía multiescala calculado de la distribución de partículas; $0 < \eta \leq 0.2$ controla la suavidad de actualización. Esto permite que barras/grumos **eleven** $\alpha$ localmente mientras que brotes/turbulencia pueden **reducirlo**.

**10.6 Pipeline de observación simulada**

Para cada instantánea:

1.  Proyectar al cielo con inclinación $i,\ PA$, distancia; aplicar PSF y haz.

2.  Agregar ruido gaussiano ajustado al S/N del sondeo; incluir difuminado de haz y dispersión instrumental.

3.  Extraer perfiles de rotación/dispersión exactamente como en Sec. 5 (mismos anillos, mismas correcciones).

4.  Construir mapas de estructura (entropía, modos, grumosidad) y recuperar $\widehat{\alpha}(L)$ vía el **mismo** mapa de indicadores usado en datos reales.

5.  Ejecutar pruebas de pendiente y colapso; calcular residuos de bTFR y diagnósticos irrelevantes a lentes.

Esto asegura comparabilidad **de extremo a extremo** y expone sesgos de la medición, no solo de la física.

**10.7 Pruebas de recuperación de parámetros**

**Objetivo.** Verificar que el pipeline recupera la **verdad** ($\alpha_{true}$, pendientes, colapsos) dentro de la tolerancia.

-   Métrica de recuperación: ${\Delta\alpha(L) = \widehat{\alpha}(L) - \alpha}_{true}(L)$; reportar mediana y dispersión del 68% por bin.

-   **Tolerancia:** mediana $\mid \Delta\alpha \mid \leq 0.2$ y residuos de pendiente ${\mid m - (1 - \alpha}_{true}) \mid \leq 0.1.$

-   **Curvas de sensibilidad:** variar FWHM del PSF, S/N, inclinación, haz y ancho de anillo para mapear regiones donde la recuperación se vuelve **sesgada** o **inestable**.

-   **Casos adversariales:** escalones abruptos de $\alpha$ dentro de un bin; flujos no circulares fuertes; discos alabeados; $\alpha(\phi)$ anisotrópico. Registrar con qué frecuencia el colapso falla cuando $\alpha$ varía dentro de un bin---esto establece las **reglas de agrupamiento**.

**10.9 Discriminantes contra materia oscura/MOND in silico**

-   **Prueba de degeneración de halo.** Ajustar halos de materia oscura estándar a las **mismas** curvas simuladas; mostrar que muchos halos ajustan $v(R)$, pero **ninguno** reproduce **colapsos** por bin ligados al campo $\mathbf{\alpha}$ **conocido** (signatura única de RTM).

-   **Clasificador MOND.** Generar simulaciones donde el $m$ externo = 0 pero las pendientes **internas** siguen el $\alpha$ impuesto; confirmar que una simple ley de aceleración tipo MOND no puede producir las correlaciones **estructura--pendiente** observadas a mapas bariónicos fijos.

**10.10 Pruebas de estrés y casos límite**

-   **Nulos keplerianos.** Análogos de binarias amplias: fijar $\alpha \rightarrow 1$ y estructura despreciable; confirmar pendientes clásicas y que la estimación de $\widehat{\alpha}$ revierte a la unidad.

-   **Discos ultra-difusos.** $\alpha$ global $\simeq 1$ con turbulencia irregular; probar tasa de falsos positivos para $\alpha > 1$ espurios debidos al ruido.

-   **Trampas de alto** $\mathbf{\alpha}$. Bolsas de $\alpha$ muy alto (régimen sobre-restringido) pueden congelar la evolución local; verificar que el pipeline señale bins que no colapsan (modo de falla del modelo, no éxito).

**10.11 Entregables**

-   **Código abierto**: integradores conscientes de $\alpha$ (S1--S4), módulos de actualización de α, herramientas de observación simulada y cuadernos de análisis; contenedores con versión fija.

-   **Catálogos simulados**: verdad por anillo ($\alpha_{true},\ v,\sigma$), valores observados (con ruido), $\widehat{\alpha}$ recuperado, pendientes, métricas de colapso, banderas QA.

-   **Tablas de sesgo**: funciones para sesgos inducidos por haz/inclinación/indicador y umbrales de exclusión recomendados.

**10.12 Criterios de éxito (para la suite de simulación)**

-   El pipeline **recupera** $\alpha$ y pendientes dentro de la tolerancia a través de regímenes realistas de S/N y resolución.

-   Los **colapsos** ${v\ R}^{\widehat{\alpha} - 1}$ son planos en bins donde $\alpha$ es verdaderamente constante; fallan donde $\alpha$ varía---diagnóstico, no un error.

-   Los discriminantes distintivos de RTM (colapso por bin; acoplamiento estructura--pendiente) **sobreviven** a los efectos de observación simulada, mientras que las líneas base de materia oscura/MOND **no pueden** reproducirlos sin parámetros ad hoc ligados a la estructura.

**Resultado.** Con estas simulaciones (i) validamos que **relojes re-temporizados** solos pueden reproducir la fenomenología de pendiente/colapso bajo campos de $\alpha$ controlados; (ii) cuantificamos dónde el pipeline de datos es **confiable** o **sesgado**; y (iii) producimos **benchmarks simulados** públicos para que grupos independientes puedan intentar **recuperación ciega** de $\alpha$ y desafiar a RTM en terreno neutral.

**11. Discriminantes vs. materia oscura y MOND**

Este capítulo enumera **pruebas decisivas y pre-registradas** que separan RTM de (i) **RG+bariones+halos de materia oscura** y (ii) **dinámica modificada tipo MOND**. Nos enfocamos en cantidades donde RTM hace declaraciones a **nivel de pendiente** o **nivel de colapso** que las líneas base no predicen **sin ajuste ad hoc** ligado a la estructura bariónica.

**11.1 Qué predice realmente cada marco**

-   **RTM (este trabajo):** Dentro de anillos con coherencia fija,

$$\frac{\partial\log v}{\partial\log L} = 1 - \alpha,\quad vL^{\alpha - 1}\text{ es plano (colapso)}$$

> Los residuos de escalamientos globales (p. ej., bTFR) se correlacionan con $\alpha$ **derivado de la estructura**, no con parámetros de masa oculta.

-   **RG + halos de materia oscura:** Reproduce casi cualquier **forma** de $v(L)$ ajustando concentración/tamaño de núcleo del halo y acoplamiento barión--halo. **No** predice genéricamente **colapsos** por anillo ligados a **textura** medida independientemente a menos que los parámetros de halo estén **forzados** a covariar con esas texturas.

-   **MOND/leyes de aceleración:** Predice una relación específica entre **aceleración** y **velocidad** (p. ej., $v^{4}{\propto GM}_{\alpha_{0}}$ en el régimen profundo); puede ajustar planos externos y relaciones tipo Tully--Fisher. **No** predice acoplamiento **estructura--pendiente** a bariones fijos, ni colapsos por bin condicionados por indicadores de coherencia.

**11.2 Clasificador de pendiente de rotación (por bin)**

**Prueba D-R1 (Identidad de pendiente).** Para cada bin de coherencia $B$,

$$m_{B}\frac{\partial\ log\ v}{\partial\ log\ L}? = {1 - \widehat{\alpha}}_{B}$$

-   **RTM APROBADO:** la identidad se cumple dentro de $\pm 0.2$ y el **colapso** aprueba (∣$m_{c}$ ∣≤0.1).

-   **Materia oscura/MOND:** pueden igualar **o** la pendiente **o** el colapso por bin con ajuste, pero no pueden predecir **ambos** a través de bins **desde** $\widehat{\mathbf{\alpha}}$ **independiente** sin incorporar $\widehat{\alpha}$ en la ley de masa/aceleración.

**Regla de decisión:** Si $\geq 70\%$ de los bins a través de la muestra satisfacen pendiente+colapso **usando** $\widehat{\mathbf{\alpha}}$ **solo de la estructura**, clasificar **RTM-favorecido**.

**11.3 Acoplamiento estructura--pendiente vs. parámetros ocultos**

**Prueba D-R2 (Correlación parcial).** Regresar residuos de pendiente interna $\Delta m$ sobre:

-   **(A)** Indicadores de $\widehat{\alpha}$ (potencia de barra, entropía multiescala, grumosidad),

-   **(B)** Parámetros de halo de materia oscura (concentración $c$, tamaño de núcleo $r_{c}$),

-   **(C)** Indicadores MOND (aceleración en el radio de muestreo, elección de función $\mu$).

**Predicción RTM:** $r$ parcial significativo para el conjunto (A), pero **no** para (B) una vez fijados los bariones; (C) débil/ausente tras controlar por $\widehat{\alpha}$.

**Clasificador:** Si ${Adj\ R}_{A}^{2} - {Adj\ R}_{B,C}^{2} \geq 0.1$ a través de la muestra, contar como **victoria de RTM**.

**11.4 Anatomía residual de la bTFR**

**Prueba D-TF1 (Vínculo residuo--coherencia).** Con $v$ medida en un radio fiducial fijo $R_{f}$:

-   Regresar $\Delta\ log\ v$ sobre $\delta_{\alpha} \equiv \widehat{\alpha}(R_{f}) - 1$ controlando por tamaño/densidad superficial.

-   Re-medir $v$ en el **radio de pendiente cero** $R_{0}$ por galaxia (donde $m \simeq 0$ en un bin de bajo $\alpha$) y repetir.

**Predicción RTM:** Correlación fuerte en $R_{f}$, correlación **que desaparece** en $R_{0}$

**Predicción de materia oscura:** Los residuos se correlacionan con $c$/espín del halo, no necesariamente con $\delta_{\alpha}$; la correlación **no** desaparece en $R_{0}$ a menos que los parámetros se reajusten.

**Predicción MOND:** Los residuos están ligados al muestreo de aceleración; sin rol especial para $\delta_{\alpha}$ o $R_{0}$

**11.5 Colapso por bin como restricción funcional**

**Prueba D-C1 (Colapso funcional).** En cada bin, ajustar la pendiente residual de

$${y(L) = v(L)\ L}^{{\widehat{\alpha}}_{B} - 1}$$

-   **RTM:** meta-pendiente agrupada $\overline{m}$ a través de bins $\approx 0$, heterogeneidad $I^{2}$ pequeña.

-   **Materia oscura/MOND:** Sin razón para $\overline{m} \rightarrow 0$ **condicionado** por $\widehat{\alpha}$ a menos que los parámetros ocultos estén ajustados para **rastrear** indicadores de estructura---una suposición adicional que probamos directamente (abajo).

**Verificación anti-trampa (D-C1b).** Forzar que los parámetros de halo sean funciones explícitas de los mismos indicadores usados para construir $\widehat{\alpha}$; medir si esta **imitación** también reproduce **D-R1** (identidad de pendiente) y **D-TF1** (desaparición del residuo en $R_{0}$) sin sobreajuste (validación cruzada entre galaxias). Si no, **RTM gana** por parsimonia.

**11.6 Elípticas y perfiles de dispersión**

**Prueba D-E1 (Identidad de pendiente de dispersión).** En bins de coherencia de elípticas,

$$\frac{\partial\ log\ \sigma}{\partial\ log\ L}? = 1 - \widehat{\alpha}$$

-   **RTM:** identidad + colapso de ${\sigma r}^{\widehat{\alpha} - 1}$

-   **Materia oscura/MOND:** requieren ajustes de anisotropía y perfil de masa no relacionados con la estructura medida; no predicen **vínculo directo** con mapas de $\widehat{\alpha}$.

**Clasificador:** Contar tasa de APROBADO por bin; $> 60\%$ a través de la muestra de elípticas señala **RTM-favorecido**.

**11.7 Verificaciones cruzadas lentes--cinemática (recapitulación como discriminantes)**

-   **Anillos/cuádruples de lentes fuertes (D-L1):** Después de la reinterpretación RTM de la cinemática estelar/gaseosa con $\widehat{\alpha}$, la masa encerrada en $R_{E}$ debe coincidir con la de lentes dentro de $\leq 15\%$. Desplazamientos sistemáticos **después** del condicionamiento por $\alpha$ desfavorecen RTM a escalas galácticas.

-   **Apilamientos de lentes débiles (D-L2):** A bariones fijos, los perfiles de cizallamiento **no** se dividen por clase de coherencia, pero las pendientes cinemáticas **sí**; si el cizallamiento se divide por coherencia, esto sugiere que la masa real covaría con la estructura → **límite de alcance** para RTM.

**11.8 Puntuación tripartita y superficie de decisión**

Definimos un **triplete de puntuación** por galaxia (o por tipo de bin):

-   $S_{RTM} \in \lbrack 0,1\rbrack$: fracción de pruebas (D-R1, D-C1, D-TF1, D-E1, D-L1/L2 cuando estén disponibles) que **APRUEBAN**.

-   $S_{DM} \in \lbrack 0,1\rbrack$: fracción de pruebas mejor explicadas por ajustes de halo **sin** usar indicadores de estructura (o requiriéndolos solo post hoc).

-   $S_{MOND} \in \lbrack 0,1\rbrack$: fracción explicada por escalamientos de solo aceleración.

**Superficie de decisión:**

-   **RTM respaldado** si $S_{RTM} - \max(S_{DM},\ S_{MOND}) \geq 0.2$ a través de la muestra (con IC de bootstrap > 0).

-   **Indeterminado** si las diferencias < 0.2.

-   **RTM desfavorecido** si $S_{RTM} \leq \max(S_{DM},\ S_{MOND}) - 0.2$

Reportamos estos con incertidumbres y realizamos sensibilidad de **eliminación de un indicador** para asegurar que la ventaja de RTM no sea impulsada por una sola característica frágil.

**11.9 Casos límite donde los discriminantes se difuminan**

-   **S0/Sa muy suaves con textura mínima:** $\widehat{\alpha}$ →1 globalmente; todos los modelos predicen pendientes externas casi planas. Los discriminantes se desplazan a **desaparición del residuo de bTFR en** $\mathbf{R}_{\mathbf{0}}$ y verificaciones de **colapso**.

-   **Discos altamente alabeados o fuertemente no axisimétricos:** análisis sectorial reemplaza anillos circulares; las predicciones RTM siguen cumpliendo **por sector**, pero los ajustes de materia oscura/MOND ganan margen de maniobra extra. Los tratamos como **TENTATIVO** a menos que los colapsos sectoriales tengan éxito.

-   **Regímenes dominados por cúmulos:** las lentes demandan masa extra; RTM queda **fuera de alcance** (no intenta corregir los presupuestos de masa de cúmulos).

**11.10 Guía práctica para lectores y revisores**

1.  **Buscar pendientes y colapsos, no solo ajustes.** Un modelo que ajusta una curva no es suficiente; RTM reclama **identidades** (pendiente $= \ 1 - \alpha$) y **planitud** tras el reescalamiento.

2.  **Exigir independencia de** $\widehat{\mathbf{\alpha}}$ **.** Si un modelo de comparación toma prestados los mismos indicadores de estructura para ajustar sus parámetros libres, requerir validación en **muestra retenida** entre galaxias.

3.  **Confiar en las lentes como guardarraíl.** Si la cinemática RTM contradice las lentes después del condicionamiento por $\alpha$, la contradicción es real---contar esto contra RTM, no contra la curvatura.

**11.11 Conclusión**

RTM compite en **parsimonia** y **estructura predictiva**: una vez que $\widehat{\alpha}(L)$ se mide de la **luz/textura**, hace declaraciones de **pendiente y colapso por bin** sin **masa libre adicional**. La materia oscura y MOND pueden ajustar muchas formas pero carecen de estos **invariantes condicionados por estructura**. Si los datos pasan las pruebas de pendiente/colapso de RTM, muestran residuos de bTFR que **desaparecen** en el radio de pendiente cero, y permanecen **consistentes con las lentes**, RTM gana poder explicativo a **escalas galácticas**. Si no, los discriminantes aquí proveen un camino principado y cuantitativo para decir **dónde termina RTM**---y por qué.

**12. Falsificación y condiciones de alcance**

Este capítulo declara---por anticipado---**cómo puede fallar RTM** a escalas galácticas y **dónde no debería aplicarse**. El objetivo es hacer el programa *decidible*: un lector debería poder ejecutar el pipeline y concluir **respaldado**, **acotado** o **falsificado** sin margen de interpretación.

**12.1 Qué cuenta como falsificación (por galaxia, por bin)**

Un bin de coherencia $B$ (anillos adyacentes con $\widehat{\alpha}$ similar) produce **RTM RECHAZADO** si **cualquiera** de lo siguiente se cumple bajo buen QA (Sec. 5 y 9):

1.  **La identidad de pendiente falla:** La pendiente EIV robusta $m_{B} = \partial\ \log\ v/\partial l\ og\ L$ **no** satisface

$$m_{B} = 1 - {\widehat{\alpha}}_{B}$$

dentro de ±0.2 **y** los IC al 95% no se solapan.

2.  **El colapso falla:** Después de reescalar con el ${\widehat{\alpha}}_{B}$ derivado de la pendiente

$${y(L) = v(L)L}^{{\widehat{\alpha}}_{B} - 1}$$

tiene una pendiente residual log--log ∣ $m_{c}$ ∣>0.1 con IC excluyendo cero.

3.  **Desacuerdo de indicadores:** El $\widehat{\alpha}$ basado en indicadores y el ${\widehat{\alpha}}_{B}$ derivado de la pendiente discrepan por $> 0.4$ sin evidencia de deriva de $\alpha$ intra-bin (es decir, el desacuerdo no se explica por heterogeneidad del bin).

Una galaxia es **RTM RECHAZADO** si $\geq 2$ bins fallan (o el único bin usable falla) mientras el QA pasa (verificaciones de resolución, difuminado de haz, inclinación y deriva asimétrica).

**12.2 Qué cuenta como apoyo (por galaxia, por muestra)**

**Por galaxia:** **RTM RESPALDADO** si ≥2 bins **APRUEBAN** tanto (i) la identidad de pendiente (±0.2 con solapamiento de IC) **como** (ii) la planitud de colapso $(|m_{c}|\  \leq \ 0.1$ con IC incluyendo 0), sin banderas QA severas. Un apoyo **PARCIAL** requiere al menos APROBADO en pendiente con colapso **TENTATIVO**, o viceversa, y sin banderas rojas de QA.

**A través de la muestra:** RTM está **respaldado** a escalas galácticas si:

-   ≥70% de todos los bins evaluados **APRUEBAN** pendiente+colapso;

-   El **acoplamiento estructura--pendiente** (Sec. 6, D2) es significativo tras controles de masa/tamaño;

-   La **correlación residuo de bTFR--**$\mathbf{\delta}_{\mathbf{\alpha}}$ está presente a un radio fijo y desaparece en el radio de pendiente cero (Sec. 6, D3);

-   Las verificaciones **lentes--cinemática** aprueban a ≤15% de tolerancia donde sea aplicable (Sec. 7).

La falla de cualesquiera **dos** de los cuatro criterios entre galaxias bajo buen QA constituye **RTM DESFAVORECIDO** a escalas galácticas.

**12.3 Condiciones de alcance (dónde RTM debería/no debería usarse)**

**Régimen válido (alcance previsto):**

-   Dinámica **a escala galáctica** de estrellas/gas donde una única **longitud dominante** por anillo es definible y los **indicadores de coherencia** estructural son medibles (barras, espirales, grumos, espesor, textura cinemática).

-   **Pruebas de baja curvatura:** RTM solo re-temporiza los **relojes orbitales/de relajación**; no altera la curvatura del espacio-tiempo.

**Fuera de alcance o regímenes de precaución:**

-   **Escalas de cúmulos:** presupuestos de masa de lentes fuertes/débiles + rayos X que exceden los bariones; **no** se espera que RTM elimine estas brechas.

-   **Flujos relativistas/campos fuertes:** cerca de SMBH o en jets donde la dilatación temporal de RG domina; la re-temporización por $\alpha$ no sustituye a la RG.

-   $\mathbf{\alpha}$ **no axisimétrico, rápidamente variable:** bins con fuerte anisotropía azimutal o $\nabla\alpha$ pronunciado dentro del bin (se requiere análisis sectorial; por defecto **TENTATIVO**).

-   **Datos con pobre resolución:** PSF/haz tan grandes que los anillos tienen <3 elementos de resolución, o incertidumbres de inclinación/PA dominan los errores de pendiente.

**12.4 Taxonomía de fallas (qué significa una falla y qué hacer)**

-   **Tipo A --- Desajuste de pendiente, buen colapso.**\
    *Interpretación:* Los indicadores de $\widehat{\alpha}$ están mal calibrados; el entorno es coherente, pero el mapa estructura→α es incorrecto.\
    *Acción:* Reajustar el mapa de indicadores solo en **galaxias de calibración**; **no** reclamar RTM hasta que la identidad de pendiente se cumpla con mapas revisados.

-   **Tipo B --- Falla de colapso, identidad de pendiente se cumple.**\
    *Interpretación:* $\alpha$ varía dentro del bin o las correcciones geométricas son incompletas.\
    *Acción:* Estrechar bins, adoptar análisis **sectorial**, o mejorar correcciones de haz/alabeo.

-   **Tipo C --- Tanto pendiente como colapso fallan.**\
    *Interpretación:* RTM no describe la dinámica en ese régimen (falsificación verdadera) o el QA es inadecuado.\
    *Acción:* Si el QA pasa, registrar como **bin falsificado**; reclasificar galaxia si múltiples bins fallan.

-   **Tipo D --- Inconsistencia de lentes.**\
    *Interpretación:* La reinterpretación cinemática de RTM contradice la masa basada en curvatura.\
    *Acción:* Contar contra RTM a **escala galáctica**; marcar cúmulos como **fuera de alcance** por diseño.

**12.5 Guardarraíles contra sobreajuste**

-   **Mapas congelados.** Los mapeos indicador→$\alpha$ se **congelan** antes de analizar objetivos científicos; cualquier ajuste post hoc debe re-validarse en galaxias **retenidas**.

-   **Pruebas retenidas.** El acoplamiento estructura--pendiente y las verificaciones de colapso deben replicarse en un subconjunto retenido con umbrales idénticos.

-   **Anti-fuga.** $\widehat{\alpha}$ **no puede** inferirse de la cinemática misma en el análisis principal (sin circularidad); debe provenir de mapas de **luz/textura**.

**12.6 Controles negativos y expectativas nulas**

-   **Regímenes tipo kepleriano:** binarias amplias, planetas externos, periferia de cúmulos globulares---RTM debe revertir a pendientes clásicas; cualquier desviación indica error del pipeline.

-   **Discos S0/Sa sin rasgos:** los indicadores deberían producir $\hat{\alpha} \to 1$ globalmente; los bins externos deberían APROBAR el colapso con $m \approx 0$.

-   **Nulos simulados:** conjuntos de datos simulados con $\alpha \equiv 1$ en todos lados deben devolver pendientes $m \approx 0$ y **ninguna** correlación espuria con métricas de textura.

**12.7 Contingencias si RTM es acotado, no falsificado**

Si RTM pasa pendiente/colapso **solo** para ciertas morfologías o rangos de masa, reportaremos **curvas de alcance**:

-   **Alcance morfológico:** fracción de APROBADO de bins vs. tipo de Hubble (con barra, sin barra, LSB, enana).

-   **Alcance de densidad superficial:** fracción de APROBADO vs. $\Sigma_{\star}$ o fracción de gas.

-   **Alcance de corrimiento al rojo:** fracción de APROBADO vs. tiempo de retrospección (donde existan datos IFU/HI).

Estas curvas son resultados legítimos; delimitan **dónde** la re-temporización por coherencia importa.

**12.8 Resumen en una figura (para revisores)**

Incluiremos un resumen de una página por muestra:

1.  **Arriba-izquierda:** distribuciones de $\widehat{\alpha}(R)$ a través de galaxias.

2.  **Arriba-derecha:** Gráfico de identidad de pendiente por bin: $m$ vs. $1 - \widehat{\alpha}$ con línea 1:1 (color = estado QA).

3.  **Abajo-izquierda:** Distribución de meta-pendiente de colapso (debería tener pico en 0).

4.  **Abajo-derecha:** Residuos lentes--cinemática (donde estén disponibles) y relación residuo de bTFR--$\delta_{\alpha}$ en $R_{f}$ y en $R_{0}$

Un lector puede juzgar **de un vistazo** si RTM se cumple, está acotado o falla.

**12.9 Conclusión**

RTM será declarado **respaldado** solo si las **identidades de pendiente** y los **colapsos** se cumplen bin a bin con $\widehat{\alpha}$ medido **independientemente** de la estructura, y si las **lentes** permanecen consistentes a escalas galácticas. Está **falsificado** si pendientes y colapsos fallan ampliamente bajo buen QA o si las brechas lentes--cinemática persisten **después** del condicionamiento por $\alpha$. Está **acotado** si el éxito se localiza en morfologías o entornos específicos. Este capítulo hace esos resultados **pre-registrados e inequívocos**---para que la comunidad pueda decidir, no solo ajustar.

**13. Discusión**

Esta sección sintetiza qué significaría la **Astronomía Rítmica** si las pruebas pre-registradas **pasan**, cómo interpretar resultados **mixtos** y qué enseña una **falla**. Cerramos mapeando los próximos pasos más decisivos y clarificando los límites conceptuales.

**13.1 Si el programa de pendiente--colapso pasa**

Un hallazgo consistente de que, dentro de anillos con coherencia fija,

$$\frac{\partial\log v}{\partial\log L} = 1 - \widehat{\alpha}\quad\text{y}\quad vL^{\widehat{\alpha} - 1} \approx \text{const}$$

establecería que los **relojes cinemáticos** de una galaxia están co-gobernados por un **campo organizacional** $\alpha(L)$ medible solo de la *estructura bariónica*. Las ganancias prácticas son inmediatas:

-   **Diversidad predictiva.** Las formas de curvas internas a masa fija dejan de ser dispersión residual; se convierten en *varianza predicha* una vez que $\widehat{\alpha}$ se mapea desde barras, espirales, grumos, espesor y textura cinemática.

-   **Anatomía de la bTFR clarificada.** Los residuos a $M_{b}$ fijo heredan una geometría simple: medir en el radio de pendiente cero (donde $\widehat{\alpha} \rightarrow 1$) y la relación se estrecha; muestrear dentro de zonas coherentes y aparece un sesgo predecible.

-   **Parsimonia vs. ajuste post-hoc.** Los halos de materia oscura (o las interpolaciones de MOND) pueden ajustar muchas formas, pero no ligan **a priori** *colapsos funcionales* por anillo a textura medida independientemente. RTM agregaría una restricción estructural faltante.

**13.2 Si vemos apoyo parcial**

Un patrón común que anticipamos es **coincidencias de pendiente** con **colapsos imperfectos** en bins donde $\alpha$ deriva a través del anillo o las sistemáticas geométricas (haz, inclinación, alabeos) permanecen. Esto no es trivial; es diagnóstico:

-   **Qué ajustar.** Estrechar bins, adoptar análisis sectorial o mejorar correcciones de haz/alabeo. Reverificar con mapas de $\widehat{\alpha}$ con eliminación de un indicador.

-   **Qué reportar.** Llamar a estos **PARCIAL** por diseño (Sección 12), y publicar los modos de falla. Un campo aprende más rápido de "casi" limpios que de victorias ambiguas.

**13.3 Si el programa falla limpiamente**

Si (i) las pendientes no igualan $1 - \widehat{\alpha}$, (ii) los colapsos muestran inclinación residual significativa, y (iii) los residuos de la bTFR ignoran $\delta_{\alpha}$ **después del QA**, entonces **RTM no es la abstracción correcta para la cinemática galáctica**. Esto sigue siendo valioso:

-   **Límite aprendido.** La re-temporización por coherencia puede ser poderosa en sistemas de laboratorio (química, redes), pero insuficiente para flujos autogravitantes una vez que la curvatura y la geometría tridimensional dominan.

-   **Disciplina reutilizable.** Las verificaciones de pendiente primero + colapso, congelamiento de indicadores y pre-registro permanecen como una plantilla para otras hipótesis conscientes de la estructura en astronomía.

**13.4 Aclaraciones conceptuales (qué es RTM---y qué no)**

-   **No una nueva fuerza ni masa oculta.** Las fuerzas y la curvatura siguen siendo RG; RTM re-temporiza procesos **operacionales** incrustados en medios estructurados.

-   **Sin almuerzo gratis en cúmulos.** Donde las lentes demandan masa más allá de los bariones (cúmulos ricos), RTM está fuera de alcance a menos que se acompañe de materia adicional genuina.

-   **Sin circularidad.** $\widehat{\alpha}$ viene de la **luz/textura**, no de la cinemática; las pendientes/colapsos se predicen entonces, no se ajustan.

**13.5 Relación con la dinámica clásica de discos**

RTM no reemplaza el análisis de Jeans; lo **complementa** con una restricción sobre cómo las *escalas temporales* varían con la escala cuando el medio está jerárquicamente organizado. En la práctica:

-   Tratar $\widehat{\alpha}$ como un **campo de hiperparámetros** que regulariza modelos dinámicos: priors sobre el comportamiento de pendiente permitido por anillo.

-   Usar RTM para **elegir radios** para escalamientos globales (p. ej., donde $\widehat{\alpha} \rightarrow 1$ para la bTFR), reduciendo sistemáticas entre muestras.

**13.6 Fuentes de falsos positivos y cómo nos protegimos contra ellos**

-   **Difuminado de haz / errores de inclinación.** Estos aplanan pendientes pero no inducen genéricamente **colapsos por bin** después del reescalamiento $L^{\widehat{\alpha} - 1}$; nuestras correcciones EIV y umbrales QA abordan esto.

-   **Flujos no circulares.** Las barras y alabeos complican $v(R)$. Manejamos esto con análisis sectorial y al incluir **textura cinemática** como un indicador negativo en $\widehat{\alpha}$.

-   **Fuga de indicadores.** Si los indicadores codifican accidentalmente la cinemática (p. ej., usando campos de velocidad), aparece circularidad. Separamos estrictamente las entradas de **estructura** de las salidas de **dinámica** (Sec. 5, 9).

**13.7 Qué *significa* físicamente un** $\mathbf{\alpha}$ **medido**

A través del corpus RTM, mayor $\alpha$ refleja mayor **persistencia** y **jerarquía**: tiempos de permanencia más largos, menos vías efectivas, mezcla más lenta. En discos, eso se traduce en:

-   **Barras/bulbos/grumos internos:** $\alpha$ elevado → relojes orbitales locales más lentos → ascensos internos más pronunciados o aplanamiento retardado.

-   **Periferias difusas:** α→1 → asíntotas planas sin invocar masa extra *si* la curvatura no necesita aumentar (la consistencia con las lentes es el guardarraíl).

Esta es una imagen unificadora: **diseñar el tiempo** mediante el **diseño de la estructura**.

**13.8 Intersecciones con retroalimentación y turbulencia**

El enfriamiento, la retroalimentación y la turbulencia ya dan forma a la estructura del disco. RTM postula que su **resultado organizativo neto**---no cada detalle microfísico---entra en la dinámica principalmente a través de $\alpha$:

-   **Retroalimentación que destruye el orden** impulsa α↓ (los discos externos se asientan más rápido, los grumos mueren antes).

-   **Características coherentes de larga vida** (barras, anillos) impulsan α↑ (los relojes internos se ralentizan, la diversidad aumenta).\
    Esto provee una **estadística resumen** para modelos subgrilla en simulaciones: en lugar de ajustar muchas perillas, ajustar cómo **desplazan** $\mathbf{\alpha}$.

**13.9 ¿Qué convencería a un escéptico?**

Tres gráficos:

1.  **Identidad de pendiente:** puntos de $m$ medido vs. $1 - \widehat{\alpha}$ abrazando la línea 1:1 a través de muchas galaxias.

2.  **Colapso funcional:** distribuciones de pendientes residuales por bin centradas en 0 con ICs estrechos.

3.  **Armonía de lentes:** $M_{kin}^{RTM}$ coincidiendo con $M_{lens}$ a escalas galácticas mientras las brechas de cúmulos permanecen.

Si estos se replican con indicadores congelados y muestras retenidas, RTM supera el listón.

**13.10 Próximas decisiones (qué haríamos *después* de los primeros resultados)**

-   **Si APROBADO:** Expandir a sondeos ricos en IFU, publicar mapas abiertos de $\widehat{\alpha}$, y avanzar en **evolución** (cómo los campos de $\alpha$ se aplanan con el tiempo cósmico). Explorar predicciones condicionadas por simetría (p. ej., fase de barra vs. $\nabla\alpha$).

-   **Si PARCIAL:** Enfocarse en sectores y estructura vertical; refinar definiciones de bin; probar regímenes de enanas/LSB donde $\alpha$ está cerca de la unidad para aislar asíntotas limpias.

-   **Si RECHAZADO:** Publicar el resultado negativo con el pre-registro completo, luego reutilizar el pipeline como un **arnés de consistencia** para cualquier propuesta futura consciente de la estructura.

**13.11 Significancia más amplia**

Independientemente del resultado, este trabajo trae una metodología de **grado de laboratorio**---inferencia de pendiente primero, verificaciones de colapso, umbrales pre-registrados---a la astronomía extragaláctica. La idea de que la **organización controla los relojes** es o bien un unificador poderoso (si es respaldado) o un callejón sin salida claramente circunscrito (si es falsificado). En ambos casos, el campo gana: o bien un nuevo eje (coherencia) en sus relaciones de escalamiento, o bien una comprensión más aguda de por qué la **masa** y la **curvatura** solas deben seguir llevando la carga.

**14. Conclusiones y perspectivas**

La **Astronomía Rítmica** avanza una descripción falsificable y de pendiente primero de la dinámica galáctica: una vez que un **campo de coherencia** $\alpha(L)$ se mide solo de la estructura bariónica, los relojes orbitales obedecen

$$v(L) = \kappa L^{1 - \alpha(L)}\quad \Rightarrow \quad\frac{\partial\log v}{\partial\log L} = 1 - \alpha/2,$$

y los **colapsos por bin** ${v\, L}^{\alpha - 1} \approx const$ deben aparecer cuando α es localmente constante. A diferencia de las parametrizaciones de materia oscura o las modificaciones de la ley de aceleración, RTM predice **identidades funcionales por bin** condicionadas por estructura medida independientemente, y mantiene la **curvatura** (lentes) en RG estándar.

**14.1 Qué contaría como éxito**

-   Las **pendientes de rotación/dispersión** coinciden con $1 - \widehat{\alpha}$ a través de anillos agrupados por coherencia con ICs pequeños.

-   Los **colapsos** son planos dentro de bins después del reescalamiento $L^{\widehat{\alpha} - 1}$.

-   Los **residuos de la bTFR** se correlacionan con $\delta_{\alpha}$ a radio de muestreo fijo y **desaparecen** en el radio de pendiente cero.

-   La **reconciliación lentes--cinemática** se cumple a ≤15% a escala galáctica, mientras los cúmulos permanecen como límite de alcance.

Si estos se replican con **mapas de indicadores congelados**, muestras retenidas y cuadernos abiertos, RTM gana un lugar junto al modelado de masa como una **ley temporal condicionada por estructura** para galaxias.

**14.2 Qué aprendimos incluso si los resultados son mixtos**

-   La disciplina de **pendiente/colapso** separa la geometría/sistemáticas de las regularidades dinámicas verdaderas.

-   Los resultados negativos o parciales **agudizan los límites**: donde $\alpha$ no puede estimarse de manera estable, o donde las lentes demandan masa independientemente de la coherencia, RTM está **acotado**.

**14.3 Próximos pasos inmediatos (90--180 días)**

1.  **Conjunto de calibración** (∼20 galaxias): congelar mapas de característica→$\alpha$; publicar pre-registro.

2.  **Muestra de prueba central** (∼150 discos + 40 elípticas): ejecutar pendiente/colapso por bin; publicar catálogos por anillo y banderas QA.

3.  **Verificaciones cruzadas de lentes**: 10--15 lentes fuertes con IFU; apilamientos de lentes débiles divididos por clase de coherencia.

4.  **Benchmarks de simulación**: simulaciones públicas conscientes de α con verdad conocida para desafíos de recuperación ciega.

**14.4 Riesgos y mitigaciones**

-   **Fragilidad de indicadores** → dos familias de mapas (paramétrico + ensamble de rangos), verificaciones de estabilidad con eliminación de un indicador.

-   **Sesgos de haz/inclinación** → correcciones EIV, umbrales de resolución, análisis sectorial para casos alabeados/no circulares.

-   **P-hacking** → umbrales pre-registrados, replicación retenida y código/datos públicos.

**14.5 Implicaciones más amplias**

-   Si es respaldado, $\alpha$ se convierte en un **nuevo eje** en las relaciones de escalamiento---vinculando **textura** (barras, espirales, grumos, espesor) con **temporización** (pendientes, perfiles de dispersión), y proporcionando un objetivo compacto para modelos subgrilla en simulaciones ("**diseñar el tiempo de la galaxia**").

-   Si es acotado o falsificado, la comunidad gana una **plantilla transparente** para probar ideas conscientes de la estructura sin confundir relojes y curvatura.

**Conclusión.** RTM no reemplaza la gravedad ni el modelado de masa bariónica; agrega un **reloj condicionado por coherencia** que puede ser probado correcto o incorrecto con datos actuales. Las signaturas decisivas son **pendientes** y **colapsos** ligados a **estructura medida independientemente**, con las **lentes** como guardarraíl. Cualquier resultado---apoyo o falla bien documentada---mueve la dinámica extragaláctica hacia adelante con palancas más claras, límites más claros y un camino reproducible que otros pueden auditar.

**Apéndice A --- Derivaciones e identidades**

**A.1 De la ley temporal RTM a las leyes de rotación/dispersión**

RTM postula un **tiempo operacional** para procesos a escala $`L`$

``` math
T(L) = T_{0}\left( \frac{L}{L_{0}} \right)^{\alpha(L)}\Theta
```

donde $`\alpha(L)`$ es el **exponente de coherencia** y $`\Theta`$ es adimensional y se trata como constante **dentro de un bin de coherencia** (Sec. 5). Para órbitas casi circulares,

``` math
T = \frac{2\pi L}{v}\quad \Rightarrow \quad v(L) = \kappa L^{1 - \alpha(L)/2},\quad\kappa \equiv \frac{2\pi L_{0}}{T_{0}\Theta}
```

Tomando derivadas **dentro de un bin** donde $`\alpha`$ es aproximadamente constante,
                                                           
 ``` math                                                     
 \frac{\partial\log v}{\partial\log L} = 1 - \alpha\text{/}2  
 ```
(A1)

que es la **ley de pendiente** usada a lo largo del texto.

Para sistemas soportados por dispersión (capa esférica de espesor $`\sim L`$), una velocidad aleatoria característica escala como $`L/T`$, dando

``` math
\left. \ \frac{\partial\log\sigma\ }{\partial\log L\ } \right|_{\text{bin}} = 1 - \alpha
```
(A2)

$`{\sigma(L)\  \propto \ L}^{1 - \alpha(L)} \Rightarrow`$

**A.2 Verificación de colapso**

Definir la **variable colapsada**

``` math
{y(L) \equiv v(L)\ L}^{\alpha/2 - 1}
```

Si $`\alpha`$ es constante dentro del bin, entonces $`y(L) = \kappa =`$ constante y


``` math
\left. \ \frac{\partial\log y\ }{\partial\log L\ } \right|_{\text{bin}} = 0
```
(A3)

La misma forma se cumple para dispersiones con $`{y(L) = \sigma(L)\ L}^{\alpha - 1}`$

**A.3 Movimientos no circulares y sistemáticas geométricas (primer orden)**

Sea $`v_{\text{obs}}^{2} = v_{\phi}^{2} + \delta v_{\text{nc}}^{2}`$ donde $`{\delta v}_{nc}`$ codifica las correcciones de flujo de barra/espiral y deriva asimétrica. Si $`{\delta v}_{nc}/v_{\phi}`$ varía lentamente con $`L`$ dentro de un bin, la pendiente de $`\log_{vobs}`$ versus $`log\ L`$ está perturbada en $`\mathcal{O}\left( \frac{\partial\log\delta v_{\text{nc}}}{\partial\log L} \right)`$, es decir, principalmente un cambio de **intersección**.\
Esto justifica el enfoque de **pendiente primero** y el **refinamiento sectorial** cuando la no circularidad es fuerte.

**A.4 Casos axisimétricos vs. esféricos**

-   **Discos delgados.** Usando geometría de anillos inclinados, la escala característica local es el radio del anillo $L = R$; los resultados (A1--A3) aplican por anillo.

-   **Sistemas esféricos.** Con modelado de Jeans, reemplazar el tiempo dinámico $t_{dyn}{\sim (G\rho)}^{- 1/2}$ por el **operacional** ${T \propto L}^{\alpha}$ cambia solo la **tasa** a la cual las órbitas mezclan fase; la identidad de pendiente medible (A2) permanece por bin siempre que la anisotropía varíe lentamente a través del bin.

**A.5 Cuando** $\mathbf{\alpha}$ **varía dentro de un bin**

Sea $(L) = \alpha_{B} + \delta\alpha(L)$ con $\mid \delta\alpha \mid \ll 1$ a través del ancho $\Delta\ \log\ L$. Entonces

$$\frac{\partial\log y}{\partial\log L} = \underset{= 0}{\overset{\left( 1 - \alpha_{B} \right) + \left( \alpha_{B} - 1 \right)}{︸}} - \delta\alpha(L)$$

así que la pendiente residual de colapso es aproximadamente ${- \langle\ \delta\alpha\rangle}_{B}$. Este es el diagnóstico usado para estrechar bins (o sectorizar) hasta que el residuo sea consistente con 0.

**Apéndice B --- Construcción de** $\widehat{\mathbf{\alpha}}$ **a partir de observables**

**Objetivo.** Mapear **indicadores de estructura** multiescala a un exponente de coherencia por anillo $\widehat{\alpha}$ con incertidumbre, usando solo **luz/textura** (sin cinemática), y luego verificar con pendientes y colapsos.

**B.1 Conjunto de características**

Para cada anillo deproyectado $A_{j}$ (Sec. 5):

1.  **Entropía multiescala** $\mathbf{E}$ **.** Calcular pirámide de wavelets à trous $I_{s}$ sobre escalas $s$, luego entropía $H_{s}$. Definir $E^{\star} = 1 - zscore(\sum_{s}\ w_{s}\ H_{s})$. Menor entropía → mayor orden → mayor $\alpha$.

2.  **Potencia de modos de Fourier** $P_{m}$ **.** Del brillo superficial deproyectado, medir la potencia fraccional en modos $m = 2$ y $m = 2 - 4$ (espiral): $C_{mode} = \sum_{m \in \{ 2,3,4\}}\ P_{m}$

3.  **Grumosidad/Suavidad** $Q$. Usar CAS o Gini--$M_{20}$ para formar $Q = 1 - S$ (más suave → más coherente).

4.  **Índice fractal/turbulento** $D$ (gas). Pendiente de función de estructura $\zeta$ o dimensión fractal $D$; convertir a $C_{D}$ tal que más orden a gran escala ⇒ mayor $C_{D}$

5.  **Espesor/Asimetría** $T$. De indicadores verticales o relaciones de ejes corregidas; definir $C_{T}$ (más delgado/simétrico → mayor $C_{T}$).

6.  **Textura cinemática** $K$ (indicador negativo). Potencia de flujo no circular de campos de velocidad residuales; usar $C_{K} = 1 - NCF$ cuando esté disponible, u omitir para mapeo puramente fotométrico.

> $z_{j} = \left\lbrack E^{*},C_{\text{mode}},Q,C_{D},C_{T},C_{K} \right\rbrack$ con covarianza $\Sigma_{j}$

**B.2 Mapeo monótono a** $\widehat{\mathbf{\alpha}}$

Dos opciones intercambiables, pre-registradas:

-   **Mapa monótono paramétrico:**

> $\widetilde{\alpha} = \alpha_{0} + \sum_{k}^{}{w_{k}g_{k}\left( z_{k} \right)},\quad w_{k} \geq 0,g_{k}$ monótono (identidad/logístico). Regularizar con $\sum_{}^{}w_{k} = 1$ y prior $\alpha \in \lbrack 0.8,3.2\rbrack$

-   **Ensamble de rangos:**

$\widetilde{\alpha} = \alpha_{0} + \lambda\backslash median_{k}\ rank\left( z_{k} \right),$ robusto a valores atípicos y escala.

Incertidumbres del método delta (paramétrico) o bootstrap (rangos).

**B.3 Agrupamiento por coherencia y contracción**

-   **Restricción de contigüidad.** Agrupar anillos **adyacentes** por $\widehat{\alpha}$ (Ward 1-D), asegurando contigüidad radial.

-   **Reconciliación de pendiente.** En cada bin $B$, ajustar $m_{B}$ y establecer ${\widehat{\alpha}}_{B}{= 1 - m}_{B}$. Contraer ${\widehat{\alpha}}_{j}$ por anillo hacia ${\widehat{\alpha}}_{B}$ con pesos ${\propto 1/SE}^{2}$.

**B.4 Umbrales QA**

-   Resolución: ≥3 elementos de resolución por anillo.

-   Corrección de difuminado de haz <20% (marcar TENTATIVO si 20--35%).

-   Robustez de indicadores: desplazamiento por eliminación de un indicador $\leq 0.2$ en $\widehat{\alpha}$.

-   Estacionariedad: pendiente del PSD o textura debe ser aproximadamente ley de potencia en la banda (rechazar curvatura fuerte).

**Apéndice C ---** Algoritmos de simulación conscientes de $\mathbf{\alpha}$

**C.1 Principio**

Mantener las **fuerzas** estándar; aplicar **reescalamiento temporal** localmente:

$$dt^{'}(x) = dt\left( \frac{L(x)}{L_{0}} \right)^{\alpha(x) - \alpha_{0}}$$

Los integradores avanzan estados con $dt'$ (re-temporización), no cambiando la gravedad.

**C.2 Órbitas no colisionales (S1)**

-   Potencial: disco de Miyamoto--Nagai + bulbo de Hernquist (opcionalmente agregar NFW para comparaciones base).

-   Partículas: ${N \sim 10}^{6}$ trazadores; paso leapfrog/simpléctico con $dt'$ adaptativo.

-   Campos de $\alpha$: picos radiales analíticos, gradientes o patrones azimutales $m = 2$.

-   Salidas: curvas de rotación por sector; pendientes y colapsos por bin.

**C.3 Disco delgado con respuesta viva (S2)**

-   Autogravedad en grilla 2D (solucionador de Poisson por FFT o grilla polar).

-   Gas vía esquema de partículas adhesivas para disipación.

-   $\alpha(x,t)$: fijo o **acoplado a estructura**

-   $\alpha^{n + 1} = (1 - \eta)\alpha^{n} + \eta\left\lbrack 1 + \lambda_{1}\widetilde{\Sigma} + \lambda_{2}\left( 1 - \widetilde{E} \right) \right\rbrack$

-   Diagnósticos: fuerza de barra vs. $\nabla\alpha$, tiempos de vida de grumos vs. $\alpha$ local.

**C.4 Cubos simulados IFU/HI (S3)**

-   Proyectar instantáneas con inclinación/PA; construir mapas de momentos 0/1/2.

-   Convolucionar con PSF/haz; agregar ruido; ejecutar el **mismo** pipeline de extracción de anillos y $\widehat{\alpha}$ que para datos reales.

**C.5 Estabilidad y guardas tipo CFL**

-   Imponer $\mid \nabla\ \ln dt' \mid \lesssim 0.5$ por celda; subciclar de lo contrario.

-   Monitorear deriva de energía y momento angular; ajustar $dt$ para que la re-temporización no rompa el comportamiento simpléctico.

**C.6 Pruebas de recuperación**

-   Tolerancia: mediana $\mid \widehat{\alpha} - \alpha_{true} \mid \leq 0.2$; residuo de pendiente $\mid m - (1 - \alpha_{true}) \mid \leq 0.1$; meta-pendiente de colapso $\mid \overline{m} \mid \leq 0.05$

-   Mapas de sesgo vs. PSF, S/N, inclinación y ancho de bin; registrar umbrales de exclusión.

**Apéndice D --- Plantilla de pre-registro y recetas de figuras**

**D.1 Pre-registro (a publicar antes del análisis)**

**Título:** Astronomía Rítmica: pruebas de pendiente/colapso con anillos condicionados por coherencia.

**Criterios de valoración primarios:**

-   H-RC: En cada bin, $m = 1 - \widehat{\alpha}$ dentro de ±0.2 (solapamiento de IC al 95%).

-   H-CL: En cada bin, la pendiente residual de ${y = vL}^{\widehat{\alpha} - 1}$ es $\mid m_{c} \mid \leq 0.1$ con IC incluyendo 0.

-   H-TF: Los residuos de la bTFR $\Delta\ log\ v$ se correlacionan con $\delta_{\alpha}$ a radio fiducial fijo y **desaparecen** en el radio de pendiente cero.

-   H-Lens (donde aplique): $\left| M_{\text{kin}}^{\text{RTM}} - M_{\text{lens}} \right|\text{/}M_{\text{lens}} \leq 0.15.$

**Exclusión/QA:**

-   PSF/haz < 0.5 del ancho del anillo; incertidumbre de inclinación < 5°; corrección de haz < 35%.

-   El anillo debe tener ≥3 elementos de resolución y ≥30 píxeles independientes.

**Mapa indicador→**$\mathbf{\alpha}$: fijar coeficientes (paramétrico) y parámetros de ensamble de rangos en el **conjunto de calibración** ($N \approx 20$), luego **congelar**.

**Plan estadístico:** Theil--Sen + SIMEX para pendientes; ICs por bootstrap (B=2000); meta de efectos aleatorios para pendientes agrupadas; FDR 5%.

**Reglas de falla:** Como en Sec. 12---dos fallas independientes entre galaxias bajo buen QA → RTM desfavorecido.

**D.2 Figuras canónicas (por galaxia)**

1.  **Estructura y mapa de** $\widehat{\mathbf{\alpha}}$: imagen deproyectada, paneles de indicadores y $\widehat{\alpha}(R)$ radial con IC.

2.  **Gráfico de pendiente:** $log\ v$ vs. $log\ R$ coloreado por bins de coherencia; anotar $m$ ajustado y $1 - \widehat{\alpha}$

3.  **Paneles de colapso:** ${vR}^{\widehat{\alpha} - 1}$ vs. $R$ por bin, con pendiente residual e IC.

4.  **Posición en la bTFR:** galaxia en la bTFR global; residuo vs. $\delta_{\alpha}$

5.  **(Si lente):** $M_{\text{kin}}^{\text{RTM}}(R)\text{ vs. }M_{\text{lens}}(R)$ con residuos.

**D.3 Figuras canónicas (nivel de muestra)**

1.  **Nube de identidad de pendiente:** todos los bins $m$ vs. $1 - \widehat{\alpha}$ con línea 1:1, sombreado por densidad.

2.  **Histograma de meta-pendiente de colapso:** distribución de pendientes residuales por bin con 0 marcado.

3.  **Anatomía residual de la bTFR:** $\Delta\ \log\ v$ vs. $\delta_{\alpha}$ en $R_{f}$ y en $R_{0}$

4.  **Reconciliación de lentes:** dispersión de ${\Delta M/M}_{lens}$ en $R_{E}$ (o bandas de perfil) con media ±IC.

5.  **Gráficos de alcance:** fracción de APROBADO vs. morfología, densidad superficial, corrimiento al rojo.

**APÉNDICE E --- Análisis empírico robusto: La base de datos SPARC y la topología bariónica**

El marco RTM propone que las curvas de rotación galáctica planas no son causadas por halos invisibles de materia oscura, sino por un cambio macroscópico en la coherencia topológica de la red bariónica ($\alpha \approx 2$). Para validar esto, analizamos galaxias de disco de la base de datos SPARC.

**E.1 Observación heurística y sesgo de atenuación**

El análisis OLS inicial fue suprimido por **sesgo de atenuación**. Una vez corregido mediante **Regresión de Distancia Ortogonal (ODR)** para absorber el 15% de varianza de hardware y observacional, el vínculo estructura-cinemática se revela como una pendiente más pronunciada de $\mathbf{- 1.169\ }\mathbf{\pm}\mathbf{0.119}$. Además, las 52 galaxias con curvas de rotación planas produjeron un exponente de coherencia derivado de $\alpha = \ 1.99$. Este resultado es algebraicamente esperado: dado que $\alpha = 2(1 - \text{pendiente})$, las curvas planas (pendiente $\approx 0$) dan $\alpha \approx 2$ por definición. El contenido empírico radica en la correlación estructura-cinemática, no en el mapeo de curva plana.

**E.2 Validación probabilística rigurosa (ODR y propagación de errores)**

Para asegurar que el vínculo estructura-cinemática representa una correlación física genuina y no un artefacto estadístico, el conjunto de datos fue sometido a un pipeline estadístico de "Equipo Rojo":

1.  **Regresión de Distancia Ortogonal (ODR):** Reemplazamos OLS con un modelo robusto de Errores en Variables (EIV) para evaluar el vínculo estructura-cinemática. Inyectamos explícitamente incertidumbres observacionales en el modelo (una varianza del $5\%$ para gradientes fotométricos y los errores de velocidad observacionales documentados), forzando las predicciones teóricas de RTM a sobrevivir la ambigüedad de la observación telescópica del mundo real.

2.  **Distribución Monte Carlo:** Para las 52 galaxias de curva plana, simulamos 52,000 puntos de datos inyectando los márgenes de error de velocidad rotacional específicos de vuelta en las derivaciones de pendiente, mapeando la distribución probabilística del exponente topológico $\alpha$.

**E.3 El vínculo estructura-cinemática (hallazgos robustos)**

Bajo propagación de errores, la correlación estructura-cinemática sobrevive:

-   **Consistencia de curva plana:** La distribución Monte Carlo para las galaxias de curva plana produce $\mathbf{\alpha}\mathbf{= \ 1.993\ }\mathbf{\pm}\mathbf{0.130}$, consistente con el mapeo algebraico $\alpha = 2(1-\text{pendiente})$ para pendientes cercanas a cero. Esto confirma la consistencia interna de la ley de velocidad RTM pero, como se señaló arriba, refleja una relación definicional más que una medición independiente.

-   **El vínculo estructura-cinemática:** El análisis ODR robusto confirma que la correlación entre estructura bariónica visible y cinemática orbital (pendiente ODR $= \  - 1.169\  \pm 0.119$) sobrevive la inyección de ruido y la propagación de errores. Este es un efecto entre tipos: galaxias con perfiles de brillo superficial más concentrados tienen pendientes cinemáticas sistemáticamente más pronunciadas.

-   **Análisis intra-tipo:** Cuando las galaxias se dividen en cuartiles por concentración estructural y se computa la correlación parcial (controlando por $v_{max}$) dentro de cada cuartil, la señal se debilita sustancialmente (Q1--Q3: $\rho < 0.17$, todos no significativos; Q4: $\rho = -0.29$, $p = 0.059$, marginal). Esto indica que la correlación general es impulsada por diferencias entre tipos más que por variación continua intra-tipo.

**E.4 Análisis de discrepancia de masa**

Una prueba más informativa explota las componentes del modelo de masa bariónica provistas por SPARC. Calculamos la velocidad bariónica $V_{bar}^2 = |V_{gas}| \cdot V_{gas} + \Upsilon_{disk} \cdot |V_{disk}| \cdot V_{disk} + \Upsilon_{bul} \cdot |V_{bul}| \cdot V_{bul}$ (con relaciones masa-luminosidad estándar $\Upsilon_{disk} = 0.5$, $\Upsilon_{bul} = 0.7$ a 3.6 $\mu$m) y la discrepancia de masa $D = V_{obs}^2 / V_{bar}^2$ para 120 galaxias con datos suficientes.

**Hallazgo clave:** A masa bariónica total fija, la **concentración** de brillo superficial de la galaxia predice significativamente la discrepancia de masa externa:

| Parámetro | $\rho$ de orden cero | $\rho$ parcial (controlando $M_{bar}$) | $p$ |
|-----------|-------------------|---------------------------------------|-----|
| Concentración | +0.04 (ns) | **+0.346** | **0.0001** |
| Pendiente SB | -0.02 (ns) | **-0.245** | **0.007** |

Comparación de modelos multivariados:

| Modelo | $R^2$ | $\Delta R^2$ | $p$ de prueba $F$ |
|-------|-------|-------------|-------------|
| Solo masa | 0.170 | — | — |
| Masa + estructura | 0.254 | +0.085 | 0.002 |
| Solo estructura | 0.013 | — | — |

**Interpretación:** La estructura bariónica agrega poder predictivo estadísticamente significativo (8.5%) a la discrepancia de masa más allá de la masa bariónica total. Las galaxias más concentradas exhiben mayores discrepancias a masa fija, consistente con la predicción de RTM de que la geometría estructural modula la relación entre la distribución bariónica y la dinámica observada.

**Limitación:** La estructura no puede predecir $v_{flat}$ más allá de la BTFR ($\Delta R^2 < 0.1\%$). Los residuos de la BTFR muestran correlación nula con los parámetros estructurales ($\rho \approx 0$). Al presente, la contribución estructural es un efecto secundario embebido dentro de la relación masa-velocidad dominante, no un reemplazo de ella.

**E.5 Conclusiones**

El análisis SPARC demuestra que la geometría estructural bariónica contiene información sobre la discrepancia de masa que la masa total sola no captura. Esto es consistente con la premisa central de RTM de que la organización topológica del medio bariónico modula la dinámica. Sin embargo, la magnitud de la contribución estructural (8.5% de varianza en $D$, contribución nula a $v_{flat}$) aún no respalda afirmaciones de eliminar la materia oscura. La cuestión de si indicadores estructurales refinados (entropía multiescala, potencia de modos de Fourier, textura cinemática) pueden aumentar sustancialmente la contribución independiente permanece como una pregunta empírica abierta abordada por la metodología propuesta en las Secciones 5--9.

**APÉNDICE F --- Validación empírica: Relajación topológica y turbulencia MHD en el viento solar**

El marco RTM dicta que la propagación de energía a través de cualquier medio está estrictamente gobernada por su coherencia topológica. Para validar esto a escalas astrofísicas, analizamos la turbulencia magnetohidrodinámica (MHD) del viento solar, un plasma no colisional donde los campos magnéticos actúan como la red estructural para el transporte de energía.

**F.1 La falacia del promedio estático**

El análisis robusto de Fase 2 muestra que el índice del viento solar no es una constante estática, sino una medida de **Relajación Topológica**. El índice evoluciona radialmente desde $\mathbf{- 1.52}$ (Topología Rígida Cerca del Sol a 0.1 UA) hasta $\mathbf{- 1.72}$ (Fluido Fractal del Espacio Profundo a 2.0 UA). Esta evolución radial está bien documentada en heliofísica (Chen 2016, Adhikari 2017, Shi 2021); la contribución de RTM es la interpretación dentro de un marco topológico unificado.

Sin embargo, tratar el viento solar en expansión como un medio estático y homogéneo introduce un defecto analítico crítico. Promediar estas métricas destruye la física dinámica subyacente y oscurece la evolución geométrica del plasma.

**F.2 Relajación topológica radial**

Para probar robustamente el marco RTM, analizamos la evolución radial del índice espectral desde 0.1 UA (Parker Solar Probe) hasta 2.0 UA (Ulysses). La trayectoria corregida por varianza demuestra que el plasma experimenta una **Relajación Topológica** macroscópica:

-   **Topología Rígida Cerca del Sol (0.1 UA):** En la vecindad inmediata del Sol, campos magnéticos intensos imponen una jerarquía rígida, altamente coherente y tipo 1D. El índice espectral empírico aquí converge a $- 1.52$, consistente con el límite teórico de Iroshnikov-Kraichnan (IK) ($- 3\text{/}2$).

-   **Fluido Fractal del Espacio Profundo (1.0 - 2.0 UA):** A medida que el plasma se expande y el campo magnético global se debilita, la restricción topológica rígida se descompone. El plasma "se relaja", fracturándose en un estado 3D isotrópico. El índice espectral cae a $- 1.68$ a $- 1.72$, consistente con el límite de turbulencia fractal de Kolmogorov ($- 5\text{/}3$).

La regresión lineal de esta relajación (pendiente = $- 0.18$ por década de UA, $R^{2} = 0.98$) demuestra que el desplazamiento espectral es sistemático, consistente con la signatura matemática de coherencia multiescala en decaimiento como predice RTM.

**F.3 Balance crítico y fricción topológica**

Evidencia adicional de geometría RTM se encuentra en la anisotropía espectral del plasma. Los datos empíricos demuestran que el espectro de energía cambia dependiendo del ángulo relativo al campo magnético local ($\theta_{B}$). La energía que atraviesa *perpendicularmente* las líneas de campo magnético encuentra "Fricción Topológica", forzando al sistema a un escalamiento fractal asimétrico conocido como Balance Crítico ($k_{\parallel} \propto k_{\bot}^{2\text{/}3}$). El plasma está geométricamente restringido por la red magnética.

**F.4 Intermitencia multifractal**

Finalmente, un análisis de las funciones de estructura de orden superior ($\zeta_{q}$) de datos MMS (Magnetospheric Multiscale) revela desviaciones severas del escalamiento monofractal lineal. Esto confirma que la energía del plasma no se disipa en una grilla perfectamente uniforme; más bien, la topología subyacente es un **multifractal**. Los vórtices de alta energía crean "agujeros" topológicos temporales o estructuras coherentes, reflejando perfectamente las concentraciones de energía discretas y heterogéneas predichas por RTM.

**Conclusión:** El viento solar se comporta como una red topológica en relajación dinámica. El mapeo de la evolución del plasma desde el límite de Iroshnikov-Kraichnan hasta el límite de Kolmogorov es consistente con la predicción de RTM de que la coherencia topológica gobierna el transporte de energía en medios no colisionales. Esta evolución radial está documentada independientemente en la literatura de heliofísica; la contribución de RTM es la interpretación unificada a través del marco del exponente de coherencia.

**APÉNDICE G: Análisis Red Team — Discrepancia de Masa, Problema de Diversidad y Predicción Directa de Velocidad**

**G.1 Metodología**

Todos los análisis usan el dataset SPARC table2.dat (Lelli et al. 2016) con razones masa-luminosidad estándar $\Upsilon_{disk} = 0.5\ M_\odot/L_\odot$ y $\Upsilon_{bul} = 0.7\ M_\odot/L_\odot$ a 3.6 $\mu$m. La velocidad bariónica se computa como $V_{bar}^2 = |V_{gas}| \cdot V_{gas} + \Upsilon_{disk} \cdot |V_{disk}| \cdot V_{disk} + \Upsilon_{bul} \cdot |V_{bul}| \cdot V_{bul}$. El brillo superficial se convierte a intensidad real mediante $I = 10^{-0.4 \cdot SB_{disk}}$. Todos los scripts están disponibles públicamente y reproducen los resultados con semillas aleatorias fijas.

**G.2 Análisis de Discrepancia de Masa**

La discrepancia de masa $D(r) = V_{obs}^2(r) / V_{bar}^2(r)$ cuantifica cuánta masa adicional (o dinámica modificada) se necesita en cada radio. Computamos $D_{outer}$ como la mediana de la discrepancia en el 40% más externo de la extensión radial de cada galaxia, donde se espera dominancia de materia oscura.

Definimos dos parámetros estructurales del perfil de SB: (a) **Concentración** $= \log_{10}(I_{inner} / I_{outer})$, donde interior y exterior se refieren al 30% más interno y más externo de los radios, y (b) **Pendiente de SB** $= d\log I / d\log r$ por regresión OLS.

**Resultados:**

| Predictor | $\rho$ de orden cero con $\log D_{outer}$ | $\rho$ parcial (controlando $M_{bar}$) | $p$ |
|-----------|----------------------------------------|---------------------------------------|-----|
| Concentración | +0.04 (ns) | **+0.346** | **0.0001** |
| Pendiente SB | $-$0.02 (ns) | **$-$0.245** | **0.007** |

Las correlaciones de orden cero son nulas porque masa y estructura están confundidas. Una vez que se controla la masa bariónica total, ambos parámetros estructurales predicen significativamente la discrepancia: galaxias más concentradas exhiben mayores discrepancias de masa a masa fija.

**Regresión multivariable:**

| Modelo | $R^2$ | $\Delta R^2$ | $F$-test $p$ |
|-------|-------|-------------|-------------|
| Solo $\log M_{bar}$ | 0.170 | — | — |
| $\log M_{bar}$ + concentración + pendiente SB | 0.254 | +0.085 | 0.002 |

La estructura añade 8.5% de varianza explicada, significativo con $p = 0.002$.

**G.3 El Problema de Diversidad**

El "problema de diversidad de curvas de rotación" (Oman et al. 2015) se refiere a la observación de que galaxias con $V_{flat}$ similar (y por lo tanto masa de halo inferida similar) muestran formas internas de curva de rotación marcadamente diferentes. El modelo estándar $\Lambda$CDM requiere dispersión en la concentración del halo para explicar esto, pero la dispersión predicha es más estrecha que la observada.

Probamos si la forma del perfil de SB predice la forma de la curva de rotación a $V_{flat}$ fijo. La **razón ascenso-a-planicie** se define como $R_{rise} = V_{inner} / V_{flat}$, donde $V_{inner}$ es la mediana de velocidad en el 40% más interno de los radios. La **pendiente interna** es $d\log V / d\log r$ computada sobre el mismo rango.

**Correlaciones parciales (controlando por $V_{flat}$), $n = 131$ galaxias:**

| Correlación | $\rho$ parcial | $p$ | ¿Significativo? |
|-------------|---------------|-----|-----------------|
| Pendiente SB $\rightarrow$ razón de ascenso | **+0.329** | **0.0001** | Sí (★★★) |
| Pendiente SB $\rightarrow$ pendiente interna | $-$0.166 | 0.059 | Marginal |
| Concentración SB $\rightarrow$ pendiente interna | $-$0.046 | 0.60 | No |

**Dentro de los terciles de $V_{flat}$:**

| Bin | $n$ | $\rho$(Pendiente SB, razón de ascenso) | $p$ |
|-----|-----|----------------------------------------|-----|
| Lentos (16–83 km/s) | 44 | +0.226 | 0.14 (ns) |
| Medios (83–163 km/s) | 43 | +0.320 | 0.037 (★) |
| Rápidos (168–331 km/s) | 44 | +0.348 | 0.021 (★) |

El efecto se replica en rotadores medios y rápidos pero no en enanas, donde los perfiles de SB son demasiado difusos para proporcionar contraste estructural significativo.

**Modelo multivariable para pendiente interna:**

| Modelo | $R^2$ | $\Delta R^2$ | $F$-test $p$ |
|-------|-------|-------------|-------------|
| Solo $V_{flat}$ | 0.326 | — | — |
| $V_{flat}$ + concentración SB + pendiente SB | 0.365 | +0.039 | 0.022 |

La estructura bariónica añade 3.9% de varianza explicada a la predicción de forma interna de la curva de rotación más allá de la masa, significativo con $p = 0.022$.

**Interpretación.** A masa fija, la distribución lumínica bariónica predice parcialmente la diversidad de curvas de rotación. Las galaxias con perfiles de SB más pronunciados (luz más concentrada centralmente) tienen rotación interna más rápida relativa a su porción plana. Esto es consistente con la predicción de RTM de que la organización topológica del medio bariónico modula la dinámica, y aborda un problema observacional específico (Oman et al. 2015) que $\Lambda$CDM no explica de forma natural a partir de las propiedades bariónicas solas.

**G.4 Prueba de Predicción Directa de Velocidad**

Como la prueba más exigente, intentamos predecir $V_{obs}(r)$ en cada radio usando solo el perfil de intensidad de SB y un parámetro libre:

$$v_{RTM}(r) = \kappa \cdot r^{1 - \alpha(r)/2}, \quad \alpha(r) = |d\log I / d\log r|$$

donde $I(r) = 10^{-0.4 \cdot SB_{disk}(r)}$ y $\kappa$ se ajusta por galaxia mediante minimización de $\chi^2$ contra $V_{obs}$ con errores observacionales.

**Comparación contra NFW y solo bariones:**

| Modelo | Parámetros libres | RMS mediano | Victorias (cara a cara vs NFW) |
|-------|-------------------|------------|-------------------------------|
| Solo bariones ($V_{bar}$) | 0 | 42.7% | — |
| **RTM** ($\kappa$) | **1** | **80.5%** | **2/135 (1.5%)** |
| Halo NFW ($V_{200}$, $c$) | 2 | 11.9% | 133/135 (98.5%) |

RTM pierde decisivamente. El gradiente bruto de SB $|d\log I/d\log r|$ diverge a radios grandes (perfil de disco exponencial), produciendo $\alpha \gg 2$ y llevando las velocidades predichas hacia cero — lo opuesto a una curva de rotación plana.

**Conclusión.** El mapeo actual de gradiente-de-SB-a-$\alpha$ es demasiado crudo para generar predicciones de velocidad competitivas. El éxito de RTM en dinámica galáctica reside en las correlaciones estructurales secundarias (Secciones G.2 y G.3), no en la predicción directa de curvas con este mapeo.

**G.5 Análisis de Dispersión de la RAR**

Probamos si la estructura local de SB reduce la dispersión de la Relación de Aceleración Radial (McGaugh et al. 2016). Usando 3,174 puntos radiales en 141 galaxias, la dispersión base de la RAR es $\sigma = 0.184$ dex. Tras corregir por gradiente y magnitud local de SB, la dispersión se reduce a $\sigma = 0.184$ dex — una reducción de 0.1%, lo cual es despreciable.

El análisis intra-galaxia muestra una tendencia débil pero significativa: la media intra-galaxia de $\rho$(gradiente SB, residuo RAR) $= +0.107$, con 64% de las galaxias mostrando correlación positiva ($t$-test $p = 0.01$). El efecto es real pero demasiado pequeño para ser operacionalmente útil.

**G.6 Resumen de Hallazgos Red Team**

| Hallazgo | Efecto | $p$ | Veredicto |
|----------|--------|-----|-----------|
| Concentración $\rightarrow$ discrepancia de masa (parcial) | $\rho = +0.346$ | 0.0001 | **Genuino, $\Delta R^2 = 8.5\%$** |
| Pendiente SB $\rightarrow$ razón de ascenso a $V_{flat}$ fijo | $\rho = +0.329$ | 0.0001 | **Genuino, problema de diversidad** |
| Predicción directa $v(r)$ vs NFW | RTM gana 2/135 | — | **RTM fracasa** |
| Reducción de dispersión RAR | 0.1% | — | **Despreciable** |
| Residuos BTFR vs estructura | $\rho \approx 0$ | $> 0.7$ | **Sin señal** |
| $\alpha = 2$ para curvas planas | Por definición | — | **Tautológico** |

**Conclusión final.** La estructura bariónica contiene información real y estadísticamente significativa sobre el patrón de discrepancia de masa y la diversidad de curvas de rotación que la masa total sola no captura. Esta información opera al nivel de $\sim 4$–$8.5\%$ de varianza explicada — secundaria pero genuina. No reemplaza los halos de materia oscura para predicción de velocidad, pero establece que la organización geométrica de la materia visible no es dinámicamente irrelevante, consistente con la premisa central de RTM.

---

*Todos los análisis son reproducibles. Scripts: rtm_pure_physics.py, rtm_flank_attack.py, astro_real_test.py. Datos: SPARC table2.dat (Lelli et al. 2016).*

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
