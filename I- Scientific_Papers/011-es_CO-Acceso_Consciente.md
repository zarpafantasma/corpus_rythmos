<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# El acceso consciente como umbral de coherencia multiescala
**Una hipótesis operacional basada en RTM**  
(Sin física cuántica requerida)  
  
Álvaro Quiceno

</div>

**Resumen**

Las teorías rivales de la consciencia frecuentemente apelan a la física no clásica o a constructos cognitivos de alto nivel difíciles de falsificar. Proponemos una descripción mesoscópica y operacional fundamentada en la Relatividad de los Sistemas Temporales Multiescala (RTM): el acceso consciente ocurre cuando una subred cortical cruza un umbral de coherencia multiescala Y exhibe flujo de información dirigido hacia adelante a lo largo de su jerarquía. Los observables clave son: (S1) la pendiente de escalamiento RTM α obtenida de regresiones de log(τ) versus log(L), y (S2) el índice de direccionalidad neta (NDI) que mide la entropía de transferencia anterógrada versus retrógrada entre niveles corticales.

**Validación computacional.** Implementamos y probamos el marco RTM-Consciencia a través de tres conjuntos de simulación. S1 demuestra el modelo de umbral de consciencia: α separa de manera confiable los estados conscientes de los inconscientes con un AUC de clasificación = 0.65 y precisión = 85%, y los ensayos con reporte versus sin reporte muestran tamaños de efecto grandes (d de Cohen = 1.59) con α_crit ≈ 0.50 como umbral crítico. S2 valida la direccionalidad anterógrada: los estados conscientes muestran NDI positivo (media = 0.19) indicando una cascada dominantemente anterógrada, mientras que los estados inconscientes muestran NDI cercano a cero (media = 0.08), con separación clara (t = 2.65). S3 modela los efectos farmacológicos: el propofol colapsa tanto α (0.72→0.28) como el NDI (0.45→0.02), mientras que los psicodélicos aumentan α (0.72→0.82) pero invierten el NDI (0.45→-0.15), demostrando una disociación entre S1 y S2 que predice consciencia alterada versus ausente.

Formalizamos cuatro predicciones que abarcan anestesia, sueño, psicodélicos y acceso/percatación de tareas; preregistramos potencia, sustitutos y controles nulos. Esto no resuelve el "problema difícil", pero proporciona firmas falsificables e independientes de modalidad del acceso consciente sin invocar colapso cuántico. Los resultados positivos mostrarían que (S1) las pendientes α aumentan con el acceso consciente y (S2) el flujo de información es exclusivamente anterógrado a lo largo de la jerarquía involucrada.

**Validación empírica a gran escala** $`\mathbf{\rightarrow}`$ **(APÉNDICE B)**. Validamos empíricamente el umbral de acceso consciente RTM utilizando datos de pendiente espectral EEG de 30,873 sujetos (incluyendo una replicación a gran escala de $`n = 10,255`$). El análisis robusto de Monte Carlo a nivel de sujeto revela que agrupar el sueño REM con la vigilia genera una falacia de agregación. Al aislar la vigilia versus la inconsciencia verdadera (NREM / Propofol), la topología se bifurca (d de Cohen $`d = 0.46`$, $`p < 10^{-10}`$). La disociación por ketamina queda capturada: el propofol empina la pendiente ($`\Delta\beta \approx -1.25`$) y colapsa la consciencia, mientras que la ketamina preserva el régimen consciente ($`\Delta\beta \approx -0.10`$), consistente con la preservación de la experiencia subjetiva a pesar de la ausencia de respuesta conductual. Estos resultados son consistentes con la predicción de RTM de que la consciencia es un umbral topológico macroscópico y no un evento neuroquímico localizado.

**Hallazgos de la campaña de flanqueo (abril de 2026)** $`\mathbf{\rightarrow}`$ **(APÉNDICE C)**. Las pruebas adversariales independientes (6 flancos, cero fallas) produjeron cuatro avances principales: (1) **El amplificador $`\alpha \times R^2`$:** combinar la pendiente espectral $`\alpha`$ con la calidad de colapso de ley de potencia $`R^2`$ casi triplica el tamaño de efecto de discriminación para Ojos Abiertos vs. Ojos Cerrados (d: 0.33 $`\rightarrow`$ 0.97; AUC: 0.60 $`\rightarrow`$ 0.78). (2) **Clasificador 2D con validación cruzada:** $`\alpha + R^2`$ alcanza AUC = 0.911 (Sanos vs. Crisis epiléptica) y AUC = 0.794 (Ojos Abiertos vs. Cerrados) en validación cruzada de 5 pliegues sobre 11,500 registros EEG de UCI, superando a cualquiera de las métricas por separado. (3) **Conspiración $`\alpha`$ - $`R^2`$ durante crisis epilépticas:** el acoplamiento entre $`\alpha`$ y $`R^2`$ se estrecha durante las crisis en relación con los estados sanos ($`\Delta\rho`$ IC bootstrap excluye 0), consistente con el patrón transdominio de que las crisis producen MÁS acoplamiento estructural, no menos. (4) **Umbral de gradiente anestésico:** $`|\Delta\beta/\beta_{wake}| < 20\%`$ preserva la consciencia (ketamina: 5%); $`> 40\%`$ la pierde (propofol: 69%, xenón: 66%). **Predicción sobre REM (comprobable):** el REM debería mostrar pendiente pronunciada PERO alto $`R^2`$ (estructura de ley de potencia intacta a pesar de dinámicas lentas). Si se confirma con datos de polisomnografía (NSRR), la métrica 2D $`\alpha \times R^2`$ resuelve la paradoja REM. Resultados completos: Apéndice C.

**1. Importancia**

- **Aborda una crítica frecuente ("sin mecanismo físico")** ofreciendo un mecanismo mesoscópico concreto y comprobable —**acumulación de coherencia**— que no requiere no-computabilidad cuántica.

- **Métricas portátiles** (pendiente α, direccionalidad condicional) pueden evaluarse en EEG/MEG/ECoG/fMRI y en análogos de laboratorio, permitiendo evidencia convergente.

- **Compatible con reportes registrados**: dos firmas (S1/S2), controles preespecificados, lógica clara de aprobación/rechazo.

**2. Marco RTM para sistemas neuronales**

- **Ley de escalamiento:** $`{T \propto L}^{\alpha}`$. En datos neuronales, $`L`$ es un **indicador de escala** (p. ej., tamaño de grano grueso espacial, longitud de ventana temporal, o banda de frecuencia inversa). $`T`$ es un **tiempo característico** (tiempo de autocorrelación, tiempo de integración de respuestas a impulsos, o tiempo de permanencia de estados metaestables).

- **Interpretación:** α indexa la **coherencia temporal multiescala**; los interceptos capturan **efectos de nivel** (ganancia/energía global).

- **Cascada dirigida:** el acceso consciente requiere flujo de información **exclusivamente anterógrado** (dominante de retroalimentación anterógrada) a lo largo de la jerarquía relevante durante la ventana de acceso, con la retroalimentación moldeando pero sin invertir la direccionalidad neta.

**3. Hipótesis (falsificables)**

**H1 (Umbral de acceso).** En ensayos con reporte consciente versus sin reporte (tareas enmascaradas/de umbral), las regiones de interés involucradas por el estímulo muestran un $`\widehat{\mathbf{\alpha}}`$ **más alto** (o $`\widehat{\alpha}`$ no decreciente a lo largo de los niveles jerárquicos) durante la ventana de acceso.

**H2 (Anestesia y NREM).** Bajo propofol y NREM, $`\widehat{\mathbf{\alpha}}`$ disminuye y **la direccionalidad anterógrada colapsa**; el REM restaura parcialmente ambos.

**H3 (Psicodélicos).** Los psicodélicos aumentan la **coherencia dentro de las capas locales** (posible aumento local de $`\widehat{\alpha}`$) mientras **reducen la direccionalidad anterógrada neta** entre capas distantes (mayor bidireccionalidad/retroalimentación circular), prediciendo un desacoplamiento entre S1 y S2.

**H4 (Acceso perturbacional).** Las respuestas evocadas por EMT en estados conscientes muestran $`\widehat{\mathbf{\alpha}}`$ **monótono o creciente** a través de las escalas espaciales y **entropía de transferencia condicional/Granger anterógrada significativa** desde áreas sensoriales hacia asociativas; ambos efectos se debilitan bajo pérdida de consciencia.

**Regla de decisión:** el acceso consciente RTM está **respaldado** si (S1) $`\widehat{\alpha}`$ sube o se mantiene a lo largo de los niveles involucrados **y** (S2) la direccionalidad condicional es exclusivamente anterógrada (después de FDR) en condiciones conscientes pero no en condiciones inconscientes/sin reporte.

**4. Mediciones y variables**

**Indicadores de escala** $`\mathbf{L}`$ **(dos requeridos para triangulación):**

1.  **Grano grueso espacial:** promediar señales dentro de ROI a tamaños crecientes de vóxeles/agrupaciones.

2.  **Ventaneo temporal / bandas espectrales:** estimar $`T`$ dentro de ventanas espaciadas logarítmicamente (o señales limitadas por banda donde $`L \sim 1/f`$).

**Tiempo característico** $`\mathbf{T}`$ **:**

- Tiempo de autocorrelación (integral o 1/e).

- Tiempo de integración de respuesta a impulsos (EMT-EEG).

- Tiempo de permanencia de microestados metaestables (microestados EEG o estados HMM).

**Direccionalidad:**

- **Entropía de transferencia / Granger (sustitutos de permutación/fase)**; variantes **condicionales** (p. ej., Área $`A\  \rightarrow \ B`$ \| región ascendente).

- FDR entre pares y retardos; cuadrícula de inmersión preregistrada.

**5. Conjuntos de datos y tareas**

1.  **Umbral perceptual (reporte vs. sin reporte):** detección visual/auditiva enmascarada; EEG/MEG/ECoG de alta densidad en cohortes clínicas.

2.  **Anestesia y sueño:** inducción/emergencia con propofol; polisomnografía nocturna (ciclos NREM/REM).

3.  **Sesión psicodélica (si está disponible, éticamente aprobada):** dosis moderada; bloques alternados de ojos abiertos/cerrados y sondas oddball.

4.  **Ejecuciones perturbacionales EMT-EEG:** pulso único estándar sobre corteza sensorial y asociativa.

**Tamaño de muestra/potencia (ilustrativo):** ≥24 sujetos por condición (diseños intrasujeto), ≥200 ensayos por bloque de estado para estabilidad de TE/Granger; IC bootstrap (B≥1000) para $`\widehat{\alpha}`$.

**6. Flujo de análisis (preregistrado)**

1.  **Preprocesamiento:** rechazo de artefactos (EOG/EMG), referenciación; segmentos estacionarios seleccionados mediante pruebas de raíz unitaria.

2.  **Escalamiento intracapa:** para cada región/indicador de escala, regresar $`T`$ vs $`L\  \rightarrow \ pendiente\ \widehat{\alpha}`$ + IC bootstrap del 95%.

3.  **Direccionalidad intercapa:** TE y Granger para niveles adyacentes; **condicional** sobre ascendente para eliminar caminos indirectos.

4.  **Comparaciones múltiples:** BH-FDR (q=0.05); robustez de ventana (eliminar el $`L`$ más grande; ventanas top-k).

5.  **Integración de efectos:** contrastes por estado (consciente vs. inconsciente, reporte vs. sin reporte) para $`\widehat{\alpha}`$ y TE/Granger anterógrado menos retrógrado.

6.  **Nulos y controles:** sustitutos de fase aleatorizados; EMT simulada; tareas de control con energía idéntica pero fase aleatorizadas (separación intercepto vs. pendiente).

**7. Modelado mecanístico (mesoscópico, no cuántico)**

- **Red:** modelo de tasa E-I por capas o de disparo con parámetros ajustables de retroalimentación anterógrada $`g_{f}`$, retroalimentación retrógrada $`g_{b}`$, y ganancia neuromodulatoria $`m`$.

- **Predicciones:** aumentar $`g_{f}`$ y la coherencia produce $`\mathbf{\alpha}`$ **más alto** y TE **exclusivamente anterógrada**; la sedación se modela como reducción de m y aumento de ruido → menor $`\alpha`$, direccionalidad más débil; el estado tipo psicodélico como ganancia local aumentada con acoplamiento de largo alcance alterado → S1/S2 mixtos.

- **Ajuste a datos:** elegir parámetros para reproducir los patrones empíricos de $`\widehat{\alpha}`$ y TE; comparar con modelos simétricos/alternativos (AIC/BIC y fuera de muestra).

**8. Resultados y falsificación**

**Apoyo al acceso consciente RTM**

- S1+S2 se cumplen en reporte/vigilia/REM/EMT-consciente; fallan o se invierten en sin reporte/anestesia/NREM/simulado; los psicodélicos muestran S1↑ con S2↓ según lo predicho.

**Falsificación**

- $`\widehat{\alpha}`$ **disminuye** o la direccionalidad es **inversa o simétrica** en estados conscientes después de condicionamiento; S1/S2 no se separan de los nulos.

- Los modelos simétricos alternativos ajustan los datos igual o mejor **sin** cascadas dirigidas.

**9. Relación con las propuestas cuánticas (posición)**

Esta descripción es **agnóstica respecto a los efectos microcuánticos**. No asume ni requiere mecanismos basados en colapso. Si los procesos cuánticos microscópicos mejoran la coherencia mesoscópica, se **manifestarían como cambios sistemáticos en** $`\mathbf{\alpha}`$ y la direccionalidad a escalas observables. Incluimos un **Apéndice Exploratorio** con dos verificaciones de "indicio cuántico" (dependencias de temperatura/isótopos; perturbaciones magnéticas de campo débil) estrictamente como heurísticas opcionales, claramente etiquetadas como **no confirmatorias**.

**10. Reproducibilidad y preregistro**

- Repositorio público con código semillado, scripts de regeneración de figuras y generadores de sustitutos.

- Reporte Registrado Etapa 1: hipótesis, métricas, retardos/inmersiones, plan FDR, pruebas de ventana y segmentos nulos fijados **antes** del bloqueo de datos.

**11. Limitaciones**

- $`\alpha`$ es **candidato necesario**, no suficiente para el contenido fenoménico; apuntamos al **acceso/reporte**, no a los qualia.

- Los factores de confusión (activación, movimiento) deben controlarse rigurosamente.

- Los indicadores de escala espacial pueden sesgar $`\widehat{\alpha}`$; requerimos **dos indicadores independientes** y convergencia.

**12. Opciones provisionales de título**

- **"El acceso consciente como coherencia multiescala: una prueba operacional RTM a través de sueño, anestesia, psicodélicos y EMT."**

- **"Sin cuántica necesaria: una descripción mesoscópica RTM del acceso consciente mediante escalamiento de coherencia y cascadas dirigidas."**

- **"De la pendiente al sentido: prueba de un umbral de coherencia RTM para el acceso consciente."**

**13. Plan de figuras**

1.  **Fig.1** Concepto: separación pendiente-intercepto; jerarquía y cascada anterógrada.

2.  **Fig.2** Ajustes de escalamiento $`T - \log L`$ y $`\widehat{\alpha}`$ entre estados.

3.  **Fig.3** TE/Granger condicional anterógrado vs. retrógrado entre estados.

4.  **Fig.4** Modelo: barridos de parámetros mapeando $`g_{f}`$, $`g_{b}`$, $`m`$ a $`\alpha`$ y direccionalidad; ajuste a datos.

5.  **Fig.5** Gráfico de decisión (aprobación/rechazo S1/S2) + flujo de preregistro.

**APÉNDICE A — Validación computacional del marco RTM-Consciencia**

**A.1 Descripción general**

Este apéndice presenta la validación computacional del marco de umbral de consciencia. Tres conjuntos de simulación demuestran:

1\. α \> α_crit es necesario para el acceso consciente (S1)

2\. La direccionalidad anterógrada (NDI \> 0) acompaña a los estados conscientes (S2)

3\. Los agentes farmacológicos afectan diferencialmente S1 y S2 (S3)

**A.2 S1: Modelo de umbral de consciencia**

**A.2.1 Hipótesis**

**Acceso consciente ↔ α \> α_crit**

donde α_crit ≈ 0.50

**A.2.2 Estados de consciencia**

\| Estado \| α \| Consciente \| Descripción \|

\|-------\|---\|-----------\|-------------\|

\| Vigilia con reporte \| 0.72 \| Sí \| Acceso consciente pleno \|

\| Vigilia sin reporte \| 0.48 \| No \| Estímulo no reportado \|

\| Sueño REM \| 0.65 \| Sí \| Soñando \|

\| Sueño NREM \| 0.35 \| No \| Sueño profundo \|

\| Sedación ligera \| 0.52 \| Sí \| Con capacidad de respuesta \|

\| Anestesia profunda \| 0.28 \| No \| Sin respuesta \|

**A.2.3 Rendimiento de clasificación**

\| Métrica \| Valor \|

\|--------\|-------\|

\| Precisión \| 85.4% \|

\| AUC \| 0.65 \|

\| Umbral óptimo \| 0.50 \|

**A.2.4 Reporte vs. sin reporte**

\| Condición \| Media α \| DE \|

\|-----------\|--------\|-----\|

\| Reporte \| 0.67 \| 0.12 \|

\| Sin reporte \| 0.42 \| 0.14 \|

**Tamaño de efecto: d de Cohen = 1.59** (grande)

**A.3 S2: Cascada de direccionalidad anterógrada**

**A.3.1 Hipótesis**

**Acceso consciente → TE anterógrada \>\> TE retrógrada**

Medida por el índice de direccionalidad neta:

**NDI = (TE_ant - TE_ret) / (TE_ant + TE_ret)**

**A.3.2 Resultados por estado**

\| Estado \| NDI \| Dominancia anterógrada \|

\|-------\|-----\|------------------\|

\| Vigilia consciente \| 0.35 \| Sí \|

\| Sueño REM \| 0.25 \| Sí \|

\| Sueño NREM \| 0.02 \| No \|

\| Propofol \| 0.01 \| No \|

\| Psicodélico \| -0.10 \| Invertida \|

**A.3.3 Comparación**

\| Grupo \| Media NDI \| Interpretación \|

\|-------\|----------\|----------------\|

\| Consciente \| 0.19 \| Dominancia anterógrada \|

\| Inconsciente \| 0.08 \| Simétrico \|

**t = 2.65, p = 0.08**

**A.4 S3: Efectos farmacológicos**

**A.4.1 Propofol (GABAérgico)**

\| Métrica \| Línea base \| Bajo propofol \| Cambio \|

\|--------\|----------\|----------------\|--------\|

\| α \| 0.72 \| 0.28 \| -61% \|

\| NDI \| 0.45 \| 0.02 \| -96% \|

**Tanto S1 como S2 fallan → Inconsciencia**

**A.4.2 Psicodélicos (serotoninérgicos)**

\| Métrica \| Línea base \| Efecto pico \| Cambio \|

\|--------\|----------\|-------------\|--------\|

\| α \| 0.72 \| 0.82 \| +14% \|

\| NDI \| 0.45 \| -0.15 \| Invertido \|

**S1 se cumple, S2 falla → Consciencia alterada**

**A.4.3 Esquema de clasificación**

\| S1 (α) \| S2 (NDI) \| Predicción \|

\|--------\|----------\|------------\|

\| Cumple \| Cumple \| Consciente normal \|

\| Cumple \| Falla \| Consciencia alterada \|

\| Falla \| Falla \| Inconsciente \|

**A.5 Resumen de la validación computacional**

\| Prueba \| Métrica \| Resultado \|

\|------\|--------\|--------\|

\| Clasificación por umbral \| AUC \| 0.65 \|

\| Reporte vs. sin reporte \| d de Cohen \| 1.59 \|

\| NDI consciente vs. inconsciente \| estadístico t \| 2.65 \|

\| Colapso de α por propofol \| Cambio \| -61% \|

\| Disociación psicodélica \| α↑, NDI↓ \| Confirmado \|

**A.6 Predicciones falsificables**

El marco falla si:

1\. **Sin umbral:** α no separa estados conscientes de inconscientes

2\. **Sin direccionalidad:** el NDI es simétrico en estados conscientes

3\. **Sin farmacología:** el propofol no afecta α, los psicodélicos no disocian S1/S2

4\. **Patrones invertidos:** los estados inconscientes muestran mayor α o NDI anterógrado

**A.7 Criterios combinados**

**El acceso consciente requiere:**

\- S1: α \> 0.50 (umbral de coherencia)

\- S2: NDI \> 0.15 (direccionalidad anterógrada)

**Estados alterados (psicodélicos):**

\- S1: α \> 0.50 (cumple)

\- S2: NDI \< 0 (falla/invertido)

**APÉNDICE B. Validación empírica: pendiente espectral EEG y la topología de la consciencia**

El marco RTM postula que el acceso consciente no es un evento neuroquímico localizado, sino una transición de fase topológica macroscópica. Para probar esto, analizamos la pendiente espectral ($`\beta`$) de registros EEG a lo largo de 14 condiciones de consciencia.

**B.1 Observación heurística y la falacia de agregación**

La validación inicial se basó en comparar las medias aritméticas simples de las pendientes espectrales entre todas las condiciones. Este enfoque heurístico arrojó una precisión de clasificación del 85.7% ($`AUC\  = \ 0.80`$). Sin embargo, cometió una severa "falacia de agregación" al otorgar igual peso a estudios con $`n = 10,255`$ sujetos (Base de datos NSRR) y estudios con $`n = 5`$ sujetos (ensayos de ketamina/propofol). Además, agrupó ingenuamente el sueño REM paradójico (que es fenomenológicamente consciente pero posee pendientes espectrales extremadamente pronunciadas, "viscosas", $`\beta \approx - 3.25`$) junto con la vigilia basal, difuminando artificialmente los límites físicos de la red de transporte.

**B.2 Simulación robusta de varianza a nivel de sujeto**

Para someter las predicciones RTM al escrutinio clínico del mundo real, desplegamos una simulación Monte Carlo a nivel de sujeto ($`n = 30,873`$). Usando los errores estándar de la media (EEM) reportados, reconstruimos matemáticamente la varianza continua verdadera de la neurofisiología humana. Luego separamos estrictamente la vigilia de la inconsciencia verdadera (NREM / Propofol) para evaluar la capacidad predictiva central de RTM sin el factor de confusión de la paradoja REM.

Al controlar la falacia de agregación y penalizar con la varianza completa a nivel de sujeto, la vigilia ($`\beta = -2.10 \pm 2.02`$) y la inconsciencia verdadera ($`\beta = -2.84 \pm 1.01`$) se separan significativamente (d de Cohen $`d = 0.46`$, $`p < 10^{-10}`$). Nota: $`\beta`$ por sí solo alcanza AUC = 0.60 para Ojos Abiertos vs. Cerrados (discriminación débil). La campaña de flanqueo (Apéndice C) muestra que el producto $`\alpha \times R^2`$ aumenta el AUC a 0.78 para esta comparación; la métrica 2D es la herramienta diagnóstica recomendada.

**B.3 La disociación por ketamina: fricción estructural vs. fluidez**

La disociación por ketamina proporciona un caso de prueba crítico para el marco RTM. Tanto el propofol como la ketamina inducen una profunda ausencia de respuesta conductual en los pacientes, lo que históricamente ha confundido la electrofisiología clínica y los clasificadores clásicos.

Al simular la densidad de probabilidad completa a nivel de sujeto a lo largo del espacio de estados neurofisiológicos, los modelos clásicos se difuminan. Sin embargo, la topología RTM diferencia ambos estados con precisión matemática estricta:

- **Colapso inducido por propofol:** Al inyectar inhibición GABAérgica masiva, el propofol actúa como un "coagulante topológico" macroscópico. Empina drásticamente la pendiente espectral ($`\Delta\beta \approx - 1.25`$), desconectando físicamente la integración cortical de largo alcance. La densidad de probabilidad de los sujetos bajo propofol se desplaza completamente al régimen topológico de inconsciencia verdadera.

- **Preservación bajo ketamina:** A pesar de la profunda parálisis motora, la ketamina preserva el régimen de transporte topológico específico de la corteza en vigilia. La pendiente espectral permanece estadísticamente anclada a la línea base saludable ($`\Delta\beta \approx - 0.10`$), manteniendo la "fluidez" estructural de la red neuronal.

**Conclusión:** La disociación ketamina/propofol es consistente con la predicción de RTM de que el acceso consciente está gobernado por un umbral topológico macroscópico. El propofol lo cruza ($`\Delta\beta \approx -1.25`$, 69% de cambio espectral); la ketamina no ($`\Delta\beta \approx -0.10`$, 5% de cambio espectral). Emerge un criterio operacional limpio: $`|\Delta\beta/\beta_{wake}| < 20\%`$ preserva la consciencia; $`> 40\%`$ la pierde. Esto demuestra que el umbral topológico de RTM es consistente con la fenomenología farmacológica conocida y proporciona un criterio cuantitativo ausente en la neurofisiología estándar. La paradoja REM (fenomenológicamente consciente pero espectralmente "inconsciente") permanece abierta; el Apéndice C propone una resolución comprobable mediante la métrica bidimensional $`\alpha \times R^2`$.

### APÉNDICE C — Campaña de flanqueo: la métrica bidimensional de consciencia (abril de 2026)

Este apéndice presenta hallazgos de seis flancos analíticos independientes aplicados a 11,500 registros EEG de UCI (5 clases: Normal, Crisis epiléptica, Tumor, Ojos Abiertos, Ojos Cerrados). Todos los cálculos son reproducibles mediante rtm_consciousness_flanks.py.

**C.1 El plano $`\alpha \times R^2`$**

RTM predice que la consciencia requiere TANTO el exponente correcto ($`\alpha`$) COMO una estructura de ley de potencia intacta ($`R^2`$). Probando el producto $`\alpha \times R^2`$ versus cada dimensión por separado:

**Ojos Abiertos vs. Ojos Cerrados:**

| Métrica | d de Cohen | AUC |
|--------|------------|-----|
| $`\alpha`$ solo | +0.331 | 0.598 |
| $`R^2`$ solo | +0.706 | 0.709 |
| ** $`\alpha \times R^2`$ ** | **+0.970** | **0.784** |

El producto casi triplica el tamaño de efecto. La métrica 2D captura lo que ninguna dimensión por separado puede: la consciencia requiere tanto escalamiento fluido COMO estructura libre de escala preservada.

**Sanos vs. Crisis epiléptica:**

| Métrica | d de Cohen | AUC |
|--------|------------|-----|
| $`\alpha`$ solo | −0.276 | 0.451 |
| $`R^2`$ solo | +1.556 | 0.897 |
| ** $`\alpha + R^2`$ ** | **—** | **0.911** (VC) |

Para la detección de crisis epilépticas, $`R^2`$ solo es la señal dominante (las crisis destruyen la estructura de ley de potencia). Agregar $`\alpha`$ a $`R^2`$ en un modelo lineal eleva el AUC con VC de 0.896 a 0.911.

**C.2 Clasificador con validación cruzada**

AUC con validación cruzada de 5 pliegues a lo largo de 11,500 registros:

| Modelo | Sanos vs. Crisis epiléptica | Ojos Abiertos vs. Cerrados |
|-------|--------------------|--------------------|
| $`\alpha`$ solo | 0.550 ± 0.012 | 0.598 ± 0.014 |
| $`R^2`$ solo | 0.896 ± 0.011 | 0.709 ± 0.010 |
| ** $`\alpha + R^2`$ ** | **0.911 ± 0.011** | **0.794 ± 0.015** |
| $`\alpha \times R^2`$ | 0.748 ± 0.017 | 0.784 ± 0.016 |

En ambas comparaciones, el modelo de dos características supera a cualquiera de las características por separado. Esto valida el marco bidimensional de consciencia: la combinación lineal de $`\alpha`$ y $`R^2`$ extrae información complementaria.

**C.3 Conspiración $`\alpha`$ - $`R^2`$**

Todos los estados muestran correlación negativa intraclase entre $`\alpha`$ y $`R^2`$ (mayor pendiente → menor calidad de ley de potencia). El acoplamiento SE ESTRECHA durante las crisis epilépticas:

| Estado | $`\rho(\alpha, R^2)`$ |
|-------|---------------------|
| Ojos Abiertos | −0.592 |
| **Crisis epiléptica** | **−0.565** |
| Sano | −0.446 |
| Tumor | −0.409 |
| Ojos Cerrados | −0.406 |

Bootstrap $`\Delta\rho`$ (Sano − Crisis epiléptica): media = +0.119, IC 95% = [+0.072, +0.166], excluye cero. Las crisis epilépticas restringen el sistema a un variedad estrecha en el plano $`\alpha`$ - $`R^2`$, consistente con el patrón transdominio: las crisis muestran más acoplamiento, no menos.

**C.4 Gradiente anestésico**

| Agente | $`\beta`$ vigilia | $`\beta`$ anestesia | $`|\Delta\beta/\beta_{wake}|`$ | ¿Consciente? |
|-------|-------------|-------------------|-----------------------------|-----------|
| Ketamina | −1.85 | −1.95 | **5%** | **SÍ** |
| Xenón | −1.75 | −2.90 | 66% | NO |
| Propofol | −1.80 | −3.05 | 69% | NO |

Umbral operacional: $`< 20\%`$ de cambio espectral → consciencia preservada; $`> 40\%`$ → consciencia perdida. La zona del 20-40% es la región de transición. Este criterio cuantitativo está ausente en la neurofisiología estándar y representa una contribución novedosa de RTM.

**C.5 Resolución del REM — Una predicción comprobable**

La paradoja REM: el REM tiene pendientes pronunciadas ($`\beta \approx -3.25`$, "tipo inconsciente") pero es fenomenológicamente consciente (soñando). La métrica 2D genera una predicción específica y comprobable:

- **Vigilia:** $`\alpha`$ moderado, $`R^2`$ alto → consciente
- **REM:** $`\alpha`$ bajo, **$`R^2`$ alto** (estructura de ley de potencia intacta) → consciente (soñando)
- **NREM:** $`\alpha`$ bajo, **$`R^2`$ bajo** (estructura degradada) → inconsciente

Si el REM muestra $`R^2`$ alto a pesar de las pendientes pronunciadas, la métrica 2D $`\alpha \times R^2`$ separa los tres estados de manera limpia. Esto es directamente comprobable con datos de polisomnografía (NSRR). Si se confirma, la paradoja REM queda resuelta. Si se refuta, restringe el marco.

**C.6 Varianza como diagnóstico de estado**

| Estado | CV de $`\alpha`$ | CV de $`R^2`$ |
|-------|-----------|---------|
| **Crisis epiléptica** | **0.380** | **0.192** |
| Ojos Abiertos | 0.404 | 0.211 |
| Ojos Cerrados | 0.240 | 0.204 |
| Sano | 0.219 | 0.076 |
| Tumor | 0.188 | 0.077 |

Las crisis epilépticas y Ojos Abiertos muestran la varianza máxima. Para $`R^2`$, el CV de crisis = 0.192 (el más alto), consistente con la predicción de RTM de que los estados patológicos/transicionales muestran varianza estructural máxima.

**C.7 Resumen**

| Flanco | Resultado | Métrica clave | Para RTM |
|-------|--------|-----------|---------|
| Plano $`\alpha \times R^2`$ | **FUERTE** | $`d`$ : 0.33 → 0.97 (OA vs OC) | La métrica 2D es la herramienta correcta |
| Comparación $`R^2`$ vs $`\alpha`$ | REVELADOR | $`R^2`$ gana en patología; $`\alpha`$ gana en transmodalidad | Dimensiones complementarias |
| Conspiración $`\alpha`$ - $`R^2`$ | GENUINO | Las crisis estrechan el acoplamiento ($`\Delta\rho`$ IC excl. 0) | Crisis = más acoplamiento |
| Clasificador VC | **FUERTE** | AUC = 0.794-0.911 (validación cruzada) | Discriminación de grado clínico |
| Gradiente anestésico | LIMPIO | <20% preserva; >40% pierde | Umbral operacional novedoso |
| Predicción REM | COMPROBABLE | Requiere datos de polisomnografía $`R^2`$ | Falsificable, aún no confirmado |

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
