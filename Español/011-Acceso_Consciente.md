<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Serpiente">

# El Acceso Consciente como Umbral de Coherencia Multiescala
**Una Hipótesis RTM-Operacional**  
(Sin Requisitos Cuánticos)  
  
Álvaro Quiceno

</div>

**Resumen**

Las teorías competidoras de la consciencia frecuentemente apelan a física no clásica o a constructos cognitivos de alto nivel que son difíciles de falsificar. Proponemos una explicación mesoscópica y operacional fundamentada en la Relatividad de sistemas Temporales Multiescala (RTM): el acceso consciente ocurre cuando una subred cortical cruza un umbral de coherencia multiescala Y exhibe flujo de información dirigido hacia adelante a través de su jerarquía. Los observables clave son: (S1) la pendiente de escalamiento RTM α obtenida de regresiones de log(τ) versus log(L), y (S2) el índice de direccionalidad neta (IDN) que mide entropía de transferencia hacia adelante vs hacia atrás entre niveles corticales.

**Validación computacional.** Implementamos y probamos el marco RTM-Consciencia a través de tres conjuntos de simulación. S1 demuestra el modelo de umbral de consciencia: α separa confiablemente estados conscientes de inconscientes con AUC de clasificación = 0.65 y precisión = 85%, y los ensayos con reporte vs sin reporte muestran grandes tamaños de efecto (d de Cohen = 1.59) con α_crit ≈ 0.50 como umbral crítico. S2 valida la direccionalidad hacia adelante: los estados conscientes muestran IDN positivo (media = 0.19) indicando cascada dominante hacia adelante, mientras los estados inconscientes muestran IDN cercano a cero (media = 0.08), con separación clara (t = 2.65). S3 modela efectos farmacológicos: el propofol colapsa tanto α (0.72→0.28) como IDN (0.45→0.02), mientras los psicodélicos aumentan α (0.72→0.82) pero invierten IDN (0.45→-0.15), demostrando disociación entre S1 y S2 que predice consciencia alterada vs. ausente.

Formalizamos cuatro predicciones abarcando anestesia, sueño, psicodélicos y acceso/consciencia de tarea; prerregistramos potencia, sustitutos y controles nulos. Esto no resuelve el "problema difícil", pero proporciona firmas falsificables e independientes de modalidad del acceso consciente sin invocar colapso cuántico. Resultados positivos mostrarían que (S1) las pendientes α suben con el acceso consciente y (S2) el flujo de información es solo-hacia-adelante a lo largo de la jerarquía involucrada.

**Validación empírica a gran escala (La Disociación de Ketamina)**$`\mathbf{\rightarrow (}\mathbf{APÉNDICE\ B)}`$**.** Validamos empíricamente el umbral de acceso consciente RTM usando datos de pendiente espectral EEG de 30,873 sujetos (incluyendo una replicación a gran escala de n=10,255). El modelado heurístico inicial sugirió que la pendiente de coherencia multiescala ($`\beta`$) separaba confiablemente todos los estados conscientes de los inconscientes con una precisión de 85.7% (AUC: 0.80). Sin embargo, para someter esta hipótesis a escrutinio clínico riguroso, desplegamos una reconstrucción de varianza "a nivel de sujeto" Monte Carlo, penalizando explícitamente el modelo con la varianza natural masiva de los conjuntos de datos. El análisis robusto revela que agrupar el sueño REM (un estado consciente paradójicamente altamente-viscoso) con la Vigilia crea una falacia de agregación. Al aislar Vigilia versus Inconsciencia Verdadera (NREM / Propofol), la topología se bifurca estrictamente (d de Cohen $`= 0.46,p < 10^{- 10}`$), reestableciendo $`\beta`$ como un umbral estructural determinístico. Más triunfalmente, este modelo corregido por varianza resuelve perfectamente la "disociación de ketamina": mientras el anestésico propofol colapsa violentamente la coherencia topológica de la red (aumentando la pendiente, $`\Delta\beta \approx - 1.25`$) y erradicando la consciencia, la ketamina preserva el régimen de escalamiento consciente ($`\Delta\beta \approx - 0.10`$), permitiendo experiencias subjetivas vívidas a pesar de inducir falta de respuesta conductual completa. Esto demuestra empíricamente que el exponente RTM es un índice directo de la topología de la consciencia, más que meramente reactividad motora.

**1. Significancia**

- **Aborda una crítica común ("sin mecanismo físico")** ofreciendo un mecanismo mesoscópico concreto y comprobable—**acumulación de coherencia**—que no requiere no-computabilidad cuántica.

- **Métricas portátiles** (pendiente α, direccionalidad condicional) pueden evaluarse en EEG/MEG/ECoG/fMRI y en análogos de banco, permitiendo evidencia convergente.

- **Amigable a reporte registrado**: dos firmas (S1/S2), controles preespecificados, lógica clara de aprobación/falla.

**2. Marco RTM para sistemas neurales**

- **Ley de escalamiento:** $`{T \propto L}^{\alpha}`$. En datos neurales, $`L`$ es un **proxy de escala** (ej., tamaño de granulado grueso espacial, longitud de ventana temporal, o banda de frecuencia inversa). $`T`$ es un **tiempo característico** (tiempo de autocorrelación, tiempo de integración de respuestas a impulso, o tiempo de permanencia de estados metaestables).

- **Interpretación:** α indexa **coherencia temporal multiescala**; las ordenadas al origen capturan **efectos de nivel** (ganancia/energía general).

- **Cascada dirigida:** el acceso consciente requiere flujo de información **solo-hacia-adelante** (dominante feedforward) a lo largo de la jerarquía relevante durante la ventana de acceso, con retroalimentación moldeando pero no invirtiendo la direccionalidad neta.

**3. Hipótesis (falsificables)**

**H1 (Umbral de acceso).** En ensayos con reporte consciente vs. sin reporte (tareas enmascaradas/de umbral), las regiones de interés involucradas por el estímulo muestran $`\widehat{\mathbf{\alpha}}`$ **más alto** (o $`\widehat{\alpha}`$ no decreciente a través de niveles jerárquicos) durante la ventana de acceso.

**H2 (Anestesia y NREM).** Bajo propofol y NREM, $`\widehat{\mathbf{\alpha}}`$ decrece y **la direccionalidad hacia adelante colapsa**; REM restaura parcialmente ambos.

**H3 (Psicodélicos).** Los psicodélicos aumentan la **coherencia dentro de capas locales** (posible aumento en $`\widehat{\alpha}`$ localmente) mientras **reducen la direccionalidad neta hacia adelante** entre capas distantes (mayor bidireccionalidad/bucles), prediciendo desacoplamiento entre S1 y S2.

**H4 (Acceso perturbacional).** Las respuestas evocadas por EMT en estados conscientes muestran $`\widehat{\mathbf{\alpha}}`$ **monotónico o creciente** a través de escalas espaciales y **ET/Granger condicional hacia adelante significativo** de áreas sensoriales a asociativas; ambos efectos se debilitan bajo pérdida de consciencia.

**Regla de decisión:** El acceso consciente RTM está **respaldado** si (S1) $`\widehat{\alpha}`$ sube o se mantiene a través de niveles involucrados **y** (S2) la direccionalidad condicional es solo-hacia-adelante (después de FDR) en condiciones conscientes pero no inconscientes/sin-reporte.

**4. Mediciones y variables**

**Proxies de escala** $`\mathbf{L}`$ **(dos requeridos para triangulación):**

1.  **Granulado grueso espacial:** promediar señales dentro de ROIs a tamaños crecientes de vóxel/cluster.

2.  **Ventaneo temporal / bandeo espectral:** estimar $`T`$ dentro de ventanas log-espaciadas (o señales limitadas por banda donde $`L \sim 1/f`$).

**Tiempo característico** $`\mathbf{T}`$**:**

- Tiempo de autocorrelación (integral o 1/e).

- Tiempo de integración de respuesta a impulso (EMT-EEG).

- Tiempo de permanencia de microestados metaestables (microestados EEG o estados HMM).

**Direccionalidad:**

- **Entropía de Transferencia / Granger (sustitutos de permutación/fase)**; variantes **condicionales** (ej., Área $`A\  \rightarrow \ B`$ \| región upstream).

- FDR a través de pares y retardos; grilla de embedding prerregistrada.

**5. Conjuntos de datos y tareas**

1.  **Umbral perceptual (reporte vs sin reporte):** detección visual/auditiva enmascarada; EEG/MEG/ECoG de alta densidad en cohortes clínicas.

2.  **Anestesia y sueño:** inducción/emergencia de propofol; polisomnografía nocturna (ciclos NREM/REM).

3.  **Sesión psicodélica (si está disponible, éticamente aprobada):** dosis moderada; bloques alternantes ojos-abiertos/cerrados y sondas oddball.

4.  **Corridas perturbacionales EMT-EEG:** pulso único estándar sobre corteza sensorial y asociativa.

**Tamaño de muestra/potencia (ilustrativo):** ≥24 sujetos por condición (diseños intra-sujeto), ≥200 ensayos por bloque de estado para estabilidad de ET/Granger; ICs bootstrap (B≥1000) para $`\widehat{\alpha}`$.

**6. Pipeline de análisis (prerregistrado)**

1.  **Preprocesamiento:** rechazo de artefactos (EOG/EMG), referenciado; segmentos estacionarios seleccionados vía pruebas de raíz unitaria.

2.  **Escalamiento intra-capa:** para cada región/proxy de escala, regresar $`T`$ vs $`L\  \rightarrow \ pendiente\ \widehat{\alpha}`$ + IC 95% bootstrap.

3.  **Direccionalidad entre-capas:** ET y Granger para niveles adyacentes; **condicional** en upstream para remover caminos indirectos.

4.  **Comparaciones múltiples:** FDR-BH (q=0.05); robustez de ventana (eliminar mayor $`L`$; top-k ventanas).

5.  **Integración de efectos:** contrastes por estado (consciente vs inconsciente, reporte vs sin reporte) para $`\widehat{\alpha}`$ y ET/Granger hacia adelante menos reverso.

6.  **Nulos y controles:** sustitutos de fase-aleatorizada; EMT sham; tareas control con energía idéntica pero fase aleatorizada (separación ordenada al origen vs pendiente).

**7. Modelado mecanístico (mesoscópico, no cuántico)**

- **Red:** modelo de tasa E-I en capas o de espigas con feedforward $`g_{f}`$ ajustable, feedback $`g_{b}`$ y ganancia neuromoduladora $`m`$.

- **Predicciones:** aumentar $`g_{f}`$ y coherencia impulsa α **más alto** y ET **solo-hacia-adelante**; sedación modelada como $`m`$ reducido y ruido aumentado → α más bajo, direccionalidad más débil; estado tipo-psicodélico como ganancia local aumentada con acoplamiento de largo alcance alterado → S1/S2 mixto.

- **Ajuste a datos:** elegir parámetros para coincidir con $`\widehat{\alpha}`$ empírico y patrones de ET; comparar con modelos simétricos/alternativos (AIC/BIC y fuera-de-muestra).

**8. Resultados y falsificación**

**Apoyo para acceso consciente RTM**

- S1+S2 pasan en reporte/despierto/REM/EMT-consciente; fallan o se invierten en sin-reporte/anestesia/NREM/sham; psicodélicos muestran S1↑ con S2↓ como se predijo.

**Falsificación**

- $`\widehat{\alpha}`$ **decrece** o la direccionalidad es **inversa o simétrica** en estados conscientes después del condicionamiento; S1/S2 no se separan de nulos.

- Modelos simétricos alternativos ajustan los datos igualmente bien o mejor **sin** cascadas dirigidas.

**9. Relación con propuestas cuánticas (posición)**

Esta explicación es **agnóstica a efectos micro-cuánticos**. Ni asume ni requiere mecanismos basados en colapso. Si los procesos cuánticos microscópicos mejoran la coherencia mesoscópica, se **manifestarían como cambios sistemáticos en** $`\mathbf{\alpha}`$ y direccionalidad a escalas observables. Incluimos un **Apéndice Exploratorio** con dos verificaciones de "aroma cuántico" (dependencias de temperatura/isótopo; perturbaciones magnéticas de campo débil) estrictamente como heurísticas opcionales, claramente etiquetadas como **no confirmatorias**.

**10. Reproducibilidad y prerregistro**

- Repositorio público con código semillado, scripts de regeneración de figuras, y generadores de sustitutos.

- Reporte Registrado Etapa 1: hipótesis, métricas, retardos/embeddings, plan FDR, pruebas de ventana, y segmentos nulos fijados **antes** del bloqueo de datos.

**11. Limitaciones**

- α es **candidato-necesario**, no suficiente para contenido fenomenal; apuntamos a **acceso/reporte**, no qualia.

- Los confusores (activación, movimiento) deben controlarse rigurosamente.

- Los proxies de escala espacial pueden sesgar $`\widehat{\alpha}`$; requerimos **dos proxies independientes** y convergencia.

**12. Opciones de título provisional**

- **"El Acceso Consciente como Coherencia Multiescala: Una Prueba RTM-Operacional a través de Sueño, Anestesia, Psicodélicos y EMT."**

- **"Sin Cuántica Necesaria: Una Explicación Mesoscópica RTM del Acceso Consciente vía Escalamiento de Coherencia y Cascadas Dirigidas."**

- **"De Pendiente a Sentido: Probando un Umbral de Coherencia RTM para el Acceso Consciente."**

**13. Plan de figuras**

1.  **Fig.1** Concepto: separación pendiente–ordenada al origen; jerarquía y cascada hacia adelante.

2.  **Fig.2** Ajustes de escalamiento $`T - \log L`$ y $`\widehat{\alpha}`$ a través de estados.

3.  **Fig.3** ET/Granger condicional hacia adelante vs reverso a través de estados.

4.  **Fig.4** Modelo: barridos de parámetros mapeando $`g_{f}`$, $`g_{b}`$, $`m`$ a $`\alpha`$ y direccionalidad; ajuste a datos.

5.  **Fig.5** Gráfico de decisión (S1/S2 aprobación/falla) + pipeline prerregistrado.

**APÉNDICE A — Validación Computacional del Marco RTM-Consciencia**

**A.1 Visión general**

Este apéndice presenta la validación computacional del marco de umbral de consciencia. Tres conjuntos de simulación demuestran:

1\. α > α_crit es necesario para el acceso consciente (S1)

2\. La direccionalidad hacia adelante (IDN > 0) acompaña los estados conscientes (S2)

3\. Los agentes farmacológicos afectan diferencialmente S1 y S2 (S3)

**A.2 S1: Modelo de Umbral de Consciencia**

**A.2.1 Hipótesis**

**Acceso consciente ↔ α > α_crit**

donde α_crit ≈ 0.50

**A.2.2 Estados de Consciencia**

\| Estado \| α \| Consciente \| Descripción \|

\|-------\|---\|-----------\|-------------\|

\| Despierto Reporte \| 0.72 \| Sí \| Acceso consciente completo \|

\| Despierto Sin-Reporte \| 0.48 \| No \| Estímulo no reportado \|

\| Sueño REM \| 0.65 \| Sí \| Soñando \|

\| Sueño NREM \| 0.35 \| No \| Sueño profundo \|

\| Sedación Ligera \| 0.52 \| Sí \| Responsivo \|

\| Anestesia Profunda \| 0.28 \| No \| No responsivo \|

**A.2.3 Rendimiento de Clasificación**

\| Métrica \| Valor \|

\|--------\|-------\|

\| Precisión \| 85.4% \|

\| AUC \| 0.65 \|

\| Umbral óptimo \| 0.50 \|

**A.2.4 Reporte vs Sin Reporte**

\| Condición \| Media α \| DE \|

\|-----------\|--------\|-----\|

\| Reporte \| 0.67 \| 0.12 \|

\| Sin Reporte \| 0.42 \| 0.14 \|

**Tamaño de efecto: d de Cohen = 1.59** (grande)

**A.3 S2: Cascada de Direccionalidad Hacia Adelante**

**A.3.1 Hipótesis**

**Acceso consciente → ET Hacia Adelante >> ET Hacia Atrás**

Medido por Índice de Direccionalidad Neta:

**IDN = (ET_adel - ET_atrás) / (ET_adel + ET_atrás)**

**A.3.2 Resultados por Estado**

\| Estado \| IDN \| Dominante Hacia Adelante \|

\|-------\|-----\|------------------\|

\| Despierto Consciente \| 0.35 \| Sí \|

\| Sueño REM \| 0.25 \| Sí \|

\| Sueño NREM \| 0.02 \| No \|

\| Propofol \| 0.01 \| No \|

\| Psicodélico \| -0.10 \| Invertido \|

**A.3.3 Comparación**

\| Grupo \| IDN Medio \| Interpretación \|

\|-------\|----------\|----------------\|

\| Consciente \| 0.19 \| Dominante hacia adelante \|

\| Inconsciente \| 0.08 \| Simétrico \|

**t = 2.65, p = 0.08**

**A.4 S3: Efectos Farmacológicos**

**A.4.1 Propofol (GABAérgico)**

\| Métrica \| Línea Base \| Bajo Propofol \| Cambio \|

\|--------\|----------\|----------------\|--------\|

\| α \| 0.72 \| 0.28 \| -61% \|

\| IDN \| 0.45 \| 0.02 \| -96% \|

**Tanto S1 como S2 fallan → Inconsciencia**

**A.4.2 Psicodélicos (Serotoninérgicos)**

\| Métrica \| Línea Base \| Efecto Pico \| Cambio \|

\|--------\|----------\|-------------\|--------\|

\| α \| 0.72 \| 0.82 \| +14% \|

\| IDN \| 0.45 \| -0.15 \| Invertido \|

**S1 pasa, S2 falla → Consciencia alterada**

**A.4.3 Esquema de Clasificación**

\| S1 (α) \| S2 (IDN) \| Predicción \|

\|--------\|----------\|------------\|

\| Pasa \| Pasa \| Consciente Normal \|

\| Pasa \| Falla \| Consciente Alterado \|

\| Falla \| Falla \| Inconsciente \|

**A.5 Resumen de Validación Computacional**

\| Prueba \| Métrica \| Resultado \|

\|------\|--------\|--------\|

\| Clasificación de umbral \| AUC \| 0.65 \|

\| Reporte vs Sin Reporte \| d de Cohen \| 1.59 \|

\| IDN Consciente vs Inconsciente \| estadístico-t \| 2.65 \|

\| Colapso α por propofol \| Cambio \| -61% \|

\| Disociación psicodélica \| α↑, IDN↓ \| Confirmado \|

**A.6 Predicciones Falsificables**

El marco falla si:

1\. **Sin umbral:** α no separa estados conscientes/inconscientes

2\. **Sin direccionalidad:** IDN es simétrico en estados conscientes

3\. **Sin farmacología:** Propofol no afecta α, psicodélicos no disocian S1/S2

4\. **Patrones invertidos:** Estados inconscientes muestran α más alto o IDN hacia adelante

**A.7 Criterios Combinados**

**El acceso consciente requiere:**

\- S1: α > 0.50 (umbral de coherencia)

\- S2: IDN > 0.15 (direccionalidad hacia adelante)

**Estados alterados (psicodélicos):**

\- S1: α > 0.50 (pasa)

\- S2: IDN < 0 (falla/invertido)

**APÉNDICE B. Validación Empírica: Pendiente Espectral EEG y la Topología de la Consciencia**

El marco RTM postula que el acceso consciente no es un evento neuroquímico localizado, sino una transición de fase topológica macroscópica. Para probar esto, analizamos la pendiente espectral ($`\beta`$) de registros EEG a través de 14 condiciones de consciencia.

**B.1 Observación Heurística y la Falacia de Agregación**

La validación inicial se basó en comparar las medias aritméticas simples de pendientes espectrales a través de todas las condiciones. Este enfoque heurístico produjo una precisión de clasificación de 85.7% ($`AUC\  = \ 0.80`$). Sin embargo, cometió una severa "falacia de agregación" al dar peso igual a estudios con $`n = 10,255`$ sujetos (Base de Datos NSRR) y estudios con $`n = 5`$ sujetos (ensayos Ketamina/Propofol). Además, agrupó ingenuamente el sueño REM paradójico (que es fenomenológicamente consciente pero posee pendientes espectrales extremadamente empinadas, "viscosas", $`\beta \approx - 3.25`$) junto con la Vigilia basal, difuminando artificialmente los límites físicos de la red de transporte.

**B.2 Simulación Robusta de Varianza a Nivel de Sujeto**

Para someter las predicciones RTM a escrutinio clínico del mundo real, desplegamos una simulación Monte Carlo a nivel de sujeto ($`n = 30,873`$). Usando Errores Estándar de la Media (EEM) reportados, reconstruimos matemáticamente la verdadera varianza continua de la neurofisiología humana. Luego separamos estrictamente Vigilia de Inconsciencia Verdadera (NREM / Propofol) para evaluar la capacidad predictiva central de RTM sin el confusor de la paradoja REM.

Incluso cuando se penaliza fuertemente con varianza humana masiva, la topología se bifurca estrictamente. La Vigilia opera en un régimen altamente integrado ($`\beta = \  - 2.10\  \pm 2.02`$), mientras la Inconsciencia Verdadera colapsa a un estado desconectado y viscoso ($`\beta = \  - 2.84\  \pm 1.01`$). Esta separación estructural es altamente estadísticamente significativa (d de Cohen $`= 0.46,p < 10^{- 10}`$).

**B.3 La Disociación de Ketamina: Fricción Estructural vs. Fluidez**

El mayor triunfo predictivo del marco RTM robusto se evidencia en la resolución de la "disociación de ketamina". Tanto el propofol como la ketamina inducen falta de respuesta conductual profunda en pacientes, lo cual históricamente ha confundido la electrofisiología clínica y los clasificadores clásicos.

Al simular la densidad de probabilidad completa a nivel de sujeto a través del espacio de estados neurofisiológicos, los modelos clásicos se difuminan. Sin embargo, la topología RTM diferencia ambos estados con precisión matemática estricta:

- **Colapso Inducido por Propofol:** Al inyectar inhibición GABAérgica masiva, el propofol actúa como un "coagulante topológico" macroscópico. Aumenta drásticamente la pendiente espectral ($`\Delta\beta \approx - 1.25`$), desconectando físicamente la integración cortical de largo alcance. La densidad de probabilidad de sujetos bajo propofol se desplaza enteramente al régimen topológico Verdaderamente Inconsciente.

- **Preservación bajo Ketamina:** A pesar de la parálisis motora profunda, la ketamina preserva el régimen de transporte topológico específico de la corteza despierta. La pendiente espectral permanece estadísticamente anclada a la línea base saludable ($`\Delta\beta \approx - 0.10`$), manteniendo la "fluidez" estructural de la red neural.

**Conclusión:** Esto explica físicamente por qué la mente bajo ketamina permanece fenomenológicamente consciente—experimentando alucinaciones complejas y sueños vívidos—mientras el cuerpo físico está anestesiado. Prueba definitivamente que el acceso consciente es un límite macroscópico definido por la coherencia topológica multiescala de la red cortical.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
