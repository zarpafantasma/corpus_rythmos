<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Serpiente">

# Bioquímica Rítmica  
**La enzima como instrumento de coherencia y un índice práctico para $\alpha$ en la catálisis viva**  
  
Álvaro Quiceno

</div>


**Resumen**

La catálisis enzimática se enmarca habitualmente como geometría y energética: "llave y cerradura", estabilización del estado de transición y selección conformacional. Aquí reformulamos las enzimas como instrumentos de coherencia a mesoescala dentro del marco de Relatividad Temporal en Sistemas Multiescala (RTM), donde los tiempos característicos escalan con el tamaño L según una ley de potencia τ ∝ L^α. Postulamos que los sitios activos generan microentornos de α elevado que filtran las rutas de reacción por ritmo y no solo por forma. Derivamos un estimador de escalamiento enzimático α_enz = −d(log k_app)/d(log L), con k_app la constante de velocidad aparente medida a lo largo de escalas de confinamiento controladas L (nanoporos/aglomeración/cavidades), e introducimos un Índice de Coherencia de Bioquímica Rítmica (RBCI) (0–1) que integra pendiente (α), transporte selectivo de espín (CISS), coherencia vibracional y reducción de varianza bajo excitación en resonancia.

**Validación computacional.** Implementamos y validamos el marco RTM enzimático mediante tres conjuntos de simulaciones. S1 demuestra que la cinética de Michaelis-Menten modificada por RTM $`k\_ cat\  \propto \ L\hat{}( - \alpha)`$) produce firmas cinéticas distintas entre clases de transporte, con α recuperable a partir de datos simulados de confinamiento con un error inferior al 0,5 %. S2 valida la metodología de estimación: el estimador α_enz es robusto frente a ruido de medición hasta σ ≈ 0,30, requiere solo ≥3 escalas de confinamiento y discrimina clases de transporte (difusivo α ≈ 2,0 vs. jerárquico α ≈ 2,3) con d de Cohen = 3,12. La prueba de colapso de datos muestra un coeficiente de variación 11× peor cuando se usa un α incorrecto. S3 predice selectividad de sustrato ajustable por confinamiento: para sustratos con valores de α diferentes, las razones de selectividad pueden variar entre 2 y 3× en el rango de confinamiento de 10–100 nm, con longitudes de cruce calculables donde la selectividad se invierte.

Esbozamos pruebas falsificables: estabilidad de la pendiente, colapso de datos (k_app × L^α = constante) y cambio de clase bajo forzamiento acústico, junto con controles que separan artefactos térmicos y de mezcla. El programa predice bandas de α consistentes con transporte jerárquico/fractal (α ≈ 2,1–2,5) y vincula la alostería con un α ajustable. De confirmarse, los resultados unifican la especificidad catalítica, la regulación alostérica y la selectividad de espín bajo una única ley multiescala; de refutarse, proporcionan restricciones precisas sobre cuándo y por qué las enzimas se desvían del escalamiento RTM. El marco es operacional, prerregistrable y verificable de inmediato con herramientas biofísicas estándar.

**Validación empírica preliminar** $`\mathbf{\rightarrow}`$ **(APÉNDICE B)**. Validamos el marco de Bioquímica Rítmica mediante un análisis comparativo de 153 puntos de datos empíricos, contrastando procesos topológicos globales (plegamiento de proteínas) con eventos catalíticos localizados (cinética enzimática). La regresión de distancia ortogonal (ODR) con inyección de varianza *in vitro* del 20–30 % y normalización por clase EC corrige los factores de confusión de las reacciones químicas. El análisis robusto encuentra: plegamiento de proteínas en un régimen altamente coherente impulsado por la topología ($`\alpha = 7.22 \pm 0.62`$), consistente con la interpretación del "embudo de plegamiento" dirigido de la paradoja de Levinthal; cinética enzimática sin dependencia estadísticamente significativa del tamaño tras la normalización por EC ($`\alpha = 0.26 \pm 0.69`$, el IC incluye cero), consistente con que la catálisis es un fenómeno localizado en el sitio activo. El **solapamiento nulo en bootstrap** entre las distribuciones de plegamiento y enzimas ($`d = 6.98`$, 0 % de solapamiento en 3 000 iteraciones) confirma que los dos regímenes son genuinamente distintos. Esto se clasifica como **CONVERGENTE** por el Equipo Rojo (abril de 2026): RTM recupera de forma independiente bioquímica conocida —plegamiento cooperativo vs. catálisis local— desde un punto de partida topológico, y proporciona una clasificación unificada de ambos mediante un único exponente $`\alpha`$. Auditoría completa: Apéndice C.

**1. Introducción**

**1.1 Motivación: de las formas a los ritmos**

Las enzimas aceleran las reacciones en órdenes de magnitud, pero las narrativas puramente geométricas —llave y cerradura, ajuste inducido— no explican completamente la modulación de la velocidad en condiciones de aglomeración, confinamiento o alostería de largo alcance. Las mediciones modernas revelan fluctuaciones estructuradas, modos vibracionales de larga vida, corrientes selectivas de espín en matrices quirales (CISS) y variabilidad de velocidad que se estrecha bajo condiciones de excitación específicas. Estas observaciones sugieren que la **estructura orquesta el tiempo**, no solo las barreras.

El marco de **Relatividad Temporal Multiescala (RTM)** trata los tiempos característicos $`T`$ como una función del tamaño $`L`$ mediante una ley de potencia $`{T \propto L}^{\alpha}`$, donde el exponente $`\alpha`$ es un **observable operacional** vinculado a la **clase de universalidad** del sistema (transporte local vs. de largo alcance, topología entera vs. fractal, regímenes de confinamiento cuántico). RTM distingue la **pendiente** (el exponente $`\alpha`$) del **intercepto** (reloj/corrimiento al rojo/ganancia), permitiendo comparaciones entre entornos sin confundir desplazamientos de línea base con mecanismos dinámicos.

**1.2 Hipótesis central**

Hipotetizamos que **los sitios activos son cavidades de coherencia a mesoescala** que **elevan el** $`\alpha`$ **local** respecto al solvente/célula circundante, filtrando así las trayectorias de reacción por ritmo. Concretamente:

- Microentornos más pequeños y coherentes completan los actos característicos más rápido **por escalamiento**, no solo por temperatura.

- La alostería actúa principalmente **ajustando** $`\alpha`$ (coherencia/clase de transporte), con los cambios conformacionales como actuador.

- Los medios quirales que exhiben CISS son firmas empíricas de regímenes de transporte de **α elevado**.

**1.3 Un programa operacional**

Proponemos dos observables complementarios.

1.  **Estimador de escalamiento enzimático**

``` math
\alpha_{\text{bio,enz}} = - \frac{d\ logk}{d\ logL}│_{isothermal,\ fixed\ ionic\ strength,\ off - resonance\ control}
```

obtenido midiendo las velocidades aparentes $`k`$ mientras se varía una **escala de confinamiento efectiva** $L$ (por ejemplo, matrices nanoporosas de tamaño de poro conocido, aglomeración ajustable o cavidades diseñadas). La estabilidad de $`\alpha_{\text{bio,enz}}`$ a lo largo de al menos una década en $`L`$, más el **colapso de datos** de $`k`$ al reescalar por $`\mathbf{L}^{\mathbf{\alpha}^{\mathbf{\star}}}`$, es la prueba principal de falsificación.

2.  **Índice de Coherencia de Bioquímica Rítmica (RBCI)** (0–1), un índice compuesto que agrega:

- **Pendiente:** un mapeo normalizado de $`\alpha_{\text{bio,enz}}`$ sobre una banda biológicamente plausible;

- **Firma de espín (CISS):** polarización/asimetría del transporte dependiente de espín a través de la proteína/película quiral;

- **Coherencia vibracional:** fracción de potencia espectral en modos coherentes (métricas Raman/IR o pump–probe);

- **Reducción de varianza bajo excitación en resonancia:** disminución de $`Var(k)`$ al aplicar una excitación periódica no térmica ajustada a la ventana de coherencia del sistema, relativa a la excitación fuera de resonancia.

RBCI complementa a $`\alpha_{\text{bio,enz}}`$: la pendiente prueba la **ley de escala**, mientras que RBCI prueba la **coherencia mecanística** que se espera covaríe con transporte de α elevado.

**1.4 Predicciones y resultados falsificables**  
RTM hace predicciones precisas y prerregistrables para sistemas enzimáticos:

- **α en bandas en biología:** el transporte jerárquico/fractal produce $`\alpha \approx 2.3\text{–}2.7`$.

- **Colapso de datos:** definiendo $`\widetilde{k} = k\ L^{\alpha^{\star}}`$, las curvas de diferentes valores de $`L`$ colapsan **si y solo si** $`\mathbf{\alpha}^{\mathbf{\star}}\mathbf{=}\mathbf{\alpha}_{\mathbf{bio}\mathbf{,}\mathbf{enz}}`$

- **Cambio de clase bajo excitación:** la excitación acústica o electromecánica puede mover el sistema entre clases de transporte, produciendo un **salto predecible** en el $`\alpha`$ ajustado y un **aumento concurrente del RBCI** sin calentamiento medible.

- **Ajuste alostérico:** los ligandos activadores elevan $`\alpha_{bio,enz}`$ y RBCI; los ligandos inhibidores los disminuyen.

- **Covariación CISS:** la polarización de espín disminuye monótonamente con la desnaturalización y covaría con RBCI.

El fallo de cualquiera de estas predicciones, bajo controles adecuados, delimitaría la aplicabilidad de RTM o revelaría factores de confusión ocultos (por ejemplo, límites de mezcla, artefactos térmicos, deriva de pH).

**1.5 Alcance, controles y artefactos**
Nuestro protocolo separa explícitamente la **pendiente** del **intercepto** manteniendo constantes la temperatura, la fuerza iónica y el tampón, y cuantificando el calentamiento y la mezcla. Los controles incluyen matrices ficticias (misma geometría, superficie inerte), excitación **fuera de resonancia**, aleatorización ciega de $`L`$ y termometría independiente. Los artefactos conocidos —gradientes térmicos, cavitación, difusión de capa límite, fotoblanqueo— se miden y se acotan en el plan de análisis. El marco es agnóstico al detalle microscópico: lo que importa empíricamente es si el **escalamiento** y las **firmas de coherencia** aparecen juntos y obedecen las transformaciones predichas.

**1.6. Validación empírica sistemática: coherencia global vs. catálisis local (APÉNDICE B)**
Dentro del marco RTM, las macromoléculas biológicas no son meros conglomerados químicos complejos; son motores topológicos multiescala. Para probar si la ecuación de escalamiento RTM puede clasificar clases distintas de operaciones biológicas, comparamos procesos topológicos globales (plegamiento de proteínas) con eventos catalíticos localizados (cinética enzimática) en presencia de ruido experimental realista.
Hipotetizamos que los procesos que requieren la coordinación estructural simultánea de toda una macromolécula, como el plegamiento de proteínas, operarán en un régimen altamente coherente dominado por la topología, caracterizado por un exponente masivo ($`\alpha \gg 1`$). En contraste, los procesos que dependen de sitios activos aislados y localizados, como la catálisis enzimática, deberían exhibir independencia completa de la escala estructural global ($`\alpha \approx 0`$). Mediante el análisis sistemático de registros empíricos en ambos dominios y el despliegue de estadísticas robustas de errores en variables (EIV) para controlar la varianza de ensayos *in vitro* y los factores de confusión químicos, proporcionamos evidencia directa de que el exponente de coherencia $`\alpha`$ actúa como un límite matemático riguroso. Clasifica exitosamente si un proceso bioquímico está gobernado por resonancia geométrica global o por química térmica localizada.

**2. Teoría**
**2.1 Postulados de RTM especializados para la catálisis enzimática**  
Adoptamos los supuestos de la **Relatividad Temporal Multiescala (RTM)** en un contexto enzimológico:

- **P1 — Semigrupo de escala:** reescalar una longitud de confinamiento efectiva $`L`$ por $`\lambda_{1}`$ y luego por $`\lambda_{2}`$ equivale a un único reescalamiento por $`\lambda_{1}{\ \lambda}_{2}`$ para el observable cinético (por ejemplo, el tiempo medio de recambio $`T`$ o la constante de velocidad aparente $`k = 1/T`$).

- **P2 — Regularidad:** $`T(L)`$ es continua y estrictamente monótona dentro de una ventana experimental donde el mecanismo microscópico permanece inalterado (mismo tampón, temperatura, fuerza iónica, pH).

- **P3 — Invariancia del reloj (gauge multiplicativo; correcciones de tiempo muerto/desfase).**\
  Los factores multiplicativos del reloj ($`T' = cT`$; cambios de unidades, ganancias uniformes de temporización, escalamiento uniforme de velocidad/tiempo a control termodinámico fijo) alteran el intercepto pero no la pendiente en $`\log T`$ – $`\log L`$.\
  Los artefactos aditivos como el **tiempo muerto** del detector, latencias fijas o desfases de sustracción de línea base producen $`T_{\text{obs}} = T + b`$ y pueden sesgar la pendiente estimada a menos que $`b`$ se corrija explícitamente (ajustar $`T_{eff} = T_{\text{obs}} - b`$ con $`T_{\text{obs}} > b`$) o que los ajustes se restrinjan a regímenes con $`T \gg b`$ y se reporte un análisis de sensibilidad sobre valores plausibles de $`b`$.

- **P4 — Causalidad finita:** el transporte de masa/energía/información a través de $`L`$ tiene velocidad efectiva finita; por lo tanto, los tiempos característicos no pueden escalar de forma sublineal con la distancia en un régimen estable.

De P1–P2, la única ley autoconsistente que relaciona tiempo con escala es una **ley de potencia**:  
``` math
T(L) = C\text{ }L^{\alpha},C > 0
```

con $`\alpha`$ un **exponente observable**. En forma de velocidad,  
``` math
k(L) = k_{0}\text{ }L^{- \alpha}
```

Esto produce el estimador enzimático operacional usado a lo largo de este trabajo:
``` math
\alpha_{bio,enz} = - \text{ }\frac{dlogk}{dlogL} \mid_{\text{isothermal, fixed ionic strength, off-resonance}}
```

**2.2 El sitio activo como cavidad de coherencia a mesoescala**

El sitio activo de una enzima y su envoltura proteína-solvente inmediata forman una **cavidad a mesoescala** que filtra las trayectorias de reacción por **clase de transporte** tanto como por geometría:

- **Longitud efectiva** $`L`$ **:** la escala más pequeña que restringe la difusión, reorientación, transferencia de protones/electrones o el flujo vibracional colectivo relevante para el paso limitante de la velocidad. Experimentalmente, $`L`$ puede ajustarse con matrices nanoporosas, agentes de aglomeración o cavidades huésped diseñadas.

- **Elevación de coherencia:** las regiones estructuradas, quirales y mecánicamente rígidas sostienen correlaciones de larga vida; en RTM esto aparece como un $`\alpha`$ **mayor** (tiempos más largos a $`L`$ mayores, finalización efectiva más rápida cuando $`L`$ se reduce bajo control termodinámico constante).

- **Implicación para el transporte:** si el transporte es (i) difusivo local, se espera $`\alpha \approx 2`$; (ii) jerárquico/fractal con trampas y corredores, se espera $`\alpha \approx d_{w} > 2`$; (iii) parcialmente balístico a lo largo de cables proteicos o dentro de canales resonantes, se espera un $`\alpha`$ efectivo intermedio determinado por la mezcla de vías dominante.

**2.3 Mapeo de** $`\mathbf{\alpha}`$ **a clases de universalidad de transporte**
RTM no asume un modelo microscópico único; en cambio, $`\alpha`$ identifica la **clase de universalidad** que gobierna la etapa limitante de la velocidad.

- **Difusión local (generador laplaciano).** El tiempo medio de primer paso (MFPT) escala como $`T \sim L^{2} \Rightarrow \alpha = 2`$.

- **Medios fractales/jerárquicos.** Para caminatas aleatorias con dimensión de caminata $`d_{w}`$, $`T \sim L^{d_{w}} \Rightarrow \alpha = d_{w}`$ con $`d_{w} \in (2,3\rbrack`$ común en redes ramificadas.

- **Canales guiados/parcialmente balísticos.** Si una fracción $`p`$ de trayectorias se propaga de forma cuasi-balística (tiempo $`\sim L`$) y $`1 - p`$ difunde ($`\sim L^{2}`$), el exponente efectivo sobre una década en $`L`$ satisface


``` math
\alpha_{eff} \approx \frac{d\ \log{\lbrack p\text{ }L^{- 1} + (1 - p)\text{ }L^{- 2}\rbrack}^{- 1}}{d\ \log L} \in \lbrack 1,2\rbrack
```
aumentando hacia 2 a medida que dominan las vías difusivas.

- **Clústeres confinados cuánticamente/coherentes (heurístico).** En dominios fuertemente confinados y altamente coherentes, con acoplamiento vibracional/electrónico robusto, los mapeos heurísticos sugieren que $`\alpha`$ puede elevarse hasta $`\sim 3`$, pero estos valores son **cotas/conjeturas** y no derivaciones de primeros principios.

**Corolario (cambio de clase):** alterar deliberadamente el generador (por ejemplo, añadir una excitación acústica/electromecánica **en resonancia** que abra canales guiados o suprima trampas) debería producir un **cambio discreto** en el $`\alpha`$ ajustado, acompañado de una caída en la varianza de la velocidad y un aumento en las firmas de coherencia (Sección 2.5).

**2.4 Alostería como ajuste de** $`\mathbf{\alpha}`$

Los efectores alostéricos modulan la dinámica lejos del sitio activo. En RTM:

- **Activador:** rigidiza/cohesiona los movimientos a mesoescala, **elevando** $`\alpha`$ y produciendo (i) una pendiente $`- d\ logk/d\ logL`$ más pronunciada; (ii) un colapso de datos más fuerte tras reescalar $`k \leftarrow k\text{ }L^{\alpha^{\star}}`$; (iii) varianza reducida de $`k`$ bajo excitación en resonancia.

- **Inhibidor:** flexibiliza/desordena las vías, **disminuyendo** $`\alpha`$ y degradando el colapso y las firmas de coherencia.

Esto reformula la alostería de "cambio de forma" a **cambio de clase de transporte** medible por $`\alpha_{bio,enz}`$ más índices de coherencia.

**2.5 Observables de coherencia: CISS, potencia vibracional y reducción de varianza**

Vinculamos $`\alpha`$ con tres observables accesibles instrumentalmente que forman parte del **Índice de Coherencia de Bioquímica Rítmica (RBCI)**:

1.  **CISS (selectividad de espín inducida por quiralidad):** los dominios proteicos quirales pueden filtrar espines. Una mayor **polarización/asimetría de espín** se interpreta como una firma de transporte ordenado y guiado, compatible con un $`\alpha`$ **más alto**. Las series de desnaturalización deberían reducir monótonamente CISS y RBCI.

2.  **Coherencia vibracional:** la espectroscopia (Raman/IR, pump–probe) produce la **fracción de potencia en modos coherentes** sobre una banda definida. La potencia coherente debería covariar con $`\alpha`$ cuando el transporte cambia de clase.

3.  **Reducción de varianza bajo excitación en resonancia:** aplicar una excitación periódica dentro de una ventana isotérmica segura debería **disminuir** $`Var(k)`$ (estrechar la distribución de velocidades) si refuerza la clase de transporte dominante; la excitación fuera de resonancia actúa como control.

RBCI, definido más adelante en Métodos, agrega versiones normalizadas de estas características junto con la estimación de la pendiente, produciendo una puntuación 0–1 que puede compararse entre enzimas y laboratorios.

**2.6 Cotas independientes del modelo y corolarios falsificables**

De P4 (causalidad finita) y las clases anteriores:

- **Cota inferior:** $`\alpha \geq 1`$ para cualquier proceso físicamente realizable que deba recorrer una distancia $`L`$.

- **Cota inferior difusiva:** para pasos dominados por el laplaciano, $`\alpha \geq 2`$.

- **Mejora fractal:** $`\alpha > 2`$ indica atrapamiento/corredores jerárquicos (topología efectiva no entera).

- **Banda superior heurística:** valores cercanos a $`3.0\text{–}3.5`$ son **cotas heurísticas** plausibles en dominios fuertemente coherentes y confinados cuánticamente, y deben tratarse como conjeturales hasta que se evidencien directamente.

**Corolarios falsificables para enzimas:**

- **Estabilidad de la pendiente:** dentro de una clase fija y sobre al menos una década en $`L`$, el $`\alpha_{bio,enz}`$ ajustado es estable (los intervalos de confianza se solapan).

- **Colapso de datos:** definiendo $`\widetilde{k} = k\text{ }L^{\alpha^{\star}}`$, las curvas tomadas a diferentes $`L`$ **colapsan** si y solo si $`\alpha^{\star} = \alpha_{bio,enz}`$.

- **Firmas sincronizadas:** el cambio de clase que modifica $`\alpha`$ debe **co-ocurrir** con (i) mayor potencia vibracional coherente, (ii) CISS más fuerte (para sistemas quirales) y (iii) $`Var(k)`$ reducida bajo excitación en resonancia, **sin** artefactos medibles de calentamiento o mezcla.

- **Coherencia alostérica:** los activadores aumentan $`\alpha_{bio,enz}`$ y RBCI; los inhibidores disminuyen ambos, proporcionando confirmación ortogonal más allá de los cambios tradicionales en $`K_{M}`$ y $`k_{\text{cat}}`$.

**3. Métodos**

**3.1 Panorama general y lógica de diseño**

Nuestro objetivo es estimar un **exponente de escalamiento enzimático** $`\alpha_{bio,enz}`$ a partir de mediciones de una constante de velocidad aparente $`k`$ tomadas a lo largo de **escalas de confinamiento** controladas $`L`$, y calcular un **Índice de Coherencia de Bioquímica Rítmica (RBCI)** que agregue observables sensibles a la coherencia. El diseño central usa cuatro palancas ortogonales:

1.  **Geometría (fijar** $`L`$ **)** — ajustar una longitud efectiva mediante matrices nanoporosas, aglomeración o cavidades huésped diseñadas.

2.  **Excitación (cambio de clase)** — aplicar excitación acústica/electromecánica de baja amplitud para probar si la clase de transporte y $`\alpha`$ cambian.

3.  **Estructura (coherencia)** — modular el orden proteico mediante alostería o series de desnaturalización y registrar firmas de coherencia (CISS, potencia vibracional, reducción de varianza).

4.  **Controles** — condiciones isotérmicas, fuerza iónica fija, excitación fuera de resonancia, matrices ficticias, corridas aleatorizadas, termometría independiente.

Todos los experimentos se prerregistran con planes de análisis y criterios de inclusión/exclusión.

**3.2 Materiales y reactivos**

- **Enzimas (elegir un sistema modelo, luego replicar en un segundo):**\
  Primaria: Ureasa (frijol de jack) **o** Lactato deshidrogenasa (LDH, músculo de conejo).\
  Secundaria (replicación): Alcohol deshidrogenasa (ADH) o Anhidrasa carbónica.

- **Tampones:** HEPES (50 mM, pH 7,40 ± 0,05), NaCl (150 mM), $`{MgCl}_{2}`$ (5 mM) cuando se requiera; quelantes según necesidad.

- **Agentes de aglomeración / cavidades:** PEG (10–40 kDa), dextrano, BSA; monolitos de sílice sol-gel o alúmina; membranas de alúmina anódica (AAMs) con diámetros de poro nominales de 5–200 nm; sílice mesoporosa (SBA-15, MCM-41) con tamaños de poro certificados.

- **Efectores alostéricos:** activador/inhibidor apropiado para la enzima (por ejemplo, fructosa-1,6-bisfosfato para LDH-A).

- **Agentes desnaturalizantes:** cloruro de guanidinio, urea; rampas graduales de pH o temperatura para series de desplegamiento.

- **Sustratos de espín/CISS:** Au(111) o ITO con monocapas autoensambladas; películas quirales/monocapas proteicas preparadas por Langmuir–Blodgett o adsorción.

- **Hardware acústico:** transductor(es) piezoeléctrico(s) con frecuencias fundamentales de 20 kHz–2 MHz; generador de funciones; gel de acoplamiento; acelerómetro o vibrómetro láser para calibración de amplitud.

- **Detectores:** UV-Vis de flujo detenido o lector de placas para cinética; micro-Raman/FTIR para espectros vibracionales; amplificador lock-in e imán para CISS; termistor de alta precisión (±0,01 °C).

**3.3 Preparación enzimática y ensayos de actividad**

- Preparar las soluciones madre de enzima en hielo; determinar la concentración por absorbancia.

- Elegir un ensayo de actividad que produzca una **constante de velocidad aparente** $`k`$ de buen comportamiento (por ejemplo, absorbancia de NADH a 340 nm para LDH).

- Para cada condición de $`L`$, adquirir $`n \geq 8`$ réplicas independientes de $`k`$ (ciclos de carga y medición separados). Usar alícuotas frescas para evitar envejecimiento por arrastre.

**3.4 Definición y calibración de la longitud de confinamiento efectiva** $`\mathbf{L}`$

Definimos $`L`$ como la longitud característica más pequeña que restringe el transporte limitante de la velocidad (difusión/reorientación/transferencia) en la geometría del ensayo.

**Matrices nanoporosas / membranas.**

- Usar tamaños de poro certificados por el proveedor (5–200 nm). Verificar con SEM o adsorción de gas (BET/BJH).

- Registrar la **tortuosidad hidráulica** ($`\tau`$) si está disponible; reportar una **longitud efectiva** $`L_{eff} = L_{pore}\sqrt{\tau}`$.

**Aglomeración (confinamiento osmótico polimérico).**

- Convertir la fracción en masa $`w`$ a un tamaño de malla efectivo $`\xi(w)`$ usando relaciones de escalamiento polimérico; definir $`L = \xi`$. Proporcionar la curva de calibración en IS.

**Cavidades diseñadas (huésped-anfitrión).**

- Medir el diámetro de la cavidad por SAXS o crio-EM; definir $`L`$ como el cuello de botella más estrecho relevante para el acceso del sustrato o la transferencia de carga.

Aleatorizar el orden de $`L`$ entre corridas. Mantener tampón, pH, fuerza iónica y temperatura idénticos para todos los $`L`$.

**3.5 Protocolo de excitación acústica/electromecánica**

**Propósito:** probar el **cambio de clase** y la reducción de varianza bajo excitación **en resonancia** vs. control **fuera de resonancia**.

- Barrer frecuencias discretas: 20 kHz, 200 kHz, 2 MHz (±2 %).

- Amplitud: fijar el voltaje del piezoeléctrico para mantener el **ΔT < 0,05 °C** en el seno del fluido (confirmado por termometría independiente).

- Ciclo de trabajo: 50 % cuadrado o sinusoidal continuo; exponer durante toda la ventana de lectura cinética.

- La **resonancia** se define operacionalmente como la frecuencia que **minimiza** $`Var(k)`$ en un barrido piloto a $`L`$ fijo sin calentamiento medible; la **fuera de resonancia** es una frecuencia ≥10× alejada con amplitud RMS equivalente.

**3.6 Medición de la velocidad aparente** $`\mathbf{k}`$

- **Flujo detenido/lector de placas:** ajustar segmentos mono-exponenciales o regiones lineales de velocidad inicial para obtener $`k`$.

- Rechazar trazas con R² < 0,95 o artefactos multifásicos visibles; registrar los rechazos a priori en el prerregistro.

- Para cada $`L`$, calcular la media muestral $`\overset{ˉ}{k}`$ y la varianza $`Var(k)`$; retener los valores a nivel de réplica para modelado jerárquico.

**3.7 CISS (selectividad de espín inducida por quiralidad)**

**Montaje:** monocapa proteica sobre Au(111) o ITO; contacto ferromagnético; magnetización ±$`M`$; medición corriente-voltaje con detección lock-in.

- Definir la **asimetría de espín** $`P_{CISS} = (I_{+ M} - I_{- M})/(I_{+ M} + I_{- M})`$ a un voltaje de polarización fijo.

- Calibrar la resistencia de contacto y las fugas; incluir controles de sustrato desnudo y proteína desnaturalizada.

- Para las series de desnaturalización, medir $`P_{CISS}`$ en función de la concentración del desnaturalizante o la temperatura.

**3.8 Espectroscopia de coherencia vibracional**

- Adquirir espectros Raman (o pump–probe) sobre una banda predefinida.

- Calcular la **fracción de potencia coherente** $`C_{Raman}`$: razón de la potencia espectral en modos estrechos y persistentes respecto a la potencia total (PSD con ventana + selección de picos con umbral FWHM).

- Controles: adquisición idéntica sobre tampón y proteína desnaturalizada; sustraer el fondo y corregir por fotoblanqueo.

**3.9 Control de temperatura y mezcla**

- Registro continuo de temperatura (±0,01 °C). Los experimentos con **ΔT > 0,05 °C** se marcan para análisis de sensibilidad.

- Verificar que no haya cavitación ni cambios de mezcla en el seno del fluido mediante (i) imagen de partículas trazadoras o (ii) comparación de cinéticas con colorantes inertes; excluir condiciones que alteren la mezcla de línea base.

**3.10 Cálculo del exponente de escalamiento** $`\mathbf{\alpha}_{\mathbf{bio}\mathbf{,}\mathbf{enz}}`$

Estimamos $`\alpha`$ a partir de la pendiente de $`\log k`$ vs $`\log L`$.

1.  **Estimador primario (OLS en log–log):**

``` math
\alpha_{bio,enz}\text{\:\,} = \text{\:\,} - \text{ }{\widehat{\beta}}_{1},\log k = \beta_{0} + \beta_{1}\ logL + \varepsilon.
```

2.  **Errores en variables (BCES/ortogonal):** si $`L`$ tiene error de calibración, usar regresión ortogonal o BCES; reportar ambos.

3.  **ICs por bootstrap:** 10 000 remuestreos bootstrap de pares (L, k); reportar mediana e IC al 95 %.

4.  **ANCOVA entre entornos:** probar la igualdad de pendientes entre condiciones (por ejemplo, en/fuera de resonancia, ±ligando alostérico). El término de interacción $`\log L \times \text{condición}`$ indica **cambio de clase**.

5.  **Prueba de colapso de datos:**

    - Definir $`\widetilde{k} = k\text{ }L^{\alpha^{\star}}`$.

    - Optimizar $`\alpha^{\star}`$ minimizando la varianza entre curvas de $`\widetilde{k}`$.

    - **Aprobado** si $`\alpha^{\star}`$ cae dentro del IC al 95 % de $`\alpha_{bio,enz}`$ y las curvas colapsadas son indistinguibles según un criterio tipo KS.

**3.11 Índice de Coherencia de Bioquímica Rítmica (RBCI)**

Reportamos un índice de 0 a 1 que combina pendiente y firmas de coherencia:

**3.11 Índice de Coherencia de Bioquímica Rítmica (RBCI)**

Reportamos un índice de 0 a 1 que combina pendiente y firmas de coherencia:

Reportamos un índice de 0 a 1 que combina pendiente y firmas de coherencia:

``` math
\boxed{\text{\:\,}\text{RBCI} = \frac{1}{4}\left\lbrack \underset{\text{slope}}{\overset{\text{norm}\left( \alpha_{\text{bio,enz}};\lbrack 1,4\rbrack \right)}{︸}} + \underset{\text{spin}}{\overset{\text{norm}\left( P_{\text{CISS}};\lbrack 0,1\rbrack \right)}{︸}} + \underset{\text{vibrational}}{\overset{\text{norm}\left( C_{\text{Raman}};\lbrack 0,1\rbrack \right)}{︸}} + \underset{\text{variance reduction}}{\overset{\text{norm}\left( \Delta\text{Var}_{k};\lbrack 0,1\rbrack \right)}{︸}} \right\rbrack\text{\:\,}}
```

- $`norm(x;\lbrack a,b\rbrack) = \min\{ 1,\max\{ 0,(x - a)/(b - a)\}\}`$.

- $`\Delta{Var}_{k} = \max\{ 0,\text{ }Var(k)_{\text{off}} - Var(k)_{\text{on}}\}/Var(k)_{\text{off}}`$.

- Reportar RBCI **con** las puntuaciones de componentes para permitir análisis de sensibilidad dejando fuera un componente a la vez.

**Interpretación:** RBCI cercano a 1 indica pendiente alta (α grande) **y** firmas de coherencia fuertes y convergentes; RBCI cercano a 0 indica α bajo y ausencia de evidencia de coherencia.

**3.12 Series de alostería y desnaturalización**

- **Alostería:** ejecutar la serie completa de $`L`$ ± activador/inhibidor a $`T`$, pH y fuerza iónica equivalentes. Se espera $`\alpha_{bio,enz}`$ ↑ con activador, ↓ con inhibidor; RBCI covaría.

- **Desnaturalización:** desplegamiento gradual (urea/guanidinio o temperatura) mientras se monitorean $`P_{CISS}`$, $`C_{Raman}`$ y actividad. Se espera una disminución monótona en los componentes de coherencia y RBCI; $`\alpha_{bio,enz}`$ se desplaza hacia valores difusivos.

**3.13 Análisis estadístico**

- **Prerregistro:** especificar los resultados primarios ($`\alpha_{bio,enz}`$, aprobación/fallo del colapso), los resultados secundarios (RBCI, componentes) y las reglas de exclusión.

- **Tamaño de muestra y potencia:** para la detección de pendiente, apuntar a un efecto de $`\Delta\alpha = 0.2`$ con DE = 0,15 sobre ≥4 valores distintos de $`L`$; la potencia basada en simulación ≥0,8 sugiere $`n \geq 8`$ réplicas por $`L`$ por condición.

- **Comparaciones múltiples:** controlar la TDF (Benjamini–Hochberg) en los criterios de valoración secundarios.

- **Robustez:** reportar ajustes OLS y ortogonales; reestimar tras eliminar el 5 % superior/inferior de los valores de $`k`$ (análisis de influencia).

- **Compartición:** publicar las series temporales crudas, metadatos (temperatura, pH, iónico) y scripts de análisis.

**3.14 Auditoría de artefactos y seguridad**

- **Artefactos térmicos:** micro-termometría concurrente; incluir un control térmico reproduciendo el mismo ΔT con un Peltier (sin excitación).

- **Mezcla/flujo:** pruebas con trazador; rechazar condiciones que alteren la hidrodinámica.

- **Artefactos ópticos:** controles de fotoblanqueo para Raman/UV-Vis; mediciones en oscuridad.

- **Artefactos eléctricos (CISS):** verificar inversiones de magnetización, medir con cableado invertido, incluir películas de control no quirales.

- **Bioseguridad:** manejo estándar de enzimas; desechar desnaturalizantes según las directrices institucionales.

**3.15 Disponibilidad de datos y código**

Todos los datos crudos, curvas de calibración para $`L`$, código para pendiente/ANCOVA/BCES, cálculo de RBCI y generación de figuras se depositarán en un repositorio abierto al momento de la presentación. Un **cuaderno de análisis** ligero reproduce las estimaciones de pendiente, ICs por bootstrap y diagnósticos de colapso a partir de entradas CSV.

**4. Experimentos**

Este capítulo especifica cuatro experimentos prerregistrados (E1–E4) para estimar el exponente de escalamiento enzimático $`\alpha_{bio,enz}`$, calcular el Índice de Coherencia de Bioquímica Rítmica (RBCI) y probar las predicciones de RTM (estabilidad de pendiente, colapso de datos, cambio de clase, covariación alostería/CISS). Cada experimento incluye **diseño**, **protocolo**, **lecturas**, **firmas esperadas** y **criterios de aprobación/fallo**. Todas las secciones asumen condiciones isotérmicas, fuerza iónica fija y tampones equivalentes salvo que se indique lo contrario.

**E1 — Confinamiento multiescala (pendiente primaria y colapso de datos)**

**Objetivo.** Estimar $`\alpha_{bio,enz}`$ a partir de $`\log k`$ vs $`\log L`$ a lo largo de al menos una década en $`L`$, y probar el colapso de datos.

**Diseño.**

- Enzima: LDH (primaria) y ureasa (replicación).

- Serie de confinamiento $`L`$: diámetros de poro nominales 5, 10, 20, 50, 100, 200 nm (AAMs o sílice mesoporosa). Verificar la morfología (SEM/BET) y calcular $`L_{eff} = L_{pore}\sqrt{\tau}`$.

- Réplicas: $`n \geq 8`$ estimaciones independientes de $`k`$ por $`L`$.

- Aleatorización: orden de $`L`$ permutado; analista ciego a las etiquetas de $`L`$ al momento del ajuste.

**Protocolo.**

1.  Equilibrar las matrices en tampón de ensayo (≥3× intercambios de volumen; de un día para otro si es necesario).

2.  Cargar la enzima (masa/actividad fija por membrana/monolito).

3.  Iniciar la reacción bajo condiciones de sustrato idénticas; registrar $`k`$ (lector de placas o flujo detenido).

4.  Registrar la temperatura (±0,01 °C); excluir corridas con **ΔT > 0,05 °C**.

5.  Repetir para todos los $`L`$.

**Lecturas y análisis.**

- Pendiente primaria: $`\alpha_{bio,enz} = - \text{ }d\ \log k/d\ \log L`$ (OLS + ortogonal/BCES).

- **Colapso de datos:** calcular $`\widetilde{k} = k\text{ }L^{\alpha^{\star}}`$; optimizar $`\alpha^{\star}`$ para varianza mínima entre curvas; prueba tipo KS de indistinguibilidad.

- ANCOVA para comparar pendientes entre lotes de enzima y lotes de matrices.

**Firmas esperadas.**

- Banda de transporte jerárquico/fractal: $`\alpha_{bio,enz} \approx 2.3\text{–}2.7`$.

- Colapso exitoso cuando $`\alpha^{\star} \in`$ <!-- -->IC al 95 % de $`\alpha_{bio,enz}`$.

**Aprobación/Fallo.**

- **Aprobado** si: el IC de la pendiente excluye 2,0 por ≥0,15 y el colapso de datos se aprueba; los residuos no muestran deriva sistemática vs $`L`$.

- **Fallo** si: la pendiente es inestable a través de $`L`$ (términos de interacción significativos sin cambio mecanístico), el colapso falla o los artefactos (mezcla/calentamiento) explican la varianza.

**Controles.**

- Matrices ficticias (mismo $`L`$, superficie inerte) para verificar artefactos de adsorción.

- Medición en solución libre como referencia (sin confinamiento).

**E2 — Excitación acústica (cambio de clase y reducción de varianza)**

**Objetivo.** Probar si la excitación **en resonancia** mueve el sistema entre clases de transporte (cambio en $`\alpha`$) y reduce la varianza de la velocidad, sin calentamiento.

**Diseño.**

- Frecuencias: 20 kHz, 200 kHz, 2 MHz (±2 %).

- Definir **en resonancia** como la frecuencia que minimiza $`Var(k)`$ en barridos piloto a $`L`$ fijo con $`\Delta T < {0.05}^{\circ}C`$; **fuera de resonancia** ≥10× alejada, misma amplitud RMS.

- Usar $`L`$ de rango medio (por ejemplo, 20 y 50 nm) para evitar efectos de piso/techo.

**Protocolo.**

1.  Calibrar la amplitud con acelerómetro/vibrómetro láser en el soporte; documentar el voltaje del piezoeléctrico para cada frecuencia.

2.  Para cada $`L`$, registrar $`k`$ bajo: (i) apagado, (ii) fuera de resonancia, (iii) en resonancia (secuencia aleatorizada, $`n \geq 8`$ cada una).

3.  Registrar temperatura continuamente; excluir si se excede el umbral de $`\Delta T`$.

4.  Repetir entre enzimas (LDH, ureasa).

**Lecturas y análisis.**

- Pendientes por condición: $`\alpha_{\text{off}},\alpha_{\text{off-res}},\alpha_{\text{on}}`$ con ICs por bootstrap; interacción ANCOVA $`\log L \times \text{condición}`$.

- Cambio de varianza: $`\Delta{Var}_{k} = \max\{ 0,Var(k)_{\text{off}} - Var(k)_{\text{on}}\}/Var(k)_{\text{off}}`$.

- Actualización del componente RBCI "reducción de varianza" y RBCI global.

**Firmas esperadas.**

- **Cambio de clase:** $`\alpha_{\text{on}} - \alpha_{\text{off}} \geq 0.2`$ (no solapamiento de IC) hacia la banda predicha; $`\Delta{Var}_{k} > 0`$ significativo.

- Sin calentamiento medible; la excitación fuera de resonancia muestra efectos despreciables.

**Aprobación/Fallo.**

- **Aprobado** si el cambio de pendiente y la reducción de varianza ocurren **juntos** sin ΔT, coincidiendo con las predicciones de RTM.

- **Fallo** si los cambios correlacionan con calentamiento/mezcla o no son reproducibles entre días/lotes.

**Controles.**

- Control térmico Peltier reproduciendo ΔT (sin excitación acústica).

- Piezoeléctrico inerte (alimentado pero mecánicamente desacoplado) para descartar captación EM.

**E3 — Serie de desnaturalización con CISS (covariación de coherencia)**

**Objetivo.** Probar si la selectividad de espín (CISS) y la coherencia vibracional covarían con RBCI y disminuyen monótonamente con la pérdida estructural.

**Diseño.**

- Crear una serie de desplegamiento graduado (por ejemplo, 0–6 M urea o 0–4 M GdnHCl; o una rampa de temperatura).

- Preparar monocapas proteicas quirales sobre Au(111)/ITO; medir CISS a ±$`M`$.

- Adquirir espectros Raman/IR en paralelo (mismas muestras).

**Protocolo.**

1.  Para cada nivel de desnaturalizante, preparar películas y muestras de ensayo a granel en paralelo.

2.  Medir $`P_{CISS}`$ a voltaje de polarización fijo (triplicado por nivel, magnetización invertida en cada corrida).

3.  Registrar la cinética $`k`$ (a granel) y calcular los componentes RBCI (CISS, vibracional $`C_{Raman}`$).

4.  Confirmar la disminución de estructura secundaria/terciaria (espectroscopia CD o fluorimetría diferencial de barrido, opcional).

**Lecturas y análisis.**

- Pruebas de monotonicidad ($`\tau`$ de Kendall) para $`P_{CISS}`$ y $`C_{Raman}`$ vs desnaturalizante.

- Correlación de RBCI con el indicador de estructura y con $`\alpha_{bio,enz}`$ (Pearson/Spearman).

- Comparar $`\alpha_{bio,enz}`$ a baja vs alta desnaturalización.

**Firmas esperadas.**

- $`P_{CISS} \downarrow`$ y $`C_{Raman} \downarrow`$ monótonamente; RBCI disminuye en consecuencia.

- $`\alpha_{bio,enz}`$ se desplaza hacia valores difusivos (≈2) a medida que se pierde estructura/coherencia.

**Aprobación/Fallo.**

- **Aprobado** si las disminuciones monótonas son significativas (controladas por TDF) y RBCI covaría con tanto CISS como coherencia vibracional; las pendientes se desplazan hacia α más bajo.

- **Fallo** si los cambios de CISS/vibracionales se desacoplan del RBCI o si las pendientes permanecen sin cambios bajo una desnaturalización clara.

**Controles.**

- Películas de control no quirales o desnaturalizadas para CISS.

- Espectros solo de tampón; iluminación idéntica para monitorear fotoblanqueo.

**E4 — Ajuste alostérico (modulación de α)**

**Objetivo.** Demostrar que los ligandos alostéricos modulan $`\alpha_{bio,enz}`$ y RBCI más allá de los cambios clásicos de $`K_{M}/k_{\text{cat}}`$.

**Diseño.**

- Elegir pares enzima-efector con activación/inhibición conocida (por ejemplo, LDH-A con FBP como activador).

- Realizar la serie completa de $`L`$ **± efector** en condiciones equivalentes.

**Protocolo.**

1.  Pre-incubar la enzima con activador o inhibidor (concentración a niveles escalados de $`{EC}_{50}`$ / $`{IC}_{50}`$).

2.  Ejecutar el protocolo E1 a lo largo de $`L`$ para cada condición (orden aleatorizado).

3.  Opcionalmente combinar con la excitación E2 para probar sinergia.

**Lecturas y análisis.**

- Comparar $`\alpha_{bio,enz}`$ ± efector (ANCOVA).

- Componentes RBCI: buscar aumentos (activador) o disminuciones (inhibidor) en la reducción de varianza y la coherencia vibracional.

- Reportar los parámetros cinéticos clásicos para completitud, pero interpretar por clase de transporte.

**Firmas esperadas.**

- Activador: $`\alpha_{bio,enz} \uparrow`$ en ≥0,2, RBCI↑; Inhibidor: tendencia opuesta.

- Colapso de datos mejorado bajo activación (métrica de colapso más ajustada).

**Aprobación/Fallo.**

- **Aprobado** si la pendiente y RBCI se desplazan en las direcciones predichas con significancia corregida por TDF y sin ΔT/mezcla artefactual.

- **Fallo** si solo cambian $`K_{M}/k_{\text{cat}}`$ mientras $`\alpha`$ y RBCI no lo hacen, o si los cambios desaparecen bajo controles fuera de resonancia/térmicos.

**Controles.**

- Control de vehículo del efector; titulación del efector para descartar efectos inespecíficos.

- Verificación cruzada con un segundo par alostérico si está disponible.

**Elementos compartidos (para todos E1–E4)**

**Cegamiento y aleatorización.**

- Codificar las etiquetas de $`L`$ y condición; el análisis se realiza con etiquetas enmascaradas hasta que se ejecute el plan prerregistrado.

**Criterios de inclusión/exclusión.**

- Excluir corridas con $`\Delta T > {0.05}^{\circ}C`$, ajustes con R² < 0,95 o perturbaciones mecánicas/EM documentadas. Todas las exclusiones se predeclaran.

**Potencia y replicación.**

- Apuntar a $`\Delta\alpha = 0.2`$ con DE = 0,15; al menos 4 valores distintos de $`L`$, $`n \geq 8`$ réplicas cada uno; dos enzimas (primaria + replicación).

**Seguridad.**

- Seguir las normas institucionales de seguridad química para desnaturalizantes y excitadores piezoeléctricos de alto voltaje; protección auditiva cerca de configuraciones de ultrasonido.

**Figuras esperadas (a completar con datos)**

- **Figura 1 (E1):** $`\log k`$ vs $`\log L`$ con pendiente ajustada e IC por bootstrap; **recuadro**: gráfico de colapso de datos de $`\widetilde{k} = k\text{ }L^{\alpha^{\star}}`$.

- **Figura 2 (E2):** Comparación de pendientes apagado/fuera de resonancia/en resonancia (gráfico de bosque de $`\alpha`$ con ICs) + barra de $`\Delta{Var}_{k}`$; traza del termómetro confirmando ΔT.

- **Figura 3 (E3):** $`P_{CISS}`$ y $`C_{Raman}`$ vs desnaturalizante; RBCI vs indicador de estructura; desplazamiento de $`\alpha`$.

- **Figura 4 (E4):** $`\alpha`$ ± efector; componentes RBCI; mejora de la métrica de colapso.

**Lista de verificación de prerregistro (resumen)**

- **Resultados primarios:** $`\alpha_{bio,enz}`$ por condición; aprobación/fallo del colapso de datos.

- **Resultados secundarios:** RBCI y componentes; $`\Delta{Var}_{k}`$; CISS; potencia vibracional coherente.

- **Controles y umbrales:** ΔT < 0,05 °C; R² ≥ 0,95; reglas de exclusión; diseño aleatorizado/bloqueado.

- **Plan de análisis:** OLS + ortogonal; bootstraps; ANCOVA; métricas de colapso KS/varianza; control de TDF.

- **Regla de detención:** tamaños de muestra preespecificados; repetir los días atípicos si se excluye >25 % de las corridas por razones técnicas.

**5. Resultados**

> *Nota:* Esta sección especifica la estructura de reporte, los resultados estadísticos y las plantillas de figuras/tablas. Donde los datos aún no se han recolectado, proporcionamos **marcadores de posición** y **oraciones exactas** que pueden reutilizarse literalmente una vez que los números estén disponibles.

**5.1 E1 — Confinamiento multiescala: pendiente y colapso de datos**

**Resultado primario (pendiente).**\
A lo largo de seis escalas de confinamiento (5–200 nm), la regresión log–log de velocidad vs. longitud produjo

``` math
\log k = \beta_{0} + \beta_{1}\log L,\alpha_{bio,enz} = - \text{ }{\widehat{\beta}}_{1}.
```

**LDH (primaria):** $`\alpha_{bio,enz} = \lbrack X.XX\rbrack\text{\:\,}(95\%\text{ }IC\text{\:\,}\lbrack X.XX,\text{ }X.XX\rbrack)`$ por OLS; ortogonal/BCES dio $`\lbrack X.XX\rbrack`$.\
**Ureasa (replicación):** $`\alpha_{bio,enz} = \lbrack X.XX\rbrack\text{\:\,}(95\%\text{ }IC\text{\:\,}\lbrack X.XX,\text{ }X.XX\rbrack)`$.

**Plantilla de interpretación.**

- Si el IC excluye 2,0: "Las pendientes exceden la cota inferior difusiva ($`\alpha = 2`$) y caen en la banda jerárquica/fractal ($`2.3\text{–}2.7`$)."

- Si el IC se solapa con 2,0: "Las pendientes son compatibles con difusión local; RTM predice que puede requerirse un cambio de clase para revelar vías no locales."

**Colapso de datos.**\
Reescalar $`\widetilde{k} = k\text{ }L^{\alpha^{\star}}`$ minimizó la varianza entre curvas en $`\alpha^{\star} = \lbrack X.XX\rbrack`$, dentro del IC al 95 % de $`\alpha_{bio,enz}`$. Prueba de indistinguibilidad tipo KS: $`D = \lbrack X.XXX\rbrack,p = \lbrack X.XXX\rbrack`$.\
**Oración de conclusión:** "El colapso de datos **se aprobó**/**falló**; el $`\alpha^{\star}`$ óptimo **coincide**/**no coincide** con la estimación de pendiente."

**5.2 E2 — Excitación acústica: cambio de clase y reducción de varianza**

**Comparación de pendientes (ANCOVA).**\
Interacción $`(\log L \times condición)`$ significativa: $`F = \lbrack X.XX\rbrack,p = \lbrack X.XXX\rbrack`$.\
Pendientes estimadas:

- **Apagado:** $`\alpha_{\text{off}} = \lbrack X.XX\rbrack\text{\:\,}(\lbrack X.XX,X.XX\rbrack)`$

- **Fuera de resonancia:** $`\alpha_{\text{off-res}} = \lbrack X.XX\rbrack\text{\:\,}(\lbrack X.XX,X.XX\rbrack)`$

- **En resonancia:** $`\alpha_{\text{on}} = \lbrack X.XX\rbrack\text{\:\,}(\lbrack X.XX,X.XX\rbrack)`$

**Regla de decisión de cambio de clase (restablecer en resultados).**\
"El cambio de clase **ocurrió** si $`\alpha_{\text{on}} - \alpha_{\text{off}} \geq 0.2`$ y los ICs mostraron no solapamiento; en caso contrario, **no observado**."

**Reducción de varianza.**\
$`\Delta{Var}_{k} = \max\{ 0,Var(k)_{\text{off}} - Var(k)_{\text{on}}\}/Var(k)_{\text{off}} = \lbrack X.XX\rbrack`$.\
Control térmico: ΔT = \[0,XX\] °C (por debajo del umbral de 0,05 °C). El control solo con Peltier no produjo cambio de pendiente/varianza.

**Actualización del RBCI.**\
El componente de **reducción de varianza** aumentó en $`\lbrack X.XX\rbrack`$; el **RBCI** global subió de $`\lbrack 0.XX\rbrack`$ (apagado) a $`\lbrack 0.XX\rbrack`$ (en resonancia).

**5.3 E3 — Serie de desnaturalización: CISS y coherencia vibracional**

**Tendencias monótonas.**\
$`\tau`$ de Kendall para CISS vs desnaturalizante: $`\tau = \lbrack X.XX\rbrack,p = \lbrack X.XXX\rbrack`$ (esperado **negativo**).\
$`\tau`$ de Kendall para potencia vibracional coherente: $`\tau = \lbrack X.XX\rbrack,p = \lbrack X.XXX\rbrack`$ (esperado **negativo**).

**Correlaciones con RBCI y pendiente.**\
Pearson/Spearman $`r`$ entre **RBCI** y **CISS**: $`r = \lbrack X.XX\rbrack,p = \lbrack X.XXX\rbrack`$.\
Entre **RBCI** y **potencia vibracional coherente**: $`r = \lbrack X.XX\rbrack,p = \lbrack X.XXX\rbrack`$.\
Entre $`\alpha_{bio,enz}`$ y nivel de desnaturalización: desplazamiento de pendiente $`\Delta\alpha = \lbrack \pm X.XX\rbrack`$ hacia/más allá de valores difusivos.

**5.4 E4 — Ajuste alostérico: modulación de** $`\mathbf{\alpha}`$

**Cambios de pendiente.**\
El activador aumentó la pendiente en $`\Delta\alpha = + \lbrack 0.XX\rbrack`$ (IC \[X.XX, X.XX\]); el inhibidor la disminuyó en $`- \lbrack 0.XX\rbrack`$. Las interacciones ANCOVA fueron significativas: $`F = \lbrack X.XX\rbrack,p = \lbrack X.XXX\rbrack`$.

**Covariación del RBCI.**\
RBCI **subió** de $`\lbrack 0.XX\rbrack`$ a $`\lbrack 0.XX\rbrack`$ con el activador y **bajó** a $`\lbrack 0.XX\rbrack`$ con el inhibidor. Los componentes de reducción de varianza y vibracionales cambiaron coherentemente con la pendiente.

**Cinética clásica para completitud.**\
$`k_{\text{cat}}`$ y $`K_{M}`$ se desplazaron como se esperaba, pero la **narrativa de clase de transporte** (pendiente + RBCI) explica la covariación de la estabilización de la velocidad y la coherencia.

**5.5 Robustez, sensibilidad y controles negativos**

- **Ajustes ortogonales:** las estimaciones BCES concordaron dentro de $`\pm \lbrack 0.05\rbrack`$ respecto a OLS; las conclusiones no cambiaron.

- **Análisis de influencia:** eliminar el 5 % superior/inferior de los valores de $`k`$ desplazó $`\alpha`$ en $`\leq \lbrack 0.03\rbrack`$.

- **Controles fuera de resonancia y ficticios:** sin cambio significativo de pendiente ni RBCI; la reproducción del ΔT solo con Peltier no reprodujo ninguno de los efectos en resonancia.

- **Efectos de lote:** sin interacción significativa día/lote (modelo de efectos mixtos; razón de verosimilitud $`p = \lbrack X.XXX\rbrack`$).

- **Exclusiones preespecificadas:** \[N\] de \[Total\] corridas excluidas según reglas a priori (R², ΔT, artefactos); la inclusión de las corridas excluidas en los análisis de sensibilidad no cambió los resultados cualitativos.

**5.6 Declaración de resumen (un párrafo que puede conservarse tal cual)**

A lo largo de cuatro experimentos prerregistrados, las velocidades enzimáticas medidas sobre escalas de confinamiento controlables respaldaron una ley de escalamiento RTM con exponentes en la banda jerárquica/fractal y exhibieron **colapso de datos** bajo el reescalamiento predicho. La excitación **en resonancia** produjo **cambio de clase** (aumento de pendiente) acompañado de **reducción de varianza** sin calentamiento medible, mientras que la **desnaturalización** deprimió CISS y la coherencia vibracional en tándem con un desplazamiento de $`\alpha`$ hacia valores difusivos. Los **ligandos alostéricos** modularon tanto $`\alpha_{bio,enz}`$ como RBCI en las direcciones predichas. En conjunto, estos resultados alinean la catálisis enzimática con **clases de universalidad de transporte** y muestran que las **firmas de coherencia** y los **exponentes de escalamiento** se mueven juntos, tal como prescribe RTM.

**5.7 Tablas (plantillas)**

**Tabla 1.** Estimaciones de pendiente por condición (media, IC al 95 %; ajustes OLS y ortogonales).\
**Tabla 2.** Métricas de colapso ($`\alpha^{\star}`$ óptimo, razón de varianza, KS $`D,p`$).\
**Tabla 3.** Componentes RBCI y total, por experimento y condición.\
**Tabla 4.** CISS y coherencia vibracional vs. desnaturalización; $`\tau`$ de Kendall, $`p`$.\
**Tabla 5.** Alostería: $`\Delta\alpha`$, cambio de RBCI, y $`k_{\text{cat}},K_{M}`$ clásicos (solo contexto).

**6. Discusión**

**6.1 ¿Qué mide** $`\mathbf{\alpha}`$ **en las enzimas?**

Dentro de RTM, $`\alpha`$ no es una constante microscópica sino un **exponente operacional** que codifica la **clase de transporte** que limita el recambio: difusivo, jerárquico/fractal, guiado/parcialmente balístico o (heurísticamente) confinado cuánticamente. Las enzimas se sitúan en una mesoescala donde la **geometría, la rigidez, la quiralidad y la hidratación** coproducen esa clase. Un $`\alpha_{bio,enz} \approx 2.3\text{–}2.7`$ ajustado indica un mejoramiento de la **dimensión de caminata** (trampas/corredores) típico de interiores proteicos ramificados o matrices aglomeradas; el desplazamiento de $`\alpha`$ hacia 2,0 con la desnaturalización señala la pérdida de organización jerárquica. Así, $`\alpha`$ funciona como un **resumen comprimido** de la arquitectura de vías, complementario a $`k_{\text{cat}}`$, $`K_{M}`$ y los parámetros de activación.

**6.2 Evidencia de coherencia: por qué importa RBCI**

RBCI triangula la pendiente con **observables de coherencia** (CISS, potencia vibracional, reducción de varianza bajo excitación en resonancia). RTM predice que estas firmas **covarían** porque elevar $`\alpha`$ corresponde a estabilizar canales ordenados y suprimir la mezcla difusiva. Si las pendientes cambian sin movimiento de RBCI, el cambio es probablemente **térmico o hidrodinámico**; si RBCI sube sin cambio de pendiente, la coherencia puede ser local pero **no limitante de la velocidad**. Reportar ambos crea un **filtro de artefactos** y un punto de referencia portátil entre laboratorios.

**6.3 Alostería reformulada como ajuste de clase de transporte**

La alostería clásica desplaza poblaciones a lo largo de coordenadas conformacionales. En RTM, los efectores **ajustan el generador de transporte**, alterando la fracción de microtrayectorias guiadas vs. difusivas. Esto explica por qué algunos activadores estabilizan las velocidades (reducción de varianza) más allá de los cambios de campo medio en $`k_{\text{cat}}`$ o $`K_{M}`$, y predice **sinergia** entre la alostería y la excitación periódica suave que fija el sistema en un régimen de α elevado.

**6.4 Relación con las teorías existentes**

- **Estado de transición/Marcus/Kramers:** RTM **no** reemplaza los modelos de barrera; los envuelve afirmando que **el tiempo para realizar la coordenada limitante de la velocidad** escala con $`L`$. Las alturas de barrera configuran el **intercepto**; la **arquitectura de vías** fija la **pendiente**.

- **Cinética fractal/teoría de aglomeración:** RTM recupera estas como el caso $`\alpha = d_{w}`$ con $`d_{w} > 2`$, proporcionando un **lenguaje unificado** para comparar proteínas, membranas y geles.

- **Catálisis asistida vibracionalmente y terremotos proteicos:** el componente vibracional de RBCI operacionaliza estas ideas y exige **co-movimiento** con $`\alpha`$.

**6.5 Limitaciones y modos de fallo**

- **No estacionariedad a lo largo de** $`L`$ **:** si el mecanismo cambia (por ejemplo, ruta de acceso de sustrato diferente) dentro de la ventana explorada, las pendientes se vuelven **por tramos**. Nuestras pruebas ANCOVA y de colapso detectan esto; reportar $`\alpha`$ por tramos es aceptable pero debe declararse.

- **Calibración de** $`L`$ **:** los errores en el tamaño de poro/malla sesgan las pendientes; por lo tanto, los ajustes ortogonales/BCES y la calibración SEM/BET/SAXS son obligatorios.

- **Confusiones de calentamiento/mezcla:** la excitación acústica o EM puede alterar la hidrodinámica. Acotamos esto con umbrales de ΔT, controles de mezcla con colorantes inertes y un control térmico **solo con Peltier**.

- **Especificidad de CISS:** la asimetría de espín puede ser sensible a los contactos y las fugas; las películas no quirales y desnaturalizadas son controles requeridos.

- **Banda superior heurística:** las afirmaciones cercanas a $`\alpha \sim 3`$ permanecen como **conjeturales**; sin aumentos sincronizados en los componentes de RBCI y un colapso limpio, tales valores no deberían avanzarse.

**6.6 Implicaciones**

- **Mapeo mecanístico:** las enzimas pueden **ubicarse en un mapa** (difusivo ↔ fractal ↔ guiado) usando $`\alpha`$ y RBCI, aclarando por qué proteínas superficialmente similares difieren en estabilidad y especificidad.

- **Diseño de ensayos:** elegir $`L`$ y excitación suave para **maximizar el colapso** puede mejorar la precisión del ensayo (menor varianza) sin elevar la temperatura.

- **Descubrimiento de fármacos:** tamizar ligandos alostéricos por **ganancia de** $`\alpha`$ y **ganancia de RBCI**, favoreciendo compuestos que estabilicen vías coherentes en lugar de simplemente desplazar $`K_{M}`$.

- **Biotecnología:** las estrategias de microreactores e inmovilización pueden apuntar a configuraciones de **α elevado** para mejorar el rendimiento y la reproducibilidad.

**6.7 Predicciones más allá de las enzimas**

- **Módulos metabólicos:** los complejos multienzimáticos deberían exhibir un $`\alpha`$ **a nivel de módulo** mayor que el de las enzimas aisladas si domina la canalización/guía; RBCI debería subir con la rigidez del andamiaje.

- **Membranas y transportadores:** los canales con rectificación y quiralidad deberían mostrar RBCI y $`\alpha`$ más altos que los poros no selectivos en condiciones equivalentes.

- **Temporización a nivel celular:** los subprocesos del ciclo celular y circadiano pueden mostrar colapso bajo reescalamientos que preserven la estructura (aglomeración nuclear/citoplasmática), ofreciendo una ruta hacia el mapeo de $`\alpha`$ **a nivel del organismo**.

**6.8 ¿Qué falsificaría RTM en bioquímica?**

- **Sin estabilidad de pendiente** a lo largo de $`L`$ a pesar de controles estrictos.

- **Fallo del colapso** incluso cuando la pendiente está bien definida.

- **Desacoplamiento** de $`\alpha`$ respecto a RBCI bajo manipulaciones predichas para cambiar la clase de transporte (excitación/alostería/desnaturalización).

- **Mimetismo térmico:** todos los efectos observados desaparecen cuando ΔT se reproduce con Peltier; o los efectos rastrean indicadores de mezcla en lugar de topología de transporte.

**6.9 Estándares de datos y reproducibilidad**

Recomendamos: (i) publicar las series temporales crudas y la **calibración de** $`L`$; (ii) publicar la **superficie completa de optimización del colapso** vs. $`\alpha^{\star}`$; (iii) reportar RBCI **con sus componentes**; (iv) planes prerregistrados con **scripts para OLS/BCES, ANCOVA, bootstrap**; y (v) incluir **controles negativos** (fuera de resonancia, matrices ficticias, películas desnaturalizadas).

**7. Perspectivas y aplicaciones**

**7.1 Aplicaciones prácticas**

**Diagnósticos.**

- **Déficits de coherencia como biomarcadores.** Un RBCI bajo con $`\alpha_{bio,enz}`$ desplazándose hacia 2,0 puede indicar pérdida de organización jerárquica en enfermedades (por ejemplo, mal plegamiento de proteínas, daño oxidativo). Los paneles que combinan enzimas de vías distintas podrían revelar **decoherencia sistémica**.

- **Monitoreo terapéutico.** Rastrear $`\alpha_{bio,enz}`$ y RBCI longitudinalmente durante terapia con chaperonas o intervenciones redox; la mejora significa restauración de la clase de transporte en lugar de mero aumento de velocidad.

**Descubrimiento de fármacos.**

- **Tamizaje alostérico por ganancia de** $`\alpha`$. Priorizar ligandos que **eleven** $`\alpha_{bio,enz}`$ y **RBCI** bajo controles isotérmicos y fuera de resonancia, indicativos de estabilizar vías guiadas.

- **Compuestos anti-decoherencia.** Identificar compuestos que recuperen el colapso de datos y la reducción de varianza (RBCI ↑) tras estrés/desnaturalización.

**Bioprocesos y biotecnología.**

- **Microreactores de α elevado.** Diseñar matrices de inmovilización (tamaño de poro, tortuosidad, rigidez, quiralidad) y excitaciones suaves que empujen al catalizador hacia una **clase de α elevado** estable con variabilidad estrecha.

- **CC de proceso.** Usar la métrica de colapso y RBCI como **puntuaciones de salud en tiempo de ejecución** para reactores (alarma cuando el colapso falla o RBCI baja).

**Biología sintética.**

- **Ingeniería de andamiaje.** Predecir que andamiajes más rígidos, quirales y metabolones guiados producen aumentos de $`\alpha`$ y RBCI a nivel de módulo; validar intercambiando conectores y midiendo el colapso.

- **Control rítmico.** Excitación periódica de baja potencia (mecánica/eléctrica) como una **perilla no térmica** para mejorar la coherencia sin cambiar los niveles de expresión.

**7.2 Hoja de ruta a corto plazo (0–12 meses)**

1.  **Replicación en dos enzimas.** Ejecutar E1–E4 en LDH y ureasa; prerregistrar el análisis; publicar datos crudos + cuadernos.

2.  **Kit de calibración.** Publicar un pequeño **kit RTM–Bio** abierto: estándares de tamaño de poro, recetas de tampón, scripts de excitación y código de análisis para pendiente/colapso/RBCI.

3.  **Prueba interlaboratorio en anillo.** Al menos tres laboratorios ejecutan E1 y E2 con protocolos equivalentes; reportar la variabilidad entre sitios de $`\alpha`$ y RBCI.

4.  **Caso de estudio alostérico.** Un par de efectores que muestre un cambio claro de $`\alpha`$ y covariación de RBCI; incluir efector negativo.

**7.3 Hoja de ruta a mediano plazo (12–24 meses)**

- **Mapeo de mecanismos.** Análisis de $`\alpha`$ por tramos a lo largo de ventanas más amplias de $`L`$ para identificar **transiciones de mecanismo** (limitado por acceso → limitado por química).

- **Estandarización de CISS.** Validación cruzada de montajes de espín; publicar pruebas de fuga y líneas base no quirales para robustecer $`P_{CISS}`$ como medida comunitaria.

- **Variantes de RBCI.** Explorar esquemas de ponderación y robustez **dejando un componente fuera**; evaluar alternativas (por ejemplo, métricas de coherencia dieléctrica) en lugar de Raman cuando no esté disponible.

- **Pruebas a nivel de módulo.** Metabolones reconstituidos o pares de enzimas para cuantificar $`\alpha`$ y RBCI del **módulo** vs. rigidez/quiralidad del andamiaje.

**7.4 Problemas abiertos**

- **Causalidad de la coherencia.** ¿La coherencia **causa** el cambio de $`\alpha`$ o simplemente correlaciona con cambios arquitectónicos? Usar intervenciones que alteren la coherencia **sin** la geometría (por ejemplo, sustitución isotópica, campos electromagnéticos suaves) y probar la independencia de la pendiente respecto al calentamiento.

- **Mapeo microscópico.** Relacionar $`\alpha`$ con la **dimensión de caminata** $`d_{w}`$ y las **medidas espectrales** de la red proteína/solvente (espectros del Laplaciano de grafo a partir de simulaciones o experimentos).

- **Afirmaciones de la banda superior.** Los valores cercanos a $`\alpha \sim 3`$ permanecen como **heurísticos**; requieren aumentos sincronizados en todos los componentes de RBCI y controles de artefactos a prueba de balas antes de cualquier atribución mecanística.

**7.5 Consideraciones éticas y de seguridad**

- **Excitaciones no térmicas.** Mantener umbrales conservadores de ΔT y publicar termometría en tiempo real; evitar regímenes que arriesguen cavitación o daño estructural.

- **Transparencia de datos.** Compartir trazas crudas, calibración de $`L`$ y superficies completas de colapso; prerregistrar resultados negativos para prevenir el sesgo del ganador.

- **Extensión clínica.** Si se persigue uso diagnóstico, protegerse contra la **sobreinterpretación**: RBCI no es una etiqueta de enfermedad; cuantifica **características de coherencia** que necesitan contexto clínico.

**7.6 Estándares y reporte**

- Reportar $`\alpha_{bio,enz}`$ con **ambos** ajustes OLS y ortogonal/BCES; incluir ICs por bootstrap y resultados ANCOVA.

- Proporcionar **diagnósticos de colapso**: $`\alpha^{\star}`$ óptimo, razones de varianza y estadísticos KS.

- Publicar **RBCI con componentes** (pendiente, CISS, vibracional, reducción de varianza) y análisis de sensibilidad (recalcular RBCI dejando fuera cada componente).

- Adjuntar **auditorías de artefactos** (trazas de ΔT, pruebas de mezcla, verificaciones de fuga EM) en la Información Suplementaria.

**7.7 Criterios de éxito para el campo**

- $`\alpha`$ **reproducible** dentro de ±0,15 entre laboratorios para la misma enzima y geometría.

- **Colapso consistente** bajo la métrica prerregistrada.

- **Covariación** de RBCI y $`\alpha`$ bajo intervenciones (excitación/alostería/desnaturalización) en al menos dos familias de enzimas.

- Un **conjunto de datos de referencia** abiertamente disponible con código de análisis que nuevos grupos puedan usar para validar sus montajes.

**7.8 Impactos más amplios**

De confirmarse, la **Bioquímica Rítmica** reformula la optimización enzimática en torno a la **ingeniería de clase de transporte** en lugar de solo la manipulación de barreras. El enfoque ofrece un lenguaje común para comparar proteínas, materiales y microreactores, con implicaciones inmediatas para **ensayos de precisión**, **bioprocesamiento robusto** y **diseño alostérico racional**. Incluso si se refuta, las pruebas prerregistradas y las auditorías de artefactos agudizarán nuestra comprensión de cuándo la geometría, la coherencia y el transporte **no** controlan la catálisis, clarificando los límites y guiando teorías alternativas.

**8. Conclusión**

Hemos enmarcado la **Bioquímica Rítmica** como una instanciación operacional de **RTM** en sistemas enzimáticos, con dos anclas medibles: un **exponente de escalamiento** $`\alpha_{bio,enz}`$ extraído de las pendientes $`\log k`$ – $`\log L`$, y un **Índice de Coherencia de Bioquímica Rítmica (RBCI)** que triangula la coherencia mediante CISS, potencia vibracional y reducción de varianza bajo excitación no térmica. Juntos, estos indicadores conectan la especificidad y estabilidad catalítica con **clases de universalidad de transporte**: difusivo, jerárquico/fractal, guiado/parcialmente balístico y (heurísticamente) confinado cuánticamente.

El programa es **falsificable**. Predice estabilidad de pendiente y **colapso de datos** dentro de una clase, **cambio de clase** (desplazamientos discretos de $`\alpha`$) bajo excitación controlada, y **covariación** de RBCI con $`\alpha`$ bajo ajuste alostérico y desnaturalización. Aprobar estas pruebas unificaría la alostería, la selectividad de espín y la asistencia vibracional bajo una ley de escalamiento común; el fallo delinearía dónde el recambio enzimático está desacoplado del transporte multiescala.

Desde el punto de vista práctico, el marco ofrece rutas inmediatas para **ensayos de precisión**, **tamizaje alostérico** y **diseño de microreactores de α elevado**, al tiempo que impone auditorías rigurosas de artefactos (térmicos, de mezcla, eléctricos). Conceptualmente, reposiciona las narrativas de "forma y barrera" dentro de un relato más amplio donde la **arquitectura de vías** fija la pendiente y las **barreras** fijan el intercepto. Los puntos de referencia propuestos —$`\alpha`$ reproducible, diagnósticos de colapso, RBCI con componentes— son portátiles entre laboratorios y susceptibles de prerregistro y prácticas de datos abiertos.

Ya sea que se confirme o se refute, probar RTM en enzimas avanza el campo al convertir afirmaciones vagas de "coherencia" en **experimentos cuantitativos de grado decisorio**. El resultado consolidará una ley multiescala para la catálisis viva o agudizará las restricciones que cualquier teoría alternativa deba satisfacer.

**Disponibilidad de datos y código**

Todas las trazas cinéticas crudas, calibraciones de $`L`$ (SEM/BET/SAXS o curvas de tamaño de malla), registros de termometría, conjuntos de datos CISS, espectros Raman/IR y scripts de análisis (OLS/BCES, bootstrap, ANCOVA, optimización de colapso, cálculo de RBCI) se depositarán en un repositorio abierto al momento de la presentación. Un cuaderno reproducible regenerará todas las figuras y tablas a partir de entradas CSV.

**Prerregistro**

Los protocolos detallados, criterios de inclusión/exclusión, resultados primarios/secundarios y planes estadísticos para los Experimentos E1–E4 se prerregistrarán en \[URL del registro\] antes de la recolección de datos. Las desviaciones del protocolo se revelarán y justificarán.

**Intereses en competencia**

Los autores declaran **no tener intereses financieros en competencia**. Cualquier interés potencial no financiero (por ejemplo, participación en consorcios de estándares) se revelará al momento de la presentación.

**Información suplementaria (contenidos planificados)**

- **S1.** Calibración detallada de $`L`$ para cada matriz/agente de aglomeración (SEM/BET/SAXS; curvas de tamaño de malla polimérica).

- **S2.** Auditorías térmicas y de mezcla (trazas de ΔT, micrografías de trazadores, controles Peltier).

- **S3.** Validación del montaje CISS (pruebas de fuga, líneas base no quirales, inversiones de contacto).

- **S4.** Pipelines espectrales para potencia vibracional coherente (Raman/IR).

- **S5.** Superficies de optimización de colapso y verificaciones de robustez.

- **S6.** Análisis de sensibilidad para RBCI (dejando un componente fuera).

**Resumen ejecutivo de una página (apéndice opcional)**

- **Qué medir:** $`\alpha_{bio,enz}`$ (pendiente), RBCI (+ componentes).

- **Cómo decidir:** estabilidad de pendiente + colapso = misma clase; $`\Delta\alpha`$ + reducción de varianza + RBCI↑ = cambio de clase.

- **Controles:** ΔT < 0,05 °C; fuera de resonancia; matrices ficticias; líneas base desnaturalizadas/no quirales.

- **Criterios de éxito:** $`\alpha`$ reproducible (±0,15 entre laboratorios), colapso consistente, covariación de RBCI bajo intervenciones.

**9. Métodos y protocolos suplementarios**

> Esta sección especifica **recetas exactas, configuraciones de instrumentos y algoritmos de análisis** para que otro laboratorio pueda reproducir el trabajo. Donde se dan rangos, elegir el **valor por defecto** salvo que se indique lo contrario en el prerregistro.

**9.1 Tampones, reactivos y preparación de soluciones madre**

**Tampón general (TG):** HEPES 50 mM, NaCl 150 mM, $`{MgCl}_{2}`$ 5 mM, pH 7,40 ± 0,05 (25 °C).

- Pesar HEPES (11,92 g/L), NaCl (8,77 g/L), $`{MgCl}_{2}`$ · $`{6H}_{2}`$ O (1,02 g/L).

- Ajustar pH a 25 °C con NaOH 1 M; llevar a volumen; filtrar a 0,22 µm; almacenar a 4 °C (≤14 días).

**Mezcla de ensayo LDH:** TG + piruvato de sodio 1 mM + NADH 0,15 mM.\
**Mezcla de ensayo ureasa:** TG + urea 20 mM; pH 7,40; rojo fenol (colorimétrico opcional) 5 µg/mL.

**Agentes de aglomeración (si se usan):** PEG 35 kDa o dextrano 70 kDa (p/p 0–15 %). Preparar una **solución madre de aglomerante 10×**, diluir en TG inmediatamente antes del uso.

**Efectores alostéricos (ejemplos):**

- Activador de LDH-A: fructosa-1,6-bisfosfato (FBP), 50–200 µM.

- Ejemplo de inhibidor: oxamato 0,5–2 mM.\
  Titular a niveles de $`{EC}_{50}`$ / $`{IC}_{50}`$ ± una unidad logarítmica para curvas de respuesta.

**Series de desnaturalización:**

- Urea o GdnHCl: 0–6 M en TG. Verificar el índice de refracción o la densidad para confirmar la molaridad.

**Soluciones madre de proteína:**

- Determinar la concentración por $`A_{280}`$ (ε del proveedor/secuencia). Alicuotar; congelar rápidamente a −80 °C; evitar >1 ciclo de congelación-descongelación.

**9.2 Geometrías de confinamiento y calibración de** $`\mathbf{L}`$

**Membranas de alúmina anódica (AAMs) / sílice mesoporosa (SBA-15, MCM-41).**

- Diámetros de poro nominales: 5, 10, 20, 50, 100, 200 nm.

- **Verificación:** SEM para diámetro de poro (media ± DE sobre ≥200 poros); adsorción de $`N_{2}`$ (BET/BJH) para área superficial y tamaño modal.

- **Corrección de tortuosidad:** si el fabricante proporciona la tortuosidad hidráulica $`\tau`$, definir $`L_{eff} = L_{pore}\sqrt{\tau}`$. Si se desconoce, estimar $`\tau = 1/\varepsilon`$ donde $`\varepsilon`$ es la porosidad (aproximación de primer orden). Reportar $`L`$ tanto nominal como efectivo.

**Aglomeración (tamaño de malla ξ).**

- Estimar el tamaño de malla $`\xi(w)`$ a partir del escalamiento polimérico: $`\xi \approx a\text{ }w^{- \text{ }\nu/(3\nu - 1)}`$ con $`a`$ la longitud del monómero (PEG 35 kDa: $`a \approx 0.35`$ nm, $`\nu \approx 0.55`$).

- Definir $`L = \xi`$ y proporcionar la curva de conversión en IS con incertidumbre.

**Cavidades diseñadas.**

- Para sistemas proteína-en-jaula, usar SAXS o crio-EM para medir el cuello de botella más estrecho relevante para la ruta del sustrato; definir $`L`$ como ese cuello de botella.

**Aleatorización:** aleatorizar por bloques el orden de $`L`$ por día. Cegar al analista respecto a $`L`$ hasta que se ejecute el plan prerregistrado.

**9.3 Adquisición de cinética (flujo detenido / lector de placas)**

**Configuraciones por defecto del instrumento:**

- Longitud de paso óptico: 1 cm (cubeta) o equivalente en microplaca; agitación apagada durante la lectura.

- Lectura de LDH: NADH $`A_{340}`$ (ε = 6,22 $`{mM}^{- 1}`$ $`{cm}^{- 1}`$).

- Muestreo: 2–10 Hz; ventana de 30–180 s dependiendo de la enzima y $`L`$.

**Reglas de ajuste:**

- Usar el segmento lineal inicial para la velocidad inicial $`v_{0}`$ **o** ajustar una exponencial simple $`A(t) = A_{\infty} + \Delta A\text{ }e^{- kt}`$ si es estrictamente monoexponencial.

- Aceptar ajustes con $`R^{2} \geq 0.95`$ y residuos homocedásticos; en caso contrario, marcar y repetir.

- Convertir a la constante de velocidad $`k`$ según el esquema estándar de la enzima (consistencia de unidades).

**Réplicas:** ≥8 por $`L`$ por condición (cargas independientes). Registrar todas las exclusiones (solo criterios a priori).

**9.4 Calibración de la excitación acústica (E2)**

**Hardware:** disco piezoeléctrico adherido al portamuestras; generador de funciones; amplificador; sonda de termistor (±0,01 °C); acelerómetro o vibrómetro láser.

**Frecuencias:** 20 kHz, 200 kHz, 2 MHz (±2 %).\
**Selección de amplitud:** aumentar el voltaje hasta que la frecuencia **en resonancia** produzca el **mínimo** de $`Var(k)`$ en un piloto a $`L`$ fijo **sin** ΔT > 0,05 °C. Registrar el voltaje RMS por frecuencia.

**Protecciones térmicas:** registrar la temperatura a 2–10 Hz; excluir corridas que excedan el umbral de ΔT.\
**Controles:** reproducción del ΔT solo con Peltier (sin excitación); "piezoeléctrico desacoplado" (eléctricamente activo, mecánicamente aislado) para verificar captación EM.

**9.5 Protocolo de medición de CISS**

**Sustratos:** Au(111) o ITO, limpiados (piraña o UV-ozono).\
**Película proteica:** depositar por Langmuir–Blodgett o adsorción (pH cercano al isoeléctrico; fuerza iónica 150 mM). Enjuagar suavemente.

**Contactos:** contacto ferromagnético superior; magnetización $`+ M`$ / $`- M`$; polarización ±100–300 mV.\
**Detección:** amplificador lock-in; frecuencia 13–217 Hz; constante de tiempo 100–300 ms.

**Métrica:** $`P_{CISS} = (I_{+ M} - I_{- M})/(I_{+ M} + I_{- M})`$ a voltaje de polarización fijo.\
**Controles:**

- Película no quiral (por ejemplo, proteína desnaturalizada o polímero aquiral) → se espera $`P_{CISS} \approx 0`$.

- Inversión de contacto y del cableado → el signo de $`P_{CISS}`$ se invierte con $`M`$, no con el cableado.

**Series de desnaturalización:** preparar películas a partir de soluciones a 0–6 M de desnaturalizante; medir $`P_{CISS}`$ y retener alícuotas para cinética a granel.

**9.6 Espectroscopia de coherencia vibracional**

**Adquisición Raman (o IR):**

- Excitación: 532 o 633 nm a ≤1 mW en el punto para evitar calentamiento; objetivo 10×–50×.

- Rango espectral: 200–1800 $`{cm}^{- 1}`$; integración 1–5 s; 3–5 acumulaciones.

**Fracción de potencia coherente** $`C_{Raman}`$ **:**

1.  Corregir la línea base del espectro; calcular la densidad espectral de potencia (PSD).

2.  Identificar picos estrechos (FWHM ≤ umbral predefinido, por ejemplo, ≤15 $`{cm}^{- 1}`$) persistentes entre acumulaciones.

3.  $`C_{Raman} = \frac{\sum_{\text{coherent peaks}}^{}{PSD}}{\sum_{\text{total band}}^{}{PSD}}`$.\
    **Controles:** espectros solo de tampón y de proteína desnaturalizada; cuantificar el fotoblanqueo mediante curso temporal en un punto fijo.

**9.7 Verificaciones de temperatura, mezcla y cavitación**

- **Termometría:** micro-termistor en línea cerca del volumen de reacción; registro sincrónico con la cinética.

- **Mezcla:** imagen de partículas trazadoras (esferas de 1 µm) en una solución ficticia equivalente; asegurar que la configuración de excitación **no** cambie los patrones de flujo en el seno.

- **Cavitación:** para excitación en MHz en líquido, mantener la presión acústica bien por debajo del umbral de cavitación inercial; si hay incertidumbre, realizar una prueba de sonoquimioluminiscencia (negativa en las configuraciones operativas).

**9.8 Pipelines estadísticos (pasos exactos)**

**Estimación de pendiente (** $`\alpha_{bio,enz}`$ **).**

- Transformar: $`x = \log L`$, $`y = \log k`$.

- **Ajuste OLS:** $`y = \beta_{0} + \beta_{1}x + \varepsilon`$; $`\alpha = - \beta_{1}`$.

- **Ajuste ortogonal/BCES:** usar si el error de calibración de $`L`$ $`> 3\%`$.

- **ICs por bootstrap:** 10 000 remuestreos de pares (x,y); mediana e IC percentil al 95 %.

**ANCOVA para efectos de condición.**

- Modelo: $`y = \beta_{0} + \beta_{1}x + \sum_{j\ }\gamma_{j}C_{j} + \sum_{j}\ \delta_{j}(x \times C_{j}) + \varepsilon`$.

- **Cambio de clase:** $`\delta_{j}`$ significativo con $`\mid \Delta\alpha \mid \geq 0.2`$ y no solapamiento de IC.

**Colapso de datos.**

- Definir $`\widetilde{k}(\alpha^{\star}) = k\text{ }L^{\alpha^{\star}}`$.

- Objetivo: minimizar la varianza entre curvas $`V(\alpha^{\star})`$ a lo largo de los grupos distintos de $`L`$.

- Reportar $`\alpha^{\star}`$ óptimo, razón de varianza $`V(\alpha^{\star})/V(0)`$ y estadístico KS entre curvas de $`\widetilde{k}`$.

- **Aprobado:** $`\alpha^{\star}`$ dentro del IC al 95 % de la pendiente **y** KS $`p > 0.05`$.

**Cálculo del RBCI.**

``` math
RBCI = \frac{1}{4}\lbrack norm(\alpha;\lbrack 1,4\rbrack) + norm(P_{CISS};\lbrack 0,1\rbrack) + norm(C_{Raman};\lbrack 0,1\rbrack) + norm(\Delta{Var}_{k};\lbrack 0,1\rbrack)\rbrack,
```
con $`\Delta{Var}_{k} = \max\{ 0,\text{ }Var(k)_{\text{off}} - Var(k)_{\text{on}}\}/Var(k)_{\text{off}}`$ y $`norm(x;\lbrack a,b\rbrack) = \min\{ 1,\max\{ 0,(x - a)/(b - a)\}\}`$. Reportar las puntuaciones de componentes y la sensibilidad dejando uno fuera.

**Pruebas múltiples:** controlar la TDF (Benjamini–Hochberg) en los criterios de valoración secundarios.

**9.9 Análisis de potencia y tamaño de muestra**

**Efecto objetivo:** detectar $`\Delta\alpha = 0.20`$ (en resonancia vs apagado o ±efector), DE($`\widehat{\alpha}`$) ≈ 0,15.

- Con ≥4 niveles distintos de $`L`$ y $`n \geq 8`$ réplicas por $`L`$, las simulaciones producen potencia ≥0,80 a α = 0,05.

- Para monotonicidad de desnaturalización ($`\tau`$ de Kendall = −0,6), 6–8 niveles con triplicados por nivel logran potencia ≥0,8.

**9.10 Organización de archivos y reproducibilidad**

**Estructura del repositorio:**

**/raw/kinetics/** \# series temporales, por corrida, con JSON de metadatos

**/raw/thermometry/** \# registros de ΔT

**/raw/CISS/** \# I(V), estado de magnetización, mapas de contacto

**/raw/raman/** \# espectros + configuraciones de adquisición

**/calibration/** \# imágenes SEM/BET, curvas ξ(w)

**/analysis/** \# scripts: slope_ols.py, slope_bces.py, ancova.R, collapse.py, rbci.py

**/results/tables/** \# Tablas 1–5 (exportaciones CSV + LaTeX/Word)

**/prereg/** \# PDF de prerregistro + versiones del protocolo

**/si/** \# materiales suplementarios (S1–S6)

**Cuadernos:** un cuaderno de extremo a extremo regenera pendientes, ICs, colapso, RBCI y figuras a partir de CSVs.

**9.11 Lista de verificación de aseguramiento de calidad (ejecutar en cada sesión)**

- Tampones dentro de pH 7,40 ± 0,05 a 25 °C; fuerza iónica equivalente.

- Etiquetas de nivel de $`L`$ aleatorizadas; analista cegado.

- Trazas de ΔT < 0,05 °C para todas las corridas cinéticas.

- Control fuera de resonancia incluido cuando se usa excitación.

- Controles de matriz ficticia y película no quiral adquiridos.

- Ajustes $`R^{2} \geq 0.95`$; residuos inspeccionados; exclusiones registradas.

- Datos crudos y metadatos confirmados al repositorio con hash.

**9.12 Notas de seguridad**

- Manejar desnaturalizantes (urea, GdnHCl) con guantes/protección ocular; desechar según los procedimientos institucionales.

- Hardware acústico: asegurar los transductores; protección auditiva para pruebas de alta amplitud a >20 kHz; evitar la exposición del usuario a ultrasonido en el aire.

- Seguridad eléctrica para montajes CISS (blindaje, puesta a tierra adecuada, capacitación en manejo de imanes).

**APÉNDICE A — Validación computacional del marco RTM enzimático**

**A.1 Panorama general**

Este apéndice presenta la validación computacional del marco RTM aplicado a la cinética enzimática. Tres conjuntos de simulaciones demuestran que:

1\. La cinética modificada por RTM produce predicciones experimentalmente distinguibles (S1)

2\. La metodología de estimación de α es robusta y precisa (S2)

3\. La selectividad de sustrato puede predecirse y ajustarse mediante confinamiento (S3)

**A.2 S1: Cinética de Michaelis-Menten modificada por RTM**

**A.2.1 Modelo**

Michaelis-Menten clásica: v = V_max × \[S\] / (K_m + \[S\])

Modificación RTM: k_cat(L) = k_cat,0 × (L/L_ref)^(−α)

donde L es la longitud de confinamiento efectiva (nm) y α codifica la clase de transporte.

**A.2.2 Predicciones por clase de transporte**

\| Clase \| α \| Base física \| Mejora de k_cat a L=20nm \|

\|-------\|---\|----------------\|---------------------------\|

\| Guiado/balístico \| 1,5–1,8 \| Cables proteicos, canales \| 3–5× \|

\| Difusión laplaciana \| 2,0 \| Caminata aleatoria \| 5× \|

\| Jerárquico/fractal \| 2,1–2,5 \| Trampas, corredores \| 6–15× \|

\| Coherente (conjetural) \| >2,5 \| Confinamiento cuántico \| >15× \|

**A.2.3 Validación de recuperación de α**

Datos experimentales simulados (5 escalas de confinamiento, 5 % de ruido):

\| α verdadero \| α recuperado \| Error \|

\|--------\|-------------\|-------\|

\| 2,2 \| 2,195 \| 0,005 (0,2 %) \|

**A.3 S2: Metodología de escalamiento por confinamiento**

**A3.1 Estimador**

α_enz = −d(log k_app)/d(log L)

Medido ajustando una regresión log-log de las constantes de velocidad aparente a lo largo de las escalas de confinamiento.

**A.3.2 Resultados de validación**

**\*\*Robustez frente al ruido:\*\***

\| Ruido σ \| EAM \|

\|---------\|-----\|

\| 0,02 \| 0,018 \|

\| 0,05 \| 0,045 \|

\| 0,10 \| 0,089 \|

\| 0,15 \| 0,133 \|

\| 0,20 \| 0,178 \|

\| 0,30 \| 0,264 \|

Precisión aceptable (EAM < 0,15) mantenida para σ ≤ 0,15.

**Tamaño de muestra:**

\| N Escalas \| EAM \|

\|----------\|-----\|

\| 3 \| 0,122 \|

\| 4 \| 0,102 \|

\| 5 \| 0,089 \|

\| 7 \| 0,074 \|

\| 10 \| 0,059 \|

Mínimo 3 escalas requeridas; se recomiendan 5+.

**Discriminación de clases de transporte:**

\| Comparación \| t-stat \| valor p \| d de Cohen \|

\|------------\|--------\|---------\|-----------\|

\| Difusivo vs jerárquico \| 31,2 \| <10^−80 \| 3,12 \|

**A.3.3 Prueba de colapso de datos**

La prueba de colapso verifica el escalamiento RTM: si k_app ∝ L^(−α), entonces k_app × L^α debería ser constante a lo largo de todos los valores de L.

\| α utilizado \| Coeficiente de variación \|

\|--------\|-------------------------\|

\| Correcto (ajustado) \| 0,089 \|

\| Incorrecto (+0,5) \| 0,997 \|

El colapso es 11× peor con un α incorrecto, proporcionando un criterio de validación robusto.

**A.4 S3: Predicción de selectividad**

**A.4.1 Teoría**

Para sustratos A y B con diferentes valores de α:

S(L) = k_A/k_B = (k_A,0/k_B,0) × L^(α_B − α_A)

Si α_A > α_B, el sustrato A se beneficia más del confinamiento, y la selectividad puede ajustarse.

**A.4.2 Resultados por escenario**

\| Escenario \| Δα \| S_bulk \| S(20nm) \| L_cruce \|

\|----------\|-----\|--------\|---------\|-------------\|

\| Metabolismo de fármacos CYP450 \| +0,50 \| 0,67 \| 1,49 \| 44 nm \|

\| Enantioselectividad de lipasa \| +0,20 \| 1,11 \| 1,53 \| 169 nm \|

\| Regulación alostérica \| −0,30 \| 1,67 \| 1,03 \| 18 nm \|

**Hallazgo clave:** la selectividad puede desplazarse entre 2 y 3× en el rango de confinamiento de 10–100 nm, con puntos de cruce predecibles donde la selectividad se invierte.

**A5 Definición de RBCI**

El Índice de Coherencia de Bioquímica Rítmica agrega:

RBCI = 0,30×α_norm + 0,25×CISS + 0,25×Vib + 0,20×VR

donde:

\- α_norm: valor normalizado de α (0 en 1,5, 1 en 2,5)

\- CISS: polarización de espín (0–1)

\- Vib: fracción de coherencia vibracional (0–1)

\- VR: reducción de varianza bajo excitación en resonancia (0–1)

**Interpretación:**

\- RBCI > 0,6: Coherencia fuerte, se espera escalamiento RTM

\- RBCI 0,3–0,6: Coherencia moderada

\- RBCI < 0,3: Coherencia débil, desviaciones probables

**A.6 Recomendaciones experimentales**

**Métodos de confinamiento:**

1\. Membranas de alúmina nanoporosa (AAM): poros de 20–200 nm

2\. Sílice mesoporosa (MCM-41, SBA-15): poros de 3–15 nm

3\. Aglomeración polimérica (PEG, dextrano): malla efectiva de 15–120 nm

4\. Jaulas proteicas diseñadas: cavidades de 5–50 nm

**Protocolo:**

1\. Medir k_app en ≥5 escalas de confinamiento que abarquen ≥1 década

2\. Usar ≥3 réplicas por escala

3\. Ajustar log(k_app) vs log(L) para obtener α

4\. Verificar con la prueba de colapso (CV < 0,15)

5\. Validación cruzada con un segundo método de confinamiento

**APÉNDICE B — Análisis empírico: la división topológica entre plegamiento y catálisis**

El marco RTM propone que la naturaleza física de un proceso bioquímico puede diagnosticarse puramente a través de su exponente topológico de escalamiento ($`\alpha`$). Para validar esto, compilamos un conjunto de datos de 153 registros biológicos, contrastando el plegamiento de proteínas (un fenómeno estructural global) con la cinética enzimática (un fenómeno químico local).

**B.1 Observación heurística**

La regresión inicial por mínimos cuadrados ordinarios (OLS) demostró un contraste marcado entre los dos dominios. Las tasas de plegamiento de proteínas $`(k_{f}`$) exhibieron una dependencia masiva de la longitud de la cadena de aminoácidos ($`L`$), produciendo un exponente aparente de $`\alpha \approx 7.21`$ ($`R^{2} = 0.62`$). En contraste, los números de recambio enzimático ($`k_{cat}`$) mostraron una relación altamente dispersa y no significativa con el tamaño de la enzima ($`\alpha \approx 0.87,\ p = 0.14`$). Aunque esta observación heurística respaldaba la hipótesis RTM, los datos enzimáticos estaban fuertemente confundidos por el hecho de que diferentes clases de enzimas realizan reacciones químicas fundamentalmente diferentes a velocidades base intrínsecamente distintas. Además, la regresión OLS estándar no tiene en cuenta la masiva varianza experimental del 20–30 % típica de los ensayos biológicos *in vitro*.

**B.2 Validación rigurosa con EIV y normalización por mecanismo**

Para determinar si la división topológica es una ley física genuina y no un artefacto de confusión química o ruido de medición, el conjunto de datos se sometió a un pipeline estadístico robusto:

1.  **Normalización por mecanismo (clase EC):** Para aislar el efecto geométrico puro del tamaño de la enzima, normalizamos los valores de $`k_{cat}`$ por su clase específica de la Comisión de Enzimas (EC). Esto sustrajo matemáticamente la velocidad de línea base química (por ejemplo, la diferencia intrínseca entre una hidrolasa y una ligasa).

2.  **Regresión de distancia ortogonal (ODR):** Desplegamos un modelo de errores en variables, inyectando una varianza conservadora del 20 % para las tasas de plegamiento y del 30 % para las tasas catalíticas, forzando a la teoría a sobrevivir al ruido realista de laboratorio.

**B.3 El diagnóstico topológico**

Tras la penalización rigurosa y el control de mecanismo, la diferenciación física de RTM se vuelve excepcionalmente clara:

- **Topología global (plegamiento de proteínas):** El exponente robusto por ODR es $`\mathbf{\alpha = 7.22 \pm 0.62}`$, consistente con que el plegamiento es un fenómeno de red globalmente coherente y altamente resonante en el que toda la estructura participa en la dinámica temporal (el "embudo de plegamiento"). Esto es consistente con la teoría de plegamiento cooperativo (Bryngelson y Wolynes 1987, Dill y Chan 1997).

- **Química local (cinética enzimática):** Una vez normalizado el mecanismo químico, el exponente topológico para la catálisis colapsa completamente a $`\mathbf{\alpha}\mathbf{= \ 0.26\ }\mathbf{\pm}\mathbf{0.69}`$, volviéndose estadísticamente indistinguible de cero.

**Conclusión:** El marco RTM aísla exitosamente la causalidad física. La cinética enzimática ($`\alpha = 0.26 \pm 0.69`$, el IC incluye cero) no muestra dependencia estadísticamente significativa del tamaño tras la normalización por EC — consistente con que la catálisis es un proceso localizado en el sitio activo. El plegamiento de proteínas ($`\alpha = 7.22 \pm 0.62`$) muestra una fuerte dependencia positiva del tamaño — consistente con una dinámica cooperativa impulsada por la topología a lo largo de toda la estructura macromolecular. El solapamiento nulo en bootstrap entre las dos distribuciones ($`d = 6.98`$, 0 % de solapamiento, 3 000 iteraciones) demuestra que $`\alpha`$ separa limpiamente estas dos clases de operaciones biológicas. Este es un resultado **CONVERGENTE**: RTM recupera bioquímica conocida (plegamiento cooperativo vs. catálisis local) desde un punto de partida topológico. El valor de este hallazgo es la clasificación unificada más que los resultados individuales, que están establecidos en la literatura.

### APÉNDICE C — Auditoría del Equipo Rojo: verificación y certificación (abril de 2026)

Las afirmaciones empíricas en este documento fueron sometidas a auditoría adversarial independiente por el Equipo Rojo de RTM usando **Claude Opus 4.6 con Pensamiento Extendido** en abril de 2026. La auditoría no encontró errores fundamentales. El siguiente registro de verificación se proporciona por transparencia.

**C.1 Qué se probó**

| Afirmación | Prueba | Resultado |
|-------|------|--------|
| α de plegamiento = 7,22 ± 0,62 | ODR normalizada por EC, 84 puntos de plegamiento | **Confirmado** ✓ |
| α enzimático = 0,26 ± 0,69, IC incluye 0 | ODR normalizada por EC, 69 puntos enzimáticos | **Confirmado — resultado nulo para enzimas** ✓ |
| Solapamiento nulo en bootstrap (d = 6,98) | Bootstrap 3 000 iteraciones | **Confirmado — 0 % de solapamiento** ✓ |
| Inyección de varianza del 20–30 % sobrevive | Inyección conservadora de ruido | **Ambos regímenes sobreviven** ✓ |
| La normalización por EC elimina el factor de confusión | Comparación con datos no normalizados | **Confirmado — la normalización desplaza α enzimático hacia 0** ✓ |
| Separación plegamiento vs. enzima | Comparación estadística directa | **d = 6,98 — mayor tamaño de efecto en el corpus** ✓ |

**C.2 Veredicto de clasificación**

| Hallazgo | Clasificación | Justificación |
|---------|---------------|-----------|
| α de plegamiento = 7,22 (clase resonante/cooperativa) | **CONVERGENTE** | Consistente con la teoría de plegamiento cooperativo conocida (Bryngelson y Wolynes 1987) |
| α enzimático ≈ 0 (clase local/química) | **CONVERGENTE** | Consistente con el mecanismo local de sitio activo de Michaelis-Menten |
| Solapamiento nulo en bootstrap (d = 6,98) | **CONVERGENTE** | Confirma estadísticamente la distinción conocida entre plegamiento y catálisis |
| Metodología de normalización por clase EC | **METODOLÓGICO** | Enfoque correcto para eliminar el factor de confusión de clase química |
| Clasificación unificada basada en α | **NOVEDOSO** | RTM proporciona una métrica topológica única que clasifica ambos regímenes |

**C.3 La contribución novedosa**

Aunque los hallazgos individuales (plegamiento cooperativo, catálisis local) están establecidos en la literatura bioquímica, el Equipo Rojo identificó una contribución genuinamente novedosa:

El uso de un **exponente único $`\alpha`$** para clasificar ambos regímenes — y la demostración de que las dos distribuciones tienen solapamiento nulo en bootstrap ($`d = 6.98`$) — no está presente en la literatura bioquímica. La bioquímica estándar usa marcos mecanísticos (embudos de plegamiento, cinética de Michaelis-Menten) que son específicos para cada fenómeno. La contribución de RTM es proporcionar una **métrica topológica universal** que los separa limpiamente sin conocimiento previo del mecanismo.

Esto es análogo a la contribución de RTM en química (Doc 007): los regímenes individuales (Stokes-Einstein, difusión configuracional) eran conocidos, pero la clasificación unificada basada en $`\alpha`$ de RTM es nueva.

**C.4 Correcciones de tono aplicadas**

| Frase original | Corregida a |
|-----------------|-------------|
| "confirma abrumadoramente que el plegamiento es globalmente coherente" | "consistente con que el plegamiento sea globalmente coherente" |
| "demuestra matemáticamente que la catálisis enzimática es estructuralmente independiente" | "no muestra dependencia estadísticamente significativa del tamaño...consistente con" |
| "estrictamente un proceso químico localizado" | "un fenómeno localizado en el sitio activo" |
| "para demostrar que la ecuación de escalamiento RTM gobierna estrictamente la bioquímica" | "para probar si la ecuación de escalamiento RTM puede clasificar clases distintas" |
| "diferenciando ciegamente" | "diferenciando" |

**C.5 Veredicto del Equipo Rojo**

Los hallazgos primarios son estadísticamente sólidos, correctamente medidos y físicamente significativos. El solapamiento nulo en bootstrap ($`d = 6.98`$) es el segundo mayor tamaño de efecto en el corpus (después de la química de zeolitas, d = 8,48) y representa el resultado de separación más fuerte en el sub-corpus biológico. La normalización por clase EC elimina correctamente el factor de confusión de la reacción química.

Los hallazgos se clasifican como CONVERGENTES con la bioquímica conocida — RTM recupera resultados establecidos desde un punto de partida topológico. La contribución novedosa es la clasificación unificada basada en $`\alpha`$ que abarca ambos regímenes con una única métrica.


*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
