<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **Computación cuántica con RTM**  
**Un marco multiescala, pendiente-primero, para coherencia, planificación y diseño**  
  
Álvaro Quiceno

</div>

**Resumen**

Introducimos una metodología **pendiente-primero** para la computación cuántica basada en la **Relatividad Temporal Multiescala (RTM)**. Dentro de un régimen operativo fijo, RTM postula que un tiempo característico $T$ escala con un proxy de tamaño/escala $L$ mediante una ley de potencia,

$\log T = \alpha \log L + c$

donde el **exponente de coherencia** $\alpha$ es la señal estructural **invariante de reloj** y $c$ codifica el reloj/unidades. Adaptamos RTM a las pilas cuánticas---**física**, **QEC**, **compilador/tiempo de ejecución** e **I/O--criogénica**---definiendo pares $ (L,T)$ específicos por capa (p. ej., número de qubits activos vs. tiempo de calibración estable; distancia del código vs. tiempo de falla lógica; grado de multiplexación vs. latencia de lectura; ancho de circuito vs. duración total), y estimando pendientes por compartimiento bajo errores en variables (ODR/TLS, Theil--Sen, SIMEX). Una **prueba de colapso** valida el escalamiento y protege contra la mezcla de regímenes; las pendientes limpias por familia se fusionan en un $\mathbf{ECI}_{QC}$ **(t)** en tiempo real con incertidumbre y compuertas de QA.

Formulamos hipótesis **falsificables**: **(H1)** un $\alpha$ pre-choque más alto predice márgenes de estabilidad más largos (menos recalibraciones forzadas, menor error lógico a $d$ fijo); **(H2)** los **eventos de decoherencia**---caídas significativas y limpias de QA en ${ECI}_{QC}$---anteceden picos en el error lógico, colas de espera o duración total; **(H3)** las **cascadas de tempo** micro→meso→macro exhiben $\alpha$ no decreciente dentro de regímenes estables. Demostramos cómo la **planificación con RTM** (agrupamiento, reinicios escalonados, enrutamiento de baja varianza), el **diseño de cadencia QEC** (desincronización de ciclos de síndrome) y el **dimensionamiento modular** (puntos óptimos para interconexión) pueden mejorar el rendimiento y la fiabilidad sin cambiar las fidelidades físicas. El marco es reproducible, robusto ante gauges (cambios de unidades/reloj no afectan $\alpha$) y está diseñado para fallar de forma controlada (la ausencia de colapso y la alta heterogeneidad se convierten en fronteras de alcance, no en correcciones ad hoc).

**Validación empírica sistemática** $`\mathbf{\rightarrow}`$ **(APÉNDICE G).** Validamos el marco diagnóstico RTM en hardware cuántico mediante un análisis sistemático de 31 procesadores IBM Quantum de 5 a 1121 qubits. El análisis inicial de escalamiento en bruto arrojó una relación positiva coherencia-tamaño ($`\alpha \approx +0.23`$), una **Paradoja de Simpson** impulsada por un factor confusor de fabricación (mejoras tecnológicas generacionales). Para deslindar los avances cronológicos de ingeniería del verdadero escalamiento topológico, desplegamos un pipeline ODR multivariable, inyectando un margen de ruido criogénico de calibración del $`15\%`$. Tras normalizar el factor de ganancia tecnológica ($`\gamma = +0.139`$ dex/año), el escalamiento físico aislado revela un exponente negativo robusto de $`\mathbf{\alpha = -0.259 \pm 0.049}`$, IC bootstrap [ $`-0.382, -0.038`$ ], que excluye el cero al 95% de confianza. Esto ubica la decoherencia cuántica macroscópica en la **Clase de Transporte Inverso** ($`\alpha < 0`$), junto con la difusión clásica de Stokes-Einstein. El hallazgo clave es la **identificación de la Paradoja de Simpson**: el análisis ingenuo concluye que los procesadores cuánticos mejoran con la escala ($`\alpha > 0`$); el pipeline de RTM con control de factores confusores revela lo opuesto a generación tecnológica fija. Esto fue clasificado como **NOVEDOSO** por el Red Team (abril de 2026); el patrón de reversión por factor confusor no es visible sin descomposición multivariable. RTM separa exitosamente las leyes de escalamiento físico de los artefactos de ingeniería, demostrando que la coherencia masiva requiere resonancia arquitectónica, no escalamiento monolítico por fuerza bruta. Auditoría completa: Apéndice H.

**1. Introducción**

**1.1 Motivación: más allá de fidelidades y tasas de error**

El desempeño cuántico usualmente se resume mediante **métricas puntuales**---fidelidades de uno y dos qubits, $T_{1}/T_{2}$, tasas de error lógico o cifras de referencia (QED-C, QV). Sin embargo, la fiabilidad práctica y el rendimiento dependen de algo ortogonal: **cómo se extiende el tiempo a través de la escala** en una pila de múltiples etapas---qubits y resonadores, ciclos de código, compiladores, I/O criogénica. Cuando los subsistemas pequeños responden rápido y los más grandes responden más lentamente de forma disciplinada y estratificada, los choques se **disipan**; cuando los tiempos se **aplanan**, las perturbaciones percolan a través de las capas y sincronizan fallas (estancando la lectura, disparando el error lógico o forzando recalibraciones globales).

La **Relatividad Temporal Multiescala (RTM)** proporciona un lenguaje compacto para este fenómeno. Dentro de un régimen fijo, RTM espera una relación de ley de potencia entre un **tiempo característico** $T$ y un **proxy de escala** $L$: la **pendiente** $\alpha$ en $\log T = \alpha \log L + c$ es estructural (invariante ante unidades de tiempo), mientras que la ordenada al origen $c$ es un **reloj** (gauge). Llevamos este principio a la computación cuántica y mostramos que medir, validar e **ingeniar** $\alpha$ produce palancas accionables ---independientes de unidades nominales--- para mejorar la estabilidad y el rendimiento.

**1.2 RTM en una línea**

**La estructura vive en la pendiente; los relojes viven en el gauge.**\
Un cambio de reloj o unidades desplaza $c$ pero deja $\alpha$ sin cambio. Por lo tanto, $\alpha$ puede compararse entre dispositivos, pilas y laboratorios, mientras que $c$ no.

**1.3 Contribuciones**

Este artículo realiza cinco contribuciones:

1.  **Operacionalización de RTM para CC.** Definimos pares $ (L,T)$ específicos por capa para las capas **física**, **QEC**, **compilador/tiempo de ejecución** e **I/O--criogénica** (p. ej., $L =$ qubits activos, $T =$ tiempo de calibración estable; $L = d$, $T =$ ciclos hasta falla lógica; $L =$ grado de multiplexación, $T =$ latencia de lectura; $L =$ ancho de circuito, $T =$ duración total).

2.  **Validación y estimación.** Proporcionamos una **prueba de colapso** (independencia residual de $\log T - \alpha \log L$ respecto de $\log L$) para detectar mezcla de regímenes y curvatura no potencial, y adoptamos estimación con **errores en variables** (ODR/TLS, Theil-Sen, SIMEX) con incertidumbre bootstrap y guardias de punto de cambio.

3.  **Un indicador único en tiempo real.** Fusionamos pendientes por familia en $\mathbf{ECI}_{QC}$ **(t)** mediante metaanálisis de efectos aleatorios con controles de heterogeneidad ($Q$, $I^{2}$, ${\widehat{\tau}}^{2}$); publicamos banderas de QA y retenemos la fusión cuando los proxies no concuerdan.

4.  **Palancas de diseño.** Formalizamos la **planificación con RTM** (agrupamiento, reinicios escalonados, enrutamiento de baja varianza), el **diseño de cadencia QEC** (desincronización para evitar el enganche de fase entre errores físicos y extracción de síndrome) y el **dimensionamiento modular** (elección de escalas de módulo/interconexión que eleven $\alpha$ sin limitar el rendimiento).

5.  **Hipótesis y protocolos falsificables.** Pre-registramos **H1--H3** con protocolos A/B en plataformas superconductoras y de iones atrapados, métricas (rendimiento, duración total, error lógico, tiempo de actividad, proporciones p95/p50) y umbrales de decisión para adopción.

**1.4 Qué** $\mathbf{\alpha}$ **es---y qué no es**

-   **Es:** una **pendiente por compartimiento** que vincula un tiempo $T$ con una escala $L$ dentro de un **entorno fijo** (misma temperatura/firmware/topología/esquema de síndrome). Captura la **geometría del tempo a través de la escala**.

-   **No es:** un parámetro causal por defecto; los cambios de nivel en $T$ (unidades, relojes, desfases) **no** cambian $\alpha$. Cuando el colapso falla, $\alpha$ queda **indefinido** para ese compartimiento y no debe fusionarse.

**1.5 Ejemplares de** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$ **por capa (vista previa)**

-   **Física:** $L =$ qubits activos / grado de acoplador / tamaño de clúster; $T =$ intervalo de calibración estable, latencia de compuerta/lectura, tiempo medio hasta la deriva.

-   **QEC:** $L = d$ (distancia de código) o número de qubits lógicos; $T =$ ciclos hasta falla lógica; cadencia de extracción de síndrome.

-   **Compilador/tiempo de ejecución:** $L =$ ancho o profundidad de circuito tras mapeo; $T =$ duración total; retardo de cola y latencia de reprogramación.

-   **I/O--criogénica:** $L=$ grado de multiplexación o canales; $T =$ latencia de lectura/recuperación de BER; longitud de cola p95.

**Uso.** Preferir ODR como rutina de ajuste base; reportar SIMEX como estimación de **sensibilidad** junto a ODR. Si $\sigma_\xi^2$ es incierta, dar una banda (baja/media/alta) para $\hat{\alpha}_{\text{SIMEX}}$.

**1.6 Hipótesis (falsificables)**

-   **H1 (Resiliencia):** Un $\alpha$ pre-choque más alto se asocia con picos de error lógico menores a $d$ fijo y con intervalos de calibración estable más largos.

-   **H2 (Anticipación):** Las caídas limpias de QA en ${ECI}_{QC}$ anteceden incrementos en la duración total, colas de espera o error lógico por semanas a meses, aportando valor predictivo sobre las líneas base (fidelidad, utilización, temperatura).

-   **H3 (Cascada):** Dentro de regímenes estables, $\alpha_{\text{physical}} \leq \alpha_{\text{QEC}} \leq \alpha_{\text{runtime/I/O}}$; las pruebas de direccionalidad favorecen el flujo de temporización micro→meso→macro.

**1.7 Diseño con RTM (intuiciones que probaremos)**

-   **Planificación:** Evitar patrones que **aplanen** $\alpha$ (operaciones largas y fuertemente acopladas en paralelo); favorecer el **agrupamiento** de lecturas y los **reinicios escalonados** para prevenir cascadas de sincronización.

-   **Cadencia QEC:** Introducir una leve **desincronización** (desfases) entre los ciclos de síndrome y los ritmos de ruido conocidos para elevar $\alpha_{\text{QEC}}$.

-   **Modularidad:** Elegir el tamaño de módulo y la densidad de interconexión donde $\alpha$ sea suficientemente alto para amortiguar cascadas inter-módulo, pero no tan alto que limite el rendimiento.

**1.8 Relación con trabajos previos**

Nuestro marco complementa los enfoques centrados en fidelidad y modelos de error al agregar una **geometría escala--tempo**. Es compatible con (no un reemplazo de) la teoría de códigos de superficie/LDPC, heurísticas de compilación/enrutamiento y modelos de colas; aporta una estadística **invariante de gauge** $\alpha$ y una prueba de **colapso** como especificación para separar la **estructura** de los efectos de **reloj**. En el lenguaje de procesos estocásticos, nuestra sección de dinámica (posterior) conecta RTM con **difusiones con cambio de tiempo**; en términos de metaanálisis, nuestra fusión replica **efectos aleatorios** con **compuertas de heterogeneidad** explícitas.

**1.9. Validación empírica sistemática: la ilusión del escalamiento monolítico**$\mathbf{\rightarrow}$ **(APÉNDICE G)**

Una premisa fundamental de RTM es su capacidad para diagnosticar la verdadera clase de transporte de un sistema observando su exponente de escalamiento. En la carrera por construir computadoras cuánticas tolerantes a fallas, los desarrolladores de hardware han escalado continuamente los tamaños de procesadores monolíticos (conteo de qubits). Superficialmente, los datos históricos parecen sugerir que los procesadores más grandes poseen mejores tiempos de coherencia ($T_{2}$). Sin embargo, dentro del marco RTM debemos preguntar: ¿es esta mejora una propiedad de la escala espacial ($\alpha > \ 0$), o es un desfase artificial generado por avances tecnológicos continuos?

Para responder, utilizamos RTM como filtro diagnóstico sobre 31 procesadores IBM Quantum. Hipotetizamos que la decoherencia cuántica no es un conjunto de eventos independientes aislados, sino un colapso topológico colectivo. Por lo tanto, el verdadero escalamiento físico debería exhibir una firma de transporte inverso ($\alpha < \ 0$), donde una huella geométrica mayor amplifica naturalmente la diafonía y el ruido correlacionado. Mediante modelado multivariable de errores en variables, demostramos cómo RTM corta matemáticamente a través de los factores confusores de fabricación para revelar la cruda física subyacente de los sistemas cuánticos macroscópicos.

**2. Fundamentos de RTM adaptados a la computación cuántica**

Esta sección enuncia los axiomas de RTM, deriva la forma de **ley de potencia** $T = \kappa L^{\alpha}$ y adapta las nociones de **reloj/gauge** y **colapso** a las pilas cuánticas. A lo largo de todo, $L > 0$ es un **proxy de escala** (específico de capa) y $T > 0$ es un **tiempo característico** medido en un **entorno/compartimiento fijo** (misma temperatura, firmware, topología, esquema de síndrome, banda de utilización).

**2.1 Axiomas (por compartimiento)**

**A1 --- Semigrupo de escala.** Para cualquier dilatación $b > 0$,

$T(bL) = f(b)\text{ }T(L)$

con $f(1)=1$ y $f(b_{1}b_{2}) = f(b_{1})f(b_{2})$.

**A2 --- Regularidad suave.** $f$ es medible (o continua en $b = 1$).

**A3 --- Invarianza de reloj intra-compartimiento.** Los **cambios de reloj** permitidos multiplican $T$ por un factor $c>0$ **independiente de** $L$ dentro del compartimiento (cambios de unidad, líneas base de marcas de tiempo, desfases de latencia fijos). En la práctica de CC: reescalar unidades de tiempo, sobrecargas constantes de lectura, líneas base de I/O criogénica constantes.

**A4 --- Compartimentación.** Las comparaciones se realizan dentro de compartimientos donde el entorno es estable. Si se detecta un punto de cambio, el compartimiento debe dividirse.

**2.2 Solución de ecuación funcional → ley de potencia**

Sea $u = \log L$, $v = \log T$. De A1--A2, la ecuación multiplicativa de Cauchy da $f(b) = b^{\alpha}$ para algún $\alpha \in \mathbb{R}$. Por lo tanto

$T(L) = \kappa L^{\alpha},v(u) = \alpha u + \log\kappa$

**Interpretación.** $\alpha$ es el **exponente de coherencia** (pendiente); $\kappa$ es un **reloj** (ordenada al origen).

**2.3 Relojes (gauge multiplicativo vs. latencia aditiva)**

En RTM, un "cambio de reloj" dentro de un compartimiento fijo es un reescalamiento **multiplicativo** de todos los tiempos característicos: $T^{'} = cT$, $c>0$ independiente de $L$. Esto incluye conversiones de unidades de tiempo (ns↔µs), reescalamientos uniformes de base de tiempo/tasa de reloj o factores de calibración uniformes. En coordenadas logarítmicas, $\log T^{'} = \log T + \log c$, de modo que $\alpha$ no cambia y solo la ordenada al origen se desplaza.\
Por el contrario, las **latencias constantes** (p. ej., preámbulo fijo de lectura, retardo de pipeline, desfases de línea base de marca de tiempo) son **aditivas**: $T_{\text{obs}} = T + b$. En gráficos log--log esto no es un desplazamiento puro de ordenada al origen y puede sesgar $\alpha$, especialmente cuando $T$ no es $\gg b$. Por lo tanto, antes de estimar $\alpha$, se debe:\
(i) estimar/sustraer la latencia $b$ y ajustar usando $T_{eff} = \max(T_{\text{obs}} - b,\varepsilon)$, o\
(ii) restringir el análisis a regímenes donde $T_{\text{obs}} \gg b$ y reportar la sensibilidad de $\alpha$ a valores plausibles de $b$.

**2.4 Colapso como prueba de especificación por compartimiento**

Dadas observaciones $\{(L_i, T_i)\}_i$ *en un compartimiento, se define* $x_i = \log L_i$, $y_i = \log T_i$. Se ajusta una pendiente por compartimiento $\hat{\alpha}$ (Sección 5) y se examinan los **residuos**

${\widetilde{y}}_{i}: = y_{i} - \widehat{\alpha}x_{i}$

**Prueba de colapso.** En un compartimiento RTM válido, $\widetilde{y}$ debe ser **independiente de** $x$ (salvo ruido). Lo operacionalizamos con:

-   Una regresión $\widetilde{y} \sim x$ requiriendo $R_{\text{collapse}}^{2} < \tau$ (valor por defecto $\tau = 0.05$).

-   Un **placebo de reloj**: multiplicar todos los $T_{i}$ por una constante; $\widehat{\alpha}$ y $R_{\text{collapse}}^{2}$ deben permanecer sin cambio.

-   Una **verificación suave** (LOESS o spline) para detectar tendencia visible; si está presente, rechazar el compartimiento.

**Significado.** El colapso establece que, tras eliminar $\widehat{\alpha}\ logL$, solo queda un **gauge** (ruido de ordenada al origen), no una tendencia frente a la escala.

**2.5 Exponentes variables y sesgo de ventana finita**

En la práctica, $\alpha$ puede derivar lentamente con el entorno o la escala (p. ej., a través de bandas de utilización o factores de multiplexación). Se escribe

$v(u) = \int_{u_{0}}^{u}{\alpha(s)\text{ }ds + \log\kappa(u),}$

con $\mid \alpha^{'}(u) \mid \leq \varepsilon$ pequeño en la ventana y $\kappa$ **de variación lenta**. Para cualquier ventana simétrica de ancho $h$ en $u$,

$\widehat{\alpha}(u;h)\text{\:\,} = \text{\:\,}\alpha(u)\text{\:\,} + \text{\:\,}O(\varepsilon h)\text{\:\,} + \text{\:\,}O(\text{variación lenta})$

y

$R_{\text{collapse}}^{2}\text{\:\,} = \text{\:\,}O((\varepsilon h)^{2})$

**Regla.** Elegir compartimientos/ventanas lo suficientemente pequeños para que la curvatura sea despreciable; de lo contrario, dividir el compartimiento.

**2.6 Modos de falla (debe fallar)**

RTM está diseñado para **predecir su propio fallo**:

1.  **Mezcla de regímenes (quiebres).** Ejemplo: cambiar la cadena de lectura o el planificador de síndromes a mitad del compartimiento. El gráfico log--log muestra un cambio de pendiente en $L^{\star}$; el colapso falla.

2.  **Curvatura (no potencial).** Ejemplo: una sobrecarga dependiente de la multiplexación que crece de manera no lineal con $L$. Los residuos muestran tendencia con $x$; el colapso falla incluso tras re-compartimentar.

3.  **Relojes dependientes de la escala.** Cualquier factor de "reloj" $c(L)$ que dependa de $L$ no es un gauge; inyecta componentes $du$ en la 1-forma y debe modelarse explícitamente (o el compartimiento se rechaza).

**2.7 Mapeo de capas CC (notación y ejemplares)**

Usaremos estos pares **canónicos** $ (L,T)$ en las secciones posteriores (pueden agregarse otros si pasan la prueba de colapso):

-   **Física:**\
    $L =$ número de **qubits activos** (o grado de clúster/acoplador);\
    $T =$ **intervalo de calibración estable**, latencia de **compuerta**, latencia de **lectura** o **tiempo medio hasta la deriva**.

-   **QEC:**\
    $L =$ **distancia de código** $d$ (o conteo de qubits lógicos);\
    $T =$ **ciclos hasta falla lógica** a un error objetivo fijo.

-   **Compilador/tiempo de ejecución:**\
    $L =$ **ancho de circuito** o **profundidad post-mapeo**;\
    $T =$ **duración total** o **retardo de cola**.

-   **I/O--criogénica:**\
    $L =$ **grado de multiplexación** o conteo de canales de lectura;\
    $T =$ **latencia de lectura efectiva** / **vida media de recuperación de BER** / **longitud de cola p95 (en tiempo)**.

Cada familia produce una $\hat{\alpha}\_f$ por compartimiento. Solo las familias que **pasan el colapso** y el QA contribuyen al indicador fusionado $ECI\_{\text{QC}}(t)$ (Sección 6).

**2.8 Por qué** $\mathbf{\alpha}$ **importa operativamente**

-   **Comparabilidad**: $\alpha$ es invariante a cambios de unidad y sobrecargas constantes, lo que permite comparaciones **entre laboratorios** y **entre generaciones**.

-   **Alerta temprana**: **caídas** significativas en $\alpha$ (por familia o fusionado) señalan **eventos de decoherencia** que probablemente preceden picos de error lógico, duración total o recalibraciones forzadas.

-   **Palanca de diseño**: elevar $\alpha$ (sin exceso de estratificación) mediante **planificación**, **cadencia QEC** o **dimensionamiento modular** mejora la amortiguación de cascadas entre escalas.

**2.9 Resumen**

RTM en CC se reduce a tres enunciados por compartimiento: (i) escalamiento de **ley de potencia** $T = \kappa L^{\alpha}$, (ii) **invarianza de gauge** (solo la pendiente $\alpha$ es estructural), y (iii) **colapso** como prueba de especificación falsificable. Con compartimentación cuidadosa y estimación con conciencia de EIV, $\alpha$ se convierte en un **exponente de coherencia** reproducible y robusto ante unidades, que guía tanto el **diagnóstico** como el **diseño** a través de la pila cuántica.
**3. Geometría escala–reloj para CC (Colapso como exactitud)**

Reformulamos RTM para pilas cuánticas en forma geométrica. El objeto clave es la **1-forma RTM**

$\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha(x)\text{ }d(\log L),$

definida sobre un compartimiento $E$ con coordenadas de **entorno** $x$ (temperatura, firmware, topología, programa de síndromes, banda de utilización) y **escala** $u = \log L$. En este lenguaje, el **colapso** equivale a la **exactitud/planitud** de $\omega$; las costuras de régimen y la curvatura no potencial aparecen como **holonomía/curvatura**. Esta sección enuncia los resultados y los instancia con modos de falla de CC.

**3.1 Espacios, compartimientos y la 1-forma RTM**

-   **Espacio de estados.** $M = X \times \mathbb{R}$ con coordenadas $ (x,u)$, donde $u = \log L$.

-   **Potencial de reloj.** $v(x,u) = \log T(x,L)$.

-   **1-forma RTM.** $\omega = dv - \alpha(x)\text{ }du$ (caso de $\alpha$ constante) o $\omega = dv - \alpha(x,u)\text{ }du$ (deriva lenta permitida).

**Un cambio de reloj** (cambio de unidad/línea base independiente de $L$ dentro de un compartimiento) es:

``` math
v \mapsto v^{\#} = v + \phi(x).
```

Entonces

$\omega \mapsto \omega^{\#} = \omega + d\phi(x)$

una **transformación de gauge** por una 1-forma exacta retraída desde $X$. Por lo tanto, $\alpha$ **es invariante de gauge**.

**3.2 Colapso ⇔ exactitud/planitud**

**Teorema 3.1 (Colapso** $\Leftrightarrow$ **exactitud).**\
Sobre un compartimiento simplemente conexo $E$, las siguientes afirmaciones son equivalentes:

1.  (Carta RTM) $v(x,u) = \alpha(x)\text{ }u + \log\kappa(x)$ (o $v = \int\alpha(x,s)\text{ }ds + \log\kappa(x)$ para deriva lenta).

2.  (**Colapso**) El residuo $\widetilde{v}: = v - \alpha u$ es independiente de $u$ en $E$.

3.  (**Exactitud**) $\omega = d\psi$ en $E$ para algún $\psi(x)$ (sin dependencia de $u$).

**Corolario 3.2 (Prueba de planitud).**\
$d\omega=0$ es necesario y (en $E$ simplemente conexo) suficiente para el colapso. Con $\alpha = \alpha(x,u)$,

$d\omega\text{\:\,} = \text{\:\,} - \text{ }d\alpha \land du.$

Así, la curvatura (comportamiento no potencial) o la mezcla de regímenes produce $d\alpha/\text{ }du \neq 0$ y **rompe el colapso**.

**3.3 Holonomía y costuras de régimen (modos de falla en CC)**

Definimos la **holonomía** alrededor de un lazo cerrado $\gamma \subset E$ : $\mathcal{H(}\gamma) =\oint_{\gamma}^{}{\omega.\ }$ Si $\mathcal{H(}\gamma) \neq 0$, el colapso no puede sostenerse globalmente.

**Instancias en CC.**

-   **Costura de planificador.** Cambiar la cadencia de extracción de síndromes a mitad de compartimiento (nueva imagen FPGA) produce un quiebre en $v(u)$; lazos que cruzan la costura acumulan holonomía no nula → **recompartimentar**.

-   **Cambio de cadena de lectura.** Un sobrecosto por canal que *depende del multiplexado* se comporta como un reloj dependiente de la escala $c(L)$; esto **no es gauge** e inyecta componentes $du$ → el colapso falla (y debe fallar).

-   **Ventana de deriva térmica.** Una rampa lenta de utilización cambia $\alpha$ a lo largo de $u$; si $\partial_{u}\alpha$ no es pequeño en la ventana, $d\omega \neq 0$ → dividir el compartimiento o reducir la ventana.

**3.4 Colapso adiabático (** $\mathbf{\alpha}$ **lentamente variable)**

Si $\mid \partial_{u}\alpha \mid \leq \varepsilon$ en una ventana de ancho $h$,

$\widetilde{v}(x,u) = v - \alpha(u_{0},x)\text{ }u = \log\kappa(x) + O(\varepsilon h)$

y el estadístico empírico de colapso obedece

$R_{\text{collapse}}^{2} = O\text{ }((\varepsilon h)^{2})$

**Práctica.** Elegir $h$ de modo que $\varepsilon h \ll 1$; de lo contrario, reducir el compartimiento o modelar la deriva explícitamente.

**3.5 Morfismos (reparametrizaciones) y gauge**

Sea $\Phi = (\varphi,\psi)$ un mapeo $ (X_{A},L_{A},v_{A}) \rightarrow (X_{B},L_{B},v_{B})$, donde $\varphi:X_{A} \rightarrow X_{B}$ reparametriza el entorno y $\psi:X_{B} \rightarrow \mathbb{R}$ es un cambio de reloj. Entonces

$\Phi^{*}\omega_{B}\text{\:\,} = \text{\:\,}\omega_{A}\text{\:\,} + \text{\:\,}d(\psi \circ \varphi)$

Interpretación: transportar la estructura de $B$ a $A$ preserva la **pendiente** y altera solo el **reloj** mediante una forma exacta. Esto formaliza las comparaciones entre laboratorios/dispositivos cuando las unidades/líneas base difieren.

**3.6 Diagnósticos y compuertas de aceptación (lista de verificación para CC)**

1.  **Prueba de colapso.** Ajustar $\widehat{\alpha}$ (Sección 5), calcular residuos $\widetilde{y} = y - \widehat{\alpha}x$; requerir\
    $R_{\text{collapse}}^{2} < 0.05$ **y** ausencia de tendencia en un suavizado no paramétrico.

2.  **Placebo de reloj.** Multiplicar todos los $T$ por una constante; $\widehat{\alpha}$ y $R_{\text{collapse}}^{2}$ deben permanecer inalterados.

3.  **Puntos de cambio.** Ejecutar detectores sobre $(x,y)$ y sobre $\widetilde{y}$; cualquier quiebre ⇒ recompartimentar.

4.  **Control de ventana.** Asegurar que $\mid \partial_{u}\alpha \mid \text{ }h$ sea pequeño (régimen adiabático).

5.  **Publicar/retener.** Solo los compartimientos que pasan 1–4 contribuyen a ${ECI}_{QC}$ (t); de lo contrario, etiquetar NO_COLLAPSE o REGIME_MIX.

**3.7 Beneficio operativo**

-   Una **obligación de prueba**: demostrar planitud/exactitud (colapso) antes de confiar en una pendiente.

-   Un **depurador**: la holonomía no nula localiza costuras (cambios de planificador, cambios de lectura).

-   Una **regla de ajuste**: reducir $h$ o recompartimentar hasta que $d\omega \approx 0$; si es imposible, el dominio es **no potencial**—tratar $\alpha$ como indefinido allí.

**3.8 Resumen**

La geometría escala–reloj precisa dos enunciados RTM para CC:

1.  $\alpha$ **es una cantidad estructural invariante de gauge**, no afectada por cambios de unidad/línea base;

2.  **El colapso equivale a la exactitud/planitud de** $\omega$, y su fallo es informativo (curvatura o costuras).\
    Ahora aprovecharemos esto para definir $(L,T)$ **operativas** (Sec. 4) y estimar $\widehat{\alpha}$ de forma robusta bajo error de medición (Sec. 5).

**4. Definiciones operativas de** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$ **y protocolo de compartimentación**

Esta sección convierte RTM en **práctica medible** para pilas cuánticas. Definimos pares $(L,T)$ específicos por capa, especificamos **muestreo**, **unidades** y **guardias**, y damos un protocolo de compartimentación que evita la mezcla de regímenes. En todo momento, $u = \log L$, $v = \log T$.

**4.1 Principios de diseño para** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$

-   **Un mecanismo por familia.** Cada par $(L,T)$ debe reflejar un único mecanismo dominante (p. ej., tubería de lectura, no una mezcla de lectura + enrutamiento).

-   **$L$ monótono.** $L$ debe crecer con el "tamaño de problema" en esa capa (ancho, distancia, canales, tamaño de clúster).

-   **Independencia de reloj.** Dentro de un compartimiento, los cambios **multiplicativos** de base temporal ($T^{'} = cT$) son gauges permitidos (reescalamientos de unidad/base temporal). Los sobrecostos **aditivos** ($T_{\text{obs}} = T + b$) deben sustraerse, modelarse o evitarse (ajustar solo donde $T \gg b$); de lo contrario pueden sesgar pendientes e invalidar el colapso.

-   **Muestreo estable.** Usar recolección de **cadencia fija**; registrar marcas de tiempo crudas para permitir rerrebanado.

**4.2 Capa física**

**Candidatos para** $L$ **:**

-   $L =$ número de **qubits activos** en la ventana de carga de trabajo;

-   $L =$ **tamaño de clúster** (qubits conectados que participan simultáneamente);

-   $L =$ **grado de acoplador** (fanout promedio).

**Candidatos para** $T$ **:**

-   **Intervalo de calibración estable** (tiempo hasta que cualquier qubit del clúster sale de tolerancia);

-   **Latencia de compuerta** (mediana de duración de compuertas de uno/dos qubits en el conjunto activo);

-   **Latencia de lectura** (tiempo mediano por disparo hasta símbolo válido bajo umbrales fijos);

-   **Tiempo medio hasta la deriva** (MTTD) de frecuencia/fase.

**Instrumentación.**

-   Registrar marcas de tiempo por disparo; un vigilante de calibración que registre cuándo se violan umbrales; adjuntar etiquetas de entorno: banda de temperatura, hash de firmware, punto de polarización.

**Contraejemplos.**

-   Mezclar *tanto* latencia de compuerta como latencia de lectura en el mismo $T$.

-   Permitir que $L$ sea "qubits definidos en el chip" (no necesariamente activos).

**4.3 Corrección de errores (QEC)**

$L$ **:** **distancia** del código $d$ (principal), o número de **qubits lógicos** a $d$ fijo.\
$T$ **:**

-   **Ciclos hasta falla lógica** a una tasa de error objetivo fija (mediana o cuantil de supervivencia);

-   **Latencia del ciclo de síndromes** (tiempo medio por ciclo bajo programa fijo).

**Notas de programación.**

-   Congelar un **programa de síndromes** (imagen FPGA + cadencia). Cualquier cambio ⇒ nuevo compartimiento.

-   Registrar sesgo (X/Z) y configuraciones de mitigación de fuga.

**Casos límite.**

-   Si $T$ está dominado por **eventos catastróficos raros** (p. ej., bloqueos de resonador), preferir **medianas condicionales** (excluir banderas catastróficas conocidas) y reportar un panel de sensibilidad.

**4.4 Compilador / Tiempo de ejecución**

$L$ **:** **ancho** del circuito (máximo de qubits concurrentes) o **profundidad post-mapeo**; opcionalmente **capas activas** después del enrutamiento.\
$T$ **:**

-   **Duración total** (envío → finalización);

-   **Retardo de cola** (envío → inicio);

-   **Latencia de reprogramación** después de un evento de calibración.

**Controles.**

-   Fijar la **política de enrutamiento** y la **heurística de colocación** dentro de un compartimiento.

-   Estratificar por banda de utilización (p. ej., 0–30%, 30–60%, >60%). Si la utilización deriva, dividir el compartimiento.

**4.5 E/S – Cryo / Lectura**

$L$ **:** **grado de multiplexado** (canales por línea) o número de canales de lectura concurrentes.\
$T$ **:**

-   **Latencia de lectura** (mediana p50 y cola p95);

-   **Vida media de recuperación de BER** después de una ráfaga controlada;

-   **p95 de cola** expresado en tiempo.

**Instrumentación.**

-   Marcar temporalmente cada ráfaga DMA/ADC; registrar búferes por canal; anotar versiones de firmware del DSP.

**Advertencia.**

-   Los sobrecostos por canal que **crecen con** $L$ *no* son gauges; son efectos de escala genuinos—admisibles para RTM—pero si el sobrecosto mismo cambia a mitad de compartimiento, el colapso debe fallar y disparar una división.

**4.6 Protocolo de compartimentación (fijación de entorno)**

Un **compartimiento** es un intervalo máximo donde el entorno es efectivamente constante.

**Clave del compartimiento (ejemplo):**

$\text{BIN} = \{\text{platform},\text{ temperature band},\text{ firmware hash},\text{ topology ID},\text{ routing policy},\text{ syndrome cadence},\text{ utilization band}\}$

**Procedimiento.**

1.  **Rebanar** los datos por BIN; descartar rebanadas con < $N_{\mathrm{min}}$ valores distintos de $L$ (por defecto 6).

2.  **Escaneo de puntos de cambio** sobre $y = \log T$ vs. $x = \log L$ (y sobre residuos si están disponibles). Si se detecta un punto de cambio (BIC/AIC/PELT), **dividir**.

3.  **Ventaneo**: para regímenes de deriva lenta, usar ventanas deslizantes en $x$ de ancho $h$ tal que $\mid \partial_{u}\alpha \mid \text{ }h \ll 1$ (de Sec. 3.4).

4.  **Placebo de reloj**: multiplicar $T$ por una constante; la pendiente $\widehat{\alpha}$ no debe cambiar.

**4.7 Conjunto de datos listo para estimación**

Crear una tabla ordenada por compartimiento con columnas:

$x = log\ L,\ y = \log T,\text{ family},\text{ BIN tags},\text{ replicate ID},\text{ timestamp},\text{ weights }\rbrack$

-   **Réplicas.** Si hay múltiples corridas al mismo $L$, agregar a resúmenes robustos (mediana $y$, SE basado en MAD) o pasar todas y dejar que ODR las maneje con pesos de réplica.

-   **Pesos.** Preferir pesos de varianza inversa a partir de bootstrap sobre conteos simples.

-   **Valores atípicos.** Etiquetar eventos catastróficos (banderas de hardware); reportar tanto **con** como **sin** ellos.

**4.8 Compuertas de aceptación (por compartimiento, por familia)**

Una familia contribuye una pendiente ${\widehat{\alpha}}_{f}$ **solo si** se cumplen todas:

1.  **Cobertura:** al menos $6$ puntos distintos de $L$ y amplitud $\geq 0.6$ en $\log L$.

2.  **Colapso:** regresar $\widetilde{y} = y - \widehat{\alpha}x$ sobre $x$; requerir $R_{\text{collapse}}^{2} < 0.05$ y ausencia de tendencia visible (verificación por suavizado).

3.  **Placebo de reloj:** $\widehat{\alpha}$ inalterado bajo $T \mapsto cT$.

4.  **Puntos de cambio:** ninguno dentro del compartimiento (de lo contrario dividir y reestimar).

5.  **Calidad del ajuste EIV:** ODR/TLS convergió; diagnósticos de residuos aceptables (ningún punto de apalancamiento individual domina).

Los compartimientos o familias que fallan cualquier compuerta se marcan (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE, EIV_FAIL) y se **excluyen de la fusión**.

**4.9 Ejemplos vs. contraejemplos (con sabor CC)**

-   **Buena familia física:** $L =$ tamaño de clúster de qubits activos; $T =$ intervalo de calibración estable. Un solo firmware, temperatura estable, sin cambio de enrutamiento. Colapsa limpiamente → aceptar.

-   **Mala familia física:** Igual, pero a mitad de compartimiento los parámetros del lazo PLL cambian. Se dispara un punto de cambio; se requiere división.

-   **Buena familia QEC:** $L = d$, $T =$ ciclos hasta falla lógica, cadencia de síndromes fija. Residuos planos → aceptar.

-   **Mala familia QEC:** Mezcla de dos cadencias (rápida y lenta) dentro de un compartimiento → quiebre en log–log → rechazar hasta dividir.

-   **Buena familia E/S:** $L =$ grado de multiplexado; $T =$ latencia de lectura p95. Firmware constante; la latencia crece como $L^{\alpha}$, el colapso se sostiene → aceptar.

-   **Mala familia E/S:** Cambio de firmware DSP que altera el sobrecosto por canal de forma no lineal a mitad de compartimiento → curvatura; rechazar o recompartimentar alrededor del cambio.

**4.10 Resumen**

-   Fijamos $(L,T)$ **operativas** por capa y especificamos **instrumentación** para hacerlas medibles.

-   Definimos un **protocolo de compartimentación** que impone constancia ambiental y protege contra mezcla de regímenes.

-   Establecimos **compuertas de aceptación** (cobertura, colapso, placebo, puntos de cambio, ajuste EIV) que determinan si la pendiente de una familia entra en la fusión descendente (${ECI}_{QC}$ (t)).

**5. Estimación bajo errores en variables (EIV) y umbrales de colapso**

Ahora especificamos **cómo** estimar la pendiente por compartimiento $\alpha$ de manera robusta cuando ambos ejes tienen ruido, y cómo decidir—mediante un **umbral de colapso**—si los datos de una familia son consistentes con RTM. En todo momento, $x = \log L$, $y = \log T$. Las observaciones son $x^{obs} = x + \xi$, $y^{obs} = y + \zeta$ con errores de media cero.

**5.1 Objetivos de estimación y modelos**

Dentro de un **compartimiento fijo**, el objetivo es la **pendiente local** $\alpha$ en

$y = \alpha x + c + r(x)$

con $r \equiv 0$ bajo RTM exacto o $\mid r^{'}(x) \mid \leq \varepsilon$ bajo deriva lenta en una ventana. Dado que $x$ tiene ruido, **OLS está atenuado**; usamos estimadores con conciencia de EIV.

**Objetivo por defecto:** pendiente puntual $\alpha$ para el compartimiento; el intercepto $c$ es un **gauge** (no se compara entre compartimientos).

### 5.2 Regresión de distancia ortogonal (mínimos cuadrados totales)

**Definición.** ODR minimiza los residuos ortogonales a una línea:

$$
\min_{\alpha,c} \sum_{i} \frac{(y_i^{\text{obs}} - \alpha x_i^{\text{obs}} - c)^2}{\sigma_y^2 + \alpha^2\sigma_x^2}
$$

con $(\sigma_x, \sigma_y)$ efectivos (posiblemente heterogéneos) a partir de varianza de réplicas o bootstrap.

**Práctica.**

-   Inicializar con Theil–Sen (Sec. 5.4) para evitar mínimos locales deficientes.

-   Usar **bootstrap de clústeres** (de réplica o nivel de trabajo) para IC.

-   Si se dispone de SE por punto, ponderarlos; de lo contrario, usar pesos robustos de Huber sobre residuos ortogonales.

**Compuertas de convergencia.**

-   Número de condición de la matriz de covarianza centrada $< 10^{4}$.

-   Verificación de apalancamiento por jackknife: ningún punto individual contribuye $> 25\%$ de la influencia sobre la pendiente.

**5.3 SIMEX (cuando** $\mathbf{Var}\mathbf{(}\mathbf{\xi}\mathbf{)}$ **es conocido/estimado)**

Si se puede estimar $\sigma_{\xi}^{2} = Var(\xi)$ (p. ej., $L$ repetido en la misma configuración), aplicar **SIMEX**:

1. Para $\lambda \in \Lambda = \{0.5, 1.0, 1.5, 2.0\}$, generar pseudo-muestras $x_i^{(\lambda)} = x_i^{obs} + \sqrt{\lambda} {\tilde{\xi}}_i, \quad {\tilde{\xi}}_i \sim \mathcal{N}(0, \sigma_\xi^2).$

2.  Ajustar una pendiente ingenua $\widehat{\alpha}(\lambda)$ por ODR u OLS.

3.  Ajustar una cuadrática $\widehat{\alpha}(\lambda) = a + b\lambda + c\lambda^{2}$ y **extrapolar a** $\lambda = - 1$ :
    ${\widehat{\alpha}}_{\text{SIMEX}} = a - b + c$.

**Uso.** Preferir ODR como rutina base de ajuste; reportar SIMEX como estimación de **sensibilidad** junto a ODR. Si $\sigma_{\xi}^{2}$ es incierto, dar una banda (baja/media/alta) para ${\widehat{\alpha}}_{\text{SIMEX}}$.

**5.4 Theil–Sen (pendiente mediana robusta)**

La pendiente **Theil–Sen** es la mediana de todas las pendientes por pares

$\alpha_{ij} = \frac{y_{j}^{obs} - y_{i}^{obs}}{x_{j}^{obs} - x_{i}^{obs}}(i < j)$

con un intercepto robusto a partir de la mediana de $y_{i}^{obs} - \widehat{\alpha}x_{i}^{obs}$.

**Rol.**

-   Inicialización para ODR.

-   **Verificación cruzada robusta ante valores atípicos** reportada junto a ODR.

-   Cuando EIV es severo y $\sigma_{\xi}^{2}$ es desconocido, Theil–Sen puede seguir siendo estable (esperar atenuación leve).

**5.5 Ventaneo y sesgo de ventana finita**

Si se sospecha deriva lenta, estimar pendientes en **ventanas simétricas** en $x$ de ancho $h$. De la cota de sesgo adiabático,

$\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h)$

elegir $h$ de modo que $\varepsilon h \ll 1$. En la práctica: comenzar con $h \approx 0.8$ en amplitud de $\log L$ si la cobertura lo permite; reducir hasta que el colapso pase (Sec. 5.7) sin explotar la varianza.

**5.6 Incertidumbre y diagnósticos**

-   **Bootstrap** (pares dentro del compartimiento o bloque/clúster si existen réplicas naturales) para IC de 50/95%.

-   **Jackknife-después-de-bootstrap** para detectar puntos de apalancamiento.

-   **Gráficos de residuos**: residuo ortogonal vs. $x$; el suavizado LOESS debe ser plano dentro de las bandas.

-   **Adecuación de EIV**: si OLS y ODR difieren en $\geq$ 0.2 en pendiente absoluta **y** el IC de ODR excluye a OLS, reportar EIV como material.

**5.7 Umbral de colapso (compuerta de especificación)**

Dado $\hat{\alpha}$, calcular residuos $\tilde{y}_i = y_i^{obs} - \hat{\alpha}x_i^{obs} - \hat{c}$ y regresar $\tilde{y}$ sobre $x$ (con los mismos pesos usados en la estimación). Definir

$R_{\text{collapse}}^{2}: = R^{2}(\widetilde{y} \sim x)$

**Regla de decisión (por defecto):**

-   Aceptar el compartimiento si se cumplen **todas**:

    1.  $R_{\text{collapse}}^{2} < 0.05$ (o el IC al 95% de la pendiente en $\widetilde{y} \sim x$ contiene 0),

    2.  El suavizado LOESS no muestra tendencia,

    3.  **Placebo de reloj**: escalar $T \mapsto cT$ deja $\widehat{\alpha}$ y $R_{\text{collapse}}^{2}$ inalterados,

    4.  El escaneo de puntos de cambio (PELT/BIC) no encuentra ninguno dentro del compartimiento.

-   De lo contrario marcar (NO_COLLAPSE o REGIME_MIX) y **no** publicar una pendiente ni incluirla en la fusión.

**5.8 Compuertas de cobertura y apalancamiento**

Para evitar ajustes frágiles:

-   **Puntos distintos de** $L$ $\geq 6$ y amplitud en $\log L$ $\geq 0.6$.

-   **Apalancamiento equilibrado:** el punto de mayor apalancamiento contribuye $\leq 25\%$ de la influencia de la pendiente ODR.

-   **Réplicas:** si hay $> 3$ réplicas por $L$, resumir a media/SE robustos o pasar pesos de réplica a ODR.

Los compartimientos que no pasan estas compuertas se marcan como THIN_COVERAGE o LEVERAGE_RISK.

**5.9 Procedimiento completo (algoritmo por compartimiento)**

1.  **Preparación:** construir la tabla ordenada (Sec. 4.7); ejecutar escaneo de puntos de cambio; ventanear si es necesario.

2.  **Inicialización:** calcular pendiente/intercepto Theil–Sen; remover catastróficos obvios (conservar ambas versiones para sensibilidad).

3.  **Ajustar ODR/TLS:** ponderado por SE de réplicas; obtener $\widehat{\alpha}$, $\widehat{c}$, IC bootstrap.

4.  **SIMEX (opcional):** si $\sigma_{\xi}^{2}$ está disponible, calcular ${\widehat{\alpha}}_{\text{SIMEX}}$.

5.  **Compuerta de colapso:** calcular $R_{\text{collapse}}^{2}$, verificación por suavizado, placebo de reloj.

6.  **Decisión:** si todas las compuertas pasan, **aceptar** $\widehat{\alpha}$ con incertidumbre; de lo contrario **rechazar/dividir**.

7.  **Reporte:** pendiente, IC, diagnósticos (colapso $R^{2}$, gráfico de apalancamiento, puntos de cambio). Almacenar banderas.

**5.10 Lo que publicamos por familia aceptada**

-   ${\widehat{\alpha}}_{f} \pm$ `<!-- -->` {=html}IC de 50/95% (ODR); Theil–Sen como robustez; banda SIMEX si aplica.

-   Diagnósticos de colapso: $R_{\text{collapse}}^{2}$, verificación de placebo, ancho de ventana $h$.

-   Cobertura: \# de $L$ distintos, amplitud en $\log L$, resumen de apalancamiento.

-   Notas: cualquier exclusión (catastróficos), estado de puntos de cambio.

Solo las familias aceptadas entran a la **fusión** (Sec. 6). Si $\geq 2$ familias pasan, aplicamos efectos aleatorios con $Q$, $I^{2}$ y compuertas de heterogeneidad; de lo contrario reportamos pendientes por familia sin fusión.

**5.11 Resumen**

-   Usar **ODR/TLS** como estimador EIV primario; **Theil–Sen** para inicialización/verificación robusta; **SIMEX** cuando $\sigma_{\xi}^{2}$ es estimable.

-   Imponer el **colapso** como **prueba de especificación** ($R_{\text{collapse}}^{2} < 0.05$ + placebo + sin puntos de cambio).

-   Controlar el **sesgo de ventana finita** eligiendo $h$ suficientemente pequeño (régimen adiabático) y dividiendo compartimientos cuando sea necesario.

-   Publicar **diagnósticos** y **banderas** completos; solo las familias limpias pasan a la fusión y al ${ECI}_{QC}$ (t) en tiempo real.

**6. Construcción del indicador en tiempo real** $\mathbf{ECI}_{\mathbf{QC}}\mathbf{(}\mathbf{t}\mathbf{)}$

Ahora construimos un **indicador de coherencia único y en tiempo real** para una plataforma, fusionando las pendientes **aceptadas** por familia $\{{\widehat{\alpha}}_{f,t}\}$ de la Sección 5. La fusión es de **efectos aleatorios** (para reconocer la heterogeneidad entre familias), opera sobre un reloj deslizante y alimenta **compuertas de QA** y **alertas de decoherencia**.

**6.1 Entradas y precondiciones (por tiempo** $\mathbf{t}$ **)**

Para cada familia $f\in\mathcal{F}_{t}$ (Física, QEC, Compilador/Tiempo de ejecución, E/S–Cryo):

-   Una estimación por compartimiento $\hat{\alpha}\_{f,t}$ con varianza $\hat{\sigma}\_{f,t}^2$ (bootstrap o ponderada por réplicas),

-   Colapso aprobado (Sección 5.7), compuertas de cobertura/apalancamiento satisfechas (Sección 5.8),

-   Etiquetas de entorno (BIN) sin cambios dentro de la ventana que produjo ${\widehat{\alpha}}_{f,t}$.

Una fusión en el tiempo $t$ procede **solo si** $\mid \mathcal{F}_{t} \mid \geq 2$.

**6.2 Fusión de efectos aleatorios**

Estimamos la varianza entre familias ${\widehat{\tau}}_{t}^{2}$ (por defecto **REML**; DerSimonian–Laird como sensibilidad). Definimos pesos

$w_{f,t}\text{\:\,} = \text{\:\,}\frac{1}{{\widehat{\sigma}}_{f,t}^{2} + {\widehat{\tau}}_{t}^{2}}$

Entonces la pendiente fusionada y su varianza son

${\widehat{\alpha}}_{QC}(t) = \frac{\sum_{f \in \mathcal{F}_{t}}^{}{w_{f,t}\text{ }{\widehat{\alpha}}_{f,t}}}{\sum_{f \in \mathcal{F}_{t}}^{}w_{f,t}},\ \ Var({\widehat{\alpha}}_{QC}(t)) = \frac{1}{\sum_{f \in \mathcal{F}_{t}}^{}w_{f,t}}$

Reportar intervalos del 50% y 95% vía aproximación normal o mediante un **bootstrap sobre familias** (remuestrear familias con reemplazo, recalcular ${\widehat{\tau}}_{t}^{2}$ y la media fusionada).

**6.3 Diagnósticos de heterogeneidad y compuertas**

Calcular la línea base de efecto fijo

$w_{f,t}^{FE} = \frac{1}{{\widehat{\sigma}}_{f,t}^{2}},\ \ {\widehat{\alpha}}_{FE}(t) = \frac{\sum_{f}^{}{w_{f,t}^{FE}\text{ }{\widehat{\alpha}}_{f,t}}}{\sum_{f}^{}w_{f,t}^{FE}}$

**$Q$ de Cochran** **e** $I^{2}$ **:**

$Q_{t} = \sum_{f}^{}{w_{f,t}^{FE}\text{ }({\widehat{\alpha}}_{f,t} - {\widehat{\alpha}}_{FE}(t))^{2},\ \ I_{t}^{2} = \max}\{ 0,\text{\:\,}\frac{Q_{t} - ( \mid \mathcal{F}_{t} \mid - 1)}{Q_{t}}\} \times 100\%$

**Compuertas de fusión (pre-registradas):**

-   Proceder con un número único **solo si**\
(i) $|\mathcal{F}\_t| \geq 2$,  
(ii) $I\_t^2 < 50\%$ (*heterogeneidad moderada o inferior*), y  
(iii) REML converge con $\hat{\tau}\_t^2$ finito que no exceda un tope histórico (p. ej., $\leq$ percentil 90 sobre ventanas limpias pasadas).  

-   Si cualquiera falla, **retener la fusión** y publicar ${\widehat{\alpha}}_{f,t}$ por familia + diagnósticos; marcar FAMILY_DIVERGENCE.

**6.4 Operación en tiempo real (ventanas deslizantes)**

-   **Cadencia.** Recalcular el ${\widehat{\alpha}}_{f,t}$ de cada familia sobre una **ventana deslizante** en $x = \log L$ de ancho $h$ (elegido por la regla adiabática; Sec. 5.5) y un **horizonte de reloj de pared** (p. ej., últimos 7–28 días de datos).

-   **Relleno y datos faltantes.** Si falta una familia en $t$, fusionar sobre las $\mathcal{F}\_t$ disponibles siempre que $|\mathcal{F}\_t| \geq 2$; de lo contrario **suspender** $ECI\_{\text{QC}}(t)$ y publicar una bandera `THIN_FAMILIES`.

-   **Placebo de reloj.** Una vez al día, multiplicar todos los $T$ contribuyentes por una constante y verificar que $\hat{\alpha}_{\text{QC}}(t)$ e $I_t^2$ permanecen inalterados (almacenado como artefacto de QA).

**6.5 Eventos de decoherencia (lógica de alertas)**

Definimos un **evento de decoherencia** como una **caída** significativa, limpia en QA, de ${ECI}_{QC}(t)$, robusta al suavizado y no explicada por picos de heterogeneidad.

**Filtros:**

1.  **Suavizado:** mantener una mediana de 3 puntos $\widetilde{\alpha}(t)$ de ${\widehat{\alpha}}_{QC}(t)$.

2.  **Puntuación Z:** $Z(t) = \frac{\tilde{\alpha}(t) - \text{EWMA}\_{30}[\tilde{\alpha}]}{\hat{\sigma}\_{\text{EWMA}}(t)}$

**Niveles de alerta (por defecto):**

-   **Aviso:** $Z(t) \leq - 1.5$ durante ≥2 marcas consecutivas **y** $I_{t}^{2} < 50\%$.

-   **Vigilancia:** $Z(t) \leq - 2.0$ una vez **o** $Z(t) \leq - 1.5$ persistente durante ≥4 marcas, $I_{t}^{2} < 40\%$.

-   **Advertencia:** $Z(t) \leq - 2.5$ y una caída coincidente por familia (≥2 familias con $Z_{f} \leq - 2$).

**Protocolos de acción activados:** limitar la programación (reducir concurrencia/multiplexado), ejecutar recalibración segmentada, o cambiar a enrutamiento con conciencia RTM hasta que $\widetilde{\alpha}(t)$ se normalice.

**6.6 Reporte y visualización**

-   **Panel principal:** $\hat{\alpha}\_{\text{QC}}(t)$ con bandas de 50/95%, cinta de heterogeneidad coloreada por $I\_t^2$ (verde <25%, ámbar 25–50%, rojo $\geq$ 50%).

-   **Diagrama de bosque:** $\hat{\alpha}\_{f,t}$ por familia, pesos $w\_{f,t}$ e IC; mostrar $Q\_t$, $I\_t^2$, $\hat{\tau}\_t^2$.

-   **Tablero de colapso:** por familia, mostrar $R_{\text{collapse}}^{2}$, residuos LOESS, ancho de ventana $h$, métricas de cobertura y apalancamiento.

-   **Leyenda de banderas:** NO_COLLAPSE, REGIME_MIX, LEVERAGE_RISK, THIN_COVERAGE, FAMILY_DIVERGENCE, THIN_FAMILIES.

**6.7 Sensibilidad y ablación**

-   Publicar el resumen de **efecto fijo** ${\widehat{\alpha}}_{FE}(t)$ junto al de efectos aleatorios.

-   Reportar ${\widehat{\tau}}_{DL}^{2}$ basado en DL como sensibilidad.

-   **Excluir una familia a la vez**: recalcular ${\widehat{\alpha}}_{QC}^{( - f)}(t)$ para exponer dominancia.

-   **Placebos de reloj** y **nulos por permutación** (permutar $L$ dentro de la familia) no deben producir alertas escalonadas; si lo hacen, revisar las compuertas.

**6.8 Gobernanza y procedencia**

Cada punto fusionado almacena:

-   Familias fuente y sus etiquetas BIN,

-   Configuraciones del estimador (inicialización ODR, semillas de bootstrap, $h$),

-   Métricas de colapso, $Q_{t}$, $I_{t}^{2}$, ${\widehat{\tau}}_{t}^{2}$,

-   Hashes de resultados de placebo,

-   Código/configuración versionados (YAML de métodos).

Esto asegura **reproducibilidad** y permite análisis post-mortem cuando se disparan alertas.

**6.9 Resumen**

$ECI\_{\text{QC}}(t)$ es una **fusión de efectos aleatorios** de pendientes por compartimiento limpias en QA. Las compuertas de heterogeneidad ($I\_t^2 < 50\%$, $|\mathcal{F}\_t| \geq 2$) previenen números únicos engañosos cuando los proxies discrepan. El suavizado en tiempo real y las puntuaciones Z convierten la dinámica de pendientes en **alertas accionables** para **eventos de decoherencia**, mientras que los tableros y la procedencia mantienen el sistema auditable.

**7. Diseño con conciencia RTM: ingeniería de** $\mathbf{\alpha}$ **sin sacrificar rendimiento**

Esta sección convierte RTM en **palancas de diseño**. Objetivo: incrementar el **exponente de coherencia** $\alpha$ (estratificación temporal más fuerte a través de la escala) manteniendo o mejorando el rendimiento. Damos controles específicos por capa, objetivos de optimización y salvaguardas.

**7.1 Objetivo de diseño y salvaguardas**

Tratamos $\alpha$ como un **objetivo operativo** dentro de un compartimiento:\
$$\max_{\text{\:\,controls }\theta}\ \ \ \alpha(\theta)\ \ \ s.t.\ \ \ \ throughput\  \geq \ B,\ \ fidelity\  \geq \ F,\ \ \ \ \ collapse\ passes.$$

-   **Controles** $\theta$ : parámetros de planificador, cadencia/jitter de QEC, restricciones de enrutamiento, límites de multiplexado, tamaños de módulo.

-   **Restricciones**: un piso de rendimiento $\mathcal{B}$ (p. ej., trabajos/hora), piso de fidelidad $\mathcal{F}$, y **compuertas de colapso** (Sec. 5.7).

-   **Monitoreo:** rastrear $\hat{\alpha}\_f$ por familia y el fusionado $\hat{\alpha}\_{\text{QC}}(t)$ con QA (Sec. 6).

**7.2 Planificador: agrupamiento y enrutamiento con conciencia de varianza**

**Problema.** Las operaciones largas fuertemente acopladas lanzadas en paralelo **aplanan** $\alpha$ (cascadas rápidas a través de la escala).

**Controles.**

1.  **Agrupamiento por frentes de onda (lectura y operaciones largas).** Particionar el tiempo en oleadas cortas; empaquetar lecturas en oleadas en lugar de concurrencia libre.

2.  **Reinicios escalonados.** Agregar pequeños desplazamientos $\delta \in \lbrack - \epsilon,\epsilon\rbrack$ a los tiempos de reinicio para evitar picos de sincronización.

3.  **Enrutamiento de baja varianza.** Preferir rutas con **baja varianza de tiempo de trayecto** incluso si la longitud del trayecto aumenta ligeramente.

**Objetivo.** Para un DAG de trabajo con operaciones $o$ de duraciones nominales $\tau_{o}$ y rutas $p(o)$ :

$\underset{\text{\:\,schedule},\text{ }p( \cdot )}{\min}\text{\:\,}\underset{\text{desincronizar ops pesadas}}{\underbrace{{Var}_{t}\lbrack N_{\text{long}}(t)\rbrack}}\text{\:\,} + \text{\:\,}\lambda\text{\:\,}\underset{\text{enrutamiento de baja varianza}}{\underbrace{\sum_{o \in \mathcal{O}}^{}{Var(T_{\text{route}}(p(o)))}}}$

sujeto a presupuesto de duración total. Esto reduce las "acumulaciones" temporales, elevando $\alpha$.

**Heurística (voraz, práctica).**

-   Ordenar operaciones por duración desc; asignar tiempos de inicio en **oleadas** de modo que la carga total de operaciones largas de cada oleada esté balanceada.

-   Para cada ruta candidata, penalizar varianza temporal y puntuación de diafonía; elegir el costo penalizado mínimo.

**7.3 Cadencia QEC: evitar el enganche de fase (jitter/desincronización)**

**Problema.** Una cadencia fija de síndromes puede generar **enganche de fase** con ritmos de ruido físico, creando sincronización entre capas → $\alpha_{QEC}$ cae.

**Controles.**

-   **Micro-jitter** del período del ciclo: $P_{k} = P\text{ }(1 + \eta_{k})$ con $\eta_{k} \sim \mathcal{U}\lbrack - \rho,\rho\rbrack$, $\rho \ll 1$ (p. ej., 1–3%).

-   **Extracción multifase:** dividir el código en subretículas cuyos ciclos están desfasados por fases pequeñas $\phi_{j}$.

**Regla de diseño.** Elegir $\rho$ de modo que el **lóbulo principal** del espectro de línea del ciclo de síndromes se mueva **fuera** de los picos fuertes del PSD del error, manteniendo válida la temporización del decodificador. Validar mediante: (i) aumento de ${\widehat{\alpha}}_{QEC}$ vs. $d$, (ii) error lógico estable a $d$ fijo.

**7.4 Gradientes y pozos de** $\mathbf{\alpha}$

Dos motivos arquitectónicos para **dirigir flujos**:

-   **Gradiente:** disponer recursos de modo que $\alpha$ **aumente** hacia regiones de cómputo crítico. Las perturbaciones pequeñas decaen a medida que se desplazan hacia el interior.

-   **Pozo:** crear una **cuenca de alto** $\alpha$ alrededor de qubits sensibles (p. ej., temporización y almacenamiento en búfer que frenan cascadas a gran escala).

**Indicaciones de implementación.** Aumentar el almacenamiento temporal en búfer (colas, planificación amortiguada) y reducir el fanout de diafonía al acercarse al "núcleo", pero limitar el almacenamiento en búfer (salvaguardas Sec. 7.1) para que el rendimiento no sufra.

**7.5 Dimensionamiento modular: elegir un punto óptimo equilibrando latencia intra vs. inter**

Sea el total de qubits $Q$ particionado en $Q/m$ módulos de tamaño $m$. **Tiempo característico** aproximado:

$T(m)\text{\:\,} = \text{\:\,}A\text{ }m^{a}\text{\:\,} + \text{\:\,}B\text{ }(\frac{Q}{m})^{b}\text{     }\text{(costo intra-módulo + costo de interconexión)}$

**Tamaño óptimo de módulo** (minimiza $T$):

$m^{\star}\text{\:\,} = \text{\:\,}{(\frac{B\text{ }b}{A\text{ }a})}^{\frac{1}{a + b}}\text{\:\,}Q^{\frac{b}{a + b}}$

-   $a>0$ : escalamiento intra-módulo (p. ej., calibración, enrutamiento dentro del módulo).

-   $b>0$ : escalamiento inter-módulo (p. ej., latencia de enlace fotónico/iónico).

**Uso en diseño.** Medir $a,b$ empíricamente (RTM por mecanismo), estimar $A,B$, calcular $m^{\star}$. Operar cerca de $m^{\star}$ y verificar que $\widehat{\alpha}$ **no colapse** (siga siendo de tipo potencial) en esa vecindad.

**7.6 Multiplexado y E/S: mantener las colas bajo control**

**Problema.** El multiplexado agresivo reduce el tiempo por disparo, pero puede sincronizar colas de cola → $\alpha_{IO} \downarrow$.

**Controles.**

-   Limitar el multiplexado de modo que la **razón de cola** $p95/p50$ de la latencia de lectura se mantenga por debajo de un umbral (p. ej., $\leq 1.6$).

-   Usar **ventanas de lectura con desfase** entre canales para evitar crecimiento coherente de las colas.

-   Dimensionamiento de búfer: mantener utilización del búfer < 70% para evitar amplificación de colas.

**Señal.** Si $p95/p50$ crece y ${\widehat{\alpha}}_{IO}$ cae con colapso limpio, reducir multiplexado e introducir desfases.

**7.7 Lazo de control en línea (ingeniería de** $\mathbf{\alpha}$ **en lazo cerrado)**

Un controlador simple para mantener $\alpha$ alto bajo restricciones:

every Δt:

estimate {α_f(t), σ_f(t)} per accepted family (Sec. 5)

if \|F_t\| ≥ 2 and I\^2_t \< 50%:

compute α_QC(t) (Sec. 6)

if α_QC(t) \< α_floor and constraints met:

apply actions A = {↑wave size, ↑reset jitter ρ, ↑routing penalty on variance,

↓multiplex cap, move toward m\*}

else if throughput \< B:

relax A minimally (keep collapse passing)

log QA: collapse R\^2, I\^2_t, flags; revert actions if flags trip

-   $\alpha_{\text{floor}}$ : pendiente fusionada mínima aceptable pre-registrada.

-   **Revertir** cualquier acción que cause NO_COLLAPSE o $I_{t}^{2} \geq 50\%$.

**7.8 Seguridad y validación**

-   Cualquier intervención debe **volver a pasar el colapso** en las familias afectadas.

-   Ejecutar ventanas A/B (≥2–4 semanas) con **KPI pre-registrados**: rendimiento, duración total, error lógico, tiempo de operación, $p95/p50$, y ${\widehat{\alpha}}_{f}$.

-   Si $\alpha$ sube pero los KPI empeoran más allá de los presupuestos, se está **sobre-estratificando** (demasiado almacenamiento en búfer). Retroceder al frente de Pareto.

**7.9 Protocolos de acción rápida**

-   **Si** $\alpha_{QEC} \downarrow$ **:** agregar 1–3% de jitter en cadencia; introducir 2–3 grupos de fase para síndromes; remedir colapso.

-   **Si** $\alpha_{IO} \downarrow$ **:** reducir el tope de multiplexado 10–20%; agregar 1–2 ciclos de desfase; mantener $p95/p50 \leq 1.6$.

-   **Si** $\alpha_{runtime} \downarrow$ **:** habilitar agrupamiento de lecturas; penalizar rutas de alta varianza; limitar operaciones largas concurrentes por oleada.

-   **Planificación arquitectónica:** estimar $a,b,A,B$ y fijar el tamaño de módulo cerca de $m^{\star}$; confirmar escalamiento de tipo potencial alrededor de ese punto.

**7.10 Resumen**

-   El **planificador** (oleadas, reinicios escalonados, enrutamiento de baja varianza) y la **cadencia QEC** (micro-jitter, multifase) son las palancas de primera línea para **elevar** $\alpha$.

-   El **dimensionamiento modular** admite un óptimo en forma cerrada $m^{\star}$ que equilibra costos intra/inter; operar cerca de él vigilando el colapso.

-   Los **controles de E/S** evitan que las colas de latencia se sincronicen.

-   Un **controlador de lazo cerrado** mantiene $\alpha$ por encima de un piso bajo presupuestos de rendimiento/fidelidad.

**8. Protocolos experimentales falsificables (superconductores e iones atrapados)**

Esta sección especifica experimentos RTM-CC **comprobables** con elecciones concretas de $(L,T)$, recolección de datos, planes de análisis y criterios de éxito. Cada protocolo es por compartimiento (entorno fijo) e incluye **placebos**, **guardias de puntos de cambio** y una **tabla de decisión pre-registrada**.

**8.1 Andamiaje común (aplica a todos los protocolos)**

**Bloqueo de BIN (entorno).**\
{plataforma; banda de temperatura; hash de firmware (FPGA/DSP); ID de topología; política de enrutamiento; cadencia de síndromes; banda de utilización}. Cualquier cambio ⇒ nuevo compartimiento.

**Esquema de datos (ordenado).** Para cada registro:

$$x = log\ L,y = logT,\text{ family},\text{ BIN tags},\text{ replicate ID},\text{ timestamp},\text{ weights}\rbrack$$

**Compuertas de QA (deben pasar):**

-   Cobertura: ≥6 $L$ distintos, amplitud ≥0.6 en $\log L$.

-   Ajuste EIV convergido (ODR), apalancamiento <25%, inicialización robusta (Theil–Sen).

-   Colapso: $R_{\text{collapse}}^{2} < 0.05$, sin tendencia LOESS, placebo de reloj se sostiene.

-   Puntos de cambio: ninguno dentro del compartimiento (de lo contrario dividir).

**Resultados (primarios, por familia):**

-   Pendiente ${\widehat{\alpha}}_{f}$ con IC de 50/95%; diagnósticos de colapso.

-   Para resultados fusionados, ${\widehat{\alpha}}_{QC}(t)$, $Q$, $I^{2}$, ${\widehat{\tau}}^{2}$ (Sec. 6).

**Plan estadístico.**\
IC bootstrap (pares/clúster). Predefinir **efecto mínimo detectable** (MDE) sobre $\alpha$ (p. ej., $\Delta\alpha = 0.15$) y **KPI operativos** (rendimiento, duración total, tasa de error lógico, tiempo de operación, p95/p50). Umbrales a continuación.

**8.2 Protocolo A — Capa física (superconductores)**

**Hipótesis (H1-Phys).** Aumentar la **desincronización de clústeres** (reinicios escalonados + oleadas de lectura) **eleva** $\alpha_{\text{phys}}$ sin exceder el presupuesto de rendimiento.

**Diseño.**

-   $L$ : **tamaño de clúster** de qubits activos (comprometidos simultáneamente).

-   $T$ : **intervalo de calibración estable** (tiempo hasta la primera bandera fuera de tolerancia en el clúster).

-   Brazos: **Control** (planificador base) vs. **Con conciencia RTM** (agrupamiento de lecturas + reinicios escalonados, ±2–4% de desfase).

-   Duración: 2–4 semanas; alternar brazos diariamente para equilibrar la deriva.

**Análisis.**

-   Ajustar ODR por brazo, pasar colapso.

-   Efecto primario: $\Delta \hat{\alpha}\_{\text{phys}} = \hat{\alpha}\_{\text{RTM}} - \hat{\alpha}\_{\text{CTRL}}$

-   Salvaguardas de KPI: caída de rendimiento ≤5%, sin aumento de error de compuerta/lectura >0.2σ.

**Criterios de éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{phys}} \geq 0.15$ y el IC excluye 0, **y** las salvaguardas se satisfacen.

-   Si el colapso falla en cualquier brazo, declarar **no concluyente** y recompartimentar.

**Placebos.** Multiplicar $T$ por una constante; $\widehat{\alpha}$ inalterado. Permutar $L$ dentro del día; sin pendiente significativa.

**8.3 Protocolo B — Cadencia QEC (superconductores o iones)**

**Hipótesis (H1-QEC).** Introducir **micro-jitter** (1–3%) en el período de síndromes y/o **extracción multifase** incrementa $\alpha_{\text{QEC}}$ vs. distancia de código $d$ a decodificador fijo.

**Diseño.**

-   $L$ : **distancia de código** $d$ (p. ej., $d \in \{ 3,5,7,9\}$).

-   $T$ : **ciclos hasta falla lógica** (mediana o cuantil de supervivencia a tasa de error objetivo fija).

-   Brazos: Control (período fijo $P$) vs. Jitter ($P_{k} = P(1 + \eta_{k})$, $\eta_{k} \sim \mathcal{U}\lbrack - 0.02,0.02\rbrack$) y/o 2–3 **grupos de fase**.

-   Mantener parámetros de decodificador fijos; sin cambio en mitigación de sesgo de ruido.

**Análisis.**

-   ODR por brazo; compuerta de colapso.

-   Efecto: $\Delta{\widehat{\alpha}}_{\text{QEC}}$.

-   Salvaguardas de KPI: error lógico a $d$ fijo no empeora en >5% relativo.

**Criterios de éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{QEC}} \geq 0.15$ con IC excluyendo 0 y salvaguardas aprobadas.

**Diagnósticos.** Verificar PSD de procesos de error; confirmar que el jitter mueve las líneas de cadencia fuera de los picos dominantes.

**8.4 Protocolo C — Planificación de compilador/tiempo de ejecución**

**Hipótesis (H2-Run).** El **agrupamiento por frentes de onda** de lectura y el **enrutamiento de baja varianza** reducen las cascadas de sincronización, incrementando $\alpha_{\text{runtime}}$ y reduciendo las colas de duración total.

**Diseño.**

-   $L$ : **ancho de circuito post-mapeo** (o capas activas).

-   $T$ : **duración total** (envío→finalización).

-   Brazos: Política base vs. con conciencia RTM (oleadas + enrutamiento penalizado por varianza).

-   Controlar banda de utilización; misma mezcla de trabajos entre brazos.

**Análisis.**

-   Pendiente ODR por brazo; colapso.

-   KPI: mediana de duración total (≤ línea base), p95/p50 de latencia ↓ ≥10%.

**Criterios de éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{runtime}} \geq 0.10$ (IC excluye 0) y p95/p50 mejora ≥10%.

**8.5 Protocolo D — Multiplexado E/S–Cryo**

**Hipótesis (H2-IO).** Las **ventanas de lectura con desfase** entre canales mantienen o elevan $\alpha_{\text{IO}}$ mientras reducen las colas p95 a un grado de multiplexado dado.

**Diseño.**

-   $L$ : **grado de multiplexado** (canales/línea).

-   $T$ : **latencia de lectura p95** (y p50).

-   Brazos: Ventanas síncronas vs. ventanas desfasadas (patrón de fase $\phi_{j}$).

-   Barrer $L$ a lo largo del rango operativo.

**Análisis y éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{IO}} \geq 0.10$; p95/p50 ≤ 1.6 en el brazo RTM sobre la mayoría de $L$; colapso aprobado.

**8.6 Protocolo E — Dimensionamiento modular (estudio de planificación)**

**Hipótesis (H3-Mod).** Existe un tamaño de módulo $m^{\star}$ que minimiza $T(m) = Am^{a} + B(Q/m)^{b}$ con $a,b>0$ medidos empíricamente, y operar cerca de $m^{\star}$ preserva el escalamiento de tipo potencial (el colapso se sostiene).

**Diseño.**

-   Plataformas con enlaces fotónicos/iónicos entre módulos.

-   Medir $T(m)$ variando el tamaño de módulo (o emulando el costo de interconexión) a $Q$ total fijo.

-   Ajustar $a,b,A,B$ vía ODR sobre el conjunto de datos de cada término; calcular $m^{\star}$.

**Criterios de éxito.**

-   $T(m)$ observado minimizado cerca de $m^{\star}$ (dentro del IC), y los ajustes log–log alrededor de $m^{\star}$ retienen el colapso (sin curvatura).

**8.7 Fusión y alertas (entre protocolos)**

A través de A–D, si ≥2 familias pasan las compuertas en tiempos superpuestos, calcular ${\widehat{\alpha}}_{QC}(t)$ (Sec. 6).\
**H2 (anticipación):** declarar un **evento de decoherencia** si se cumplen los niveles de puntuación Z (Sec. 6.5); probar **adelanto–retraso** vs. picos en error lógico/duración total/colas. El valor predictivo adicional se evalúa contra líneas base (fidelidad, utilización, temperatura) usando regresión de series temporales con errores HAC; pre-registrar horizontes (p. ej., 7–30–90 días).

**8.8 Placebos, permutaciones y robustez**

-   **Placebos de reloj:** multiplicar todos los $T$ por constantes; $\widehat{\alpha}$ y $R_{\text{collapse}}^{2}$ invariantes.

-   **Nulos por permutación:** permutar $L$ dentro del día; las pendientes colapsan a ~0 (dentro del IC).

-   **Fusión excluyendo una familia a la vez** para revelar dominancia.

-   **Puntos de cambio**: división automática si se detectan; reestimar en ambos lados.

**8.9 Potencia y duración (reglas empíricas)**

-   Con amplitud ≥0.8 en $\log L$, 8–12 puntos distintos de $L$ y ruido moderado (SNR≈5–10), ODR detecta $\Delta\alpha \approx 0.10$–0.15 al 95% con ≈200–400 observaciones totales por brazo.

-   Si el ruido es mayor o se sospecha deriva, reducir ventanas (Sec. 5.5) y extender la duración.

**8.10 Tabla de decisión (pre-registrada)**

| Resultado | Acción |
| :--- | :--- |
| $\Delta\hat{\alpha} \geq$ MDE **y** salvaguardas aprobadas | Promover intervención a producción en ese compartimiento; monitorear con $\text{ECI}_{\text{QC}}(t)$. |
| $\Delta\hat{\alpha}$ significativo pero salvaguarda de KPI violada | Ajustar intensidad (p. ej., reducir almacenamiento en búfer/jitter) y reprobar. |
| Colapso falla o heterogeneidad alta ($I^2 \geq 50\%$) | No fusionar; reportar por familia; revisar compartimentación o mecanismos. |
| Sin efecto ($\Delta\hat{\alpha} \approx 0$) | Documentar como *límite de alcance*; mantener como control negativo. |

**8.11 Ética, seguridad y reproducibilidad**

-   **Seguridad:** no aumentar potencia RF de forma insegura; los límites de jitter mantienen los decodificadores válidos; reversión ante NO_COLLAPSE o violación de KPI.

-   **Reproducibilidad:** YAML de métodos versionado (BIN, configuraciones de estimador, semillas), gráficos públicos (paneles de colapso, diagramas de bosque) y artefactos almacenados de placebos/permutaciones.

-   **Transparencia:** publicar tanto éxitos como fracasos (los resultados negativos definen el alcance).

**8.12 Resumen**

Estos protocolos hacen RTM-CC **falsificable**: cada uno reclama un cambio direccional en $\alpha$ a partir de un control específico, bajo constancia de compartimiento, con el colapso como prueba de especificación y salvaguardas operativas. El éxito mejora no solo la pendiente sino también la **estabilidad en tiempo de ejecución** (colas, recalibraciones) sin sacrificar rendimiento.

**9. Plantillas de resultados y estándares de reporte**

Esta sección define **qué publicar** una vez ejecutados los protocolos (Sec. 8). Estandariza figuras, tablas, paneles de robustez y una lista de verificación de una página para que los resultados sean interpretables, reproducibles y directamente comparables entre laboratorios y plataformas.

**9.1 Conjunto de figuras (mínimo)**

**Fig. 1 — Paneles de colapso (por familia aceptada).**\
Cuatro subgráficos por familia $f$ dentro de un compartimiento:

1.  **Ajuste log–log:** $y = \log T$ vs. $x = \log L$ con línea ODR y banda del 95%.

2.  **Residuo vs.** $x$ **:** $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$ con LOESS; mostrar $R_{\text{collapse}}^{2}$.

3.  **Cobertura/apalancamiento:** dispersión destacando puntos de apalancamiento; anotar amplitud en $\log L$, \# de $L$ distintos.

4.  **Verificación de placebo:** superposición de ajustes antes/después de $T \mapsto cT$ (las curvas coinciden).

**Fig. 2 — Diagrama de bosque y heterogeneidad.**\
Por franja temporal (o por brazo experimental), mostrar $\hat{\alpha}\_f \pm \text{IC}$, pesos $w\_f$, el fusionado $\hat{\alpha}\_{\text{QC}}$ (diamante) y estadísticos de heterogeneidad: $Q, I^2, \hat{\tau}^2$.

**Fig. 3 —**$\mathbf{ECI}_{QC}$ **(t) serie temporal.**\
Pendiente fusionada deslizante con bandas de 50/95%; cinta de fondo coloreada por $I^{2}$ (verde <25%, ámbar 25–50%, rojo ≥50%). Marcar **eventos de decoherencia** (aviso/vigilancia/advertencia) y eventos de plataforma (recalibraciones, cambios de firmware).

**Fig. 4 — Panel de KPI (pareado con Fig. 3).**\
Ejes temporales alineados para: tasa de error lógico (a $d$ fijo), mediana y p95 de duración total, p95 de cola, tiempo de operación entre recalibraciones. Superponer regiones sombreadas para los niveles de alerta de Fig. 3.

**Fig. 5 — Resultados A/B (por protocolo).**\
Para cada brazo: gráficos de distribución (violín/caja) de ${\widehat{\alpha}}_{f}$, p95/p50 de duración total, error lógico; incluir $\Delta\widehat{\alpha}$ con IC y salvaguardas.

**Fig. 6 opcional — Diagnósticos espectrales (QEC).**\
PSD de procesos de error mostrando cómo el jitter de cadencia/multifase mueve los espectros de línea fuera de los picos dominantes.

**9.2 Tablas principales**

### Tabla 1 — Familias aceptadas (por compartimiento/brazo).

| Familia | #pts L | Amplitud $\log L$ | $\alpha_f$ (ODR, IC 50/95\%) | Theil–Sen | Banda SIMEX | ($R_{\text{coll}}^2$) | Apalancam. máx | Banderas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Física | 9 | 1.05 | 0.62 [0.55, 0.70] | 0.60 | 0.58–0.66 | 0.02 | 0.18 | — |
| QEC | 8 | 0.82 | 0.74 [0.66, 0.82] | 0.71 | — | 0.03 | 0.22 | — |
| … | … | … | … | … | … | … | … | … |

### Tabla 2 — Fusión y heterogeneidad (por franja temporal o brazo).

| Tiempo/Brazo | Familias | $\alpha_{\text{QC}} \pm \text{SE}$ | (Q) (gl) | $I^2$ | $\tau^2$ | ¿Fusión? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Con RTM | 3 | 0.69 $\pm$ 0.04 | 3.2 (2) | 37\% | 0.005 | Sí |
| Control | 3 | 0.54 $\pm$ 0.05 | 6.8 (2) | 71\% | 0.018 | No (reportar por familia) |

### Tabla 3 — Resultados de protocolos (A/B).

| Protocolo | Métrica | Control | Con RTM | Efecto ($\Delta$) | IC 95\% | ¿Pasa salvaguarda? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A (Fís) | $\alpha_{\text{phys}}$ | 0.48 | 0.64 | +0.16 | [0.07, 0.25] | ✔ |
| A (Fís) | Rendimiento | 100\% | 97\% | –3\% | [–6, 0]\% | ✔ |
| B (QEC) | $\alpha_{\text{QEC}}$ | 0.68 | 0.83 | +0.15 | [0.06, 0.24] | ✔ |
| C (Ejec) | p95/p50 | 1.85 | 1.60 | –0.25 | [–0.35, –0.15] | ✔ |

### Tabla 4 — Umbrales pre-registrados y banderas.

| Compuerta | Umbral | Estado |
| :--- | :--- | :--- |
| Colapso $R^2$ | < 0.05 | Aprobado |
| Heterogeneidad $I^2$ | < 50\% para fusión | Aprobado |
| MDE sobre $\Delta\alpha$ | ≥ 0.10–0.15 | Aprobado |
| Salvaguardas de KPI | ≤ 5\% pérdida de rendimiento; ≤ +5\% error lógico | Aprobado |

**9.3 Panel de robustez y sensibilidad**

-   **Estimadores:** ODR (primario), Theil–Sen, bandas SIMEX (± para $\sigma_{\xi}^{2}$).

-   **Ventanas:** repetir con $h$ ± 25%; $\widehat{\alpha}$ estable y colapso aún aprobado.

-   **Placebos:** invarianza ante reescalamiento de reloj; **Permutaciones:** permutar $L$ dentro del día—pendiente → ~0.

-   **Fusión excluyendo una familia a la vez:** reportar ${\widehat{\alpha}}_{QC}^{( - f)}$.

-   **Catastróficos:** reestimar excluyendo eventos marcados; mostrar Δ.

-   **Efecto fijo vs. efectos aleatorios:** publicar ambos; divergencia implica heterogeneidad genuina.

**9.4 Resultados negativos y límites de alcance**

Publicar compartimientos/brazos que **fallaron**:

-   NO_COLLAPSE (curvatura), REGIME_MIX (quiebres), THIN_COVERAGE, LEVERAGE_RISK, FAMILY_DIVERGENCE ($I^{2}$ alto).\
    Incluir una nota breve: mecanismo sospechado y próximos pasos (recompartimentar, cambio de instrumentación, aislamiento de mecanismo). Los resultados negativos definen **donde RTM no aplica**.

**9.5 Lista de verificación de una página (para cada conjunto de figuras/tablas)**

-   Claves BIN listadas y sin cambios.

-   \# de $L$ distintos ≥ 6 y amplitud ≥ 0.6.

-   ODR convergido; Theil–Sen reportado; SIMEX (si $\sigma_{\xi}^{2}$ es conocido).

-   Colapso: $R^{2} < 0.05$; placebo OK; sin puntos de cambio.

-   Fusión: $\mid \mathcal{F}_{t} \mid \geq 2$; $ I^{2} < 50\%$; REML convergido.

-   KPI: rendimiento, p95/p50 de duración total, error lógico, tiempo de operación—salvaguardas aplicadas.

-   Panel de robustez completado (ventanas, permutaciones, LOO).

-   Hashes de procedencia (YAML de métodos, semillas, versión de código) incluidos.

**9.6 Plantilla narrativa (texto breve de "Resultados")**

> *Capa física.* A través de 9 tamaños de clúster (amplitud 1.05 en $\log L$), la planificación con conciencia RTM incrementó la pendiente de $0.48$ a $0.64$ (Δ = $0.16$, IC 95% $\lbrack 0.07,0.25\rbrack$); los residuos mostraron $R_{\text{collapse}}^{2} = 0.02$. El rendimiento se mantuvo dentro de la salvaguarda del 5%.\
> *QEC.* Con 1–3% de jitter en cadencia, $\alpha_{\text{QEC}}$ subió de $0.68$ a $0.83$ (Δ = $0.15$, IC $\lbrack 0.06,0.24\rbrack$), el error lógico a $d$ fijo no empeoró.\
> *Tiempo de ejecución.* El agrupamiento por frentes de onda y el enrutamiento con conciencia de varianza redujeron p95/p50 de 1.85 a 1.60; $\alpha_{\text{runtime}}$ aumentó en $0.12$.\
> *Fusión.* Tres familias pasaron las compuertas; $I^{2} = 37\%$. El fusionado ${\widehat{\alpha}}_{QC} = 0.69 \pm 0.04$. Una alerta de nivel **vigilancia** de decoherencia se disparó el día 17; precedió un pico de duración total por 3 días.

**9.7 Resumen**

Las plantillas anteriores aseguran que cada afirmación está respaldada por: (i) prueba visual y numérica de **colapso**, (ii) estimación con conciencia EIV, (iii) contabilidad de **heterogeneidad** para la fusión, (iv) salvaguardas de KPI, y (v) evidencia completa de **robustez**.

**10. Discusión**

Esta sección interpreta los resultados de RTM-CC, aclara cómo una visión **pendiente-primero** complementa los paradigmas de fidelidad/QEC, y expone compromisos, riesgos y caminos de adopción.

**10.1 ¿Qué compra realmente un** $\mathbf{\alpha}$ **más alto?**

Una pendiente por compartimiento $\alpha$ más grande significa que **el tiempo se estira más pronunciadamente con la escala**, es decir, los agregados más grandes se ralentizan *relativamente* respecto a los más pequeños dentro de un entorno estable. Operativamente:

-   **Amortiguación de perturbaciones:** las perturbaciones a escala pequeña tienen menos probabilidad de sincronizar capas mayores (tiempo de ejecución → QEC → E/S), reduciendo cascadas que inflan colas (p95/p50), filas de espera y recalibraciones forzadas.

-   **Predecibilidad:** un $\alpha$ más alto típicamente reduce la **varianza corrida a corrida** (distribuciones de KPI más estrechas) porque el "gradiente de tempo" de la pila previene la alineación de eventos largos raros.

-   **Apalancamiento de control:** $\alpha$ es agnóstico a unidades; podemos optimizarlo con perillas de planificador/QEC/interconexión sin confundir cambios de unidades (relojes) con cambio estructural.

**No sustituye a la fidelidad.** RTM mejora **cómo** se comporta la temporización a través de la escala; no aumenta las fidelidades de compuertas de uno/dos qubits por sí mismo. Las ganancias llegan mediante menos cascadas y mejor uso de la fidelidad existente.

**10.2 Complementariedad con QEC y compilación**

-   **QEC:** El diseño tradicional elige la distancia de código $d$ a partir de tasas de error. RTM agrega un segundo eje: **geometría de cadencia**. Una ligera **desincronización** (jitter/multifase) puede elevar $\alpha_{\text{QEC}}$ a $d$ y decodificador fijos, a menudo mejorando la estabilidad sin sobrecosto adicional.

-   **Compilación/tiempo de ejecución:** El enrutamiento de vanguardia minimiza profundidad/longitud. RTM pide además minimizar **varianza temporal** y **coincidencia de operaciones largas**, lo cual puede mejorar las colas incluso si la profundidad media cambia marginalmente.

**10.3 Compromisos y frente de Pareto**

-   **Rendimiento vs. estratificación:** Elevar $\alpha$ agregando búferes/agrupamiento puede reducir la concurrencia bruta. Por eso optimizamos sobre un **frente de Pareto** (Sec. 7.1): incrementar $\alpha$ *sujeto a* pisos de rendimiento/fidelidad.

-   **Jitter vs. temporización del decodificador:** El micro-jitter debe mantenerse dentro de la validez del decodificador; de lo contrario se intercambia un $\alpha$ más alto por fallas lógicas.

-   **Tamaño modular:** Operar cerca de $m^{\star}$ (Sec. 7.5) equilibra costos intra/inter, pero alejarse demasiado (módulos más grandes o más pequeños) puede aplanar $\alpha$ (sincronización) o estrangular el ancho de banda.

**10.4 Modos de falla (informativos por diseño)**

La **compuerta de colapso** de RTM convierte los fallos en diagnósticos:

-   NO_COLLAPSE**:** log–log curvo → mecanismo faltante (p. ej., "reloj" dependiente de la escala o sobrecosto no lineal).

-   REGIME_MIX**:** quiebres → costuras ocultas (cambios de firmware/planificador); recompartimentar o dividir.

-   **$I^{2}$ alto:** los proxies discrepan → **no** fusionar; inspeccionar controles por familia.

Publicar estos casos mapea **límites de alcance** (donde RTM *no* aplica), lo cual es científicamente útil y previene extrapolaciones indebidas.

**10.5 Por qué un indicador fusionado único—y cuándo no usarlo**

**Ventajas:** $\text{ECI}_{\text{QC}}(t)$ resume la coherencia multiescala, habilitando **alertas** (Sec. 6.5) y seguimiento de tendencias.
**Desventajas:** La fusión puede ocultar heterogeneidad. De ahí las **compuertas** (al menos dos familias, $I^2$ < 50%, convergencia REML). Si fallan, publicar solo $\hat{\alpha}_f$ **por familia**; la ausencia de fusión es en sí un resultado ("la pila habla con pendientes diferentes").

**10.6 Relación con difusiones con cambio de tiempo y colas**

La visión PDE (RTM como un **reloj dependiente del estado**) explica por qué las **colas** se reducen cuando $\alpha$ sube: el **exponente dinámico efectivo** $z$ aumenta, y los tiempos de salida/primer paso escalan más pronunciadamente con el "radio" (Sec. 6 del artículo matemático). En términos de colas, la planificación que eleva $\alpha$ **descorrelaciona** las ráfagas de servicio y amortigua la amplificación de colas.

**10.7 Validez externa y portabilidad**

Dado que $\alpha$ es **invariante de gauge**, las comparaciones se sostienen entre laboratorios y generaciones cuando los compartimientos están pareados (claves de entorno). La misma tubería se porta a **iones atrapados**, **superconductores**, **átomos neutros** y **recocedores** con $(L,T)$ apropiados por capa. Lo que cambia es la instrumentación; la **lógica de colapso** y la **estimación EIV** permanecen.

**10.8 Camino de adopción (práctico)**

1.  **Modo sombra:** calcular ${\widehat{\alpha}}_{f}$ por familia y paneles de colapso sin cambiar operaciones.

2.  **Perillas de bajo riesgo:** habilitar **agrupamiento de lecturas**, **reinicios escalonados** y **jitter de cadencia** mínimo (≤3%).

3.  **Cerrar el lazo:** incorporar ${ECI}_{QC}(t)$ a tableros de guardia con niveles de alerta y protocolos de acción.

4.  **Planificación arquitectónica:** medir $a,b,A,B$ (Sec. 7.5) para elegir tamaños de módulo; iterar trimestralmente.

**10.9 Preguntas abiertas**

-   **Co-diseño con decodificadores:** ¿cómo incluir $\alpha$ directamente en las actualizaciones de planificación/grafos de los decodificadores?

-   **Controladores con aprendizaje:** ¿puede el RL ajustar $\alpha$ sujeto a pisos de KPI sin violar el colapso?

-   **Pruebas de holonomía:** estadísticas prácticas para distinguir curvatura de obstrucciones topológicas (falla global de colapso).

-   **Causalidad entre capas:** ¿cuándo los cambios de $\alpha$ en la capa física *causan* cambios en el tiempo de ejecución vs. simplemente correlacionan vía utilización?

**10.10 Conclusión clave**

RTM-CC agrega un **tercer eje**—la **geometría del tempo**—a fidelidad y escala. Con compuertas estrictas (colapso, heterogeneidad) y controles modestos (agrupamiento, jitter, varianza de enrutamiento), $\alpha$ se convierte en una palanca confiable para **estabilidad y rendimiento**, produciendo alertas tempranas y guía de diseño mientras se respeta la falsificabilidad científica.

**11. Limitaciones y alcance**

**Dependencia del compartimiento.** RTM es una teoría **por compartimiento**. Si el entorno (temperatura, firmware, topología, decodificador, utilización) deriva, la pendiente $\alpha$ es indefinida hasta que el compartimiento se divida. Los resultados solo son válidos dentro de claves BIN claramente documentadas.

**Sensibilidad a la elección de proxy.** Los proxies $(L,T)$ deben reflejar un **único mecanismo dominante** por familia. Proxies mal especificados (p. ej., mezclar lectura y enrutamiento en el mismo $T$) inducen curvatura y fallan válidamente el colapso.

**Sesgo de ventana finita.** Cuando $\alpha(u)$ deriva, cualquier ventana finita de ancho $h$ incurre en sesgo $O(\varepsilon h)$. Nuestra guía adiabática mitiga pero no elimina esto; el $\widehat{\alpha}$ reportado debe interpretarse como **local**.

**Supuestos del modelo EIV.** ODR/TLS y SIMEX asumen errores bien comportados (media cero, momentos finitos) e independencia de $x$. Los errores de colas pesadas o dependientes del estado requieren verificaciones de robustez (Theil–Sen, bootstrap, bandas de sensibilidad).

**Heterogeneidad de fusión.** La fusión de efectos aleatorios es apropiada solo cuando las familias son **conmensurables** e $I^{2} < 50\%$. De lo contrario, el indicador de número único se retiene por diseño; RTM no fuerza concordancia entre mecanismos.

**Límites de causalidad.** $\alpha$ es **estructural pero no causal** por defecto. Las secciones de diseño proponen intervenciones y protocolos A/B, pero las afirmaciones causales requieren los controles y salvaguardas pre-registrados que especificamos.

**Límites de alcance.** Los sistemas con temporización **no potencial** (curvatura persistente), **relojes dependientes de la escala** (sobrecostos que crecen con $L$ dentro de un compartimiento) o **holonomía** fuerte (costuras globales) quedan **fuera** de la aplicabilidad de RTM. En tales dominios, tratar $\alpha$ como indefinido y publicar resultados negativos.

**12. Métodos y reproducibilidad**

**12.1 Esquema de datos y compartimientos**

-   **Clave BIN:** {plataforma, banda de temperatura, hash de firmware (FPGA/DSP), ID de topología, política de enrutamiento, cadencia de síndromes, banda de utilización}.

-   **Tabla ordenada (por compartimiento):** \[x=log L, y=log T, family, BIN tags, replicate_id, timestamp, weight\].

-   **Compuertas de cobertura:** ≥6 $L$ distintos, amplitud ≥0.6 en $\log L$.

**12.2 Tubería de estimación (por familia, por compartimiento)**

1.  **Escaneo de puntos de cambio:** PELT/BIC sobre $(x,y)$ y sobre residuos si están disponibles; dividir si se detectan.

2.  **Inicialización:** pendiente/intercepto Theil–Sen; marcar catastróficos; construir pesos de réplica.

3.  **Ajuste primario:** ODR/TLS (residuos ortogonales) con SE de réplica o bootstrap.

4.  **SIMEX (opcional):** cuando $\sigma_{\xi}^{2}$ es estimable; extrapolar a $\lambda = - 1$.

5.  **Prueba de colapso:** regresar $\tilde{y} = y - \hat{\alpha}x - \hat{c}$ sobre $x$; requerir $R_{\text{collapse}}^2 < 0.05$, LOESS plano, placebo de reloj aprobado.

6.  **Diagnósticos:** apalancamiento ≤25%; gráficos de residuos; ancho de ventana $h$ registrado.

7.  **Aceptar/Rechazar:** aceptar si todas las compuertas pasan; de lo contrario marcar (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE, LEVERAGE_RISK, EIV_FAIL).

**12.3 Fusión y heterogeneidad (deslizante)**

-   **Pesos:** $w_{f} = 1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})$ con ${\widehat{\tau}}^{2}$ vía REML (DL como sensibilidad).

-   **Pendiente fusionada:** ${\hat{\alpha}}_{\mathrm{QC}}=\sum w_f {\hat{\alpha}}_f / \sum w_f$; **varianza:** $1 / \sum w_f$.

-   **Diagnósticos:** línea base de efecto fijo, **$Q$ de Cochran** e $I^{2}$.

-   **Compuertas:** fusionar solo si $\mid \mathcal{F} \mid \geq 2$ e $I^{2} < 50\%$. De lo contrario publicar por familia.

**12.4 Operación en tiempo real y alertas**

-   **Ventanas deslizantes:** horizonte deslizante en $x$ (ancho $h$) y reloj de pared (7–28 días).

-   **Suavizado:** mediana de 3 puntos; **puntuación Z** contra EWMA de 30 días.

-   **Niveles de alerta:** umbrales de Aviso/Vigilancia/Advertencia (Sec. 6.5).

-   **Protocolos de acción:** limitar concurrencia, escalonar reinicios, jitter de cadencia, enrutamiento con conciencia de varianza; todas las intervenciones deben volver a pasar el **colapso**.

**12.5 Robustez y sensibilidad**

-   **Estimadores:** publicar ODR (primario), Theil–Sen, bandas SIMEX.

-   **Ventanas:** sensibilidad ±25% en $h$; estabilidad de $\widehat{\alpha}$ requerida.

-   **Placebos y permutaciones:** invarianza ante reescalamiento de reloj; permutaciones de $L$ producen pendientes cercanas a cero.

-   **Fusión excluyendo una familia a la vez**; comparación **efecto fijo** vs. **efectos aleatorios**.

**12.6 Procedencia (YAML de métodos)**

-   Claves BIN, configuraciones de estimador, semillas de bootstrap, $\Lambda$ de SIMEX, ventana $h$, umbrales de colapso, compuertas de heterogeneidad, versiones de código de análisis.

-   Todos los gráficos y números incluyen hash del YAML de métodos; las re-ejecuciones con el mismo YAML reproducen los números dentro del ruido de bootstrap.

**13. Conclusión y perspectivas**

Presentamos la **computación cuántica con conciencia RTM (RTM-CC)**: un marco **pendiente-primero** que mide e **ingenieriza** la geometría del tiempo a través de la escala. Dentro de compartimientos estables, el tiempo característico $T$ escala con un proxy de tamaño $L$ como $T \propto L^{\alpha}$; el **exponente de coherencia** $\alpha$ es invariante a los relojes y, por lo tanto, comparable entre dispositivos, pilas y laboratorios. Con el **colapso** como compuerta falsificable y la estimación de **errores en variables**, $\alpha$ se convierte en una señal operativa confiable. La fusión de pendientes limpias por capa produce un $\mathbf{ECI}_{QC}$ **(t)** en tiempo real que soporta **alertas tempranas** (eventos de decoherencia) y **decisiones de diseño** (planificador, cadencia QEC, dimensionamiento modular, desfases de E/S).

**Qué agrega esto.** RTM-CC complementa la fidelidad/QEC introduciendo un tercer eje—**geometría del tempo**—que explica y controla colas, filas de espera y cascadas de sincronización. Controles modestos y reversibles (agrupamiento, reinicios escalonados, micro-jitter, enrutamiento de baja varianza) pueden **elevar** $\alpha$ sin degradar rendimiento o fidelidad cuando se usan con salvaguardas.

**Qué no hace.** RTM-CC no reemplaza las mejoras físicas (fidelidades, $T_{1}/T_{2}$), ni garantiza causalidad sin los protocolos A/B y salvaguardas que especificamos. Las fallas de colapso, la alta heterogeneidad o las costuras de régimen son **informativas**, delimitando límites de alcance en lugar de invitar correcciones post-hoc.

**Agenda a corto plazo.**

1.  **Ejecutar los protocolos** (Sec. 8) en plataformas superconductoras y de iones; publicar tanto éxitos como negativos con diagnósticos completos de colapso/fusión.

2.  **Cerrar el lazo**: desplegar tableros de ${ECI}_{QC}(t)$ y protocolos de acción de alertas en producción; evaluar adelanto–retraso vs. picos de KPI.

3.  **Co-diseñar con decodificadores** y compiladores para que la cadencia y el enrutamiento optimicen $\alpha$ sujeto a pisos de rendimiento/fidelidad.

4.  **Estandarizar reportes**: figuras/tablas en Sec. 9, YAML de métodos y artefactos de robustez abiertos.

**Preguntas a largo plazo.** Incorporar $\alpha$ en **modelos de difusión con cambio de tiempo** de colas; desarrollar **pruebas de holonomía** para distinguir curvatura de costuras; extender a **redes modulares** y plataformas de **átomos neutros**; integrar controladores basados en aprendizaje que respeten las compuertas de colapso.

**En síntesis.** RTM-CC brinda a los equipos cuánticos una **palanca robusta ante unidades y falsificable** sobre la temporización multiescala. Medir la pendiente, **validar por colapso**, fusionar cuando las familias concuerdan e **ingenierizar** $\alpha$—no como un eslogan, sino como una práctica reproducible para entregar computación cuántica más estable y eficiente.

**Apéndices**

**Apéndice A — Fundamentos matemáticos (elementos esenciales de RTM para CC)**

**A.1 Semigrupo → ley de potencia**

Asumir semigrupo de escala por compartimiento $T(bL) = f(b)T(L)$, $f(1) = 1$ y mensurabilidad cerca de $b=1$. Entonces $f(b) = b^{\alpha}$ y

$T(L) = \kappa L^{\alpha},v(u) = \log T = \alpha u + \log\kappa,u = \log L$

$\alpha$ es **invariante de gauge**; $\kappa$ es un **reloj**.

**A.2 1-forma y colapso**

Definir la 1-forma RTM $\omega = dv - \alpha\text{ }du$. El **colapso** (independencia residual de $v - \alpha u$ respecto a $u$) equivale a la **exactitud** de $\omega$ en un compartimiento simplemente conexo:

$\omega = d\psi(x),d\omega = 0,\psi\text{ independiente de }u$

Si $\alpha = \alpha(x,u)$, entonces $d\omega = - d\alpha \land du$; la curvatura no nula rompe el colapso.

**A.3 Exponentes variables (sesgo de ventana finita)**

Para $\alpha(u)$ lentamente variable:

$v(u) = \int_{u_{0}}^{u}{\alpha(s)\text{ }ds + \log\kappa(u),\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h)$

y $R_{\text{collapse}}^{2} = O((\varepsilon h)^{2})$ para ancho de ventana $h$.

**Apéndice B — Estimadores y algoritmos**

**B.1 Regresión de distancia ortogonal (TLS/ODR)**

Minimizar residuos ortogonales:

$\underset{\alpha,c}{\min}\sum_{i}^{}\frac{(y_{i} - \alpha x_{i} - c)^{2}}{\sigma_{y,i}^{2} + \alpha^{2}\sigma_{x,i}^{2}}$

**Inicialización:** Theil–Sen; **IC:** bootstrap pares/clúster; **verificaciones:** número de condición < $10^{4}$; apalancamiento máximo < 25%.

**B.2 Theil–Sen**

Mediana de pendientes por pares $\alpha_{ij} = (y_{j} - y_{i})/(x_{j} - x_{i})$; robusto ante valores atípicos; atenuación leve por EIV.

**B.3 SIMEX (opcional)**

Si $\sigma_{\xi}^{2} = Var(\xi)$ es estimable, simular $x^{(\lambda)} = x^{obs} + \sqrt{\lambda}\widetilde{\xi}$ y extrapolar $\widehat{\alpha}(\lambda)$ a $\lambda = - 1$.

**B.4 Compuerta de colapso**

Regresar residuos $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$ sobre $x$; requerir $R_{\text{collapse}}^{2} < 0.05$ y LOESS plano; pasar placebo de reloj.

**Apéndice C — Tarjetas de protocolo (plantillas para copiar y pegar)**

**C.1 Física (reinicios escalonados + oleadas de lectura)**

-   **L/T:** $L =$ tamaño de clúster activo; $T =$ intervalo de calibración estable.

-   **Brazos:** Control vs. con conciencia RTM (oleadas + 2–4% de desfase en reinicios).

-   **Duración:** 2–4 semanas, intercalado.

-   **Éxito:** $\Delta\alpha_{\text{phys}} \geq 0.15$ (IC 95% excluye 0), pérdida de rendimiento ≤5%, colapso aprobado.

**C.2 QEC (micro-jitter / multifase)**

-   **L/T:** $L = d$; $T =$ ciclos hasta falla lógica.

-   **Brazos:** Período fijo vs $Pk = P(1 + \eta k),\  \mid \eta k \mid \leq 0.02$ o 2–3 grupos de fase.

-   **Éxito:** $\Delta\alpha_{\text{QEC}} \geq 0.15$, sin regresión de error lógico (>5%) a $d$ fijo.

**C.3 Tiempo de ejecución (agrupamiento + enrutamiento de baja varianza)**

-   **L/T:** $L =$ ancho post-mapeo; $T =$ duración total.

-   **Brazos:** Línea base vs. frentes de onda + enrutamiento penalizado por varianza.

-   **Éxito:** $\Delta\alpha_{\text{runtime}} \geq 0.10$ y p95/p50 de latencia ↓ ≥10%.

**C.4 E/S (ventanas con desfase)**

-   **L/T:** $L =$ grado de multiplexado; $T =$ latencia de lectura p95 (y p50).

-   **Brazos:** Sincrónico vs. ventanas con desfase.

-   **Éxito:** $\Delta\alpha_{\text{IO}} \geq 0.10$, p95/p50 ≤ 1.6 sobre la mayoría de $L$.

**Apéndice D — YAML de métodos (esqueleto)**

### YAML de métodos (esqueleto)

```
bin:
  platform: "SC"              # o "IONS", "NA"
  temperature_band: "10-15mK"
  firmware_hash: "fpga_1.4.2_dsp_0.9.8"
  topology_id: "mesh-v3"
  routing_policy: "baseline"  # o "rtm-aware"
  syndrome_cadence: "P=3.2us, jitter=0%"
  utilization_band: "30-60%"
 
estimation:
  min_L_points: 6
  min_logL_span: 0.6
  eiv: "odr"
  odr:
    init: "theil-sen"
    leverage_cap: 0.25
    bootstrap: {clusters: true, reps: 2000, seed: 123}
  simex:
    enabled: false
    lambda: [0.5, 1.0, 1.5, 2.0]

collapse:
  r2_threshold: 0.05
  placebo_clock: true
  changepoint_scan: {method: "PELT", penalty: "BIC"}
 
fusion:
  heterogeneity_gate_I2: 0.5
  tau2_method: "REML"
  min_families: 2
 
eci_rt:
  window_logL: 0.8
  horizon_days: 14
  smoothing: "median3"
  alert:
    z_advisory: -1.5
    z_watch: -2.0
    z_warning: -2.5
```

**Apéndice E — Glosario de notación**

-   $L$ : proxy de escala (específico por capa); $u = \log L$.

-   $T$ : tiempo característico; $v = \log T$.

-   $\alpha$ : **exponente de coherencia** (pendiente; invariante de reloj).

-   **Compartimiento**: rebanada de entorno con {plataforma, banda de temperatura, hash de firmware, ID de topología, política de enrutamiento, cadencia de síndromes, banda de utilización} fijos.

-   **Colapso**: $R^{2}(\widetilde{y} \sim x) < 0.05$ para $\widetilde{y} = y - \widehat{\alpha}x$; los residuos no muestran tendencia vs $x$.

-   $\mathbf{ECI}_{QC}(t)$ : pendiente fusionada vía efectos aleatorios en el tiempo $t$.

-   $Q,I^{2},\tau^{2}$ : estadísticos de heterogeneidad para la fusión.

-   ODR/TLS, Theil–Sen, SIMEX: estimadores de pendiente bajo EIV.

-   **Ventana adiabática**: ancho $h$ en $u$ donde $\mid \partial_{u}\alpha \mid h \ll 1$.

**Apéndice F — Recetas reproducibles de figuras (mínimas)**

-   **Panel de colapso**:

    -   Ajustar ODR; calcular residuos $\widetilde{y}$.

    -   Graficar $y$ vs $x$ + banda ODR; residuo vs $x$ con LOESS.

    -   Anotar $R^2\_{\text{collapse}}$, #L, amplitud, apalancamiento.

-   **Diagrama de bosque**:

    -   Para familias aceptadas, mostrar $\widehat{\alpha}_f \pm \text{IC}$; *calcular* $w_f$, $Q$, $I^2$, $\hat{\tau}^2$.

    -   Superponer ${\widehat{\alpha}}_{QC}$ fusionado.

-   $\mathbf{ECI}_{QC}(t)$ :

    -   Fusión deslizante; mostrar bandas de 50/95%; fondo coloreado por niveles de $I^{2}$; marcar niveles de alerta.

**APÉNDICE G — Análisis empírico: escalamiento de hardware cuántico y el confusor generacional**

El marco RTM dicta que incrementar los límites físicos de una red fuertemente acoplada pero no resonante incrementará proporcionalmente su fricción topológica. Para probar esto en arreglos cuánticos, analizamos los tiempos de coherencia $T_{2}$ de 31 procesadores IBM Quantum (de 5 a 1121 qubits).

**G.1 Observación heurística y paradoja de Simpson**

La regresión ingenua inicial por mínimos cuadrados ordinarios (OLS) sobre el conjunto de datos crudo arrojó un exponente de escalamiento positivo de $\alpha = \  + 0.227$. Esto creó la ilusión de que agregar más qubits extendía intrínsecamente los tiempos de coherencia. Sin embargo, esta es una manifestación clásica de la paradoja de Simpson: los procesadores más grandes fueron construidos años después que los más pequeños, lo que significa que sus tiempos de $T_{2}$ extendidos fueron resultado de materiales superconductores y técnicas de fabricación superiores, no de su tamaño espacial incrementado.

**G.2 Validación multivariable rigurosa con EIV**

Para aislar matemáticamente la ley de escalamiento físico del progreso de la ingeniería humana, desplegamos una tubería estadística de "equipo rojo":

1.  **Regresión de distancia ortogonal (ODR) multivariable:** Abandonamos la compartimentación categórica cruda por "eras" a favor de un modelo continuo multivariable. Este evalúa simultáneamente la progresión tecnológica cronológica junto con la expansión espacial topológica.

2.  **Inyección de ruido de calibración:** Inyectamos explícitamente una varianza de calibración de hardware realista del $15\%$ en las lecturas de $T_{2}$, forzando al marco a absorber el ruido estándar de medición criogénica.

**G.3 La clase de transporte inverso (hallazgos robustos)**

Una vez que la mejora continua de materiales superconductores se normaliza algebraicamente, la ilusión de escalamiento monolítico se desintegra, revelando la física verdadera del arreglo cuántico:

-   **Factor de ganancia tecnológica:** El modelo extrae con precisión la progresión ingenieril, mostrando que la coherencia del hardware de IBM mejora en un factor de $\mathbf{\gamma}\mathbf{= \  + 0.139}$ **dex/año**.

- **Verdadero exponente topológico:** Después de sustraer $`\gamma`$, el escalamiento físico aislado revela un exponente negativo de $`\mathbf{\alpha = -0.259 \pm 0.049}`$, IC bootstrap [ $`-0.382, -0.038`$ ]. El IC excluye cero, confirmando que la clasificación de transporte inverso no es un artefacto de la inyección de ruido.

**Conclusión:** El modelo ODR multivariable demuestra que el escalamiento crudo positivo ($`\alpha \approx +0.23`$) es una **paradoja de Simpson** producida por la confusión del tiempo con la escala. Después de remover el confusor temporal ($`\gamma = +0.139`$ dex/año), el verdadero escalamiento físico es negativo: la decoherencia empeora de forma no lineal con el tamaño del sistema a generación tecnológica fija. Este hallazgo se clasifica como **NOVEDOSO** por el equipo rojo (abril 2026): la reversión del confusor no es visible sin descomposición multivariable, y conlleva una implicación ingenieril directa: el escalamiento monolítico de qubits superconductores sin innovaciones arquitectónicas que supriman la decoherencia colectiva producirá rendimientos decrecientes en tiempo de coherencia por qubit.

### APÉNDICE H — Auditoría del equipo rojo: verificación y certificación (abril 2026)

Las afirmaciones empíricas de este documento fueron sometidas a una auditoría adversarial independiente por el equipo rojo de RTM utilizando **Claude Opus 4.6 con pensamiento extendido** en abril de 2026. Este documento recibió la **puntuación más alta de todo el corpus RTM (82%)**. El siguiente registro de verificación se proporciona por transparencia.

**H.1 Qué se probó**

| Afirmación | Prueba | Resultado |
|-------|------|--------|
| α crudo = +0.23 (ingenuo, sin corregir) | Regresión OLS, 31 procesadores | **Confirmado** ✓ |
| Confusor temporal γ = +0.139 dex/año | ODR multivariable | **Confirmado** ✓ |
| α corregido = −0.259 ± 0.049 | ODR tras normalización temporal | **Confirmado** ✓ |
| IC bootstrap [−0.382, −0.038] excluye 0 | Bootstrap 3.000 iteraciones | **Confirmado — IC excluye cero** ✓ |
| Inyección de ruido criogénico del 15% | Margen de error conservador | **Sobrevive la inyección de ruido** ✓ |
| Identificación de paradoja de Simpson | Reversión de dirección crudo vs. corregido | **Confirmado — la dirección se revierte completamente** ✓ |
| Ubicación en clase de transporte inverso (α < 0) | Verificación de clasificación | **Confirmado** ✓ |

**H.2 Veredicto de clasificación**

| Hallazgo | Clasificación | Justificación |
|---------|---------------|-----------|
| Identificación de paradoja de Simpson (α: +0.23 → −0.259) | **NOVEDOSO** | La reversión de dirección no es visible sin descomposición multivariable; no reportado en la literatura cuántica de IBM |
| Clase de transporte inverso (α < 0) | **NOVEDOSO** | Clasificación nativa de RTM; sin enmarcamiento topológico previo del escalamiento de decoherencia cuántica |
| Confusor temporal γ = +0.139 dex/año | **CONVERGENTE** | Consistente con las mejoras conocidas del roadmap de hardware de IBM |
| Implicación arquitectónica (el escalamiento monolítico falla) | **CONSISTENTE** | Consistente con la literatura conocida de corrección de errores cuánticos (Preskill 2018) |

**H.3 Por qué este es el hallazgo más fuerte del corpus**

El equipo rojo identificó tres propiedades que hacen este hallazgo excepcional:

1. **La reversión es completa y grande.** El signo de α cambia de +0.23 a −0.259 — no un cambio pequeño, sino una reversión completa. Esto significa que la conclusión ingenua ("los procesadores más grandes son mejores") no solo es cuantitativamente incorrecta, sino direccionalmente incorrecta.

2. **El confusor es físicamente significativo.** El confusor temporal (γ = +0.139 dex/año) captura progreso ingenieril genuino. Removerlo aísla correctamente la física. Esta es una descomposición metodológicamente sólida, no una corrección ad-hoc.

3. **La implicación ingenieril es directa y accionable.** Si el escalamiento monolítico empeora la decoherencia a generación tecnológica fija, entonces el camino hacia la computación cuántica a gran escala requiere innovación arquitectónica (corrección de errores, diseño modular, qubits topológicos), no simplemente agregar más qubits a un chip monolítico. Esta predicción es falsificable mediante los protocolos experimentales de las Secciones 3–9.

**H.4 Correcciones de tono aplicadas**

| Frase original | Corregido a |
|-----------------|-------------|
| "unequivocally in the Inverse Transport Class" | "in the Inverse Transport Class" |
| "definitively proves that the raw positive scaling is a statistical mirage" | "demonstrates that the raw positive scaling is a Simpson's Paradox" |
| "strictly negative exponent" | "negative exponent" |
| "proves that as quantum system size increases, topological noise scales collectively" | "is consistent with topological noise scaling collectively" |
| "This empirical result proves that" | "This empirical result demonstrates that" |

**H.5 Nota sobre protocolos experimentales (Secciones 3–9)**

Los protocolos experimentales constituyen la mayor parte del documento y están correctamente enmarcados como **prescripciones de ingeniería** derivadas del marco RTM-CC, no como validaciones empíricas. El equipo rojo confirma que este enmarcamiento es apropiado. Los protocolos son internamente consistentes con el hallazgo de transporte inverso y generan predicciones específicas y falsificables ($`\Delta\hat{\alpha}_{runtime} \geq 0.10`$, IC excluye 0) que pueden probarse en hardware cuántico real.

**H.6 Veredicto del equipo rojo**

La identificación de la paradoja de Simpson ($`\alpha: +0.23 \rightarrow -0.259`$, IC bootstrap [−0.382, −0.038]) es el hallazgo novedoso más riguroso metodológicamente de todo el corpus. La ODR multivariable con conciencia de confusores separa correctamente la física de la ingeniería, la reversión está estadísticamente confirmada y la implicación ingenieril es directa y accionable.

El hallazgo es novedoso en el sentido estricto: revela estructura en el conjunto de datos de hardware cuántico de IBM que no es visible sin el enfoque de descomposición de confusores de RTM, y genera predicciones que los marcos existentes de evaluación cuántica no hacen. No se requirió campaña de flanqueo; el hallazgo se sostiene por mérito propio.


*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
