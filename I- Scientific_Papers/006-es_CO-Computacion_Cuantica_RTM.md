<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **Computación Cuántica Consciente de RTM**  
**Un Marco Multiescala, Pendiente-Primero para Coherencia, Programación y Diseño**  
  
Álvaro Quiceno

</div>

**Resumen**

Introducimos una metodología **pendiente-primero** para computación cuántica basada en **Relatividad Temporal Multiescala (RTM)**. Dentro de un régimen operacional fijo, RTM postula que un tiempo característico $T$ escala con un proxy de tamaño/escala $L$ mediante una ley de potencia,

$$\log T\text{\:\,} = \text{\:\,}\alpha\text{ }\log L\text{\:\,} + \text{\:\,}c,
$$

donde el **exponente de coherencia** $\alpha$ es la señal estructural **invariante de reloj** y $c$ codifica reloj/unidades. Adaptamos RTM a pilas cuánticas---**física**, **QEC**, **compilador/tiempo de ejecución**, e **I/O--cryo**---definiendo pares $(L,T)$ específicos por capa (ej., número de qubits activos vs. tiempo de calibración estable; distancia de código vs. tiempo de falla lógica; grado de multiplexación vs. latencia de lectura; ancho de circuito vs. makespan), y estimando pendientes por bin bajo errores en variables (ODR/TLS, Theil--Sen, SIMEX). Una **prueba de colapso** valida el escalamiento y protege contra mezcla de regímenes; las pendientes limpias por familia se fusionan en un $\mathbf{ECI}_{QC}$**(t)** en tiempo real con incertidumbre y puertas de QA.

Formulamos hipótesis **falsificables**: **(H1)** mayor $\alpha$ pre-choque predice márgenes de estabilidad más largos (menos recalibraciones forzadas, menor error lógico a $d$ fijo); **(H2)** los **eventos de decoherencia**---caídas significativas con QA limpio en ${ECI}_{QC}$---preceden picos en error lógico, colas, o makespan; **(H3)** las **cascadas de tempo** micro→meso→macro exhiben $\alpha$ no decreciente dentro de regímenes estables. Demostramos cómo la **programación consciente de RTM** (agrupamiento, reinicios escalonados, enrutamiento de baja varianza), el **diseño de cadencia QEC** (desincronización de ciclos de síndrome), y el **dimensionamiento modular** (puntos óptimos para interconexión) pueden mejorar el rendimiento y la confiabilidad sin cambiar las fidelidades físicas. El marco es reproducible, robusto al gauge (los cambios de unidad/reloj no afectan $\alpha$), y diseñado para fallar graciosamente (no-colapso y alta heterogeneidad se convierten en límites de alcance, no en arreglos post-hoc).

**Validación empírica sistemática**$\mathbf{\rightarrow}$**(APÉNDICE G)**. Validamos el marco diagnóstico RTM en hardware cuántico a través de un análisis sistemático de 31 procesadores IBM Quantum que abarcan de 5 a 1121 qubits. El análisis inicial de escalamiento crudo sugirió una relación positiva coherencia-a-tamaño ($\alpha \approx + 0.23$); sin embargo, RTM aísla esto como una ilusión estadística impulsada por un factor de confusión de manufactura (mejoras tecnológicas generacionales). Para desenredar definitivamente los avances de ingeniería cronológicos del verdadero escalamiento de transporte topológico, desplegamos una tubería de Regresión de Distancia Ortogonal (ODR) Multivariable, inyectando un margen realista de ruido de calibración criogénica del $15\%$. Al normalizar algebraicamente el factor de ganancia tecnológica ($\gamma = \  + 0.139$ dex/año), el verdadero escalamiento topológico revela un exponente estrictamente negativo de $\mathbf{\alpha}\mathbf{= \  - 0.259\ }\mathbf{\pm}\mathbf{0.049}$. Esto coloca la decoherencia cuántica macroscópica inequívocamente en la Clase de Transporte Inverso ($\alpha < \ 0$), junto con la difusión clásica de Stokes-Einstein. Este resultado empírico prueba que a medida que aumenta el tamaño del sistema cuántico ($N$), el ruido topológico (diafonía, defectos correlacionados) escala colectivamente en lugar de independientemente, causando que el sistema decohere más rápido. RTM separa exitosamente las leyes de escalamiento físico subyacentes de los artefactos de ingeniería, demostrando que la coherencia masiva requiere resonancia arquitectónica, no meramente escalamiento monolítico por fuerza bruta.

**1. Introducción**

**1.1 Motivación: más allá de fidelidades y tasas de error**

El rendimiento cuántico usualmente se resume mediante **métricas puntuales**---fidelidades de uno y dos qubits, $T_{1}/T_{2}$, tasas de error lógico, o figuras de benchmark (QED-C, QV). Sin embargo, la confiabilidad práctica y el rendimiento dependen de algo ortogonal: **cómo se estira el tiempo a través de la escala** en una pila de múltiples etapas---qubits y resonadores, ciclos de código, compiladores, I/O criogénico. Cuando los subsistemas pequeños responden rápidamente y los más grandes responden más lentamente de manera disciplinada y estratificada, los choques se **disipan**; cuando los tiempos **se aplanan**, las perturbaciones percolan a través de capas y sincronizan fallas (estancando lecturas, elevando el error lógico, o forzando recalibraciones globales).

La **Relatividad Temporal Multiescala (RTM)** proporciona un lenguaje compacto para este fenómeno. Dentro de un régimen fijo, RTM espera una relación de ley de potencia entre un **tiempo característico** $T$ y un **proxy de escala** $L$: la **pendiente** $\alpha$ en $\log T = \alpha \log L + c$ es estructural (invariante a unidades de tiempo), mientras que la ordenada al origen $c$ es un **reloj** (gauge). Traemos este principio a la computación cuántica y mostramos que medir, validar e **ingenierar** $\alpha$ produce palancas accionables, independientes de unidades nominales, para mejorar la estabilidad y el rendimiento.

**1.2 RTM en una línea**

**La estructura vive en la pendiente; los relojes viven en el gauge.**\
Un cambio de reloj o unidades desplaza $c$ pero deja $\alpha$ sin cambiar. Así $\alpha$ puede compararse entre dispositivos, pilas y laboratorios, mientras que $c$ no.

**1.3 Contribuciones**

Este artículo hace cinco contribuciones:

1.  **Operacionalización de RTM para QC.** Definimos pares $(L,T)$ específicos por capa para capas **física**, **QEC**, **compilador/tiempo de ejecución**, e **I/O--cryo** (ej., $L =$ qubits activos, $T =$ tiempo de calibración estable; $L = d$, $T =$ ciclos hasta falla lógica; $L =$ grado de multiplexación, $T =$ latencia de lectura; $L =$ ancho de circuito, $T =$ makespan).

2.  **Validación y estimación.** Proporcionamos una **prueba de colapso** (independencia residual de $\log T - \alpha \log L$ respecto a $\log L$) para detectar mezcla de regímenes y curvatura no-potencia, y adoptamos estimación de **errores en variables** (ODR/TLS, Theil-Sen, SIMEX) con incertidumbre bootstrap y guardias de puntos de cambio.

3.  **Un indicador único en tiempo real.** Fusionamos las pendientes por familia en $\mathbf{ECI}_{QC}$**(t)** vía meta-análisis de efectos aleatorios con controles de heterogeneidad ($Q$, $I^{2}$, ${\widehat{\tau}}^{2}$); publicamos banderas de QA y retenemos la fusión cuando los proxies discrepan.

4.  **Palancas de diseño.** Formalizamos la **programación consciente de RTM** (agrupamiento, reinicios escalonados, enrutamiento de baja varianza), el **diseño de cadencia QEC** (desincronización para evitar bloqueo de fase entre errores físicos y extracción de síndrome), y el **dimensionamiento modular** (elegir escalas de módulo/interconexión que eleven $\alpha$ sin estrangular el rendimiento).

5.  **Hipótesis falsificables y protocolos.** Pre-registramos **H1--H3** con protocolos A/B en plataformas superconductoras y de iones atrapados, métricas (rendimiento, makespan, error lógico, tiempo de actividad, ratios p95/p50), y umbrales de decisión para adopción.

**1.4 Qué es** $\mathbf{\alpha}$**---y qué no es**

-   **Es:** una **pendiente por bin** que vincula un tiempo $T$ a una escala $L$ dentro de un **ambiente fijo** (misma temperatura/firmware/topología/programa de síndrome). Captura la **geometría del tempo a través de la escala**.

-   **No es:** un parámetro causal por defecto; los cambios de nivel en $T$ (unidades, relojes, desplazamientos) **no** cambian $\alpha$. Cuando el colapso falla, $\alpha$ está **indefinido** para ese bin y no debe fusionarse.

**1.5 Ejemplos de** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$ **específicos por capa (vista previa)**

-   **Física:** $L =$ qubits activos / grado de acoplador / tamaño de cluster; $T =$ intervalo de calibración estable, latencia de compuerta/RO, tiempo medio hasta deriva.

-   **QEC:** $L = d$ (distancia de código) o número de qubits lógicos; $T =$ ciclos hasta falla lógica; cadencia de extracción de síndrome.

-   **Compilador/tiempo de ejecución:** $L =$ ancho de circuito o profundidad post-mapeo; $T =$ makespan; retraso de cola y latencia de reprogramación.

-   **I/O--cryo:** $L=$ grado de multiplexación o canales; $T =$ latencia de lectura/recuperación de BER; longitud de cola p95.

**Uso.** Prefiera ODR como la rutina de ajuste base; reporte SIMEX como una estimación de **sensibilidad** junto a ODR. Si $\sigma_\xi^2$ es incierto, dé una banda (bajo/med/alto) para $\hat{\alpha}_{\text{SIMEX}}$.

**1.6 Hipótesis (falsificables)**

-   **H1 (Resiliencia):** Mayor $\alpha$ pre-choque se asocia con picos de error lógico más pequeños a $d$ fijo e intervalos de calibración estable más largos.

-   **H2 (Anticipación):** Las caídas de ${ECI}_{QC}$ con QA limpio preceden aumentos en makespan, colas, o error lógico por semanas a meses, añadiendo valor predictivo sobre líneas base (fidelidad, utilización, temperatura).

-   **H3 (Cascada):** Dentro de regímenes estables, $\alpha_{\text{física}} \leq \alpha_{\text{QEC}} \leq \alpha_{\text{tiempo de ejecución/I/O}}$; las pruebas de direccionalidad favorecen el flujo de tiempo micro→meso→macro.

**1.7 Diseño consciente de RTM (intuiciones que probaremos)**

-   **Programación:** Evite patrones que **aplanen** $\alpha$ (operaciones largas, fuertemente acopladas en paralelo); favorezca el **agrupamiento** de lecturas y reinicios **escalonados** para prevenir cascadas de sincronización.

-   **Cadencia QEC:** Introduzca ligera **desincronización** (desplazamientos de fase) entre ciclos de síndrome y ritmos de ruido conocidos para elevar $\alpha_{\text{QEC}}$.

-   **Modularidad:** Elija el tamaño de módulo y la densidad de interconexión donde $\alpha$ sea lo suficientemente alto para amortiguar cascadas inter-módulo pero no tan alto que el rendimiento se estrangule.

**1.8 Relación con trabajo previo**

Nuestro marco complementa los enfoques centrados en fidelidad y modelos de error añadiendo una **geometría escala--tempo**. Es compatible con (no un reemplazo de) la teoría de código surface/LDPC, heurísticas de compilación/enrutamiento, y modelos de colas; contribuye una estadística **invariante de gauge** $\alpha$ y una prueba de especificación de **colapso** para separar efectos de **estructura** de efectos de **reloj**. En el lenguaje de procesos estocásticos, nuestra sección de dinámica (más adelante) conecta RTM con **difusiones con cambio de tiempo**; en términos de meta-análisis, nuestra fusión imita **efectos aleatorios** con **puertas de heterogeneidad** explícitas.

**1.9. Validación Empírica Sistemática: La Ilusión del Escalamiento Monolítico**$\mathbf{\rightarrow}$**(APÉNDICE G)**

Una premisa fundamental de RTM es su capacidad para diagnosticar la verdadera clase de transporte de un sistema observando su exponente de escalamiento. En la carrera por construir computadoras cuánticas tolerantes a fallas, los desarrolladores de hardware han escalado continuamente los tamaños de procesadores monolíticos (conteos de qubits). Superficialmente, los datos históricos parecen sugerir que los procesadores más grandes poseen mejores tiempos de coherencia ($T_{2}$). Sin embargo, dentro del marco RTM, debemos preguntar: ¿es esta mejora una propiedad de la escala espacial ($\alpha > \ 0$), o es un desplazamiento artificial generado por avances tecnológicos continuos?

Para responder esto, utilizamos RTM como un filtro diagnóstico en 31 procesadores IBM Quantum. Hipotetizamos que la decoherencia cuántica no es un conjunto de eventos independientes aislados, sino un colapso topológico colectivo. Por tanto, el verdadero escalamiento físico debería exhibir una firma de transporte Inverso ($\alpha < \ 0$), donde una huella geométrica más grande naturalmente amplifica la diafonía y el ruido correlacionado. Al desplegar modelado de Errores en Variables multivariable, demostramos cómo RTM corta matemáticamente a través de los factores de confusión de manufactura para revelar la física cruda y subyacente de los sistemas cuánticos macroscópicos.

**2. Fundamentos RTM Adaptados a Computación Cuántica**

Esta sección declara los axiomas RTM, deriva la forma de **ley de potencia** $T = \kappa L^{\alpha}$, y adapta las nociones de **reloj/gauge** y **colapso** a pilas cuánticas. A lo largo, $L > 0$ es un **proxy de escala** (específico de capa) y $T > 0$ es un **tiempo característico** medido en un **ambiente/bin fijo** (misma temperatura, firmware, topología, programa de síndrome, banda de utilización).

**2.1 Axiomas (por bin)**

**A1 --- Semigrupo de escala.** Para cualquier dilatación $b > 0$,

$$T(bL) = f(b)\text{ }T(L),
$$

con $f(1) = 1$ y $f(b_{1}b_{2}) = f(b_{1})f(b_{2})$.

**A2 --- Regularidad leve.** $f$ es medible (o continua en $b = 1$).

**A3 --- Invariancia de reloj dentro del bin.** Los **cambios de reloj** permitidos multiplican $T$ por un factor $c > 0$ **independiente de** $L$ dentro del bin (cambios de unidad, líneas base de marca de tiempo, desplazamientos de latencia fija). En práctica QC: reescalamiento de unidades de tiempo, gastos generales de lectura constantes, líneas base de I/O cryo constantes.

**A4 --- Binning.** Las comparaciones se hacen dentro de bins donde el ambiente es estable. Si se detecta un punto de cambio, el bin debe dividirse.

**2.2 Solución de ecuación funcional → ley de potencia**

Sea $u = \log L$, $v = \log T$. De A1--A2, la ecuación de Cauchy multiplicativa da $f(b) = b^{\alpha}$ para algún $\alpha \in \mathbb{R}$. Por tanto

$$T(L) = \kappa L^{\alpha},v(u) = \alpha u + \log\kappa.
$$

**Interpretación.** $\alpha$ es el **exponente de coherencia** (pendiente); $\kappa$ es un **reloj** (ordenada al origen).

**2.3 Relojes (gauge multiplicativo vs. latencia aditiva)**

En RTM, un "cambio de reloj" dentro de un bin fijo es un reescalamiento **multiplicativo** de todos los tiempos característicos: $T^{'} = cT$, $c > 0$ independiente de $L$. Esto incluye conversiones de unidades de tiempo (ns↔µs), reescalamientos uniformes de base de tiempo/tasa de tick, o factores de calibración uniformes. En coordenadas logarítmicas, $\log T^{'} = \log T + \log c$, así que $\alpha$ permanece sin cambios y solo la ordenada al origen se desplaza.\
Por contraste, las **latencias constantes** (ej., preámbulo de lectura fijo, retraso de pipeline, desplazamientos de línea base de marca de tiempo) son **aditivas**: $T_{\text{obs}} = T + b$. En gráficos log--log esto no es un desplazamiento puro de ordenada al origen y puede sesgar $\alpha$, especialmente cuando $T$ no es $\gg b$. Por tanto, antes de estimar $\alpha$, ya sea:\
(i) estime/reste la latencia $b$ y ajuste usando $T_{eff} = \max(T_{\text{obs}} - b,\varepsilon)$, o\
(ii) restrinja el análisis a regímenes donde $T_{\text{obs}} \gg b$ y reporte la sensibilidad de $\alpha$ a valores plausibles de $b$.

**2.4 Colapso como prueba de especificación por bin**

Dadas observaciones $\{(L_i, T_i)\}_i$ *en un bin, defina* $x_i = \log L_i$, $y_i = \log T_i$. Ajuste una pendiente por bin $\hat{\alpha}$ (Sección 5) y examine los **residuos**

$${\widetilde{y}}_{i}: = y_{i} - \widehat{\alpha}x_{i}.
$$

**Prueba de colapso.** En un bin RTM válido, $\widetilde{y}$ debería ser **independiente de** $x$ (hasta el ruido). Lo operacionalizamos con:

-   Una regresión $\widetilde{y} \sim x$ y requerir $R_{\text{colapso}}^{2} < \tau$ (por defecto $\tau = 0.05$).

-   Un **placebo de reloj**: multiplicar todos los $T_{i}$ por una constante; $\widehat{\alpha}$ y $R_{\text{colapso}}^{2}$ deben permanecer sin cambios.

-   Una **verificación suave** (LOESS o spline) para tendencia visible; si está presente, rechace el bin.

**Significado.** El colapso establece que, después de remover $\widehat{\alpha}\ logL$, solo queda un **gauge** (ruido de ordenada al origen), no una tendencia vs. escala.

**2.5 Exponentes variables y sesgo de ventana finita**

En la práctica, $\alpha$ puede derivar lentamente con el ambiente o la escala (ej., a través de bandas de utilización o factores de multiplexación). Escriba

$$v(u) = \int_{u_{0}}^{u}{\alpha(s)\text{ }ds + \log\kappa(u),}
$$

con $\mid \alpha^{'}(u) \mid \leq \varepsilon$ pequeño en la ventana y $\kappa$ **lentamente variable**. Para cualquier ventana simétrica de ancho $h$ en $u$,

$$\widehat{\alpha}(u;h)\text{\:\,} = \text{\:\,}\alpha(u)\text{\:\,} + \text{\:\,}O(\varepsilon h)\text{\:\,} + \text{\:\,}O(\text{variación-lenta}),
$$

y

$$R_{\text{colapso}}^{2}\text{\:\,} = \text{\:\,}O((\varepsilon h)^{2}).
$$

**Regla.** Elija bins/ventanas lo suficientemente pequeños que la curvatura sea despreciable; de lo contrario divida el bin.

**2.6 Modos de falla (debería fallar)**

RTM está diseñado para **predecir su propio fallo**:

1.  **Mezcla de regímenes (quiebres).** Ejemplo: cambiar la cadena de lectura o el programador de síndrome a mitad de bin. El gráfico log--log muestra un cambio de pendiente en $L^{\star}$; el colapso falla.

2.  **Curvatura (no-potencia).** Ejemplo: un gasto general dependiente de multiplexación que crece no linealmente con $L$. Los residuos muestran tendencia con $x$; el colapso falla incluso después de rebinear.

3.  **Relojes dependientes de escala.** Cualquier factor de "reloj" $c(L)$ que dependa de $L$ no es un gauge; inyecta componentes $du$ en la 1-forma y debe modelarse explícitamente (o el bin se rechaza).

**2.7 Mapeo de capas QC (notación y ejemplos)**

Usaremos estos pares $(L,T)$ **canónicos** en secciones posteriores (otros pueden añadirse si pasan el colapso):

-   **Física:**\
    $L =$ número de **qubits activos** (o grado de cluster/acoplador);\
    $T =$**intervalo de calibración estable**, latencia de **compuerta**, latencia de **lectura**, o **tiempo medio hasta deriva**.

-   **QEC:**\
    $L =$**distancia de código** $d$ (o conteo de qubits lógicos);\
    $T =$**ciclos hasta falla lógica** a error objetivo fijo.

-   **Compilador/Tiempo de ejecución:**\
    $L =$**ancho de circuito** o **profundidad post-mapeo**;\
    $T =$**makespan** o **retraso de cola**.

-   **I/O--Cryo:**\
    $L =$**grado de multiplexación** o conteo de canales de lectura;\
    $T =$**latencia de lectura efectiva** / **vida media de recuperación de BER** / **longitud de cola p95 (en tiempo)**.

Cada familia produce un $\hat{\alpha}\_f$ por bin. Solo las familias que **pasan el colapso** y QA contribuyen al indicador fusionado $ECI\_{\text{QC}}(t)$ (Sección 6).

**2.8 Por qué** $\mathbf{\alpha}$ **importa operacionalmente**

-   **Comparabilidad**: $\alpha$ es invariante a cambios de unidad y gastos generales constantes, permitiendo comparación **entre laboratorios** y **entre generaciones**.

-   **Alerta temprana**: **caídas** significativas en $\alpha$ (por familia o fusionado) señalan **eventos de decoherencia** que probablemente preceden picos en error lógico, makespan, o recalibraciones forzadas.

-   **Palanca de diseño**: elevar $\alpha$ (sin sobre-estratificar) vía **programación**, **cadencia QEC**, o **dimensionamiento de módulo** mejora el amortiguamiento de cascadas entre escalas.

**2.9 Resumen**

RTM en QC se reduce a tres declaraciones por bin: (i) escalamiento de **ley de potencia** $T = \kappa L^{\alpha}$, (ii) **invariancia de gauge** (solo la pendiente $\alpha$ es estructural), y (iii) **colapso** como prueba de especificación falsificable. Con binning cuidadoso y estimación consciente de EIV, $\alpha$ se convierte en un **exponente de coherencia** reproducible y robusto a unidades que guía tanto **diagnósticos** como **diseño** a través de la pila cuántica.

**3. Geometría Escala--Reloj para QC (Colapso como Exactitud)**

Replanteamos RTM para pilas cuánticas en forma geométrica. El objeto clave es la **1-forma RTM**

$$\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha(x)\text{ }d(\log L),
$$

definida en un bin $E$ con coordenadas de **ambiente** $x$ (temperatura, firmware, topología, programa de síndrome, banda de utilización) y **escala** $u = \log L$. En este lenguaje, **colapso** es equivalente a **exactitud/planitud** de $\omega$; las costuras de régimen y la curvatura no-potencia aparecen como **holonomía/curvatura**. Esta sección declara los resultados y los instancia con modos de falla QC.

**3.1 Espacios, bins, y la 1-forma RTM**

-   **Espacio de estados.** $M = X \times \mathbb{R}$ con coordenadas $(x,u)$, donde $u = \log L$.

-   **Potencial de reloj.** $v(x,u) = \log T(x,L)$.

-   **1-forma RTM.** $\omega = dv - \alpha(x)\text{ }du$ (caso $\alpha$ constante) o $\omega = dv - \alpha(x,u)\text{ }du$ (deriva lenta permitida).

**Un cambio de reloj** (desplazamiento de unidad/línea base independiente de $L$ dentro de un bin) es:

``` math
v \mapsto v^{\#} = v + \phi(x).
```

Entonces

``` math
$$
\omega \mapsto \omega^{\#} = \omega + d\phi(x)
$$
```
una **transformación de gauge** por una 1-forma exacta traída de $X$. Por tanto $\alpha$ **es invariante de gauge**.

**3.2 Colapso ⇔ exactitud/planitud**

**Teorema 3.1 (Colapso** $\Leftrightarrow$ **exactitud).**\
En un bin simplemente conexo $E$, los siguientes son equivalentes:

1.  (Carta RTM) $v(x,u) = \alpha(x)\text{ }u + \log\kappa(x)$ (o $v = \int\alpha(x,s)\text{ }ds + \log\kappa(x)$ para deriva lenta).

2.  (**Colapso**) El residuo $\widetilde{v}: = v - \alpha u$ es independiente de $u$ en $E$.

3.  (**Exactitud**) $\omega = d\psi$ en $E$ para algún $\psi(x)$ (sin dependencia de $u$).

**Corolario 3.2 (Prueba de planitud).**\
$d\omega = 0$ es necesario y (en $E$ simplemente conexo) suficiente para el colapso. Con $\alpha = \alpha(x,u)$,

$$d\omega\text{\:\,} = \text{\:\,} - \text{ }d\alpha \land du.
$$

Así la curvatura (comportamiento no-potencia) o la mezcla de regímenes da $d\alpha/\text{ }du \neq 0$ y **rompe el colapso**.

**3.3 Holonomía y costuras de régimen (modos de falla QC)**

Defina la **holonomía** alrededor de un lazo cerrado $\gamma \subset E$: $\mathcal{H(}\gamma) = \oint_{\gamma}^{}{\omega.\ }$ Si $\mathcal{H(}\gamma) \neq 0$, el colapso no puede mantenerse globalmente.

**Instancias QC.**

-   **Costura de programador.** Cambiar la cadencia de extracción de síndrome a mitad de bin (nueva imagen FPGA) produce un quiebre en $v(u)$; los lazos que cruzan la costura recogen holonomía no nula → **rebinear**.

-   **Intercambio de cadena de lectura.** Un gasto general por canal que *depende de la multiplexación* se comporta como un reloj dependiente de escala $c(L)$; esto **no es gauge** e inyecta componentes $du$ → el colapso falla (y debería).

-   **Ventana de deriva térmica.** Una rampa de utilización lenta cambia $\alpha$ a través de $u$; si $\partial_{u}\alpha$ no es pequeño en la ventana, $d\omega \neq 0$ → dividir el bin o reducir la ventana.

**3.4 Colapso adiabático (** $\mathbf{\alpha}$ **lentamente variable)**

Si $\mid \partial_{u}\alpha \mid \leq \varepsilon$ en una ventana de ancho $h$,

$$\widetilde{v}(x,u) = v - \alpha(u_{0},x)\text{ }u = \log\kappa(x) + O(\varepsilon h),
$$

y la estadística de colapso empírica obedece

$$R_{\text{colapso}}^{2} = O\text{ }((\varepsilon h)^{2}).
$$

**Práctica.** Elija $h$ tal que $\varepsilon h \ll 1$; de lo contrario, reduzca el bin o modele la deriva explícitamente.

**3.5 Morfismos (reparametrizaciones) y gauge**

Sea $\Phi = (\varphi,\psi)$ mapea $(X_{A},L_{A},v_{A}) \rightarrow (X_{B},L_{B},v_{B})$, donde $\varphi:X_{A} \rightarrow X_{B}$ reparametriza el ambiente y $\psi:X_{B} \rightarrow \mathbb{R}$ es un cambio de reloj. Entonces

$$\Phi^{*}\omega_{B}\text{\:\,} = \text{\:\,}\omega_{A}\text{\:\,} + \text{\:\,}d(\psi \circ \varphi).
$$

Interpretación: transportar la estructura de $B$ a $A$ preserva la **pendiente** y altera solo el **reloj** por una forma exacta. Esto formaliza las comparaciones entre laboratorios/dispositivos cuando las unidades/líneas base difieren.

**3.6 Diagnósticos y puertas de aceptación (lista de verificación QC)**

1.  **Prueba de colapso.** Ajuste $\widehat{\alpha}$ (Sección 5), calcule los residuos $\widetilde{y} = y - \widehat{\alpha}x$; requiera\
    $R_{\text{colapso}}^{2} < 0.05$ **y** sin tendencia en un suavizado no paramétrico.

2.  **Placebo de reloj.** Multiplique todos los $T$ por una constante; $\widehat{\alpha}$ y $R_{\text{colapso}}^{2}$ deben permanecer sin cambios.

3.  **Puntos de cambio.** Ejecute detectores en $(x,y)$ y en $\widetilde{y}$; cualquier quiebre ⇒ rebinear.

4.  **Control de ventana.** Asegure que $\mid \partial_{u}\alpha \mid \text{ }h$ sea pequeño (régimen adiabático).

5.  **Publicar/retener.** Solo los bins que pasan 1--4 contribuyen a ${ECI}_{QC}$(t); de lo contrario etiquete NO_COLAPSO o MEZCLA_REGIMEN.

**3.7 Qué nos compra esto operacionalmente**

-   Una **obligación de prueba**: mostrar planitud/exactitud (colapso) antes de confiar en una pendiente.

-   Un **depurador**: la holonomía no nula localiza costuras (intercambios de programador, cambios de lectura).

-   Una **regla de ajuste**: reduzca $h$ o rebinee hasta que $d\omega \approx 0$; si es imposible, el dominio es **no-potencia**---trate $\alpha$ como indefinido allí.

**3.8 Resumen**

La geometría escala--reloj hace precisas dos declaraciones RTM para QC:

1.  $\alpha$ **es una cantidad estructural invariante de gauge**, no afectada por cambios de unidad/línea base;

2.  **Colapso equivale a exactitud/planitud de** $\omega$, y su fallo es informativo (curvatura o costuras).\
    Ahora aprovecharemos esto para definir $(L,T)$ **operacionales** (Sec. 4) y para estimar $\widehat{\alpha}$ robustamente bajo error de medición (Sec. 5).

**4. Definiciones Operacionales de** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$ **y Protocolo de Binning**

Esta sección convierte RTM en **práctica medible** para pilas cuánticas. Definimos pares $(L,T)$ específicos por capa, especificamos **muestreo**, **unidades**, y **guardias**, y damos un protocolo de binning que evita la mezcla de regímenes. A lo largo, $u = \log L$, $v = \log T$.

**4.1 Principios de diseño para** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$

-   **Un mecanismo por familia.** Cada par $(L,T)$ debería reflejar un mecanismo dominante único (ej., pipeline de lectura, no una mezcla de lectura + enrutamiento).

-   **$L$ monótono.** $L$ debería aumentar con el "tamaño del problema" en esa capa (ancho, distancia, canales, tamaño de cluster).

-   **Independencia de reloj.** Dentro de un bin, los cambios de base de tiempo **multiplicativos** ($T^{'} = cT$) son gauges permitidos (reescalamientos de unidad/base de tiempo). Los gastos generales **aditivos** ($T_{\text{obs}} = T + b$) deben restarse, modelarse, o evitarse (ajustar solo donde $T \gg b$); de lo contrario pueden sesgar las pendientes e invalidar el colapso.

-   **Muestreo estable.** Use recolección de **cadencia fija**; registre marcas de tiempo crudas para permitir re-segmentación.

**4.2 Capa física**

**Candidatos para** $L$**:**

-   $L =$ número de **qubits activos** en la ventana de carga de trabajo;

-   $L =$**tamaño de cluster** (qubits conectados participando simultáneamente);

-   $L =$**grado de acoplador** (fanout promedio).

**Candidatos para** $T$**:**

-   **Intervalo de calibración estable** (tiempo hasta que cualquier qubit en el cluster salga de tolerancia);

-   **Latencia de compuerta** (duración mediana de compuerta de uno/dos qubits a través del conjunto activo);

-   **Latencia de lectura** (tiempo por disparo mediano hasta símbolo válido bajo umbrales fijos);

-   **Tiempo medio hasta deriva** (MTTD) para frecuencia/fase.

**Instrumentación.**

-   Registre marcas de tiempo por disparo; un vigilante de calibración registrando cuando se violan los umbrales; adjunte etiquetas de ambiente: banda de temperatura, hash de firmware, punto de polarización.

**No-ejemplos.**

-   Mezclar *tanto* latencia de compuerta como latencia de lectura en el mismo $T$.

-   Dejar que $L$ sea "qubits definidos en el chip" (no necesariamente activos).

**4.3 Corrección de Errores (QEC)**

$L$**:** **distancia** de código $d$ (primario), o número de **qubits lógicos** a $d$ fijo.\
$T$**:**

-   **Ciclos hasta falla lógica** a un error objetivo fijo (mediana o cuantil de supervivencia);

-   **Latencia de ciclo de síndrome** (tiempo medio por ciclo bajo programa fijo).

**Notas de programación.**

-   Congele un **programa de síndrome** (imagen FPGA + cadencia). Cualquier cambio ⇒ nuevo bin.

-   Registre el sesgo (X/Z) y la configuración de mitigación de fuga.

**Casos límite.**

-   Si $T$ está dominado por **eventos catastróficos raros** (ej., enganche de resonador), prefiera **medianas condicionales** (excluya banderas catastróficas conocidas) y reporte un panel de sensibilidad.

**4.4 Compilador / Tiempo de ejecución**

$L$**:** **ancho** de circuito (máximo de qubits concurrentes) o **profundidad post-mapeo**; opcionalmente **capas activas** después de enrutamiento.\
$T$**:**

-   **Makespan** (envío → completación);

-   **Retraso de cola** (envío → inicio);

-   **Latencia de reprogramación** después de un evento de calibración.

**Controles.**

-   Fije la **política de enrutamiento** y la **heurística de colocación** dentro de un bin.

-   Estratifique por banda de utilización (ej., 0--30%, 30--60%, \>60%). Si la utilización deriva, divida el bin.

**4.5 I/O -- Cryo / Lectura**

$L$**:** **grado de multiplexación** (canales por línea) o número de canales de lectura concurrentes.\
$T$**:**

-   **Latencia de lectura** (mediana p50 y cola p95);

-   **Vida media de recuperación de BER** después de una ráfaga controlada;

-   **Cola p95** expresada en tiempo.

**Instrumentación.**

-   Marque la hora de cada ráfaga DMA/ADC; registre buffers por canal; anote versiones de firmware de DSP.

**Advertencia.**

-   Los gastos generales por canal que **crecen con** $L$ *no* son gauges; son efectos de escala genuinos---permisibles para RTM---pero si el gasto general mismo cambia a mitad de bin, el colapso debería fallar y disparar una división.

**4.6 Protocolo de binning (fijación de ambiente)**

Un **bin** es un intervalo máximo donde el ambiente es efectivamente constante.

**Clave de bin (ejemplo):**

$$\text{BIN} = \{\text{plataforma},\text{ banda de temperatura},\text{ hash de firmware},\text{ ID de topología},\text{ política de enrutamiento},\text{ cadencia de síndrome},\text{ banda de utilización}\}.
$$

**Procedimiento.**

1.  **Segmente** los datos por BIN; descarte segmentos con < $N_{\mathrm{min}}$ valores distintos de $L$ (por defecto 6).

2.  **Escaneo de punto de cambio** en $y = \log T$ vs. $x = \log L$ (y en residuos si están disponibles). Si se detecta un punto de cambio (BIC/AIC/PELT), **divida**.

3.  **Ventaneo**: para regímenes que derivan lentamente, use ventanas deslizantes en $x$ de ancho $h$ tal que $\mid \partial_{u}\alpha \mid \text{ }h \ll 1$ (de Sec. 3.4).

4.  **Placebo de reloj**: multiplique $T$ por una constante; la pendiente $\widehat{\alpha}$ no debe cambiar.

**4.7 Conjunto de datos listo para estimación**

Cree una tabla ordenada por bin con columnas:

$$x = log\ L,\ y = \log T,\text{ familia},\text{ etiquetas BIN},\text{ ID de réplica},\text{ marca de tiempo},\text{ pesos }\rbrack.
$$

-   **Réplicas.** Si hay múltiples corridas al mismo $L$, agregue a resúmenes robustos (mediana $y$, SE basado en MAD) o pase todas y deje que ODR las maneje con pesos de réplica.

-   **Pesos.** Prefiera pesos de varianza inversa de bootstrap sobre conteos simples.

-   **Valores atípicos.** Etiquete eventos catastróficos (banderas de hardware); reporte tanto **con** como **sin** ellos.

**4.8 Puertas de aceptación (por bin, por familia)**

Una familia contribuye una pendiente ${\widehat{\alpha}}_{f}$ **solo si** todo se cumple:

1.  **Cobertura:** al menos $6$ puntos distintos de $L$ y span $\geq 0.6$ en $\log L$.

2.  **Colapso:** regresione $\widetilde{y} = y - \widehat{\alpha}x$ sobre $x$; requiera $R_{\text{colapso}}^{2} < 0.05$ y sin tendencia visible (verificación suave).

3.  **Placebo de reloj:** $\widehat{\alpha}$ sin cambios bajo $T \mapsto cT$.

4.  **Puntos de cambio:** ninguno dentro del bin (de lo contrario divida y re-estime).

5.  **Calidad de ajuste EIV:** ODR/TLS convergió; diagnósticos residuales aceptables (ningún punto de palanca único domina).

Los bins o familias que fallan cualquier puerta se etiquetan (NO_COLAPSO, MEZCLA_REGIMEN, COBERTURA_DELGADA, FALLA_EIV) y **se excluyen de la fusión**.

**4.9 Ejemplos vs. no-ejemplos (con sabor QC)**

-   **Buena familia física:** $L =$ tamaño de cluster de qubits activos; $T =$ intervalo de calibración estable. Firmware único, temperatura estable, sin cambio de enrutamiento. Colapsa limpiamente → aceptar.

-   **Mala familia física:** Lo mismo, pero a mitad de bin los parámetros del bucle PLL cambian. Se dispara punto de cambio; división requerida.

-   **Buena familia QEC:** $L = d$, $T =$ ciclos hasta falla lógica, cadencia de síndrome fija. Residuos planos → aceptar.

-   **Mala familia QEC:** Mezcla de dos cadencias (rápida y lenta) dentro de un bin → quiebre en log--log → rechazar hasta dividir.

-   **Buena familia I/O:** $L =$ grado de multiplexación; $T =$ latencia de lectura p95. Firmware constante; la latencia sube como $L^{\alpha}$, el colapso se mantiene → aceptar.

-   **Mala familia I/O:** Cambio de firmware DSP que cambia el gasto general por canal no linealmente a mitad de bin → curvatura; rechazar o rebinear alrededor del cambio.

**4.10 Resumen**

-   Fijamos $(L,T)$ **operacionales** por capa y especificamos **instrumentación** para hacerlos medibles.

-   Definimos un **protocolo de binning** que impone constancia del ambiente y protege contra mezcla de regímenes.

-   Establecimos **puertas de aceptación** (cobertura, colapso, placebo, puntos de cambio, ajuste EIV) que determinan si la pendiente de una familia entra en la fusión descendente (${ECI}_{QC}$(t)).

**5. Estimación Bajo Errores en Variables (EIV) y Umbrales de Colapso**

Ahora especificamos **cómo** estimar la pendiente por bin $\alpha$ robustamente cuando ambos ejes son ruidosos, y cómo decidir---vía un **umbral de colapso**---si los datos de una familia son consistentes con RTM. A lo largo, $x = \log L$, $y = \log T$. Las observaciones son $x^{obs} = x + \xi$, $y^{obs} = y + \zeta$ con errores de media cero.

**5.1 Objetivos de estimación y modelos**

Dentro de un **bin fijo**, el objetivo es la **pendiente local** $\alpha$ en

$$y = \alpha x + c + r(x),
$$

con $r \equiv 0$ bajo RTM exacto o $\mid r^{'}(x) \mid \leq \varepsilon$ bajo deriva lenta en una ventana. Porque $x$ es ruidoso, **OLS está atenuado**; usamos estimadores conscientes de EIV.

**Objetivo por defecto:** pendiente puntual $\alpha$ para el bin; la ordenada al origen $c$ es un **gauge** (no se compara entre bins).

### 5.2 Regresión de Distancia Ortogonal (Mínimos Cuadrados Totales)

**Definición.** ODR minimiza los residuos ortogonales a una línea:

$$
\min_{\alpha,c} \sum_{i} \frac{(y_i^{\text{obs}} - \alpha x_i^{\text{obs}} - c)^2}{\sigma_y^2 + \alpha^2\sigma_x^2}
$$

con $(\sigma_x, \sigma_y)$ efectivos (posiblemente heterogéneos) de varianza de réplica o bootstrap.

**Práctica.**

-   Inicialice con Theil--Sen (Sec. 5.4) para evitar mínimos locales pobres.

-   Use **bootstrap de cluster/parejas** (réplica o nivel de trabajo) para ICs.

-   Si hay SEs por punto disponibles, péselos; de lo contrario use pesos robustos de Huber sobre residuos ortogonales.

**Puertas de convergencia.**

-   Número de condición de la matriz de covarianza centrada $< 10^{4}$.

-   Verificación de palanca jackknife: ningún punto individual contribuye $> 25\%$ de la influencia de la pendiente.

**5.3 SIMEX (cuando** $\mathbf{Var}\mathbf{(}\mathbf{\xi}\mathbf{)}$ **es conocida/estimada)**

Si puede estimar $\sigma_{\xi}^{2} = Var(\xi)$ (ej., $L$ repetido en la misma configuración), aplique **SIMEX**:

1. Para $\lambda \in \Lambda = \{0.5, 1.0, 1.5, 2.0\}$, genere pseudo-muestras
``` math
$$x_i^{(\lambda)} = x_i^{obs} + \sqrt{\lambda} {\tilde{\xi}}_i, \quad {\tilde{\xi}}_i \sim \mathcal{N}(0, \sigma_\xi^2).$$
```

3.  Ajuste una pendiente ingenua $\widehat{\alpha}(\lambda)$ por ODR u OLS.

4.  Ajuste una cuadrática $\widehat{\alpha}(\lambda) = a + b\lambda + c\lambda^{2}$ y **extrapole a** $\lambda = - 1$:\
    ${\widehat{\alpha}}_{\text{SIMEX}} = a - b + c$.

**Uso.** Prefiera ODR como la rutina de ajuste base; reporte SIMEX como una estimación de **sensibilidad** junto a ODR. Si $\sigma_{\xi}^{2}$ es incierto, dé una banda (bajo/med/alto) para ${\widehat{\alpha}}_{\text{SIMEX}}$.

**5.4 Theil--Sen (pendiente mediana robusta)**

La pendiente **Theil--Sen** es la mediana de todas las pendientes por pares

$$\alpha_{ij} = \frac{y_{j}^{obs} - y_{i}^{obs}}{x_{j}^{obs} - x_{i}^{obs}}(i < j),
$$

con una ordenada al origen robusta de la mediana de $y_{i}^{obs} - \widehat{\alpha}x_{i}^{obs}$.

**Rol.**

-   Inicialización para ODR.

-   Verificación cruzada **robusta a valores atípicos** reportada junto a ODR.

-   Cuando EIV es severo y $\sigma_{\xi}^{2}$ es desconocido, Theil--Sen puede seguir siendo estable (espere atenuación leve).

**5.5 Ventaneo y sesgo de ventana finita**

Si se sospecha deriva lenta, estime pendientes en **ventanas simétricas** en $x$ de ancho $h$. Del límite de sesgo adiabático,

$$\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h),
$$

elija $h$ tal que $\varepsilon h \ll 1$. Prácticamente: comience con $h \approx 0.8$ en span de $\log L$ si la cobertura lo permite; reduzca hasta que el colapso pase (Sec. 5.7) sin explotar la varianza.

**5.6 Incertidumbre y diagnósticos**

-   **Bootstrap** (parejas dentro de bin o bloque/cluster si existen réplicas naturales) para ICs 50/95%.

-   **Jackknife-después-de-bootstrap** para detectar puntos de palanca.

-   **Gráficos residuales**: residuo ortogonal vs. $x$; el suavizado LOESS debe ser plano dentro de bandas.

-   **Adecuación de EIV**: si OLS y ODR difieren por $\geq$ 0.2 pendiente absoluta **y** el IC de ODR excluye OLS, reporte EIV como material.

**5.7 Umbral de colapso (puerta de especificación)**

Dado $\hat{\alpha}$, calcule los residuos $\tilde{y}_i = y_i^{obs} - \hat{\alpha}x_i^{obs} - \hat{c}$ y regresione $\tilde{y}$ sobre $x$ (con los mismos pesos usados en la estimación). Defina

$$R_{\text{colapso}}^{2}: = R^{2}(\widetilde{y} \sim x).
$$

**Regla de decisión (por defecto):**

-   Acepte el bin si **todo** se cumple:

    1.  $R_{\text{colapso}}^{2} < 0.05$ (o el IC al 95% de la pendiente en $\widetilde{y} \sim x$ contiene 0),

    2.  El suavizado LOESS no muestra tendencia,

    3.  **Placebo de reloj**: escalar $T \mapsto cT$ deja $\widehat{\alpha}$ y $R_{\text{colapso}}^{2}$ sin cambios,

    4.  El escaneo de punto de cambio (PELT/BIC) no encuentra ninguno dentro del bin.

-   De lo contrario etiquete (NO_COLAPSO o MEZCLA_REGIMEN) y **no** publique una pendiente ni la incluya en la fusión.

**5.8 Puertas de cobertura y palanca**

Para evitar ajustes frágiles:

-   **Puntos** $L$ **distintos** $\geq 6$ y span de $\log L$ $\geq 0.6$.

-   **Palanca equilibrada:** el punto de palanca más grande contribuye $\leq 25\%$ de la influencia de la pendiente ODR.

-   **Réplicas:** si hay $> 3$ réplicas por $L$, ya sea resuma a una media/SE robusta o pase pesos de réplica a ODR.

Los bins que fallan estas puertas se etiquetan COBERTURA_DELGADA o RIESGO_PALANCA.

**5.9 Juntándolo todo (algoritmo por bin)**

1.  **Preparación:** construya la tabla ordenada (Sec. 4.7); ejecute escaneo de punto de cambio; ventanee si es necesario.

2.  **Inicialización:** calcule pendiente/ordenada al origen Theil--Sen; remueva catastróficos obvios (mantenga ambas versiones para sensibilidad).

3.  **Ajuste ODR/TLS:** ponderado por SEs de réplica; obtenga $\widehat{\alpha}$, $\widehat{c}$, ICs bootstrap.

4.  **SIMEX (opcional):** si $\sigma_{\xi}^{2}$ está disponible, calcule ${\widehat{\alpha}}_{\text{SIMEX}}$.

5.  **Puerta de colapso:** calcule $R_{\text{colapso}}^{2}$, verificación suave, placebo de reloj.

6.  **Decisión:** si todas las puertas pasan, **acepte** $\widehat{\alpha}$ con incertidumbre; de lo contrario **rechace/divida**.

7.  **Reporte:** pendiente, IC, diagnósticos (colapso $R^{2}$, gráfico de palanca, puntos de cambio). Almacene banderas.

**5.10 Qué publicamos por familia aceptada**

-   ${\widehat{\alpha}}_{f} \pm$`<!-- -->`{=html}IC 50/95% (ODR); Theil--Sen como robustez; banda SIMEX si aplica.

-   Diagnósticos de colapso: $R_{\text{colapso}}^{2}$, verificación de placebo, ancho de ventana $h$.

-   Cobertura: \# distintos $L$, span de $\log L$, resumen de palanca.

-   Notas: cualquier exclusión (catastróficos), estado de punto de cambio.

Solo las familias aceptadas entran en **fusión** (Sec. 6). Si $\geq 2$ familias pasan, aplicamos efectos aleatorios con $Q$, $I^{2}$ y puertas de heterogeneidad; de lo contrario reportamos pendientes por familia sin fusión.

**5.11 Resumen**

-   Use **ODR/TLS** como el estimador EIV primario; **Theil--Sen** para inicialización/verificación robusta; **SIMEX** cuando $\sigma_{\xi}^{2}$ es estimable.

-   Haga cumplir el **colapso** como una **prueba de especificación** ($R_{\text{colapso}}^{2} < 0.05$ + placebo + sin puntos de cambio).

-   Controle el **sesgo de ventana finita** eligiendo $h$ lo suficientemente pequeño (régimen adiabático) y dividiendo bins cuando sea necesario.

-   Publique **diagnósticos** y **banderas** completos; solo las familias limpias proceden a la fusión y al ${ECI}_{QC}$(t) en tiempo real.

**6. Construyendo el Indicador en Tiempo Real** $\mathbf{ECI}_{\mathbf{QC}}\mathbf{(}\mathbf{t}\mathbf{)}$

Ahora construimos un indicador de coherencia **único, en tiempo real** para una plataforma fusionando las pendientes por familia **aceptadas** $\{{\widehat{\alpha}}_{f,t}\}$ de la Sección 5. La fusión es de **efectos aleatorios** (para reconocer la heterogeneidad entre familias), corre en un reloj rodante, e impulsa **puertas de QA** y **alertas de decoherencia**.

**6.1 Entradas y precondiciones (por tiempo** $\mathbf{t}$**)**

Para cada familia $f \in \mathcal{F}_{t}$ (Física, QEC, Compilador/Tiempo de ejecución, I/O--Cryo):

-   Una estimación por bin $\hat{\alpha}\_{f,t}$ con varianza $\hat{\sigma}\_{f,t}^2$ (bootstrap o ponderada por réplica),

-   Colapso pasado (Sección 5.7), puertas de cobertura/palanca satisfechas (Sección 5.8),

-   Etiquetas de ambiente (BIN) sin cambios dentro de la ventana que produjo ${\widehat{\alpha}}_{f,t}$.

Una fusión en tiempo $t$ procede **solo si** $\mid \mathcal{F}_{t} \mid \geq 2$.

**6.2 Fusión de efectos aleatorios**

Estimamos la varianza entre familias ${\widehat{\tau}}_{t}^{2}$ (por defecto **REML**; DerSimonian--Laird como sensibilidad). Defina pesos

$$w_{f,t}\text{\:\,} = \text{\:\,}\frac{1}{{\widehat{\sigma}}_{f,t}^{2} + {\widehat{\tau}}_{t}^{2}}.
$$

Entonces la pendiente fusionada y su varianza son

$${\widehat{\alpha}}_{QC}(t) = \frac{\sum_{f \in \mathcal{F}_{t}}^{}{w_{f,t}\text{ }{\widehat{\alpha}}_{f,t}}}{\sum_{f \in \mathcal{F}_{t}}^{}w_{f,t}},\ \ Var({\widehat{\alpha}}_{QC}(t)) = \frac{1}{\sum_{f \in \mathcal{F}_{t}}^{}w_{f,t}}.
$$

Reporte intervalos 50% y 95% vía aproximación normal o por un **bootstrap-sobre-familias** (remuestree familias con reemplazo, recalcule ${\widehat{\tau}}_{t}^{2}$ y la media fusionada).

**6.3 Diagnósticos de heterogeneidad y puertas**

Calcule la línea base de efecto fijo

$$w_{f,t}^{FE} = \frac{1}{{\widehat{\sigma}}_{f,t}^{2}},\ \ {\widehat{\alpha}}_{FE}(t) = \frac{\sum_{f}^{}{w_{f,t}^{FE}\text{ }{\widehat{\alpha}}_{f,t}}}{\sum_{f}^{}w_{f,t}^{FE}}.
$$

**Q de Cochran** y $I^{2}$**:**

$$Q_{t} = \sum_{f}^{}{w_{f,t}^{FE}\text{ }({\widehat{\alpha}}_{f,t} - {\widehat{\alpha}}_{FE}(t))^{2},\ \ I_{t}^{2} = \max}\{ 0,\text{\:\,}\frac{Q_{t} - ( \mid \mathcal{F}_{t} \mid - 1)}{Q_{t}}\} \times 100\%.
$$

**Puertas de fusión (pre-registradas):**

-   Proceda con un número único **solo si**\
(i) $|\mathcal{F}\_t| \geq 2$,  
(ii) $I\_t^2 < 50\%$ (*heterogeneidad moderada o menor*), y  
(iii) REML converge con $\hat{\tau}\_t^2$ finito que no excede un tope histórico (ej., $\leq$ percentil 90 sobre ventanas limpias pasadas).  

-   Si alguno falla, **retenga la fusión** y publique ${\widehat{\alpha}}_{f,t}$ por familia + diagnósticos; etiquete DIVERGENCIA_FAMILIAS.

**6.4 Operación en tiempo real (ventanas rodantes)**

-   **Cadencia.** Recalcule el ${\widehat{\alpha}}_{f,t}$ de cada familia en una **ventana rodante** en $x = \log L$ de ancho $h$ (elegido por la regla adiabática; Sec. 5.5) y un **horizonte de tiempo de reloj** (ej., últimos 7--28 días de datos).

-   **Rellenado y faltantes.** Si una familia falta en $t$, fusione sobre el $\mathcal{F}\_t$ disponible siempre que $|\mathcal{F}\_t| \geq 2$; de lo contrario **suspenda** $ECI\_{\text{QC}}(t)$ y publique una bandera `FAMILIAS_DELGADAS`.

-   **Placebo de reloj.** Una vez al día, multiplique todos los $T$ contribuyentes por una constante y verifique que $\hat{\alpha}_{\text{QC}}(t)$ e $I_t^2$ permanezcan sin cambios (almacenado como un artefacto de QA).

**6.5 Eventos de decoherencia (lógica de alerta)**

Definimos un **evento de decoherencia** como una **caída** significativa, con QA limpio en ${ECI}_{QC}(t)$, robusta al suavizado y no explicada por picos de heterogeneidad.

**Filtros:**

1.  **Suavizado:** mantenga una mediana de 3 puntos $\widetilde{\alpha}(t)$ de ${\widehat{\alpha}}_{QC}(t)$.

2.  **Puntuación Z:** $Z(t) = \frac{\tilde{\alpha}(t) - \text{EWMA}\_{30}[\tilde{\alpha}]}{\hat{\sigma}\_{\text{EWMA}}(t)}$

**Niveles de alerta (por defecto):**

-   **Aviso:** $Z(t) \leq - 1.5$ por ≥2 ticks consecutivos **y** $I_{t}^{2} < 50\%$.

-   **Vigilancia:** $Z(t) \leq - 2.0$ una vez **o** persistente $Z(t) \leq - 1.5$ por ≥4 ticks, $I_{t}^{2} < 40\%$.

-   **Advertencia:** $Z(t) \leq - 2.5$ y una caída coincidente por familia (≥2 familias con $Z_{f} \leq - 2$).

**Manuales de juego disparados:** estrangule la programación (reduzca concurrencia/multiplexación), ejecute recalibración segmentada, o cambie a enrutamiento consciente de RTM hasta que $\widetilde{\alpha}(t)$ se normalice.

**6.6 Reporte y visualización**

-   **Panel primario:** $\hat{\alpha}\_{\text{QC}}(t)$ con bandas 50/95%, cinta de heterogeneidad coloreada por $I\_t^2$ (verde <25%, ámbar 25-50%, rojo $\geq$ 50%).

-   **Gráfico de bosque:** $\hat{\alpha}\_{f,t}$ por familia, pesos $w\_{f,t}$, e ICs; muestre $Q\_t$, $I\_t^2$, $\hat{\tau}\_t^2$.

-   **Panel de colapso:** por familia, muestre $R_{\text{colapso}}^{2}$, residuos LOESS, ancho de ventana $h$, métricas de cobertura y palanca.

-   **Leyenda de banderas:** NO_COLAPSO, MEZCLA_REGIMEN, RIESGO_PALANCA, COBERTURA_DELGADA, DIVERGENCIA_FAMILIAS, FAMILIAS_DELGADAS.

**6.7 Sensibilidad y ablación**

-   Publique el resumen de **efecto fijo** ${\widehat{\alpha}}_{FE}(t)$ junto con efectos aleatorios.

-   Reporte ${\widehat{\tau}}_{DL}^{2}$ basado en DL como sensibilidad.

-   **Dejar-una-familia-fuera**: recalcule ${\widehat{\alpha}}_{QC}^{( - f)}(t)$ para exponer dominancia.

-   Los **placebos de reloj** y **nulos de mezcla** (mezcle $L$ dentro de familia) no deben producir alertas escalonadas; si lo hacen, revise las puertas.

**6.8 Gobernanza y procedencia**

Cada punto fusionado almacena:

-   Familias fuente y sus etiquetas BIN,

-   Configuraciones de estimador (inicialización ODR, semillas bootstrap, $h$),

-   Métricas de colapso, $Q_{t}$, $I_{t}^{2}$, ${\widehat{\tau}}_{t}^{2}$,

-   Hashes de resultado de placebo,

-   Código/configuración versionados (YAML de métodos).

Esto asegura **reproducibilidad** y permite post-mortems cuando se disparan alertas.

**6.9 Resumen**

$ECI\_{\text{QC}}(t)$ es una **fusión de efectos aleatorios** de pendientes por bin con QA limpio. Las puertas de heterogeneidad ($I\_t^2 < 50\%$, $|\mathcal{F}\_t| \geq 2$) previenen números únicos engañosos cuando los proxies discrepan. El suavizado en tiempo real y las puntuaciones Z convierten la dinámica de pendiente en **alertas accionables** para **eventos de decoherencia**, mientras que los paneles y la procedencia mantienen el sistema auditable.

**7. Diseño Consciente de RTM: Ingeniería de** $\mathbf{\alpha}$ **sin Sacrificar Rendimiento**

Esta sección convierte RTM en **palancas de diseño**. Objetivo: aumentar el **exponente de coherencia** $\alpha$ (estratificación de tempo más fuerte a través de la escala) mientras mantiene o mejora el rendimiento. Damos controles específicos por capa, objetivos de optimización, y barandillas.

**7.1 Objetivo de diseño y barandillas**

Tratamos $\alpha$ como un **objetivo operacional** dentro de un bin:\
$$\max_{\text{\:\,controles }\theta}\ \ \ \alpha(\theta)\ \ \ s.t.\ \ \ \ rendimiento\  \geq \ B,\ \ fidelidad\  \geq \ F,\ \ \ \ \ colapso\ pasa.$$

-   **Controles** $\theta$: parámetros de programador, cadencia/jitter QEC, restricciones de enrutamiento, límites de multiplexación, tamaños de módulo.

-   **Restricciones**: un piso de rendimiento $\mathcal{B}$ (ej., trabajos/hora), piso de fidelidad $\mathcal{F}$, y **puertas de colapso** (Sec. 5.7).

-   **Monitor:** rastree $\hat{\alpha}\_f$ por familia y el $\hat{\alpha}\_{\text{QC}}(t)$ fusionado con QA (Sec. 6).

**7.2 Programador: agrupamiento y enrutamiento consciente de varianza**

**Problema.** Las operaciones largas, fuertemente acopladas lanzadas en paralelo **aplanan** $\alpha$ (cascadas rápidas a través de la escala).

**Controles.**

1.  **Agrupamiento de frente de onda (lectura y ops largas).** Particione el tiempo en olas cortas; empaque las lecturas en olas en lugar de concurrencia libre.

2.  **Reinicios escalonados.** Añada pequeños desplazamientos $\delta \in \lbrack - \epsilon,\epsilon\rbrack$ a los tiempos de reinicio para evitar picos de sincronización.

3.  **Enrutamiento de baja varianza.** Prefiera rutas con **baja varianza de tiempo de ruta** incluso si la longitud de ruta aumenta ligeramente.

**Objetivo.** Para un DAG de trabajo con ops $o$ que tienen duraciones nominales $\tau_{o}$ y rutas $p(o)$:

$$\underset{\text{\:\,programa},\text{ }p( \cdot )}{\min}\text{\:\,}\underset{\text{desincronizar ops pesadas}}{\underbrace{{Var}_{t}\lbrack N_{\text{largo}}(t)\rbrack}}\text{\:\,} + \text{\:\,}\lambda\text{\:\,}\underset{\text{enrutamiento de baja varianza}}{\underbrace{\sum_{o \in \mathcal{O}}^{}{Var(T_{\text{ruta}}(p(o)))}}}.
$$

sujeto a presupuesto de makespan. Esto reduce "apilamientos" temporales, elevando $\alpha$.

**Heurística (voraz, práctica).**

-   Ordene ops por duración desc; asigne tiempos de inicio en **olas** de modo que la carga total de ops largas de cada ola esté balanceada.

-   Para cada ruta candidata, penalice la varianza de tiempo y la puntuación de diafonía; elija el costo penalizado mínimo.

**7.3 Cadencia QEC: evitar el bloqueo de fase (jitter/desincronización)**

**Problema.** Una cadencia de síndrome fija puede **bloquear fase** con ritmos de ruido físico, creando sincronización entre capas → $\alpha_{QEC}$ cae.

**Controles.**

-   **Micro-jitter** el período de ciclo: $P_{k} = P\text{ }(1 + \eta_{k})$ con $\eta_{k} \sim \mathcal{U}\lbrack - \rho,\rho\rbrack$, $\rho \ll 1$ (ej., 1--3%).

-   **Extracción multi-fase:** divida el código en subretículas cuyos ciclos están desplazados por pequeñas fases $\phi_{j}$.

**Regla de diseño.** Elija $\rho$ tal que el **lóbulo principal** del espectro de línea del ciclo de síndrome se mueva **fuera** de picos fuertes de la PSD de error mientras mantiene válido el tiempo del decodificador. Valide por: (i) aumento de ${\widehat{\alpha}}_{QEC}$ vs. $d$, (ii) error lógico estable a $d$ fijo.

**7.4 Gradientes y pozos de** $\mathbf{\alpha}$

Dos motivos arquitectónicos para **dirigir flujos**:

-   **Gradiente:** organice recursos de modo que $\alpha$ **aumente** hacia regiones de cómputo críticas. Las perturbaciones pequeñas decaen a medida que viajan hacia adentro.

-   **Pozo:** cree una **cuenca de alto** $\alpha$ alrededor de qubits sensibles (ej., relojeo y buffering que desacelera cascadas de gran escala).

**Pistas de implementación.** Aumente el buffering temporal (colas, programación amortiguada) y reduzca el fanout de diafonía a medida que se acerca al "núcleo", pero limite el buffering (barandillas Sec. 7.1) para que el rendimiento no sufra.

**7.5 Dimensionamiento modular: elegir un punto óptimo balanceando latencia intra vs. inter**

Sea el total de qubits $Q$ particionado en $Q/m$ módulos de tamaño $m$. Aproxime el **tiempo característico**:

$$T(m)\text{\:\,} = \text{\:\,}A\text{ }m^{a}\text{\:\,} + \text{\:\,}B\text{ }(\frac{Q}{m})^{b}\text{     }\text{(costo intra-módulo + costo de interconexión)}.
$$

**Tamaño de módulo óptimo** (minimiza $T$):

$$m^{\star}\text{\:\,} = \text{\:\,}{(\frac{B\text{ }b}{A\text{ }a})}^{\frac{1}{a + b}}\text{\:\,}Q^{\frac{b}{a + b}}.\ 
$$

-   $a > 0$: escalamiento intra-módulo (ej., calibración, enrutamiento dentro del módulo).

-   $b > 0$: escalamiento inter-módulo (ej., latencia de enlace fotónico/iónico).

**Uso de diseño.** Mida $a,b$ empíricamente (RTM por mecanismo), estime $A,B$, calcule $m^{\star}$. Opere cerca de $m^{\star}$ y verifique que $\widehat{\alpha}$ **no colapse** (siga siendo tipo potencia) en ese vecindario.

**7.6 Multiplexación e I/O: mantener las colas bajo control**

**Problema.** La multiplexación agresiva reduce el tiempo por disparo, pero puede sincronizar colas de cola → $\alpha_{IO} \downarrow$.

**Controles.**

-   Limite la multiplexación tal que el **ratio de cola** $p95/p50$ de latencia de lectura se mantenga por debajo de un umbral (ej., $\leq 1.6$).

-   Use **ventanas de lectura con desplazamiento de fase** a través de canales para evitar crecimiento de cola coherente.

-   Dimensionamiento de buffer: mantenga utilización de buffer \< 70% para evitar amplificación de cola.

**Señal.** Si $p95/p50$ crece y ${\widehat{\alpha}}_{IO}$ cae con colapso limpio, reduzca la multiplexación e introduzca desplazamientos.

**7.7 Bucle de control en línea (ingeniería de** $\mathbf{\alpha}$ **en lazo cerrado)**

Un controlador simple para mantener $\alpha$ alto bajo restricciones:

cada Δt:

estimar {α_f(t), σ_f(t)} por familia aceptada (Sec. 5)

si \|F_t\| ≥ 2 y I\^2_t \< 50%:

calcular α_QC(t) (Sec. 6)

si α_QC(t) \< α_piso y restricciones cumplidas:

aplicar acciones A = {↑tamaño de ola, ↑jitter de reinicio ρ, ↑penalización de enrutamiento por varianza,

↓límite de multiplex, moverse hacia m\*}

de lo contrario si rendimiento \< B:

relajar A mínimamente (mantener colapso pasando)

registrar QA: colapso R\^2, I\^2_t, banderas; revertir acciones si se disparan banderas

-   $\alpha_{\text{piso}}$: pendiente fusionada mínima aceptable pre-registrada.

-   **Revertir** cualquier acción que cause NO_COLAPSO o $I_{t}^{2} \geq 50\%$.

**7.8 Seguridad y validación**

-   Cualquier intervención debe **re-pasar el colapso** en las familias afectadas.

-   Ejecute ventanas A/B (≥2--4 semanas) con **KPIs pre-registrados**: rendimiento, makespan, error lógico, tiempo de actividad, ratios $p95/p50$, y ${\widehat{\alpha}}_{f}$.

-   Si $\alpha$ sube pero los KPIs empeoran más allá de los presupuestos, está **sobre-estratificando** (demasiado buffering). Retroceda a la frontera de Pareto.

**7.9 Manuales de juego de inicio rápido**

-   **Si** $\alpha_{QEC} \downarrow$**:** añada 1--3% de jitter de cadencia; introduzca 2--3 grupos de fase para síndrome; re-mida el colapso.

-   **Si** $\alpha_{IO} \downarrow$**:** reduzca el límite de multiplex 10--20%; añada 1--2 desplazamientos de ciclo; mantenga $p95/p50 \leq 1.6$.

-   **Si** $\alpha_{runtime} \downarrow$**:** habilite agrupamiento de lectura; penalice rutas de alta varianza; limite ops largas concurrentes por ola.

-   **Planificación arquitectónica:** estime $a,b,A,B$ y fije el tamaño de módulo cerca de $m^{\star}$; confirme escalamiento tipo potencia alrededor de ese punto.

**7.10 Resumen**

-   **Programador** (olas, reinicios escalonados, enrutamiento de baja varianza) y **cadencia QEC** (micro-jitter, multi-fase) son palancas de primera línea para **elevar** $\alpha$.

-   **Dimensionamiento modular** admite un óptimo de forma cerrada $m^{\star}$ balanceando costos intra/inter; opere cerca mientras observa el colapso.

-   **Controles de I/O** evitan que las colas de latencia se sincronicen.

-   Un **controlador en lazo cerrado** mantiene $\alpha$ por encima de un piso bajo presupuestos de rendimiento/fidelidad.

**8. Protocolos Experimentales Falsificables (Superconductor e Iones Atrapados)**

Esta sección especifica experimentos RTM-QC **testeables** con elecciones concretas de $(L,T)$, recolección de datos, planes de análisis, y criterios de éxito. Cada protocolo es por bin (ambiente fijo) e incluye **placebos**, **guardias de punto de cambio**, y una **tabla de decisión pre-registrada**.

**8.1 Andamiaje común (aplica a todos los protocolos)**

**Bloqueo de BIN (ambiente).**\
{plataforma; banda de temperatura; hash de firmware (FPGA/DSP); ID de topología; política de enrutamiento; cadencia de síndrome; banda de utilización}. Cualquier cambio ⇒ nuevo bin.

**Esquema de datos (ordenado).** Para cada registro:

$$x = log\ L,y = logT,\text{ familia},\text{ etiquetas BIN},\text{ ID de réplica},\text{ marca de tiempo},\text{ pesos}\rbrack$$

**Puertas de QA (deben pasar):**

-   Cobertura: ≥6 distintos $L$, span ≥0.6 en $\log L$.

-   Ajuste EIV convergido (ODR), palanca \<25%, inicialización robusta (Theil--Sen).

-   Colapso: $R_{\text{colapso}}^{2} < 0.05$, sin tendencia LOESS, placebo de reloj se mantiene.

-   Puntos de cambio: ninguno dentro del bin (de lo contrario divida).

**Resultados (primarios, por familia):**

-   Pendiente ${\widehat{\alpha}}_{f}$ con IC 50/95%; diagnósticos de colapso.

-   Para resultados fusionados, ${\widehat{\alpha}}_{QC}(t)$, $Q$, $I^{2}$, ${\widehat{\tau}}^{2}$ (Sec. 6).

**Plan estadístico.**\
ICs bootstrap (parejas/cluster). Predefina **efecto mínimo detectable** (MDE) sobre $\alpha$ (ej., $\Delta\alpha = 0.15$) y **KPIs operacionales** (rendimiento, makespan, tasa de error lógico, tiempo de actividad, p95/p50). Umbrales abajo.

**8.2 Protocolo A --- Capa física (Superconductor)**

**Hipótesis (H1-Fís).** Aumentar la **desincronización de cluster** (reinicios escalonados + olas de lectura) **eleva** $\alpha_{\text{fís}}$ sin exceder el presupuesto de rendimiento.

**Diseño.**

-   $L$: **tamaño de cluster** de qubits activos (involucrados simultáneamente).

-   $T$: **intervalo de calibración estable** (tiempo hasta primera bandera fuera de tolerancia en cluster).

-   Brazos: **Control** (programador base) vs. **Consciente de RTM** (agrupamiento de lectura + reinicios escalonados, ±2--4% desplazamientos).

-   Duración: 2--4 semanas; intercale brazos diariamente para balancear deriva.

**Análisis.**

-   Ajuste ODR por brazo, pase colapso.

-   Efecto primario: $\Delta \hat{\alpha}\_{\text{fís}} = \hat{\alpha}\_{\text{RTM}} - \hat{\alpha}\_{\text{CTRL}}$

-   Barandillas de KPI: caída de rendimiento ≤5%, sin aumento en error de compuerta/RO \>0.2σ.

**Criterios de éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{fís}} \geq 0.15$ e IC excluye 0, **y** barandillas satisfechas.

-   Si el colapso falla en cualquier brazo, declare **inconclusivo** y rebinee.

**Placebos.** Multiplique $T$ por una constante; $\widehat{\alpha}$ sin cambios. Mezcle $L$ dentro del día; sin pendiente significativa.

**8.3 Protocolo B --- Cadencia QEC (Superconductor o Iones)**

**Hipótesis (H1-QEC).** Introducir **micro-jitter** (1--3%) en el período de síndrome y/o **extracción multi-fase** aumenta $\alpha_{\text{QEC}}$ vs. distancia de código $d$ a decodificador fijo.

**Diseño.**

-   $L$: **distancia de código** $d$ (ej., $d \in \{ 3,5,7,9\}$).

-   $T$: **ciclos hasta falla lógica** (mediana o cuantil de supervivencia a error objetivo fijo).

-   Brazos: Control (período fijo $P$) vs. Jitter ($P_{k} = P(1 + \eta_{k})$, $\eta_{k} \sim \mathcal{U}\lbrack - 0.02,0.02\rbrack$) y/o 2--3 **grupos de fase**.

-   Mantenga parámetros de decodificador fijos; sin cambio en mitigación de sesgo de ruido.

**Análisis.**

-   ODR por brazo; puerta de colapso.

-   Efecto: $\Delta{\widehat{\alpha}}_{\text{QEC}}$.

-   Barandillas de KPI: error lógico a $d$ fijo no peor por \>5% relativo.

**Criterios de éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{QEC}} \geq 0.15$ con IC excluyendo 0 y barandillas pasan.

**Diagnósticos.** Verifique PSD de procesos de error; confirme que el jitter mueve las líneas de cadencia fuera de picos dominantes.

**8.4 Protocolo C --- Programación de compilador/tiempo de ejecución**

**Hipótesis (H2-Ejecución).** El **agrupamiento de frente de onda** de lectura y el **enrutamiento de baja varianza** reducen cascadas de sincronización, aumentando $\alpha_{\text{ejecución}}$ y bajando colas de makespan.

**Diseño.**

-   $L$: **ancho de circuito post-mapeo** (o capas activas).

-   $T$: **makespan** (envío→completación).

-   Brazos: Política base vs. Consciente de RTM (olas + enrutamiento penalizado por varianza).

-   Controle banda de utilización; misma mezcla de trabajos a través de brazos.

**Análisis.**

-   Pendiente ODR por brazo; colapso.

-   KPIs: mediana de makespan (≤ base), p95/p50 latencia ↓ ≥10%.

**Criterios de éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{ejecución}} \geq 0.10$ (IC excluye 0) y p95/p50 mejora ≥10%.

**8.5 Protocolo D --- Multiplexación I/O--Cryo**

**Hipótesis (H2-IO).** Las **ventanas de lectura con desplazamiento de fase** a través de canales mantienen o elevan $\alpha_{\text{IO}}$ mientras reducen colas p95 a un grado de multiplexación dado.

**Diseño.**

-   $L$: **grado de multiplexación** (canales/línea).

-   $T$: **latencia de lectura p95** (y p50).

-   Brazos: Ventanas sincronizadas vs. ventanas desplazadas (patrón de fase $\phi_{j}$).

-   Barra $L$ a través del rango operacional.

**Análisis y éxito.**

-   $\Delta{\widehat{\alpha}}_{\text{IO}} \geq 0.10$; p95/p50 ≤ 1.6 en brazo RTM sobre la mayoría de $L$; colapso pasa.

**8.6 Protocolo E --- Dimensionamiento modular (estudio de planificación)**

**Hipótesis (H3-Mod).** Existe un tamaño de módulo $m^{\star}$ que minimiza $T(m) = Am^{a} + B(Q/m)^{b}$ con $a,b > 0$ medidos empíricamente, y operar cerca de $m^{\star}$ preserva el escalamiento tipo potencia (colapso se mantiene).

**Diseño.**

-   Plataformas con enlaces fotónicos/iónicos entre módulos.

-   Mida $T(m)$ variando tamaño de módulo (o emulando costo de interconexión) a $Q$ total fijo.

-   Ajuste $a,b,A,B$ vía ODR en el conjunto de datos de cada término; calcule $m^{\star}$.

**Criterios de éxito.**

-   $T(m)$ observado minimizado cerca de $m^{\star}$ (dentro del IC), y los ajustes log--log alrededor de $m^{\star}$ retienen colapso (sin curvatura).

**8.7 Fusión y alertas (cross-protocolo)**

A través de A--D, si ≥2 familias pasan puertas en tiempos superpuestos, calcule ${\widehat{\alpha}}_{QC}(t)$ (Sec. 6).\
**H2 (anticipación):** declare un **evento de decoherencia** si se cumplen los niveles de puntuación Z (Sec. 6.5); pruebe **adelanto--retraso** vs. picos en error lógico/makespan/colas. El valor predictivo aditivo se evalúa contra líneas base (fidelidad, utilización, temperatura) usando regresión de series de tiempo con errores HAC; pre-registre horizontes (ej., 7--30--90 días).

**8.8 Placebos, mezclas, y robustez**

-   **Placebos de reloj:** multiplique todos los $T$ por constantes; $\widehat{\alpha}$ y $R_{\text{colapso}}^{2}$ invariantes.

-   **Nulos de mezcla:** permute $L$ dentro del día; las pendientes colapsan a \~0 (dentro del IC).

-   **Fusión dejar-una-familia-fuera** para revelar dominancia.

-   **Puntos de cambio**: división automática si se detecta; re-estime en ambos lados.

**8.9 Potencia y duración (reglas empíricas)**

-   Con span ≥0.8 en $\log L$, 8--12 puntos distintos de $L$, y ruido moderado (SNR≈5--10), ODR detecta $\Delta\alpha \approx 0.10$--0.15 al 95% con ≈200--400 observaciones totales por brazo.

-   Si el ruido es mayor o se sospecha deriva, reduzca las ventanas (Sec. 5.5) y extienda la duración.

**8.10 Tabla de decisión (pre-registrada)**

| Resultado | Acción |
| :--- | :--- |
| $\Delta\hat{\alpha} \geq$ MDE **y** barandillas pasan | Promueva la intervención a producción en ese bin; monitoree con $\text{ECI}_{\text{QC}}(t)$. |
| $\Delta\hat{\alpha}$ significativo pero barandilla de KPI violada | Ajuste la intensidad (ej., reduzca buffering/jitter) y re-pruebe. |
| Colapso falla o heterogeneidad alta ($I^2 \geq 50\%$) | No fusione; reporte por familia; revise binning o mecanismos. |
| Sin efecto ($\Delta\hat{\alpha} \approx 0$) | Documente como *límite de alcance*; mantenga como control negativo. |

**8.11 Ética, seguridad, y reproducibilidad**

-   **Seguridad:** sin aumento inseguro de potencia RF; límites de jitter mantienen decodificadores válidos; retroceso en NO_COLAPSO o violación de KPI.

-   **Reproducibilidad:** YAML de métodos versionado (BIN, configuraciones de estimador, semillas), gráficos públicos (paneles de colapso, gráficos de bosque), y artefactos de placebo/mezcla almacenados.

-   **Transparencia:** publique tanto éxitos como fracasos (los resultados negativos definen alcance).

**8.12 Resumen**

Estos protocolos hacen RTM-QC **falsificable**: cada uno afirma un cambio direccional en $\alpha$ de un control específico, bajo constancia de bin, con colapso como prueba de especificación y barandillas operacionales. El éxito mejora no solo la pendiente sino también la **estabilidad en tiempo de ejecución** (colas, recalibraciones) sin sacrificar rendimiento.

**9. Plantillas de Resultados y Estándares de Reporte**

Esta sección define **qué publicar** una vez que se ejecutan los protocolos (Sec. 8). Estandariza figuras, tablas, paneles de robustez, y una lista de verificación de una página para que los resultados sean interpretables, reproducibles, y directamente comparables entre laboratorios y plataformas.

**9.1 Conjunto de figuras (mínimo)**

**Fig. 1 --- Paneles de colapso (por familia aceptada).**\
Cuatro pequeños múltiplos por familia $f$ dentro de un bin:

1.  **Ajuste log--log:** $y = \log T$ vs. $x = \log L$ con línea ODR y banda al 95%.

2.  **Residuo vs.** $x$**:** $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$ con LOESS; muestre $R_{\text{colapso}}^{2}$.

3.  **Cobertura/palanca:** dispersión resaltando puntos de palanca; anote span en $\log L$, \# distintos $L$.

4.  **Verificación de placebo:** superposición de ajustes antes/después de $T \mapsto cT$ (curvas coinciden).

**Fig. 2 --- Gráfico de bosque y heterogeneidad.**\
Por segmento de tiempo (o por brazo experimental), muestre $\hat{\alpha}\_f \pm \text{IC}$, pesos $w\_f$, el $\hat{\alpha}\_{\text{QC}}$ fusionado (diamante), y estadísticas de heterogeneidad: $Q, I^2, \hat{\tau}^2$.

**Fig. 3 ---** $\mathbf{ECI}_{QC}$**(t) serie temporal.**\
Pendiente fusionada rodante con bandas 50/95%; cinta de fondo coloreada por $I^{2}$ (verde \<25%, ámbar 25--50%, rojo ≥50%). Marque **eventos de decoherencia** (aviso/vigilancia/advertencia) y eventos de plataforma (recalibraciones, cambios de firmware).

**Fig. 4 --- Panel de KPI (emparejado con Fig. 3).**\
Ejes de tiempo alineados para: tasa de error lógico (a $d$ fijo), mediana de makespan y p95, cola p95, tiempo de actividad entre recalibraciones. Superponga regiones sombreadas para niveles de alerta de Fig. 3.

**Fig. 5 --- Resultados A/B (por protocolo).**\
Para cada brazo: gráficos de distribución (violín/caja) de ${\widehat{\alpha}}_{f}$, makespan p95/p50, error lógico; incluya $\Delta\widehat{\alpha}$ con IC y barandillas.

**Fig. 6 opcional --- Diagnósticos espectrales (QEC).**\
PSD de procesos de error mostrando cómo el jitter de cadencia/multi-fase mueve espectros de línea fuera de picos dominantes.

### Tabla 1 — Familias aceptadas (por bin/brazo).

| Familia | #pts L | span $\log L$ | $\alpha_f$ (ODR, IC 50/95\%) | Theil–Sen | banda SIMEX | ($R_{\text{col}}^2$) | Palanca máx | Banderas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Física | 9 | 1.05 | 0.62 [0.55, 0.70] | 0.60 | 0.58–0.66 | 0.02 | 0.18 | — |
| QEC | 8 | 0.82 | 0.74 [0.66, 0.82] | 0.71 | — | 0.03 | 0.22 | — |
| … | … | … | … | … | … | … | … | … |

### Tabla 2 — Fusión y heterogeneidad (por segmento de tiempo o brazo).

| Tiempo/Brazo | Familias | $\alpha_{\text{QC}} \pm \text{SE}$ | (Q) (gl) | $I^2$ | $\tau^2$ | ¿Fusión? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Consciente de RTM | 3 | 0.69 $\pm$ 0.04 | 3.2 (2) | 37\% | 0.005 | Sí |
| Control | 3 | 0.54 $\pm$ 0.05 | 6.8 (2) | 71\% | 0.018 | No (reporte por familia) |

### Tabla 3 — Resultados de protocolo (A/B).

| Protocolo | Métrica | Control | Consciente de RTM | Efecto ($\Delta$) | IC 95\% | ¿Pasa barandilla? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A (Fís) | $\alpha_{\text{fís}}$ | 0.48 | 0.64 | +0.16 | [0.07, 0.25] | ✔ |
| A (Fís) | Rendimiento | 100\% | 97\% | –3\% | [–6, 0]\% | ✔ |
| B (QEC) | $\alpha_{\text{QEC}}$ | 0.68 | 0.83 | +0.15 | [0.06, 0.24] | ✔ |
| C (Ejec) | p95/p50 | 1.85 | 1.60 | –0.25 | [–0.35, –0.15] | ✔ |

### Tabla 4 — Umbrales y banderas pre-registrados.

| Puerta | Umbral | Estado |
| :--- | :--- | :--- |
| Colapso $R^2$ | < 0.05 | Pasa |
| Heterogeneidad $I^2$ | < 50\% para fusión | Pasa |
| MDE sobre $\Delta\alpha$ | ≥ 0.10–0.15 | Pasa |
| Barandillas de KPI | ≤ 5\% pérdida de rendimiento; ≤ +5\% error lógico | Pasa |

**9.3 Panel de robustez y sensibilidad**

-   **Estimadores:** ODR (primario), Theil--Sen, bandas SIMEX (± bandas para $\sigma_{\xi}^{2}$).

-   **Ventanas:** repita con $h$± 25%; $\widehat{\alpha}$ estable y colapso aún pasando.

-   **Placebos:** reescalamiento de reloj invariante; **Mezclas:** permute $L$ dentro del día---pendiente → \~0.

-   **Fusión dejar-una-familia-fuera:** reporte ${\widehat{\alpha}}_{QC}^{( - f)}$.

-   **Catastróficos:** re-estime excluyendo eventos etiquetados; muestre Δ.

-   **Efecto fijo vs. efectos aleatorios:** publique ambos; divergencia implica heterogeneidad genuina.

**9.4 Resultados negativos y límites de alcance**

Publique bins/brazos que **fallaron**:

-   NO_COLAPSO (curvatura), MEZCLA_REGIMEN (quiebres), COBERTURA_DELGADA, RIESGO_PALANCA, DIVERGENCIA_FAMILIAS (alto $I^{2}$).\
    Incluya una nota corta: mecanismo sospechado y próximos pasos (rebinear, cambio de instrumentación, aislamiento de mecanismo). Los resultados negativos definen **dónde RTM no aplica**.

**9.5 Lista de verificación de una página (para cada conjunto de figuras/tablas)**

-   Claves de BIN listadas y sin cambios.

-   \# distintos $L$≥ 6 y span ≥ 0.6.

-   ODR convergido; Theil--Sen reportado; SIMEX (si $\sigma_{\xi}^{2}$ conocido).

-   Colapso: $R^{2} < 0.05$; placebo OK; sin puntos de cambio.

-   Fusión: $\mid \mathcal{F}_{t} \mid \geq 2$; $I^{2} < 50\%$; REML convergido.

-   KPIs: rendimiento, makespan p95/p50, error lógico, tiempo de actividad---barandillas aplicadas.

-   Panel de robustez completado (ventanas, mezclas, LOO).

-   Hashes de procedencia (YAML de métodos, semillas, versión de código) incluidos.

**9.6 Plantilla narrativa (texto corto de "Resultados")**

> *Capa física.* A través de 9 tamaños de cluster (span 1.05 en $\log L$), la programación consciente de RTM aumentó la pendiente de $0.48$ a $0.64$ (Δ = $0.16$, IC 95% $\lbrack 0.07,0.25\rbrack$); los residuos mostraron $R_{\text{colapso}}^{2} = 0.02$. El rendimiento permaneció dentro de la barandilla del 5%.\
> *QEC.* Con 1--3% de jitter de cadencia, $\alpha_{\text{QEC}}$ subió de $0.68$ a $0.83$ (Δ = $0.15$, IC $\lbrack 0.06,0.24\rbrack$), el error lógico a $d$ fijo no empeoró.\
> *Tiempo de ejecución.* El agrupamiento de frente de onda y el enrutamiento consciente de varianza redujeron p95/p50 de 1.85 a 1.60; $\alpha_{\text{ejecución}}$ aumentó en $0.12$.\
> *Fusión.* Tres familias pasaron las puertas; $I^{2} = 37\%$. El ${\widehat{\alpha}}_{QC} = 0.69 \pm 0.04$ fusionado. Una alerta de **vigilancia** de decoherencia se disparó el día 17; precedió un pico de makespan por 3 días.

**9.7 Resumen**

Las plantillas arriba aseguran que cada afirmación esté respaldada por: (i) prueba visual y numérica de **colapso**, (ii) estimación consciente de EIV, (iii) contabilidad de **heterogeneidad** para fusión, (iv) barandillas de KPI, y (v) evidencia completa de **robustez**.

**10. Discusión**

Esta sección interpreta los resultados de RTM-QC, clarifica cómo una vista **pendiente-primero** complementa los paradigmas de fidelidad/QEC, y expone los compromisos, riesgos, y caminos de adopción.

**10.1 ¿Qué compra realmente un** $\mathbf{\alpha}$ **más alto?**

Una pendiente por bin más grande $\alpha$ significa que **el tiempo se estira más abruptamente con la escala**, es decir, los agregados más grandes se desaceleran *relativamente* a los más pequeños dentro de un ambiente estable. Operacionalmente:

-   **Amortiguamiento de choque:** las perturbaciones a pequeña escala tienen menos probabilidad de sincronizar capas más grandes (tiempo de ejecución → QEC → I/O), reduciendo cascadas que inflan colas (p95/p50), colas, y recalibraciones forzadas.

-   **Predecibilidad:** mayor $\alpha$ típicamente reduce la **varianza de corrida a corrida** (distribuciones de KPI más estrechas) porque el "gradiente de tempo" de la pila previene la alineación de eventos largos raros.

-   **Palanca de control:** $\alpha$ es agnóstico a unidades; podemos optimizarlo con perillas de programador/QEC/interconexión sin confundir cambios de unidad (relojes) con cambio estructural.

**No es un sustituto de la fidelidad.** RTM mejora **cómo** se comporta el tiempo a través de la escala; no aumenta las fidelidades de uno/dos qubits por sí mismo. Las ganancias llegan a través de menos cascadas y mejor uso de la fidelidad existente.

**10.2 Complementariedad con QEC y compilación**

-   **QEC:** El diseño tradicional elige la distancia de código $d$ de las tasas de error. RTM añade un segundo eje: **geometría de cadencia**. Ligera **desincronización** (jitter/multi-fase) puede elevar $\alpha_{\text{QEC}}$ a $d$ y decodificador fijos, a menudo mejorando la estabilidad sin gastos generales adicionales.

-   **Compilación/tiempo de ejecución:** El enrutamiento de última generación minimiza profundidad/longitud. RTM también pide minimizar la **varianza de tiempo** y la **coincidencia de ops largas**, lo cual puede mejorar las colas incluso si la profundidad media cambia marginalmente.

**10.3 Compromisos y frente de Pareto**

-   **Rendimiento vs. estratificación:** Elevar $\alpha$ añadiendo buffers/agrupamiento puede reducir la concurrencia cruda. Por tanto optimizamos en un **frente de Pareto** (Sec. 7.1): aumentar $\alpha$ *sujeto a* pisos de rendimiento/fidelidad.

-   **Jitter vs. tiempo de decodificador:** El micro-jitter debe permanecer dentro de la validez del decodificador; de lo contrario intercambia mayor $\alpha$ por fallas lógicas.

-   **Tamaño modular:** Operar cerca de $m^{\star}$ (Sec. 7.5) balancea costos intra/inter, pero derivar demasiado lejos (módulos más grandes o más pequeños) puede ya sea aplanar $\alpha$ (sincronización) o estrangular el ancho de banda.

**10.4 Modos de falla (informativos por diseño)**

La puerta de **colapso** de RTM convierte las fallas en diagnósticos:

-   **NO_COLAPSO:** log--log curvo → mecanismo faltante (ej., "reloj" dependiente de escala o gasto general no lineal).

-   **MEZCLA_REGIMEN:** quiebres → costuras ocultas (intercambios de firmware/programador); rebinee o divida.

-   **Alto** $I^{2}$**:** los proxies discrepan → **no** fusione; inspeccione controles por familia.

Publicar estos casos mapea **límites de alcance** (donde RTM *no* aplica), lo cual es científicamente útil y previene el exceso de alcance.

**10.5 Por qué un indicador único fusionado---y cuándo no usarlo**

**Pros:** $\text{ECI}_{\text{QC}}(t)$ resume la coherencia multiescala, habilitando **alertas** (Sec. 6.5) y seguimiento de tendencias.
**Contras:** La fusión puede ocultar heterogeneidad. Por tanto las **puertas** (al menos dos familias, $I^2$ < 50%, convergencia REML). Si fallan, publique **por familia** $\hat{\alpha}_f$ solamente; la falta de fusión es en sí un resultado ("la pila está hablando con diferentes pendientes").

**10.6 Relación con difusiones con cambio de tiempo y colas**

La vista PDE (RTM como un **reloj dependiente del estado**) explica por qué las **colas** se encogen cuando $\alpha$ sube: el **exponente dinámico efectivo** $z$ aumenta, y los tiempos de salida/primer paso escalan más abruptamente con el "radio" (Sec. 6 del artículo de matemáticas). En términos de colas, la programación que eleva $\alpha$ **descorrelaciona** las ráfagas de servicio y amortigua la amplificación de cola.

**10.7 Validez externa y portabilidad**

Porque $\alpha$ es **invariante de gauge**, las comparaciones se mantienen entre laboratorios y generaciones cuando los bins coinciden (claves de ambiente). La misma tubería se porta a **iones atrapados**, **superconductores**, **átomos neutros**, y **recocedores** con $(L,T)$ apropiados por capa. Lo que cambia es la instrumentación; la **lógica de colapso** y la **estimación EIV** permanecen.

**10.8 Camino de adopción (práctico)**

1.  **Modo sombra:** calcule ${\widehat{\alpha}}_{f}$ por familia y paneles de colapso sin cambiar las operaciones.

2.  **Perillas de bajo riesgo:** habilite **agrupamiento de lectura**, **reinicios escalonados**, y pequeño **jitter de cadencia** (≤3%).

3.  **Cierre el lazo:** traiga ${ECI}_{QC}(t)$ a paneles de guardia con niveles de alerta y manuales de juego.

4.  **Planificación arquitectónica:** mida $a,b,A,B$ (Sec. 7.5) para elegir tamaños de módulo; itere trimestralmente.

**10.9 Preguntas abiertas**

-   **Co-diseño de decodificador:** ¿cómo incluir $\alpha$ directamente en las actualizaciones de programación/grafo de los decodificadores?

-   **Controladores de aprendizaje:** ¿puede RL ajustar $\alpha$ sujeto a pisos de KPI sin violar el colapso?

-   **Pruebas de holonomía:** estadísticas prácticas para distinguir curvatura de obstrucciones topológicas (falla de colapso global).

-   **Causalidad entre capas:** ¿cuándo los cambios de $\alpha$ en la capa física *causan* cambios en tiempo de ejecución vs. solo correlacionan vía utilización?

**10.10 Conclusión**

RTM-QC añade un **tercer eje**---la **geometría del tempo**---a fidelidad y escala. Con puertas estrictas (colapso, heterogeneidad) y controles modestos (agrupamiento, jitter, varianza de enrutamiento), $\alpha$ se convierte en una palanca confiable para **estabilidad y rendimiento**, produciendo alertas tempranas y guía de diseño mientras respeta la falsificabilidad científica.

**11. Limitaciones y Alcance**

**Dependencia de bin.** RTM es una teoría **por bin**. Si el ambiente (temperatura, firmware, topología, decodificador, utilización) deriva, la pendiente $\alpha$ está indefinida hasta que el bin se divida. Los resultados solo son válidos dentro de claves de BIN claramente documentadas.

**Sensibilidad a elección de proxy.** Los proxies $(L,T)$ deben reflejar un **mecanismo dominante único** por familia. Proxies mal especificados (ej., mezclar lectura y enrutamiento en el mismo $T$) inducen curvatura y válidamente fallan el colapso.

**Sesgo de ventana finita.** Cuando $\alpha(u)$ deriva, cualquier ventana finita de ancho $h$ incurre sesgo $O(\varepsilon h)$. Nuestra guía adiabática mitiga pero no elimina esto; el $\widehat{\alpha}$ reportado debe interpretarse como **local**.

**Supuestos del modelo EIV.** ODR/TLS y SIMEX asumen errores bien comportados (media cero, momentos finitos) e independencia de $x$. Errores de colas pesadas o dependientes del estado requieren verificaciones de robustez (Theil--Sen, bootstrap, bandas de sensibilidad).

**Heterogeneidad de fusión.** La fusión de efectos aleatorios es apropiada solo cuando las familias son **conmensurables** e $I^{2} < 50\%$. De lo contrario el indicador de número único se retiene por diseño; RTM no fuerza acuerdo entre mecanismos.

**Límites de causalidad.** $\alpha$ es **estructural pero no causal** por defecto. Las secciones de diseño proponen intervenciones y protocolos A/B, pero las afirmaciones causales requieren los controles y barandillas pre-registrados que especificamos.

**Límites de alcance.** Los sistemas con tiempo **no-potencia** (curvatura persistente), **relojes dependientes de escala** (gastos generales que crecen con $L$ dentro de un bin), o **holonomía** fuerte (costuras globales) están **fuera** de la aplicabilidad de RTM. En tales dominios, trate $\alpha$ como indefinido y publique resultados negativos.

**12. Métodos y Reproducibilidad**

**12.1 Esquema de datos y BINs**

-   **Clave de BIN:** {plataforma, banda de temperatura, hash de firmware (FPGA/DSP), ID de topología, política de enrutamiento, cadencia de síndrome, banda de utilización}.

-   **Tabla ordenada (por bin):** \[x=log L, y=log T, familia, etiquetas BIN, id_réplica, marca_tiempo, peso\].

-   **Puertas de cobertura:** ≥6 distintos $L$, span ≥0.6 en $\log L$.

**12.2 Tubería de estimación (por familia, por bin)**

1.  **Escaneo de punto de cambio:** PELT/BIC en $(x,y)$ y en residuos si disponibles; divida si se detecta.

2.  **Inicialización:** pendiente/ordenada al origen Theil--Sen; etiquete catastróficos; construya pesos de réplica.

3.  **Ajuste primario:** ODR/TLS (residuos ortogonales) con SEs de réplica o bootstrap.

4.  **SIMEX (opcional):** cuando $\sigma_{\xi}^{2}$ es estimable; extrapole a $\lambda = - 1$.

5.  **Prueba de colapso:** regresione $\tilde{y} = y - \hat{\alpha}x - \hat{c}$ sobre $x$; requiera $R_{\text{colapso}}^2 < 0.05$, LOESS plano, placebo de reloj se mantiene.

6.  **Diagnósticos:** palanca ≤25%; gráficos residuales; ancho de ventana $h$ registrado.

7.  **Aceptar/Rechazar:** acepte si todas las puertas pasan; de lo contrario etiquete (NO_COLAPSO, MEZCLA_REGIMEN, COBERTURA_DELGADA, RIESGO_PALANCA, FALLA_EIV).

**12.3 Fusión y heterogeneidad (rodante)**

-   **Pesos:** $w_{f} = 1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})$ con ${\widehat{\tau}}^{2}$ vía REML (DL como sensibilidad).

-   **Pendiente fusionada:** ${\hat{\alpha}}_{\mathrm{QC}}=\sum w_f {\hat{\alpha}}_f / \sum w_f$; **varianza:** $1 / \sum w_f$.

-   **Diagnósticos:** línea base de efecto fijo, **Q de Cochran** e $I^{2}$.

-   **Puertas:** fusione solo si $\mid \mathcal{F} \mid \geq 2$ e $I^{2} < 50\%$. De lo contrario publique por familia.

**12.4 Operación en tiempo real y alertas**

-   **Ventanas rodantes:** horizonte deslizante en $x$ (ancho $h$) y reloj de pared (7--28 días).

-   **Suavizado:** mediana de 3 puntos; **Puntuación Z** contra EWMA de 30 días.

-   **Niveles de alerta:** umbrales de Aviso/Vigilancia/Advertencia (Sec. 6.5).

-   **Manuales de juego:** estrangule concurrencia, escalone reinicios, jitter de cadencia, enrutamiento consciente de varianza; todas las intervenciones deben re-pasar **colapso**.

**12.5 Robustez y sensibilidad**

-   **Estimadores:** publique ODR (primario), Theil--Sen, bandas SIMEX.

-   **Ventanas:** sensibilidad ±25% $h$; estabilidad de $\widehat{\alpha}$ requerida.

-   **Placebos y mezclas:** invariancia de reescalamiento de reloj; mezclas de $L$ producen pendientes cercanas a cero.

-   **Fusión dejar-una-familia-fuera**; comparación **efecto fijo** vs **efectos aleatorios**.

**12.6 Procedencia (YAML de métodos)**

-   Claves de BIN, configuraciones de estimador, semillas bootstrap, $\Lambda$ SIMEX, ventana $h$, umbrales de colapso, puertas de heterogeneidad, versiones de código de análisis.

-   Todos los gráficos y números incluyen hash del YAML de métodos; las re-ejecuciones con el mismo YAML reproducen números dentro del ruido de bootstrap.

**13. Conclusión y Perspectiva**

Presentamos la **computación cuántica consciente de RTM (RTM-QC)**: un marco **pendiente-primero** que mide e **ingeniería** la geometría del tiempo a través de la escala. Dentro de bins estables, el tiempo característico $T$ escala con un proxy de tamaño $L$ como $T \propto L^{\alpha}$; el **exponente de coherencia** $\alpha$ es invariante a relojes y por tanto comparable entre dispositivos, pilas, y laboratorios. Con **colapso** como puerta falsificable y estimación de **errores en variables**, $\alpha$ se convierte en una señal operacional confiable. Fusionar pendientes limpias por capa produce un $\mathbf{ECI}_{QC}$**(t)** en tiempo real que soporta **alertas tempranas** (eventos de decoherencia) y **decisiones de diseño** (programador, cadencia QEC, dimensionamiento modular, desplazamientos de I/O).

**Qué añade esto.** RTM-QC complementa fidelidad/QEC introduciendo un tercer eje---**geometría de tempo**---que explica y controla colas, colas de espera, y cascadas de sincronización. Controles modestos y reversibles (agrupamiento, reinicios escalonados, micro-jitter, enrutamiento de baja varianza) pueden **elevar** $\alpha$ sin degradar rendimiento o fidelidad cuando se usan con barandillas.

**Qué no hace.** RTM-QC no reemplaza mejoras físicas (fidelidades, $T_{1}/T_{2}$), ni garantiza causalidad sin los protocolos A/B y barandillas que especificamos. Las fallas de colapso, alta heterogeneidad, o costuras de régimen son **informativas**, delineando límites de alcance en lugar de invitar arreglos post-hoc.

**Agenda a corto plazo.**

1.  **Ejecute los protocolos** (Sec. 8) en plataformas superconductoras e iónicas; publique tanto éxitos como negativos con diagnósticos completos de colapso/fusión.

2.  **Cierre el lazo**: despliegue paneles de ${ECI}_{QC}(t)$ y manuales de juego de alerta en producción; evalúe adelanto--retraso vs. picos de KPI.

3.  **Co-diseñe con decodificadores** y compiladores de modo que cadencia y enrutamiento optimicen $\alpha$ sujeto a pisos de rendimiento/fidelidad.

4.  **Estandarice el reporte**: figuras/tablas en Sec. 9, YAML de métodos, y artefactos de robustez abiertos.

**Preguntas a largo plazo.** Incorpore $\alpha$ en **modelos de difusión con cambio de tiempo** de colas; desarrolle **pruebas de holonomía** para distinguir curvatura de costuras; extienda a **redes modulares** y plataformas de **átomos neutros**; integre controladores basados en aprendizaje que respeten las puertas de colapso.

**Conclusión.** RTM-QC da a los equipos cuánticos una **palanca robusta a unidades y falsificable** sobre el tiempo multiescala. Mida la pendiente, **valide por colapso**, fusione cuando las familias concuerden, e **ingeniería** $\alpha$---no como un eslogan, sino como una práctica reproducible para entregar computación cuántica más estable y eficiente.

**Apéndices**

**Apéndice A --- Antecedentes Matemáticos (esenciales RTM para QC)**

**A.1 Semigrupo → ley de potencia**

Asuma semigrupo de escala por bin $T(bL) = f(b)T(L)$, $f(1) = 1$, y mensurabilidad cerca de $b = 1$. Entonces $f(b) = b^{\alpha}$ y

$$T(L) = \kappa L^{\alpha},v(u) = \log T = \alpha u + \log\kappa,u = \log L.
$$

$\alpha$ es **invariante de gauge**; $\kappa$ es un **reloj**.

**A.2 1-forma y colapso**

Defina la 1-forma RTM $\omega = dv - \alpha\text{ }du$. **Colapso** (independencia residual de $v - \alpha u$ respecto a $u$) es equivalente a **exactitud** de $\omega$ en un bin simplemente conexo:

$$\omega = d\psi(x),d\omega = 0,\psi\text{ independiente de }u.
$$

Si $\alpha = \alpha(x,u)$, entonces $d\omega = - d\alpha \land du$; curvatura no nula rompe el colapso.

**A.3 Exponentes variables (sesgo de ventana finita)**

Para $\alpha(u)$ lentamente variable:

$$v(u) = \int_{u_{0}}^{u}{\alpha(s)\text{ }ds + \log\kappa(u),\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h),}
$$

y $R_{\text{colapso}}^{2} = O((\varepsilon h)^{2})$ para ancho de ventana $h$.

**Apéndice B --- Estimadores y Algoritmos**

**B.1 Regresión de Distancia Ortogonal (TLS/ODR)**

Minimice residuos ortogonales:

$$\underset{\alpha,c}{\min}\sum_{i}^{}\frac{(y_{i} - \alpha x_{i} - c)^{2}}{\sigma_{y,i}^{2} + \alpha^{2}\sigma_{x,i}^{2}}.
$$

**Inicialización:** Theil--Sen; **ICs:** bootstrap parejas/cluster; **verificaciones:** número de condición \< $10^{4}$; palanca máx \< 25%.

**B.2 Theil--Sen**

Mediana de pendientes por pares $\alpha_{ij} = (y_{j} - y_{i})/(x_{j} - x_{i})$; robusto a valores atípicos; atenuación EIV leve.

**B.3 SIMEX (opcional)**

Si $\sigma_{\xi}^{2} = Var(\xi)$ es estimable, simule $x^{(\lambda)} = x^{obs} + \sqrt{\lambda}\widetilde{\xi}$ y extrapole $\widehat{\alpha}(\lambda)$ a $\lambda = - 1$.

**B.4 Puerta de colapso**

Regresione residuos $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$ sobre $x$; requiera $R_{\text{colapso}}^{2} < 0.05$ y LOESS plano; pase placebo de reloj.

**Apéndice C --- Tarjetas de Protocolo (plantillas copiar-pegar)**

**C.1 Física (reinicios escalonados + olas de lectura)**

-   **L/T:** $L =$ tamaño de cluster activo; $T =$ intervalo de calibración estable.

-   **Brazos:** Control vs Consciente de RTM (olas + desplazamientos de reinicio 2--4%).

-   **Duración:** 2--4 semanas, intercalados.

-   **Éxito:** $\Delta\alpha_{\text{fís}} \geq 0.15$ (IC 95% excluye 0), pérdida de rendimiento ≤5%, colapso pasa.

**C.2 QEC (micro-jitter / multi-fase)**

-   **L/T:** $L = d$; $T =$ ciclos hasta falla lógica.

-   **Brazos:** Período fijo vs $Pk = P(1 + \eta k),\  \mid \eta k \mid \leq 0.02$ o 2--3 grupos de fase.

-   **Éxito:** $\Delta\alpha_{\text{QEC}} \geq 0.15$, sin regresión de error lógico (\>5%) a $d$ fijo.

**C.3 Tiempo de ejecución (agrupamiento + enrutamiento de baja varianza)**

-   **L/T:** $L =$ ancho post-mapeo; $T =$ makespan.

-   **Brazos:** Base vs frente de onda + enrutamiento penalizado por varianza.

-   **Éxito:** $\Delta\alpha_{\text{ejecución}} \geq 0.10$ y latencia p95/p50 ↓ ≥10%.

**C.4 I/O (ventanas con desplazamiento de fase)**

-   **L/T:** $L =$ grado de multiplexación; $T =$ latencia de lectura p95 (y p50).

-   **Brazos:** Sincronizado vs ventanas con desplazamiento de fase.

-   **Éxito:** $\Delta\alpha_{\text{IO}} \geq 0.10$, p95/p50 ≤ 1.6 sobre la mayoría de $L$.

**Apéndice D --- YAML de Métodos (esqueleto)**

### YAML de Métodos (esqueleto)

```
bin:
  plataforma: "SC"              # o "IONES", "NA"
  banda_temperatura: "10-15mK"
  hash_firmware: "fpga_1.4.2_dsp_0.9.8"
  id_topologia: "mesh-v3"
  politica_enrutamiento: "base"  # o "consciente-rtm"
  cadencia_sindrome: "P=3.2us, jitter=0%"
  banda_utilizacion: "30-60%"
 
estimacion:
  min_puntos_L: 6
  min_span_logL: 0.6
  eiv: "odr"
  odr:
    init: "theil-sen"
    limite_palanca: 0.25
    bootstrap: {clusters: true, reps: 2000, semilla: 123}
  simex:
    habilitado: false
    lambda: [0.5, 1.0, 1.5, 2.0]

colapso:
  umbral_r2: 0.05
  placebo_reloj: true
  escaneo_punto_cambio: {metodo: "PELT", penalidad: "BIC"}
 
fusion:
  puerta_heterogeneidad_I2: 0.5
  metodo_tau2: "REML"
  min_familias: 2
 
eci_rt:
  ventana_logL: 0.8
  horizonte_dias: 14
  suavizado: "mediana3"
  alerta:
    z_aviso: -1.5
    z_vigilancia: -2.0
    z_advertencia: -2.5
```

**Apéndice E --- Glosario de Notación**

-   $L$: proxy de escala (específico de capa); $u = \log L$.

-   $T$: tiempo característico; $v = \log T$.

-   $\alpha$: **exponente de coherencia** (pendiente; invariante de reloj).

-   **Bin**: segmento de ambiente con {plataforma, banda de temperatura, hash de firmware, ID de topología, política de enrutamiento, cadencia de síndrome, banda de utilización} fijos.

-   **Colapso**: $R^{2}(\widetilde{y} \sim x) < 0.05$ para $\widetilde{y} = y - \widehat{\alpha}x$; los residuos no muestran tendencia vs $x$.

-   $\mathbf{ECI}_{QC}(t)$: pendiente fusionada vía efectos aleatorios en tiempo $t$.

-   $Q,I^{2},\tau^{2}$: estadísticas de heterogeneidad para fusión.

-   ODR/TLS, Theil--Sen, SIMEX: estimadores de pendiente bajo EIV.

-   **Ventana adiabática**: ancho $h$ en $u$ donde $\mid \partial_{u}\alpha \mid h \ll 1$.

**Apéndice F --- Recetas de Figuras Reproducibles (minimal)**

-   **Panel de colapso**:

    -   Ajuste ODR; calcule residuos $\widetilde{y}$.

    -   Grafique $y$ vs $x$+ banda ODR; residuo vs $x$ con LOESS.

    -   Anote $R^2\_{\text{colapso}}$, #L, span, palanca.

-   **Gráfico de bosque**:

    -   Para familias aceptadas, muestre $\widehat{\alpha}_f \pm \text{IC}$; *calcule* $w_f$, $Q$, $I^2$, $\hat{\tau}^2$.

    -   Superponga ${\widehat{\alpha}}_{QC}$ fusionado.

-   $\mathbf{ECI}_{QC}(t)$:

    -   Fusión rodante; muestre bandas 50/95%; fondo coloreado por niveles de $I^{2}$; marque niveles de alerta.

**APÉNDICE G --- Análisis Empírico: Escalamiento de Hardware Cuántico y el Factor de Confusión Generacional**

El marco RTM dicta que aumentar los límites físicos de una red fuertemente acoplada pero no resonante aumentará proporcionalmente su fricción topológica. Para probar esto en arreglos cuánticos, analizamos los tiempos de coherencia $T_{2}$ de 31 procesadores IBM Quantum (5 a 1121 qubits).

**G.1 Observación Heurística y Paradoja de Simpson**

La regresión inicial ingenua de Mínimos Cuadrados Ordinarios (OLS) sobre el conjunto de datos crudo produjo un exponente de escalamiento positivo de $\alpha = \  + 0.227$. Esto creó la ilusión de que añadir más qubits extendía intrínsecamente los tiempos de coherencia. Sin embargo, esta es una manifestación clásica de la Paradoja de Simpson: los procesadores más grandes fueron construidos años después que los más pequeños, lo que significa que sus tiempos $T_{2}$ extendidos fueron el resultado de materiales superconductores y técnicas de fabricación superiores, no de su mayor tamaño espacial.

**G.2 Validación EIV Multivariable Rigurosa**

Para aislar matemáticamente la ley de escalamiento físico del progreso de ingeniería humana, desplegamos una tubería estadística "Equipo Rojo":

1.  **Regresión de Distancia Ortogonal (ODR) Multivariable:** Abandonamos el burdo "binning por era" categórico en favor de un modelo multivariable continuo. Este evalúa simultáneamente la progresión tecnológica cronológica junto con la expansión espacial topológica.

2.  **Inyección de Ruido de Calibración:** Inyectamos explícitamente una varianza realista de calibración de hardware del $15\%$ en las lecturas de $T_{2}$, forzando al marco a absorber el ruido de medición criogénica estándar.

**G.3 La Clase de Transporte Inverso (Hallazgos Robustos)**

Una vez que la mejora continua de materiales superconductores se normaliza algebraicamente, la ilusión del escalamiento monolítico se destruye, revelando la verdadera física del arreglo cuántico:

-   **Factor de Ganancia Tecnológica:** El modelo extrae precisamente la progresión de ingeniería, mostrando que la coherencia del hardware de IBM mejora por un factor de $\mathbf{\gamma}\mathbf{= \  + 0.139}$ **dex/año**.

-   **Verdadero Exponente Topológico:** Después de restar $\gamma$, el escalamiento físico aislado revela un exponente robusto, estrictamente negativo de $\mathbf{\alpha}\mathbf{= \  - 0.259\ }\mathbf{\pm}\mathbf{0.049}$.

**Conclusión:** La decoherencia cuántica macroscópica reside seguramente dentro de la **Clase de Transporte Inverso** ($\alpha < \ 0$). RTM valida empíricamente que la decoherencia en arreglos de procesadores grandes no es un fenómeno localizado por qubit, sino una fuga topológica colectiva masiva: la coherencia estructural naturalmente y predeciblemente se degrada a medida que aumenta el tamaño de la red geométrica.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
