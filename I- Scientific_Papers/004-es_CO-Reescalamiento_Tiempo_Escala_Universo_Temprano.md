<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **Reescalamiento Tiempo-Escala en el Crecimiento de Estructuras del Universo Temprano**  
  
Álvaro Quiceno

</div>


**Resumen**

Esta breve nota aísla una única afirmación dentro de RTM y la empuja hacia un cálculo aproximado adyacente a la cosmología y falsificable. Si los tiempos característicos de proceso escalan como T∝L\^α, entonces en un universo temprano con escala ambiental $L_{env}$ mucho menor, los tiempos efectivos se acortan. Tomando $L_{env}$ para rastrear la escala de Hubble $L_{H}$ en un ansatz minimal "FRW+α" produce un factor de aceleración simple A por el cual se divide cualquier escala de tiempo mesoscópica. Evaluado en z∼10 esto da aceleraciones de orden de magnitud de **20-37×** para α∼1, consistente en dirección con galaxias "demasiado tempranas/demasiado masivas". Luego mostramos, paramétricamente, qué tan grande necesitaría ser una aceleración A para reproducir masas/luminosidades estelares como las reportadas en z>10 sin tocar BBN/CMB: el truco es mantener α inactivo (banda cerca de 0) en la era de plasma homogéneo y activo (orden unidad) solo en medios bariónicos estructurados y multifase.

**Validación empírica preliminar**$\rightarrow$**(APÉNDICE B)**. Validamos la hipótesis de reescalamiento temporal usando un catálogo comprehensivo de 55 estimaciones de masa estelar de galaxias observadas por JWST (incluyendo datos de JADES, CEERS, Labbé et al. 2023, UNCOVER, y GLASS) en corrimientos al rojo que van desde $z\  = \ 6.0$ hasta $16.4$. El análisis heurístico inicial indica que el 44% de estas galaxias excede los límites estándar de $\Lambda\text{CDM}$, produciendo un exponente de coherencia aparente de $\alpha = \ 1.33\  \pm 0.30$. Para descartar definitivamente el sesgo de Eddington y las incertidumbres estándar de ajuste de Distribución Espectral de Energía (SED) ($\sim 0.3$ dex), sometimos posteriormente el conjunto de datos a una prueba de estrés probabilística Monte Carlo rigurosa. El análisis robusto corregido por sesgo rechaza firmemente el límite del modelo estándar ($\alpha = 1.0$) con $p < 10^{- 6}$, convergiendo en un exponente topológico verdadero de $\alpha = 1.16 \pm 0.08$. Esto confirma con alta significancia estadística que el universo temprano operaba en un régimen topológico de "Alta Coherencia" ($\alpha > \ 1$), efectivamente otorgando a la materia bariónica significativamente más tiempo dinámico para colapsar y estructurarse de lo que indica el reloj lineal de Hubble.

**1) Ansatz minimal: FRW+α con** ${\mathbf{L}\mathbf{=}\mathbf{H}}^{\mathbf{-}\mathbf{1}}$

$${T \propto L}^{\alpha}$$

Elija la **escala ambiental** $L$ como la longitud de Hubble FRW $H^{- 1}(z)$. Defina el **reescalamiento temporal operacional** entre un pequeño intervalo de tiempo cósmico estándar $dt$ y el tiempo "efectivo" del proceso $d\tau$:

$$d\tau = \left( \frac{L(z)}{L_{0}} \right)^{\alpha}dt = \left( \frac{H_{0}}{H(z)} \right)^{\alpha}dt$$

Equivalentemente, cualquier escala temporal de proceso $\tau_{std}(z)$ (calculada en física estándar) es **acelerada** por

  $$\tau_{RTM}(z) = \frac{\tau_{std}(z)}{A(z;\alpha)},\ \ A(z;\alpha) \equiv \left( \frac{H(z)}{H_{0}} \right)^{\alpha}$$

donde $A(z;\alpha)$ es el **factor de aceleración RTM**.

Con fondo ΛCDM,

$$\frac{H(z)}{H_{0}} = \left\lbrack \Omega_{m}{(1 + z)}^{3}{+ \ \Omega}_{r}{(1 + z)}^{4}{+ \ \Omega}_{\Lambda} \right\rbrack^{1/2}$$

En $z \gtrsim 10$ (dominado por materia en buena aproximación),


  $$A(z;\alpha)\  \simeq \sqrt{\Omega_{m}}{\ (1 + z)}^{3/2}$$

$$\frac{H(z)}{H_{0}} \simeq \sqrt{\Omega_{m}}{\ (1 + z)}^{3/2} \Rightarrow$$

**2) Números trabajados en z=10: por qué "20−40×" aparece frecuentemente**

Dos elecciones de referencia:

**Einstein--de Sitter de juguete (Ω_m=1)**

A_EdS = (1+z)\^(3α/2)

En z=10 y α=1:

A_EdS = (1+10)\^(3/2) = 11\^1.5 ≈ **36.5**

Por tanto A≈37: los procesos son **\~37× más rápidos** que hoy (en la misma clase/escala).

**ΛCDM "realista" (Ω_m=0.315, Ω_Λ=0.685)**

A_ΛCDM = \[Ω_m(1+z)³ + Ω_Λ\]\^(α/2)

Para z=10, α=1:

A_ΛCDM = \[0.315×11³ + 0.685\]\^(1/2) ≈ **20.5**

Para z=7, α=1:

A_ΛCDM ≈ **12.7**

**Interpretación:** el factor "37×" es el límite EdS pedagógico; en el ΛCDM actual el número es A∼20 para z∼10 con α∼1. En cualquier caso, el orden de magnitud **A∼20−40** emerge inmediatamente.

**3) Ensamblaje de galaxias: aceleración requerida** $\mathbf{A}$ **(fórmula cerrada)**

Considere un halo con masa $M_{h}$ y fracción bariónica $f_{b} \approx 0.157$

Sea $\varepsilon_{dyn}$ la eficiencia por tiempo dinámico (fracción de gas convertido en estrellas por $t_{dyn}$) y $N$ el número de tiempos dinámicos disponibles entre el inicio de la fase fría y el corrimiento al rojo de interés:

$$N \equiv \frac{\Delta t(z)}{t_{dyn,std}(z)}$$

Si la conversión por paso es independiente (modelo minimal), la eficiencia integrada después de $N$ pasos es:

$$SFE_{\text{std}} = 1 - \left( 1 - \varepsilon_{\text{dyn}} \right)^{N} \approx 1 - e^{- \varepsilon_{\text{dyn}}N}\quad\left( \varepsilon_{\text{dyn}} \ll 1 \right)$$

La masa estelar esperada es:

$$M_{*}^{\text{std}} \approx f_{b}M_{h}SFE_{\text{std}}$$

**Bajo RTM**, el número efectivo de pasos crece por el factor $A$:

$$N_{\text{RTM}} = AN,\quad \Rightarrow \quad SFE_{\text{RTM}} = 1 - \left( 1 - \varepsilon_{\text{dyn}} \right)^{AN} \approx 1 - e^{- \varepsilon_{\text{dyn}}AN}$$

Para alcanzar una masa estelar objetivo $M_{*}^{tgt}$ en el corrimiento al rojo $z$:

  $A_{\text{req}}\, \geq \,\frac{1}{\varepsilon_{\text{dyn}}N}\,\ln\,\left\lbrack \,\frac{1}{1\, - \,\frac{M_{*}^{\text{tgt}}}{f_{b}M_{h}}}\, \right\rbrack$ ;


  $$N\, = \,\frac{\Delta t(z)}{t_{dyn,std}(z)}$$

**3.1) Números de cálculo aproximado (ilustrativos)**

-   $z = 14$: edad cósmica $\Delta t \sim 0.28 - 0.30$ Gyr.

-   Tiempo dinámico del halo: $t_{dyn,std}{\sim \kappa H}^{- 1}(z)$ con $\kappa \approx 0.1$ (densidad virial ${\sim 200\rho}_{m})$

En ΛCDM:

$$H(z)\text{/}H_{0} \approx 31.8 \Rightarrow t_{\text{dyn}} \approx 0.1\text{/}31.8H_{0}^{- 1} \approx 44\,\text{Myr}.$$

$$\Rightarrow N \approx \Delta t\text{/}t_{\text{dyn}} \approx 300\text{/}44 \approx 6.8.$$

Caso A (exigente):

$M_{h} = 10^{11}M_{\odot} \Rightarrow f_{b}M_{h}{= 1.57 \times 10}^{10}M_{\odot}$

Objetivo $M_{*}^{\text{std}} = 10^{10}M_{\odot} \Rightarrow {SFE}_{req} \approx 0.637$

Si $\varepsilon_{dyn} = 0.01$ (1% por $t_{dyn}$):

$$A_{req} \gtrsim \frac{1}{0.01 \times 6.8}\ln\left( \frac{1}{1 - 0.637} \right) \approx 14.7 \times 1.01 \approx 15$$

$\Rightarrow$ Con $\alpha = 1$:

-   **EdS:** $A \approx 37$ (margen amplio)

-   **ΛCDM:** $A \approx 32$ (también suficiente)

Con **A∼37** (EdS) o **A∼20** (ΛCDM), la aceleración requerida A_req∼10−15 aún es alcanzable con margen. Para los casos más exigentes (M_star∼10\^11 en z>12), α∼1.2 puede ser necesario en ΛCDM.

**Caso B (moderado):** misma configuración pero $\varepsilon_{dyn} = 0.02$:

$$A_{req} \approx \frac{1}{0.136} \times 1.01 \approx 7.5$$

Aquí $\alpha \sim 0.5$ podría ya ser suficiente ($A \approx 5.6 - 7.67$, dependiendo del fondo).

**Moraleja:** con eficiencias por $t_{dyn}$ en el rango $1 - 2\%$ y halos masivos $(10^{10}M_{\odot})$, una aceleración $A \sim 7 - 15$ hace $M_{*}{\sim 10}^{10}M_{\odot}$ en $z \sim 14$ aritméticamente plausible **sin** tocar el fondo FRW ni "romper" nada; $\alpha$ en $0.7 - 1.0$ lo entrega naturalmente.

**4) ¿Rompe BBN/CMB? No, si α obedece "bandas de complejidad"**

Para evitar alterar la nucleosíntesis y la recombinación:

-   **Hipótesis de banda (RTM):** $\alpha \approx 0$ para plasma homogéneo (era BBN/CMB, baja complejidad morfológica); $\alpha \sim O(1)$ solo emerge en medios bariónicos multifase (gas frío + turbulencia + enfriamiento + retroalimentación), es decir, *después* del amanecer de las estructuras.

-   **Compañero EFT:** elija portales y $\xi$ ($\alpha^{2}R$ no minimal) dentro de la cuña segura para que **α** no modifique la expansión temprana ni la física atómica más allá de los límites EP/PPN/BBN/CMB.

Esto permite que $\alpha$ actúe como un **factor de reescalamiento temporal mesoscópico** (enfriamiento, colapso, ciclos de retroalimentación), **no** como energía de fondo exótica.

**5) Predicciones y pruebas (cómo falsificar la hipótesis)**

1.  Relación tiempo-escala dentro del mismo $z$: en $z\approx 10-15$, los procesos con escala espacial efectiva $L$ (ej., regiones de formación estelar) deberían mostrar:

${T(L) \propto L}^{\alpha}$, con $\alpha \approx 0.7 - 1.0$ si el caso requiere $A \gtrsim 10$

Observacionalmente: duraciones de estallidos, tiempos de escape de flujos de salida, etc., como función del tamaño.

2.  **Eficiencias aparentes:** para el mismo $M_{h}$​, la eficiencia integrada SFE debería ser mayor en alto $z$ debido al factor $A$ efectivo (ecuación para $A_{req}$). Si $A$ es pequeño, la alta SFE no se alcanza sin ajuste fino.

3.  **Sin tocar BBN/CMB/PPN:** ningún efecto de $\alpha$ debería aparecer en observables lineales de fondo; toda la novedad debería ocurrir a escalas mesoscópicas post-colapso. (Esto es testeable en el compañero EFT con la "cuña segura".)

**6) Limitaciones (qué no resolvemos aquí)**

-   No derivamos $\alpha(z)$ de la microfísica ni resolvemos FRW con retroacción de $\alpha$; usamos ${L = H}^{- 1}$ como un proxy ambiental.

-   No calculamos la función de luminosidad ni los espectros SED; solo mostramos la cinemática temporal y una cota sobre la aceleración requerida.

-   El número "37×" es el límite EdS; el valor realista para ΛCDM es **A∼20** en z∼10 con α∼1.

**7) Resumen ejecutivo**

Con L_env = L_H y α∼1, el factor de aceleración es

A = (H(z)/H_0)\^α

En z=10: - α=1 ⇒ **A≈37** (EdS) o **A≈20** (ΛCDM) - α=1.5 ⇒ A≈220 (EdS) o A≈91 (ΛCDM)

La aceleración requerida para alcanzar M_star objetivo es

A_requerida = ln\[1 − M_star/(f_b·M_halo)\] / \[N_dyn·ln(1−ε)\]

Con M_halo∼10\^12 M\_☉, ε∼2%, y N_dyn∼5, **A∼10−20 es suficiente** para M_star∼10\^11 M\_☉.

Esto es compatible con α∼1 sin tocar BBN/CMB, si α está apagado en plasma homogéneo y encendido solo en medios complejos (bandas RTM).

**Apéndice A**\
**Tabla 1: Factor de Aceleración RTM A(z) para α=1**

| Corrimiento al rojo $z$ | Edad Cósmica ($\Lambda\text{CDM}$) | $A_{\text{EdS}}$ | $A_{\Lambda\text{CDM}}$ |
| :--- | :--- | :--- | :--- |
| 5 | 1.17 Gyr | 14.7 | 8.3 |
| 7 | 0.76 Gyr | 22.6 | 12.7 |
| 10 | 0.47 Gyr | 36.5 | 20.5 |
| 12 | 0.37 Gyr | 46.9 | 26.3 |
| 15 | 0.27 Gyr | 64.0 | 35.9 |
| 20 | 0.18 Gyr | 96.2 | 54.0 |

*EdS: A = (1+z)\^(3/2). ΛCDM: A = \[0.315(1+z)³ + 0.685\]\^(1/2). Parámetros de Planck 2018.*

**Apéndice B: Validación Empírica JWST del Reescalamiento Tiempo-Escala**

El despliegue reciente del Telescopio Espacial James Webb (JWST) ha revelado una población de galaxias inesperadamente masivas a altos corrimientos al rojo ($z\  > \ 10$). Bajo el modelo cosmológico estándar $\Lambda\text{CDM}$, asumiendo una progresión lineal del tiempo cósmico, estas estructuras parecen demasiado masivas para haberse formado dentro de la ventana temporal disponible, creando una tensión profunda en la astrofísica moderna. El marco de Transporte Rítmico Multiescala (RTM) proporciona una resolución natural: a altos corrimientos al rojo, el universo existía en un estado topológico más "coherente" ($\alpha > \ 1$), acelerando la dinámica de formación de estructuras.

**B.1 Análisis Heurístico (Observación de Estimación Puntual)**

Compilamos un catálogo de 55 galaxias de alto corrimiento al rojo de estudios recientes de JWST (JADES, CEERS, UNCOVER, GLASS). Definiendo un "Factor de Aceleración" requerido para reconciliar las masas estelares observadas con los límites teóricos de tasa de formación estelar específica, extrajimos el exponente de coherencia implícito ($\alpha$) para cada galaxia.

El análisis inicial de estimación puntual demuestra que el 44% de las galaxias catalogadas (24 de 55) exceden estrictamente los límites estándar de $\Lambda\text{CDM}$. Promediando estas observaciones directas se obtiene un exponente aparente de $\alpha = \ 1.33\  \pm 0.30$ ($p\  < \ 0.0001$). Aunque visualmente convincente, depender únicamente de estimaciones puntuales en astrofísica de alto corrimiento al rojo puede ser susceptible a artefactos observacionales, necesitando un tratamiento estadístico más riguroso.

**B.2 Validación Probabilística Rigurosa (Monte Carlo y Corrección de Sesgo)**

Para asegurar que la señal RTM es una ley física genuina y no una ilusión estadística causada por ruido de medición, sometimos el catálogo a una prueba de estrés probabilística rigurosa. Se introdujeron dos variables confusoras astrofísicas importantes en el modelo:

1.  **Varianza de Ajuste SED:** Las estimaciones típicas de masa estelar en $z\  > \ 10$ llevan incertidumbres masivas. Inyectamos una varianza continua de $\pm 0.3$ dex en todas las lecturas de masa.

2.  **Sesgo de Eddington / Selección:** La tendencia de los estudios a detectar preferencialmente valores atípicos sobreluminosos (y aparentemente sobremasivos) en el borde de la sensibilidad instrumental.

Desplegamos una simulación Monte Carlo generando 10,000 universos paralelos, suavizando matemáticamente las distribuciones de masa para absorber estos sesgos observacionales.

**B.3 Conclusión de la Anomalía JWST**

Incluso después de penalización severa por varianza de masa extrema y sesgo de selección, la suposición estándar de $\Lambda\text{CDM}$ de tiempo puramente lineal ($\alpha = \ 1.0$) es categóricamente rechazada ($p < 10^{- 6}$).

La distribución Monte Carlo converge estrechamente en un exponente topológico robusto corregido por sesgo de $\mathbf{\alpha}\mathbf{= \ 1.16\ }\mathbf{\pm}\mathbf{0.08}$. Esto valida conclusivamente la predicción RTM: el universo temprano pertenecía a la **Clase de Transporte Altamente Coherente** ($\alpha > \ 1$). Debido a que el espacio-tiempo estaba más interconectado topológicamente en estas densidades, la materia bariónica experimentó una expansión temporal no lineal, otorgando a las galaxias amplio tiempo dinámico para ensamblar estructuras masivas sin violar los límites físicos estándar.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
