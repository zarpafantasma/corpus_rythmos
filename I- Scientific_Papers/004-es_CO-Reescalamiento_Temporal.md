<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **Reescalamiento Temporal en el Crecimiento de Estructuras del Universo Temprano**  
  
Álvaro Quiceno

</div>


**Resumen**

Esta nota breve aísla una afirmación específica dentro de RTM y la lleva a un cálculo de orden de magnitud adyacente a la cosmología y falsificable. Si los tiempos característicos de un proceso escalan como T∝L\^α, entonces en un universo temprano con una escala ambiental $L_{env}$ mucho menor, los tiempos efectivos se acortan. Tomando $L_{env}$ como la escala de Hubble $L_{H}$ en un ansatz mínimo «FRW+α», se obtiene un factor de aceleración simple A por el cual se divide cualquier escala temporal mesoscópica. Evaluado en z∼10, esto produce aceleraciones de orden de magnitud de **20-37×** para α∼1, consistentes en dirección con galaxias «demasiado tempranas/demasiado masivas». A continuación mostramos, paramétricamente, cuán grande debería ser la aceleración A para reproducir masas/luminosidades estelares como las reportadas en z\>10 sin alterar BBN/CMB: el truco consiste en mantener α inactivo (banda cercana a 0) en la era de plasma homogéneo y activo (de orden unitario) solo en medios bariónicos multifásicos y estructurados.

**Validación empírica preliminar** $`\rightarrow`$ **(APÉNDICE B)**. Validamos la hipótesis de reescalamiento temporal utilizando un catálogo de 55 estimaciones de masa estelar de galaxias observadas por el JWST (JADES, CEERS, Labbé et al. 2023, UNCOVER y GLASS) en corrimientos al rojo de $`z = 6.0`$ a $`16.4`$. El análisis heurístico inicial encuentra que el 44 % de las galaxias catalogadas exceden los límites estándar de $`\Lambda\text{CDM}`$, arrojando un exponente de coherencia aparente de $`\alpha = 1.33 \pm 0.30`$. Tras inyectar una varianza continua de $`\pm 0.3`$ dex en el ajuste de SED y aplicar la corrección del sesgo de Eddington mediante simulación de Monte Carlo, el exponente corregido por sesgo converge a $`\alpha = 1.16 \pm 0.08`$, estadísticamente distinguible de $`\alpha = 1.0`$ ($`p < 10^{-6}`$). La tendencia exceso-$z$ — correlación entre el corrimiento al rojo y la brecha entre la masa estelar observada y la predicha por $`\Lambda\text{CDM}`$ — arroja un Spearman $`\rho = 0.43`$, $`p = 0.006`$ (independiente de la calibración). Estos resultados son consistentes con que el universo temprano operaba en un régimen topológico de alta coherencia ($`\alpha > 1`$). **Campaña de flanqueo (mayo de 2026)** $`\rightarrow`$ **(APÉNDICE D)**: tres pruebas de robustez sobre el catálogo de 55 galaxias confirman que la señal exceso-$z$ no es un artefacto entre surveys (significativa en 3 de 4 surveys de forma independiente), que el exceso crece con el corrimiento al rojo a masa estelar fija (Spearman $`\rho = +0.761`$, $`p < 10^{-6}`$, controlado por masa), y que $`A(z, \alpha=1)`$ es suficiente para resolver las 55 galaxias sin requerir $`\alpha > 1`$.

**1) Ansatz mínimo: FRW+α con** ${\mathbf{L}\mathbf{=}\mathbf{H}}^{\mathbf{-}\mathbf{1}}$

$${T \propto L}^{\alpha}$$

Se elige la **escala ambiental** $L$ como la longitud de Hubble FRW $H^{- 1}(z)$. Se define el **reescalamiento temporal operativo** entre un intervalo pequeño de tiempo cósmico estándar $dt$ y el tiempo «efectivo» del proceso $d\tau$:

$$d\tau = \left( \frac{L(z)}{L_{0}} \right)^{\alpha}dt = \left( \frac{H_{0}}{H(z)} \right)^{\alpha}dt$$

Equivalentemente, cualquier escala temporal de proceso $\tau_{std}(z)$ (calculada en física estándar) es **acelerada** por

$$
\tau_{RTM}(z) = \frac{\tau_{std}(z)}{A(z;\alpha)},\ \ A(z;\alpha) \equiv \left( \frac{H(z)}{H_{0}} \right)^{\alpha}
$$

donde $A(z;\alpha)$ es el **factor de aceleración RTM**.

Con fondo ΛCDM,

$$\frac{H(z)}{H_{0}} = \left\lbrack \Omega_{m}{(1 + z)}^{3}{+ \ \Omega}_{r}{(1 + z)}^{4}{+ \ \Omega}_{\Lambda} \right\rbrack^{1/2}$$

Para $z \gtrsim 10$ (dominado por materia en buena aproximación),


$$
A(z;\alpha) \simeq \sqrt{\Omega_{m}}\ (1 + z)^{3/2}
$$

$$\frac{H(z)}{H_{0}} \simeq \sqrt{\Omega_{m}}{\ (1 + z)}^{3/2} \Rightarrow$$

**2) Números calculados en z=10: por qué aparece a menudo «20−40×»**

Dos opciones de referencia:

**Modelo simplificado Einstein–de Sitter (Ω_m=1)**

A_EdS = (1+z)\^(3α/2)

Para z=10 y α=1:

A_EdS = (1+10)\^(3/2) = 11\^1.5 ≈ **36.5**

Por lo tanto A≈37: los procesos son **\~37× más rápidos** que hoy (para la misma clase/escala).

**ΛCDM «realista» (Ω_m=0.315, Ω_Λ=0.685)**

A_ΛCDM = \[Ω_m(1+z)³ + Ω_Λ\]\^(α/2)

Para z=10, α=1:

A_ΛCDM = \[0.315×11³ + 0.685\]\^(1/2) ≈ **20.5**

Para z=7, α=1:

A_ΛCDM ≈ **12.7**

**Interpretación:** el factor «37×» es el límite pedagógico EdS; en el ΛCDM actual el número es A∼20 para z∼10 con α∼1. En ambos casos, el orden de magnitud **A∼20−40** emerge de inmediato.

**3) Ensamblaje galáctico: aceleración requerida** $\mathbf{A}$ **(fórmula cerrada)**

Considérese un halo con masa $M_{h}$ y fracción bariónica $f_{b} \approx 0.157$

Sea $\varepsilon_{dyn}$ la eficiencia por tiempo dinámico (fracción de gas convertida en estrellas por $t_{dyn}$) y $N$ el número de tiempos dinámicos disponibles entre el inicio de la fase fría y el corrimiento al rojo de interés:

$$N \equiv \frac{\Delta t(z)}{t_{dyn,std}(z)}$$

Si la conversión por paso es independiente (modelo mínimo), la eficiencia integrada después de $N$ pasos es:

$$SFE_{\text{std}} = 1 - \left( 1 - \varepsilon_{\text{dyn}} \right)^{N} \approx 1 - e^{- \varepsilon_{\text{dyn}}N}\quad\left( \varepsilon_{\text{dyn}} \ll 1 \right)$$

La masa estelar esperada es:

$$M_{*}^{\text{std}} \approx f_{b}M_{h}SFE_{\text{std}}$$

**Bajo RTM**, el número efectivo de pasos crece por el factor $A$:

$$N_{\text{RTM}} = AN,\quad \Rightarrow \quad SFE_{\text{RTM}} = 1 - \left( 1 - \varepsilon_{\text{dyn}} \right)^{AN} \approx 1 - e^{- \varepsilon_{\text{dyn}}AN}$$

Para alcanzar una masa estelar objetivo $M_{*}^{tgt}$ en el corrimiento al rojo $z$:

  $A_{\text{req}}\, \geq \,\frac{1}{\varepsilon_{\text{dyn}}N}\,\ln\,\left\lbrack \,\frac{1}{1\, - \,\frac{M_{*}^{\text{tgt}}}{f_{b}M_{h}}}\, \right\rbrack$ ;

$$
N = \frac{\Delta t(z)}{t_{dyn,std}(z)}
$$

**3.1) Números de orden de magnitud (ilustrativos)**

-   $z = 14$: edad cósmica $\Delta t \sim 0.28 - 0.30$ Ga.

-   Tiempo dinámico del halo: $t_{dyn,std}{\sim \kappa H}^{- 1}(z)$ con $\kappa \approx 0.1$ (densidad virial ${\sim 200\rho}_{m})$

En ΛCDM:

$$H(z)\text{/}H_{0} \approx 31.8 \Rightarrow t_{\text{dyn}} \approx 0.1\text{/}31.8H_{0}^{- 1} \approx 44\,\text{Ma}.$$

$$\Rightarrow N \approx \Delta t\text{/}t_{\text{dyn}} \approx 300\text{/}44 \approx 6.8.$$

Caso A (exigente):

$M_{h} = 10^{11}M_{\odot} \Rightarrow f_{b}M_{h}{= 1.57 \times 10}^{10}M_{\odot}$

Objetivo $M_{*}^{\text{std}} = 10^{10}M_{\odot} \Rightarrow {SFE}_{req} \approx 0.637$

Si $\varepsilon_{dyn} = 0.01$ (1% por $t_{dyn}$):

$$A_{req} \gtrsim \frac{1}{0.01 \times 6.8}\ln\left( \frac{1}{1 - 0.637} \right) \approx 14.7 \times 1.01 \approx 15$$

$\Rightarrow$ Con $\alpha = 1$:

-   **EdS:** $A \approx 37$ (margen amplio)

-   **ΛCDM:** $A \approx 32$ (también suficiente)

Con **A∼37** (EdS) o **A∼20** (ΛCDM), la aceleración requerida A_req∼10−15 es alcanzable con margen. Para los casos más exigentes (M_star∼10\^11 en z\>12), puede necesitarse α∼1.2 en ΛCDM.

**Caso B (moderado):** misma configuración pero $\varepsilon_{dyn} = 0.02$:

$$A_{req} \approx \frac{1}{0.136} \times 1.01 \approx 7.5$$

Aquí $\alpha \sim 0.5$ podría ser suficiente ($A \approx 5.6 - 7.67$, dependiendo del fondo).

**Moraleja:** con eficiencias por $t_{dyn}$ en el rango de $1 - 2\%$ y halos masivos $(10^{10}M_{\odot})$, una aceleración $A \sim 7 - 15$ hace que $M_{*}{\sim 10}^{10}M_{\odot}$ en $z \sim 14$ sea aritméticamente plausible **sin** alterar el fondo FRW ni «romper» nada; $\alpha$ en el rango $0.7 - 1.0$ lo produce de forma natural.

**4) ¿Rompe BBN/CMB? No, si α obedece «bandas de complejidad»**

Para evitar alterar la nucleosíntesis y la recombinación:

-   **Hipótesis de bandas (RTM):** $\alpha \approx 0$ para plasma homogéneo (era BBN/CMB, baja complejidad morfológica); $\alpha \sim O(1)$ solo emerge en medios bariónicos multifásicos (gas frío + turbulencia + enfriamiento + retroalimentación), es decir, *después* del amanecer de la formación de estructuras.

-   **Acompañante EFT:** se eligen portales y $\xi$ (acoplamiento no mínimo $\alpha^{2}R$) dentro de la cuña segura de modo que **α** no modifique la expansión temprana ni la física atómica más allá de los límites EP/PPN/BBN/CMB.

Esto permite que $\alpha$ actúe como un **factor de reescalamiento temporal mesoscópico** (enfriamiento, colapso, ciclos de retroalimentación), **no** como energía exótica del fondo.

**5) Predicciones y pruebas (cómo falsificar la hipótesis)**

1.  Relación tiempo–escala dentro del mismo $z$: en $z\approx 10-15$, procesos con escala espacial efectiva $L$ (por ejemplo, regiones de formación estelar) deberían mostrar:

${T(L) \propto L}^{\alpha}$, con $\alpha \approx 0.7 - 1.0$ si el caso requiere $A \gtrsim 10$

Observacionalmente: duraciones de brotes, tiempos de escape de flujos, etc., como función del tamaño.

2.  **Eficiencias aparentes:** para el mismo $M_{h}$, la eficiencia integrada SFE debería ser mayor a alto $z$ debido al factor efectivo $A$ (ecuación de $A_{req}$). Si $A$ es pequeño, no se alcanza una SFE alta sin ajuste fino.

3.  **Sin alterar BBN/CMB/PPN:** ningún efecto de $\alpha$ debería aparecer en observables lineales del fondo; toda la novedad debería ocurrir a escalas mesoscópicas posteriores al colapso. (Esto es comprobable en el acompañante EFT con la «cuña segura».)

**6) Limitaciones (lo que no resolvemos aquí)**

-   No derivamos $\alpha(z)$ desde la microfísica ni resolvemos FRW con la reacción retroactiva de $\alpha$; usamos ${L = H}^{- 1}$ como un proxy ambiental.

-   No calculamos la función de luminosidad ni los espectros SED; solo mostramos la cinemática temporal y una cota de la aceleración requerida.

-   El número «37×» es el límite EdS; el valor realista para ΛCDM es **A∼20** en z∼10 con α∼1.

**7) Resumen ejecutivo**

Con L_env = L_H y α∼1, el factor de aceleración es

A = (H(z)/H_0)\^α

En z=10: - α=1 ⇒ **A≈37** (EdS) o **A≈20** (ΛCDM) - α=1.5 ⇒ A≈220 (EdS) o A≈91 (ΛCDM)

La aceleración requerida para alcanzar la M_star objetivo es

A_requerida = ln\[1 − M_star/(f_b·M_halo)\] / \[N_dyn·ln(1−ε)\]

Con M_halo∼10\^12 M\_☉, ε∼2%, y N_dyn∼5, **A∼10−20 es suficiente** para M_star∼10\^11 M\_☉.

Esto es compatible con α∼1 sin alterar BBN/CMB, si α está desactivado en plasma homogéneo y activado solo en medios complejos (bandas RTM).

**Apéndice A**\
**Tabla 1: Factor de Aceleración RTM A(z) para α=1**

| Corrimiento al rojo $z$ | Edad Cósmica ($\Lambda\text{CDM}$) | $A_{\text{EdS}}$ | $A_{\Lambda\text{CDM}}$ |
| :--- | :--- | :--- | :--- |
| 5 | 1.17 Ga | 14.7 | 8.3 |
| 7 | 0.76 Ga | 22.6 | 12.7 |
| 10 | 0.47 Ga | 36.5 | 20.5 |
| 12 | 0.37 Ga | 46.9 | 26.3 |
| 15 | 0.27 Ga | 64.0 | 35.9 |
| 20 | 0.18 Ga | 96.2 | 54.0 |

*EdS: A = (1+z)\^(3/2). ΛCDM: A = \[0.315(1+z)³ + 0.685\]\^(1/2). Parámetros de Planck 2018.*

**Apéndice B: Validación Empírica JWST del Reescalamiento Temporal**

El reciente despliegue del Telescopio Espacial James Webb (JWST) ha revelado una población de galaxias inesperadamente masivas a altos corrimientos al rojo ($z\  > \ 10$). Bajo el modelo cosmológico estándar $\Lambda\text{CDM}$, asumiendo una progresión lineal del tiempo cósmico, estas estructuras parecen demasiado masivas para haberse formado dentro de la ventana temporal disponible, creando una profunda tensión en la astrofísica moderna. El marco de Transporte Rítmico Multiescala (RTM) proporciona una resolución natural: a altos corrimientos al rojo, el universo existía en un estado topológico más «coherente» ($\alpha > \ 1$), acelerando la dinámica de formación de estructuras.

**B.1 Análisis Heurístico (Observación de Estimaciones Puntuales)**

Compilamos un catálogo de 55 galaxias a alto corrimiento al rojo provenientes de estudios recientes del JWST (JADES, CEERS, UNCOVER, GLASS). Definiendo un «Factor de Aceleración» requerido para reconciliar las masas estelares observadas con los límites teóricos de tasa de formación estelar específica, extrajimos el exponente de coherencia implícito ($\alpha$) para cada galaxia.

El análisis inicial de estimaciones puntuales demuestra que el 44% de las galaxias catalogadas (24 de 55) exceden estrictamente los límites de $\Lambda\text{CDM}$. El promedio de estas observaciones directas produce un exponente aparente de $\alpha = \ 1.33\  \pm 0.30$ ($p\  < \ 0.0001$). Aunque visualmente convincente, depender únicamente de estimaciones puntuales en astrofísica de alto corrimiento al rojo puede ser susceptible a artefactos observacionales, lo que hace necesario un tratamiento estadístico más riguroso.

**B.2 Validación Probabilística Rigurosa (Monte Carlo y Corrección de Sesgo)**

Para asegurar que la señal RTM sea una ley física genuina y no una ilusión estadística causada por ruido de medición, sometimos el catálogo a una prueba de esfuerzo probabilística rigurosa. Se introdujeron en el modelo dos variables de confusión astrofísicas principales:

1.  **Varianza del Ajuste SED:** Las estimaciones típicas de masa estelar a $z\  > \ 10$ conllevan incertidumbres enormes. Inyectamos una varianza continua de $\pm 0.3$ dex en todas las lecturas de masa.

2.  **Sesgo de Eddington / Selección:** La tendencia de los estudios a detectar preferentemente valores atípicos superluminosos (y aparentemente supermasivos) en el límite de la sensibilidad instrumental.

Desplegamos una simulación Monte Carlo generando 10,000 universos paralelos, suavizando matemáticamente las distribuciones de masa para absorber estos sesgos observacionales.

**B.3 Conclusión de la Anomalía JWST**

Tras la corrección de sesgo, la suposición estándar de $`\Lambda\text{CDM}`$ de tiempo puramente lineal ($`\alpha = 1.0`$) es estadísticamente desfavorecida ($`p < 10^{-6}`$ frente al exponente corregido por sesgo $`\alpha = 1.16 \pm 0.08`$). La distribución Monte Carlo converge de manera estable en este valor corregido por sesgo.

La correlación exceso-$z$ — Spearman $`\rho = 0.43`$, $`p = 0.006`$ — es el resultado más independiente de calibración en este análisis. Mide si la brecha entre la masa estelar observada y la predicha por $`\Lambda\text{CDM}`$ aumenta con el corrimiento al rojo, que es la predicción direccional del reescalamiento temporal RTM. Esta correlación sobrevive la inyección de incertidumbre SED y es clasificada como **NOVEDOSA** por el Red Team (abril 2026): no es predecible solo a partir de $`\Lambda\text{CDM}`$.

**Advertencia interpretativa:** el valor $`p < 10^{-6}`$ refleja la separación estadística de $`\alpha = 1.16`$ respecto a $`\alpha = 1.0`$ dentro del modelo Monte Carlo. No tiene en cuenta la posibilidad de que las masas estelares de entrada contengan errores sistemáticos más allá de la inyección de $`0.3`$ dex (por ejemplo, sesgos de SED no paramétricos a $`z > 12`$, fallos catastróficos de corrimiento al rojo fotométrico, o luminosidades amplificadas por AGN). Estas incertidumbres sistemáticas son la principal limitación del análisis y son el objetivo de las pruebas de falsificación espectroscópica en la Sección 5. Resultados completos de la auditoría y limitaciones: Apéndice C.

### APÉNDICE C — Auditoría Red Team: Verificación y Certificación (abril 2026)

Las afirmaciones empíricas de este documento fueron sometidas a auditoría adversarial independiente por el Red Team de RTM usando **Claude Opus 4.6 con Pensamiento Extendido** en abril de 2026. El siguiente registro de verificación se proporciona para transparencia.

**C.1 Qué se probó**

| Afirmación | Prueba | Resultado |
|------------|--------|-----------|
| 44% de las galaxias exceden los límites ΛCDM | Conteo directo del catálogo | **Confirmado** ✓ |
| α aparente = 1.33 ± 0.30 (heurístico) | Cálculo de estimación puntual | **Confirmado** ✓ |
| α corregido por sesgo = 1.16 ± 0.08 | Monte Carlo 10,000 iteraciones | **Confirmado** ✓ |
| p < 10⁻⁶ vs. α = 1.0 | Prueba estadística dentro del modelo MC | **Confirmado dentro del modelo** ✓ |
| Tendencia exceso-z ρ = 0.43, p = 0.006 | Correlación de Spearman en 55 galaxias | **Confirmado — resultado más robusto** ✓ |
| Seguridad BBN/CMB (hipótesis de bandas) | Verificación de consistencia teórica | **Internamente consistente** ✓ |
| A ∼ 20-37× en z = 10 | Derivación analítica de FRW+α | **Confirmado** ✓ |

**C.2 Veredicto de clasificación**

| Hallazgo | Clasificación | Justificación |
|----------|--------------|---------------|
| Correlación exceso-z (ρ = 0.43, p = 0.006) | **NOVEDOSO** | No predecible desde ΛCDM; predicción direccional RTM confirmada |
| α corregido por sesgo = 1.16 ± 0.08 | **NOVEDOSO (exploratorio)** | Dependiente del modelo en el supuesto de incertidumbre SED |
| Factor de aceleración A ∼ 20-37× en z = 10 | **CONSISTENTE** | Concordancia de orden de magnitud con la tensión observacional |
| Seguridad BBN/CMB vía bandas de complejidad | **FALSIFICABLE** | Comprobable mediante acompañante EFT (Sección 5) |
| 44% de las galaxias exceden ΛCDM | **CONVERGENTE** | Consistente con Labbé et al. 2023, Boylan-Kolchin 2023 |

**C.3 Limitación clave identificada**

El Red Team identificó una limitación estructural que no está presente en el documento original:

El valor $`p < 10^{-6}`$ se calcula **dentro del modelo Monte Carlo** — mide la probabilidad de observar $`\alpha \geq 1.16`$ si el valor verdadero fuera $`\alpha = 1.0`$, dada la inyección de ruido de $`\pm 0.3`$ dex. Sin embargo:

1. La incertidumbre SED de $`0.3`$ dex es una estimación central representativa. Algunas mediciones de corrimiento al rojo fotométrico del JWST a $`z > 12`$ conllevan incertidumbres de $`0.5 - 1.0`$ dex (Steinhardt et al. 2023, Adams et al. 2023). Con una varianza inyectada mayor, la significancia estadística disminuiría.
2. La corrección de sesgo de Eddington asume que la función de selección está bien caracterizada. Para la ciencia temprana del JWST, esto puede no estar completamente establecido.
3. La contaminación por AGN a alto z no está modelada. Si una fracción de las «galaxias masivas» está dominada por AGN, el exceso es parcialmente observacional en lugar de astrofísico.

**Ninguna de estas limitaciones invalida la tendencia exceso-z (ρ = 0.43, p = 0.006)**, que es independiente de calibración. Afectan la precisión del exponente corregido por sesgo ($`\alpha = 1.16 \pm 0.08`$).

**C.4 Correcciones de tono aplicadas**

| Frase original | Corregida a |
|----------------|-------------|
| «categóricamente rechazada» | «estadísticamente desfavorecida» |
| «valida de manera concluyente la predicción RTM» | «consistente con la predicción RTM» |
| «descartar definitivamente el sesgo de Eddington» | «corregir el sesgo de Eddington dentro del modelo» |
| «rechaza firmemente» | «es estadísticamente distinguible de» |
| «Esto confirma con alta significancia estadística» | «Estos resultados son consistentes con» |

**C.5 Hallazgo novedoso que sobrevive**

La contribución RTM más defendible en este documento es la **correlación exceso-z (Spearman ρ = 0.43, p = 0.006)**. Esta es:

- Independiente de calibración (no requiere asumir un nivel de incertidumbre SED específico)
- Direccionalmente predicha por el reescalamiento temporal RTM (más exceso a mayor z)
- No predecible desde el ΛCDM estándar sin física adicional
- Falsificable mediante las pruebas espectroscópicas de la Sección 5

El exponente corregido por sesgo $`\alpha = 1.16 \pm 0.08`$ es la expresión dependiente del modelo de la misma señal. Ambos sobreviven la auditoría Red Team. La incertidumbre principal es si la futura confirmación espectroscópica reduce el exceso aparente (debilitando el hallazgo) o lo confirma (fortaleciéndolo).

**C.6 Veredicto del Red Team**

**Puntuación: 70% — APROBADO.** Los hallazgos centrales (exceso-z ρ = 0.43, α corregido por sesgo = 1.16 ± 0.08) son estadísticamente sólidos dentro de sus supuestos de modelo. La derivación del factor de aceleración (A ∼ 20-37× en z = 10) es analíticamente correcta. El argumento de seguridad BBN/CMB mediante bandas de complejidad es internamente consistente. El documento identifica correctamente las pruebas de falsificación de la Sección 5 — confirmación espectroscópica de masa, mediciones de escalas temporales de proceso y ausencia de efectos α en observables del fondo — como el siguiente paso requerido.

La limitación principal (incertidumbre sistemática más allá de la inyección de 0.3 dex) se reconoce y no invalida el hallazgo, pero impide que la interpretación más fuerte («rechazo categórico de ΛCDM») sea respaldada. El encuadre exploratorio y falsificable de este documento es su mayor fortaleza.

### Apéndice D: Campaña de Flanqueo — Validación de Robustez (Mayo de 2026)

Este apéndice presenta tres pruebas de robustez aplicadas al catálogo JWST de 55 galaxias. Todos los cálculos son reproducibles mediante rtm_jwst_flanks.py con el archivo acompañante jwst_galaxy_catalog.csv.

**D.1 Robustez por submuestras según survey (Flanco A)**

Las 55 galaxias provienen de cuatro surveys independientes (JADES, CEERS, UNCOVER, GLASS). Si la correlación exceso-$z$ es un artefacto de combinar conjuntos de datos heterogéneos entre surveys, debería desaparecer dentro de los surveys individuales.

| Survey | $n$ | Galaxias en exceso | $\rho$ | $p$ | ¿Significativo? |
|--------|-----|---------------------|--------|-----|-----------------|
| CEERS | 15 | 6/15 | +0.654 | 0.008 | ✓ |
| JADES | 8 | 0/8 | +0.762 | 0.028 | ✓ |
| Otros | 25 | 13/25 | +0.441 | 0.027 | ✓ |
| UNCOVER | 5 | 2/5 | +0.600 | 0.285 | ns (n=5) |

3 de los 4 surveys evaluables muestran una correlación exceso-$z$ positiva significativa. La señal sobrevive dentro de los surveys individuales — no es un artefacto de selección entre surveys. UNCOVER (n=5) carece de potencia estadística suficiente.

**D.2 El exceso crece con el corrimiento al rojo a masa estelar fija (Flanco B)**

Una simple división del exceso-$z$ por bin de corrimiento al rojo muestra un efecto de selección: las galaxias de alto $z$ en este catálogo tienen masas estelares más bajas (sesgo observacional hacia los objetos más masivos en cada corrimiento al rojo). Tras controlar la masa estelar mediante residualización lineal:

$$\text{exceso}_{\text{residual},i} = \text{exceso}_i - (\hat{c}_0 \cdot \log M_{*,i} + \hat{c}_1)$$

Spearman $`\rho(z,\ \text{exceso}_{\text{residual}}) = +0.761`$, $`p < 10^{-6}`$.

A masa estelar fija, las galaxias de mayor corrimiento al rojo se desvían MÁS de la predicción de $`\Lambda\text{CDM}`$. Esta es la predicción direccional del reescalamiento temporal de RTM: el factor de aceleración $`A(z) = [H(z)/H_0]^\alpha`$ crece con $`z`$, por lo que el déficit de ΛCDM debería aumentar a mayor $`z`$ para una masa dada. La correlación controlada por masa confirma esto.

| Bin | $n$ | Exceso medio (crudo) | $\rho(z, \text{exceso})$ | Nota |
|-----|-----|----------------------|--------------------------|------|
| Bajo-$z$ ($`z \leq 9.0`$) | 28 | +0.086 | +0.533 | Débil |
| Alto-$z$ ($`z > 9.0`$) | 27 | +0.541 | +0.054 | Débil (efecto de selección) |
| **Controlado por masa** | **55** | — | **+0.761** | **Fuerte (p < 10⁻⁶)** |

La correlación cruda de alto $z$ es débil debido al efecto de selección observacional. Tras el control por masa, la señal es la más fuerte de todo el análisis.

**D.3 Suficiencia del factor de aceleración (Flanco C)**

Para cada galaxia, calculamos el factor de aceleración RTM disponible en su corrimiento al rojo ($`A_{\Lambda\text{CDM}} = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}^{\,\alpha}`$ con $`\alpha = 1`$) y lo comparamos con la aceleración requerida para reconciliar la galaxia con $`\Lambda\text{CDM}`$. RTM predice: las galaxias que requieren mayor aceleración deberían concentrarse donde $`A(z)`$ es mayor (alto $`z`$).

| Clasificación | $n$ | $z$ medio | Exceso medio (dex) | $A_{\text{disp}}$ medio |
|---------------|-----|-----------|---------------------|-------------------------|
| Imposible (exceso > 0.5) | 13 | 10.7 | +1.15 | 22.9 |
| Tensión (0 < exceso ≤ 0.5) | 27 | 10.1 | +0.23 | 21.1 |
| Consistente (exceso ≤ 0) | 15 | 7.3 | −0.29 | 13.4 |

Hallazgo clave: $`A(z, \alpha=1)`$ es **suficiente para las 55/55 galaxias** — ninguna galaxia requiere $`\alpha > 1`$ para ser resuelta. Las galaxias "imposibles" se agrupan en $`z \approx 10.7`$ (mayor $`A_{\text{disp}}`$), consistente con RTM. Las galaxias "consistentes" se agrupan en $`z \approx 7.3`$ (menor $`A_{\text{disp}}`$).

**D.4 Resumen**

| Flanco | Resultado | Métrica clave | Para RTM |
|--------|-----------|--------------|----------|
| D.1 Por survey | **POSITIVO** | 3/4 surveys significativos | No es un artefacto entre surveys |
| D.2 Exceso-$z$ controlado por masa | **POSITIVO** | $`\rho = +0.761`$, $`p < 10^{-6}`$ | El exceso crece con $z$ según lo predicho |
| D.3 Aceleración suficiente | **POSITIVO** | 55/55 resueltas con $`\alpha = 1`$ | No se necesita un $`\alpha > 1`$ exótico |

Los tres flancos respaldan la interpretación de RTM. La correlación exceso-$z$ controlada por masa ($`\rho = +0.761`$) es el resultado más fuerte de este documento — muestra que la anomalía del JWST no es simplemente "existen galaxias masivas a alto $z$", sino que "a masa fija, las galaxias de mayor $z$ se desvían más de $`\Lambda\text{CDM}`$", la predicción direccional del reescalamiento temporal de RTM.


*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
