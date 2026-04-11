<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **Geometría Escala–Reloj**  
**Una Fundación Matemática para RTM**  
  
Álvaro Quiceno


</div>

**Resumen**

Desarrollamos una fundación rigurosa para la **Relatividad Temporal Multiescala (RTM)** como una teoría matemática del **escalamiento tiempo–escala**. Partiendo de un axioma de semigrupo de escala $`T(bL) = f(b)\text{ }T(L)`$ con regularidad moderada, derivamos la **ley de potencia** $`T(L) = \kappa L^{\alpha}`$, identificando el **exponente de coherencia** $`\alpha`$ como una **pendiente invariante de reloj** y $`\kappa`$ como un factor de gauge (reloj). Reformulamos RTM en forma geométrica vía la 1-forma

``` math
\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha\text{ }d(\log L),
```

y demostramos que el **colapso**—independencia residual de $`\log T - \alpha\ \log L`$ respecto a $`{log\ }L`$—es equivalente a la **exactitud/planitud** de $`\omega`$ en un bin; la mezcla de regímenes y alternativas no potenciales aparecen como **holonomía/curvatura**. Embebemos RTM en **variación regular** con **exponentes variables**, cuantificando el sesgo de ventana finita y mostrando que las estadísticas de colapso escalan con la curvatura. Un operador de **renormalización** (dilatación de escala + re-gauge) tiene leyes de potencia como **puntos fijos** y es **contractivo** en clases Hölder/Zygmund; los relojes lentamente variables yacen en una **variedad central**, y los exponentes lentamente variables producen **atracción adiabática**. En dinámica, RTM actúa como un **reloj dependiente del espacio** para difusiones y formas de Dirichlet, dando exponentes de similaridad $`z = m + \alpha`$ y leyes de tiempo de salida $`T \sim R^{\text{ }m + \alpha}`$ con cotas de error adiabáticas. Para inferencia bajo errores en variables, mostramos consistencia de **ODR/TLS**, **SIMEX**, y **Theil–Sen** para $`\alpha`$ local, y formalizamos la estadística de colapso como una prueba de especificación contra curvatura. Un empaquetado teórico-categorial hace de los relojes un gauge y de la pendiente el invariante de móduli, clarificando el comportamiento funtorial bajo productos y engrosamiento.  Concluimos con contraejemplos constructivos y problemas abiertos (pruebas de holonomía, configuraciones de grafos, problemas inversos, ruido de colas pesadas).

**1. Introducción**

**1.1 Problema y perspectiva**

Muchos sistemas exhiben una relación sistemática entre un **tiempo característico** $`T`$ y un **proxy de escala** $`L`$: las unidades más grandes operan con relojes más lentos, las unidades más pequeñas con relojes más rápidos. RTM postula que **dentro de un entorno fijo** esta relación es **multiplicativamente consistente** bajo reescalamiento de $`L`$. La práctica empírica—vista a través de la física, biología y economía—es examinar la pendiente de $`\log T`$ vs. $`{log\ }L`$ y probar si los residuos "colapsan" después de eliminar la tendencia por esa pendiente.

Este artículo suministra una **columna vertebral matemática** para esa práctica. Nuestra afirmación central es que la **pendiente** $`\alpha`$ es el objeto estructural (invariante bajo cambios de reloj), mientras que los **relojes** son un gauge. Con esta separación, RTM se convierte en una teoría limpia que vincula: (i) ecuaciones funcionales → leyes de potencia, (ii) una **1-forma/conexión** cuya planitud codifica el colapso, (iii) **variación regular** con exponentes variables para cuantificar efectos de ventana finita, (iv) **renormalización** como dinámica de dilatación de escala con puntos fijos de ley de potencia, (v) **difusiones con relojes dependientes del espacio**, y (vi) **identificabilidad estadística** bajo error de medición.

> [!NOTE]
> **Nota de alcance.** Este documento opera bajo los Supuestos 1–6 del Doc 001 (Sec. 2.1): la densidad local $\rho$ y la temperatura $\Theta$ son uniformes dentro de cada bin. Bajo estas condiciones, la ley maestra completa $T/T_0 = (L/L_0)^\alpha \cdot \Theta(\mathcal{T})/\sqrt{\rho/\rho_0}$ se reduce a $T = \kappa L^\alpha$. Para el tratamiento de $\rho$ y $\Theta$ variables, ver Doc 001 Sec. 2.1–2.2. Para aplicaciones empíricas donde estos supuestos pueden violarse, la prueba de colapso (Sec. 3.2, 7) proporciona un diagnóstico falsificable.

**1.2 Contribuciones**

1.  **Semigrupo → Ley de potencia (Sec. 2).**\
    De $`T(bL) = f(b)\text{ }T(L)`$ con $`f`$ medible/continua, obtenemos $`f(b) = b^{\alpha}`$ y $`T(L) = \kappa L^{\alpha}`$. La pendiente $`\alpha`$ es **invariante de reloj**; la ordenada al origen $`\log\kappa`$ es el reloj.

2.  **Geometría escala–reloj (Sec. 3).**\
    Con $`\omega = d\ \log T - \alpha\text{ }d\ \log L`$, demostramos\
    **colapso ⇔ exactitud/planitud** de $`\omega`$ en un bin. La mezcla de regímenes y el comportamiento no potencial se manifiestan como **holonomía** ($`\oint\omega \neq 0`$). Cuantificamos el **colapso adiabático** cuando $`\alpha`$ deriva lentamente.

3.  **Variación regular con exponente variable (Sec. 4).**\
    Formalizamos $`T(L;x) = L^{\alpha(x)}\mathcal{l}(L;x)`$ con variación lenta **uniforme**, derivamos el **sesgo de ventana finita** $`O( \parallel \partial_{u}\alpha \parallel \text{ }h)`$, y mostramos que las estadísticas de colapso escalan como la curvatura $`\sim h^{2}`$.

4.  **Renormalización y estabilidad (Sec. 5).**\
    Un operador de dilatación-más-re-gauge tiene leyes de potencia como **puntos fijos**; en clases Hölder/Zygmund es una **contracción**, dando **atracción local** a la variedad de ley de potencia. Los relojes lentamente variables forman una **variedad central**; los exponentes lentamente variables producen **seguimiento adiabático**.

5.  **Difusiones RTM y formas de Dirichlet (Sec. 6).**\
    Sea la conductividad $`L(x)^{- \alpha(x)}`$. Con $`\alpha`$ constante, las soluciones obedecen autosimilaridad con exponente dinámico $`z = m + \alpha`$ y escalamiento de tiempo de salida $`T \sim R^{\text{ }z}`$; con deriva lenta obtenemos **cotas de error adiabáticas**. Las difusiones RTM son **movimientos brownianos con cambio de tiempo**.

6.  **Identificabilidad e inferencia bajo EIV (Sec. 7).**\
    Mostramos que **ODR/TLS** y **SIMEX** recuperan consistentemente $`\alpha`$ local bajo supuestos estándar; **Theil–Sen** proporciona verificaciones robustas. La **estadística de colapso** es una prueba de especificación contra curvatura incluso con error de medición.

7.  **Empaquetado categorial (Sec. 8).**\
    Los objetos portan $`\omega`$ y un gauge; la **pendiente** es el invariante de móduli, los relojes son gauge; los productos suman pendientes; el engrosamiento es un funtor con puntos fijos de ley de potencia.

8.  **Ejemplos, contraejemplos, problemas abiertos (Sec. 9).**\
    Damos fallas constructivas (quiebres, log–log curvo), y listamos direcciones abiertas (pruebas de holonomía, grafos, problemas inversos, errores de colas pesadas).

**1.3 Relación con trabajo previo**

Nuestro uso de ecuaciones de Cauchy multiplicativas y **variación regular** sigue a Karamata/de Haan, pero adaptado a **exponentes variables** uniformes en el entorno. El enmarcado de 1-forma es paralelo a las ideas estándar de **gauge/conexión**, aquí especializadas a un fibrado escala–reloj de modo que "colapso" es **exactitud**. El operador de renormalización es clásico en espíritu pero aplicado a **funciones log-tiempo** con **equivalencia de gauge**; la contracción en espacios Hölder/Zygmund proporciona un camino limpio desde el escalamiento empírico a una imagen dinámica de punto fijo. Los resultados de difusión conectan con **procesos con cambio de tiempo** y teoría elíptica de coeficientes variables, dando exponentes de similaridad explícitos vinculados a $`\alpha`$. Sobre inferencia, nuestras declaraciones sitúan estimadores de **errores en variables** dentro de la estructura de invariancia RTM, clarificando qué es y qué no es identificable.

**1.4 Alcance y falsificabilidad**

RTM está destinado para **bins**—dominios donde el entorno es suficientemente estable para que los relojes sean $`L`$-independientes. La teoría **predice sus propios modos de falla**: la curvatura no potencial o las mezclas de regímenes producen holonomía y estadísticas de colapso no nulas. Estos son **límites de alcance**, no defectos.

**1.5 Hoja de ruta del artículo**

- **Sec. 2:** axioma de semigrupo ⇒ ley de potencia; invariancia de reloj.

- **Sec. 3:** geometría escala–reloj; colapso = exactitud; holonomía y colapso adiabático.

- **Sec. 4:** variación regular con exponente variable; sesgo de ventana finita; pruebas de curvatura.

- **Sec. 5:** puntos fijos de renormalización; contracción y variedad central; seguimiento adiabático.

- **Sec. 6:** difusiones RTM/formas de Dirichlet; exponente de similaridad $`z = m + \alpha`$; tiempos de salida; cambio de tiempo.

- **Sec. 7:** identificabilidad y consistencia bajo EIV; colapso como prueba de especificación.

- **Sec. 8:** formulación categorial y propiedades funtoriales.

- **Sec. 9:** ejemplos, contraejemplos, problemas abiertos; conclusión concisa.

**Principio guía:** *la estructura vive en la pendiente; los relojes viven en el gauge.* El resto del artículo hace esto preciso a través de análisis, geometría, dinámica e inferencia.

**2. Semigrupo de Escala → Ley de Potencia (Fundamentos)**

Esta sección formaliza el axioma de escalamiento detrás de RTM y deriva la forma de ley de potencia $`T(L) = \kappa L^{\alpha}`$. También aislamos el **reloj** como un gauge multiplicativo y demostramos que la **pendiente** $`\alpha`$ es el invariante estructural. A lo largo de esta sección, $`L \in \mathbb{R}_{> 0}`$ denota una variable de tamaño/escala y $`T(L) \in \mathbb{R}_{> 0}`$ un tiempo característico.

**2.1 Axiomas y consecuencias**

Separamos **simetría de escala** de **elección de reloj**.

**Axioma 2.1 (Semigrupo de escala).**\
Existe una familia de mapas $`\{ S_{b}\}_{b > 0}`$ (escalamientos por factor $`b`$) y una función $`f:\mathbb{R}_{> 0} \rightarrow \mathbb{R}_{> 0}`$ tales que para todo $`b > 0`$ y todo $`L > 0`$,

``` math
T(S_{b}L) = T(bL) = f(b)\text{ }T(L),
```

con $`f(1) = 1`$ y

``` math
f(b_{1}b_{2}) = f(b_{1})\text{ }f(b_{2})\ \ \ \ \ \ \ \ (\text{composición de semigrupo}).
```

**Axioma 2.2 (Regularidad moderada).**\
Ya sea (i) $`f`$ es medible en $`\mathbb{R}_{> 0}`$, o (ii) $`f`$ es continua en $`b = 1`$.\
(Cualquier regularidad estándar—Baire/medible/localmente acotada—servirá.)

**Definición 2.3 (Transformación de reloj).**\
Un cambio de unidades de medición o temporización base es un mapa $`T \mapsto T^{\#}`$ de la forma $`T^{\#}(L) = c\text{ }T(L)`$ para alguna constante $`c > 0`$ (o, más generalmente, $`c = c(x)`$ dependiendo de un parámetro de **entorno** externo $`x`$, pero *independiente de* $`L`$ dentro de un entorno fijo).

**2.2 Solución de la ecuación funcional**

**Lema 2.4 (Cauchy multiplicativo).**\
Bajo los Axiomas 2.1–2.2, $`f(b) = b^{\alpha}`$ para algún $`\alpha \in \mathbb{R}`$.

*Demostración.* Sea $`g(\log b): = \log f(b)`$. La ley de semigrupo da $`g(u + v) = g(u) + g(v)`$. La medibilidad (o continuidad en 0) fuerza $`g(u) = \alpha u`$ para algún $`\alpha \in \mathbb{R}`$. Exponenciando, $`f(b) = e^{g(\log b)} = b^{\alpha}`$.

**Teorema 2.5 (Representación de ley de potencia).**\
Fije cualquier $`L_{0} > 0`$. Bajo los Axiomas 2.1–2.2,

``` math
T(L) = T(L_{0})\text{ }(\frac{L}{L_{0}})^{\alpha} = \kappa\text{ }L^{\alpha},\ \ \ \ \ \ \ \ \text{donde        }\kappa: = T(L_{0})L_{0}^{- \alpha}.
```

*Demostración.* Aplicar el Lema 2.4 con $`b = L/L_{0}`$:

``` math
T(L) = T\text{ }((L/L_{0})L_{0}) = f(L/L_{0})\text{ }T(L_{0}) = (L/L_{0})^{\alpha}T(L_{0}).
```

Reorganizar para definir $`\kappa`$.

**Corolario 2.6 (Forma log-lineal).**\
$`\log T = \alpha\ \log L + \log\kappa`$. Por tanto la **pendiente** $`\alpha`$ captura el escalamiento, mientras que la **ordenada al origen** $`\log\kappa`$ captura el reloj.

**2.3 Invariancia de reloj e identificabilidad de** $`\mathbf{\alpha}`$

**Proposición 2.7 (Invariancia de reloj de la pendiente).**\
Si $`T^{\#}(L) = c\text{ }T(L)`$ con $`c > 0`$ independiente de $`L`$ (dentro de un entorno fijo), entonces

``` math
\log T^{\#} = \alpha\log L + (\log\kappa + \log c),
```

así que la pendiente de regresión de $`{log\ }T^{\#}`$ sobre $`\log L`$ es igual a $`\alpha`$.

*Demostración.* Inmediato del corolario.

**Observación 2.8 (Relojes dependientes del entorno).**\
Si el factor de reloj depende de una etiqueta externa $`x`$ pero no de $`L`$—es decir, $`T^{\#}(L;x) = c(x)\text{ }T(L;x)`$—entonces dentro de cualquier bin de $`x`$-entorno fijo la pendiente permanece $`\alpha(x)`$, mientras que la ordenada al origen se desplaza por $`\log c(x)`$.

**Proposición 2.9 (Unicidad salvo reloj).**\
Si $`T_{1}(L) = \kappa_{1}L^{\alpha_{1}}`$ y $`T_{2}(L) = \kappa_{2}L^{\alpha_{2}}`$ satisfacen $`T_{2}(L) = c\text{ }T_{1}(L)`$ para todo $`L`$ con algún $`c > 0`$, entonces $`\alpha_{1} = \alpha_{2}`$ y $`c = \kappa_{2}/\kappa_{1}`$.

*Demostración.* Tomar logaritmos y comparar coeficientes de $`\log L`$.

**2.4 Generalización de variación regular (opcional pero útil)**

El escalamiento exacto puede relajarse a **variación regular**, que cubre relojes lentamente variables y leyes de potencia asintóticas.

**Definición 2.10 (Variación regular de Karamata).**\
Una función positiva $`T`$ es *regularmente variable de índice* $`\alpha`$ si para todo $`b > 0`$,

``` math
\underset{L \rightarrow \infty}{\lim}\frac{T(bL)}{T(L)} = b^{\alpha}.
```

Equivalentemente, $`T(L) = L^{\alpha}\text{ }\mathcal{l}(L)`$ con $`\mathcal{l}`$ *lentamente variable* ($`\mathcal{l}(bL)\mathcal{/l}(L) \rightarrow 1`$).

**Teorema 2.11 (RTM bajo variación regular).**\
Si $`T`$ es regularmente variable con índice $`\alpha`$, entonces las pendientes log–log locales sobre ventanas compactas de $`\log L`$ convergen a $`\alpha`$. Los cambios de reloj que son lentamente variables (ej., $`\mathcal{l}`$) perturban la ordenada al origen asintóticamente pero no la pendiente.

*Esbozo.* Representación de Karamata estándar y argumentos tauberianos; la consistencia de la pendiente local sigue de la convergencia uniforme de razones.

**Observación 2.12 (Sesgo de ventana finita).**\
Cuando $`\mathcal{l}`$ no es plano en el rango observado, $`\widehat{\alpha}`$ está sesgado por $`O\ (\sup \mid log\mathcal{l \mid})`$ a través de la ventana. Esto motiva la **fijación de entorno** y ventanas estrechas en RTM empírico.

**2.5 Condiciones necesarias y suficientes para escalamiento de ley de potencia**

La siguiente declaración empaqueta la idea de **prueba de colapso** de RTM a nivel algebraico.

**Proposición 2.13 (Equivalencia de ley de potencia y log-afinidad).**\
Para un entorno fijo, lo siguiente es equivalente:

1.  $`T(L) = \kappa L^{\alpha}`$ para algunos $`\kappa > 0,\alpha \in \mathbb{R}`$.

2.  Existen constantes $`\alpha,c`$ tales que $`\log T - \alpha\ \log L \equiv c`$ para todo $`L`$.

3.  Para cualquier $`L_{1} \neq L_{2}`$,

``` math
\frac{\log T(L_{2}) - \log T(L_{1})}{\log L_{2} - \log L_{1}} \equiv \alpha\ (\text{independiente del par}).
```

*Demostración.* (1)⇒(2) es Cor. 2.6; (2)⇒(3) restando; (3)⇒(2) fijando $`L_{1}`$ e integrando la constancia de la derivada discreta; luego exponenciar para obtener (1).

**Corolario 2.14 (Prueba de especificación por bin).**\
Dadas observaciones $`\{(L_{i},T_{i})\}`$ en un entorno fijo, si existe una pendiente consistente $`\alpha`$ tal que los residuos $`{\widetilde{y}}_{i}: = \log T_{i} - \alpha\ \log L_{i}`$ son **independientes de** $`\log L_{i}`$ (salvo ruido), la especificación de ley de potencia RTM no se rechaza para ese bin. Cualquier tendencia sistemática de $`\widetilde{y}`$ vs. $`\log L`$ falsifica el escalamiento exacto de ley de potencia en ese bin.

**2.6 Contraejemplos y alcance**

**Contraejemplo 2.15 (Mezclas de regímenes).**\
Sea $`T(L) = \kappa_{1}L^{\alpha_{1}}`$ para $`L \leq L^{\star}`$ y $`T(L) = \kappa_{2}L^{\alpha_{2}}`$ para $`L > L^{\star}`$ con $`\alpha_{1} \neq \alpha_{2}`$. Entonces ningún solo $`\alpha`$ ajusta globalmente; cualquier intento mostrará cambios de tendencia residual en $`L^{\star}`$. (Esto es **mezcla de regímenes** y debe dividirse en bins.)

**Contraejemplo 2.16 (Curvatura).**\
Sea $`\log T = g(\log L)`$ con $`g^{''} \neq 0`$ en un intervalo. Entonces la pendiente discreta en Prop. 2.13 depende del par, violando la condición de ley de potencia; el colapso debe fallar en ese intervalo.

**2.7 Estimación de muestra finita con error de medición (preparación para después)**

Mientras que las demostraciones anteriores son declaraciones de funciones exactas, RTM empírico enfrenta $`L,T`$ ruidosos. Escriba $`x = \log L`$, $`y = \log T`$, y observe

``` math
x^{obs} = x + \xi,y^{obs} = y + \zeta,
```

con errores de media cero. Los mínimos cuadrados ordinarios atenúan la pendiente cuando $`\xi \neq 0`$. Mostraremos después (Sección 7) que la **regresión de distancia ortogonal** (mínimos cuadrados totales) y **SIMEX** producen $`\widehat{\alpha}`$ consistente bajo condiciones estándar; los resultados de invariancia (Props. 2.7–2.9) siguen manteniéndose exactamente porque el reloj multiplica $`T`$, no $`L`$.

**2.8 Resumen de la Sección 2**

- El **semigrupo de escala** + regularidad moderada fuerza una **ley de potencia** $`T = \kappa L^{\alpha}`$.

- La **pendiente** $`\alpha`$ es **invariante de reloj** e identifica la estructura; la **ordenada al origen** $`\log\kappa`$ codifica el reloj.

- La **variación regular** extiende la teoría al escalamiento asintótico con relojes lentamente variables.

- El criterio de **colapso** de RTM es la declaración algebraica de que $`\log T - \alpha\ \log L`$ es constante (sin tendencia vs. $`\log L`$) dentro de un bin.

- Las mezclas de regímenes y la curvatura proporcionan **contraejemplos** limpios, justificando el agrupamiento en bins y las pruebas de especificación.

**3. Geometría Escala–Reloj (Colapso como Exactitud)**

Damos una formulación geométrica de RTM que separa la **pendiente** (estructura) del **reloj** (gauge). El objeto clave es la 1-forma

``` math
\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha(x)\text{ }d(\log L),
```

definida sobre un espacio producto donde $`x`$ indexa *entorno* y $`L > 0`$ es *escala*. El criterio de **colapso** de RTM se convierte en la declaración de que $`\omega`$ es **exacta/plana** en un bin. Esta sección hace eso preciso y demuestra las equivalencias.

**3.1 Espacios, coordenadas y bins**

- Sea $`X`$ un espacio de **entorno** suave (o al menos topológico) que recolecta condiciones de fondo (régimen de política, tecnología, microestructura).

- Sea $`S = \mathbb{R}_{> 0}`$ la línea de **escala** con coordenada $`L`$; escriba $`u = \log L \in \mathbb{R}`$.

- Sea $`Y = \mathbb{R}_{> 0}`$ la línea de **tiempo relojado** con coordenada $`T`$; escriba $`v = \log T \in \mathbb{R}`$.

Trabajamos en la variedad $`M = X \times S`$ con coordenadas $`(x,u)`$. Un **bin** es un conjunto abierto conexo por caminos $`E \subset M`$ en el que "el entorno es suficientemente fijo" en el sentido de RTM (sin rupturas de régimen). En $`E`$, asuma un **campo de coherencia** localmente integrable $`\alpha:E \rightarrow \mathbb{R}`$.

**3.2 La 1-forma RTM**

**Definición 3.1 (1-forma RTM).**\
En $`E \subset M`$, defina

``` math
\omega\text{\:\,} = \text{\:\,}dv\text{\:\,} - \text{\:\,}\alpha(x,u)\text{ }du.
```

Aquí $`\alpha`$ puede depender de $`x`$ y (opcionalmente) de $`u`$ si permitimos exponentes lentamente variables; $`\alpha`$ constante es el caso ideal de RTM.

**Transformaciones de reloj (gauge).**\
Un **cambio de reloj** multiplica el tiempo bruto por un factor positivo independiente de $`L`$ dentro del bin:

``` math
v \mapsto v^{\#}\text{\:\,} = \text{\:\,}v + \phi(x),\phi:X \rightarrow \mathbb{R.}
```

Bajo esto,

``` math
\omega \mapsto \omega^{\#}\text{\:\,} = \text{\:\,}d(v + \phi(x)) - \alpha\text{ }du\text{\:\,} = \text{\:\,}\omega + d\phi(x).
```

Así $`\omega`$ está definida **salvo adición de 1-formas exactas retrotaídas desde** $`X`$—una libertad de gauge estándar.

**Proposición 3.2 (La pendiente es invariante de gauge).**\
Los cambios de reloj $`v \mapsto v + \phi(x)`$ no alteran el coeficiente $`\alpha`$ de $`du`$. Por tanto $`\alpha`$ es un objeto invariante de gauge, mientras que $`v`$ y $`\omega`$ se desplazan por formas exactas.

*Demostración.* Inmediato de la regla de transformación.

**3.3 Colapso como exactitud**

El **colapso** de RTM declara que, dentro de un bin, después de eliminar $`\alpha\text{ }u`$ la variación restante de $`v`$ es constante (salvo ruido), es decir, los residuos no tienen tendencia con $`u`$.

**Teorema 3.3 (Colapso ⇔ exactitud).**\
Sea $`E \subset M`$ simplemente conexo. Lo siguiente es equivalente:

1.  (*Carta de ley de potencia*) Existe una función $`\kappa:E \rightarrow \mathbb{R}_{> 0}`$ tal que

``` math
v(x,u)\text{\:\,} = \text{\:\,}\alpha(x)\text{ }u\text{\:\,} + \text{\:\,}\log\kappa(x).
```

(Caso de $`\alpha`$ constante; para $`\alpha(x)`$ variable reemplace $`\alpha(x)\text{ }u`$ por $`\int_{}^{u}{\alpha(x,s)\, ds.}`$)

2.  (*Colapso*) Para algún $`\alpha`$ como arriba, el **residuo** $`\widetilde{v}: = v - \alpha u`$ es independiente de $`u`$ en $`E`$ (es decir, una función solo de $`x`$).

3.  (*Exactitud*) La 1-forma $`\omega = dv - \alpha\text{ }du`$ es **exacta** en $`E`$: $`\omega = d\psi`$ para algún potencial escalar $`\psi(x)`$ (sin dependencia de $`u`$).

*Demostración.* (1) ⇒ (2) es inmediato: $`\widetilde{v} = \log\kappa(x)`$. (2) ⇒ (3): si $`\widetilde{v} = \psi(x)`$, entonces $`d\widetilde{v} = dv - \alpha\text{ }du = d\psi(x)`$. (3) ⇒ (1): la exactitud y la conexión simple implican $`\widetilde{v} = \psi(x) + C`$, por tanto $`v = \alpha u + \log\kappa(x)`$.

**Corolario 3.4 (Prueba de planitud).**\
En $`E`$ simplemente conexo, el colapso se cumple si y solo si $`d\omega = 0`$. En coordenadas locales,

``` math
d\omega\text{\:\,} = \text{\:\,} - \text{ }d\alpha \land du.
```

Así una condición **necesaria y suficiente** para el colapso es que $`\partial\alpha/\partial u = 0`$ y que cualquier dependencia en $`x`$ de $`\alpha`$ no cree holonomía alrededor de lazos con extensión en $`u`$. Para $`\alpha`$ constante, $`d\omega = 0`$ automáticamente.

*Observación.* Si $`\alpha = \alpha(x)`$ solo, $`d\omega = - (\partial\alpha/\partial x)\text{ }dx \land du`$. La planitud entonces requiere que a lo largo de cualquier lazo en $`E`$ con extensión no nula en $`u`$, la variación en $`x`$ se integre a cero—equivalentemente, que el campo sea **independiente del camino** después de fijar el gauge. En la práctica trabajamos en bins pequeños donde $`\alpha`$ es aproximadamente constante, así que $`d\omega \approx 0`$.

**3.4 Holonomía, mezcla de regímenes, y por qué el colapso puede (y debe) fallar**

**Definición 3.5 (Holonomía de la conexión RTM).**\
Dado un lazo cerrado $`\gamma \subset E`$, defina la holonomía

``` math
\mathcal{H(}\gamma)\text{\:\,} = \text{\:\,}\oint_{\gamma}^{}\omega\text{\:\,} = \text{\:\,}\oint_{\gamma}^{}{(dv - \alpha\text{ }du)}.
```

- Si $`\mathcal{H(}\gamma) = 0`$ para todos los lazos (es decir, $`d\omega = 0`$), los residuos son independientes del camino y el colapso puede tener éxito.

- Si $`\mathcal{H(}\gamma) \neq 0`$ para algún lazo, el bin contiene **regímenes incompatibles** o curvatura genuina (comportamiento no potencial): el colapso debe fallar.

**Proposición 3.6 (Las mezclas y la curvatura inducen holonomía).**\
Suponga que $`E`$ contiene subregiones con exponentes diferentes $`\alpha_{1} \neq \alpha_{2}`$ a través de una costura en $`u`$ o $`x`$. Cualquier lazo que rodee la costura produce $`\mathcal{H(}\gamma) \neq 0`$. Por tanto el colapso a través de todo $`E`$ es imposible; el conjunto debe reagruparse en bins.

*Esbozo.* Integre $`\omega`$ por piezas; el salto en $`\alpha`$ contribuye una integral no nula de $`(\alpha_{2} - \alpha_{1})\text{ }du`$.

**3.5 Caso de exponente variable y colapso adiabático**

Empíricamente, $`\alpha`$ puede derivar lentamente con $`u`$ o $`x`$. Entonces el colapso exacto no puede sostenerse globalmente, pero el **colapso adiabático** puede sostenerse en ventanas cortas.

**Proposición 3.7 (Aproximación adiabática).**\
Si $`\alpha(x,u)`$ es $`C^{1}`$ y $`\parallel \partial\alpha/\partial u \parallel \leq \varepsilon`$ en $`E`$, entonces sobre cualquier ventana en $`u`$ de ancho $`h`$,

``` math
\widetilde{v}(x,u)\text{\:\,} = \text{\:\,}v - \alpha(u_{0},x)\text{ }u\text{\:\,} = \text{\:\,}\log\kappa(x)\text{\:\,} + \text{\:\,}O(\varepsilon h),
```

uniformemente para $`u \in \lbrack u_{0} - h/2,u_{0} + h/2\rbrack`$. Consecuentemente, la estadística empírica de colapso $`R^{2}(\widetilde{v} \sim u)`$ es $`O(\varepsilon^{2}h^{2})`$.

*Esbozo.* Expansión de Taylor de primer orden de $`\alpha(u)`$ alrededor de $`u_{0}`$; acotar la tendencia residual.

*Interpretación.* Esto justifica la **doctrina de agrupamiento en bins** de RTM: hacer ventanas suficientemente pequeñas para que la curvatura sea despreciable; el colapso entonces prueba la planitud aproximada.

**3.6 Identificabilidad bajo gauge (vista global)**

**Proposición 3.8 (Clase de equivalencia de gauge).**\
Dos campos de tiempo $`v_{1},v_{2}`$ en $`E`$ definen el mismo $`\alpha`$ si y solo si sus 1-formas RTM difieren por un retrotracto exacto desde $`X`$:

``` math
dv_{2} - \alpha\text{ }du\text{\:\,} = \text{\:\,}dv_{1} - \alpha\text{ }du + d\phi(x).
```

Equivalentemente, $`v_{2} = v_{1} + \phi(x)`$. Así los **móduli** de estructuras RTM en $`E`$ son el cociente

``` math
\mathcal{M(}E)\text{\:\,} \cong \text{\:\,}\{(\alpha,v)\}/\{ v \sim v + \phi(x)\}.
```

La pendiente $`\alpha`$ clasifica la órbita; los relojes viven en la fibra de gauge.

*Consecuencia.* Cualquier procedimiento empírico que estime $`\alpha`$ a partir de pendientes en $`u`$ es automáticamente invariante de gauge; los procedimientos que usan niveles en $`v`$ no lo son.

**3.7 Diagnósticos prácticos en lenguaje geométrico**

- La **estadística de colapso** $`\Delta_{\text{colapso}} = R^{2}(\widetilde{v} \sim u)`$ es un **proxy de curvatura**; valores grandes indican $`d\omega \neq 0`$ o mezcla de regímenes.

- Los **placebos de reloj** (cambiar unidades de tiempo) implementan $`v \mapsto v + \text{const}`$: no deberían cambiar $`\alpha`$ ni $`\Delta_{\text{colapso}}`$.

- **Reagrupar en bins** corresponde a restringir a subdominios donde $`d\omega \approx 0`$.

- El **libro de ordenadas al origen** es un registro de los gauges elegidos $`\phi(x)`$ a través de conjuntos de datos.

**3.8 Resumen**

- RTM se expresa naturalmente vía la 1-forma $`\omega = d\ \log T - \alpha\text{ }d\ \log L`$.

- Los **cambios de reloj** son transformaciones de gauge $`\omega \mapsto \omega + d\phi(x)`$; la **pendiente** $`\alpha`$ es invariante de gauge.

- **Colapso ⇔ exactitud/planitud** de $`\omega`$ en un bin; la holonomía/curvatura explica cuándo debe fallar el colapso.

- El **colapso adiabático** se sostiene en ventanas pequeñas cuando $`\alpha`$ varía lentamente, cuantificando el sesgo de ventana finita.

**4. Variación Regular con Exponente Variable (Análisis)**

La Sección 2 derivó leyes de potencia exactas de la simetría de escala. Empíricamente, RTM a menudo se sostiene **localmente** mientras los exponentes derivan **lentamente** a través de entornos o a través del eje de escala. Esta sección sitúa RTM dentro de la **variación regular** con índices **espacialmente variables**, da teoremas de representación, y cuantifica el sesgo de ventana finita y el vínculo entre **estadísticas de colapso** y **curvatura**.

A lo largo de esta sección, escriba $`x \in X`$ para entorno, $`L > 0`$ para escala, $`u = \log L`$, y $`v(x,u) = \log T(x,L)`$.

**4.1 Variación regular clásica (recapitulación)**

Una $`T:\mathbb{R}_{> 0} \rightarrow \mathbb{R}_{> 0}`$ medible es **regularmente variable** de índice $`\alpha \in \mathbb{R}`$ si

``` math
\underset{L \rightarrow \infty}{\lim}\frac{T(bL)}{T(L)} = b^{\alpha}\forall b > 0.
```

Entonces (Karamata–de Haan) existe una $`\mathcal{l}`$ **lentamente variable** tal que $`T(L) = L^{\alpha}\mathcal{l}(L)`$, con $`\mathcal{l}(bL)\mathcal{/l}(L) \rightarrow 1`$ para cada $`b`$ fijo. En escalas log–log,

``` math
v(u) = \alpha u + \log\mathcal{l}(e^{u}),{\ \ \ \ \ \ \ \partial}_{u}v(u) = \alpha + o(1).
```

Por tanto las pendientes locales convergen a $`\alpha`$ cuando $`u \rightarrow \infty`$.

**Cotas de Potter.** Para todo $`\epsilon > 0`$, existe $`U`$ tal que para $`u \geq U`$,

``` math
\mid \log\mathcal{l}(e^{u + h}) - \log\mathcal{l}(e^{u}) \mid \leq \epsilon \mid h \mid + o(1),\ \ h\text{ acotado,}
```

lo cual controlará el sesgo en ventanas finitas.

**4.2 RTM con exponente variable** $`\mathbf{\alpha(x)}`$

Ahora permitimos que el exponente varíe con el entorno $`x`$ (y después lentamente con $`u`$).

**Definición 4.1 (Variación regular puntual en** $`x`$**).**\
$`T( \cdot ;x)`$ es **regularmente variable en** $`\infty`$ con índice $`\alpha(x)`$ si para cada $`x`$ fijo y $`b > 0`$,

``` math
$$\lim_{L \to \infty} \frac{T(bL; x)}{T(L; x)} = b^{\alpha(x)}.$$
```

Equivalentemente,

``` math
T(L;x) = L^{\alpha(x)}\text{ }\mathcal{l}(L;x),
```

donde $`\mathcal{l}( \cdot ;x)`$ es lentamente variable **uniformemente en conjuntos compactos de** $`x`$ (UCS): para cada compacto $`K \subset X`$ y $`b > 0`$,

``` math
$$
\sup_{x \in K} \left| \frac{\ell(bL; x)}{\ell(L; x)} - 1 \right| \underset{L \to \infty}{\longrightarrow} 0.
$$
```

**Proposición 4.2 (Pendiente local uniforme).**\
Bajo variación lenta UCS,

``` math
\partial_{u}v(x,u) = \alpha(x) + r(x,u),\sup_{x \in K} \mid r(x,u) \mid \rightarrow 0(u \rightarrow \infty)
```

para cada compacto $`K \subset X`$. Así en bins de gran escala, las **pendientes por bin** convergen uniformemente a $`\alpha(x)`$.

*Esbozo.* Tomar logaritmos, diferenciar en $`u`$; la propiedad UCS da pequeñez uniforme del incremento de $`\log\mathcal{l}`$.

**4.3 Deriva en** $`\mathbf{\alpha}`$ **a través de la escala:** $`\mathbf{\alpha(x,u)}`$

Empíricamente, los exponentes pueden **derivar lentamente con** $`u`$ (fenómenos de rango finito, regímenes en evolución). Modele

``` math
v(x,u) = \int_{u_{0}}^{u}{\alpha(x,s)\text{ }ds + \log\kappa(x,u),}
```

con $`\kappa`$ lentamente variable en el sentido de que para $`h`$ acotado,

``` math
\sup_{x \in K} \mid \log\kappa(x,u + h) - \log\kappa(x,u) \mid \leq \epsilon \mid h \mid + o(1),
```

y asuma **adiabaticidad**:

``` math
\sup_{x \in K} \mid \partial_{u}\alpha(x,u) \mid \leq \varepsilon\ \ \ \text{(pequeño)}.
```

**Teorema 4.3 (Representación adiabática y cota de sesgo).**\
Sea $`\widehat{\alpha}(x;u,h)`$ cualquier estimador de **pendiente local simétrica** en la ventana $`\lbrack u - h/2,\text{ }u + h/2\rbrack`$ (ej., ODR/TLS/Theil–Sen). Bajo las condiciones de deriva lenta y variación lenta,

``` math
\widehat{\alpha}(x;u,h)\text{\:\,} = \text{\:\,}\alpha(x,u)\text{\:\,} + \text{\:\,}O\text{ }(\varepsilon h)\text{\:\,} + \text{\:\,}O\text{ }(\epsilon).
```

Por tanto el sesgo de ventana finita es lineal en la **curvatura** $`\partial_{u}\alpha`$ y acotado por la variación lenta de $`\kappa`$.

*Esbozo.* Expandir $`v(x,u + s)`$ en Taylor a primer orden en $`s`$ con resto $`\frac{1}{2}(\partial_{u}\alpha)\text{ }s^{2}`$; las ventanas simétricas cancelan términos impares; las cotas de Potter manejan $`\kappa`$.

**Corolario 4.4 (Estadística de colapso bajo deriva lenta).**\
Sea $`\widetilde{v}(x,u) = v(x,u) - \widehat{\alpha}(x;u,h)\text{ }u`$ dentro de la ventana. Entonces

``` math
R^{2}\text{ }(\widetilde{v} \sim u)\text{\:\,} = \text{\:\,}O\text{ }((\varepsilon h)^{2}) + O(\epsilon^{2}),
```

es decir, la **falla de colapso** escala cuadráticamente con el ancho de ventana y la curvatura de $`\alpha`$.

**4.4 Prueba de especificación: curvatura vs. ley de potencia**

Suponga que la relación verdadera es **no potencial** con $`g`$ dos veces diferenciable:

``` math
v(u) = g(u),g^{''}(u) \equiv \not{}0.
```

Sea $`\widehat{\alpha}(u,h)`$ la pendiente de mínimos cuadrados local en $`\lbrack u - h/2,\text{ }u + h/2\rbrack`$.

**Lema 4.5 (Error de linealización local).**

``` math
$$\sup_{|s| \leq h/2} | g(u+s) - (g(u) + \hat{\alpha}(u,h)s) | \geq c | g''(u) | h^2$$
```

para alguna constante universal $`c > 0`$. Consecuentemente,

``` math
R^{2}(\widetilde{v} \sim u)\text{\:\,} \geq \text{\:\,}c'\text{ } \mid g^{''}(u) \mid^{2}\text{ }h^{2} + o(h^{2}),
```

así que la curvatura persistente fuerza una **estadística de colapso no nula** cuando $`h \rightarrow 0`$ solo linealmente, dando una **prueba de especificación** práctica contra leyes de potencia.

*Esbozo.* Alternancia de Chebyshev / cotas de resto de Taylor; la varianza residual de regresión acotada por abajo por la energía de curvatura.

**4.5 Mezclas multi-régimen e identificabilidad**

Sea $`v(u) = \alpha_{1}u + c_{1}`$ en $`\lbrack u_{-},u^{\star}\rbrack`$ y $`v(u) = \alpha_{2}u + c_{2}`$ en $`\lbrack u^{\star},u_{+}\rbrack`$ con $`\alpha_{1} \neq \alpha_{2}`$.

**Proposición 4.6 (Holonomía inevitable / falla de colapso).**\
Cualquier pendiente de ventana única sobre $`\lbrack u_{-},u_{+}\rbrack`$ exhibe tendencia residual con magnitud $`\Omega( \mid \alpha_{2} - \alpha_{1} \mid \text{ } \mid u_{+} - u_{-} \mid )`$. Así los **re-bins** que respeten las fronteras de régimen son necesarios; de lo contrario la geometría de Sec. 3 produce holonomía no nula.

*Esbozo.* La exactitud lineal por piezas implica quiebre en $`u^{\star}`$; cualquier ajuste afín único deja residuos sistemáticos con cambio de signo.

**4.6 Errores en variables bajo variación regular**

Sea $`x = \log L`$, $`y = v(x) = \alpha(x_{0})x + c + r(x)`$ en un bin fijo alrededor de $`x_{0}`$, con $`\mid r'(x) \mid \leq \varepsilon`$ (curvatura pequeña). Las observaciones satisfacen

``` math
x^{obs} = x + \xi,y^{obs} = y + \zeta,
```

con errores de media cero, $`\xi`$ independiente de $`x`$, y $`\zeta`$ ruido independiente.

**Teorema 4.7 (Consistencia de ODR/SIMEX con deriva lenta).**\
Si $`\mathbb{E}\xi^{2} < \infty`$, $`\mathbb{E}\zeta^{2} < \infty`$, y la curvatura $`\varepsilon \rightarrow 0`$ con el ancho de ventana, entonces:

- La pendiente **ODR/TLS** $`{\widehat{\alpha}}_{ODR} \rightarrow \alpha(x_{0})`$ en probabilidad cuando $`n \rightarrow \infty`$, ventana $`h \rightarrow 0`$ con $`nh \rightarrow \infty`$.

- **SIMEX** es consistente siempre que una estimación precisa de $`Var(\xi)`$ esté disponible; el error de extrapolación SIMEX es $`o(1)`$ bajo el mismo régimen.

*Esbozo.* Asintóticas EIV estándar en modelos lineales locales más control de sesgo del Teorema 4.3.

**4.7 Juntando todo (reglas operacionales)**

1.  **Doctrina de agrupamiento en bins.** Elegir ventanas suficientemente pequeñas para que $`\partial_{u}\alpha`$ sea despreciable: sesgo $`= O(\varepsilon h)`$.

2.  **Colapso como proxy de curvatura.** Usar $`R^{2}(\widetilde{v} \sim u)`$ para detectar curvatura $`g^{''}`$ o mezclas de regímenes; los umbrales escalan como $`h^{2}`$.

3.  **Robustez de reloj.** Los relojes lentamente variables $`\kappa`$ alteran ordenadas al origen, no pendientes; las cotas de Potter controlan su contribución.

4.  **Consciente de EIV.** Usar ODR/SIMEX/Theil–Sen; asegurar $`nh \rightarrow \infty`$ para consistencia mientras $`h \rightarrow 0`$ para control de sesgo.

5.  **Detección de mezcla.** Los quiebres ($`\alpha`$ por piezas) implican holonomía no nula (Sec. 3) y fuerzan reagrupamiento en bins.

**4.8 Resumen**

- RTM encaja naturalmente dentro de la **variación regular**: potencia exacta cuando $`\alpha`$ es constante; **localmente similar a potencia** cuando $`\alpha`$ deriva lentamente.

- El **sesgo de pendiente** de ventana finita es $`O(\partial_{u}\alpha \cdot h)`$; las fallas de colapso escalan como $`O((\partial_{u}\alpha)^{2}h^{2})`$.

- La **curvatura** o las **mezclas** producen residuos de colapso persistentes; esto produce una **prueba de especificación** con principios y justifica el agrupamiento en bins.

- Con error de medición, **ODR/SIMEX** permanecen consistentes para el exponente local bajo regímenes estándar.

**5. Renormalización en Escalas: Puntos Fijos y Estabilidad**

La ley de potencia de RTM emerge de la simetría de escala. Ahora reformulamos esto como un problema de **renormalización** en un espacio de funciones: reescalar el argumento $`L \mapsto bL`$ y re-calibrar el reloj para que el resultado pueda compararse en la escala original. **Las leyes de potencia son puntos fijos** de este operador; bajo contractibilidad moderada, los flujos se aproximan a la variedad de ley de potencia, dando una justificación dinámica para RTM.

**5.1 Espacios de funciones y el operador de renormalización**

Sea $`\mathcal{F}`$ una clase de funciones positivas $`T:\mathbb{R}_{> 0} \rightarrow \mathbb{R}_{> 0}`$ con $`\log T \in C_{\text{loc}}^{1}`$. Fije $`b > 1`$ (una dilatación). Para una **elección de gauge** dada $`f(b) > 0`$ (el factor de reloj), defina:

``` math
(\mathcal{R}_{b}T)(L)\text{\:\,}: = \text{\:\,}\frac{T(bL)}{f(b)}.
```

Gauges típicos:

- **Gauge de pendiente exacta** cuando $`\alpha`$ es conocido: $`f(b) = b^{\alpha}`$.

- **Gauge auto-normalizante**: $`f(b) = T(bL_{0})/T(L_{0})`$ para un $`L_{0} > 0`$ de referencia (así $`(\mathcal{R}_{b}T)(L_{0}) = T(L_{0})`$).

- **Gauge de momento**: elegir $`f(b)`$ para que un funcional elegido $`\Phi\lbrack T\rbrack`$ sea invariante (ej., $`\Phi\lbrack T\rbrack = \int w(L)\log T(L)dL`$).

Trabajaremos principalmente con el **gauge auto-normalizante**; los resultados se trasladan a otros gauges por equivalencia (Observación 5.2).

**Métrica.** En $`I \subset (0,\infty)`$ compacto, use

``` math
d_{I}(T_{1},T_{2})\text{\:\,} = \text{\:\,}\sup_{L \in I} \mid \log T_{1}(L) - \log T_{2}(L) \mid .
```

En $`\mathcal{F}`$, considere la familia proyectiva $`d = \sum_{k = 1}^{\infty}{2^{- k}d_{I_{k}}}`$ para compactos anidados $`I_{k} = \lbrack e^{- k},e^{k}\rbrack`$.

**5.2 Los puntos fijos son leyes de potencia**

**Proposición 5.1 (Puntos fijos).**\
Sea $`f(b) = b^{\alpha}`$ para algún $`\alpha \in \mathbb{R}`$. Entonces $`T`$ es un punto fijo de $`\mathcal{R}_{b}`$ para todo $`b > 1`$,

``` math
\mathcal{R}_{b}T = T\forall b > 1,
```

si y solo si $`T(L) = \kappa L^{\alpha}`$ para algún $`\kappa > 0`$.

*Demostración.* Si $`T(L) = \kappa L^{\alpha}`$, entonces $`T(bL)/b^{\alpha} = \kappa(bL)^{\alpha}/b^{\alpha} = \kappa L^{\alpha} = T(L)`$. Recíprocamente, asuma $`\mathcal{R}_{b}T = T`$ para todo $`b`$. Entonces $`T(bL) = b^{\alpha}T(L)`$; por el Teorema 2.5, $`T(L) = \kappa L^{\alpha}`$.

**Observación 5.2 (Equivalencia de gauge).**\
Con el gauge auto-normalizante $`f(b) = T(bL_{0})/T(L_{0})`$, los puntos fijos satisfacen $`T(bL)/T(bL_{0}) = T(L)/T(L_{0})`$, es decir,

``` math
\frac{T(bL)}{T(L)} = \frac{T(bL_{0})}{T(L_{0})} = b^{\alpha},
```

así que los puntos fijos son de nuevo precisamente leyes de potencia. Así los **puntos fijos son invariantes de gauge salvo reloj**.

**5.3 Linealización y estabilidad cerca de una ley de potencia**

Sea $T^\star(L) = \kappa L^\alpha$ un punto fijo (para gauge $f(b) = b^\alpha$). Escriba las perturbaciones en espacio log:

``` math
\log T(L) = \log T^{\star}(L) + \varepsilon(L),\varepsilon:\mathbb{R}_{> 0} \rightarrow \mathbb{R.}
```

Entonces

``` math
\log(\mathcal{R}_{b}T)(L) = \log T^{\star}(L) + \varepsilon(bL) - \log f(b) + \log\kappa + \alpha\log L - (\log\kappa + \alpha\log L).
```

Por tanto, para el gauge de pendiente exacta ($`f(b) = b^{\alpha}`$):

``` math
\varepsilon\text{\:\,} \mapsto \text{\:\,}\mathcal{L}_{b}\varepsilon\ \ \ \text{con   }(\mathcal{L}_{b}\varepsilon)(L) = \varepsilon(bL).
```

Así la renormalización linealizada actúa por **composición con dilatación**.

**Lema 5.3 (Contracción en clases Hölder/Zygmund).**\
Sea $`\mathcal{C}^{0,\beta}`$ funciones Hölder de $`u = \log L`$ con seminorma $`\lbrack\varepsilon\rbrack_{\beta} = {\sup}_{u \neq v}\frac{\mid \varepsilon(e^{u}) - \varepsilon(e^{v}) \mid}{\mid u - v \mid^{\beta}}`$. Entonces para cualquier $`b > 1`$,

``` math
\lbrack\mathcal{L}_{b}\varepsilon\rbrack_{\beta} = b^{- \beta}\lbrack\varepsilon\rbrack_{\beta}.
```

Si usamos la norma $`\parallel \varepsilon \parallel_{C^{0,\beta}(I)} = {\sup}_{I} \mid \varepsilon \mid + diam(I)^{\beta}\lbrack\varepsilon\rbrack_{\beta}`$ en un intervalo compacto $`I`$ estable bajo $`u \mapsto u + \log b`$, el operador es una **contracción estricta** con factor $`b^{- \beta} < 1`$.

*Demostración.* En coordenadas $`u`$, $`(\mathcal{L}_{b}\varepsilon)(e^{u}) = \varepsilon(e^{u + \log b})`$; las diferencias se contraen por $`b^{- \beta}`$.

**Teorema 5.4 (Estabilidad local de leyes de potencia).**\
Fije un $`I \subset \mathbb{R}`$ compacto (en $`u = \log L`$), y sea el gauge $`f(b) = b^{\alpha}`$. Si $`\varepsilon \in C^{0,\beta}`$ en $`I' = \{ u + \log b^{n}:\text{ }u \in I,\text{ }n = 0,1,2,\ldots\text{ }\}`$ con norma pequeña, entonces los iterados satisfacen

``` math
$$
\| \varepsilon_n \|_{C^{0,\beta}(I)} \leq b^{-n\beta} \| \varepsilon_0 \|_{C^{0,\beta}(I')} \underset{n \to \infty}{\longrightarrow} 0,
$$
```

es decir, $`\mathcal{R}_{b}^{n}T \rightarrow T^{\star}`$ uniformemente en $`I`$ en espacio log. Por tanto **las leyes de potencia son localmente atractivas** en topologías Hölder/Zygmund.

*Interpretación.* Las perturbaciones Hölder pequeñas son **amortiguadas** por reescalamiento repetido (se "desplazan a la derecha" en $`u`$ y se suavizan). Este es el análogo dinámico de la **variación regular**.

**5.4 Relojes lentamente variables y variedades centrales**

Sea $`T(L) = L^{\alpha}\kappa(L)`$ con $`\kappa`$ **lentamente variable**. En espacio log: $`\varepsilon(u) = \log\kappa(e^{u})`$ con $`\varepsilon(u + h) - \varepsilon(u) \rightarrow 0`$ cuando $`u \rightarrow \infty`$.

**Proposición 5.5 (Variedad central de factores lentamente variables).**\
Bajo el gauge auto-normalizante $`f(b) = T(bL_{0})/T(L_{0})`$, la dinámica de renormalización en $`\varepsilon`$ es

``` math
(\mathcal{L}_{b}\varepsilon)(u) = \varepsilon(u + \log b) - \varepsilon(u_{0} + \log b) + \varepsilon(u_{0}),
```

que preserva el "ancla" $`\varepsilon(u_{0})`$ y desplaza las **diferencias** a lo largo de $`u`$. Si $`\varepsilon`$ es lentamente variable, entonces para cualquier $`I`$ compacto,

``` math
$$
\sup_{u \in I} \left| (\mathcal{L}_{b}^{n}\varepsilon)(u) - \varepsilon(u_{0}) \right| \underset{n \to \infty}{\longrightarrow} 0.
$$
```

Así $`L^{\alpha}\kappa(L)`$ fluye hacia la **hoja de ley de potencia** determinada por el gauge elegido; el factor lentamente variable se sienta en una **variedad central** (dirección neutra) que es cocientada por la auto-normalización.

*Esbozo.* Sumas telescópicas de incrementos lentos; la convergencia uniforme en compactos sigue de las cotas de Potter.

**5.5 Exponentes variables y atracción adiabática**

Sea $`T(L) = \exp(\int_{u_{0}}^{\log L}{\alpha(s)\text{ }ds)\text{ }\kappa(L)}`$ con $`\alpha`$ $C^{1}$ y $`\mid \alpha'(u) \mid \leq \varepsilon`$ pequeño en una banda que abarca $`I' = \cup_{n \geq 0}^{}{(I + n\ \log b).}`$

**Teorema 5.6 (Estabilidad adiabática hacia una ley de potencia a la deriva).**\
Bajo el gauge de pendiente exacta $`f(b) = b^{\alpha(u_{0})}`$ o el gauge auto-normalizante, los iterados satisfacen en cualquier $`I`$ compacto:

``` math
\sup_{u \in I} \mid \log(\mathcal{R}_{b}^{n}T)(e^{u}) - (\alpha(u_{0})\text{ }u + C_{n}) \mid \text{\:\,} \leq \text{\:\,}C\text{ }\varepsilon\text{ }n\ \log b\text{\:\,} + \text{\:\,}o(1),
```

donde $`C_{n}`$ es una constante (dependiente del gauge). Para $`I`$ fijo, cuando $`n`$ crece el lado derecho permanece **pequeño** siempre que la deriva acumulada $`\varepsilon\text{ }n\ \log b`$ sea pequeña—este es el **régimen adiabático**. Por tanto en ventanas finitas el flujo **sigue** una ley de potencia local con exponente cercano a $`\alpha(u_{0})`$.

*Esbozo.* Descomponer $`\int_{u_{0}}^{u + n\log b}{\alpha(s)ds = \alpha(u_{0})(u + n\log b - u_{0}) + \int\alpha'(s)(u + n\log b - s)ds}`$. El resto escala con $`\varepsilon n\ \log b`$; la variación lenta de $`\kappa`$ se maneja como en 5.5.

*Interpretación.* Si $`\alpha`$ deriva lentamente, la renormalización aún empuja hacia **comportamiento local de ley de potencia** en cualquier ventana fija—precisamente la configuración empírica de RTM.

**5.6 Alternativas no potenciales: la curvatura genera modos inestables**

Considere $`v(u) = g(u)`$ con curvatura $`g^{''} \equiv \not{}0`$. En coordenadas de perturbación relativas a $`\alpha`$, sea $`\varepsilon(u) = g(u) - (\alpha u + c)`$. Entonces bajo $`\mathcal{L}_{b}`$,

``` math
\varepsilon(u) \mapsto \varepsilon(u + \log b).
```

Si $`g^{''}`$ es persistente (ej., periódica o polinomial), los residuos desplazados **no decaen** en ventanas fijas; solo "se trasladan". La contracción del Lema 5.3 falla en $`C^{0}`$ a menos que cocientemos por deriva y curvatura. Consecuentemente:

**Proposición 5.7 (La curvatura como modo no decayente).**\
Si $`g^{''}`$ no se anula en infinito (o decae demasiado lentamente), entonces para cualquier gauge, existe una ventana compacta $`I`$ y $`\delta > 0`$ tales que

``` math
$$\inf_{n \geq 0} \sup_{u \in I} |\varepsilon_n(u)| \geq \delta,$$
```

es decir, **la renormalización no contrae** a una ley de potencia en esa ventana. Esto se alinea con la **falla de colapso** (Sec. 4.4).

*Conclusión.* La curvatura persistente es una **característica inestable** bajo el flujo RG—precisamente lo que detecta nuestra prueba de colapso.

**5.7 Resumen e implicaciones**

- El operador de renormalización $`\mathcal{R}_{b}`$ formaliza **"hacer zoom hacia afuera en escala y recalibrar el reloj"**.

- **Las leyes de potencia son puntos fijos**, independientes del gauge (salvo una constante multiplicativa).

- En topologías Hölder/Zygmund, $`\mathcal{R}_{b}`$ es una **contracción**, dando **atracción local** a la variedad de ley de potencia.

- Los **relojes lentamente variables** yacen en una **variedad central** y son neutralizados por la auto-normalización; los flujos convergen a una ley de potencia representativa.

- Los **exponentes lentamente a la deriva** producen **atracción adiabática**: en cualquier ventana fija, los iterados siguen una ley de potencia local con error pequeño y controlado.

- La **curvatura** en $`g = \log T`$ es un **modo no decayente**; bajo RG persiste como traslación, exactamente reflejando la **falla de colapso**.

**6. Difusiones Dependientes de Escala y Formas de Dirichlet**

Esta sección muestra cómo un exponente RTM actúa como un **campo de reloj local** en dinámica estocástica/EDP. Construimos difusiones en espacios métricos cuyo *tiempo efectivo* se estira con la escala, derivamos leyes autosimilares cuando $`\alpha`$ es constante, y demostramos aproximaciones **adiabáticas** cuando $`\alpha`$ varía lentamente. Esto vincula RTM a sub/super-difusión *sin* comprometerse a priori con operadores fraccionales.

**6.1 Configuración métrica–medida y conductividad RTM**

Sea $`(M,d,\mu)`$ un espacio de medida métrico completo y separable con una forma de Dirichlet regular, fuertemente local $`\mathcal{(E,D})`$ en $`L^{2}(\mu)`$ y carré-du-champ $`\Gamma`$. Para intuición, $`M = \mathbb{R}^{m}`$ con $`\Gamma(u) = \mid \nabla u \mid^{2}`$.

Sea $`L:M \rightarrow (0,\infty)`$ un **proxy de escala** (ej., radio de vecindad local, grado, o densidad gruesa) y $`\alpha:M \rightarrow \mathbb{R}`$ un **campo de coherencia**. Defina una **conductividad RTM**

``` math
\mathsf{D}(x)\text{\:\,} = \text{\:\,}L(x)^{- \alpha(x)}\ \ \ \ \ \ \ (\text{relojes más lentos a mayor escala si }\alpha > 0).
```

**Definición 6.1 (Forma de Dirichlet RTM).**

``` math
\mathcal{E}_{\alpha}(u,v)\text{\:\,} = \text{\:\,}\int_{M}^{}{\mathsf{D}(x)\text{ }\Gamma(u,v)(x)\text{ }d\mu(x),\mathcal{D}_{\alpha} = \mathcal{D}.}
```

Esto es cerrado, simétrico, y genera un semigrupo de Markov conservativo $`(P_{t}^{\alpha})_{t \geq 0}`$ con generador

``` math
\mathcal{L}_{\alpha}u\text{\:\,} = \text{\:\,}\nabla \cdot (\mathsf{D}\text{ }\nabla u)(\text{en el caso }\mathbb{R}^{m}).
```

**6.2 Exponente constante: escalamiento autosimilar**

Asuma $`\alpha(x) \equiv \alpha`$ y $`L(x) = \lambda\text{ } \mid x \mid_{R}`$ para alguna cuasi-norma homogénea $`\mid \cdot \mid_{R}`$ de grado 1 bajo un grupo de dilatación $`x \mapsto b\text{ }x`$.

**Teorema 6.2 (Similaridad RTM).**\
Sea $`u(t,x)`$ solución de $`\partial_{t}u = \mathcal{L}_{\alpha}u`$ con datos iniciales integrables. Entonces para cualquier $`b > 0`$,

``` math
u(t,x)\text{\:\,} = \text{\:\,}b^{m}\text{ }u\text{ }(b^{m + \alpha}\text{ }t,\text{\:\,}b\text{ }x)
```

en el sentido de distribuciones. En particular, el núcleo de calor $`p^{\alpha}(t,x,y)`$ obedece

``` math
p^{\alpha}(t,x,y)\text{\:\,} = \text{\:\,}t^{- \frac{m}{m + \alpha}}\text{ }\Phi\text{ }(\frac{d(x,y)}{t^{1/(m + \alpha)}})
```

para algún perfil $`\Phi`$ (colas tipo gaussiano cuando $`M = \mathbb{R}^{m}`$). Así el **radio de difusión** escala como

``` math
r(t)\text{\:\,} \asymp \text{\:\,}t^{1/(m + \alpha)}\ \ \  \Longleftrightarrow \ \ \ t\text{\:\,} \asymp \text{\:\,}r^{\text{ }m + \alpha}.
```

*Interpretación.* El **exponente dinámico efectivo** es $`z = m + \alpha`$: el tiempo crece con la escala como $`T \sim L^{\text{ }z}`$. Cuando $`m`$ es fijo, variar $`\alpha`$ cambia el **gradiente de reloj** a través de la escala.

*Esbozo.* Invariancia de $`\mathcal{E}_{\alpha}`$ bajo $`x \mapsto bx`$, $`t \mapsto b^{m + \alpha}t`$, y conservación de masa dan la ley de escalamiento.

**6.3 Exponente lentamente variable: relojes adiabáticos**

Sea $`\alpha \in C^{1}(M)`$ y $`L \in C^{1}(M)`$. Considere la EDP inhomogénea

``` math
\partial_{t}u\text{\:\,} = \text{\:\,}\nabla \cdot (L(x)^{- \alpha(x)}\nabla u).
```

**Supuesto (deriva adiabática).** Existe $`\varepsilon \ll 1`$ y una cobertura de $`M`$ por parches $`U_{k}`$ de diámetro $`h`$ tales que

``` math
$$\sup_{x \in U_k} \| \nabla \alpha(x) \| \leq \varepsilon, \sup_{x \in U_k} \| \nabla \log L(x) \| \leq \varepsilon.$$
```

**Teorema 6.3 (Autosimilaridad local, error adiabático).**\
Fije un parche $`U`$ y un punto de referencia $`x_{0} \in U`$. Sea $`\alpha_{0} = \alpha(x_{0})`$, $`L_{0} = L(x_{0})`$. Para tiempos $`t`$ tales que el radio de difusión $`r(t) \ll h`$,

``` math
u(t,x)\text{\:\,} = \text{\:\,}(P_{t}^{\alpha_{0}}\text{ }u_{0})(x)\text{\:\,} + \text{\:\,}\mathcal{O}(\varepsilon\text{ }t\text{ }r(t)^{- 2})\ \ \ \ \ \ \ \text{uniformemente para }x \in U,
```

con $`r(t) \asymp t^{1/(m + \alpha_{0})}`$. Equivalentemente, en **ventanas de observación finitas**, la solución es aproximada por un modelo de $`\alpha_{0}`$ constante con un **error adiabático** lineal en la curvatura del campo de reloj.

*Esbozo.* Expansión de Duhamel alrededor del operador de coeficientes congelados en $`x_{0}`$; las cotas de conmutador producen el error establecido usando estimaciones gaussianas locales.

**Corolario 6.4 (Consistencia adiabática ECI).**\
Las estimaciones locales de la **pendiente tiempo–escala** a partir de características de la solución (ej., radio de bola de calor vs. tiempo, tiempos de primer paso) convergen a $`\alpha(x_{0})`$ con sesgo $`O(\varepsilon)`$ cuando la ventana de observación se reduce, coincidiendo con las cotas de sesgo estadístico de la Sección 4.

**6.4 Tiempos de primer paso y de salida**

Sea $`\tau_{R} = \inf\{ t > 0:\text{ }X_{t} \notin B(x_{0},R)\}`$ para la difusión generada por $`\mathcal{L}_{\alpha}`$.

**Proposición 6.5 (Escalamiento de tiempo de salida RTM).**\
Si $`\alpha \equiv \alpha_{0}`$ y $`L(x) \propto \mid x - x_{0} \mid`$ cerca de $`x_{0}`$, entonces

``` math
\mathbb{E}_{x_{0}}\text{ }\tau_{R}\text{\:\,} \asymp \text{\:\,}R^{\text{ }m + \alpha_{0}}.
```
Bajo deriva adiabática, para $`R \ll h`$,

``` math
\mathbb{E}_{x_{0}}\text{ }\tau_{R}\text{\:\,} = \text{\:\,}R^{\text{ }m + \alpha_{0}}\text{\:\,}(1 + O(\varepsilon R)).
```

Por tanto la familia de $`\alpha`$ constante de RTM **realiza** exponentes de sub/super-difusión vía modulación de reloj local en lugar de saltos no locales.

**6.5 Punto de vista espectral**

Sea $`\{ - \mathcal{L}_{\alpha}\varphi_{k} = \lambda_{k}\varphi_{k}\}`$ la resolución espectral (en un dominio acotado con frontera de Dirichlet).

**Teorema 6.7 (Ley tipo Weyl con reloj RTM).**\
Si $`L,\alpha`$ son suaves y acotados por arriba/abajo en $`\Omega \subset \mathbb{R}^{m}`$,

``` math
N(\lambda)\text{\:\,}: = \text{\:\,}\#\{ k:\lambda_{k} \leq \lambda\}\text{\:\,} \sim \text{\:\,}C_{m}\int_{\Omega}^{}{(\lambda\text{ }L(x)^{\alpha(x)})^{m/2}\text{ }dx(\lambda \rightarrow \infty).}
```

Consecuentemente, los modos propios de alta frecuencia "sienten" el **reloj local** como un multiplicador de densidad $`L^{\alpha}`$.

*Esbozo.* Ley de Weyl local vía medida semiclásica con símbolo principal de coeficiente variable $`\mid \xi \mid^{2}L(x)^{- \alpha(x)}`$.

**6.6 Representación estocástica y cambio de tiempo**

Sea $`B_{t}`$ movimiento browniano en $`M`$ (para $`M = \mathbb{R}^{m}`$). Defina un **funcional aditivo**

``` math
A_{t}\text{\:\,} = \text{\:\,}\int_{0}^{t}{\mathsf{D}(B_{s})\text{ }ds =}\int_{0}^{t}{L(B_{s})^{- \alpha(B_{s})}\text{ }ds,}
```

y su inverso continuo por la derecha $`T(t) = \inf\{ s:A_{s} > t\}`$.

**Proposición 6.8 (Representación de cambio de reloj).**\
La difusión $`X_{t} = B_{T(t)}`$ tiene generador $`\mathcal{L}_{\alpha}`$. Así las difusiones RTM son **movimientos brownianos con cambio de tiempo** con un *reloj dependiente del estado*.

*Consecuencias.* Muchas propiedades (martingalas, cotas de Harnack) se levantan del movimiento browniano a través del cambio de tiempo, clarificando cuándo RTM hereda regularidad clásica.

**6.7 Resumen**

- RTM entra en la teoría de difusión como un **reloj dependiente del espacio** $`L^{- \alpha}`$ que multiplica la conductividad.

- **$`\alpha`$ constante** produce **similaridad exacta** con exponente dinámico $`z = m + \alpha`$ y tiempos de salida $`T \sim R^{\text{ }z}`$.

- **$`\alpha`$ lentamente variable** admite aproximaciones **adiabáticas**; las estimaciones locales de la pendiente tiempo–escala son consistentes con sesgo controlado.

- RTM proporciona un camino alternativo a la **difusión anómala** y una interpretación **espectral** limpia; las difusiones RTM son **movimientos brownianos con cambio de tiempo**.

**7. Identificabilidad y Consistencia Estadística**

Esta sección formaliza **qué es identificable** a partir de datos finitos y ruidosos y da resultados de **consistencia** para estimadores de pendiente comunes usados en RTM: **regresión de distancia ortogonal (ODR/TLS)**, **SIMEX**, y **Theil–Sen**. También enmarcamos la **estadística de colapso** como una prueba de especificación contra alternativas no potenciales con error de medición.

Configuración: en un bin fijo (entorno), el modelo ideal es

``` math
y = \log T = \alpha x + c + r(x),\ \ x = \log L,
```

donde $`r \equiv 0`$ (RTM exacto) o $`\mid r'(x) \mid \leq \varepsilon`$ (deriva lenta / curvatura). Las observaciones son ruidosas:

``` math
x^{obs} = x + \xi,{\ \ \ \ \ \ \ y}^{obs} = y + \zeta,
```

con $`\mathbb{E}\lbrack\xi\rbrack = \mathbb{E}\lbrack\zeta\rbrack = 0`$, varianzas finitas, e independencia/regularidad moderada dada abajo.

**7.1 Qué es (y qué no es) identificable**

**Proposición 7.1 (La pendiente es invariante de reloj; la ordenada al origen no).**\
Si el reloj reescala como $`y^{\#} = y + \phi`$ con $`\phi`$ constante en $`x`$ dentro del bin, entonces cualquier estimador de pendiente basado en contrastes en $`x`$ (ODR, SIMEX, Theil–Sen) no cambia, mientras que la ordenada al origen se desplaza por $`\phi`$.\
*Implicación.* Solo $`\alpha`$ es un objetivo intrínseco; las ordenadas al origen son artefactos de gauge (reloj).

**Proposición 7.2 (Identificabilidad salvo curvatura).**\
Si $`r \equiv 0`$, $`\alpha`$ está puntualmente identificado de la distribución conjunta de $`(x^{obs},y^{obs})`$ dada la estructura de error de medición. Si $`\mid r' \mid \leq \varepsilon`$, entonces el objetivo identificado es la **pendiente local** $`\alpha(u_{0})`$ salvo sesgo $`O(\varepsilon h)`$ para ancho de ventana $`h`$ (Sección 4).

**7.2 Regresión de Distancia Ortogonal (Mínimos Cuadrados Totales)**

Asuma muestra i.i.d. $`\{(x_{i}^{obs},y_{i}^{obs})\}_{i = 1}^{n}`$, con\
(i) $`x_{i}`$ soportada en un intervalo compacto $`\lbrack a,b\rbrack`$, densidad acotada por abajo desde 0;\
(ii) $`\xi_{i},\zeta_{i}`$ independientes de $`x_{i}`$ y entre sí, media 0, segundos momentos finitos;\
(iii) $`r \equiv 0`$ (o $`\mid r' \mid \leq \varepsilon`$ pequeño en la ventana).

**Teorema 7.3 (Consistencia de ODR/TLS).**\
Bajo (i)–(ii) con $`r \equiv 0`$, la pendiente ODR $`{\widehat{\alpha}}_{ODR}`$ es **consistente** para $`\alpha`$, y $`\sqrt{n}({\widehat{\alpha}}_{ODR} - \alpha)`$ es asintóticamente normal con varianza determinada por los segundos momentos de $`(x,\xi,\zeta)`$. Si además $`\mid r' \mid \leq \varepsilon`$ en una ventana de ancho $`h`$, entonces

``` math
{\widehat{\alpha}}_{ODR}\text{\:\,} = \text{\:\,}\alpha(u_{0})\text{\:\,} + \text{\:\,}O_{p}(\varepsilon h)\text{\:\,} + \text{\:\,}O_{p}(n^{- 1/2}).
```

*Esbozo.* Asintóticas TLS clásicas (vector propio de matriz de covarianza centrada). La curvatura contribuye un sesgo determinístico de orden $`\varepsilon h`$.

**Observación.** OLS está **atenuado** cuando $`\xi \neq 0`$; ODR es el remedio EIV correcto cuando la razón de error no es extrema.

**7.3 SIMEX (Simulación–Extrapolación)**

Asuma que conocemos o podemos estimar $`\sigma_{\xi}^{2} = Var(\xi)`$. Defina pseudo-muestras

``` math
x_{i}^{(\lambda)} = x_{i}^{obs} + \sqrt{\lambda}\text{ }{\widetilde{\xi}}_{i},\lambda \in \Lambda \subset \lbrack 0,\Lambda_{\max}\rbrack,
```
con $`{\widetilde{\xi}}_{i} \sim N(0,\sigma_{\xi}^{2})`$ frescas; ajuste pendientes ingenuas $`\widehat{\alpha}(\lambda)`$ (ej., OLS u ODR) y extrapole un polinomio de bajo orden a $`\lambda = - 1`$ para obtener $`{\widehat{\alpha}}_{SIMEX}`$.

**Teorema 7.4 (Consistencia de SIMEX).**\
Si $\sigma_{\xi}^2$ es estimado consistentemente y $r \equiv 0$, entonces $\hat{\alpha}_{SIMEX} \xrightarrow{p} \alpha$. Con $|r'| \leq \varepsilon$, en una ventana $h$,

``` math
$$\hat{\alpha}_{SIMEX} = \alpha(u_0) + O_p(\varepsilon h) + o_p(1).$$
```

*Esbozo.* Teoría SIMEX estándar: el sesgo por error de medición es una función suave de $`\lambda`$; extrapolar a $`- 1`$ lo elimina.

**Nota práctica.** Cuando $`\sigma_{\xi}^{2}`$ está mal especificado, SIMEX puede sobre/sub-corregir; usar como **cota de sensibilidad** junto con ODR.

**7.4 Theil–Sen (Pendiente mediana robusta)**

Defina la mediana de pendientes por pares en $`(x^{obs},y^{obs})`$. Bajo ruido simétrico y sin curvatura, Theil–Sen es $`\sqrt{n}`$**-consistente** y robusto a valores atípicos.

**Proposición 7.5 (Envolvente de robustez).**\
Si una fracción $`\pi < 0.29`$ de observaciones son valores atípicos arbitrarios, la pendiente de Theil–Sen aún converge a $`\alpha`$ (punto de ruptura ~29%). Con curvatura pequeña $`\mid r' \mid \leq \varepsilon`$, el sesgo es $`O(\varepsilon h)`$.

*Uso.* Reportar Theil–Sen como **verificación robusta**; fusionar con ODR vía meta-análisis para proteger contra colas pesadas.

**7.5 Inferencia: incertidumbre y práctica de muestra pequeña**

- **Cluster/bootstrap** por entidad o trayectoria para capturar dependencia serial en $`x`$ y $`T`$.

- **Bootstrap de residuo ortogonal** es apropiado para ODR; **bootstrap de pares** para Theil–Sen.

- Reportar **ICs de percentil** y **diagnósticos de influencia** (puntos de apalancamiento en $`x`$).

- Mantener un **libro de pendiente–ordenada**: pendientes con ICs, ordenadas al origen (gauge), y cambios de reloj/unidad conocidos.

**7.6 Estadística de colapso con error de medición**

Sea $`{\widetilde{y}}_{i} = y_{i}^{obs} - \widehat{\alpha}\text{ }x_{i}^{obs}`$ dentro de un bin, y regrese $`\widetilde{y}`$ sobre $`x^{obs}`$. Defina

``` math
\Delta_{\text{colapso}}\text{\:\,}: = \text{\:\,}R^{2}(\widetilde{y} \sim x^{obs}).
```

**Teorema 7.6 (Prueba de especificación bajo EIV).**\
Asuma $r \equiv 0$ (ley de potencia verdadera), $\hat{\alpha}$ es consistente, y $\xi, \zeta$ son de media cero con varianzas finitas. Entonces $\Delta_{\text{colapso}} \xrightarrow{p} 0$.

Si $`v = g(x)`$ con $`g^{''} \neq 0`$ en el bin y suavidad moderada, entonces para cualquier estimador de pendiente consistente,

``` math
\underset{n \rightarrow \infty}{lim\, inf}\Delta_{\text{colapso}}\text{\:\,} \geq \text{\:\,}c\text{ }\mathbb{E}\lbrack g^{''}(X)^{2}\rbrack\text{ }h^{2}\text{(hasta términos de varianza de error)},
```

así que la estadística permanece **acotada por abajo desde 0** en el límite mientras la curvatura persista sobre el ancho de ventana $`h`$.

*Esbozo.* Bajo la hipótesis nula, los residuos son independientes en media de $`x`$; con curvatura, la regresión captura un componente lineal no nulo proporcional a las segundas derivadas (Sección 4.4), con EIV inflando la varianza pero sin borrar la deriva.

**Práctica.** Pre-registrar un umbral de colapso (ej., $`< 0.05`$); acompañar con gráficos de **residuo vs.** $`x`$ (suavizado no paramétrico) y una verificación de **placebo de reloj**.

**7.7 Selección de ventana y puntos de cambio**

- **Compromiso sesgo–varianza:** Elegir ancho de ventana $`h`$ tal que $`nh \rightarrow \infty`$ (varianza ↓) mientras $`h \rightarrow 0`$ (sesgo $`O(\varepsilon h)`$↓).

- **Puntos de cambio:** Usar PELT/Bai–Perron en pares $`(x^{obs},y^{obs})`$ o en residuos preliminares para evitar mezclar regímenes (los quiebres violan el colapso).

- **Puertas de cobertura:** Rechazar bins con muy pocos $`x`$-spans efectivos (apalancamiento delgado → pendiente inestable).

**7.8 Fusión multi-proxy con incertidumbre**

Dadas estimaciones a nivel de familia $`({\widehat{\alpha}}_{f},{\widehat{\sigma}}_{f}^{2})`$ que pasaron el colapso, aplicar fusión de **efectos aleatorios**:

``` math
{\widehat{\alpha}}_{RE} = \frac{\sum_{f}^{}{w_{f}{\widehat{\alpha}}_{f}}}{\sum_{f}^{}w_{f}},w_{f} = \frac{1}{{\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2}},{\widehat{\tau}}^{2} = \max\left\{ \frac{Q - (F - 1)}{\sum w_{f} - \sum w_{f}^{2}/\sum w_{f}},0 \right\}.
```

Reportar $`Q`$, $`{\widehat{\tau}}^{2}`$, e influencia **leave-one-family-out**. Alto $`{\widehat{\tau}}^{2}`$ ⇒ publicar $`{\widehat{\alpha}}_{f}`$ a nivel de familia en lugar de un solo número.

**7.9 Banderas rojas de muestra finita (diagnósticos prácticos)**

- **Atenuación progresiva:** pendiente OLS ≪ pendiente ODR.

- **Escasez de apalancamiento:** La mayor parte del apalancamiento de puntos $`x`$ extremos; ejecutar **jackknife** eliminándolos.

- **Alto** $`\Delta_{\text{colapso}}`$**:** tendencia residual vs. $`x`$ → probable curvatura o mezcla de régimen.

- **Falla de reloj:** Cambio de unidad/reloj altera la pendiente → reagrupar en bins; la pendiente debe ser **invariante de reloj**.

**7.10 Resumen**

- En un bin, $`\alpha`$ es el único estimando invariante de gauge.

- **ODR/TLS** y **SIMEX** son consistentes para $`\alpha`$ bajo supuestos EIV estándar; **Theil–Sen** es una verificación robusta.

- El sesgo de ventana finita por deriva/curvatura es $`O(\varepsilon h)`$; manejar con agrupamiento en bins y puntos de cambio.

- La **estadística de colapso** es una prueba de especificación: tiende a 0 bajo el modelo RTM y permanece positiva con curvatura no potencial—incluso con error de medición.

- Publicar **incertidumbre, diagnósticos de colapso, y heterogeneidad**; cuando la heterogeneidad de fusión es alta, preferir pendientes a nivel de familia sobre un solo índice.

**8. Un Empaquetado Teórico-Categorial de RTM**

Esta sección formaliza RTM como una **teoría de gauge en un fibrado escala–reloj**. El objetivo no es abstracción por sí misma, sino un lenguaje limpio para invariantes (pendiente), gauges (relojes), y funtorialidad bajo cambios de variables, engrosamiento, y construcciones de producto.

**8.1 La categoría RTM**

Un objeto de **RTM** es una terna $`\mathsf{A} = (X,L,v)`$ donde:

- $`X`$ es un espacio topológico (segundo contable) de **entornos**;

- $`L:X \rightarrow \mathbb{R}_{> 0}`$ es un mapa de **escala** continuo (o un factor trivial con coordenada $`u = \log L`$);

- $`v:X \rightarrow \mathbb{R}`$ es un **potencial de reloj** continuo $`v = \log T`$.

Asociada a $`\mathsf{A}`$ está la **1-forma RTM**

``` math
\omega_{\mathsf{A}}\text{\:\,} = \text{\:\,}dv - \alpha\text{ }d(\log L),
```

para algún $`\alpha`$ (constante o un campo en $`X`$), definida salvo **gauge**: $`v \sim v + \phi`$ con $`\phi:X \rightarrow \mathbb{R}`$.

Un **morfismo** $`\Phi:\mathsf{A} \rightarrow \mathsf{B}`$ es un par $`(\varphi,\psi)`$ con $`\varphi:X \rightarrow Y`$ continuo y $`\psi:Y \rightarrow \mathbb{R}`$ tal que

``` math
\Phi^{\text{*}}\omega_{\mathsf{B}}\text{\:\,} = \text{\:\,}\omega_{\mathsf{A}} + d(\psi \circ \varphi),
```
es decir, $`\Phi`$ retrotrae la 1-forma del objetivo a la 1-forma de la fuente **salvo gauge**. La composición es $`(\varphi_{2},\psi_{2}) \circ (\varphi_{1},\psi_{1}) = (\varphi_{2} \circ \varphi_{1},\text{\:\,}\psi_{1} + \psi_{2} \circ \varphi_{1})`$.

**Interpretación.** Diferentes relojes corresponden a desplazamientos de gauge verticales $`v \mapsto v + \phi`$. Los morfismos son reparametrizaciones **compatibles con el reloj** de entorno/escala.

**8.2 Grupo de gauge y móduli**

El **grupo de gauge** $`\mathcal{G}_{X} = C^{0}(X,\mathbb{R})`$ actúa sobre objetos por $`v \mapsto v + \phi`$. Dos objetos son **equivalentes de gauge** si están relacionados por esta acción.

**Proposición 8.1 (Pendiente como invariante de móduli).**\
Si $`\omega_{\mathsf{A}}`$ y $`\omega_{\mathsf{B}}`$ son equivalentes de gauge en $`X`$, entonces sus campos $`\alpha`$ coinciden (c.t.p.). Recíprocamente, campos $`\alpha`$ iguales definen la misma clase en el **espacio de móduli**

``` math
\mathfrak{M}(X) = \{\text{objetos en }X\}/\mathcal{G}_{X}.
```
Así $`\lbrack\mathsf{A}\rbrack \in \mathfrak{M}(X)`$ está únicamente determinado por $`\alpha`$ y la clase de cohomología de de Rham $`\lbrack\omega_{\mathsf{A}}\rbrack \in H^{1}(X;\mathbb{R})`$; en bins simplemente conexos, $`\lbrack\omega\rbrack = 0`$ y la clase está completamente determinada por $`\alpha`$.

*Consecuencia.* En un bin (simplemente conexo), la **pendiente** es el único dato intrínseco; los relojes son puro gauge.

**8.3 Colapso = trivialización del fibrado RTM**

Sea $`\pi:X \times \mathbb{R} \rightarrow X`$ el fibrado de línea trivial con coordenada de fibra $`u = \log L`$. Considere la 1-forma de conexión

``` math
\omega = dv - \alpha(x)\text{ }du.
```

**Teorema 8.2 (Colapso ⇔ trivialización plana).**\
En un bin simplemente conexo $`E \subset X \times \mathbb{R}`$, lo siguiente es equivalente:

1.  Existe una **sección global** $`s(x) = (x,u)`$ y un gauge $`\phi`$ tal que, en la trivialización con potencial $`v^{\phi} = v + \phi`$, $`v^{\phi}(x,u) = \alpha(x)u + c(x)`$ (carta RTM).

2.  La conexión $`\omega`$ es **plana** en $`E`$: $`d\omega = 0`$ y su holonomía se anula.

3.  El **colapso** empírico de la Sección 3 se sostiene en $`E`$.

Esto reempaqueta la Sección 3 en lenguaje categorial: el colapso es la existencia de una trivialización que endereza $`v`$ en una función afín de $`u`$.

**8.4 Productos, sumas, y engrosamiento (estructura monoidal)**

Defina un **producto monoidal** $`\mathsf{A} \otimes \mathsf{B}`$ para subsistemas independientes:

``` math
(X_{A},L_{A},v_{A}) \otimes (X_{B},L_{B},v_{B})\text{\:\,} = \text{\:\,}(X_{A} \times X_{B},\text{\:\,}L_{A}L_{B},\text{\:\,}v_{A} + v_{B}),
```

con $`\alpha_{\otimes} = \alpha_{A} + \alpha_{B}`$ si cada uno obedece una ley de potencia.

**Proposición 8.3 (Aditividad bajo composición independiente).**\
Si ambos factores están en forma RTM exacta $`v_{i} = \alpha_{i}u_{i} + c_{i}`$ con $`u = \log(L_{A}L_{B}) = u_{A} + u_{B}`$, entonces

``` math
v_{\otimes}\text{\:\,} = \text{\:\,}(\alpha_{A} + \alpha_{B})\text{ }u\text{\:\,} + \text{\:\,}(c_{A} + c_{B}),
```

así que las pendientes **se suman** bajo composición multiplicativa de escalas. Las transformaciones de gauge se distribuyen.

**Funtor de engrosamiento.**\
Sea $`\mathcal{C}_{b}`$ que mapea $`(X,L,v)`$ a $`(X,\text{ }bL,\text{ }v - \log f_{b})`$. Con una elección de gauge $`f_{b}`$ (Sección 5), $`\mathcal{C}_{b}`$ es un **endofuntor** de **RTM**; los objetos de ley de potencia son sus **puntos fijos**.

**8.5 Transformaciones naturales y elecciones de reloj**

Dos elecciones de gauge $`f_{b}`$ y $`g_{b}`$ para engrosamiento definen endofuntores $`\mathcal{C}_{b}^{(f)}`$ y $`\mathcal{C}_{b}^{(g)}`$. El mapa

``` math
\eta_{b}:\mathcal{C}_{b}^{(f)} \Rightarrow \mathcal{C}_{b}^{(g)},\ \ \eta_{b}(\mathsf{A}) = (\text{id}_{X},\psi_{b}),\text{ }{\ \ \ \ \ \ \ \psi}_{b} = \log f_{b} - \log g_{b}
```

es una **transformación natural**, es decir, un desplazamiento de gauge funtorial que conmuta con morfismos. Esto codifica la declaración "cambiar el gauge de renormalización es solo un cambio de reloj".

**8.6 Curvatura y obstrucciones (cohomología)**

Sea $`\Omega^{1}(E)`$ 1-formas en un bin $`E \subset X \times \mathbb{R}`$. La curvatura es

``` math
\mathcal{F}\text{\:\,} = \text{\:\,}d\omega\text{\:\,} = \text{\:\,} - d\alpha \land du.
```

- Si $`\mathcal{F} \neq 0`$, el colapso está **obstruido**; la mezcla de regímenes o curvatura genuina persiste.

- Si $`\mathcal{F} = 0`$ pero $`H^{1}(E) \neq 0`$, el colapso aún puede fallar globalmente debido a **holonomía** (clase de cohomología no trivial). El colapso local siempre se sostiene.

**Proposición 8.4 (Obstrucción cohomológica).**\
El colapso se sostiene globalmente en $`E`$ si y solo si $`\mathcal{F} = 0`$ y $`\lbrack\omega\rbrack = 0`$ en $`H^{1}(E)`$. De lo contrario uno solo puede colapsar **localmente** (en cartas simplemente conexas), consistente con el agrupamiento práctico en bins.

**8.7 Observables como funtores**

Un **observable** (ej., exponente de tiempo de salida, exponente de radio de difusión) es un funtor $`\mathcal{O}:RTM \rightarrow \mathcal{C}`$ (conjuntos, grupos, números) que satisface:

- **Invariancia de gauge:** $`\mathcal{O}(v + \phi) = \mathcal{O}(v)`$;

- **Aditividad monoidal:** $`\mathcal{O}(\mathsf{A} \otimes \mathsf{B}) = \mathcal{O}(\mathsf{A}) + \mathcal{O}(\mathsf{B})`$ cuando está definido.

**Ejemplo.** El **funtor de pendiente** $`\mathcal{S}`$ mapea $`(X,L,v) \mapsto \alpha`$ (como una función en $`X`$) y es el observable terminal invariante de gauge en bins con $`H^{1} = 0`$.

**8.8 Resumen**

- **RTM** forma una categoría donde los objetos portan una **1-forma de gauge** $`\omega = d\ \log T - \alpha\text{ }d\ \log L`$.

- Los **morfismos** son reparametrizaciones que preservan $`\omega`$ salvo formas exactas; el **grupo de gauge** actúa por desplazamientos de reloj.

- La **pendiente** $`\alpha`$ es el invariante de móduli; el **colapso** es igual a **trivialización plana** (curvatura y holonomía cero).

- Los **productos** y el **engrosamiento** son funtoriales; las leyes de potencia son puntos fijos de endofuntores de engrosamiento.

- La **cohomología** captura obstrucciones globales al colapso; el agrupamiento en bins proporciona cartas simplemente conexas donde el colapso es factible.

**9. Ejemplos, Contraejemplos, y Problemas Abiertos**

Cerramos la exposición matemática con ejemplos trabajados que satisfacen RTM exactamente, contraejemplos controlados que **deben** fallar el colapso, y una lista corta de problemas abiertos sugeridos por el marco escala–reloj.

**9.1 Ejemplos RTM exactos (el colapso se sostiene)**

**Ejemplo 9.1 (Ley de potencia pura)**

Sea $`T(L) = \kappa L^{\alpha}`$ en $`L > 0`$. Entonces $`v = \log T = \alpha\log L + \log\kappa`$, $`\omega = dv - \alpha\text{ }d(\log L) = 0`$.

- **Colapso:** trivial (residuo constante).

- **Renormalización:** punto fijo de $`\mathcal{R}_{b}`$ para todo $`b > 0`$.

**Ejemplo 9.2 (Reloj lentamente variable)**

Sea $`T(L) = L^{\alpha}\mathcal{l}(L)`$ con $`\mathcal{l}`$ lentamente variable (Karamata). En cualquier ventana finita de $`\log{\ L}`$,

``` math
v(u) = \alpha u + \varepsilon(u),\varepsilon(u + h) - \varepsilon(u) \rightarrow 0.
```

- **Colapso:** se sostiene hasta $`O(\sup \mid \varepsilon(u + h) - \varepsilon(u) \mid )`$.

- **RG:** $`\mathcal{R}_{b}`$ auto-normalizado fluye a la hoja de ley de potencia (Sec. 5.5).

**Ejemplo 9.3 (Medios homogéneos por piezas en EDP)**

En $`\mathbb{R}^{m}`$, tome $`\alpha(x) \equiv \alpha_{0}`$ y $`L(x) = c \mid x \mid`$. El núcleo de calor de $`\partial_{t}u = \nabla \cdot (L(x)^{- \alpha_{0}}\nabla u)`$ satisface similaridad con exponente $`z = m + \alpha_{0}`$ (Sec. 6.2).

- **Observable:** tiempo de salida $`\mathbb{E}\tau_{R} \asymp R^{\text{ }z}`$ recupera $`\alpha_{0}`$.

**9.2 Fallas controladas (el colapso debe fallar)**

**Contraejemplo 9.4 (Costura de régimen / quiebre)**

``` math
$$T(L) = \begin{cases} \kappa_1 L^{\alpha_1}, & L \leq L^\star, \\ \kappa_2 L^{\alpha_2}, & L > L^\star, \alpha_1 \neq \alpha_2. \end{cases}$$
```

- **Geometría:** $`\omega`$ es exacta en cada lado, pero los lazos que cruzan $`L^{\star}`$ tienen holonomía no nula $`\oint\omega = (\alpha_{2} - \alpha_{1})\text{ }d(\log L)`$.

- **Empíricos:** los residuos muestran cambio de signo; $`\Delta_{\text{colapso}}`$ acotado por abajo desde 0 a menos que **reagrupemos en bins**.

**Contraejemplo 9.5 (Relación log–log curva)**

Sea $`v(u) = u + \sin u`$ así $`T(L) = L\text{ }e^{\sin(\log L)}`$.

- **Curvatura:** $`g^{''}(u) = - \sin u \neq 0`$ ⇒ estadística de colapso escala como $`c\text{ }h^{2}`$ (Sec. 4.4).

- **RG:** los residuos se trasladan bajo $`\mathcal{R}_{b}`$, nunca contrayéndose en una ventana fija (Prop. 5.7).

**Contraejemplo 9.6 (Reloj dependiendo de la escala)**

Si un factor de "reloj" depende secretamente de $`L`$: $`T^{\#}(L) = c(L)\text{ }T(L)`$ con $`c`$ no constante, entonces

``` math
v^{\#}(u) = \alpha u + \log\kappa + \log c(e^{u}),
```

y $`\omega^{\#} = \omega + d\ \log c(e^{u})`$ adquiere un **componente** $`du`$.

- **Interpretación:** esto **no** es un gauge permisible en RTM (los relojes deben ser $`L`$-independientes dentro del bin). El colapso debería y fallará—señalando correctamente la mala especificación.

**9.3 Construcciones compuestas trabajadas**

**Construcción 9.7 (Sistemas producto → aditividad de pendiente)**

Sea $T_{A}(L_{A}) = \kappa_{A}L_{A}^{\alpha_{A}}$, $T_{B}(L_{B}) = \kappa_{B}L_{B}^{\alpha_{B}}$. Para composición independiente con escala total $L = L_{A} L_{B}$ y tiempo $T = T_{A}T_{B}$:

``` math
$$
T(L) = \kappa_{A}\kappa_{B} L^{\alpha_{A} + \alpha_{B}}, \quad \alpha_{\text{total}} = \alpha_{A} + \alpha_{B}
$$
```

(Sec. 8.4). Esto modela etapas en cascada cuyos tiempos característicos se multiplican.


**Construcción 9.8 (Empalme adiabático)**

Particione el eje de escala en ventanas $\{I_k\}$ donde $\| \partial_u \alpha \| \leq \varepsilon$. En cada ventana, ajuste $\alpha_k$; defina un modelo **adiabático por piezas**

$$v(u) = \sum_k \mathbf{1}_{u \in I_k} (\alpha_k(u - u_k) + c_k),$$

con restricciones de continuidad en las costuras.

* **Error:** $O(\varepsilon |I_k|)$ por parche; el colapso se sostiene localmente, falla globalmente si $\alpha$ deriva.

**9.4 Puentes a otras teorías**

- **Variación regular / Karamata–de Haan.** RTM vive en la **variedad de ley de potencia**; los relojes lentamente variables son la **variedad central** (Sec. 5.5).

- **Grupo de renormalización.** Los puntos fijos de RTM son puntos fijos RG; la curvatura es un modo no decayente.

- **Difusiones con cambio de tiempo.** Las EDPs RTM son movimientos brownianos con **relojes dependientes del estado** (Sec. 6.7).

- **Lenguaje de gauge/conexión.** Colapso $`\Leftrightarrow`$ conexión plana; la holonomía captura mezcla de regímenes (Secs. 3, 8).

**9.5 Problemas abiertos**

1.  **Umbrales de colapso precisos.** Demostrar cotas de muestra finita, no asintóticas, vinculando $`\Delta_{\text{colapso}}`$ a curvatura $`g^{''}`$ bajo EIV, con constantes óptimas.

2.  **Detección de holonomía.** Construir pruebas estadísticas que distingan curvatura de **obstrucciones topológicas** ($`H^{1}`$ no trivial) usando integrales de lazo de 1-formas residuales.

3.  **Variación regular con exponente variable en grafos.** Sea $`L`$ grado o longitud de camino en un grafo aleatorio; establecer ley de grandes números para $`\widehat{\alpha}`$ local.

4.  **Problemas inversos.** De datos de tiempo de salida $`\mathbb{E}_{x}\tau_{R} \asymp R^{\text{ }m + \alpha(x)}`$, reconstruir $`\alpha(x)`$ (unicidad tipo Calderón con coeficientes dependientes de escala).

5.  **Gauges globales en bins no simplemente conexos.** Clasificar cuándo existe un reloj global (clase de cohomología nula de $`\omega`$), y dar algoritmos constructivos para trivializar si es posible.

6.  **Más allá de leyes de potencia.** Caracterizar las clases de curvatura mínimas $`g`$ para las cuales RG se vuelve contractivo después de reparametrizaciones **no lineales** (ej., log–poly o gauges spline).

7.  **Asintóticas bajo error de medición de colas pesadas.** Extender la consistencia de ODR/SIMEX a ruido $`\alpha`$-estable; cuantificar envolventes de robustez.

8.  **Campos acoplados.** Analizar EDPs con retroalimentación $`L = L(u,x)`$ (el proxy de escala depende del estado), produciendo **relojes no lineales** y potenciales bifurcaciones en $`\alpha`$.

**10. Conclusión Matemática**

Proporcionamos una columna vertebral rigurosa para RTM:

- De un **semigrupo de escala** y regularidad moderada, el tiempo característico obedece una **ley de potencia** $`T = \kappa L^{\alpha}`$; la **pendiente** $`\alpha`$ es una cantidad estructural **invariante de reloj** (Sec. 2).

- RTM se expresa más naturalmente vía la **1-forma** $`\omega = d\ \log T - \alpha\text{ }d\ \log L`$; el **colapso** es igual a **exactitud/planitud**, mientras que la mezcla de regímenes y la curvatura aparecen como **holonomía** (Sec. 3).

- La **variación regular** con exponentes (lentamente) **variables** explica las estimaciones de ventana finita y el sesgo; las estadísticas de colapso cuantifican la curvatura (Sec. 4).

- Un operador de **renormalización** en funciones tiene **leyes de potencia como puntos fijos** y es **contractivo** en clases Hölder/Zygmund; los relojes lentamente variables forman una **variedad central** (Sec. 5).

- En dinámica, los exponentes RTM actúan como **campos de reloj locales** para difusiones/EDPs, produciendo exponentes de similaridad $`z = m + \alpha`$ y aproximaciones **adiabáticas** cuando $`\alpha`$ deriva (Sec. 6).

- Estadísticamente, **ODR/SIMEX/Theil–Sen** recuperan consistentemente $`\alpha`$ local bajo EIV, y la **estadística de colapso** es una prueba de especificación contra curvatura—incluso con ruido (Sec. 7).

- Una formulación **categorial** empaqueta invariancia, gauges, y engrosamiento funtorialmente (Sec. 8).

El programa produce un principio compacto: **la estructura vive en la pendiente**, los relojes viven en el gauge. Donde los bins son estables y el colapso se sostiene, RTM da una descripción falsificable y transportable de cómo **el tiempo se estira con la escala**. Donde el colapso falla, RTM proporciona un **diagnóstico**—curvatura o mezcla de regímenes—no un parche. Los problemas abiertos arriba delinean un camino para profundizar la teoría (gauges no simplemente conexos, problemas inversos, grafos, colas pesadas) y conectarla con análisis y probabilidad más amplios.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*