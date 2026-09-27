<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/apollo.png" width="200" alt="Diagrama de Apolo">

# La Gramática del Ritmo
**Unificando Vida, Mente y Materia bajo RTM**
  
Álvaro Quiceno


</div>

**Resumen**

El tiempo no simplemente pasa; **aprende** las formas a través de las cuales se mueve. Un latido del corazón, un pensamiento, un paso a través de una habitación—cada uno mantiene el tempo con el tamaño de la estructura que lo porta. Este libro nombra esa relación **Relatividad Temporal Multiescala (RTM)**: dentro de una ventana coherente, la duración propia $`T`$ de un proceso sigue la longitud característica $`L`$ que lo sostiene, y la relación es una pendiente $`\alpha`$ que dice cómo el tiempo se apoya en la forma.

Mantenemos la poesía, y mantenemos la prueba. La columna vertebral científica es simple y estricta: $`\alpha`$ **se estima únicamente como una pendiente log–log multipunto de** $`\log T`$ **sobre** $`\log L`$ dentro de ventanas que pasan verificaciones de colapso y regularidad. Cuando el mundo elige otras gramáticas—atajos de mundo pequeño, por ejemplo—lo decimos ($`T \sim \log L`$) en lugar de forzar una ley de potencia. Señales como pendientes espectrales $`\beta`$, enganche de fase, coherencia, o medidas de eco/retardo son bienvenidas como compañeras **auxiliares** de la historia—iluminando, nunca sustituyendo, y nunca convertidas en $`\alpha`$ por fórmula universal.

Con esa disciplina, RTM se convierte en una lente para la vida, la memoria, la inteligencia, la conciencia y el diseño: cómo las **bandas-**$`\alpha`$ estabilizan el significado, cómo los gradientes $`\nabla\alpha`$ canalizan energía e información, cómo el fracaso (NO_COLLAPSE, LOG-SCALING, MULTI-REGIME) enseña dónde el relato debe cambiar de escala. El arco filosófico es que la coherencia no es un ornamento sino un pacto entre estructura y tiempo; el arco empírico es que este pacto puede ser **medido**, **falsificado** y **reparado**. Escribimos en dos voces—lírica y técnica—para que el lector pueda sentir el ritmo y verificarlo, en el mismo aliento.

**Capítulo 1 · Vida — Ritmo Que Sostiene**

**Epígrafe.** *Un cuerpo es un metrónomo tallado por la distancia.*

**1.1 Una promesa entre estructura y tiempo (preludio poético)**

La vida no es una línea recta—es un **tempo sostenido**. Un capilar recuerda cuánto tiempo debe permanecer la sangre; un pulmón recuerda cuánto tiempo debe quedarse el aire; una extremidad recuerda cuánto tiempo debe tomar un paso. La duración no es arbitraria. Se apoya en el tamaño de la forma que la porta. Cuando la forma se deshilacha, el tiempo olvida. Cuando la forma regresa, el tiempo vuelve a marcar el tiempo.

**1.2 La afirmación (enunciado técnico)**

Dentro de una ventana coherente—un mecanismo, una geometría efectiva—la **duración propia** $`T`$ de un proceso escala con una **longitud** característica $`L`$ como

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (pendiente multipunto estimada en una ventana válida de colapso).}
```

Estimamos $`\alpha`$ únicamente mediante regresión de $`\log T`$ sobre $`\log L`$ con errores en variables (ODR/TLS o Theil–Sen, SIMEX opcional), y solo cuando los **diagnósticos de colapso** (linealidad, estabilidad, heterocedasticidad acotada) pasan. Cuando el mundo elige otras gramáticas (ej., topologías de atajo), reportamos $`T \sim \log L`$ en lugar de forzar una potencia.

**Política (pendiente primero).** Pendientes espectrales $`\beta`$, enganche de fase, coherencia, medidas de eco/retardo, e índices similares son **auxiliares**: correlatos útiles bajo modelos explícitos, nunca sustitutos de $`\alpha`$, nunca convertidos en $`\alpha`$ por fórmulas universales.

**1.3 Qué significan "tamaño" y "tiempo" en sistemas vivos**

- **Longitud estructural** $`L`$**:** una escala ligada al mecanismo—longitud de trayectoria o radio de vaso, longitud de membrana o cable dendrítico, extensión de fibra/axón, longitud de onda cortical $`\lambda/2`$, longitud corporal del organismo, o un sustituto validado que preserve la geometría.

- **Tiempo propio** $`T`$**:** la duración del proceso coherente con esa estructura—tiempo de circulación, tiempo medio de difusión o reacción, período de oscilación $`T = 1/f`$, ciclo de marcha, tiempo de etapa de desarrollo, tiempo de recambio.

- **Ventanas:** mantener mecanismo y métrica constantes (ley de transporte, carga, topología). Si los mecanismos se mezclan o las métricas cambian, reportar **MULTI-REGIME** o **NO_COLLAPSE** en lugar de promediar.

**1.4 Recuadro Lírico · Sobre la respiración**

La respiración es una bisagra entre distancias. El pecho se abre a la medida de sus huesos; los alvéolos cuentan en silencio, como cuentas de rosario. La sangre no se apresura porque pueda—permanece porque debe, cumpliendo una cita con la longitud de sus corredores. Lo que llamamos "reposo" es el reloj de arena que un cuerpo puede levantar sin derramar.

*(Puente: en los capítulos siguientes, elegir la escala estructural* $`L`$ *fija el intervalo que un proceso puede soportar; la pendiente* $`\alpha`$ *dice cómo* $`T`$ *mantiene el paso con* $`L`$ *dentro de ese intervalo.)*

**1.5 Medición y estimación (cómo mantenemos la promesa)**

**Construcción del conjunto de datos.** Ensamblar observaciones pareadas $`\{(L_{i},T_{i})\}`$ a través de individuos, tejidos, ROIs, o ventanas temporales donde el mecanismo es estable. Pre-registrar:

1.  la definición de $`L`$ y $`T`$,

2.  límites de bins y reglas de puntos de cambio,

3.  exclusiones (patología, artefactos).

**Estimador.** Ajustar la pendiente de $`\log T`$ sobre $`\log L`$ con:

- **ODR/TLS** (por defecto, ambos ejes ruidosos),

- **Theil–Sen** (robusto),

- **SIMEX** (opcional, cuando el error de medición es estimable).

**Diagnósticos de colapso.** Aceptar una ventana si: (i) los residuos no muestran curvatura, (ii) la pendiente leave-one-out permanece dentro de la tolerancia, (iii) la heterocedasticidad está acotada, (iv) no hay pendientes segmentadas ocultas. De lo contrario reportar **NO_COLLAPSE**; si semi-log es lineal, reportar **LOG-SCALING**.

**Reporte.** Proporcionar $`\widehat{\alpha}`$, IC 95%, estimador, $`n`$, especificaciones de bins, y un panel pequeño de residuos/colapso.

**1.6 Lentes mecanísticos (por qué** $`\mathbf{\alpha}`$ **toma los valores que toma)**

- **Dominado por difusión:** $`T \sim L^{2}/D \Rightarrow \alpha \approx 2`$ cuando la difusividad es aproximadamente estable en escala dentro del bin.

- **Advectivo/flujo:** tránsito gobernado por conductos coherentes tiende hacia $`\alpha \approx 1`$.

- **Disparo/balístico:** propagación dominada por velocidades casi constantes también tiende a $`\alpha \approx 1`$, a menos que la **métrica efectiva** cambie (redes de atajo), en cuyo caso $`T \sim \log L`$ (no-potencia).

- **Control anidado:** reclutamiento/poda cambia el $`L`$ activo; las pendientes pueden segmentarse entre bins (reportar **MULTI-REGIME** con puntos de quiebre).

Estos son indicadores; $`\alpha`$ siempre se mide, nunca se asume.

**1.7 Bosquejos de casos (ilustrativos, agnósticos a datos)**

**1.7.1 Temporización circulatoria**

Sea $`L`$ la longitud de trayectoria característica del vaso (geodésica de red); $`T`$ el tiempo de tránsito medio de dilución del indicador. Esperar $`\alpha \approx 1`$ en flujo coherente; la curvatura sugiere mezcla de difusión capilar o deriva de velocidad dependiente de carga.

**1.7.2 Difusión celular**

$`L`$ como trayectoria dendrítica o alcance de morfógeno; $`T`$ como tiempo medio de equilibración. Bins limpios a menudo muestran $`\alpha \approx 2`$; si $`D`$ deriva con $`L`$, las pendientes se segmentan.

**1.7.3 Ritmos neurales (nivel de fuente)**

$`T = 1/f`$ para picos de banda; $`L = \lambda/2`$ desde longitud de onda modelada en fuente. Estimar $`\alpha`$ vía EIV a través de ROIs/ventanas; reportar PLV/coherencia y $`\beta`$ espectral como **auxiliares**, no como $`\alpha`$.

**1.7.4 Comportamiento (marcha, respiración)**

$`L`$ como longitud de extremidad o trayectoria de vía aérea; $`T`$ como período de ciclo. $`\alpha \approx 1`$ es común dentro de bins de postura constante. Las transiciones (caminar→correr; voluntario→pautado) a menudo rompen el colapso.

**1.8 Recuadro Técnico · Estimando** $`\mathbf{\alpha}`$ **en sistemas vivos (protocolo)**

**Entradas:** pares $`(L,T)`$ y una ventana pre-registrada.\
**Pasos:**

1.  Ajustar pendiente EIV (ODR/TLS; robustez Theil–Sen).

2.  Bootstrap de ICs; registrar $`n`$ y extensión en $`L`$.

3.  Ejecutar diagnósticos de colapso (curvatura, estabilidad LOO, verificaciones de heterocedasticidad).

4.  Reportar $`\widehat{\alpha}`$ con IC y un mini-panel de residuos/colapso.\
    **Resultados:** $`\widehat{\alpha}`$ si colapso pasa; **NO_COLLAPSE** o **LOG-SCALING** de lo contrario.

*Plantilla de leyenda (pendiente):* "Escalamiento de $`T`$ con $`L`$ en \[sistema\]. Pendiente EIV $`\widehat{\alpha}`$ (IC 95%) en un bin válido de colapso (n = …). Los residuos no muestran curvatura."

**1.9 Recuadro Técnico · Usando auxiliares sin romper el hechizo**

- **Reportar** $`\beta`$ espectral, PLV/coherencia, potencias de banda, estadísticas de ráfaga/ISI con métodos e ICs.

- **No convertir** auxiliares a $`\alpha`$ (ej., no universal $`\alpha = 1 + \beta/2`$).

- **Usar** auxiliares como covariables para interpretar cambios en $`\widehat{\alpha}`$, o para seleccionar bins; **informan** la pendiente, no la **definen**.

*Plantilla de leyenda (auxiliar):* "Pendiente espectral $`\beta`$ de VFC (auxiliar; no $`\alpha`$). Método: \[..\]; IC: \[..\]. Sin conversión."

**1.10 Modos de fallo y significados**

- **NO_COLLAPSE:** la línea no se sostiene → mecanismos mezclados o métrica incorrecta. Remedio: re-binear, refinar $`L`$/$`T`$, o reportar el negativo.

- **LOG-SCALING:** $`T \sim \log L`$ debido a geometrías de atajo; etiquetar explícitamente; no inferir $`\alpha`$.

- **MULTI-REGIME:** pendientes lineales por tramos en log–log → reportar $`\alpha`$ segmentado con puntos de quiebre y notas de mecanismo.

**1.11 Coda lírica · Lo que la vida protege**

La vida protege **intervalos**. No solo un latido del corazón, sino la longitud **correcta** de un latido del corazón; no solo una respiración, sino la longitud **correcta** de una respiración. La palabra para esa corrección aquí es $`\alpha`$—un nombre modesto para un pacto: qué tan lejos alcanza una estructura y cuánto tiempo se permite a un proceso convertirse en sí mismo.

**1.12 Lo que este capítulo no hace**

- **No** infiere $`\alpha`$ de fórmulas de un solo punto (ej., $`\log(T_{i}/T_{0})/log(L_{i}/L_{0})`$).

- **No** trata $`\beta`$ espectral, PLV/coherencia, o eco/retardo como $`\alpha`$.

- **No** fuerza leyes de potencia cuando la topología implica $`T \sim \log L`$.

**1.13 Conclusión clave**

Sostener la vida es mantener una **banda-**$`\alpha`$ funcional: un intervalo estructural $`L`$ y un intervalo temporal $`T`$ donde la pendiente es real, verificable, y amable con el organismo. El resto del libro mide cómo esa banda se encuentra, se mantiene, se pierde, y se vuelve a encontrar.

**Capítulo 2 · Memoria — Pliegues en el Tiempo**

**Epígrafe.** *Recordar es mantener una puerta abierta entre dos momentos.*

**2.1 Preludio poético · El pliegue**

La memoria es un pliegue que el presente hace para tocarse a sí mismo después. Una dendrita guarda una señal bajo su rama; un mapa en la corteza pliega el mundo para que pueda ser transportado. No almacenamos cosas—almacenamos **duraciones** moldeadas por **distancias**. Cuando el pliegue se sostiene, el futuro llega ya medio recordado.

*(Puente: en RTM, un "pliegue" es el emparejamiento de una escala estructural* $`L`$ *con una duración propia* $`T`$ *que mantiene el paso a través de una pendiente estable* $`\alpha`$ *medida en una ventana válida de colapso.)*

**2.2 La afirmación (enunciado técnico)**

Dentro de un mecanismo consistente (sináptico, a nivel de sistemas, o conductual), los tiempos de memoria $`T`$ escalan con longitudes estructurales $`L`$ como

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (pendiente multipunto en una ventana válida de colapso)}.
```

Estimamos $`\alpha`$ mediante regresión con errores en variables (ODR/TLS o Theil–Sen; SIMEX opcional) y publicamos **NO_COLLAPSE** cuando los diagnósticos fallan. Topologías de recuperación tipo atajo se reportan como $`T \sim \log L`$ (no-potencia), no forzadas en una ley de potencia.

**Política (pendiente primero).** Pendiente espectral $`\beta`$, enganche de fase/coherencia, tasas de ripple/ráfaga, y métricas de retardo son **auxiliares**: pueden co-variar con $`\alpha`$ bajo modelos explícitos pero nunca se convierten en $`\alpha`$ por fórmula universal.

**2.3 Qué significan "tamaño" y "tiempo" para la memoria**

- **Escala estructural** $`L`$**:** longitud de cuello de espina o volumen de cabeza; longitud de cable dendrítico/orden de rama; ancho de minicolumna; radio de ensamble/mapa; longitud de axón/vía; extensión de proyección hipocampo–cortical. Elegir una por conjunto de datos y mantenerla fija dentro de la ventana.

- **Tiempo propio** $`T`$**:** ancho de ventana STDP/plasticidad; retardo de consolidación; vida media de retención; período de ciclo de repetición; decaimiento de memoria de trabajo a criterio; latencia de recuperación a precisión fija.

**Ventanas.** Mantener mecanismo, estado, y tarea constantes. Mezclar (ej., consolidación más ensayo) rompe el colapso—reportar **MULTI-REGIME** o segmentar pendientes con un punto de quiebre registrado.

**2.4 Recuadro Lírico · La biblioteca de distancias**

Una biblioteca no son estantes de objetos sino corredores de **alcance**. Volvemos a un pensamiento caminando por su pasillo de nuevo. Cuanto más largo el corredor, más tiempo debe dejarse encendida la luz. Algunos libros están al alcance de la mano; otros requieren una linterna y un paso paciente.

*(Puente: longitud de corredor* $`L`$*; tiempo de luz encendida* $`T`$*. Donde la relación es estable, la pendiente log–log es* $`\alpha`$*.)*

**2.5 Lentes mecanísticos (por qué** $`\mathbf{\alpha}`$ **toma estos valores)**

- **Memoria de difusión–reacción (biofísica local):** cuando la señalización se propaga a través de espinas y dendritas, $`T \sim L^{2}/D`$ y $`\alpha \approx 2`$ en compartimentos homogéneos.

- **Enrutamiento e integración (bucles y trayectorias):** conducción con integración a través de una ruta de longitud $`L`$ tiende hacia $`\alpha \approx 1`$ si las velocidades y umbrales son estables en escala.

- **Recuperación por atajo (redes asociativas):** geodésicas efectivas colapsan vía hubs/índices; la temporización se comporta $`T \sim \log L`$ (etiquetar **LOG-SCALING**).

- **Interacciones de dos almacenes:** almacenes lábiles/rápidos y estables/lentos a menudo producen pendientes **segmentadas** (punto de quiebre = cambio de almacén).

Estos son guías; $`\alpha`$ se mide de datos en una ventana definida.

**2.6 Construcción de conjuntos de datos para escalamiento de memoria**

**Observaciones pareadas** $`\{(L_{i},T_{i})\}`$**:**

- **Memoria sináptica:** $`L`$=medida de longitud espina/dendrítica; $`T`$=constante de plasticidad o decaimiento bioquímico de la misma clase de compartimento.

- **Consolidación de sistemas:** $`L`$=longitud de vía o radio de mapa; $`T`$=retardo hasta independencia hipocampal bajo regímenes controlados de sueño/vigilia.

- **Memoria conductual:** $`L`$=longitud de fragmento/secuencia con ensayo/retroalimentación fijos; $`T`$=vida media de retención o latencia de recuerdo.

**Pre-registro:** definiciones de escala y reloj; límites de bin; controles de estado; criterios de exclusión (farmacología, patología, efectos de techo/piso).

**2.7 Estimando** $`\mathbf{\alpha}`$ **(métodos)**

- **Estimador:** pendiente EIV de $`\log T`$ sobre $`\log L`$: **ODR/TLS** (por defecto), **Theil–Sen** (robusto), **SIMEX** opcional para corrección de error.

- **Diagnósticos de colapso:** (i) linealidad en log–log (sin curvatura en residuos), (ii) estabilidad leave-one-out de pendiente, (iii) heterocedasticidad acotada, (iv) sin puntos de cambio ocultos.

- **Reporte:** $`\widehat{\alpha}`$, IC 95%, estimador, $`n`$, extensión en $`L`$, especificación de bin, mini-panel de residuos/colapso.

- **Modos de fallo:** **NO_COLLAPSE** (mezcla de mecanismos), **LOG-SCALING** (topología de atajo), **MULTI-REGIME** (pendientes segmentadas con punto de quiebre e interpretación).

**2.8 Bosquejos de casos (ilustrativos)**

**2.8.1 Consolidación espina-a-soma**

Emparejar longitud de trayectoria soma–espina $`L`$ con tiempo de consolidación $`T`$. Esperar $`\alpha \approx 2`$ si difusión–reacción domina; desviaciones señalan transporte activo o cambios de compartimentación.

**2.8.2 Radio de mapa y retardo de sistemas**

Sea $`L`$ el radio de un mapa cortical o la ruta efectiva hipocampo–cortical; $`T`$ el retardo hasta toma de control cortical. $`\alpha \approx 1`$ casi lineal es plausible; pendientes segmentadas pueden marcar dependencias de etapa de sueño.

**2.8.3 Extensión de memoria de trabajo**

Usar tamaño de fragmento $`L`$ bajo interferencia fija; $`T`$=decaimiento a criterio. A menudo **MULTI-REGIME**: fragmentos pequeños se mantienen vía ensayo (una pendiente), fragmentos más largos fallan colapso o cambian mecanismo.

**2.8.4 Búsqueda asociativa con índices**

Si la recuperación usa hubs o índices direccionables, la temporización sigue $`T \sim \log L`$; reportar parámetros de ajuste semi-log y abstenerse de $`\alpha`$.

**2.9 Recuadro Técnico · Midiendo** $`\mathbf{\alpha}`$ **en memoria (protocolo)**

**Entradas:** pares $`(L,T)`$ dentro de una ventana pre-registrada.\
**Pasos:**

1.  Ajustar pendiente **ODR/TLS** en $`(\log L,\log T)`$; agregar **Theil–Sen** como robustez.

2.  Bootstrap ICs 95%; registrar $`n`$, extensión, y estimador.

3.  Ejecutar diagnósticos de colapso (curvatura de residuos, deriva de pendiente LOOCV, límites de heterocedasticidad).

4.  Reportar $`\widehat{\alpha}`$ con un recuadro de residuos/colapso.\
    **Resultados:** $`\widehat{\alpha}`$ (válido de colapso) o **NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**.

*Plantilla de leyenda (pendiente):* "Tiempo de memoria $`T`$ vs. escala estructural $`L`$ en \[sistema\]. Pendiente EIV $`\widehat{\alpha}`$ (IC 95%) en un bin válido de colapso (n = …). Los residuos no muestran curvatura."

**2.10 Recuadro Técnico · Usando auxiliares sin romper el pliegue**

- **Reportar** $`\beta`$ espectral, PLV/coherencia, tasas de ripple/ráfaga con métodos e ICs.

- **No convertir** auxiliares a $`\alpha`$.

- **Usar** auxiliares como covariables para interpretar cambios en $`\widehat{\alpha}`$ o para seleccionar bins; **iluminan** el pliegue, no lo **definen**.

*Plantilla de leyenda auxiliar:* "Pendiente espectral $`\beta`$ durante consolidación. Auxiliar (no $`\alpha`$); sin conversión. Método/IC reportados."

**2.11 Modos de fallo y significados**

- **NO_COLLAPSE:** mezcla de procesos (ej., ensayo más consolidación). Remedio: bins más estrechos, $`L`$ específico al mecanismo.

- **LOG-SCALING:** recuperación por atajo; tratar con ajustes semi-log; no inferir $`\alpha`$.

- **MULTI-REGIME:** punto de cambio entre almacenes; reportar pendientes segmentadas con interpretación.

**2.12 Coda lírica · Lo que realmente guarda el recordar**

Lo que guardamos no es la imagen sino el **intervalo** necesario para encontrarla de nuevo. Guardamos el tiempo que una estructura nos pide—el minuto que un corredor requiere, el aliento que toma una frase. Recordar es honrar ese intercambio: una longitud por una duración, una distancia por una estadía.

**2.13 Lo que este capítulo no hace**

- **No** infiere $`\alpha`$ de razones de un solo punto.

- **No** convierte $`\beta`$ espectral, PLV/coherencia, o métricas de ripple en $`\alpha`$.

- **No** fuerza leyes de potencia donde la recuperación se comporta $`T \sim \log L`$.

**2.14 Conclusión clave**

La memoria es una **disciplina de pliegues**—emparejando extensiones estructurales $`L`$ con duraciones $`T`$ para que una pendiente $`\alpha`$ se sostenga. La poesía es la imagen de un corredor iluminado justo el tiempo suficiente; la ciencia es la pendiente que prueba que la luz estaba ajustada a la longitud del corredor.

**Capítulo 3 · Inteligencia — La Danza Adaptativa**

**Epígrafe.** *Una mente es una puerta que aprende el tamaño de las habitaciones que entra.*

**3.1 Preludio poético · El salto que aterriza**

La inteligencia es el arte de **llegar**—no solo moverse. Es el don de elegir una estructura que sostendrá el tiempo que estás a punto de gastar. El experto no simplemente piensa más rápido; se **coloca en el lugar del tamaño correcto**. Desde allí, las respuestas se sienten cercanas, no porque la distancia desapareció, sino porque distancia y duración hicieron un pacto.

*(Puente: nombraremos ese pacto por una pendiente* $`\alpha`$ *que liga el tiempo de proceso* $`T`$ *al tamaño estructural* $`L`$ *dentro de una ventana coherente.)*

**3.2 La afirmación (enunciado técnico)**

Dentro de un mecanismo estable (arquitectura, política, costo), el comportamiento inteligente opera en ventanas donde

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (pendiente multipunto en una ventana válida de colapso).}
```

Estimamos $`\alpha`$ únicamente mediante regresión con errores en variables (ODR/TLS o Theil–Sen; SIMEX opcional) y publicamos **NO_COLLAPSE** cuando los diagnósticos fallan. Cuando la métrica efectiva es tipo atajo (ej., recuperación indexada), reportamos $`T \sim \log L`$ en lugar de forzar una ley de potencia.

**Política (pendiente primero).** Puntuaciones de información, $`\beta`$ espectral, sincronía/coherencia, confianza, y proxies de recompensa son **auxiliares**—covariables interpretativas, nunca sustitutos de $`\alpha`$, nunca convertidos a $`\alpha`$ por fórmulas universales.

**3.3 Qué significan "tamaño" y "tiempo" para la inteligencia**

- **Escala estructural** $`L`$**:** granularidad de representación (radio de campo receptivo, ventana de contexto), tamaño de problema (ancho de cuadrícula, conteo de cláusulas), radio de búsqueda u horizonte de planificación, tamaño de módulo/ensamble, espacio de trabajo/longitud de trayectoria del actuador.

- **Tiempo propio** $`T`$**:** latencia de decisión, tiempo-a-criterio, tiempo de asentamiento a tolerancia, tiempo de convergencia para una política/estimador, tiempo de reloj hasta rendimiento estable.

**Ventanas.** Mantener arquitectura, optimizador/política, distribución de tareas, y costo constantes. Cambios de estrategia o cambios de currículo a mitad de ventana rompen el colapso—reportar **MULTI-REGIME** en lugar de promediar.

**3.4 Recuadro Lírico · Eligiendo la habitación**

Algunas habitaciones invitan al silencio; otras piden frases más largas. La inteligencia es saber **a qué habitación** pertenece la pregunta. Elige un espacio demasiado pequeño y el pensamiento tartamudea; demasiado grande y hace eco hacia la indecisión. La habitación correcta responde en un solo aliento.

*(Puente: la habitación es* $`L`$*; el aliento es* $`T`$*; la respuesta que encaja es la pendiente* $`\alpha`$ *medida en una ventana válida de colapso.)*

**3.5 Lentes mecanísticos (por qué** $`\mathbf{\alpha}`$ **toma estos valores)**

- **Agregación y enrutamiento:** computación que crece proporcionalmente con la extensión representacional a menudo produce $`\alpha \approx 1`$ cuando los anchos de banda son estables en escala.

- **Combinatoria domada:** ramificación ingenua empuja $`T`$ a supralineal; heurísticas/abstracciones aprendidas pueden comprimir la métrica **efectiva** a grafos de atajo, produciendo $`T \sim \log L`$ (no-potencia).

- **Jerarquías de control:** controladores multinivel seleccionan un $`L`$ que mantiene el tiempo asequible; cambios de régimen aparecen como pendientes segmentadas (**MULTI-REGIME**).

- **Dinámicas de aprendizaje:** ventanas de batch/replay y retardos de objetivo definen $`L`$ operativo en espacio de parámetros/estado; dentro de configuraciones fijas, escalamiento casi de potencia puede aparecer.

Estos son guías; $`\alpha`$ siempre se mide.

**3.6 Construcción de conjuntos de datos para escalamiento de inteligencia**

Construir observaciones pareadas $`\{(L_{i},T_{i})\}`$ **variando paramétricamente una escala estructural** mientras se mantienen los mecanismos constantes.

**Ejemplos**

- **Modelos de secuencia:** $`L`$=longitud de contexto; $`T`$=pasos/tiempo de reloj hasta una puntuación de validación fija con arquitectura/optimizador congelados.

- **Aprendizaje por refuerzo:** $`L`$=horizonte de planificación; $`T`$=episodios-a-criterio bajo exploración y modelado de recompensa fijos.

- **Robótica/control:** $`L`$=longitud de trayectoria o radio de espacio de trabajo; $`T`$=tiempo de asentamiento a tolerancia con ganancias de controlador fijas.

- **Resolución de problemas humanos:** $`L`$=tamaño de problema (ancho de cuadrícula, conteo de cláusulas, longitud de secuencia); $`T`$=latencia mediana de solución con instrucciones fijas.

**Pre-registrar:** definiciones de escala/reloj; límites de bin; reglas de parada/timeout; exclusiones (reinicios de política, saltos de currículo).

**3.7 Estimando** $`\mathbf{\alpha}`$ **(métodos)**

- **Estimador:** regresión EIV de $`\log T`$ sobre $`\log L`$: **ODR/TLS** (por defecto), **Theil–Sen** (robusto), **SIMEX** opcional.

- **Diagnósticos de colapso:** linealidad en log–log; estabilidad leave-one-condition-out; heterocedasticidad acotada; sin puntos de cambio ocultos.

- **Reporte:** $`\widehat{\alpha}`$, IC 95%, estimador, $`n`$, extensión en $`L`$, especificación de bin, mini-panel de residuos/colapso.

- **Modos de fallo:** **NO_COLLAPSE** (mezcla/curvatura), **LOG-SCALING** (métrica de atajo), **MULTI-REGIME** (pendientes segmentadas con puntos de quiebre).

**3.8 Bosquejos de casos**

**3.8.1 Extensión de contexto vs. tiempo de aprendizaje (modelos de secuencia)**

Mantener arquitectura/optimizador fijos; variar tokens de contexto $`L`$. Sea $`T`$=tiempo para alcanzar un umbral de validación. Bins limpios a menudo muestran $`\alpha \approx 1 - 1.5`$; cambios de optimizador/currículo crean pendientes segmentadas.

**3.8.2 Horizonte de planificación vs. episodios (RL)**

$`L`$=horizonte de búsqueda; $`T`$=episodios-a-criterio bajo exploración fija. Heurísticas/abstracciones a veces inducen **LOG-SCALING**; de lo contrario pendientes supralineales aparecen cuando la ramificación domina.

**3.8.3 Radio de espacio de trabajo vs. tiempo de asentamiento (robots)**

$`L`$=radio o longitud de trayectoria; $`T`$=tiempo de asentamiento. $`\alpha`$ casi lineal es común con controladores bien ajustados; límites de torque o re-planificación aparecen como puntos de quiebre.

**3.8.4 Insight humano bajo presión de tiempo**

$`L`$=escala del problema (ej., alcance del silogismo); $`T`$=tiempo-a-primera-solución con precisión fija. Deriva de estrategia → **NO_COLLAPSE**; recuperación de plantilla en $`L`$ grande puede voltear a **LOG-SCALING**.

**3.9 Recuadro Técnico · Diseño experimental para** $`\mathbf{\alpha}`$

**Objetivo:** maximizar potencia para una pendiente verdadera; evitar ajustes falsos por mezcla.

1.  **Extensión y granularidad:** ≥0.6–1.0 décadas en $`L`$; ≥6 niveles distintos de $`L`$.

2.  **Constancia:** congelar optimizador/política/configuraciones dentro del bin; registrar desviaciones.

3.  **Réplicas:** ≥3 corridas por $`L`$; usar EIV y considerar **SIMEX** cuando el error crece con $`L`$.

4.  **Diagnósticos:** pre-registrar pruebas de colapso; graficar residuos; ejecutar verificaciones de puntos de cambio.

5.  **Resultados:** $`\widehat{\alpha}`$ con IC (válido de colapso) o un negativo etiquetado (**NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**).

**3.10 Usando auxiliares sin romper las reglas**

**Auxiliares:** información predictiva, $`\beta`$ espectral, sincronía/coherencia, medidas de gradiente/entropía, confianza.

- Reportar métodos e incertidumbre; usar como covariables para **interpretar** $`\widehat{\alpha}`$ o para elegir bins.

- **No** convertir auxiliares a $`\alpha`$ o promediarlos en "$`\alpha`$ compuesto".

*Plantilla de leyenda auxiliar:* "Información predictiva vs. contexto. Auxiliar (no $`\alpha`$); sin conversión. Parámetros e IC reportados."

**3.11 Modos de fallo específicos de inteligencia**

- **Deriva de estrategia dentro de un bin:** cambios de política o deriva de instrucciones → curvatura; re-binear o imponer política fija.

- **Topes de recursos ocultos:** límites de memoria/cómputo se activan en $`L`$ grande; emergen pendientes segmentadas—reportar puntos de quiebre.

- **Artefactos de atajo:** indexación/recuperación comprime la distancia → **LOG-SCALING**; etiquetar explícitamente.

- **Confusiones de recompensa:** el modelado de recompensa altera $`T`$ sin cambiar $`L`$; tratar como mecanismo diferente o controlarlo.

**3.12 Coda lírica · La danza**

Una buena respuesta comienza antes de ser hablada. La mente entra en la **habitación que encaja**, y el tiempo, aliviado, mantiene el paso. La danza no es velocidad sino **ajuste**—una pendiente sostenida firme mientras la música cambia.

**3.13 Lo que este capítulo no hace**

- **No** infiere $`\alpha`$ de razones de un solo punto o métricas auxiliares.

- **No** mezcla regímenes para forzar una línea.

- **No** impone leyes de potencia donde la topología implica $`T \sim \log L`$.

**3.14 Conclusión clave**

La inteligencia es **mantenimiento de banda adaptativo**: elegir $`L`$ y $`T`$ para que un $`\alpha`$ coherente exista y se sostenga. Medirlo con estimación pendiente-primero en ventanas válidas de colapso; dejar que los auxiliares canten armonía, no lleven la melodía.

**Capítulo 4 · Conciencia — La Ventana Integrativa**

**Epígrafe.** *El mundo no llega todo de una vez, sino como un acorde que aprendes a sostener.*

**4.1 Preludio poético · Cómo se sostiene el acorde**

Ser consciente es **mantener** un acorde sin que se deshilache. Sensaciones, memorias e intenciones llegan con diferentes longitudes, pidiendo diferentes cantidades de tiempo. Cuando el acorde se sostiene, un momento se convierte en una habitación donde muchas voces pueden estar juntas. Cuando se desliza, la habitación se hace añicos en ruido o sueño.

*(Puente: hablaremos de habitaciones por una escala estructural* $`L`$*, de cuánto tiempo pueden mantenerse por una duración* $`T`$*, y del pacto entre ellas por una pendiente* $`\alpha`$ *medida en una ventana válida de colapso.)*

**4.2 La afirmación (enunciado técnico)**

Dentro de un estado coherente—tarea fija, arousal, y geometría efectiva—el acceso consciente opera en ventanas donde

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (pendiente multipunto estimada en una ventana válida de colapso)}.
```

Estimamos $`\alpha`$ mediante regresión con errores en variables (ODR/TLS o Theil–Sen; SIMEX opcional), y publicamos **NO_COLLAPSE** cuando los diagnósticos fallan. Donde las topologías efectivas proporcionan atajos (difusión dominada por hubs), reportamos $`T \sim \log L`$ en lugar de forzar una ley de potencia.

**Política (pendiente primero).** Pendiente espectral $`\beta`$, PLV/coherencia, índices perturbacionales, umbrales de ignición, y medidas de retardo/eco son **auxiliares**—correlatos interpretativos bajo modelos explícitos, nunca sustitutos de $`\alpha`$, nunca convertidos a $`\alpha`$ por fórmula universal.

**4.3 Qué significan "tamaño" y "tiempo" para la conciencia**

- **Escala estructural** $`L`$**:** longitud de onda espacial modelada en fuente $`\lambda/2`$ de ensambles activos; radio de coaliciones funcionales; longitud de trayectoria hub-a-hub en un grafo de difusión efectivo; extensión de bucle tálamo–cortical; ancho de "mapa" representacional.

- **Tiempo propio** $`T`$**:** tiempo de acumulación de evidencia; latencia de acceso o reporte corregida por retardo motor; tiempo de permanencia de ignición; ventana de vinculación; ventana de atención sostenida.

**Ventanas.** Mantener estado y mecanismo constantes: bloque de tarea, banda de arousal, y configuración de red. Cambios de estrategia (ej., cambios atencionales) o derivas de estado rompen el colapso—reportar **MULTI-REGIME** o **NO_COLLAPSE**.

**4.4 Recuadro Lírico · La ventana**

Una ventana no es un agujero en una pared; es una promesa de que **lo que encaja será sostenido**. Demasiado pequeña, y el mundo se corta en fragmentos. Demasiado grande, y el mundo pasa sin ser captado. La conciencia es el oficio de elegir un vidrio que mantiene el viento y deja entrar la luz.

*(Puente: tamaño del vidrio* $`L`$*; tiempo de sostén* $`T`$*; dentro de un vidrio funcional,* $`T`$ *mantiene el paso con* $`L`$ *por una pendiente* $`\alpha`$*.)*

**4.5 Lentes mecanísticos (por qué** $`\mathbf{\alpha}`$ **toma estos valores)**

- **Acumulación recurrente:** integración a través de ensambles con velocidades y umbrales casi constantes a menudo produce $`\alpha \approx 1`$.

- **Integración difusiva:** cuando la evidencia se propaga sobre distancia representacional, $`\alpha \approx 2`$ puede aparecer.

- **Difusión global:** enrutamiento vía hubs de largo alcance aumenta $`L`$; bucles eficientes pueden preservar $`\alpha \approx 1`$, mientras cuellos de botella producen pendientes segmentadas (**MULTI-REGIME**).

- **Geometría de atajo:** atajos de hub fuertes comprimen la distancia efectiva, produciendo $`T \sim \log L`$ (no-potencia); etiquetar explícitamente.

- **Control de ganancia e inhibición:** cambios en excitabilidad desplazan interceptos (relojes) más que pendientes; asimetrías persistentes pueden revelar mezcla de régimen oculta.

Estos son indicadores; $`\alpha`$ siempre se mide de datos en una ventana definida.

**4.6 Construcción de conjuntos de datos para acceso consciente**

Construir observaciones pareadas $`\{(L_{i},T_{i})\}`$ bajo estado y mecanismo controlados.

**Paradigmas**

1.  **Acceso perceptual:** $`L = \lambda/2`$ desde extensión de ensamble modelada en fuente; $`T`$=latencia de acceso o reporte a precisión fija (corregida por motor).

2.  **Perturbar-y-medir (ej., EMT/perturbación):** $`L`$=radio/trayectoria de ensamble evocado; $`T`$=tiempo-a-respuesta-estable o permanencia de ignición dentro de límites de criterio.

3.  **Vinculación y atención:** $`L`$=mapa de características o extensión espacial de contenido vinculado; $`T`$=ventana de vinculación o permanencia atencional.

**Pre-registrar:** definiciones de escala/reloj; límites de bin; exclusiones (deriva de arousal, lapsos); corrección de retardo motor; manejo de artefactos.

**4.7 Estimando** $`\mathbf{\alpha}`$ **(métodos)**

- **Estimador:** regresión EIV en $`(\log L,\log T)`$ usando **ODR/TLS** (por defecto) o **Theil–Sen** (robusto); **SIMEX** opcional si la estructura de error es caracterizable.

- **Diagnósticos de colapso:** (i) linealidad log–log (sin curvatura de residuos), (ii) estabilidad de pendiente leave-one-condition-out, (iii) heterocedasticidad acotada, (iv) sin puntos de cambio ocultos.

- **Reporte:** $`\widehat{\alpha}`$, IC 95%, estimador, $`n`$, extensión en $`L`$, especificación de bin, mini-panel de residuos/colapso.

- **Modos de fallo:** **NO_COLLAPSE** (mezcla/curvatura), **LOG-SCALING** (métrica de atajo), **MULTI-REGIME** (pendientes segmentadas con puntos de quiebre).

**4.8 Bosquejos de casos (ilustrativos)**

**4.8.1 Extensión de ensamble vs. latencia de acceso**

Variar configuraciones de estímulo para reclutar ensambles de diferentes extensiones espaciales; modelar en fuente para obtener $`L = \lambda/2`$. Definir $`T`$ como latencia de acceso corregida por motor. Bins limpios a menudo muestran $`\alpha \approx 1`$; curvatura o pendientes segmentadas indican cuellos de botella o deriva de estado.

**4.8.2 Alcance de difusión vs. permanencia de ignición**

Definir $`L`$ por longitud de trayectoria hub-a-hub en un grafo de difusión efectivo; medir $`T`$ como permanencia de ignición (sobre un umbral pre-registrado). Integración eficiente puede mantener pendientes casi lineales; enrutamiento con cuellos de botella segmenta la pendiente.

**4.8.3 Enmascaramiento, sedación, y transiciones de estado**

Dentro de pasos estrechos de estado, $`L`$ desde ensamble modelado en fuente y $`T`$ como ventana de acceso revelan si la pendiente persiste. Cruzar regímenes produce **NO_COLLAPSE** o puntos de quiebre (reportar **MULTI-REGIME**). Estados profundos pueden mostrar **LOG-SCALING** si solo quedan atajos.

**4.9 Recuadro Técnico · Midiendo** $`\mathbf{\alpha}`$ **durante acceso consciente**

**Entradas:** pares $`(L,T)`$ dentro de un bin de estado constante.\
**Pasos:**

1.  Modelar ensambles en fuente para obtener $`L = \lambda/2`$; definir $`T`$ como tiempo de acceso/vinculación/ignición con corrección de motor.

2.  Ajustar pendiente **ODR/TLS** de $`\log T`$ sobre $`\log L`$; agregar **Theil–Sen** como robustez.

3.  Bootstrap ICs; ejecutar diagnósticos de colapso (curvatura de residuos, estabilidad LOO, verificaciones de heterocedasticidad).

4.  Reportar $`\widehat{\alpha}`$, IC, estimador, $`n`$, extensión, detalles de bin, y un recuadro de residuos/colapso.\
    **Resultados:** $`\widehat{\alpha}`$ (válido de colapso) o **NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**.

*Plantilla de leyenda (pendiente):* "Tiempo de acceso consciente $`T`$ vs. escala de ensamble $`L`$. Pendiente EIV $`\widehat{\alpha}`$ (IC 95%) en un bin válido de colapso (n = …). Los residuos no muestran curvatura."

**4.10 Recuadro Técnico · Señales auxiliares sin exceso**

**Auxiliares:** pendiente espectral $`\beta`$, PLV/coherencia, complejidad perturbacional, umbrales de ignición, componentes ERP.

- Reportar parámetros e incertidumbre; usar como covariables para interpretar $`\widehat{\alpha}`$ o para seleccionar bins.

- **No** convertir auxiliares en $`\alpha`$ o promediarlos en "$`\alpha`$ compuesto".

*Plantilla de leyenda auxiliar:* "PLV durante acceso. Auxiliar (no $`\alpha`$); sin conversión. Parámetros e IC reportados."

**4.11 Placebos y controles (reloj vs. estructura)**

- **Placebo de reloj:** reescalar marcas de tiempo o agregar pequeña fluctuación; $`\widehat{\alpha}`$ debe permanecer dentro del IC si la pendiente refleja estructura, no artefactos de reloj.

- **Placebo de estructura:** barajar patrones espaciales mientras se mantiene el reloj; el colapso debe fallar o la pendiente debe derivar si la estructura impulsa $`\alpha`$.

- **Control de retardo motor:** restar o fijar retardo motor/de reporte a través de $`L`$ para que cambios de intercepto no se hagan pasar por cambios de pendiente.

**4.12 Modos de fallo y significados**

- **Deriva de arousal dentro del bin** → curvatura; re-binear por bandas de estado más estrechas.

- **Fuga de espacio de sensores** → $`L`$ sesgado; preferir modelos de fuente o sustitutos controlados por fuga.

- **Contaminación de atajo** → $`T \sim \log L`$; etiquetar **LOG-SCALING**, no inferir $`\alpha`$.

- **Puntos de cambio ocultos** → **MULTI-REGIME**; reportar puntos de quiebre con interpretación.

**4.13 Coda lírica · El acorde que puedes sostener**

La conciencia no es una luz encendida sino un **sostenimiento** en el cual muchas longitudes acuerdan hablar por la misma cantidad de tiempo. El milagro es humilde: elige un vidrio, mantén una promesa, deja que el acorde resuene lo suficiente para significar.

**4.14 Lo que este capítulo no hace**

- **No** infiere $`\alpha`$ de $`\beta`$ espectral, PLV/coherencia, PCI, umbrales de ignición, o métricas de eco/retardo.

- **No** acepta fórmulas de un solo punto como $`\alpha`$.

- **No** fuerza leyes de potencia donde la topología produce $`T \sim \log L`$.

**4.15 Conclusión clave**

La conciencia es una **ventana integrativa**: una extensión estructural $`L`$ sostenida por una duración $`T`$ para que exista una pendiente medible $`\alpha`$. La lírica cuenta por qué nos importa; la pendiente cuenta si la ventana realmente se sostiene.

**Capítulo 5 · Intuición — Saltos Trans-Escala**

**Epígrafe.** *A veces el camino es corto porque el cuerpo lo recuerda.*

**5.1 Preludio poético · Cómo la respuesta llega temprano**

La intuición es el **atajo silencioso** que no miente. Se siente como saltar, pero es mayormente **recordar el tamaño correcto**—pararse en una forma que ya sabe cuánto tiempo debe tomar el trabajo. Lo que parece magia es un ajuste practicado: el mundo ofrece una extensión, y nosotros respondemos con el tiempo que esa extensión puede sostener.

*(Puente: nombraremos la extensión por una escala estructural* $`L`$*, el tiempo de respuesta por una duración* $`T`$*, y su pacto por una pendiente* $`\alpha`$ *medida en una ventana válida de colapso.)*

**5.2 La afirmación (enunciado técnico)**

Dentro de un mecanismo estable (tarea, representación, política de control), el desempeño intuitivo ocupa ventanas donde

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (pendiente multipunto en una ventana válida de colapso).}
```

Estimamos $`\alpha`$ con regresión con errores en variables (ODR/TLS o Theil–Sen; SIMEX opcional) y publicamos **NO_COLLAPSE** cuando los diagnósticos fallan. Cuando la métrica efectiva proporciona atajos duros (direccionamiento, hashing, recuperación directa), reportamos $`T \sim \log L`$ en lugar de forzar una ley de potencia.

**Política (pendiente primero).** Confianza, pendiente espectral $`\beta`$, PLV/coherencia, y puntuaciones heurísticas son **auxiliares**: covariables interpretativas, nunca sustitutos de $`\alpha`$, nunca convertidos a $`\alpha`$ por fórmula universal.

**5.3 Qué significan "tamaño" y "tiempo" para la intuición**

- **Escala estructural** $`L`$**:** tamaño de fragmento o granularidad representacional; número de elementos recuperados en un esquema; radio de vecindario de búsqueda; extensión de plantilla en espacio motor; ventana de contexto en juicio rápido.

- **Tiempo propio** $`T`$**:** tiempo-a-primera-acción; latencia de decisión a precisión fija; latencia de solución bajo exposición breve; tiempo de estabilización después de una conjetura de un solo intento.

**Ventanas.** Mantener instrucciones, pago, y recursos constantes (sin cambios de estrategia a mitad de ensayo). Si las estrategias se mezclan (recuerdo → búsqueda), reportar **MULTI-REGIME** o segmentar pendientes; no promediar.

**5.4 Recuadro Lírico · La mirada del tamaño correcto**

Una buena mirada no es velocidad sino **ajuste**. Colocas tu atención en un cuenco donde el problema puede sentarse. Si el cuenco es demasiado pequeño, la verdad se derrama; demasiado ancho, y el sabor se diluye. El cuenco correcto sostiene el sabor lo suficiente para saber.

*(Puente: el cuenco es* $`L`$*; el tiempo de saboreo es* $`T`$*; la proporción sostenida es* $`\alpha`$*.)*

**5.5 Lentes mecanísticos (por qué** $`\mathbf{\alpha}`$ **toma estos valores)**

- **Recuperación de plantilla:** reutilizar un mapeo almacenado a escala $`L`$ a menudo produce temporización casi lineal ($`\alpha \approx 1`$) cuando los anchos de banda de acceso y actuación son estables.

- **Razonamiento fragmentado:** aumentar el tamaño del fragmento eleva $`L`$ mientras evita búsqueda serial; las pendientes dependen del costo de ensamblaje del fragmento—puntos de quiebre marcan el límite de fragmentos utilizables (**MULTI-REGIME**).

- **Kernels de similitud e indexación:** si el acceso es efectivamente logarítmico en extensión (ej., memoria direccionable), la temporización se comporta $`T \sim \log L`$ (no-potencia).

- **Puerta y avanza:** relojes mejorados reducen interceptos ($`a`$ más rápido en $`\log T = a + \alpha\log L`$) sin cambiar $`\alpha`$; distinguir cambios de reloj de cambios de pendiente.

Estos son guías; $`\alpha`$ siempre se mide de datos dentro de una ventana definida.

**5.6 Construcción de conjuntos de datos para escalamiento intuitivo**

Crear observaciones pareadas $`\{(L_{i},T_{i})\}`$ **variando paramétricamente una escala estructural** mientras se congelan los mecanismos.

**Ejemplos**

- **Juicios perceptuales rápidos:** $`L`$=radio de parche/campo receptivo; $`T`$=tiempo a primera categoría correcta a precisión fija.

- **Selección motora de un intento:** $`L`$=extensión de trayectoria o tamaño de campo de obstáculos; $`T`$=tiempo-a-compromiso usando una primitiva aprendida.

- **Tareas de insight cognitivo:** $`L`$=escala del problema (ancho de cuadrícula, conteo de cláusulas); $`T`$=latencia bajo exposición breve sin trabajo en borrador.

- **Recuperación experta:** $`L`$=tamaño de esquema (conteo de pistas vinculadas); $`T`$=tiempo-a-primera respuesta confiada.

**Pre-registrar:** definiciones de $`L,T`$; límites de bin; reglas de exclusión (timeouts, ensayos de baja confianza si la precisión es fija); historial de entrenamiento permitido antes de la prueba.

**5.7 Estimando** $`\mathbf{\alpha}`$ **(métodos)**

- **Estimador:** pendiente EIV de $`\log T`$ sobre $`\log L`$: **ODR/TLS** (por defecto), **Theil–Sen** (robusto), **SIMEX** opcional para corrección de error.

- **Diagnósticos de colapso:** (i) linealidad log–log sin curvatura de residuos, (ii) estabilidad de pendiente leave-one-condition-out, (iii) heterocedasticidad acotada, (iv) sin puntos de cambio ocultos.

- **Reporte:** $`\widehat{\alpha}`$, IC 95%, estimador, $`n`$, extensión en $`L`$, especificación de bin, recuadro de residuos/colapso.

- **Modos de fallo:** **NO_COLLAPSE** (estrategias mezcladas), **LOG-SCALING** (atajos verdaderos), **MULTI-REGIME** (quiebres de plantilla vs. búsqueda).

**5.8 Bosquejos de casos (ilustrativos)**

**5.8.1 Categorización rápida**

Variar $`L`$ (extensión del estímulo) a dificultad constante; $`T`$=tiempo a primera categoría correcta. Bins limpios a menudo muestran $`\alpha \approx 1`$; curvatura indica competencia entre plantillas foveales y búsqueda periférica.

**5.8.2 Selección motora de un intento**

Sea $`L`$=extensión de movimiento; $`T`$=latencia para comprometerse usando una primitiva entrenada. Pendientes casi lineales aparecen cuando la primitiva cubre la extensión; en $`L`$ grande la re-planificación inserta puntos de quiebre (**MULTI-REGIME**).

**5.8.3 Estimación numérica**

$`L`$=numerosidad o extensión espacial de un código comprimido; $`T`$=tiempo para estimar bajo visualización breve. Si la codificación es logarítmica, etiquetar **LOG-SCALING**; no inferir $`\alpha`$.

**5.8.4 Recuperación de patrón experto**

$`L`$=tamaño de esquema; $`T`$=tiempo-a-primer movimiento/respuesta. Expertos muestran interceptos más bajos (relojes más rápidos) y colapso más limpio; motivos no familiares producen pendientes segmentadas.

**5.9 Recuadro Técnico · Experimentos de intuición limpios**

**Objetivo:** aislar selección de banda rápida de búsqueda deliberada.

1.  **Bloqueo de instrucciones:** responder en primera impresión confiada; ventanas de respuesta estrictas.

2.  **Extensión y granularidad:** ≥0.6–1.0 décadas en $`L`$; ≥6 niveles distintos.

3.  **Sin ensayo dentro del bin:** aleatorizar $`L`$; prevenir deriva de estrategia.

4.  **Réplicas:** ≥3 ensayos por nivel; modelar error de medición.

5.  **Diagnósticos:** pre-registrar pruebas de colapso; verificaciones de residuos y puntos de cambio.

**Resultado:** $`\widehat{\alpha}`$ con IC (válido de colapso) **o** un negativo etiquetado (NO_COLLAPSE / LOG-SCALING / MULTI-REGIME).

**5.10 Recuadro Técnico · Separando reloj de pendiente**

Ajustar $`\log T = a + \alpha\log L`$ a través de condiciones (ej., novato vs. experto).

- **Igual** $`\alpha`$**, menor** $`a`$**:** reloj más rápido (práctica) con la misma gramática estructura–tiempo.

- **Diferente** $`\alpha`$**:** estructura activa diferente (re-binear) o cambio de mecanismo.\
  Reportar ambos parámetros con ICs; evitar confundir velocidad con coherencia.

**5.11 Usando auxiliares sin excederse**

**Auxiliares:** confianza, $`\beta`$ espectral, PLV/coherencia, diámetro pupilar, sorpresa/entropía.

- Reportar métodos e incertidumbre; usar como covariables para interpretar $`\widehat{\alpha}`$ o elegir bins.

- **Nunca** convertir auxiliares en $`\alpha`$ o promediarlos en un "$`\alpha`$ compuesto".

*Plantilla de leyenda auxiliar:* "Confianza vs. tamaño de estructura. Auxiliar (no $`\alpha`$); sin conversión. Parámetros e IC reportados."

**5.12 Modos de fallo y significados**

- **Filtración de estrategia:** búsqueda analítica se cuela → curvatura; ajustar plazos o re-binear.

- **Confusiones de pista:** $`L`$ más grande inadvertidamente facilita la tarea; igualar dificultad entre niveles.

- **Exceso de plantilla:** una plantilla encaja solo parte de la extensión; pendientes se segmentan—reportar **MULTI-REGIME**.

- **Disfraz de atajo:** direccionamiento verdadero hace $`T \sim \log L`$; etiquetar **LOG-SCALING** y parar.

**5.13 Coda lírica · La sensación de un salto verdadero**

Los mejores saltos no son lejos; son **exactos**. Aterrizas donde el suelo estaba esperando. La respuesta se siente inmediata porque la escala era correcta, y el tiempo—agradecido—no tuvo que vagar.

**5.14 Lo que este capítulo no hace**

- **No** infiere $`\alpha`$ de confianza, $`\beta`$, o coherencia.

- **No** acepta razones de un solo punto como $`\alpha`$.

- **No** fuerza leyes de potencia donde los atajos de representación implican $`T \sim \log L`$.

**5.15 Conclusión clave**

La intuición es **mantenimiento de banda rápido**: elegir una escala estructural $`L`$ y una duración $`T`$ que producen una pendiente coherente $`\alpha`$ al primer intento. Medir la pendiente con rigor; dejar que los auxiliares iluminen, no reemplacen, el salto.

**Capítulo 6 · Hacia una Cultura de Coherencia**

**Epígrafe.** *El método es la amabilidad que extendemos a la verdad.*

**6.1 Preludio poético · La casa que construimos para la evidencia**

Las historias son rápidas; la evidencia es paciente. Si queremos resultados que viajen—a través de laboratorios, años, e idiomas—debemos darle a la verdad una buena casa: habitaciones con puertas claras, ventanas que se abren, pisos que sostienen. En este libro esa casa es un pacto simple: **estructura primero, tiempo segundo, pendiente de muchos puntos, y honestidad cuando la línea no se sostiene**.

*(Puente: las reglas de la casa a continuación mantienen la lírica viva sin dejarla reescribir los datos.)*

**6.2 Los no negociables (enunciado técnico)**

1.  **Pendiente primero:** $`\alpha`$ se estima **únicamente** como una pendiente log–log multipunto de $`\log T`$ sobre $`\log L`$ dentro de ventanas válidas de colapso.

2.  **Las ventanas importan:** los bins se pre-registran (escala $`L`$, reloj $`T`$, límites, puntos de cambio, exclusiones). Mecanismos mezclados ⇒ **MULTI-REGIME** o **NO_COLLAPSE**.

3.  **Auxiliares ≠** $`\alpha`$**:** $`\beta`$ espectral, PLV/coherencia, eco/retardo, puntuaciones de información son **auxiliares**—nunca sustitutos, nunca convertidos por fórmulas universales.

4.  **Honestidad topológica:** si la geometría efectiva implica $`T \sim \log L`$, reportar **LOG-SCALING**; no forzar una ley de potencia.

5.  **Errores en variables por defecto:** ODR/TLS o Theil–Sen; SIMEX opcional donde el error de medición es estimable; siempre reportar ICs 95% y diagnósticos.

**6.3 Recuadro Lírico · Sobre mantener promesas a los datos**

La precisión es una forma de ternura. No coaccionamos una línea de una nube reacia; escuchamos dónde es recta y dónde cambia de tono. El poema puede ampliar la habitación; el método decide si la habitación existe.

**6.4 Principios de diseño para proyectos y laboratorios**

- **Pre-registro:** definir $`L`$, $`T`$, límites de bin, inclusión/exclusión, y criterios de fallo antes de ajustar pendientes.

- **Extensión y granularidad:** apuntar ≥0.6–1.0 décadas en $`L`$ con ≥6 niveles distintos dentro de un solo mecanismo.

- **Constancia dentro de bins:** fijar arquitectura/política/estado; registrar desviaciones.

- **Replicación y error:** ≥3 réplicas por nivel; modelar ruido de ambos ejes; aplicar SIMEX si el error escala con $`L`$.

- **Diagnósticos obligatorios:** verificaciones de curvatura de residuos, estabilidad leave-one-out, límites de heterocedasticidad, pruebas de puntos de cambio.

- **La segmentación supera al promedio:** si hay un punto de quiebre, reportar pendientes segmentadas con interpretación, no una línea mezclada.

- **Veracidad de leyendas:** cualquier figura que reclame $`\alpha`$ muestra la dispersión, ajuste, IC 95%, estimador, $`n`$, extensión, y al menos un panel de colapso.

**6.5 Recuadro Técnico · Estándar de reporte (lista de verificación para copiar y pegar)**

- **Definición:** cómo se computaron $`L`$ y $`T`$ y por qué coinciden con mecanismo/geometría.

- **Estimación:** estimador (ODR/TLS/Theil–Sen), método de IC, $`n`$, extensión en $`L`$.

- **Diagnósticos:** gráfico de residuos, estabilidad de pendiente LOOCV, verificación de heterocedasticidad, prueba de puntos de cambio.

- **Etiqueta de resultado:** $`\widehat{\alpha}`$ (válido de colapso) **o** NO_COLLAPSE / LOG-SCALING / MULTI-REGIME.

- **Auxiliares:** listar β, PLV/coherencia, retardos, puntuaciones de información con métodos e ICs; **etiquetar explícitamente como auxiliar**; sin conversiones a $`\alpha`$.

*Plantilla de divulgación de una línea:*\
"$`\alpha`$ estimado como pendiente EIV sobre $`\log T`$ vs $`\log L`$ (ODR/TLS), ventana válida de colapso, $`n = \ldots`$, extensión … décadas; IC 95% …; auxiliares (β/PLV/…) reportados sin conversión."

**6.6 Reproducibilidad: cómo hacer figuras reconstruibles**

- **Diseño del repositorio:**\
  /raw/ entradas → /proc/ pares $`(L,T)`$ con banderas → /qc/ diagnósticos → /features/ auxiliares → /figures/ paneles.

- **Bloqueo de entorno:** versiones, semillas, hardware; script de reconstrucción de un comando.

- **Manifiesto por figura:** YAML/JSON con entradas, parámetros, hashes, y etiqueta de resultado.

- **Negativos preservados:** NO_COLLAPSE, LOG-SCALING, MULTI-REGIME viven en el repositorio y el artículo (no en la basura).

**6.7 Ética de afirmaciones de escalamiento**

- **Sin alquimia de proxies:** β→$`\alpha`$ (o cualquier mapeo universal de proxy) está prohibido a menos que un **modelo específico** sea declarado y **validado fuera de muestra**.

- **Sin** $`\alpha`$ **de un solo punto:** razones como $`\log(T_{i}/T_{0})/log(L_{i}/L_{0})`$ no son pendientes; evaden incertidumbre y diagnósticos.

- **Incertidumbre transparente:** reportar ICs y sensibilidad; resistir redondear regímenes a enteros atractivos.

- **Los fallos enseñan:** un negativo etiquetado es conocimiento sobre límites de mecanismo, no una mancha a esconder.

**6.8 Prácticas institucionales (cómo pueden ayudar editores, IPs, y revisores)**

- **Dos preguntas primero:** (1) ¿De dónde vienen los pares $`(L,T)`$? (2) ¿Dónde está la verificación de colapso?

- **Análisis registrados:** requerir un plan de pendiente pre-registrado con criterios de fallo.

- **Crédito por negativos:** aceptar y citar reportes limpios de NO_COLLAPSE / LOG-SCALING / MULTI-REGIME.

- **Micro-subvenciones de replicación:** fondos pequeños reservados para verificación independiente de pendientes entre laboratorios.

**6.9 Recuadro Técnico · Herramientas esenciales**

- **Ajustador de pendiente:** ODR/TLS + Theil–Sen con ICs de bootstrap, opción SIMEX.

- **Kit de diagnósticos:** residuos, pruebas de rachas, detección de puntos de cambio, verificaciones de heterocedasticidad.

- **Gestor de ventanas:** impone inclusión/exclusión, rastrea resultados y etiquetas.

- **Generador de figuras:** plantillas estandarizadas de pendiente/auxiliar y leyendas.

- **Placebos:** reescalamiento de reloj/fluctuación y barajado de estructura para confirmar origen de pendiente.

**6.10 Patrones de uso incorrecto (y correcciones exactas)**

- **Inflación de proxy:** "β≈1 ⇒ $`\alpha`$≈1.5."\
  **Corrección:** re-etiquetar β como auxiliar; estimar $`\alpha`$ de $`(T,L)`$ o descartar la afirmación.

- **α de un solo punto:** "pendientes" basadas en identidad.\
  **Corrección:** recolectar ≥6 niveles de $`L`$, ajustar EIV, publicar IC y diagnósticos.

- **Agrupación entre regímenes:** líneas mezcladas.\
  **Corrección:** ajustes segmentados con puntos de quiebre y notas de mecanismo.

- **Ceguera topológica:** forzar potencia cuando $`T \sim \log L`$.\
  **Corrección:** reportar LOG-SCALING y discutir geometría.

**6.11 Recuadro Lírico · Sobre publicar negativos**

Una línea fallida es un mapa: muestra dónde el río se bifurca, dónde el terreno se eleva, dónde comienza otro sendero. Le debemos al próximo viajero un letrero, no un mito pulido.

**6.12 Limitaciones y problemas abiertos**

- **Elegir el** $`L`$ **correcto:** la selección de métrica es el paso más difícil; sustitutos de espacio de sensores pueden sesgar pendientes. Preferir métricas de nivel de fuente o físicamente fundamentadas.

- **Leyes locales:** $`\alpha`$ vive en ventanas finitas; universales globales son improbables. Aprender a narrar **dónde** la ley se sostiene.

- **Causalidad:** $`\alpha`$ es descriptivo del acoplamiento estructura–tiempo; intervenciones y placebos ayudan, pero las afirmaciones deben permanecer modestas.

- **Costo:** EIV + diagnósticos añaden sobrecarga; herramientas estandarizadas y manifiestos reducen la fricción.

**6.13 Un código de práctica compacto (colgar esto en la pared del laboratorio)**

1.  Definir $`L`$ y $`T`$ por mecanismo.

2.  Pre-registrar bins y criterios de fallo.

3.  Estimar $`\alpha`$ con EIV en $`(\log L,\log T)`$.

4.  Requerir colapso; **publicar negativos**.

5.  Etiquetar auxiliares; nunca convertirlos en $`\alpha`$.

6.  Preferir segmentación a promedio; reportar puntos de quiebre.

7.  Declarar topología cuando no-potencia (ej., $`T \sim \log L`$).

8.  Hacer cada figura reconstruible.

**6.14 Coda de cierre · La cultura como una promesa medible**

Una cultura de coherencia no es anti-poesía; es poesía con andamiaje. Mantenemos la canción—y mostramos la partitura. Cuando futuros lectores abran este libro, que escuchen ambas: el ritmo de un mundo que mantiene el tiempo con sus formas, y la prueba silenciosa de que el ritmo realmente estaba allí.

**Apéndice A · Conceptos y Definiciones (Solo Técnico)**

**A.1 Notación y Símbolos**

- $`L`$: escala estructural (longitud, longitud de onda, longitud de trayectoria, longitud de correlación, o sustituto validado con consistencia dimensional).

- $`T`$: duración propia del proceso gobernado por la estructura que define $`L`$.

- $`\alpha`$: **exponente de coherencia**, la pendiente log–log de $`T`$ sobre $`L`$ dentro de una ventana válida de colapso.

- $`(L_{i},T_{i})`$: observaciones pareadas; $`i = 1,\ldots,n`$.

- $`\widehat{\alpha}`$: pendiente estimada; $`{CI}_{95\%}`$: intervalo de confianza al 95%.

- "Ventana/bin": rango finito de $`L`$ sobre el cual mecanismo y métrica efectiva permanecen constantes.

- "Colapso": conjunto de diagnósticos de que una línea log–log es apropiada en una ventana (Sec. A.5).

- Observables auxiliares: pendiente espectral $`\beta`$, PLV/coherencia, potencias de banda, estadísticas de ráfaga/ISI, medidas de eco/retardo.

- Etiquetas de fallo: **NO_COLLAPSE**, **LOG-SCALING**, **MULTI-REGIME**.

**A.2 Escalas Estructurales y Temporales**

**A.2.1 Definiendo** $`L`$**.**\
Elegir una escala ligada al mecanismo, que preserve la geometría:

- Longitud física (ej., radio/longitud de trayectoria de vaso, longitud de cable dendrítico, longitud de extremidad).

- Longitud de onda modelada en fuente ($`L = \lambda/2`$); longitud de correlación de espectros de potencia espaciales/variogramas.

- Longitud geodésica de red (distancia efectiva a lo largo de la conectividad operativa).

- Sustituto validado (espacio de sensores solo si fuente/geometría no está disponible, marcado como sustituto).

**A.2.2 Definiendo** $`T`$**.**\
Duración del mecanismo sostenido por $`L`$: tiempo de circulación, tiempo medio de difusión/reacción, período de oscilación $`T = 1/f`$, latencia de integración/acceso (corregida por motor), tiempo de asentamiento, retardo de consolidación, vida media de retención.

**A.2.3 Reglas de inclusión.**\
Declarar cómo se computan $`L`$ y $`T`$; mantener definiciones fijas dentro de cada ventana. Excluir artefactos, patología, y tareas/mecanismos mezclados según criterios pre-registrados.

**A.3 Ventanas y Regímenes**

- **Selección de ventana.** Pre-registrar límites en $`L`$ y condiciones (estado/tarea/arquitectura) para mantener el mecanismo operativo constante.

- **Extensión y granularidad.** Apuntar ≥0.6–1.0 décadas en $`L`$ con ≥6 niveles distintos de $`L`$ dentro de una ventana.

- **Puntos de cambio.** Si las pendientes difieren entre subrangos, reportar **MULTI-REGIME** con ajustes segmentados; no promediar entre regímenes.

**A.4 Estimando** $`\mathbf{\alpha}`$ **(Errores en Variables)**

Estimamos $`\alpha`$ **únicamente** de regresiones multipunto de $`\log T`$ sobre $`\log L`$.

**A.4.1 Modelo.**

``` math
\log T_{i} = a + \alpha\text{ }\log L_{i} + \varepsilon_{i},\text{con ruido en ambos ejes.}
```

**A.4.2 Estimadores.**

- **ODR/TLS** (por defecto): tiene en cuenta el error de medición en ambos $`L`$ y $`T`$.

- **Theil–Sen** (robusto): mediana de pendientes; usar como verificación de sensibilidad.

- **SIMEX** (opcional): corrección de error cuando las varianzas de error son estimables.

**A.4.3 Incertidumbre.**\
Reportar $`\widehat{\alpha}`$ con $`{CI}_{95\%}`$ (bootstrap o asintótico), más $`n`$, extensión en $`L`$, y detalles del estimador.

**A.5 Diagnósticos de Colapso y Regularidad**

Una ventana es **válida de colapso** si todo se cumple:

1.  **Linealidad log–log:** los residuos no muestran curvatura (prueba de rachas o inspección de residuos LOESS).

2.  **Estabilidad:** la pendiente leave-one-out (o leave-one-level-out) permanece dentro de una tolerancia pre-registrada.

3.  **Límites de heterocedasticidad:** patrones de varianza dentro de límites (ej., Breusch–Pagan o verificaciones estratificadas por varianza).

4.  **Sin puntos de cambio ocultos:** ajustes segmentados no superan significativamente una sola pendiente a menos que se etiquete **MULTI-REGIME**.

5.  **Extensión/granularidad suficiente:** cumple objetivos de Sec. A.3 (si no, marcar exploratorio).

**Manejo de fallos.**

- Curvatura o inestabilidad → **NO_COLLAPSE**.

- Linealidad semi-log → **LOG-SCALING** (reportar $`T \sim \log L`$, no inferir $`\alpha`$).

- Punto(s) de quiebre claro(s) → **MULTI-REGIME** (reportar $`\alpha`$ segmentado con puntos de quiebre).

**A.6 Expectativas Mecanísticas (Guías No Vinculantes)**

- **Difusión–reacción:** $`T \sim L^{2}/D \Rightarrow \alpha \approx 2`$ cuando $`D`$ es aproximadamente constante en la ventana.

- **Advección/balístico:** velocidad/tasas de proceso casi constantes → $`\alpha \approx 1`$.

- **Atajo/compresión topológica:** distancias efectivas tipo mundo pequeño → $`T \sim \log L`$ (no-potencia).\
  Estas expectativas guían el **diseño**, no la inferencia; $`\alpha`$ debe ser **medido** según A.4–A.5.

**A.7 Relaciones de Escalamiento y Observables Operacionales**

**Definición (operacional).**\
En RTM, el exponente $`\alpha`$ se define **exclusivamente** del escalamiento tiempo–tamaño dentro de un régimen:

``` math
T \propto L^{\alpha} \Longleftrightarrow \alpha = \frac{d\log T}{d\log L}.
```

Los valores estimados de $`\alpha`$ vienen **únicamente** de pendientes log–log de $`\log T`$ vs. $`\log L`$ computadas en una **ventana finita que pasa verificaciones de colapso y regularidad** (bins pre-registrados, puntos de cambio, exclusiones; ver A.5).

**Observables auxiliares (proxies, no** $`\alpha`$**).**\
Señales de dominio como pendientes espectrales $`\beta`$ (de $`P(f) \propto f^{- \beta}`$), índices de acoplamiento entre frecuencias, estadísticas de ráfaga/ISI, tiempos de eco/retardo, y medidas de sincronía/coherencia pueden ser **correlatos** informativos bajo **modelos específicos**. **No** son $`\alpha`$ y **no hay conversión universal** (ej., $`\alpha \neq 1 + \beta/2`$ en general). Cuando se reportan, se etiquetan **auxiliar** y nunca se usan como sustitutos de $`\alpha`$.

**Estándar de reporte.** Para cada régimen reportar:

1.  la dispersión $`(\log L,\log T)`$, pendiente ajustada $`\widehat{\alpha}`$, $`{CI}_{95\%}`$, y estimador (**ODR/TLS**, **Theil–Sen**, **SIMEX** opcional);

2.  diagnósticos de colapso y especificación de ventana (bins, puntos de cambio, exclusiones, modos de fallo);

3.  cualquier análisis **correlativo** relacionando $`\widehat{\alpha}`$ con observables auxiliares (ej., $`\beta`$), marcado explícitamente como **dependiente del modelo** y, donde sea aplicable, evaluado **fuera de muestra**.

**Salvaguardas.**

- **No** sustituir $`\beta`$, medidas de eco/retardo, PLV/coherencia, u otros marcadores por $`\alpha`$.

- **No** agrupar regímenes mezclados o reportar un solo $`\alpha`$ a través de ventanas que fallan las pruebas de colapso.

- Si la **métrica efectiva es no euclidiana/topológica** (ej., geodésicas de mundo pequeño), declarar la métrica explícitamente y **reportar** $`T \sim \log L`$ en lugar de forzar un ajuste de ley de potencia.

**Puentes condicionales al modelo (opcional).**\
Si una teoría específica implica un mapeo $`\alpha = g(\beta;\text{ modelo, banda, estimador})`$, **pre-registrar** $`g`$ y criterios de fallo explícitos, estimar $`\widehat{\alpha}`$ **directamente** de $`T`$–$`L`$, y evaluar el acuerdo **fuera de muestra** entre $`g(\beta)`$ y $`\widehat{\alpha}`$. Fallo ⇒ revisar o descartar $`g`$.

**Lo que aparece en la tabla de resultados.**\
Solo valores de $`\widehat{\alpha}`$ obtenidos de ajustes **log–log** $`\log T`$–$`\log L`$ que pasan colapso se tabulan. Observables auxiliares aparecen en una columna separada (o apéndice) y **nunca** se usan como sustitutos de $`\alpha`$.

**A.8 Estándar de Reporte (Lista de Verificación para Copiar y Pegar)**

Para cada afirmación de $`\alpha`$ incluir:

- **Definición:** cómo se computaron $`L`$ y $`T`$ y por qué coinciden con el mecanismo.

- **Estimación:** estimador (ODR/TLS/Theil–Sen), $`n`$, extensión en $`L`$, $`\widehat{\alpha}`$ con $`{CI}_{95\%}`$.

- **Diagnósticos:** mostrar (o enlazar) panel de residuos/colapso; declarar pasa/falla.

- **Etiqueta de resultado:** $`\widehat{\alpha}`$ (válido de colapso) **o** NO_COLLAPSE / LOG-SCALING / MULTI-REGIME.

- **Auxiliares:** listados con métodos e ICs, etiquetados explícitamente como **auxiliar**; **sin conversión** a $`\alpha`$.

*Plantilla de leyenda (pendiente):*\
"Escalamiento de $`T`$ con $`L`$ en \[sistema\]. Pendiente EIV $`\widehat{\alpha}`$ (IC 95%) en una ventana válida de colapso (n = …; extensión … décadas). Los residuos no muestran curvatura. Estimador: \[ODR/TLS \| Theil–Sen \| SIMEX\]."

*Plantilla de leyenda (auxiliar):*\
"Pendiente espectral $`\beta`$/PLV/coherencia en \[sistema\]. **Auxiliar** (no $`\alpha`$); sin conversión. Método e IC 95% reportados."

**A.9 Reproducibilidad y Controles**

- **Diseño del repositorio:** /raw/ (entradas) → /proc/ (pares $`(L,T)`$ con banderas) → /qc/ (diagnósticos) → /features/ (auxiliares) → /figures/ (paneles).

- **Bloqueo de entorno:** versiones, semillas, hardware; script de reconstrucción de un comando.

- **Placebos:**

  - *Placebo de reloj:* reescalar marcas de tiempo o agregar fluctuación—$`\widehat{\alpha}`$ debe permanecer dentro del IC si la estructura impulsa la pendiente.

  - *Placebo de estructura:* barajar estructura espacial (mantener reloj)—el colapso debe fallar o $`\widehat{\alpha}`$ derivar si la estructura es causal.

- **Negativos preservados:** publicar y archivar resultados de **NO_COLLAPSE**, **LOG-SCALING**, **MULTI-REGIME**.

**Resumen del Apéndice A.**\
$`\alpha`$ es una **pendiente medida** de $`\log T`$ sobre $`\log L`$ dentro de ventanas válidas de colapso. Las señales auxiliares informan pero nunca lo reemplazan. Comportamientos no-potencia se etiquetan como tales. El reporte es estandarizado, los diagnósticos son obligatorios, y los negativos son resultados de primera clase.

**Apéndice B · Protocolos (Solo Técnico)**

**Política.** En todos los protocolos a continuación, $`\alpha`$ se estima **únicamente** como una pendiente log–log multipunto de $`\log T`$ sobre $`\log L`$ dentro de una ventana válida de colapso. Pendientes espectrales ($`\beta`$), PLV/coherencia, medidas de eco/retardo, e índices relacionados son **auxiliares**—se reportan con incertidumbre pero nunca se convierten a $`\alpha`$ por fórmulas universales.

**B.1 Variabilidad de la Frecuencia Cardíaca (VFC) — Observables Auxiliares (No** $`\mathbf{\alpha}`$**)**

**Propósito.** Proporcionar un pipeline estandarizado para reportar descriptores de VFC (incluyendo pendiente espectral $`\beta`$) como correlatos **auxiliares**. No se estima $`\alpha`$ de VFC.

**B.1.1 Datos y Grabación**

- **Adquisición:** ECG a ≥ 500 Hz (preferido 1 kHz).

- **Duración:** ≥ 10 min por condición (reposo/tarea), evitando grandes derivas de estado.

- **Derivaciones:** Configuración de pecho estándar; registrar respiración si está disponible.

- **Metadatos:** postura, medicación, hora del día, ejercicio/cafeína reciente.

**B.1.2 Preprocesamiento**

1.  **Detección de pico R:** detector validado; revisión manual de segmentos ambiguos.

2.  **Manejo de artefactos:** remover latidos ectópicos/falsas detecciones; interpolar huecos RR (spline cúbico o adaptativo).

3.  **Bins de estacionaridad:** dividir en ventanas fijas $`W`$ (ej., 120 s, solapamiento opcional 50%). Excluir ventanas con > 5% de artefactos o no-estacionaridad visible.

**B.1.3 Características (Auxiliares)**

- **Dominio temporal:** RR medio, SDNN, RMSSD, pNN50.

- **Dominio de frecuencia:** espectro Welch o AR en tacograma remuestreado uniformemente (≤ 4 Hz de remuestreo).

  - **Bandas:** VLF/LF/HF según puntos de corte pre-registrados; reportar potencia de banda y **pendiente espectral** $`\beta`$ en una banda especificada usando ajuste lineal robusto a $`\log P(f)`$ vs. $`\log f`$.

- **Acoplamiento respiratorio (si disponible):** coherencia/PLV RR–respiración (métodos, parámetros).

**B.1.4 Reporte**

- **Primario:** $`\beta`$ (pendiente) con IC 95%, método (Welch/AR, tamaño de ventana, solapamiento), definición de banda, configuraciones de remoción de tendencia.

- **Secundario:** potencias de banda, LF/HF, coherencia/PLV con ICs.

- **Etiquetado:** Cada leyenda de panel debe declarar **"Auxiliar (no** $`\alpha`$**); sin conversión a** $`\alpha`$**."**

**B.1.5 Reproducibilidad**

- **Archivos:** /raw/ECG, /proc/RR, /features/HRV_aux.csv (filas por ventana), /figures/HRV\_\*.

- **Manifiesto:** JSON/YAML con detector, política de interpolación, parámetros PSD, bordes de banda.

**B.2 Electroencefalografía (EEG) — Estimando** $`\mathbf{\alpha}`$

**Propósito.** Estimar $`\alpha`$ de mediciones pareadas de tiempo–tamaño usando escala espacial de nivel de fuente $`L`$ y escala temporal $`T`$. $`\beta`$ espectral, PLV/coherencia, etc., se reportan como **auxiliares**.

**B.2.1 Datos y Grabación**

- **Adquisición:** 64+ canales (≥ 500 Hz), disparadores sincronizados; registrar EOG/EMG; respiración opcional.

- **Montaje:** 10–10 o mayor densidad; modelo de cabeza individual (RMI) preferido; plantilla aceptable con advertencias.

- **Duración:** ≥ 10 min por condición para soportar ventanas estables.

- **Metadatos:** bloques de tarea, marcadores de arousal, medicación, edad, lateralidad.

**B.2.2 Preprocesamiento**

1.  **Filtrado:** 0.5–80 Hz (o según estudio); muesca en frecuencia de línea.

2.  **Manejo de artefactos:** interpolación de canales malos; ICA/SSP para remover parpadeo/cardiaco/músculo; registrar componentes retenidos.

3.  **Segmentación:** ventanas $`W`$ de 10–30 s (solapamiento 50% opcional). Excluir ventanas por umbrales de artefacto pre-registrados.

4.  **Tiempo–frecuencia para picos:** multitaper o Morlet para identificar picos de banda (theta 4–7, alfa 8–12, beta 13–30, gamma 30–45 Hz).

**B.2.3 Definiendo** $`\mathbf{(T,L)}`$

- **Escala temporal $T$:** para cada ventana y pico de banda $f^*$, establecer $T=1/f^*$ (segundos). Si múltiples picos por banda, pre-registrar una regla de centroide o criterio de dominancia.

- **Escala espacial** $`L`$**:** modelado de fuente (eLORETA/beamformer) para estimar longitud de onda espacial $`\lambda`$ (o longitud de correlación) de fuentes limitadas en banda; establecer $`L = \lambda/2`$.

  - **Estimadores para** $`\lambda`$**:**\
    (i) FWHM de blobs de activación cortical;\
    (ii) pico del espectro de potencia espacial 2D cortical (frecuencia espacial inversa);\
    (iii) rango de variograma. Pre-registrar uno y mantenerlo fijo.

> Si el modelado de fuente no está disponible, un sustituto de espacio de sensores puede usarse **solo como exploratorio** (ej., período SVD de topografía). Marcar claramente como sustituto.

**B.2.4 Ventanas y Criterios de Colapso**

Construir un conjunto de datos de puntos pareados $`\{(\log L_{i},\log T_{i})\}`$ a través de ROIs/ventanas con mecanismo/estado constante.

**Ventana válida de colapso requiere:**

1.  linealidad log–log de régimen único (sin curvatura de residuos),

2.  estabilidad de pendiente leave-one-ROI/window-out,

3.  heterocedasticidad dentro de límites preestablecidos,

4.  sin puntos de cambio ocultos,

5.  extensión ≥ 0.6 décadas en $`L`$ con ≥ 6 niveles distintos de $`L`$ (objetivo).

Los fallos se etiquetan **NO_COLLAPSE**. Si semi-log es lineal, etiquetar **LOG-SCALING** (no inferir $`\alpha`$).

**B.2.5 Estimando** $`\mathbf{\alpha}`$

- **Estimador:** regresión con errores en variables de $`\log T`$ sobre $`\log L`$: **ODR/TLS** (por defecto) o **Theil–Sen** (robusto).

- **Corrección de error:** **SIMEX** opcional cuando el error de medición es caracterizable.

- **Incertidumbre:** IC 95% (bootstrap o asintótico). Reportar $`n`$, extensión, estimador.

``` math
\widehat{\alpha} = pendiente\text{ }(\log T\text{ sobre }\log L),T \propto L^{\alpha}.
```

**B.2.6 Controles y Confusiones**

- **Dependencia de estado:** estratificar por tarea vs. reposo y ojos abiertos/cerrados; no agrupar estados mezclados.

- **Conducción de volumen:** preferir métricas de nivel de fuente; si se usa espacio de sensores, aplicar controles de fuga (ej., ortogonalización).

- **Covariables fisiológicas:** respiración/VFC incluidas como covariables **auxiliares**; nunca convertidas a $`\alpha`$.

- **Advertencia de topología:** para geometría efectiva tipo mundo pequeño, reportar $`T \sim \log L`$ en lugar de forzar potencia.

**B.2.7 Salidas**

- **Tabla de alfa (por sujeto/condición):** $`\widehat{\alpha}`$, IC, estimador, $`n`$, extensión, banderas de colapso, especificación de bin.

- **Tabla de QC:** tasas de artefactos, ventanas rechazadas, parámetros del modelo de fuente.

- **Figuras:**\
  (i) dispersión $`\log T`$ vs. $`\log L`$ con ajuste + IC 95%;\
  (ii) gráfico de residuos;\
  (iii) panel de sensibilidad leave-one-out.

- **Auxiliares (separados):** $`\beta`$ espectral, PLV/coherencia; etiquetados como **no** $`\alpha`$.

**B.2.8 Lista de Verificación del Repositorio**

- /proc/EEG/ datos preprocesados + índices de ventana.

- /src/ modelos de cabeza, campos guía, parámetros inversos.

- /features/EEG_pairs.csv con $`(\log L,\log T)`$ y metadatos.

- /qc/EEG_collapse_reports/ diagnósticos por bin.

- /figures/EEG_scaling\_\* paneles estandarizados.

- README.md con parámetros exactos, versiones, semillas.

**B.2.9 Modos de Fallo**

- **NO_COLLAPSE:** mezcla de mecanismos o métrica incorrecta; re-binear o revisar estimador de $`L`$.

- **LOG-SCALING:** topología de atajo; reportar ajuste semi-log, no $`\alpha`$.

- **MULTI-REGIME:** pendientes segmentadas con puntos de quiebre; reportar ambos segmentos e interpretación.

**(Opcional) B.3 Plantilla — Cualquier Modalidad con Pares Espacio–Tiempo**

Usar esta plantilla para otros dominios (MEG, fNIRS, robótica/control, ciclos conductuales) que pueden producir pares $`(L,T)`$ válidos.

**B.3.1 Definir** $`L`$ **y** $`T`$**.** Escala estructural ligada al mecanismo y duración propia; pre-registrar ambos y mantenerlos fijos dentro de las ventanas.

**B.3.2 Adquirir y preprocesar.** Adquisición apropiada al dominio; controles de artefactos; ventanas fijas $`W`$ con umbrales de exclusión.

**B.3.3 Construir pares.** Derivar $`(\log L,\log T)`$ a través de ≥ 6 niveles de $`L`$ con extensión objetivo ≥ 0.6 décadas.

**B.3.4 Estimar** $`\alpha`$**.** Regresión EIV (ODR/TLS; Theil–Sen), IC 95%, SIMEX opcional.

**B.3.5 Diagnósticos de colapso.** Curvatura de residuos, estabilidad de pendiente LOOCV, límites de heterocedasticidad, pruebas de puntos de cambio.

**B.3.6 Etiquetar resultados.** $`\widehat{\alpha}`$ (válido de colapso) o **NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**.

**B.3.7 Reportar auxiliares.** Métricas específicas del dominio (ej., $`\beta`$ espectral, PLV, puntuaciones de información) con ICs; explícitamente **no** $`\alpha`$.

**B.3.8 Reproducibilidad.** Estructura de carpetas, manifiestos, reconstrucción de un comando.

**Resumen del Apéndice B.** VFC se reporta solo como **auxiliar**; EEG proporciona un pipeline canónico, pendiente-primero para estimar $`\alpha`$. Todas las demás modalidades siguen el mismo plano: definir $`L`$ y $`T`$, obtener pares multipunto, verificar colapso, ajustar pendientes EIV con incertidumbre, etiquetar fallos explícitamente, y mantener los auxiliares en su propio carril.

**Apéndice C · Glosario Extendido (Solo Técnico)**

**Alcance.** Definiciones usadas a lo largo del libro. A menos que se indique lo contrario, $`\log`$ es logaritmo natural y "pendiente" significa el coeficiente de una regresión de $`\log T`$ sobre $`\log L`$. Todas las entradas evitan metáforas y se mantienen en significado operacional.

$`\mathbf{\alpha}`$ **(exponente de coherencia):**
La pendiente log–log **multipunto** de $`\log T`$ sobre $`\log L`$ estimada dentro de una **ventana válida de colapso**. Reportar como $`\widehat{\alpha}`$ con IC 95%, estimador, $`n`$, extensión.

**Observable auxiliar:**
Cualquier métrica de dominio que **no** es $`\alpha`$ (ej., pendiente espectral $`\beta`$, PLV/coherencia, potencias de banda, eco/retardo). Puede correlacionarse con $`\alpha`$ de maneras **específicas al modelo**; nunca un sustituto universal.

**Banda /:** $`\mathbf{\alpha}`$**-banda**
Un rango de escalas donde el $`\alpha`$ estimado es aproximadamente constante y pasa colapso. Los límites se determinan por diagnósticos o puntos de quiebre visibles.

**IC de Bootstrap:**
Intervalo de confianza obtenido por remuestreo (con reemplazo) de los puntos pareados $`(L,T)`$ o residuos bajo el estimador elegido.

**Punto de cambio (punto de quiebre):**
Un valor de escala donde un modelo de pendiente única es significativamente superado por pendientes segmentadas. Etiquetar el resultado **MULTI-REGIME** y reportar cada segmento.

**Colapso (diagnósticos):**
Conjunto de pruebas usadas para aceptar una ventana: linealidad log–log (sin curvatura de residuos), estabilidad leave-one-out, heterocedasticidad acotada, sin puntos de cambio ocultos, extensión/granularidad suficiente.

**Longitud de correlación:**
Una escala espacial (ej., rango de variograma o inverso del pico de frecuencia espacial) que resume la extensión de la correlación; puede usarse para definir $`L`$.

**Errores en Variables (EIV):**
Marco de regresión que tiene en cuenta el error de medición en ambos $`\log L`$ y $`\log T`$. Requerido cuando ambos ejes son ruidosos.

**Divulgación del estimador:**
La declaración explícita que acompaña a $`\widehat{\alpha}`$: estimador (ODR/TLS, Theil–Sen, uso de SIMEX), método de IC, $`n`$, extensión (décadas), definición de ventana, y diagnósticos.

**Topología/métrica efectiva:**
La geometría que gobierna la propagación (ej., geodésicas de red, atajos de mundo pequeño), que puede diferir de la distancia euclidiana y cambiar el escalamiento esperado (ej., $`T \sim \log L`$).

**FWHM (ancho a media altura):**
Medida de dispersión espacial usada en mapas de nivel de fuente; puede definir longitud de onda $`\lambda`$ y por tanto $`L = \lambda/2`$.

**Granularidad (niveles):**
El número de niveles distintos de $`L`$ en una ventana. Objetivo ≥6 niveles para estabilizar estimaciones de pendiente y diagnósticos.

**Heterocedasticidad:**
Varianza de residuos dependiente de escala; debe permanecer dentro de límites pre-registrados para que el colapso pase.

**LOG-SCALING:**
Etiqueta de resultado cuando semi-log es lineal y $`T`$ crece aproximadamente como $`\log L`$. **No** reportar $`\alpha`$ para este régimen.

$`\mathbf{\beta}`$ **(pendiente espectral):**
Exponente de un espectro de potencia $`P(f) \propto f^{- \beta}`$. **Auxiliar** por defecto. **No hay** conversión universal de $`\beta`$ a $`\alpha`$.

**EXTENSIÓN MÍNIMA:**
Extensión objetivo de $`L`$ dentro de una ventana (≥0.6–1.0 décadas) para asegurar identificabilidad de la pendiente y diagnósticos confiables.

**MULTI-REGIME:**
Etiqueta de resultado cuando pendientes segmentadas son soportadas dentro de una ventana. Reportar valores de $`\widehat{\alpha}`$ por tramos, puntos de quiebre, e interpretación.

**NO_COLLAPSE:**
Etiqueta de resultado cuando los criterios de colapso fallan (curvatura, inestabilidad, heterocedasticidad excesiva, extensión insuficiente, o puntos de cambio ocultos). No reportar $`\alpha`$.

**ODR/TLS (regresión de distancia ortogonal / mínimos cuadrados totales):**
Estimador EIV por defecto que minimiza distancias ortogonales a la línea de regresión, teniendo en cuenta error en ambos ejes.

**Placebo (reloj):**
Control que reescala marcas de tiempo o agrega pequeña fluctuación mientras mantiene estructura; $`\widehat{\alpha}`$ debe permanecer dentro del IC si la pendiente refleja estructura, no artefactos de reloj.

**Placebo (estructura):**
Control que perturba patrones espaciales o emparejamientos mientras mantiene el reloj; el colapso debe fallar o $`\widehat{\alpha}`$ derivar si la estructura impulsa la pendiente.

**PLV / coherencia:**
Valor de enganche de fase y medidas de coherencia. Descriptores **auxiliares** de sincronía; no sustitutos de $`\alpha`$.

**Pre-registro (ventana):**
Declaración previa de definiciones de $`L`$ y $`T`$, límites de bin, inclusión/exclusión, estimador, diagnósticos, y criterios de fallo antes de ajustar $`\alpha`$.

**Curvatura de residuos (prueba de rachas/LOESS):**
Desviación de linealidad en log–log; evidencia contra un modelo de pendiente única dentro de la ventana.

**SIMEX:**
Procedimiento de simulación–extrapolación para corregir sesgo por error de medición cuando su varianza es estimable. Usado sobre ajustes ODR/TLS u otros EIV.

**Longitud de onda modelada en fuente** $`\mathbf{(\lambda)}`$ :
Período espacial de una fuente oscilatoria estimada en la corteza (ej., vía eLORETA/beamformer); $`L = \lambda/2`$.

**Extensión (décadas):**
El ancho de valores de $`L`$ en una ventana medido en órdenes de magnitud. Reportado junto con $`\widehat{\alpha}`$.

**Sustituto (espacio de sensores):**
Un proxy estructural no-fuente para $`L`$ (ej., período SVD topográfico). Permitido solo con advertencias explícitas y usualmente etiquetado como exploratorio.

**Theil–Sen:**
Estimador de pendiente robusto (mediana de pendientes por pares); usado como verificación de sensibilidad contra ODR/TLS o cuando hay valores atípicos.

**Tiempo** $`\mathbf{T}`$ **(duración propia):**
Duración del mecanismo gobernado por la estructura que define $`L`$ (ej., período de oscilación $`1/f`$, tiempo de tránsito, tiempo de asentamiento, retardo de consolidación).

**Variograma (rango):**
Estadística espacial usada para estimar longitud de correlación. El rango (donde el variograma se estabiliza) puede definir $`L`$.

**Ventana (bin):**
Rango finito de $`L`$ (y mecanismo/estado fijos) sobre el cual se prueba un modelo de pendiente única. Debe pasar diagnósticos de colapso para reportar $`\alpha`$.

**Regla de uso (global).**\
Solo valores de $`\widehat{\alpha}`$ derivados de pendientes log–log **multipunto** dentro de ventanas **válidas de colapso** aparecen en tablas de resultados. Todas las demás métricas son **auxiliares** y se reportan separadamente con sus propios métodos e incertidumbre.
  
**Apéndice D · Resultados Negativos y Modos de Fallo (Solo Técnico)**

**Alcance.** Cómo detectar, etiquetar, y reportar cuando un régimen de ley de potencia **no** se sostiene, y cómo aprender de ese resultado sin forzar $`\alpha`$.

**D.1 Etiquetas de Resultado (fuente única de verdad)**

- **NO_COLLAPSE** — El ajuste log–log falla los diagnósticos de colapso en la ventana pre-registrada (curvatura, inestabilidad, heterocedasticidad, o extensión insuficiente).

- **LOG-SCALING** — Semi-log es lineal: $`T \sim a + b\log L`$. Reportar $`b`$ e IC; **no** inferir $`\alpha`$.

- **MULTI-REGIME** — Lineal por tramos en log–log con ≥1 punto(s) de quiebre significativo(s). Reportar $`\widehat{\alpha}`$ segmentado, puntos de quiebre, ICs.

- **AUXILIARY-ONLY** — Sin pares $`(L,T)`$ válidos o colapso; reportar métricas auxiliares (ej., $`\beta`$, PLV) **explícitamente como auxiliar**, con métodos e ICs.

**Regla:** Estas etiquetas son **resultados finales**, no marcadores de posición. Aparecen en tablas, figuras, y el manifiesto del repositorio.

**D.2 Diagnósticos de Colapso (recapitulación, estricto)**

Una ventana se **acepta** solo si **todos** pasan:

1.  **Linealidad (log–log):** rachas de residuos/LOESS no muestran curvatura.

2.  **Estabilidad:** deriva de pendiente leave-one-level-out ≤ tolerancia pre-registrada.

3.  **Heterocedasticidad:** dentro de límites (ej., *p* de Breusch–Pagan > umbral o verificación estratificada por varianza).

4.  **Puntos de cambio:** sin puntos de quiebre ocultos que superen una pendiente única a menos que se etiquete **MULTI-REGIME**.

5.  **Extensión y granularidad:** objetivo ≥0.6–1.0 décadas en $`L`$ con ≥6 niveles distintos.

Fallar cualquiera → **NO_COLLAPSE** (a menos que semi-log lineal ⇒ **LOG-SCALING**).

**D.3 Árbol de Decisión (operacional)**

1.  Ajustar EIV en $`(\log L,\log T)`$ (ODR/TLS; sensibilidad Theil–Sen).

2.  Ejecutar diagnósticos (D.2).

    - **Pasan todos** → reportar $`\widehat{\alpha}`$ con IC.

    - **Curvatura / inestabilidad / hetero / muy pocos niveles** → **NO_COLLAPSE**.

    - **Mejor modelo es semi-log** → **LOG-SCALING** (reportar $`b`$).

    - **Punto(s) de quiebre claro(s)** → **MULTI-REGIME** (reportar segmentos).

3.  Si **sin** $`(L,T)`$ válidos → **AUXILIARY-ONLY** (sin afirmación de $`\alpha`$).

**D.4 Plantillas de Reporte**

**D.4.1 NO_COLLAPSE (plantilla de leyenda de figura)**

*"Pares tiempo–tamaño para \[sistema\] dentro de la ventana pre-registrada. Ajuste log–log falla colapso (curvatura de residuos / pendiente inestable / heterocedasticidad). Resultado: **NO_COLLAPSE**. Reportamos descriptores auxiliares separadamente (no* $`\alpha`$*)."*

**Campos de fila de tabla:** sistema \| especificación de ventana \| $`n`$ \| extensión (décadas) \| estimador \| diagnósticos (razón de fallo) \| resultado = NO_COLLAPSE.

**D.4.2 LOG-SCALING (plantilla de leyenda de figura)**

*"La temporización sigue una ley logarítmica en \[sistema\]:* $`T \sim a + b\ \log L`$ *(semi-log lineal, IC 95% para* $`b`$*). Ajuste de ley de potencia rechazado por colapso. Resultado: **LOG-SCALING** (no-potencia)."*

**Campos de fila de tabla:** sistema \| ventana \| $`n`$ \| pendiente semi-log $`b`$ (IC) \| prueba vs. ley de potencia \| resultado = LOG-SCALING.

**D.4.3 MULTI-REGIME (plantilla de leyenda de figura)**

"La temporización sigue una ley logarítmica en [sistema]: $T \sim a + b \log L$ (semi-log lineal, IC 95% para $b$). Ajuste de ley de potencia rechazado por colapso. Resultado: **LOG-SCALING** (no-potencia)."

**Campos de fila de tabla:** sistema | ventana | $n$ | pendiente semi-log $b$ (IC) | prueba vs. ley de potencia | resultado = LOG-SCALING.

**D.4.4 AUXILIARY-ONLY (plantilla de leyenda de figura)**

*"Sin pares tiempo–tamaño válidos para \[sistema\]. Reportamos métricas auxiliares (ej.,* $`\beta`$ *espectral, PLV/coherencia) con métodos e ICs. Resultado: **AUXILIARY-ONLY** (sin* $`\alpha`$ *afirmado)."*

**Campos de fila de tabla:** sistema \| razón (sin pares / $`L`$ faltante / $`T`$ faltante) \| auxiliares reportados (con métodos) \| resultado = AUXILIARY-ONLY.

**D.5 Guía de Solución de Problemas (de síntoma a acción)**

| **Síntoma** | **Causa probable** | **Acción** |
|----|----|----|
| Curvatura de residuos (hacia arriba) | mezcla de mecanismos; punto de cambio faltante | ajustar ventana; probar pendientes segmentadas; si significativo → MULTI-REGIME; sino NO_COLLAPSE |
| Curvatura de residuos (hacia abajo) | métrica incorrecta para $`L`$; saturación | redefinir $`L`$ (nivel de fuente, geodésica); re-binear |
| Gran deriva de pendiente LOOCV | muy pocos niveles; apalancamiento de valor atípico | aumentar niveles; estimador robusto; re-medir |
| Heterocedasticidad explota | error de medición crece con $`L`$ | modelar error; SIMEX; estrechar extensión; si persiste → NO_COLLAPSE |
| Buen ajuste semi-log, mal log–log | topología de atajo | etiquetar LOG-SCALING; no forzar $`\alpha`$ |
| Segmentos limpios, peor ajuste único | transición de régimen | MULTI-REGIME con IC de punto de quiebre e interpretación |
| Sin $`T`$ o $`L`$ utilizables | brecha de adquisición | re-recolectar o degradar a AUXILIARY-ONLY |

**D.6 Placebos y Controles (hacer los negativos significativos)**

- **Placebo de reloj:** reescalar marcas de tiempo o agregar pequeña fluctuación; si $`\widehat{\alpha}`$ o $`b`$ cambia más allá del IC, se sospecha artefacto de temporización.

- **Placebo de estructura:** barajar estructura espacial / etiquetas de ROI mientras se mantiene el reloj; la pendiente debe desaparecer o derivar si la estructura impulsa la temporización.

- **Corrección de motor/reporte (comportamiento):** restar retardos constantes para evitar artefactos de intercepto haciéndose pasar por pendiente.

- **Estratificación de estado (neuro/fisio):** re-binear por arousal/tarea; agrupar estados a menudo causa NO_COLLAPSE.

**D.7 Repositorio y Manifiesto (negativos auditables)**

- **Carpetas:**\
  /proc/pairs/ $`(L,T)`$ con banderas; /qc/ reportes de colapso; /figures/negatives/ paneles; /features/aux/ métricas auxiliares.

- **Campos del manifiesto (por análisis):** sistema, window_spec, niveles, span_decades, estimador, diagnósticos, outcome_label, ic, puntos de quiebre, notas.

- **Reconstrucción:** script de un comando reproduce **tanto** positivos como negativos.

**D.8 Ética y Comunicación**

- **Equivalencia de valor:** Un **NO_COLLAPSE** limpio es tan informativo como una pendiente positiva—mapea el límite del mecanismo.

- **Sin eufemismos:** No renombrar **LOG-SCALING** como "escalamiento débil." Declararlo claramente y modelarlo apropiadamente.

- **Sin alquimia de proxy:** Un resultado negativo de $`\alpha`$ **no** es licencia para convertir auxiliares (ej., $`\beta`$) en $`\alpha`$. Mantener carriles claros.

**D.9 Mini-Ejemplos Trabajados (abstraídos)**

1.  **Tránsito vascular (curvatura):** Curvatura log–log detectada; pendientes segmentadas pasan por fragmento. **Resultado:** MULTI-REGIME con IC de $`L^{*}`$; nota de mecanismo: transición capilar → venoso.

2.  **Acceso EEG (atajo):** Semi-log lineal; $`b = 0.34`$ (IC). Ajuste de potencia falla. **Resultado:** LOG-SCALING. Nota: enrutamiento dominado por hub.

3.  **Recuerdo conductual (pendiente inestable):** Deriva LOOCV excede tolerancia; extensión < 0.4 décadas. **Resultado:** NO_COLLAPSE; plan: aumentar niveles, fijar política de ensayo.

4.  **Conjunto de datos solo VFC:** Sin pares $`T,L`$; $`\beta`$ de PSD auxiliar y PLV reportados con ICs. **Resultado:** AUXILIARY-ONLY.

**D.10 FAQ (respuestas rápidas de modo de fallo)**

- **P:** ¿Puedo promediar dos regímenes para reportar un $`\alpha`$?\
  **R:** No. Usar **MULTI-REGIME** con puntos de quiebre.

- **P:** Semi-log funciona; ¿puedo aún citar $`\alpha`$?\
  **R:** No. Etiquetar **LOG-SCALING** y reportar $`b`$.

- **P:** Colapso falla pero los auxiliares se ven geniales—¿puedo inferir $`\alpha`$ de $`\beta`$?\
  **R:** No. Los auxiliares permanecen auxiliares; sin conversión universal.

- **P:** Mi extensión es pequeña (0.3 décadas) pero el ajuste se ve recto—¿OK reportar?\
  **R:** Solo exploratorio; expandir extensión o marcar **NO_COLLAPSE** (granularidad insuficiente).

**Resumen del Apéndice D.**\
Los resultados negativos son **evidencia**: localizan límites, revelan métricas incorrectas, exponen mezcla de regímenes, y señalan gramáticas alternativas (ej., $`T \sim \log L`$). Tratar **NO_COLLAPSE**, **LOG-SCALING**, **MULTI-REGIME**, y **AUXILIARY-ONLY** como resultados publicables de primera clase con el mismo cuidado—diagnósticos, leyendas, manifiestos—que hallazgos positivos de $`\alpha`$.

*© 2026 Álvaro José Quiceno Rendón. Este documento se distribuye bajo una licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).*
