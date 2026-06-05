# VI - Simulaciones Computacionales

Esta carpeta contiene **validaciones computacionales** del marco teórico RTM. Cada simulación prueba predicciones específicas de los Artículos 001–017, aportando evidencia reproducible de que las matemáticas producen resultados consistentes y físicamente significativos.

---

## ⚠️ Naturaleza de Esta Sección

Esta carpeta se enfoca exclusivamente en los **Artículos 001, 016 y 017** — los tres artículos cuyas validaciones son necesariamente *computacionales* en lugar de *empíricas*.

**¿Por qué estos tres?**

| Artículo | Tema | Por Qué Computacional |
|----------|------|-----------------------|
| **001** | Ley de escala central RTM (T ∝ L^α) | Prueba la emergencia matemática de α desde la topología de red — no se necesitan datos externos |
| **016** | Propulsión de vacío Aetherion | No existe aparato experimental para probar efectos de gradiente de vacío |
| **017** | Marco de Campo Unificado RTM | Formalismo QFT, holografía AdS/CFT — valida la consistencia teórica |

Los **Artículos 003–015** tienen sus validaciones en la **Carpeta VII (Validaciones Empíricas y Heurísticas)** porque esos dominios disponen de conjuntos de datos reales contra los cuales se pueden poner a prueba las predicciones RTM: latencias de la corteza visual, catálogos de ondas gravitacionales, HRV cardíaco, intensificación de huracanes, caídas de mercado, etc.

**La distinción crítica:**
- **Carpeta VI (aquí):** ¿Funcionan las matemáticas internamente? ¿Se preservan las leyes termodinámicas? ¿Convergen las soluciones numéricas?
- **Carpeta VII:** ¿Coinciden las matemáticas con la realidad? ¿Los exponentes de escala observados coinciden con las predicciones RTM?

Los Artículos 001, 016 y 017 solo pueden responder la primera pregunta. La validación empírica de la propulsión de vacío, los saltos de rama y los efectos del campo unificado queda pendiente para experimentos futuros que aún no existen.

---

## Estructura

Las simulaciones de cada artículo incluyen:
- **Scripts Python** (`.py`) — ejecución independiente
- **Notebooks Jupyter** (`.ipynb`) — exploración interactiva
- **Dockerfiles** — ejecución en contenedores reproducibles
- **Carpetas de salida** — datos CSV, figuras PNG/PDF, archivos de texto con resúmenes
- **Archivos README** — contexto teórico e interpretación de resultados
- **Auditorías Red Team** (cuando corresponda) — pruebas adversariales para detectar violaciones termodinámicas

---

## 001 - Relatividad Temporal Multiescala (RTM)

**Propósito:** Validar la ley de escala central T ∝ L^α en diferentes regímenes de transporte.

| Simulación | α Objetivo | Descripción | Resultado |
|------------|------------|-------------|-----------|
| `01_ballistic_1d` | 1.0 | Propagación a velocidad constante (cota inferior) | α = 1.0000 ± 0.0001 ✓ |
| `02_diffusive_1d` | 2.0 | Movimiento Browniano / caminata aleatoria | α = 2.0000 ± 0.0002 ✓ |
| `03_flat_small_world` | ~1.0 | Red de mundo pequeño Watts-Strogatz | Confirma topología plana |
| `04_sierpinski_fractal` | ~2.58 | Junta de Sierpiński (d_f = log3/log2) | α ≈ 2.58 ✓ |
| `05_vascular_tree` | ~2.3 | Red de ramificación con ley de Murray | Coincide con escala biológica |
| `06_hierarchical_small_world` | ~2.0 | Red modular jerárquica | α ≈ 2.0 ✓ |
| `07_holographic_decay` | 3.0 | P(r) ∝ r⁻³ conexiones de largo alcance | α = 2.95 ± 0.07, IC 95% incluye 3.0 ✓ |
| `08_quantum_confined` | ~3.5 | Fronteras de pared dura + confinamiento armónico | α = 3.52 ± 0.05 (prueba de concepto) |

**Hallazgo clave:** La ley de escala RTM recupera correctamente los regímenes de transporte conocidos (balístico α=1, difusivo α=2) y predice con éxito los regímenes intermedios/extremos determinados por la topología de red.

---

## 016 - Aetherion, El Saltador

**Propósito:** Validar la propulsión por gradiente de vacío y la física de los saltos de rama.

### Capítulo I: Capacitor Topológico (Almacenamiento de Energía)

| Simulación | Afirmación V1 | Hallazgo Red Team |
|------------|---------------|-------------------|
| `S1_1D_slab` | P ∝ (∇α)² extrae potencia | **Potencia neta = 0** — el gradiente estático almacena energía, no la extrae |
| `S2_2D_simulation` | El gradiente radial produce empuje | **Las fuerzas se cancelan geométricamente** — requiere pulsos asimétricos |
| `S3_scaling_analysis` | La potencia escala con el gradiente | **La tensión almacenada escala como Δα³** — los gradientes pronunciados suprimen el ruido térmico |

**Veredicto:** Los metamateriales estáticos actúan como "Capacitores Topológicos" — resortes espaciales cargados que almacenan energía del vacío pero requieren pulsos dinámicos para liberarla. **Primera Ley de la Termodinámica preservada.**

### Capítulo II: Propulsión y Dinámica

| Simulación | Afirmación V1 | Hallazgo Red Team |
|------------|---------------|-------------------|
| `S1_static_thrust` | Empuje continuo sin consumo | **Falacia de Bootstrap** — la fuerza estática es tensión interna, no momento |
| `S2_OMV_vibration` | La oscilación produce empuje | **La rectificación ponderomotriz funciona** — vibración → empuje DC ✓ |
| `S3_TPH_structural` | Los pulsos piezoeléctricos producen impulso | **~123 pN·s de impulso por pulso** mediante onda de choque asimétrica ✓ |
| `S4_levitation_hover` | Levitación estática posible | **Requiere control activo de pulso TPH** con bucle de retroalimentación PD |
| `S5_inertial_mitigation` | Escudo α=50 reduce 100g → 2g | **Funciona pero introduce sacudidas** — requiere amortiguadores mecánicos |

**Veredicto:** La propulsión requiere **ruptura dinámica de simetría** (oscilación OMV o pulsos TPH), no gradientes estáticos. **Tercera Ley de Newton preservada.**

### Capítulo III: Saltos de Rama (FTL)

| Simulación | Afirmación V1 | Hallazgo Red Team |
|------------|---------------|-------------------|
| `S1_multiwell_potential` | El potencial polinomial funciona | **Falló** — reemplazado por potencial Sine-Gordon ✓ |
| `S2_1D_branch_jump` | Salto controlado a Rama 1 | **Riesgo de avalancha** — requiere amortiguación topológica |
| `S3_3D_verification` | Los resultados 1D se extienden a 3D | **La tensión superficial domina** — requiere pulso supercrítico |
| `S4_jump_threshold` | Escala lineal con radio | **Aplica teoría de nucleación** — solo macroscópico (R > 1m) |
| `S5_grid_convergence` | Verificación de artefactos numéricos | **Convergido** — marco de EDP matemáticamente sólido ✓ |

**Veredicto:** El salto de rama es una **transición de fase macroscópica violenta**, no un truco matemático sin fricción. Requiere energía inmensa, amortiguación masiva y es estrictamente imposible a escalas microscópicas. **Restricciones de la Teoría Cuántica de Campos preservadas.**

---

## 017 - Marco de Campo Unificado RTM

**Propósito:** Validar el campo-α como campo cuántico dinámico con comportamiento QFT correcto.

### Sección 3.1.3: Correcciones Cuánticas

| Simulación | Prueba | Resultado |
|------------|--------|-----------|
| `S1_coleman_weinberg` | Potencial efectivo a un lazo | Los mínimos se desplazan Δα ≈ ±0.04 ✓ |
| `S2_quantum_bands` | Estructura de bandas bajo correcciones cuánticas | Las 5 bandas se desplazan con μ ✓ |
| `S3_rg_flow` | Funciones β para acoplamientos RTM | Todos los acoplamientos corren correctamente ✓ |
| `S4_two_loop` | Convergencia de la teoría de perturbaciones | |V₂| << |V₁| << |V_tree| ✓ |

### Sección 3.3: Holografía AdS/CFT

| Simulación | Prueba | Resultado |
|------------|--------|-----------|
| `S1_ads_alpha_profile` | Campo-α en el volumen AdS | Dependencia radial correcta ✓ |
| `S2_holographic_rg_flow` | Flujo RG de volumen a frontera | Las funciones β coinciden ✓ |
| `S3_boundary_correlators` | Funciones de dos puntos | Escala del VEV correcta ✓ |
| `S4_bh_thermodynamics` | Entropía de agujero negro vs α | Bekenstein-Hawking modificado ✓ |

### Sección 3.5: Unificación RG

| Simulación | Prueba | Resultado |
|------------|--------|-----------|
| `S1_gauge_rge_running` | Acoplamientos de gauge del ME con α | Corrida modificada ✓ |
| `S2_threshold_matching` | Efectos de umbral de banda-α | Catálogo generado ✓ |
| `S3_unification_fit` | Escala GUT con correcciones α | *Obsoleto* — ver Red Team |
| `S4_alpha_shift_effect` | Sensibilidad de la unificación a α | Validado con correcciones ✓ |

### Sección 4: Soluciones Numéricas de Campo

| Simulación | Prueba | Resultado |
|------------|--------|-----------|
| `S1_block_matrix_solver` | Solucionador de EDP acopladas | Soluciones 1D/2D verificadas ✓ |
| `S2_field_profiles_power` | Escala de potencia con gradiente | P ∝ (∇α)² confirmado ✓ |
| `S3_mesh_convergence` | Estabilidad numérica | Independiente de la malla ✓ |
| `S4_sierpinski_fractal` | Efectos de topología fractal | α coincide con d_f ✓ |
| `S5_vascular_tree` | Transporte en red biológica | Escala de Murray recuperada ✓ |

### Sección 6.3: Firmas Experimentales

| Simulación | Predice | Magnitud |
|------------|---------|----------|
| `S1_calorimetric_power` | Salida térmica en gradiente | P ∝ Δα² |
| `S2_rf_suppression` | Reducción de ruido EM en regiones-α | Corte dependiente de frecuencia |
| `S3_photon_delay` | Retardo de luz a través del gradiente | Δt ∝ Δα · L |
| `S4_multimodal_validation` | Detección de firma combinada | Protocolo multiinstrumento |

---

## Metodología Red Team

Varias simulaciones incluyen **auditorías Red Team** — pruebas adversariales diseñadas para detectar:

1. **Falacias de sobreunidad:** Afirmaciones de energía de la nada
2. **Falacias de Bootstrap:** Momento sin reacción
3. **Sesgo de confirmación:** Métricas que solo muestran resultados favorables
4. **Artefactos numéricos:** Resultados que dependen del tamaño de la malla

Las auditorías Red Team inyectan:
- Ruido térmico (5-15%)
- Defectos de fabricación (ruido espacial)
- Latencia de sensores (retardos de control realistas)
- Contabilidad termodinámica estricta

**Patrón de hallazgos:** Las simulaciones originales (V1) con frecuencia tenían matemáticas correctas pero interpretaciones físicas incorrectas. Las correcciones Red Team preservaron las matemáticas mientras corregían la física.

---

## Artículos 003–015: Validaciones Empíricas (Ver Carpeta VII)

Los Artículos 003–015 se validan contra **datos del mundo real** y por tanto residen en la **Carpeta VII (Validaciones Empíricas y Heurísticas)**:

| Artículo | Dominio | Fuente de Datos |
|----------|---------|-----------------|
| 003 | Corteza Visual | Tamaños de campo receptivo, latencias de respuesta |
| 004 | Cosmología | Galaxias de alto corrimiento al rojo del JWST |
| 005 | Ondas Gravitacionales | Catálogos LIGO/Virgo/KAGRA (O1-O4) |
| 006 | Computación Cuántica | Decoherencia en procesadores cuánticos de IBM |
| 007 | Química | Difusión en zeolitas, redes de transporte |
| 008 | Bioquímica | Cinética enzimática, plegamiento de proteínas |
| 009 | Homeostasis | Variabilidad de la frecuencia cardíaca (PhysioNet) |
| 010 | Neurociencia | Estados EEG (sueño, meditación, epilepsia) |
| 011 | Consciencia | Marcadores de profundidad de anestesia |
| 012 | Ecología/Epidemiología | Base de datos AnAge, propagación de COVID-19 |
| 013 | Meteorología | Huracanes IBTrACS, extremos climáticos |
| 014 | Astronomía | Galaxias SPARC, plasma de viento solar |
| 015 | Economía | Caídas de Bitcoin (datos de Binance) |

Estos artículos ponen a prueba las predicciones RTM contra verdad de campo externa. Los Artículos 001, 016 y 017 (esta carpeta) prueban la consistencia matemática interna porque aún no existe verdad de campo externa para sus predicciones.

---

## Reproducibilidad

Todas las simulaciones están diseñadas para ser reproducibles:

```bash
# Opción 1: Python directo
pip install -r requirements.txt
python nombre_simulacion.py

# Opción 2: Jupyter
jupyter notebook nombre_simulacion.ipynb

# Opción 3: Docker (recomendado)
docker build -t rtm-nombre-simulacion .
docker run --rm -v $(pwd)/output:/app/output rtm-nombre-simulacion
```

Las semillas aleatorias están fijadas. Todas las dependencias están ancladas. Los contenedores Docker garantizan entornos de ejecución idénticos.

---

## Interpretación de Resultados

**Lo que estas simulaciones prueban:**
- Las ecuaciones RTM son matemáticamente consistentes
- Las leyes de escala emergen de la topología según lo predicho
- Se satisfacen las restricciones termodinámicas
- El formalismo QFT está implementado correctamente

**Lo que estas simulaciones NO prueban:**
- Que los gradientes de vacío existan en el mundo real
- Que los metamateriales puedan crear gradientes-α
- Que el salto de rama sea físicamente posible
- Que el campo-α sea una entidad física real

Las simulaciones validan la *lógica interna* de la RTM. La validación empírica requiere experimentos que aún no existen.

---

## Citación

Si utilizas este trabajo, por favor cita:

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

---

## Licencia

© 2026 Álvaro José Quiceno Rendón  
Distribuido bajo [Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
Nota: **Utiliza el identificador DOI de Zenodo más reciente.**
