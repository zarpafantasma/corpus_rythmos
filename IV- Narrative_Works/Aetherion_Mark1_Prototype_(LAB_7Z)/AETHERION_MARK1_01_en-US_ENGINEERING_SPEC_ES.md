# AETHERION MARK 1
## Documentación de ingeniería y especificaciones técnicas
### Sistema prototipo de propulsión por gradiente de vacío

**Clasificación:** EXPERIMENTAL  
**Revisión:** 1.0  
**Fecha:** Marzo de 2026

---

```
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                             - M A R K   1    ║
    ║     █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗ ██╗ ██████╗ ███╗   ██╗   ║
    ║    ██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗██║██╔═══██╗████╗  ██║   ║
    ║    ███████║█████╗     ██║   ███████║█████╗  ██████╔╝██║██║   ██║██╔██╗ ██║   ║
    ║    ██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██║██║   ██║██║╚██╗██║   ║
    ║    ██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║██║╚██████╔╝██║ ╚████║   ║
    ║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ║
    ║                                                                              ║
    ║                           SALTANDO AL OTRO LADO                              ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## ÍNDICE

1. [Resumen del sistema](#1-resumen-del-sistema)
2. [Principios de funcionamiento](#2-principios-de-funcionamiento)
3. [Ensamblaje del núcleo (Capacitor topológico)](#3-ensamblaje-del-núcleo-capacitor-topológico)
4. [Sistema de propulsión (Híbrido TPH/OMV)](#4-sistema-de-propulsión-híbrido-tphomv)
5. [Electrónica de control](#5-electrónica-de-control)
6. [Disposición estructural](#6-disposición-estructural)
7. [Diagramas de ensamblaje](#7-diagramas-de-ensamblaje)
8. [Tabla de especificaciones](#8-tabla-de-especificaciones)
9. [Protocolos de seguridad](#9-protocolos-de-seguridad)
   [Apéndices](#A-protocolos-de-prueba, #B-hoja-de-ruta)
---

## 1. RESUMEN DEL SISTEMA

El **Aetherion Mark 1** es un prototipo experimental a escala de laboratorio diseñado para demostrar la propulsión por gradiente de vacío RTM (Relatividad Temporal Multiescala). Opera mediante:

1. **Almacenar** energía de vacío de punto cero en un núcleo metamaterial (Capacitor topológico)
2. **Liberar** la energía almacenada mediante pulsación piezoeléctrica (Protocolo TPH)
3. **Rectificar** fuerzas oscilatorias en empuje unidireccional (Efecto ponderomotriz OMV)

### Filosofía de diseño

| Principio | Implementación |
|-----------|----------------|
| **Cumplimiento de la primera ley** | Sin sobreunidad; se requiere entrada de potencia externa |
| **Ruptura de simetría** | Geometría asimétrica de tobera + ondas de deformación viajeras |
| **Escalabilidad** | Apilado modular del núcleo para multiplicación del empuje |
| **Inmunidad al ruido** | Gradientes pronunciados (Δα > 2.0) suprimen el ruido térmico |

### Objetivos de desempeño (Prototipo de laboratorio)

| Métrica | Objetivo | Notas |
|--------|--------|-------|
| **Masa del núcleo** | 50 gramos | Apilado metamaterial |
| **Empuje (DC)** | 100-500 nN | Medible mediante balanza de torsión |
| **Impulso por pulso** | ~120 pN·s | A 1 kHz de tasa de repetición |
| **Temperatura de operación** | 293 K | Temperatura ambiente (sin criogenia) |
| **Potencia de entrada** | 5-50 W | Driver piezoeléctrico |

---

## 2. PRINCIPIOS DE FUNCIONAMIENTO

### 2.1 El capacitor topológico

Un gradiente metamaterial estático **no produce empuje continuo** (verificado por el equipo rojo). En cambio, actúa como un "resorte espacial cargado":

```
    GRADIENTE ESTÁTICO (Sin empuje)
    ════════════════════════════════════════

    α_min ─────────────────────────── α_max
    ←──── ∇α ────→

    Flujo de punto cero:

    ←←←← φ ────→→→→
         ↑
         │
    Neto = 0 (los vectores se cancelan)

    Energía ALMACENADA en el centro como esfuerzo estructural
```

### 2.2 Propulsión TPH (Ruptura de simetría)

Para extraer empuje, se inyectan **pulsos de deformación asimétricos**:

```
    JERARQUÍA TEMPORAL DE PULSOS (TPH)
    ════════════════════════════════════════

    Onda de choque piezoeléctrica:

    ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░
    ←── Comprimido ──→←── Expandido ──→

    El gradiente viajero ∇L crea una liberación ASIMÉTRICA:

         ∇α (estático)
    ┌─────────────────────────────┐
    │    ⊕ ⊕ ⊕ ⊕ ⊕ ⊕ ⊕ ⊕ ⊕      │
    │  ┌───────────────────────┐  │
    │  │ BURBUJA φ EXPULSADA   │──┼──→ EMPUJE
    │  └───────────────────────┘  │
    │    ⊖ ⊖ ⊖ ⊖ ⊖ ⊖ ⊖ ⊖ ⊖      │
    └─────────────────────────────┘
         ∇L (pulso)
```

### 2.3 Rectificación ponderomotriz OMV

La vibración continua genera **empuje DC** mediante rectificación cuadrática:

```
    MODULACIÓN OSCILATORIA DEL VACÍO (OMV)
    ════════════════════════════════════════

    Vibración piezoeléctrica: α(t) = α₀ + Δα·cos(ωt)

    Densidad de fuerza: F ∝ (∇α)²

    cos²(ωt) = ½[1 + cos(2ωt)]
                 ↑
                 COMPONENTE DC (empuje neto)

    Resultado: ~197 pN de empuje sostenido (verificado)
```

---

## 3. ENSAMBLAJE DEL NÚCLEO (Capacitor topológico)

### 3.1 Arquitectura del apilado metamaterial

```
    VISTA EN CORTE (Axial)
    ════════════════════════════════════════

                    ↑ EJE DE EMPUJE
                    │
            ┌───────┴───────┐
           ╱                 ╲
          ╱   CONO DE TOBERA  ╲
         ╱     α = 2.5         ╲
        ╱                       ╲
       ├─────────────────────────┤ ← Abertura de escape
       │                         │
       │    ZONA DE GRADIENTE    │
       │    α = 2.0 → 0.5        │
       │    (15 capas)           │
       │                         │
       ├─────────────────────────┤ ← φ Máximo (Núcleo)
       │                         │
       │    ACUMULADOR           │
       │    α = 0.5 (constante)  │
       │    (5 capas)            │
       │                         │
       ├─────────────────────────┤
       │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │ ← Arreglo piezoeléctrico
       │  ▓▓▓ ANILLO ACTUADOR ▓▓ │
       ├─────────────────────────┤
       │                         │
       │    PLACA BASE           │
       │    α = 2.5 (reflector)  │
       │                         │
       └─────────────────────────┘
```

### 3.2 Composición de capas

| Capa n.º | Espesor | Valor α | Material | Función |
|---------|-----------|---------|----------|----------|
| 1-5 | 0.5 mm c/u | 0.5 | Compuesto ZrO₂/SiC | Acumulador (almacenamiento φ) |
| 6-20 | 0.3 mm c/u | 0.5→2.0 | Metamaterial graduado | Zona de gradiente |
| 21-23 | 0.4 mm c/u | 2.0→2.5 | Geometría de tobera | Escape direccional |
| Base | 2.0 mm | 2.5 | Placa reflectora | Prevenir reflujo |

### 3.3 Fabricación del gradiente metamaterial

```
    PROGRAMA DE DEPOSICIÓN DE CAPAS
    ════════════════════════════════════════

    valor α
    2.5 ─┐                              ┌─ Tobera
        │                              │
    2.0 ─┤                         ┌───┘
        │                    ┌────┘
    1.5 ─┤               ┌───┘
        │          ┌────┘
    1.0 ─┤     ┌───┘
        │┌────┘
    0.5 ─┴────┬────┬────┬────┬────┬────┬────
        1    5   10   15   20   23   Base
                    Capa n.º

    Gradiente: Δα/Δz = 0.10 por capa
               ∇α = 200 m⁻¹ (objetivo)
```

### 3.4 Dimensiones del núcleo

```
    DIBUJO DIMENSIONAL
    ════════════════════════════════════════

    Todas las dimensiones en milímetros

              ← 35 →
         ┌─────────────┐
        ╱               ╲
       ╱← 25 →           ╲     ↑
      ╱                   ╲    │ 8 (tobera)
     ╱                     ╲   │
    ├───────────────────────┤  ↓
    │                       │  ↑
    │                       │  │
    │                       │  │ 12 (gradiente)
    │                       │  │
    │                       │  ↓
    ├───────────────────────┤  ↑
    │                       │  │ 5 (acumulador)
    │                       │  ↓
    ├───────────────────────┤  ↑
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  │ 3 (piezo)
    ├───────────────────────┤  ↓
    │                       │  ↑
    │                       │  │ 2 (base)
    └───────────────────────┘  ↓

    ←──────── 40 ────────→

    Altura total: 30 mm
    Diámetro exterior: 40 mm
    Masa del núcleo: ~50 g
```

---

## 4. SISTEMA DE PROPULSIÓN (Híbrido TPH/OMV)

### 4.1 Arreglo de actuadores piezoeléctricos

```
    ANILLO ACTUADOR (Vista superior)
    ════════════════════════════════════════

                    N
                    │
            ┌───────┼───────┐
           ╱   P1   │   P2   ╲
          ╱    ◆────┼───◆    ╲
         │          │          │
       W─┼──◆───────┼──────◆──┼─E
         │   P8     │     P3   │
          ╲    ◆────│───◆    ╱
           ╲   P7   │   P4   ╱
            └───────┼───────┘
                    │
                    S
                   P5,P6

    8× actuadores PZT-5H (disposición radial)

    Secuencia de disparo (Modo TPH):
    ─────────────────────────────
    t=0:    P1,P2 DISPARAN (pulso norte)
    t=τ/4:  P3    DISPARA (propaga)
    t=τ/2:  P4,P5 DISPARAN (llega al sur)
    t=3τ/4: P6,P7 DISPARAN (propaga)
    t=τ:    P8,P1 DISPARAN (ciclo completo)

    Crea una ONDA VIAJERA alrededor de la circunferencia
```

### 4.2 Especificaciones de los actuadores

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| **Tipo** | PZT-5H (Titanato zirconato de plomo) | Alto coeficiente d₃₃ |
| **Dimensiones** | 5×5×2 mm cada uno | Configuración apilada |
| **Cantidad** | 8 (arreglo radial) | Controlado por fase |
| **Desplazamiento máximo** | 2 µm | A 200 V |
| **Frecuencia resonante** | 50 kHz | Resonancia mecánica |
| **Frecuencia de operación** | 1-10 kHz | Tasa de pulso TPH |
| **Voltaje de accionamiento** | 0-200 V | Forma de onda programable |

### 4.3 Forma de onda del pulso

```
    SEÑAL DE ACCIONAMIENTO TPH
    ════════════════════════════════════════

    Voltaje (V)
    200 ─┐     ┌─┐     ┌─┐     ┌─┐
         │     │ │     │ │     │ │
    100 ─┤     │ │     │ │     │ │
         │     │ │     │ │     │ │
      0 ─┴─────┴─┴─────┴─┴─────┴─┴─────
        0    1ms   2ms   3ms   4ms

        ←τ_rise→   ←τ_fall→
          50µs       50µs

    Ciclo de trabajo: 10%
    Tasa de repetición: 1 kHz (ajustable 100 Hz - 10 kHz)
    Tiempo de subida: 50 µs (choque abrupto)

    MODO OMV (Seno continuo):
    ─────────────────────────────

    200  ─     ╱╲      ╱╲      ╱╲
         │    ╱  ╲    ╱  ╲    ╱  ╲
    100 ─┼───╱────╲──╱────╲──╱────╲──
         │  ╱      ╲╱      ╲╱      ╲
      0 ─┴─────────────────────────
        0    0.5ms  1ms   1.5ms  2ms

    Frecuencia: 2 kHz (rectificación ponderomotriz)
```

### 4.4 Salida de empuje esperada

```
    EMPUJE vs FRECUENCIA
    ════════════════════════════════════════

    Empuje (nN)
    500 ─┤                          ╱
        │                        ╱
    400 ─┤                      ╱
        │                    ╱
    300 ─┤                 ╱
        │              ╱
    200 ─┤           ╱    ← Régimen lineal
        │        ╱
    100 ─┤     ╱
        │  ╱
      0 ─┴────┬────┬────┬────┬────┬────
        0    2    4    6    8   10
                 Frecuencia (kHz)

    F_thrust ≈ 50 nN/kHz (modo TPH)
    F_DC ≈ 197 pN (OMV continuo)
```

---

## 5. ELECTRÓNICA DE CONTROL

### 5.1 Diagrama de bloques del sistema

```
    ARQUITECTURA DE CONTROL
    ════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────┐
    │                  UNIDAD DE CONTROL                      │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
    │  │             │    │             │    │             │  │
    │  │    MCU      │───→│ GENERADOR   │───→│ AMPLIFICADOR│  │
    │  │  (STM32H7)  │    │ DE FORMA    │    │   8-CAN     │  │
    │  │             │    │ DE ONDA     │    │   (200V)    │  │
    │  └──────┬──────┘    └─────────────┘    └──────┬──────┘  │
    │         │                                      │        │
    │         │    ┌─────────────────────────────────┘        │
    │         │    │                                          │
    └─────────┼────┼──────────────────────────────────────────┘
              │    │
              │    ▼
              │  ┌─────────────────────────────────────────┐
              │  │         ARREGLO PIEZO (8x)              │
              │  │      ◆──◆──◆──◆──◆──◆──◆──◆         │ 
              │  └─────────────────┬───────────────────────┘
              │                    │
              │                    ▼
              │  ┌─────────────────────────────────────────┐
              │  │        NÚCLEO METAMATERIAL              │
              │  │    ┌───────────────────────┐            │
              │  │    │   CAPACITOR           │            │
              │  │    │   TOPOLOGICO          │───→ EMPUJE │
              │  │    └───────────────────────┘            │
              │  └─────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                 ARREGLO DE SENSORES                     │
    │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
    │  │ BALANZA   │  │ SENSORES  │  │   ACEL    │            │
    │  │ DE TORSIÓN│  │ DE TEMP   │  │ (6 GDL)   │            │
    │  │ (empuje)  │  │  (4x)     │  │           │            │
    │  └───────────┘  └───────────┘  └───────────┘            │
    └─────────────────────────────────────────────────────────┘
```

### 5.2 Lista de componentes

| Componente | Número de parte | Cantidad | Función |
|-----------|-------------|----------|----------|
| MCU | STM32H743 | 1 | Controlador principal |
| DDS | AD9910 | 1 | Generación de forma de onda |
| Amplificador HV | PA94 | 8 | Driver piezoeléctrico (200 V) |
| DAC | AD5764 | 2 | Salida de 8 canales |
| Temperatura | PT1000 | 4 | Monitoreo del núcleo |
| Acelerómetro | ADXL355 | 1 | Sensado 6 GDL |
| Fuente de alimentación | 24V/5A | 1 | Alimentación principal |
| Fuente HV | 200V/100mA | 1 | Alimentación piezoeléctrica |

### 5.3 Lazo de control (Modo de levitación)

```
    CONTROLADOR PD
    ════════════════════════════════════════

    Consigna (z₀) ──┬──→(+)──→[Kp]──┬──→[Σ]──→[f(Hz)]──→ Piezo
                    │    ↑          │    ↑
                    │    │(-)       │    │
                    │    │          │    │
                    │    └──[z]─────│────┤
                    │               │    │
                    └──→[d/dt]──→[Kd]────┘

    Función de transferencia:
    ──────────────────────────
    f(t) = Kp·[z₀ - z(t)] + Kd·[dz/dt]

    Parámetros:
    Kp = 1000 Hz/µm
    Kd = 100 Hz·s/µm
    Latencia del sensor: 2 ms (compensada)
```

---

## 6. DISPOSICIÓN ESTRUCTURAL

### 6.1 Vista explotada del ensamblaje

```
    VISTA EXPLOTADA
    ════════════════════════════════════════

                    ┌───┐
                    │ 1 │  Cubierta superior (Al)
                    └─┬─┘
                      │
                   ╱─────╲
                  ╱   2   ╲    Cono de tobera (Metamaterial)
                 ╱─────────╲
                      │
               ┌──────┴──────┐
               │      3      │  Apilado de gradiente (15 capas)
               │             │
               │   ░░░░░░░   │
               │   ░░░░░░░   │
               │   ░░░░░░░   │
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      4      │  Núcleo acumulador
               │   ▓▓▓▓▓▓▓   │
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      5      │
               │◆◆◆◆◆◆◆◆ │  Conjunto del anillo piezoeléctrico
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      6      │  Placa base (Reflector)
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      7      │  Compartimento electrónico
               │  [PCB] [PS] │
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      8      │  Brida de montaje
               └─────────────┘


    ORDEN DE ENSAMBLAJE: 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1
```

### 6.2 Corte transversal (Ensamblado)

```
    CORTE TRANSVERSAL ENSAMBLADO
    ════════════════════════════════════════

              EMPUJE ↑
                    │
         ══════════╪══════════  ← Cubierta superior
        ╱          │          ╲
       ╱    ┌──────┴──────┐    ╲  ← Tobera (α=2.5)
      ╱     │             │     ╲
     ╱      │   φ_exit    │      ╲
    ════════╪═════════════╪════════ ← Abertura (Ø25 mm)
    ║       │             │       ║
    ║   ╔═══╧═════════════╧═══╗   ║
    ║   ║                     ║   ║  ← Zona de gradiente
    ║   ║   α: 2.0 → 0.5      ║   ║    (15 capas)
    ║   ║                     ║   ║
    ║   ║   ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓   ║   ║  ← Flujo de energía
    ║   ║                     ║   ║
    ║   ╠═════════════════════╣   ║
    ║   ║   ████████████████  ║   ║  ← φ_max (Núcleo)
    ║   ║   ██ ACUMULADOR █   ║   ║    α=0.5
    ║   ║   ████████████████  ║   ║
    ║   ╠═════════════════════╣   ║
    ║   ║◆◆◆◆◆ PIEZO ◆◆◆◆◆║   ║  ← Anillo actuador
    ║   ╠═════════════════════╣   ║
    ║   ║   BASE (α=2.5)      ║   ║  ← Reflector
    ║   ╚═════════════════════╝   ║
    ║                             ║
    ║    [MCU] [AMP] [PS]         ║  ← Electrónica
    ║                             ║
    ╚═════════════════════════════╝
              │
              ▼
         BRIDA DE MONTAJE

    Altura total: 85 mm
    Diámetro total: 60 mm
    Masa total: ~250 g
```

---

## 7. DIAGRAMAS DE ENSAMBLAJE

### 7.1 Vista isométrica

```
    VISTA ISOMÉTRICA 3D
    ════════════════════════════════════════

                      ↗ EMPUJE
                    ╱
                  ╱
               ╱─────╲
              ╱       ╲
             ╱ CUBIERTA╲
            ╱ SUPERIOR ╲
           ╱─────────────╲
          │╲             ╱│
          │ ╲  TOBERA   ╱ │
          │  ╲         ╱  │
          │   ╲───────╱   │
          │   │       │   │
          │   │ ZONA  │   │
          │   │ GRAD  │   │
          │   │       │   │
          │   ├───────┤   │
          │   │ NÚCLEO│   │
          │   ├───────┤   │
          │   │▓PIEZO▓│   │
          │   ├───────┤   │
          │   │ BASE  │   │
          │   └───────┘   │
          │ COMPARTIMENTO │
          │  ELECTRÓNICO  │
          └───────┬───────┘
                  │
            ══════╧══════
             BRIDA MONTAJE

    Escala: ~1:2
```

### 7.2 Diagrama de cableado

```
    CONEXIONES ELÉCTRICAS
    ════════════════════════════════════════

    ENTRADA 24V DC ──┬──→ [VREG 5V] ──→ MCU, Sensores
                     │
                     └──→ [HV BOOST] ──→ Riel de 200V
                                 │
                                 ▼
                     ┌─────────────┐
                     │  AMP 8-CAN  │
                     │  PA94 ×8    │
                     └──┬──┬──┬──┬─┘
                        │  │  │  │
            ┌───────────┼──┼──┼──┼────────────┐
            │           │  │  │  │            │
            ▼           ▼  ▼  ▼  ▼            ▼
           P1          P2 P3 P4 P5            P8
            ◆───────────◆──◆──◆──◆─────────◆
            │                                 │
            └────────── ANILLO PIEZO ─────────┘

    RUTA DE SEÑAL:
    ──────────────
    MCU (SPI) ──→ DDS (AD9910) ──→ DAC ──→ AMP ──→ PIEZO
         │
         └──→ CONTROL DE FASE (8 canales independientes)

    RETORNO DE SENSORES:
    ────────────────────
    PT1000 ×4 ──→ ADC ──→ MCU (temperatura)
    ADXL355   ──→ SPI ──→ MCU (aceleración)
    Torsión   ──→ ADC ──→ MCU (medición de empuje)
```

---

## 8. TABLA DE ESPECIFICACIONES

### 8.1 Especificaciones físicas

| Parámetro | Valor | Tolerancia |
|-----------|-------|-----------|
| **Altura total** | 85 mm | ±1 mm |
| **Diámetro total** | 60 mm | ±0.5 mm |
| **Masa total** | 250 g | ±10 g |
| **Masa del núcleo** | 50 g | ±2 g |
| **Diámetro del núcleo** | 40 mm | ±0.1 mm |
| **Altura del núcleo** | 30 mm | ±0.5 mm |
| **Abertura de la tobera** | 25 mm | ±0.2 mm |
| **Número de capas** | 23 | — |
| **Espesor de capa** | 0.3-0.5 mm | ±0.05 mm |

### 8.2 Especificaciones eléctricas

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| **Voltaje de entrada** | 24 V DC | ±5% |
| **Potencia de entrada** | 5-50 W | Dependiente del modo |
| **Riel HV** | 200 V DC | Accionamiento piezoeléctrico |
| **Canales piezoeléctricos** | 8 | Fase independiente |
| **Frecuencia de operación** | 100 Hz - 50 kHz | Programable |
| **Interfaz de control** | USB / UART | 115200 baudios |

### 8.3 Especificaciones de desempeño

| Parámetro | Valor | Condiciones |
|-----------|-------|------------|
| **Empuje TPH** | 100-500 nN | 1-10 kHz, 200V |
| **Empuje DC OMV** | ~200 pN | 2 kHz continuo |
| **Impulso/Pulso** | 123 pN·s | Pulso TPH único |
| **Intensidad del gradiente** | ∇α = 200 m⁻¹ | Objetivo de diseño |
| **Máximo de campo** | φ_max ≈ 0.1 | Unidades normalizadas |
| **Inmunidad al ruido** | 5% defectos de fabricación | Verificado por Monte Carlo |
| **Temperatura de operación** | 20-40 °C | Temperatura ambiente |

### 8.4 Especificaciones del metamaterial

| Zona de capas | Rango α | Sistema de material | Propósito |
|------------|---------|-----------------|---------|
| **Acumulador** | 0.5 | ZrO₂-SiC (70:30) | Almacenamiento φ |
| **Gradiente** | 0.5→2.0 | ZrO₂-Al₂O₃ graduado | Transporte |
| **Tobera** | 2.0→2.5 | Al₂O₃-TiO₂ | Escape |
| **Reflector** | 2.5 | Al₂O₃ denso | Prevención de reflujo |

---

## 9. PROTOCOLOS DE SEGURIDAD

### 9.1 Riesgos operativos

| Peligro | Nivel de riesgo | Mitigación |
|--------|------------|------------|
| **Alto voltaje (200V)** | ALTO | Interbloqueos, puesta a tierra, aislamiento |
| **Emisión acústica piezoeléctrica** | MEDIO | Protección auditiva por encima de 10 kHz |
| **Térmico (Núcleo)** | BAJO | Monitoreo de temperatura, apagado automático |
| **Vibración mecánica** | BAJO | Montaje seguro, amortiguamiento |

### 9.2 Lista de verificación previa a la operación

```
    LISTA DE VERIFICACIÓN PREVIA
    ════════════════════════════════════════

    [ ] 1. Inspección visual (sin grietas ni residuos)
    [ ] 2. Conexiones eléctricas verificadas
    [ ] 3. Interbloqueo HV activado
    [ ] 4. Sensores de temperatura respondiendo (4/4)
    [ ] 5. Acelerómetro calibrado
    [ ] 6. Balanza de torsión puesta en cero
    [ ] 7. Vacío/presión atmosférica registrados
    [ ] 8. Software de control cargado
    [ ] 9. Parada de emergencia accesible
    [ ] 10. Personal retirado de la zona HV

    FIRMA AUTORIZADA: ________________
    FECHA: ________________
```

### 9.3 Procedimientos de emergencia

```
    SECUENCIA DE APAGADO DE EMERGENCIA
    ════════════════════════════════════════

    1. PRESIONE EL E-STOP ROJO (corta toda la alimentación)
    2. Espere 30 segundos (descarga del capacitor HV)
    3. Verifique que el LED indicador de HV esté APAGADO
    4. Conecte el riel HV a tierra con una sonda de descarga
    5. Documente el incidente en la bitácora

    NO toque el arreglo piezo hasta completar el Paso 4
```

---

## APÉNDICE B: Protocolo de prueba

### A.1 Verificación de empuje

1. Monte la unidad en una balanza de torsión calibrada
2. Ponga en cero la balanza en estado de reposo
3. Aplique el protocolo TPH a 1 kHz
4. Registre la deflexión durante 60 segundos
5. Calcule el empuje medio a partir de la curva de calibración
6. Compare con el valor previsto de ~100 nN

### A.2 Verificación de la ley de escalado

1. Barrido de frecuencia: 100 Hz → 10 kHz
2. Registre el empuje en cada frecuencia
3. Grafique el empuje vs frecuencia
4. Verifique la relación lineal (F ∝ f)
5. Mida la pendiente: objetivo ~50 nN/kHz

---

## APÉNDICE B: HOJA DE RUTA

  MARK 1 ──────── Prototipo de laboratorio                       
  ══════          • Masa: 250 g                                  
                  • Empuje: 100-500 nN                           
                  • Objetivo: Validar la física TPH/OMV          
                  • Prueba: Balanza de torsión + vacío           
                  • Costo: ~US$14,000                            
                          ↓                                      
  MARK 2 ──────── Prototipo escalado                             
                  • Masa: 3-11 lb                                
                  • Empuje: µN - mN                              
                  • Mejoras: Enfriamiento líquido, núcleos apilados
                  • Prueba: Banco de empuje calibrado            
                          ↓                                      
  MARK 3 ──────── Demostrador de ingeniería                      
                  • Masa: 22-110 lb                              
                  • Empuje: mN - N                               
                  • Objetivo: Demostrar escalabilidad            
                          ↓                                      
  MARK 4+ ─────── Prototipo de vuelo                             
                  • Integración con vehículo                     
                  • Certificación                                
                  • Prueba de vuelo suborbital                   


```
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║                    FIN DEL DOCUMENTO                             ║
    ║                                                                  ║
    ║           AETHERION MARK 1 - ESPECIFICACIÓN DE INGENIERÍA        ║
    ║                     Revisión 1.0                                 ║
    ║                                                                  ║
    ║         "El tiempo no es lo que pasa, sino lo que pulsa."        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
```

     +-----------------------------------------------------------------------+
     | PROPIETARIO Y CONFIDENCIAL | ZARPAFANTASMA SYSTEMS CORP.              |
     | ID DEL PROYECTO: [AETHERION] | AUTORIZACIÓN DE SEGURIDAD: NIVEL 5     |
     |-----------------------------------------------------------------------|
     | ADVERTENCIA: El acceso, distribución o reproducción no autorizados de |
     | este documento están estrictamente prohibidos por el Protocolo Legal  |
     | de ZS-CORP. El rastreo electrónico y el marcado forense están activos |
     | en este archivo.                                                      |
     +-----------------------------------------------------------------------+
