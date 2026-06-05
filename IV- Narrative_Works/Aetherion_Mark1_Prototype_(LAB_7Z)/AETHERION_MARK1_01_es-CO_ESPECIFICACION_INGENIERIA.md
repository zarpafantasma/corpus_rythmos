# AETHERION MARK 1
## Documentación de Ingeniería y Especificaciones Técnicas
### Prototipo de Sistema de Propulsión por Gradiente de Vacío

**Clasificación:** EXPERIMENTAL  
**Revisión:** 1.0  
**Fecha:** Marzo 2026

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

## TABLA DE CONTENIDOS

1. [Descripción General del Sistema](#1-descripcion-general-del-sistema)
2. [Principios de Operación](#2-principios-de-operacion)
3. [Ensamble del Núcleo (Capacitor Topológico)](#3-ensamble-del-nucleo)
4. [Sistema de Propulsión (Híbrido TPH/OMV)](#4-sistema-de-propulsion)
5. [Electrónica de Control](#5-electronica-de-control)
6. [Distribución Estructural](#6-distribucion-estructural)
7. [Diagramas de Ensamble](#7-diagramas-de-ensamble)
8. [Tabla de Especificaciones](#8-tabla-de-especificaciones)
9. [Protocolos de Seguridad](#9-protocolos-de-seguridad)
10. [Apéndices]  (#A-protocolos-de-prueba, #B-hoja-de-ruta)
   

---

## 1. DESCRIPCIÓN GENERAL DEL SISTEMA

El **Aetherion Mark 1** es un prototipo experimental a escala de laboratorio diseñado para demostrar la propulsión por gradiente de vacío RTM (Relatividad Temporal Multiescala). Opera mediante:

1. **Almacenamiento** de energía de punto cero del vacío en un núcleo de metamaterial (Capacitor Topológico)
2. **Liberación** de la energía almacenada mediante pulsación piezoeléctrica (Protocolo TPH)
3. **Rectificación** de fuerzas oscilatorias en empuje unidireccional (Efecto Ponderomotriz OMV)

### Filosofía de Diseño

| Principio | Implementación |
|-----------|----------------|
| **Cumplimiento de la Primera Ley** | Sin sobreunidad; se requiere entrada de energía externa |
| **Ruptura de Simetría** | Geometría de tobera asimétrica + ondas de deformación viajeras |
| **Escalabilidad** | Apilamiento modular de núcleos para multiplicación de empuje |
| **Inmunidad al Ruido** | Gradientes pronunciados (Δα > 2.0) suprimen el ruido térmico |

### Objetivos de Rendimiento (Prototipo de Laboratorio)

| Métrica | Objetivo | Notas |
|---------|----------|-------|
| **Masa del Núcleo** | 50 gramos | Pila de metamaterial |
| **Empuje (DC)** | 100–500 nN | Medible mediante balanza de torsión |
| **Impulso por pulso** | ~120 pN·s | A tasa de repetición de 1 kHz |
| **Temperatura de Operación** | 293 K | Temperatura ambiente (sin criogenia) |
| **Potencia de Entrada** | 5–50 W | Controlador piezoeléctrico |

---

## 2. PRINCIPIOS DE OPERACIÓN

### 2.1 El Capacitor Topológico

Un gradiente estático de metamaterial **no produce empuje continuo** (verificado por el Equipo Rojo). En cambio, actúa como un "resorte espacial cargado":

```
    GRADIENTE ESTÁTICO (Sin Empuje)
    ════════════════════════════════════════
    
    α_min ─────────────────────────── α_max
    ←──── ∇α ────→
    
    Flujo de Punto Cero:
    
    ←←←← φ ────→→→→
         ↑
         │
    Neto = 0 (Vectores se cancelan)
    
    Energía ALMACENADA en el centro como tensión estructural
```

### 2.2 Propulsión TPH (Ruptura de Simetría)

Para extraer empuje, se inyectan **pulsos de deformación asimétricos**:

```
    JERARQUÍA DE PULSOS TEMPORALES (TPH)
    ════════════════════════════════════════
    
    Onda de Choque Piezoeléctrica:
    
    ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░
    ←── Comprimido ──→←── Expandido ──→
    
    El gradiente viajero ∇L crea una liberación ASIMÉTRICA:
    
         ∇α (estático)
    ┌─────────────────────────────┐
    │    ⊕ ⊕ ⊕ ⊕ ⊕ ⊕ ⊕ ⊕ ⊕      │
    │  ┌───────────────────────┐  │
    │  │   BURBUJA φ EXPULSADA │──┼──→ EMPUJE
    │  └───────────────────────┘  │
    │    ⊖ ⊖ ⊖ ⊖ ⊖ ⊖ ⊖ ⊖ ⊖      │
    └─────────────────────────────┘
         ∇L (pulso)
```

### 2.3 Rectificación Ponderomotriz OMV

La vibración continua genera **empuje en DC** mediante rectificación cuadrática:

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

## 3. ENSAMBLE DEL NÚCLEO (Capacitor Topológico)

### 3.1 Arquitectura de la Pila de Metamaterial

```
    VISTA EN CORTE TRANSVERSAL (Axial)
    ════════════════════════════════════════
    
                    ↑ EJE DE EMPUJE
                    │
            ┌───────┴───────┐
           ╱                 ╲
          ╱   CONO DE TOBERA  ╲
         ╱     α = 2.5         ╲
        ╱                       ╲
       ├─────────────────────────┤ ← Apertura de Escape
       │                         │
       │    ZONA DE GRADIENTE    │
       │    α = 2.0 → 0.5        │
       │    (15 capas)           │
       │                         │
       ├─────────────────────────┤ ← Máximo φ (Núcleo)
       │                         │
       │    ACUMULADOR           │
       │    α = 0.5 (constante)  │
       │    (5 capas)            │
       │                         │
       ├─────────────────────────┤
       │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ← Arreglo Piezoeléctrico
       │ ▓▓▓ ANILLO ACTUADOR ▓▓▓ │
       ├─────────────────────────┤
       │                         │
       │    PLACA BASE           │
       │    α = 2.5 (reflector)  │
       │                         │
       └─────────────────────────┘
```

### 3.2 Composición de Capas

| Capa # | Espesor | Valor α | Material | Función |
|--------|---------|---------|----------|---------|
| 1–5 | 0.5 mm c/u | 0.5 | Compuesto ZrO₂/SiC | Acumulador (almacenamiento φ) |
| 6–20 | 0.3 mm c/u | 0.5→2.0 | Metamaterial gradado | Zona de gradiente |
| 21–23 | 0.4 mm c/u | 2.0→2.5 | Geometría de tobera | Escape direccional |
| Base | 2.0 mm | 2.5 | Placa reflectora | Prevención de flujo inverso |

### 3.3 Fabricación del Gradiente de Metamaterial

```
    PROGRAMA DE DEPOSICIÓN DE CAPAS
    ════════════════════════════════════════
    
    Valor α
    2.5 ─┐                           ┌─ Tobera
         │                           │
    2.0 ─┤                       ┌───┘
         │                  ┌────┘
    1.5 ─┤              ┌───┘
         │         ┌────┘
    1.0 ─┤     ┌───┘
         │┌────┘
    0.5 ─┴────┬────┬────┬────┬────┬────┬────
        1    5   10   15   20   23   Base
                    Capa #
    
    Gradiente: Δα/Δz = 0.10 por capa
              ∇α = 200 m⁻¹ (objetivo)
```

### 3.4 Dimensiones del Núcleo

```
    PLANO DIMENSIONAL
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
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  │ 3 (piezoeléctrico)
    ├───────────────────────┤  ↓
    │                       │  ↑
    │                       │  │ 2 (base)
    └───────────────────────┘  ↓
    
    ←──────── 40 ────────→
    
    Altura Total: 30 mm
    Diámetro Exterior: 40 mm
    Masa del Núcleo: ~50 g
```

---

## 4. SISTEMA DE PROPULSIÓN (Híbrido TPH/OMV)

### 4.1 Arreglo de Actuadores Piezoeléctricos

```
    ANILLO ACTUADOR (Vista Superior)
    ════════════════════════════════════════
    
                    N
                    │
            ┌───────┼───────┐
           ╱   P1   │   P2   ╲
          ╱    ◆────┼───◆    ╲
         │          │          │
       O─┼──◆───────┼──────◆──┼─E
         │   P8     │     P3   │
          ╲    ◆────│───◆    ╱
           ╲   P7   │   P4   ╱
            └───────┼───────┘
                    │
                    S
                   P5,P6
    
    8× Actuadores PZT-5H (disposición radial)
    
    Secuencia de Disparo (Modo TPH):
    ─────────────────────────────────
    t=0:    P1,P2 DISPARAN (pulso Norte)
    t=τ/4:  P3    DISPARA  (propagación)
    t=τ/2:  P4,P5 DISPARAN (llega al Sur)
    t=3τ/4: P6,P7 DISPARAN (propagación)
    t=τ:    P8,P1 DISPARAN (ciclo completo)
    
    Genera ONDA VIAJERA alrededor de la circunferencia
```

### 4.2 Especificaciones del Actuador

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| **Tipo** | PZT-5H (Titanato Zirconato de Plomo) | Alto coeficiente d₃₃ |
| **Dimensiones** | 5×5×2 mm cada uno | Configuración en pila |
| **Cantidad** | 8 (arreglo radial) | Fase controlada independientemente |
| **Desplazamiento Máx.** | 2 µm | A 200 V |
| **Frecuencia de Resonancia** | 50 kHz | Resonancia mecánica |
| **Frecuencia de Operación** | 1–10 kHz | Tasa de pulso TPH |
| **Voltaje de Excitación** | 0–200 V | Forma de onda programable |

### 4.3 Forma de Onda del Pulso

```
    SEÑAL DE EXCITACIÓN TPH
    ════════════════════════════════════════
    
    Voltaje (V)
    200 ─┐     ┌─┐     ┌─┐     ┌─┐
         │     │ │     │ │     │ │
    100 ─┤     │ │     │ │     │ │
         │     │ │     │ │     │ │
      0 ─┴─────┴─┴─────┴─┴─────┴─┴─────
        0    1ms   2ms   3ms   4ms
        
        ←τ_subida→   ←τ_bajada→
            50µs          50µs
    
    Ciclo de Trabajo: 10%
    Tasa de Repetición: 1 kHz (ajustable 100 Hz – 10 kHz)
    Tiempo de Subida: 50 µs (choque pronunciado)
    
    MODO OMV (Senoidal Continua):
    ──────────────────────────────
    
    200  ─     ╱╲      ╱╲      ╱╲
         │    ╱  ╲    ╱  ╲    ╱  ╲
    100 ─┼───╱────╲──╱────╲──╱────╲──
         │  ╱      ╲╱      ╲╱      ╲
      0 ─┴─────────────────────────
        0    0.5ms  1ms   1.5ms  2ms
        
    Frecuencia: 2 kHz (rectificación ponderomotriz)
```

### 4.4 Empuje Esperado

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
    
    F_empuje ≈ 50 nN/kHz (modo TPH)
    F_DC ≈ 197 pN (OMV continuo)
```

---

## 5. ELECTRÓNICA DE CONTROL

### 5.1 Diagrama de Bloques del Sistema

```
    ARQUITECTURA DE CONTROL
    ════════════════════════════════════════
    
    ┌─────────────────────────────────────────────────────────┐
    │                     UNIDAD DE CONTROL                   │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
    │  │             │    │             │    │             │  │
    │  │    MCU      │───→│  GENERADOR  │───→│  AMPLIF.    │  │
    │  │  (STM32H7)  │    │  DE FORMA   │    │  8 CANALES  │  │
    │  │             │    │  DE ONDA    │    │   (200V)    │  │
    │  │             │    │   (DDS)     │    │             │  │
    │  └──────┬──────┘    └─────────────┘    └──────┬──────┘  │
    │         │                                     │         │
    │         │    ┌────────────────────────────────┘         │
    │         │    │                                          │
    └─────────┼────┼──────────────────────────────────────────┘
              │    │
              │    ▼
              │  ┌─────────────────────────────────────────┐
              │  │         ARREGLO PIEZOELÉCTRICO (8×)     │
              │  │      ◆──◆──◆──◆──◆──◆──◆──◆         │ 
              │  └─────────────────┬───────────────────────┘
              │                    │
              │                    ▼
              │  ┌─────────────────────────────────────────┐
              │  │         NÚCLEO DE METAMATERIAL          │
              │  │    ┌───────────────────────┐            │
              │  │    │   CAPACITOR           │            │
              │  │    │   TOPOLÓGICO          │───→ EMPUJE │
              │  │    └───────────────────────┘            │
              │  └─────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    ARREGLO DE SENSORES                  │
    │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
    │  │ BALANZA   │  │  SENSORES │  │   ACELER. │            │
    │  │ TORSIÓN   │  │   TEMP.   │  │  (6-DOF)  │            │
    │  │ (empuje)  │  │   (4×)    │  │           │            │
    │  └───────────┘  └───────────┘  └───────────┘            │
    └─────────────────────────────────────────────────────────┘
```

### 5.2 Lista de Componentes

| Componente | Número de Parte | Cantidad | Función |
|------------|-----------------|----------|---------|
| MCU | STM32H743 | 1 | Controlador principal |
| DDS | AD9910 | 1 | Generación de forma de onda |
| Amplificador AV | PA94 | 8 | Controlador piezoeléctrico (200 V) |
| DAC | AD5764 | 2 | Salida de 8 canales |
| Temperatura | PT1000 | 4 | Monitoreo del núcleo |
| Acelerómetro | ADXL355 | 1 | Sensado 6-DOF |
| Fuente de Poder | 24 V/5 A | 1 | Potencia principal |
| Fuente AV | 200 V/100 mA | 1 | Potencia piezoeléctrica |

### 5.3 Lazo de Control (Modo Levitación)

```
    CONTROLADOR PD
    ════════════════════════════════════════
    
    Referencia (z₀) ──┬──→(+)──→[Kp]──┬──→[Σ]──→[f(Hz)]──→ Piezo
                      │    ↑          │    ↑
                      │    │(-)       │    │
                      │    │          │    │
                      │    └──[z]─────│────┤
                      │               │    │
                      └──→[d/dt]──→[Kd]────┘
    
    Función de Transferencia:
    ──────────────────────────
    f(t) = Kp·[z₀ - z(t)] + Kd·[dz/dt]
    
    Parámetros:
    Kp = 1000 Hz/µm
    Kd = 100 Hz·s/µm
    Latencia del sensor: 2 ms (compensada)
```

---

## 6. DISTRIBUCIÓN ESTRUCTURAL

### 6.1 Vista Explosionada del Ensamble

```
    VISTA EXPLOSIONADA
    ════════════════════════════════════════
    
                    ┌───┐
                    │ 1 │  Cubierta Superior (Al)
                    └─┬─┘
                      │
                   ╱─────╲
                  ╱   2   ╲    Cono de Tobera (Metamaterial)
                 ╱─────────╲
                      │
               ┌──────┴──────┐
               │      3      │  Pila de Gradiente (15 capas)
               │             │
               │   ░░░░░░░   │
               │   ░░░░░░░   │
               │   ░░░░░░░   │
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      4      │  Núcleo Acumulador
               │   ▓▓▓▓▓▓▓   │
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      5      │
               │◆◆◆◆◆◆◆◆ │  Ensamble del Anillo Piezoeléctrico
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      6      │  Placa Base (Reflector)
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      7      │  Bahía de Electrónica
               │  [PCB] [PS] │
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │      8      │  Brida de Montaje
               └─────────────┘
    
    
    ORDEN DE ENSAMBLE: 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1
```

### 6.2 Corte Transversal (Ensamblado)

```
    CORTE TRANSVERSAL ENSAMBLADO
    ════════════════════════════════════════
    
              EMPUJE ↑
                    │
         ══════════╪══════════  ← Cubierta Superior
        ╱          │          ╲
       ╱    ┌──────┴──────┐    ╲  ← Tobera (α=2.5)
      ╱     │             │     ╲
     ╱      │   φ_salida  │      ╲
    ════════╪═════════════╪════════ ← Apertura (Ø25 mm)
    ║       │             │       ║
    ║   ╔═══╧═════════════╧═══╗   ║
    ║   ║                     ║   ║  ← Zona de Gradiente
    ║   ║   α: 2.0 → 0.5      ║   ║    (15 capas)
    ║   ║                     ║   ║
    ║   ║   ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓   ║   ║  ← Flujo de energía
    ║   ║                     ║   ║
    ║   ╠═════════════════════╣   ║
    ║   ║   ████████████████  ║   ║  ← φ_max (Núcleo)
    ║   ║   ██  ACUMULADOR █  ║   ║    α=0.5
    ║   ║   ████████████████  ║   ║
    ║   ╠═════════════════════╣   ║
    ║   ║◆◆◆◆◆ PIEZO ◆◆◆◆║   ║  ← Anillo Actuador
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
    
    Altura Total: 85 mm
    Diámetro Total: 60 mm
    Masa Total: ~250 g
```

---

## 7. DIAGRAMAS DE ENSAMBLE

### 7.1 Vista Isométrica

```
    VISTA ISOMÉTRICA 3D
    ════════════════════════════════════════
    
                      ↗ EMPUJE
                    ╱
                  ╱
               ╱─────╲
              ╱  CUBI.  ╲
             ╱  SUPERIOR ╲
            ╱─────────────╲
          │╲             ╱│
          │ ╲  TOBERA   ╱ │
          │  ╲         ╱  │
          │   ╲───────╱   │
          │   │       │   │
          │   │ ZONA  │   │
          │   │ GRAD. │   │
          │   │       │   │
          │   ├───────┤   │
          │   │ NÚCL. │   │
          │   ├───────┤   │
          │   │▓PIEZO▓│   │
          │   ├───────┤   │
          │   │ BASE  │   │
          │   └───────┘   │
          │  ELECTRÓNICA  │
          │    BAHÍA      │
          └───────┬───────┘
                  │
            ══════╧══════
             BRIDA DE MONTAJE
    
    Escala: ~1:2
```

### 7.2 Diagrama de Cableado

```
    CONEXIONES ELÉCTRICAS
    ════════════════════════════════════════
    
    24V DC ENTRADA ──┬──→ [REG. 5V] ──→ MCU, Sensores
                     │
                     └──→ [ELEVADOR AV] ──→ Raíl 200V
                                 │
                                 ▼
                          ┌─────────────┐
                          │  AMPLIF. 8 CH│
                          │  PA94 ×8    │
                          └──┬──┬──┬──┬─┘
                             │  │  │  │
             ┌───────────────┼──┼──┼──┼────────────┐
             │               │  │  │  │            │
             ▼               ▼  ▼  ▼  ▼            ▼
            P1              P2 P3 P4 P5            P8
             ◆───────────────◆──◆──◆──◆─────────◆
             │                                     │
             └────────── ANILLO PIEZO ─────────────┘
    
    RUTA DE SEÑAL:
    ──────────────
    MCU (SPI) ──→ DDS (AD9910) ──→ DAC ──→ AMPLIF. ──→ PIEZO
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

### 8.1 Especificaciones Físicas

| Parámetro | Valor | Tolerancia |
|-----------|-------|------------|
| **Altura Total** | 85 mm | ±1 mm |
| **Diámetro Total** | 60 mm | ±0.5 mm |
| **Masa Total** | 250 g | ±10 g |
| **Masa del Núcleo** | 50 g | ±2 g |
| **Diámetro del Núcleo** | 40 mm | ±0.1 mm |
| **Altura del Núcleo** | 30 mm | ±0.5 mm |
| **Apertura de Tobera** | 25 mm | ±0.2 mm |
| **Número de Capas** | 23 | — |
| **Espesor de Capa** | 0.3–0.5 mm | ±0.05 mm |

### 8.2 Especificaciones Eléctricas

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| **Voltaje de Entrada** | 24 V DC | ±5% |
| **Potencia de Entrada** | 5–50 W | Dependiente del modo |
| **Raíl AV** | 200 V DC | Excitación piezoeléctrica |
| **Canales Piezoeléctricos** | 8 | Fase independiente |
| **Frecuencia de Operación** | 100 Hz – 50 kHz | Programable |
| **Interfaz de Control** | USB / UART | 115200 baudios |

### 8.3 Especificaciones de Rendimiento

| Parámetro | Valor | Condiciones |
|-----------|-------|-------------|
| **Empuje TPH** | 100–500 nN | 1–10 kHz, 200 V |
| **Empuje DC OMV** | ~200 pN | 2 kHz continuo |
| **Impulso/Pulso** | 123 pN·s | Pulso TPH único |
| **Fuerza del Gradiente** | ∇α = 200 m⁻¹ | Objetivo de diseño |
| **Máximo de Campo** | φ_max ≈ 0.1 | Unidades normalizadas |
| **Inmunidad al Ruido** | 5% defectos de fabricación | Verificado por Monte Carlo |
| **Temp. de Operación** | 20–40 °C | Temperatura ambiente |

### 8.4 Especificaciones del Metamaterial

| Zona de Capa | Rango α | Sistema de Material | Propósito |
|--------------|---------|---------------------|-----------|
| **Acumulador** | 0.5 | ZrO₂-SiC (70:30) | Almacenamiento φ |
| **Gradiente** | 0.5→2.0 | ZrO₂-Al₂O₃ gradado | Transporte |
| **Tobera** | 2.0→2.5 | Al₂O₃-TiO₂ | Escape |
| **Reflector** | 2.5 | Al₂O₃ denso | Prevención de flujo inverso |

---

## 9. PROTOCOLOS DE SEGURIDAD

### 9.1 Riesgos Operacionales

| Riesgo | Nivel de Riesgo | Mitigación |
|--------|-----------------|------------|
| **Alto Voltaje (200 V)** | ALTO | Enclavamientos, puesta a tierra, aislamiento |
| **Emisión Acústica Piezoeléctrica** | MEDIO | Protección auditiva por encima de 10 kHz |
| **Térmica (Núcleo)** | BAJO | Monitoreo de temperatura, apagado automático |
| **Vibración Mecánica** | BAJO | Montaje seguro, amortiguación |

### 9.2 Lista de Verificación Pre-Operación

```
    LISTA DE VERIFICACIÓN PRE-VUELO
    ════════════════════════════════════════
    
    [ ] 1. Inspección visual (sin grietas ni residuos)
    [ ] 2. Conexiones eléctricas verificadas
    [ ] 3. Enclavamiento AV activado
    [ ] 4. Sensores de temperatura respondiendo (4/4)
    [ ] 5. Acelerómetro calibrado
    [ ] 6. Balanza de torsión en cero
    [ ] 7. Presión de vacío/atmosférica registrada
    [ ] 8. Software de control cargado
    [ ] 9. Parada de emergencia accesible
    [ ] 10. Personal despejado de la zona AV
    
    FIRMA AUTORIZADA: ________________
    FECHA: ________________
```

### 9.3 Procedimientos de Emergencia

```
    SECUENCIA DE APAGADO DE EMERGENCIA
    ════════════════════════════════════════
    
    1. PRESIONAR BOTÓN ROJO E-STOP (corta toda la energía)
    2. Esperar 30 segundos (descarga del capacitor AV)
    3. Verificar que el LED indicador AV esté APAGADO
    4. Poner a tierra el raíl AV con sonda de descarga
    5. Documentar el incidente en el registro
    
    NO tocar el arreglo piezoeléctrico hasta completar el Paso 4
```

---

## APÉNDICE A: Protocolo de Pruebas

### A.1 Verificación de Empuje

1. Montar la unidad en una balanza de torsión calibrada
2. Llevar la balanza a cero en estado quiescente
3. Aplicar el protocolo TPH a 1 kHz
4. Registrar la deflexión durante 60 segundos
5. Calcular el empuje promedio a partir de la curva de calibración
6. Comparar con el valor predicho de ~100 nN

### A.2 Verificación de la Ley de Escala

1. Barrer frecuencia: 100 Hz → 10 kHz
2. Registrar el empuje en cada frecuencia
3. Graficar empuje vs frecuencia
4. Verificar relación lineal (F ∝ f)
5. Medir la pendiente: objetivo ~50 nN/kHz

---

## APÉNDICE B: HOJA DE RUTA

  MARK 1 ──────── Prototipo de Laboratorio
  ══════          • Masa: 250 g
                  • Empuje: 100–500 nN
                  • Objetivo: Validar física TPH/OMV
                  • Prueba: Balanza de torsión + vacío
                  • Costo: ~$14,000 USD
                          ↓
  MARK 2 ──────── Prototipo Escalado
                  • Masa: 3–11 Lb
                  • Empuje: µN – mN
                  • Mejoras: Enfriamiento líquido, núcleos apilados
                  • Prueba: Soporte de empuje calibrado
                          ↓
  MARK 3 ──────── Demostrador de Ingeniería
                  • Masa: 22–110 Lb
                  • Empuje: mN – N
                  • Objetivo: Demostrar escalabilidad
                          ↓
  MARK 4+ ─────── Prototipo de Vuelo
                  • Integración con vehículo
                  • Certificación
                  • Vuelo de prueba suborbital


```
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║                      FIN DEL DOCUMENTO                           ║
    ║                                                                  ║
    ║           AETHERION MARK 1 - ESPECIFICACIÓN DE INGENIERÍA        ║
    ║                        Revisión 1.0                              ║
    ║                                                                  ║
    ║         "El tiempo no es lo que pasa, sino lo que pulsa."        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
```

     +-----------------------------------------------------------------------+
     | PROPIETARIO Y CONFIDENCIAL | ZARPAFANTASMA SYSTEMS CORP.              |
     | ID DE PROYECTO: [AETHERION] | AUTORIZACIÓN DE SEGURIDAD: NIVEL 5      |
     |-----------------------------------------------------------------------|
     | ADVERTENCIA: El acceso no autorizado, distribución o reproducción de  |
     | este documento está estrictamente prohibido por el Protocolo Legal    |
     | ZS-CORP. El rastreo electrónico y la marca de agua forense están      |
     | activos en este archivo.                                              |
     +-----------------------------------------------------------------------+
