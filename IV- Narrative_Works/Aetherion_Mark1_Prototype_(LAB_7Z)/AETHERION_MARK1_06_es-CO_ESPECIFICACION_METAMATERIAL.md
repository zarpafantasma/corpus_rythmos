# AETHERION MARK 1
## Especificación de Fabricación del Núcleo de Metamaterial

**ID del Documento:** ATP-MK1-MTL-001  
**Revisión:** 1.0  
**Clasificación:** ESPECIFICACIÓN PARA PROVEEDOR  
**Fecha:** Febrero 2026  

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    DOCUMENTO DE FABRICACIÓN PERSONALIZADA                    ║
║                                                                              ║
║     Este documento contiene especificaciones completas para fabricar         ║
║     el Núcleo de Metamaterial de Gradiente Topológico Aetherion Mark 1.      ║
║                                                                              ║
║     Destinatarios Previstos:                                                 ║
║       • Fabricantes de cerámicas avanzadas                                   ║
║       • Instalaciones de fabricación de metamateriales                       ║
║       • Laboratorios de deposición de películas delgadas                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## TABLA DE CONTENIDOS

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Fundamento Teórico](#2-fundamento-teórico)
3. [Arquitectura del Núcleo](#3-arquitectura-del-núcleo)
4. [Especificaciones de Capas](#4-especificaciones-de-capas)
5. [Composiciones de Materiales](#5-composiciones-de-materiales)
6. [Requisitos Dimensionales](#6-requisitos-dimensionales)
7. [Perfil de Gradiente](#7-perfil-de-gradiente)
8. [Métodos de Fabricación](#8-métodos-de-fabricación)
9. [Control de Calidad](#9-control-de-calidad)
10. [Criterios de Aceptación](#10-criterios-de-aceptación)
11. [Manejo y Almacenamiento](#11-manejo-y-almacenamiento)
12. [Entregables](#12-entregables)

---

## 1. RESUMEN EJECUTIVO

### 1.1 Descripción del Componente

El Núcleo de Metamaterial Aetherion Mark 1 es una **pila cerámica graduada** diseñada para crear un gradiente espacial en el exponente topológico efectivo (α) de la interacción vacío-materia. Este gradiente permite el acoplamiento con las fluctuaciones del campo de punto cero para investigación experimental de propulsión.

### 1.2 Requisitos Clave

| Parámetro | Valor | ¿Crítico? |
|-----------|-------|-----------|
| Total de capas | 23 | Sí |
| Diámetro del núcleo | 40.0 mm | Sí |
| Altura total | ~15 mm | Sí |
| Rango de gradiente α | 0.5 → 2.5 | **CRÍTICO** |
| Temperatura de operación | 20-100°C | Sí |
| Monotonicidad del gradiente | Estrictamente creciente | **CRÍTICO** |

### 1.3 Aplicación

Prototipo de laboratorio para investigación de propulsión por gradiente de vacío. El núcleo será sometido a:
- Estrés mecánico piezoeléctrico (vibración de 1-50 kHz)
- Ciclado térmico (ambiente a 100°C)
- Ambiente de vacío (10⁻³ a 10⁻⁶ Torr)

---

## 2. FUNDAMENTO TEÓRICO

### 2.1 El Exponente Topológico (α)

En la Relatividad Temporal Multiescala (RTM), el parámetro **α** caracteriza la intensidad de acoplamiento local entre la materia y las fluctuaciones del campo de vacío. La relación es:

```
Densidad de energía: ε ∝ ∇α × (gradientes de campo)

Donde:
  α < 1  → Transporte sub-difusivo (acumulación de energía)
  α = 1  → Transporte balístico (propagación lineal)
  α > 1  → Transporte super-difusivo (dispersión de energía)
  α ≈ 2  → Atractor jerárquico (tipo gravitacional)
```

### 2.2 Realización Física de α

El exponente topológico α se realiza a través de **propiedades microestructurales**:

| Propiedad del Material | Efecto sobre α |
|------------------------|----------------|
| Porosidad (mayor) | Aumenta α |
| Tamaño de grano (menor) | Disminuye α |
| Constante dieléctrica (mayor) | Disminuye α |
| Densidad (mayor) | Disminuye α |
| Orden cristalino (mayor) | Disminuye α |

### 2.3 El Requisito de Gradiente

```
¿POR QUÉ UN GRADIENTE?
═══════════════════════════════════════════════════════════════

Un material con α uniforme almacena energía de punto cero simétricamente:

    α Uniforme:     ← φ →  (las fuerzas se cancelan, sin efecto neto)

Un GRADIENTE en α crea distribución asimétrica de energía:

    α Bajo ──────────────────────── α Alto
           φ se acumula aquí → se expulsa aquí →
           
Esto permite transferencia DIRECCIONAL de energía/momento.
```

### 2.4 Justificación de Selección de Materiales

El valor de α se diseña mezclando fases cerámicas con diferentes propiedades:

| Fase | Propiedades | Efecto sobre α |
|------|-------------|----------------|
| **ZrO₂** (Zirconia) | Alta densidad, alto ε | α Bajo (~0.5) |
| **SiC** (Carburo de Silicio) | Alta conductividad térmica | Estabiliza α |
| **Al₂O₃** (Alúmina) | Densidad moderada, estable | α Medio (~1.5) |
| **TiO₂** (Titania) | Variantes de alta porosidad | α Alto (~2.5) |

---

## 3. ARQUITECTURA DEL NÚCLEO

### 3.1 Zonas Funcionales

El núcleo consiste en **cuatro zonas funcionales**:

```
SECCIÓN TRANSVERSAL (no a escala)
═══════════════════════════════════════════════════════════════

                ↑ DIRECCIÓN DEL EMPUJE
                │
        ┌───────┴───────┐
       ╱                 ╲
      ╱   ZONA 4: TOBERA  ╲     α = 2.0 → 2.5
     ╱     (3 capas)       ╲    Escape direccional
    ╱                       ╲
   ├─────────────────────────┤  ← Apertura de escape Ø25mm
   │                         │
   │   ZONA 3: GRADIENTE     │  α = 0.5 → 2.0
   │     (15 capas)          │  Región de transporte de energía
   │                         │
   ├─────────────────────────┤  ← Ubicación de φ_max
   │                         │
   │   ZONA 2: ACUMULADOR    │  α = 0.5 (constante)
   │     (5 capas)           │  Almacenamiento de energía de punto cero
   │                         │
   ├─────────────────────────┤
   │   ZONA 1: BASE          │  α = 2.5 (constante)
   │     (1 capa)            │  Reflector / bloqueo de retroflujo
   └─────────────────────────┘
```

### 3.2 Funciones de las Zonas

| Zona | Capas | Valor α | Función |
|------|-------|---------|---------|
| **1: Base** | 1 | 2.5 | Previene fuga de energía hacia atrás |
| **2: Acumulador** | 5 | 0.5 | Almacena energía del campo de punto cero |
| **3: Gradiente** | 15 | 0.5→2.0 | Transporta energía hacia el escape |
| **4: Tobera** | 3 | 2.0→2.5 | Dirige la liberación de momento |

---

## 4. ESPECIFICACIONES DE CAPAS

### 4.1 Programa Completo de Capas

```
ESPECIFICACIÓN CAPA POR CAPA
═══════════════════════════════════════════════════════════════

Capa  │ Zona        │ Espesor   │ Valor α │ Composición
──────┼─────────────┼───────────┼─────────┼─────────────────────
  1   │ Base        │  2.00 mm  │  2.50   │ Al₂O₃ Denso (100%)
──────┼─────────────┼───────────┼─────────┼─────────────────────
  2   │ Acumulador  │  0.50 mm  │  0.50   │ ZrO₂-SiC (70:30)
  3   │ Acumulador  │  0.50 mm  │  0.50   │ ZrO₂-SiC (70:30)
  4   │ Acumulador  │  0.50 mm  │  0.50   │ ZrO₂-SiC (70:30)
  5   │ Acumulador  │  0.50 mm  │  0.50   │ ZrO₂-SiC (70:30)
  6   │ Acumulador  │  0.50 mm  │  0.50   │ ZrO₂-SiC (70:30)
──────┼─────────────┼───────────┼─────────┼─────────────────────
  7   │ Gradiente   │  0.30 mm  │  0.60   │ ZrO₂-Al₂O₃ (90:10)
  8   │ Gradiente   │  0.30 mm  │  0.70   │ ZrO₂-Al₂O₃ (85:15)
  9   │ Gradiente   │  0.30 mm  │  0.80   │ ZrO₂-Al₂O₃ (80:20)
 10   │ Gradiente   │  0.30 mm  │  0.90   │ ZrO₂-Al₂O₃ (75:25)
 11   │ Gradiente   │  0.30 mm  │  1.00   │ ZrO₂-Al₂O₃ (70:30)
 12   │ Gradiente   │  0.30 mm  │  1.10   │ ZrO₂-Al₂O₃ (65:35)
 13   │ Gradiente   │  0.30 mm  │  1.20   │ ZrO₂-Al₂O₃ (60:40)
 14   │ Gradiente   │  0.30 mm  │  1.30   │ ZrO₂-Al₂O₃ (55:45)
 15   │ Gradiente   │  0.30 mm  │  1.40   │ ZrO₂-Al₂O₃ (50:50)
 16   │ Gradiente   │  0.30 mm  │  1.50   │ ZrO₂-Al₂O₃ (45:55)
 17   │ Gradiente   │  0.30 mm  │  1.60   │ ZrO₂-Al₂O₃ (40:60)
 18   │ Gradiente   │  0.30 mm  │  1.70   │ ZrO₂-Al₂O₃ (35:65)
 19   │ Gradiente   │  0.30 mm  │  1.80   │ ZrO₂-Al₂O₃ (30:70)
 20   │ Gradiente   │  0.30 mm  │  1.90   │ ZrO₂-Al₂O₃ (25:75)
 21   │ Gradiente   │  0.30 mm  │  2.00   │ ZrO₂-Al₂O₃ (20:80)
──────┼─────────────┼───────────┼─────────┼─────────────────────
 22   │ Tobera      │  0.40 mm  │  2.17   │ Al₂O₃-TiO₂ (75:25)
 23   │ Tobera      │  0.40 mm  │  2.33   │ Al₂O₃-TiO₂ (50:50)
 24   │ Tobera      │  0.40 mm  │  2.50   │ Al₂O₃-TiO₂ (25:75)
──────┴─────────────┴───────────┴─────────┴─────────────────────

TOTAL: 24 capas, ~14.7 mm de altura (excluyendo geometría de tobera)
```

### 4.2 Tolerancias Críticas

| Parámetro | Nominal | Tolerancia | Notas |
|-----------|---------|------------|-------|
| Espesor de capa | Según especificación | ±0.05 mm | Crítico para el gradiente |
| Valor α | Según especificación | ±0.05 | Verificado por medición de ε |
| Composición | Según especificación | ±2% en peso | Crítico |
| Unión de interfaces | — | Sin delaminación | Debe sobrevivir ciclado térmico |

---

## 5. COMPOSICIONES DE MATERIALES

### 5.1 Materiales Base

| Material | Pureza | Tamaño de Partícula | Fuente |
|----------|--------|---------------------|--------|
| **ZrO₂** (Estabilizado con Itria) | ≥99.5% | 0.5-1.0 µm | Tosoh TZ-3Y |
| **SiC** (fase-α) | ≥99.0% | 0.5-2.0 µm | Superior Graphite |
| **Al₂O₃** (fase-α) | ≥99.8% | 0.3-0.5 µm | Sumitomo AKP-30 |
| **TiO₂** (Anatasa) | ≥99.5% | 0.1-0.3 µm | Evonik P25 |

### 5.2 Detalles de Composición por Zona

#### Zona 1: Reflector Base

```
CAPA 1: Alúmina Densa
────────────────────────
Composición: 100% Al₂O₃
Densidad objetivo: ≥98% teórica
Porosidad: <2%
Propósito: Barrera de α alto, previene retroflujo

Notas de procesamiento:
- Usar polvo fino (0.3 µm)
- Sinterizar a mínimo 1600°C
- α objetivo = 2.5 vía alta densidad
```

#### Zona 2: Acumulador

```
CAPAS 2-6: Compuesto ZrO₂-SiC
──────────────────────────────
Composición: 70 wt% ZrO₂ + 30 wt% SiC
Densidad objetivo: 95-97% teórica
Porosidad: 3-5%
Propósito: Almacenamiento de energía con α bajo

Notas de procesamiento:
- ZrO₂ proporciona α bajo (alto ε, alta densidad)
- SiC proporciona estabilidad térmica
- Prensado en caliente a 1500°C, 20 MPa
- α objetivo = 0.5
```

#### Zona 3: Gradiente

```
CAPAS 7-21: ZrO₂-Al₂O₃ Graduado
─────────────────────────────────
Composición: Variable (ver Programa de Capas)
Densidad objetivo: 94-96% teórica
Porosidad: 4-6%
Propósito: Gradiente monotónico de α

Notas de procesamiento:
- Cada capa preparada independientemente
- La co-sinterización puede causar difusión - controlar cuidadosamente
- Capa 7: 90:10 ZrO₂:Al₂O₃ → α ≈ 0.6
- Capa 21: 20:80 ZrO₂:Al₂O₃ → α ≈ 2.0
- Interpolación lineal entre ellas

CÁLCULO DE α:
  α(capa) = 0.5 + (capa - 6) × 0.1
  
  Capa 7:  α = 0.5 + 1×0.1 = 0.6
  Capa 14: α = 0.5 + 8×0.1 = 1.3
  Capa 21: α = 0.5 + 15×0.1 = 2.0
```

#### Zona 4: Tobera

```
CAPAS 22-24: Al₂O₃-TiO₂ Alto-α
───────────────────────────────
Composición: Variable (ver Programa de Capas)
Densidad objetivo: 90-94% teórica
Porosidad: 6-10%
Propósito: Región de escape con α alto

Notas de procesamiento:
- TiO₂ aumenta porosidad → mayor α
- Temperatura de sinterización menor (1400°C) para retener porosidad
- Capa 24 (más externa): α ≈ 2.5

GEOMETRÍA: Las capas de tobera deben formar cono truncado
- Diámetro interno: 25 mm (apertura de escape)
- Diámetro externo: 35-40 mm (coincide con el núcleo)
- Esto puede lograrse mediante:
  a) Mecanizado después del sinterizado, o
  b) Prensado de cuerpo verde con forma
```

### 5.3 Tabla de Composición (Porcentaje en Peso)

| Capa | ZrO₂ | SiC | Al₂O₃ | TiO₂ | α Objetivo |
|------|------|-----|-------|------|------------|
| 1 | — | — | 100 | — | 2.50 |
| 2-6 | 70 | 30 | — | — | 0.50 |
| 7 | 90 | — | 10 | — | 0.60 |
| 8 | 85 | — | 15 | — | 0.70 |
| 9 | 80 | — | 20 | — | 0.80 |
| 10 | 75 | — | 25 | — | 0.90 |
| 11 | 70 | — | 30 | — | 1.00 |
| 12 | 65 | — | 35 | — | 1.10 |
| 13 | 60 | — | 40 | — | 1.20 |
| 14 | 55 | — | 45 | — | 1.30 |
| 15 | 50 | — | 50 | — | 1.40 |
| 16 | 45 | — | 55 | — | 1.50 |
| 17 | 40 | — | 60 | — | 1.60 |
| 18 | 35 | — | 65 | — | 1.70 |
| 19 | 30 | — | 70 | — | 1.80 |
| 20 | 25 | — | 75 | — | 1.90 |
| 21 | 20 | — | 80 | — | 2.00 |
| 22 | — | — | 75 | 25 | 2.17 |
| 23 | — | — | 50 | 50 | 2.33 |
| 24 | — | — | 25 | 75 | 2.50 |

---

## 6. REQUISITOS DIMENSIONALES

### 6.1 Geometría del Núcleo

```
DIBUJO DIMENSIONAL (Todas las dimensiones en mm)
═══════════════════════════════════════════════════════════════

                    ← 25.0 ±0.2 →
               ┌─────────────────┐
              ╱                   ╲
             ╱                     ╲     ↑
            ╱       TOBERA          ╲    │ 1.2
           ╱      (Capas 22-24)      ╲   │ (3 × 0.4)
          ╱                           ╲  ↓
         ├─────────────────────────────┤ ← Apertura de escape
         │                             │ ↑
         │                             │ │
         │      ZONA GRADIENTE         │ │ 4.5
         │      (Capas 7-21)           │ │ (15 × 0.3)
         │                             │ │
         │                             │ ↓
         ├─────────────────────────────┤
         │      ACUMULADOR             │ ↑
         │      (Capas 2-6)            │ │ 2.5
         │                             │ │ (5 × 0.5)
         ├─────────────────────────────┤ ↓
         │      BASE                   │ ↑
         │      (Capa 1)               │ │ 2.0
         └─────────────────────────────┘ ↓
         
         ←───────── 40.0 ±0.1 ─────────→

ALTURA TOTAL DE LA PILA: 10.2 mm (sin cono de tobera)
TOTAL CON TOBERA: ~12-14 mm (dependiendo de geometría del cono)
```

### 6.2 Tolerancias Dimensionales

| Dimensión | Nominal | Tolerancia | Método de Medición |
|-----------|---------|------------|-------------------|
| Diámetro externo | 40.0 mm | ±0.1 mm | Calibrador/CMM |
| Diámetro de apertura | 25.0 mm | ±0.2 mm | Óptico/CMM |
| Altura total | Según pila | ±0.5 mm | Calibrador |
| Espesor de capa | Según especif. | ±0.05 mm | Micrómetro en muestras |
| Planitud (base) | — | <0.05 mm | Mesa de planitud |
| Paralelismo | — | <0.1 mm | CMM |
| Concentricidad | — | <0.2 mm | CMM |

### 6.3 Requisitos de Superficie

| Superficie | Ra (µm) | Notas |
|------------|---------|-------|
| Base (inferior) | <1.6 | Interfaz de montaje |
| Cilíndrica externa | <3.2 | Ajuste con carcasa |
| Apertura (interna) | <6.3 | No crítica |
| Interfaces de capas | N/A | Unidas, no expuestas |

---

## 7. PERFIL DE GRADIENTE

### 7.1 Perfil α vs Posición

```
VALOR α VS POSICIÓN AXIAL
═══════════════════════════════════════════════════════════════

α
2.5 ─┬─────────────────────────────────────────────────┬─ Tobera
     │                                            ╱╱╱╱╱│
2.0 ─┤                                      ╱╱╱╱╱      │
     │                                ╱╱╱╱╱            │
1.5 ─┤                          ╱╱╱╱╱                  │─ Gradiente
     │                    ╱╱╱╱╱                        │  (LINEAL)
1.0 ─┤              ╱╱╱╱╱                              │
     │        ╱╱╱╱╱                                    │
0.5 ─┼───────────────────────┬─────────────────────────┤─ Acumulador
     │        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                         │
     │                       │                         │
2.5 ─┼───────────────────────┴─────────────────────────┤─ Base
     │                                                 │
     └────────┬────────┬────────┬────────┬────────┬────┘
              0        5       10       15       20   z (mm)
              
     BASE    ACUM       ZONA GRADIENTE        TOBERA
     
CRÍTICO: El gradiente en la "Zona Gradiente" DEBE ser:
  • Monotónicamente creciente (sin reversiones)
  • Aproximadamente lineal (Δα/Δz ≈ 0.1 por capa de 0.3mm)
  • Suave (sin discontinuidades de paso >0.15)
```

### 7.2 Verificación del Gradiente

El gradiente se verificará midiendo la **constante dieléctrica efectiva (ε)** de muestras testigo de cada lote de capas:

```
CORRELACIÓN α-ε (Empírica)
═══════════════════════════════════════════════════════════════

Para sistema ZrO₂-Al₂O₃ a 1 kHz, 25°C:

  α ≈ 3.0 - 0.1 × ε_eff

Donde:
  ε_eff = constante dieléctrica efectiva
  
Ejemplo:
  Capa con ε = 25 → α ≈ 3.0 - 2.5 = 0.5 ✓
  Capa con ε = 10 → α ≈ 3.0 - 1.0 = 2.0 ✓

Medición:
  • Preparar disco testigo: 10mm diám × 1mm espesor
  • Depositar electrodos de oro por sputtering en ambas caras
  • Medir capacitancia a 1 kHz
  • Calcular ε a partir de la geometría
```

---

## 8. MÉTODOS DE FABRICACIÓN

### 8.1 Flujo de Proceso Recomendado

```
FLUJO DE PROCESO DE FABRICACIÓN
═══════════════════════════════════════════════════════════════

┌─────────────────┐
│  PREP. POLVO    │  Mezclar composiciones según especif. de capa
│  (Por capa)     │  Molienda de bolas 24h en etanol
└────────┬────────┘
         ↓
┌─────────────────┐
│  FORMACIÓN      │  Opción A: Colado en cinta (preferido)
│  BARBOTINA/CINTA│  Opción B: Colado por deslizamiento
└────────┬────────┘
         ↓
┌─────────────────┐
│  LAMINACIÓN     │  Apilar capas con aglutinante fugitivo
│  CUERPO VERDE   │  Aplicar presión uniaxial de 10-20 MPa
└────────┬────────┘
         ↓
┌─────────────────┐
│  QUEMADO DE     │  Rampa lenta: 1°C/min hasta 600°C
│  AGLUTINANTE    │  Mantener 2h a 600°C
└────────┬────────┘
         ↓
┌─────────────────┐
│  SINTERIZADO    │  Zona 1-2: 1550°C, 2h (densificar)
│  (Multi-etapa)  │  Zona 3: 1500°C, 2h (ε controlado)
│                 │  Zona 4: 1400°C, 2h (retener porosidad)
└────────┬────────┘
         ↓
┌─────────────────┐
│  MECANIZADO     │  Rectificar DE a 40.0mm
│  (Si necesario) │  Mecanizar geometría de cono de tobera
└────────┬────────┘
         ↓
┌─────────────────┐
│  CC / PRUEBA    │  Inspección dimensional
│                 │  Medición de ε de muestra testigo
│                 │  Prueba de ciclo térmico
└─────────────────┘
```

### 8.2 Métodos Alternativos

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Colado en Cinta + Laminación** | Mejor control de capas, escalable | Configuración compleja |
| **Colado por Deslizamiento Secuencial** | Equipo simple | Variación en espesor de capa |
| **Prensado en Caliente (capa por capa)** | Alta densidad | Lento, costoso |
| **Sinterizado por Plasma Pulsado** | Rápido, controlado | Limitado a tamaños pequeños |
| **Manufactura Aditiva** | Geometrías complejas | Control de porosidad difícil |

### 8.3 Atmósfera de Sinterizado

| Zona | Atmósfera | Razón |
|------|-----------|-------|
| Zonas 1-3 | Aire | Sinterizado estándar de óxidos |
| Zona 4 (TiO₂) | Aire o N₂ | Prevenir reducción |

---

## 9. CONTROL DE CALIDAD

### 9.1 Inspección en Proceso

| Etapa | Inspección | Criterio de Aceptación |
|-------|------------|------------------------|
| Polvo | Tamaño de partícula (SEM) | Según especif. ±20% |
| Cuerpo verde | Espesor | Según especif. ±0.1 mm |
| Post-sinterizado | Densidad (Arquímedes) | Según especif. ±2% |
| Post-sinterizado | Dimensiones | Según especificación |
| Final | Visual | Sin grietas, astillas |

### 9.2 Pruebas de Muestras Testigo

Para cada composición de capa, preparar muestras testigo:

```
REQUISITOS DE MUESTRAS TESTIGO
═══════════════════════════════════════════════════════════════

Cantidad: 3 muestras por composición de capa (mínimo)
Geometría: Disco, 10mm diámetro × 1mm espesor

Pruebas:
  1. Densidad (método de Arquímedes)
     - Reportar: g/cm³ y % teórico
     
  2. Constante dieléctrica (1 kHz, 25°C)
     - Depositar electrodos de Au por sputtering
     - Medir con medidor LCR
     - Reportar: ε_r
     
  3. Valor α calculado
     - Usar α ≈ 3.0 - 0.1 × ε_r
     - Reportar: α
     
  4. Microestructura (opcional, 1 por zona)
     - SEM de sección transversal pulida
     - Reportar tamaño de grano, porosidad
```

### 9.3 Inspección Final del Ensamble

| Prueba | Método | Criterio de Aceptación |
|--------|--------|------------------------|
| Dimensiones | CMM o calibradores de precisión | Según Sección 6 |
| Masa | Balanza de precisión | 45-55 g |
| Visual | Aumento 10× | Sin defectos visibles |
| Delaminación | Escaneo C ultrasónico (opcional) | Sin vacíos >1mm |
| Ciclo térmico | 5× ciclos, 25→100→25°C | Sin agrietamiento |

---

## 10. CRITERIOS DE ACEPTACIÓN

### 10.1 Requisitos Obligatorios

Todos los siguientes DEBEN cumplirse para aceptación:

```
CRITERIOS DE ACEPTACIÓN OBLIGATORIOS
═══════════════════════════════════════════════════════════════

□ 1. Las 24 capas presentes y unidas
□ 2. Sin grietas o astillas visibles
□ 3. Diámetro externo: 40.0 ±0.1 mm
□ 4. Diámetro de apertura: 25.0 ±0.2 mm  
□ 5. Altura total dentro de ±0.5 mm del diseño
□ 6. Masa: 45-55 g
□ 7. Planitud de base: <0.05 mm
□ 8. Sobrevive 5× ciclos térmicos (25-100-25°C)

□ 9. VERIFICACIÓN DE GRADIENTE α:
     • Testigo Zona 2: α = 0.50 ±0.05
     • Testigo Zona 3 Capa 11: α = 1.00 ±0.10
     • Testigo Zona 3 Capa 21: α = 2.00 ±0.10
     • Testigo Zona 4: α = 2.50 ±0.15
     • El gradiente es monotónicamente creciente

□ 10. Paquete de documentación completo
```

### 10.2 Requisitos Deseables

Estos son objetivos, no obligatorios:

| Requisito | Objetivo | Notas |
|-----------|----------|-------|
| Acabado superficial (Ra) | <1.6 µm en base | Mejora el montaje |
| Concentricidad | <0.1 mm | Mejora la simetría |
| Uniformidad de espesor de capa | <5% variación | Mejora el gradiente |

---

## 11. MANEJO Y ALMACENAMIENTO

### 11.1 Precauciones de Manejo

```
⚠️ REQUISITOS DE MANEJO
═══════════════════════════════════════════════════════════════

• Manipular con guantes limpios y sin pelusa
• Soportar desde la base - no sujetar el cono de tobera
• Evitar impactos - las cerámicas son frágiles
• No apilar múltiples unidades
• Transportar en contenedor con revestimiento de espuma
```

### 11.2 Condiciones de Almacenamiento

| Parámetro | Requisito |
|-----------|-----------|
| Temperatura | 15-30°C |
| Humedad | <60% HR |
| Empaque | Bolsa sellada con desecante |
| Vida útil | Indefinida si se almacena apropiadamente |

### 11.3 Envío

- Cajas individuales acolchadas con espuma
- Marcar como "FRÁGIL - CERÁMICA"
- Incluir paquetes de gel de sílice

---

## 12. ENTREGABLES

### 12.1 Entregables Físicos

| Artículo | Cantidad | Notas |
|----------|----------|-------|
| Ensamble de Núcleo de Metamaterial | 1 | Unidad de vuelo |
| Ensamble de Núcleo de Repuesto | 1 | Respaldo (recomendado) |
| Muestras Testigo | 3 por tipo de capa | Para verificación de CC |

### 12.2 Entregables de Documentación

| Documento | Contenido |
|-----------|-----------|
| Certificado de Conformidad | Declaración de que la unidad cumple especificación |
| Reporte de Inspección Dimensional | Todas las mediciones según Sección 6 |
| Datos de Muestras Testigo | Densidad, ε, α calculado |
| Certificados de Materiales | Trazabilidad de lotes de polvo |
| Hoja de Ruta de Proceso | Números de lote, fechas, operadores |

### 12.3 Programa de Entrega

| Hito | Tiempo de Entrega Típico |
|------|--------------------------|
| Acuse de recibo de orden | 1 semana |
| Adquisición de polvos | 2-3 semanas |
| Fabricación de cuerpo verde | 2 semanas |
| Sinterizado | 1-2 semanas |
| CC y documentación | 1 semana |
| **Total** | **6-8 semanas** |

---

## APÉNDICE A: TARJETA DE REFERENCIA RÁPIDA

```
┌─────────────────────────────────────────────────────────────────┐
│        METAMATERIAL AETHERION MARK 1 - REF. RÁPIDA              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GEOMETRÍA                                                      │
│    Diámetro externo: 40.0 mm                                    │
│    Apertura: 25.0 mm                                            │
│    Altura: ~12-14 mm                                            │
│    Masa: 45-55 g                                                │
│    Capas: 24                                                    │
│                                                                 │
│  GRADIENTE α                                                    │
│    Base: 2.5 (alto, bloquea retroflujo)                         │
│    Acumulador: 0.5 (bajo, almacena energía)                     │
│    Gradiente: 0.5 → 2.0 (lineal, transporta)                    │
│    Tobera: 2.0 → 2.5 (alto, escape)                             │
│                                                                 │
│  MATERIALES                                                     │
│    ZrO₂: α Bajo (0.5)                                           │
│    Al₂O₃: α Medio (1.5)                                         │
│    TiO₂: α Alto (2.5)                                           │
│    SiC: Aditivo de estabilidad térmica                          │
│                                                                 │
│  REQUISITOS CRÍTICOS                                            │
│    ✓ Gradiente α monotónico (¡sin reversiones!)                 │
│    ✓ Sin delaminación                                           │
│    ✓ Sobrevive operación a 100°C                                │
│    ✓ Verificado por medición de ε de muestra testigo            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## APÉNDICE B: INFORMACIÓN DE CONTACTO

```
CONSULTAS TÉCNICAS
═══════════════════════════════════════════════════════════════

Para preguntas sobre esta especificación, contactar:

  Proyecto: Aetherion Mark 1
  Documento: ATP-MK1-MTL-001
  
  [A LLENAR POR EL CLIENTE]
  Contacto Técnico: _______________________
  Correo: _______________________
  Teléfono: _______________________
  
HISTORIAL DE REVISIONES
═══════════════════════════════════════════════════════════════

Rev  │ Fecha      │ Descripción              │ Autor
─────┼────────────┼──────────────────────────┼─────────
1.0  │ 2026-02-28 │ Versión inicial          │ Equipo RTM
```

---

```
═══════════════════════════════════════════════════════════════
                       FIN DEL DOCUMENTO
                              
          ESPECIFICACIÓN DE METAMATERIAL AETHERION MARK 1
                     ATP-MK1-MTL-001 Rev 1.0
                              
            "El gradiente es el motor del transporte."
═══════════════════════════════════════════════════════════════
```



     +-----------------------------------------------------------------------+
     | PROPIETARIO Y CONFIDENCIAL | ZARPAFANTASMA SYSTEMS CORP.              |
     | ID DE PROYECTO: [AETHERION] | NIVEL DE AUTORIZACIÓN: NIVEL 5          |
     |-----------------------------------------------------------------------|
     | ADVERTENCIA: El acceso, distribución o reproducción no autorizada de  |
     | este documento está estrictamente prohibida según el Protocolo Legal  |
     | de ZS-CORP. El rastreo electrónico y las marcas de agua forenses      |
     | están activos en este archivo.                                        |
     +-----------------------------------------------------------------------+
