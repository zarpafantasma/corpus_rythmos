# AETHERION MARK 1
## Protocolo de Pruebas y Procedimientos de Validación

**ID del Documento:** ATP-MK1-001  
**Revisión:** 1.0  
**Clasificación:** OPERACIONAL  
**Fecha:** Febrero 2026  

---

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   ⚠️  DOCUMENTO DE SEGURIDAD CRÍTICO — LEA COMPLETAMENTE ANTES DE OPERAR ⚠️   ║
║                                                                                ║
║     Este protocolo incorpora las restricciones del Equipo Rojo Asesor.         ║
║     El incumplimiento puede resultar en:                                       ║
║       • Daño permanente al equipo (despolarización térmica)                    ║
║       • Lesiones al personal (trauma acústico)                                 ║
║       • Daño a las instalaciones de prueba (falla por resonancia)              ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## TABLA DE CONTENIDOS

1. [Alcance y Objetivos](#1-alcance-y-objetivos)
2. [Equipo Requerido](#2-equipo-requerido)
3. [Requisitos de Seguridad](#3-requisitos-de-seguridad)
4. [Configuración Previa a la Prueba](#4-configuración-previa-a-la-prueba)
5. [Procedimientos de Calibración](#5-procedimientos-de-calibración)
6. [Secuencias de Prueba](#6-secuencias-de-prueba)
7. [Recolección de Datos](#7-recolección-de-datos)
8. [Criterios de Aprobación/Rechazo](#8-criterios-de-aprobaciónrechazo)
9. [Procedimientos de Emergencia](#9-procedimientos-de-emergencia)
10. [Procedimientos Post-Prueba](#10-procedimientos-post-prueba)

---

## 1. ALCANCE Y OBJETIVOS

### 1.1 Propósito

Este documento establece el protocolo completo de pruebas para validar el prototipo del propulsor de gradiente de vacío Aetherion Mark 1. El objetivo principal es medir el empuje ponderomotriz generado mediante los modos operacionales TPH (Jerarquía de Pulsos Temporales) y OMV (Modulación Oscilatoria del Vacío).

### 1.2 Objetivos de la Prueba

| ID | Objetivo | Métrica de Éxito |
|----|----------|------------------|
| **T-01** | Verificar generación de impulso TPH | Deflexión medible en balanza de torsión |
| **T-02** | Verificar empuje DC en OMV | Deflexión sostenida > piso de ruido |
| **T-03** | Validar escalamiento del empuje con frecuencia | Relación lineal F ∝ f |
| **T-04** | Validar escalamiento del empuje con voltaje | Relación cuadrática F ∝ V² |
| **T-05** | Confirmar estabilidad térmica | Arreglo piezo < 90°C durante operación |
| **T-06** | Verificar respuesta del sistema de control | Todos los modos conmutan correctamente |

### 1.3 Fases de Prueba

```
FASE 0: Configuración y Calibración (Día 1)
    ↓
FASE 1: Verificación Eléctrica — Sin generación de empuje (Día 1)
    ↓
FASE 2: Pruebas Atmosféricas — Detección inicial de empuje (Día 2)
    ↓
FASE 3: Pruebas en Vacío — Mediciones de precisión (Día 3-5)
    ↓
FASE 4: Barrido Paramétrico — Caracterización completa (Día 6-10)
```

---

## 2. EQUIPO REQUERIDO

### 2.1 Artículos de Prueba

| Artículo | Especificación | Cantidad |
|----------|----------------|----------|
| Unidad Aetherion Mark 1 | Según especificación de ingeniería | 1 |
| Arreglo PZT-5H de repuesto | 8× actuadores, precableados | 1 juego |
| Masas de calibración | 1mg, 10mg, 100mg, 1g | 1 juego |

### 2.2 Infraestructura de Prueba

| Artículo | Especificación | Propósito |
|----------|----------------|-----------|
| **Balanza de Torsión** | Resolución < 10 nN | Medición de empuje |
| **Cámara de Vacío** | Capacidad 10⁻³ a 10⁻⁶ Torr | Eliminar arrastre del aire |
| **Sensor de Desplazamiento Óptico** | Resolución < 0.1 µm | Lectura de balanza |
| **Mesa de Aislamiento** | Neumática, corte < 1 Hz | Aislamiento de vibraciones |
| **Jaula de Faraday** | Cerramiento completo | Blindaje EMI |

### 2.3 Instrumentación

| Instrumento | Modelo (Ejemplo) | Propósito |
|-------------|------------------|-----------|
| Osciloscopio | Keysight DSOX3024T | Verificación de forma de onda |
| Multímetro | Fluke 87V | Voltaje/corriente |
| Cámara Térmica | FLIR E8-XT | Temperatura de piezos |
| Medidor de Nivel Sonoro | Extech 407730 | Monitoreo acústico |
| Adquisición de Datos | NI USB-6009 | Registro multicanal |

### 2.4 Equipo de Seguridad

| Artículo | Especificación | Ubicación |
|----------|----------------|-----------|
| **Protección Auditiva** | NRR 30+ dB | Todo el personal |
| **Sonda de Descarga AV** | Capacidad 200V | Cerca de estación de prueba |
| **Extintor** | CO2, Clase C | Dentro de 3m |
| **Botiquín de Primeros Auxilios** | Industrial estándar | Sala de control |
| **Botón de Parada de Emergencia** | Cableado directo, hongo rojo | Consola Y cámara de prueba |

---

## 3. REQUISITOS DE SEGURIDAD

### 3.1 Requisitos de Personal

| Rol | Mínimo # | Responsabilidades |
|-----|----------|-------------------|
| **Director de Prueba** | 1 | Autoridad general, decisiones CONTINUAR/DETENER |
| **Operador de Prueba** | 1 | Operación de consola, registro de datos |
| **Oficial de Seguridad** | 1 | Monitorear térmico/acústico, autoridad de E-Stop |

```
⚠️  MÍNIMO 2 PERSONAS REQUERIDAS PARA CUALQUIER PRUEBA EN VIVO
⚠️  NINGÚN PERSONAL EN LA CÁMARA DE PRUEBA DURANTE LA OPERACIÓN
```

### 3.2 Protocolo de Seguridad Térmica

**Referencia:** Asesoría del Equipo Rojo §1 — Riesgo de Despolarización Térmica

| Parámetro | Límite | Acción si se Excede |
|-----------|--------|---------------------|
| Temperatura del Arreglo Piezo | < 90°C | E-STOP AUTOMÁTICO |
| Temperatura del Arreglo Piezo | < 70°C | ADVERTENCIA, reducir ciclo de trabajo |
| Temperatura Ambiente de Cámara | < 40°C | Pausar pruebas, ventilar |

**Límites Obligatorios de Ciclo de Trabajo:**

```
┌───────────────────────────────────────────────────────────────────┐
│  PROTOCOLO TÉRMICO MARK 1 (Hasta enfriamiento líquido en Mark 2)  │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  DISPARO:      5-10 segundos MÁXIMO                               │
│  ENFRIAMIENTO: 60 segundos MÍNIMO                                 │
│                                                                   │
│  Ciclo de Trabajo = 10s / 70s = 14.3% MÁXIMO                      │
│                                                                   │
│  El MCU debe aplicar esto automáticamente.                        │
│  La anulación manual está PROHIBIDA.                              │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.3 Protocolo de Seguridad Acústica

**Referencia:** Asesoría del Equipo Rojo §2 — Riesgos Acústicos

| Condición | Requisito |
|-----------|-----------|
| Frecuencia de Prueba 1-10 kHz | Protección auditiva NRR 30+ |
| Potencia > 10W | NINGÚN personal en cámara de prueba |
| Cualquier disparo en vivo | Operación remota SOLAMENTE |

**Monitoreo Acústico:**

```
Umbrales de Acción por Nivel Sonoro:
─────────────────────────────────
< 85 dB    Operación normal
85-100 dB  Protección auditiva obligatoria
100-120 dB Solo operación remota (cabina de control)
> 120 dB   E-STOP INMEDIATO — investigar
```

### 3.4 Protocolo de Seguridad Eléctrica

| Riesgo | Mitigación |
|--------|------------|
| Línea AV de 200V | Gabinete con interbloqueo, sonda de descarga |
| Descarga de Capacitores | Esperar 30 segundos después de apagar |
| Fallas a Tierra | Protección GFCI en todos los circuitos |

### 3.5 Lista de Verificación de Seguridad Pre-Prueba

```
┌─────────────────────────────────────────────────────────────────┐
│           VERIFICACIÓN DE SEGURIDAD PRE-PRUEBA                  │
│                                                                 │
│  □ 1. Todo el personal informado del plan de prueba de hoy      │
│  □ 2. Director de Prueba ha confirmado estado CONTINUAR         │
│  □ 3. Salidas de emergencia despejadas y señalizadas            │
│  □ 4. Botones E-Stop probados (ambas ubicaciones)               │
│  □ 5. Extintor inspeccionado y accesible                        │
│  □ 6. Protección auditiva distribuida                           │
│  □ 7. Sonda de descarga AV lista                                │
│  □ 8. Cámara térmica encendida y apuntando                      │
│  □ 9. Cabina de control aislada (puerta cerrada)                │
│  □ 10. Sistema de comunicación probado (intercomunicador/radio) │
│                                                                 │
│  Firma Director de Prueba: ___________________ Fecha: ______    │
│  Firma Oficial de Seguridad: _________________ Fecha: ______    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. CONFIGURACIÓN PREVIA A LA PRUEBA

### 4.1 Configuración Mecánica (Día 1 Mañana)

**Paso 1: Preparación de la Balanza de Torsión**

```
1.1  Nivelar la mesa de aislamiento (nivel de burbuja, las 4 esquinas)
1.2  Instalar balanza de torsión en el centro de la mesa
1.3  Verificar que la fibra/alambre esté sin daños y con tensión adecuada
1.4  Instalar sensor de desplazamiento óptico
1.5  Poner a cero el sensor con la balanza en reposo
1.6  Registrar temperatura ambiente: _______ °C
1.7  Registrar presión ambiente: _______ mbar
```

**Paso 2: Montaje del Mark 1**

```
⚠️  CRÍTICO: ¡NO montar la unidad de forma rígida!

2.1  Instalar almohadillas de aislamiento Sorbothane en el brazo de la balanza
     - 4× almohadillas, 10mm de espesor, Shore 50A
     - Propósito: Filtrar vibración acústica de la medición de empuje

2.2  Colocar la unidad Mark 1 sobre las almohadillas de aislamiento
2.3  Asegurar con sujeción ligera (no apretar excesivamente)
2.4  Verificar eje de empuje alineado con eje de sensibilidad de la balanza
2.5  Fotografiar configuración de montaje para registros
```

**Paso 3: Configuración de Cámara de Vacío (si aplica)**

```
3.1  Instalar escudo de policarbonato contra explosiones dentro de la cámara
3.2  Pasar cables a través de pasamuros de vacío
3.3  Verificar todos los sellos y O-rings
3.4  Conectar bomba de desbaste
3.5  Conectar bomba turbo (si se requiere alto vacío)
3.6  Instalar medidor de presión
```

### 4.2 Configuración Eléctrica (Día 1 Tarde)

**Paso 4: Conexiones de Alimentación**

```
4.1  Verificar fuente de 24V APAGADA
4.2  Conectar fuente de 24V a entrada del Mark 1
4.3  Conectar USB/UART a computadora de control
4.4  Conectar puntas del osciloscopio:
     - CH1: Salida DDS (forma de onda de referencia)
     - CH2: Canal piezo 1 (verificar amplificación)
4.5  Conectar alimentación de cámara térmica al monitor
4.6  Verificar E-Stop cableado en serie con fuente AV
```

**Paso 5: Verificación del Sistema de Control**

```
5.1  ENCENDER fuente de 24V
5.2  Verificar línea de 5V: _______ V (esperado: 5.0 ± 0.1V)
5.3  Verificar línea AV (sin carga): _______ V (esperado: 200 ± 5V)
5.4  Iniciar software de control
5.5  Verificar comunicación USB establecida
5.6  Leer todos los valores de sensores:
     - Temp 1: ___°C  Temp 2: ___°C  Temp 3: ___°C  Temp 4: ___°C
     - Acel X: ___g  Acel Y: ___g  Acel Z: ___g
5.7  Verificar interbloqueo térmico: Establecer umbral en 90°C
5.8  Verificar limitador de ciclo de trabajo: 10s ENCENDIDO / 60s APAGADO
```

---

## 5. PROCEDIMIENTOS DE CALIBRACIÓN

### 5.1 Calibración de Balanza de Torsión

**Objetivo:** Establecer factor de conversión nN/µm

**Procedimiento:**

```
CAL-1: Calibración Estática con Masas Conocidas

1. Registrar posición de línea base: X₀ = _______ µm
2. Aplicar masa de calibración de 1 mg en punto de empuje
   - Fuerza gravitacional: F = 9.81 µN
   - Registrar deflexión: X₁ = _______ µm
   - ΔX = X₁ - X₀ = _______ µm
   
3. Calcular sensibilidad: S = F / ΔX = _______ nN/µm

4. Repetir con masa de 10 mg:
   - F = 98.1 µN
   - ΔX = _______ µm
   - S = _______ nN/µm
   
5. Verificar linealidad (ambos valores de S deben coincidir dentro del 5%)

6. Registrar factor de calibración final:
   
   ┌─────────────────────────────────────┐
   │  FACTOR DE CALIBRACIÓN              │
   │  S = _________ nN/µm                │
   │  Fecha: __________                  │
   │  Técnico: __________                │
   └─────────────────────────────────────┘
```

### 5.2 Calibración de Respuesta Piezo

**Objetivo:** Verificar que los 8 canales respondan correctamente

**Procedimiento:**

```
CAL-2: Prueba de Canal Individual

Para cada canal P1 a P8:

1. Establecer frecuencia: 1 kHz
2. Establecer voltaje: 50V (25% potencia — seguro para prueba sostenida)
3. Establecer modo: Solo canal individual
4. Disparar por 1 segundo
5. Observar en osciloscopio:
   - Forma de onda: □ Correcta  □ Distorsionada
   - Amplitud: _______ V (esperado: 50 ± 2V)
   - Frecuencia: _______ Hz (esperado: 1000 ± 1 Hz)
6. Registrar aumento de temperatura piezo: ΔT = _______ °C

Resultados de Prueba de Canales:
┌────────┬─────────────┬───────────┬────────┬────────┐
│ Canal  │ Forma Onda  │ Amplitud  │ Frec   │ ΔT(°C) │
├────────┼─────────────┼───────────┼────────┼────────┤
│ P1     │ □OK □FALLA  │           │        │        │
│ P2     │ □OK □FALLA  │           │        │        │
│ P3     │ □OK □FALLA  │           │        │        │
│ P4     │ □OK □FALLA  │           │        │        │
│ P5     │ □OK □FALLA  │           │        │        │
│ P6     │ □OK □FALLA  │           │        │        │
│ P7     │ □OK □FALLA  │           │        │        │
│ P8     │ □OK □FALLA  │           │        │        │
└────────┴─────────────┴───────────┴────────┴────────┘

Todos los canales deben pasar. Cualquier falla = NO CONTINUAR con las pruebas.
```

### 5.3 Calibración de Alineación de Fase

**Objetivo:** Verificar generación de onda viajera para modo TPH

**Procedimiento:**

```
CAL-3: Prueba de Secuencia de Fase

1. Establecer modo: TPH
2. Establecer frecuencia: 1 kHz
3. Establecer voltaje: 50V
4. Conectar osciloscopio:
   - CH1: P1 (trigger)
   - CH2: P3 (desfase de 90° esperado)
   
5. Disparar por 1 segundo
6. Medir retardo de fase: Δφ = _______ ° (esperado: 90 ± 5°)

7. Repetir para P1 vs P5 (180° esperado): Δφ = _______ °
8. Repetir para P1 vs P7 (270° esperado): Δφ = _______ °

Alineación de fase: □ PASA (todos dentro de ±5°)  □ FALLA
```

---

## 6. SECUENCIAS DE PRUEBA

### 6.1 Fase 1: Verificación Eléctrica (Sin Empuje)

**Objetivo:** Confirmar todos los sistemas funcionales antes de generación de empuje

| ID Prueba | Descripción | Duración | Voltaje | Resultado Esperado |
|-----------|-------------|----------|---------|-------------------|
| EV-01 | Secuencia de encendido | — | — | Todos los LEDs, sensores activos |
| EV-02 | Prueba de comunicación | — | — | USB responde |
| EV-03 | Lectura térmica | 60s | 0V | Estable, < 30°C |
| EV-04 | Ping de canal individual | 100ms | 50V | Forma de onda en osciloscopio |
| EV-05 | Ping de todos los canales | 100ms | 50V | 8 formas de onda verificadas |
| EV-06 | Prueba de E-Stop | — | 50V | Corte instantáneo de energía |
| EV-07 | Interbloqueo térmico | Sim | — | MCU dispara a 90°C |

### 6.2 Fase 2: Detección de Empuje Atmosférico

**Objetivo:** Primera medición de empuje (arrastre de aire presente pero aceptable para detección)

```
⚠️  EL PERSONAL DEBE SALIR DE LA CÁMARA DE PRUEBA
⚠️  OPERACIÓN REMOTA SOLAMENTE DESDE ESTE PUNTO
```

**Secuencia de Prueba AT-01: Detección Modo OMV**

```
1. Evacuar cámara de prueba (solo personal, no vacío)
2. Sellar puerta de la cámara
3. Armar sistema desde cabina de control
4. Establecer parámetros:
   - Modo: OMV (senoidal continuo)
   - Frecuencia: 2 kHz
   - Voltaje: 100V (50% potencia)
   - Duración: 5 segundos

5. Registrar línea base pre-disparo:
   - Posición de balanza: X₀ = _______ µm
   - Temperatura piezo: T₀ = _______ °C
   - Nivel sonoro ambiente: _______ dB

6. Comando: DISPARAR

7. Durante el disparo, registrar:
   - Deflexión de balanza (en vivo): _______ µm
   - Nivel sonoro (pico): _______ dB
   
8. Después del disparo, registrar:
   - Posición de balanza (estabilizada): X₁ = _______ µm
   - Temperatura piezo: T₁ = _______ °C
   - Aumento de temperatura: ΔT = T₁ - T₀ = _______ °C

9. Calcular empuje:
   - Deflexión: ΔX = X₁ - X₀ = _______ µm
   - Empuje: F = ΔX × S = _______ nN

10. Esperar 60 segundos (enfriamiento obligatorio)

11. Repetir 3 veces para estadísticas
```

**Secuencia de Prueba AT-02: Detección Modo TPH**

```
Mismo procedimiento que AT-01, pero:
- Modo: TPH (pulsado)
- Frecuencia: 1 kHz
- Voltaje: 100V
- Duración: 5 segundos

Esperado: Deflexión tipo impulso seguida de decaimiento
```

### 6.3 Fase 3: Pruebas en Vacío

**Objetivo:** Medición de precisión sin interferencia aerodinámica

**Niveles de Vacío:**

| Nivel | Presión | Propósito |
|-------|---------|-----------|
| Desbaste | 10⁻¹ Torr | Eliminar convección |
| Medio | 10⁻³ Torr | Eliminar la mayor parte del arrastre |
| Alto | 10⁻⁶ Torr | Precisión máxima |

**Secuencia de Prueba VT-01: Línea Base en Vacío**

```
1. Bombear cámara a 10⁻³ Torr
2. Esperar 10 minutos para equilibrio térmico
3. Registrar piso de ruido de línea base:
   - Deriva de balanza en 60s: _______ µm
   - Ruido RMS: _______ µm
   - Ruido de fuerza equivalente: _______ nN

4. Esto establece el umbral de detección
```

**Secuencia de Prueba VT-02: Empuje OMV en Vacío**

```
1. Verificar presión: < 10⁻³ Torr
2. Establecer parámetros:
   - Modo: OMV
   - Frecuencia: 2 kHz  
   - Voltaje: 150V (75% potencia)
   - Duración: 10 segundos

3. Disparar y registrar:
   - Deflexión pico: _______ µm
   - Deflexión sostenida: _______ µm
   - Empuje: _______ nN

4. Comparar con prueba atmosférica — la señal debería ser más limpia
```

### 6.4 Fase 4: Barrido Paramétrico

**Objetivo:** Caracterización completa de dependencias del empuje

**Matriz de Prueba:**

```
BARRIDO DE FRECUENCIA (V fijo = 100V, Modo OMV)
┌──────────┬──────────┬──────────┬──────────┐
│ f (kHz)  │ ΔX (µm)  │ F (nN)   │ Temp (°C)│
├──────────┼──────────┼──────────┼──────────┤
│ 0.5      │          │          │          │
│ 1.0      │          │          │          │
│ 2.0      │          │          │          │
│ 5.0      │          │          │          │
│ 10.0     │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
Esperado: Relación lineal F ∝ f

BARRIDO DE VOLTAJE (f fijo = 2 kHz, Modo OMV)
┌──────────┬──────────┬──────────┬──────────┐
│ V (V)    │ ΔX (µm)  │ F (nN)   │ Temp (°C)│
├──────────┼──────────┼──────────┼──────────┤
│ 50       │          │          │          │
│ 100      │          │          │          │
│ 150      │          │          │          │
│ 200      │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
Esperado: Relación cuadrática F ∝ V²

COMPARACIÓN DE MODOS (f fijo = 2 kHz, V = 150V)
┌──────────┬──────────┬──────────┬──────────┐
│ Modo     │ ΔX (µm)  │ F (nN)   │ Carácter │
├──────────┼──────────┼──────────┼──────────┤
│ OMV      │          │          │ DC       │
│ TPH      │          │          │ Impulso  │
│ Híbrido  │          │          │ Ambos    │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 7. RECOLECCIÓN DE DATOS

### 7.1 Canales de Datos Requeridos

| Canal | Sensor | Tasa de Muestreo | Unidades |
|-------|--------|------------------|----------|
| CH1 | Posición de balanza | 1 kHz | µm |
| CH2 | Temp Piezo 1 | 10 Hz | °C |
| CH3 | Temp Piezo 2 | 10 Hz | °C |
| CH4 | Temp Piezo 3 | 10 Hz | °C |
| CH5 | Temp Piezo 4 | 10 Hz | °C |
| CH6 | Presión de cámara | 1 Hz | Torr |
| CH7 | Temp ambiente | 1 Hz | °C |
| CH8 | Nivel sonoro | 100 Hz | dB |
| CH9 | Acel X | 1 kHz | g |
| CH10 | Acel Y | 1 kHz | g |
| CH11 | Acel Z | 1 kHz | g |
| CH12 | Voltaje de comando | 10 kHz | V |

### 7.2 Formato de Archivo de Datos

```
Nombre de archivo: AETHERION_MK1_PRUEBA_{FECHA}_{ID_SECUENCIA}.csv

Encabezado:
# Datos de Prueba Aetherion Mark 1
# Fecha: AAAA-MM-DD HH:MM:SS
# ID de Prueba: {ID_SECUENCIA}
# Modo: {TPH/OMV/HIBRIDO}
# Frecuencia: {f} kHz
# Voltaje: {V} V
# Duración: {t} s
# Presión: {P} Torr
# Factor de Calibración: {S} nN/µm

Columnas:
marca_tiempo_ms, posicion_um, temp1_C, temp2_C, temp3_C, temp4_C, 
presion_torr, ambiente_C, sonido_dB, acel_x_g, acel_y_g, acel_z_g, cmd_V
```

### 7.3 Registro Obligatorio

Cada prueba DEBE registrar:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRADA DE BITÁCORA                      │
├─────────────────────────────────────────────────────────────┤
│ Fecha: ____________  Hora: ____________                     │
│ ID de Prueba: ____________                                  │
│ Director de Prueba: ____________                            │
│ Operador: ____________                                      │
│                                                             │
│ Parámetros:                                                 │
│   Modo: □ OMV  □ TPH  □ Híbrido                             │
│   Frecuencia: ________ kHz                                  │
│   Voltaje: ________ V                                       │
│   Duración: ________ s                                      │
│   Presión de Cámara: ________ Torr                          │
│                                                             │
│ Resultados:                                                 │
│   Deflexión: ________ µm                                    │
│   Empuje Calculado: ________ nN                             │
│   Temp Piezo Pico: ________ °C                              │
│   Nivel Sonoro Pico: ________ dB                            │
│                                                             │
│ Anomalías: _____________________________________________    │
│ __________________________________________________________  │
│                                                             │
│ Archivo de Datos: ________________________________________  │
│                                                             │
│ Firmas:                                                     │
│   Director de Prueba: _________________ Fecha: __________   │
│   Operador: _________________ Fecha: __________             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. CRITERIOS DE APROBACIÓN/RECHAZO

### 8.1 Criterios de Éxito Primarios

| ID | Criterio | Umbral | Resultado |
|----|----------|--------|-----------|
| **P1** | Empuje OMV medible | > 50 nN @ 2kHz/150V | □ PASA □ FALLA |
| **P2** | Impulso TPH medible | Deflexión visible | □ PASA □ FALLA |
| **P3** | Empuje escala con frecuencia | R² > 0.9 (lineal) | □ PASA □ FALLA |
| **P4** | Empuje escala con voltaje | R² > 0.9 (cuadrático) | □ PASA □ FALLA |
| **P5** | Estabilidad térmica | T máx < 90°C | □ PASA □ FALLA |
| **P6** | Repetibilidad | σ/μ < 20% | □ PASA □ FALLA |

### 8.2 Criterios de Éxito Secundarios

| ID | Criterio | Umbral | Resultado |
|----|----------|--------|-----------|
| **S1** | SNR Vacío vs Atmosférico | > 3× mejora | □ PASA □ FALLA |
| **S2** | Superioridad modo híbrido | F_híbrido > F_OMV | □ PASA □ FALLA |
| **S3** | Dirección controlable | 8 vectores distintos | □ PASA □ FALLA |

### 8.3 Condiciones de Falla Automática

```
🛑 TERMINACIÓN INMEDIATA DE PRUEBA SI:

• Temperatura piezo excede 90°C
• Nivel sonoro excede 130 dB
• Humo, chispas o llamas visibles
• Grieta o ruptura de cámara de vacío
• Fibra de balanza se rompe
• Cualquier personal en cámara durante disparo
• Pérdida de comunicación con cabina de control
```

### 8.4 Predicciones Teóricas

La RTM predice los siguientes valores de empuje (para comparación):

| Modo | Frecuencia | Voltaje | Empuje Predicho |
|------|------------|---------|-----------------|
| OMV | 2 kHz | 150V | ~150-300 nN |
| OMV | 2 kHz | 200V | ~200-500 nN |
| TPH | 1 kHz | 150V | ~100 nN promedio |
| TPH | 10 kHz | 200V | ~500 nN promedio |

```
Si el empuje medido es:
  • Dentro de 50-200% de la predicción → VALIDACIÓN FUERTE
  • Dentro de 10-500% de la predicción → VALIDACIÓN PARCIAL (investigar)
  • Fuera de 10-500% → INVESTIGAR SISTEMÁTICOS
  • Cero o negativo → REVISAR CONFIGURACIÓN (no invalida la teoría)
```

---

## 9. PROCEDIMIENTOS DE EMERGENCIA

### 9.1 Activación de E-Stop

```
CUÁNDO PRESIONAR E-STOP:
• Temperatura excede 90°C (si el interbloqueo automático falla)
• Daño visible al equipo
• Ruido o vibración fuerte inesperado
• Fuego o humo
• Emergencia de personal
• Pérdida de respuesta del sistema de control

PROCEDIMIENTO E-STOP:
1. Presionar botón rojo tipo hongo (cualquier ubicación)
2. TODA LA ENERGÍA se corta inmediatamente
3. Anunciar "E-STOP ACTIVADO" por intercomunicador
4. Esperar 30 segundos (descarga de capacitores)
5. NO entrar a la cámara hasta autorización
6. Documentar razón en bitácora de prueba
```

### 9.2 Descontrol Térmico

```
SI LA TEMPERATURA PIEZO EXCEDE 100°C:

1. Presionar E-STOP inmediatamente
2. NO abrir la cámara (riesgo de choque térmico)
3. Esperar 10 minutos para enfriamiento pasivo
4. Monitorear cámara térmica para decaimiento
5. Cuando T < 50°C, es seguro acercarse
6. Inspeccionar arreglo piezo por daños
7. Si se sospecha despolarización:
   - Probar respuesta piezo a bajo voltaje
   - Si no hay respuesta, reemplazar arreglo
```

### 9.3 Emergencia Acústica

```
SI EL NIVEL SONORO EXCEDE 120 dB INESPERADAMENTE:

1. Presionar E-STOP inmediatamente
2. NO entrar a la cámara
3. Verificar daños inducidos por resonancia:
   - Integridad de cámara de vacío
   - Agrietamiento de campana de vidrio
   - Fibra de balanza
4. Si ocurrió falla de vidrio:
   - Evacuar área inmediata
   - No tocar vidrio roto bajo estrés de vacío
   - Llamar al equipo de seguridad
```

### 9.4 Incendio

```
SI SE OBSERVA FUEGO O HUMO:

1. Presionar E-STOP
2. Evacuar a todo el personal
3. Si es pequeño y contenido: extintor de CO2
4. Si se propaga: Evacuar el edificio, llamar a bomberos
5. NO usar agua en fuego eléctrico
```

### 9.5 Lesión de Personal

```
SI SE SOSPECHA DAÑO AUDITIVO:
1. Retirar a la persona del ambiente acústico
2. No gritarle (daño adicional)
3. Buscar atención médica
4. Documentar incidente

SI HAY DESCARGA ELÉCTRICA:
1. NO tocar a la víctima si aún está en contacto
2. Cortar energía si es seguro hacerlo
3. Llamar a servicios de emergencia
4. Si está capacitado, iniciar RCP si es necesario
```

---

## 10. PROCEDIMIENTOS POST-PRUEBA

### 10.1 Post-Prueba Inmediata

```
DESPUÉS DE CADA DISPARO DE PRUEBA:

□ 1. Esperar 60 segundos mínimo (enfriamiento)
□ 2. Verificar temperatura piezo < 50°C antes de siguiente prueba
□ 3. Registrar todos los datos del DAQ
□ 4. Guardar archivo de datos con convención de nombres apropiada
□ 5. Llenar entrada de bitácora de prueba
□ 6. Verificar cualquier anomalía
```

### 10.2 Procedimientos de Fin de Día

```
FIN DE SESIÓN DE PRUEBA:

□ 1. Completar entrada final de bitácora
□ 2. Apagar fuente AV
□ 3. Apagar fuente de 24V
□ 4. Si hay vacío: ventilar cámara lentamente
□ 5. Abrir puerta de cámara
□ 6. Inspección visual de unidad Mark 1
□ 7. Fotografiar cualquier desgaste o daño
□ 8. Asegurar todos los archivos de datos (respaldo en nube)
□ 9. Restablecer E-Stop y verificación del sistema de seguridad
□ 10. Cerrar laboratorio con llave
```

### 10.3 Análisis Post-Campaña

Después de completar todas las fases de prueba:

```
LISTA DE VERIFICACIÓN DE ANÁLISIS:

□ Compilar todos los archivos de datos
□ Graficar empuje vs frecuencia (verificar F ∝ f)
□ Graficar empuje vs voltaje (verificar F ∝ V²)
□ Calcular media y desviación estándar para cada condición
□ Comparar con predicciones teóricas RTM
□ Identificar cualquier error sistemático
□ Documentar lecciones aprendidas
□ Preparar informe final de prueba
```

### 10.4 Plantilla de Informe Final

```
INFORME DE CAMPAÑA DE PRUEBAS AETHERION MARK 1

1. Resumen Ejecutivo
   - Hallazgos clave
   - Aprobación/Rechazo en criterios primarios
   
2. Configuración de Prueba
   - Equipo utilizado
   - Resultados de calibración
   
3. Resultados
   - Mediciones de empuje (todas las condiciones)
   - Verificación de leyes de escalamiento
   - Desempeño térmico
   
4. Análisis
   - Comparación con predicciones RTM
   - Análisis de error
   - Incertidumbres sistemáticas
   
5. Conclusiones
   - ¿Los datos apoyan el empuje ponderomotriz RTM?
   - Recomendaciones para Mark 2
   
6. Apéndices
   - Todos los archivos de datos crudos
   - Registros de calibración
   - Bitácoras de prueba
   - Fotografías
```

---

## APÉNDICE A: Tarjetas de Referencia Rápida

### A.1 Comandos de Consola

```
COMANDOS DE SOFTWARE DE CONTROL:

arm                     # Armar sistema para disparo
disarm                  # Desarmar sistema
fire <duracion_ms>      # Disparar por duración especificada
set mode <omv|tph|hibrido>
set freq <Hz>
set voltage <V>
set phase <ch> <deg>    # Establecer fase de canal individual
status                  # Imprimir todas las lecturas de sensores
temp                    # Imprimir temperaturas piezo
estop                   # E-stop por software
reset                   # Restablecer después de E-stop
log start <nombrearchivo>
log stop
help
```

### A.2 Secuencia de Disparo Normal

```
LISTA DE VERIFICACIÓN DE DISPARO ESTÁNDAR:

1. Verificar cámara sellada
2. Verificar personal fuera
3. > arm
4. > set mode omv
5. > set freq 2000
6. > set voltage 150
7. > log start prueba_001.csv
8. > fire 5000
9. [Esperar finalización]
10. > log stop
11. > disarm
12. Esperar 60s de enfriamiento
```

### A.3 Contactos de Emergencia

```
┌─────────────────────────────────────────┐
│       CONTACTOS DE EMERGENCIA           │
├─────────────────────────────────────────┤
│ Emergencia Bomberos/Médica: 911         │
│ Seguridad de Instalaciones: [INSERTAR]  │
│ Gerente de Laboratorio: [INSERTAR]      │
│ Líder de Proyecto: [INSERTAR]           │
│ Control de Intoxicaciones: [INSERTAR]   │
└─────────────────────────────────────────┘
```

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      FIN DEL PROTOCOLO DE PRUEBAS                            ║
║                                                                              ║
║                    AETHERION MARK 1 — ATP-MK1-001                            ║
║                           Revisión 1.0                                       ║
║                                                                              ║
║              "El tiempo no es lo que pasa, sino lo que pulsa."               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
