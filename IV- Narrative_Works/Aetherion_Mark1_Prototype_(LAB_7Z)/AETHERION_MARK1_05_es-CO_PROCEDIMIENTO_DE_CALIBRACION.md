# AETHERION MARK 1
## Manual de Procedimientos de Calibración

**ID del Documento:** ATP-MK1-CAL-001  
**Revisión:** 1.0  
**Clasificación:** OPERACIONAL  
**Fecha:** Febrero 2026  

---

## TABLA DE CONTENIDOS

1. [Descripción General](#1-descripción-general)
2. [Equipo Requerido](#2-equipo-requerido)
3. [CAL-1: Calibración de Balanza de Torsión](#3-cal-1-calibración-de-balanza-de-torsión)
4. [CAL-2: Verificación del Arreglo Piezoeléctrico](#4-cal-2-verificación-del-arreglo-piezoeléctrico)
5. [CAL-3: Calibración de Alineación de Fase](#5-cal-3-calibración-de-alineación-de-fase)
6. [CAL-4: Calibración de Sensores de Temperatura](#6-cal-4-calibración-de-sensores-de-temperatura)
7. [CAL-5: Calibración de Frecuencia DDS](#7-cal-5-calibración-de-frecuencia-dds)
8. [CAL-6: Calibración de Amplificador AV](#8-cal-6-calibración-de-amplificador-av)
9. [Programa de Calibración](#9-programa-de-calibración)
10. [Registros de Calibración](#10-registros-de-calibración)

---

## 1. DESCRIPCIÓN GENERAL

### 1.1 Propósito

Este documento establece los procedimientos de calibración para todos los sistemas de medición y control del prototipo Aetherion Mark 1. Una calibración adecuada asegura:

- Mediciones precisas de empuje
- Interbloqueos de seguridad térmica confiables
- Señales de excitación piezoeléctricas correctas
- Resultados de prueba repetibles

### 1.2 Jerarquía de Calibración

```
┌─────────────────────────────────────────────────────────────┐
│                   CADENA DE CALIBRACIÓN                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Estándares Trazables NIST                                  │
│         ↓                                                   │
│  Instrumentos de Referencia del Laboratorio                 │
│         ↓                                                   │
│  Accesorios de Calibración Aetherion                        │
│         ↓                                                   │
│  Sensores y Actuadores Instalados                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Definiciones de Estado de Calibración

| Estado | Definición | Acción |
|--------|------------|--------|
| **CALIBRADO** | Dentro de especificación | Proceder con pruebas |
| **VENCIDO** | Intervalo de calibración expirado | Recalibrar antes de usar |
| **FUERA DE TOLERANCIA** | Falló calibración | Reparar/reemplazar antes de usar |
| **SOLO REFERENCIA** | No para uso cuantitativo | Documentar limitación |

---

## 2. EQUIPO REQUERIDO

### 2.1 Estándares de Calibración

| Artículo | Especificación | Intervalo de Calibración |
|----------|----------------|--------------------------|
| Juego de Masas (ASTM Clase 4) | 1mg a 100g | 12 meses |
| Termómetro de Precisión | ±0.1°C, trazable NIST | 12 meses |
| Contador de Frecuencia | Resolución 10 Hz, 0.1 ppm | 12 meses |
| Multímetro Digital | 6½ dígitos, trazable NIST | 12 meses |
| Osciloscopio | 200 MHz, calibrado | 12 meses |

### 2.2 Accesorios de Calibración

| Artículo | Descripción | Referencia BOM |
|----------|-------------|----------------|
| Juego de Masas de Calibración | 1mg, 10mg, 100mg, 1g, 10g | DOC-002 |
| Referencia de Punto de Hielo | Baño de hielo 0.00°C | — |
| Referencia de Punto de Ebullición | Punto de vapor 100.0°C | — |
| Carga de Prueba Piezo | Resistiva 10 MΩ | — |

---

## 3. CAL-1: CALIBRACIÓN DE BALANZA DE TORSIÓN

### 3.1 Propósito

Establecer el factor de sensibilidad fuerza-desplazamiento (S) en nN/µm.

### 3.2 Frecuencia

- **Inicial:** Antes del primer uso
- **Periódica:** Cada campaña de pruebas o mensualmente
- **Por evento:** Después de cualquier ajuste de balanza o reemplazo de fibra

### 3.3 Prerrequisitos

- [ ] Balanza instalada en mesa de aislamiento
- [ ] Sensor óptico en cero
- [ ] Vibración ambiental < 1 µm RMS
- [ ] Temperatura estable (±1°C durante 1 hora)

### 3.4 Procedimiento

```
CAL-1: CALIBRACIÓN ESTÁTICA DE BALANZA DE TORSIÓN
═══════════════════════════════════════════════════════════════

PASO 1: MEDICIÓN DE LÍNEA BASE
──────────────────────────────
1.1  Permitir que la balanza se estabilice durante 30 minutos
1.2  Registrar posición de línea base durante 60 segundos
1.3  Calcular media de línea base: X₀ = _______ µm
1.4  Calcular ruido RMS: σ₀ = _______ µm
1.5  Fuerza de piso de ruido: F_ruido = σ₀ × S_nominal = _______ nN

     Aceptación: σ₀ < 0.5 µm (si se usa 10 nN/µm nominal)

PASO 2: APLICAR MASAS DE CALIBRACIÓN
────────────────────────────────────
Para cada masa de calibración (m):

2.1  Calcular fuerza gravitacional: F = m × g
     (Usar valor local de g, típicamente 9.80665 m/s²)
     
2.2  Colocar cuidadosamente la masa en el punto de aplicación de empuje
2.3  Esperar 30 segundos para estabilización
2.4  Registrar posición deflectada durante 30 segundos
2.5  Calcular media: X_m = _______ µm
2.6  Remover masa, verificar retorno a línea base

TABLA DE DATOS:
┌────────────┬───────────┬───────────┬───────────┬───────────┐
│ Masa (mg)  │ Fuerza(nN)│ X₀ (µm)   │ X_m (µm)  │ ΔX (µm)   │
├────────────┼───────────┼───────────┼───────────┼───────────┤
│ 1.000      │ 9.807     │           │           │           │
│ 10.00      │ 98.07     │           │           │           │
│ 100.0      │ 980.7     │           │           │           │
│ 1000       │ 9807      │           │           │           │
│ 10000      │ 98070     │           │           │           │
└────────────┴───────────┴───────────┴───────────┴───────────┘

PASO 3: CALCULAR SENSIBILIDAD
─────────────────────────────
3.1  Graficar F vs ΔX (debe ser lineal)
3.2  Realizar regresión lineal: F = S × ΔX + b
3.3  Sensibilidad: S = _______ nN/µm
3.4  Intercepto: b = _______ nN (debe ser ~0)
3.5  R² = _______ (aceptación: R² > 0.9999)

PASO 4: VERIFICAR LINEALIDAD
────────────────────────────
4.1  Calcular residuos para cada punto
4.2  Residuo máximo: _______ nN
4.3  Residuo < 1% de escala completa: □ PASA  □ FALLA

PASO 5: DOCUMENTAR RESULTADOS
─────────────────────────────
┌─────────────────────────────────────────────────────────────┐
│          CERTIFICADO DE CALIBRACIÓN DE BALANZA DE TORSIÓN   │
├─────────────────────────────────────────────────────────────┤
│ Fecha: ________________  Hora: ________________             │
│ Técnico: ________________                                   │
│                                                             │
│ RESULTADOS:                                                 │
│   Sensibilidad (S): _____________ nN/µm                     │
│   Piso de Ruido: _____________ nN                           │
│   Linealidad (R²): _____________                            │
│   Rango de Medición: 0 a _____________ nN                   │
│                                                             │
│ ESTADO:  □ CALIBRADO   □ FUERA DE TOLERANCIA                │
│                                                             │
│ Próxima Calibración: ________________                       │
│                                                             │
│ Firma: _______________________                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 Criterios de Aceptación

| Parámetro | Requisito |
|-----------|-----------|
| Linealidad (R²) | > 0.9999 |
| Piso de ruido | < 10 nN |
| Histéresis | < 2% |
| Deriva de cero | < 5 nN/hora |

---

## 4. CAL-2: VERIFICACIÓN DEL ARREGLO PIEZOELÉCTRICO

### 4.1 Propósito

Verificar que los 8 canales piezoeléctricos respondan correctamente a las señales de excitación.

### 4.2 Frecuencia

- **Inicial:** Antes del primer uso
- **Periódica:** Antes de cada campaña de pruebas
- **Por evento:** Después de cualquier evento térmico o sospecha de despolarización

### 4.3 Procedimiento

```
CAL-2: VERIFICACIÓN DE CANALES PIEZOELÉCTRICOS
═══════════════════════════════════════════════════════════════

CONFIGURACIÓN DE EQUIPO:
────────────────────────
- Osciloscopio CH1: Salida de referencia DDS
- Osciloscopio CH2: Canal piezo bajo prueba
- Cámara térmica apuntando al arreglo piezo
- Temperatura inicial de piezo: T₀ = _______ °C

PARÁMETROS DE PRUEBA:
─────────────────────
- Frecuencia: 1 kHz (frecuencia de prueba segura)
- Voltaje: 50 V (25% potencia - seguro para prueba extendida)
- Duración: 1 segundo por canal

PROCEDIMIENTO:
──────────────
Para cada canal P1 a P8:

1. Seleccionar modo de canal individual
2. Aplicar señal de prueba durante 1 segundo
3. Capturar forma de onda en osciloscopio
4. Medir y registrar:
   - Amplitud de salida
   - Forma de onda
   - Fase relativa a referencia
   - Aumento de temperatura

TABLA DE DATOS:
┌─────────┬───────────┬───────────┬───────────┬───────────┬────────┐
│ Canal   │ Amplitud  │ Forma de  │ Fase (°)  │ ΔT (°C)   │ Estado │
│         │ (V p-p)   │ Onda      │           │           │        │
├─────────┼───────────┼───────────┼───────────┼───────────┼────────┤
│ P1      │           │ □OK □MAL  │           │           │ □P □F  │
│ P2      │           │ □OK □MAL  │           │           │ □P □F  │
│ P3      │           │ □OK □MAL  │           │           │ □P □F  │
│ P4      │           │ □OK □MAL  │           │           │ □P □F  │
│ P5      │           │ □OK □MAL  │           │           │ □P □F  │
│ P6      │           │ □OK □MAL  │           │           │ □P □F  │
│ P7      │           │ □OK □MAL  │           │           │ □P □F  │
│ P8      │           │ □OK □MAL  │           │           │ □P □F  │
└─────────┴───────────┴───────────┴───────────┴───────────┴────────┘

CRITERIOS DE ACEPTACIÓN:
────────────────────────
- Amplitud: 50 ± 2 V (dentro del 4%)
- Forma de onda: Senoidal limpia, sin distorsión
- Fase: Según programado ± 5°
- ΔT: < 5°C para prueba de 1 segundo
- Los 8 canales deben PASAR

VERIFICACIÓN DE DESPOLARIZACIÓN:
────────────────────────────────
Si algún canal muestra:
- Amplitud < 40 V (>20% pérdida)
- Distorsión severa de forma de onda
- Sin salida

→ El canal puede estar despolarizado. Reemplazar elemento piezo.
```

### 4.4 Criterios de Aceptación

| Parámetro | Requisito | Acción si Falla |
|-----------|-----------|-----------------|
| Amplitud | 50 ± 2 V | Verificar amplificador, cableado |
| Forma de onda | Senoidal limpia | Verificar cortocircuitos, daños |
| Todos los canales funcionales | 8/8 | Reemplazar elementos fallidos |

---

## 5. CAL-3: CALIBRACIÓN DE ALINEACIÓN DE FASE

### 5.1 Propósito

Verificar relaciones de fase correctas para generación de onda viajera TPH.

### 5.2 Procedimiento

```
CAL-3: VERIFICACIÓN DE ALINEACIÓN DE FASE
═══════════════════════════════════════════════════════════════

REQUISITOS DE FASE MODO TPH:
────────────────────────────
Espaciado entre canales: 45° (360° / 8 canales)

Fases esperadas:
  P1: 0°    P2: 45°   P3: 90°   P4: 135°
  P5: 180°  P6: 225°  P7: 270°  P8: 315°

PROCEDIMIENTO DE MEDICIÓN:
──────────────────────────
1. Establecer modo: TPH
2. Establecer frecuencia: 1 kHz
3. Establecer voltaje: 50 V
4. Conectar osciloscopio:
   - CH1: P1 (referencia de trigger)
   - CH2: Canal bajo prueba

5. Medir retardo de tiempo (Δt) entre flancos de subida
6. Calcular fase: φ = (Δt / T) × 360°
   donde T = 1/f = 1 ms a 1 kHz

TABLA DE DATOS:
┌─────────┬───────────┬───────────┬───────────┬───────────┬────────┐
│ Canal   │ Fase      │ Fase      │ Δt (µs)   │ Error (°) │ Estado │
│         │ Esperada  │ Medida (°)│           │           │        │
├─────────┼───────────┼───────────┼───────────┼───────────┼────────┤
│ P1      │ 0         │ 0 (ref)   │ 0         │ 0         │ REF    │
│ P2      │ 45        │           │           │           │ □P □F  │
│ P3      │ 90        │           │           │           │ □P □F  │
│ P4      │ 135       │           │           │           │ □P □F  │
│ P5      │ 180       │           │           │           │ □P □F  │
│ P6      │ 225       │           │           │           │ □P □F  │
│ P7      │ 270       │           │           │           │ □P □F  │
│ P8      │ 315       │           │           │           │ □P □F  │
└─────────┴───────────┴───────────┴───────────┴───────────┴────────┘

ACEPTACIÓN: Error de fase < ±5° para todos los canales

PROCEDIMIENTO DE AJUSTE (si es necesario):
──────────────────────────────────────────
1. Conectar al MCU vía UART
2. Usar comando: phase <canal> <grados>
3. Ejemplo: phase 2 47  (ajusta P2 a 47°)
4. Volver a medir e iterar hasta estar dentro de especificación
```

---

## 6. CAL-4: CALIBRACIÓN DE SENSORES DE TEMPERATURA

### 6.1 Propósito

Calibrar sensores RTD PT1000 para monitoreo térmico preciso.

### 6.2 Procedimiento

```
CAL-4: CALIBRACIÓN DE TEMPERATURA PT1000
═══════════════════════════════════════════════════════════════

PUNTOS DE REFERENCIA:
─────────────────────
- Punto de hielo: 0.00°C (baño de agua con hielo)
- Ambiente: ~25°C (termómetro de referencia)
- Punto caliente: ~50°C (baño de agua calentada)

EQUIPO:
───────
- Termómetro de referencia trazable NIST (±0.1°C)
- Baño de agua con hielo (0.00 ± 0.02°C)
- Baño de agua con temperatura controlada
- Mecanismo de agitación

PROCEDIMIENTO:
──────────────
Para cada sensor (SN-001 a SN-004):

1. PUNTO DE HIELO (0°C)
   1.1 Sumergir sensor y referencia en baño de hielo
   1.2 Esperar 5 minutos para equilibrio
   1.3 Registrar referencia: T_ref = _______ °C
   1.4 Registrar sensor: T_sensor = _______ °C
   1.5 Error a 0°C: _______ °C

2. PUNTO AMBIENTE (~25°C)
   2.1 Colocar sensor y referencia a temperatura ambiente
   2.2 Esperar 10 minutos para equilibrio
   2.3 Registrar referencia: T_ref = _______ °C
   2.4 Registrar sensor: T_sensor = _______ °C
   2.5 Error a 25°C: _______ °C

3. PUNTO CALIENTE (~50°C)
   3.1 Sumergir en baño de agua calentada a 50°C
   3.2 Esperar 5 minutos para equilibrio
   3.3 Registrar referencia: T_ref = _______ °C
   3.4 Registrar sensor: T_sensor = _______ °C
   3.5 Error a 50°C: _______ °C

4. CALCULAR CORRECCIÓN
   4.1 Graficar sensor vs referencia
   4.2 Ajustar corrección lineal: T_corregida = a × T_sensor + b
   4.3 Registrar coeficientes: a = _______, b = _______ °C

TABLA DE DATOS:
┌────────┬───────────┬───────────┬───────────┬───────────┬────────┐
│ Sensor │ Error @0° │ Error @25°│ Error @50°│ Error Máx │ Estado │
├────────┼───────────┼───────────┼───────────┼───────────┼────────┤
│ T1     │           │           │           │           │ □P □F  │
│ T2     │           │           │           │           │ □P □F  │
│ T3     │           │           │           │           │ □P □F  │
│ T4     │           │           │           │           │ □P □F  │
└────────┴───────────┴───────────┴───────────┴───────────┴────────┘

ACEPTACIÓN: Error máximo < ±1.0°C en el rango 0-50°C

Nota: El umbral crítico es 90°C. Si los sensores no pueden verificarse
a esta temperatura, aplicar compensación conservadora en firmware.
```

---

## 7. CAL-5: CALIBRACIÓN DE FRECUENCIA DDS

### 7.1 Propósito

Verificar que el DDS AD9910 genere frecuencias precisas.

### 7.2 Procedimiento

```
CAL-5: CALIBRACIÓN DE FRECUENCIA DDS
═══════════════════════════════════════════════════════════════

EQUIPO:
───────
- Contador de frecuencia (resolución 10 Hz, precisión 0.1 ppm)
- Osciloscopio para verificación de forma de onda

FRECUENCIAS DE PRUEBA:
──────────────────────
Cubrir rango operacional: 100 Hz a 50 kHz

PROCEDIMIENTO:
──────────────
1. Conectar contador de frecuencia a salida DDS
2. Para cada frecuencia de prueba:
   2.1 Comandar frecuencia vía UART: freq <Hz>
   2.2 Esperar 1 segundo para estabilización
   2.3 Registrar lectura del contador
   2.4 Calcular error

TABLA DE DATOS:
┌────────────┬────────────┬────────────┬────────────┬────────┐
│ Comandada  │ Medida     │ Error      │ Error      │ Estado │
│ (Hz)       │ (Hz)       │ (Hz)       │ (ppm)      │        │
├────────────┼────────────┼────────────┼────────────┼────────┤
│ 100        │            │            │            │ □P □F  │
│ 500        │            │            │            │ □P □F  │
│ 1000       │            │            │            │ □P □F  │
│ 2000       │            │            │            │ □P □F  │
│ 5000       │            │            │            │ □P □F  │
│ 10000      │            │            │            │ □P □F  │
│ 20000      │            │            │            │ □P □F  │
│ 50000      │            │            │            │ □P □F  │
└────────────┴────────────┴────────────┴────────────┴────────┘

ACEPTACIÓN: Error < 100 ppm (0.01%) en todo el rango
```

---

## 8. CAL-6: CALIBRACIÓN DE AMPLIFICADOR AV

### 8.1 Propósito

Verificar que los amplificadores PA94 entreguen el voltaje de salida correcto.

### 8.2 Procedimiento

```
CAL-6: CALIBRACIÓN DE AMPLIFICADOR AV
═══════════════════════════════════════════════════════════════

⚠️  ALTO VOLTAJE - Usar procedimientos de seguridad apropiados

EQUIPO:
───────
- Sonda AV (1000:1, calibrada)
- Osciloscopio
- Carga ficticia resistiva (10 kΩ, 25W)

PROCEDIMIENTO:
──────────────
1. Desconectar arreglo piezo
2. Conectar carga ficticia a salida del amplificador
3. Conectar sonda AV a la salida
4. Para cada voltaje comandado:
   4.1 Establecer voltaje vía UART: voltage <V>
   4.2 Establecer frecuencia: 1 kHz
   4.3 Disparar durante 1 segundo
   4.4 Medir salida pico a pico
   4.5 Calcular RMS: V_rms = V_pp / (2√2)

TABLA DE DATOS:
┌────────────┬────────────┬────────────┬────────────┬────────┐
│ Comandado  │ Medido     │ Error      │ Error      │ Estado │
│ (V)        │ V_pp (V)   │ (V)        │ (%)        │        │
├────────────┼────────────┼────────────┼────────────┼────────┤
│ 50         │            │            │            │ □P □F  │
│ 100        │            │            │            │ □P □F  │
│ 150        │            │            │            │ □P □F  │
│ 200        │            │            │            │ □P □F  │
└────────────┴────────────┴────────────┴────────────┴────────┘

ACEPTACIÓN: Error < ±5% en todo el rango

COINCIDENCIA ENTRE CANALES:
───────────────────────────
Repetir para los 8 canales con configuración de 150V:

┌─────────┬────────────┬────────────┬────────┐
│ Canal   │ V Medido   │ Desviación │ Estado │
├─────────┼────────────┼────────────┼────────┤
│ CH1     │            │            │ □P □F  │
│ CH2     │            │            │ □P □F  │
│ CH3     │            │            │ □P □F  │
│ CH4     │            │            │ □P □F  │
│ CH5     │            │            │ □P □F  │
│ CH6     │            │            │ □P □F  │
│ CH7     │            │            │ □P □F  │
│ CH8     │            │            │ □P □F  │
└─────────┴────────────┴────────────┴────────┘

ACEPTACIÓN: Variación entre canales < ±3%
```

---

## 9. PROGRAMA DE CALIBRACIÓN

### 9.1 Intervalos de Calibración

| Procedimiento | Intervalo | Eventos Disparadores |
|---------------|-----------|----------------------|
| CAL-1 Balanza de Torsión | Mensual / Por campaña | Reemplazo de fibra, reubicación |
| CAL-2 Verificación Piezo | Por campaña | Evento térmico, sospecha de daño |
| CAL-3 Alineación de Fase | Por campaña | Actualización de firmware, cambio de cableado |
| CAL-4 Sensores de Temperatura | 6 meses | Reemplazo de sensor |
| CAL-5 Frecuencia DDS | 12 meses | Actualización de firmware |
| CAL-6 Amplificadores AV | 6 meses | Reemplazo de componentes |

### 9.2 Registro de Estado de Calibración

```
AETHERION MARK 1 - REGISTRO DE ESTADO DE CALIBRACIÓN
═══════════════════════════════════════════════════════════════

┌───────────┬────────────┬────────────┬───────────┬───────────┐
│Procedim.  │ Última Cal │ Próx Venc  │ Estado    │ Técnico   │
├───────────┼────────────┼────────────┼───────────┼───────────┤
│ CAL-1     │            │            │           │           │
│ CAL-2     │            │            │           │           │
│ CAL-3     │            │            │           │           │
│ CAL-4     │            │            │           │           │
│ CAL-5     │            │            │           │           │
│ CAL-6     │            │            │           │           │
└───────────┴────────────┴────────────┴───────────┴───────────┘
```

---

## 10. REGISTROS DE CALIBRACIÓN

### 10.1 Retención de Registros

Todos los registros de calibración deberán conservarse por:
- **Mínimo:** Duración de la campaña de pruebas + 2 años
- **Formato:** PDF o copias en papel firmadas
- **Ubicación:** Carpeta de documentación del proyecto

### 10.2 Documentación Requerida

Cada calibración deberá incluir:
- [ ] Hoja de datos de calibración completada
- [ ] Certificados de estándares de referencia
- [ ] Firma del técnico y fecha
- [ ] Determinación de Pasa/Falla
- [ ] Acción correctiva (si aplica)

### 10.3 Plantilla de Certificado de Calibración

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           CERTIFICADO DE CALIBRACIÓN AETHERION MARK 1           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Procedimiento: ____________________  Revisión: _______         │
│                                                                 │
│  ID del Equipo: ____________________                            │
│  Número de Serie: ____________________                          │
│                                                                 │
│  Fecha de Calibración: ____________________                     │
│  Vencimiento de Calibración: ____________________               │
│                                                                 │
│  Estándares de Referencia Utilizados:                           │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│                                                                 │
│  Resultados:                                                    │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│                                                                 │
│  Estado:  □ PASA - Dentro de Tolerancia                         │
│           □ FALLA - Fuera de Tolerancia                         │
│           □ LIMITADO - Ver notas                                │
│                                                                 │
│  Notas:                                                         │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│                                                                 │
│  Calibrado Por: ____________________  Fecha: __________         │
│                                                                 │
│  Revisado Por: ____________________  Fecha: __________          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## APÉNDICE A: REFERENCIA RÁPIDA

### A.1 Lista de Verificación de Calibración (Pre-Campaña de Pruebas)

```
LISTA DE VERIFICACIÓN DE CALIBRACIÓN PRE-CAMPAÑA
══════════════════════════════════════════════════

□ CAL-1: Sensibilidad de balanza de torsión verificada
         S = _______ nN/µm

□ CAL-2: Los 8 canales piezo funcionales
         P1□ P2□ P3□ P4□ P5□ P6□ P7□ P8□

□ CAL-3: Alineación de fase dentro de ±5°
         Error máximo = _______ °

□ CAL-4: Sensores de temperatura dentro de ±1°C
         Error máximo = _______ °C

□ CAL-5: Frecuencia DDS dentro de 100 ppm
         Error máximo = _______ ppm

□ CAL-6: Amplificadores AV dentro de ±5%
         Error máximo = _______ %

TODAS LAS CALIBRACIONES VIGENTES: □ SÍ  □ NO

Verificado Por: _________________ Fecha: _________
```

---

```
═══════════════════════════════════════════════════════════════
                     FIN DEL DOCUMENTO
              AETHERION MARK 1 - MANUAL DE CALIBRACIÓN
                   ATP-MK1-CAL-001 Rev 1.0
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
