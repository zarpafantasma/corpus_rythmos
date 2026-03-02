# AETHERION MARK 1
## Calibration Procedures Manual

**Document ID:** ATP-MK1-CAL-001  
**Revision:** 1.0  
**Classification:** OPERATIONAL  
**Date:** February 2026  

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [Required Equipment](#2-required-equipment)
3. [CAL-1: Torsion Balance Calibration](#3-cal-1-torsion-balance-calibration)
4. [CAL-2: Piezoelectric Array Verification](#4-cal-2-piezoelectric-array-verification)
5. [CAL-3: Phase Alignment Calibration](#5-cal-3-phase-alignment-calibration)
6. [CAL-4: Temperature Sensor Calibration](#6-cal-4-temperature-sensor-calibration)
7. [CAL-5: DDS Frequency Calibration](#7-cal-5-dds-frequency-calibration)
8. [CAL-6: HV Amplifier Calibration](#8-cal-6-hv-amplifier-calibration)
9. [Calibration Schedule](#9-calibration-schedule)
10. [Calibration Records](#10-calibration-records)

---

## 1. OVERVIEW

### 1.1 Purpose

This document establishes calibration procedures for all measurement and control systems in the Aetherion Mark 1 prototype. Proper calibration ensures:

- Accurate thrust measurements
- Reliable thermal safety interlocks
- Correct piezoelectric drive signals
- Repeatable test results

### 1.2 Calibration Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                   CALIBRATION CHAIN                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NIST Traceable Standards                                   │
│         ↓                                                   │
│  Laboratory Reference Instruments                           │
│         ↓                                                   │
│  Aetherion Calibration Fixtures                             │
│         ↓                                                   │
│  Installed Sensors & Actuators                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Calibration Status Definitions

| Status | Definition | Action |
|--------|------------|--------|
| **CALIBRATED** | Within specification | Proceed with testing |
| **DUE** | Calibration interval expired | Recalibrate before use |
| **OUT OF TOLERANCE** | Failed calibration | Repair/replace before use |
| **REFERENCE ONLY** | Not for quantitative use | Document limitation |

---

## 2. REQUIRED EQUIPMENT

### 2.1 Calibration Standards

| Item | Specification | Calibration Interval |
|------|---------------|---------------------|
| Mass Set (ASTM Class 4) | 1mg to 100g | 12 months |
| Precision Thermometer | ±0.1°C, NIST traceable | 12 months |
| Frequency Counter | 10 Hz resolution, 0.1 ppm | 12 months |
| Digital Multimeter | 6½ digit, NIST traceable | 12 months |
| Oscilloscope | 200 MHz, calibrated | 12 months |

### 2.2 Calibration Fixtures

| Item | Description | BOM Reference |
|------|-------------|---------------|
| Calibration Mass Set | 1mg, 10mg, 100mg, 1g, 10g | DOC-002 |
| Ice Point Reference | 0.00°C ice bath | — |
| Boiling Point Reference | 100.0°C steam point | — |
| Piezo Test Load | 10 MΩ resistive | — |

---

## 3. CAL-1: TORSION BALANCE CALIBRATION

### 3.1 Purpose

Establish the force-to-displacement sensitivity factor (S) in nN/µm.

### 3.2 Frequency

- **Initial:** Before first use
- **Periodic:** Every test campaign or monthly
- **Event-driven:** After any balance adjustment or fiber replacement

### 3.3 Prerequisites

- [ ] Balance installed on isolation table
- [ ] Optical sensor zeroed
- [ ] Ambient vibration < 1 µm RMS
- [ ] Temperature stable (±1°C over 1 hour)

### 3.4 Procedure

```
CAL-1: TORSION BALANCE STATIC CALIBRATION
═══════════════════════════════════════════════════════════════

STEP 1: BASELINE MEASUREMENT
─────────────────────────────
1.1  Allow balance to settle for 30 minutes
1.2  Record baseline position for 60 seconds
1.3  Calculate mean baseline: X₀ = _______ µm
1.4  Calculate RMS noise: σ₀ = _______ µm
1.5  Noise floor force: F_noise = σ₀ × S_nominal = _______ nN

     Acceptance: σ₀ < 0.5 µm (if using 10 nN/µm nominal)

STEP 2: APPLY CALIBRATION MASSES
────────────────────────────────
For each calibration mass (m):

2.1  Calculate gravitational force: F = m × g
     (Use local g value, typically 9.80665 m/s²)
     
2.2  Carefully place mass at thrust application point
2.3  Wait 30 seconds for settling
2.4  Record deflected position for 30 seconds
2.5  Calculate mean: X_m = _______ µm
2.6  Remove mass, verify return to baseline

DATA TABLE:
┌────────────┬───────────┬───────────┬───────────┬───────────┐
│ Mass (mg)  │ Force (nN)│ X₀ (µm)   │ X_m (µm)  │ ΔX (µm)   │
├────────────┼───────────┼───────────┼───────────┼───────────┤
│ 1.000      │ 9.807     │           │           │           │
│ 10.00      │ 98.07     │           │           │           │
│ 100.0      │ 980.7     │           │           │           │
│ 1000       │ 9807      │           │           │           │
│ 10000      │ 98070     │           │           │           │
└────────────┴───────────┴───────────┴───────────┴───────────┘

STEP 3: CALCULATE SENSITIVITY
─────────────────────────────
3.1  Plot F vs ΔX (should be linear)
3.2  Perform linear regression: F = S × ΔX + b
3.3  Sensitivity: S = _______ nN/µm
3.4  Intercept: b = _______ nN (should be ~0)
3.5  R² = _______ (acceptance: R² > 0.9999)

STEP 4: VERIFY LINEARITY
────────────────────────
4.1  Calculate residuals for each point
4.2  Max residual: _______ nN
4.3  Residual < 1% of full scale: □ PASS  □ FAIL

STEP 5: DOCUMENT RESULTS
────────────────────────
┌─────────────────────────────────────────────────────────────┐
│              TORSION BALANCE CALIBRATION CERTIFICATE        │
├─────────────────────────────────────────────────────────────┤
│ Date: ________________  Time: ________________              │
│ Technician: ________________                                │
│                                                             │
│ RESULTS:                                                    │
│   Sensitivity (S): _____________ nN/µm                      │
│   Noise Floor: _____________ nN                             │
│   Linearity (R²): _____________                             │
│   Measurement Range: 0 to _____________ nN                  │
│                                                             │
│ STATUS:  □ CALIBRATED   □ OUT OF TOLERANCE                  │
│                                                             │
│ Next Calibration Due: ________________                      │
│                                                             │
│ Signature: _______________________                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 Acceptance Criteria

| Parameter | Requirement |
|-----------|-------------|
| Linearity (R²) | > 0.9999 |
| Noise floor | < 10 nN |
| Hysteresis | < 2% |
| Zero drift | < 5 nN/hour |

---

## 4. CAL-2: PIEZOELECTRIC ARRAY VERIFICATION

### 4.1 Purpose

Verify all 8 piezoelectric channels respond correctly to drive signals.

### 4.2 Frequency

- **Initial:** Before first use
- **Periodic:** Before each test campaign
- **Event-driven:** After any thermal event or suspected depolarization

### 4.3 Procedure

```
CAL-2: PIEZOELECTRIC CHANNEL VERIFICATION
═══════════════════════════════════════════════════════════════

EQUIPMENT SETUP:
────────────────
- Oscilloscope CH1: DDS reference output
- Oscilloscope CH2: Piezo channel under test
- Thermal camera aimed at piezo array
- Initial piezo temperature: T₀ = _______ °C

TEST PARAMETERS:
────────────────
- Frequency: 1 kHz (safe test frequency)
- Voltage: 50 V (25% power - safe for extended test)
- Duration: 1 second per channel

PROCEDURE:
──────────
For each channel P1 through P8:

1. Select single channel mode
2. Apply test signal for 1 second
3. Capture oscilloscope waveform
4. Measure and record:
   - Output amplitude
   - Waveform shape
   - Phase relative to reference
   - Temperature rise

DATA TABLE:
┌─────────┬───────────┬───────────┬───────────┬───────────┬────────┐
│ Channel │ Amplitude │ Waveform  │ Phase (°) │ ΔT (°C)   │ Status │
│         │ (V p-p)   │           │           │           │        │
├─────────┼───────────┼───────────┼───────────┼───────────┼────────┤
│ P1      │           │ □OK □BAD  │           │           │ □P □F  │
│ P2      │           │ □OK □BAD  │           │           │ □P □F  │
│ P3      │           │ □OK □BAD  │           │           │ □P □F  │
│ P4      │           │ □OK □BAD  │           │           │ □P □F  │
│ P5      │           │ □OK □BAD  │           │           │ □P □F  │
│ P6      │           │ □OK □BAD  │           │           │ □P □F  │
│ P7      │           │ □OK □BAD  │           │           │ □P □F  │
│ P8      │           │ □OK □BAD  │           │           │ □P □F  │
└─────────┴───────────┴───────────┴───────────┴───────────┴────────┘

ACCEPTANCE CRITERIA:
────────────────────
- Amplitude: 50 ± 2 V (within 4%)
- Waveform: Clean sinusoid, no distortion
- Phase: As programmed ± 5°
- ΔT: < 5°C for 1-second test
- All 8 channels must PASS

DEPOLARIZATION CHECK:
─────────────────────
If any channel shows:
- Amplitude < 40 V (>20% loss)
- Severe waveform distortion
- No output

→ Channel may be depolarized. Replace piezo element.
```

### 4.4 Acceptance Criteria

| Parameter | Requirement | Action if Failed |
|-----------|-------------|------------------|
| Amplitude | 50 ± 2 V | Check amplifier, wiring |
| Waveform | Clean sinusoid | Check for shorts, damage |
| All channels functional | 8/8 | Replace failed elements |

---

## 5. CAL-3: PHASE ALIGNMENT CALIBRATION

### 5.1 Purpose

Verify correct phase relationships for TPH traveling wave generation.

### 5.2 Procedure

```
CAL-3: PHASE ALIGNMENT VERIFICATION
═══════════════════════════════════════════════════════════════

TPH MODE PHASE REQUIREMENTS:
────────────────────────────
Channel spacing: 45° (360° / 8 channels)

Expected phases:
  P1: 0°    P2: 45°   P3: 90°   P4: 135°
  P5: 180°  P6: 225°  P7: 270°  P8: 315°

MEASUREMENT PROCEDURE:
──────────────────────
1. Set mode: TPH
2. Set frequency: 1 kHz
3. Set voltage: 50 V
4. Connect oscilloscope:
   - CH1: P1 (trigger reference)
   - CH2: Channel under test

5. Measure time delay (Δt) between rising edges
6. Calculate phase: φ = (Δt / T) × 360°
   where T = 1/f = 1 ms at 1 kHz

DATA TABLE:
┌─────────┬───────────┬───────────┬───────────┬───────────┬────────┐
│ Channel │ Expected  │ Measured  │ Δt (µs)   │ Error (°) │ Status │
│         │ Phase (°) │ Phase (°) │           │           │        │
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

ACCEPTANCE: Phase error < ±5° for all channels

ADJUSTMENT PROCEDURE (if needed):
─────────────────────────────────
1. Connect to MCU via UART
2. Use command: phase <channel> <degrees>
3. Example: phase 2 47  (adjusts P2 to 47°)
4. Re-measure and iterate until within spec
```

---

## 6. CAL-4: TEMPERATURE SENSOR CALIBRATION

### 6.1 Purpose

Calibrate PT1000 RTD sensors for accurate thermal monitoring.

### 6.2 Procedure

```
CAL-4: PT1000 TEMPERATURE CALIBRATION
═══════════════════════════════════════════════════════════════

REFERENCE POINTS:
─────────────────
- Ice point: 0.00°C (ice-water bath)
- Ambient: ~25°C (reference thermometer)
- Warm point: ~50°C (heated water bath)

EQUIPMENT:
──────────
- NIST-traceable reference thermometer (±0.1°C)
- Ice-water bath (0.00 ± 0.02°C)
- Temperature-controlled water bath
- Stirring mechanism

PROCEDURE:
──────────
For each sensor (SN-001 through SN-004):

1. ICE POINT (0°C)
   1.1 Immerse sensor and reference in ice bath
   1.2 Wait 5 minutes for equilibration
   1.3 Record reference: T_ref = _______ °C
   1.4 Record sensor: T_sensor = _______ °C
   1.5 Error at 0°C: _______ °C

2. AMBIENT POINT (~25°C)
   2.1 Place sensor and reference at ambient
   2.2 Wait 10 minutes for equilibration
   2.3 Record reference: T_ref = _______ °C
   2.4 Record sensor: T_sensor = _______ °C
   2.5 Error at 25°C: _______ °C

3. WARM POINT (~50°C)
   3.1 Immerse in heated water bath at 50°C
   3.2 Wait 5 minutes for equilibration
   3.3 Record reference: T_ref = _______ °C
   3.4 Record sensor: T_sensor = _______ °C
   3.5 Error at 50°C: _______ °C

4. CALCULATE CORRECTION
   4.1 Plot sensor vs reference
   4.2 Fit linear correction: T_corrected = a × T_sensor + b
   4.3 Record coefficients: a = _______, b = _______ °C

DATA TABLE:
┌────────┬───────────┬───────────┬───────────┬───────────┬────────┐
│ Sensor │ Error @0° │ Error @25°│ Error @50°│ Max Error │ Status │
├────────┼───────────┼───────────┼───────────┼───────────┼────────┤
│ T1     │           │           │           │           │ □P □F  │
│ T2     │           │           │           │           │ □P □F  │
│ T3     │           │           │           │           │ □P □F  │
│ T4     │           │           │           │           │ □P □F  │
└────────┴───────────┴───────────┴───────────┴───────────┴────────┘

ACCEPTANCE: Max error < ±1.0°C across range 0-50°C

Note: Critical threshold is 90°C. If sensors cannot be verified
at this temperature, apply conservative offset in firmware.
```

---

## 7. CAL-5: DDS FREQUENCY CALIBRATION

### 7.1 Purpose

Verify AD9910 DDS generates accurate frequencies.

### 7.2 Procedure

```
CAL-5: DDS FREQUENCY CALIBRATION
═══════════════════════════════════════════════════════════════

EQUIPMENT:
──────────
- Frequency counter (10 Hz resolution, 0.1 ppm accuracy)
- Oscilloscope for waveform verification

TEST FREQUENCIES:
─────────────────
Cover operating range: 100 Hz to 50 kHz

PROCEDURE:
──────────
1. Connect frequency counter to DDS output
2. For each test frequency:
   2.1 Command frequency via UART: freq <Hz>
   2.2 Wait 1 second for settling
   2.3 Record counter reading
   2.4 Calculate error

DATA TABLE:
┌────────────┬────────────┬────────────┬────────────┬────────┐
│ Commanded  │ Measured   │ Error      │ Error      │ Status │
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

ACCEPTANCE: Error < 100 ppm (0.01%) across range
```

---

## 8. CAL-6: HV AMPLIFIER CALIBRATION

### 8.1 Purpose

Verify PA94 amplifiers deliver correct output voltage.

### 8.2 Procedure

```
CAL-6: HV AMPLIFIER CALIBRATION
═══════════════════════════════════════════════════════════════

⚠️  HIGH VOLTAGE - Use appropriate safety procedures

EQUIPMENT:
──────────
- HV probe (1000:1, calibrated)
- Oscilloscope
- Resistive dummy load (10 kΩ, 25W)

PROCEDURE:
──────────
1. Disconnect piezo array
2. Connect dummy load to amplifier output
3. Connect HV probe to output
4. For each commanded voltage:
   4.1 Set voltage via UART: voltage <V>
   4.2 Set frequency: 1 kHz
   4.3 Fire for 1 second
   4.4 Measure peak-to-peak output
   4.5 Calculate RMS: V_rms = V_pp / (2√2)

DATA TABLE:
┌────────────┬────────────┬────────────┬────────────┬────────┐
│ Commanded  │ Measured   │ Error      │ Error      │ Status │
│ (V)        │ V_pp (V)   │ (V)        │ (%)        │        │
├────────────┼────────────┼────────────┼────────────┼────────┤
│ 50         │            │            │            │ □P □F  │
│ 100        │            │            │            │ □P □F  │
│ 150        │            │            │            │ □P □F  │
│ 200        │            │            │            │ □P □F  │
└────────────┴────────────┴────────────┴────────────┴────────┘

ACCEPTANCE: Error < ±5% across range

CHANNEL MATCHING:
─────────────────
Repeat for all 8 channels at 150V setting:

┌─────────┬────────────┬────────────┬────────┐
│ Channel │ Measured V │ Deviation  │ Status │
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

ACCEPTANCE: Channel-to-channel variation < ±3%
```

---

## 9. CALIBRATION SCHEDULE

### 9.1 Calibration Intervals

| Procedure | Interval | Trigger Events |
|-----------|----------|----------------|
| CAL-1 Torsion Balance | Monthly / Per campaign | Fiber replacement, relocation |
| CAL-2 Piezo Verification | Per campaign | Thermal event, suspected damage |
| CAL-3 Phase Alignment | Per campaign | Firmware update, wiring change |
| CAL-4 Temperature Sensors | 6 months | Sensor replacement |
| CAL-5 DDS Frequency | 12 months | Firmware update |
| CAL-6 HV Amplifiers | 6 months | Component replacement |

### 9.2 Calibration Status Log

```
AETHERION MARK 1 - CALIBRATION STATUS LOG
═══════════════════════════════════════════════════════════════

┌───────────┬────────────┬────────────┬───────────┬───────────┐
│ Procedure │ Last Cal   │ Next Due   │ Status    │ Technician│
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

## 10. CALIBRATION RECORDS

### 10.1 Record Retention

All calibration records shall be retained for:
- **Minimum:** Duration of test campaign + 2 years
- **Format:** PDF or signed paper copies
- **Location:** Project documentation folder

### 10.2 Required Documentation

Each calibration shall include:
- [ ] Completed calibration data sheet
- [ ] Reference standard certificates
- [ ] Technician signature and date
- [ ] Pass/Fail determination
- [ ] Corrective action (if any)

### 10.3 Calibration Certificate Template

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              AETHERION MARK 1 CALIBRATION CERTIFICATE           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Procedure: ____________________  Revision: _______             │
│                                                                 │
│  Equipment ID: ____________________                             │
│  Serial Number: ____________________                            │
│                                                                 │
│  Calibration Date: ____________________                         │
│  Calibration Due: ____________________                          │
│                                                                 │
│  Reference Standards Used:                                      │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│                                                                 │
│  Results:                                                       │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│                                                                 │
│  Status:  □ PASS - Within Tolerance                             │
│           □ FAIL - Out of Tolerance                             │
│           □ LIMITED - See notes                                 │
│                                                                 │
│  Notes:                                                         │
│    ___________________________________________________________  │
│    ___________________________________________________________  │
│                                                                 │
│  Calibrated By: ____________________  Date: __________          │
│                                                                 │
│  Reviewed By: ____________________  Date: __________            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## APPENDIX A: QUICK REFERENCE

### A.1 Calibration Checklist (Pre-Test Campaign)

```
PRE-CAMPAIGN CALIBRATION CHECKLIST
══════════════════════════════════════════

□ CAL-1: Torsion balance sensitivity verified
         S = _______ nN/µm

□ CAL-2: All 8 piezo channels functional
         P1□ P2□ P3□ P4□ P5□ P6□ P7□ P8□

□ CAL-3: Phase alignment within ±5°
         Max error = _______ °

□ CAL-4: Temperature sensors within ±1°C
         Max error = _______ °C

□ CAL-5: DDS frequency within 100 ppm
         Max error = _______ ppm

□ CAL-6: HV amplifiers within ±5%
         Max error = _______ %

ALL CALIBRATIONS CURRENT: □ YES  □ NO

Verified By: _________________ Date: _________
```

---

```
═══════════════════════════════════════════════════════════════
                     END OF DOCUMENT
              AETHERION MARK 1 - CALIBRATION MANUAL
                   ATP-MK1-CAL-001 Rev 1.0
═══════════════════════════════════════════════════════════════
```
