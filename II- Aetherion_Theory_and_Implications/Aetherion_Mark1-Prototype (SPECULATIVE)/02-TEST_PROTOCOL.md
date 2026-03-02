# AETHERION MARK 1
## Test Protocol & Validation Procedures

**Document ID:** ATP-MK1-001  
**Revision:** 1.0  
**Classification:** OPERATIONAL  
**Date:** February 2026  

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ⚠️  CRITICAL SAFETY DOCUMENT — READ COMPLETELY BEFORE ANY OPERATION  ⚠️   ║
║                                                                              ║
║     This protocol incorporates Red Team Advisory constraints.                ║
║     Failure to comply may result in:                                         ║
║       • Permanent equipment damage (thermal depolarization)                  ║
║       • Personnel injury (acoustic trauma)                                   ║
║       • Test facility damage (resonance-induced failure)                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## TABLE OF CONTENTS

1. [Scope & Objectives](#1-scope--objectives)
2. [Required Equipment](#2-required-equipment)
3. [Safety Requirements](#3-safety-requirements)
4. [Pre-Test Setup](#4-pre-test-setup)
5. [Calibration Procedures](#5-calibration-procedures)
6. [Test Sequences](#6-test-sequences)
7. [Data Collection](#7-data-collection)
8. [Pass/Fail Criteria](#8-passfail-criteria)
9. [Emergency Procedures](#9-emergency-procedures)
10. [Post-Test Procedures](#10-post-test-procedures)

---

## 1. SCOPE & OBJECTIVES

### 1.1 Purpose

This document establishes the complete test protocol for validating the Aetherion Mark 1 vacuum gradient thruster prototype. The primary objective is to measure ponderomotive thrust generated via TPH (Temporal Pulse Hierarchy) and OMV (Oscillatory Modulation of Vacuum) operational modes.

### 1.2 Test Objectives

| ID | Objective | Success Metric |
|----|-----------|----------------|
| **T-01** | Verify TPH impulse generation | Measurable deflection on torsion balance |
| **T-02** | Verify OMV DC thrust | Sustained deflection > noise floor |
| **T-03** | Validate thrust scaling with frequency | Linear F ∝ f relationship |
| **T-04** | Validate thrust scaling with voltage | Quadratic F ∝ V² relationship |
| **T-05** | Confirm thermal stability | Piezo array < 90°C during operation |
| **T-06** | Verify control system response | All modes switch correctly |

### 1.3 Test Phases

```
PHASE 0: Setup & Calibration (Day 1)
    ↓
PHASE 1: Electrical Verification — No thrust generation (Day 1)
    ↓
PHASE 2: Atmospheric Testing — Initial thrust detection (Day 2)
    ↓
PHASE 3: Vacuum Testing — Precision measurements (Day 3-5)
    ↓
PHASE 4: Parametric Sweep — Full characterization (Day 6-10)
```

---

## 2. REQUIRED EQUIPMENT

### 2.1 Test Articles

| Item | Specification | Quantity |
|------|---------------|----------|
| Aetherion Mark 1 Unit | Per engineering spec | 1 |
| Spare PZT-5H Array | 8× actuators, pre-wired | 1 set |
| Calibration Masses | 1mg, 10mg, 100mg, 1g | 1 set |

### 2.2 Test Infrastructure

| Item | Specification | Purpose |
|------|---------------|---------|
| **Torsion Balance** | Resolution < 10 nN | Thrust measurement |
| **Vacuum Chamber** | 10⁻³ to 10⁻⁶ Torr capable | Eliminate air drag |
| **Optical Displacement Sensor** | Resolution < 0.1 µm | Balance readout |
| **Isolation Table** | Pneumatic, < 1 Hz cutoff | Vibration isolation |
| **Faraday Cage** | Full enclosure | EMI shielding |

### 2.3 Instrumentation

| Instrument | Model (Example) | Purpose |
|------------|-----------------|---------|
| Oscilloscope | Keysight DSOX3024T | Waveform verification |
| Multimeter | Fluke 87V | Voltage/current |
| Thermal Camera | FLIR E8-XT | Piezo temperature |
| Sound Level Meter | Extech 407730 | Acoustic monitoring |
| Data Acquisition | NI USB-6009 | Multi-channel logging |

### 2.4 Safety Equipment

| Item | Specification | Location |
|------|---------------|----------|
| **Hearing Protection** | NRR 30+ dB | All personnel |
| **HV Discharge Probe** | 200V rated | Near test station |
| **Fire Extinguisher** | CO2, Class C | Within 3m |
| **First Aid Kit** | Standard industrial | Control room |
| **Emergency Stop Button** | Hardwired, red mushroom | Console AND test chamber |

---

## 3. SAFETY REQUIREMENTS

### 3.1 Personnel Requirements

| Role | Minimum # | Responsibilities |
|------|-----------|------------------|
| **Test Director** | 1 | Overall authority, GO/NO-GO decisions |
| **Test Operator** | 1 | Console operation, data logging |
| **Safety Officer** | 1 | Monitor thermal/acoustic, E-stop authority |

```
⚠️  MINIMUM 2 PERSONNEL REQUIRED FOR ANY LIVE-FIRE TEST
⚠️  NO PERSONNEL IN TEST CHAMBER DURING OPERATION
```

### 3.2 Thermal Safety Protocol

**Reference:** Red Team Advisory §1 — Thermal Depolarization Risk

| Parameter | Limit | Action if Exceeded |
|-----------|-------|-------------------|
| Piezo Array Temperature | < 90°C | AUTO E-STOP |
| Piezo Array Temperature | < 70°C | WARNING, reduce duty cycle |
| Ambient Chamber Temp | < 40°C | Pause testing, ventilate |

**Mandatory Duty Cycle Limits:**

```
┌─────────────────────────────────────────────────────────────┐
│  MARK 1 THERMAL PROTOCOL (Until liquid cooling in Mark 2)   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FIRE:     5-10 seconds MAX                                 │
│  COOLDOWN: 60 seconds MINIMUM                               │
│                                                             │
│  Duty Cycle = 10s / 70s = 14.3% MAXIMUM                     │
│                                                             │
│  MCU must enforce this automatically.                       │
│  Manual override is PROHIBITED.                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Acoustic Safety Protocol

**Reference:** Red Team Advisory §2 — Acoustic Hazards

| Condition | Requirement |
|-----------|-------------|
| Test Frequency 1-10 kHz | Hearing protection NRR 30+ |
| Power > 10W | NO personnel in test chamber |
| Any live-fire | Remote operation ONLY |

**Acoustic Monitoring:**

```
Sound Level Action Thresholds:
─────────────────────────────────
< 85 dB    Normal operation
85-100 dB  Hearing protection mandatory
100-120 dB Remote operation only (control booth)
> 120 dB   IMMEDIATE E-STOP — investigate
```

### 3.4 Electrical Safety Protocol

| Hazard | Mitigation |
|--------|------------|
| 200V HV Rail | Interlocked enclosure, discharge probe |
| Capacitor Discharge | 30-second wait after power-off |
| Ground Faults | GFCI protection on all circuits |

### 3.5 Pre-Test Safety Checklist

```
┌─────────────────────────────────────────────────────────────┐
│              PRE-TEST SAFETY VERIFICATION                   │
│                                                             │
│  □ 1. All personnel briefed on today's test plan            │
│  □ 2. Test Director has confirmed GO status                 │
│  □ 3. Emergency exits clear and marked                      │
│  □ 4. E-Stop buttons tested (both locations)                │
│  □ 5. Fire extinguisher inspected and accessible            │
│  □ 6. Hearing protection distributed                        │
│  □ 7. HV discharge probe ready                              │
│  □ 8. Thermal camera powered and aimed                      │
│  □ 9. Control booth isolated (door closed)                  │
│  □ 10. Communication system tested (intercom/radio)         │
│                                                             │
│  Test Director Signature: ___________________ Date: ______  │
│  Safety Officer Signature: __________________ Date: ______  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. PRE-TEST SETUP

### 4.1 Mechanical Setup (Day 1 Morning)

**Step 1: Torsion Balance Preparation**

```
1.1  Level the isolation table (bubble level, all 4 corners)
1.2  Install torsion balance on table center
1.3  Verify fiber/wire is undamaged and properly tensioned
1.4  Install optical displacement sensor
1.5  Zero the sensor with balance at rest
1.6  Record ambient temperature: _______ °C
1.7  Record ambient pressure: _______ mbar
```

**Step 2: Mark 1 Mounting**

```
⚠️  CRITICAL: Do NOT hard-mount the unit!

2.1  Install Sorbothane isolation pads on balance arm
     - 4× pads, 10mm thick, Shore 50A
     - Purpose: Filter acoustic vibration from thrust measurement

2.2  Place Mark 1 unit on isolation pads
2.3  Secure with light clamping (do not over-torque)
2.4  Verify thrust axis aligned with balance sensitivity axis
2.5  Photograph mounting configuration for records
```

**Step 3: Vacuum Chamber Setup (if applicable)**

```
3.1  Install polycarbonate blast shield inside chamber
3.2  Route cables through vacuum feedthroughs
3.3  Verify all seals and O-rings
3.4  Connect roughing pump
3.5  Connect turbo pump (if high vacuum required)
3.6  Install pressure gauge
```

### 4.2 Electrical Setup (Day 1 Afternoon)

**Step 4: Power Connections**

```
4.1  Verify 24V supply OFF
4.2  Connect 24V supply to Mark 1 input
4.3  Connect USB/UART to control computer
4.4  Connect oscilloscope probes:
     - CH1: DDS output (reference waveform)
     - CH2: Piezo channel 1 (verify amplification)
4.5  Connect thermal camera feed to monitor
4.6  Verify E-Stop wired in series with HV supply
```

**Step 5: Control System Verification**

```
5.1  Power ON 24V supply
5.2  Verify 5V rail: _______ V (expected: 5.0 ± 0.1V)
5.3  Verify HV rail (no load): _______ V (expected: 200 ± 5V)
5.4  Launch control software
5.5  Verify USB communication established
5.6  Read all sensor values:
     - Temp 1: ___°C  Temp 2: ___°C  Temp 3: ___°C  Temp 4: ___°C
     - Accel X: ___g  Accel Y: ___g  Accel Z: ___g
5.7  Verify thermal interlock: Set threshold to 90°C
5.8  Verify duty cycle limiter: 10s ON / 60s OFF
```

---

## 5. CALIBRATION PROCEDURES

### 5.1 Torsion Balance Calibration

**Objective:** Establish nN/µm conversion factor

**Procedure:**

```
CAL-1: Static Calibration with Known Masses

1. Record baseline position: X₀ = _______ µm
2. Apply 1 mg calibration mass at thrust point
   - Gravitational force: F = 9.81 µN
   - Record deflection: X₁ = _______ µm
   - ΔX = X₁ - X₀ = _______ µm
   
3. Calculate sensitivity: S = F / ΔX = _______ nN/µm

4. Repeat with 10 mg mass:
   - F = 98.1 µN
   - ΔX = _______ µm
   - S = _______ nN/µm
   
5. Verify linearity (both S values should match within 5%)

6. Record final calibration factor:
   
   ┌─────────────────────────────────────┐
   │  CALIBRATION FACTOR                 │
   │  S = _________ nN/µm                │
   │  Date: __________                   │
   │  Technician: __________             │
   └─────────────────────────────────────┘
```

### 5.2 Piezo Response Calibration

**Objective:** Verify all 8 channels respond correctly

**Procedure:**

```
CAL-2: Individual Channel Test

For each channel P1 through P8:

1. Set frequency: 1 kHz
2. Set voltage: 50V (25% power — safe for sustained test)
3. Set mode: Single channel only
4. Fire for 1 second
5. Observe on oscilloscope:
   - Waveform shape: □ Correct  □ Distorted
   - Amplitude: _______ V (expected: 50 ± 2V)
   - Frequency: _______ Hz (expected: 1000 ± 1 Hz)
6. Record piezo temperature rise: ΔT = _______ °C

Channel Test Results:
┌────────┬───────────┬───────────┬────────┬────────┐
│ Channel│ Waveform  │ Amplitude │ Freq   │ ΔT(°C) │
├────────┼───────────┼───────────┼────────┼────────┤
│ P1     │ □OK □FAIL │           │        │        │
│ P2     │ □OK □FAIL │           │        │        │
│ P3     │ □OK □FAIL │           │        │        │
│ P4     │ □OK □FAIL │           │        │        │
│ P5     │ □OK □FAIL │           │        │        │
│ P6     │ □OK □FAIL │           │        │        │
│ P7     │ □OK □FAIL │           │        │        │
│ P8     │ □OK □FAIL │           │        │        │
└────────┴───────────┴───────────┴────────┴────────┘

All channels must pass. Any failure = NO-GO for testing.
```

### 5.3 Phase Alignment Calibration

**Objective:** Verify traveling wave generation for TPH mode

**Procedure:**

```
CAL-3: Phase Sequence Test

1. Set mode: TPH
2. Set frequency: 1 kHz
3. Set voltage: 50V
4. Connect oscilloscope:
   - CH1: P1 (trigger)
   - CH2: P3 (90° offset expected)
   
5. Fire for 1 second
6. Measure phase delay: Δφ = _______ ° (expected: 90 ± 5°)

7. Repeat for P1 vs P5 (180° expected): Δφ = _______ °
8. Repeat for P1 vs P7 (270° expected): Δφ = _______ °

Phase alignment: □ PASS (all within ±5°)  □ FAIL
```

---

## 6. TEST SEQUENCES

### 6.1 Phase 1: Electrical Verification (No Thrust)

**Objective:** Confirm all systems functional before thrust generation

| Test ID | Description | Duration | Voltage | Expected Result |
|---------|-------------|----------|---------|-----------------|
| EV-01 | Power-on sequence | — | — | All LEDs, sensors active |
| EV-02 | Communication test | — | — | USB responds |
| EV-03 | Thermal readout | 60s | 0V | Stable, < 30°C |
| EV-04 | Single channel ping | 100ms | 50V | Oscilloscope waveform |
| EV-05 | All channels ping | 100ms | 50V | 8 waveforms verified |
| EV-06 | E-Stop test | — | 50V | Instant power cut |
| EV-07 | Thermal interlock | Sim | — | MCU triggers at 90°C |

### 6.2 Phase 2: Atmospheric Thrust Detection

**Objective:** First thrust measurement (air drag present but acceptable for detection)

```
⚠️  PERSONNEL MUST EXIT TEST CHAMBER
⚠️  REMOTE OPERATION ONLY FROM THIS POINT
```

**Test Sequence AT-01: OMV Mode Detection**

```
1. Evacuate test chamber (personnel only, not vacuum)
2. Seal chamber door
3. Arm system from control booth
4. Set parameters:
   - Mode: OMV (continuous sine)
   - Frequency: 2 kHz
   - Voltage: 100V (50% power)
   - Duration: 5 seconds

5. Record pre-fire baseline:
   - Balance position: X₀ = _______ µm
   - Piezo temperature: T₀ = _______ °C
   - Ambient sound level: _______ dB

6. Command: FIRE

7. During firing, record:
   - Balance deflection (live): _______ µm
   - Sound level (peak): _______ dB
   
8. After firing, record:
   - Balance position (settled): X₁ = _______ µm
   - Piezo temperature: T₁ = _______ °C
   - Temperature rise: ΔT = T₁ - T₀ = _______ °C

9. Calculate thrust:
   - Deflection: ΔX = X₁ - X₀ = _______ µm
   - Thrust: F = ΔX × S = _______ nN

10. Wait 60 seconds (mandatory cooldown)

11. Repeat 3 times for statistics
```

**Test Sequence AT-02: TPH Mode Detection**

```
Same procedure as AT-01, but:
- Mode: TPH (pulsed)
- Frequency: 1 kHz
- Voltage: 100V
- Duration: 5 seconds

Expected: Impulse-like deflection followed by decay
```

### 6.3 Phase 3: Vacuum Testing

**Objective:** Precision measurement without aerodynamic interference

**Vacuum Levels:**

| Level | Pressure | Purpose |
|-------|----------|---------|
| Rough | 10⁻¹ Torr | Eliminate convection |
| Medium | 10⁻³ Torr | Eliminate most drag |
| High | 10⁻⁶ Torr | Ultimate precision |

**Test Sequence VT-01: Vacuum Baseline**

```
1. Pump chamber to 10⁻³ Torr
2. Wait 10 minutes for thermal equilibration
3. Record baseline noise floor:
   - Balance drift over 60s: _______ µm
   - RMS noise: _______ µm
   - Equivalent force noise: _______ nN

4. This establishes detection threshold
```

**Test Sequence VT-02: Vacuum OMV Thrust**

```
1. Verify pressure: < 10⁻³ Torr
2. Set parameters:
   - Mode: OMV
   - Frequency: 2 kHz  
   - Voltage: 150V (75% power)
   - Duration: 10 seconds

3. Fire and record:
   - Peak deflection: _______ µm
   - Sustained deflection: _______ µm
   - Thrust: _______ nN

4. Compare to atmospheric test — should be cleaner signal
```

### 6.4 Phase 4: Parametric Sweep

**Objective:** Full characterization of thrust dependencies

**Test Matrix:**

```
FREQUENCY SWEEP (Fixed V = 100V, OMV Mode)
┌──────────┬──────────┬──────────┬──────────┐
│ f (kHz)  │ ΔX (µm)  │ F (nN)   │ Temp (°C)│
├──────────┼──────────┼──────────┼──────────┤
│ 0.5      │          │          │          │
│ 1.0      │          │          │          │
│ 2.0      │          │          │          │
│ 5.0      │          │          │          │
│ 10.0     │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
Expected: Linear relationship F ∝ f

VOLTAGE SWEEP (Fixed f = 2 kHz, OMV Mode)
┌──────────┬──────────┬──────────┬──────────┐
│ V (V)    │ ΔX (µm)  │ F (nN)   │ Temp (°C)│
├──────────┼──────────┼──────────┼──────────┤
│ 50       │          │          │          │
│ 100      │          │          │          │
│ 150      │          │          │          │
│ 200      │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
Expected: Quadratic relationship F ∝ V²

MODE COMPARISON (Fixed f = 2 kHz, V = 150V)
┌──────────┬──────────┬──────────┬──────────┐
│ Mode     │ ΔX (µm)  │ F (nN)   │ Character│
├──────────┼──────────┼──────────┼──────────┤
│ OMV      │          │          │ DC       │
│ TPH      │          │          │ Impulse  │
│ Hybrid   │          │          │ Both     │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 7. DATA COLLECTION

### 7.1 Required Data Channels

| Channel | Sensor | Sample Rate | Units |
|---------|--------|-------------|-------|
| CH1 | Balance position | 1 kHz | µm |
| CH2 | Piezo Temp 1 | 10 Hz | °C |
| CH3 | Piezo Temp 2 | 10 Hz | °C |
| CH4 | Piezo Temp 3 | 10 Hz | °C |
| CH5 | Piezo Temp 4 | 10 Hz | °C |
| CH6 | Chamber pressure | 1 Hz | Torr |
| CH7 | Ambient temp | 1 Hz | °C |
| CH8 | Sound level | 100 Hz | dB |
| CH9 | Accel X | 1 kHz | g |
| CH10 | Accel Y | 1 kHz | g |
| CH11 | Accel Z | 1 kHz | g |
| CH12 | Command voltage | 10 kHz | V |

### 7.2 Data File Format

```
Filename: AETHERION_MK1_TEST_{DATE}_{SEQUENCE_ID}.csv

Header:
# Aetherion Mark 1 Test Data
# Date: YYYY-MM-DD HH:MM:SS
# Test ID: {SEQUENCE_ID}
# Mode: {TPH/OMV/HYBRID}
# Frequency: {f} kHz
# Voltage: {V} V
# Duration: {t} s
# Pressure: {P} Torr
# Calibration Factor: {S} nN/µm

Columns:
timestamp_ms, position_um, temp1_C, temp2_C, temp3_C, temp4_C, 
pressure_torr, ambient_C, sound_dB, accel_x_g, accel_y_g, accel_z_g, cmd_V
```

### 7.3 Mandatory Logging

Every test MUST record:

```
┌─────────────────────────────────────────────────────────────┐
│                    TEST LOG ENTRY                           │
├─────────────────────────────────────────────────────────────┤
│ Date: ____________  Time: ____________                      │
│ Test ID: ____________                                       │
│ Test Director: ____________                                 │
│ Operator: ____________                                      │
│                                                             │
│ Parameters:                                                 │
│   Mode: □ OMV  □ TPH  □ Hybrid                              │
│   Frequency: ________ kHz                                   │
│   Voltage: ________ V                                       │
│   Duration: ________ s                                      │
│   Chamber Pressure: ________ Torr                           │
│                                                             │
│ Results:                                                    │
│   Deflection: ________ µm                                   │
│   Calculated Thrust: ________ nN                            │
│   Peak Piezo Temp: ________ °C                              │
│   Peak Sound Level: ________ dB                             │
│                                                             │
│ Anomalies: _____________________________________________    │
│ __________________________________________________________  │
│                                                             │
│ Data File: ________________________________________________ │
│                                                             │
│ Signatures:                                                 │
│   Test Director: _________________ Date: __________         │
│   Operator: _________________ Date: __________              │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. PASS/FAIL CRITERIA

### 8.1 Primary Success Criteria

| ID | Criterion | Threshold | Result |
|----|-----------|-----------|--------|
| **P1** | Measurable OMV thrust | > 50 nN @ 2kHz/150V | □ PASS □ FAIL |
| **P2** | Measurable TPH impulse | Visible deflection | □ PASS □ FAIL |
| **P3** | Thrust scales with frequency | R² > 0.9 (linear) | □ PASS □ FAIL |
| **P4** | Thrust scales with voltage | R² > 0.9 (quadratic) | □ PASS □ FAIL |
| **P5** | Thermal stability | Max T < 90°C | □ PASS □ FAIL |
| **P6** | Repeatability | σ/μ < 20% | □ PASS □ FAIL |

### 8.2 Secondary Success Criteria

| ID | Criterion | Threshold | Result |
|----|-----------|-----------|--------|
| **S1** | Vacuum vs Atmospheric SNR | > 3× improvement | □ PASS □ FAIL |
| **S2** | Hybrid mode superiority | F_hybrid > F_OMV | □ PASS □ FAIL |
| **S3** | Direction controllable | 8 distinct vectors | □ PASS □ FAIL |

### 8.3 Automatic Failure Conditions

```
🛑 IMMEDIATE TEST TERMINATION IF:

• Piezo temperature exceeds 90°C
• Sound level exceeds 130 dB
• Visible smoke, sparks, or flames
• Vacuum chamber crack or breach
• Balance fiber breaks
• Any personnel in chamber during fire
• Communication loss with control booth
```

### 8.4 Theoretical Predictions

RTM predicts the following thrust values (for comparison):

| Mode | Frequency | Voltage | Predicted Thrust |
|------|-----------|---------|------------------|
| OMV | 2 kHz | 150V | ~150-300 nN |
| OMV | 2 kHz | 200V | ~200-500 nN |
| TPH | 1 kHz | 150V | ~100 nN average |
| TPH | 10 kHz | 200V | ~500 nN average |

```
If measured thrust is:
  • Within 50-200% of prediction → STRONG VALIDATION
  • Within 10-500% of prediction → PARTIAL VALIDATION (investigate)
  • Outside 10-500% → INVESTIGATE SYSTEMATICS
  • Zero or negative → CHECK SETUP (does not invalidate theory)
```

---

## 9. EMERGENCY PROCEDURES

### 9.1 E-Stop Activation

```
WHEN TO HIT E-STOP:
• Temperature exceeds 90°C (if auto-interlock fails)
• Visible equipment damage
• Unexpected loud noise or vibration
• Fire or smoke
• Personnel emergency
• Loss of control system response

E-STOP PROCEDURE:
1. Press red mushroom button (either location)
2. ALL POWER is immediately severed
3. Announce "E-STOP ACTIVATED" on intercom
4. Wait 30 seconds (capacitor discharge)
5. Do NOT enter chamber until cleared
6. Document reason in test log
```

### 9.2 Thermal Runaway

```
IF PIEZO TEMPERATURE EXCEEDS 100°C:

1. Hit E-STOP immediately
2. Do NOT open chamber (thermal shock risk)
3. Wait 10 minutes for passive cooling
4. Monitor thermal camera for decay
5. When T < 50°C, safe to approach
6. Inspect piezo array for damage
7. If depolarization suspected:
   - Test piezo response at low voltage
   - If no response, replace array
```

### 9.3 Acoustic Emergency

```
IF SOUND LEVEL EXCEEDS 120 dB UNEXPECTEDLY:

1. Hit E-STOP immediately
2. Do NOT enter chamber
3. Check for resonance-induced damage:
   - Vacuum chamber integrity
   - Bell jar cracking
   - Balance fiber
4. If glass failure occurred:
   - Evacuate immediate area
   - Do not touch broken glass under vacuum stress
   - Call safety team
```

### 9.4 Fire

```
IF FIRE OR SMOKE OBSERVED:

1. Hit E-STOP
2. Evacuate all personnel
3. If small and contained: CO2 extinguisher
4. If spreading: Evacuate building, call fire department
5. Do NOT use water on electrical fire
```

### 9.5 Personnel Injury

```
IF HEARING DAMAGE SUSPECTED:
1. Remove person from acoustic environment
2. Do not shout at them (further damage)
3. Seek medical attention
4. Document incident

IF ELECTRICAL SHOCK:
1. Do NOT touch victim if still in contact
2. Cut power if safely possible
3. Call emergency services
4. If trained, begin CPR if needed
```

---

## 10. POST-TEST PROCEDURES

### 10.1 Immediate Post-Test

```
AFTER EVERY TEST FIRING:

□ 1. Wait 60 seconds minimum (cooldown)
□ 2. Verify piezo temperature < 50°C before next test
□ 3. Record all data from DAQ
□ 4. Save data file with proper naming convention
□ 5. Fill out test log entry
□ 6. Check for any anomalies
```

### 10.2 End of Day Procedures

```
END OF TEST SESSION:

□ 1. Complete final test log entry
□ 2. Power down HV supply
□ 3. Power down 24V supply
□ 4. If vacuum: vent chamber slowly
□ 5. Open chamber door
□ 6. Visual inspection of Mark 1 unit
□ 7. Photograph any wear or damage
□ 8. Secure all data files (backup to cloud)
□ 9. E-Stop reset and safety system check
□ 10. Lock laboratory
```

### 10.3 Post-Campaign Analysis

After completing all test phases:

```
ANALYSIS CHECKLIST:

□ Compile all data files
□ Plot thrust vs frequency (verify F ∝ f)
□ Plot thrust vs voltage (verify F ∝ V²)
□ Calculate mean and standard deviation for each condition
□ Compare to RTM theoretical predictions
□ Identify any systematic errors
□ Document lessons learned
□ Prepare final test report
```

### 10.4 Final Report Template

```
AETHERION MARK 1 TEST CAMPAIGN REPORT

1. Executive Summary
   - Key findings
   - Pass/Fail on primary criteria
   
2. Test Configuration
   - Equipment used
   - Calibration results
   
3. Results
   - Thrust measurements (all conditions)
   - Scaling law verification
   - Thermal performance
   
4. Analysis
   - Comparison to RTM predictions
   - Error analysis
   - Systematic uncertainties
   
5. Conclusions
   - Does data support RTM ponderomotive thrust?
   - Recommendations for Mark 2
   
6. Appendices
   - All raw data files
   - Calibration records
   - Test logs
   - Photographs
```

---

## APPENDIX A: Quick Reference Cards

### A.1 Console Commands

```
CONTROL SOFTWARE COMMANDS:

arm                     # Arm system for firing
disarm                  # Disarm system
fire <duration_ms>      # Fire for specified duration
set mode <omv|tph|hybrid>
set freq <Hz>
set voltage <V>
set phase <ch> <deg>    # Set individual channel phase
status                  # Print all sensor readings
temp                    # Print piezo temperatures
estop                   # Software E-stop
reset                   # Reset after E-stop
log start <filename>
log stop
help
```

### A.2 Normal Firing Sequence

```
STANDARD FIRING CHECKLIST:

1. Verify chamber sealed
2. Verify personnel clear
3. > arm
4. > set mode omv
5. > set freq 2000
6. > set voltage 150
7. > log start test_001.csv
8. > fire 5000
9. [Wait for completion]
10. > log stop
11. > disarm
12. Wait 60s cooldown
```

### A.3 Emergency Contacts

```
┌─────────────────────────────────────┐
│       EMERGENCY CONTACTS            │
├─────────────────────────────────────┤
│ Fire/Medical Emergency: 911         │
│ Facility Security: [INSERT]         │
│ Lab Manager: [INSERT]               │
│ Project Lead: [INSERT]              │
│ Poison Control: 1-800-222-1222      │
└─────────────────────────────────────┘
```

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         END OF TEST PROTOCOL                                 ║
║                                                                              ║
║                    AETHERION MARK 1 — ATP-MK1-001                            ║
║                           Revision 1.0                                       ║
║                                                                              ║
║              "Time is not what passes, but what pulses."                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
