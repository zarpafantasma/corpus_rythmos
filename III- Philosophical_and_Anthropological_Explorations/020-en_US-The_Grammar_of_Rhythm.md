<div align="center">

<img src="https://codeberg.org/Zarpa_Fantasma/corpus_rythmos/raw/branch/main/media/apollo.png" width="200" alt="Diagrama de Apolo">

# The Grammar of Rhythm
**Unifying Life, Mind, and Matter under RTM**
  
Álvaro Quiceno


</div>

**Abstract**

Time does not simply pass; it **learns** the shapes it moves through. A heartbeat, a thought, a step across a room—each keeps tempo with the size of the structure that carries it. This book names that relation **Multiscale Temporal Relativity (RTM)**: within a coherent window, the proper duration $`T`$ of a process follows the characteristic length $`L`$ that sustains it, and the relation is a slope $`\alpha`$ that says how time leans on form.

We keep the poetry, and we keep the proof. The scientific spine is simple and strict: $`\alpha`$ **is estimated only as a multi-point log–log slope of** $`\log T`$ **on** $`\log L`$ inside windows that pass collapse and regularity checks. When the world chooses other grammars—small-world shortcuts, for example—we say so ($`T \sim \log L`$) rather than forcing power. Signals like spectral slopes $`\beta`$, phase-locking, coherence, or echo/delay measures are welcomed as **auxiliary** companions to the story—illuminating, never substituting, and never converted into $`\alpha`$ by universal formula.

With that discipline, RTM becomes a lens for life, memory, intelligence, consciousness, and design: how $`\alpha`$**-bands** stabilize meaning, how gradients $`\nabla\alpha`$ channel energy and information, how failure (NO_COLLAPSE, LOG-SCALING, MULTI-REGIME) teaches where the tale must change scales. The philosophical arc is that coherence is not an ornament but a covenant between structure and time; the empirical arc is that this covenant can be **measured**, **falsified**, and **repaired**. We write in two voices—lyric and technical—so that the reader can feel the rhythm and verify it, in the same breath.

**Chapter 1 · Life — Rhythm That Sustains**

**Epigraph.** *A body is a metronome carved by distance.*

**1.1 A promise between structure and time (poetic prelude)**

Life is not a straight line—it is a **held tempo**. A capillary remembers how long blood should linger; a lung remembers how long air should stay; a limb remembers how long a step should take. Duration is not arbitrary. It leans on the size of the form that bears it. When form frays, time forgets. When form returns, time keeps time again.

**1.2 The claim (technical statement)**

Within a coherent window—one mechanism, one effective geometry—the **proper duration** $`T`$ of a process scales with a characteristic **length** $`L`$ as

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (multi-point slope estimated on a collapse-valid window).}
```

We estimate $`\alpha`$ only by regressing $`\log T`$ on $`\log L`$ with errors-in-variables (ODR/TLS or Theil–Sen, optional SIMEX), and only when **collapse diagnostics** (linearity, stability, bounded heteroscedasticity) pass. When the world chooses other grammars (e.g., shortcut topologies), we report $`T \sim \log L`$ instead of forcing a power.

**Policy (slope-first).** Spectral slopes $`\beta`$, phase locking, coherence, echo/delay measures, and similar indices are **auxiliary**: useful correlates under explicit models, never substitutes for $`\alpha`$, never converted into $`\alpha`$ by universal formulas.

**1.3 What “size” and “time” mean in living systems**

- **Structural length** $`L`$**:** a mechanism-tied scale—vessel path length or radius, membrane or dendritic cable length, fiber/axon span, cortical wavelength $`\lambda/2`$, organismal body length, or a validated surrogate that preserves geometry.

- **Proper time** $`T`$**:** the process duration coherent with that structure—circulation time, diffusion or reaction half-time, oscillation period $`T = 1/f`$, gait cycle, developmental stage time, turnover time.

- **Windows:** keep mechanism and metric constant (transport law, loading, topology). If mechanisms mix or metrics shift, report **MULTI-REGIME** or **NO_COLLAPSE** rather than averaging.

**1.4 Lyric Box · On breath**

Breath is a hinge between distances. The chest opens to the measure of its bones; alveoli count quietly, like beads. Blood does not rush because it can—it lingers because it must, keeping an appointment with the length of its corridors. What we name “rest” is the hourglass a body can lift without spilling.

*(Bridge: in the chapters ahead, choosing the structural scale* $`L`$ *fixes the span a process can bear; the slope* $`\alpha`$*tells how* $`T`$ *keeps pace with* $`L`$ *within that span.)*

**1.5 Measurement & estimation (how we keep the promise)**

**Dataset construction.** Assemble paired observations $`\{(L_{i},T_{i})\}`$across individuals, tissues, ROIs, or time windows where the mechanism is stable. Pre-register:

1.  the definition of $`L`$ and $`T`$,

2.  bin bounds and changepoint rules,

3.  exclusions (pathology, artifacts).

**Estimator.** Fit the slope of $`\log T`$ on $`\log L`$ with:

- **ODR/TLS** (default, both axes noisy),

- **Theil–Sen** (robust),

- **SIMEX** (optional, when measurement error is estimable).

**Collapse diagnostics.** Accept a window if: (i) residuals show no curvature, (ii) leave-one-out slope stays within tolerance, (iii) heteroscedasticity is bounded, (iv) no hidden segmented slopes. Else report **NO_COLLAPSE**; if semi-log is linear, report **LOG-SCALING**.

**Reporting.** Provide $`\widehat{\alpha}`$, 95% CI, estimator, $`n`$, bin specs, and a small residual/collapse panel.

**1.6 Mechanistic lenses (why** $`\mathbf{\alpha}`$**takes the values it takes)**

- **Diffusion-dominated:** $`T \sim L^{2}/D \Rightarrow \alpha \approx 2`$ when diffusivity is approximately scale-stable in the bin.

- **Advective/flow:** transit governed by coherent conduits trends toward $`\alpha \approx 1`$.

- **Trigger/ballistic:** propagation dominated by near-constant speeds also trends to $`\alpha \approx 1`$, unless the **effective metric** changes (shortcut networks), in which case $`T \sim \log L`$ (non-power).

- **Nested control:** recruitment/pruning changes the active $`L`$; slopes can segment across bins (report **MULTI-REGIME** with breakpoints).

These are signposts; $`\alpha`$ is always measured, never assumed.

**1.7 Case sketches (illustrative, data-agnostic)**

**1.7.1 Circulatory timing**

Let $`L`$ be characteristic vessel path length (network geodesic); $`T`$the indicator-dilution mean transit time. Expect $`\alpha \approx 1`$ in coherent flow; curvature suggests capillary diffusion mixing or loading-dependent velocity drift.

**1.7.2 Cellular diffusion**

$`L`$ as dendritic path or morphogen reach; $`T`$ as equilibration half-time. Clean bins often show $`\alpha \approx 2`$; if $`D`$ drifts with $`L`$, slopes segment.

**1.7.3 Neural rhythms (source-level)**

$`T = 1/f`$ for band peaks; $`L = \lambda/2`$ from source-modeled wavelength. Estimate $`\alpha`$ via EIV across ROIs/windows; report PLV/coherence and spectral $`\beta`$ as **auxiliary**, not $`\alpha`$.

**1.7.4 Behavior (gait, respiration)**

$`L`$ as limb length or airway path; $`T`$ as cycle period. $`\alpha \approx 1`$ is common within posture-constant bins. Transitions (walk→run; voluntary→paced) often break collapse.

**1.8 Technical Box · Estimating** $`\mathbf{\alpha}`$**in living systems (protocol)**

**Inputs:** paired $`(L,T)`$and a pre-registered window.\
**Steps:**

1.  Fit EIV slope (ODR/TLS; Theil–Sen robustness).

2.  Bootstrap CIs; record $`n`$and span in $`L`$.

3.  Run collapse diagnostics (curvature, LOO stability, heteroscedasticity checks).

4.  Report $`\widehat{\alpha}`$ with CI and a residual/collapse mini-panel.\
    **Outcomes:** $`\widehat{\alpha}`$ if collapse passes; **NO_COLLAPSE** or **LOG-SCALING** otherwise.

*Caption template (slope):* “Scaling of $`T`$ with $`L`$ in \[system\]. EIV slope $`\widehat{\alpha}`$ (95% CI) on a collapse-valid bin (n = …). Residuals show no curvature.”

**1.9 Technical Box · Using auxiliaries without breaking the spell**

- **Report** spectral $`\beta`$, PLV/coherence, band powers, burst/ISI statistics with methods and CIs.

- **Do not convert** auxiliaries to $`\alpha`$ (e.g., no universal $`\alpha = 1 + \beta/2`$).

- **Use** auxiliaries as covariates to interpret changes in $`\widehat{\alpha}`$, or to select bins; they **inform** the slope, they do not **define** it.

*Caption template (auxiliary):* “HRV spectral slope $`\beta`$ (auxiliary; not $`\alpha`$). Method: \[..\]; CI: \[..\]. No conversion.”

**1.10 Failure modes & meanings**

- **NO_COLLAPSE:** the line won’t hold → mixed mechanisms or wrong metric. Remedy: re-bin, refine $`L`$/$`T`$, or report the negative.

- **LOG-SCALING:** $`T \sim \log L`$ due to shortcut geometries; label explicitly; do not infer $`\alpha`$.

- **MULTI-REGIME:** piecewise-linear slopes on log–log → report segmented $`\alpha`$ with breakpoints and mechanism notes.

**1.11 Lyric coda · What life protects**

Life protects **intervals**. Not just a heartbeat, but the **right** length of a heartbeat; not just a breath, but the **right** length of a breath. The word for that rightness here is $`\alpha`$—a modest name for a covenant: how far a structure reaches and how long a process is allowed to become itself.

**1.12 What this chapter does not do**

- It does **not** infer $`\alpha`$ from single-point formulas (e.g., $`\log(T_{i}/T_{0})/log(L_{i}/L_{0})`$).

- It does **not** treat spectral $`\beta`$, PLV/coherence, or echo/delay as $`\alpha`$.

- It does **not** force power laws when topology implies $`T \sim \log L`$.

**1.13 Take-home**

To sustain life is to keep a workable $`\alpha`$**-band**: a structural span $`L`$ and a temporal span $`T`$ where the slope is real, testable, and kind to the organism. The rest of the book measures how that band is found, kept, lost, and found again.

**Chapter 2 · Memory — Folds in Time**

**Epigraph.** *To remember is to keep a door open between two moments.*

**2.1 Poetic prelude · The fold**

Memory is a fold the present makes to touch itself later. A dendrite tucks a signal under its branch; a map in cortex creases the world so it can be carried. We do not store things—we store **durations** shaped by **distances**. When the fold holds, the future arrives already half-remembered.

*(Bridge: in RTM, a “fold” is the pairing of a structural scale* $`L`$ *with a proper duration* $`T`$ *that keeps pace through a stable slope* $`\alpha`$ *measured on a collapse-valid window.)*

**2.2 The claim (technical statement)**

Within a consistent mechanism (synaptic, systems-level, or behavioral), memory times $`T`$ scale with structural lengths $`L`$ as

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (multi-point slope on a collapse-valid window)}.
```

We estimate $`\alpha`$ by errors-in-variables regression (ODR/TLS or Theil–Sen; optional SIMEX) and publish **NO_COLLAPSE** when diagnostics fail. Shortcut-like retrieval topologies are reported as $`T \sim \log L`$ (non-power), not forced into a power law.

**Policy (slope-first).** Spectral slope $`\beta`$, phase-locking/coherence, ripple/burst rates, and delay metrics are **auxiliary**: they may co-vary with $`\alpha`$under explicit models but are never converted into $`\alpha`$ by a universal formula.

**2.3 What “size” and “time” mean for memory**

- **Structural scale** $`L`$**:** spine neck length or head volume; dendritic cable length/branch order; minicolumn width; ensemble/map radius; axon/pathway length; hippocampo–cortical projection span. Choose one per dataset and keep it fixed within the window.

- **Proper time** $`T`$**:** STDP/plasticity window width; consolidation delay; retention half-life; replay cycle period; working-memory decay to criterion; retrieval latency at fixed accuracy.

**Windows.** Keep mechanism, state, and task constant. Mixing (e.g., consolidation plus rehearsal) breaks collapse—report **MULTI-REGIME** or segment slopes with a registered breakpoint.

**2.4 Lyric Box · The library of distances**

A library is not shelves of objects but corridors of **reach**. We return to a thought by walking its hallway again. The longer the corridor, the longer the light must be left on. Some books are at arm’s length; others require a lantern and a patient stride.

*(Bridge: corridor length* $`L`$*; light-on time* $`T`$*. Where the relation is stable, the log–log slope is* $`\alpha`$*.)*

**2.5 Mechanistic lenses (why** $`\mathbf{\alpha}`$**takes these values)**

- **Diffusion–reaction memory (local biophysics):** when signaling spreads through spines and dendrites, $`T \sim L^{2}/D`$and $`\alpha \approx 2`$ in homogeneous compartments.

- **Routing & integration (loops and paths):** conduction with integration across a route of length $`L`$trends toward $`\alpha \approx 1`$ if velocities and thresholds are scale-stable.

- **Shortcut retrieval (associative nets):** effective geodesics collapse via hubs/indices; timing behaves $`T \sim \log L`$ (label **LOG-SCALING**).

- **Two-store interactions:** labile/fast and stable/slow stores often produce **segmented** slopes (breakpoint = store switch).

These are guides; $`\alpha`$ is measured from data in a defined window.

**2.6 Building datasets for memory scaling**

**Paired observations** $`\{(L_{i},T_{i})\}`$**:**

- **Synaptic memory:** $`L`$=spine/dendritic length measure; $`T`$=plasticity or biochemical decay constant from the same compartment class.

- **Systems consolidation:** $`L`$=pathway length or map radius; $`T`$=delay to hippocampal independence under controlled sleep/wake regimes.

- **Behavioral memory:** $`L`$=chunk/sequence length with fixed rehearsal/feedback; $`T`$=retention half-life or recall latency.

**Pre-registration:** scale & clock definitions; bin limits; state controls; exclusion criteria (pharmacology, pathology, ceiling/floor effects).

**2.7 Estimating** $`\mathbf{\alpha}`$ **(methods)**

- **Estimator:** EIV slope of $`\log T`$on $`\log L`$: **ODR/TLS** (default), **Theil–Sen** (robust), optional **SIMEX** for error correction.

- **Collapse diagnostics:** (i) linearity on log–log (no curvature in residuals), (ii) leave-one-out stability of slope, (iii) bounded heteroscedasticity, (iv) no hidden changepoints.

- **Reporting:** $`\widehat{\alpha}`$, 95% CI, estimator, $`n`$, span in $`L`$, bin specification, residual/collapse mini-panel.

- **Failure modes:** **NO_COLLAPSE** (mechanism mixing), **LOG-SCALING** (shortcut topology), **MULTI-REGIME** (segmented slopes with a breakpoint and interpretation).

**2.8 Case sketches (illustrative)**

**2.8.1 Spine-to-soma consolidation**

Pair soma–spine path length $`L`$ with consolidation time $`T`$. Expect $`\alpha \approx 2`$ if diffusion–reaction dominates; deviations signal active transport or compartmentalization changes.

**2.8.2 Map radius and systems delay**

Let $`L`$ be the radius of a cortical map or the effective hippocampo–cortical route; $`T`$ the delay to cortical takeover. Near-linear $`\alpha \approx 1`$ is plausible; segmented slopes may mark sleep-stage dependencies.

**2.8.3 Working memory span**

Use chunk size $`L`$ under fixed interference; $`T`$=decay to criterion. Often **MULTI-REGIME**: small chunks hold via rehearsal (one slope), longer chunks fail collapse or switch mechanism.

**2.8.4 Associative lookup with indices**

If retrieval uses hubs or addressable indices, timing follows $`T \sim \log L`$; report semi-log fit parameters and refrain from $`\alpha`$.

**2.9 Technical Box · Measuring** $`\mathbf{\alpha}`$**in memory (protocol)**

**Inputs:** paired $`(L,T)`$ within a pre-registered window.\
**Steps:**

1.  Fit **ODR/TLS** slope on $`(\log L,\log T)`$; add **Theil–Sen** as robustness.

2.  Bootstrap 95% CIs; record $`n`$, span, and estimator.

3.  Run collapse diagnostics (residual curvature, LOOCV slope drift, heteroscedasticity bounds).

4.  Report $`\widehat{\alpha}`$ with a residual/collapse inset.\
    **Outcomes:** $`\widehat{\alpha}`$ (collapse-valid) or **NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**.

*Caption template (slope):* “Memory time $`T`$ vs. structural scale $`L`$ in \[system\]. EIV slope $`\widehat{\alpha}`$ (95% CI) on a collapse-valid bin (n = …). Residuals show no curvature.”

**2.10 Technical Box · Using auxiliaries without breaking the fold**

- **Report** spectral $`\beta`$, PLV/coherence, ripple/burst rates with methods and CIs.

- **Do not convert** auxiliaries to $`\alpha`$.

- **Use** auxiliaries as covariates to interpret changes in $`\widehat{\alpha}`$ or to select bins; they **illuminate** the fold, they do not **define** it.

*Auxiliary caption template:* “Spectral slope $`\beta`$ during consolidation. Auxiliary (not $`\alpha`$); no conversion. Method/CI reported.”

**2.11 Failure modes & meanings**

- **NO_COLLAPSE:** mixing processes (e.g., rehearsal plus consolidation). Remedy: narrower bins, mechanism-specific $`L`$.

- **LOG-SCALING:** shortcut retrieval; treat with semi-log fits; do not infer $`\alpha`$.

- **MULTI-REGIME:** changepoint between stores; report segmented slopes with interpretation.

**2.12 Lyric coda · What remembering really keeps**

What we keep is not the picture but the **interval** needed to find it again. We keep the time a structure asks of us—the minute a corridor requires, the breath a sentence takes. To remember is to honor that exchange: a length for a duration, a distance for a stay.

**2.13 What this chapter does not do**

- It does **not** infer $`\alpha`$ from single-point ratios.

- It does **not** convert spectral $`\beta`$, PLV/coherence, or ripple metrics into $`\alpha`$.

- It does **not** force power laws where retrieval behaves $`T \sim \log L`$.

**2.14 Take-home**

Memory is a **discipline of folds**—pairing structural spans $`L`$ with durations $`T`$ so that a slope $`\alpha`$ holds. The poetry is the image of a corridor lit just long enough; the science is the slope that proves the light was set to the corridor’s length.

**Chapter 3 · Intelligence — The Adaptive Dance**

**Epigraph.** *A mind is a doorway that learns the size of the rooms it enters.*

**3.1 Poetic prelude · The leap that lands**

Intelligence is the art of **arriving**—not just moving. It is the knack for choosing a structure that will bear the time you are about to spend. The expert does not merely think faster; they **stand in the right-sized place**. From there, answers feel close, not because distance vanished, but because distance and duration made a pact.

*(Bridge: we will name that pact by a slope* $`\alpha`$ *that ties process time* $`T`$ *to structural size* $`L`$ *inside a coherent window.)*

**3.2 The claim (technical statement)**

Within a stable mechanism (architecture, policy, cost), intelligent behavior operates in windows where

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (multi-point slope on a collapse-valid window).}
```

We estimate $`\alpha`$ only by errors-in-variables regression (ODR/TLS or Theil–Sen; optional SIMEX) and publish **NO_COLLAPSE** when diagnostics fail. When the effective metric is shortcut-like (e.g., indexed retrieval), we report $`T \sim \log L`$ instead of forcing a power law.

**Policy (slope-first).** Information scores, spectral $`\beta`$, synchrony/coherence, confidence, and reward proxies are **auxiliary**—interpretive covariates, never substitutes for $`\alpha`$, never converted to $`\alpha`$ by universal formulas.

**3.3 What “size” and “time” mean for intelligence**

- **Structural scale** $`L`$**:** representation granularity (receptive-field radius, context window), problem size (grid width, clause count), search radius or planning horizon, module/ensemble size, actuator workspace/path length.

- **Proper time** $`T`$**:** decision latency, time-to-criterion, settling time to tolerance, convergence time for a policy/estimator, wall-clock to stable performance.

**Windows.** Hold architecture, optimizer/policy, task distribution, and cost constant. Strategy switches or curriculum changes mid-window break collapse—report **MULTI-REGIME** rather than averaging.

**3.4 Lyric Box · Choosing the room**

Some rooms invite silence; others ask for longer sentences. Intelligence is knowing **which room** the question belongs to. Choose a space too small and thinking stutters; too large and it echoes into indecision. The right room answers back in one breath.

*(Bridge: the room is* $`L`$*; the breath is* $`T`$*; the reply that fits is the slope* $`\alpha`$*measured on a collapse-valid window.)*

**3.5 Mechanistic lenses (why** $`\mathbf{\alpha}`$ **takes these values)**

- **Aggregation & routing:** computation that grows proportionally with representational span often yields $`\alpha \approx 1`$ when bandwidths are scale-stable.

- **Combinatorics tamed:** naive branching pushes $`T`$ supralinear; heuristics/learned abstractions can compress the **effective** metric to shortcut graphs, producing $`T \sim \log L`$ (non-power).

- **Control hierarchies:** multi-level controllers select an $`L`$ that keeps time affordable; regime switches appear as segmented slopes (**MULTI-REGIME**).

- **Learning dynamics:** batch/replay windows and target lags define operative $`L`$ in parameter/state space; within fixed settings, near-power scaling can appear.

These are guides; $`\alpha`$ is always measured.

**3.6 Building datasets for intelligence scaling**

Construct paired observations $`\{(L_{i},T_{i})\}`$ by **parametrically varying one structural scale** while holding mechanisms constant.

**Examples**

- **Sequence models:** $`L`$=context length; $`T`$=steps/wall-clock to a fixed validation score with architecture/optimizer frozen.

- **Reinforcement learning:** $`L`$=planning horizon; $`T`$=episodes-to-criterion under fixed exploration and reward shaping.

- **Robotics/control:** $`L`$=path length or workspace radius; $`T`$=settling time to tolerance with fixed controller gains.

- **Human problem solving:** $`L`$=problem size (grid width, clause count, sequence length); $`T`$=median solution latency at fixed instructions.

**Pre-register:** scale/clock definitions; bin limits; stopping/timeout rules; exclusions (policy resets, curriculum jumps).

**3.7 Estimating** $`\mathbf{\alpha}`$**(methods)**

- **Estimator:** EIV regression of $`\log T`$ on $`\log L`$: **ODR/TLS** (default), **Theil–Sen** (robust), optional **SIMEX**.

- **Collapse diagnostics:** linearity on log–log; leave-one-condition-out stability; bounded heteroscedasticity; no hidden changepoints.

- **Reporting:** $`\widehat{\alpha}`$, 95% CI, estimator, $`n`$, span in $`L`$, bin specification, residual/collapse mini-panel.

- **Failure modes:** **NO_COLLAPSE** (mixing/curvature), **LOG-SCALING** (shortcut metric), **MULTI-REGIME** (segmented slopes with breakpoints).

**3.8 Case sketches**

**3.8.1 Context span vs. learning time (sequence models)**

Keep architecture/optimizer fixed; vary context tokens $`L`$. Let $`T`$=time to hit a validation threshold. Clean bins often show $`\alpha \approx 1 - 1.5`$; optimizer/curriculum shifts create segmented slopes.

**3.8.2 Planning horizon vs. episodes (RL)**

$`L`$=search horizon; $`T`$=episodes-to-criterion under fixed exploration. Heuristics/abstractions sometimes induce **LOG-SCALING**; otherwise supralinear slopes appear when branching dominates.

**3.8.3 Workspace radius vs. settling time (robots)**

$`L`$=radius or path length; $`T`$=settling time. Near-linear $`\alpha`$ is common with well-tuned controllers; torque limits or re-planning show up as breakpoints.

**3.8.4 Human insight under time pressure**

$`L`$=problem scale (e.g., syllogism scope); $`T`$=time-to-first-solution at fixed accuracy. Strategy drift → **NO_COLLAPSE**; template retrieval at large $`L`$ may flip to **LOG-SCALING**.

**3.9 Technical Box · Experiment design for** $`\mathbf{\alpha}`$

**Goal:** maximize power for a true slope; avoid false fits from mixing.

1.  **Span & granularity:** ≥0.6–1.0 decades in $`L`$; ≥6 distinct $`L`$ levels.

2.  **Constancy:** freeze optimizer/policy/settings inside the bin; log deviations.

3.  **Replicates:** ≥3 runs per $`L`$; use EIV and consider **SIMEX** when error grows with $`L`$.

4.  **Diagnostics:** pre-register collapse tests; plot residuals; run changepoint checks.

5.  **Outcomes:** $`\widehat{\alpha}`$ with CI (collapse-valid) or a labeled negative (**NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**).

**3.10 Using auxiliaries without breaking the rules**

**Auxiliaries:** predictive information, spectral $`\beta`$, synchrony/coherence, gradient/entropy measures, confidence.

- Report methods and uncertainty; use as covariates to **interpret** $`\widehat{\alpha}`$ or to choose bins.

- **Do not** convert auxiliaries to $`\alpha`$ or average them into “composite $`\alpha`$”.

*Auxiliary caption template:* “Predictive information vs. context. Auxiliary (not $`\alpha`$); no conversion. Parameters and CI reported.”

**3.11 Failure modes specific to intelligence**

- **Strategy drift inside a bin:** policy changes or instruction creep → curvature; re-bin or enforce fixed policy.

- **Hidden resource caps:** memory/compute limits activate at large $`L`$; segmented slopes emerge—report breakpoints.

- **Shortcut artifacts:** indexing/retrieval compresses distance → **LOG-SCALING**; label explicitly.

- **Reward confounds:** reward shaping alters $`T`$ without changing $`L`$; treat as a different mechanism or control for it.

**3.12 Lyric coda · The dance**

A good answer begins before it is spoken. The mind steps into the **room that fits**, and time, relieved, keeps pace. The dance is not speed but **fit**—a slope held steady while the music changes.

**3.13 What this chapter does not do**

- It does **not** infer $`\alpha`$ from single-point ratios or auxiliary metrics.

- It does **not** mix regimes to force a line.

- It does **not** impose power laws where topology implies $`T \sim \log L`$.

**3.14 Take-home**

Intelligence is **adaptive band-keeping**: picking $`L`$ and $`T`$ so a coherent $`\alpha`$ exists and holds. Measure it with slope-first estimation on collapse-valid windows; let auxiliaries sing harmony, not carry the tune.

**Chapter 4 · Consciousness — The Integrative Window**

**Epigraph.** *The world arrives not all at once, but as a chord you learn to hold.*

**4.1 Poetic prelude · How the chord holds**

To be conscious is to **keep** a chord from fraying. Sensations, memories, and intentions arrive at different lengths, asking for different amounts of time. When the chord holds, a moment becomes a room where many voices can stand together. When it slips, the room shatters into noise or sleep.

*(Bridge: we will speak of rooms by a structural scale* $`L`$*, of how long they can be kept by a duration* $`T`$*, and of the pact between them by a slope* $`\alpha`$ *measured on a collapse-valid window.)*

**4.2 The claim (technical statement)**

Within a coherent state—fixed task, arousal, and effective geometry—conscious access operates in windows where

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (multi-point slope estimated on a collapse-valid window)}.
```

We estimate $`\alpha`$by errors-in-variables regression (ODR/TLS or Theil–Sen; optional SIMEX), and we publish **NO_COLLAPSE** when diagnostics fail. Where effective topologies provide shortcuts (hub-dominated broadcast), we report $`T \sim \log L`$ rather than forcing a power law.

**Policy (slope-first).** Spectral slope $`\beta`$, PLV/coherence, perturbational indices, ignition thresholds, and delay/echo measures are **auxiliary**—interpretive correlates under explicit models, never substitutes for $`\alpha`$, never converted to $`\alpha`$ by universal formula.

**4.3 What “size” and “time” mean for consciousness**

- **Structural scale** $`L`$**:** source-modeled spatial wavelength $`\lambda/2`$ of active assemblies; radius of functional coalitions; hub-to-hub path length in an effective broadcast graph; thalamo–cortical loop span; representational “map” width.

- **Proper time** $`T`$**:** evidence accumulation time; access or report latency corrected for motor delay; ignition dwell time; binding window; sustained attention window.

**Windows.** Keep state and mechanism constant: task block, arousal band, and network configuration. Strategy swaps (e.g., attention shifts) or state drifts break collapse—report **MULTI-REGIME** or **NO_COLLAPSE**.

**4.4 Lyric Box · The window**

A window is not a hole in a wall; it is a promise that **what fits will be held**. Too small, and the world is chopped into shards. Too large, and the world washes through ungrasped. Consciousness is the craft of choosing a pane that keeps the wind and lets in the light.

*(Bridge: pane size* $`L`$*; holding time* $`T`$*; within a workable pane,* $`T`$*keeps pace with* $`L`$*by a slope* $`\alpha`$*.)*

**4.5 Mechanistic lenses (why** $`\mathbf{\alpha}`$**takes these values)**

- **Recurrent accumulation:** integration across assemblies with near-constant velocities and thresholds often yields $`\alpha \approx 1`$.

- **Diffusive integration:** when evidence spreads over representational distance, $`\alpha \approx 2`$can appear.

- **Global broadcast:** routing via long-range hubs increases $`L`$; efficient loops can preserve $`\alpha \approx 1`$, while bottlenecks produce segmented slopes (**MULTI-REGIME**).

- **Shortcut geometry:** strong hub shortcuts compress effective distance, producing $`T \sim \log L`$ (non-power); label explicitly.

- **Gain control & inhibition:** changes in excitability shift intercepts (clocks) more than slopes; persistent asymmetries may reveal hidden regime mixing.

These are signposts; $`\alpha`$ is always measured from data in a defined window.

**4.6 Building datasets for conscious access**

Construct paired observations $`\{(L_{i},T_{i})\}`$ under controlled state and mechanism.

**Paradigms**

1.  **Perceptual access:** $`L = \lambda/2`$ from source-modeled assembly extent; $`T`$=access or report latency at fixed accuracy (motor-corrected).

2.  **Perturb-and-measure (e.g., TMS/perturbation):** $`L`$=evoked assembly radius/path; $`T`$=time-to-stable response or ignition dwell within criterion bounds.

3.  **Binding & attention:** $`L`$=feature-map or spatial span of bound content; $`T`$=binding window or attentional dwell.

**Pre-register:** scale/clock definitions; bin limits; exclusions (arousal drift, lapses); motor-delay correction; artifact handling.

**4.7 Estimating** $`\mathbf{\alpha}`$ **(methods)**

- **Estimator:** EIV regression on $`(\log L,\log T)`$ using **ODR/TLS** (default) or **Theil–Sen** (robust); optional **SIMEX** if error structure is characterizable.

- **Collapse diagnostics:** (i) log–log linearity (no residual curvature), (ii) leave-one-condition-out slope stability, (iii) bounded heteroscedasticity, (iv) no hidden changepoints.

- **Reporting:** $`\widehat{\alpha}`$, 95% CI, estimator, $`n`$, span in $`L`$, bin specification, residual/collapse mini-panel.

- **Failure modes:** **NO_COLLAPSE** (mixing/curvature), **LOG-SCALING** (shortcut metric), **MULTI-REGIME** (segmented slopes with breakpoints).

**4.8 Case sketches (illustrative)**

**4.8.1 Assembly span vs. access latency**

Vary stimulus configurations to recruit assemblies of differing spatial spans; source-model to obtain $`L = \lambda/2`$. Define $`T`$as motor-corrected access latency. Clean bins often show $`\alpha \approx 1`$; curvature or segmented slopes indicate bottlenecks or state drift.

**4.8.2 Broadcast reach vs. ignition dwell**

Define $`L`$ by hub-to-hub path length in an effective broadcast graph; measure $`T`$ as ignition dwell (above a pre-registered threshold). Efficient integration can maintain near-linear slopes; bottlenecked routing segments the slope.

**4.8.3 Masking, sedation, and state transitions**

Within narrow state steps, $`L`$ from source-modeled assembly and $`T`$ as access window reveal whether slope persists. Crossing regimes yields **NO_COLLAPSE** or breakpoints (report **MULTI-REGIME**). Deep states may show **LOG-SCALING** if only shortcuts remain.

**4.9 Technical Box · Measuring** $`\mathbf{\alpha}`$**during conscious access**

**Inputs:** paired $`(L,T)`$within a state-constant bin.\
**Steps:**

1.  Source-model assemblies to get $`L = \lambda/2`$; define $`T`$ as access/binding/ignition time with motor correction.

2.  Fit **ODR/TLS** slope of $`\log T`$ on $`\log L`$; add **Theil–Sen** as robustness.

3.  Bootstrap CIs; run collapse diagnostics (residual curvature, LOO stability, heteroscedasticity checks).

4.  Report $`\widehat{\alpha}`$, CI, estimator, $`n`$, span, bin details, and a residual/collapse inset.\
    **Outcomes:** $`\widehat{\alpha}`$ (collapse-valid) or **NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**.

*Caption template (slope):* “Conscious access time $`T`$ vs. assembly scale $`L`$. EIV slope $`\widehat{\alpha}`$ (95% CI) on a collapse-valid bin (n = …). Residuals show no curvature.”

**4.10 Technical Box · Auxiliary signals without overreach**

**Auxiliaries:** spectral slope $`\beta`$, PLV/coherence, perturbational complexity, ignition thresholds, ERP components.

- Report parameters and uncertainty; use as covariates to interpret $`\widehat{\alpha}`$ or to select bins.

- **Do not** convert auxiliaries into $`\alpha`$or average them into “composite $`\alpha`$”.

*Auxiliary caption template:* “PLV during access. Auxiliary (not $`\alpha`$); no conversion. Parameters and CI reported.”

**4.11 Placebos & controls (clock vs. structure)**

- **Clock placebo:** rescale timestamps or add small jitter; $`\widehat{\alpha}`$ should remain within CI if slope reflects structure, not clock artifacts.

- **Structure placebo:** shuffle spatial patterns while keeping the clock; collapse should fail or slope should drift if structure drives $`\alpha`$.

- **Motor-delay control:** subtract or fix motor/report delay across $`L`$ so intercept changes do not masquerade as slope shifts.

**4.12 Failure modes & meanings**

- **Arousal drift inside the bin** → curvature; re-bin by tighter state bands.

- **Sensor-space leakage** → biased $`L`$; prefer source models or leakage-controlled surrogates.

- **Shortcut contamination** → $`T \sim \log L`$; label **LOG-SCALING**, do not infer $`\alpha`$.

- **Hidden changepoints** → **MULTI-REGIME**; report breakpoints with interpretation.

**4.13 Lyric coda · The chord you can keep**

Consciousness is not a light turned on but a **holding** in which many lengths agree to speak for the same amount of time. The miracle is humble: pick a pane, keep a promise, let the chord ring long enough to mean.

**4.14 What this chapter does not do**

- It does **not** infer $`\alpha`$from spectral $`\beta`$, PLV/coherence, PCI, ignition thresholds, or echo/delay metrics.

- It does **not** accept single-point formulas as $`\alpha`$.

- It does **not** force power laws where topology yields $`T \sim \log L`$.

**4.15 Take-home**

Consciousness is an **integrative window**: a structural span $`L`$ held for a duration $`T`$ so that a measurable slope $`\alpha`$ exists. The lyric tells why we care; the slope tells whether the window truly holds.

**Chapter 5 · Intuition — Cross-Scale Leaps**

**Epigraph.** *Sometimes the path is short because the body remembers it.*

**5.1 Poetic prelude · How the answer arrives early**

Intuition is the **quiet shortcut** that doesn’t lie. It feels like leaping, but it is mostly **remembering the right size**—standing in a shape that already knows how long the work should take. What looks like magic is a practiced fit: the world offers a span, and we answer with the time that span can hold.

*(Bridge: we will name the span by a structural scale* $`L`$*, the answer-time by a duration* $`T`$*, and their pact by a slope* $`\alpha`$*measured on a collapse-valid window.)*

**5.2 The claim (technical statement)**

Within a stable mechanism (task, representation, control policy), intuitive performance occupies windows where

``` math
T \propto L^{\alpha},\alpha = \frac{d\log T}{d\log L}\text{ (multi-point slope on a collapse-valid window).}
```

We estimate $`\alpha`$ with errors-in-variables regression (ODR/TLS or Theil–Sen; optional SIMEX) and publish **NO_COLLAPSE** when diagnostics fail. When the effective metric provides hard shortcuts (addressing, hashing, direct retrieval), we report $`T \sim \log L`$ rather than forcing a power law.

**Policy (slope-first).** Confidence, spectral slope $`\beta`$, PLV/coherence, and heuristic scores are **auxiliary**: interpretive covariates, never substitutes for $`\alpha`$, never converted to $`\alpha`$ by universal formula.

**5.3 What “size” and “time” mean for intuition**

- **Structural scale** $`L`$**:** chunk size or representational granularity; number of retrieved elements in a schema; search neighborhood radius; template span in motor space; context window in rapid judgment.

- **Proper time** $`T`$**:** time-to-first-action; decision latency at fixed accuracy; solution latency under brief exposure; stabilization time after a one-shot guess.

**Windows.** Hold instructions, payoff, and resources constant (no mid-trial strategy shifts). If strategies mix (recall → search), report **MULTI-REGIME** or segment slopes; do not average.

**5.4 Lyric Box · The right-sized glance**

A good glance is not speed but **fit**. You place your attention in a bowl the problem can sit in. If the bowl is too small, the truth spills; too wide, and the taste thins. The right bowl holds flavor long enough to know.

*(Bridge: the bowl is* $`L`$*; the savoring time is* $`T`$*; the held proportion is* $`\alpha`$*.)*

**5.5 Mechanistic lenses (why** $`\mathbf{\alpha}`$ **takes these values)**

- **Template retrieval:** reusing a stored mapping at scale $`L`$ often yields near-linear timing ($`\alpha \approx 1`$) when access and actuation bandwidths are stable.

- **Chunked reasoning:** increasing chunk size raises $`L`$ while avoiding serial search; slopes depend on chunk assembly cost—breakpoints mark the limit of usable chunks (**MULTI-REGIME**).

- **Similarity kernels & indexing:** if access is effectively logarithmic in span (e.g., addressable memory), timing behaves $`T \sim \log L`$ (non-power).

- **Gate & go:** improved clocks reduce intercepts (faster $`a`$ in $`\log T = a + \alpha\log L`$) without changing $`\alpha`$; distinguish clock shifts from slope changes.

These are guides; $`\alpha`$is always measured from data within a defined window.

**5.6 Building datasets for intuitive scaling**

Create paired observations $`\{(L_{i},T_{i})\}`$by **parametrically varying one structural scale** while freezing mechanisms.

**Examples**

- **Perceptual snap judgments:** $`L`$=patch/receptive-field radius; $`T`$=time to first correct category at fixed accuracy.

- **Motor one-shots:** $`L`$=trajectory span or obstacle field size; $`T`$=time-to-commit using a learned primitive.

- **Cognitive insight tasks:** $`L`$=problem scale (grid width, clause count); $`T`$=latency under brief exposure with no scratch work.

- **Expert retrieval:** $`L`$=schema size (linked cue count); $`T`$=time-to-first confident response.

**Pre-register:** definitions of $`L,T`$; bin bounds; exclusion rules (timeouts, low-confidence trials if accuracy is fixed); training history allowed before testing.

**5.7 Estimating** $`\mathbf{\alpha}`$**(methods)**

- **Estimator:** EIV slope of $`\log T`$on $`\log L`$: **ODR/TLS** (default), **Theil–Sen** (robust), optional **SIMEX** for error correction.

- **Collapse diagnostics:** (i) log–log linearity with no residual curvature, (ii) leave-one-condition-out slope stability, (iii) bounded heteroscedasticity, (iv) no hidden changepoints.

- **Reporting:** $`\widehat{\alpha}`$, 95% CI, estimator, $`n`$, span in $`L`$, bin specification, residual/collapse inset.

- **Failure modes:** **NO_COLLAPSE** (mixed strategies), **LOG-SCALING** (true shortcuts), **MULTI-REGIME** (template vs. search breaks).

**5.8 Case sketches (illustrative)**

**5.8.1 Rapid categorization**

Vary $`L`$ (stimulus span) at constant difficulty; $`T`$=time to first correct category. Clean bins often show $`\alpha \approx 1`$; curvature indicates competition between foveal templates and peripheral search.

**5.8.2 One-shot motor selection**

Let $`L`$=movement span; $`T`$=latency to commit using a trained primitive. Near-linear slopes appear when the primitive covers the span; at large $`L`$ re-planning inserts breakpoints (**MULTI-REGIME**).

**5.8.3 Numerical estimation**

$`L`$=numerosity or spatial extent of a compressed code; $`T`$=time to estimate under brief display. If coding is logarithmic, label **LOG-SCALING**; do not infer $`\alpha`$.

**5.8.4 Expert pattern retrieval**

$`L`$=schema size; $`T`$=time-to-first move/answer. Experts show lower intercepts (faster clocks) and cleaner collapse; unfamiliar motifs produce segmented slopes.

**5.9 Technical Box · Clean intuition experiments**

**Goal:** isolate fast band-selection from deliberate search.

1.  **Instruction lock:** respond on first confident impression; strict response windows.

2.  **Span & granularity:** ≥0.6–1.0 decades in $`L`$; ≥6 distinct levels.

3.  **No rehearsal inside the bin:** randomize $`L`$; prevent strategy drift.

4.  **Replicates:** ≥3 trials per level; model measurement error.

5.  **Diagnostics:** pre-register collapse tests; residuals and changepoint checks.

**Outcome:** $`\widehat{\alpha}`$ with CI (collapse-valid) **or** a labeled negative (NO_COLLAPSE / LOG-SCALING / MULTI-REGIME).

**5.10 Technical Box · Separating clock from slope**

Fit $`\log T = a + \alpha\log L`$ across conditions (e.g., novice vs. expert).

- **Equal** $`\alpha`$**, lower** $`a`$**:** faster clock (practice) with same structure–time grammar.

- **Different** $`\alpha`$**:** different active structure (re-bin) or mechanism shift.\
  Report both parameters with CIs; avoid conflating speed with coherence.

**5.11 Using auxiliaries without overreach**

**Auxiliaries:** confidence, spectral $`\beta`$, PLV/coherence, pupil diameter, surprise/entropy.

- Report methods and uncertainty; use as covariates to interpret $`\widehat{\alpha}`$ or choose bins.

- **Never** convert auxiliaries into $`\alpha`$ or average them into a “composite $`\alpha`$”.

*Auxiliary caption template:* “Confidence vs. structure size. Auxiliary (not $`\alpha`$); no conversion. Parameters and CI reported.”

**5.12 Failure modes & meanings**

- **Strategy leakage:** analytic search creeps in → curvature; tighten deadlines or re-bin.

- **Cue confounds:** larger $`L`$ inadvertently eases the task; match difficulty across levels.

- **Template overreach:** one template fits only part of the span; slopes segment—report **MULTI-REGIME**.

- **Shortcut masquerade:** true addressing makes $`T \sim \log L`$; label **LOG-SCALING** and stop.

**5.13 Lyric coda · The feel of a true leap**

The best leaps are not far; they are **exact**. You land where the ground was waiting. The answer feels immediate because the scale was right, and time—grateful—did not have to wander.

**5.14 What this chapter does not do**

- It does **not** infer $`\alpha`$ from confidence, $`\beta`$, or coherence.

- It does **not** accept single-point ratios as $`\alpha`$.

- It does **not** force power laws where representation shortcuts imply $`T \sim \log L`$.

**5.15 Take-home**

Intuition is **fast band-keeping**: choosing a structural scale $`L`$ and a duration $`T`$ that yield a coherent slope $`\alpha`$ on the first try. Measure the slope with rigor; let auxiliaries illuminate, not replace, the leap.

**Chapter 6 · Toward a Culture of Coherence**

**Epigraph.** *Method is the kindness we extend to truth.*

**6.1 Poetic prelude · The house we build for evidence**

Stories are quick; evidence is patient. If we want results that travel—across labs, years, and languages—we must give truth a good house: rooms with clear doors, windows that open, floors that hold. In this book that house is a simple covenant: **structure first, time second, slope from many points, and honesty when the line won’t hold**.

*(Bridge: the house rules below keep the lyric alive without letting it rewrite the data.)*

**6.2 The non-negotiables (technical statement)**

1.  **Slope-first:** $`\alpha`$ is estimated **only** as a multi-point log–log slope of $`\log T`$on $`\log L`$ within collapse-valid windows.

2.  **Windows matter:** bins are pre-registered (scale $`L`$, clock $`T`$, bounds, changepoints, exclusions). Mixed mechanisms ⇒ **MULTI-REGIME** or **NO_COLLAPSE**.

3.  **Auxiliaries ≠** $`\alpha`$**:** spectral $`\beta`$, PLV/coherence, echo/delay, information scores are **auxiliary**—never substitutes, never converted by universal formulas.

4.  **Topology honesty:** if effective geometry implies $`T \sim \log L`$, report **LOG-SCALING**; do not force a power law.

5.  **Errors-in-variables by default:** ODR/TLS or Theil–Sen; SIMEX optional where measurement error is estimable; always report 95% CIs and diagnostics.

**6.3 Lyric Box · On keeping promises to data**

Precision is a form of tenderness. We do not coerce a line out of a reluctant cloud; we listen for where it is straight and where it changes key. The poem can widen the room; the method decides whether the room exists.

**6.4 Design principles for projects and labs**

- **Pre-registration:** define $`L`$, $`T`$, bin bounds, inclusion/exclusion, and failure criteria before fitting slopes.

- **Span & granularity:** target ≥0.6–1.0 decades in $`L`$ with ≥6 distinct levels inside a single mechanism.

- **Constancy inside bins:** fix architecture/policy/state; log deviations.

- **Replication & error:** ≥3 replicates per level; model both-axis noise; apply SIMEX if error scales with $`L`$.

- **Diagnostics mandatory:** residual curvature checks, leave-one-out stability, heteroscedasticity bounds, changepoint tests.

- **Segmentation beats averaging:** if there’s a breakpoint, report segmented slopes with interpretation, not a blended line.

- **Caption truthfulness:** any figure claiming $`\alpha`$ shows the scatter, fit, 95% CI, estimator, $`n`$, span, and at least one collapse panel.

**6.5 Technical Box · Reporting standard (copy-paste checklist)**

- **Definition:** how $`L`$and $`T`$ were computed and why they match mechanism/geometry.

- **Estimation:** estimator (ODR/TLS/Theil–Sen), CI method, $`n`$, span in $`L`$.

- **Diagnostics:** residual plot, LOOCV slope stability, heteroscedasticity check, changepoint test.

- **Outcome label:** $`\widehat{\alpha}`$ (collapse-valid) **or** NO_COLLAPSE / LOG-SCALING / MULTI-REGIME.

- **Auxiliaries:** list β, PLV/coherence, delays, info scores with methods and CIs; **explicitly label as auxiliary**; no conversions to $`\alpha`$.

*One-line disclosure template:*\
“$`\alpha`$ estimated as EIV slope on $`\log T`$ vs $`\log L`$ (ODR/TLS), collapse-valid window, $`n = \ldots`$, span … decades; 95% CI …; auxiliaries (β/PLV/…) reported without conversion.”

**6.6 Reproducibility: how to make figures rebuildable**

- **Repository layout:**\
  /raw/ inputs → /proc/ paired $`(L,T)`$ with flags → /qc/ diagnostics → /features/ auxiliaries → /figures/ panels.

- **Environment lock:** versions, seeds, hardware; one-command rebuild script.

- **Manifest per figure:** YAML/JSON with inputs, parameters, hashes, and outcome label.

- **Negatives preserved:** NO_COLLAPSE, LOG-SCALING, MULTI-REGIME live in the repo and the paper (not the trash).

**6.7 Ethics of scaling claims**

- **No alchemy from proxies:** β→$`\alpha`$ (or any universal proxy mapping) is off-limits unless a **specific model** is stated and **validated out-of-sample**.

- **No single-point** $`\alpha`$**:** ratios like $`\log(T_{i}/T_{0})/log(L_{i}/L_{0})`$ are not slopes; they bypass uncertainty and diagnostics.

- **Transparent uncertainty:** report CIs and sensitivity; resist rounding regimes to attractive integers.

- **Failures teach:** a labeled negative is knowledge about mechanism boundaries, not a blemish to hide.

**6.8 Institutional practices (how editors, PIs, and reviewers can help)**

- **Two questions first:** (1) Where do the $`(L,T)`$ pairs come from? (2) Where is the collapse check?

- **Registered analyses:** require a pre-registered slope plan with failure criteria.

- **Credit for negatives:** accept and cite clean NO_COLLAPSE / LOG-SCALING / MULTI-REGIME reports.

- **Replication micro-grants:** small funds reserved for independent slope verification across labs.

**6.9 Technical Box · Tooling essentials**

- **Slope fitter:** ODR/TLS + Theil–Sen with bootstrapped CIs, SIMEX option.

- **Diagnostics kit:** residuals, runs tests, changepoint detection, heteroscedasticity checks.

- **Window manager:** enforces inclusion/exclusion, tracks outcomes and labels.

- **Figure maker:** standardized slope/auxiliary templates and captions.

- **Placebos:** clock rescaling/jitter and structure shuffles to confirm slope origin.

**6.10 Patterns of misuse (and exact fixes)**

- **Proxy inflation:** “β≈1 ⇒ $`\alpha`$≈1.5.”\
  **Fix:** relabel β as auxiliary; estimate $`\alpha`$from $`(T,L)`$ or drop the claim.

- **Single-point α:** identity-based “slopes.”\
  **Fix:** collect ≥6 levels of $`L`$, fit EIV, publish CI and diagnostics.

- **Pooling across regimes:** blended lines.\
  **Fix:** segmented fits with breakpoints and mechanism notes.

- **Topology blindness:** forcing power when $`T \sim \log L`$.\
  **Fix:** report LOG-SCALING and discuss geometry.

**6.11 Lyric Box · On publishing negatives**

A failed line is a map: it shows where the river forks, where the ground lifts, where another trail begins. We owe the next traveler a signpost, not a polished myth.

**6.12 Limitations & open problems**

- **Choosing the right** $`L`$**:** metric selection is the hardest step; sensor-space surrogates can bias slopes. Prefer source-level or physically grounded metrics.

- **Local laws:** $`\alpha`$ lives in finite windows; global universals are unlikely. Learn to narrate **where** the law holds.

- **Causality:** $`\alpha`$ is descriptive of structure–time coupling; interventions and placebos help, but claims should stay modest.

- **Cost:** EIV + diagnostics add overhead; standardized tooling and manifests reduce friction.

**6.13 A compact code of practice (post this on your lab wall)**

1.  Define $`L`$ and $`T`$ by mechanism.

2.  Pre-register bins and failure criteria.

3.  Estimate $`\alpha`$with EIV on $`(\log L,\log T)`$.

4.  Require collapse; **publish negatives**.

5.  Label auxiliaries; never convert them into $`\alpha`$.

6.  Prefer segmentation to averaging; report breakpoints.

7.  State topology when non-power (e.g., $`T \sim \log L`$).

8.  Make every figure rebuildable.

**6.14 Closing coda · Culture as a measurable promise**

A culture of coherence is not anti-poetry; it is poetry with scaffolding. We keep the song—and we show the score. When future readers open this book, may they hear both: the rhythm of a world that keeps time with its forms, and the quiet proof that the rhythm was really there.

**Appendix A · Concepts & Definitions (Technical Only)**

**A.1 Notation and Symbols**

- $`L`$: structural scale (length, wavelength, path length, correlation length, or validated surrogate with dimensional consistency).

- $`T`$: proper duration of the process governed by the structure that defines $`L`$.

- $`\alpha`$: **coherence exponent**, the log–log slope of $`T`$ on $`L`$ within a collapse-valid window.

- $`(L_{i},T_{i})`$: paired observations; $`i = 1,\ldots,n`$.

- $`\widehat{\alpha}`$: estimated slope; $`{CI}_{95\%}`$: 95% confidence interval.

- “Window/bin”: finite range of $`L`$over which mechanism and effective metric remain constant.

- “Collapse”: set of diagnostics that a log–log line is appropriate in a window (Sec. A.5).

- Auxiliary observables: spectral slope $`\beta`$, PLV/coherence, band powers, burst/ISI statistics, echo/delay measures.

- Failure labels: **NO_COLLAPSE**, **LOG-SCALING**, **MULTI-REGIME**.

**A.2 Structural and Temporal Scales**

**A.2.1 Defining** $`L`$**.**\
Choose a mechanism-tied, geometry-preserving scale:

- Physical length (e.g., vessel radius/path length, dendritic cable length, limb length).

- Source-modeled wavelength ($`L = \lambda/2`$); correlation length from spatial power spectra/variograms.

- Network geodesic length (effective distance along the operative connectivity).

- Validated surrogate (sensor-space only if source/geometry is unavailable, flagged as surrogate).

**A.2.2 Defining** $`T`$**.**\
Duration of the mechanism sustained by $`L`$: circulation time, diffusion/reaction half-time, oscillation period $`T = 1/f`$, integration/access latency (motor-corrected), settling time, consolidation delay, retention half-life.

**A.2.3 Inclusion rules.**\
State how $`L`$ and $`T`$ are computed; keep definitions fixed within each window. Exclude artifacts, pathology, and mixed tasks/mechanisms per pre-registered criteria.

**A.3 Windows and Regimes**

- **Window selection.** Pre-register bounds in $`L`$ and conditions (state/task/architecture) to keep the operative mechanism constant.

- **Span and granularity.** Target ≥0.6–1.0 decades in $`L`$ with ≥6 distinct $`L`$ levels inside a window.

- **Changepoints.** If slopes differ across subranges, report **MULTI-REGIME** with segmented fits; do not average across regimes.

**A.4 Estimating** $`\mathbf{\alpha}`$**(Errors-in-Variables)**

We estimate $`\alpha`$**only** from multi-point regressions of $`\log T`$ on $`\log L`$.

**A.4.1 Model.**

``` math
\log T_{i} = a + \alpha\text{ }\log L_{i} + \varepsilon_{i},\text{with noise on both axes.}
```

**A.4.2 Estimators.**

- **ODR/TLS** (default): accounts for measurement error in both $`L`$ and $`T`$.

- **Theil–Sen** (robust): median-of-slopes; use as a sensitivity check.

- **SIMEX** (optional): error correction when error variances are estimable.

**A.4.3 Uncertainty.**\
Report $`\widehat{\alpha}`$ with $`{CI}_{95\%}`$ (bootstrap or asymptotic), plus $`n`$, span in $`L`$, and estimator details.

**A.5 Collapse and Regularity Diagnostics**

A window is **collapse-valid** if all hold:

1.  **Log–log linearity:** residuals show no curvature (runs test or LOESS residual inspection).

2.  **Stability:** leave-one-out (or leave-one-level-out) slope remains within a pre-registered tolerance.

3.  **Heteroscedasticity bounds:** variance patterns within limits (e.g., Breusch–Pagan or variance-stratified checks).

4.  **No hidden changepoints:** segmented fits do not significantly outperform a single slope unless labeled **MULTI-REGIME**.

5.  **Sufficient span/granularity:** meets Sec. A.3 targets (else mark exploratory).

**Failure handling.**

- Curvature or instability → **NO_COLLAPSE**.

- Semi-log linearity → **LOG-SCALING** (report $`T \sim \log L`$, do not infer $`\alpha`$).

- Clear breakpoint(s) → **MULTI-REGIME** (report segmented $`\alpha`$ with breakpoints).

**A.6 Mechanistic Expectations (Non-binding Guides)**

- **Diffusion–reaction:** $`T \sim L^{2}/D \Rightarrow \alpha \approx 2`$ when $`D`$ is roughly constant in the window.

- **Advection/ballistic:** near-constant velocity/process rates → $`\alpha \approx 1`$.

- **Shortcut/topological compression:** small-world-like effective distances → $`T \sim \log L`$ (non-power).\
  These expectations guide **design**, not inference; $`\alpha`$ must be **measured** per A.4–A.5.

**A.7 Scaling Relations & Operational Observables**

**Definition (operational).**\
In RTM, the exponent $`\alpha`$is defined **exclusively** from time–size scaling within a regime:

``` math
T \propto L^{\alpha} \Longleftrightarrow \alpha = \frac{d\log T}{d\log L}.
```

Estimated values of $`\alpha`$ come **only** from log–log slopes of $`\log T`$ vs. $`\log L`$ computed on a **finite window that passes collapse and regularity checks** (pre-registered bins, changepoints, exclusions; see A.5).

**Auxiliary observables (proxies, not** $`\alpha`$**).**\
Domain signals such as spectral slopes $`\beta`$(from $`P(f) \propto f^{- \beta}`$), cross-frequency coupling indices, burst/ISI statistics, echo/delay times, and synchrony/coherence measures can be informative **correlates** under **specific models**. They are **not** $`\alpha`$ and there is **no universal conversion** (e.g., $`\alpha \neq 1 + \beta/2`$in general). When reported, they are labeled **auxiliary** and never used as substitutes for $`\alpha`$.

**Reporting standard.** For each regime report:

1.  the $`(\log L,\log T)`$ scatter, fitted slope $`\widehat{\alpha}`$, $`{CI}_{95\%}`$, and estimator (**ODR/TLS**, **Theil–Sen**, optional **SIMEX**);

2.  collapse diagnostics and window specification (bins, changepoints, exclusions, failure modes);

3.  any **correlative** analyses relating $`\widehat{\alpha}`$ to auxiliary observables (e.g., $`\beta`$), explicitly marked **model-dependent** and, where applicable, evaluated **out-of-sample**.

**Guardrails.**

- Do **not** substitute $`\beta`$, echo/delay measures, PLV/coherence, or other markers for $`\alpha`$.

- Do **not** pool mixed regimes or report a single $`\alpha`$ across windows that fail collapse tests.

- If the **effective metric is non-Euclidean/topological** (e.g., small-world geodesics), state the metric explicitly and **report** $`T \sim \log L`$ rather than forcing a power-law fit.

**Model-conditional bridges (optional).**\
If a specific theory implies a mapping $`\alpha = g(\beta;\text{ model, band, estimator})`$, **pre-register** $`g`$and explicit failure criteria, estimate $`\widehat{\alpha}`$ **directly** from $`T`$–$`L`$, and evaluate **out-of-sample** agreement between $`g(\beta)`$and $`\widehat{\alpha}`$. Failure ⇒ revise or drop $`g`$.

**What appears in the results table.**\
Only $`\widehat{\alpha}`$ values obtained from **log–log** $`\log T`$–$`\log L`$ fits that pass collapse are tabulated. Auxiliary observables appear in a separate column (or appendix) and are **never** used as surrogates for $`\alpha`$.

**A.8 Reporting Standard (Copy-Paste Checklist)**

For every $`\alpha`$-claim include:

- **Definition:** how $`L`$ and $`T`$ were computed and why they match the mechanism.

- **Estimation:** estimator (ODR/TLS/Theil–Sen), $`n`$, span in $`L`$, $`\widehat{\alpha}`$with $`{CI}_{95\%}`$.

- **Diagnostics:** show (or link) residual/collapse panel; state pass/fail.

- **Outcome label:** $`\widehat{\alpha}`$ (collapse-valid) **or** NO_COLLAPSE / LOG-SCALING / MULTI-REGIME.

- **Auxiliaries:** listed with methods and CIs, explicitly labeled **auxiliary**; **no conversion** to $`\alpha`$.

*Caption template (slope):*\
“Scaling of $`T`$ with $`L`$ in \[system\]. EIV slope $`\widehat{\alpha}`$ (95% CI) on a collapse-valid window (n = …; span … decades). Residuals show no curvature. Estimator: \[ODR/TLS \| Theil–Sen \| SIMEX\].”

*Caption template (auxiliary):*\
“Spectral slope $`\beta`$/PLV/coherence in \[system\]. **Auxiliary** (not $`\alpha`$); no conversion. Method and 95% CI reported.”

**A.9 Reproducibility and Controls**

- **Repository layout:** /raw/ (inputs) → /proc/ (paired $`(L,T)`$with flags) → /qc/ (diagnostics) → /features/ (auxiliaries) → /figures/ (panels).

- **Environment lock:** versions, seeds, hardware; one-command rebuild script.

- **Placebos:**

  - *Clock placebo:* rescale timestamps or add jitter—$`\widehat{\alpha}`$ should remain within CI if structure drives the slope.

  - *Structure placebo:* shuffle spatial structure (hold clock) — collapse should fail or $`\widehat{\alpha}`$ drift if structure is causal.

- **Negatives preserved:** publish and archive **NO_COLLAPSE**, **LOG-SCALING**, **MULTI-REGIME** outcomes.

**Appendix A Summary.**\
$`\alpha`$ is a **measured slope** of $`\log T`$ on $`\log L`$ within collapse-valid windows. Auxiliary signals inform but never replace it. Non-power behaviors are labeled as such. Reporting is standardized, diagnostics are mandatory, and negatives are first-class results.

**Appendix B · Protocols (Technical Only)**

**Policy.** In all protocols below, $`\alpha`$ is estimated **only** as a multi-point log–log slope of $`\log T`$ on $`\log L`$ within a collapse-valid window. Spectral slopes ($`\beta`$), PLV/coherence, echo/delay measures, and related indices are **auxiliary**—they are reported with uncertainty but never converted to $`\alpha`$ by universal formulas.

**B.1 Heart Rate Variability (HRV) — Auxiliary Observables (Not** $`\mathbf{\alpha}`$**)**

**Purpose.** Provide a standardized pipeline to report HRV descriptors (including spectral slope $`\beta`$) as **auxiliary** correlates. No $`\alpha`$ is estimated from HRV.

**B.1.1 Data & Recording**

- **Acquisition:** ECG at ≥ 500 Hz (preferred 1 kHz).

- **Duration:** ≥ 10 min per condition (rest/task), avoiding large state drifts.

- **Leads:** Standard chest configuration; record respiration if available.

- **Metadata:** posture, medication, time of day, recent exercise/caffeine.

**B.1.2 Pre-processing**

1.  **R-peak detection:** validated detector; manual review of ambiguous segments.

2.  **Artifact handling:** remove ectopic beats/false detections; interpolate RR gaps (cubic spline or adaptive).

3.  **Stationarity bins:** split into fixed windows $`W`$ (e.g., 120 s, optional 50% overlap). Exclude windows with \> 5% artifacts or visible nonstationarity.

**B.1.3 Features (Auxiliaries)**

- **Time-domain:** mean RR, SDNN, RMSSD, pNN50.

- **Frequency-domain:** Welch or AR spectrum on evenly resampled tachogram (≤ 4 Hz resampling).

  - **Bands:** VLF/LF/HF per pre-registered cutoffs; report band power and **spectral slope** $`\beta`$ on a specified band using robust linear fit to $`\log P(f)`$ vs. $`\log f`$.

- **Respiratory coupling (if available):** RR–respiration coherence/PLV (methods, parameters).

**B.1.4 Reporting**

- **Primary:** $`\beta`$(slope) with 95% CI, method (Welch/AR, window size, overlap), band definition, detrending settings.

- **Secondary:** band powers, LF/HF, coherence/PLV with CIs.

- **Labeling:** Every panel caption must state **“Auxiliary (not** $`\alpha`$**); no conversion to** $`\alpha`$**.”**

**B.1.5 Reproducibility**

- **Files:** /raw/ECG, /proc/RR, /features/HRV_aux.csv (per-window rows), /figures/HRV\_\*.

- **Manifest:** JSON/YAML with detector, interpolation policy, PSD params, band edges.

**B.2 Electroencephalography (EEG) — Estimating** $`\mathbf{\alpha}`$

**Purpose.** Estimate $`\alpha`$ from paired time–size measurements using source-level spatial scale $`L`$ and temporal scale $`T`$. Spectral $`\beta`$, PLV/coherence, etc., are reported as **auxiliary**.

**B.2.1 Data & Recording**

- **Acquisition:** 64+ channels (≥ 500 Hz), synchronized triggers; record EOG/EMG; optional respiration.

- **Montage:** 10–10 or higher density; individual head model (MRI) preferred; template acceptable with caveats.

- **Duration:** ≥ 10 min per condition to support stable windowing.

- **Metadata:** task blocks, arousal markers, medication, age, handedness.

**B.2.2 Pre-processing**

1.  **Filtering:** 0.5–80 Hz (or per study); notch at line frequency.

2.  **Artifact handling:** bad-channel interpolation; ICA/SSP to remove blink/cardiac/muscle; log retained components.

3.  **Segmentation:** windows $`W`$of 10–30 s (50% overlap optional). Exclude windows by pre-registered artifact thresholds.

4.  **Time–frequency for peaks:** multitaper or Morlet to identify band peaks (theta 4–7, alpha 8–12, beta 13–30, gamma 30–45 Hz).

**B.2.3 Defining** $`\mathbf{(T,L)}`$

- **Temporal scale $T$:** for each window and band peak $f\*$  , set $T=1/f\*$  (seconds). If multiple peaks per band, pre-register a centroid rule or dominance criterion.

- **Spatial scale** $`L`$**:** source modeling (eLORETA/beamformer) to estimate spatial wavelength $`\lambda`$ (or correlation length) of band-limited sources; set $`L = \lambda/2`$.

  - **Estimators for** $`\lambda`$**:**\
    (i) FWHM of cortical activation blobs;\
    (ii) peak of the cortical 2D spatial power spectrum (inverse spatial frequency);\
    (iii) variogram range. Pre-register one and keep it fixed.

> If source modeling is unavailable, a sensor-space surrogate may be used **only as exploratory** (e.g., topography SVD period). Flag clearly as surrogate.

**B.2.4 Windowing & Collapse Criteria**

Build a dataset of paired points $`\{(\log L_{i},\log T_{i})\}`$across ROIs/windows with constant mechanism/state.

**Collapse-valid window requires:**

1.  single-regime log–log linearity (no residual curvature),

2.  leave-one-ROI/window-out slope stability,

3.  heteroscedasticity within pre-set bounds,

4.  no hidden changepoints,

5.  span ≥ 0.6 decades in $`L`$ with ≥ 6 distinct $`L`$ levels (target).

Failures are labeled **NO_COLLAPSE**. If semi-log is linear, label **LOG-SCALING** (do not infer $`\alpha`$).

**B.2.5 Estimating** $`\mathbf{\alpha}`$

- **Estimator:** errors-in-variables regression of $`\log T`$ on $`\log L`$: **ODR/TLS** (default) or **Theil–Sen** (robust).

- **Error correction:** **SIMEX** optional when measurement error is characterizable.

- **Uncertainty:** 95% CI (bootstrap or asymptotic). Report $`n`$, span, estimator.

``` math
\widehat{\alpha} = slope\text{ }(\log T\text{ on }\log L),T \propto L^{\alpha}.
```

**B.2.6 Controls & Confounds**

- **State dependence:** stratify by task vs. rest and eyes-open/closed; do not pool mixed states.

- **Volume conduction:** prefer source-level metrics; if sensor-space used, apply leakage controls (e.g., orthogonalization).

- **Physiology covariates:** respiration/HRV included as **auxiliary** covariates; never converted to $`\alpha`$.

- **Topology caveat:** for small-world-like effective geometry, report $`T \sim \log L`$ instead of forcing power.

**B.2.7 Outputs**

- **Alpha table (per subject/condition):** $`\widehat{\alpha}`$, CI, estimator, $`n`$, span, collapse flags, bin specification.

- **QC table:** artifact rates, rejected windows, source-model parameters.

- **Figures:**\
  (i) $`\log T`$ vs. $`\log L`$ scatter with fit + 95% CI;\
  (ii) residual plot;\
  (iii) leave-one-out sensitivity panel.

- **Auxiliaries (separate):** spectral $`\beta`$, PLV/coherence; labeled **not** $`\alpha`$.

**B.2.8 Repository Checklist**

- /proc/EEG/ pre-processed data + window indices.

- /src/ head models, lead fields, inverse parameters.

- /features/EEG_pairs.csv with $`(\log L,\log T)`$ and metadata.

- /qc/EEG_collapse_reports/ diagnostics per bin.

- /figures/EEG_scaling\_\* standardized panels.

- README.md with exact parameters, versions, seeds.

**B.2.9 Failure Modes**

- **NO_COLLAPSE:** mechanism mixing or wrong metric; re-bin or revise $`L`$ estimator.

- **LOG-SCALING:** shortcut topology; report semi-log fit, no $`\alpha`$.

- **MULTI-REGIME:** segmented slopes with breakpoints; report both segments and interpretation.

**(Optional) B.3 Template — Any Modality with Space–Time Pairs**

Use this template for other domains (MEG, fNIRS, robotics/control, behavioral cycles) that can produce valid $`(L,T)`$ pairs.

**B.3.1 Define** $`L`$**and** $`T`$**.** Mechanism-tied structural scale and proper duration; pre-register both and keep them fixed within windows.

**B.3.2 Acquire & preprocess.** Domain-appropriate acquisition; artifact controls; fixed windows $`W`$ with exclusion thresholds.

**B.3.3 Build pairs.** Derive $`(\log L,\log T)`$ across ≥ 6 levels of $`L`$ with target span ≥ 0.6 decades.

**B.3.4 Estimate** $`\alpha`$**.** EIV regression (ODR/TLS; Theil–Sen), 95% CI, SIMEX optional.

**B.3.5 Collapse diagnostics.** Residual curvature, LOOCV slope stability, heteroscedasticity bounds, changepoint tests.

**B.3.6 Label outcomes.** $`\widehat{\alpha}`$ (collapse-valid) or **NO_COLLAPSE / LOG-SCALING / MULTI-REGIME**.

**B.3.7 Report auxiliaries.** Domain-specific metrics (e.g., spectral $`\beta`$, PLV, information scores) with CIs; explicitly **not** $`\alpha`$.

**B.3.8 Reproducibility.** Folder structure, manifests, one-command rebuild.

**Appendix B Summary.** HRV is reported as **auxiliary** only; EEG provides a canonical, slope-first pipeline to estimate $`\alpha`$. All other modalities follow the same blueprint: define $`L`$ and $`T`$, obtain multi-point pairs, verify collapse, fit EIV slopes with uncertainty, label failures explicitly, and keep auxiliaries in their own lane.

**Appendix C · Extended Glossary (Technical Only)**

**Scope.** Definitions used across the book. Unless stated otherwise, $`\log`$is natural log and “slope” means the coefficient from a regression of $`\log T`$ on $`\log L`$. All entries avoid metaphor and keep to operational meaning.

$`\mathbf{\alpha}`$ **(coherence exponent):**
The **multi-point** log–log slope of $`\log T`$ on $`\log L`$ estimated inside a **collapse-valid window**. Report as $`\widehat{\alpha}`$ with 95% CI, estimator, $`n`$, span.

**Auxiliary observable:**
Any domain metric that is **not** $`\alpha`$(e.g., spectral slope $`\beta`$, PLV/coherence, band powers, echo/delay). May correlate with $`\alpha`$ in **model-specific** ways; never a universal substitute.

**Band /:** $`\mathbf{\alpha}`$**-band**
A range of scales where estimated $`\alpha`$ is approximately constant and passes collapse. Boundaries are determined by diagnostics or visible breakpoints.

**Bootstrap CI:**
Confidence interval obtained by resampling (with replacement) the paired $`(L,T)`$points or residuals under the chosen estimator.

**Changepoint (breakpoint):**
A scale value where a single-slope model is significantly outperformed by segmented slopes. Label the outcome **MULTI-REGIME** and report each segment.

**Collapse (diagnostics):**
Set of tests used to accept a window: log–log linearity (no residual curvature), leave-one-out stability, bounded heteroscedasticity, no hidden changepoints, sufficient span/granularity.

**Correlation length:**
A spatial scale (e.g., variogram range or inverse peak of spatial frequency) summarizing the extent of correlation; may be used to define $`L`$.

**Errors-in-Variables (EIV):**
Regression framework that accounts for measurement error in both $`\log L`$ and $`\log T`$. Required when both axes are noisy.

**Estimator disclosure:**
The explicit statement accompanying $`\widehat{\alpha}`$: estimator (ODR/TLS, Theil–Sen, SIMEX use), CI method, $`n`$, span (decades), window definition, and diagnostics.

**Effective topology / metric:**
The geometry that governs propagation (e.g., network geodesics, small-world shortcuts), which may differ from Euclidean distance and change the expected scaling (e.g., $`T \sim \log L`$).

**FWHM (full width at half maximum):**
Spatial dispersion measure used on source-level maps; can define wavelength $`\lambda`$ and hence $`L = \lambda/2`$.

**Granularity (levels):**
The number of distinct $`L`$ levels in a window. Target ≥6 levels to stabilize slope estimates and diagnostics.

**Heteroscedasticity:**
Scale-dependent variance of residuals; must remain within pre-registered bounds for collapse to pass.

**LOG-SCALING:**
Outcome label when semi-log is linear and $`T`$ grows approximately as $`\log L`$. Do **not** report $`\alpha`$ for this regime.

$`\mathbf{\beta}`$ **(spectral slope):**
Exponent from a power spectrum $`P(f) \propto f^{- \beta}`$. **Auxiliary** by default. There is **no universal** conversion from $`\beta`$ to $`\alpha`$.

**MINIMUM SPAN:**
Target span of $`L`$ within a window (≥0.6–1.0 decades) to ensure identifiability of the slope and reliable diagnostics.

**MULTI-REGIME:**
Outcome label when segmented slopes are supported within a window. Report piecewise $`\widehat{\alpha}`$ values, breakpoints, and interpretation.

**NO_COLLAPSE:**
Outcome label when the collapse criteria fail (curvature, instability, excessive heteroscedasticity, insufficient span, or hidden changepoints). Do not report $`\alpha`$.

**ODR/TLS (orthogonal distance regression / total least squares):**
Default EIV estimator that minimizes distances orthogonal to the regression line, accounting for error on both axes.

**Placebo (clock):**
Control that rescales timestamps or adds small jitter while keeping structure; $`\widehat{\alpha}`$ should remain within CI if the slope reflects structure, not clock artifacts.

**Placebo (structure):**
Control that perturbs spatial patterns or pairings while keeping the clock; collapse should fail or $`\widehat{\alpha}`$ drift if structure drives the slope.

**PLV / coherence:**
Phase-locking value and coherence measures. **Auxiliary** descriptors of synchrony; not substitutes for $`\alpha`$.

**Pre-registration (window):**
Prior declaration of $`L`$ and $`T`$ definitions, bin bounds, inclusion/exclusion, estimator, diagnostics, and failure criteria before fitting $`\alpha`$.

**Residual curvature (runs/LOESS test):**
Deviation from linearity on log–log; evidence against a single-slope model inside the window.

**SIMEX:**
Simulation–extrapolation procedure to correct bias from measurement error when its variance is estimable. Used on top of ODR/TLS or other EIV fits.

**Source-modeled wavelength** $`\mathbf{(\lambda)}`$ :
Spatial period of an oscillatory source estimated on the cortex (e.g., via eLORETA/beamformer); $`L = \lambda/2`$.

**Span (decades):**
The width of $`L`$ values in a window measured in orders of magnitude. Reported alongside $`\widehat{\alpha}`$.

**Surrogate (sensor-space):**
A non-source structural proxy for $`L`$(e.g., topographic SVD period). Allowed only with explicit caveats and usually labeled exploratory.

**Theil–Sen:**
Robust slope estimator (median of pairwise slopes); used as a sensitivity check against ODR/TLS or when outliers are a concern.

**Time** $`\mathbf{T}`$ **(proper duration):**
Duration of the mechanism governed by the structure defining $`L`$ (e.g., oscillation period $`1/f`$, transit time, settling time, consolidation delay).

**Variogram (range):**
Spatial statistic used to estimate correlation length. The range (where the variogram plateaus) can define $`L`$.

**Window (bin):**
Finite range of $`L`$ (and fixed mechanism/state) over which a single slope model is tested. Must pass collapse diagnostics to report $`\alpha`$.

**Usage rule (global).**\
Only values of $`\widehat{\alpha}`$ derived from **multi-point** log–log slopes within **collapse-valid** windows appear in results tables. All other metrics are **auxiliary** and reported separately with their own methods and uncertainty.
  
**Appendix D · Negative Results & Failure Modes (Technical Only)**

**Scope.** How to detect, label, and report when a power-law regime **does not** hold, and how to learn from that outcome without forcing $`\alpha`$.

**D.1 Outcome Labels (single-source-of-truth)**

- **NO_COLLAPSE** — The log–log fit fails collapse diagnostics in the pre-registered window (curvature, instability, heteroscedasticity, or insufficient span).

- **LOG-SCALING** — Semi-log is linear: $`T \sim a + b\log L`$. Report $`b`$and CI; do **not** infer $`\alpha`$.

- **MULTI-REGIME** — Piecewise linear on log–log with ≥1 significant breakpoint(s). Report segmented $`\widehat{\alpha}`$, breakpoints, CIs.

- **AUXILIARY-ONLY** — No valid $`(L,T)`$pairs or collapse; report auxiliary metrics (e.g., $`\beta`$, PLV) **explicitly as auxiliary**, with methods and CIs.

**Rule:** These labels are **final results**, not placeholders. They appear in tables, figures, and the repository manifest.

**D.2 Collapse Diagnostics (recap, strict)**

A window is **accepted** only if **all** pass:

1.  **Linearity (log–log):** residual runs/LOESS show no curvature.

2.  **Stability:** leave-one-level-out slope drift ≤ pre-registered tolerance.

3.  **Heteroscedasticity:** within bounds (e.g., Breusch–Pagan *p* \> threshold or variance-stratified check).

4.  **Changepoints:** no hidden breakpoints outperform a single slope unless labeling **MULTI-REGIME**.

5.  **Span & granularity:** target ≥0.6–1.0 decades in $`L`$ with ≥6 distinct levels.

Fail any → **NO_COLLAPSE** (unless semi-log linear ⇒ **LOG-SCALING**).

**D.3 Decision Tree (operational)**

1.  Fit EIV on $`(\log L,\log T)`$(ODR/TLS; Theil–Sen sensitivity).

2.  Run diagnostics (D.2).

    - **Pass all** → report $`\widehat{\alpha}`$ with CI.

    - **Curvature / instability / hetero / too few levels** → **NO_COLLAPSE**.

    - **Best model is semi-log** → **LOG-SCALING** (report $`b`$).

    - **Clear breakpoint(s)** → **MULTI-REGIME** (report segments).

3.  If **no valid** $`(L,T)`$→ **AUXILIARY-ONLY** (no $`\alpha`$ claim).

**D.4 Reporting Templates**

**D.4.1 NO_COLLAPSE (figure caption template)**

*“Time–size pairs for \[system\] within the pre-registered window. Log–log fit fails collapse (residual curvature / unstable slope / heteroscedasticity). Outcome: **NO_COLLAPSE**. We report auxiliary descriptors separately (not* $`\alpha`$*).”*

**Table row fields:** system \| window spec \| $`n`$\| span (decades) \| estimator \| diagnostics (fail reason) \| outcome = NO_COLLAPSE.

**D.4.2 LOG-SCALING (figure caption template)**

*“Timing follows a logarithmic law in \[system\]:* $`T \sim a + b\ \log L`$*(semi-log linear, 95% CI for* $`b`$*). Power-law fit rejected by collapse. Outcome: **LOG-SCALING** (non-power).”*

**Table row fields:** system \| window \| $`n`$\| semi-log slope $`b`$ (CI) \| test vs. power-law \| outcome = LOG-SCALING.

**D.4.3 MULTI-REGIME (figure caption template)**

"Timing follows a logarithmic law in [system]: $T \sim a + b \log L$ (semi-log linear, 95% CI for $b$). Power-law fit rejected by collapse. Outcome: **LOG-SCALING** (non-power)."

**Table row fields:** system | window | $n$ | semi-log slope $b$ (CI) | test vs. power-law | outcome = LOG-SCALING.

**D.4.4 AUXILIARY-ONLY (figure caption template)**

*“No valid time–size pairs for \[system\]. We report auxiliary metrics (e.g., spectral* $`\beta`$*, PLV/coherence) with methods and CIs. Outcome: **AUXILIARY-ONLY** (no* $`\alpha`$ *claimed).”*

**Table row fields:** system \| reason (no pairs / missing $`L`$/ missing $`T`$) \| auxiliaries reported (with methods) \| outcome = AUXILIARY-ONLY.

**D.5 Troubleshooting Guide (from symptom to action)**

| **Symptom** | **Likely cause** | **Action** |
|----|----|----|
| Residual curvature (upward) | mechanism mixing; missing changepoint | tighten window; test segmented slopes; if significant → MULTI-REGIME; else NO_COLLAPSE |
| Residual curvature (downward) | wrong metric for $`L`$; saturation | redefine $`L`$ (source-level, geodesic); re-bin |
| Large LOOCV slope drift | too few levels; outlier leverage | increase levels; robust estimator; re-measure |
| Heteroscedasticity explodes | measurement error grows with $`L`$ | model error; SIMEX; narrow span; if persists → NO_COLLAPSE |
| Good semi-log fit, bad log–log | shortcut topology | label LOG-SCALING; do not force $`\alpha`$ |
| Clean segments, worse single fit | regime transition | MULTI-REGIME with breakpoint CI & interpretation |
| No usable $`T`$ or $`L`$ | acquisition gap | re-collect or downgrade to AUXILIARY-ONLY |

**D.6 Placebos & Controls (make negatives meaningful)**

- **Clock placebo:** rescale timestamps or add small jitter; if $`\widehat{\alpha}`$ or $`b`$ shifts beyond CI, timing artifact suspected.

- **Structure placebo:** shuffle spatial structure / ROI labels while keeping the clock; slope should vanish or drift if structure drives timing.

- **Motor / report correction (behavior):** subtract constant delays to avoid intercept artifacts masquerading as slope.

- **State stratification (neuro/phys):** re-bin by arousal/task; pooling states often causes NO_COLLAPSE.

**D.7 Repository & Manifest (auditable negatives)**

- **Folders:**\
  /proc/pairs/ $`(L,T)`$with flags; /qc/ collapse reports; /figures/negatives/ panels; /features/aux/ auxiliary metrics.

- **Manifest fields (per analysis):** system, window_spec, levels, span_decades, estimator, diagnostics, outcome_label, ci, breakpoints, notes.

- **Rebuild:** one-command script reproduces **both** positives and negatives.

**D.8 Ethics & Communication**

- **Equivalence of value:** A clean **NO_COLLAPSE** is as informative as a positive slope—it maps the boundary of mechanism.

- **No euphemisms:** Do not rename **LOG-SCALING** as “weak scaling.” State it plainly and model it appropriately.

- **No proxy alchemy:** A negative $`\alpha`$ result is **not** a license to convert auxiliaries (e.g., $`\beta`$) into $`\alpha`$. Keep lanes clear.

**D.9 Worked Mini-Examples (abstracted)**

1.  **Vascular transit (curvature):** Log–log curvature detected; segmented slopes pass per-chunk. **Outcome:** MULTI-REGIME with $`L^{*}`$ CI; mechanism note: capillary → venous transition.

2.  **EEG access (shortcut):** Semi-log linear; $`b = 0.34`$(CI). Power fit fails. **Outcome:** LOG-SCALING. Note: hub-dominated routing.

3.  **Behavioral recall (unstable slope):** LOOCV drift exceeds tolerance; span \< 0.4 decades. **Outcome:** NO_COLLAPSE; plan: increase levels, fix rehearsal policy.

4.  **HRV-only dataset:** No $`T,L`$ pairs; auxiliary PSD $`\beta`$ and PLV reported with CIs. **Outcome:** AUXILIARY-ONLY.

**D.10 FAQ (failure-mode quick answers)**

- **Q:** Can I average two regimes to report one $`\alpha`$?\
  **A:** No. Use **MULTI-REGIME** with breakpoints.

- **Q:** Semi-log works; can I still quote $`\alpha`$?\
  **A:** No. Label **LOG-SCALING** and report $`b`$.

- **Q:** Collapse fails but auxiliaries look great—can I infer $`\alpha`$from $`\beta`$?\
  **A:** No. Auxiliaries remain auxiliaries; no universal conversion.

- **Q:** My span is small (0.3 decades) but fit looks straight—OK to report?\
  **A:** Exploratory only; expand span or mark **NO_COLLAPSE** (insufficient granularity).

**Appendix D Summary.**\
Negative outcomes are **evidence**: they locate boundaries, reveal wrong metrics, expose regime mixing, and point to alternate grammars (e.g., $`T \sim \log L`$). Treat **NO_COLLAPSE**, **LOG-SCALING**, **MULTI-REGIME**, and **AUXILIARY-ONLY** as publishable first-class results with the same care—diagnostics, captions, manifests—as positive $`\alpha`$-findings.

*© 2026 Álvaro José Quiceno Rendón. This document is distributed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.*

