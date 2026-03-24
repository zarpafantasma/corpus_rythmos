<div align="center">

<img src="https://raw.githubusercontent.com/zarpafantasma/corpus_rythmos/main/media/serpent1.png" width="200" alt="Diagrama de Snake">

# **RTM-Aware Quantum Computing**  
**A Multiscale, Slope-First Framework for Coherence, Scheduling, and Design**  
  
Álvaro Quiceno

</div>

**Abstract**

We introduce a **slope-first** methodology for quantum computing based on **Multiscale Temporal Relativity (RTM)**. Inside a fixed operational regime, RTM posits that a characteristic time $T$ scales with a size/scale proxy $L$ by a power law,

$$\log T\text{\:\,} = \text{\:\,}\alpha\text{ }\log L\text{\:\,} + \text{\:\,}c,
$$

where the **coherence exponent** $\alpha$ is the **clock-invariant** structural signal and $c$ encodes clock/units. We adapt RTM to quantum stacks---**physical**, **QEC**, **compiler/runtime**, and **I/O--cryo**---by defining layer-specific $(L,T)$ pairs (e.g., number of active qubits vs. stable calibration time; code distance vs. logical-failure time; multiplexing degree vs. readout latency; circuit width vs. makespan), and estimating binwise slopes under errors-in-variables (ODR/TLS, Theil--Sen, SIMEX). A **collapse test** validates scaling and guards against regime mixing; clean family-wise slopes are fused into a real-time $\mathbf{ECI}_{QC}$**(t)** with uncertainty and QA gates.

We formulate **falsifiable** hypotheses: **(H1)** higher pre-shock $\alpha$ predicts longer stability margins (fewer forced recalibrations, lower logical error at fixed $d$); **(H2)** **decoherence events**---significant QA-clean drops in ${ECI}_{QC}$---lead spikes in logical error, queueing, or makespan; **(H3)** micro→meso→macro **tempo cascades** exhibit non-decreasing $\alpha$ within stable regimes. We demonstrate how **RTM-aware scheduling** (batching, staggered resets, low-variance routing), **QEC cadence design** (desynchronization of syndrome cycles), and **modular sizing** (sweet spots for interconnect) can improve throughput and reliability without changing physical fidelities. The framework is reproducible, gauge-robust (unit/clock changes do not affect $\alpha$), and designed to fail gracefully (no-collapse and high heterogeneity become scope boundaries, not post-hoc fixes).

**Systematic empirical validation**$\mathbf{\rightarrow}$**(APPENDIX G)**. We validate the RTM diagnostic framework in quantum hardware through a systematic analysis of 31 IBM Quantum processors spanning 5 to 1121 qubits. Initial raw scaling analysis suggested a positive coherence-to-size relationship ($\alpha \approx + 0.23$); however, RTM isolates this as a statistical illusion driven by a manufacturing confounder (generational technology improvements). To definitively untangle chronological engineering advancements from true topological transport scaling, we deployed a Multivariable Orthogonal Distance Regression (ODR) pipeline, injecting a realistic $15\%$ cryogenic calibration noise margin. When algebraically normalizing the technological gain factor ($\gamma = \  + 0.139$ dex/year), the true topological scaling reveals a strictly negative exponent of $\mathbf{\alpha}\mathbf{= \  - 0.259\ }\mathbf{\pm}\mathbf{0.049}$. This places macroscopic quantum decoherence unequivocally in the Inverse Transport Class ($\alpha < \ 0$), alongside classical Stokes-Einstein diffusion. This empirical result proves that as quantum system size ($N$) increases, topological noise (crosstalk, correlated defects) scales collectively rather than independently, causing the system to decohere faster. RTM successfully separates underlying physical scaling laws from engineering artifacts, demonstrating that massive coherence requires architectural resonance, not merely brute-force monolithic scaling.

**1. Introduction**

**1.1 Motivation: beyond fidelities and error rates**

Quantum performance is usually summarized by **point metrics**---single- and two-qubit fidelities, $T_{1}/T_{2}$, logical error rates, or benchmark figures (QED-C, QV). Yet practical reliability and throughput hinge on something orthogonal: **how timing stretches across scale** in a multistage stack---qubits and resonators, code cycles, compilers, cryogenic I/O. When small subsystems respond quickly and larger ones respond more slowly in a disciplined, layered fashion, shocks are **dissipated**; when timings **flatten**, disturbances percolate across layers and synchronize failures (stalling readout, spiking logical error, or forcing global recalibrations).

**Multiscale Temporal Relativity (RTM)** provides a compact language for this phenomenon. Inside a fixed regime, RTM expects a power-law relation between a **characteristic time** $T$ and a **scale proxy** $L$: the **slope** $\alpha$ in $\log T = \alpha\ \log L + c$is structural (invariant to time units), while the intercept $c$is a **clock** (gauge). We bring this principle to quantum computing and show that measuring, validating, and **engineering** $\alpha$ yields actionable levers---independent of nominal units---to improve stability and throughput.

**1.2 RTM in one line**

**Structure lives in the slope; clocks live in the gauge.**\
A change of clock or units shifts $c$ but leaves $\alpha$ unchanged. Thus $\alpha$ can be compared across devices, stacks, and labs, while $c$ cannot.

**1.3 Contributions**

This paper makes five contributions:

1.  **Operationalization of RTM for QC.** We define layer-specific $(L,T)$ pairs for **physical**, **QEC**, **compiler/runtime**, and **I/O--cryo** layers (e.g., $L =$ active qubits, $T =$ stable calibration time; $L = d$, $T =$ cycles to logical failure; $L =$ multiplexing degree, $T =$ readout latency; $L =$ circuit width, $T =$ makespan).

2.  **Validation & estimation.** We provide a **collapse test** (residual independence of $\log T - \alpha\ \log L$from ${log\ }L$) to detect regime mixing and non-power curvature, and adopt **errors-in-variables** estimation (ODR/TLS, Theil--Sen, SIMEX) with bootstrap uncertainty and changepoint guards.

3.  **A single real-time indicator.** We fuse family-wise slopes into $\mathbf{ECI}_{QC}$**(t)** via random-effects meta-analysis with heterogeneity controls ($Q$, $I^{2}$, ${\widehat{\tau}}^{2}$); we publish QA flags and withhold fusion when proxies disagree.

4.  **Design levers.** We formalize **RTM-aware scheduling** (batching, staggered resets, low-variance routing), **QEC cadence design** (desynchronization to avoid phase lock between physical errors and syndrome extraction), and **modular sizing** (choosing module/interconnect scales that elevate $\alpha$ without throttling throughput).

5.  **Falsifiable hypotheses & protocols.** We pre-register **H1--H3** with A/B protocols on superconducting and trapped-ion platforms, metrics (throughput, makespan, logical error, uptime, p95/p50 ratios), and decision thresholds for adoption.

**1.4 What** $\mathbf{\alpha}$ **is---and is not**

-   **Is:** a **binwise slope** linking a time $T$ to a scale $L$ inside a **fixed environment** (same temperature/firmware/topology/syndrome schedule). It captures the **geometry of tempo across scale**.

-   **Is not:** a causal parameter by default; level changes in $T$ (units, clocks, offsets) do **not** change $\alpha$. When collapse fails, $\alpha$ is **undefined** for that bin and should not be fused.

**1.5 Layer-specific** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$**exemplars (preview)**

-   **Physical:** $L =$ active qubits / coupler degree / cluster size; $T =$ stable calibration interval, gate/RO latency, mean time to drift.

-   **QEC:** $L = d$ (code distance) or number of logical qubits; $T =$ cycles to logical failure; cadence of syndrome extraction.

-   **Compiler/runtime:** $L =$ circuit width or depth after mapping; $T =$ makespan; queueing delay and rescheduling latency.

-   **I/O--cryo:** $L =$multiplexing degree or channels; $T =$ readout latency/BER recovery; p95 queue length.

Each layer yields a slope ${\widehat{\alpha}}_{f}$; after QA and collapse, we fuse them into ${ECI}_{QC}$(t) with uncertainty bands. Clean **decoherence events** are significant drops in ${ECI}_{QC}$ over pre-registered horizons.

**1.6 Hypotheses (falsifiable)**

-   **H1 (Resilience):** Higher pre-shock $\alpha$ associates with smaller logical-error spikes at fixed $d$ and longer stable calibration intervals.

-   **H2 (Anticipation):** QA-clean ${ECI}_{QC}$ drops lead increases in makespan, queueing, or logical error by weeks to months, adding predictive value over baselines (fidelity, utilization, temperature).

-   **H3 (Cascade):** Within stable regimes, $\alpha_{\text{physical}} \leq \alpha_{\text{QEC}} \leq \alpha_{\text{runtime/I/O}}$; directionality tests favor micro→meso→macro timing flow.

**1.7 RTM-aware design (intuitions we will test)**

-   **Scheduling:** Avoid patterns that **flatten** $\alpha$ (long, tightly coupled operations in parallel); favor **batching** readouts and **staggered** resets to prevent synchronization cascades.

-   **QEC cadence:** Introduce slight **desynchronization** (phase offsets) between syndrome cycles and known noise rhythms to raise $\alpha_{\text{QEC}}$.

-   **Modularity:** Choose module size and interconnect density where $\alpha$ is high enough to damp inter-module cascades but not so high that throughput is throttled.

**1.8 Relation to prior work**

Our framework complements fidelity-centric and error-model approaches by adding a **scale--tempo geometry**. It is compatible with (not a replacement for) surface/LDPC code theory, compilation/routing heuristics, and queueing models; it contributes a **gauge-invariant** statistic $\alpha$ and a **collapse** specification test to separate **structure** from **clock** effects. In the language of stochastic processes, our dynamics section (later) connects RTM to **time-changed diffusions**; in meta-analysis terms, our fusion mimics **random-effects** with explicit **heterogeneity gates**.

**1.9. Systematic Empirical Validation: The Illusion of Monolithic Scaling**$\mathbf{\rightarrow}$**(APPENDIX G)**

A fundamental premise of RTM is its ability to diagnose the true transport class of a system by observing its scaling exponent. In the race to build fault-tolerant quantum computers, hardware developers have continuously scaled up monolithic processor sizes (qubit counts). Superficially, historical data seems to suggest that larger processors possess better coherence times ($T_{2}$). However, within the RTM framework, we must ask: is this improvement a property of the spatial scale ($\alpha > \ 0$), or is it an artificial offset generated by continuous technological advancements?

To answer this, we utilize RTM as a diagnostic filter on 31 IBM Quantum processors. We hypothesize that quantum decoherence is not a set of isolated independent events, but a collective topological collapse. Therefore, true physical scaling should exhibit an Inverse transport signature ($\alpha < \ 0$), where a larger geometric footprint naturally amplifies crosstalk and correlated noise. By deploying multivariable Errors-in-Variables modeling, we demonstrate how RTM mathematically cuts through manufacturing confounders to reveal the stark, underlying physics of macroscopic quantum systems.

**2. RTM Foundations Adapted to Quantum Computing**

This section states the RTM axioms, derives the **power-law** form $T = \kappa L^{\alpha}$, and tailors **clock/gauge** and **collapse** notions to quantum stacks. Throughout, $L > 0$ is a **scale proxy** (layer-specific) and $T > 0$ is a **characteristic time** measured in a **fixed environment/bin** (same temperature, firmware, topology, syndrome schedule, utilization band).

**2.1 Axioms (binwise)**

**A1 --- Scale semigroup.** For any dilation $b > 0$,

$$T(bL) = f(b)\text{ }T(L),
$$

with $f(1) = 1$ and $f(b_{1}b_{2}) = f(b_{1})f(b_{2})$.

**A2 --- Mild regularity.** $f$ is measurable (or continuous at $b = 1$).

**A3 --- Clock invariance in-bin.** Allowed **clock changes** multiply $T$ by a factor $c > 0$**independent of** $L$ inside the bin (unit changes, timestamp baselines, fixed-latency offsets). In QC practice: rescaling time units, constant readout overheads, constant cryo I/O baselines.

**A4 --- Binning.** Comparisons are made within bins where environment is stable. If a changepoint is detected, the bin must be split.

**2.2 Functional-equation solution → power law**

Let $u = \log L$, $v = \log T$. From A1--A2, the multiplicative Cauchy equation gives $f(b) = b^{\alpha}$ for some $\alpha \in \mathbb{R}$. Hence

$$T(L) = \kappa L^{\alpha},v(u) = \alpha u + \log\kappa.
$$

**Interpretation.** $\alpha$ is the **coherence exponent** (slope); $\kappa$ is a **clock** (intercept).

**2.3 Clocks (multiplicative gauge vs. additive latency)**

In RTM, a "clock change" inside a fixed bin is a **multiplicative** rescaling of all characteristic times: $T^{'} = cT$, $c > 0$ independent of $L$. This includes time-unit conversions (ns↔µs), uniform timebase/tick-rate rescalings, or uniform calibration factors. In log coordinates, $\log T^{'} = \log T + \log c$, so $\alpha$ is unchanged and only the intercept shifts.\
By contrast, **constant latencies** (e.g., fixed readout preamble, pipeline delay, timestamp baseline offsets) are **additive**: $T_{\text{obs}} = T + b$. On log--log plots this is not a pure intercept shift and can bias $\alpha$, especially when $T$ is not $\gg b$. Therefore, before estimating $\alpha$, either:\
(i) estimate/subtract the latency $b$ and fit using $T_{eff} = \max(T_{\text{obs}} - b,\varepsilon)$, or\
(ii) restrict analysis to regimes where $T_{\text{obs}} \gg b$and report sensitivity of $\alpha$ to plausible $b$.

**2.4 Collapse as a binwise specification test**

Given observations $\{(L_{i},T_{i})\}_{i}$ in a bin, define $x_{i} = \log L_{i}$, $y_{i} = \log T_{i}$. Fit a binwise slope $\widehat{\alpha}$ (Section 5) and examine **residuals**

$${\widetilde{y}}_{i}: = y_{i} - \widehat{\alpha}x_{i}.
$$

**Collapse test.** In a valid RTM bin, $\widetilde{y}$ should be **independent of** $x$ (up to noise). We operationalize with:

-   A regression $\widetilde{y} \sim x$ and require $R_{\text{collapse}}^{2} < \tau$ (default $\tau = 0.05$).

-   A **clock placebo**: multiply all $T_{i}$ by a constant; $\widehat{\alpha}$ and $R_{\text{collapse}}^{2}$ must be unchanged.

-   A **smooth check** (LOESS or spline) for visible trend; if present, reject the bin.

**Meaning.** Collapse establishes that, after removing $\widehat{\alpha}\ logL$, only a **gauge** remains (intercept noise), not a trend vs. scale.

**2.5 Variable exponents and finite-window bias**

In practice, $\alpha$ can drift slowly with environment or scale (e.g., across utilization bands or multiplexing factors). Write

$$v(u) = \int_{u_{0}}^{u}{\alpha(s)\text{ }ds + \log\kappa(u),}
$$

with $\mid \alpha^{'}(u) \mid \leq \varepsilon$ small on the window and $\kappa$ **slowly varying**. For any symmetric window of width $h$ in $u$,

$$\widehat{\alpha}(u;h)\text{\:\,} = \text{\:\,}\alpha(u)\text{\:\,} + \text{\:\,}O(\varepsilon h)\text{\:\,} + \text{\:\,}O(\text{slow-variation}),
$$

and

$$R_{\text{collapse}}^{2}\text{\:\,} = \text{\:\,}O((\varepsilon h)^{2}).
$$

**Rule.** Choose bins/windows small enough that curvature is negligible; otherwise split the bin.

**2.6 Failure modes (should fail)**

RTM is designed to **predict its own failure**:

1.  **Regime mixing (kinks).** Example: changing the readout chain or syndrome scheduler mid-bin. The log--log plot shows a slope change at $L^{\star}$; collapse fails.

2.  **Curvature (non-power).** Example: a multiplexing-dependent overhead that grows nonlinearly with $L$. Residuals trend with $x$; collapse fails even after rebinning.

3.  **Scale-dependent clocks.** Any "clock" factor $c(L)$ that depends on $L$ is not a gauge; it injects $du$-components into the 1-form and must be modeled explicitly (or the bin rejected).

**2.7 QC layer mapping (notation and exemplars)**

We will use these **canonical** $(L,T)$ pairs in later sections (others may be added if they pass collapse):

-   **Physical**:\
    $L =$number of **active qubits** (or cluster/coupler degree);\
    $T =$**stable calibration interval**, **gate** latency, **readout** latency, or **mean time to drift**.

-   **QEC**:\
    $L =$**code distance** $d$ (or logical-qubit count);\
    $T =$**cycles to logical failure** at fixed target error.

-   **Compiler/Runtime**:\
    $L =$**circuit width** or **post-mapping depth**;\
    $T =$**makespan** or **queueing delay**.

-   **I/O--Cryo**:\
    $L =$**multiplexing degree** or readout-channel count;\
    $T =$**effective readout latency** / **BER-recovery half-life** / **p95 queue length (in time)**.

Each family produces a binwise ${\widehat{\alpha}}_{f}$. Only families that **pass collapse** and QA contribute to the fused indicator $\mathbf{ECI}_{QC}$**(t)** (Section 6).

**2.8 Why** $\mathbf{\alpha}$**matters operationally**

-   **Comparability**: $\alpha$ is invariant to unit changes and constant overheads, enabling **cross-lab** and **cross-generation** comparison.

-   **Early warning**: significant **drops** in $\alpha$ (per family or fused) signal **decoherence events** likely to precede spikes in logical error, makespan, or forced recalibrations.

-   **Design lever**: raising $\alpha$ (without over-layering) via **scheduling**, **QEC cadence**, or **module sizing** improves damping of cross-scale cascades.

**2.9 Summary**

RTM in QC reduces to three binwise statements: (i) **power-law** scaling $T = \kappa L^{\alpha}$, (ii) **gauge invariance** (only the slope $\alpha$ is structural), and (iii) **collapse** as a falsifiable specification test. With careful binning and EIV-aware estimation, $\alpha$ becomes a reproducible, unit-robust **coherence exponent** that guides both **diagnostics** and **design** across the quantum stack.

**3. Scale--Clock Geometry for QC (Collapse as Exactness)**

We recast RTM for quantum stacks in geometric form. The key object is the **RTM 1-form**

$$\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha(x)\text{ }d(\log L),
$$

defined on a bin $E$ with **environment** coordinates $x$ (temperature, firmware, topology, syndrome schedule, utilization band) and **scale** $u = \log L$. In this language, **collapse** is equivalent to **exactness/flatness** of $\omega$; regime seams and non-power curvature appear as **holonomy/curvature**. This section states the results and instantiates them with QC failure modes.

**3.1 Spaces, bins, and the RTM 1-form**

-   **State space.** $M = X \times \mathbb{R}$ with coordinates $(x,u)$, where $u = \log L$.

-   **Clock potential.** $v(x,u) = \log T(x,L)$.

-   **RTM 1-form.** $\omega = dv - \alpha(x)\text{ }du$ (constant-$\alpha$ case) or $\omega = dv - \alpha(x,u)\text{ }du$ (slow drift allowed).

**A clock change** (unit/baseline shift independent of $L$ inside a bin) is:

``` math
v \mapsto v^{\#} = v + \phi(x).
```

Then

``` math
$$
\omega \mapsto \omega^{\#} = \omega + d\phi(x)
$$
```
a **gauge transformation** by an exact 1-form pulled back from $X$. Hence $\alpha$ **is gauge-invariant**.

**3.2 Collapse ⇔ exactness/flatness**

**Theorem 3.1 (Collapse** $\Leftrightarrow$ **exactness).**\
On a simply connected bin $E$, the following are equivalent:

1.  (RTM chart) $v(x,u) = \alpha(x)\text{ }u + \log\kappa(x)$(or $v = \int\alpha(x,s)\text{ }ds + \log\kappa(x)$ for slow drift).

2.  (**Collapse**) Residual $\widetilde{v}: = v - \alpha u$ is independent of $u$ in $E$.

3.  (**Exactness**) $\omega = d\psi$ on $E$ for some $\psi(x)$(no $u$-dependence).

**Corollary 3.2 (Flatness test).**\
$d\omega = 0$ is necessary and (on simply connected $E$) sufficient for collapse. With $\alpha = \alpha(x,u)$,

$$d\omega\text{\:\,} = \text{\:\,} - \text{ }d\alpha \land du.
$$

Thus curvature (non-power behavior) or regime mixing gives $d\alpha/\text{ }du \neq 0$ and **breaks collapse**.

**3.3 Holonomy and regime seams (QC failure modes)**

Define **holonomy** around a closed loop $\gamma \subset E$: $\mathcal{H(}\gamma) = \oint_{\gamma}^{}{\omega.\ }$ If $\mathcal{H(}\gamma) \neq 0$, collapse cannot hold globally.

**QC instances.**

-   **Scheduler seam.** Changing the syndrome-extraction cadence mid-bin (new FPGA image) produces a kink in $v(u)$; loops that cross the seam pick up nonzero holonomy → **rebin**.

-   **Readout chain swap.** A per-channel overhead that *depends on multiplexing* behaves like a scale-dependent clock $c(L)$; this is **not gauge** and injects $du$-components → collapse fails (and should).

-   **Thermal drift window.** A slow utilization ramp changes $\alpha$ across $u$; if $\partial_{u}\alpha$ is not small on the window, $d\omega \neq 0$→ split the bin or shrink the window.

**3.4 Adiabatic collapse (slowly varying** $\mathbf{\alpha}$**)**

If $\mid \partial_{u}\alpha \mid \leq \varepsilon$ on a window of width $h$,

$$\widetilde{v}(x,u) = v - \alpha(u_{0},x)\text{ }u = \log\kappa(x) + O(\varepsilon h),
$$

and the empirical collapse statistic obeys

$$R_{\text{collapse}}^{2} = O\text{ }((\varepsilon h)^{2}).
$$

**Practice.** Choose $h$ so that $\varepsilon h \ll 1$; otherwise, reduce the bin or model the drift explicitly.

**3.5 Morphisms (reparametrizations) and gauge**

Let $\Phi = (\varphi,\psi)$ map $(X_{A},L_{A},v_{A}) \rightarrow (X_{B},L_{B},v_{B})$, where $\varphi:X_{A} \rightarrow X_{B}$ reparametrizes environment and $\psi:X_{B} \rightarrow \mathbb{R}$ is a clock change. Then

$$\Phi^{*}\omega_{B}\text{\:\,} = \text{\:\,}\omega_{A}\text{\:\,} + \text{\:\,}d(\psi \circ \varphi).
$$

Interpretation: transporting the structure from $B$ to $A$ preserves **slope** and alters only the **clock** by an exact form. This formalizes cross-lab/device comparisons when units/baselines differ.

**3.6 Diagnostics and acceptance gates (QC checklist)**

1.  **Collapse test.** Fit $\widehat{\alpha}$ (Section 5), compute residuals $\widetilde{y} = y - \widehat{\alpha}x$; require\
    $R_{\text{collapse}}^{2} < 0.05$ **and** no trend in a nonparametric smooth.

2.  **Clock placebo.** Multiply all $T$ by a constant; $\widehat{\alpha}$ and $R_{\text{collapse}}^{2}$ must be unchanged.

3.  **Changepoints.** Run detectors on $(x,y)$ and on $\widetilde{y}$; any kink ⇒ rebin.

4.  **Window control.** Ensure $\mid \partial_{u}\alpha \mid \text{ }h$is small (adiabatic regime).

5.  **Publish/withhold.** Only bins passing 1--4 contribute to ${ECI}_{QC}$(t); otherwise label NO_COLLAPSE or REGIME_MIX.

**3.7 What this buys us operationally**

-   A **proof-obligation**: show flatness/exactness (collapse) before trusting a slope.

-   A **debugger**: nonzero holonomy localizes seams (scheduler swaps, readout changes).

-   A **tuning rule**: reduce $h$ or rebin until $d\omega \approx 0$; if impossible, the domain is **non-power**---treat $\alpha$ as undefined there.

**3.8 Summary**

The scale--clock geometry makes two RTM statements precise for QC:

1.  $\alpha$ **is a gauge-invariant structural quantity**, unaffected by unit/baseline changes;

2.  **Collapse equals exactness/flatness of** $\omega$, and its failure is informative (curvature or seams).\
    We will now leverage this to define **operational** $(L,T)$ (Sec. 4) and to estimate $\widehat{\alpha}$ robustly under measurement error (Sec. 5).

**4. Operational** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$**Definitions and Binning Protocol**

This section turns RTM into **measurable practice** for quantum stacks. We define layer-specific $(L,T)$ pairs, specify **sampling**, **units**, and **guards**, and give a binning protocol that avoids regime mixing. Throughout, $u = \log L$, $v = \log T$.

**4.1 Design principles for** $\mathbf{(}\mathbf{L}\mathbf{,}\mathbf{T}\mathbf{)}$

-   **One mechanism per family.** Each $(L,T)$ pair should reflect a single dominant mechanism (e.g., readout pipeline, not a mixture of readout + routing).

-   **Monotone** $L$**.** $L$ should increase with "problem size" at that layer (width, distance, channels, cluster size).

-   **Clock independence.** Within a bin, **multiplicative** timebase changes ($T^{'} = cT$) are allowed gauges (unit/timebase rescalings). **Additive** overheads ($T_{\text{obs}} = T + b$) must be subtracted, modeled, or avoided (fit only where $T \gg b$); otherwise they may bias slopes and invalidate collapse.

-   **Steady sampling.** Use **fixed cadence** collection; record raw timestamps to allow reslicing.

**4.2 Physical layer**

**Candidates for** $L$**:**

-   $L =$number of **active qubits** in the workload window;

-   $L =$**cluster size** (connected qubits participating simultaneously);

-   $L =$**coupler degree** (average fanout).

**Candidates for** $T$**:**

-   **Stable calibration interval** (time until any qubit in the cluster exits tolerance);

-   **Gate latency** (median single/two-qubit gate duration across the active set);

-   **Readout latency** (median per-shot time to valid symbol under fixed thresholds);

-   **Mean time to drift** (MTTD) for frequency/phase.

**Instrumentation.**

-   Log per-shot timestamps; a calibration watchdog recording when thresholds are breached; attach environment tags: temperature band, firmware hash, bias point.

**Non-examples.**

-   Mixing *both* gate latency and readout latency in the same $T$.

-   Letting $L$ be "qubits defined on chip" (not necessarily active).

**4.3 Error Correction (QEC)**

$L$**:** code **distance** $d$ (primary), or number of **logical qubits** at fixed $d$.\
$T$**:**

-   **Cycles to logical failure** at a fixed target error (median or survival quantile);

-   **Syndrome-cycle latency** (mean time per cycle under fixed schedule).

**Scheduling notes.**

-   Freeze a **syndrome schedule** (FPGA image + cadence). Any change ⇒ new bin.

-   Record bias (X/Z) and leakage mitigation settings.

**Edge cases.**

-   If $T$ is dominated by **rare catastrophic events** (e.g., resonator latch-ups), prefer **conditional medians** (exclude known catastrophic flags) and report a sensitivity panel.

**4.4 Compiler / Runtime**

$L$**:** circuit **width** (max concurrent qubits) or **post-mapping depth**; optionally **active layers** after routing.\
$T$**:**

-   **Makespan** (submission → completion);

-   **Queueing delay** (submission → start);

-   **Rescheduling latency** after a calibration event.

**Controls.**

-   Fix **routing policy** and **placement heuristic** inside a bin.

-   Stratify by utilization band (e.g., 0--30%, 30--60%, \>60%). If utilization drifts, split the bin.

**4.5 I/O -- Cryo / Readout**

$L$**:** **multiplexing degree** (channels per line) or number of concurrent readout channels.\
$T$**:**

-   **Readout latency** (median p50 and tail p95);

-   **BER recovery half-life** after a controlled burst;

-   **Queue p95** expressed in time.

**Instrumentation.**

-   Timestamp every DMA/ADC burst; log per-channel buffers; annotate firmware versions of DSP.

**Caveat.**

-   Per-channel overheads that **grow with** $L$ are *not* gauges; they are genuine scale effects---permissible for RTM---but if the overhead itself changes mid-bin, collapse should fail and trigger a split.

**4.6 Binning protocol (environment fixing)**

A **bin** is a maximal interval where the environment is effectively constant.

**Bin key (example):**

$$\text{BIN} = \{\text{platform},\text{ temperature band},\text{ firmware hash},\text{ topology ID},\text{ routing policy},\text{ syndrome cadence},\text{ utilization band}\}.
$$

**Procedure.**

1.  **Slice** data by BIN; discard slices with \<$N_{\min}$ distinct $L$ values (default 6).

2.  **Changepoint scan** on $y = \log T$ vs. $x = \log L$ (and on residuals if available). If a changepoint is detected (BIC/AIC/PELT), **split**.

3.  **Windowing**: for slowly drifting regimes, use sliding windows in $x$ of width $h$ such that $\mid \partial_{u}\alpha \mid \text{ }h \ll 1$ (from Sec. 3.4).

4.  **Clock placebo**: multiply $T$ by a constant; the slope $\widehat{\alpha}$ must not change.

**4.7 Estimation-ready dataset**

Create a tidy table per bin with columns:

$$x = log\ L,\ y = \log T,\text{ family},\text{ BIN tags},\text{ replicate ID},\text{ timestamp},\text{ weights }\rbrack.
$$

-   **Replicates.** If multiple runs at same $L$, aggregate to robust summaries (median $y$, MAD-based SE) or pass all and let ODR handle them with replicate weights.

-   **Weights.** Prefer inverse-variance weights from bootstrap over simple counts.

-   **Outliers.** Tag catastrophic events (hardware flags); report both **with** and **without** them.

**4.8 Acceptance gates (per bin, per family)**

A family contributes a slope ${\widehat{\alpha}}_{f}$ **only if** all hold:

1.  **Coverage:** at least $6$ distinct $L$ points and span $\geq 0.6$ in $\log L$.

2.  **Collapse:** regress $\widetilde{y} = y - \widehat{\alpha}x$ on $x$; require $R_{\text{collapse}}^{2} < 0.05$ and no visible trend (smooth check).

3.  **Clock placebo:** $\widehat{\alpha}$ unchanged under $T \mapsto cT$.

4.  **Changepoints:** none within bin (else split and re-estimate).

5.  **EIV fit quality:** ODR/TLS converged; residual diagnostics acceptable (no single leverage point dominates).

Bins or families failing any gate are flagged (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE, EIV_FAIL) and **excluded from fusion**.

**4.9 Examples vs. non-examples (QC-flavored)**

-   **Good physical family:** $L =$ active-qubit cluster size; $T =$ stable calibration interval. Single firmware, stable temperature, no routing change. Collapses cleanly → accept.

-   **Bad physical family:** Same, but mid-bin the PLL loop parameters change. Changepoint triggers; split required.

-   **Good QEC family:** $L = d$, $T =$ cycles to logical failure, fixed syndrome cadence. Residuals flat → accept.

-   **Bad QEC family:** Mix of two cadences (fast and slow) inside one bin → kink in log--log → reject until split.

-   **Good I/O family:** $L =$ multiplexing degree; $T =$ readout latency p95. Firmware constant; latency rises as $L^{\alpha}$, collapse holds → accept.

-   **Bad I/O family:** Switch of DSP firmware that changes per-channel overhead nonlinearly mid-bin → curvature; reject or rebin around the switch.

**4.10 Summary**

-   We fixed **operational** $(L,T)$ per layer and specified **instrumentation** to make them measurable.

-   We defined a **binning protocol** that enforces environment constancy and guards against regime mixing.

-   We set **acceptance gates** (coverage, collapse, placebo, changepoints, EIV fit) that determine whether a family's slope enters downstream fusion (${ECI}_{QC}$(t)).

**5. Estimation Under Errors-in-Variables (EIV) and Collapse Thresholds**

We now specify **how** to estimate the binwise slope $\alpha$ robustly when both axes are noisy, and how to decide---via a **collapse threshold**---whether a family's data are RTM-consistent. Throughout, $x = \log L$, $y = \log T$. Observations are $x^{obs} = x + \xi$, $y^{obs} = y + \zeta$ with mean-zero errors.

**5.1 Estimation targets and models**

Inside a **fixed bin**, the target is the **local slope** $\alpha$ in

$$y = \alpha x + c + r(x),
$$

with $r \equiv 0$ under exact RTM or $\mid r^{'}(x) \mid \leq \varepsilon$ under slow drift on a window. Because $x$ is noisy, **OLS is attenuated**; we use EIV-aware estimators.

**Default target:** point slope $\alpha$ for the bin; intercept $c$ is a **gauge** (not compared across bins).

**5.2 Orthogonal Distance Regression (Total Least Squares)**

**Definition.** ODR minimizes orthogonal residuals to a line:

$$\underset{\alpha,c}{\min}\sum_{i}^{}\frac{(y_{i}^{obs} - \alpha x_{i}^{obs} - c)^{2}}{\sigma_{y}^{2} + \alpha^{2}\sigma_{x}^{2}}
$$with effective (possibly heterogeneous) $(\sigma_{x},\sigma_{y})$ from replicate variance or bootstrap.

**Practice.**

-   Initialize by Theil--Sen (Sec. 5.4) to avoid poor local minima.

-   Use **cluster/bootstrap** (replicate or job-level) for CIs.

-   If per-point SEs are available, weight them; else use robust Huber weights on orthogonal residuals.

**Convergence gates.**

-   Condition number of the centered covariance matrix $< 10^{4}$.

-   Jackknife leverage check: no single point contributes $> 25\%$ of slope influence.

**5.3 SIMEX (when** $\mathbf{Var}\mathbf{(}\mathbf{\xi}\mathbf{)}$**is known/estimated)**

If you can estimate $\sigma_{\xi}^{2} = Var(\xi)$ (e.g., repeated $L$ at the same setting), apply **SIMEX**:

1.  For $\lambda \in \Lambda = \{ 0.5,1.0,1.5,2.0\}$, generate pseudo-samples\
    $x_{i}^{(\lambda)} = x_{i}^{obs} + \sqrt{\lambda}\text{ }{\widetilde{\xi}}_{i}$, ${\widetilde{\xi}}_{i} \sim \mathcal{N}(0,\sigma_{\xi}^{2})$.

2.  Fit a naive slope $\widehat{\alpha}(\lambda)$ by ODR or OLS.

3.  Fit a quadratic $\widehat{\alpha}(\lambda) = a + b\lambda + c\lambda^{2}$ and **extrapolate to** $\lambda = - 1$:\
    ${\widehat{\alpha}}_{\text{SIMEX}} = a - b + c$.

**Use.** Prefer ODR as the base fitting routine; report SIMEX as a **sensitivity** estimate next to ODR. If $\sigma_{\xi}^{2}$ is uncertain, give a band (low/med/high) for ${\widehat{\alpha}}_{\text{SIMEX}}$.

**5.4 Theil--Sen (robust median slope)**

The **Theil--Sen** slope is the median of all pairwise slopes

$$\alpha_{ij} = \frac{y_{j}^{obs} - y_{i}^{obs}}{x_{j}^{obs} - x_{i}^{obs}}(i < j),
$$

with a robust intercept from the median of $y_{i}^{obs} - \widehat{\alpha}x_{i}^{obs}$.

**Role.**

-   Initialization for ODR.

-   **Outlier-robust** cross-check reported alongside ODR.

-   When EIV is severe and $\sigma_{\xi}^{2}$ is unknown, Theil--Sen may still be stable (expect mild attenuation).

**5.5 Windowing and finite-window bias**

If slow drift is suspected, estimate slopes on **symmetric windows** in $x$ of width $h$. From the adiabatic bias bound,

$$\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h),
$$

choose $h$ so that $\varepsilon h \ll 1$. Practically: start with $h \approx 0.8$ in $\log L$ span if coverage allows; shrink until collapse passes (Sec. 5.7) without exploding variance.

**5.6 Uncertainty and diagnostics**

-   **Bootstrap** (pairs within bin or block/cluster if natural replicates exist) for 50/95% CIs.

-   **Jackknife-after-bootstrap** to detect leverage points.

-   **Residual plots**: orthogonal residual vs. $x$; LOESS smooth must be flat within bands.

-   **EIV adequacy**: if OLS and ODR differ by $\geq$`<!-- -->`{=html}0.2 absolute slope **and** ODR CI excludes OLS, report EIV as material.

**5.7 Collapse threshold (specification gate)**

Given $\widehat{\alpha}$, compute residuals ${\widetilde{y}}_{i} = y_{i}^{obs} - \widehat{\alpha}x_{i}^{obs} - \widehat{c}$ and regress $\widetilde{y}$ on $x$ (with the same weights used in estimation). Define

$$R_{\text{collapse}}^{2}: = R^{2}(\widetilde{y} \sim x).
$$

**Decision rule (default):**

-   Accept the bin if **all** hold:

    1.  $R_{\text{collapse}}^{2} < 0.05$ (or the 95% CI of the slope in $\widetilde{y} \sim x$ contains 0),

    2.  LOESS smooth shows no trend,

    3.  **Clock placebo**: scaling $T \mapsto cT$ leaves $\widehat{\alpha}$ and $R_{\text{collapse}}^{2}$ unchanged,

    4.  Changepoint scan (PELT/BIC) finds none inside the bin.

-   Otherwise flag (NO_COLLAPSE or REGIME_MIX) and **do not** publish a slope or include it in fusion.

**5.8 Coverage and leverage gates**

To avoid brittle fits:

-   **Distinct** $L$**points** $\geq 6$ and $\log L$ span $\geq 0.6$.

-   **Balanced leverage:** the largest leverage point contributes $\leq 25\%$ of the ODR slope influence.

-   **Replicates:** if $> 3$ replicates per $L$, either summarize to a robust mean/SE or pass replicate weights to ODR.

Bins failing these gates are flagged THIN_COVERAGE or LEVERAGE_RISK.

**5.9 Putting it together (per-bin algorithm)**

1.  **Prep:** build the tidy table (Sec. 4.7); run changepoint scan; window if needed.

2.  **Init:** compute Theil--Sen slope/intercept; remove obvious catastrophics (keep both versions for sensitivity).

3.  **Fit ODR/TLS:** weighted by replicate SEs; obtain $\widehat{\alpha}$, $\widehat{c}$, bootstrap CIs.

4.  **SIMEX (optional):** if $\sigma_{\xi}^{2}$ is available, compute ${\widehat{\alpha}}_{\text{SIMEX}}$.

5.  **Collapse gate:** compute $R_{\text{collapse}}^{2}$, smooth check, placebo clock.

6.  **Decision:** if all gates pass, **accept** $\widehat{\alpha}$ with uncertainty; else **reject/split**.

7.  **Report:** slope, CI, diagnostics (collapse $R^{2}$, leverage plot, changepoints). Store flags.

**5.10 What we publish per accepted family**

-   ${\widehat{\alpha}}_{f} \pm$`<!-- -->`{=html}50/95% CI (ODR); Theil--Sen as robustness; SIMEX band if applicable.

-   Collapse diagnostics: $R_{\text{collapse}}^{2}$, placebo check, window width $h$.

-   Coverage: \# distinct $L$, $\log L$ span, leverage summary.

-   Notes: any exclusions (catastrophics), changepoint status.

Only accepted families enter **fusion** (Sec. 6). If $\geq 2$ families pass, we apply random-effects with $Q$, $I^{2}$ and heterogeneity gates; otherwise we report family-wise slopes without fusion.

**5.11 Summary**

-   Use **ODR/TLS** as the primary EIV estimator; **Theil--Sen** for robust init/check; **SIMEX** when $\sigma_{\xi}^{2}$ is estimable.

-   Enforce **collapse** as a **specification test** ($R_{\text{collapse}}^{2} < 0.05$ + placebo + no changepoints).

-   Control **finite-window bias** by choosing $h$ small enough (adiabatic regime) and splitting bins when needed.

-   Publish complete **diagnostics** and **flags**; only clean families proceed to fusion and to the real-time ${ECI}_{QC}$(t).

**6. Building the Real-Time Indicator** $\mathbf{ECI}_{\mathbf{QC}}\mathbf{(}\mathbf{t}\mathbf{)}$

We now construct a **single, real-time** coherence indicator for a platform by fusing the **accepted** family-wise slopes $\{{\widehat{\alpha}}_{f,t}\}$ from Section 5. The fusion is **random-effects** (to acknowledge between-family heterogeneity), runs on a rolling clock, and drives **QA gates** and **decoherence alerts**.

**6.1 Inputs and preconditions (per time** $\mathbf{t}$**)**

For each family $f \in \mathcal{F}_{t}$ (Physical, QEC, Compiler/Runtime, I/O--Cryo):

-   A binwise estimate ${\widehat{\alpha}}_{f,t}$ with variance ${\widehat{\sigma}}_{f,t}^{2}$ (bootstrap or replicate-weighted),

-   Collapse passed (Section 5.7), coverage/leverage gates satisfied (Section 5.8),

-   Environment tags (BIN) unchanged within the window that produced ${\widehat{\alpha}}_{f,t}$.

A fusion at time $t$proceeds **only if** $\mid \mathcal{F}_{t} \mid \geq 2$.

**6.2 Random-effects fusion**

We estimate the between-family variance ${\widehat{\tau}}_{t}^{2}$ (default **REML**; DerSimonian--Laird as sensitivity). Define weights

$$w_{f,t}\text{\:\,} = \text{\:\,}\frac{1}{{\widehat{\sigma}}_{f,t}^{2} + {\widehat{\tau}}_{t}^{2}}.
$$

Then the fused slope and its variance are

$${\widehat{\alpha}}_{QC}(t) = \frac{\sum_{f \in \mathcal{F}_{t}}^{}{w_{f,t}\text{ }{\widehat{\alpha}}_{f,t}}}{\sum_{f \in \mathcal{F}_{t}}^{}w_{f,t}},\ \ Var({\widehat{\alpha}}_{QC}(t)) = \frac{1}{\sum_{f \in \mathcal{F}_{t}}^{}w_{f,t}}.
$$

Report 50% and 95% intervals via normal approximation or by a **bootstrap-over-families** (resample families with replacement, recompute ${\widehat{\tau}}_{t}^{2}$ and the fused mean).

**6.3 Heterogeneity diagnostics and gates**

Compute the fixed-effect baseline

$$w_{f,t}^{FE} = \frac{1}{{\widehat{\sigma}}_{f,t}^{2}},\ \ {\widehat{\alpha}}_{FE}(t) = \frac{\sum_{f}^{}{w_{f,t}^{FE}\text{ }{\widehat{\alpha}}_{f,t}}}{\sum_{f}^{}w_{f,t}^{FE}}.
$$

**Cochran's** $Q$ **and** $I^{2}$**:**

$$Q_{t} = \sum_{f}^{}{w_{f,t}^{FE}\text{ }({\widehat{\alpha}}_{f,t} - {\widehat{\alpha}}_{FE}(t))^{2},\ \ I_{t}^{2} = \max}\{ 0,\text{\:\,}\frac{Q_{t} - ( \mid \mathcal{F}_{t} \mid - 1)}{Q_{t}}\} \times 100\%.
$$

**Fusion gates (pre-registered):**

-   Proceed with a single number **only if**\
    (i) $\mid \mathcal{F}_{t} \mid \geq 2$,\
    (ii) $I_{t}^{2} < 50\%$ *(moderate or lower heterogeneity)*, and\
    (iii) REML converges with finite ${\widehat{\tau}}_{t}^{2}$ not exceeding a historical cap (e.g., ≤ 90th percentile over past clean windows).

-   If any fails, **withhold fusion** and publish family-wise ${\widehat{\alpha}}_{f,t}$+ diagnostics; flag FAMILY_DIVERGENCE.

**6.4 Real-time operation (rolling windows)**

-   **Cadence.** Recompute each family's ${\widehat{\alpha}}_{f,t}$ on a **rolling window** in $x = \log L$ of width $h$ (chosen by the adiabatic rule; Sec. 5.5) and a **wall-clock horizon** (e.g., last 7--28 days of data).

-   **Backfill and missingness.** If a family is missing at $t$, fuse over the available $\mathcal{F}_{t}$ provided $\mid \mathcal{F}_{t} \mid \geq 2$; otherwise **suspend** ${ECI}_{QC}(t)$ and publish a THIN_FAMILIES flag.

-   **Clock placebo.** Once per day, multiply all contributing $T$ by a constant and verify ${\widehat{\alpha}}_{QC}(t)$ and $I_{t}^{2}$ are unchanged (stored as a QA artifact).

**6.5 Decoherence events (alerting logic)**

We define a **decoherence event** as a significant, QA-clean **drop** in ${ECI}_{QC}(t)$, robust to smoothing and not explained by heterogeneity spikes.

**Filters:**

1.  **Smoothing:** maintain a 3-point median $\widetilde{\alpha}(t)$ of ${\widehat{\alpha}}_{QC}(t)$.

2.  **Z-score:** $Z(t) = \frac{\widetilde{\alpha}(t) - {EWMA}_{30}\lbrack\widetilde{\alpha}\rbrack}{{\widehat{\sigma}}_{EWMA}(t)}$.

**Alert tiers (default):**

-   **Advisory:** $Z(t) \leq - 1.5$ for ≥2 consecutive ticks **and** $I_{t}^{2} < 50\%$.

-   **Watch:** $Z(t) \leq - 2.0$ once **or** persistent $Z(t) \leq - 1.5$ for ≥4 ticks, $I_{t}^{2} < 40\%$.

-   **Warning:** $Z(t) \leq - 2.5$ and a coincident family-wise drop (≥2 families with $Z_{f} \leq - 2$).

**Playbooks triggered:** throttle scheduling (reduce concurrency/multiplexing), run segmented recalibration, or switch to RTM-aware routing until $\widetilde{\alpha}(t)$ normalizes.

**6.6 Reporting and visualization**

-   **Primary panel:** ${\widehat{\alpha}}_{QC}(t)$ with 50/95% bands, heterogeneity ribbon colored by $I_{t}^{2}$ (green \<25%, amber 25--50%, red ≥50%).

-   **Forest plot:** per-family ${\widehat{\alpha}}_{f,t}$, weights $w_{f,t}$, and CIs; show $Q_{t}$, $I_{t}^{2}$, ${\widehat{\tau}}_{t}^{2}$.

-   **Collapse dashboard:** per family, show $R_{\text{collapse}}^{2}$, LOESS residuals, window width $h$, coverage and leverage metrics.

-   **Flags legend:** NO_COLLAPSE, REGIME_MIX, LEVERAGE_RISK, THIN_COVERAGE, FAMILY_DIVERGENCE, THIN_FAMILIES.

**6.7 Sensitivity and ablation**

-   Publish the **fixed-effect** summary ${\widehat{\alpha}}_{FE}(t)$ alongside random-effects.

-   Report DL-based ${\widehat{\tau}}_{DL}^{2}$ as a sensitivity.

-   **Leave-one-family-out**: recompute ${\widehat{\alpha}}_{QC}^{( - f)}(t)$ to expose dominance.

-   **Clock placebos** and **shuffle nulls** (shuffle $L$ within family) must not produce tiered alerts; if they do, review gates.

**6.8 Governance and provenance**

Every fused point stores:

-   Source families and their BIN tags,

-   Estimator settings (ODR init, bootstrap seeds, $h$),

-   Collapse metrics, $Q_{t}$, $I_{t}^{2}$, ${\widehat{\tau}}_{t}^{2}$,

-   Placebo outcome hashes,

-   Versioned code/config (methods YAML).

This ensures **reproducibility** and enables post-mortems when alerts fire.

**6.9 Summary**

${ECI}_{QC}(t)$ is a **random-effects fusion** of QA-clean, binwise slopes. Heterogeneity gates ($I_{t}^{2} < 50\%$, $\mid \mathcal{F}_{t} \mid \geq 2$) prevent misleading single numbers when proxies disagree. Real-time smoothing and Z-scores turn slope dynamics into **actionable alerts** for **decoherence events**, while dashboards and provenance keep the system auditable.

**7. RTM-Aware Design: Engineering** $\mathbf{\alpha}$**without Sacrificing Throughput**

This section turns RTM into **design levers**. Goal: increase the **coherence exponent** $\alpha$ (stronger tempo stratification across scale) while keeping or improving throughput. We give layer-specific controls, optimization targets, and guardrails.

**7.1 Design objective and guardrails**

We treat $\alpha$ as an **operational objective** within a bin:\
$$\max_{\text{\:\,controls }\theta}\ \ \ \alpha(\theta)\ \ \ s.t.\ \ \ \ throughput\  \geq \ B,\ \ fidelity\  \geq \ F,\ \ \ \ \ collapse\ passes.$$

-   **Controls** $\theta$: scheduler parameters, QEC cadence/jitter, routing constraints, multiplexing limits, module sizes.

-   **Constraints**: a throughput floor $\mathcal{B}$ (e.g., jobs/hour), fidelity floor $\mathcal{F}$, and **collapse gates** (Sec. 5.7).

-   **Monitor**: track per-family ${\widehat{\alpha}}_{f}$ and the fused ${\widehat{\alpha}}_{QC}(t)$ with QA (Sec. 6).

**7.2 Scheduler: batching & variance-aware routing**

**Problem.** Long, tightly coupled operations launched in parallel **flatten** $\alpha$ (fast cascades across scale).

**Controls.**

1.  **Wavefront batching (readout & long ops).** Partition time into short waves; pack readouts into waves instead of free-running concurrency.

2.  **Staggered resets.** Add small offsets $\delta \in \lbrack - \epsilon,\epsilon\rbrack$ to reset times to avoid synch peaks.

3.  **Low-variance routing.** Prefer routes with **low path-time variance** even if path length increases slightly.

**Objective.** For a job DAG with ops $o$having nominal durations $\tau_{o}$ and routes $p(o)$:

$$\underset{\text{\:\,schedule},\text{ }p( \cdot )}{\min}\text{\:\,}\underset{\text{desynchronize heavy ops}}{\underbrace{{Var}_{t}\lbrack N_{\text{long}}(t)\rbrack}}\text{\:\,} + \text{\:\,}\lambda\text{\:\,}\underset{\text{low-variance routing}}{\underbrace{\sum_{o \in \mathcal{O}}^{}{Var(T_{\text{route}}(p(o)))}}}.
$$

subject to makespan budget. This reduces temporal "pile-ups," lifting $\alpha$.

**Heuristic (greedy, practical).**

-   Sort ops by duration desc; assign start times into **waves** so that each wave's total long-op load is balanced.

-   For each route candidate, penalize time-variance and crosstalk score; pick minimum penalized cost.

**7.3 QEC cadence: avoid phase-lock (jitter/desynchronization)**

**Problem.** A fixed syndrome cadence can **phase-lock** with physical noise rhythms, creating cross-layer synchronization → $\alpha_{QEC}$ falls.

**Controls.**

-   **Micro-jitter** the cycle period: $P_{k} = P\text{ }(1 + \eta_{k})$ with $\eta_{k} \sim \mathcal{U}\lbrack - \rho,\rho\rbrack$, $\rho \ll 1$ (e.g., 1--3%).

-   **Multi-phase extraction:** split the code into sublattices whose cycles are offset by small phases $\phi_{j}$.

**Design rule.** Choose $\rho$ so that the **main lobe** of the syndrome cycle's line spectrum moves **off** strong peaks of the error PSD while keeping decoder timing valid. Validate by: (i) increased ${\widehat{\alpha}}_{QEC}$ vs. $d$, (ii) stable logical error at fixed $d$.

**7.4 Gradients and wells of** $\mathbf{\alpha}$

Two architectural motifs to **steer flows**:

-   **Gradient:** arrange resources so $\alpha$ **increases** towards critical compute regions. Small disturbances decay as they travel inward.

-   **Well:** create a **high-**$\alpha$ **basin** around sensitive qubits (e.g., clocking and buffering that slow large-scale cascades).

**Implementation cues.** Increase temporal buffering (queues, damped scheduling) and reduce crosstalk fanout as you approach the "core," but cap buffering (Sec. 7.1 guardrails) so throughput doesn't suffer.

**7.5 Modular sizing: pick a sweet spot by balancing intra vs. inter latency**

Let total qubits $Q$ be partitioned into $Q/m$ modules of size $m$. Approximate **characteristic time**:

$$T(m)\text{\:\,} = \text{\:\,}A\text{ }m^{a}\text{\:\,} + \text{\:\,}B\text{ }(\frac{Q}{m})^{b}\text{     }\text{(intra-module cost + interconnect cost)}.
$$

**Optimal module size** (minimizes $T$):

$$m^{\star}\text{\:\,} = \text{\:\,}{(\frac{B\text{ }b}{A\text{ }a})}^{\frac{1}{a + b}}\text{\:\,}Q^{\frac{b}{a + b}}.\ 
$$

-   $a > 0$: intra-module scaling (e.g., calibration, routing within module).

-   $b > 0$: inter-module scaling (e.g., photonic/ion link latency).

**Design use.** Measure $a,b$ empirically (RTM per mechanism), estimate $A,B$, compute $m^{\star}$. Operate near $m^{\star}$ and verify that $\widehat{\alpha}$ **does not collapse** (still power-like) in that neighborhood.

**7.6 Multiplexing & I/O: hold tails in check**

**Problem.** Aggressive multiplexing reduces per-shot time, but can synchronize queue tails → $\alpha_{IO} \downarrow$.

**Controls.**

-   Cap multiplexing such that the **tail ratio** $p95/p50$ of readout latency stays below a threshold (e.g., $\leq 1.6$).

-   Use **phase-offset readout windows** across channels to avoid coherent tail growth.

-   Buffer sizing: maintain buffer utilization \< 70% to avoid tail amplification.

**Signal.** If $p95/p50$ grows and ${\widehat{\alpha}}_{IO}$ drops with clean collapse, back off multiplexing and introduce offsets.

**7.7 Online control loop (closed-loop** $\mathbf{\alpha}$ **engineering)**

A simple controller to keep $\alpha$ high under constraints:

every Δt:

estimate {α_f(t), σ_f(t)} per accepted family (Sec. 5)

if \|F_t\| ≥ 2 and I\^2_t \< 50%:

compute α_QC(t) (Sec. 6)

if α_QC(t) \< α_floor and constraints met:

apply actions A = {↑wave size, ↑reset jitter ρ, ↑routing penalty on variance,

↓multiplex cap, move toward m\*}

else if throughput \< B:

relax A minimally (keep collapse passing)

log QA: collapse R\^2, I\^2_t, flags; revert actions if flags trip

-   $\alpha_{\text{floor}}$: pre-registered minimal acceptable fused slope.

-   **Revert** any action that causes NO_COLLAPSE or $I_{t}^{2} \geq 50\%$.

**7.8 Safety and validation**

-   Any intervention must **re-pass collapse** in the affected families.

-   Run A/B windows (≥2--4 weeks) with **pre-registered** KPIs: throughput, makespan, logical error, uptime, $p95/p50$, and ${\widehat{\alpha}}_{f}$.

-   If $\alpha$rises but KPIs worsen beyond budgets, you are **over-layering** (too much buffering). Roll back to the Pareto frontier.

**7.9 Quick-start playbooks**

-   **If** $\alpha_{QEC} \downarrow$**:** add 1--3% cadence jitter; introduce 2--3 phase groups for syndrome; re-measure collapse.

-   **If** $\alpha_{IO} \downarrow$**:** reduce multiplex cap 10--20%; add 1--2 cycle offsets; keep $p95/p50 \leq 1.6$.

-   **If** $\alpha_{runtime} \downarrow$**:** enable readout batching; penalize high-variance routes; cap concurrent long ops per wave.

-   **Architectural planning:** estimate $a,b,A,B$ and set module size near $m^{\star}$; confirm power-like scaling around that point.

**7.10 Summary**

-   **Scheduler** (waves, staggered resets, low-variance routing) and **QEC cadence** (micro-jitter, multi-phase) are first-line levers to **raise** $\alpha$.

-   **Modular sizing** admits a closed-form optimum $m^{\star}$ balancing intra/inter costs; operate near it while watching collapse.

-   **I/O controls** keep latency tails from synchronizing.

-   A **closed-loop controller** maintains $\alpha$ above a floor under throughput/fidelity budgets.

**8. Falsifiable Experimental Protocols (Superconducting & Trapped-Ion)**

This section specifies **testable** RTM-QC experiments with concrete $(L,T)$ choices, data collection, analysis plans, and success criteria. Each protocol is binwise (fixed environment) and includes **placebos**, **changepoint guards**, and a **pre-registered** decision table.

**8.1 Common scaffolding (applies to all protocols)**

**BIN (environment) lock.**\
$\{$platform; temperature band; firmware hash (FPGA/DSP); topology ID; routing policy; syndrome cadence; utilization band$\}$. Any change ⇒ new bin.

**Data schema (tidy).** For each record:

$$x = log\ L,y = logT,\text{ family},\text{ BIN tags},\text{ replicate ID},\text{ timestamp},\text{ weights}\rbrack$$

**QA gates (must pass):**

-   Coverage: ≥6 distinct $L$, span ≥0.6 in $\log L$.

-   EIV fit converged (ODR), leverage \<25%, robust init (Theil--Sen).

-   Collapse: $R_{\text{collapse}}^{2} < 0.05$, no LOESS trend, clock placebo holds.

-   Changepoints: none inside bin (else split).

**Outcomes (primary, per family):**

-   Slope ${\widehat{\alpha}}_{f}$ with 50/95% CI; collapse diagnostics.

-   For fused results, ${\widehat{\alpha}}_{QC}(t)$, $Q$, $I^{2}$, ${\widehat{\tau}}^{2}$ (Sec. 6).

**Statistical plan.**\
Bootstrap CIs (pairs/cluster). Predefine **minimal detectable effect** (MDE) on $\alpha$ (e.g., $\Delta\alpha = 0.15$) and **operational KPIs** (throughput, makespan, logical error rate, uptime, p95/p50). Thresholds below.

**8.2 Protocol A --- Physical layer (Superconducting)**

**Hypothesis (H1-Phys).** Increasing **cluster desynchronization** (staggered resets + readout waves) **raises** $\alpha_{\text{phys}}$ without exceeding throughput budget.

**Design.**

-   $L$: active-qubit **cluster size** (simultaneously engaged).

-   $T$: **stable calibration interval** (time to first out-of-tolerance flag in cluster).

-   Arms: **Control** (baseline scheduler) vs. **RTM-aware** (readout batching + staggered resets, ±2--4% offsets).

-   Duration: 2--4 weeks; interleave arms daily to balance drift.

**Analysis.**

-   Fit ODR per arm, pass collapse.

-   Primary effect: $\Delta{\widehat{\alpha}}_{\text{phys}} = {\widehat{\alpha}}_{\text{RTM}} - {\widehat{\alpha}}_{\text{CTRL}}$.

-   KPI guardrails: throughput drop ≤5%, no increase in gate/RO error \>0.2σ.

**Success criteria.**

-   $\Delta{\widehat{\alpha}}_{\text{phys}} \geq 0.15$ and CI excludes 0, **and** guardrails satisfied.

-   If collapse fails in any arm, declare **inconclusive** and rebin.

**Placebos.** Multiply $T$ by a constant; $\widehat{\alpha}$ unchanged. Shuffle $L$ within day; no significant slope.

**8.3 Protocol B --- QEC cadence (Superconducting or Ions)**

**Hypothesis (H1-QEC).** Introducing **micro-jitter** (1--3%) in syndrome period and/or **multi-phase extraction** increases $\alpha_{\text{QEC}}$ vs. code distance $d$at fixed decoder.

**Design.**

-   $L$: **code distance** $d$ (e.g., $d \in \{ 3,5,7,9\}$).

-   $T$: **cycles to logical failure** (median or survival quantile at fixed target error).

-   Arms: Control (fixed period $P$) vs. Jitter ($P_{k} = P(1 + \eta_{k})$, $\eta_{k} \sim \mathcal{U}\lbrack - 0.02,0.02\rbrack$) and/or 2--3 **phase groups**.

-   Keep decoder parameters fixed; no change in noise bias mitigation.

**Analysis.**

-   ODR per arm; collapse gate.

-   Effect: $\Delta{\widehat{\alpha}}_{\text{QEC}}$.

-   KPI guardrails: logical error at fixed $d$ not worse by \>5% relative.

**Success criteria.**

-   $\Delta{\widehat{\alpha}}_{\text{QEC}} \geq 0.15$ with CI excluding 0 and guardrails pass.

**Diagnostics.** Check PSD of error processes; confirm jitter moves cadence lines off dominant peaks.

**8.4 Protocol C --- compiler/runtime scheduling**

**Hypothesis (H2-Run).** **Wavefront batching** of readout and **low-variance routing** reduce synchronization cascades, increasing $\alpha_{\text{runtime}}$ and lowering makespan tails.

**Design.**

-   $L$: **post-mapping circuit width** (or active layers).

-   $T$: **makespan** (submit→complete).

-   Arms: Baseline policy vs. RTM-aware (waves + variance-penalized routing).

-   Control utilization band; same job mix across arms.

**Analysis.**

-   ODR slope per arm; collapse.

-   KPIs: median makespan (≤ baseline), p95/p50 latency ↓ ≥10%.

**Success criteria.**

-   $\Delta{\widehat{\alpha}}_{\text{runtime}} \geq 0.10$(CI excludes 0) and p95/p50 improves ≥10%.

**8.5 Protocol D --- I/O--Cryo multiplexing**

**Hypothesis (H2-IO).** **Phase-offset readout windows** across channels maintain or raise $\alpha_{\text{IO}}$ while reducing p95 tails at a given multiplexing degree.

**Design.**

-   $L$: **multiplexing degree** (channels/line).

-   $T$: **readout latency p95** (and p50).

-   Arms: Synchronous windows vs. offset windows (phase pattern $\phi_{j}$).

-   Sweep $L$ across operational range.

**Analysis & success.**

-   $\Delta{\widehat{\alpha}}_{\text{IO}} \geq 0.10$; p95/p50 ≤ 1.6 in RTM arm over majority of $L$; collapse passes.

**8.6 Protocol E --- Modular sizing (planning study)**

**Hypothesis (H3-Mod).** There exists a module size $m^{\star}$ that minimizes $T(m) = Am^{a} + B(Q/m)^{b}$ with empirically measured $a,b > 0$, and operating near $m^{\star}$ preserves power-like scaling (collapse holds).

**Design.**

-   Platforms with photonic/ion links between modules.

-   Measure $T(m)$ by varying module size (or emulating interconnect cost) at fixed total $Q$.

-   Fit $a,b,A,B$ via ODR on each term's dataset; compute $m^{\star}$.

**Success criteria.**

-   Observed $T(m)$ minimized near $m^{\star}$ (within CI), and log--log fits around $m^{\star}$ retain collapse (no curvature).

**8.7 Fusion and alerting (cross-protocol)**

Across A--D, if ≥2 families pass gates at overlapping times, compute ${\widehat{\alpha}}_{QC}(t)$ (Sec. 6).\
**H2 (anticipation):** declare a **decoherence event** if Z-score tiers (Sec. 6.5) are met; test **lead--lag** vs. spikes in logical error/makespan/queues. Additive predictive value is assessed against baselines (fidelity, utilization, temperature) using time series regression with HAC errors; pre-register horizons (e.g., 7--30--90 days).

**8.8 Placebos, shuffles, and robustness**

-   **Clock placebos:** multiply all $T$ by constants; $\widehat{\alpha}$ and $R_{\text{collapse}}^{2}$ invariant.

-   **Shuffle nulls:** permute $L$ within day; slopes collapse to \~0 (within CI).

-   **Leave-one-family-out** fusion to reveal dominance.

-   **Changepoints**: automatic split if detected; re-estimate on both sides.

**8.9 Power and duration (rules of thumb)**

-   With span ≥0.8 in $\log L$, 8--12 distinct $L$ points, and moderate noise (SNR≈5--10), ODR detects $\Delta\alpha \approx 0.10$--0.15 at 95% with ≈200--400 total observations per arm.

-   If noise is higher or drift suspected, shrink windows (Sec. 5.5) and extend duration.

**8.10 Decision table (pre-registered)**

| Outcome | Action |
| :--- | :--- |
| $\Delta\hat{\alpha} \geq$ MDE **and** guardrails pass | Promote intervention to production in that bin; monitor with $\text{ECI}_{\text{QC}}(t)$. |
| $\Delta\hat{\alpha}$ significant but KPI guardrail violated | Tune intensity (e.g., reduce buffering/jitter) and retest. |
| Collapse fails or heterogeneity high ($I^2 \geq 50\%$) | Do not fuse; report family-wise; revisit binning or mechanisms. |
| No effect ($\Delta\hat{\alpha} \approx 0$) | Document as *scope boundary*; keep as negative control. |

**8.11 Ethics, safety, and reproducibility**

-   **Safety:** no unsafe RF power increase; jitter bounds keep decoders valid; rollback on NO_COLLAPSE or KPI breach.

-   **Reproducibility:** versioned methods YAML (BIN, estimator settings, seeds), public plots (collapse panels, forest plots), and stored placebo/shuffle artifacts.

-   **Transparency:** publish both successes and failures (negative results define scope).

**8.12 Summary**

These protocols make RTM-QC **falsifiable**: each claims a directional change in $\alpha$ from a specific control, under binwise constancy, with collapse as a specification test and operational guardrails. Success improves not only the slope but also **run-time stability** (tails, recalibrations) without sacrificing throughput.

**9. Results Templates and Reporting Standards**

This section defines **what to publish** once the protocols (Sec. 8) are run. It standardizes figures, tables, robustness panels, and a one-page checklist so that results are interpretable, reproducible, and directly comparable across labs and platforms.

**9.1 Figure set (minimum)**

**Fig. 1 --- Collapse panels (per accepted family).**\
Four small multiples per family $f$within a bin:

1.  **Log--log fit:** $y = \log T$ vs. $x = \log L$ with ODR line and 95% band.

2.  **Residual vs.** $x$**:** $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$ with LOESS; show $R_{\text{collapse}}^{2}$.

3.  **Coverage/leverage:** scatter highlighting leverage points; annotate span in $\log L$, \# distinct $L$.

4.  **Placebo check:** overlay of fits before/after $T \mapsto cT$ (curves coincide).

**Fig. 2 --- Forest plot & heterogeneity.**\
Per time slice (or per experiment arm), show ${\widehat{\alpha}}_{f} \pm$CI, weights $w_{f}$, the fused ${\widehat{\alpha}}_{QC}$ (diamond), and heterogeneity stats: $Q$, $I^{2}$, ${\widehat{\tau}}^{2}$.

**Fig. 3 ---**$\mathbf{ECI}_{QC}$**(t) time series.**\
Rolling fused slope with 50/95% bands; background ribbon colored by $I^{2}$ (green \<25%, amber 25--50%, red ≥50%). Mark **decoherence events** (advisory/watch/warning) and platform events (recalibrations, firmware changes).

**Fig. 4 --- KPI panel (paired with Fig. 3).**\
Aligned time axes for: logical error rate (at fixed $d$), makespan median and p95, queue p95, uptime between recalibrations. Overlay shaded regions for alert tiers from Fig. 3.

**Fig. 5 --- A/B outcomes (per protocol).**\
For each arm: distribution plots (violin/box) of ${\widehat{\alpha}}_{f}$, makespan p95/p50, logical error; include $\Delta\widehat{\alpha}$ with CI and guardrails.

**Optional Fig. 6 --- Spectral diagnostics (QEC).**\
PSD of error processes showing how cadence jitter/multi-phase moves line spectra off dominant peaks.

**9.2 Core tables**

**Table 1 --- Accepted families (per bin/arm).**

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Family**   **#L pts**   $\mathbf{\log}\mathbf{L}$ **span**   ${\widehat{\mathbf{\alpha}}}_{\mathbf{f}}$ **(ODR, 50/95% CI)**   **Theil--Sen**   **SIMEX band**   **(**$\mathbf{R}_{\mathbf{coll}}^{\mathbf{2}}$**)**   **Leverage max**   **Flags**
  ------------ ------------ ------------------------------------ ----------------------------------------------------------------- ---------------- ---------------- ----------------------------------------------------- ------------------ -----------
  Physical     9            1.05                                 0.62 \[0.55, 0.70\]                                               0.60             0.58--0.66       0.02                                                  0.18               ---

  QEC          8            0.82                                 0.74 \[0.66, 0.82\]                                               0.71             ---              0.03                                                  0.22               ---

  ...          ...          ...                                  ...                                                               ...              ...              ...                                                   ...                ...
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table 2 --- Fusion & heterogeneity (per time slice or arm).**

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Time/Arm**   **Families**   ${\widehat{\mathbf{\alpha}}}_{\mathbf{QC}}$ **± SE**   **(Q) (df)**   $$\mathbf{I}^{\mathbf{2}}$$   $${\widehat{\mathbf{\tau}}}^{\mathbf{2}}$$   **Fusion?**
  -------------- -------------- ------------------------------------------------------ -------------- ----------------------------- -------------------------------------------- -----------------------------
  RTM-aware      3              0.69 ± 0.04                                            3.2 (2)        37%                           0.005                                        Yes

  Control        3              0.54 ± 0.05                                            6.8 (2)        71%                           0.018                                        **No** (report family-wise)
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table 3 --- Protocol outcomes (A/B).**

  ----------------------------------------------------------------------------------------------------------------------------------------------------------
  **Protocol**   **Metric**                                        **Control**   **RTM-aware**   **Effect (Δ)**   **95% CI**           **Pass guardrail?**
  -------------- ------------------------------------------------- ------------- --------------- ---------------- -------------------- ---------------------
  A (Phys)       $${\widehat{\mathbf{\alpha}}}_{\mathbf{phys}}$$   0.48          0.64            +0.16            \[0.07, 0.25\]       ✔

  A (Phys)       Throughput                                        100%          97%             --3%             \[--6, 0\]%          ✔

  B (QEC)        $${\widehat{\mathbf{\alpha}}}_{\mathbf{QEC}}$$    0.68          0.83            +0.15            \[0.06, 0.24\]       ✔

  C (Run)        p95/p50                                           1.85          1.60            --0.25           \[--0.35, --0.15\]   ✔
  ----------------------------------------------------------------------------------------------------------------------------------------------------------

**Table 4 --- Pre-registered thresholds & flags.**

  ------------------------------------------------------------------------------
  **Gate**                **Threshold**                             **Status**
  ----------------------- ----------------------------------------- ------------
  Collapse $R^{2}$        \< 0.05                                   Pass

  Heterogeneity $I^{2}$   \< 50% for fusion                         Pass

  MDE on $\Delta\alpha$   ≥ 0.10--0.15                              Pass

  KPI guardrails          ≤5% throughput loss; ≤+5% logical error   Pass
  ------------------------------------------------------------------------------

**9.3 Robustness and sensitivity panel**

-   **Estimators:** ODR (primary), Theil--Sen, SIMEX (± bands for $\sigma_{\xi}^{2}$).

-   **Windows:** repeat with $h$± 25%; $\widehat{\alpha}$ stable and collapse still passing.

-   **Placebos:** clock rescaling; **Shuffles:** permute $L$ within-day---slope → \~0.

-   **Leave-one-family-out fusion:** report ${\widehat{\alpha}}_{QC}^{( - f)}$.

-   **Catastrophics:** re-estimate excluding flagged events; show Δ.

-   **Fixed-effect vs. random-effects:** publish both; divergence implies genuine heterogeneity.

**9.4 Negative results & scope boundaries**

Publish bins/arms that **failed**:

-   NO_COLLAPSE (curvature), REGIME_MIX (kinks), THIN_COVERAGE, LEVERAGE_RISK, FAMILY_DIVERGENCE (high $I^{2}$).\
    Include a short note: suspected mechanism and next steps (rebin, instrumentation change, mechanism isolation). Negative results define **where RTM does not apply**.

**9.5 One-page checklist (for each figure/table set)**

-   BIN keys listed and unchanged.

-   \# distinct $L$≥ 6 and span ≥ 0.6.

-   ODR converged; Theil--Sen reported; SIMEX (if $\sigma_{\xi}^{2}$ known).

-   Collapse: $R^{2} < 0.05$; placebo OK; no changepoints.

-   Fusion: $\mid \mathcal{F}_{t} \mid \geq 2$; $I^{2} < 50\%$; REML converged.

-   KPIs: throughput, makespan p95/p50, logical error, uptime---guardrails applied.

-   Robustness panel completed (windows, shuffles, LOO).

-   Provenance hashes (methods YAML, seeds, code version) included.

**9.6 Narrative template (short "Results" text)**

> *Physical layer.* Across 9 cluster sizes (span 1.05 in $\log L$), RTM-aware scheduling increased the slope from $0.48$to $0.64$ (Δ = $0.16$, 95% CI $\lbrack 0.07,0.25\rbrack$); residuals showed $R_{\text{collapse}}^{2} = 0.02$. Throughput remained within the 5% guardrail.\
> *QEC.* With 1--3% cadence jitter, $\alpha_{\text{QEC}}$ rose from $0.68$ to $0.83$ (Δ = $0.15$, CI $\lbrack 0.06,0.24\rbrack$), logical error at fixed $d$ did not worsen.\
> *Runtime.* Wavefront batching and variance-aware routing reduced p95/p50 from 1.85 to 1.60; $\alpha_{\text{runtime}}$ increased by $0.12$.\
> *Fusion.* Three families passed gates; $I^{2} = 37\%$. The fused ${\widehat{\alpha}}_{QC} = 0.69 \pm 0.04$. A **watch**-level decoherence alert fired on day 17; it preceded a makespan spike by 3 days.

**9.7 Summary**

The templates above ensure every claim is backed by: (i) **collapse** visual and numeric proof, (ii) EIV-aware estimation, (iii) **heterogeneity** accounting for fusion, (iv) KPI guardrails, and (v) complete **robustness** evidence.

**10. Discussion**

This section interprets RTM-QC results, clarifies how a **slope-first** view complements fidelity/QEC paradigms, and lays out trade-offs, risks, and adoption paths.

**10.1 What does a higher** $\mathbf{\alpha}$ **actually buy?**

A larger binwise slope $\alpha$ means **time stretches more steeply with scale**, i.e., larger aggregates slow down *relative* to smaller ones within a stable environment. Operationally:

-   **Shock damping:** disturbances at small scale are less likely to synchronize larger layers (runtime → QEC → I/O), reducing cascades that inflate tails (p95/p50), queues, and forced recalibrations.

-   **Predictability:** higher $\alpha$ typically reduces **run-to-run variance** (narrower KPI distributions) because the stack's "tempo gradient" prevents alignment of rare long events.

-   **Control leverage:** $\alpha$ is unit-agnostic; we can optimize it with scheduler/QEC/interconnect knobs without conflating unit changes (clocks) with structural change.

**Not a substitute for fidelity.** RTM improves **how** timing behaves across scale; it does not increase single/two-qubit fidelities by itself. Gains arrive through fewer cascades and better use of existing fidelity.

**10.2 Complementarity with QEC and compilation**

-   **QEC:** Traditional design picks code distance $d$ from error rates. RTM adds a second axis: **cadence geometry**. Slight **desynchronization** (jitter/multi-phase) can raise $\alpha_{\text{QEC}}$ at fixed $d$and decoder, often improving stability without extra overhead.

-   **Compilation/runtime:** State-of-the-art routing minimizes depth/length. RTM asks also to minimize **time-variance** and **coincidence of long ops**, which can improve tails even if mean depth changes marginally.

**10.3 Trade-offs and Pareto front**

-   **Throughput vs. layering:** Raising $\alpha$ by adding buffers/batching can reduce raw concurrency. We therefore optimize on a **Pareto front** (Sec. 7.1): increase $\alpha$ *subject to* throughput/fidelity floors.

-   **Jitter vs. decoder timing:** Micro-jitter must stay within decoder validity; otherwise you trade higher $\alpha$ for logical failures.

-   **Modular size:** Operating near $m^{\star}$ (Sec. 7.5) balances intra/inter costs, but drifting too far (bigger or smaller modules) can either flatten $\alpha$ (synchronization) or throttle bandwidth.

**10.4 Failure modes (informative by design)**

RTM's **collapse** gate turns failures into diagnostics:

-   NO_COLLAPSE**:** curved log--log → missing mechanism (e.g., scale-dependent "clock" or nonlinear overhead).

-   REGIME_MIX**:** kinks → hidden seams (firmware/scheduler swaps); rebin or split.

-   **High** $I^{2}$**:** proxies disagree → do **not** fuse; inspect per-family controls.

Publishing these cases maps **scope boundaries** (where RTM does *not* apply), which is scientifically useful and prevents overreach.

**10.5 Why a single fused indicator---and when not to use it**

**Pros:** ${ECI}_{QC}(t)$ summarizes multiscale coherence, enabling **alerts** (Sec. 6.5) and trend tracking.\
**Cons:** Fusion can hide heterogeneity. Hence the **gates** (at least two families, $I^{2} < 50\%$, REML convergence). If they fail, publish **family-wise** ${\widehat{\alpha}}_{f}$ only; the lack of fusion is itself a result ("the stack is speaking with different slopes").

**10.6 Relation to time-changed diffusions and queueing**

The PDE view (RTM as a **state-dependent clock**) explains why **tails** shrink when $\alpha$ rises: the **effective dynamic exponent** $z$increases, and exit/first-passage times scale more steeply with "radius" (Sec. 6 of the math paper). In queueing terms, scheduling that raises $\alpha$ **decorrelates** service bursts and dampens tail amplification.

**10.7 External validity and portability**

Because $\alpha$ is **gauge-invariant**, comparisons hold across labs and generations when bins are matched (environment keys). The same pipeline ports to **trapped ions**, **superconducting**, **neutral atoms**, and **annealers** with layer-appropriate $(L,T)$. What changes is instrumentation; the **collapse logic** and **EIV estimation** remain.

**10.8 Adoption path (practical)**

1.  **Shadow mode:** compute per-family ${\widehat{\alpha}}_{f}$ and collapse panels without changing operations.

2.  **Low-risk knobs:** enable **readout batching**, **staggered resets**, and tiny **cadence jitter** (≤3%).

3.  **Close the loop:** bring ${ECI}_{QC}(t)$ into on-call dashboards with alert tiers and playbooks.

4.  **Architectural planning:** measure $a,b,A,B$ (Sec. 7.5) to choose module sizes; iterate quarterly.

**10.9 Open questions**

-   **Decoder co-design:** how to include $\alpha$ directly in decoders' scheduling/graph updates?

-   **Learning controllers:** can RL tune $\alpha$ subject to KPI floors without violating collapse?

-   **Holonomy tests:** practical statistics to distinguish curvature from topological obstructions (global collapse failure).

-   **Cross-layer causality:** when do $\alpha$ changes at the physical layer *cause* changes at runtime vs. just correlate via utilization?

**10.10 Takeaway**

RTM-QC adds a **third axis**---the **geometry of tempo**---to fidelity and scale. With strict gates (collapse, heterogeneity) and modest controls (batching, jitter, routing variance), $\alpha$ becomes a reliable lever for **stability and throughput**, yielding early warnings and design guidance while respecting scientific falsifiability.

**11. Limitations & Scope**

**Bin dependence.** RTM is a **binwise** theory. If the environment (temperature, firmware, topology, decoder, utilization) drifts, the slope $\alpha$ is undefined until the bin is split. Results are only valid within clearly documented BIN keys.

**Proxy choice sensitivity.** $(L,T)$ proxies must reflect a **single dominant mechanism** per family. Mis-specified proxies (e.g., mixing readout and routing in the same $T$) induce curvature and validly fail collapse.

**Finite-window bias.** When $\alpha(u)$ drifts, any finite window of width $h$incurs $O(\varepsilon h)$ bias. Our adiabatic guidance mitigates but does not eliminate it; reported $\widehat{\alpha}$ should be interpreted as **local**.

**EIV model assumptions.** ODR/TLS and SIMEX assume well-behaved errors (mean-zero, finite moments) and independence from $x$. Heavy-tailed or state-dependent errors require robustness checks (Theil--Sen, bootstrap, sensitivity bands).

**Fusion heterogeneity.** Random-effects fusion is appropriate only when families are **commensurate** and $I^{2} < 50\%$. Otherwise the single-number indicator is withheld by design; RTM does not force agreement across mechanisms.

**Causality limits.** $\alpha$ is **structural but not causal** by default. Design sections propose interventions and A/B protocols, yet causal claims require the pre-registered controls and guardrails we specify.

**Scope boundaries.** Systems with **non-power** timing (persistent curvature), **scale-dependent clocks** (overheads that grow with $L$ inside a bin), or strong **holonomy** (global seams) lie **outside** RTM's applicability. In such domains, treat $\alpha$ as undefined and publish negative results.

**12. Methods & Reproducibility**

**12.1 Data schema and BINs**

-   **BIN key:** {platform, temperature band, firmware hash (FPGA/DSP), topology ID, routing policy, syndrome cadence, utilization band}.

-   **Tidy table (per bin):** \[x=log L, y=log T, family, BIN tags, replicate_id, timestamp, weight\].

-   **Coverage gates:** ≥6 distinct $L$, span ≥0.6 in $\log L$.

**12.2 Estimation pipeline (per family, per bin)**

1.  **Changepoint scan:** PELT/BIC on $(x,y)$ and on residuals if available; split if detected.

2.  **Init:** Theil--Sen slope/intercept; flag catastrophics; build replicate weights.

3.  **Primary fit:** ODR/TLS (orthogonal residuals) with replicate or bootstrap SEs.

4.  **SIMEX (optional):** when $\sigma_{\xi}^{2}$ is estimable; extrapolate to $\lambda = - 1$.

5.  **Collapse test:** regress $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$on $x$; require $R_{\text{collapse}}^{2} < 0.05$, flat LOESS, clock placebo holds.

6.  **Diagnostics:** leverage ≤25%; residual plots; window width $h$logged.

7.  **Accept/Reject:** accept if all gates pass; else flag (NO_COLLAPSE, REGIME_MIX, THIN_COVERAGE, LEVERAGE_RISK, EIV_FAIL).

**12.3 Fusion and heterogeneity (rolling)**

-   **Weights:** $w_{f} = 1/({\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2})$ with ${\widehat{\tau}}^{2}$ via REML (DL as sensitivity).

-   **Fused slope:** ${\widehat{\alpha}}_{QC} = \sum w_{f}{\widehat{\alpha}}_{f}/\sum w_{f}$; **variance:** $1/\sum w_{f}$.

-   **Diagnostics:** fixed-effect baseline, **Cochran's** $Q$ and $I^{2}$.

-   **Gates:** fuse only if $\mid \mathcal{F} \mid \geq 2$ and $I^{2} < 50\%$. Otherwise publish family-wise.

**12.4 Real-time operation and alerts**

-   **Rolling windows:** sliding horizon in $x$ (width $h$) and wall-clock (7--28 days).

-   **Smoothing:** 3-point median; **Z-score** against 30-day EWMA.

-   **Alert tiers:** Advisory/Watch/Warning thresholds (Sec. 6.5).

-   **Playbooks:** throttle concurrency, stagger resets, cadence jitter, variance-aware routing; all interventions must re-pass **collapse**.

**12.5 Robustness & sensitivity**

-   **Estimators:** publish ODR (primary), Theil--Sen, SIMEX bands.

-   **Windows:** ±25% $h$ sensitivity; $\widehat{\alpha}$ stability required.

-   **Placebos & shuffles:** clock rescaling invariance; $L$-shuffles yield near-zero slopes.

-   **Leave-one-family-out** fusion; **fixed-effect** vs **random-effects** comparison.

**12.6 Provenance (methods YAML)**

-   BIN keys, estimator settings, bootstrap seeds, SIMEX $\Lambda$, window $h$, collapse thresholds, heterogeneity gates, versions of analysis code.

-   All plots and numbers include hash of the methods YAML; re-runs with the same YAML reproduce numbers within bootstrap noise.

**13. Conclusion & Outlook**

We presented **RTM-aware quantum computing (RTM-QC)**: a **slope-first** framework that measures and **engineers** the geometry of time across scale. Inside stable bins, the characteristic time $T$ scales with a size proxy $L$ as $T \propto L^{\alpha}$; the **coherence exponent** $\alpha$ is invariant to clocks and thus comparable across devices, stacks, and labs. With **collapse** as a falsifiable gate and **errors-in-variables** estimation, $\alpha$becomes a reliable operational signal. Fusing clean, layer-wise slopes yields a real-time $\mathbf{ECI}_{QC}$**(t)** that supports **early warnings** (decoherence events) and **design decisions** (scheduler, QEC cadence, modular sizing, I/O offsets).

**What this adds.** RTM-QC complements fidelity/QEC by introducing a third axis---**tempo geometry**---that explains and controls tails, queues, and synchronization cascades. Modest, reversible controls (batching, staggered resets, micro-jitter, low-variance routing) can **raise** $\alpha$ without degrading throughput or fidelity when used with guardrails.

**What it does not do.** RTM-QC does not replace physical improvements (fidelities, $T_{1}/T_{2}$), nor does it guarantee causality without the A/B protocols and guardrails we specify. Failures to collapse, high heterogeneity, or regime seams are **informative**, delineating scope boundaries rather than inviting post-hoc fixes.

**Near-term agenda.**

1.  **Run the protocols** (Sec. 8) on superconducting and ion platforms; publish both successes and negatives with full collapse/fusion diagnostics.

2.  **Close the loop**: deploy ${ECI}_{QC}(t)$ dashboards and alert playbooks in production; evaluate lead--lag vs. KPI spikes.

3.  **Co-design with decoders** and compilers so cadence and routing optimize $\alpha$ subject to throughput/fidelity floors.

4.  **Standardize reporting**: figures/tables in Sec. 9, methods YAML, and open robustness artifacts.

**Longer-term questions.** Incorporate $\alpha$ into **time-changed diffusion models** of queues; develop **holonomy tests** to distinguish curvature from seams; extend to **modular networks** and **neutral-atom** platforms; integrate learning-based controllers that respect collapse gates.

**Bottom line.** RTM-QC gives quantum teams a **unit-robust, falsifiable lever** over multiscale timing. Measure the slope, **validate by collapse**, fuse when families agree, and **engineer** $\alpha$---not as a slogan, but as a reproducible practice to deliver more stable and efficient quantum computation.

**Appendices**

**Appendix A --- Mathematical Background (RTM essentials for QC)**

**A.1 Semigroup → power law**

Assume binwise scale semigroup $T(bL) = f(b)T(L)$, $f(1) = 1$, and measurability near $b = 1$. Then $f(b) = b^{\alpha}$ and

$$T(L) = \kappa L^{\alpha},v(u) = \log T = \alpha u + \log\kappa,u = \log L.
$$

$\alpha$ is **gauge-invariant**; $\kappa$ is a **clock**.

**A.2 1-form & collapse**

Define the RTM 1-form $\omega = dv - \alpha\text{ }du$. **Collapse** (residual independence of $v - \alpha u$ from $u$) is equivalent to **exactness** of $\omega$ on a simply connected bin:

$$\omega = d\psi(x),d\omega = 0,\psi\text{ independent of }u.
$$

If $\alpha = \alpha(x,u)$, then $d\omega = - d\alpha \land du$; nonzero curvature breaks collapse.

**A.3 Variable exponents (finite-window bias)**

For slowly varying $\alpha(u)$:

$$v(u) = \int_{u_{0}}^{u}{\alpha(s)\text{ }ds + \log\kappa(u),\widehat{\alpha}(u;h) = \alpha(u) + O(\varepsilon h),}
$$

and $R_{\text{collapse}}^{2} = O((\varepsilon h)^{2})$ for window width $h$.

**Appendix B --- Estimators & Algorithms**

**B.1 Orthogonal Distance Regression (TLS/ODR)**

Minimize orthogonal residuals:

$$\underset{\alpha,c}{\min}\sum_{i}^{}\frac{(y_{i} - \alpha x_{i} - c)^{2}}{\sigma_{y,i}^{2} + \alpha^{2}\sigma_{x,i}^{2}}.
$$

**Init:** Theil--Sen; **CIs:** bootstrap pairs/cluster; **checks:** condition number \< $10^{4}$; max leverage \< 25%.

**B.2 Theil--Sen**

Median of pairwise slopes $\alpha_{ij} = (y_{j} - y_{i})/(x_{j} - x_{i})$; robust to outliers; mild EIV attenuation.

**B.3 SIMEX (optional)**

If $\sigma_{\xi}^{2} = Var(\xi)$ is estimable, simulate $x^{(\lambda)} = x^{obs} + \sqrt{\lambda}\widetilde{\xi}$ and extrapolate $\widehat{\alpha}(\lambda)$ to $\lambda = - 1$.

**B.4 Collapse gate**

Regress residuals $\widetilde{y} = y - \widehat{\alpha}x - \widehat{c}$ on $x$; require $R_{\text{collapse}}^{2} < 0.05$ and flat LOESS; pass clock placebo.

**Appendix C --- Protocol Cards (copy--paste templates)**

**C.1 Physical (staggered resets + readout waves)**

-   **L/T:** $L =$active cluster size; $T =$ stable calibration interval.

-   **Arms:** Control vs RTM-aware (waves + 2--4% reset offsets).

-   **Duration:** 2--4 weeks, interleaved.

-   **Success:** $\Delta\alpha_{\text{phys}} \geq 0.15$ (95% CI excludes 0), throughput loss ≤5%, collapse passes.

**C.2 QEC (micro-jitter / multi-phase)**

-   **L/T:** $L = d$; $T =$ cycles to logical failure.

-   **Arms:** Fixed period vs $Pk = P(1 + \eta k),\  \mid \eta k \mid \leq 0.02$ or 2--3 phase groups.

-   **Success:** $\Delta\alpha_{\text{QEC}} \geq 0.15$, no logical-error regression (\>5%) at fixed $d$.

**C.3 Runtime (batching + low-variance routing)**

-   **L/T:** $L =$ post-mapping width; $T =$ makespan.

-   **Arms:** Baseline vs wavefront + variance-penalized routing.

-   **Success:** $\Delta\alpha_{\text{runtime}} \geq 0.10$ and p95/p50 latency ↓ ≥10%.

**C.4 I/O (phase-offset windows)**

-   **L/T:** $L =$ multiplexing degree; $T =$ readout latency p95 (and p50).

-   **Arms:** Synchronous vs phase-offset windows.

-   **Success:** $\Delta\alpha_{\text{IO}} \geq 0.10$, p95/p50 ≤ 1.6 over majority of $L$.

**Appendix D --- Methods YAML (skeleton)**

bin:

platform: \"SC\" \# or \"IONS\", \"NA\"

temperature_band: \"10-15mK\"

firmware_hash: \"fpga_1.4.2_dsp_0.9.8\"

topology_id: \"mesh-v3\"

routing_policy: \"baseline\" \# or \"rtm-aware\"

syndrome_cadence: \"P=3.2us, jitter=0%\"

utilization_band: \"30-60%\"

estimation:

min_L_points: 6

min_logL_span: 0.6

eiv: \"odr\"

odr:

init: \"theil-sen\"

leverage_cap: 0.25

bootstrap: {clusters: true, reps: 2000, seed: 123}

simex:

enabled: false

lambda: \[0.5,1.0,1.5,2.0\]

collapse:

r2_threshold: 0.05

placebo_clock: true

changepoint_scan: {method: \"PELT\", penalty: \"BIC\"}

fusion:

heterogeneity_gate_I2: 0.5

tau2_method: \"REML\"

min_families: 2

eci_rt:

window_logL: 0.8

horizon_days: 14

smoothing: \"median3\"

alert:

z_advisory: -1.5

z_watch: -2.0

z_warning: -2.5

**Appendix E --- Notation Glossary**

-   $L$: scale proxy (layer-specific); $u = \log L$.

-   $T$: characteristic time; $v = \log T$.

-   $\alpha$: **coherence exponent** (slope; clock-invariant).

-   **Bin**: environment slice with fixed {platform, temperature band, firmware hash, topology ID, routing policy, syndrome cadence, utilization band}.

-   **Collapse**: $R^{2}(\widetilde{y} \sim x) < 0.05$ for $\widetilde{y} = y - \widehat{\alpha}x$; residuals show no trend vs $x$.

-   $\mathbf{ECI}_{QC}(t)$: fused slope via random-effects at time $t$.

-   $Q,I^{2},\tau^{2}$: heterogeneity statistics for fusion.

-   ODR/TLS, Theil--Sen, SIMEX: slope estimators under EIV.

-   **Adiabatic window**: width $h$in $u$ where $\mid \partial_{u}\alpha \mid h \ll 1$.

**Appendix F --- Reproducible Figure Recipes (minimal)**

-   **Collapse panel**:

    -   Fit ODR; compute residuals $\widetilde{y}$.

    -   Plot $y$vs $x$+ ODR band; residual vs $x$ with LOESS.

    -   Annotate $R_{\text{collapse}}^{2}$, \#$L$, span, leverage.

-   **Forest plot**:

    -   For accepted families, display ${\widehat{\alpha}}_{f} \pm$CI; compute $w_{f}$, $Q$, $I^{2}$, ${\widehat{\tau}}^{2}$.

    -   Overlay fused ${\widehat{\alpha}}_{QC}$.

-   $\mathbf{ECI}_{QC}(t)$:

    -   Rolling fusion; show 50/95% bands; background colored by $I^{2}$ tiers; mark alert tiers.

**APPENDIX G --- Empirical Analysis: Quantum Hardware Scaling and the Generational Confounder**

The RTM framework dictates that increasing the physical bounds of a tightly coupled but non-resonant network will proportionally increase its topological friction. To test this in quantum arrays, we analyzed the $T_{2}$ coherence times of 31 IBM Quantum processors (5 to 1121 qubits).

**G.1 Heuristic Observation and Simpson's Paradox**

Initial naive Ordinary Least Squares (OLS) regression on the raw dataset yielded a positive scaling exponent of $\alpha = \  + 0.227$. This created the illusion that adding more qubits intrinsically extended coherence times. However, this is a classic manifestation of Simpson\'s Paradox: larger processors were built years later than smaller ones, meaning their extended $T_{2}$ times were the result of superior superconducting materials and fabrication techniques, not their increased spatial size.

**G.2 Rigorous Multivariable EIV Validation**

To mathematically isolate the physical scaling law from human engineering progress, we deployed a \"Red Team\" statistical pipeline:

1.  **Multivariable Orthogonal Distance Regression (ODR):** We abandoned crude categorical \"era binning\" in favor of a continuous multivariable model. This simultaneously evaluates chronological technology progression alongside topological spatial expansion.

2.  **Calibration Noise Injection:** We explicitly injected a realistic $15\%$ hardware calibration variance into the $T_{2}$ readings, forcing the framework to absorb standard cryogenic measurement noise.

**G.3 The Inverse Transport Class (Robust Findings)**

Once the continuous improvement of superconducting materials is algebraically normalized, the illusion of monolithic scaling shatters, revealing the true physics of the quantum array:

-   **Technological Gain Factor:** The model precisely extracts the engineering progression, showing that IBM hardware coherence improves by a factor of $\mathbf{\gamma}\mathbf{= \  + 0.139}$ **dex/year**.

-   **True Topological Exponent:** After subtracting $\gamma$, the isolated physical scaling reveals a robust, strictly negative exponent of $\mathbf{\alpha}\mathbf{= \  - 0.259\ }\mathbf{\pm}\mathbf{0.049}$.

**Conclusion:** Macroscopic quantum decoherence resides securely inside the **Inverse Transport Class** ($\alpha < \ 0$). RTM empirically validates that decoherence in large processor arrays is not a localized, per-qubit phenomenon, but a massive collective topological leakage: structural coherence naturally and predictably degrades as the geometric network size increases.

*© 2026 Álvaro José Quiceno Rendón. This document is distributed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.*
