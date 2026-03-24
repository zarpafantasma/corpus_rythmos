![](media/image1.jpeg){width="2.058333333333333in" height="2.058333333333333in"}

**Scale--Clock Geometry**

A Mathematical Foundation for RTM

Álvaro Quiceno

**Abstract**

We develop a rigorous foundation for **Multiscale Temporal Relativity (RTM)** as a mathematical theory of **time--scale scaling**. Starting from a scale--semigroup axiom $T(bL) = f(b)\text{ }T(L)$ with mild regularity, we derive the **power law** $T(L) = \kappa L^{\alpha}$, identifying the **coherence exponent** $\alpha$ as a **clock-invariant** slope and $\kappa$ as a gauge (clock) factor. We recast RTM in geometric form via the 1-form

$$\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha\text{ }d(\log L),$$

and prove that **collapse**---residual independence of $\log T - \alpha\ \log L$ from ${log\ }L$---is equivalent to **exactness/flatness** of $\omega$ on a bin; regime mixing and non-power alternatives appear as **holonomy/curvature**. We embed RTM in **regular variation** with **variable exponents**, quantifying finite-window bias and showing that collapse statistics scale with curvature. A **renormalization** operator (scale dilation + re-gauge) has power laws as **fixed points** and is **contractive** in Hölder/Zygmund classes; slowly varying clocks lie on a **center manifold**, and slowly drifting exponents yield **adiabatic attraction**. In dynamics, RTM acts as a **space-dependent clock** for diffusions and Dirichlet forms, giving similarity exponents $z = m + \alpha$ and exit-time laws $T \sim R^{\text{ }m + \alpha}$ with adiabatic error bounds. For inference under errors-in-variables, we show consistency of **ODR/TLS**, **SIMEX**, and **Theil--Sen** for local $\alpha$, and formalize the collapse statistic as a specification test against curvature. A category-theoretic packaging makes clocks a gauge and slope the moduli invariant, clarifying functorial behavior under products and coarse-graining. We conclude with constructive counterexamples and open problems (holonomy tests, graph settings, inverse problems, heavy-tailed noise).

**1. Introduction**

**1.1 Problem and viewpoint**

Many systems exhibit a systematic relation between a **characteristic time** $T$ and a **scale proxy** $L$: larger units operate on slower clocks, smaller units on faster clocks. RTM posits that **inside a fixed environment** this relation is **multiplicatively consistent** under rescaling of $L$. The empirical practice---seen across physics, biology, and economics---is to examine the slope of $\log T$ vs. ${log\ }L$ and test whether residuals "collapse" after detrending by that slope.

This paper supplies a **mathematical backbone** for that practice. Our central claim is that the **slope** $\alpha$ is the structural object (invariant under clock changes), while **clocks** are a gauge. With this separation, RTM becomes a clean theory linking: (i) functional equations → power laws, (ii) a **1-form/connection** whose flatness encodes collapse, (iii) **regular variation** with variable exponents to quantify finite-window effects, (iv) **renormalization** as scale-dilation dynamics with power-law fixed points, (v) **diffusions with space-dependent clocks**, and (vi) **statistical identifiability** under measurement error.

> [!NOTE]
> **Scope note:** This document operates under **Assumptions 1–6** of Doc 001 (Sec. 2.1): local density **ρ** and temperature **Θ** are uniform within each bin.
>
> Under these conditions, the full master law:
> **T/T₀ = (L/L₀)^α · Θ(𝒯) / √(ρ/ρ₀)**
> reduces to:
> **T = κL^α**
>
> For treatment of variable **ρ** and **Θ**, see Doc 001 Sec. 2.1–2.2. For empirical applications where these assumptions may be violated, the **collapse test** (Sec. 3.2, 7) provides a falsifiable diagnostic.

**1.2 Contributions**

1.  **Semigroup → Power law (Sec. 2).**\
    From $T(bL)=f(b)T(L)$ with measurable/continuous $f$, we obtain $f(b) = b^{\alpha}$ and $T(L) = \kappa L^{\alpha}$. The slope $\alpha$ is **clock-invariant**; intercept $\log\kappa$ is the clock.

2.  **Scale--clock geometry (Sec. 3).**\
    With $\omega = d\ \log T - \alpha\text{ }d\ \log L$, we prove\
    **collapse ⇔ exactness/flatness** of $\omega$ on a bin. Regime mixing and non-power behavior manifest as **holonomy** ($\oint\omega \neq 0$). We quantify **adiabatic collapse** when $\alpha$ drifts slowly.

3.  **Variable-exponent regular variation (Sec. 4).**\
    We formalize $T(L;x) = L^{\alpha(x)}\mathcal{l}(L;x)$ with **uniform** slow variation, derive **finite-window bias** $O( \parallel \partial_{u}\alpha \parallel \text{ }h)$, and show collapse statistics scale like curvature $\sim h^{2}$.

4.  **Renormalization & stability (Sec. 5).**\
    A dilation-plus-re-gauge operator has power laws as **fixed points**; in Hölder/Zygmund classes it is a **contraction**, giving **local attraction** to the power-law manifold. Slowly varying clocks form a **center manifold**; slowly drifting exponents yield **adiabatic tracking**.

5.  **RTM diffusions & Dirichlet forms (Sec. 6).**\
    Let conductivity be $L(x)^{- \alpha(x)}$. With $\alpha$ constant, solutions obey self-similarity with dynamic exponent $z = m + \alpha$ and exit-time scaling $T \sim R^{\text{ }z}$; with slow drift we obtain **adiabatic error bounds**. RTM diffusions are **time-changed Brownian motions**.

6.  **Identifiability & inference under EIV (Sec. 7).**\
    We show **ODR/TLS** and **SIMEX** consistently recover local $\alpha$ under standard assumptions; **Theil--Sen** provides robust checks. The **collapse statistic** is a specification test against curvature even with measurement error.

7.  **Categorical packaging (Sec. 8).**\
    Objects carry $\omega$ and a gauge; **slope** is the moduli invariant, clocks are gauge; products add slopes; coarse-graining is a functor with power-law fixed points.

8.  **Examples, counterexamples, open problems (Sec. 9).**\
    We give constructive failures (kinks, curved log--log), and list open directions (holonomy tests, graphs, inverse problems, heavy-tailed errors).

**1.3 Relation to prior work**

Our use of multiplicative Cauchy equations and **regular variation** follows Karamata/de Haan, but adapted to **variable exponents** uniform in environment. The 1-form framing parallels standard **gauge/connection** ideas, here specialized to a scale--clock bundle so that "collapse" is **exactness**. The renormalization operator is classical in spirit but applied to **log-time functions** with **gauge equivalence**; the contraction on Hölder/Zygmund spaces provides a clean path from empirical scaling to a dynamical fixed-point picture. The diffusion results connect to **time-changed processes** and variable-coefficient elliptic theory, giving explicit similarity exponents tied to $\alpha$. On inference, our statements place **errors-in-variables** estimators inside the RTM invariance structure, clarifying what is and isn't identifiable.

**1.4 Scope and falsifiability**

RTM is intended for **bins**---domains where environment is stable enough that clocks are $L$-independent. The theory **predicts its own failure modes**: non-power curvature or regime mixtures produce holonomy and non-vanishing collapse statistics. These are **scope boundaries**, not defects.

**1.5 Paper roadmap**

-   **Sec. 2:** semigroup axiom ⇒ power law; clock invariance.

-   **Sec. 3:** scale--clock geometry; collapse = exactness; holonomy and adiabatic collapse.

-   **Sec. 4:** variable-exponent regular variation; finite-window bias; curvature tests.

-   **Sec. 5:** renormalization fixed points; contraction and center manifold; adiabatic tracking.

-   **Sec. 6:** RTM diffusions/Dirichlet forms; similarity exponent $z = m + \alpha$; exit times; time change.

-   **Sec. 7:** identifiability and consistency under EIV; collapse as specification test.

-   **Sec. 8:** categorical formulation and functorial properties.

-   **Sec. 9:** examples, counterexamples, open problems; concise conclusion.

**Guiding principle:** *structure lives in the slope; clocks live in the gauge.* The remainder of the paper makes this precise across analysis, geometry, dynamics, and inference.

**2. Scale Semigroup → Power Law (Foundations)**

This section formalizes the scaling axiom behind RTM and derives the power-law form $T(L) = \kappa L^{\alpha}$. We also isolate the **clock** as a multiplicative gauge and prove that **slope** $\alpha$ is the structural invariant. Throughout, $L\in\mathbb R_0$ denotes a size/scale variable and $T(L)\in\mathbb R_(>0)$ a characteristic time.

**2.1 Axioms and consequences**

We separate **scale symmetry** from **clock choice**.

**Axiom 2.1 (Scale semigroup).**\
There exists a family of maps $\{ S_{b}\}_{b > 0}$ (scalings by factor $b$) and a function $f:\mathbb{R}_{> 0} \rightarrow \mathbb{R}_{> 0}$ such that for all $b > 0$ and all $L > 0$,

$$T(S_{b}L) = T(bL) = f(b)\text{ }T(L),
$$

with $f(1) = 1$ and

$$f(b_{1}b_{2}) = f(b_{1})\text{ }f(b_{2})\ \ \ \ \ \ \ \ (\text{semigroup composition}).
$$

**Axiom 2.2 (Mild regularity).**\
Either (i) $f$ is measurable on $\mathbb{R}_{> 0}$, or (ii) $f$is continuous at $b = 1$.\
(Any standard regularity---Baire/measurable/locally bounded---will do.)

**Definition 2.3 (Clock transform).**\
A change of measurement units or baseline timing is a map $T \mapsto T^{\#}$ of the form $T^{\#}(L) = c\text{ }T(L)$ for some constant $c > 0$ (or, more generally, $c = c(x)$ depending on an external **environment** parameter $x$, but *independent of* $L$ within a fixed environment).

**2.2 Functional-equation solution**

**Lemma 2.4 (Multiplicative Cauchy).**\
Under Axioms 2.1--2.2, $f(b) = b^{\alpha}$ for some $\alpha \in \mathbb{R}$.

*Proof.* Set $g(\log b): = \log f(b)$. The semigroup law gives $g(u + v) = g(u) + g(v)$. Measurability (or continuity at 0) forces $g(u) = \alpha u$ for some $\alpha \in \mathbb{R}$. Exponentiating, $f(b) = e^{g(\log b)} = b^{\alpha}$.

**Theorem 2.5 (Power-law representation).**\
Fix any $L_{0} > 0$. Under Axioms 2.1--2.2,

$$T(L) = T(L_{0})\text{ }(\frac{L}{L_{0}})^{\alpha} = \kappa\text{ }L^{\alpha},\ \ \ \ \ \ \ \ \text{where}\text{        }\kappa: = T(L_{0})L_{0}^{- \alpha}.
$$

*Proof.* Apply Lemma 2.4 with $b = L/L_{0}$:

$$T(L) = T\text{ }((L/L_{0})L_{0}) = f(L/L_{0})\text{ }T(L_{0}) = (L/L_{0})^{\alpha}T(L_{0}).
$$

Rearrange to define $\kappa$.

**Corollary 2.6 (Log-linear form).**\
$\log T = \alpha\ \log L + \log\kappa$. Hence **slope** $\alpha$ captures scaling, while **intercept** $\log\kappa$ captures the clock.

**2.3 Clock invariance and identifiability of** $\mathbf{\alpha}$

**Proposition 2.7 (Clock invariance of the slope).**\
If $T^{\#}(L) = c\text{ }T(L)$ with $c > 0$ independent of $L$ (within a fixed environment), then

$$\log T^{\#} = \alpha\log L + (\log\kappa + \log c),
$$

so the regression slope of ${log\ }T^{\#}$ on $\log L$equals $\alpha$.

*Proof.* Immediate from the corollary.

**Remark 2.8 (Environment-dependent clocks).**\
If the clock factor depends on an external label $x$ but not on $L$---i.e. $T^{\#}(L;x) = c(x)\text{ }T(L;x)$---then within any fixed $x$-environment bin the slope stays $\alpha(x)$, while the intercept shifts by $\log c(x)$.

**Proposition 2.9 (Uniqueness up to clock).**\
If $T_{1}(L) = \kappa_{1}L^{\alpha_{1}}$ and $T_{2}(L) = \kappa_{2}L^{\alpha_{2}}$ satisfy $T_{2}(L) = c\text{ }T_{1}(L)$ for all $L$ with some $c > 0$, then $\alpha_{1} = \alpha_{2}$ and $c = \kappa_{2}/\kappa_{1}$.

*Proof.* Take logs and compare coefficients of $\log L$.

**2.4 Regular-variation generalization (optional but useful)**

The exact scaling can be relaxed to **regular variation**, which covers slowly varying clocks and asymptotic power laws.

**Definition 2.10 (Karamata regular variation).**\
A positive function $T$ is *regularly varying of index* $\alpha$ if for all $b > 0$,

$$\underset{L \rightarrow \infty}{\lim}\frac{T(bL)}{T(L)} = b^{\alpha}.
$$

Equivalently, $T(L) = L^{\alpha}\text{ }\mathcal{l}(L)$ with $\mathcal{l}$ *slowly varying* ($\mathcal{l}(bL)\mathcal{/l}(L) \rightarrow 1$).

**Theorem 2.11 (RTM under regular variation).**\
If $T$ is regularly varying with index $\alpha$, then local log--log slopes over compact $\log L$-windows converge to $\alpha$. Changes of clock that are slowly varying (e.g., $\mathcal{l}$) perturb the intercept asymptotically but not the slope.

*Sketch.* Standard Karamata representation and Tauberian arguments; consistency of local slope follows from uniform convergence of ratios.

**Remark 2.12 (Finite-window bias).**\
When $\mathcal{l}$ is not flat on the observed range, $\widehat{\alpha}$ is biased by $O\ (\sup \mid log\mathcal{l \mid})$ across the window. This motivates **environment fixing** and narrow windows in empirical RTM.

**2.5 Necessary and sufficient conditions for power-law scaling**

The next statement packages RTM's **collapse test** idea at the algebraic level.

**Proposition 2.13 (Equivalence of power-law and log-affinity).**\
For a fixed environment, the following are equivalent:

1.  $T(L) = \kappa L^{\alpha}$ for some $\kappa > 0,\alpha \in \mathbb{R}$.

2.  There exist constants $\alpha,c$ such that $\log T - \alpha\ \log L \equiv c$ for all $L$.

3.  For any $L_{1} \neq L_{2}$,

$$\frac{\log T(L_{2}) - \log T(L_{1})}{\log L_{2} - \log L_{1}} \equiv \alpha\ (\text{independent of the pair}).
$$

*Proof.* (1)⇒(2) is Cor. 2.6; (2)⇒(3) by subtracting; (3)⇒(2) by fixing $L_{1}$and integrating constancy of the discrete derivative; then exponentiate to get (1).

**Corollary 2.14 (Binwise specification test).**\
Given observations $\{(L_{i},T_{i})\}$ in a fixed environment, if a consistent slope $\alpha$ exists such that the residuals ${\widetilde{y}}_{i}: = \log T_{i} - \alpha\ \log L_{i}$ are **independent of** $\log L_{i}$(up to noise), the RTM power-law specification is not rejected for that bin. Any systematic trend of $\widetilde{y}$ vs. $\log L$ falsifies exact power-law scaling in that bin.

**2.6 Counterexamples and scope**

**Counterexample 2.15 (Mixtures of regimes).**\
Let $T(L) = \kappa_{1}L^{\alpha_{1}}$ for $L \leq L^{\star}$ and $T(L) = \kappa_{2}L^{\alpha_{2}}$ for $L > L^{\star}$ with $\alpha_{1} \neq \alpha_{2}$. Then no single $\alpha$ fits globally; any attempt will show residual trend changes at $L^{\star}$. (This is **regime mixing** and should be split into bins.)

**Counterexample 2.16 (Curvature).**\
Let $\log T = g(\log L)$ with $g^{''} \neq 0$ on an interval. Then the discrete slope in Prop. 2.13 depends on the pair, violating the power-law condition; collapse must fail on that interval.

**2.7 Finite-sample estimation with measurement error (set-up for later)**

While proofs above are exact-function statements, empirical RTM faces noisy $L,T$. Write $x = \log L$, $y = \log T$, and observe

$$x^{obs} = x + \xi,y^{obs} = y + \zeta,
$$

with mean-zero errors. Ordinary least squares attenuates the slope when $\xi \neq 0$. We will later show (Section 7) that **orthogonal distance regression** (total least squares) and **SIMEX** yield consistent $\widehat{\alpha}$ under standard conditions; the invariance results (Props. 2.7--2.9) still hold exactly because clock multiplies $T$, not $L$.

**2.8 Summary of Section 2**

-   The **scale semigroup** + mild regularity forces a **power law** $T = \kappa L^{\alpha}$.

-   **Slope** $\alpha$ is **clock-invariant** and identifies the structure; **intercept** $\log\kappa$ encodes the clock.

-   **Regular variation** extends the theory to asymptotic scaling with slowly varying clocks.

-   RTM's **collapse** criterion is the algebraic statement that $\log T - \alpha\ \log L$ is constant (no trend vs. $\log L$) inside a bin.

-   Regime mixtures and curvature provide clean **counterexamples**, justifying binning and specification tests.

**3. Scale--Clock Geometry (Collapse as Exactness)**

We give a geometric formulation of RTM that separates **slope** (structure) from **clock** (gauge). The key object is the 1-form

$$\omega\text{\:\,} = \text{\:\,}d(\log T)\text{\:\,} - \text{\:\,}\alpha(x)\text{ }d(\log L),
$$

defined on a product space where $x$ indexes *environment* and $L > 0$ is *scale*. RTM's **collapse** criterion becomes the statement that $\omega$ is **exact/flat** on a bin. This section makes that precise and proves the equivalences.

**3.1 Spaces, coordinates, and bins**

-   Let $X$ be a smooth (or at least topological) **environment space** collecting background conditions (policy regime, technology, microstructure).

-   Let $S = \mathbb{R}_{> 0}$ be the **scale** line with coordinate $L$; write $u = \log L \in \mathbb{R}$.

-   Let $Y = \mathbb{R}_{> 0}$ be the **clocked time** line with coordinate $T$; write $v = \log T \in \mathbb{R}$.

We work on the manifold $M = X \times S$ with coordinates $(x,u)$. A **bin** is a path-connected open set $E \subset M$ on which "environment is fixed enough" in the RTM sense (no regime breaks). On $E$, assume a locally integrable **coherence field** $\alpha:E \rightarrow \mathbb{R}$.

**3.2 The RTM 1-form and clock (gauge) transformations**

**Definition 3.1 (RTM 1-form).**\
On $E \subset M$, define

$$\omega\text{\:\,} = \text{\:\,}dv\text{\:\,} - \text{\:\,}\alpha(x,u)\text{ }du.
$$

Here $\alpha$ may depend on $x$and (optionally) on $u$ if we allow slowly varying exponents; constant-$\alpha$ is the ideal RTM case.

**Clock (gauge) transformations.**\
A **clock change** multiplies raw time by a positive factor independent of $L$ inside the bin:

$$v \mapsto v^{\#}\text{\:\,} = \text{\:\,}v + \phi(x),\phi:X \rightarrow \mathbb{R.}
$$

Under this,

$$\omega \mapsto \omega^{\#}\text{\:\,} = \text{\:\,}d(v + \phi(x)) - \alpha\text{ }du\text{\:\,} = \text{\:\,}\omega + d\phi(x).
$$

Thus $\omega$ is defined **up to addition of exact 1-forms pulled back from** $X$---a standard gauge freedom.

**Proposition 3.2 (Slope is gauge-invariant).**\
Clock changes $v \mapsto v + \phi(x)$ do not alter the $\alpha$-coefficient of $du$. Hence $\alpha$ is a gauge-invariant object, while $v$ and $\omega$ shift by exact forms.

*Proof.* Immediate from the transformation rule.

**3.3 Collapse as exactness**

RTM's **collapse** states that, within a bin, after removing $\alpha\text{ }u$ the remaining variation of $v$ is constant (up to noise), i.e. residuals do not trend with $u$.

**Theorem 3.3 (Collapse ⇔ exactness).**\
Let $E \subset M$ be simply connected. The following are equivalent:

1.  (*Power-law chart*) There exists a function $\kappa:E \rightarrow \mathbb{R}_{> 0}$ such that

$$v(x,u)\text{\:\,} = \text{\:\,}\alpha(x)\text{ }u\text{\:\,} + \text{\:\,}\log\kappa(x).
$$

(Constant-$\alpha$ case; for variable $\alpha(x)$ replace $\alpha(x)\text{ }u$ by $\int_{}^{u}{\alpha(x,s)\, ds.}$)

2.  (*Collapse*) For some $\alpha$ as above, the **residual** $\widetilde{v}: = v - \alpha u$ is independent of $u$ on $E$ (i.e., a function of $x$ only).

3.  (*Exactness*) The 1-form $\omega = dv - \alpha\text{ }du$ is **exact** on $E$: $\omega = d\psi$ for some scalar potential $\psi(x)$ (no $u$-dependence).

*Proof.* (1) ⇒ (2) is immediate: $\widetilde{v} = \log\kappa(x)$. (2) ⇒ (3): if $\widetilde{v} = \psi(x)$, then $d\widetilde{v} = dv - \alpha\text{ }du = d\psi(x)$. (3) ⇒ (1): exactness and simple connectivity imply $\widetilde{v} = \psi(x) + C$, hence $v = \alpha u + \log\kappa(x)$.

**Corollary 3.4 (Flatness test).**\
On simply connected $E$, collapse holds iff $d\omega = 0$. In local coordinates,

$$d\omega\text{\:\,} = \text{\:\,} - \text{ }d\alpha \land du.
$$

Thus a **necessary and sufficient** condition for collapse is that $\partial\alpha/\partial u = 0$ and that any $x$-dependence of $\alpha$ does not create holonomy around loops with $u$-extent. For constant $\alpha$, $d\omega = 0$ automatically.

*Remark.* If $\alpha = \alpha(x)$ only, $d\omega = - (\partial\alpha/\partial x)\text{ }dx \land du$. Flatness then requires that along any loop in $E$ with nonzero $u$-extent, the $x$-variation integrates to zero---equivalently, that the field be **path-independent** after gauge fixing. In practice we work on small bins where $\alpha$ is approximately constant, so $d\omega \approx 0$.

**3.4 Holonomy, regime mixing, and why collapse can (and should) fail**

**Definition 3.5 (Holonomy of the RTM connection).**\
Given a closed loop $\gamma \subset E$, define the holonomy

$$\mathcal{H(}\gamma)\text{\:\,} = \text{\:\,}\oint_{\gamma}^{}\omega\text{\:\,} = \text{\:\,}\oint_{\gamma}^{}{(dv - \alpha\text{ }du)}.
$$

-   If $\mathcal{H(}\gamma) = 0$ for all loops (i.e., $d\omega = 0$), residuals are path-independent and collapse can succeed.

-   If $\mathcal{H(}\gamma) \neq 0$ for some loop, the bin contains **incompatible regimes** or genuine curvature (non-power behavior): collapse must fail.

**Proposition 3.6 (Mixtures and curvature induce holonomy).**\
Suppose $E$ contains subregions with different exponents $\alpha_{1} \neq \alpha_{2}$ across a seam in $u$ or $x$. Any loop encircling the seam yields $\mathcal{H(}\gamma) \neq 0$. Hence collapse across the whole $E$ is impossible; the set must be rebinned.

*Sketch.* Integrate $\omega$ piecewise; the jump in $\alpha$ contributes a nonzero integral of $(\alpha_{2} - \alpha_{1})\text{ }du$.

**3.5 Variable-exponent case and adiabatic collapse**

Empirically, $\alpha$ may drift slowly with $u$ or $x$. Then exact collapse cannot hold globally, but **adiabatic collapse** can hold on short windows.

**Proposition 3.7 (Adiabatic approximation).**\
If $\alpha(x,u)$ is $C^{1}$ and $\parallel \partial\alpha/\partial u \parallel \leq \varepsilon$ on $E$, then over any $u$-window of width $h$,

$$\widetilde{v}(x,u)\text{\:\,} = \text{\:\,}v - \alpha(u_{0},x)\text{ }u\text{\:\,} = \text{\:\,}\log\kappa(x)\text{\:\,} + \text{\:\,}O(\varepsilon h),
$$

uniformly for $u \in \lbrack u_{0} - h/2,u_{0} + h/2\rbrack$. Consequently, the empirical collapse statistic $R^{2}(\widetilde{v} \sim u)$ is $O(\varepsilon^{2}h^{2})$.

*Sketch.* First-order Taylor expansion of $\alpha(u)$ around $u_{0}$; bound the residual trend.

*Interpretation.* This justifies RTM's **binning doctrine**: make windows small enough that curvature is negligible; collapse then tests approximate flatness.

**3.6 Identifiability under gauge (global view)**

**Proposition 3.8 (Gauge equivalence class).**\
Two time fields $v_{1},v_{2}$ on $E$define the same $\alpha$ iff their RTM 1-forms differ by an exact pullback from $X$:

$$dv_{2} - \alpha\text{ }du\text{\:\,} = \text{\:\,}dv_{1} - \alpha\text{ }du + d\phi(x).
$$

Equivalently, $v_{2} = v_{1} + \phi(x)$. Thus the **moduli** of RTM structures on $E$ is the quotient

$$\mathcal{M(}E)\text{\:\,} \cong \text{\:\,}\{(\alpha,v)\}/\{ v \sim v + \phi(x)\}.
$$

Slope $\alpha$ classifies the orbit; clocks live in the gauge fiber.

*Consequence.* Any empirical procedure that estimates $\alpha$ from slopes in $u$ is automatically gauge-invariant; procedures that use levels in $v$ are not.

**3.7 Practical diagnostics in geometric language**

-   **Collapse statistic** $\Delta_{\text{collapse}} = R^{2}(\widetilde{v} \sim u)$ is a **curvature proxy**; large values indicate $d\omega \neq 0$ or regime mixing.

-   **Clock placebos** (changing time units) implement $v \mapsto v + \text{const}$: they should not change $\alpha$ or $\Delta_{\text{collapse}}$.

-   **Rebinning** corresponds to restricting to subdomains where $d\omega \approx 0$.

-   **Ledger of intercepts** is a record of chosen gauges $\phi(x)$ across datasets.

**3.8 Summary**

-   RTM is naturally expressed via the 1-form $\omega = d\ \log T - \alpha\text{ }d\ \log L$.

-   **Clock changes** are gauge transformations $\omega \mapsto \omega + d\phi(x)$; **slope** $\alpha$ is gauge-invariant.

-   **Collapse ⇔ exactness/flatness** of $\omega$ on a bin; holonomy/curvature explains when collapse must fail.

-   **Adiabatic collapse** holds on small windows when $\alpha$ varies slowly, quantifying finite-window bias.

**4. Variable-Exponent Regular Variation (Analysis)**

Section 2 derived exact power laws from scale symmetry. Empirically, RTM often holds **locally** while exponents drift **slowly** across environments or across the scale axis. This section places RTM within **regular variation** with **spatially varying** indices, gives representation theorems, and quantifies finite-window bias and the linkage between **collapse statistics** and **curvature**.

Throughout, write $x \in X$ for environment, $L > 0$ for scale, $u = \log L$, and $v(x,u) = \log T(x,L)$.

**4.1 Classical regular variation (recap)**

A measurable $T:\mathbb{R}_{> 0} \rightarrow \mathbb{R}_{> 0}$ is **regularly varying** of index $\alpha \in \mathbb{R}$if

$$\underset{L \rightarrow \infty}{\lim}\frac{T(bL)}{T(L)} = b^{\alpha}\forall b > 0.
$$

Then (Karamata--de Haan) there exists a **slowly varying** $\mathcal{l}$ such that $T(L) = L^{\alpha}\mathcal{l}(L)$, with $\mathcal{l}(bL)\mathcal{/l}(L) \rightarrow 1$ for each fixed $b$. On log--log scales,

$$v(u) = \alpha u + \log\mathcal{l}(e^{u}),{\ \ \ \ \ \ \ \partial}_{u}v(u) = \alpha + o(1).
$$

Hence local slopes converge to $\alpha$ as $u \rightarrow \infty$.

**Potter bounds.** For every $\epsilon > 0$, there exists $U$ such that for $u \geq U$,

$$\mid \log\mathcal{l}(e^{u + h}) - \log\mathcal{l}(e^{u}) \mid \leq \epsilon \mid h \mid + o(1),\ \ h\text{ bounded,}
$$

which will control bias on finite windows.

**4.2 RTM with variable exponent** $\mathbf{\alpha}\mathbf{(}\mathbf{x}\mathbf{)}$

We now allow the exponent to vary with environment $x$(and later slowly with $u$).

**Definition 4.1 (Pointwise regular variation in** $x$**).**\
$T( \cdot ;x)$ is **regularly varying at** $\infty$ with index $\alpha(x)$ if for each fixed $x$and $b > 0$,

$$\underset{L \rightarrow \infty}{\lim}\frac{T(bL;x)}{T(L;x)} = b^{\alpha(x)}.
$$

Equivalently,

$$T(L;x) = L^{\alpha(x)}\text{ }\mathcal{l}(L;x),
$$

where $\mathcal{l}( \cdot ;x)$ is slowly varying **uniformly on compact sets of** $x$(UCS): for each compact $K \subset X$ and $b > 0$,

$$\underset{x \in K}{\sup} \mid \frac{\mathcal{l}(bL;x)}{\mathcal{l}(L;x)} - 1 \mid \underset{\phantom{L \rightarrow \infty}}{\longrightarrow}0.
$$

**Proposition 4.2 (Uniform local slope).**\
Under UCS slow variation,

$$\partial_{u}v(x,u) = \alpha(x) + r(x,u),\sup_{x \in K} \mid r(x,u) \mid \rightarrow 0(u \rightarrow \infty)
$$

for each compact $K \subset X$. Thus in large-scale bins, **binwise slopes** converge uniformly to $\alpha(x)$.

*Sketch.* Take logs, differentiate in $u$; the UCS property gives uniform smallness of the increment of $\log\mathcal{l}$.

**4.3 Drift in** $\mathbf{\alpha}$**across scale:** $\mathbf{\alpha}\mathbf{(}\mathbf{x}\mathbf{,}\mathbf{u}\mathbf{)}$

Empirically, exponents can **drift slowly with** $u$ (finite-range phenomena, evolving regimes). Model

$$v(x,u) = \int_{u_{0}}^{u}{\alpha(x,s)\text{ }ds + \log\kappa(x,u),}
$$

with $\kappa$ slowly varying in the sense that for bounded $h$,

$$\sup_{x \in K} \mid \log\kappa(x,u + h) - \log\kappa(x,u) \mid \leq \epsilon \mid h \mid + o(1),$$

$$
$$

and assume **adiabaticity**:

$$\sup_{x \in K} \mid \partial_{u}\alpha(x,u) \mid \leq \varepsilon\ \ \ \text{(small)}.
$$

**Theorem 4.3 (Adiabatic representation and bias bound).**\
Let $\widehat{\alpha}(x;u,h)$ be any **symmetric local slope** estimator on the window $\lbrack u - h/2,\text{ }u + h/2\rbrack$(e.g., ODR/TLS/Theil--Sen). Under the slow-drift and slow-variation conditions,

$$\widehat{\alpha}(x;u,h)\text{\:\,} = \text{\:\,}\alpha(x,u)\text{\:\,} + \text{\:\,}O\text{ }(\varepsilon h)\text{\:\,} + \text{\:\,}O\text{ }(\epsilon).
$$

Hence finite-window bias is linear in **curvature** $\partial_{u}\alpha$ and bounded by the slow variation of $\kappa$.

*Sketch.* Taylor expand $v(x,u + s)$ to first order in $s$with remainder $\frac{1}{2}(\partial_{u}\alpha)\text{ }s^{2}$; symmetric windows cancel odd terms; Potter bounds handle $\kappa$.

**Corollary 4.4 (Collapse statistic under slow drift).**\
Let $\widetilde{v}(x,u) = v(x,u) - \widehat{\alpha}(x;u,h)\text{ }u$ within the window. Then

$$R^{2}\text{ }(\widetilde{v} \sim u)\text{\:\,} = \text{\:\,}O\text{ }((\varepsilon h)^{2}) + O(\epsilon^{2}),
$$

i.e., the **collapse failure** scales quadratically with window width and the curvature of $\alpha$.

**4.4 Specification test: curvature vs. power law**

Suppose the true relation is **non-power** with twice differentiable $g$:

$$v(u) = g(u),g^{''}(u) \equiv \not{}0.
$$

Let $\widehat{\alpha}(u,h)$ be the local least-squares slope on $\lbrack u - h/2,\text{ }u + h/2\rbrack$.

**Lemma 4.5 (Local linearization error).**

$$\sup_{\mid s \mid \leq h/2} \mid g(u + s) - (g(u) + \widehat{\alpha}(u,h)\text{ }s) \mid \text{\:\,} \geq \text{\:\,}c\text{ } \mid g^{''}(u) \mid \text{ }h^{2}\ 
$$

for some universal constant $c > 0$. Consequently,

$$R^{2}(\widetilde{v} \sim u)\text{\:\,} \geq \text{\:\,}c^{'}\text{ } \mid g^{''}(u) \mid^{2}\text{ }h^{2} + o(h^{2}),
$$

so persistent curvature forces **non-vanishing collapse statistic** as $h \rightarrow 0$ only linearly, giving a practical **specification test** against power laws.

*Sketch.* Chebyshev alternation / Taylor remainder bounds; the regression residual variance lower-bounded by curvature energy.

**4.5 Multi-regime mixtures and identifiability**

Let $v(u) = \alpha_{1}u + c_{1}$ on $\lbrack u_{-},u^{\star}\rbrack$ and $v(u) = \alpha_{2}u + c_{2}$ on $\lbrack u^{\star},u_{+}\rbrack$ with $\alpha_{1} \neq \alpha_{2}$.

**Proposition 4.6 (Unavoidable holonomy / collapse failure).**\
Any single-window slope over $\lbrack u_{-},u_{+}\rbrack$ exhibits residual trend with magnitude $\Omega( \mid \alpha_{2} - \alpha_{1} \mid \text{ } \mid u_{+} - u_{-} \mid )$. Thus **rebins** that respect regime boundaries are necessary; otherwise the geometry of Sec. 3 yields nonzero holonomy.

*Sketch.* Piecewise linear exactness implies kink at $u^{\star}$; any single affine fit leaves systematic sign-changing residuals.

**4.6 Errors-in-variables under regular variation**

Let $x = \log L$, $y = v(x) = \alpha(x_{0})x + c + r(x)$ in a fixed bin around $x_{0}$, with $\mid r^{'}(x) \mid \leq \varepsilon$(small curvature). Observations satisfy

$$x^{obs} = x + \xi,y^{obs} = y + \zeta,
$$

with mean-zero errors, $\xi$ independent of $x$, and $\zeta$ independent noise.

**Theorem 4.7 (Consistency of ODR/SIMEX with slow drift).**\
If $\mathbb{E}\xi^{2} < \infty$, $\mathbb{E}\zeta^{2} < \infty$, and curvature $\varepsilon \rightarrow 0$ with window width, then:

-   **ODR/TLS** slope ${\widehat{\alpha}}_{ODR} \rightarrow \alpha(x_{0})$ in probability as $n \rightarrow \infty$, window $h \rightarrow 0$ with $nh \rightarrow \infty$.

-   **SIMEX** is consistent provided an accurate estimate of $Var(\xi)$ is available; the SIMEX extrapolation error is $o(1)$ under the same regime.

*Sketch.* Standard EIV asymptotics on local linear models plus bias control from Theorem 4.3.

**4.7 Putting it together (operational rules)**

1.  **Binning doctrine.** Choose windows small enough that $\partial_{u}\alpha$ is negligible: bias $= O(\varepsilon h)$.

2.  **Collapse as curvature proxy.** Use $R^{2}(\widetilde{v} \sim u)$to detect curvature $g^{''}$or regime mixtures; thresholds scale like $h^{2}$.

3.  **Clock robustness.** Slowly varying clocks $\kappa$alter intercepts, not slopes; Potter bounds control their contribution.

4.  **EIV aware.** Use ODR/SIMEX/Theil--Sen; ensure $nh \rightarrow \infty$ for consistency while $h \rightarrow 0$ for bias control.

5.  **Mixture detection.** Kinks (piecewise $\alpha$) imply nonzero holonomy (Sec. 3) and force rebinning.

**4.8 Summary**

-   RTM fits naturally inside **regular variation**: exact power when $\alpha$ is constant; **locally power-like** when $\alpha$ drifts slowly.

-   Finite-window **slope bias** is $O(\partial_{u}\alpha \cdot h)$; collapse failures scale like $O((\partial_{u}\alpha)^{2}h^{2})$.

-   **Curvature** or **mixtures** produce persistent collapse residuals; this yields a principled **specification test** and justifies binning.

-   With measurement error, **ODR/SIMEX** remain consistent for the local exponent under standard regimes.

**5. Renormalization on Scales: Fixed Points and Stability**

RTM's power law emerges from scale symmetry. We now recast this as a **renormalization** problem on a function space: rescale the argument $L \mapsto bL$ and re-gauge the clock so the result can be compared at the original scale. **Power laws are fixed points** of this operator; under mild contractivity, flows approach the power-law manifold, giving a dynamical justification for RTM.

**5.1 Function spaces and the renormalization operator**

Let $\mathcal{F}$ be a class of positive functions $T:\mathbb{R}_{> 0} \rightarrow \mathbb{R}_{> 0}$ with $\log T \in C_{\text{loc}}^{1}$. Fix $b > 1$ (a dilation). For a given **gauge choice** $f(b) > 0$ (the clock factor), define:

$$(\mathcal{R}_{b}T)(L)\text{\:\,}: = \text{\:\,}\frac{T(bL)}{f(b)}.
$$

Typical gauges:

-   **Exact-slope gauge** when $\alpha$ is known: $f(b) = b^{\alpha}$.

-   **Self-normalizing gauge**: $f(b) = T(bL_{0})/T(L_{0})$ for a reference $L_{0} > 0$ (so $(\mathcal{R}_{b}T)(L_{0}) = T(L_{0})$).

-   **Moment gauge**: choose $f(b)$ so a chosen functional $\Phi\lbrack T\rbrack$ is invariant (e.g., $\Phi\lbrack T\rbrack = \int w(L)\log T(L)dL$).

We will work primarily with the **self-normalizing gauge**; results carry to other gauges by equivalence (Remark 5.2).

**Metric.** On compact $I \subset (0,\infty)$, use

$$d_{I}(T_{1},T_{2})\text{\:\,} = \text{\:\,}\sup_{L \in I} \mid \log T_{1}(L) - \log T_{2}(L) \mid .
$$

On $\mathcal{F}$, consider the projective family $d = \sum_{k = 1}^{\infty}{2^{- k}d_{I_{k}}}$ for nested compacts $I_{k} = \lbrack e^{- k},e^{k}\rbrack$.

**5.2 Fixed points are power laws**

**Proposition 5.1 (Fixed points).**\
Let $f(b) = b^{\alpha}$ for some $\alpha \in \mathbb{R}$. Then $T$ is a fixed point of $\mathcal{R}_{b}$ for all $b > 1$,

$$\mathcal{R}_{b}T = T\forall b > 1,
$$

iff $T(L) = \kappa L^{\alpha}$ for some $\kappa > 0$.

*Proof.* If $T(L) = \kappa L^{\alpha}$, then $T(bL)/b^{\alpha} = \kappa(bL)^{\alpha}/b^{\alpha} = \kappa L^{\alpha} = T(L)$. Conversely, assume $\mathcal{R}_{b}T = T$ for all $b$. Then $T(bL) = b^{\alpha}T(L)$; by Theorem 2.5, $T(L) = \kappa L^{\alpha}$.

**Remark 5.2 (Gauge equivalence).**\
With the self-normalizing gauge $f(b) = T(bL_{0})/T(L_{0})$, fixed points satisfy $T(bL)/T(bL_{0}) = T(L)/T(L_{0})$, i.e.,

$$\frac{T(bL)}{T(L)} = \frac{T(bL_{0})}{T(L_{0})} = b^{\alpha},
$$

so fixed points are again precisely power laws. Thus **fixed points are gauge-invariant up to clock**.

**5.3 Linearization and stability near a power law**

Let $T^{\star}(L) = \kappa L^{\alpha}\ $be a fixed point (for gauge $f(b) = b^{\alpha}$). Write perturbations in log-space:

$$\log T(L) = \log T^{\star}(L) + \varepsilon(L),\varepsilon:\mathbb{R}_{> 0} \rightarrow \mathbb{R.}
$$

Then

$$\log(\mathcal{R}_{b}T)(L) = \log T^{\star}(L) + \varepsilon(bL) - \log f(b) + \log\kappa + \alpha\log L - (\log\kappa + \alpha\log L).
$$

Hence, for the exact-slope gauge ($f(b) = b^{\alpha}$):

$$\varepsilon\text{\:\,} \mapsto \text{\:\,}\mathcal{L}_{b}\varepsilon\ \ \ \text{with}\text{   }(\mathcal{L}_{b}\varepsilon)(L) = \varepsilon(bL).
$$

Thus the linearized renormalization acts by **composition with dilation**.

**Lemma 5.3 (Contraction on Hölder/Zygmund classes).**\
Let $\mathcal{C}^{0,\beta}$ be Hölder functions of $u = \log L$ with seminorm $\lbrack\varepsilon\rbrack_{\beta} = {\sup}_{u \neq v}\frac{\mid \varepsilon(e^{u}) - \varepsilon(e^{v}) \mid}{\mid u - v \mid^{\beta}}$. Then for any $b > 1$,

$$\lbrack\mathcal{L}_{b}\varepsilon\rbrack_{\beta} = b^{- \beta}\lbrack\varepsilon\rbrack_{\beta}.
$$

If we use the norm $\parallel \varepsilon \parallel_{C^{0,\beta}(I)} = {\sup}_{I} \mid \varepsilon \mid + diam(I)^{\beta}\lbrack\varepsilon\rbrack_{\beta}$ on a compact interval $I$ stable under $u \mapsto u + \log b$, the operator is a **strict contraction** with factor $b^{- \beta} < 1$.

*Proof.* In $u$-coordinates, $(\mathcal{L}_{b}\varepsilon)(e^{u}) = \varepsilon(e^{u + \log b})$; differences shrink by $b^{- \beta}$.

**Theorem 5.4 (Local stability of power laws).**\
Fix a compact $I \subset \mathbb{R}$(in $u = \log L$), and let the gauge be $f(b) = b^{\alpha}$. If $\varepsilon \in C^{0,\beta}$on $I^{'} = \{ u + \log b^{n}:\text{ }u \in I,\text{ }n = 0,1,2,\ldots\text{ }\}$ with small norm, then the iterates satisfy

$$\parallel \varepsilon_{n} \parallel_{C^{0,\beta}(I)}\text{\:\,} \leq \text{\:\,}b^{- n\beta}\text{ } \parallel \varepsilon_{0} \parallel_{C^{0,\beta}(I^{'})}\text{\:\,}\underset{\phantom{n \rightarrow \infty}}{\longrightarrow}\text{\:\,}0,
$$

i.e., $\mathcal{R}_{b}^{n}T \rightarrow T^{\star}$ uniformly on $I$ in log-space. Hence **power laws are locally attractive** in Hölder/Zygmund topologies.

*Interpretation.* Small Hölder perturbations are **damped** by repeated rescaling (they get "shifted right" in $u$ and smoothed). This is the dynamical counterpart of **regular variation**.

**5.4 Slowly varying clocks and center manifolds**

Let $T(L) = L^{\alpha}\kappa(L)$ with **slowly varying** $\kappa$. In log-space: $\varepsilon(u) = \log\kappa(e^{u})$with $\varepsilon(u + h) - \varepsilon(u) \rightarrow 0$as $u \rightarrow \infty$.

**Proposition 5.5 (Center manifold of slowly varying factors).**\
Under the self-normalizing gauge $f(b) = T(bL_{0})/T(L_{0})$, the renormalization dynamics on $\varepsilon$ is

$$(\mathcal{L}_{b}\varepsilon)(u) = \varepsilon(u + \log b) - \varepsilon(u_{0} + \log b) + \varepsilon(u_{0}),
$$

which preserves the "anchor" $\varepsilon(u_{0})$ and shifts **differences** along $u$. If $\varepsilon$ is slowly varying, then for any compact $I$,

$$\sup_{u \in I} \mid (\mathcal{L}_{b}^{n}\varepsilon)(u) - \varepsilon(u_{0}) \mid \underset{\phantom{n \rightarrow \infty}}{\longrightarrow}0.
$$

Thus $L^{\alpha}\kappa(L)$ flows to the **power-law leaf** determined by the chosen gauge; the slowly varying factor sits on a **center manifold** (neutral direction) that is quotiented out by self-normalization.

*Sketch.* Telescoping sums of slow increments; uniform convergence on compacts follows from Potter bounds.

**5.5 Variable exponents and adiabatic attraction**

Let $T(L) = \exp(\int_{u_{0}}^{\log L}{\alpha(s)\text{ }ds)\text{ }\kappa(L)}$ with $\alpha C^{1}$ and $\mid \alpha^{'}(u) \mid \leq \varepsilon$ small on a band encompassing $I^{'} = \cup_{n \geq 0}^{}{(I + n\ \log b).}$

**Theorem 5.6 (Adiabatic stability to a drifting power law).**\
Under the exact-slope gauge $f(b) = b^{\alpha(u_{0})}$ or the self-normalizing gauge, the iterates satisfy on any compact $I$:

$$\sup_{u \in I} \mid \log(\mathcal{R}_{b}^{n}T)(e^{u}) - (\alpha(u_{0})\text{ }u + C_{n}) \mid \text{\:\,} \leq \text{\:\,}C\text{ }\varepsilon\text{ }n\ \log b\text{\:\,} + \text{\:\,}o(1),
$$

where $C_{n}$ is a (gauge-dependent) constant. For fixed $I$, as $n$ grows the right-hand side remains **small** provided the cumulative drift $\varepsilon\text{ }n\ \log b$is small---this is the **adiabatic regime**. Hence on finite windows the flow **tracks** a local power law with exponent near $\alpha(u_{0})$.

*Sketch.* Decompose $\int_{u_{0}}^{u + n\log b}{\alpha(s)ds = \alpha(u_{0})(u + n\log b - u_{0}) + \int\alpha^{'}(s)(u + n\log b - s)ds}$. The remainder scales with $\varepsilon n\ \log b$; slow variation of $\kappa$ handled as in 5.5.

*Interpretation.* If $\alpha$ drifts slowly, renormalization still pushes toward **local power-law behavior** on any fixed window---precisely the empirical RTM setting.

**5.6 Non-power alternatives: curvature generates unstable modes**

Consider $v(u) = g(u)$ with curvature $g^{''} \equiv \not{}0$. In perturbation coordinates relative to $\alpha$, set $\varepsilon(u) = g(u) - (\alpha u + c)$. Then under $\mathcal{L}_{b}$,

$$\varepsilon(u) \mapsto \varepsilon(u + \log b).
$$

If $g^{''}$ is persistent (e.g., periodic or polynomial), the shifted residuals **do not decay** on fixed windows; they only "translate." The contraction of Lemma 5.3 fails in $C^{0}$ unless we quotient out by drift and curvature. Consequently:

**Proposition 5.7 (Curvature as a non-decaying mode).**\
If $g^{''}$ does not vanish at infinity (or decays too slowly), then for any gauge, there exists a compact window $I$ and $\delta > 0$ such that

$$\inf_{n \geq 0}\ \sup_{u \in I} \mid \varepsilon_{n}(u) \mid \text{\:\,} \geq \text{\:\,}\delta,\ \ 
$$

i.e., **renormalization does not contract** to a power law on that window. This aligns with **collapse failure** (Sec. 4.4).

*Conclusion.* Persistent curvature is an **unstable feature** under RG flow---precisely what our collapse test detects.

**5.7 Summary and implications**

-   The renormalization operator $\mathcal{R}_{b}$ formalizes **"zooming out in scale and re-gauging the clock"**.

-   **Power laws are fixed points**, independent of gauge (up to a multiplicative constant).

-   In Hölder/Zygmund topologies, $\mathcal{R}_{b}$ is a **contraction**, giving **local attraction** to the power-law manifold.

-   **Slowly varying clocks** lie on a **center manifold** and are neutralized by self-normalization; flows converge to a representative power law.

-   **Slowly drifting exponents** yield **adiabatic attraction**: on any fixed window, iterates track a local power law with small, controlled error.

-   **Curvature** in $g = \log T$ is a **non-decaying mode**; under RG it persists as translation, exactly mirroring **collapse failure**.

**6. Scale-Dependent Diffusions and Dirichlet Forms**

This section shows how an RTM exponent acts as a **local clock field** in stochastic/PDE dynamics. We construct diffusions on metric spaces whose *effective time* stretches with scale, derive self-similar laws when $\alpha$ is constant, and prove **adiabatic** approximations when $\alpha$ varies slowly. This links RTM to sub/super-diffusion *without* committing a priori to fractional operators.

**6.1 Metric--measure setting and RTM conductivity**

Let $(M,d,\mu)$ be a complete, separable metric measure space with a regular, strongly local Dirichlet form $\mathcal{(E,}\mathcal{D})$ on $L^{2}(\mu)$ and carré-du-champ $\Gamma$. For intuition, $M = \mathbb{R}^{m}$ with $\Gamma(u) = \mid \nabla u \mid^{2}$.

Let $L:M \rightarrow (0,\infty)$ be a **scale proxy** (e.g., local neighborhood radius, degree, or coarse density) and $\alpha:M \rightarrow \mathbb{R}$a **coherence field**. Define an **RTM conductivity**

$$\mathsf{D}(x)\text{\:\,} = \text{\:\,}L(x)^{- \alpha(x)}\ \ \ \ \ \ \ (\text{slower clocks at larger scale if }\alpha > 0).
$$

**Definition 6.1 (RTM Dirichlet form).**

$$\mathcal{E}_{\alpha}(u,v)\text{\:\,} = \text{\:\,}\int_{M}^{}{\mathsf{D}(x)\text{ }\Gamma(u,v)(x)\text{ }d\mu(x),\mathcal{D}_{\alpha} = \mathcal{D}.}
$$

This is closed, symmetric, and generates a conservative Markov semigroup $(P_{t}^{\alpha})_{t \geq 0}$ with generator

$$\mathcal{L}_{\alpha}u\text{\:\,} = \text{\:\,}\nabla \cdot (\mathsf{D}\text{ }\nabla u)(\text{in }\mathbb{R}^{m}\text{ case}).$$

**6.2 Constant exponent: self-similar scaling**

Assume $\alpha(x) \equiv \alpha$ and $L(x) = \lambda\text{ } \mid x \mid_{R}$ for some homogeneous quasi-norm $\mid \cdot \mid_{R}$ of degree 1 under a dilation group $x \mapsto b\text{ }x$.

**Theorem 6.2 (RTM similarity).**\
Let $u(t,x)$ solve $\partial_{t}u = \mathcal{L}_{\alpha}u$ with integrable initial data. Then for any $b > 0$,

$$u(t,x)\text{\:\,} = \text{\:\,}b^{m}\text{ }u\text{ }(b^{m + \alpha}\text{ }t,\text{\:\,}b\text{ }x)
$$

in the sense of distributions. In particular, the heat kernel $p^{\alpha}(t,x,y)$ obeys

$$p^{\alpha}(t,x,y)\text{\:\,} = \text{\:\,}t^{- \frac{m}{m + \alpha}}\text{ }\Phi\text{ }(\frac{d(x,y)}{t^{1/(m + \alpha)}})
$$

for some profile $\Phi$ (Gaussian-like tails when $M = \mathbb{R}^{m}$). Thus the **diffusion radius** scales as

$$r(t)\text{\:\,} \asymp \text{\:\,}t^{1/(m + \alpha)}\ \ \  \Longleftrightarrow \ \ \ t\text{\:\,} \asymp \text{\:\,}r^{\text{ }m + \alpha}.
$$

*Interpretation.* The **effective dynamic exponent** is $z = m + \alpha$: time grows with scale as $T \sim L^{\text{ }z}$. When $m$is fixed, varying $\alpha$ changes the **clock gradient** across scale.

*Sketch.* Invariance of $\mathcal{E}_{\alpha}$ under $x \mapsto bx$, $t \mapsto b^{m + \alpha}t$, and mass conservation give the scaling law.

**6.3 Slowly varying exponent: adiabatic clocks**

Let $\alpha \in C^{1}(M)$ and $L \in C^{1}(M)$. Consider the inhomogeneous PDE

$$\partial_{t}u\text{\:\,} = \text{\:\,}\nabla \cdot (L(x)^{- \alpha(x)}\nabla u).
$$

**Assumption (adiabatic drift).** There exists $\varepsilon \ll 1$ and a covering of $M$ by patches $U_{k}$ of diameter $h$ such that

$$\sup_{x \in U_{k}} \parallel \nabla\alpha(x) \parallel \leq \varepsilon,\sup_{x \in U_{k}} \parallel \nabla logL(x) \parallel \leq \varepsilon.\ 
$$

**Theorem 6.3 (Local self-similarity, adiabatic error).**\
Fix a patch $U$ and a reference point $x_{0} \in U$. Let $\alpha_{0} = \alpha(x_{0})$, $L_{0} = L(x_{0})$. For times $t$ such that the diffusion radius $r(t) \ll h$,

$$u(t,x)\text{\:\,} = \text{\:\,}(P_{t}^{\alpha_{0}}\text{ }u_{0})(x)\text{\:\,} + \text{\:\,}\mathcal{O}(\varepsilon\text{ }t\text{ }r(t)^{- 2})\ \ \ \ \ \ \ \text{uniformly for }x \in U,
$$

with $r(t) \asymp t^{1/(m + \alpha_{0})}$. Equivalently, on **finite observation windows**, the solution is approximated by a constant-$\alpha_{0}$ model with an **adiabatic error** linear in the local curvature of the clock field.

*Sketch.* Duhamel expansion around the frozen-coefficient operator at $x_{0}$; commutator bounds yield the stated error using local Gaussian estimates.

**Corollary 6.4 (Adiabatic ECI consistency).**\
Local estimates of the **time--scale slope** from solution features (e.g., heat-ball radius vs. time, first-passage times) converge to $\alpha(x_{0})$ with bias $O(\varepsilon)$ as the observation window shrinks, matching the statistical bias bounds of Section 4.

**6.4 First-passage and exit times**

Let $\tau_{R} = \inf\{ t > 0:\text{ }X_{t} \notin B(x_{0},R)\}$ for the diffusion generated by $\mathcal{L}_{\alpha}$.

**Proposition 6.5 (RTM exit-time scaling).**\
If $\alpha \equiv \alpha_{0}$ and $L(x) \propto \mid x - x_{0} \mid$ near $x_{0}$, then

$$\mathbb{E}_{x_{0}}\text{ }\tau_{R}\text{\:\,} \asymp \text{\:\,}R^{\text{ }m + \alpha_{0}}.
$$Under adiabatic drift, for $R \ll h$,

$$\mathbb{E}_{x_{0}}\text{ }\tau_{R}\text{\:\,} = \text{\:\,}c(x_{0})\text{ }R^{\text{ }m + \alpha(x_{0})}(1 + O(\varepsilon R)).
$$

*Interpretation.* Exit-time exponents **directly encode** the local RTM slope. These are operational observables for empirical $\alpha$ in spatial systems.

**6.5 Sub-/super-diffusion without fractional Laplacians**

Classically, anomalous diffusion uses fractional operators $( - \Delta)^{\beta}$. Here, RTM achieves **effective anomaly** via **space-dependent clocks** instead:

-   If $\alpha > 0$, large scales are **slower** (sub-diffusive w.r.t. Euclidean time).

-   If $\alpha < 0$, large scales are **faster** (super-diffusive).

**Proposition 6.6 (Equivalence in self-similar class).**\
In $\mathbb{R}^{m}$ with $\alpha \equiv \alpha_{0}$, the heat kernel has the same similarity exponent as a fractional diffusion with order $\beta = \frac{2}{m + \alpha_{0}}$ in the sense that the mean squared displacement satisfies

$$\mathbb{E} \mid X_{t} \mid^{2}\text{\:\,} \asymp \text{\:\,}t^{2/(m + \alpha_{0})}\text{\:\,} = \text{\:\,}t^{\beta}.
$$

Hence RTM's constant-$\alpha$ family **realizes** sub/super-diffusion exponents via local clock modulation rather than nonlocal jumps.

**6.6 Spectral viewpoint**

Let $\{ - \mathcal{L}_{\alpha}\varphi_{k} = \lambda_{k}\varphi_{k}\}$ be the spectral resolution (on a bounded domain with Dirichlet boundary).

**Theorem 6.7 (Weyl-type law with RTM clock).**\
If $L,\alpha$ are smooth and bounded above/below on $\Omega \subset \mathbb{R}^{m}$,

$$N(\lambda)\text{\:\,}: = \text{\:\,}\#\{ k:\lambda_{k} \leq \lambda\}\text{\:\,} \sim \text{\:\,}C_{m}\int_{\Omega}^{}{(\lambda\text{ }L(x)^{\alpha(x)})^{m/2}\text{ }dx(\lambda \rightarrow \infty).}
$$

Consequently, high-frequency eigenmodes "feel" the **local clock** as a density multiplier $L^{\alpha}$.

*Sketch.* Local Weyl law via semiclassical measure with variable coefficient principal symbol $\mid \xi \mid^{2}L(x)^{- \alpha(x)}$.

**6.7 Stochastic representation and time change**

Let $B_{t}$ be Brownian motion on $M$(for $M = \mathbb{R}^{m}$). Define an **additive functional**

$$A_{t}\text{\:\,} = \text{\:\,}\int_{0}^{t}{\mathsf{D}(B_{s})\text{ }ds =}\int_{0}^{t}{L(B_{s})^{- \alpha(B_{s})}\text{ }ds,}$$

and its right-continuous inverse $T(t) = \inf\{ s:A_{s} > t\}$.

**Proposition 6.8 (Clock-change representation).**\
The diffusion $X_{t} = B_{T(t)}$ has generator $\mathcal{L}_{\alpha}$. Thus RTM diffusions are **time-changed Brownian motions** with a *state-dependent clock*.

*Consequences.* Many properties (martingales, Harnack bounds) lift from Brownian motion through the time change, clarifying when RTM inherits classical regularity.

**6.8 Summary**

-   RTM enters diffusion theory as a **space-dependent clock** $L^{- \alpha}$ multiplying conductivity.

-   **Constant** $\alpha$yields **exact similarity** with dynamic exponent $z = m + \alpha$ and exit times $T \sim R^{\text{ }z}$.

-   **Slowly varying** $\alpha$ admits **adiabatic** approximations; local estimates of the time--scale slope are consistent with controlled bias.

-   RTM provides an alternative path to **anomalous diffusion** and a clean **spectral** interpretation; RTM diffusions are **time-changed Brownian motions**.

**7. Identifiability & Statistical Consistency**

This section formalizes **what is identifiable** from finite, noisy data and gives **consistency** results for common slope estimators used in RTM: **orthogonal distance regression (ODR/TLS)**, **SIMEX**, and **Theil--Sen**. We also cast the **collapse statistic** as a specification test against non-power alternatives with measurement error.

Setup: in a fixed bin (environment), the ideal model is

$$y = \log T = \alpha x + c + r(x),\ \ x = \log L,
$$

where $r \equiv 0$ (exact RTM) or $\mid r^{'}(x) \mid \leq \varepsilon$ (slow drift / curvature). Observations are noisy:

$$x^{obs} = x + \xi,{\ \ \ \ \ \ \ y}^{obs} = y + \zeta,
$$

with $\mathbb{E}\lbrack\xi\rbrack = \mathbb{E}\lbrack\zeta\rbrack = 0$, finite variances, and mild independence/regularity given below.

**7.1 What is (and is not) identifiable**

**Proposition 7.1 (Slope is clock-invariant; intercept is not).**\
If the clock rescales as $y^{\#} = y + \phi$ with $\phi$ constant in $x$ inside the bin, then any slope estimator based on contrasts in $x$ (ODR, SIMEX, Theil--Sen) is unchanged, while the intercept shifts by $\phi$.\
*Implication.* Only $\alpha$ is an intrinsic target; intercepts are gauge (clock) artifacts.

**Proposition 7.2 (Identifiability up to curvature).**\
If $r \equiv 0$, $\alpha$ is point-identified from the joint distribution of $(x^{obs},y^{obs})$ given measurement error structure. If $\mid r^{'} \mid \leq \varepsilon$, then the identified target is the **local slope** $\alpha(u_{0})$up to bias $O(\varepsilon h)$ for window width $h$ (Section 4).

**7.2 Orthogonal Distance Regression (Total Least Squares)**

Assume i.i.d. sample $\{(x_{i}^{obs},y_{i}^{obs})\}_{i = 1}^{n}$, with\
(i) $x_{i}$ supported on a compact interval $\lbrack a,b\rbrack$, density bounded away from 0;\
(ii) $\xi_{i},\zeta_{i}$ independent of $x_{i}$ and of each other, mean 0, finite second moments;\
(iii) $r \equiv 0$ (or $\mid r^{'} \mid \leq \varepsilon$ small on window).

**Theorem 7.3 (Consistency of ODR/TLS).**\
Under (i)--(ii) with $r \equiv 0$, the ODR slope ${\widehat{\alpha}}_{ODR}$ is **consistent** for $\alpha$, and $\sqrt{n}({\widehat{\alpha}}_{ODR} - \alpha)$ is asymptotically normal with variance determined by the second moments of $(x,\xi,\zeta)$. If moreover $\mid r^{'} \mid \leq \varepsilon$ on a window of width $h$, then

$${\widehat{\alpha}}_{ODR}\text{\:\,} = \text{\:\,}\alpha(u_{0})\text{\:\,} + \text{\:\,}O_{p}(\varepsilon h)\text{\:\,} + \text{\:\,}O_{p}(n^{- 1/2}).$$

*Sketch.* Classical TLS asymptotics (eigenvector of centered covariance matrix). Curvature contributes a deterministic bias of order $\varepsilon h$.

**Remark.** OLS is **attenuated** when $\xi \neq 0$; ODR is the correct EIV remedy when the error ratio is not extreme.

**7.3 SIMEX (Simulation--Extrapolation)**

Assume we know or can estimate $\sigma_{\xi}^{2} = Var(\xi)$. Define pseudo-samples

$$x_{i}^{(\lambda)} = x_{i}^{obs} + \sqrt{\lambda}\text{ }{\widetilde{\xi}}_{i},\lambda \in \Lambda \subset \lbrack 0,\Lambda_{\max}\rbrack,
$$with fresh ${\widetilde{\xi}}_{i} \sim N(0,\sigma_{\xi}^{2})$; fit naive slopes $\widehat{\alpha}(\lambda)$ (e.g., OLS or ODR) and extrapolate a low-order polynomial to $\lambda = - 1$ to obtain ${\widehat{\alpha}}_{SIMEX}$.

**Theorem 7.4 (Consistency of SIMEX).**\
If $\sigma_{\xi}^{2}$ is consistently estimated and $r \equiv 0$, then ${\widehat{\alpha}}_{SIMEX}\overset{\phantom{p}}{\rightarrow}\alpha$. With $\mid r^{'} \mid \leq \varepsilon$, in a window $h$,

$${\widehat{\alpha}}_{SIMEX}\text{\:\,} = \text{\:\,}\alpha(u_{0}) + O_{p}(\varepsilon h) + o_{p}(1).
$$

*Sketch.* Standard SIMEX theory: the measurement error bias is a smooth function of $\lambda$; extrapolating to $- 1$ removes it.

**Practical note.** When $\sigma_{\xi}^{2}$ is misspecified, SIMEX may over/under-correct; use as a **sensitivity bound** alongside ODR.

**7.4 Theil--Sen (Robust median slope)**

Define the median of pairwise slopes on $(x^{obs},y^{obs})$. Under symmetric noise and no curvature, Theil--Sen is $\sqrt{n}$**-consistent** and robust to outliers.

**Proposition 7.5 (Robustness envelope).**\
If a fraction $\pi < 0.29$ of observations are arbitrary outliers, Theil--Sen's slope still converges to $\alpha$(breakdown \~29%). With small curvature $\mid r^{'} \mid \leq \varepsilon$, bias is $O(\varepsilon h)$.

*Use.* Report Theil--Sen as a **robust check**; fuse with ODR via meta-analysis to guard against heavy tails.

**7.5 Inference: uncertainty and small-sample practice**

-   **Cluster/bootstrap** by entity or track to capture serial dependence in $x$ and $T$.

-   **Orthogonal residual bootstrap** is appropriate for ODR; **pairs bootstrap** for Theil--Sen.

-   Report **percentile CIs** and **influence diagnostics** (leverage points in $x$).

-   Maintain a **slope--intercept ledger**: slopes with CIs, intercepts (gauge), and known clock/unit shifts.

**7.6 Collapse statistic with measurement error**

Let ${\widetilde{y}}_{i} = y_{i}^{obs} - \widehat{\alpha}\text{ }x_{i}^{obs}$ within a bin, and regress $\widetilde{y}$ on $x^{obs}$. Define

$$\Delta_{\text{collapse}}\text{\:\,}: = \text{\:\,}R^{2}(\widetilde{y} \sim x^{obs}).
$$

**Theorem 7.6 (Specification test under EIV).**\
Assume $r \equiv 0$ (true power law), $\widehat{\alpha}\ $is consistent, and $\xi,\zeta$ are mean-zero with finite variances. Then $\Delta_{\text{collapse}}\overset{\phantom{p}}{\rightarrow}0$.\
If $v = g(x)$ with $g^{''} \neq 0$ on the bin and mild smoothness, then for any consistent slope estimator,

$$\underset{n \rightarrow \infty}{lim\, inf}\Delta_{\text{collapse}}\text{\:\,} \geq \text{\:\,}c\text{ }\mathbb{E}\lbrack g^{''}(X)^{2}\rbrack\text{ }h^{2}\text{(up to error-variance terms)},
$$

so the statistic stays **bounded away from 0** in the limit as long as curvature persists over window width $h$.

*Sketch.* Under the null, residuals are mean-independent of $x$; with curvature, regression captures a non-vanishing linear component proportional to second derivatives (Section 4.4), with EIV inflating variance but not erasing the drift.

**Practice.** Pre-register a collapse threshold (e.g., $< 0.05$); accompany with **residual vs.** $x$ plots (nonparametric smooth) and a **clock-placebo** check.

**7.7 Window selection and changepoints**

-   **Bias--variance trade-off:** Choose window width $h$ so that $nh \rightarrow \infty$(variance ↓) while $h \rightarrow 0$(bias $O(\varepsilon h)$↓).

-   **Changepoints:** Use PELT/Bai--Perron on $(x^{obs},y^{obs})$ pairs or on preliminary residuals to avoid mixing regimes (kinks violate collapse).

-   **Coverage gates:** Reject bins with too few effective $x$-spans (thin leverage → unstable slope).

**7.8 Multi-proxy fusion with uncertainty**

Given family-wise estimates $({\widehat{\alpha}}_{f},{\widehat{\sigma}}_{f}^{2})$ that passed collapse, apply **random-effects** fusion:

$${\widehat{\alpha}}_{RE} = \frac{\sum_{f}^{}{w_{f}{\widehat{\alpha}}_{f}}}{\sum_{f}^{}w_{f}},w_{f} = \frac{1}{{\widehat{\sigma}}_{f}^{2} + {\widehat{\tau}}^{2}},{\widehat{\tau}}^{2} = \max\left\{ \frac{Q - (F - 1)}{\sum w_{f} - \sum w_{f}^{2}/\sum w_{f}},0 \right\}.
$$

Report $Q$, ${\widehat{\tau}}^{2}$, and **leave-one-family-out** influence. High ${\widehat{\tau}}^{2}$⇒ publish family-wise ${\widehat{\alpha}}_{f}$ instead of a single number.

**7.9 Finite-sample red flags (practical diagnostics)**

-   **Attenuation creep:** OLS slope ≪ ODR slope.

-   **Leverage scarcity:** Most leverage from extreme $x$ points; run **jackknife** by removing them.

-   **High** $\Delta_{\text{collapse}}$**:** residual trend vs. $x$→ likely curvature or regime mix.

-   **Clock failure:** Unit/clock change alters slope → rebin; the slope must be **clock-invariant**.

**7.10 Summary**

-   In a bin, $\alpha$ is the only gauge-invariant estimand.

-   **ODR/TLS** and **SIMEX** are consistent for $\alpha$under standard EIV assumptions; **Theil--Sen** is a robust check.

-   Finite-window bias from drift/curvature is $O(\varepsilon h)$; manage with binning and changepoints.

-   The **collapse statistic** is a specification test: it tends to 0 under the RTM model and stays positive with non-power curvature---even with measurement error.

-   Publish **uncertainty, collapse diagnostics, and heterogeneity**; when fusion heterogeneity is high, prefer family-wise slopes over a single index.

**8. A Category-Theoretic Packaging of RTM**

This section formalizes RTM as a **gauge theory on a scale--clock bundle**. The goal is not abstraction for its own sake, but a clean language for invariants (slope), gauges (clocks), and functoriality under changes of variables, coarse-graining, and product constructions.

**8.1 The category RTM**

An object of **RTM** is a triple $\mathsf{A} = (X,L,v)$ where:

-   $X$ is a (second countable) topological space of **environments**;

-   $L:X \rightarrow \mathbb{R}_{> 0}$ is a continuous **scale** map (or a trivial factor with coordinate $u = \log L$);

-   $v:X \rightarrow \mathbb{R}$ is a continuous **clock potential** $v = \log T$.

Associated to $\mathsf{A}$ is the **RTM 1-form**

$$\omega_{\mathsf{A}}\text{\:\,} = \text{\:\,}dv - \alpha\text{ }d(\log L),
$$

for some $\alpha$ (constant or a field on $X$), defined up to **gauge**: $v \sim v + \phi$ with $\phi:X \rightarrow \mathbb{R}$.

A **morphism** $\Phi:\mathsf{A} \rightarrow \mathsf{B}$ is a pair $(\varphi,\psi)$ with $\varphi:X \rightarrow Y$continuous and $\psi:Y \rightarrow \mathbb{R}$ such that

$$\Phi^{\text{*}}\omega_{\mathsf{B}}\text{\:\,} = \text{\:\,}\omega_{\mathsf{A}} + d(\psi \circ \varphi),
$$i.e., $\Phi$ pulls back the target's 1-form to the source's 1-form **up to gauge**. Composition is $(\varphi_{2},\psi_{2}) \circ (\varphi_{1},\psi_{1}) = (\varphi_{2} \circ \varphi_{1},\text{\:\,}\psi_{1} + \psi_{2} \circ \varphi_{1})$.

**Interpretation.** Different clocks correspond to vertical gauge shifts $v \mapsto v + \phi$. Morphisms are **clock-compatible** reparametrizations of environment/scale.

**8.2 Gauge group and moduli**

The **gauge group** $\mathcal{G}_{X} = C^{0}(X,\mathbb{R})$ acts on objects by $v \mapsto v + \phi$. Two objects are **gauge equivalent** if related by this action.

**Proposition 8.1 (Slope as a moduli invariant).**\
If $\omega_{\mathsf{A}}$ and $\omega_{\mathsf{B}}$ are gauge equivalent on $X$, then their $\alpha$ fields agree (a.e.). Conversely, equal $\alpha$ fields define the same class in the **moduli space**

$$\mathfrak{M}(X) = \{\text{objects on }X\}/\mathcal{G}_{X}.
$$Thus $\lbrack\mathsf{A}\rbrack \in \mathfrak{M}(X)$ is uniquely determined by $\alpha$ and the de Rham cohomology class $\lbrack\omega_{\mathsf{A}}\rbrack \in H^{1}(X;\mathbb{R})$; in simply connected bins, $\lbrack\omega\rbrack = 0$ and the class is fully determined by $\alpha$.

*Consequence.* In a bin (simply connected), **slope** is the only intrinsic datum; clocks are pure gauge.

**8.3 Collapse = trivialization of the RTM bundle**

Let $\pi:X \times \mathbb{R} \rightarrow X$ be the trivial line bundle with fiber coordinate $u = \log L$. Consider the connection 1-form

$$\omega = dv - \alpha(x)\text{ }du.
$$

**Theorem 8.2 (Collapse ⇔ flat trivialization).**\
On a simply connected bin $E \subset X \times \mathbb{R}$, the following are equivalent:

1.  There exists a **global section** $s(x) = (x,u)$ and a gauge $\phi$ such that, in the trivialization with potential $v^{\phi} = v + \phi$, $v^{\phi}(x,u) = \alpha(x)u + c(x)$ (RTM chart).

2.  The connection $\omega$ is **flat** on $E$: $d\omega = 0$ and its holonomy vanishes.

3.  The empirical **collapse** of Section 3 holds on $E$.

This repackages Section 3 in categorical language: collapse is existence of a trivialization that straightens $v$ into an affine function of $u$.

**8.4 Products, sums, and coarse-graining (monoidal structure)**

Define a **monoidal product** $\mathsf{A} \otimes \mathsf{B}$ for independent subsystems:

$$(X_{A},L_{A},v_{A}) \otimes (X_{B},L_{B},v_{B})\text{\:\,} = \text{\:\,}(X_{A} \times X_{B},\text{\:\,}L_{A}L_{B},\text{\:\,}v_{A} + v_{B}),
$$

with $\alpha_{\otimes} = \alpha_{A} + \alpha_{B}$ if each obeys a power law.

**Proposition 8.3 (Additivity under independent composition).**\
If both factors are in exact RTM form $v_{i} = \alpha_{i}u_{i} + c_{i}$with $u = \log(L_{A}L_{B}) = u_{A} + u_{B}$, then

$$v_{\otimes}\text{\:\,} = \text{\:\,}(\alpha_{A} + \alpha_{B})\text{ }u\text{\:\,} + \text{\:\,}(c_{A} + c_{B}),
$$

so slopes **add** under multiplicative composition of scales. Gauge transforms distribute.

**Coarse-graining functor.**\
Let $\mathcal{C}_{b}$ map $(X,L,v)$ to $(X,\text{ }bL,\text{ }v - \log f_{b})$. With a gauge choice $f_{b}$ (Section 5), $\mathcal{C}_{b}$ is an **endofunctor** of **RTM**; power-law objects are its **fixed points**.

**8.5 Natural transformations and clock choices**

Two gauge choices $f_{b}$ and $g_{b}$ for coarse-graining define endofunctors $\mathcal{C}_{b}^{(f)}$ and $\mathcal{C}_{b}^{(g)}$. The map

$$\eta_{b}:\mathcal{C}_{b}^{(f)} \Rightarrow \mathcal{C}_{b}^{(g)},\ \ \eta_{b}(\mathsf{A}) = (\text{id}_{X},\psi_{b}),\text{ }{\ \ \ \ \ \ \ \psi}_{b} = \log f_{b} - \log g_{b}$$

is a **natural transformation**, i.e., a functorial gauge shift commuting with morphisms. This encodes the statement "changing the renormalization gauge is just a clock change."

**8.6 Curvature and obstructions (cohomology)**

Let $\Omega^{1}(E)$ be 1-forms on a bin $E \subset X \times \mathbb{R}$. The curvature is

$$\mathcal{F}\text{\:\,} = \text{\:\,}d\omega\text{\:\,} = \text{\:\,} - d\alpha \land du.$$

-   If $\mathcal{F} \neq 0$, collapse is **obstructed**; regime mixing or genuine curvature persists.

-   If $\mathcal{F} = 0$ but $H^{1}(E) \neq 0$, collapse may still fail globally due to **holonomy** (nontrivial cohomology class). Local collapse always holds.

**Proposition 8.4 (Cohomological obstruction).**\
Collapse holds globally on $E$ iff $\mathcal{F} = 0$ and $\lbrack\omega\rbrack = 0$ in $H^{1}(E)$. Otherwise one can only collapse **locally** (on simply connected charts), consistent with practical binning.

**8.7 Observables as functors**

An **observable** (e.g., exit-time exponent, diffusion radius exponent) is a functor $\mathcal{O}:RTM \rightarrow \mathcal{C}$ (sets, groups, numbers) satisfying:

-   **Gauge invariance:** $\mathcal{O}(v + \phi) = \mathcal{O}(v)$;

-   **Monoidal additivity:** $\mathcal{O}(\mathsf{A} \otimes \mathsf{B}) = \mathcal{O}(\mathsf{A}) + \mathcal{O}(\mathsf{B})$ when defined.

**Example.** The **slope functor** $\mathcal{S}$ maps $(X,L,v) \mapsto \alpha$ (as a function on $X$) and is the terminal gauge-invariant observable in bins with $H^{1} = 0$.

**8.8 Summary**

-   **RTM** forms a category where objects carry a **gauge 1-form** $\omega = d\ \log T - \alpha\text{ }d\ \log L$.

-   **Morphisms** are reparametrizations that preserve $\omega$ up to exact forms; the **gauge group** acts by clock shifts.

-   **Slope** $\alpha$ is the moduli invariant; **collapse** equals **flat trivialization** (zero curvature and holonomy).

-   **Products** and **coarse-graining** are functorial; power laws are fixed points of coarse-graining endofunctors.

-   **Cohomology** captures global obstructions to collapse; binning provides simply connected charts where collapse is feasible.

**9. Examples, Counterexamples, and Open Problems**

We close the mathematical exposition with worked examples that satisfy RTM exactly, controlled counterexamples that **must** fail collapse, and a short list of open problems suggested by the scale--clock framework.

**9.1 Exact RTM examples (collapse holds)**

**Example 9.1 (Pure power law)**

Let $T(L) = \kappa L^{\alpha}$ on $L > 0$. Then $v = \log T = \alpha\log L + \log\kappa$, $\omega = dv - \alpha\text{ }d(\log L) = 0$.

-   **Collapse:** trivial (residual constant).

-   **Renormalization:** fixed point of $\mathcal{R}_{b}$ for all $b > 0$.

**Example 9.2 (Slowly varying clock)**

Let $T(L) = L^{\alpha}\mathcal{l}(L)$ with $\mathcal{l}$ slowly varying (Karamata). On any finite $\log{\ L}$-window,

$$v(u) = \alpha u + \varepsilon(u),\varepsilon(u + h) - \varepsilon(u) \rightarrow 0.$$

-   **Collapse:** holds up to $O(\sup \mid \varepsilon(u + h) - \varepsilon(u) \mid )$.

-   **RG:** self-normalized $\mathcal{R}_{b}$ flows to the power-law leaf (Sec. 5.5).

**Example 9.3 (Piecewise homogeneous media in PDE)**

On $\mathbb{R}^{m}$, take $\alpha(x) \equiv \alpha_{0}$ and $L(x) = c \mid x \mid$. The heat kernel of $\partial_{t}u = \nabla \cdot (L(x)^{- \alpha_{0}}\nabla u)$ satisfies similarity with exponent $z = m + \alpha_{0}$ (Sec. 6.2).

-   **Observable:** exit time $\mathbb{E}\tau_{R} \asymp R^{\text{ }z}$ recovers $\alpha_{0}$.

**9.2 Controlled failures (collapse must fail)**

**Counterexample 9.4 (Regime seam / kink)**

$$T(L) = \{\begin{matrix}
\kappa_{1}L^{\alpha_{1}}, & L \leq L^{\star}, \\
\kappa_{2}L^{\alpha_{2}},\ \  & L > L^{\star},\alpha_{1} \neq \alpha_{2}.
\end{matrix}$$

-   **Geometry:** $\omega$ is exact on each side, but loops crossing $L^{\star}$ have nonzero holonomy $\oint\omega = (\alpha_{2} - \alpha_{1})\text{ }d(\log L)$.

-   **Empirics:** residuals show sign change; $\Delta_{\text{collapse}}$ bounded away from 0 unless we **rebin**.

**Counterexample 9.5 (Curved log--log relation)**

Let $v(u) = u + \sin u$ so $T(L) = L\text{ }e^{\sin(\log L)}$.

-   **Curvature:** $g^{''}(u) = - \sin u \neq 0$⇒ collapse statistic scales like $c\text{ }h^{2}$(Sec. 4.4).

-   **RG:** residuals translate under $\mathcal{R}_{b}$, never contracting on a fixed window (Prop. 5.7).

**Counterexample 9.6 (Clock depending on scale)**

If a "clock" factor secretly depends on $L$: $T^{\#}(L) = c(L)\text{ }T(L)$ with non-constant $c$, then

$$v^{\#}(u) = \alpha u + \log\kappa + \log c(e^{u}),$$

and $\omega^{\#} = \omega + d\ \log c(e^{u})$ acquires a $du$**-component**.

-   **Interpretation:** this is **not** a permissible gauge in RTM (clocks must be $L$-independent in-bin). Collapse should and will fail---correctly flagging misspecification.

**9.3 Worked composite constructions**

**Construction 9.7 (Product systems → slope additivity)**

Let $T_{A}(L_{A}) = \kappa_{A}L_{A}^{\alpha_{A}}$, $T_{B}(L_{B}) = \kappa_{B}L_{B}^{\alpha_{B}}$. For independent composition with total scale $L = L_{A}\ L_{B}$ and time $T = T_{A}T_{B}$:

$$T(L) = \kappa_{A}\kappa_{B}\text{ }L^{\alpha_{A} + \alpha_{B}},\alpha_{\text{total}} = \alpha_{A} + \alpha_{B}$$

(Sec. 8.4). This models cascaded stages whose characteristic times multiply.

**Construction 9.8 (Adiabatic patching)**

Partition the scale axis into windows $\{ I_{k}\}$ where $\parallel \partial_{u}\alpha \parallel \leq \varepsilon$. On each window, fit $\alpha_{k}$; define a **piecewise-adiabatic** model

$$v(u) = \sum_{k}^{}{\mathbf{1}_{u \in I_{k}}(\alpha_{k}(u - u_{k}) + c_{k})},$$

with continuity constraints at seams.

-   **Error:** $O(\varepsilon \mid I_{k} \mid )$ per patch; collapse holds locally, fails globally if $\alpha$drifts.

**9.4 Bridges to other theories**

-   **Regular variation / Karamata--de Haan.** RTM lives on the **power-law manifold**; slowly varying clocks are the **center manifold** (Sec. 5.5).

-   **Renormalization group.** RTM fixed points are RG fixed points; curvature is a non-decaying mode.

-   **Time-changed diffusions.** RTM PDEs are Brownian motions with **state-dependent clocks** (Sec. 6.7).

-   **Gauge/connection language.** Collapse $\Leftrightarrow$ flat connection; holonomy captures regime mixing (Secs. 3, 8).

**9.5 Open problems**

1.  **Sharp collapse thresholds.** Prove finite-sample, non-asymptotic bounds linking $\Delta_{\text{collapse}}$ to curvature $g^{''}$ under EIV, with optimal constants.

2.  **Holonomy detection.** Construct statistical tests that distinguish curvature from **topological** obstructions (nontrivial $H^{1}$) using loop integrals of residual 1-forms.

3.  **Variable-exponent regular variation on graphs.** Let $L$ be degree or path length on a random graph; establish law of large numbers for local $\widehat{\alpha}$.

4.  **Inverse problems.** From exit-time data $\mathbb{E}_{x}\tau_{R} \asymp R^{\text{ }m + \alpha(x)}$, reconstruct $\alpha(x)$ (Calderón-type uniqueness with scale-dependent coefficients).

5.  **Global gauges on non-simply connected bins.** Classify when a global clock exists (vanishing cohomology class of $\omega$), and give constructive algorithms to trivialize if possible.

6.  **Beyond power laws.** Characterize the minimal curvature classes $g$for which RG becomes contracting after **nonlinear** reparametrizations (e.g., log--poly or spline gauges).

7.  **Asymptotics under heavy-tailed measurement error.** Extend ODR/SIMEX consistency to $\alpha$-stable noise; quantify robustness envelopes.

8.  **Coupled fields.** Analyze PDEs with feedback $L = L(u,x)$ (scale proxy depends on the state), yielding **nonlinear clocks** and potential bifurcations in $\alpha$.

**10. Mathematical Conclusion**

We provided a rigorous backbone for RTM:

-   From a **scale semigroup** and mild regularity, characteristic time obeys a **power law** $T = \kappa L^{\alpha}$; the **slope** $\alpha$ is a **clock-invariant** structural quantity (Sec. 2).

-   RTM is most naturally expressed via the **1-form** $\omega = d\ \log T - \alpha\text{ }d\ \log L$; **collapse** equals **exactness/flatness**, while regime mixing and curvature appear as **holonomy** (Sec. 3).

-   **Regular variation** with (slowly) **variable exponents** explains finite-window estimates and bias; collapse statistics quantify curvature (Sec. 4).

-   A **renormalization** operator on functions has **power laws as fixed points** and is **contractive** in Hölder/Zygmund classes; slowly varying clocks form a **center manifold** (Sec. 5).

-   In dynamics, RTM exponents act as **local clock fields** for diffusions/PDEs, yielding similarity exponents $z = m + \alpha$ and **adiabatic** approximations when $\alpha$ drifts (Sec. 6).

-   Statistically, **ODR/SIMEX/Theil--Sen** consistently recover local $\alpha$under EIV, and the **collapse statistic** is a specification test against curvature---even with noise (Sec. 7).

-   A **categorical** formulation packages invariance, gauges, and coarse-graining functorially (Sec. 8).

The program yields a compact principle: **structure lives in the slope**, clocks live in the gauge. Where bins are stable and collapse holds, RTM gives a falsifiable, transportable description of how **time stretches with scale**. Where collapse fails, RTM provides a **diagnostic**---curvature or regime mixing---not a fudge. The open problems above outline a path to deepen the theory (non-simply connected gauges, inverse problems, graphs, heavy tails) and to connect it with broader analysis and probability.

*© 2026 Álvaro José Quiceno Rendón. This document is distributed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.*
