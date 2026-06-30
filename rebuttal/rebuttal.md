## Official comment: common response on Theorem 5.1

We thank the reviewers for pointing out that the current statement of Theorem 5.1 is stronger than what the proof establishes. We agree with this concern.

The current theorem is written as a uniform approximation statement for the continuous PDE solution,
$$
\sup_{z \in K}\|u(z)-\hat u(z)\|_2 < \varepsilon.
$$
However, the proof in Appendix C actually establishes a more specific architectural expressivity claim: given the local stencil values used by a consistent explicit finite-difference discretization, FieldFormer can approximate the corresponding finite-stencil update map
$$
u_h^{n+1}(\mathbf{i})
=
\psi\!\left(
\{u_h^{n-\ell}(\mathbf{i}+\mathbf{s}) : (\mathbf{s},\ell)\in\mathcal{S}\}
\right)
$$
uniformly over a compact set of stencil-value tuples. In other words, the proof controls the network approximation error for the discrete local update map, not the discretization error between the continuous PDE solution $u$ and a numerical solution $u_h$.

We will revise the paper to make this distinction explicit. Specifically, we will:

1. Retitle Theorem 5.1 as an expressivity result for **finite-stencil local dynamics**, rather than a direct continuous-PDE recovery result.
2. Restate the theorem in terms of the discrete update map $\psi$ and the compact set of reachable stencil tuples $\mathcal{V}_K$:
   $$
   \sup_{v\in \mathcal{V}_K}
   \left\|
   \psi(v)-\mathrm{FF}_\theta(E(v))
   \right\|_2
   < \varepsilon,
   $$
   where $E(v)$ denotes the FieldFormer tokenization of the stencil context.
3. Revise the appendix proof so that its conclusion is the above finite-stencil approximation result, removing the unsupported final step that currently jumps from approximating $\psi$ to approximating the continuous solution $u(z)$.
4. Add a short remark clarifying that a continuous-PDE approximation statement would require an additional numerical-analysis argument, for example
   $$
   \|u-\hat u\|
   \le
   \|u-u_h\|
   +
   \|u_h-\hat u\|,
   $$
   where the first term is a discretization/convergence error controlled by consistency, stability, regularity, and grid refinement assumptions, and the second term is the FieldFormer approximation error. Since the purpose of Theorem 5.1 is to establish architectural compatibility with local finite-stencil dynamics, we believe the finite-stencil statement is the most accurate and defensible version.

This revision preserves the intended role of the theorem: it shows that FieldFormer has sufficient representational capacity to model local update rules induced by finite-stencil discretizations when the relevant local context is available. It does not claim that FieldFormer overcomes discretization error or the observability limits of sparse sensing.

## Official comment: common response on Theorem 5.3

We thank the reviewers for identifying a gap in the proof of Theorem 5.3. We agree that the current proof is too strong as written.

There are two issues. First, the symmetrization argument only implies a two-world statement. If $V$ and $T V$ agree on the observed coordinates but induce different outputs, then an estimator using only the observed data cannot know which completion is the true one. The triangle inequality gives
$$
\max\left\{
\mathbb{E}\|\hat u(V_{\mathrm{obs}})-\psi(V)\|_2,
\mathbb{E}\|\hat u(V_{\mathrm{obs}})-\psi(TV)\|_2
\right\}
\ge
\frac{1}{2}\mathbb{E}\|\psi(V)-\psi(TV)\|_2.
$$
It does not by itself imply the same lower bound for the original world $V$ alone. We will therefore revise the theorem as a two-world/minimax non-identifiability statement: no estimator can be uniformly accurate over two observationally indistinguishable completions when their induced updates are separated.

Second, the proof currently sums coordinate-wise influence margins. As the reviewers point out, this is not valid in general because coordinate effects may interact or cancel. We will replace the coordinate-wise influence assumption with a set-level missed-support assumption. For a missed spatial set $J\subseteq\mathcal{S}_x$, let
$$
A(J)=\sum_{s\in J}\bar a_s.
$$
The revised assumption will state that if the spatial set $J$ is unobserved, then there exist two observationally indistinguishable stencil completions, differing only on $J\times\mathcal{T}$, whose local updates are separated by an amount proportional to the missed influence mass:
$$
\mathbb{E}\|\psi(V)-\psi(T_JV)\|_2 \ge c A(J).
$$
Under this assumption, the two-world lower bound becomes
$$
\max\{\mathrm{risk}(V),\mathrm{risk}(T_JV)\}
\ge
\frac{c}{2}A(J).
$$
Averaging over random longitudinal sensor placement with $J=I^c$ then gives
$$
\mathbb{E}_I[A(I^c)]
=
1-\frac{m_x}{N_x},
$$
so the same dependence on expected missed influence mass is retained, with the factor $1/2$ absorbed into the constant.

This revision changes the interpretation of Theorem 5.3 from an average-case risk lower bound under the original data distribution alone to a conditional non-identifiability result: if sparse sensing leaves a missed spatial support that can produce separated but observationally indistinguishable completions, then any estimator using only the sensed data must incur error on at least one such completion. This is the recoverability limitation we intended to formalize.

## Response to Reviewer hGFy

Thank you for the thoughtful and constructive review.

### R1.Q1: Scope of Theorem 5.1

We agree. As detailed in the official comment above, the current theorem statement overreaches: the proof supports finite-stencil update-map approximation, not direct uniform approximation of the continuous PDE solution.

We will revise Theorem 5.1 and the appendix proof accordingly, so the theorem states the finite-stencil expressivity result directly and removes the unsupported final step from discrete update approximation to continuous-solution approximation. We will also add the discretization-error caveat described in the official comment.

### R1.Q2: Miss-coverage lower bound

We agree with your concern. As detailed in the official comment on Theorem 5.3 above, the current proof incorrectly treats coordinate-wise lower bounds as additive for a single scalar risk.

We will revise the theorem and proof in two ways: state the result as a two-world/minimax non-identifiability bound, and replace the coordinate-wise influence condition with a set-level missed-support condition. This avoids summing separate coordinate perturbation bounds and makes the dependence on the missed spatial influence mass explicit.

### R1.Q3: Empirical claim strength on synthetic sensor-space tasks

Our empirical wording does distinguish "strictly best" from "consistently high-performing." Our intended claim is not that FieldFormer is the top row on every synthetic sensor-space metric, but that it remains consistently comparable to the best baseline across all three synthetic PDE benchmarks.

The results support this interpretation. On Heat, FieldFormer obtains RMSE/MAE of $0.09888/0.07876$, compared with the best baseline at $0.09786/0.07786$. On Pollution, FieldFormer obtains $0.1154/0.09216$, compared with the best baseline at $0.1147/0.09153$. On SWE, FieldFormer obtains $0.04644/0.03703$, compared with the best baseline at $0.04623/0.03683$. Thus, across Heat, Pollution, and SWE, FieldFormer is within roughly one percent of the best method while substantially outperforming several alternatives, especially on SWE.

The text also emphasizes that FieldFormer achieves near-best, consistently competitive sensor-space performance across the synthetic benchmarks, while its strongest empirical gains appear in the real-world persistent sensor-network setting.

## Response to Reviewer NuTG

Thank you for the careful reading and for articulating the theoretical concern so precisely.

### R2.Q1: Theorem 5.1 and the missing discretization-error term

We agree with your diagnosis. The decomposition you propose is exactly the distinction we should have made explicit: the current proof controls the network approximation term for a discrete update rule, but not the discretization error between $u$ and $u_h$.

As detailed in the official comment above, we will revise Theorem 5.1 to state approximation of the finite-stencil update map $\psi$ directly, and we will align the appendix proof with that statement. We think this is the most defensible fix: the intended claim is architectural compatibility with local finite-stencil dynamics, not a new convergence theorem for continuous PDE solvers.

### R2.Q2: Theorem 5.3 lower-bound proof

We agree with both issues you identify. The symmetrization step supports a $\max(a,b)\ge D/2$ two-world conclusion, not a lower bound for $a$ alone, and the coordinate-wise perturbation inequalities cannot be summed without an additional non-cancellation condition.

As described in the official comment on Theorem 5.3 above, we will revise the theorem as a two-world/minimax non-identifiability statement under a set-level missed-support influence assumption. The proof will then use the valid max lower bound and will no longer derive cumulative missed influence by summing coordinate-wise inequalities.

### R2.Q3: Learned metric and characteristic-aligned transport

Thank you for this suggestion. A characteristic-aligned metric of the form $\Delta x \approx v\Delta t$ is well motivated for wave-like or advective transport when a meaningful transport velocity $v$ is known or can be reliably estimated.

Our setting is broader. In the real-world atmospheric and pollution datasets, a single velocity satisfying $\Delta x=v\Delta t$ is not available: transport can be heterogeneous, variable-dependent, affected by forcing and sources, and only indirectly observed through sparse sensors. The strict characteristic form is therefore most appropriate for wave-like transport in the narrower sense, but is less directly applicable to diffusion, mixed advection--diffusion, multivariate atmospheric variables, and real monitoring networks.

The learned $\gamma$-scaled metric is intended as a more general and data-adaptive mechanism. It does not explicitly model a tilted characteristic relation $\Delta x-v\Delta t$; instead, it learns anisotropic spatial and temporal scaling without requiring a specified velocity field. This makes the same architecture applicable across diffusion-like, transport-like, and heterogeneous real-world regimes. We will clarify this design trade-off in the paper and note characteristic-aligned metrics as a promising specialization when reliable velocity information is available.

### R2.W2: Unseen-sensor prediction and global reconstruction

Thank you for raising this weakness. We agree with the main point: FieldFormer is designed to exploit local observational support, so when an entire sensor site is held out, or when evaluation moves far from observed sensors, its locality-aware advantage can diminish. This is exactly why we distinguish sensor-space imputation from sensor-holdout/global reconstruction.

One thing to note is that Section 6.1 already states that ImputeFormer is adapted as a fixed-node masked-imputation baseline over the deployed sensor network, rather than as a continuous coordinate-query field model. This matters for Table 4: in the held-out sensor setting, ImputeFormer still operates with a fixed node set and fixed temporal windows, giving it a transductive fixed-topology advantage for this task. It is therefore a strong fixed-node imputer, but it is not solving exactly the same field-estimation problem as other baselines, which operate on the entire field.

At the same time, Table 4 does not show that Senseiver, or any other global-prior baseline, is consistently best. Senseiver is best only on atmospheric AT RMSE; ImputeFormer is stronger on the remaining AT/RH metrics and on PM$_{10}$/PM$_{2.5}$, while SVGP obtains the best $V_y$ wind metrics. This mixed pattern is the central point we want to emphasize more clearly: once evaluation moves away from local observational support, performance becomes strongly prior-dependent, and no single model reliably solves global reconstruction across all regimes.

FieldFormer performs best in the intended sensor-space regime where local support is available, while fixed-node or global-latent priors can be advantageous in some unseen-sensor settings. This does not contradict our claim; it reinforces our framing that global reconstruction under extreme sparsity is underconstrained and prior-dependent.

## Response to Reviewer jzvM

Thank you for the detailed review and for identifying several places where the framing and experimental disclosure can be improved.

### R3.Q1: Uncertainty-expressive baselines and framing of identifiability

Response to be added.

### R3.Q2: Local decomposition versus architecture

Response to be added.

### R3.Q3: Bootstrap standard deviations in tables

Response to be added.

### R3.Q4: Baseline comparison transparency

Response to be added.
