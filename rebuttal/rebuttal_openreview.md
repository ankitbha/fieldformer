## Official comment: summary of revision changes

We thank the reviewers for the detailed comments. The revision makes targeted changes to the theory, empirical protocol, ablations, efficiency reporting, and positioning. We summarize the main changes below.

**Theory.** Reviewers hGFy and NuTG identified that Theorem 5.1 was broader than the proof supported. We have retitled and restated it as an expressivity result for finite-stencil local dynamics. The revised theorem claims approximation of the discrete local update map supplied with the relevant stencil values, not direct uniform approximation of the continuous PDE solution. The appendix proof now ends at this finite-stencil approximation result, and the main text explicitly separates network approximation error from discretization error.

For Theorem 5.3, reviewers hGFy and NuTG pointed out that the symmetrization step only supports a two-world/minimax conclusion, and that coordinate-wise influence margins cannot be summed without stronger assumptions. The revised theorem is now a missed-support non-identifiability result: when sparse sensing leaves two observationally indistinguishable stencil completions whose updates differ, no estimator using only the observed coordinates can be accurate on both. We also replaced the coordinate-wise summation argument with a set-level missed-support ambiguity margin.

**Global reconstruction and sensor holdout.** Reviewer NuTG emphasized that FieldFormer depends on local observational support and that stronger global priors can help in sensor-holdout/global reconstruction. The revision now states this explicitly. The sensor-holdout discussion notes ImputeFormer's fixed-node/transductive advantage and that Senseiver is not uniformly best. Global reconstruction is framed as prior-dependent extrapolation, while sensor-space imputation remains the primary identifiable task.

**Uncertainty-aware methods.** Reviewer jzvM requested uncertainty-expressive baselines. We added related work on diffusion/score-based imputation, Neural Processes and convolutional variants, and deep ensembles. We also added a five-member Fourier-MLP deep ensemble baseline and report ensemble-mean point predictions under the same RMSE/MAE protocol. We do not report interval coverage or predictive-distribution metrics because the benchmark evaluates deterministic point reconstruction against single realized fields, not calibrated distributional completion. That would require a different evaluation target and protocol. The revision clarifies that SVGP and the deep ensemble are evaluated by posterior/predictive mean for point accuracy, and that bootstrap standard deviations measure metric-estimate uncertainty.

**Local decomposition ablations.** Reviewer jzvM asked for clearer separation between local decomposition and transformer aggregation. We extended FieldFormer-MLP ablations to the two real-world datasets and added an explicit three-way interpretation in Section 6.5: FieldFormer vs. FieldFormer-MLP isolates local transformer aggregation under the same query-specific local neighborhood, while FieldFormer vs. Senseiver contrasts query-specific local decomposition with global sensor-set attention.

**Reporting, transparency, and efficiency.** Tables 1--8 now report bootstrap standard deviations where the captions claim mean plus standard deviation. We added an appendix protocol table documenting each method's input representation, objective/loss, masking strategy, checkpoint criterion, tuning choices, and output interface. We also added an inference-efficiency benchmark reporting time per query, throughput, setup time, and peak GPU memory. FieldFormer is not claimed to be faster than direct coordinate fields, but among context-conditioned transformer/grid/set methods it is substantially faster than Senseiver and RecFNO because it uses fixed-size local attention.

**Motivation and exposition.** We strengthened the introduction with application-domain citations for persistent sparse sensor networks, including environmental monitoring, atmospheric sensing, pollution tracking, industrial IoT, and scientific field deployments. We also tightened terminology around the velocity-scaled metric, fixed maximal sparse context, and token geometry; clarified the distinction from characteristic-aligned metrics that require a known velocity field; and softened wording that could imply unsupported global recovery.

Overall, the revision narrows the formal claims to what is proved, adds empirical transparency and ablations, and sharpens the intended scope: FieldFormer is a locality-aware, mesh-free transformer for point reconstruction in persistent sparse sensor networks, with global reconstruction treated as prior-dependent extrapolation.

## Common response on Theorem 5.1

The reviewers are correct that the current statement of Theorem 5.1 is stronger than what the proof establishes.

The current theorem is written as a uniform approximation statement for the continuous PDE solution:

\\[
\\sup_{z \\in K}\\Vert u(z)-\\hat u(z)\\Vert_2 \\lt \\varepsilon.
\\]

The proof in Appendix C establishes a more specific architectural expressivity claim: given the local stencil values used by an explicit finite-difference discretization, FieldFormer can approximate the corresponding finite-stencil update map

\\[
u_h^{n+1}(\\mathbf{i}) = \\psi(\\{u_h^{n-\\ell}(\\mathbf{i}+\\mathbf{s}) : (\\mathbf{s},\\ell)\\in\\mathcal{S}\\})
\\]

uniformly over a compact set of stencil-value tuples. Thus the proof controls the network approximation error for the discrete local update map, not the discretization error between the continuous PDE solution \\(u\\) and a numerical solution \\(u_h\\).

We will revise Theorem 5.1 to state this result directly. The theorem will be retitled as an expressivity result for **finite-stencil local dynamics** and restated as

\\[
\\sup_{v \\in {\\mathcal V}_{K}} \\Vert \\psi(v)-F_{\\theta}(E(v)) \\Vert_{2} \\lt \\varepsilon.
\\]

Here \\(\\mathcal{V}_K\\) is the compact set of reachable stencil-value tuples and \\(E(v)\\) is the FieldFormer tokenization. We will also revise the appendix proof to end at this finite-stencil approximation bound, deleting the unsupported jump to continuous-solution approximation.

Finally, we will add a remark separating approximation and discretization:

\\[
\\Vert u-\\hat u\\Vert \\le \\Vert u-u_h\\Vert + \\Vert u_h-\\hat u\\Vert.
\\]

The first term requires standard numerical-analysis assumptions such as consistency, stability, regularity, and grid refinement; Theorem 5.1 only addresses the second term. This preserves the intended role of the theorem: architectural compatibility with local finite-stencil dynamics when the relevant local context is available.

## Common response on Theorem 5.3

The reviewers are correct that the current proof of Theorem 5.3 is too strong as written.

First, the symmetrization argument only implies a two-world statement. If \\(V\\) and \\(T V\\) agree on the observed coordinates but induce different outputs, an estimator using only the observed data cannot know which completion is true. The triangle inequality gives a lower bound on the larger of the two risks:

\\[
\\max\\{R(V),R(TV)\\} \\ge \\frac{1}{2}\\Vert \\psi(V)-\\psi(TV)\\Vert_2.
\\]

It does not imply the same lower bound for the original world \\(V\\) alone. We will therefore revise the theorem as a two-world/minimax non-identifiability statement: no estimator can be uniformly accurate over two observationally indistinguishable completions when their updates are separated.

Second, the proof currently sums coordinate-wise influence margins, which is not valid in general because coordinate effects may interact or cancel. We will replace this with a set-level missed-support assumption. For a missed spatial set \\(J\\), let

\\[
A(J)=\\sum_{s \\in J}\\bar a_s.
\\]

The revised assumption states that if \\(J\\) is unobserved, then there exist two observationally indistinguishable stencil completions, differing only on the missed support, whose updates are separated by at least a constant times \\(A(J)\\):

\\[
\\mathbb{E}\\Vert \\psi(V)-\\psi(T_JV)\\Vert_2 \\ge cA(J).
\\]

Under this assumption, the two-world lower bound says that the larger of the two risks is at least a constant times \\(A(J)\\):

\\[
\\max\\{R(V),R(T_JV)\\} \\ge \\frac{c}{2}A(J).
\\]

Averaging over random longitudinal sensor placement with \\(J\\) equal to the missed spatial set gives

\\[
\\mathbb{E}_I[A(I^c)] = 1-\\frac{m_x}{N_x},
\\]

so the original dependence is retained while the theorem is stated as a valid two-world/minimax result.

The revised theorem is therefore not an average-case risk lower bound under the original data distribution alone. It is a conditional non-identifiability result: if sparse sensing leaves missed spatial support that can produce separated but observationally indistinguishable completions, any estimator using only the sensed data must fail on at least one completion.

## Response to Reviewer hGFy

Thank you for the thoughtful review.

### R1.Q1: Scope of Theorem 5.1

We agree. As detailed in the official comment, the current theorem statement overreaches: the proof supports finite-stencil update-map approximation, not direct uniform approximation of the continuous PDE solution.

We will revise Theorem 5.1 and the appendix proof so the theorem states the finite-stencil expressivity result directly. We will also add the discretization-error caveat separating \\(\\|u-u_h\\|\\) from the FieldFormer approximation term.

### R1.Q2: Miss-coverage lower bound

We agree. As detailed in the official comment on Theorem 5.3, the current proof incorrectly turns a two-world symmetrization argument into a one-world risk bound and also treats coordinate-wise lower bounds as additive.

We will revise the theorem as a two-world/minimax non-identifiability bound and replace the coordinate-wise influence condition with a set-level missed-support condition. This avoids the invalid summation step while preserving the intended dependence on missed spatial influence mass.

### R1.Q3: Empirical claim strength on synthetic sensor-space tasks

Our empirical wording does distinguish "strictly best" from "consistently high-performing." Our intended claim is not that FieldFormer is the top row on every synthetic sensor-space metric, but that it remains consistently comparable to the best baseline across all three synthetic PDE benchmarks.

The results support this interpretation. On Heat, FieldFormer obtains RMSE/MAE of 0.09888/0.07876, compared with the best baseline at 0.09786/0.07786. On Pollution, FieldFormer obtains 0.1154/0.09216, compared with 0.1147/0.09153. On SWE, FieldFormer obtains 0.04644/0.03703, compared with 0.04623/0.03683. Thus, across Heat, Pollution, and SWE, FieldFormer is within roughly one percent of the best method while substantially outperforming several alternatives, especially on SWE.

We will keep this distinction explicit: FieldFormer is consistently competitive on synthetic sensor-space prediction, while its strongest empirical gains appear in the real-world persistent sensor-network setting.

## Response to Reviewer NuTG

Thank you for the careful reading.

### R2.Q1: Theorem 5.1 and the missing discretization-error term

We agree with your diagnosis. The decomposition you propose is exactly the distinction needed: the current proof controls the network approximation term for a discrete update rule, but not the discretization error between \\(u\\) and \\(u_h\\).

As detailed in the official comment, we will revise Theorem 5.1 to state approximation of the finite-stencil update map \\(\\psi\\) directly and align the appendix proof with that statement. This is the most defensible fix: the intended claim is architectural compatibility with local finite-stencil dynamics, not a new convergence theorem for continuous PDE solvers.

### R2.Q2: Theorem 5.3 lower-bound proof

We agree with both issues you identify. The symmetrization step supports a \\(\\max(a,b)\\ge D/2\\) two-world conclusion, not a lower bound for \\(a\\) alone, and the coordinate-wise perturbation inequalities cannot be summed without an additional non-cancellation condition.

As described in the official comment on Theorem 5.3 above, we will revise the theorem as a two-world/minimax non-identifiability statement under a set-level missed-support influence assumption. The proof will then use the valid max lower bound and will no longer derive cumulative missed influence by summing coordinate-wise inequalities.

### R2.Q3: Learned metric and characteristic-aligned transport

A characteristic-aligned metric of the form \\(\\Delta x \\approx v\\Delta t\\) is well motivated for wave-like or advective transport when a meaningful transport velocity \\(v\\) is known or can be reliably estimated.

Our setting is broader. In the real-world atmospheric and pollution datasets, a single velocity satisfying \\(\\Delta x=v\\Delta t\\) is not available: transport can be heterogeneous, variable-dependent, affected by forcing and sources, and only indirectly observed through sparse sensors. The strict characteristic form is therefore most appropriate for wave-like transport in the narrower sense, but less directly applicable to diffusion, mixed advection--diffusion, multivariate atmospheric variables, and real monitoring networks.

The learned \\(\\gamma\\)-scaled metric is intended as a more general and data-adaptive mechanism. It does not explicitly model a tilted characteristic relation \\(\\Delta x-v\\Delta t\\); instead, it learns anisotropic spatial and temporal scaling without requiring a specified velocity field. This makes the same architecture applicable across diffusion-like, transport-like, and heterogeneous real-world regimes. We will clarify this design trade-off in the paper and note characteristic-aligned metrics as a promising specialization when reliable velocity information is available.

### R2.W2: Unseen-sensor prediction and global reconstruction

FieldFormer is designed to exploit local observational support, so when an entire sensor site is held out, or when evaluation moves far from observed sensors, its locality-aware advantage can diminish. This is exactly why we distinguish sensor-space imputation from sensor-holdout/global reconstruction.

Section 6.1 already states that ImputeFormer is adapted as a fixed-node masked-imputation baseline over the deployed sensor network, rather than as a continuous coordinate-query field model. This matters for Table 4: in the held-out sensor setting, ImputeFormer still operates with a fixed node set and fixed temporal windows, giving it a transductive fixed-topology advantage. It is therefore a strong fixed-node imputer, but it is not solving exactly the same field-estimation problem as coordinate-query baselines.

Table 4 also does not show that Senseiver, or any other global-prior baseline, is consistently best. Senseiver is best only on atmospheric AT RMSE; ImputeFormer is stronger on the remaining AT/RH metrics and on PM10/PM2.5, while SVGP obtains the best \\(V_y\\) wind metrics. This mixed pattern is the main point: once evaluation moves away from local observational support, performance becomes strongly prior-dependent, and no single model reliably solves global reconstruction across all regimes.

FieldFormer performs best in the intended sensor-space regime where local support is available, while fixed-node or global-latent priors can be advantageous in some unseen-sensor settings. This does not contradict our claim; it reinforces our framing that global reconstruction under extreme sparsity is underconstrained and prior-dependent.

## Response to Reviewer jzvM

Thank you for the detailed review. We address the main concerns below and will incorporate the corresponding paper revisions.

### R3.Q1: Uncertainty-expressive baselines and framing of identifiability

Our argument is not "sensor-space prediction or nothing." The intended claim is narrower: under extreme sparse sensing, **point-estimate global reconstruction** is underidentified unless one adds strong priors or represents uncertainty over multiple plausible completions.

Uncertainty-expressive methods are complementary to this framing. Our tables evaluate point predictions using RMSE/MAE against one realized target field, so distributional methods must still be reduced to a point estimate, typically a predictive mean or median. That evaluates point accuracy, not the full quality of calibrated or multimodal uncertainty.

We will revise the SVGP discussion accordingly: SVGP is probabilistic, but our benchmark uses its posterior mean to match the point-prediction protocol, so it should not be presented as covering the full class of uncertainty-aware reconstruction methods. We are also running a **deep ensemble of Fourier-MLP** as an additional predictive baseline, reporting the ensemble mean under the same RMSE/MAE protocol and uncertainty diagnostics such as predictive variance or interval coverage where appropriate.

### R3.Q2: Local decomposition versus architecture

The current experiments already separate these factors more than the paper makes explicit.

FieldFormer-MLP is the controlled ablation for **local decomposition without transformer aggregation**: it keeps the same query-specific local context, but replaces the transformer aggregator. Senseiver provides the complementary comparison: it is a global attention-based set model that does not use FieldFormer's query-specific local neighborhood, learned local metric, or per-query decomposition. It is not a one-switch ablation, but it is the relevant no-local-decomposition attention comparator already in the experiments.

Together, FieldFormer, FieldFormer-MLP, and Senseiver compare local decomposition versus global attention, and transformer aggregation versus MLP aggregation within a local context. We will make this logic explicit and extend the FieldFormer-MLP ablation to the two real-world datasets.

### R3.Q3: Bootstrap standard deviations in tables

The reported standard deviations are **bootstrap standard deviations of the RMSE/MAE estimates**, not predictive uncertainty. They measure how much the aggregate metric changes when resampling the held-out evaluation set; they do not measure epistemic uncertainty over possible global field completions or per-query predictive variance.

Small bootstrap standard deviations are expected here because the held-out sets are large and the processes are smooth or highly structured. They show metric stability under data resampling, not certainty about global reconstruction.

### R3.Q4: Baseline comparison transparency

The comparison is like-for-like at the task/evaluation level: methods use the same sparse observations, train/validation/test split, missing-entry masks, validation-RMSE checkpoint selection, and RMSE/MAE evaluator on the same held-out targets. These details are available in the anonymous repository provided with the submission.

The training objectives and hyperparameters are not literally identical, because the baselines have different native forms: coordinate neural fields, grid models, fixed-node imputers, global set models, SVGP, and PINN variants require different inputs and losses. The comparison controls the data split and evaluation target, while allowing method-appropriate training. We will add a compact appendix table listing each baseline's input representation, objective, masking strategy, checkpoint metric, and main tuning choices.

### R3.Q5: Scalability and efficiency claims

We will add an inference-efficiency comparison. The main deployment cost in our setting is prediction under sparse persistent sensor networks, so we will report wall-clock prediction time per evaluated query and peak GPU memory where available, using trained checkpoints, the same evaluation script, device, and batching protocol as the RMSE/MAE results. We will frame this as empirical throughput under the studied sparse-sensing protocol, not as an architecture-intrinsic complexity theorem.

### R3.Q6: Paper revisions for scope, related work, and exposition

We will also revise the paper for scope and exposition: add application-domain references for persistent sparse sensor networks, expand related work on diffusion/score-based imputation such as CSDI, define paper-specific terms at first use, and clarify how the learned metric affects neighborhood construction and transformer aggregation.
