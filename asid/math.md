# Grouped Fisher information: model and code

This describes the current finite-difference implementation. It is a **local,
model-based** information estimate, not a measured posterior or an estimate of
information in every physical degree of freedom.

## Observation model and score

Let `phi` be the P grouped log parameters and `u` the controller trajectory.
At a fixed initial state, Warp predicts selected node positions, flattened into
`h(phi, u) in R^D`. The likelihood assumed by the implementation is

```
y = h(phi, u) + epsilon,       epsilon ~ N(0, R),
J = d h / d phi               (D x P),
score = d log p(y | phi, u) / d phi = J^T R^-1 (y - h),
F(phi, u) = E_y[score score^T] = J^T R^-1 J.
```

The last line assumes `R` is known, positive definite, and independent of the
parameters. The expectation is over observation noise **conditional on phi and
u**. If covariance depends on the parameters, extra terms are required; if
model error, tracking errors or observations are correlated, the appropriate
non-diagonal `R` must be used. The implementation assumes `R = sigma^2 I`:
independent, equally noisy 3D coordinates at every selected node and frame.
It does not model visibility, tracking correlation, simulator mismatch or an
expectation over uncertain parameter values. In particular, the nodes of one
object are observation locations, **not posterior parameter particles**.

The score itself is not computed or sampled. Gaussian algebra integrates it
out to give `J^T J / sigma^2`. This is the expected FIM for the stated model,
not the empirical Fisher of a single residual or the exact nonlinear posterior
curvature. For `T` frames and `N` nodes, the unreduced FIM sums the outer
products of all `3TN` sensitivity rows. More independent observations add
information; duplicating correlated observations as though independent
instead overcounts it. Subsampling by `stride` changes which observations are
used; no division by the number of frames or nodes occurs.

`Observation.vector` in `asid/fim/operators.py` selects frames, flattens node
coordinates and detaches to NumPy. `jacobian` returns **parameter-by-observation**
rows `A = J^T`, calculated by finite differences; `Fisher.from_jacobian` in
`asid/fim/fim.py` computes `(A / sigma) @ (A / sigma).T`. `sigma` is the
standard deviation of the *returned observation coordinates*, in metres.
With `reduce="mean"`, the predicted positions are averaged over selected
frames before flattening. For independent per-frame noise of standard deviation
`sigma_frame`, pass `sigma = sigma_frame / sqrt(T)` for the mean. In general
its covariance is `T^-2 sum_{t,s} Cov(epsilon_t, epsilon_s)`, which need not
be diagonal or isotropic and is not supported by the scalar `sigma` interface.
Mean reduction also changes the signal Jacobian: it is not equivalent to
summing per-frame FIMs unless the sensitivities have suitable structure.
All current command-line runs use `reduce="none"`.

## Spatial grouping and coordinates

`asid/fim/group.py` clusters nodes geometrically and assigns object springs by
midpoint to the nearest cluster. For `g` groups,

```
phi = [log k_1, ..., log k_g, log c, log m_1, ..., log m_g],
P = 2g + 1,       k_s = exp(phi_k[group(s)]),
                   m_i = exp(phi_m[group(i)]),   c = exp(phi_c).
```

Object springs in a group share stiffness, nodes in a group share mass, and the
dashpot is global; controller tethers remain nominal. The initial point uses
uniform per-node mass and stiffness. Friction and within-group variation are
not estimated. For a full log-parameter field `eta` and grouping map `B`,
`eta = B phi` here, so `J_phi = J_eta B` and `F_phi = B^T F_eta B`.
Finite differences perturb `phi` directly; neither the full physical-space
Jacobian nor full physical-space FIM needs to be formed. Equivalently,
differentiating in physical rather than log coordinates includes the exponential
chain rule. The result is conditional on this grouping; low grouped posterior
variance is not proof that unmodeled spatial contrasts are known. Changing the
grouping changes the model and its prior, so reported magnitudes across group
counts are not directly comparable without a consistent prior construction.

## Prior, gain and sketch

`Prior.std` supplies independent grouped log-coordinate standard deviations
`s` (currently 0.5 for stiffness and damping, 0.3 for mass). Let `S = diag(s)`.
The whitened information and Gaussian/Laplace precision are

```
H = S F S,        P = I + H,        Cov(phi | y) approximately S P^-1 S.
```

`Fisher.whitened`, `posterior_std`, and `eigenvalues` implement these
operations. These standard deviations are conditional on the local linear,
Gaussian and grouping assumptions. They are not a calibrated uncertainty
estimate for the complete nonlinear simulator.

For a Gaussian linear model, expected information gain from the prior is
`0.5 * logdet(I + H)` nats; for a nonlinear model the local FIM expression is
an approximation. `Fisher.logdet`, `Sketch.gain`, and the planning scripts
currently return **twice** that quantity, `logdet(I + H)`, as a ranking score.
The factor of two does not affect rankings, but scores must not be presented as
mutual information in nats. With existing whitened precision `P_cur` and new
information `H_a`, the corresponding score is
`logdet(P_cur + H_a) - logdet(P_cur)`; the local Gaussian gain in nats is half
of it. Past information must be accumulated without counting the same data
twice and is local to its parameter/state linearization.

`asid/fim/sketch.py` draws an orthonormal `P x r` random basis `Q`, then obtains
`D = (J S / sigma) Q` through `2r` directional finite-difference rollouts. It
forms `T = D^T D`, score `logdet(I + T)`, and `U = Q T^(1/2)` so the stored
approximation is `U U^T = Q Q^T H Q Q^T`. In exact arithmetic this score is no
larger than the full score, but finite-difference noise can invalidate that
comparison. The projected matrix `Q Q^T H Q Q^T` is not generally bounded
above by `H` in PSD order. Therefore, after conditioning on a non-identity
precision, the sketched marginal gain can be either above or below the true
marginal gain. A random subspace is not guaranteed to capture the informative
spectrum or rank close candidates correctly. Reusing the same `Q` over actions
and rounds confines accumulated approximate information to its span, which can
understate information and distort the conditioned scores. Compare rank, seed,
group resolution and selected small full-grouped references before relying on
it. Dense grouped reference storage is `O(P^2)`; sketch storage includes `U`
of size `P x r` and directional observations of size `D x r`. Simulator
rollout cost and observation storage can still be substantial.

## Per-observation-node information map

For the unreduced observation model with independent isotropic noise,
`point_information` in `asid/fim/viz.py` extracts the Jacobian of all selected
frames and Cartesian coordinates for each observed node `i`. Its displayed
value is that node's **standalone** local Gaussian gain under the original
prior, `0.5 logdet(I + S J_i^T J_i S / sigma^2)` nats. The implementation
uses the smaller of parameter-space and observation-space determinants.
`scripts/run_fim_case.py` writes `point_information.npy` and maps the values
to node positions. These gains do not add to the joint information gain:
observations from different nodes can be redundant. The calculation inherits
the stated noise, linearization and grouping assumptions. The existing
stiffness/mass overlays instead repeat grouped `diag(F)` values across group
members and have a different meaning.

## Simulator limitations

`asid/fim/simulator.py` uses deterministic spring-force accumulation and a
smooth ground-penalty surrogate, while the upstream trainer can use hard ground
contact. The Fisher therefore describes the surrogate dynamics and chosen
noise model, not necessarily MuJoCo or the trainer. Contact transitions and
finite differences can remain step-size-sensitive. A local nominal-point FIM
does not average candidate information over a belief about physical parameters.
Use the existing step-size and grouped-reference checks when feasible; longer
realistic checks must be run separately under the project's runtime rules.

## References

- [ASID: Active Exploration for System Identification in Robotic Manipulation](https://arxiv.org/html/2404.12308v2), Gaussian transition model and A-optimal exploration.
- [Optimal Experimental Design for Infinite-dimensional Bayesian Inverse Problems](https://arxiv.org/html/2005.12998v3), prior-preconditioned information and uncertainty.
- [Randomized matrix-free trace and log-determinant estimators](https://arxiv.org/html/1605.04893v2), low-rank log-determinant approximations and their assumptions.
