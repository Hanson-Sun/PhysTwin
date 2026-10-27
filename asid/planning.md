# Planning: current prototype and replacement

This documents the existing prototype, not a completed continuous trajectory
planner. The full implementation plan is in [PLAN.md](PLAN.md); the Fisher model
and its assumptions are in [math.md](math.md).

## Current behavior

`asid/planning/actions.py` generates candidate pushes by pitching and yawing a
recorded controller/contact trajectory about the initial node centroid. Its
contact check compares the candidate with the initial node positions.
`scripts/plan_actions.py` evaluates each candidate at the nominal grouped
parameters using directional finite-difference Fisher sketches and selects a
set greedily with `asid/planning/select.py`. It uses a log-determinant score
conditioned on the already selected sketch factors. The score is twice local
Gaussian information gain in nats; it is not a measurement of true nonlinear
expected information gain.

`scripts/plan_sequential.py` re-simulates candidates from the previous
selected push's simulated final positions, conditions on accumulated sketched
precision, and picks one action per round. It does not estimate parameters
from observations: the mean remains fixed at `partition.phi0`. The restart
sets carried positions at rest, rather than preserving velocities and contact
state. The same random sketch subspace is used across rounds, so accumulated
information remains confined to that subspace. These are approximate offline
experiments, not a MuJoCo execution or continual-learning loop.

## Phase 2 prototype: sampled vs gradient refinement

`asid/planning/trajectory.py` adds a bounded prototype. It linearly
interpolates a small set of translation-offset knots over a controller/contact
template, keeping the first offset zero. `rigid_template` builds a compatible
template from initial point clouds and known controller positions without
inferring a controller origin from tracked points. It rejects sampled
candidates outside workspace or speed limits.
Given a trajectory, `fisher_score` averages half-logdet local Gaussian gains
over explicitly supplied grouped-parameter hypotheses and weights, using a
prior-whitened low-rank directional finite-difference sketch and current
precision `P = I + U_current U_current^T`. For each candidate and parameter
hypothesis `j`, the finite-difference sketch returns a factor `U_j`; its
score is `0.5 logdet(I + U_j^T P^-1 U_j)`. Hypothesis scores are averaged
with normalized user-provided weights. With one nominal hypothesis this is a
local approximation, not full Bayesian expected information gain. The same
sketch directions are used for comparison. The selected trajectory is the
best feasible candidate found, not a global optimum; the low-rank and
local-belief approximations in [math.md](math.md) apply.

The sampler includes the zero-offset template, then draws `candidates - 1`
independent uniform knot arrays in `[-offset_limit, offset_limit]`. It rejects
candidates if any controller/contact point leaves the axis-aligned workspace
or exceeds `max_speed * dt` per frame, then keeps the maximum score. Gradient
refinement, when requested, compares central finite differences of the full
score at `h` and `h/2`; a consistent gradient proposes a normalized step of
length `h`, projected to knot bounds and accepted only if feasible and better.
This is a finite-difference optimization of the score, not action autodiff.

Optional gradient refinement uses central finite differences of the entire
score and compares action gradients at `h` and `h/2`. If they disagree beyond
tolerance or violate bounds, it retains the sampled result. On a small
contact-active sloth window (100 nodes, two groups, 6 frames, rank 2, two
candidate trajectories), the sampled scores were 8.2224 and 8.0653 nats;
gradient refinement was rejected as unstable and the best sampled candidate
was retained. This is a feasibility probe, not evidence that the sampled
winner is robust at higher rank or representative of a complete push. The finite-difference gradient cost grows
with the number of action knots and
sketch rank; no speedup has been established. The existing control/contact
path does not expose end-to-end autodiff.

`asid/planning/mujoco.py` executes a translation-only trajectory on a mocap
interactor after checking the supplied initial pose, fixed orientation, rigid
controller/contact motion and timestep alignment. It reflects PhysTwin's
z-down displacement into MuJoCo's z-up frame, linearly interpolates each
segment at MuJoCo's physics timestep, and returns commands, timestamps and
RGB-D frames. A focused test executes a selected synthetic trajectory.
`scripts/plan_mujoco_smoke.py` scores three candidate pushes on a 200-node
sloth reconstruction, then executes the best candidate in a MuJoCo sloth
scene and captures five RGB-D frames. The smoke test represents the claw by
only two pad centres in Warp, and initial object states are not an exact match;
it validates plumbing, not transfer of predicted information or contact
fidelity. The recorded sloth controller tracks are not rigid translations (up
to 13–18 cm of residual in the first six frames); the adaptor rejects them
rather than silently mapping a different motion. A calibrated sim2sim test
requires a validated tool surface and matching initial object state.

## What Phase 2 must replace

- Extend the current sampled, bounded translation-knot prototype beyond a
  fixed template when needed; compare multiple feasible candidates and return
  the best found trajectory.
- Condition the information objective on the current model, belief and state;
  handle grouping, observation noise and approximation error explicitly.
- Validate action gradients of the entire information objective against
  independent finite differences, including contact. Direct differentiation
  through parameter sensitivities generally needs mixed derivatives; the
  current rollout detaches outputs and is not such a gradient path.
- Improve the coarse matched-tool smoke test into a calibrated sim2sim case,
  with tool surface, pose alignment and current object state checked in both
  simulators.
- Document the actual optimizer, parameterization, constraints, approximation
  and tests here once implemented. Do not describe a local numerical optimum as
  a globally optimal trajectory.

The reusable candidate generation, sketch scoring and plotting components can
remain a baseline when their assumptions and accuracy are checked. Full
sim2sim online parameter updates and convergence belong to Phase 3.
