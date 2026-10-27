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

## What Phase 2 must replace

- Replace selection restricted to rigid copies of a recorded push with
  bounded continuous trajectory parameters and a planner that compares
  multiple feasible candidates and returns the best found trajectory.
- Condition the information objective on the current model, belief and state;
  handle grouping, observation noise and approximation error explicitly.
- Validate action gradients of the entire information objective against
  independent finite differences, including contact. Direct differentiation
  through parameter sensitivities generally needs mixed derivatives; the
  current rollout detaches outputs and is not such a gradient path.
- Map the chosen trajectory, controller geometry, timing and coordinate frames
  into actual MuJoCo controls, and capture aligned observations. The present
  scripts simulate candidate actions only in the Warp surrogate.
- Document the actual optimizer, parameterization, constraints, approximation
  and tests here once implemented. Do not describe a local numerical optimum as
  a globally optimal trajectory.

The reusable candidate generation, sketch scoring and plotting components can
remain a baseline when their assumptions and accuracy are checked. Full
sim2sim online parameter updates and convergence belong to Phase 3.
