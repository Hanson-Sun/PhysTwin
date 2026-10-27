# Grouped Fisher information

Fisher information of the spring-mass parameters, grouped spatially on a
500-node subsample of a PhysTwin case.

Overview, output meanings and measured costs: [`asid/README.md`](../README.md).
This file keeps the method details and the validation gates. See
[`asid/math.md`](../math.md) for the observation model, score, and limitations.
The reported log-determinant scores are twice the local Gaussian information gain.

## Run

```bash
conda run -n phystwin-cu132 env PYTHONPATH=. \
  python scripts/run_fim_case.py --groups 50 --frames 60

python scripts/plot_fim.py \
  experiments-fim/sim_soft_sloth_push/g50_f60
```

`run_fim_case.py` writes arrays, `summary.json`, and
`figs/fim_partition.png` under `experiments-fim/<case>/<run_id>/`.
New runs also save `figs/fim_matrix.png`, `point_information.npy`, and
`figs/point_information.png`. The map shows standalone local Gaussian gain
per observed node; values do not add to the joint gain. See [`math.md`](../math.md).
`plot_fim.py` redraws the grouped-parameter overlay from saved arrays.

## Method

- Parameters `phi = [log k (g), log c (1), log m (g)]`, `P = 2g + 1`.
- `k` applies to object springs; controller tethers stay nominal.
- `m` is per node; nominal masses are uniform.
- Jacobian columns by central finite differences of the node trajectories;
  observation noise `sigma = 5 mm`.
- `F = J J^T / sigma^2`; the overlay colors spring midpoints by
  `log10 diag(F)_k` and nodes by `log10 diag(F)_m`.
- Prior `diag(1/s^2)`, `s = [0.5 (k), 0.5 (c), 0.3 (m)]`.
- Sketch: `H = J_w^T J_w`, `Q = orth(Omega)`, `T = (J_w Q)^T (J_w Q)`,
  `gain = log det(I + T)`, `U = Q T^{1/2}`.

## Actions

`scripts/plan_actions.py` builds the recorded push's rigid copies: a pitch about
the horizontal axis, then a rotation about the vertical axis, both about the
object centroid. The family is every `(azimuth, tilt)` pair in `--tilts`, and
every action must reach the contact radius. Each action is sketched at the same
`phi0` with a shared random subspace (one per action under
`--subspace independent`); selection is greedy on the Woodbury marginal gain.

```bash
conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/plan_actions.py \
  --groups 10 --rank 12 --azimuth-count 3 --tilts 0 -20 --select 3 \
  --run-id mix_g10_r12

conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/validate_actions.py \
  experiments-fim/sim_soft_sloth_push/mix_g10_r12 --reuse-fisher
```

`validate_actions.py` computes the exact `F_a` for every action, reports the
Spearman correlation of the gains and whether the greedy sets agree, and repeats
the greedy over the exact `F` restricted to the sketch subspace. `--reuse-fisher`
reloads saved matrices instead of re-running rollouts.

`scripts/plot_fim_heatmaps.py <run>` draws one column per action: top-view
heatmaps of `log10 diag(F)` over the object, springs on the first row and nodes
on the second, on a scale shared across actions. `--relative` subtracts each
action's own mean, so the spatial pattern is comparable when the magnitudes are
not.

## Sequential selection

After a set of pushes the posterior precision in whitened coordinates is
`P = I + sum_a H_a`, so a new action is worth `log det(I + P^{-1} H_a)`.
`scripts/plan_sequential.py` re-simulates every candidate from the deformed
state left by the chosen pushes, picks the best, executes it and repeats. The
parameter mean stays at the assumed truth; the posterior precision and the body
state carry between rounds.

```bash
conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/plan_sequential.py \
  --groups 10 --rank 12 --rounds 3 --azimuth-count 6 --run-id seq_g10_r3

python scripts/plot_fim_heatmaps.py \
  experiments-fim/sim_soft_sloth_push/mix_g20_r12 [--relative]
```

## Validation

- `theta(phi0)` reproduces the nominal simulator.
- Central differences at `h = 0.01` change by 8–16% when the step doubles.
- `F` is symmetric PSD. On the g=50 case, logdet = 216.1; rank 20 captures
  46%, rank 50 is needed for ~95%.
- Azimuth gate (g=10, rank 12, 6 pushes): Spearman 1.0 against exact, same top
  action and same greedy set (`az240`, `az120`, `az060`). Sketch gain is ~0.64
  of exact.
- Mixed gate (g=10, rank 12, 3 azimuths x tilts {0, -20}): Spearman 0.943, one
  inversion (`az240_t-20` vs `az000`); top-1 and the greedy set agree (`az240`,
  `az120`, `az120_t-20`). Tilted pushes are weaker alone, so pitching adds no
  complementarity.
- Marginal gain collapses after the best push: 308.9, 18.5, 3.5 exact.
- A shared subspace caps the joint gain at its rank (0.59 of exact here) but
  does not change the greedy set; `--subspace independent` lifts the cap.
- Finer groups (g=20, 41 parameters): same winner and greedy set, Spearman
  0.943. Exact logdets 571.5, 525.2, 262.6, 163.3, 160.0, 138.2; marginals
  571.5, 37.0, 5.0. Finer groups scale the information (~1.85x) without
  changing the decision; the shared-subspace cap falls to 0.31.
- Sequential design (g=10, 3 rounds): `az240`, `az180`, `az300` with
  conditioned gains 199.7, 28.6, 26.4. At the deformed state the exact
  conditioned gain of `az180` is 55.0 against 46.6 for `az120`, so the change
  of second pick is real.

```bash
python scripts/validate_sketch.py \
  experiments-fim/sim_soft_sloth_push/g50_f60 --rank 50
```

## Caveats

- The analysis uses a smooth penalty ground contact (`soft_ground`). The
  upstream hard-contact trainer is unchanged.
- Finite differences are of the smoothed simulator. Derivatives through contact
  and action gradients are not validated by this path.
- The logdet spectrum has a heavy tail. Rank 50 still reaches only 77% of the
  exact gain because q=0 has no power steps; q=1 or rank near P is needed for
  absolute gains.
- The sketch is local to `(phi, state)`; refresh factors when either moves.
- Absolute sketch gains are 0.64 of exact at g=10 and 0.35–0.58 at g=20: the
  fraction falls as the parameter count grows. Compare actions, not values.
- See [`PLAN.md`](../PLAN.md) for the phased review and implementation plan.
