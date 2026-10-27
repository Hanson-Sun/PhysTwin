# ASID: information-driven push selection

What is new here: a way to score candidate pushes by how much they would teach
about the material parameters of a PhysTwin soft body, and to pick a sequence of
pushes given what the earlier ones already taught. The model is the existing Warp
spring-mass simulator; the new code is the analysis around it.

All numbers below are from `data/different_types/sim_soft_sloth_push`, a 500-node
subsample, 60 frames, 167 substeps, on one RTX 3060.

## Contributions

1. **Grouped Fisher information.** The material parameters are grouped spatially
   into $g$ regions, giving $P = 2g + 1$ parameters instead of one per spring and
   node. The exact Fisher $F = J J^\top/\sigma^2$ is formed from a finite-difference
   Jacobian of the node trajectories. Gives a per-region information map
   (`node_groups`, `fim_diag`) and a posterior standard deviation per parameter.
2. **Low-rank sketch.** A randomized $q = 0$ estimator produces a factor $U$
   with $H \approx U U^\top$ from $2\ell$ rollouts per action instead of $2P$, so
   a push can be scored for a fraction of the exact cost. At $\ell = 12$ it
   reproduces the exact action ranking (Spearman 0.94–1.0, same top action and
   greedy set at $g = 10$ and $g = 20$).
3. **Selection.** Greedy selection on the Woodbury marginal gain, and a myopic
   sequential rule that re-simulates every candidate from the deformed body and
   scores $\log\det(I + P^{-1} H_a)$ against the accumulated posterior precision.
   On the sloth the sequential set is `az240 → az180 → az300`, which differs from
   the simultaneous greedy set at the second push.
4. **Harness.** Rollout wrapper with parameter writes, action swaps and start
   from a deformed state; run directories that record config, arrays and a
   manifest; figure scripts.

## How it works

1. **Reduced case.** Seeded farthest-point subsample of the case points, radius
   graph for object springs, controller tethers, dense contact trajectory from
   the recorded push. Ground contact is a smooth penalty (`soft_ground`), which
   is what makes finite differences stable; the upstream hard-contact trainer is
   untouched.
2. **Parameters.** $\varphi = [\log k\ (g),\ \log c\ (1),\ \log m\ (g)]$,
   $\theta = \exp(B\varphi)$. $k$ applies to object springs, tethers stay
   nominal; $c$ is the global dashpot; $m$ is per node. Prior std
   $0.5, 0.5, 0.3$ in log units.
3. **Exact Fisher.** Central differences of the observed frames (stride 2,
   $\sigma = 5$ mm), $F = J J^\top/\sigma^2$. Whitened $H = \Sigma F \Sigma$
   with $\Sigma = \operatorname{diag}(\text{prior std})$, so
   $\log\det(I + H)$ is information in units of prior precision.
4. **Sketch.** $Q = \operatorname{orth}(\Omega)$ (Rademacher, fixed seed),
   $T = (J_w Q)^\top (J_w Q)$, $\text{gain} = \log\det(I + T)$,
   $U = Q T^{1/2}$. $J_w Q$ is $\ell$ central differences along whitened
   directions.
5. **Selection.** Keep factors, never reassemble $F$:
   $\text{gain}(a) = \log\det(I + U_a^\top H_\text{cur}^{-1} U_a)$,
   $H_\text{cur} = I + \sum_a U_a U_a^\top$.
6. **Sequential.** Execute the chosen push, carry the body state, update
   $P \leftarrow P + H_a$, then re-simulate every remaining candidate from that
   state and repeat.

## Outputs and how to read them

Run directories are `experiments-fim/<case>/<run_id>/`. `config.json` is the
authoritative record of what was run; the folder name is only a label.

| File | Meaning | How to read it |
|---|---|---|
| `summary.json` | Dense grouped finite-difference run summary | `logdet` is twice the local Gaussian information gain in nats (0 = nothing learned); `eigenvalues` = whitened spectrum; `rollout_seconds` = measured cost per rollout |
| `fisher.npy` | Exact $F$ in $\varphi$ order ($k$, $c$, $m$) | $P \times P$; diagonal entries are per-parameter information, off-diagonal are correlations |
| `fim_diag.npy` | $\operatorname{diag}(F)$, flattened | Only comparable across runs with the same grouping and noise model |
| `point_information.npy` | Standalone local Gaussian information gain per observed node | New dense grouped runs only; gains in nats are not additive across nodes |
| `posterior_std.npy` | Posterior std after the selected set | In log units; divide by the prior std for the fraction of prior uncertainty left (0.34 = std is 34% of the prior) |
| `node_groups.npy`, `spring_groups.npy`, `node_centroids.npy`, `phi0.npy` | The spatial partition and its nominal parameters | Which pieces share one parameter; required to expand any per-group value back onto the body |
| `gains.json` | Sketch gain per action, plus `selected` | `gain` ranks actions; `marginal_gain` is the gain of adding that action given the earlier picks, so `199 → 8.1 → 1.7` means the second and third add ~4% and ~1%. `clearance` is the minimum distance from the contact points to any node in metres ($\le 0.02$ is contact) and `contact_frames` counts frames inside that radius; both are diagnostics, not the score |
| `comparison.json` | Exact vs sketch (`validate_actions.py`) | `spearman` = rank agreement; `selection_match` = greedy sets agree; `shared_subspace.capture` = fraction of the exact joint gain a rank-$\ell$ shared subspace keeps |
| `rounds.json` | Sequential scores per round | Scores are comparable within a round, not across rounds ($P$ grows); `posterior_std` blocks show what is still unknown |
| `conditioned_exact.json` | Exact re-check at the deformed state | Confirms a sequential choice that differs from the simultaneous one |
| `final_state.npy` | Node positions $(N, 3)$ after the last push | Input to the next round |
| `actions/<name>/` | `U.npy` factor, `JQ.npy` = $J_w Q$, `controller.npy`/`contact.npy`; `fisher.npy` (exact) once `validate_actions.py` has run | The trajectories are the ones actually simulated, so any result can be reproduced without regenerating actions |
| `figs/` | `fim_partition.png`, `fim_matrix.png`, `point_information.png`, action and sequential plots | Matrix = grouped FIM; point map = standalone per-node local gain; partition and action overlays map grouped diagonals, not per-observation gain |

A sketch gain is a lower bound on the exact one and is not on the same scale:
use it to order actions, and the exact numbers to state magnitudes.

## Sketch accuracy (measured)

`scripts/analyze_sketch.py` replays the saved exact Fishers (`mix_g10_r12`,
`mix_g20_r12`) through the sketch's subspace choice. With $H$ known,
$\text{gain} = \log\det(I + Q^\top H Q)$ isolates the subspace from
differentiation noise, so every number below is an upper bound on a real
sketch. Rollouts are per action, at 2 per direction.

Capture of the exact gain at $q = 0$, over 5 seeds × 6 actions:

| rank $\ell$ | $g = 10$ ($P = 21$) | $g = 20$ ($P = 41$) | rollouts |
|---:|---:|---:|---:|
| 4 | 25% | 17% | 8 |
| 8 | 46% | 30% | 16 |
| 12 | 65% | 43% | 24 |
| 20 | 97% | 62% | 40 |
| $P$ | 100% | 100% | $2P$ |

These answer two different questions, and the answers differ sharply.

**Ordering is safe.** At $\ell = 12$ the Spearman correlation with the exact
gains is 1.00 at $g = 10$ and 0.98–1.00 at $g = 20$, the top action matches in
every seed, and the greedy set of three matches exactly. This holds down to
$\ell = 4$. That is what the sketch is for.

**Absolute gains are not.** The log-det spectrum has a heavy tail, so
$\ell = 12$ recovers 65% and 43% of the total. Magnitudes need $\ell$ close to
$P$. Note what that implies at $g = 10$: the exact Fisher costs 42 rollouts, so
no sketch cheaper than that is within 3% of it.

**Power iterations do not close the gap.** $q = 1$ adds 5–9 points at fixed
rank; $q = 2$ adds under 1 point on top of $q = 1$. At an equal rollout budget
rank wins, because $q$ multiplies cost by $q + 1$ while rank adds it linearly:

| budget | $q = 0$ | $q = 1$ | $q = 2$ | best |
|---:|---:|---:|---:|---|
| 24 | 43% | 29% | 21% | $q = 0$, rank 12 |
| 48 | 71% | 50% | 37% | $q = 0$, rank 24 |
| 72 | 93% | 67% | 51% | $q = 0$, rank 36 |

($g = 20$.) There is a second reason to leave $q = 0$: every power iteration
needs a $J^\top$ application, and the harness steps the simulator with
`requires_grad=False`, so $q > 0$ has no cheap implementation here at all.

**Shared vs independent subspace.** `build_dictionary` defaults to `shared`,
one subspace for every action. That is the right default for comparing
standalone gains, but it caps *accumulated* information at `rank` dimensions:

| actions | exact | shared | independent |
|---:|---:|---:|---:|
| 1 | 571.5 | 180.1 (32%) | 180.0 (32%) |
| 2 | 608.4 | 189.1 (31%) | 345.8 (57%) |
| 3 | 613.5 | 191.3 (31%) | 447.5 (73%) |

($g = 20$, exact greedy order.) With `shared` the posterior stops growing after
the first action, so the reported posterior std stays near the prior and
redundancy between pushes is under-penalised. `plan_sequential.py` passes one
seed to every action and every round, which pins it to a single subspace, so
this applies there by construction. The ordering still matched the exact one
on the six-action set, but the consolidated information it reports is about a
third of the truth.

**One caveat on the recorded numbers.** The offline model reproduces the
gains `plan_actions.py` actually wrote to within 8% on average (Spearman 0.94),
and the recorded values run up to 16% *higher*: finite-difference noise adds
energy, so a real sketch gain is optimistic against its own subspace. Do not
compare a sketch gain to an exact one as if the difference were the subspace.

```bash
conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/analyze_sketch.py
```

## Cost (measured)

One 60-frame, 500-node rollout: **5.15 s** in release mode (83 s in debug).
`run_fim_case.py` does $2P$ rollouts; the sketch does $2\ell$ per action.

| Run | Exact $F$ | Sketch, $\ell = 12$ |
|---|---|---|
| $g = 10$ ($P = 21$) | 3.6–4.0 min per action | 2.1 min per action |
| $g = 20$ ($P = 41$) | 7–9 min per action | 2.1 min per action |
| $g = 50$ ($P = 101$) | 17.3 min per action | not run |

Full runs: 6-action screen at $g = 10$ or $g = 20$, 13–14 min; exact validation
of those 6 actions, 24 min ($g = 10$) and 54 min ($g = 20$); three-round
sequential selection at $g = 10$, 42 min. Figures regenerate in seconds.

## Reproduce

```bash
conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/run_fim_case.py \
  --groups 50 --frames 60 --run-id g50_f60

conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/plan_actions.py \
  --groups 10 --rank 12 --azimuth-count 3 --tilts 0 -20 --select 3 --run-id mix_g10_r12

conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/validate_actions.py \
  experiments-fim/sim_soft_sloth_push/mix_g10_r12

conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/plan_sequential.py \
  --groups 10 --rank 12 --rounds 3 --azimuth-count 6 --run-id seq_g10_r3
```

Use focused tests with `conda run -n phystwin-cu132`; name individual test
modules because some simulator tests require CUDA and saved case data.

## Limits

- The design conditions on the posterior precision and the deformed state; the
  parameter mean stays at the assumed truth, so $J$ is never re-evaluated at an
  updated estimate.
- $F$ is that of the smoothed surrogate, not the hard-contact trainer.
- Sensitivities are finite differences of the smoothed surrogate; gradients
  through contact and candidate action optimization have not been validated.
- Absolute sketch gains are 0.43–0.65 of exact at the default rank and fall as
  $P$ grows; only rankings transfer. See "Sketch accuracy" for the measured
  rank-versus-power-iteration frontier.
- Selection is myopic over a fixed action dictionary; motion shape (speed,
  depth) is not optimised.

Math and assumptions: [`math.md`](math.md). Detailed method and caveats:
[`asid/fim/README.md`](fim/README.md).
