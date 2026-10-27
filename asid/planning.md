# Planning: one information-seeking trajectory per ASID iteration

An ASID iteration selects **one** tool trajectory, executes it, refits the
material parameters, recomputes the Fisher information and plans again. This
document defines the decision space, the objective, the search, and exactly how
the trajectory that MuJoCo replays is produced. The Fisher model itself is in
[math.md](math.md); the open items are in [PLAN.md](PLAN.md).

## Decision space: a spline tool motion

`asid/planning/spline.py`. A trajectory is `K` waypoints in PhysTwin's z-down
world frame, each with a position offset, a time fraction and a claw
orientation:

```
spec = { positions[K,3], times[K] (0 = t_0 < ... < t_K-1 = 1), yaw[K], pitch[K] }
```

With normalized time `u` over the plan horizon and shape-preserving (PCHIP)
splines `S_pos`, `S_yaw`, `S_pitch` through the knots,

```
p(u)   = S_pos(u)          positions off the recorded tool path
psi(u) = S_yaw(u)          yaw about z
th(u)  = S_pitch(u)        pitch about y
R(u)   = Rz(psi(u)) Ry(th(u))
x_i(u) = R(u) (x_i^0 - c) + c + p(u)      for every controller and contact point
```

so the tool is translated **and rotated rigidly** in every frame: the 30
controller points and the contact cloud keep their shape, and the claw can
approach off-axis, rotate while in contact and leave along a different path than
it arrived. That is the whole point of the parameterization — the previous
family could only re-orient the recorded straight stroke on the table.

`times` is a warp of the waypoint positions onto the horizon, so the spacing of
the knots *is* the speed profile: a short first interval and a long second one
means "get there fast, then creep", and repeated waypoints with a long interval
mean the claw dwells. The interior times are searched like any other control; they
are sorted and clipped to `[1e-3, 1-1e-3]` on projection so the spline stays
monotone (`ordered_times`).

Offsets are relative to the recorded push, so a zero spec reproduces the
recorded trajectory exactly (`Spec.start`, covered by a test), and the first
waypoint is pinned to the start pose: the tool is already there.

Serialized controls, the vector CEM actually searches over:

```
v = [ positions.ravel(), times[1:-1], yaw, pitch ]     dims = 6K - 2
```

## Selection: the screen proposes, the exact Fisher decides

The rank-`l` sketch is a *screen*, and its finite-difference protocol differs from
the exact one: it perturbs `phi` along `step * W q` (whitened step `step`), while
`jacobian` perturbs one raw coordinate by `step`. On close candidates that bias
is decisive. Measured at 124 frames, `g = 20`: the sketch ranked
`az270 233.23 > az090 230.75`, the exact Fisher ranked `az090 692.67 > 628.37`,
and the same rank-12 subspace evaluated with the *exact* Fisher reproduces the
exact order (203.79 vs 223.15) — so the inversion is the estimator's step, not
subspace coverage (0 of 200 random subspaces inverted the order at any rank from
8 to 41). The pipeline therefore screens with the sketch, computes the **exact**
Fisher for the best `--exact-top N` candidates, and starts the iteration from the
exact-best of those. Exact gains are cached by `exact_key` (case, config, `phi0`,
trajectory bytes, step, sigma — no rank or seed), and the run writes each
`actions/<name>/fisher.npy` so validation and heatmaps reuse them.

## Objective

The screened action family supplies an initializer, not the answer. For a
candidate trajectory `a`, with the current whitened precision
`H_cur = I + U_cur U_cur^T` (`U_cur` = accumulated factors, empty on the first
iteration), the score is the Woodbury marginal gain of the candidate's own
factor `U_a`:

```
score(a) = logdet(I + U_a^T H_cur^{-1} U_a)
         = logdet(I + U_cur^T U_cur + U_a^T U_a) - logdet(I + U_cur^T U_cur)
```

`U_a` comes from `randomized_sketch`: a rank-`l` Rademacher subspace `Q`, prior-
whitened directions `W Q`, central finite differences of the observation with
step `step`, divided by `sigma`, giving `T = JQ^T JQ` and `U_a = Q V sqrt(Λ)`.
Every candidate is scored at the nominal grouped parameters `phi0` with common
sketch directions, so candidates are compared with common random numbers; the
score is twice the local Gaussian information gain in nats under the sketch and
Gaussian-observation approximations discussed in [math.md](math.md). It is an
ordering criterion, not a calibrated expected information gain.

A factor is **local to one trajectory, body state and parameter point**, so it
is never reused across candidates. What is cached is only what does not change
between them: one `Rollout` instance reused through `ActionRunner`
(`asid/fim/runner.py`), and sketch factors on disk keyed by a hash over the
case, node count, every `SimConfig` field, `phi0`, both trajectory arrays, rank,
step, sigma and seed (`asid/fim/cache.py`, `CACHE_VERSION` invalidates stale
entries).

## Constraints

A candidate is infeasible (`objective` returns `None` and CEM drops it, never a
penalty that could masquerade as a low score) if:

- any tool point leaves the workspace box: the bounds of the object together
  with the recorded tool path, padded by `margin` (default 0.15 m). Object
  bounds alone would reject the demonstration's own approach, which starts half
  a stroke away;
- the per-frame displacement exceeds `max_speed dt`, with `max_speed` taken as
  1.5× the largest per-frame step of the recorded push, so a plan may be more
  dynamic than the demonstration but not arbitrarily fast;
- the closest object point never comes within the contact radius during the
  horizon. The score is *exactly flat* before contact (measured: bit-identical
  observations at 24 frames), so without this guard a claw that misses the object
  ties at the nominal score and floods the elites.

`yaw` and `pitch` are bounded to ±30°, the wrist slew to 1.5 rad/s, and interior
times must keep a minimum gap. The interpolation is shape-preserving, so the
waypoints bound the motion that is actually commanded; the envelope is still
checked on the **sampled per-frame** motion, because a natural cubic does not do
that — measured 0.25 m of offset and 53° of yaw from a bound of 0.05 m and 30°,
which made the diagnostics claim a tighter motion than the plan was. The first
waypoint is pinned in position *and* orientation — at 124 frames an unpinned
wrist started the replay 15.5° / 9.8° off the recorded pose — so the plan departs
from where the tool actually is and only then curves and rotates.

## Search: CEM over the spline

`asid/planning/cem.py`, a diagonal-Gaussian cross-entropy method:

```
mean <- initializer (the best screened rigid action, zero spline)
repeat iterations (default 3):
    evaluate mean and samples (default 8) drawn as mean + std * N(0, I)
    refit mean to the top elites (default 2), std <- 0.7 std + 0.3 std_elites
    clip to bounds, project (sorted times), floor std
```

Elite ranking is scale-free, so the log-determinant score needs no rescaling;
the initializer is always evaluated and always competes, so the plan can never
score below the screened discrete baseline. Cost is `1 + samples * iterations`
rollouts (25 with the defaults) and does not grow with the parameter dimension —
the expensive coordinate count lives in the sketch, not in the search.

Gradient-based search would be preferable and the trajectory parameterization
supports it, but the Warp rollout is built with `disable_backward=True`: node
observations are detached, so there is no differentiating path from tool motion
to the Fisher score. Derivative-free CEM over ~17 controls in one shot replaces
what used to be 33 candidate-rounds of nested sampling.

## Screened initializer

`generate_actions` still builds the azimuth/tilt family — rigid rotations of the
recorded push — and `asid/planning/dictionary.py` scores it with the same sketch.
Its best member initializes CEM, and the CEM result is compared against it in the
run output. The family is therefore an initialization heuristic for "which
direction is worth pushing from", not the planner's decision space.

## Execution and replay

`scripts/plan_push_trajectory.py` writes `trajectory.json`:

```
spec, samples {offsets[T,3], yaw[T], pitch[T]}, start_action, start_gain (sketch),
screen, selection_basis, exact {action: exact log-det}, exact_scores {start, plan,
delta}, score (sketch), history, evaluations, knots, rank, groups, frames, nodes,
basis
```

`exact_scores` is the comparison that matters: the exact Fisher log-det of the
screened initializer and of the CEM plan itself, evaluated at the same `phi0`,
under the same observation model.

`scripts/generate_sim_data.py --trajectory <json>` resamples `samples` onto the
scripted waypoint grid and applies them with `mujoco_sim.interactor.apply_tool_motion`,
so the capture is produced by the ordinary data-generation path: the case's own
model, cameras, 848×480, `capture_every` and fps, and `color/{0,1,2}.mp4` is
directly comparable with the dataset videos.

The frame change is a reflection. With `M = diag(1, 1, -1)`,
`M Rz(psi) Ry(th) M = Rz(psi) Ry(-th)`: positions reflect in z, yaw keeps its
sign, pitch negates, and the claw's base quaternion is composed with
`Rz(psi) Ry(-th)` about the tool origin (`spline.mirror` and
`apply_tool_motion`, both covered by tests — silently mirroring a push the wrong
way would look plausible and be wrong).

The replay is **open loop**: the plan comes from the reduced Warp model at
`phi0`, not from a MuJoCo contact response.

## Cost

A 60-frame, 500-node rollout is 5.15 s and one sketch costs `2l` rollouts, so a
sketch is about 2.1 min at `l = 12`. At the defaults the screen costs 6 sketches
(~13 min, ~4 min once cached) and CEM costs `1 + 8 * 3 = 25` sketches (~52 min).
The camera-ready saving is in the search structure, not in the sketch: one
screen plus one refinement replaces the previous 33 rounds of nested sampling.
Reduce `--samples`/`--iterations` or `--frames` to budget a faster plan; the
cache only ever helps across repeated screens of the same family.

## Files

| Path | Role |
|---|---|
| `asid/planning/spline.py` | Spec, interpolation, rigid sampling, mirror |
| `asid/planning/envelope.py` | Workspace, offset/orientation, slew and contact limits |
| `asid/planning/search.py` | `plan_spline`: CEM over a spec inside an envelope |
| `asid/planning/cem.py` | Bounded CEM with infeasible-sample rejection |
| `asid/planning/actions.py` | Screened azimuth/tilt family and contact checks |
| `asid/planning/dictionary.py`, `select.py` | Sketch factors, marginal-gain selection |
| `asid/planning/measure.py` | `Measure` (gain per candidate) and `Decision` (the pick) |
| `asid/fim/runner.py` | `ActionRunner`: one simulator re-armed per trajectory |
| `asid/fim/cache.py` | `FactorCache`, `ExactCache` and their keys |
| `scripts/plan_push_trajectory.py` | Screen → CEM → `trajectory.json` |
| `scripts/generate_sim_data.py` | `--trajectory` replay through the capture path |

Each concern has one owner: :mod:`asid.planning.measure` decides and records
the gain of a candidate (sketch screen, exact arbitration, the labels a
comparison reads), :mod:`asid.planning.envelope` owns feasibility and the box
the search draws from, :mod:`asid.planning.search` owns the search, and
:mod:`asid.fim.runner` owns rollout reuse. The scripts only parse arguments,
wire those together and write artifacts.

## Not claimed

- The score is not a calibrated expected information gain, and it is evaluated at
  `phi0` unless `--current-factors` is supplied.
- CEM returns the best evaluated feasible trajectory, not a global optimum.
- No sim2sim contact match: the Warp model is a reduced spring-mass graph and the
  replay is open loop.
- `plan_sequential.py`, `plot_sequential.py` and `asid/planning/sequential.py`
  remain as offline set-selection studies and are not part of the loop.
