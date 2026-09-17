# ASID Integration Plan

## 1. Objective

Integrate active system identification (ASID) with the existing Warp/PhysTwin and MuJoCo pipeline so that the system can select interactions that reduce uncertainty over physical parameters.

The intended simulated loop is:

```text
MuJoCo ground-truth environment
        |
        | execute selected action and render RGB-D
        v
RGB-D preprocessing and tracking
        |
        | estimate observed object motion
        v
Warp/PhysTwin parameter estimator
        |
        | update physical parameter belief
        v
Warp/BOBA differentiable planner
        |
        | select an informative action
        v
MuJoCo ground-truth environment
```

MuJoCo is the ground-truth environment for the initial project scope. Warp/PhysTwin is the learned differentiable simulator used to predict outcomes and plan experiments. BOBA provides the fast batched Warp execution needed for evaluating candidate actions and sensitivities.

The first implementation should be chunk-based: execute an interaction chunk, update the physical model, then select the next chunk. Streaming estimation can be considered later.

## 2. Current Components

The repository currently provides the following pieces:

- MuJoCo scene generation and RGB-D capture.
- Scripted interactor trajectories such as object pushes.
- PhysTwin-style RGB-D preprocessing and point tracking.
- Warp spring-mass simulation.
- CMA initialization and differentiable Warp parameter training.
- BOBA/Boba batched and accelerated runtime paths.
- Gaussian rendering and runtime visualization.
- Synthetic rigid, rope, and asymmetric-mass box assets.

The following pieces are not yet connected:

- Online action selection.
- A persistent physical-parameter belief.
- Analytical information-gain computation.
- A headless batched planning API.
- Automatic execution of planner-selected trajectories in MuJoCo.
- Incremental processing and incorporation of new observation chunks.
- Articulated claw control and pickup actions.

The current `interactive_playground.py` primarily replays recorded controller trajectories. It is not yet an ASID or MPPI planner.

## 3. Parameter Representation

Warp may contain tens of thousands of local parameters, for example one stiffness value per spring. These parameters should remain available in the simulator, but the information objective must avoid constructing an unnecessarily large dense covariance matrix.

Use optimization-friendly physical parameters, preferably with positivity-preserving parameterizations:

```text
log_spring_stiffness
log_damping
friction_parameterization
collision parameters
```

The current training code already uses log stiffness in parts of the Warp simulator, so the active-identification parameterization should follow the same convention.

The initial estimator can maintain:

```text
parameter estimate:      theta
parameter uncertainty:   diagonal variance Sigma_diag
```

Later, correlated uncertainty can be represented with a low-rank approximation.

Although all local parameters are retained, nearby parameters may be difficult to identify independently. The system should therefore also support structured parameter views:

- spatial spring clusters,
- object-part groups,
- low-frequency spatial bases,
- graph Laplacian modes,
- learned low-rank material modes.

These structured views should be used for reporting and regularization even if the underlying Warp parameters remain local.

## 4. Initial Real-to-Sim Calibration

The first interaction is a fixed, minimally informative trajectory, such as:

```text
approach -> push -> retract
```

The existing synthetic pipeline can generate this trajectory with MuJoCo and export:

- RGB frames,
- depth frames,
- camera intrinsics,
- camera-to-world poses,
- frame metadata.

The RGB-D pipeline produces tracked object/controller data in `final_data.pkl`.

The initial Warp calibration then estimates parameters by fitting simulated motion to observed motion:

```text
find theta such that
    Warp(state_0, action_chunk, theta)
    ~= observed_object_motion
```

This produces an initial parameter estimate. It is not a learned action policy and does not yet perform active exploration.

## 5. Analytical Information Gain

For an action sequence `u`, let the differentiable Warp simulator predict an observation vector:

```text
y(u, theta)
```

The parameter sensitivity is:

```text
J(u) = d y(u, theta) / d theta
```

If `R` is the observation-noise covariance, the local Fisher Information Matrix is:

```text
F(u) = J(u)^T R^-1 J(u)
```

No sampling of the full parameter space is required. The first implementation should use the diagonal Fisher approximation:

```text
F_ii(u) = sum_k J_ki(u)^2 / R_kk
```

With a diagonal prior parameter variance `Sigma_ii`, an approximate per-parameter uncertainty reduction is:

```text
gain_i(u) = 0.5 * log(1 + Sigma_ii * F_ii(u))
```

The total information score can be:

```text
IG(u) = sum_i weight_i * gain_i(u)
```

The weights can prioritize:

- currently uncertain parameters,
- parameters in a selected object region,
- parameters relevant to a downstream manipulation task,
- parameters that have not yet been sufficiently excited.

The diagonal approximation is scalable, but it does not represent parameter correlations. After the diagonal implementation is validated, add low-rank or matrix-free Fisher approximations where needed.

### Information objectives to support

- **D-optimal:** maximize `log det(I + Sigma F)`.
- **A-optimal:** minimize the trace of posterior covariance.
- **E-optimal:** reduce the largest remaining uncertainty direction.
- **Diagonal gain:** sum per-parameter variance reductions.
- **Task-weighted gain:** prioritize parameters that affect a target behavior.

The initial implementation should use diagonal gain because it directly supports tens of thousands of parameters and produces interpretable per-spring information maps.

## 6. Computing Sensitivities Efficiently

Do not materialize a dense `parameter_count x parameter_count` Fisher matrix.

Candidate implementations:

### Jacobian blocks

Compute observation Jacobian blocks and accumulate squared sensitivities:

```text
F_diag += sum(J_block^2 / observation_variance)
```

### Forward-mode JVPs

Use Jacobian-vector products when the parameter dimension is large and the observation dimension is relatively small.

### Reverse-mode VJPs

Use vector-Jacobian products to obtain:

```text
J^T v
```

for observation probes and accumulate diagonal estimates.

### Low-rank sensitivity directions

Use a small number of dominant directions to approximate:

```text
F ~= V Lambda V^T
```

This is useful when local parameters are strongly correlated.

The BOBA forward path is optimized for fast inference and rendering. ASID requires a separate headless planning path that preserves Warp autodiff state or exposes efficient JVP/VJP operations. CUDA graph capture and differentiable tape usage must be tested separately; a fast forward graph is not automatically a differentiable planning graph.

## 7. Headless Warp Planning API

Add a planning-oriented interface separate from the interactive renderer:

```python
predicted_states = planner.rollout_batch(
    initial_state=state,
    action_batch=actions,
    parameter_batch=parameters,
    horizon=horizon,
)
```

Recommended tensor shapes:

```text
action_batch:      [B, H, C, action_dim]
parameter_batch:   [B, parameter_dim]
predicted_states:  [B, H, N, 3]
predicted_observations: [B, H, observation_dim]
```

Where:

- `B` is the number of independent candidate rollouts.
- `H` is the fixed planning horizon.
- `C` is the number of controllers.
- `N` is the number of simulated object nodes.

Every candidate must start from the same initial state unless the planner explicitly requests otherwise. State buffers must be reset between candidate rollouts.

Initially, the planner should operate on object-node states or compact point-cloud features rather than rendered RGB images. Rendering can be added later when image-space objectives are required.

## 8. Action Representation

For pushing, use a bounded incremental action:

```text
action_t = [dx, dy, dz, droll, dpitch, dyaw]
```

For an articulated claw, extend it with:

```text
action_t = [dx, dy, dz, droll, dpitch, dyaw, gripper_opening]
```

Bound each action to prevent unstable or unrealistic experiments. For example:

```text
translation: a few millimeters per control step
rotation:    a few degrees per control step
gripper:     joint-limit constrained
```

The planner should output an action chunk or trajectory that can be converted into the existing MuJoCo interactor trajectory format.

## 9. Action Optimization

MPPI is not required for parameter information computation. It samples action sequences, not physical parameter vectors, so it may remain useful when the action space is low-dimensional and contact dynamics are non-smooth.

However, because Warp is differentiable, the first planner should also support direct action optimization:

```text
u* = argmax_u IG(u)
```

A gradient-based planner can:

1. initialize an action sequence,
2. roll out Warp,
3. compute parameter sensitivities,
4. compute the information score,
5. differentiate the score with respect to actions,
6. update the action sequence,
7. project actions back into safety limits.

MPPI or CEM can later be compared against this method for robustness around contacts and local optima.

A combined objective can include both task and information terms:

```text
score(u) = task_reward(u)
           + information_weight * IG(u)
           - control_cost(u)
           - collision_penalty(u)
```

For pure system identification, task reward may be omitted or replaced with a safe-contact constraint.

## 10. MuJoCo Execution Loop

MuJoCo executes the action selected by Warp. The current MuJoCo wrapper already supports direct mocap pose updates and scripted trajectories.

The active loop should be:

```python
belief = initialize_belief()

for episode in range(num_episodes):
    action = planner.select_action(belief)

    rgbd_chunk = mujoco_env.execute_action(action)

    observation = observation_processor.process(rgbd_chunk)

    belief = estimator.update(
        belief=belief,
        action=action,
        observation=observation,
    )
```

MuJoCo should use hidden ground-truth parameters. The estimator must only receive the same observations that would be available from the camera pipeline. Ground-truth object states can be recorded for debugging and evaluation, but must not be used by the policy or estimator during normal operation.

## 11. Incremental Estimation

After each action chunk:

1. Process RGB-D frames.
2. Segment the object and controller.
3. Track object points.
4. Convert observations to the Warp estimator format.
5. Append the new chunk to the accumulated dataset.
6. Warm-start parameter optimization from the previous estimate.
7. Update the parameter uncertainty.
8. Recompute the next information-seeking action.

The first implementation can refit all accumulated chunks. Later, use a sliding window or streaming estimator.

A simple update is:

```text
previous estimate -> optimize on previous + new chunk -> updated estimate
```

The estimator should report:

- parameter estimate,
- per-parameter uncertainty,
- per-parameter information gained from the latest chunk,
- fit error on the latest chunk,
- fit error across all chunks.

## 12. ASID Belief Options

### Initial approach: diagonal Laplace approximation

Maintain:

```text
mean theta
variance per parameter
```

After estimating the parameters, approximate curvature using the Gauss-Newton or Fisher diagonal.

### Ensemble approach

Maintain several parameter vectors around the current estimate. This is useful for nonlinear uncertainty, but it should not be the primary method for handling tens of thousands of independent samples.

### Later approach: low-rank covariance

Maintain a small number of uncertainty directions for correlated parameters:

```text
Sigma ~= D + U U^T
```

This is a practical compromise between a diagonal approximation and a dense covariance matrix.

## 13. Success Metrics

Measure both estimation quality and information efficiency:

### Parameter estimation

- error in global stiffness,
- error in local stiffness fields,
- friction error,
- damping error,
- mass or center-of-mass error where modeled.

### Predictive quality

- object trajectory error,
- point-cloud Chamfer error,
- contact prediction error,
- held-out action prediction error.

### Active exploration

- information gained per episode,
- uncertainty reduction per frame,
- number of episodes to reach a target error,
- action safety violations,
- compute time per planner decision.

### Manipulation quality

- push displacement error,
- grasp success rate,
- pickup success rate,
- object drop rate,
- final pose error.

## 14. Recommended Milestones

### Milestone 1: Fixed-action calibration

- Generate one MuJoCo interaction chunk.
- Fit Warp parameters.
- Compare Warp prediction to MuJoCo ground truth.
- Validate parameter error and trajectory error.

### Milestone 2: Differentiable sensitivity test

- Select one fixed action.
- Compute Warp sensitivities to every spring parameter.
- Produce a per-spring Fisher/information map.
- Compare analytical sensitivities against finite differences on a small test case.

### Milestone 3: Action-dependent information score

- Evaluate several manually specified actions.
- Confirm that different actions emphasize different physical parameters.
- Validate that the predicted high-information action actually reduces estimation error in MuJoCo.

### Milestone 4: Batched headless BOBA planner

- Add independent candidate action batches.
- Reuse fixed-shape CUDA graphs where compatible.
- Verify that candidate states do not interfere with one another.
- Benchmark throughput and memory.

### Milestone 5: Gradient-based action optimization

- Optimize short action sequences directly against the information score.
- Add collision, workspace, and smoothness constraints.
- Compare against random actions and manually designed actions.

### Milestone 6: Closed-loop ASID

- Execute the selected action in MuJoCo.
- Process the resulting RGB-D chunk.
- Update parameters and uncertainty.
- Repeat for multiple episodes.

### Milestone 7: MPPI/CEM comparison

- Add MPPI or CEM as an alternative action optimizer.
- Compare robustness and compute cost against gradient-based planning.
- Use MPPI only if it provides a measurable benefit for contact-rich action optimization.

### Milestone 8: Articulated pickup

- Add finger joints and actuators.
- Add gripper opening to the action interface.
- Add contact and pickup-success signals.
- Train and evaluate pickup only after pushing-based ASID is stable.

## 15. Main Risks

### High-dimensional parameter correlations

Diagonal Fisher information may overstate identifiability when neighboring parameters have similar effects. Use spatial grouping or low-rank methods when this appears.

### Differentiability through contacts

Contact transitions can produce noisy or discontinuous gradients. Use short horizons, smooth contact settings, gradient clipping, and compare gradients with finite differences.

### BOBA graph limitations

Fixed CUDA graphs may not support all dynamic action, parameter, or tape configurations. Maintain a separate correctness-first planning path before optimizing it with CUDA graphs.

### Observation mismatch

The information score is only useful if Warp predictions and processed MuJoCo observations use the same representation. Start with tracked object points before adding rendered images.

### Dataset drift

The preprocessing pipeline must preserve consistent camera frames, coordinate conventions, segmentation labels, and controller trajectories across chunks.

### Overfitting to one action type

Use a curriculum of pushes, presses, sliding contacts, and later lifts. Randomize object pose, friction, damping, stiffness, and mass distribution during evaluation.

## 16. Initial Recommendation

The first active-identification prototype should use:

```text
full local Warp parameter vector
log-parameterization for positive quantities
diagonal Fisher information
per-parameter uncertainty estimates
headless differentiable Warp rollouts
short-horizon gradient-based action optimization
MuJoCo as the hidden-parameter ground-truth environment
```

Do not begin with a dense covariance matrix or tens of thousands of sampled parameter hypotheses. First verify that analytical sensitivities identify meaningful physical regions and that actions predicted to be informative actually reduce parameter error after execution in MuJoCo.

BOBA remains central: its fast batched forward execution makes it practical to evaluate candidate actions and sensitivity calculations. The main required extension is a planning API that supports independent candidate action sequences, parameter gradients, state resets, and headless operation.
