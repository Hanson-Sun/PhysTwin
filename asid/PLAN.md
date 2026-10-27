# ASID Review and Planning Implementation

## Working plan and orientation — 2026-10-26

**Status:** Phase 1 review is committed. The Gaussian/grouped-FIM derivation is documented in `math.md`; the separate adjoint experiment was removed. Focused Fisher, observation, visualization and Warp tests pass in WSL's `phystwin-cu132` environment, and a small real-case run produced matrix and per-node gain figures. Phase 2 prototype includes sampled/gradient comparison, a translation-only MuJoCo adaptor, and a reproducible coarse sloth FIM-to-MuJoCo smoke test. Reliable contact gradients and calibrated sim2sim contact/observation alignment remain open; see `planning.md`. Phase 3 must not claim a full loop until incremental observation processing and warm-start training are demonstrated. Use an interactive WSL bash shell to load conda initialization. The original request and proposal remain below as the source of requirements.

### Premise

Physical RSI closes a chunk-based loop: use an estimated PhysTwin/Warp model to choose an informative interaction, execute it in MuJoCo, use the observed motion to update the model, then repeat. The first target is simulated deformable-object pushing, not real-world robotics or streaming inference. Information estimates must reflect the actual observation and uncertainty model rather than assume that all visual points or parameters are independently identifiable.

### Repository findings (inspection, not yet verification)

- The checkout is on `PRSI/2-asid-integration`, not a branch literally named `asid`. Relative to `PRSI/1-phystwin-real2sim`, three committed changes concern branch conventions, push cases, and environment setup; the ASID implementation is predominantly untracked. Preserve the current branch and pre-existing work; confirm ownership and commit scope before each requested commit. Do not push. The proposal is included below; no separate `proposal.md` was found in the workspace.
- `asid/fim/operators.py` flattens selected node positions and computes parameter sensitivities by finite differences. `asid/fim/fim.py` forms `F = J J^T / sigma^2` with parameter indices in rows, and `sketch.py` approximates a prior-whitened FIM in a random subspace. The grouped log parameters are object-spring stiffness, one dashpot, and node mass; friction is not currently part of this FIM.
- These are trajectory-observation sensitivities, not explicitly sampled likelihood-score outer products. Their equivalence to a Fisher matrix depends on the Gaussian observation assumptions, fixed noise, and correct treatment of aggregation. Frame selection, `reduce="mean"`, parameter coordinates, and observation correlations all need explicit review. Current gain code reports `logdet`, not the half-logdet in Gaussian mutual information.
- The currently modified tracked Warp simulator includes deterministic spring forces and soft-ground contact used by the ASID harness. Review the entire diff and its relevance before staging; do not assume all pre-existing changes belong in the commit.
- The separate adjoint experiment and exclusive scripts/tests were removed as agreed; the full grouped finite-difference matrix remains as a reference. The existing contact-adjoint issue is not independently verified here, and the ordinary Warp training path is retained.
- The original `asid/fim/viz.py` and `asid/planning/fimmap.py` map **grouped parameter-FIM diagonals** to springs/nodes. A 100-node, 2-group, 6-frame sloth smoke run also produced a grouped FIM matrix plot and standalone per-observation-node local Gaussian information-gain map; the resulting values were finite and nonnegative. This is a small execution check, not a full-scale validation.
- Current planning is a prototype: `actions.py` rotates a recorded trajectory; `plan_actions.py` greedily selects a fixed dictionary; `plan_sequential.py` updates sketch precision and a simulated deformed position while keeping parameters fixed. It uses the same sketch subspace across rounds, and starts carried positions at rest. Replace the fixed-dictionary-only selection and simulated-only execution/parameter handling for the actual planner; retain reusable actions, scoring, or visualization only when validated. No MuJoCo execution adaptor or learning loop is wired into `asid/`.
- `mujoco_sim/interactor.py`, `mujoco_sim/simulation.py`, and `mujoco_sim/phystwin_export.py` provide trajectory, execution, and RGB-D export building blocks. The trainer saves parameter and optimizer checkpoints, but the displayed checkpoint restore is for inference; warm-started continued training and tracked-observation continuity still need proof.

### Large-parameter design constraint

The underlying model may have huge numbers of spring/node parameters; the first working method must scale with a **much smaller, explicitly chosen spatial grouping/basis**, not with the full dense physical-parameter FIM. Preserve dense local simulator parameters through the existing sparse group-to-parameter mapping, but estimate uncertainty only for the chosen grouped coordinates. Grouping is a modeling assumption: within-group contrasts and other omitted modes are not identified, so a small grouped posterior variance does **not** imply the full field is known.

Let the full log-parameter field be `eta = eta0 + B z`, with `z` the grouped coefficients and `B` a sparse spatial map. For observation Jacobian `J_eta`, the grouped FIM is `F_z = B^T J_eta^T R^-1 J_eta B`, obtained from sensitivities with respect to `z` without forming a full-space FIM. If working in physical rather than log coordinates, include the exponential chain rule. Define the prior on `z` and the normalization of `B` explicitly, so comparisons do not change arbitrarily with group size or units. The existing `Partition` mapping is the starting point; verify how it handles different nominal values and parameter families.

For a small enough grouped dimension, a dense grouped FIM is the simplest reference, **not** a license to materialize a dense matrix over all springs/nodes. If the grouped dimension itself becomes large, use the validated directional finite-difference sketch and factor/solve-based scoring; account for storage and rollout costs as functions of group count and sketch rank. Matrix-free FIM actions, Krylov/low-rank logdet estimates, or observation-space determinant identities are future options only if the required JVP/VJP and covariance operations are verified; do not assume Warp's current contact adjoint supplies them. A fixed sketch is not guaranteed accurate or equivalent to Nyström, and the documented shared-subspace cap matters for sequential uncertainty.

Validate the **scientific grouping** separately from numerical sketch accuracy: compare representative nested/coarser and finer groupings and selected within-group contrast directions; check that candidate rankings, posterior summaries, and held-out predictions are stable enough for the intended use. For close candidate scores, repeat with higher rank or independent seeds and use a higher-fidelity grouped reference where affordable; if still unresolved, report the ordering as uncertain. Record actual rollout cost, memory, and dimensionality, without claiming scalability or speedups that were not measured. Start with one principled working configuration, not a framework of speculative alternatives.

### Phase 1 — review, cleanup, and first commit

1. **Specify and verify the likelihood and FIM.** In `asid/math.md`, derive the score and expected FIM for `y = h(phi, u) + epsilon`, `epsilon ~ N(0, R)` with parameter-independent covariance: `score = J^T R^-1 (y - h)` and `F = J^T R^-1 J`, using observation-by-parameter `J` and explicitly mapping it to the transposed code convention. Cover the prior/log coordinates, units, frame and node sums, independent versus correlated observations, stride, and how averaging frames changes covariance. Distinguish simulated object nodes from samples over uncertain parameter hypotheses; do not call a deterministic nominal-point FIM an expectation over parameter beliefs. Describe the conditions under which a half-logdet is an approximate Gaussian expected information gain and distinguish it from a ranking-only logdet score.
2. **Focused correctness gates.** Compare against a small analytic linear-Gaussian model (score, FIM, scaling with noise and repeated independent samples, correlations, and mean reduction); verify finite differences, symmetry/PSD, the grouped pullback and prior whitening. Check grouped-space versus full-space interpretation on a tiny synthetic example, plus sketch/rank stability on a manageable reference. Fix causes of discrepancies and record the observation assumptions and any approximations. Use only tests small enough for the runtime rule.
3. **Remove the adjoint path.** The exclusive experiment, scripts, tests and harness wiring have been removed; ordinary training and the finite-difference FIM reference remain. Re-run targeted checks and search for dangling references.
4. **Visualization gate.** Reuse existing saved artifacts if suitable, or generate a small fixture. Confirm execution of a parameter-FIM matrix visualization and object-mapped per-point information. Define the latter explicitly (e.g. each observed point's contribution to a stated scalar criterion, with time/noise aggregation); label grouped parameter diagonals separately. Verify outputs are finite, interpretable, and linked to the same observation/model assumptions. Record commands and outputs.
5. **Planning audit and commit.** Record in `asid/planning.md` what the prototype does and what Phase 2 replaces, without claiming it is a solved continuous optimizer. Review staged and unstaged changes and recent commit style; stage only relevant, owned Phase 1 hunks and files, commit without AI/tool attribution, and do not push. If ownership or the branch is ambiguous, stop and ask rather than staging unrelated work.

**Exit criteria:** math-to-code mapping and focused tests agree; intended adjoint removal has no dangling references; both visualizations actually run and have defined semantics; current planning limitations are documented; Phase 1 changes are committed.

### Phase 2 — trajectory planning and second commit

**Prototype and sim2sim limitation:** A sampled translation-knot planner and a gradient-consistency gate are implemented; the contact-case check selected the sampled result after rejecting an unstable gradient. A focused MuJoCo test executes a selected synthetic trajectory with z reflection and physics-step timing. A sloth smoke run scores three grouped-FIM candidates and sends the winner to MuJoCo, using two pad centres as coarse tool geometry; this is not a calibrated contact or initial-state match. Actual recorded sloth controller tracks are not translation-rigid (first-six-frame maximum residual 0.127 m for sparse controller points and 0.182 m for dense contact points), so the adaptor rejects them. `gmsh` was installed into `phystwin-cu132` for the soft scene. A calibrated execution needs matching tool surface, world pose, and object state; do not approximate the tracked cloud as a rigid trajectory without validating alignment.

**Derivative probe, 2026-10-26:** The current rollout converts action arrays to fresh tensors, controller/contact state arrays have gradients disabled, controller target copies disable backward, and rollout observations are detached. Thus end-to-end action autodiff is not wired up. A small GPU probe used the sloth case with 100 nodes, 2 spatial groups, 6 frames starting near recorded frame 64, 100 substeps, a rank-2 finite-difference FIM sketch, and a rigid x-offset as the action variable. Central differences of the complete half-logdet score produced derivatives of 579, 1497, 712 and 6836 nats/m at steps 2, 1, 0.5 and 0.25 mm respectively; moving the controller 1 m away gave zero for the 2 and 1 mm checks. Runs completed successfully in WSL; logs are `logs/asid_action_gradient_probe.log` and `logs/asid_action_gradient_smallsteps.log`. This is one short restarted contact window, not a proof that all contact gradients fail. It shows the current full-objective finite-difference gradient is not reliable on this test. Do not claim a validated gradient planner; ask whether to repair the derivative/contact path or use a sampled optimizer before proceeding.

1. **Agree on the prototype contract.** Use one short, bounded pushing trajectory parameterized by a few control waypoints, with a fixed initial state and explicit timing, workspace, speed, and contact constraints. Inputs must include current parameter estimate/belief, existing information or prior precision, observation/noise model, and state; output includes a feasible trajectory, score, and optimizer diagnostics. Define precisely how Warp controller/contact points map to MuJoCo poses, coordinate reflection, timing and observations. Select the best found trajectory, not a guaranteed global optimum.
2. **Choose a consistent local objective.** Start with prior/posterior-conditioned D-optimal gain `0.5 * (logdet(P + H(u)) - logdet(P))` in whitened coordinates for positive-definite precision `P`, using stable factorizations. The factor `0.5` is required when calling it Gaussian information gain. Define a finite set of current parameter hypotheses and weights if approximating an expectation over beliefs; otherwise clearly label a nominal local approximation. Avoid conflating observed FIM, prior precision, and predicted candidate information or double-counting data; validate whether and how past information is relinearized at a new parameter estimate. Keep A-optimal/Schur alternatives as documented comparisons, not extra initial features.
3. **Prototype candidate baseline first.** Reuse the validated finite-difference/sketch machinery for a small, reproducible set of feasible trajectories; common random directions can reduce comparison noise, but the documented shared-subspace accumulation cap must not masquerade as posterior certainty. Compare against a small dense *grouped* reference and test the effect of group resolution and sketch rank on close candidates. Keep full-parameter storage out of the FIM path; record group count, rank, rollouts, memory, action scores, and elapsed time without asserting a speedup.
4. **Gate gradient optimization on actual derivatives.** Direct differentiation of a parameter-sensitivity FIM with respect to trajectory controls generally needs mixed parameter/action derivatives. Current rollouts detach to NumPy/CPU, and the existing contact adjoint has documented problems. On a tiny contact-free and contact-active case, test whether the chosen route yields finite, reproducible action gradients matching independent directional finite differences of the *whole objective*. Candidate routes include differentiable parameter sensitivities if supported or finite-difference parameter sensitivities combined with valid first-order action derivatives. If neither passes, ask before settling for derivative-free search; preserve the sampled baseline and document the blocker rather than presenting it as gradient optimization.
5. **Optimize and execute.** Initialize from multiple feasible sampled candidates, run the validated gradient method with projection/constraints, include the original seeds in the final comparison, and return the best feasible result. Adapt that exact trajectory to the existing MuJoCo interactor, preserving spatial conventions, timing, and the actual executed controls. Test a minimal MuJoCo sim2sim trajectory and observation capture, including mismatch/failure handling.
6. **Document and commit.** Complete `asid/planning.md` with the likelihood, objective, belief approximation, optimization, initialization, constraints, action mapping, limitations, and reproducible commands. Focused tests cover objective conditioning, optimization versus initial candidates, gradient checks, and adaptor timing/frame conversion. Review and commit only Phase 2 work, without attribution or push.

**Exit criteria:** planner scores multiple actions and selects the best found; validated gradients improve or retain the best sampled candidate; a selected trajectory executes in MuJoCo with aligned outputs; planning math and limitations are documented; Phase 2 is committed. If derivatives cannot meet the gate, surface that limitation and seek a decision instead of claiming completion.

### Phase 3 — iterative learning loop and third commit (after Phase 2)

1. Provide one configuration-driven entry point and a per-iteration result containing the updated Warp model/parameters, updated information representation, and next trajectory. Bootstrap from an initial model, parameter belief, state, and data; specify first-action behavior.
2. Execute the selected trajectory in MuJoCo, capture RGB-D and actual controller motion, and route observations through the existing preprocessing/tracking and training interfaces. Ensure units, camera/world frames, point correspondence, and state/velocity continuity are explicit; do not leak MuJoCo ground truth to the estimator. If full preprocessing cannot yet support incremental chunks, expose that as a blocker rather than silently substituting ground truth.
3. Warm-start training from the prior parameters; carry optimizer state where the existing trainer supports it, explicitly verify checkpoint restoration, and retain prior chunks or an equivalent principled continual-learning objective. Keep the FIM, parameterization, and updated model aligned; avoid counting the same observations repeatedly when refitting. Recompute the next plan from the updated state and belief.
4. Define convergence in config using normalized parameter change over consecutive iterations **and** a fit/identifiability check, plus maximum-iteration and failure stops. A small parameter change alone can mean non-identifiability, and FIM-implied covariance is only approximate. Persist diagnostics and artifacts for reproducibility.
5. Run a minimal focused loop test that demonstrates warm-start, fresh observations, FIM/model update, next action and stopping; record longer realistic commands separately for the user. Commit only Phase 3 work, with no attribution or push.

**Exit criteria:** one config starts the automated loop; each iteration provides the three requested outputs; continuation and stopping are demonstrated; limitations are explicit; Phase 3 is committed.

### Research basis and engineering decisions

- [ASID paper](https://arxiv.org/html/2404.12308v2) and [official code](https://github.com/WEIRDLabUW/asid): A-optimal `tr(F^-1)` and exploration under parameter uncertainty; the paper uses policy optimization and permits finite-difference sensitivities. Do not assume direct action gradients are part of ASID.
- [SPI-Active paper](https://arxiv.org/html/2505.14266v1) and [repository](https://github.com/LeCAR-Lab/SPI-Active): sampling-based parameter and command optimization with an A-optimal criterion; repository availability of the full active method must be checked before relying on it.
- [Schur-complement trajectory design](https://arxiv.org/html/2610.10905v1): separate excitation from parameter coupling; useful if aggregate scores hide confounding. [Fisher trajectory synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4293777/) and [contact-aware Fisher design](https://arxiv.org/html/2505.12214v1/) illustrate alternative criteria and contact assumptions.
- [Constrained SVGD system identification](https://arxiv.org/html/2109.08815v2) and [original SVGD](https://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm): posterior inference, not itself an action objective; defer multi-particle inference unless the simpler belief proves inadequate.
- [Bayesian optimal experimental design review](https://arxiv.org/html/2005.12998v3) and [randomized matrix-free log-determinant estimators](https://arxiv.org/html/1605.04893v2): prior-preconditioned, low-rank/operator methods can scale when the spectrum and available derivatives permit; approximation quality and omitted uncertainty must be checked, not assumed.
- [PhysTwin](https://github.com/jianghanxiao/phystwin), [Boba](https://github.com/jianxiapyh/Boba-Public), and [PhysTwin-MPC](https://github.com/Jianghanxiao/PhysTwin-MPC): reuse existing code where appropriate; batching or differentiable physics alone does not prove action/FIM-gradient support. Do not add submodules unless they demonstrably save work.

### Operating rules and open decisions

- Run all development and scripts in WSL, using an interactive WSL bash shell to initialize conda and `conda run -n phystwin-cu132` for Python/scripts. Cap individual commands/processes below two minutes, use separate background processes for blocking work, and write genuinely longer commands in an MD file for the user to run; do not leave a background job running past the limit. Prefer tiny focused checks.
- Preserve the pre-existing work and branch. No push, no tool/agent signatures in commits, no unrequested dependencies or submodules.
- The user agreed to remove the separate reverse-adjoint experiment while retaining the grouped finite-difference reference.
- **Before implementing the planner:** agree on the first MuJoCo case and actionable trajectory bounds if existing case geometry and controls do not determine them. If valid contact action gradients cannot be established, present the alternatives and ask before changing the gradient-first requirement.

---

## Original request and proposal

# ASID Review and Planning Implementation

## Context
We recently implemented the FIM estimation component of Physical RSI. The proposal is attached
(`proposal.md`). Development is in WSL; use conda env `phystwin-cu132` for all scripts.
The work lives on the `asid` branch, in the `asid/` folder.

## Phase 0: Orient (do in parallel with subagents)
1. Read the proposal and summarize the premise in a few lines.
2. Review the `asid/` folder and the FIM estimation code.
3. Review the uncommitted and branch changes on `asid` to see what is implemented.

## Phase 1: Review and cleanup
Verify or fix each item, then commit the ASID additions (do not push).

1. **FIM correctness.** I'm unsure the FIM estimation is right.
   - Check the math (score definition, Gaussian-noise reduction, scaling, aggregation over time/particles).
   - Write `asid/math.md` explaining the math and how each piece maps to code.
2. **Remove the "exact path".** I think it's overkill. Delete it and all related code, tests and docs.
3. **Visualization.** Confirm it runs and can show (a) the FIM and (b) per-point information gain
   mapped onto the object.
4. **Current planning code.** I believe it's only preliminary test code. Confirm, and list what
   should be replaced.

## Phase 2: ASID trajectory planning
Implement a planner that takes the current FIM and returns the optimal action/trajectory.

- Evaluate multiple candidate actions and select the one with the highest expected information
  gain (EIG), using a scalar criterion of the FIM (e.g. tr(F⁻¹) per ASID, or logdet / Schur-complement
  variants; see sources).
- Reference approach is sampling-based (MPPI), which may be too slow. Since the scene is fully
  differentiable, prefer **gradient-based optimization of the action trajectory** against the FIM
  objective, with sampled candidates as initialization or as a baseline.
- Provide an adaptor that executes the trajectory in MuJoCo (sim2sim).
- Deliver `asid/planning.md` documenting the planning math and exactly how the optimal trajectory
  is computed.
- Commit (do not push).

## Phase 3: Iterative learning loop (only after Phase 2 is done)
Each iteration returns: (1) updated Warp model, (2) updated FIM, (3) next optimal trajectory.

- Loop: plan → execute in MuJoCo → collect data → update the model → recompute FIM → repeat until
  parameters converge (define a convergence criterion, e.g. FIM-implied covariance or parameter change).
- Warm-start from the previous parameters (online/continual updating, not retraining from scratch).
- Ergonomic and fully automated: a single entry point with a config.
- Commit (do not push).

## Research (use subagents; cite sources in the docs)
Do not reinvent the wheel. Base the design on existing methods and code. Starting points
(see the source list I've provided): ASID (FIM objective), SPI-Active, Schur-complement
trajectory design, Fisher-based exploration, SVGD for system identification, and the
PhysTwin/Boba codebases. Pull relevant repos as submodules only if they save real work.

## Working rules
- Concise, clean, idiomatic Python; simplest principled solution. Ask me when a design choice is unclear.
- Keep tests minimal and focused on core functionality.
- Don't run anything over 2 minutes. Put long-running commands in an MD file for me to run later;
  run blocking tasks in separate processes (async).
- Parallelize with subagents where possible.
- No tool/agent signature in commit messages.
- DO NOT REINVENT THE WHEEL. base assumptions and ideas on existing work and code. If you are unsure, ask, or do web research with subagents.
- try to iterate quickly and get a working prototype first, then optimize and clean up later.
- keep and maintain a clear and detailed plan so you never lose track of the goal/


# PROPOSAL: Physical RSI: Active Information-Seeking for Self-Improving Robotic World Models

## Introduction
Estimating physical properties of unknown objects (mass, friction, stiffness, damping) from visual data fundamentally requires physical interaction. While there has been recent work that converts video into interactive 3D simulations (*PhysGaussian*, *PhysTwin*), these methods are **passive**: they require a human to pre-record a video of interactions, with no active feedback loop and no way for the model to decide it needs more data. Conversely, robotic **Active System Identification** (e.g., *ASID*) closes part of that loop with exploratory pokes, but relies on objects with known parametric shapes and treats visual reconstruction (where used at all) as a separate, fixed preprocessing step rather than an integrated differentiable representation the robot can act on and update.

Neither of these is really a *self-improving* system in the sense we want: a robot that builds its own world model, acts to find the parts of that model it's most uncertain about, updates the model from what it observes, and repeats. We want something that closes the loop between "what I believe about the world" and "what I do to test that belief" without a human in it. Essentially, this is the physical analogue of recursive self-improvement (RSI). Instead of a model improving its own weights or code, here the robot is improving its own *world model* of a physical object, and using that improved world model to choose better actions, which in turn produce better data to improve the world model again.

This project proposes an end-to-end framework that unifies real-to-sim reconstruction and active information-seeking control into exactly that loop. The system should execute targeted physical interactions (e.g., pokes/presses), and optimize physical parameters across *episodic interaction chunks*, using each round of interaction to eliminate physical uncertainty and sharpen the world model for the next round. If possible, the system should also be simulation-agnostic, allowing it to be applied to any black-box real-to-sim model without requiring simulator-specific modifications.

## Implementation Details

To restrict the scope of the project, we will design the pipeline to work in a simulated environment, which would allow us to generate large amounts of good training data without real-world problems.

The core components of the project already exist in isolation: real-to-sim pipelines, differentiable physics simulation, and active exploration. Ideally, we would build on top of existing work and integrate these components into a single closed self-improvement loop. Essentially, the end goal would be to have something that looks like this:

\begin{center}
\includegraphics[width=0.6\linewidth]{image.png}
\end{center}

Note that we will initially approach this problem in a **chunk-based** manner, where we optimize the parameters (i.e., update the world model) after a window of time, then re-plan the next interaction from the updated model. If feasible, we can explore a **streaming** approach where we optimize the parameters in real-time as the interactions are happening, so the loop between acting and updating the world model gets tighter and tighter.

The project will be structured into two main phases:

1. **Phase 1: real-to-sim**: Build a differentiable simulation pipeline that can take real-world RGB-D video of an object being poked and estimate its physical parameters (mass, friction, stiffness, damping) by fitting a differentiable physics model to the observed motion. This is effectively the "world model" half of the loop.

   - Data generation with MuJoCo or Isaac Sim
   - We will focus on PhysTwin as the differentiable physics simulator. PhysTwin recently released versions that support fast batched simulation (*Boba*) and experimental code for MPPI planning.
   - this section should be straightforward given I have experience working with PhysTwin

2. **Phase 2: active exploration**: Extend the pipeline to an active perception setting where the system can autonomously select actions (e.g., pokes) to maximize information gain about the object's physical properties, using the differentiable physics simulator to predict outcomes of potential actions before taking them. This is the "self-improving" half of the loop.

   - the main guiding methodology of this section will be motivated by the Active System Identification (ASID) paper.

     - Will probably need to look into other alternatives like QOED or ASBI due to the number of parameters we are trying to estimate. Similarly, would be worth it to look into deep-learning based methods like iDAD.

   - we should be able to measure the information gain of an action by computing the Fisher Information Matrix (FIM) of each parameter (spring constants, damping, mass distribution, friction, etc).

     - May be important to look at faster alternative methods since we do have a fully differentiable physics model: diagonal empirical FIM, SWAG (Stochastic Weight Averaging-Gaussian), etc.

   - we will then combine this with MPPI (or other sampling methods) to sample candidate actions and select the one with the highest expected information gain, closing the loop back into Phase 1's parameter update.

There are other adjacent ideas that may be worth exploring if time allows:

1. abstracting the differentiable physics model: we can see if it's possible to make the active exploration agnostic to the differentiable physics model, so we can treat it like a black box and just use the gradients (or gradient-free signals) to optimize the parameters. This is what would let the physical RSI loop generalize beyond PhysTwin to other real-to-sim backends.
2. streaming vs chunk-based optimization: instead of optimizing the parameters after a full poke, we can try to optimize them in a streaming fashion (e.g., using MHE) to see if we can get better convergence and stability.
3. incorporating uncertainty into the differentiable physics model: we can try to model the uncertainty of the parameters and use that to guide the active exploration. For example, we can use a Bayesian approach to estimate the posterior distribution of the parameters given the observations, and use that to compute the expected information gain of each action.
4. Bring it to the real world! This is really the point of framing it as physical RSI in the first place: a robot that keeps closing the loop on its own world model as it encounters new objects, without needing a human to hand it fresh labeled interaction data every time.

## The plan

The overall 8 month plan is as follows:

* **Months 1–2: Data Collection and Preprocessing**

  * Learning and getting familiar with all the technical components.
  * Set up synthetic data generation pipeline using MuJoCo or Isaac Sim.
  * Collect a diverse dataset of RGB-D videos of objects being poked, along with ground truth physical parameters.
  * Set up PhysTwin and ensure it can be used as a differentiable physics simulator for parameter estimation.

* **Months 3–4: Active Exploration**

  * Look into the Active System Identification (ASID) paper and understand how to compute the Fisher Information Matrix (FIM) for each parameter.
  * Modify the differentiable physics simulator to compute the FIM for each parameter.
  * Implement MPPI (or other sampling methods) to sample candidate actions and select the one with the highest expected information gain.

* **Months 5–6: Testing**

  * Integrate the full pipeline and test on synthetic data.
  * Time for any bug fixing and debugging.
  * Preliminary experiments to evaluate the performance of the active exploration strategy in terms of parameter estimation accuracy and convergence speed.

* **Months 7–8: Benchmarks, Visualizations & Thesis Writing**

  * Compare against **Random Poking** and **Heuristic Poking** baselines.
  * Ablation studies to evaluate the impact of different components (e.g., differentiable physics model, active exploration strategy) on performance.
  * Render results, finalize data, write thesis...

## References

1. **PhysTwin:** Jiang, et al. *"PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos."* ICCV 2025.
2. **Fast PhysTwin (Boba)**: Pang, et al. *"Boba: Batched Simulation for Physics-Based Gaussian Digital Twins."* ECCV 2026.
3. **ASID:** Gupta, et al. *"ASID: Active Exploration for System Identification in Robotic Manipulation."* ICLR 2024. *(Active system identification and information-theoretic exploration).*
4. **ASBI**: Kim and Matsubara. *"ASBI: Leveraging Informative Real-World Data for Active Black-Box Simulator Tuning"*
5. **Quasi-Optimal Experimental Design**: Yu, et al. *"Learning What Matters: Adaptive Information Theoretic Objectives for Robot Exploration"* RSS 2026
6. **iDAD**: Ivanova, et al. *"Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods"* NeurIPS 2021

<!-- pandoc proposal.md -o proposal.tex --standalone -V header-includes="\usepackage{graphicx}" -->