# PLANS.md

## Current project status

Status: Milestone 0 closed on `main` on 2026-08-31 after all component and integration gates passed. Optional Milestone 0_a1 closed on renamed branch `m0_a1_visualizer` on 2026-09-01 and remains unmerged. Milestone 1 Component 1.1 D1/E1 is complete on branch `m1_ledger` and awaiting its user checkpoint; Component 1.2 has not started.

The methodology and high-level implementation backbone are defined in `PROJECT_CONTEXT.md`. The milestone sequence below is provisional at project start. Codex must enter Phase A for only the next milestone requested by the user and must not implement it until Phase C approval.

## Milestone philosophy

A **milestone** is a scientifically meaningful capability or validation result.

A **phase** is the human-in-the-loop state inside a milestone. Phases A–C are shared system-level planning gates. For implementation-dominant milestones, approved components are then closed incrementally:

- Phase A: inspect and propose the subsystem as a whole, including a complete Type 2 milestone-global versus Type 1 component-local question inventory;
- Phase B: confirm question ownership, resolve the milestone-global backbone first, then discuss component-local architecture, interfaces, semantics, component order, and tests serially;
- Phase C: freeze the approved system plan;
- Phase D_i: implement component `i`;
- Phase E_i: validate component `i` immediately;
- component checkpoint: user accepts/corrects/amends before the next component;
- Phase D_integration: compose accepted components;
- Phase E_integration: validate the complete stack;
- Phase F: user review and milestone closure.

This is one milestone, not a collection of component milestones. Component acceptance is strong but revisable if later integration exposes a real interface/coupling defect. Scientific-experiment milestones may use a single D/E pair when component cycling would be artificial.

Repository-wide planning uses the question-ownership taxonomy defined in `AGENTS.md`: Type 2 milestone-global contracts close before dependent Type 1 component-local choices; mixed questions are split explicitly. Closed global contracts become inputs to component studies. A local conflict reopens the smallest affected global contract rather than being hidden in component implementation or validation.

For Milestones 0 and 1, the formal Phase-C plan should be **component-first**: after a short milestone-wide architecture/interface preamble, keep each component's functionality, I/O, source/reference basis, differentiability semantics, implementation, tests, visual diagnostics if useful, and checkpoint criteria together in the same component section.

Do not conflate implementation progress with scientific acceptance.

## Milestone 0 — reproducible differentiable LTM baseline

Status: Closed on 2026-08-31. All Component 0.1–0.7 and integration D/E gates passed and were accepted. Authoritative plan and closure record: `docs/milestones/milestone_0_ltm_baseline.md`.

### Goal and shared architecture gate

Establish a trustworthy continuous-mass first-order network-loading backbone before adding mesoscopic order/lane refinement. Phase A/B must plan the following components as one system, including their interfaces, update ordering, reference equations, differentiability path, and dependency order. Exact milestone-specific choices such as discrete-time indexing, node SCIR, and signal/node API remain Phase A/B decisions rather than pre-migration commitments.

### Component 0.1 — traffic units and triangular fundamental diagram

Desired role: define consistent traffic units and link parameters/derived quantities required by LTM, including capacity, jam storage, free-flow travel delay, and backward-wave delay.

Checkpoint intent: hand/reference checks on units and derived values; reject inconsistent FD/link configurations before later components depend on them.

### Component 0.2 — cumulative-count LTM link

Desired role: map cumulative boundary histories and fixed link/FD parameters to sending/receiving quantities, then update cumulative counts from accepted boundary flows without integerizing mass.

Checkpoint intent: free-flow and congested hand cases, conservation/indexing checks, and local finite-difference/autograd checks where a continuous control/state dependency exists.

### Component 0.3 — fixed movement-demand projection

Desired role: project incoming sending demand and fixed turning/route proportions into oriented movement demands while keeping scenario semantics separate from flow realization.

Checkpoint intent: exact movement-total accounting and linear reference cases; no route-order ledger yet.

### Component 0.4 — generic first-order node solver

Desired role: map oriented demands, downstream supplies, and the approved priority/constraint inputs to admissible realized movement flows satisfying the selected generic-node requirements.

Checkpoint intent: one-to-one, diverge, merge, active-constraint and conservation cases; inspect the selected SCIR/active-set behavior before signal coupling.

### Component 0.5 — continuum fixed-time signal service

Desired role: convert approved fixed-time phase/green parameters and movement permissions/saturation flows into continuous movement service/capacity constraints consumed by the node model.

Checkpoint intent: isolated capacity/service cases and gradient checks with respect to green split. Treat correctness of AD for the continuum model separately from fidelity of the continuum approximation to on/off signals.

### Component 0.6 — deterministic network rollout and state update scheduler

Desired role: compose link sending/receiving, movement projection, node/signal service, boundary-flow updates, and deterministic scenario/config/result state into a fixed update order.

Checkpoint intent: short hand-auditable rollouts with explicit state histories and conservation.

### Component 0.7 — diagnostic objective, metrics, and gradient/reference harness

Desired role: provide a minimal scalar diagnostic objective plus traffic metrics and reusable reference/finite-difference tools needed to validate end-to-end signal sensitivity. This does not freeze the final scientific signal-optimization objective.

Checkpoint intent: directional finite-difference agreement in small fixed cases away from known active-set boundaries, plus reproducible diagnostic outputs.

### Integration gate

After Components 0.1–0.7 have each passed their D_i/E_i user checkpoint, integrate the full macro stack and verify:

- end-to-end mass conservation and demand/supply feasibility;
- expected propagation and node behavior on a minimal network;
- deterministic reproducibility;
- basic green-split sensitivity with approved finite-difference checks;
- no hidden integerization or silent graph detachment.

### Non-goals

- no ordered route ledger;
- no local lane grid;
- no lane-choice equilibrium;
- no turn-pocket finite-order experiment;
- no SUMO comparison;
- no full signal optimization study.

### Admission to next milestone

Satisfied on 2026-08-31. The optional Milestone 0_a1 tooling branch remains separate from the scientific closure.

## Milestone 0_a1 — headless macro-baseline visualizer add-on

Status: Closed on 2026-09-01 on separate branch `m0_a1` at closure commit `17d421c`; intentionally not merged into `main`. This `main` entry documents the optional tool's existence and branch location without importing its code or branch-specific milestone archive.

### Goal

Provide a small deterministic Matplotlib-based, headless visualization layer over the accepted Milestone 0 simulator and recorded rollout/gradient evidence. It should make the simulation process traceable and create fixed comparison views for later macro–meso experiments without implementing a second traffic simulator or changing Milestone 0 equations.

### Provisional scope

- Conda-managed Matplotlib dependency installation planned and recorded in the future approved D/E pass;
- noninteractive/headless rendering from the accepted simulator and immutable result histories;
- simple topology view plus essential link occupancy/storage, boundary flow, node movement flow, source-queue, signal-service, conservation, and gradient information;
- deterministic layout/configuration, units, time convention, scales, provenance, and non-overwriting artifacts;
- arbitrary handcrafted scenarios only within the accepted simulator's eventual explicitly approved initialization contract;
- semantic data/render checks and a small set of ordinary, signalized, and spillback-oriented visual fixtures.

### Explicit non-goals

- no interactive GUI, editor, live control, or SUMO/Vissim/MATSim-style application;
- no duplicate simulation/update logic inside plotting code;
- no change to traffic equations, gradient semantics, or accepted Milestone 0 evidence;
- no claim that aggregate occupancy is a spatial queue profile;
- no Milestone 1 route/order/lane functionality.

### Gate

Satisfied on branch `m0_a1_visualizer` on 2026-09-01 after its Phases A–F, three component checkpoints, and D/E integration passed. The tool remains optional and branch-isolated. Its absence from `main` does not block Milestone 1.

## Milestone 1 — ordered continuous route-mass ledger

Status: Component 1.1 D1/E1 complete on branch `m1_ledger` on 2026-09-02 and awaiting user acceptance. Its focused tests and the full regression suite pass. Component 1.2 has not started. Authoritative plan: `docs/milestones/milestone_1_ordered_route_mass_ledger.md`.

### Start gate and handoff

Phases A–C are complete. The Phase A/B archives remain under `temp_content/milestone_1_ordered_route_mass_ledger/`; the approved Phase C plan governs implementation. No Milestone 1 implementation is authorized until the user explicitly opens the first approved D_i pass.

### Goal and shared architecture gate

Add persistent route identity and exact continuous-mass ordering on top of the accepted LTM physical backbone without integerizing traffic flow. Phase A/B must plan route semantics, ledger ownership, split/forward/merge behavior, node projection, and aggregate coupling together before implementing the components separately.

### Component 1.1 — global route table and route-position semantics

Desired role: store each route once and define how a ledger entry references its route and current progress without duplicating full future movement sequences.

Checkpoint intent: deterministic route lookup/progression and explicit handling of route endpoints and node transitions.

### Component 1.2 — ordered continuous route-mass ledger state

Desired role: represent persistent ordered entries conceptually as `(route_id, route_position, mass)` with strictly continuous positive mass and stable FIFO order.

Checkpoint intent: construction/invariant tests for ordered mass totals and structural metadata; no floor/round-based transfers.

### Component 1.3 — route-to-current-movement projection

Desired role: derive the current movement from route metadata through a fixed node-local mapping while retaining persistent route identity for downstream use.

Checkpoint intent: hand-built routes/nodes map to the intended current movements; projection does not mutate route identity.

### Component 1.4 — continuous split, forward, and exact adjacent merge operations

Desired role: split continuous ordered mass when required, advance accepted mass across nodes/links, and losslessly merge adjacent structurally identical route states without changing order semantics or total mass.

Checkpoint intent: exact conservation and order preservation through split/forward/merge sequences, including partial-entry transfers.

### Component 1.5 — ledger coupling to aggregate LTM/node transfer

Desired role: let accepted aggregate boundary/node flow determine **how much** continuous mass advances while the ordered ledger determines **which route mass** advances, without changing the already validated aggregate physical flow in matched cases.

Checkpoint intent: aggregate cumulative counts remain unchanged by metadata bookkeeping, route-mass totals equal aggregate transferred mass, and route order survives transitions where semantics require it.

### Component 1.6 — ordered movement diagnostic view

Desired role: expose deterministic node-local ordered movement/mass views for tests and later FIFO-service development without replacing the persistent route ledger with movement-only storage.

Checkpoint intent: simple route sequences produce human-auditable movement-order traces and preserve correspondence to underlying route entries.

### Integration gate

After Components 1.1–1.6 pass their D_i/E_i user checkpoints, integrate the ledger with the accepted macro stack and verify:

- route-mass conservation equals aggregate traffic-mass conservation;
- matched aggregate physical counts/flows are unchanged when ledger metadata is behaviorally passive;
- exact order survives approved split/forward/merge cases;
- no integer vehicle transfer or lossy persistent compression is introduced;
- deterministic ordered movement views are available for the next local-zone milestone.

### Admission to next milestone

Do not proceed until every required ledger component checkpoint and the final ledger/macro integration gate are accepted.

## Milestone 2 — per-lane cumulative-mass interaction zone

Status: Provisional; blocked by Milestone 1 closure.

### Goal

Introduce the local per-lane mesoscopic refinement and couple it conservatively to LTM links.

### Conceptual scope

- per-lane cumulative-mass/FIFO-rank grid near downstream node;
- movement-composition tensors/views;
- mapping between incoming route-mass ledger and lane-local state;
- lane storage based on geometry/jam density;
- local-zone receiving feedback to upstream LTM;
- no clipping when the physical queue exceeds the detailed zone.

### Key validation

- mass conservation across LTM/local-zone boundary;
- local storage saturation reduces upstream receiving correctly;
- neutralized local detail reproduces matched aggregate behavior;
- tensor shapes, ordering, and physical units are explicit and tested.

## Milestone 3 — lane allocation, finite storage, and exact FIFO service

Status: Provisional; blocked by Milestone 2 closure.

### Goal

Implement the signal-relevant mesoscopic behaviors that justify the local refinement.

### Conceptual scope

- choose/freeze a minimal established lane-allocation/resistance rule;
- separate desired allocation from realized lane accessibility;
- finite turn-pocket/shared-lane storage;
- exact ordered-prefix service using lane-local route/movement order;
- movement-specific signal eligibility;
- partial/localized blocking rather than indiscriminate full-link FIFO.

### Required diagnostic cases

At minimum include:

- `[T(5), L(5)]` versus `[L(5), T(5)]` under the same aggregate movement totals;
- shared-lane head-of-line red-movement blockage;
- finite turn-pocket overflow affecting upstream accessibility;
- a no-blockage case that agrees with the aggregate baseline.

### Open decisions to resolve in Phase A/B

- exact lane-allocation/equilibrium formulation;
- exact interaction between lane allocation and known current blockage;
- service operator details at mixed/active-prefix boundaries;
- whether startup lost time enters this milestone or remains deferred.

## Milestone 4 — differentiability and matched macro–meso sensitivity study

Status: Provisional; blocked by Milestone 3 closure.

### Goal

Determine whether the mesoscopic representation changes gradients in the intended cases and whether those gradients are numerically credible.

### Conceptual scope

- matched macro baseline sharing demand, FD, signal parameterization, and objective;
- central finite-difference directional checks in fixed deterministic scenarios;
- active-set/boundary diagnostics;
- same-macro/different-order causal cases;
- gradient norm/alignment/direction-improvement diagnostics;
- characterize when macro and meso gradients agree versus diverge.

### Key scientific question

Does retained order/lane/storage information create different signal sensitivities specifically where the aggregate representation cannot distinguish the scenarios?

## Milestone 5 — fixed-time signal optimization in mesoscopic scenarios

Status: Provisional; blocked by Milestone 4 closure.

### Goal

Use validated gradients for downstream optimization rather than only internal derivative checks.

### Initial control scope

- fixed phase sequence/topology;
- green-split optimization first;
- deterministic or fixed scenario-bank sample-average objective;
- held-out scenario evaluation;
- identical optimizer/update budgets and objective definitions for matched macro/meso methods unless a shared tuning protocol is explicitly approved.

### Measurements

- objective improvement;
- queue/wait/throughput or other approved traffic metrics;
- convergence and failures;
- gradient direction usefulness;
- runtime and memory;
- sensitivity to initial signal plan;
- scenario-to-scenario variability if sampling is used.

### Deferred branches

- cycle length until lost time/within-cycle effects are modeled;
- offset/coordinated-network optimization;
- stochastic route choice;
- learned controllers.

## Milestone 6 — cross-fidelity transfer to a richer simulator

Status: Provisional; blocked by Milestone 5 closure.

### Goal

Test whether macro and mesoscopic gradient-guided signal changes transfer differently to a higher-fidelity reference simulator, initially expected to be SUMO unless Phase A evidence supports another reference.

### Conceptual scope

- construct matched reference scenarios;
- evaluate small perturbations along macro and mesoscopic gradient directions;
- evaluate final optimized signal plans on held-out reference scenarios;
- keep reference-simulator calibration/parameter choices common across compared methods;
- distinguish representation error, scenario variance, and optimization artifacts.

### Primary interpretation

The main claim should concern **useful cross-fidelity sensitivity**, not merely lower loss in the differentiable simulator.

## Conditional future milestones

These are deferred and should not be planned automatically:

- startup/clearance lost-time modeling and cycle-length optimization;
- multi-intersection offset optimization;
- metadata-compression/defragmentation study;
- controlled scenario stochasticity and gradient-variance reduction;
- macro-gradient control variates;
- endogenous route/lane-choice differentiation;
- comparison with differentiable agent/platoon approaches;
- performance scaling and larger networks.

## Minimal repository responsibility map

The exact layout may adapt during Phase A, but responsibilities should remain separated:

- `src/.../fd`: traffic-flow parameters/units;
- `src/.../ltm`: long-link cumulative-count propagation;
- `src/.../nodes`: node demand/supply/admissibility logic;
- `src/.../routes`: route tables and movement projection;
- `src/.../ledger`: ordered continuous route-mass metadata;
- `src/.../local_zone`: per-lane cumulative-mass state and LTM coupling;
- `src/.../lane_allocation`: desired versus realized lane use;
- `src/.../signals`: continuum fixed-time service;
- `src/.../simulation`: update schedule/rollout;
- `src/.../objectives`: signal-control objective and metric components;
- `src/.../gradients`: numerical checks/diagnostics;
- `experiments`: approved scientific runs;
- `configs`: immutable/reproducible experiment configuration;
- `tests`: reference, conservation, behavior, and gradient tests;
- `reports`: durable scientific evidence;
- `temp_content`: user-facing planning and phase dialogue;
- `docs/milestones`: approved implementation plans.

## First Codex task

When the repository is initialized, the first substantive Codex task should be:

1. read `AGENTS.md`, `PROJECT_CONTEXT.md`, and `PLANS.md`;
2. inspect the repository, Python environment, available compute, and relevant installed packages;
3. open **Milestone 0 Phase A only**;
4. write the proposal under `temp_content/milestone_0_ltm_baseline/`;
5. identify unresolved equation/indexing/API decisions and acceptance tests;
6. stop for user discussion and approval;
7. do not implement Milestone 0 in the same task unless the user explicitly waives the gate.
