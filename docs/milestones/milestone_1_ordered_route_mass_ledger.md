# Milestone 1 — ordered continuous route-mass ledger

Status: **Closed**

Approval date: 2026-09-02

Phase: Phase F closed on 2026-09-03

Branch: developed and closed on `m1_ledger`, based on `main` commit `c6a2d75`; fast-forward merged into `main` at `309cba6` on 2026-09-03; `m1_ledger` retained

## 1. Scientific purpose

Add persistent route identity, continuous FIFO-rank order, and recursively propagated boundary timing over the accepted Milestone 0 LTM backbone without integerizing traffic mass or changing accepted aggregate flows.

Milestone 1 tests a representation prerequisite:

> Can continuous vehicle-equivalent mass retain exact route progress and deterministic ordered timing metadata while remaining conservative, differentiable within fixed structural regimes, and behaviorally passive relative to the matched macro model?

This milestone does not test whether order improves signal-control gradients. It establishes the trusted semantic state required for later per-lane storage and FIFO/blockage milestones.

## 2. Authority, starting state, and scope

This plan incorporates the closed global and component contracts in `temp_content/milestone_1_ordered_route_mass_ledger/phase_B_revision.md`. It supersedes the provisional interface and candidate choices in the Phase A proposal.

Starting state:

- Milestone 0 is closed under `docs/milestones/milestone_0_ltm_baseline.md`.
- The accepted macro `NetworkDefinition`, `Scenario`, `simulation_step`, `RolloutResult`, LTM, movement, node, signal, objective, and gradient behavior remain unchanged.
- Macro arrivals are interval-boundary masses: `arrivals[k]` is added to source availability at `t_k` and may be admitted during `[t_k,t_{k+1}]`.
- The optional visualizer remains isolated on `m0_a1_visualizer` and is not a dependency.
- Authoritative environment: `/home/zpz/miniconda3/envs/diff_minimal_meso`, Python 3.12.13, PyTorch 2.5.1 with CUDA 12.4 build, pytest 9.0.3. E2 reverified the executable target, runtime imports, Conda metadata, and unchanged environment history after one anomalous isolated command result; the E2 validation record documents that investigation.

### In scope

- immutable global route table with compact route references and positions;
- nonempty simple connected fixed routes, including one-link routes;
- persistent per-source and per-link ordered continuous route-mass state;
- actual-discharge-derived affine timing segments;
- deterministic movement projection and disposable adjacent movement-run shorthand;
- continuous partial split, movement-accessible extraction, route progression, deterministic merge ordering, and exact adjacent defragmentation;
- independent mesoscopic scenario/state/step/rollout records that reuse unchanged macro closures;
- route/macro conservation, aggregate equivalence, order/timing, replay, and stable-regime AD/JVP/FD evidence;
- immutable JSON plus Markdown integration evidence.

### Explicit non-goals

- no order-induced service restriction, lane state, lane allocation, shared-lane blockage, finite turn-pocket storage, or lane-changing model;
- no modification of Milestone 0 turning fractions, node equations, LTM equations, continuum signal semantics, or macro public records;
- no dynamic/endogenous route choice, repeated-link/cyclic routes, nonempty physical-link initialization, or microscopic vehicles;
- no sub-packet interleaving for overlapping incoming time profiles;
- no thresholded timing merge, lossy compression, multi-segment resident entry, smoothing, surrogate gradient, or differentiation through permutation changes;
- no batching redesign, padding capacity, compilation, performance optimization, new dependency, visualizer merge, or Milestone 2 work;
- no claim of mesoscopic forward advantage or gradient usefulness.

## 3. Milestone-wide frozen contracts

### 3.1 Units, time, and notation

| Quantity | Symbol | Unit | Runtime form |
|---|---|---|---|
| interval boundary | `t_k = k*dt` | s | derived structural float |
| continuous mass/rank | `m`, `q`, `z` | veh-eq | scalar float64 tensor |
| rate | `C` | veh-eq/s | inherited structural/tensor value |
| interval transfer | `F_k` | veh-eq | scalar/vector float64 tensor |
| generation/eligibility time | `t_front`, `t_tail` | s | scalar float64 tensor |
| route/link/position/movement | `r,a,p,mv` | index/ID | structural Python values |

No mass is rounded, floored, integerized, clipped, silently dropped, or created.

For accepted boundary mass `F_k>0` over `[t_k,t_{k+1}]`, offset rank `z` has reconstructed actual crossing time

```text
t_cross(z) = t_k + dt * z / F_k,  0 <= z <= F_k.
```

A positive sub-block spanning ranks `[z_0,z_1]` receives `[t_cross(z_0),t_cross(z_1)]`; therefore its duration is strictly positive. Zero transfer creates no entry.

A resident entry on link `a` stores its earliest downstream-boundary eligibility segment:

```text
t_eligible_front = t_actual_entry_front + tau_f[a]
t_eligible_tail  = t_actual_entry_tail  + tau_f[a].
```

When it actually leaves `a`, its outgoing segment is reconstructed from the accepted discharge rank in that step; the next link's free-flow time is then added. Static origin-generation time is never propagated in place of actual admission/discharge.

### 3.2 Source-time convention

The accepted macro scheduler makes all `Scenario.arrivals[k]` available at `t_k`. To retain individual/cohort chronology without changing that scheduler:

- meso route-demand blocks assigned to step `k` represent demand generated **by** boundary `t_k` and added at `t_k`;
- absolute generation endpoints are structural chronology for ordering the source queue and must satisfy `generation_front < generation_tail <= t_k` for a positive block;
- step 0 may use pre-horizon generation times ending at or before `t_0=0`; this does not imply preloaded physical links or a nonempty macro initial state;
- actual admission during `[t_k,t_{k+1}]`, not generation time, creates the first-link resident eligibility segment;
- macro and meso source availability remain exactly aligned at every boundary.

This is a discrete-time demand-bucketing convention, not a claim that the macro solver resolves continuous source generation inside a step.

### 3.3 Passive coupling and sending eligibility

At each input link and step:

1. accepted macro sending `S_i` defines the mass length of the aggregate FIFO tranche physically eligible to send;
2. the macro node solution supplies movement quotas `f_m` whose sum for input `i` equals accepted input outflow;
3. the ledger scans only within the first `S_i` mass of the link ledger;
4. within that tranche it consumes the earliest mass matching each movement quota, preserving relative order of transferred and residual subsequences;
5. insufficient matching mass is a fail-fast macro/ledger compatibility error.

This movement-accessible extraction adds no order-induced blockage and never searches travel-ineligible resident mass. Macro totals remain authoritative.

### 3.4 Deterministic output order

For packages entering one outbound link:

- sort whole packages by leading actual discharge/eligibility time;
- if positive-duration profiles overlap, append the later-leading whole package after the earlier-leading package; do not synthesize interleaving;
- equal leading times use the position of the incoming link in `NodeMovementMap.input_link_ids`;
- preserve order within each incoming stream;
- record the tie convention in artifacts.

This is a behaviorally passive scaffold. It is not physical lane-entry chronology and may be replaced only by reopening G2 in a later lane-local milestone.

### 3.5 Entry and exact-merge semantics

One resident entry is one continuous FIFO-rank block, one route reference/position, and one affine eligibility-time segment. It is not a list of blocks or timing segments.

For adjacent entries

```text
E1 = (r,p,m1,a,b)
E2 = (r,p,m2,b,c),
```

exact merge to `(r,p,m1+m2,a,c)` is legal only if the combined affine line reproduces the shared boundary exactly:

```text
(b-a) * m2 == (c-b) * m1
```

using strict stored float64 equality and with no time gap. Failure to prove equality leaves both entries unchanged. False nonmerge is safe; tolerance-based merge is prohibited.

### 3.6 Structural versus differentiated state

Structural/nondifferentiated:

- route strings, compact indices, positions, link/movement IDs;
- tuple lengths and order, source/node ownership, input tie priority;
- scan, split-boundary, active-entry, output-order, and exact-merge topology;
- diagnostics and failure reasons.

Gradient-carrying when their upstream inputs require gradients:

- mass, generation/eligibility endpoints, admitted/movement/sink quotas;
- split residual/transferred mass and interpolated times;
- ledger/movement totals and coupled aggregate quantities.

Within fixed structural topology, arithmetic remains in PyTorch. Do not use `.item()`, NumPy, `detach`, or tensor reconstruction on selected arithmetic. Comparisons may define a structural active set, and no derivative is claimed through topology/permutation changes.

Frozen dataclasses are not direct `torch.func.jvp` primals in installed PyTorch. JVP/FD wrappers accept tensor leaves, construct fixed structural records inside the transformed function, and return tensors. Reverse mode operates normally through tensor fields. No custom pytree registration is required.

### 3.7 Independent composition roots and update order

```text
shared unchanged closures                       meso-owned semantics
FD / LTM / movement map / node / signal   +   routes / ledger / timed demand
                 \                              /
                  meso rollout composition root

macro rollout remains separately runnable and unchanged
```

At interval `k`:

1. add meso route-demand blocks for boundary `t_k` to ordered source queues;
2. call unchanged macro `simulation_step` from the pre-update macro state;
3. use `source_admitted` to consume source-ledger prefixes and create first-link resident entries with admission-rank-derived eligibility times;
4. use each node's accepted movement flows and pre-update link ledgers to perform sending-tranche validation and movement-accessible extraction;
5. reconstruct actual discharge segments from each input's selected total order; progress transferred route positions; add outbound free-flow delays;
6. order/append packages at each output using Section 3.4, then optionally apply explicit exact merge;
7. use `sink_outflow` to consume eligible sink-link prefix mass and record completed route mass/timing;
8. construct the next immutable meso state and step evidence.

All ledger operations use pre-update state. Newly admitted or node-transferred mass cannot traverse the newly entered complete link in the same step.

## 4. Approved files and dependency order

Expected additions during authorized D/E passes:

```text
src/diff_minimal_meso/routes.py
src/diff_minimal_meso/ledger.py
src/diff_minimal_meso/route_simulation.py
tests/references/ledger_scalar_reference.py
tests/test_routes.py
tests/test_ledger_state.py
tests/test_ledger_projection.py
tests/test_ledger_operations.py
tests/test_route_simulation.py
tests/test_ledger_diagnostics.py
tests/test_milestone_1_integration.py
configs/milestone_1/minimal_ordered_route.json
scripts/run_milestone_1_diagnostics.py
```

`src/diff_minimal_meso/__init__.py` may add exports only after the owning component passes. Do not modify accepted macro records or equations. A behavior-preserving helper extraction from macro code requires an explicit plan-conformance note and tests; prefer direct reuse first.

Implementation order is Components 1.1 through 1.6, each D_i then E_i then user checkpoint, followed by D_integration/E_integration. Later components may import only accepted earlier components.

## 5. Component 1.1 — route table and route-position semantics

### Responsibility and interface

`routes.py` owns immutable structural records conceptually equivalent to:

```text
RouteDefinition(route_id: str, link_ids: tuple[int, ...])
RouteTable(routes: tuple[RouteDefinition, ...], route_id_to_index: Mapping[str,int])
```

Required pure queries validate and return current link, next link/movement, terminal status, and progressed position. External IDs are nonempty unique strings; entries store compact integer route indices and integer positions.

Valid routes are nonempty simple connected ordered paths. Every link is in range; no link repeats; each adjacent pair occurs as exactly one configured movement; first link has source receiving ownership; final link has sink sending ownership. One-link source-to-sink routes are valid.

The existing `NodeMovementMap.input_link_ids` tuple position is the node-local exact-time tie priority. No separate priority metadata is added.

### Reference basis and differentiability

Use established fixed-route/multi-commodity LTM semantics: a route is stored once and local state carries only reference plus position. All Component 1.1 values are structural and nondifferentiated.

### Implementation and literal checks

- Valid chain: route `R-main=(0,1,2)` maps positions `0,1,2`, movements `(0,1)` and `(1,2)`, and terminates at configured sink.
- Valid one-link route: source/sink-owned link `3` has no next movement and is terminal.
- Mapped IDs: routes `R-main` and `R-alt` map to compact indices independent of declaration-facing names; duplicate names fail.
- Invalid cases: empty route; out-of-range link; repeated link `(0,1,0)`; missing/duplicate adjacency movement; wrong source first link; wrong sink final link; Boolean/noninteger link; out-of-range position; progression beyond terminal.
- Relabeling: consistently permuting links and node maps produces correspondingly mapped queries and preserves input-tuple tie rank.

### Acceptance and checkpoint

All valid positions have one unambiguous owner and next movement or valid terminal. Invalid topology fails before simulation. Route identity never changes during progression.

After D1/E1, write `phase_D/phase_D1_routes_worklog.md` and `phase_E/phase_E1_routes_validation.md`, report evidence, and stop for user acceptance.

## 6. Component 1.2 — immutable ordered ledger state

### Responsibility and interface

`ledger.py` owns conceptually:

```text
LedgerEntry(
    route_index: int,
    route_position: int,
    mass: Tensor[],
    eligible_front_s: Tensor[],
    eligible_tail_s: Tensor[],
)
OrderedLedger(owner_link_index: int, entries: tuple[LedgerEntry, ...])
```

The descriptive Phase B `t_front/t_tail` fields are named `eligible_front_s/eligible_tail_s` in resident state to prevent confusion with source generation or actual discharge records.

Validation requires scalar finite float64 tensors on one device, `mass>0`, `eligible_tail_s>eligible_front_s`, valid route/position, and route current link equal to ledger owner. Empty ledger uses `entries=()`; no public zero entry exists. Construction is immutable/out-of-place.

### Reference basis and differentiability

Continuous multi-commodity queue mass remains tensor-valued while route/order metadata is structural. Tuple/dataclass containers do not detach tensors. Tensor-leaf wrappers are the transform boundary for JVP.

### Implementation and literal checks

- Construct owner link 0 ledger with `R1@0: mass=.125, eligibility=[2,2.5]` and `R2@0: mass=.375, eligibility=[2.5,3.5]`; preserve order and total `.5`.
- Reverse gradient of total mass is `[1,1]`; timing-derived scalar gradients remain connected.
- Tensor-leaf JVP of total/split-compatible arithmetic matches hand directional derivatives.
- Empty ledger total is scalar zero on an explicitly supplied dtype/device context or via state-owned template; do not invent CPU tensors on a CUDA path.
- Reject zero/negative/nonfinite mass; equal/reversed/nonfinite times; nonscalar/mismatched dtype/device; Boolean/negative/out-of-range route metadata; owner mismatch; public zero residual.
- Immutability attempts fail or leave originals byte/value-identical.

### Acceptance and checkpoint

Construction preserves exact declared order and graph paths; totals match hand sums; no integer conversion exists. Installed reverse/JVP probe behavior is reproduced in component tests.

After D2/E2, write the corresponding worklog/validation and stop.

## 7. Component 1.3 — route-to-movement projection

### Responsibility and interface

Derive a disposable node-facing sequence:

```text
MovementRun(movement_index: int, mass: Tensor[])
project_movement_runs(ledger, node, route_table) -> tuple[MovementRun, ...]
movement_totals(runs, movement_count) -> Tensor[M]
```

Projection maps each resident entry's `(route,position)` through fixed `(input_link,next_link)` movement lookup, then run-length encodes only adjacent equal movement labels. It stores no route pointers and never mutates or replaces the ledger. Terminal or wrong-node entries fail; sink handling is separate.

### Differentiability

Movement labels and run boundaries are structural. Run mass is the tensor sum of contributing entry masses, preserving gradients within the fixed run topology.

### Literal checks

```text
ledger: [R1:T(.5), R2:L(.25), R3:T(.75), R4:T(1.75)]
runs:   [T(.5), L(.25), T(2.5)]
totals: T=3.0, L=.25
```

- Same totals with `[L(.25),T(3.0)]` remain order-distinct.
- Reprojecting the unchanged ledger yields identical runs.
- Run total equals ledger total and movement totals equal the entry-wise reference projection.
- Reverse/JVP of run and grouped sums match the linear hand map.
- Reject terminal/wrong-owner/undefined movement without omission.

### Acceptance and checkpoint

The shorthand contains only movement and mass, preserves movement-block order, and is exactly reproducible from the authoritative ledger. After D3/E3, report and stop.

## 8. Component 1.4 — split, extraction, progression, ordering, and exact merge

### Responsibility and interfaces

Implement pure, separately testable operations plus a composed transaction:

- affine entry split by positive mass;
- prefix/sending-tranche construction;
- stable movement-accessible extraction by quotas;
- route progression after node crossing;
- actual-discharge segment assignment by selected rank;
- whole-package output ordering/tie handling;
- `merge_adjacent_exact` and merge diagnostics;
- result records containing residual, transferred, and evidence.

For entry `(M,a,b)` split at `0<x<M`:

```text
t_x = a + (x/M)*(b-a)
head = (x,a,t_x)
tail = (M-x,t_x,b).
```

Arithmetic remains tensor-native. Structural branching does not detach the selected mass arithmetic.

### Literal reference cases

1. Split `M=2`, `[10,12]` at `.75` -> head `.75,[10,10.75]`, tail `1.25,[10.75,12]`; mass/time continuity exact within hand tolerance.
2. Zero quota returns the identical logical ledger and empty transfer; full quota omits zero residual; negative quota and overdraw fail.
3. Movement scan over `[T(.5),L(.25),T(.75)]` with eligible tranche `1.5` and quotas `T=.8,L=.2` consumes `T(.5),L(.2),T(.3)` in selected ledger order and leaves `L(.05),T(.45)`; both subsequence orders and total conservation are checked.
4. A quota whose matching mass is insufficient inside the eligible tranche fails even if matching mass exists later in the resident ledger.
5. Progress only transferred entries; terminal transfer is routed to sink completion, never progressed beyond route end.
6. Discharge `F=2` during `[4,5]`: selected rank blocks `[0,.5]`, `[.5,2]` receive actual times `[4,4.25]`, `[4.25,5]`; adding outbound `tau_f=3` yields eligibility `[7,7.25]`, `[7.25,8]`.
7. Two outbound packages `[7,8]` and `[7.5,8.5]` serialize as whole packages by leading time; exact equal fronts use input tuple rank.
8. Exact merge positive: `m1=1,[0,1]` plus `m2=2,[1,3]` -> `m=3,[0,3]`.
9. Safe nonmerge: route mismatch, position mismatch, nonadjacency, time gap `[0,1]+[2,3]`, or rate breakpoint `m1=1,[0,1]+m2=1,[1,3]`.
10. Long deterministic split/progress/append/merge chain preserves mass and original immutable inputs.

An independent standard-library scalar oracle must not import production ledger operators.

### Gradient and diagnostic checks

- reverse/JVP/central FD agree away from equality, split-exhaustion, ordering, and merge-topology boundaries;
- declare active signatures for selected entry indices, split locations, ordering permutation, and merge decisions;
- record adjacent compatible pairs examined, exact merges, before/after counts, and safe-nonmerge reasons;
- telemetry is observational and cannot change behavior or remove the operator.

### Acceptance and checkpoint

For every successful operation, residual plus transferred mass equals input at `rtol=1e-10, atol=1e-12`; order/timing/route positions match references; invalid requests fail; no lossy merge occurs.

After D4/E4, report and stop.

## 9. Component 1.5 — meso scenario and macro/ledger coupling

### Responsibility and interfaces

`route_simulation.py` owns conceptually:

```text
SourceRouteBlock(route_index, mass, generation_front_s, generation_tail_s)
MesoScenario(macro_fields, route_arrivals[T][B_source])
MesoState(macro_state, source_ledgers, link_ledgers, completed_route_mass)
MesoBoundaryTransfer(...)
MesoStepResult(macro_step_result, transfers, end_state)
MesoRolloutResult(macro_rollout_result, ledger_history, step_results)
```

`MesoScenario` is authoritative. Its pure macro adapter sums route block masses per `[k,source]` and constructs the unchanged macro `Scenario`; it is not a third demand record. Other macro fields are shared/copied immutably with recorded provenance.

The meso composition root is separate and calls accepted macro functions. Macro state/results are referenced, not numerically duplicated. Boundary transfer records retain sufficient entry-level evidence for conservation/replay even though Component 1.3 solver shorthand stores no backward pointers.

### Literal cases and checks

1. Source bucket: at `t_0=0`, two pre-boundary generation blocks `R1:.125,[-2,-1]` and `R2:.375,[-1,0]` sum to macro arrival `.5` and preserve source order.
2. Blocked source: generated mass waits; first-link eligibility uses actual later admission interval plus link `tau_f`, not generation endpoints.
3. One-link fractional pulse reproduces macro `n_in/n_out`, source queue, and sink counts exactly; route completion mass is `.125`.
4. Two-link chain preserves the Milestone 0 no-same-step-cascade trace and progresses route position exactly once per crossing.
5. Diverge: two routes sharing an input realize fixed macro movement quotas by G1 scan; ledger and macro boundary masses agree or fail fast on incompatible eligible composition.
6. Merge: two inputs entering one output use actual discharge-derived order, overlap serialization, and input-tuple exact tie priority.
7. Restricted signal case preserves macro physical-green reverse/JVP/FD sensitivities while ledger transfers equal accepted movement quotas.
8. Matched macro run uses the adapter output; all aggregate step/state/result tensors are `torch.equal` where the unchanged macro operation path is identical.
9. Invalid cases: route/source mismatch, route arrival sum mismatch by construction attempt, timestamp outside its source bucket, ledger/macro state mismatch, insufficient eligible movement mass, route completion at wrong sink.
10. Deterministic repeat and mapped relabeling preserve mapped route/order results.

### Conservation identities

At every boundary and horizon:

```text
route mass generated
= source-ledger mass + resident-link-ledger mass + completed route mass

sum route source admission = macro source_admitted
sum route movement transfer by movement = macro movement_flow
sum route sink completion = macro sink_outflow
sum resident route mass by link = macro occupancy
```

Use `rtol=1e-10, atol=1e-12` for ledger identities. Macro equivalence targets exact equality because the macro composition is unchanged; any deviation requires a documented arithmetic cause and plan amendment before weakening the gate.

### Acceptance and checkpoint

The meso and macro stacks run independently with distinct entry points. Meso demand has one authority, all quotas replay conservatively, timing is recursive, macro results/gradients remain unchanged, and no same-step cascade occurs.

After D5/E5, report and stop.

## 10. Component 1.6 — ordered diagnostics

### Responsibility and schema

Expose read-only diagnostic builders with two explicit levels:

```text
ordered_entries:
  link, ordinal, route_id/index, route_position, movement_or_terminal,
  mass, eligible_front_s, eligible_tail_s

movement_totals:
  node/input/movement, total_mass
```

The detailed view is one-for-one with persistent entries and authoritative for audit. The summary is derived and may not replace or coalesce the detailed view. Presentation details are delegated but must remain deterministic, unit-labeled, and machine/human consistent.

### Literal checks

- Render `[T(.5,R1),L(.25,R2),T(.75,R3)]` and separately `T=1.25,L=.25`.
- Same totals/different order produce different `ordered_entries` and equal summaries.
- Every detailed row round-trips to the underlying entry values; totals conserve mass.
- Empty, terminal, source-queued, and completed states have explicit schemas rather than omission.
- JSON and Markdown are generated from one evidence object and agree field-for-field on checked values.
- Diagnostics do not mutate tensors, detach production results, or affect simulator behavior.

### Acceptance and checkpoint

Order, route, timing, and totals are simultaneously auditable; output is deterministic; no diagnostic aggregation becomes persistent state.

After D6/E6, report and stop.

## 11. Integration D/E gate

After all six component checkpoints are accepted, D_integration may add only approved orchestration, config, runner, and integration tests.

### Required scenarios

1. **Ordinary multi-node route case:** at least two sources, a merge, a diverge, fractional masses, a blocked source interval, overlapping inbound timing, route completion, and at least one exact-merge and several safe-nonmerge reasons.
2. **Restricted signal case:** fixed phase topology, differentiable physical green, multiple routes/movements, passive ledger coupling, stable active regime.

### Required acceptance evidence

- mathematical scalar/reference agreement for route queries, splits, timing, ordering, and exact merge;
- route/source/link/sink conservation at every step and over the horizon;
- sending/receiving eligibility and movement quota correspondence;
- exact matched macro aggregate results and unchanged accepted macro test suite;
- deterministic same-flow/different-order traces and mapped relabeling;
- exact-only merge telemetry with reasons, interpreted descriptively only;
- reverse mode and JVP agreement; central FD agreement on at least three adjacent stable step sizes away from topology boundaries;
- finite gradients and recorded active signatures;
- no integer conversion, lossy compression, hidden macro mode, same-step cascade, or production-data mutation;
- immutable non-overwriting JSON evidence plus Markdown summary under `reports/milestone_1_ordered_route_mass_ledger/<run_id>/`.

The integration runner records command, git commit/dirty status, Python/PyTorch versions, device, dtype, config hash/content, timestamps, tolerances, active signatures, quantitative checks, merge telemetry, and artifact paths.

After E_integration, write `phase_D/phase_D_integration_worklog.md` and `phase_E/phase_E_integration_validation.md`, report every criterion item by item, and stop for Phase F. Do not start Milestone 2.

## 12. Numerical policy and commands

- Mainline and reference dtype: `torch.float64`.
- Structural route/link indices: Python integers; tensor indices only where inherited APIs require CPU `torch.long`.
- Hand/reference and ledger conservation: `rtol=1e-10`, `atol=1e-12`.
- Declared mass aggregation validation: `rtol=0`, `atol=1e-9` only at authored configuration boundaries; no repair/normalization.
- Stable reverse/JVP: `1e-10 + 1e-10*scale`.
- Stable AD/FD: `1e-8 + 1e-6*scale`, with a predeclared step scan and three adjacent stable rows.
- Exact merge comparisons are semantics and use exact equality; test tolerances never decide merging.
- CPU float64 is the reference authority. Retain the existing CUDA device-path check; if sandboxing masks required CUDA, retry with minimum escalation before concluding unavailability.

Per-component commands:

```bash
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_routes.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_ledger_state.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_ledger_projection.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_ledger_operations.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_route_simulation.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_ledger_diagnostics.py
```

Integration/final commands:

```bash
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_milestone_1_integration.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m compileall -q src tests
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python scripts/run_milestone_1_diagnostics.py --config configs/milestone_1/minimal_ordered_route.json --output-root reports/milestone_1_ordered_route_mass_ledger
```

## 13. Component D/E reporting and error policy

For every authorized component:

1. re-read this plan and implement only that component;
2. run its literal/reference, invalid, conservation, behavior, and applicable gradient checks;
3. write the D worklog and E validation report;
4. diagnose and fix syntax, harness, or implementation-conformance errors within this plan;
5. stop for the user checkpoint before the next component.

If evidence challenges scientific semantics or an acceptance contract, stop, document the affected contract and smallest amendment candidates, and wait. Do not hide a defect in an adapter, relax a tolerance, introduce clipping/smoothing, or alter behavior from diagnostic results.

## 14. Stop conditions, reopen conditions, and deferred items

Stop and reopen the smallest contract if implementation would require:

- ledger order to change macro service or introduce blockage (G1);
- sub-packet interleaving or physical lane-order claims (G2);
- modifying macro records/equations or adding a macro/meso mode switch (G3);
- multi-segment entries, tolerance merge, gap deletion, or lossy compression (G4);
- weakening immutable evidence or authoritative environment/provenance (G5);
- repeated-link routes or dynamic route choice (Component 1.1);
- zero-mass/zero-duration resident entries or padding capacity (Component 1.2);
- persistent movement-only state or nondeterministic ledger replay (Component 1.3);
- quota clamping, ineligible scanning, or behavior-changing merge telemetry (Component 1.4);
- continuously generated within-step macro arrivals or a second authored demand truth (Component 1.5);
- diagnostics that coalesce authoritative entries or affect simulation (Component 1.6).

Deferred: lane bins, lane occupation, implicit lane-level aggregation/reordering, FIFO blockage, finite storage, lossy compression experiments, performance packing, nonempty physical initialization, visualization integration, and all later milestones.

## 15. Phase C consistency audit

### Resolved transcription ambiguities

1. **Resident time-field meaning:** Phase B used generic `t_front/t_tail`; this plan names them downstream `eligible_front_s/eligible_tail_s`. Actual crossing times are step-transfer records and generate the next eligibility segment.
2. **Source generation versus M0 boundary arrivals:** absolute generation profiles are bucketed as demand available by `t_k`; they do not alter the accepted macro within-step scheduler. Actual admission starts physical link timing.
3. **Movement shorthand correspondence:** no backward pointer is stored. Deterministic rescan of the unchanged ledger applies solver quotas; explicit boundary-transfer records preserve evidence after the operation.
4. **Exact merge diagnostics:** telemetry includes safe-nonmerge reasons but has no behavioral authority.

### Contradictions

No contradiction remains between this plan, `AGENTS.md`, fixed `PROJECT_CONTEXT.md` invariants, the accepted Milestone 0 plan, or the closed Phase B contracts.

### Remaining ambiguities

No scientific or engineering ambiguity remains that blocks implementation. Exact class/function names may be adjusted mechanically during a component pass only when responsibilities, fields, units, equations, and acceptance behavior remain identical; otherwise reopen the plan.

## 16. Phase F corrective D/E amendment

The Phase F closure audit found no contradiction in the accepted traffic or differentiation semantics, but identified five narrower implementation/evidence gaps. The user selected FQ1 Candidate A, FQ2 Candidate A, revised FQ3 Candidate A, delegated FQ4 to Codex (resolved as revised Candidate A), and selected FQ5 Candidate A. The detailed questions, alternatives, decisions, and qualifications remain archived in `temp_content/milestone_1_ordered_route_mass_ledger/phase_F_closure_audit_open_questions.md` and `phase_F_dialogue.md`.

This amendment does not reopen G1–G5, change macro authority, alter ledger selection/order/timing arithmetic, modify exact-merge predicates, or change acceptance tolerances. It authorizes only the following bounded add-on sequence once the user explicitly opens it.

### 16.1 Phase D_amend1 — corrective implementation

Component 1.5 work:

- enforce at every boundary the complete generated-by-route identity across source queues, resident ledgers, and completed route mass, together with aggregate completion versus macro sink exit;
- retain graph-connected actual crossing-time intervals, aligned one-for-one with every positive source, node, and sink `MesoBoundaryTransfer`, strictly as observational evidence; resident eligibility remains the sole persistent behavioral timing authority;
- add public-boundary shape/state validation only where malformed source/link ledger tuples or completed-route tensors are currently reachable; and
- do not change transfer selection, accepted macro flow, route progression, ordering, mass, or eligibility arithmetic.

Integration and evidence work:

- retain passive per-step/per-link telemetry from the exact-merge calls made by actual integration orchestration;
- remove the unused configured FD-step list, serialize the step sizes and stability/pass rows actually returned by the accepted directional-check harness, and record effective tolerances;
- identify macro exactness as backed by the existing fail-fast equivalence checks; no duplicate macro measurement logic is required;
- record final artifact paths mechanically where available; and
- add the approved reachable coupled invalid-case tests plus construction/property evidence for unrepresentable aggregate-arrival disagreement.

Write `temp_content/milestone_1_ordered_route_mass_ledger/phase_D/phase_D_amend1_worklog.md`. D_amend1 implements only this approved amendment and does not run or claim the E_amend1 acceptance gate.

### 16.2 Phase E_amend1 — corrective validation

Validation scope:

- run boundary-by-boundary aggregate and per-route conservation/reference checks;
- check source/node/sink actual-interval values, alignment, graph connectivity, and unchanged behavior;
- directly test reachable Component 1.5 and coupled invalid states, and provide construction/property evidence for states made unrepresentable by authoritative constructors;
- demonstrate at least one successful exact merge through the actual integration orchestration, correct installation of the returned ledger, conservation/order/timing preservation, and observational telemetry with no behavioral authority;
- pass boundary-wise conservation and all approved coupled invalid-case checks;
- pass directional autograd/finite-difference validation with the actual executed scan rows recorded;
- rerun the full test suite and compilation check; and
- create a new non-overwriting JSON/Markdown evidence artifact, retaining earlier artifacts as immutable superseded provenance.

Write `temp_content/milestone_1_ordered_route_mass_ledger/phase_E/phase_E_amend1_validation.md`; place durable evidence under `reports/milestone_1_ordered_route_mass_ledger/`. Stop after E_amend1 for Phase F user review. Do not close Milestone 1 or start Milestone 2 automatically.

### 16.3 Corrective error and reopen policy

Syntactic, harness, schema-transcription, and implementation-conformance defects may be corrected within the applicable D/E pass and retested. If evidence instead requires changing a scientific semantic, accepted component interface, exact-merge predicate, macro-passivity contract, differentiability boundary, or acceptance threshold, stop and reopen the smallest affected contract under the ordinary milestone workflow.

## 17. Closure and handoff

Phase C is complete and this file is authoritative for the Milestone 1 implementation pass.

Components 1.1 through 1.6 are user-accepted and the initial D/E integration tests pass. Phase F subsequently resolved all five evidence-conformance questions and approved the bounded amendment in Section 16.

D_amend1 implemented the bounded amendment and E_amend1 passed its focused, full-regression, compilation, and immutable-artifact checks. The worklog is `temp_content/milestone_1_ordered_route_mass_ledger/phase_D/phase_D_amend1_worklog.md`; the validation record is `temp_content/milestone_1_ordered_route_mass_ledger/phase_E/phase_E_amend1_validation.md`; durable v3 evidence is under `reports/milestone_1_ordered_route_mass_ledger/reference_cpu_float64_v3/`.

The user explicitly closed Milestone 1 on 2026-09-03. The durable closure and no-rerun handoff record is `temp_content/milestone_1_ordered_route_mass_ledger/phase_F_closure.md`. It records accepted results, artifact checksums, superseded evidence, uncommitted-worktree state, residual risks, and verification instructions.

Milestone 2 is now eligible to begin at Phase A only. When explicitly opened, create `m2_interaction_zone` from merged `main` at or after `309cba6` and switch to that branch. Milestone 1 experiments and tests do not need to be rerun merely to open Milestone 2 planning; verify the recorded artifacts/checksums and rerun only under the conditions stated in the closure record. Do not implement Milestone 2 before its Phase A–C gates are complete.
