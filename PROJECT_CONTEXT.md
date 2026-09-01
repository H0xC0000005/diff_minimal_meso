# PROJECT_CONTEXT.md

## Project status

Status: Milestone 0 closed on `main` on 2026-08-31 after all seven component checkpoints and D/E integration passed. Optional Milestone 0_a1 closed separately on branch `m0_a1` on 2026-09-01 and remains intentionally unmerged. Milestone 1 has not started; its next allowed action is Phase A planning on a new branch created from current `main`.

The repository follows the milestone sequence in `PLANS.md`. Closed Milestone 0 is governed by `docs/milestones/milestone_0_ltm_baseline.md`. Milestone 1 must preserve that accepted macro baseline and follow the full gated workflow before implementation. The optional visualizer is available on branch `m0_a1` for reference but is not part of the Milestone 1 branch basis.

## 1. Research motivation

Differentiable traffic simulation is useful only if the resulting gradients support decisions in the traffic system one ultimately cares about. A purely macroscopic differentiable model is computationally attractive, but it can erase finite movement order, lane-specific storage, shared-lane blocking, and turn-pocket effects. A fully microscopic differentiable model retains those details but introduces discrete vehicle generation, leader/lane-change topology, categorical decisions, and much larger computational graphs.

This project studies a middle point:

> retain only the discrete semantic traffic structure needed for signal-relevant mesoscopic behavior, while keeping traffic quantity and most physical dynamics continuous and differentiable.

The intended contribution is methodological and empirical rather than a claim that each traffic component is novel.

## 2. Primary research question

> How much traffic discreteness is actually required to preserve useful mesoscopic gradients for downstream fixed-time signal optimization?

The central representation hypothesis is:

> Continuous traffic mass plus selectively retained route/movement ordering can preserve important mesoscopic queue and blocking behavior without integer vehicle transfers or full microscopic agent dynamics.

The downstream hypothesis is:

> In settings where finite movement order, shared-lane FIFO blocking, or finite turn-pocket storage matter, gradients from the mesoscopic representation will predict useful high-fidelity signal perturbations better than gradients from a matched homogeneous macroscopic model.

Both statements are hypotheses to test, not assumptions to bake into evaluation.

## 3. Intended scope

### Fixed initial scope

- Dynamic network loading with first-order traffic-flow physics.
- Long links represented by LTM/Newell-style aggregate propagation.
- Local lane-aware mesoscopic refinement only near downstream nodes.
- Continuous vehicle-equivalent traffic mass.
- Fixed/exogenous route set and route choices in V1.
- Fixed lane geometry and lane permissions.
- Fixed signal phase sequence/topology.
- Initial downstream control variable: green split / fixed-time service allocation.
- Deterministic scenarios first; scenario sampling may be added later while remaining exogenous to the signal parameter during differentiation.

### Deferred unless separately approved

- microscopic car-following;
- explicit lane-changing dynamics;
- endogenous stochastic route choice;
- dynamic user equilibrium coupled into the gradient loop;
- actuated/adaptive phase sequence;
- integer vehicle generation or vehicle-by-vehicle LTM transfer;
- reinforcement learning;
- network-wide lane expansion;
- learned route/order compression;
- lossy metadata defragmentation;
- offsets across a large coordinated network;
- cycle-length optimization before lost-time/within-cycle effects are modeled credibly.

## 4. Methodological lineage and what is reused

The project deliberately uses established components where possible.

### Multi-commodity LTM / Yperman lineage

Use the decomposition:

`aggregate kinematic-wave link physics + boundary cumulative counts + route/commodity bookkeeping + node transfer`.

The key transferable principle is that route identity can ride on top of common aggregate traffic physics. The project extends this locally by retaining finite ordered route mass where movement order affects node service.

### Generic first-order node-model lineage

Node flows must respect well-grounded demand/supply, conservation, turning, and active-constraint logic. The implementation should not invent node transfer rules that violate these admissibility requirements merely for differentiability.

### Partial-FIFO / high-dimensional node lineage

Macroscopic full-link FIFO can create unrealistic blocking when only some output movements are obstructed. Partial FIFO and lane-aware node models motivate localized blocking rather than indiscriminate link-wide blocking.

A key limitation of aggregate multi-commodity formulations is isotropic mixing within a movement/stream. This project intentionally retains local finite order so that identical movement proportions with different sequences can behave differently when FIFO service makes that distinction physically relevant.

### Lane-aware macroscopic node lineage

Lane allocation/equilibrium models motivate a separation between desired lane choice and realized lane accessibility. The project may reuse an established resistance/equilibrium concept, but finite storage and ordered local traffic determine what can actually occupy/reach each lane.

### Continuum signal lineage

Continuous signal service is useful for differentiability and coarse DNL. It approximates on/off signal service rather than reproducing every switching event. The no-spillback approximation error is controlled under restrictive conditions and scales with cycle/capacity; spillback weakens the guarantee. Therefore the continuum signal should be treated as an explicit modeling approximation whose effect is validated, not as exact signal physics.

### Individual-tracking mesoscopic LTM lineage

Vehicle-tracking LTM work shows that identity/order can be layered onto LTM while preserving aggregate traffic physics, but integerized vehicle transfers create staircase behavior that is hostile to ordinary AD. This project retains continuous mass and discrete semantic metadata instead.

### Differentiable LTM / hybrid traffic lineage

Recent differentiable LTM work supports using cumulative-count traffic dynamics with piecewise differentiable `min/max` operators. Hybrid differentiable traffic work shows that discrete representation transitions may require special gradient treatment. The project seeks to avoid unnecessary discrete physical transitions by keeping the physical mass path continuous.

## 5. Core network architecture

The working architecture is:

`upstream node -> long-link LTM propagation -> inbound local interaction-zone boundary -> per-lane cumulative-mass state + ordered route-mass metadata -> lane/storage/FIFO/signal service -> downstream node/link -> LTM`

The local refinement exists because homogeneous link-level state cannot represent all signal-relevant finite order/storage effects.

### 5.1 Long-link physical backbone

For a link `a`, maintain upstream/downstream cumulative counts such as:

- `N_in_a(t)`: cumulative mass that has entered the link;
- `N_out_a(t)`: cumulative mass that has left the link.

With a triangular fundamental diagram, representative LTM sending/receiving forms are:

`S_a(t) = min(N_in_a(t + dt - tau_f) - N_out_a(t), q_max * dt)`

`R_a(t) = min(N_out_a(t + dt - tau_b) + K_a - N_in_a(t), q_max * dt)`

where `tau_f` is free-flow travel time, `tau_b` is backward-wave travel time, and `K_a` is link jam storage in vehicle-equivalent mass.

Exact indexing and discretization must be frozen by an approved milestone and verified against reference cases before larger components are built.

Cumulative counts are continuous real-valued mass states. Their updates are additive and the core constraints use sums/min/max, making the physical path piecewise differentiable almost everywhere without integer rounding.

### 5.2 Persistent route-mass ledger

Persistent semantic identity should be represented by an ordered ledger rather than duplicated per-node movement strings.

Conceptual entry:

`(route_id, route_position, mass)`

where:

- `route_id` references a route stored once in a global route table;
- `route_position` identifies progress along that route;
- `mass > 0` is continuous vehicle-equivalent mass.

The node projects route metadata to its current movement through a fixed mapping:

`movement = phi_node(route_id, route_position)`.

Current movement is therefore a local view derived from persistent route identity.

Lossless adjacent entries with identical relevant route state may be merged. Splitting continuous mass is allowed. Mass must never be dropped.

### 5.3 Per-lane local interaction zone

Only a fixed inbound zone near the node is refined by lane.

The local continuous state is conceptually:

`W[lane, ordered_mass_bin, movement]`

where bins are ordered in cumulative mass/FIFO-rank coordinates rather than fixed physical-space cells.

This grid carries continuous occupancy/composition/storage state. Physical queue length is not inferred by treating one bin as a fixed number of meters; physical storage must remain consistent with lane length, jam density, vehicle-equivalent conversion, and the LTM boundary state.

The grid must be per lane. A per-link grid would erase the lane-specific effects the refinement is intended to model.

### 5.4 Exact local order metadata

A mixed grid bin does not by itself define exact FIFO service order.

For example, a bin with `(T=4, L=1)` does not tell whether the realized order is:

- `[T(4), L(1)]`, or
- `[L(1), T(4)]`.

These can have different signal service under a red left-turn movement.

Therefore each committed lane retains an ordered route/movement-mass view for FIFO service. The grid is the continuous physical/composition state; the ordered ledger is the semantic ordering state.

The node uses the ordered prefix that is service-eligible under current movement/signal/storage constraints. Active-prefix changes are nonsmooth events, but continuous masses within a fixed structural ordering remain differentiable through arithmetic and `min/max`-type service operations.

### 5.5 Lane allocation versus realized occupancy

Do not conflate desired lane allocation with what can physically enter a lane.

Conceptually:

`perceived lane resistance -> desired allocation -> finite storage/accessibility/order constraints -> realized lane occupancy`.

Current known blockage may affect perceived resistance. Drivers need not anticipate all future overflow/order in V1.

Example: if a left-turn pocket can store 10 vehicle-equivalents but 20 left-turn equivalents seek it, the desired allocation may point left traffic toward the pocket while only 10 can enter; the remainder stays upstream and can obstruct other movement access depending on geometry/order.

The exact lane-choice/equilibrium rule is not yet fixed. It must be selected in an approved milestone from an established, minimally sufficient formulation.

### 5.6 Spillback and local-zone coupling

The local interaction zone must not clip queues when its detailed region fills.

Required causal behavior:

`local lane/pocket storage fills -> receiving capacity falls -> upstream link/node sees lower receiving -> queue propagates upstream through LTM/Newell`.

The local zone is a refinement of node-facing structure, not an independent finite buffer that may discard traffic beyond its configured length.

## 6. Signal-control model

### Initial control scope

Use fixed phase sequence/topology and optimize continuous fixed-time service parameters, beginning with green split.

A continuum signal model may convert phase green ratio into average movement service/capacity. This is attractive for AD because the control variable can influence continuous service directly.

Do not interpret continuum service as exact on/off switching. Validation should include conditions where spillback/order may expose approximation error.

### Cycle length

Cycle length is deferred as a primary optimization variable until the model includes a reason that absolute cycle length matters, such as startup lost time, clearance lost time, or richer within-cycle dynamics. Without such effects, pure green-ratio service may be scale-invariant or weakly identified with respect to cycle length.

### Offset

Network offset optimization is deferred until single/intersection fixed-time behavior and cross-fidelity gradient usefulness are established.

## 7. Differentiability design

The project uses **selective discreteness**.

### Structural/non-differentiated metadata

Initially treat as fixed:

- network topology;
- lane permissions;
- route IDs and route positions;
- route-to-movement mapping;
- movement labels;
- phase sequence;
- exact semantic ordering skeleton for a fixed scenario.

### Gradient-carrying continuous state

Examples include:

- packet/ledger masses;
- cumulative link counts;
- sending/receiving quantities;
- local grid movement masses;
- lane allocation fractions when represented continuously;
- storage/queue quantities;
- continuum signal service;
- signal timing parameters;
- downstream objective.

### Expected nonsmoothness

`min`, `max`, active constraints, storage saturation, and active-prefix changes create piecewise-smooth behavior. Zero or discontinuous derivatives at active-set boundaries are not automatically bugs.

Do not introduce smoothing solely to make gradients nonzero. Smoothing or surrogate gradients require a separate approved methodological decision and must be evaluated for forward and gradient bias.

### Scenario randomness

The core simulator need not be stochastic. For deterministic departures/routes/order, `J(g)` is deterministic.

If scenario sampling is introduced, represent exogenous scenario data by `xi` and optimize:

`F(g) = E_xi[J(g; xi)]`.

Hold sampled route/order realization fixed while differentiating with respect to signal parameters in V1. Use common random numbers / fixed scenario banks when comparing methods.

## 8. Macro versus meso comparison

A matched macroscopic baseline is required. It should share as much physics/objective/demand/signal parameterization as possible while omitting the extra mesoscopic order/storage state under study.

The comparison should isolate representation rather than unrelated numerical choices.

Key causal experiment:

- construct two scenarios with the same aggregate link demand, turning totals, and macro state;
- change only finite movement order, e.g. `[T(5), L(5)]` versus `[L(5), T(5)]`;
- the macro baseline should see the same state/gradient;
- the mesoscopic model may produce different service/sensitivity;
- a richer simulator should determine which sensitivity better predicts actual objective change.

This is a central experiment because it directly tests whether retained mesoscopic information changes the intervention in a useful way.

## 9. Validation hierarchy

### 9.1 Forward/reference correctness

Validate component equations and conservation before optimization.

Examples:

- single-link LTM propagation;
- sending/receiving under free flow and congestion;
- one-to-one node transfer;
- diverge/merge capacity constraints;
- route-mass projection and conservation;
- exact lossless split/merge of ledger entries;
- local-zone storage and spillback coupling.

### 9.2 Macro-equivalence cases

When finite ordering/lane effects are intentionally neutralized, mesoscopic results should agree with a matched aggregate model within explicit tolerance.

This establishes that local refinement does not arbitrarily change basic flow physics.

### 9.3 Meso-sensitive cases

Use deliberately small cases that expose one effect at a time:

- same movement proportions, different order;
- shared-lane red-movement head-of-line blocking;
- finite turn-pocket fill/overflow;
- desired lane allocation versus realized accessibility;
- spillback from local storage into upstream LTM.

### 9.4 Gradient correctness

For fixed structural scenarios:

- compare AD/autograd directional derivative with central finite difference in `float64`;
- test multiple directions and step sizes;
- avoid interpreting checks exactly on known active-set/event boundaries as smooth-function tests;
- inspect gradient paths and state sensitivities when discrepancies appear.

### 9.5 Gradient statistics if scenarios are sampled

Characterize scenario-to-scenario gradient variance separately from model/aggregation bias.

A deterministic macro gradient has low/no Monte Carlo variance but may still be biased relative to mesoscopic expected sensitivity because aggregation and nonlinear congestion/FIFO effects do not commute.

### 9.6 Downstream signal optimization

Use fixed scenario banks / sample-average approximation for optimization and held-out scenarios for evaluation.

Compare at minimum:

- matched macro gradient direction/optimizer;
- mesoscopic gradient direction/optimizer;
- random/reverse-direction controls for diagnostic perturbation tests when useful.

### 9.7 Cross-fidelity verification

After internal forward/gradient validation, evaluate signal perturbations/plans in a richer reference simulator such as SUMO.

Useful diagnostic for a signal vector `g`:

`d = -grad(J_meso) / ||grad(J_meso)||`

Evaluate `J_ref(g + epsilon d) - J_ref(g)` over small `epsilon` and compare against the matched macro direction, reverse direction, and controls.

The final claim should be about **transfer of useful sensitivity**, not merely optimization of the simulator that generated the gradient.

## 10. Data structures and implementation boundaries

Exact names may adapt to repository conventions, but responsibilities should remain separated.

Suggested modules:

- `network`: graph, links, lanes, movements, route table, static permissions;
- `fd`: fundamental-diagram parameters and traffic units;
- `ltm`: cumulative-count link propagation and link boundary state;
- `nodes`: generic demand/supply transfer and node constraints;
- `routes`: route table and route-to-movement projection;
- `ledger`: ordered continuous route-mass state, splitting, lossless merging;
- `local_zone`: per-lane cumulative-mass grid and coupling to link/node state;
- `lane_allocation`: desired allocation and realized accessibility logic;
- `signals`: fixed-time/continuum signal service and parameterization;
- `simulation`: deterministic rollout/update scheduling;
- `objectives`: signal-optimization objectives and component reporting;
- `gradients`: gradient diagnostics and directional checks, not alternative scientific methods unless approved;
- `experiments`: scenario generation, matched macro/meso comparisons, optimization;
- `evaluate`: held-out and cross-fidelity evaluation;
- `tests`: numerical, conservation, forward, order/storage, and gradient tests.

Keep structural metadata separate from differentiable tensors in the API so it is clear which values define scenario semantics and which values carry gradients.

## 11. Units and numerical conventions

Defaults to freeze in the first implementation milestones:

- time in seconds;
- length in meters;
- speed in meters/second;
- flow in vehicle-equivalent mass/second unless a documented interface requires another form;
- storage in vehicle-equivalent mass;
- cumulative counts continuous real values;
- explicit time-step indexing and update order documented in code/tests;
- `float64` for finite-difference and gradient-validation cases;
- production precision may be reconsidered only after numerical agreement is established.

Do not estimate local storage from a nominal 4 m vehicle length alone. Use lane geometry and jam density / effective vehicle-equivalent spacing consistently with the fundamental diagram.

## 12. Metadata fragmentation policy

V1 prioritizes correctness and auditability over aggressive compression.

Default order:

1. preserve exact ordered route-mass metadata;
2. perform exact adjacent merges when route state/suffix semantics are identical;
3. project/collapse to current movement only as a temporary node-computation view;
4. profile memory/runtime before introducing persistent lossy compression.

Potential future compression hierarchy, only if approved:

`exact route/suffix merge -> exact behavioral-prefix merge -> bounded local-behavior merge with explicit size/horizon limits`.

Do not homogenize large structured bursts merely to reduce ledger length.

## 13. Expected scientific contribution

A successful result would support both of the following:

### H1 — representation/differentiability

Signal-relevant mesoscopic behavior can be retained with continuous traffic mass plus selective discrete semantic order, avoiding integerized flow and full differentiable agent dynamics in the initial scope.

### H2 — gradient value

The additional mesoscopic state produces signal sensitivities that transfer better to a richer reference model than matched homogeneous macro gradients in cases where order/lane/storage effects matter.

A scientifically useful negative result is also possible: the extra mesoscopic representation may not improve downstream gradient transfer enough to justify its complexity. The experiment design must permit that conclusion.

## 14. Non-claims

Do not describe the project as:

- the first differentiable mesoscopic simulator;
- the first LTM with route identity;
- the first lane-aware node model;
- the inventor of continuum signal control;
- a complete replacement for microscopic simulation.

The intended novelty lies in selective-discreteness representation/differentiability and in testing whether the added mesoscopic state improves **gradient transferability/usefulness** under matched comparisons.

## 15. Change control

If interactive user instructions override a default, record the override in the current milestone phase archive and follow it.

If the user requests a change to a **Fixed** item, treat it as an explicit methodology amendment: document the affected assumptions and update `PROJECT_CONTEXT.md` only after the decision is clear.
