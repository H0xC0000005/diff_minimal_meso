# Milestone 0 — reproducible differentiable LTM baseline

Status: **Closed**

Approval date: 2026-08-27

Closure date: 2026-08-31

Phase: Phase F complete; all component and integration gates accepted

## 1. Scientific purpose

Build and validate the continuous-mass macroscopic backbone on which later ordered mesoscopic refinements will depend. Milestone 0 must establish:

1. correct triangular-FD and cumulative-count LTM propagation;
2. conservative movement projection and first-order node transfer;
3. a deliberately restricted, unique cycle-averaged signal service model;
4. deterministic network rollout with closed mass accounting; and
5. stable-active-regime derivative correctness with respect to physical green split.

The milestone establishes a trustworthy matched macro baseline. It does **not** test the project's central gradient-usefulness hypothesis.

## 2. Scope and non-goals

### In scope

- deterministic, empty-start, fixed-step dynamic network loading;
- triangular-FD LTM with exact shifted-time linear interpolation;
- continuous vehicle-equivalent mass throughout;
- fixed node-local turning fractions and compact movement maps;
- ordinary Tampère oriented-capacity-proportional Branch-T/ORCA nodes;
- restricted Han-style continuum signal service within H1–H4 below;
- fixed phase topology and physical green fractions with unit total green;
- full aggregate FIFO;
- PyTorch piecewise differentiation without smoothing or custom backward rules;
- hand/scalar/literature reference evidence, conservation, and directional AD/JVP/FD checks.

### Non-goals

- route ledger, route choice, ordered traffic metadata, lane state, partial FIFO, or turn-pocket storage;
- nonempty end-to-end initialization;
- on/off phase switching, phase order within a cycle, lost time, cycle length, offset, or actuated control;
- the original Tampère signalized internal-supply extension or COS/S-ORCA;
- signal optimization, claims of gradient usefulness, SUMO comparison, or macro–meso experiments;
- UNsim/UXsim installation or execution;
- JIT, scan, checkpointing, sparse batching, GPU optimization, or production-precision tuning.

## 3. Milestone-wide frozen contracts

### 3.1 Units and notation

| Quantity | Symbol/example | Unit | Runtime form |
|---|---|---|---|
| time | `t`, `dt`, `tau_f`, `tau_b` | s | structural Python float |
| length | `L` | m | structural Python float |
| speed | `v`, `w` | m/s | structural Python float |
| density | `k_j`, `k_c` | veh-eq/m | structural Python float |
| rate/capacity | `C`, `s_i` | veh-eq/s | structural float or float64 tensor |
| interval demand/supply/service/flow | `S`, `R`, `D`, `f` | veh-eq | float64 tensor |
| cumulative count/occupancy/queue/storage | `N`, `x`, `Q`, `K` | veh-eq | float64 tensor |
| turning fraction/green/exposure | `beta`, `g`, `e` | dimensionless | fixed or differentiable float64 tensor |

Conversions are named and explicit:

`step_mass = rate * dt`

`rate = step_mass / dt`.

Dynamic component interfaces use interval mass, never an ambiguously named `flow` rate. Continuous mass is never rounded, floored, or cast to integer.

### 3.2 Environment and import convention

- Environment: `/home/zpz/miniconda3/envs/diff_minimal_meso`.
- Verified Phase-C base: Python 3.12.13; the environment initially contained only the base Conda/Python tooling and no PyTorch or pytest.
- Intended direct project dependencies: PyTorch and pytest only. They are installed lazily after separate implementation authorization; realized version, build, channel, and exact installation commands are recorded in the applicable D/E archives.
- Realized dependency state at D2 opening: pytest 9.0.3 and GPU-capable PyTorch 2.5.1 build `py3.12_cuda12.4_cudnn9.1.0_0` from the `pytorch` channel, with `pytorch-cuda` 12.4 and `pytorch-mutex` `cuda`. MKL is pinned to 2023.1.0 because MKL 2025 caused an unresolved `iJIT_NotifyEvent` import and MKL 2020 lacked the `.so.2` ABI expected by this PyTorch build.
- GPU-capable installation is the environment requirement even where Milestone 0 validation remains CPU/float64 for deterministic reference evidence. Host GPU access may require command escalation: an escalated smoke test detected the Tesla V100-SXM2-16GB and completed a CUDA tensor operation. A sandbox-only CUDA-unavailable result must not trigger a CPU-only fallback without first retrying the required probe or command with the minimum justified escalation.
- Dependency management is Conda-only. Do not add `pyproject.toml`, editable-install machinery, or treat `src/` as a packaging protocol.
- Commands run from the repository root with `PYTHONPATH=src` and the environment's absolute Python path.
- UNsim, UXsim, and other differentiable simulators are not installed in this environment.

Planned test command after dependencies and Component D1 are authorized:

```bash
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q
```

### 3.3 Dtype, device, and tolerance policy

- All Milestone 0 implementation and evidence use `torch.float64` on CPU.
- Structural indices use `torch.int64`; masks use `torch.bool`.
- Structural scalar validation uses finite Python values before tensor arithmetic where practical.
- Simplex and turning-sum validation: `rtol=0`, `atol=1e-9`.
- Hand/reference forward equality: `rtol=1e-10`, `atol=1e-12`.
- Conservation/admissibility at test scale: residual must satisfy
  `abs(residual) <= 1e-10 + 1e-10 * max(1, reference_mass)`.
- Exact algorithmic ties use exact equality of computed float64 restriction values; no epsilon changes which constraint binds. Test tolerances are not solver semantics.
- NaN, infinity, invalid shapes, invalid topology, negative dynamic inputs, and nonpositive structural capacities fail fast.

### 3.4 Structural versus differentiated data

Structural/nondifferentiated:

- topology and link/node IDs;
- local input/output/movement order and index arrays;
- turning fractions and phase-permission matrix;
- FD/link parameters, delay indices, interpolation weights, horizon, and phase order;
- node kind, active IDs/masks, tied IDs, and pivot IDs.

Gradient-carrying when used in a differentiated rollout:

- physical green `g`;
- signal exposure/service arithmetic;
- cumulative histories, queues, demand/supply, movement demands/flows, and objective tensors.

Production arithmetic must not call `.item()`, NumPy, `detach`, or reconstruct tensors from tensor values on a gradient path. Branch selection and diagnostic IDs are structural, while arithmetic on the selected branch remains in PyTorch.

### 3.5 System data flow and update order

At interval `k`, representing `[t_k,t_{k+1}]`:

`arrivals/source availability`

`-> all-link sending/receiving from histories through t_k`

`-> movement-demand projection`

`-> fixed signal exposure/service parameters`

`-> independent node solves using the common pre-update state`

`-> boundary-flow aggregation and source/sink flows`

`-> source queue, cumulative count, and sink count updates`

`-> immutable StepResult record`.

There is no within-step topological sweep. Traffic already eligible at a link's downstream boundary may cross its node in interval `k`; mass newly entering a link cannot also exit that complete link in the same step.

## 4. Files and dependency order

Create only during the authorized component D/E cycles:

```text
src/diff_minimal_meso/__init__.py
src/diff_minimal_meso/fd.py
src/diff_minimal_meso/ltm.py
src/diff_minimal_meso/movements.py
src/diff_minimal_meso/nodes.py
src/diff_minimal_meso/signals.py
src/diff_minimal_meso/simulation.py
src/diff_minimal_meso/objectives.py
src/diff_minimal_meso/gradients.py
tests/references/ltm_scalar_reference.py
tests/references/node_scalar_reference.py
tests/test_fd.py
tests/test_ltm.py
tests/test_movements.py
tests/test_nodes.py
tests/test_signals.py
tests/test_simulation.py
tests/test_objectives_gradients.py
tests/test_milestone_0_integration.py
configs/milestone_0/minimal_signalized.json
scripts/run_milestone_0_diagnostics.py
```

Implementation/checkpoint order is strictly 0.1 through 0.7, followed by D_integration/E_integration. A later component may import only accepted earlier components.

## 5. Component 0.1 — traffic units and triangular FD

### Responsibility, records, and equations

`fd.py` owns immutable structural records:

```text
TriangularFD(
    free_speed_mps: float,
    backward_wave_speed_mps: float,  # positive magnitude
    jam_density_veh_per_m: float,
)

LinkGeometry(length_m: float)

LinkFDParameters(
    critical_density_veh_per_m: float,
    capacity_veh_per_s: float,
    jam_storage_veh: float,
    free_flow_time_s: float,
    backward_wave_time_s: float,
)
```

For `v>0`, `w>0`, `k_j>0`, and `L>0`:

`k_c = (w / (v + w)) k_j`

`C = v k_c = w (k_j - k_c)`

`K = k_j L`

`tau_f = L / v`

`tau_b = L / w`.

`derive_link_fd(fd, geometry, dt_s)` validates finiteness/positivity and the causal fixed-step condition

`dt_s <= min(tau_f, tau_b)`.

This condition ensures every end-of-interval shifted history query is no later than `t_k` and implies `C*dt <= K`. Capacity is derived, not calibrated or overridden.

### Reference basis and adaptation

Use the triangular-FD/Newell/Yperman identities. UNsim naming may inform review but no external runtime or copied implementation is used.

### Implementation pieces

- `critical_density(fd)`;
- `capacity_veh_per_s(fd)`;
- `jam_storage_veh(geometry, fd)`;
- `travel_times_s(geometry, fd)`;
- `derive_link_fd(fd, geometry, dt_s)`;
- construction validators.

All values are structural; Component 0.1 has no differentiable path in Milestone 0.

### Required checks and acceptance

Hand anchor:

`L=300`, `v=15`, `w=5`, `k_j=0.15`, `dt=10`

must yield

`k_c=0.0375`, `C=0.5625`, `K=45`, `tau_f=20`, `tau_b=60`, and `C*dt=5.625`.

Tests must cover the capacity identity, `0<k_c<k_j`, dimensional values, nonfinite/nonpositive primitives, `dt>tau_f`, and `dt>tau_b`. All hand values meet the forward-reference tolerance.

**Checkpoint 0.1:** report equations, exact hand output, rejected inputs, files changed, commands, and tests; stop for user acceptance before 0.2.

## 6. Component 0.2 — cumulative-count LTM link

### Responsibility and records

`ltm.py` owns:

```text
CumulativeLinkState(
    n_in: Tensor[H+1, A],
    n_out: Tensor[H+1, A],
)

LinkStepResult(
    sending: Tensor[A],
    receiving: Tensor[A],
    occupancy: Tensor[A],
    sending_active: BoolTensor[A, 2],   # availability, capacity
    receiving_active: BoolTensor[A, 2], # storage, capacity
)
```

Histories are boundary-time values: row `k` is the cumulative value at `t_k=k*dt`. Empty-start prehistory is identically zero. Synthetic prehistory is allowed only in isolated reference tests.

### Exact history query and equations

For a causal query `q<=t_k`, define `r=q/dt`, `l=floor(r)`, and `theta=r-l`. For `q<=0`, return the empty-start boundary value `0`; otherwise:

`H(q) = (1-theta) H[l] + theta H[l+1]`.

At an exact boundary, `theta=0`. Indices and weights are structural. A query after `t_k` is an error.

For interval `k`:

`q_f = t_{k+1} - tau_f`

`q_b = t_{k+1} - tau_b`

`raw_S_a[k] = N_in_a(q_f) - N_out_a(t_k)`

`S_a[k] = min(C_a dt, max(0, raw_S_a[k]))`

`raw_R_a[k] = N_out_a(q_b) + K_a - N_in_a(t_k)`

`R_a[k] = min(C_a dt, max(0, raw_R_a[k]))`.

Accepted boundary masses must satisfy `0<=inflow<=R` and `0<=outflow<=S`. Update out of place:

`N_in[k+1] = N_in[k] + inflow[k]`

`N_out[k+1] = N_out[k] + outflow[k]`

`occupancy[k] = N_in[k] - N_out[k]`.

The physical nonnegative maximum is part of LTM boundary semantics, not a numerical gradient repair. Materially inadmissible histories fail validation.

### Reference basis and independent oracle

- Yperman (2007) equations and Example 4.5 for cumulative/interpolation/spillback reasoning;
- compatible literal one-link cases transcribed from UNsim v0.12.0 commit `5c396357cc540f2433c3e33689b415b1df1f6e79`, without execution;
- a standard-library scalar reference that independently evaluates the frozen equations.

### Required checks and acceptance

- empty-link sending `0` and receiving `Cdt`;
- the Component 0.1 link with a pulse admitted at `t_0` has no downstream eligibility before the `tau_f=20 s` history query permits it;
- capacity-, storage-, fractional-interpolation-, and downstream-congestion tables;
- exact occupancy update identity and monotone cumulative counts;
- admissible occupancy in `[0,K]` and continuous values including `0.125`;
- production/reference agreement at forward-reference tolerance;
- reverse-mode and central FD on stable branches; event ties are forward/diagnostic tests, not smooth checks.

**Checkpoint 0.2:** report history/index table, hand/scalar/source-reference results, conservation, gradient evidence, and provenance; stop before 0.3.

## 7. Component 0.3 — fixed movement-demand projection

### Responsibility and records

`movements.py` owns the one validating construction path for:

```text
NodeMovementMap(
    input_link_ids: tuple[int, ...],
    output_link_ids: tuple[int, ...],
    movement_input_index: LongTensor[M],
    movement_output_index: LongTensor[M],
    turning_fraction: Tensor[M],
)
```

Input and output link order is explicit. Allowed movements are ordered lexicographically by `(local_input_index, local_output_index)`. Duplicate movements, missing positive-demand input rows, invalid indices, negative/nonfinite fractions, and row sums outside `1 +/- 1e-9` are rejected.

### Equation and functions

For movement `m=(i,j)`:

`D_m = beta_m S_i`.

Functions:

- `build_movement_map(...)`;
- `validate_turning_fractions(...)`;
- `project_oriented_demand(sending, movement_map)`;
- `aggregate_input_flow(movement_values, movement_map)`;
- `aggregate_output_flow(movement_values, movement_map)`.

Projection uses gather/multiply; aggregation uses out-of-place grouped tensor reduction. `beta` and indices are structural; the map is linear in sending and preserves its graph.

### Required checks and acceptance

- `S=2.5`, `beta=(0.25,0.75)` gives `D=(0.625,1.875)`;
- exact zero sending retains shape and gives zero demands;
- input movement sums equal sending within the structural tolerance;
- Jacobian equals the fixed projection matrix;
- mapped input/output relabeling produces mapped-identical results;
- every invalid mapping class above fails fast.

**Checkpoint 0.3:** report map layout, validation table, conservation, Jacobian, and permutation evidence; stop before 0.4.

## 8. Component 0.4 — first-order node solver

### Responsibility and node regimes

`nodes.py` consumes a validated `NodeMovementMap`, oriented demand `D[M]`, output receiving mass `R[J]`, positive structural input capacities, and an explicit structural node kind. It returns:

```text
NodeFlows(
    movement_flow: Tensor[M],
    input_outflow: Tensor[I],
    output_inflow: Tensor[J],
    binding_mask: BoolTensor[number_of_constraints],
    tied_constraint_ids: tuple[ConstraintID, ...],
    selected_pivot_ids: tuple[ConstraintID, ...],
)
```

`ConstraintID` contains structural `(kind, node_local_index_or_pair)`. Verbose iteration traces are test/debug-only.

Every node is constructed as exactly one of:

1. `ORDINARY_ORCA`; or
2. `RESTRICTED_CONTINUUM_SIGNAL`.

There is no runtime fallback between regimes.

### 8.1 Ordinary Branch-T/ORCA equations and algorithm

Let

`S_i = sum_j D_ij`,

`C_i_step = C_i_rate dt > 0`,

and oriented capacities

`C_ij = beta_ij C_i_step`.

Full FIFO requires one accepted input mass `x_i`:

`f_ij = beta_ij x_i`, `0<=x_i<=S_i`.

The solver implements Tampère's oriented-capacity-proportional active-set construction:

1. Initialize unresolved inputs `U` and fixed flows to zero. Resolve every exact-zero-demand input at `x_i=0`.
2. For every output with unresolved oriented capacity, compute residual receiving mass
   `R_tilde_j = R_j - sum_{i notin U} beta_ij x_i`
   and restriction level
   `alpha_j = R_tilde_j / sum_{i in U} C_ij`.
3. Let `alpha_min` be the smallest level and retain every exactly tied output ID as binding diagnostic information.
4. For a selected most-restrictive output `j*`, find unresolved competitors whose demand is below their current capacity-proportional share:
   `L = {i in U: beta_ij*>0 and S_i <= alpha_min C_i_step}`.
5. If `L` is nonempty, fix each `i in L` at `x_i=S_i`, remove it from `U`, and recompute residual supplies/restriction levels. This redistributes only unused ordinary physical receiving supply.
6. Otherwise fix every unresolved competitor of `j*`,
   `B = {i in U: beta_ij*>0}`,
   at `x_i=alpha_min C_i_step`, remove `B`, and recompute.
7. If no output has unresolved oriented capacity, fix all remaining inputs at demand. Terminate when `U` is empty.

All output residuals must remain nonnegative within validation tolerance; no corrective clipping is allowed. Exact zero dynamic demand/supply is legal. Nonpositive structural `C_i_step` is invalid.

Tie semantics are hybrid:

- all exact tied constraints remain binding and visible;
- if one pivot selected earlier in the current solve remains among a newly tied eligible set, retain it as the internal pivot;
- with no unique incumbent, use simultaneous source-consistent processing where applicable, otherwise canonical output-local order only after the production and scalar references demonstrate mapped forward equivalence;
- pivot identity does not define a unique derivative at the tie.

### 8.2 Restricted continuum-signal branch

For each homogeneous input service group `i`, Component 0.5 supplies physical exposure `g_i in [0,1]` and structural saturation rate `s_i>0`. Define:

`x_i = min(S_i, g_i s_i dt, min_{j: beta_ij>0}(g_i R_j / beta_ij))`

and

`f_ij = beta_ij x_i`.

The construction is supported only under:

- H1: all positive-turn movements of input `i` share the same `g_i`;
- H2: `sum_{i: beta_ij>0} g_i <= 1` for every output `j`;
- H3: unused service/receiving share remains unused and is not reassigned across inputs/phases;
- H4: only cycle/interval-averaged blockage represented through `R_j` is claimed.

H2 proves output feasibility because `f_ij <= g_i R_j`; summing over `i` gives at most `R_j`. The formula selects a unique forward vector for fixed inputs. It is not post-ORCA clipping, demand scaling, or the original Tampère signalized internal-supply model.

### Reference basis and independent oracle

- Tampère et al. (2011), DOI `10.1016/j.trb.2010.06.004`, is authoritative for ordinary oriented-capacity-proportional Branch T and its exact iterative solution.
- Han et al. (2014), DOI `10.1016/j.trb.2014.01.001`, supports the continuum priority/service approximation and its spillback limitations.
- A separately organized scalar solver implements the frozen ORCA/Han equations without importing `nodes.py`.
- INM/UNsim/UXsim are contextual comparators only.

### Required literal 2x2 ORCA trace

Use input step capacities `C=(6,4)`, demands `S=(6,8/5)`, turning matrix

```text
beta = [[1/2, 1/2],
        [1/4, 3/4]]
```

and receiving `R=(2,4)`.

Initial restriction levels are

`alpha_0 = 2/(3+1)=1/2`,

`alpha_1 = 4/(3+3)=2/3`.

Input 1 is demand constrained because `8/5 < (1/2)*4=2`; fix its movements at `(2/5,6/5)`. Residual receiving is `(8/5,14/5)`. For input 0, new levels are `(8/5)/3=8/15` and `(14/5)/3=14/15`; output 0 binds, giving input-0 movements `(8/5,8/5)`. Final movement matrix is

```text
f = [[8/5, 8/5],
     [2/5, 6/5]]
```

with output totals `(2,14/5)`.

### Required checks and acceptance

- SISO demand/supply cases;
- at least two each of SIMO, MISO, and MIMO, including the trace above;
- equal/unequal capacities, unused ordinary supply redistribution, dynamic zeros, exact ties, near ties, and mapped permutations;
- demand, receiving, full-FIFO, nonnegativity, and conservation inequalities;
- production/scalar agreement at forward-reference tolerance;
- exact tie diagnostics contain all tied IDs and mapped permutation produces mapped-identical forward flows;
- stable-branch AD/FD evidence; ties are boundary diagnostics;
- at least two small-network coupling cases later in integration.

**Checkpoint 0.4:** report source transcription, algorithm trace, production/scalar tables, invariants, tie/permutation evidence, and gradients; stop before 0.5. Failure of incumbent-pivot forward equivalence reopens only C0.4-Q5.

## 9. Component 0.5 — continuum fixed-time signal service

### Responsibility and records

`signals.py` owns:

```text
FixedPhasePlan(
    phase_ids: tuple[int, ...],
    movement_phase_matrix: BoolTensor[M, P],
    input_saturation_rate: Tensor[I], # veh-eq/s, structural
)

ContinuumService(
    physical_green: Tensor[P],
    movement_exposure: Tensor[M],
    input_exposure: Tensor[I],
    input_service_mass: Tensor[I],
)
```

### Equations and validation

Physical green satisfies

`g_p>=0`, `sum_p g_p=1`

within simplex validation tolerance; exact zero is allowed. Invalid green is rejected, never clipped or normalized.

With fixed Boolean permission matrix `A`:

`e_m = sum_p A_mp g_p`.

Require `0<=e_m<=1`. Multiple phases may serve a movement only as additive nonoverlapping opportunities. For every input, all positive-turn movements must have equal exposure within `1e-9`; define that value as `g_i`. Heterogeneous rows are unsupported and rejected. Validate H2 for every output.

Nominal input service mass is

`mu_i = g_i s_i dt`.

The signal component returns service parameters; it does not clip node flow or allocate receiving supply. The restricted node formula in Component 0.4 combines demand, `mu_i`, and green-weighted receiving bounds.

For a two-phase diagnostic control:

`g(theta)=(theta,1-theta)`, `0<=theta<=1`.

The local Jacobian is

`partial e_m / partial g_p = A_mp`,

`partial mu_i / partial g_p = s_i dt * partial g_i/partial g_p`.

### Reference basis and limitations

Han et al.'s continuum model supplies the average-service interpretation and diminishing-cycle motivation. Tests establish arithmetic and AD correctness, not equivalence to finite-cycle on/off control, especially with spillback.

### Required checks and acceptance

- zero, partial, and full green literal values;
- exact zero exposure for unserved movements;
- one movement served in multiple compatible phases;
- rejection of invalid simplex, overlap, H1, H2, saturation, shape, and dtype cases;
- analytical Jacobian and fixed-total direction `(1,-1)/sqrt(2)`;
- rational Han coupling: two one-movement inputs share one output with `g=(2/5,3/5)`, `R=10`, high saturation, and `S=(1,10)`, yielding `x=(1,6)` and unused receiving mass `3` that is not redistributed;
- forward/reference and stable derivative tolerances.

**Checkpoint 0.5:** report local arithmetic/Jacobian, rejection table, Han coupling trace, and continuum limitation; stop before 0.6. Failure to preserve H1–H4 reopens the smallest 0.4/0.5 contract.

## 10. Component 0.6 — deterministic rollout and scheduler

### Responsibility and records

`simulation.py` owns the smallest typed orchestration records:

```text
NetworkDefinition(
    link_parameters,
    node_movement_maps,
    node_parameters,
    phase_plans,
    source_link_index: LongTensor[B_source],
    sink_link_index: LongTensor[B_sink],
)

Scenario(
    dt_s: float,
    horizon_steps: int,
    arrivals: Tensor[T, B_source],
    source_entry_capacity: Tensor[T, B_source] | None,
    sink_receiving: Tensor[T, B_sink] | None,
)

SimulationState(
    cumulative_links: CumulativeLinkState,
    source_queue: Tensor[B_source],
    cumulative_sink_exit: Tensor[B_sink],
)

StepResult(
    source_available,
    source_admitted,
    source_queue_end,
    sending,
    receiving,
    movement_demand,
    movement_flow,
    link_inflow,
    link_outflow,
    sink_outflow,
    active_constraint_records,
)

RolloutResult(
    initial_state,
    step_results,
    cumulative_link_history,
    source_queue_history,
    cumulative_sink_history,
    terminal_state,
)
```

Component-owned parameter/map records are referenced, not duplicated. Structural metadata and continuous tensors remain separate. Core rollout performs no file I/O.

### Source, sink, and update equations

At `t_k`:

`Q_available[k] = Q[k] + arrivals[k]`.

For a source with exclusive entry to link `a`:

`source_admitted = min(Q_available, R_a, source_entry_capacity)`

where an absent entry cap is `+infinity`. A network definition that gives the same receiving resource to multiple unallocated source/internal claims is invalid.

`Q[k+1] = Q_available[k] - source_admitted[k]`.

For a sink attached to terminal link `a`:

`sink_outflow = min(S_a, sink_receiving)`

with absent sink receiving treated as `+infinity`.

Internal node movement flows aggregate to link outflow/inflow exactly once. Then Component 0.2 advances all cumulative counts out of place. Source arrivals at `t_k` may enter the first link in interval `k`; they cannot be used in that link's downstream sending computed earlier in the step.

### Required checks and acceptance

- one-link delayed pulse with boundary-time table;
- two-link chain proving no same-step complete-link cascade;
- blocked first-link receiving accumulates source queue without loss;
- ordinary and restricted-signal node rollout cases;
- for every boundary time, empty-start accounting:
  `cumulative_arrivals = source_queue + sum(link_occupancy) + cumulative_sink_exit`;
- immutable inputs, out-of-place state, complete float64 histories, deterministic CPU repeat, and intact gradient paths;
- no UNsim execution; hand/scalar traces are numerical authority.

**Checkpoint 0.6:** report full hand rollout, interval/horizon conservation, causal timing, immutability, reproducibility, and graph-path checks; stop before 0.7.

## 11. Component 0.7 — objective, metrics, and gradient harness

### Responsibility and objective

`objectives.py` computes differentiable metrics from `RolloutResult`. The temporary diagnostic total system time uses a left-interval mass convention after start-boundary arrivals join the source queue:

`TST = dt * sum_{k=0}^{T-1} [sum_b Q_available_b[k] + sum_a (N_in_a[k]-N_out_a[k])]`.

Unit: veh-eq*s. Exited mass is not included. This is the only differentiable Milestone 0 objective. Report throughput, terminal mass, queues, occupancies, and conservation residual separately without detaching before the scalar objective is formed.

`gradients.py` owns central directional differences, step scans, reverse-direction contraction, JVP, error calculation, and active-regime classification. It does not implement a custom AD engine or backward rule.

### Physical directions and predeclared numerical rule

Directions satisfy

`sum_p d_p=0`, `||d||_2=1`.

For two phases use `d=(1,-1)/sqrt(2)`. Let

`h_max = min_{p:d_p!=0}(g_p/abs(d_p))`.

Set `h_scale=min(1,h_max)` and scan

`h = h_scale * rho`,

`rho in (1e-1,1e-2,1e-3,1e-4,1e-5,1e-6)`.

Rows for which either perturbed green is infeasible are recorded and classified, not silently omitted.

For scalar objective `L(g)` and direction `d`, record:

- reverse mode: `AD_rev = grad(L,g) dot d`;
- forward mode: `AD_jvp = J(g)d` via `torch.func.jvp`;
- `FD(h) = [L(g+h d)-L(g-h d)]/(2h)`.

No separate explicit VJP is called. Reverse and JVP agree when

`abs(AD_rev-AD_jvp) <= 1e-10 + 1e-10*max(abs(AD_rev),abs(AD_jvp))`.

A stable FD row passes when baseline, `+h`, and `-h` have the same relevant active IDs/masks and

`abs(AD_rev-FD) <= 1e-8 + 1e-6*max(abs(AD_rev),abs(FD))`.

The stable scenario passes if at least three adjacent scan rows pass. All rows, objective values, derivatives, errors, feasibility, and regimes are retained. The intentional event-boundary case passes by exposing the active/tied regime and finite forward values; it is not required to meet smooth FD convergence.

### Artifact contract

Pytest is authoritative for assertions. `scripts/run_milestone_0_diagnostics.py` is a thin wrapper around tested functions and accepts:

```text
--config configs/milestone_0/minimal_signalized.json
--output-dir reports/milestone_0_ltm_baseline/<unique_run_id>
```

It refuses to overwrite an existing nonempty output directory and writes:

- `diagnostics.json`: schema version, timestamp, command, environment/package record, config content/hash, dtype/device, objective/metrics, direction, full step scan, active regimes, and acceptance items;
- `summary.md`: concise human-readable rendering of the same data.

No plotting or CSV is required in Milestone 0.

### Required checks and acceptance

- literal TST from a tiny artificial history;
- linear and polynomial toy functions validate reverse/JVP/FD harness behavior independently of traffic;
- one signalized stable-active-set scenario with finite nonzero sensitivity and enough horizon for service effects;
- one exact/near event-boundary scenario;
- no `.item()`/detach before objective differentiation;
- deterministic schema and values on the verified CPU environment;
- the numerical rules above pass without changing grid/tolerances after results.

**Checkpoint 0.7:** report objective hand value, toy harness tests, stable derivative table, boundary classification, artifact example, and reproducibility; stop before integration.

## 12. Integration D/E gate

Only after explicit acceptance of Checkpoints 0.1–0.7 may integration begin.

### Integration implementation

- add only the coupling/configuration/test/runner code already listed;
- compose accepted records and functions without semantic adapters;
- use at least two small networks: one ordinary multi-node propagation case and one restricted signalized case;
- the signalized case must contain at least two competing physical green allocations, stable nonzero TST sensitivity, and remain inside H1–H4.

### Integration acceptance

1. every component checkpoint has explicit user acceptance;
2. all component and integration pytest cases pass without weakened criteria;
3. every interval satisfies demand, receiving, service, nonnegativity, storage, FIFO, and movement accounting;
4. horizon conservation meets the frozen residual criterion;
5. fractional mass remains continuous with no integerization or silent loss;
6. ordinary-node and restricted-signal reference traces match frozen values;
7. mapped node/link ordering produces mapped-identical outputs;
8. repeated CPU/float64 runs are deterministic;
9. TST retains a graph path to physical green;
10. reverse/JVP agreement and at least three adjacent stable FD rows pass;
11. the event case is diagnosed rather than smoothed or misreported as smooth;
12. the runner writes valid immutable JSON/Markdown evidence under `reports/milestone_0_ltm_baseline/`;
13. no claim extends beyond macro forward/derivative correctness.

Approved future commands, after implementation and environment authorization:

```bash
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_milestone_0_integration.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python scripts/run_milestone_0_diagnostics.py --config configs/milestone_0/minimal_signalized.json --output-dir reports/milestone_0_ltm_baseline/<unique_run_id>
```

Phase E_integration must write durable evidence and then stop for Phase F review. Milestone 1 must not begin automatically.

## 13. Stop conditions and change control

Stop the active D/E pass and report evidence if:

- a source equation or reference case contradicts this plan or a fixed invariant;
- ORCA incumbent-pivot processing is not mapped-forward-equivalent;
- restricted Han coupling violates H1–H4, conservation, or no-redistribution;
- an intended scenario needs heterogeneous phase exposure within one physical input;
- causal interpolation would query unavailable current/future history;
- a component cannot meet conservation/reference/gradient acceptance without clipping, smoothing, normalization, or a custom gradient;
- PyTorch lacks required ordinary-operation JVP coverage after installation;
- environment or dependency evidence cannot be reproduced;
- an interface defect requires changing traffic meaning.

Reopen the smallest affected component or global decision and repeat its D/E evidence after approval. Do not hide the issue in integration glue or weaken acceptance.

## 14. Deferred and unresolved items

No design item is unresolved for the approved Milestone 0 scope.

Deferred:

- nonempty rollout initialization;
- heterogeneous per-movement phase service within one physical input;
- original Tampère signalized internal supplies and COS/S-ORCA;
- partial FIFO and lane/local-order behavior;
- softmax/logit optimizer coordinates and actual signal optimization;
- continuum versus finite-cycle on/off fidelity;
- gradient usefulness and high-fidelity transfer;
- external simulator execution and SUMO comparison;
- performance/memory modes and non-CPU precision.

## 15. Plans superseded

This file supersedes, for Milestone 0 implementation only:

- `temp_content/milestone_0_ltm_baseline/phase_A_proposal.md`;
- `temp_content/milestone_0_ltm_baseline/phase_B_revision.md`;
- `temp_content/milestone_0_ltm_baseline/phase_B_closure.md`;
- the seven studies under `temp_content/milestone_0_ltm_baseline/subcomponent_plans/`.

Those files remain the discussion/audit archive. `AGENTS.md`, `PROJECT_CONTEXT.md`, and `PLANS.md` retain their higher-level authority.

## 16. Closure decision and next allowed action

The user explicitly closed Milestone 0 on 2026-08-31 after accepting all seven component checkpoints and the complete D_integration/E_integration evidence. The durable integration report is under `reports/milestone_0_ltm_baseline/integration_e_20260831/`; the Phase F decision record is `temp_content/milestone_0_ltm_baseline/phase_F_closure.md`.

Milestone 0_a1 is a proposed headless-visualizer add-on and is not part of this scientific closure. It has not entered Phase A. Do not implement it, install Matplotlib, create its development branch, or begin Milestone 1 until explicitly requested under the applicable workflow.
