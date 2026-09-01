# Milestone 0_a1 — headless macro-baseline visualizer add-on

Status: **Closed**

Approval date: 2026-08-31

Phase: Phase C complete; implementation not started

Branch: `m0_a1`, based on closed Milestone 0 commit `729d788`

## 1. Scientific and engineering purpose

Add a deterministic, headless visualization and evidence layer to the closed Milestone 0 macroscopic simulator. The tool makes boundary states, interval flows, spillback, accounting, and the accepted total-system-time (TST) directional sensitivity inspectable without implementing a second simulator or changing traffic behavior.

The add-on is intentionally isolated on `m0_a1`. Merge into `main` is optional and requires a later user decision. Its evidence supports human inspection and later matched macro–meso comparisons; it does not establish new forward- or gradient-fidelity claims.

## 2. Scope, non-goals, and inherited contracts

### 2.1 In scope

- versioned JSON definitions for arbitrary handcrafted networks, empty-start scenarios, fixed controls, and drawing layouts within the accepted Milestone 0 contract;
- construction of the existing `NetworkDefinition`, `Scenario`, and `SignalControl` records through a visualizer-owned adapter;
- accepted rollout and optional accepted TST directional-gradient evaluation;
- immutable detached CPU visualization data, saved as the primary rendering truth;
- re-rendering from saved renderer-facing data without rerunning simulation;
- Matplotlib Agg overview and selected-frame PNG output (Candidate A);
- after Candidate A acceptance, a complete ordered `T+1` PNG sequence (Candidate B);
- ordinary-chain, signalized-merge, and exact dynamic-spillback fixtures;
- fixed scales, units, provenance, checksums, non-overwrite behavior, and deterministic artifact naming.

### 2.2 Explicit non-goals

- nonempty LTM initialization or semantic relaxation of Milestone 0 scenarios;
- interactive windows, widgets, playback, network editing, or live control;
- GIF, MP4, ffmpeg, or an animation writer;
- YAML, arbitrary Python/symbolic objectives, multiple simultaneous sensitivities, or a full flow-by-green Jacobian;
- automatic layout, map tiles, geographic projection, or large-network rendering optimization;
- spatial queue-tail reconstruction or labeling aggregate occupancy as physical queue length;
- route, lane, ordered-ledger, or other Milestone 1+ functionality;
- modification of accepted FD, LTM, movement, node, signal, scheduler, objective, or gradient semantics;
- mandatory merge into `main`.

### 2.3 Inherited numerical and environment contracts

Milestone 0 units, indexing, float64 CPU reference tolerances, continuous-mass semantics, update order, active-set interpretation, and conservation contracts remain authoritative. State exists at `T+1` boundaries; interval flow/service exists for `T` intervals. All gradient evaluation finishes before reporting tensors are detached.

Environment: `/home/zpz/miniconda3/envs/diff_minimal_meso`, managed only with Conda. Verified planning state is Python 3.12.13, PyTorch 2.5.1 CUDA 12.4, pytest 9.0.3, with Matplotlib and NumPy absent. No environment mutation occurred in Phases A–C.

## 3. Frozen system architecture

```text
versioned scenario/layout JSON
        |
        v
strict parser + stable external-ID mapping
        |
        +--> accepted Milestone 0 records and validators
        |
        v
accepted rollout + optional TST directional check
        |
        v
validated extraction, then detach().cpu()
        |
        v
versioned renderer-facing bundle/manifest
        |
        +--> re-render without rollout
        +--> Candidate A overview/selected frames
        `--> after checkpoint, Candidate B full sequence
```

The parser and renderer are adapters. They must not duplicate simulation/update equations. Drawing metadata cannot affect rollout.

### 3.1 Declarative input contract

Schema identifier: `m0-a1-scenario-v1`. JSON is the sole declarative format in V1.

The simulation portion contains:

- `dt_s` and `horizon_steps`;
- links with stable string ID/label and all five accepted `LinkFDParameters` fields;
- nodes with stable string ID, kind, ordered input/output link IDs, canonical `(input, output, beta)` movement rows, capacity inputs, and optional phase plan;
- sources with link ID, length-`T` arrival series, and optional entry-capacity series;
- sinks with link ID and optional length-`T` receiving series;
- physical-green vectors for restricted signal nodes;
- an optional sensitivity request naming one restricted node and one feasible direction.

The drawing portion contains stable vertex IDs and 2-D coordinates, ordered tail/head vertices for every link, labels, simulation-node/vertex associations, selected frame indices, render constants, and explicit comparison-group bounds where required.

Topology and drawing layout are separate namespaces with explicit references. External IDs map deterministically to the simulator's global integer ordering and are retained for reporting. The parser rejects rather than repairs duplicate or unresolved IDs, invalid topology, invalid shapes/lengths, nonfinite inputs, invalid turning sums, invalid green vectors, missing layout coverage, or mismatched control dimensions. Existing constructors remain the final scientific validators.

### 3.2 Renderer-facing data contract

Implement immutable records equivalent to:

```text
LayoutDefinition:
    vertex_ids: tuple[str, ...]
    vertex_xy: Float64Tensor[V, 2]
    link_tail_index, link_head_index: LongTensor[A]
    link_labels: tuple[str, ...]
    node_vertex_index: LongTensor[N]
    source_vertex_index: LongTensor[B_source]
    sink_vertex_index: LongTensor[B_sink]

SensitivitySummary:
    objective_name: "TST"
    objective_unit: "veh-eq*s"
    control_node_id: str
    physical_green, direction: Float64Tensor[P]
    reverse_directional, jvp_directional: float
    reverse_jvp_agree, stable_scenario_passes, event_detected: bool

VisualizationBundle:
    boundary_time_s: Float64Tensor[T+1]
    occupancy_veh, occupancy_ratio: Float64Tensor[T+1, A]
    source_queue_veh: Float64Tensor[T+1, B_source]
    cumulative_sink_veh: Float64Tensor[T+1, B_sink]
    conservation_residual_veh: Float64Tensor[T+1]
    sending_veh, receiving_veh: Float64Tensor[T, A]
    link_inflow_veh, link_outflow_veh: Float64Tensor[T, A]
    node_total_flow_veh: Float64Tensor[T, N]
    movement_flow_veh: tuple[Float64Tensor[T, M_n], ...]
    active_regime_labels: tuple[tuple[str, ...], ...]
    layout: LayoutDefinition
    sensitivity: SensitivitySummary | None
```

All continuous bundle tensors are finite CPU float64 with `requires_grad=False`; indices are CPU `torch.long`. Extraction validates shapes, stable ordering, layout coverage, occupancy ratios within `[0,1]`, time dimensions, and conservation using the closed Milestone 0 tolerance. Renderer-facing arrays and metadata are stored in a versioned manifest and are sufficient to re-render without a rollout or serialized PyTorch graph.

### 3.3 Differentiability and sensitivity contract

Configuration selects zero or one restricted signal node and one feasible direction. No selection means sensitivity is not requested and must be displayed as N/A. Arbitrary-network runs never silently choose a target.

The bundled signalized fixture deterministically selects the first restricted node in stable external-ID order and the accepted unit-L2 direction `[1/sqrt(2),-1/sqrt(2)]` in its two-phase order. A proportional sign pattern `[+1,-1]` may be shown only as a human-readable explanation and must not be confused with the evaluated vector. Arbitrary JSON must supply an already unit-L2, zero-sum feasible direction and is rejected rather than normalized. The runner varies only that node's physical-green tensor, holds all other controls fixed, and calls the accepted `directional_check` on TST. The display includes the symbolic definition `d(TST)/dg · d`, green, evaluated direction, reverse and JVP values, units, agreement, and stable/event classification. Event-boundary values receive an explicit regime-dependence warning.

### 3.4 Frame and visual-semantics contract

A horizon of `T` intervals yields `T+1` boundary frames. Frame `k<T` combines state at `t_k` with separately labeled interval-`k` quantities for `[t_k,t_{k+1}]`. Frame `T` is labeled terminal and contains no interval overlay.

The compact figure has four areas:

1. topology/state: link occupancy/storage color, realized outflow/capacity width, source queue, node flow, and restricted-node green;
2. link-by-boundary occupancy-ratio heatmap;
3. source queue, cumulative sink exit, on-network mass, and conservation trace;
4. frame status, movement/service/active-set summary, TST, and optional sensitivity.

Occupancy color is fixed to `[0,1]`. Flow width uses link `C*dt`. Unbounded quantities use explicit, validated comparison-group limits; independent auto-scaling is forbidden for scientific comparisons. State and interval flow retain distinct legends. The style, font, size, DPI, subplot geometry, color maps, and PNG metadata are fixed. No unsimulated state is interpolated.

### 3.5 Artifact and runner contract

Primary command shape:

```bash
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python \
  scripts/run_milestone_0_a1_visualizer.py \
  --config configs/milestone_0_a1/<case>.json \
  --output-dir reports/milestone_0_a1_headless_visualizer/<unique_run_id>
```

The same runner exposes an explicit re-render mode using saved manifest data and no simulator invocation. Exact CLI flags may be named mechanically during D1, but tests must demonstrate that the re-render path cannot call rollout.

Candidate A artifacts:

```text
manifest.json
summary.md
overview.png
selected_frames/frame_0000.png
selected_frames/frame_<selected>.png
selected_frames/frame_<T>_terminal.png
```

Candidate B later adds `frames/frame_0000.png` through `frames/frame_<T>_terminal.png`. The command refuses a nonempty output directory and preserves its contents. The manifest records config and schema, config hash, Git branch/commit/dirty state, command, package/backend/dtype/device details, units, render constants, layout, all plot-data arrays, sensitivity definition/result, acceptance flags, ordered artifact names, and checksums. Timestamps may appear in JSON but not in PNG pixel metadata used for determinism checks.

## 4. Component A1.1 — declarative data, bundle, and semantic fixtures

### Responsibility and I/O

Own strict JSON parsing, stable ID resolution, construction of accepted simulator records, optional TST sensitivity evaluation, detached bundle extraction, manifest-data serialization/deserialization, and the numerical fixture authorities. It must not import Matplotlib.

Inputs are a validated JSON object/path or a saved renderer-facing manifest. Outputs are accepted simulation inputs, `LayoutDefinition`, and a validated `VisualizationBundle` suitable for serialization and rendering.

### Reference basis and differentiability

Simulation values come only from closed Milestone 0 APIs. Serialization follows the accepted diagnostics runner's provenance/hash pattern but generalizes it to arbitrary accepted records. Differentiation is performed by the existing gradient harness before a single explicit reporting detach boundary; structural ID/layout operations are nondifferentiated.

### Implementation files and functions

- create `src/diff_minimal_meso/visualization_data.py` for schema constants, immutable records, parse/resolve/build functions, extraction, validation, JSON conversion, and sensitivity-summary construction;
- create `configs/milestone_0_a1/ordinary_chain.json`, `signalized_merge.json`, and `spillback_chain.json`;
- create `tests/test_visualization_data.py` and small literal fixture helpers under `tests/references/` only if useful.

Functions remain small and typed. Parsing, ID resolution, simulator-record construction, rollout invocation, sensitivity evaluation, bundle extraction, and serialization are separate pieces.

### Local checks and acceptance

- valid SISO, SIMO, MISO, and MIMO definitions reach existing record validators with deterministic ordering;
- invalid schema/IDs/shapes/units/topology/greens fail without normalization or repair;
- serialized then loaded bundles preserve values, dtypes represented in schema, IDs, shapes, units, and ordering;
- a re-render input can be loaded without rollout or autograd;
- extracted occupancy, flow, queue, sink, conservation, active regimes, and sensitivity equal literal accepted rollout values;
- no bundle tensor carries gradients or can mutate rollout histories;
- the exact spillback authority in Section 7.1 passes.

**A1.1 checkpoint:** complete D1, write `phase_D/phase_D1_visualization_data_worklog.md`; complete E1, write `phase_E/phase_E1_visualization_data_validation.md`; report evidence and stop for user acceptance.

## 5. Component A1.2 — Candidate A static Agg renderer and runner

### Responsibility and I/O

Install the approved dependency transaction, render the frozen four-area overview and selected boundary frames from a `VisualizationBundle`, support original-run and re-render modes, and atomically write the Candidate A artifact contract without overwrite.

The renderer consumes reporting data only and returns/writes figures and artifact metadata; it never calls traffic update functions. Orchestration may call A1.1 before rendering.

### Dependency and API basis

Pin the conventional Defaults-channel Conda metapackage `matplotlib=3.11.0`, accepting its PyQt/Tornado closure. Rendering nevertheless uses only Matplotlib's documented noninteractive Agg backend and figure/axes/patch/collection/color-normalization/colorbar/`savefig` APIs. Agg must be selected before importing `matplotlib.pyplot`; no display or Qt event loop may be required.

Before installation:

1. record `conda list`;
2. inspect the exact dry-run/solver transaction for `conda install -n diff_minimal_meso -c defaults matplotlib=3.11.0`;
3. stop if it unexpectedly replaces/downgrades Python 3.12.13, the PyTorch CUDA build/runtime, pytest, MKL 2023.1.0, or another critical numerical package;
4. install only after the transaction passes, retrying the same operation with minimum escalation if sandbox/network/cache access blocks it;
5. record realized versions/builds and verify imports/Agg;
6. run the full preexisting suite before accepting renderer evidence.

### Implementation files and functions

- create `src/diff_minimal_meso/visualization.py` for fixed render specification, overview/frame artist construction, scale validation, and PNG saving;
- create `scripts/run_milestone_0_a1_visualizer.py` for CLI parsing, run versus re-render orchestration, provenance, staging/output refusal, manifest/summary writing, and checksums;
- create `tests/test_visualization.py` and `tests/test_visualization_runner.py`.

Figure construction, artist data preparation, saving, provenance, and filesystem policy remain separate functions. The script contains no traffic equations.

### Local checks and acceptance

- backend is Agg and rendering succeeds without a display;
- PNGs are nonempty, readable by Pillow, and have frozen dimensions/DPI/mode;
- artist inputs equal bundle values; semantic assertions take priority over pixel snapshots;
- original-run and saved-bundle re-render produce identical plot data and artifact structure and, if stable in the realized environment, identical PNG checksums;
- selected-state/interval labels and terminal behavior are exact;
- paired cases retain identical coordinates, units, and scales;
- missing sensitivity is N/A, stable sensitivity matches A1.1, and events are warned;
- output refusal leaves a preexisting nonempty directory unchanged;
- ordinary, paired-signal, and spillback artifacts are suitable for manual inspection.

**A1.2 checkpoint:** complete D2/E2 and their separate worklog/validation files, present Candidate A artifacts, and stop. Candidate B is not authorized until the user explicitly accepts this checkpoint.

## 6. Component A1.3 — Candidate B complete frame sequence

### Responsibility and I/O

Reuse the accepted A1.2 single-frame renderer to emit one ordered PNG for every boundary `0..T`. It adds no alternate visual interpretation, video encoder, or simulator call.

### Implementation and differentiability

Extend the renderer/runner and their tests only as required for sequence orchestration and manifest bookkeeping. All inputs are already detached; this component has no gradient path.

### Local checks and acceptance

- exactly `T+1` files with deterministic zero-padded ordering and terminal naming;
- every file uses the accepted frame renderer and matches its bundle index;
- terminal frame has no interval quantities;
- ordered filenames, checksums, and total storage are recorded;
- Candidate A outputs and semantics remain unchanged;
- non-overwrite behavior remains effective.

**A1.3 checkpoint:** complete D3/E3 with separate worklog/validation records, report evidence, and stop for user acceptance before integration.

## 7. Frozen fixtures and validation hierarchy

### 7.1 Exact dynamic-spillback authority

Use two serial links and one ordinary SISO node with empty start, `dt=tau_f=tau_b=1 s`, capacity `1 veh-eq/s`, storage `2 veh-eq` per link, arrivals of `1 veh-eq` in each of six intervals, and terminal sink receiving zero.

Required boundary occupancies `[upstream, downstream]` are:

```text
[[0,0], [1,0], [1,1], [1,2], [2,2], [2,2], [2,2]]
```

Required source queue is `[0,0,0,0,0,1,2]`; downstream receiving is `[1,1,1,0,0,0]`; node flow is `[0,1,1,0,0,0]`; source admission/upstream receiving is `[1,1,1,1,0,0]`; cumulative sink exit is zero; every conservation residual is exactly zero in this fixture.

The rendered causal sequence must visibly agree: downstream storage fills, its receiving collapses, node transfer stops, upstream storage fills, and source queue then grows.

### 7.2 Required test commands

After the applicable component is authorized and available:

```bash
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_visualization_data.py
PYTHONPATH=src /home/zpz/miniconda3/envs/diff_minimal_meso/bin/python -m pytest -q tests/test_visualization.py tests/test_visualization_runner.py
```

Each E_i runs its focused tests plus relevant accepted predecessor and Milestone 0 regression tests. Use inherited float64 tolerances. Rendering never substitutes for mathematical/reference, conservation, behavior, or gradient checks.

## 8. D/E integration and milestone acceptance

After A1.1–A1.3 checkpoints are accepted, D_integration may add only coupling/orchestration required by this plan. E_integration must:

- pass the full preexisting and M0_a1 suites without altered criteria;
- generate durable ordinary, paired-signal, and spillback artifact sets;
- prove schema-to-record and rollout-to-bundle value equivalence;
- prove saved-data re-rendering without rollout;
- verify comparison coordinates, units, scales, horizon semantics, and gradient definition are identical where required;
- verify the spillback numerical and visible causal sequence;
- verify deterministic artifact names/data/checksums to the extent supported by the realized fixed environment;
- verify closed Milestone 0 histories, TST, gradients, and event classifications are unchanged;
- record packages, config, commands, Git state, quantitative results, visual inspection targets, deviations, and risks under `reports/milestone_0_a1_headless_visualizer/` and the phase archives.

Integration acceptance establishes visualization correctness, reproducibility, and diagnostic usefulness only. It does not close the milestone; Phase F user review remains mandatory.

## 9. Stop conditions and change control

Stop and reopen the smallest affected plan/component contract if:

- parsing arbitrary valid input requires scientific repair or modification of accepted simulator records/equations;
- extraction loses/reorders data, changes event labels, or detaches before gradient evaluation;
- the dependency solve changes a protected environment package unexpectedly;
- Agg requires a display/interactive backend in the realized environment;
- comparison scales cannot truthfully represent the declared group;
- occupancy is represented as physical queue length or unsimulated states are interpolated;
- Candidate B diverges from the accepted A1.2 renderer;
- output handling overwrites user data;
- the exact spillback fixture conflicts with independently rerun closed equations.

Implementation-conformance errors may be fixed within the active D_i/E_i pass and documented. Any change to traffic semantics, gradient meaning, schema scientific scope, comparison meaning, or acceptance criteria requires an explicit plan amendment and user approval.

## 10. Deferred items and open questions

Deferred: YAML, nonempty initialization, arbitrary objectives, multiple sensitivities, GUI interaction, animation/video, automatic layout, spatial queue inference, large-network optimization, and branch merge.

**Open questions blocking implementation: none.**

Amendment on 2026-08-31: Candidate A from the D1/E1 sensitivity-direction diagnosis was approved. The evaluated and serialized direction uses the accepted unit-L2 convention; `[+1,-1]` is explanatory notation only. This supersedes the earlier raw-vector wording without changing closed Milestone 0.

## 11. Implementation order and next gate

Order is strictly A1.1 D1/E1/checkpoint, A1.2 D2/E2/Candidate-A checkpoint, A1.3 D3/E3/checkpoint, D_integration/E_integration, then Phase F. A later component may depend only on accepted predecessors.

This implementation order was completed through Phase F on 2026-09-01. No further milestone action is allowed under this plan except an explicitly authorized corrective reopening or separate branch-merge decision.

## 12. Phase F closure

Closed by explicit user decision on 2026-09-01.

Components A1.1–A1.3 and D/E integration passed their approved acceptance criteria and user checkpoints. Final validation reported 164 passed tests and one preexisting optional-GPU skip. Candidate A selected-frame and Candidate B complete-sequence artifacts were generated for the ordinary, paired-signal, and spillback authorities; conservation, shared-frame renderer identity, saved-data re-rendering, deterministic bookkeeping, gradient reporting, and non-overwrite checks passed.

No scientific or engineering issue remains open for this add-on. Closure establishes visualization correctness, reproducibility, and diagnostic usefulness only; it does not add a traffic-fidelity or gradient-usefulness claim. The work remains isolated on branch `m0_a1`. Merge into `main` remains deferred and requires a separate explicit user decision.
