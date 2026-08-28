# AGENTS.md

## Purpose

This repository develops and evaluates a differentiable mesoscopic traffic simulator for fixed-time signal-control optimization.

The main scientific question is deliberately narrower than "build a differentiable traffic simulator":

> How much traffic discreteness must be retained to preserve useful mesoscopic gradients for downstream traffic-control decisions?

The working hypothesis is that signal-relevant mesoscopic effects can be represented with **continuous traffic mass plus selectively retained discrete semantic order/route metadata**, without integerizing vehicle mass or implementing full microscopic agent dynamics.

Before planning or editing code, read:

1. `PROJECT_CONTEXT.md`
2. `PLANS.md`
3. the applicable approved milestone plan under `docs/milestones/`, if one exists
4. the existing source, tests, configs, and environment files

Authority hierarchy:

- `AGENTS.md` defines repository-wide agent behavior and approval workflow.
- `PROJECT_CONTEXT.md` defines current scientific intent, stable model semantics, methodological decisions, and invariants.
- `PLANS.md` defines the project roadmap, milestone status, and high-level gates.
- an approved `docs/milestones/milestone_<N>_<short_name>.md` is authoritative for that implementation pass.
- code and tests define current implemented behavior, but they do not override an explicit scientific invariant.

If implementation evidence conflicts with `PROJECT_CONTEXT.md`, do not silently change the methodology. Record the conflict in the current phase archive, propose the smallest amendment, and wait for explicit user approval before changing scientific behavior.

## Decision-status vocabulary

Interpret project statements using these labels:

- **Fixed**: must be implemented as written unless the user explicitly approves a methodological change.
- **Default**: use unless evidence, numerical constraints, or an API limitation justify reconsideration.
- **Hypothesis**: an empirical claim to test, not an assumed truth.
- **Open**: requires a documented decision before implementation if it affects scientific meaning.
- **Deferred**: outside the current mainline.

Do not silently convert a default or hypothesis into a fixed requirement.

## Scientific invariants

The following are fixed for the initial mainline unless explicitly amended by the user:

1. Long-link traffic propagation uses an aggregate first-order LWR/LTM/Newell-style backbone rather than microscopic car-following.
2. Traffic quantity is represented as continuous vehicle-equivalent mass. Do not round or floor traffic flow to integer vehicles in the mainline simulator.
3. Routes, current movement labels, lane permissions, network topology, and fixed phase order are structural metadata in V1; they are not variables to differentiate through.
4. Persistent route identity is stored by route-table reference plus route position and continuous mass; do not duplicate a full future movement sequence in every local state.
5. Current node movement is projected from route metadata by a fixed route-to-movement mapping.
6. The local mesoscopic refinement is per lane near the downstream node. Do not silently replace it with a per-link mixed representation.
7. The local ordered coordinate is cumulative traffic mass/FIFO rank, not physical distance. Physical queue length and spillback remain governed by traffic density/storage and LTM/Newell propagation.
8. Exact local FIFO semantics are represented by ordered metadata. Do not infer exact head-of-line order only from a mixed movement-composition bin.
9. Lane-choice intent and realized lane occupancy/accessibility are distinct. Finite lane or turn-pocket storage may prevent desired allocations from being realized.
10. No traffic mass may be created, destroyed, or silently dropped by packet/grid/ledger operations.
11. Exact lossless adjacent merging of structurally identical ledger entries is allowed. Lossy behavioral compression, route mixing, or order relaxation requires an approved milestone decision.
12. Initial signal control uses fixed phase topology/order and continuous fixed-time service parameters. Green split is the first optimization variable. Cycle length is not a meaningful first target unless startup/clearance lost-time or another within-cycle effect is explicitly modeled.
13. Forward fidelity, gradient fidelity, and downstream gradient usefulness are distinct claims and require distinct validation evidence.
14. Do not claim a gradient is useful solely because the simulator is differentiable, finite gradients exist, or an internal optimization objective decreases.
15. Do not silently broaden the initial project into dynamic route choice, stochastic route choice, explicit microscopic lane changing, actuated signal sequence control, reinforcement learning, or a full network-wide lane-expansion model.

## Mandatory milestone workflow

Every scientifically distinct milestone follows a gated human-in-the-loop workflow. Planning and implementation are separate Codex tasks unless the user explicitly waives the gate for a narrowly mechanical change.

### Temporary planning and feedback files

Use `temp_content/` for editable planning/discussion material.

The global user feedback file is:

`temp_content/feedback/feedback.md`

Treat that file as a user-facing feedback inbox. Feedback is organized by Markdown milestone and phase tabs/headings. For the active task, read the corresponding milestone/phase tab and use its latest entry as the default feedback unless the user's current interactive instruction overrides it. Read any supplementary files linked from that entry. Do not overwrite or delete user feedback merely to mark it handled. Archive the interpreted feedback and response in the active milestone's phase dialogue file.

When presenting open questions for user decision, the default format for each question is:

1. **Problem statement**: define the unresolved decision as a concrete core question.
2. **Assumed model-chain function**: explain what the affected functionality is expected to do and where it sits in the approved/proposed data flow.
3. **Core gap**: identify exactly what is not yet specified or evidenced and why implementation cannot safely infer it.
4. **Candidates and recommendation**: list the smallest viable choices, briefly state each choice's advantages and disadvantages, and give a clearly labeled recommendation with its rationale.

Use this expanded format whenever the user asks to list or review open questions, unless the user explicitly requests a shorter inventory. Keep recommendations provisional until the user approves them, and preserve any decision dependencies between questions.

For each active milestone, create:

`temp_content/milestone_<N>_<short_name>/`

Archive the current milestone discussion in phase-specific files. For an implementation-dominant milestone with component checkpoints, a useful pattern is:

- `phase_A_proposal.md`
- `phase_A_dialogue.md`
- `phase_B_revision.md`
- `phase_B_dialogue.md`
- `phase_C_candidate_plan.md`
- `phase_D/phase_D1_<component>_worklog.md`
- `phase_E/phase_E1_<component>_validation.md`
- `phase_D/phase_D2_<component>_worklog.md`
- `phase_E/phase_E2_<component>_validation.md`
- `phase_D/phase_D_integration_worklog.md`
- `phase_E/phase_E_integration_validation.md`
- `phase_F_closure.md`

The exact number of component files follows the approved milestone plan. The phase archive should capture useful methodological/implementation reasoning, user decisions, rejected alternatives, component-checkpoint decisions, integration anomalies, and unresolved items. Do not dump a raw chat transcript when a concise decision record preserves the needed information.

### Phase A — inspect and propose

For the current milestone:

1. Read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, applicable approved milestone plans, and relevant code/tests/configs.
2. Inspect actual dependency APIs and current repository behavior before proposing interfaces.
3. Do not edit production project files, install packages, or run environment-modifying commands unless the user explicitly authorizes those actions for the planning task.
4. Write the proposal to the milestone's `temp_content/` folder rather than relying on console-only planning.
5. The proposal must include, where relevant:
   - verified current state and dependency behavior;
   - scope and explicit non-goals;
   - equations and traffic semantics being implemented;
   - state definitions, units, shapes, dtypes, and update order;
   - files/modules to create or modify and their responsibilities;
   - differentiability path and structural/non-differentiated metadata;
   - tests, numerical tolerances, commands, and acceptance criteria;
   - unresolved scientific and engineering choices;
   - risks and expected consequences of each unresolved option.
6. Stop after the proposal. Do not implement.

### Phase B — discuss and revise

1. Treat the proposal as unapproved until the user explicitly accepts it.
2. Read the latest applicable entry from the active milestone/phase tab in `temp_content/feedback/feedback.md`, inspect its linked supplementary files, and archive the relevant discussion in the current milestone's phase dialogue file.
3. Answer questions and revise the proposal without starting implementation.
4. Do not silently resolve choices that can change model fidelity, gradient semantics, experimental meaning, or comparison fairness.
5. When alternatives exist, present the smallest viable options and their scientific/engineering consequences.
6. Continue until scope, semantics, validation, acceptance criteria, and stop conditions are explicit.

### Phase C — freeze the approved milestone plan

After explicit approval:

1. Save the accepted plan under:

   `docs/milestones/milestone_<N>_<short_name>.md`

2. Include at least:
   - status: `Approved`;
   - approval date if known;
   - scientific purpose;
   - scope and non-goals;
   - model semantics and equations affected;
   - implementation and integration order;
   - validation and acceptance criteria;
   - stop conditions;
   - unresolved or deferred items;
   - plans superseded, if any.
3. **Plan the milestone as a system before implementing components separately.** Freeze component responsibilities, interfaces, data flow, equations, units, differentiability boundaries, dependency order, component checks, and final integration checks together in shared Phases A–C.
4. For implementation-dominant milestones, especially Milestones 0 and 1, organize the authoritative plan **component-first** rather than section-first. After a short milestone-wide scope/interface preamble, each component section should keep together its:
   - responsibility and I/O;
   - literature/reference basis and any approved adaptation;
   - differentiability semantics;
   - implementation work;
   - local reference/unit/behavior checks;
   - optional low-cost visual diagnostic where useful;
   - component acceptance criteria and user checkpoint.
   Do not scatter all component implementations into one section and all component tests into a later unrelated section.
5. The approved milestone file is authoritative for that implementation pass.
6. Do not overwrite high-level methodology in `PROJECT_CONTEXT.md` or the milestone sequence in `PLANS.md` with low-level implementation details.
7. If the approved plan conflicts with `AGENTS.md` or a fixed scientific invariant, stop and report the conflict before implementation.

### Component implementation/validation cycle — D_i -> E_i -> user checkpoint

After shared Phases A–C approve the complete subsystem architecture, implementation-dominant milestones proceed one approved component at a time in dependency order.

For each component `i`:

1. **Phase D_i — implement only that component.** Re-read the approved system plan and implement the component against the already planned interface. Do not begin the next component merely because coding succeeds.
2. **Phase E_i — validate immediately.** Run the component-specific hand/reference, unit, conservation, behavior, and gradient checks required by the approved plan. Produce any approved low-complexity diagnostic figure/trace intended for manual inspection.
3. **User checkpoint.** Report the component evidence and stop. The user decides whether the component is accepted as the trusted dependency for the next component, needs correction, or requires a plan/interface amendment.
4. Component acceptance means **correct under the currently approved interface and semantics**, not permanently immutable. If later integration exposes a genuine coupling defect, reopen the smallest affected component/interface explicitly and repeat its D_i/E_i evidence rather than hiding the fix in integration glue.
5. A scientifically meaningful ambiguity discovered during D_i or E_i still triggers the ordinary change-control rule: document the evidence, propose the smallest amendment, and wait for approval before changing behavior.

Default component-pass reporting and error handling:

- After every authorized D_i implementation, proceed directly to its E_i checks unless the user limits the task further, and always write the component's validation report under the active milestone's `phase_E/` folder.
- Codex may diagnose and fix syntactic, procedural, test-harness, and implementation-conformance errors within the approved component plan, rerunning the affected E_i checks and recording the corrections.
- If a failure shows that the approved method, scientific semantics, or acceptance contract is invalid or cannot be implemented as specified, stop the component pass. Write a diagnosis report with the evidence, affected contract, and smallest viable amendment candidates for reopening D_i/E_i; do not silently change methodology or weaken the failed criterion.

This cycle gives early component closure while preserving a system-level design. It does not create separate scientific milestones for each component.

### Phase D_integration — compose accepted components

After all required component checkpoints are accepted:

1. Integrate the accepted components according to the Phase-C architecture.
2. Add only the coupling/orchestration code required by the approved plan.
3. Do not compensate for a component/interface defect with an undocumented adapter that changes traffic semantics. Reopen the smallest affected component if necessary.
4. Do not begin a later milestone.

### Phase E_integration — validate the complete milestone stack

1. Run the approved pairwise/coupling and end-to-end integration checks.
2. Re-run critical component invariants if integration could have affected them.
3. Run the milestone-level conservation, reference/equivalence, behavior, and gradient checks applicable to the complete stack.
4. Do not weaken, remove, or reinterpret a failed acceptance criterion merely to complete the milestone.
5. Write durable validation evidence under `reports/milestone_<N>_<short_name>/` and a concise integration summary under the milestone's `temp_content/` folder.
6. Report files changed, commands run, quantitative/visual results, acceptance status item by item, deviations/amendments, assumptions, and unresolved risks.
7. Stop for Phase F. Do not start the next milestone.

For milestones that are primarily a scientific experiment rather than multi-component construction, the approved Phase-C plan may use a single Phase D followed by a single Phase E instead of artificial component cycles.

### Phase F — user review and closure

The milestone is not complete merely because code was produced and tests passed.

1. Wait for user review of the diff, evidence, and acceptance results.
2. The user decides whether the milestone:
   - passes and closes;
   - needs a corrective implementation pass;
   - requires a plan amendment;
   - should be abandoned or deferred.
3. Record the closure decision in the milestone plan or a closure section/report.
4. Plan the next milestone only after explicit closure of the current one.

### When the full gate may be skipped

The user may explicitly waive the planning gate for a narrowly mechanical task, such as:

- correcting a typo;
- applying an already approved rename;
- adding a precisely specified assertion;
- rerunning an already approved experiment configuration;
- making a local refactor with no behavioral or scientific effect.

Changes to traffic equations, node semantics, queue/storage logic, route/order handling, differentiability construction, signal parameterization, objectives, scenario splits, experiment comparisons, dependencies, or acceptance tests require the full workflow unless the user explicitly states otherwise.

## Operating rules

1. Inspect before editing.
2. Verify third-party APIs from installed packages/source or official documentation. Do not invent interfaces from memory.
3. Before a choice that can change scientific meaning, report the missing decision, the smallest viable options, and the expected consequences.
4. Prefer the smallest implementation that satisfies the approved milestone.
5. Never begin the next milestone automatically.
6. Do not add dependencies unless the current milestone requires them.
7. Do not add clipping, smoothing, randomization, fallback branches, or regularization that changes traffic semantics without approval.
8. Keep experiment settings centralized and configurable rather than duplicated across scripts.
9. Set and record random seeds for any stochastic scenario generation or sampling.
10. Use double precision for numerical gradient checks unless there is a documented reason not to.
11. Run relevant tests after each material change.
12. Never place secrets, credentials, or private data in the repository.
13. Do not modify or delete user data or existing results unless explicitly asked.
14. If a diagnostic may be blocked by sandbox/host permissions rather than true environment state, request the minimum required escalation before drawing conclusions.
15. Preserve reproducibility: record environment, package versions/commits, config, seeds, commands, and output locations for scientific runs.

## Validation discipline

Validation is hierarchical. A later layer does not substitute for an earlier one.

At minimum, applicable milestones should distinguish:

1. **Mathematical/reference correctness**
   - local LTM/node/grid updates agree with hand-computed or independently coded reference cases;
   - units, time indexing, and conservation identities are explicit.
2. **Forward invariants**
   - traffic mass and route-mass conservation;
   - sending/receiving and storage constraints respected;
   - lane permissions and FIFO/order semantics respected;
   - no silent mass loss under split/merge/defragment operations.
3. **Macro-limit/equivalence checks**
   - when local order/storage effects are disabled or homogenized by design, the mesoscopic implementation reproduces the matched aggregate baseline within declared tolerance.
4. **Meso-sensitive forward checks**
   - cases with the same aggregate state but different movement order can produce different realized service when they should;
   - finite turn-pocket/shared-lane blockage and spillback behave causally.
5. **Gradient correctness**
   - autograd/differentiated directional derivatives agree with central finite differences on small fixed scenarios away from known event boundaries;
   - gradients are finite and sensitivity paths are inspected, not assumed.
6. **Gradient usefulness**
   - compare matched macro and mesoscopic gradient directions under the same objective/scenarios;
   - evaluate downstream improvement, not only gradient agreement internal to the simulator.
7. **Cross-fidelity validation**
   - when opened by an approved milestone, evaluate candidate signal perturbations/plans in a richer reference simulator such as SUMO using held-out scenarios.

If a validation fails, report it rather than weakening the test.

## Overnight and detached execution

Long approved experiments may run detached only after the applicable milestone plan, implementation, tests, exact command, and output location are approved.

Repository-wide protocol:

1. Prepare and validate a repository-owned runner before detaching it.
2. Prefer:
   - immutable recorded configuration;
   - unique output and log paths;
   - independent atomic shards where appropriate;
   - refusal to overwrite completed shards by default;
   - explicit progress, timestamps, failure records, and exit-status files;
   - a separate aggregation-only command that cannot start training/optimization.
3. If `tmux` or another host-level persistence mechanism is used, verify it in the actual environment. Do not assume a sandbox-owned detached process will survive its owner.
4. Use a stable milestone-qualified session name and durable repository logs. Record the exact command and session name in the work report.
5. Verify the session and first expected output/progress artifact before telling the user the job is safely detached.
6. A detached process may continue computing, but Codex does not continue reasoning, interpreting results, editing reports, or making scientific decisions while the conversation is suspended.
7. On resumption, inspect exit status, logs, failures, shard completeness, and checksums before aggregation or interpretation. Do not silently retry failed or missing scientific runs.
8. Destructive restarts, overwrites, forced termination, or deletion of results require explicit approval when they may discard work.
9. Benchmark concurrency before a large CPU/GPU run. Resource scheduling may change for efficiency, but scenarios, seeds, precision, optimization budgets, and acceptance rules must not be reduced merely to shorten runtime.

## Command escalation 

By default, codex sessions are run inside the sandbox. This may render some resources by default unavailable, such as GPU. If you want to request resources that are 
unavailable inside the sandbox, or find that a command cannot access required resources unless escalated, you should raise an escalation request upon manual approval.

Default user preference: when an in-scope required command fails in a way that may be caused by the sandbox, retry the same required operation with the minimum justified escalation before proposing or attempting a behavior-changing alternative. In particular, do not replace required GPU execution with CPU execution merely because an un-escalated probe reports CUDA unavailable.

## Implementation style

Defaults unless an approved milestone says otherwise:

- Language: Python.
- Primary differentiation framework: PyTorch unless repository inspection establishes another existing choice.
- Target platform: Linux/Ubuntu server; verify exact environment before relying on version-specific behavior.
- Prefer typed, small, testable functions.
- Keep network data, LTM physics, node logic, route/ledger state, local-zone state, signal control, objectives, gradients, experiments, and evaluation in separate modules.
- Use explicit tensor shapes, units, dtypes, and device handling.
- Avoid hidden global mutable state.
- Keep structural metadata and differentiable continuous tensors clearly separated in APIs.
- Save machine-readable results in CSV/JSONL/Parquet as appropriate and human-readable summaries in Markdown or plain text.
- Do not use notebooks as the only implementation of core simulator or scientific experiments.

## Work report

At the end of each Codex task, report:

- files changed;
- assumptions made;
- commands run;
- tests and results;
- unresolved scientific or engineering risks;
- current milestone/phase status;
- the next allowed action, without starting it unless requested.
