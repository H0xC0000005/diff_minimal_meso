"""Passive macro/ordered-ledger composition for Milestone 1 Component 1.5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ledger import (
    LedgerEntry,
    OrderedLedger,
    OutboundPackage,
    SelectionEvidence,
    append_ordered_packages,
    assign_discharge_times,
    execute_node_transfer,
    merge_adjacent_exact,
    progress_transferred_entries,
    take_ledger_prefix,
)
from .routes import RouteTable, incoming_link_priority
from .simulation import (
    NetworkDefinition,
    RolloutResult,
    Scenario,
    SignalControl,
    SimulationState,
    StepResult,
    initialize_state,
    simulation_step,
)


@dataclass(frozen=True, slots=True)
class SourceRouteBlock:
    route_index: int
    mass: Tensor
    generation_front_s: Tensor
    generation_tail_s: Tensor

    def __post_init__(self) -> None:
        # Reuse the ledger's strict scalar/timing contract without storing this
        # temporary validation record as resident state.
        LedgerEntry(
            self.route_index,
            0,
            self.mass,
            self.generation_front_s,
            self.generation_tail_s,
        )


@dataclass(frozen=True, slots=True)
class MesoScenario:
    dt_s: float
    horizon_steps: int
    route_arrivals: tuple[tuple[tuple[SourceRouteBlock, ...], ...], ...]
    route_table: RouteTable
    source_entry_capacity: Tensor | None = None
    sink_receiving: Tensor | None = None

    def __post_init__(self) -> None:
        if len(self.route_arrivals) != self.horizon_steps:
            raise ValueError("route_arrivals must have one row per step")
        source_count = self.route_table.network.source_link_index.numel()
        for step, row in enumerate(self.route_arrivals):
            if not isinstance(row, tuple) or len(row) != source_count:
                raise ValueError("each route-arrival row must align with sources")
            boundary = step * self.dt_s
            lower = (step - 1) * self.dt_s
            for source, blocks in enumerate(row):
                if not isinstance(blocks, tuple):
                    raise TypeError("route-arrival cells must be tuples")
                source_link = int(self.route_table.network.source_link_index[source])
                for block in blocks:
                    if not isinstance(block, SourceRouteBlock):
                        raise TypeError("route arrivals must contain SourceRouteBlock")
                    if self.route_table.current_link(block.route_index, 0) != source_link:
                        raise ValueError("route arrival does not match source link")
                    tail = float(block.generation_tail_s)
                    front = float(block.generation_front_s)
                    if tail > boundary or (step > 0 and front < lower):
                        raise ValueError("generation timestamp is outside its source bucket")
        # The unchanged Scenario performs the remaining dt/horizon/table checks.
        self.to_macro_scenario()

    def _template(self) -> Tensor:
        for row in self.route_arrivals:
            for blocks in row:
                if blocks:
                    return blocks[0].mass
        for value in (self.source_entry_capacity, self.sink_receiving):
            if value is not None:
                return value.new_zeros(())
        raise ValueError("an all-empty meso scenario requires tensor-valued boundary context")

    def to_macro_scenario(self) -> Scenario:
        template = self._template()
        arrivals = torch.stack(
            tuple(
                torch.stack(
                    tuple(
                        torch.stack(tuple(block.mass for block in blocks)).sum()
                        if blocks
                        else template.new_zeros(())
                        for blocks in row
                    )
                )
                for row in self.route_arrivals
            )
        )
        return Scenario(
            self.dt_s,
            self.horizon_steps,
            arrivals,
            self.source_entry_capacity,
            self.sink_receiving,
        )


@dataclass(frozen=True, slots=True)
class MesoState:
    macro_state: SimulationState
    source_ledgers: tuple[OrderedLedger, ...]
    link_ledgers: tuple[OrderedLedger, ...]
    completed_route_mass: Tensor


@dataclass(frozen=True, slots=True)
class MesoBoundaryTransfer:
    kind: str
    boundary_index: int
    residual: OrderedLedger
    transferred: tuple[LedgerEntry, ...]
    evidence: SelectionEvidence


@dataclass(frozen=True, slots=True)
class MesoStepResult:
    macro_step_result: StepResult
    transfers: tuple[MesoBoundaryTransfer, ...]
    end_state: MesoState


@dataclass(frozen=True, slots=True)
class MesoRolloutResult:
    macro_rollout_result: RolloutResult
    ledger_history: tuple[MesoState, ...]
    step_results: tuple[MesoStepResult, ...]


def initialize_meso_state(scenario: MesoScenario) -> MesoState:
    macro_scenario = scenario.to_macro_scenario()
    network = scenario.route_table.network
    macro = initialize_state(network, macro_scenario)
    empty_links = tuple(
        OrderedLedger(link, (), scenario.route_table) for link in range(network.link_count)
    )
    source_ledgers = tuple(
        OrderedLedger(int(link), (), scenario.route_table)
        for link in network.source_link_index
    )
    return MesoState(
        macro,
        source_ledgers,
        empty_links,
        macro_scenario.arrivals.new_zeros((scenario.route_table.route_count,)),
    )


def _empty_evidence() -> SelectionEvidence:
    return SelectionEvidence((), (), ())


def _assert_state_correspondence(state: MesoState, scenario: MesoScenario) -> None:
    macro = state.macro_state
    occupancy = macro.cumulative_links.n_in[-1] - macro.cumulative_links.n_out[-1]
    for link, ledger in enumerate(state.link_ledgers):
        total = ledger.total_mass(like=occupancy[link])
        torch._assert(
            torch.isclose(total, occupancy[link], rtol=1.0e-10, atol=1.0e-12),
            "ledger/link occupancy mismatch",
        )
    for source, ledger in enumerate(state.source_ledgers):
        total = ledger.total_mass(like=macro.source_queue[source])
        torch._assert(
            torch.isclose(total, macro.source_queue[source], rtol=1.0e-10, atol=1.0e-12),
            "source ledger/macro queue mismatch",
        )


def meso_simulation_step(
    state: MesoState,
    scenario: MesoScenario,
    step_index: int,
    control: SignalControl | None = None,
) -> tuple[MesoState, MesoStepResult]:
    table = scenario.route_table
    network = table.network
    macro_scenario = scenario.to_macro_scenario()
    _assert_state_correspondence(state, scenario)
    next_macro, macro_step = simulation_step(
        state.macro_state, network, macro_scenario, step_index, control
    )
    interval_front = macro_scenario.arrivals.new_tensor(step_index * scenario.dt_s)
    interval_tail = macro_scenario.arrivals.new_tensor((step_index + 1) * scenario.dt_s)
    transfers: list[MesoBoundaryTransfer] = []

    source_queues = list(state.source_ledgers)
    link_residuals = list(state.link_ledgers)
    additions: list[list[LedgerEntry]] = [[] for _ in range(network.link_count)]

    for source, blocks in enumerate(scenario.route_arrivals[step_index]):
        queue = source_queues[source]
        generated = tuple(
            LedgerEntry(
                block.route_index,
                0,
                block.mass,
                block.generation_front_s,
                block.generation_tail_s,
            )
            for block in blocks
        )
        queue = OrderedLedger(queue.owner_link_index, queue.entries + generated, table)
        admitted = macro_step.source_admitted[source]
        if bool(admitted == 0.0):
            source_queues[source] = queue
            continue
        selected = take_ledger_prefix(queue, admitted)
        tau = admitted.new_tensor(
            network.link_parameters[queue.owner_link_index].free_flow_time_s
        )
        timing = assign_discharge_times(
            selected.transferred.entries,
            admitted,
            interval_front,
            interval_tail,
            tau,
        )
        additions[queue.owner_link_index].extend(timing.entries)
        source_queues[source] = selected.residual
        transfers.append(
            MesoBoundaryTransfer(
                "source", source, selected.residual, timing.entries, selected.evidence
            )
        )

    for node, movement_map in enumerate(network.node_movement_maps):
        packages_by_output: list[list[OutboundPackage]] = [
            [] for _ in range(movement_map.output_count)
        ]
        flow = macro_step.movement_flow[node]
        for local_input, input_link in enumerate(movement_map.input_link_ids):
            mask = movement_map.movement_input_index == local_input
            quota = flow[mask.to(flow.device)]
            if bool(quota.sum() == 0.0):
                continue
            device_mask = mask.to(flow.device)
            full_quota = torch.where(device_mask, flow, torch.zeros_like(flow))
            output_delays = flow.new_tensor(
                [
                    network.link_parameters[output_link].free_flow_time_s
                    for output_link in movement_map.output_link_ids
                ]
            )
            delays = output_delays[
                movement_map.movement_output_index.to(flow.device)
            ]
            result = execute_node_transfer(
                state.link_ledgers[input_link],
                node,
                macro_step.sending[input_link],
                full_quota,
                interval_front,
                interval_tail,
                delays,
                table,
            )
            link_residuals[input_link] = result.residual
            by_output: dict[int, list[LedgerEntry]] = {}
            for item in result.transferred:
                transition_position = item.route_position - 1
                transition = table.transition(item.route_index, transition_position)
                by_output.setdefault(transition.output_local_index, []).append(item)
            priority = incoming_link_priority(network, node, input_link)
            for output_local, entries in by_output.items():
                packages_by_output[output_local].append(
                    OutboundPackage(priority, tuple(entries))
                )
            transfers.append(
                MesoBoundaryTransfer(
                    "node", input_link, result.residual, result.transferred, result.evidence
                )
            )
        for output_local, packages in enumerate(packages_by_output):
            if packages:
                output_link = movement_map.output_link_ids[output_local]
                additions[output_link].extend(
                    append_ordered_packages(
                        OrderedLedger(output_link, (), table), tuple(packages)
                    ).ledger.entries
                )

    completed = state.completed_route_mass.clone()
    for sink, link in enumerate(network._sink_link_ids):
        outflow = macro_step.sink_outflow[sink]
        if bool(outflow == 0.0):
            continue
        selected = take_ledger_prefix(state.link_ledgers[link], outflow)
        progression = progress_transferred_entries(selected.transferred.entries, table)
        if progression.progressed:
            raise ValueError("sink transfer contains a nonterminal route")
        link_residuals[link] = selected.residual
        for item in progression.completed:
            completed[item.route_index] = completed[item.route_index] + item.mass
        transfers.append(
            MesoBoundaryTransfer(
                "sink", sink, selected.residual, progression.completed, selected.evidence
            )
        )

    next_ledgers: list[OrderedLedger] = []
    for link, residual in enumerate(link_residuals):
        combined = OrderedLedger(link, residual.entries + tuple(additions[link]), table)
        next_ledgers.append(merge_adjacent_exact(combined).ledger)
    next_state = MesoState(
        next_macro, tuple(source_queues), tuple(next_ledgers), completed
    )
    _assert_state_correspondence(next_state, scenario)
    result = MesoStepResult(macro_step, tuple(transfers), next_state)
    return next_state, result


def meso_rollout(
    scenario: MesoScenario, control: SignalControl | None = None
) -> MesoRolloutResult:
    macro_scenario = scenario.to_macro_scenario()
    macro_result = _macro_rollout(scenario.route_table.network, macro_scenario, control)
    initial = initialize_meso_state(scenario)
    history = [initial]
    steps = []
    state = initial
    for step in range(scenario.horizon_steps):
        state, result = meso_simulation_step(state, scenario, step, control)
        history.append(state)
        steps.append(result)
    # The separately executed macro result must be bit-identical to composition.
    if not torch.equal(
        state.macro_state.cumulative_links.n_in,
        macro_result.cumulative_link_history.n_in,
    ):
        raise AssertionError("meso composition changed macro inflow arithmetic")
    if not torch.equal(
        state.macro_state.cumulative_links.n_out,
        macro_result.cumulative_link_history.n_out,
    ):
        raise AssertionError("meso composition changed macro outflow arithmetic")
    if not torch.equal(
        state.macro_state.source_queue, macro_result.terminal_state.source_queue
    ):
        raise AssertionError("meso composition changed macro source queues")
    if not torch.equal(
        state.macro_state.cumulative_sink_exit,
        macro_result.terminal_state.cumulative_sink_exit,
    ):
        raise AssertionError("meso composition changed macro sink counts")
    for composed, independent in zip(
        steps, macro_result.step_results, strict=True
    ):
        left = composed.macro_step_result
        for name in (
            "source_available",
            "source_admitted",
            "source_queue_end",
            "sending",
            "receiving",
            "link_inflow",
            "link_outflow",
            "sink_outflow",
        ):
            if not torch.equal(getattr(left, name), getattr(independent, name)):
                raise AssertionError(f"meso composition changed macro {name}")
        for name in ("movement_demand", "movement_flow"):
            if any(
                not torch.equal(a, b)
                for a, b in zip(
                    getattr(left, name), getattr(independent, name), strict=True
                )
            ):
                raise AssertionError(f"meso composition changed macro {name}")
    return MesoRolloutResult(macro_result, tuple(history), tuple(steps))


def _macro_rollout(
    network: NetworkDefinition,
    scenario: Scenario,
    control: SignalControl | None,
) -> RolloutResult:
    from .simulation import rollout

    return rollout(network, scenario, control)
