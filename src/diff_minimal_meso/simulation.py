"""Deterministic fixed-step orchestration for the Milestone 0 macro stack."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import Tensor

from .fd import LinkFDParameters
from .ltm import (
    CumulativeLinkState,
    LinkStepResult,
    advance_link_state,
    compute_link_step,
)
from .movements import NodeMovementMap, project_oriented_demand
from .nodes import ConstraintID, NodeKind, NodeParameters, solve_node
from .signals import FixedPhasePlan, continuum_service, validate_green_split


@dataclass(frozen=True, slots=True)
class NetworkDefinition:
    """Immutable structural assembly of accepted Components 0.1--0.5."""

    link_parameters: tuple[LinkFDParameters, ...]
    node_movement_maps: tuple[NodeMovementMap, ...]
    node_parameters: tuple[NodeParameters, ...]
    phase_plans: tuple[FixedPhasePlan | None, ...]
    source_link_index: Tensor
    sink_link_index: Tensor
    _source_link_ids: tuple[int, ...] = field(init=False, repr=False)
    _sink_link_ids: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.link_parameters:
            raise ValueError("network must contain at least one link")
        node_count = len(self.node_movement_maps)
        if len(self.node_parameters) != node_count or len(self.phase_plans) != node_count:
            raise ValueError("node maps, parameters, and phase plans must have equal length")
        for node, (movement_map, parameters, phase_plan) in enumerate(
            zip(
                self.node_movement_maps,
                self.node_parameters,
                self.phase_plans,
                strict=True,
            )
        ):
            if parameters.movement_map is not movement_map:
                raise ValueError(f"node {node} parameters must reference its movement map")
            if parameters.kind is NodeKind.ORDINARY_ORCA and phase_plan is not None:
                raise ValueError("ordinary nodes must not have a phase plan")
            if parameters.kind is NodeKind.RESTRICTED_CONTINUUM_SIGNAL and phase_plan is None:
                raise ValueError("restricted signal nodes require a phase plan")
            if phase_plan is not None:
                if phase_plan.movement_phase_matrix.shape[0] != movement_map.movement_count:
                    raise ValueError("phase plan movement dimension does not match node map")
                if phase_plan.input_saturation_rate.shape != (movement_map.input_count,):
                    raise ValueError("phase plan saturation dimension does not match node map")
            for local_input, link in enumerate(movement_map.input_link_ids):
                self._validate_link_index(link)
                expected_capacity = self.link_parameters[link].capacity_veh_per_s
                actual_capacity = float(parameters.input_capacity_rate[local_input])
                if actual_capacity != expected_capacity:
                    raise ValueError("node input capacity must match its link FD capacity")
            for link in movement_map.output_link_ids:
                self._validate_link_index(link)
        self._validate_boundary_indices("source_link_index", self.source_link_index)
        self._validate_boundary_indices("sink_link_index", self.sink_link_index)
        object.__setattr__(self, "_source_link_ids", tuple(self.source_link_index.tolist()))
        object.__setattr__(self, "_sink_link_ids", tuple(self.sink_link_index.tolist()))
        self._validate_boundary_ownership()

    @property
    def link_count(self) -> int:
        return len(self.link_parameters)

    @property
    def node_count(self) -> int:
        return len(self.node_movement_maps)

    def _validate_link_index(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("movement-map link IDs must be global integer link indices")
        if value < 0 or value >= self.link_count:
            raise ValueError("movement-map link index is outside the network")

    def _validate_boundary_indices(self, name: str, value: Tensor) -> None:
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise ValueError(f"{name} must have shape [B]")
        if value.dtype != torch.long or value.device.type != "cpu":
            raise TypeError(f"{name} must be a CPU torch.long tensor")
        indices = value.tolist()
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must not contain duplicates")
        for link in indices:
            self._validate_link_index(link)

    def _validate_boundary_ownership(self) -> None:
        incoming_claims = list(self.source_link_index.tolist())
        outgoing_claims = list(self.sink_link_index.tolist())
        for movement_map in self.node_movement_maps:
            incoming_claims.extend(movement_map.output_link_ids)
            outgoing_claims.extend(movement_map.input_link_ids)
        if len(incoming_claims) != len(set(incoming_claims)):
            raise ValueError("a link receiving boundary has multiple source/node claims")
        if len(outgoing_claims) != len(set(outgoing_claims)):
            raise ValueError("a link sending boundary has multiple node/sink claims")


@dataclass(frozen=True, slots=True)
class Scenario:
    dt_s: float
    horizon_steps: int
    arrivals: Tensor
    source_entry_capacity: Tensor | None = None
    sink_receiving: Tensor | None = None

    def __post_init__(self) -> None:
        dt = _positive_finite("dt_s", self.dt_s)
        object.__setattr__(self, "dt_s", dt)
        if isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, Integral):
            raise TypeError("horizon_steps must be an integer")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        _validate_time_table("arrivals", self.arrivals, self.horizon_steps)
        for name, value in (
            ("source_entry_capacity", self.source_entry_capacity),
            ("sink_receiving", self.sink_receiving),
        ):
            if value is None:
                continue
            _validate_time_table(name, value, self.horizon_steps)
            if value.device != self.arrivals.device:
                raise ValueError(f"{name} must be on the arrivals device")


@dataclass(frozen=True, slots=True)
class SignalControl:
    """Fixed physical green tensors aligned with network node order."""

    physical_green: tuple[Tensor | None, ...]


@dataclass(frozen=True, slots=True)
class SimulationState:
    cumulative_links: CumulativeLinkState
    source_queue: Tensor
    cumulative_sink_exit: Tensor


@dataclass(frozen=True, slots=True)
class NodeActiveRecord:
    binding_mask: Tensor
    tied_constraint_ids: tuple[ConstraintID, ...]
    selected_pivot_ids: tuple[ConstraintID, ...]


@dataclass(frozen=True, slots=True)
class StepResult:
    source_available: Tensor
    source_admitted: Tensor
    source_queue_end: Tensor
    sending: Tensor
    receiving: Tensor
    movement_demand: tuple[Tensor, ...]
    movement_flow: tuple[Tensor, ...]
    link_inflow: Tensor
    link_outflow: Tensor
    sink_outflow: Tensor
    active_constraint_records: tuple[NodeActiveRecord, ...]


@dataclass(frozen=True, slots=True)
class RolloutResult:
    initial_state: SimulationState
    step_results: tuple[StepResult, ...]
    cumulative_link_history: CumulativeLinkState
    source_queue_history: Tensor
    cumulative_sink_history: Tensor
    terminal_state: SimulationState


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validate_time_table(name: str, value: Tensor, horizon_steps: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != horizon_steps:
        raise ValueError(f"{name} must have shape [T, B]")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_network_scenario(network: NetworkDefinition, scenario: Scenario) -> None:
    if scenario.arrivals.shape[1] != network.source_link_index.numel():
        raise ValueError("arrivals source dimension does not match the network")
    if scenario.source_entry_capacity is not None and (
        scenario.source_entry_capacity.shape != scenario.arrivals.shape
    ):
        raise ValueError("source_entry_capacity must match arrivals shape")
    expected_sink_shape = (scenario.horizon_steps, network.sink_link_index.numel())
    if scenario.sink_receiving is not None and scenario.sink_receiving.shape != expected_sink_shape:
        raise ValueError("sink_receiving must have shape [T, B_sink]")
    for parameters in network.link_parameters:
        if scenario.dt_s > min(parameters.free_flow_time_s, parameters.backward_wave_time_s):
            raise ValueError("scenario dt exceeds a link characteristic travel time")
        if parameters.capacity_veh_per_s * scenario.dt_s > parameters.jam_storage_veh:
            raise ValueError("scenario step capacity exceeds link storage")


def _validate_control(network: NetworkDefinition, control: SignalControl | None) -> None:
    if control is None:
        if any(parameter.kind is NodeKind.RESTRICTED_CONTINUUM_SIGNAL for parameter in network.node_parameters):
            raise ValueError("signal control is required for restricted signal nodes")
        return
    if len(control.physical_green) != network.node_count:
        raise ValueError("signal control must align with network node order")
    for node, (parameters, phase_plan, green) in enumerate(
        zip(network.node_parameters, network.phase_plans, control.physical_green, strict=True)
    ):
        if parameters.kind is NodeKind.ORDINARY_ORCA:
            if green is not None:
                raise ValueError(f"ordinary node {node} must not have physical green")
        else:
            if green is None or phase_plan is None:
                raise ValueError(f"restricted node {node} requires physical green")
            validate_green_split(green, len(phase_plan.phase_ids))


def initialize_state(network: NetworkDefinition, scenario: Scenario) -> SimulationState:
    """Return the approved empty-start state on the scenario device."""

    _validate_network_scenario(network, scenario)
    zeros = scenario.arrivals.new_zeros
    cumulative = CumulativeLinkState(
        n_in=zeros((1, network.link_count)),
        n_out=zeros((1, network.link_count)),
    )
    return SimulationState(
        cumulative_links=cumulative,
        source_queue=zeros((network.source_link_index.numel(),)),
        cumulative_sink_exit=zeros((network.sink_link_index.numel(),)),
    )


def simulation_step(
    state: SimulationState,
    network: NetworkDefinition,
    scenario: Scenario,
    step_index: int,
    control: SignalControl | None = None,
) -> tuple[SimulationState, StepResult]:
    """Advance one interval using only pre-update sending/receiving values."""

    if step_index < 0 or step_index >= scenario.horizon_steps:
        raise IndexError("step_index is outside the scenario horizon")
    if state.cumulative_links.current_index != step_index:
        raise ValueError("state history boundary does not match step_index")
    _validate_control(network, control)
    device = scenario.arrivals.device
    if state.cumulative_links.n_in.device != device:
        raise ValueError("state and scenario must use the same device")
    if state.source_queue.shape != (network.source_link_index.numel(),):
        raise ValueError("source queue shape does not match network sources")
    if state.cumulative_sink_exit.shape != (network.sink_link_index.numel(),):
        raise ValueError("sink exit shape does not match network sinks")

    limits = compute_link_step(state.cumulative_links, network.link_parameters, scenario.dt_s)
    source_available = state.source_queue + scenario.arrivals[step_index]
    link_inflow_values = [scenario.arrivals.new_zeros(()) for _ in range(network.link_count)]
    link_outflow_values = [scenario.arrivals.new_zeros(()) for _ in range(network.link_count)]
    movement_demands: list[Tensor] = []
    movement_flows: list[Tensor] = []
    active_records: list[NodeActiveRecord] = []

    for node, (movement_map, parameters, phase_plan) in enumerate(
        zip(
            network.node_movement_maps,
            network.node_parameters,
            network.phase_plans,
            strict=True,
        )
    ):
        input_index = torch.tensor(
            movement_map.input_link_ids, dtype=torch.long, device=device
        )
        output_index = torch.tensor(
            movement_map.output_link_ids, dtype=torch.long, device=device
        )
        oriented_demand = project_oriented_demand(limits.sending[input_index], movement_map)
        node_receiving = limits.receiving[output_index]
        if parameters.kind is NodeKind.ORDINARY_ORCA:
            node_flow = solve_node(
                oriented_demand, node_receiving, parameters, scenario.dt_s
            )
        else:
            assert phase_plan is not None and control is not None
            green = control.physical_green[node]
            assert green is not None
            service = continuum_service(green, phase_plan, movement_map, scenario.dt_s)
            node_flow = solve_node(
                oriented_demand,
                node_receiving,
                parameters,
                scenario.dt_s,
                input_exposure=service.input_exposure,
                input_saturation_rate=phase_plan.input_saturation_rate.to(device),
            )
        for local, link in enumerate(movement_map.input_link_ids):
            link_outflow_values[link] = node_flow.input_outflow[local]
        for local, link in enumerate(movement_map.output_link_ids):
            link_inflow_values[link] = node_flow.output_inflow[local]
        movement_demands.append(oriented_demand)
        movement_flows.append(node_flow.movement_flow)
        active_records.append(
            NodeActiveRecord(
                node_flow.binding_mask,
                node_flow.tied_constraint_ids,
                node_flow.selected_pivot_ids,
            )
        )

    source_admitted_values: list[Tensor] = []
    for source, link in enumerate(network._source_link_ids):
        candidates = [source_available[source], limits.receiving[link]]
        if scenario.source_entry_capacity is not None:
            candidates.append(scenario.source_entry_capacity[step_index, source])
        admitted = torch.min(torch.stack(tuple(candidates)))
        source_admitted_values.append(admitted)
        link_inflow_values[link] = admitted
    source_admitted = (
        torch.stack(tuple(source_admitted_values))
        if source_admitted_values
        else scenario.arrivals.new_zeros((0,))
    )
    source_queue_end = source_available - source_admitted

    sink_outflow_values: list[Tensor] = []
    for sink, link in enumerate(network._sink_link_ids):
        candidates = [limits.sending[link]]
        if scenario.sink_receiving is not None:
            candidates.append(scenario.sink_receiving[step_index, sink])
        outflow = torch.min(torch.stack(tuple(candidates)))
        sink_outflow_values.append(outflow)
        link_outflow_values[link] = outflow
    sink_outflow = (
        torch.stack(tuple(sink_outflow_values))
        if sink_outflow_values
        else scenario.arrivals.new_zeros((0,))
    )

    link_inflow = torch.stack(tuple(link_inflow_values))
    link_outflow = torch.stack(tuple(link_outflow_values))
    next_cumulative = advance_link_state(
        state.cumulative_links,
        link_inflow,
        link_outflow,
        limits,
        network.link_parameters,
    )
    next_state = SimulationState(
        cumulative_links=next_cumulative,
        source_queue=source_queue_end,
        cumulative_sink_exit=state.cumulative_sink_exit + sink_outflow,
    )
    return next_state, StepResult(
        source_available=source_available,
        source_admitted=source_admitted,
        source_queue_end=source_queue_end,
        sending=limits.sending,
        receiving=limits.receiving,
        movement_demand=tuple(movement_demands),
        movement_flow=tuple(movement_flows),
        link_inflow=link_inflow,
        link_outflow=link_outflow,
        sink_outflow=sink_outflow,
        active_constraint_records=tuple(active_records),
    )


def rollout(
    network: NetworkDefinition,
    scenario: Scenario,
    control: SignalControl | None = None,
) -> RolloutResult:
    """Run a transparent deterministic Python-loop rollout with full histories."""

    _validate_network_scenario(network, scenario)
    _validate_control(network, control)
    initial = initialize_state(network, scenario)
    state = initial
    steps: list[StepResult] = []
    source_history = [state.source_queue]
    sink_history = [state.cumulative_sink_exit]
    for step_index in range(scenario.horizon_steps):
        state, result = simulation_step(
            state, network, scenario, step_index, control
        )
        steps.append(result)
        source_history.append(state.source_queue)
        sink_history.append(state.cumulative_sink_exit)
    return RolloutResult(
        initial_state=initial,
        step_results=tuple(steps),
        cumulative_link_history=state.cumulative_links,
        source_queue_history=torch.stack(tuple(source_history)),
        cumulative_sink_history=torch.stack(tuple(sink_history)),
        terminal_state=state,
    )


def mass_balance_residual(
    result: RolloutResult, scenario: Scenario
) -> Tensor:
    """Return boundary-time arrived minus queued/on-network/exited mass."""

    cumulative_arrivals = torch.cat(
        (
            scenario.arrivals.new_zeros((1,)),
            torch.cumsum(scenario.arrivals.sum(dim=1), dim=0),
        )
    )
    queued = result.source_queue_history.sum(dim=1)
    occupancy = (
        result.cumulative_link_history.n_in
        - result.cumulative_link_history.n_out
    ).sum(dim=1)
    exited = result.cumulative_sink_history.sum(dim=1)
    return cumulative_arrivals - queued - occupancy - exited
