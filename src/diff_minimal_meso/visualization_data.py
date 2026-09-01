"""Declarative inputs and detached reporting data for Milestone 0_a1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .fd import LinkFDParameters
from .gradients import directional_check, node_active_signature
from .movements import build_movement_map
from .nodes import NodeKind, NodeParameters
from .objectives import total_system_time
from .signals import FixedPhasePlan, validate_green_split
from .simulation import (
    NetworkDefinition,
    RolloutResult,
    Scenario,
    SignalControl,
    mass_balance_residual,
    rollout,
)


SCENARIO_SCHEMA_VERSION = "m0-a1-scenario-v1"
BUNDLE_SCHEMA_VERSION = "m0-a1-bundle-v1"
CONSERVATION_ATOL = 1.0e-10
DIRECTION_ATOL = 1.0e-12


@dataclass(frozen=True, slots=True)
class LayoutDefinition:
    vertex_ids: tuple[str, ...]
    vertex_xy: Tensor
    link_tail_index: Tensor
    link_head_index: Tensor
    link_labels: tuple[str, ...]
    node_vertex_index: Tensor
    source_vertex_index: Tensor
    sink_vertex_index: Tensor
    selected_frames: tuple[int, ...]
    comparison_limits: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True, slots=True)
class SensitivityRequest:
    node_index: int
    node_id: str
    direction: Tensor


@dataclass(frozen=True, slots=True)
class VisualizationCase:
    network: NetworkDefinition
    scenario: Scenario
    control: SignalControl | None
    layout: LayoutDefinition
    link_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    sink_ids: tuple[str, ...]
    sensitivity_request: SensitivityRequest | None
    source_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SensitivitySummary:
    objective_name: str
    objective_unit: str
    control_node_id: str
    physical_green: Tensor
    direction: Tensor
    reverse_directional: float
    jvp_directional: float
    reverse_jvp_agree: bool
    stable_scenario_passes: bool
    event_detected: bool


@dataclass(frozen=True, slots=True)
class VisualizationBundle:
    boundary_time_s: Tensor
    occupancy_veh: Tensor
    occupancy_ratio: Tensor
    source_queue_veh: Tensor
    cumulative_sink_veh: Tensor
    conservation_residual_veh: Tensor
    sending_veh: Tensor
    receiving_veh: Tensor
    link_inflow_veh: Tensor
    link_outflow_veh: Tensor
    node_total_flow_veh: Tensor
    movement_flow_veh: tuple[Tensor, ...]
    active_regime_labels: tuple[tuple[str, ...], ...]
    total_system_time_veh_s: float
    layout: LayoutDefinition
    link_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    sink_ids: tuple[str, ...]
    link_capacity_step_veh: Tensor
    link_storage_veh: Tensor
    physical_green_by_node: tuple[Tensor | None, ...]
    sensitivity: SensitivitySummary | None


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _keys(value: Mapping[str, Any], required: set[str], optional: set[str], name: str) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        condition = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {condition}")
    return result


def _unique_records(records: Sequence[dict[str, Any]], name: str) -> None:
    ids = [_identifier(record.get("id"), f"{name}.id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} IDs must be unique")


def _float_tensor(value: Any, name: str, *, ndim: int | None = None) -> Tensor:
    try:
        tensor = torch.tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric values") from error
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _parse_links(records: list[Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[LinkFDParameters, ...]]:
    parsed = [_object(item, "link") for item in records]
    _unique_records(parsed, "links")
    ordered = sorted(parsed, key=lambda item: item["id"])
    ids: list[str] = []
    labels: list[str] = []
    parameters: list[LinkFDParameters] = []
    required = {
        "id", "label", "critical_density_veh_per_m", "capacity_veh_per_s",
        "jam_storage_veh", "free_flow_time_s", "backward_wave_time_s",
    }
    for index, item in enumerate(ordered):
        _keys(item, required, set(), f"links[{index}]")
        ids.append(_identifier(item["id"], "link.id"))
        labels.append(_identifier(item["label"], "link.label"))
        parameters.append(LinkFDParameters(*(
            _finite(item[key], f"link.{key}", positive=True)
            for key in (
                "critical_density_veh_per_m", "capacity_veh_per_s", "jam_storage_veh",
                "free_flow_time_s", "backward_wave_time_s",
            )
        )))
    if not ids:
        raise ValueError("links must not be empty")
    return tuple(ids), tuple(labels), tuple(parameters)


def _parse_layout(
    raw: Any,
    *,
    link_ids: tuple[str, ...],
    link_labels: tuple[str, ...],
    node_ids: tuple[str, ...],
    source_link_ids: tuple[str, ...],
    sink_link_ids: tuple[str, ...],
    horizon_steps: int,
) -> LayoutDefinition:
    layout = _object(raw, "layout")
    _keys(layout, {"vertices", "links", "node_vertices", "selected_frames", "comparison_limits"}, set(), "layout")
    vertices = [_object(item, "layout.vertex") for item in _list(layout["vertices"], "layout.vertices")]
    _unique_records(vertices, "layout.vertices")
    vertices = sorted(vertices, key=lambda item: item["id"])
    vertex_ids: list[str] = []
    coordinates: list[list[float]] = []
    for item in vertices:
        _keys(item, {"id", "x", "y"}, set(), "layout.vertex")
        vertex_ids.append(_identifier(item["id"], "layout.vertex.id"))
        coordinates.append([_finite(item["x"], "vertex.x"), _finite(item["y"], "vertex.y")])
    if not vertex_ids:
        raise ValueError("layout.vertices must not be empty")
    vertex_index = {value: index for index, value in enumerate(vertex_ids)}
    edges = [_object(item, "layout.link") for item in _list(layout["links"], "layout.links")]
    edge_by_link: dict[str, tuple[str, str]] = {}
    for item in edges:
        _keys(item, {"link_id", "tail", "head"}, set(), "layout.link")
        link = _identifier(item["link_id"], "layout.link.link_id")
        if link in edge_by_link:
            raise ValueError("layout link IDs must be unique")
        edge_by_link[link] = (
            _identifier(item["tail"], "layout.link.tail"),
            _identifier(item["head"], "layout.link.head"),
        )
    if set(edge_by_link) != set(link_ids):
        raise ValueError("layout must cover exactly every simulation link")
    try:
        tails = [vertex_index[edge_by_link[link][0]] for link in link_ids]
        heads = [vertex_index[edge_by_link[link][1]] for link in link_ids]
    except KeyError as error:
        raise ValueError(f"layout link references unknown vertex {error.args[0]!r}") from error
    node_vertices = _object(layout["node_vertices"], "layout.node_vertices")
    if set(node_vertices) != set(node_ids):
        raise ValueError("node_vertices must cover exactly every simulation node")
    try:
        node_index = [vertex_index[_identifier(node_vertices[node], f"node_vertices.{node}")] for node in node_ids]
        source_index = [vertex_index[edge_by_link[link][0]] for link in source_link_ids]
        sink_index = [vertex_index[edge_by_link[link][1]] for link in sink_link_ids]
    except KeyError as error:
        raise ValueError(f"layout references unknown vertex {error.args[0]!r}") from error
    selected = tuple(_list(layout["selected_frames"], "layout.selected_frames"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > horizon_steps for value in selected):
        raise ValueError("selected_frames must contain boundary indices in [0,T]")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_frames must not contain duplicates")
    limits_raw = _object(layout["comparison_limits"], "layout.comparison_limits")
    limits: list[tuple[str, float, float]] = []
    for name in sorted(limits_raw):
        pair = _list(limits_raw[name], f"comparison_limits.{name}")
        if len(pair) != 2:
            raise ValueError("each comparison limit must be [minimum, maximum]")
        lower, upper = (_finite(pair[0], f"{name}.minimum"), _finite(pair[1], f"{name}.maximum"))
        if lower >= upper:
            raise ValueError("comparison limit minimum must be below maximum")
        limits.append((_identifier(name, "comparison limit name"), lower, upper))
    return LayoutDefinition(
        tuple(vertex_ids), torch.tensor(coordinates, dtype=torch.float64),
        torch.tensor(tails, dtype=torch.long), torch.tensor(heads, dtype=torch.long),
        link_labels, torch.tensor(node_index, dtype=torch.long),
        torch.tensor(source_index, dtype=torch.long), torch.tensor(sink_index, dtype=torch.long),
        selected, tuple(limits),
    )


def parse_scenario_config(config: Mapping[str, Any]) -> VisualizationCase:
    """Parse one strict V1 JSON object into accepted Milestone 0 records."""

    raw = _object(dict(config), "config")
    required = {"schema_version", "dt_s", "horizon_steps", "links", "nodes", "sources", "sinks", "controls", "layout"}
    _keys(raw, required, {"sensitivity"}, "config")
    if raw["schema_version"] != SCENARIO_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCENARIO_SCHEMA_VERSION!r}")
    dt_s = _finite(raw["dt_s"], "dt_s", positive=True)
    horizon = raw["horizon_steps"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon_steps must be a positive integer")
    link_ids, link_labels, link_parameters = _parse_links(_list(raw["links"], "links"))
    link_index = {value: index for index, value in enumerate(link_ids)}

    node_records = [_object(item, "node") for item in _list(raw["nodes"], "nodes")]
    _unique_records(node_records, "nodes")
    node_records = sorted(node_records, key=lambda item: item["id"])
    node_ids = tuple(_identifier(item["id"], "node.id") for item in node_records)
    movement_maps = []
    node_parameters = []
    phase_plans = []
    for item in node_records:
        _keys(item, {"id", "kind", "input_links", "output_links", "movements", "input_capacity_rate"}, {"phase_plan"}, "node")
        try:
            inputs = tuple(link_index[_identifier(value, "node.input_link")] for value in _list(item["input_links"], "node.input_links"))
            outputs = tuple(link_index[_identifier(value, "node.output_link")] for value in _list(item["output_links"], "node.output_links"))
        except KeyError as error:
            raise ValueError(f"node references unknown link {error.args[0]!r}") from error
        rows = []
        for movement in _list(item["movements"], "node.movements"):
            movement = _object(movement, "movement")
            _keys(movement, {"input", "output", "beta"}, set(), "movement")
            try:
                rows.append((link_index[_identifier(movement["input"], "movement.input")], link_index[_identifier(movement["output"], "movement.output")], _finite(movement["beta"], "movement.beta")))
            except KeyError as error:
                raise ValueError(f"movement references unknown link {error.args[0]!r}") from error
        movement_map = build_movement_map(inputs, outputs, rows)
        kind = NodeKind(item["kind"])
        input_capacity = _float_tensor(item["input_capacity_rate"], "node.input_capacity_rate", ndim=1)
        parameters = NodeParameters(movement_map, input_capacity, kind)
        phase_raw = item.get("phase_plan")
        if phase_raw is None:
            phase_plan = None
        else:
            phase_raw = _object(phase_raw, "phase_plan")
            _keys(phase_raw, {"phase_ids", "movement_phase_matrix", "input_saturation_rate"}, set(), "phase_plan")
            phase_ids = tuple(_list(phase_raw["phase_ids"], "phase_plan.phase_ids"))
            matrix_rows = _list(phase_raw["movement_phase_matrix"], "phase_plan.movement_phase_matrix")
            if any(
                not isinstance(row, list)
                or any(not isinstance(item, bool) for item in row)
                for row in matrix_rows
            ):
                raise TypeError("movement_phase_matrix must contain JSON booleans")
            matrix = torch.tensor(matrix_rows, dtype=torch.bool)
            saturation = _float_tensor(phase_raw["input_saturation_rate"], "phase_plan.input_saturation_rate", ndim=1)
            phase_plan = FixedPhasePlan(phase_ids, matrix, saturation)
        movement_maps.append(movement_map)
        node_parameters.append(parameters)
        phase_plans.append(phase_plan)

    def source_boundaries() -> tuple[tuple[str, ...], Tensor, Tensor | None]:
        key, series_key, optional_key = "sources", "arrivals", "entry_capacity"
        records = [_object(item, key[:-1]) for item in _list(raw[key], key)]
        seen: set[str] = set()
        ordered: list[tuple[str, dict[str, Any]]] = []
        for record in records:
            _keys(record, {"link_id", series_key}, {optional_key}, key[:-1])
            link = _identifier(record["link_id"], f"{key}.link_id")
            if link in seen:
                raise ValueError(f"{key} link IDs must be unique")
            seen.add(link)
            ordered.append((link, record))
        ordered.sort(key=lambda item: item[0])
        ids = tuple(item[0] for item in ordered)
        if any(value not in link_index for value in ids):
            raise ValueError(f"{key} references an unknown link")
        primary = _float_tensor([item[1][series_key] for item in ordered], f"{key}.{series_key}", ndim=2).T.contiguous()
        if primary.shape[0] != horizon:
            raise ValueError(f"{key}.{series_key} series must have length T")
        if bool((primary < 0.0).any()):
            raise ValueError(f"{key}.{series_key} must be nonnegative")
        present = [optional_key in item[1] for item in ordered]
        if any(present) and not all(present):
            raise ValueError(f"{optional_key} must be supplied for all or no {key}")
        optional = None
        if all(present) and present:
            optional = _float_tensor([item[1][optional_key] for item in ordered], f"{key}.{optional_key}", ndim=2).T.contiguous()
            if optional.shape != primary.shape or bool((optional < 0.0).any()):
                raise ValueError(f"{key}.{optional_key} must be nonnegative with length T")
        return ids, primary, optional

    def sink_boundaries() -> tuple[tuple[str, ...], Tensor | None]:
        records = [_object(item, "sink") for item in _list(raw["sinks"], "sinks")]
        ordered: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for record in records:
            _keys(record, {"link_id"}, {"receiving"}, "sink")
            link = _identifier(record["link_id"], "sinks.link_id")
            if link in seen:
                raise ValueError("sinks link IDs must be unique")
            seen.add(link)
            ordered.append((link, record))
        ordered.sort(key=lambda item: item[0])
        ids = tuple(item[0] for item in ordered)
        if any(value not in link_index for value in ids):
            raise ValueError("sinks references an unknown link")
        present = ["receiving" in item[1] for item in ordered]
        if any(present) and not all(present):
            raise ValueError("receiving must be supplied for all or no sinks")
        if not present or not all(present):
            return ids, None
        receiving = _float_tensor([item[1]["receiving"] for item in ordered], "sinks.receiving", ndim=2).T.contiguous()
        if receiving.shape != (horizon, len(ids)) or bool((receiving < 0.0).any()):
            raise ValueError("sinks.receiving must be nonnegative with length T")
        return ids, receiving

    source_ids, arrivals, entry_capacity = source_boundaries()
    sink_ids, sink_receiving = sink_boundaries()
    scenario = Scenario(dt_s, horizon, arrivals, entry_capacity, sink_receiving)
    network = NetworkDefinition(
        tuple(link_parameters), tuple(movement_maps), tuple(node_parameters), tuple(phase_plans),
        torch.tensor([link_index[value] for value in source_ids], dtype=torch.long),
        torch.tensor([link_index[value] for value in sink_ids], dtype=torch.long),
    )

    controls_raw = _object(raw["controls"], "controls")
    if set(controls_raw) != {node_ids[index] for index, p in enumerate(node_parameters) if p.kind is NodeKind.RESTRICTED_CONTINUUM_SIGNAL}:
        raise ValueError("controls must cover exactly every restricted signal node")
    greens: list[Tensor | None] = []
    for index, parameters in enumerate(node_parameters):
        if parameters.kind is NodeKind.ORDINARY_ORCA:
            greens.append(None)
        else:
            green = _float_tensor(controls_raw[node_ids[index]], f"controls.{node_ids[index]}", ndim=1)
            phase_plan = phase_plans[index]
            assert phase_plan is not None
            validate_green_split(green, len(phase_plan.phase_ids))
            greens.append(green)
    control = SignalControl(tuple(greens)) if greens or controls_raw else None
    if all(value is None for value in greens):
        control = None

    sensitivity = None
    if "sensitivity" in raw:
        request = _object(raw["sensitivity"], "sensitivity")
        _keys(request, {"node_id", "direction"}, set(), "sensitivity")
        node_id = _identifier(request["node_id"], "sensitivity.node_id")
        if node_id not in node_ids:
            raise ValueError("sensitivity node_id is unknown")
        node = node_ids.index(node_id)
        if node_parameters[node].kind is not NodeKind.RESTRICTED_CONTINUUM_SIGNAL:
            raise ValueError("sensitivity target must be a restricted signal node")
        direction = _float_tensor(request["direction"], "sensitivity.direction", ndim=1)
        if direction.shape != greens[node].shape:
            raise ValueError("sensitivity direction must match phase count")
        if not bool(torch.isclose(direction.sum(), direction.new_zeros(()), rtol=0.0, atol=DIRECTION_ATOL)):
            raise ValueError("sensitivity direction must sum to zero")
        if not bool(torch.isclose(torch.linalg.vector_norm(direction), direction.new_ones(()), rtol=0.0, atol=DIRECTION_ATOL)):
            raise ValueError("sensitivity direction must have unit L2 norm")
        sensitivity = SensitivityRequest(node, node_id, direction)

    layout = _parse_layout(raw["layout"], link_ids=link_ids, link_labels=link_labels, node_ids=node_ids, source_link_ids=source_ids, sink_link_ids=sink_ids, horizon_steps=horizon)
    return VisualizationCase(network, scenario, control, layout, link_ids, node_ids, source_ids, sink_ids, sensitivity, raw)


def load_scenario_config(path: str | Path) -> VisualizationCase:
    return parse_scenario_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _constraint_label(record: Any) -> str:
    tied = ",".join(f"{value.kind}:{value.local_index_or_pair}" for value in record.tied_constraint_ids) or "none"
    pivots = ",".join(f"{value.kind}:{value.local_index_or_pair}" for value in record.selected_pivot_ids) or "none"
    binding = "".join("1" if bool(value) else "0" for value in record.binding_mask)
    return f"binding={binding};tied={tied};pivots={pivots}"


def evaluate_sensitivity(case: VisualizationCase) -> SensitivitySummary | None:
    request = case.sensitivity_request
    if request is None:
        return None
    assert case.control is not None
    point = case.control.physical_green[request.node_index]
    assert point is not None

    def run(green: Tensor) -> RolloutResult:
        values = list(case.control.physical_green)
        values[request.node_index] = green
        return rollout(case.network, case.scenario, SignalControl(tuple(values)))

    check = directional_check(
        lambda green: total_system_time(run(green), case.scenario),
        point,
        request.direction,
        regime_function=lambda green: node_active_signature(run(green)),
    )
    event = any(row.feasible and not row.stable_regime for row in check.rows)
    return SensitivitySummary(
        "TST", "veh-eq*s", request.node_id, point.detach().cpu().clone(),
        request.direction.detach().cpu().clone(), float(check.reverse_directional.detach()),
        float(check.jvp_directional.detach()), check.reverse_jvp_agree,
        check.stable_scenario_passes, event,
    )


def extract_visualization_bundle(
    case: VisualizationCase,
    result: RolloutResult,
    *,
    sensitivity: SensitivitySummary | None = None,
) -> VisualizationBundle:
    """Detach and validate all renderer-facing values after simulation/autograd."""

    scenario, network = case.scenario, case.network
    occupancy = result.cumulative_link_history.n_in - result.cumulative_link_history.n_out
    storage = occupancy.new_tensor([value.jam_storage_veh for value in network.link_parameters])
    steps = result.step_results
    stack = lambda name: torch.stack(tuple(getattr(step, name) for step in steps))
    movement = tuple(torch.stack(tuple(step.movement_flow[node] for step in steps)) for node in range(network.node_count))
    node_total = torch.stack(tuple(torch.stack(tuple(values[step].sum() for values in movement)) for step in range(scenario.horizon_steps)), dim=0) if network.node_count else occupancy.new_zeros((scenario.horizon_steps, 0))
    labels = tuple(tuple(_constraint_label(record) for record in step.active_constraint_records) for step in steps)

    def detached(value: Tensor) -> Tensor:
        return value.detach().cpu().clone()

    bundle = VisualizationBundle(
        boundary_time_s=torch.arange(scenario.horizon_steps + 1, dtype=torch.float64) * scenario.dt_s,
        occupancy_veh=detached(occupancy), occupancy_ratio=detached(occupancy / storage),
        source_queue_veh=detached(result.source_queue_history),
        cumulative_sink_veh=detached(result.cumulative_sink_history),
        conservation_residual_veh=detached(mass_balance_residual(result, scenario)),
        sending_veh=detached(stack("sending")), receiving_veh=detached(stack("receiving")),
        link_inflow_veh=detached(stack("link_inflow")), link_outflow_veh=detached(stack("link_outflow")),
        node_total_flow_veh=detached(node_total), movement_flow_veh=tuple(detached(value) for value in movement),
        active_regime_labels=labels, total_system_time_veh_s=float(total_system_time(result, scenario).detach()),
        layout=case.layout, link_ids=case.link_ids, node_ids=case.node_ids,
        source_ids=case.source_ids, sink_ids=case.sink_ids,
        link_capacity_step_veh=torch.tensor([value.capacity_veh_per_s * scenario.dt_s for value in network.link_parameters], dtype=torch.float64),
        link_storage_veh=detached(storage),
        physical_green_by_node=(
            tuple(None if value is None else detached(value) for value in case.control.physical_green)
            if case.control is not None
            else tuple(None for _ in range(network.node_count))
        ),
        sensitivity=sensitivity,
    )
    validate_visualization_bundle(bundle)
    return bundle


def run_visualization_case(case: VisualizationCase) -> VisualizationBundle:
    sensitivity = evaluate_sensitivity(case)
    result = rollout(case.network, case.scenario, case.control)
    return extract_visualization_bundle(case, result, sensitivity=sensitivity)


def validate_visualization_bundle(bundle: VisualizationBundle) -> None:
    tensors = [
        bundle.boundary_time_s, bundle.occupancy_veh, bundle.occupancy_ratio,
        bundle.source_queue_veh, bundle.cumulative_sink_veh,
        bundle.conservation_residual_veh, bundle.sending_veh, bundle.receiving_veh,
        bundle.link_inflow_veh, bundle.link_outflow_veh, bundle.node_total_flow_veh,
        bundle.link_capacity_step_veh, bundle.link_storage_veh, *bundle.movement_flow_veh,
        *(value for value in bundle.physical_green_by_node if value is not None),
    ]
    for value in tensors:
        if value.dtype != torch.float64 or value.device.type != "cpu" or value.requires_grad:
            raise TypeError("bundle continuous tensors must be detached CPU float64")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("bundle tensors must be finite")
    t = bundle.boundary_time_s.numel() - 1
    a = len(bundle.link_ids)
    if bundle.occupancy_veh.shape != (t + 1, a) or bundle.occupancy_ratio.shape != (t + 1, a):
        raise ValueError("occupancy arrays must have shape [T+1,A]")
    if len(bundle.physical_green_by_node) != len(bundle.node_ids):
        raise ValueError("physical greens must align with node order")
    for value in (bundle.sending_veh, bundle.receiving_veh, bundle.link_inflow_veh, bundle.link_outflow_veh):
        if value.shape != (t, a):
            raise ValueError("link interval arrays must have shape [T,A]")
    if bool((bundle.occupancy_ratio < -CONSERVATION_ATOL).any()) or bool((bundle.occupancy_ratio > 1.0 + CONSERVATION_ATOL).any()):
        raise ValueError("occupancy ratios must lie in [0,1]")
    if float(torch.max(torch.abs(bundle.conservation_residual_veh))) > CONSERVATION_ATOL:
        raise ValueError("bundle violates the Milestone 0 conservation tolerance")
    for value in tensors:
        if value.data_ptr() in {bundle.occupancy_veh.data_ptr()} and value is not bundle.occupancy_veh:
            raise ValueError("bundle tensors must not alias")


def _tensor_data(value: Tensor) -> list[Any]:
    return value.tolist()


def bundle_to_dict(bundle: VisualizationBundle) -> dict[str, Any]:
    validate_visualization_bundle(bundle)
    layout = bundle.layout
    sensitivity = None
    if bundle.sensitivity is not None:
        value = bundle.sensitivity
        sensitivity = {
            "objective_name": value.objective_name, "objective_unit": value.objective_unit,
            "control_node_id": value.control_node_id, "physical_green": _tensor_data(value.physical_green),
            "direction": _tensor_data(value.direction), "reverse_directional": value.reverse_directional,
            "jvp_directional": value.jvp_directional, "reverse_jvp_agree": value.reverse_jvp_agree,
            "stable_scenario_passes": value.stable_scenario_passes, "event_detected": value.event_detected,
        }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "metadata": {"link_ids": list(bundle.link_ids), "node_ids": list(bundle.node_ids), "source_ids": list(bundle.source_ids), "sink_ids": list(bundle.sink_ids)},
        "layout": {
            "vertex_ids": list(layout.vertex_ids), "vertex_xy": _tensor_data(layout.vertex_xy),
            "link_tail_index": _tensor_data(layout.link_tail_index), "link_head_index": _tensor_data(layout.link_head_index),
            "link_labels": list(layout.link_labels), "node_vertex_index": _tensor_data(layout.node_vertex_index),
            "source_vertex_index": _tensor_data(layout.source_vertex_index), "sink_vertex_index": _tensor_data(layout.sink_vertex_index),
            "selected_frames": list(layout.selected_frames), "comparison_limits": [list(value) for value in layout.comparison_limits],
        },
        "arrays": {name: _tensor_data(getattr(bundle, name)) for name in (
            "boundary_time_s", "occupancy_veh", "occupancy_ratio", "source_queue_veh", "cumulative_sink_veh",
            "conservation_residual_veh", "sending_veh", "receiving_veh", "link_inflow_veh", "link_outflow_veh",
            "node_total_flow_veh", "link_capacity_step_veh", "link_storage_veh",
        )},
        "movement_flow_veh": [_tensor_data(value) for value in bundle.movement_flow_veh],
        "physical_green_by_node": [None if value is None else _tensor_data(value) for value in bundle.physical_green_by_node],
        "active_regime_labels": [list(value) for value in bundle.active_regime_labels],
        "total_system_time_veh_s": bundle.total_system_time_veh_s,
        "sensitivity": sensitivity,
    }


def bundle_from_dict(raw: Mapping[str, Any]) -> VisualizationBundle:
    value = _object(dict(raw), "bundle")
    _keys(value, {"schema_version", "metadata", "layout", "arrays", "movement_flow_veh", "physical_green_by_node", "active_regime_labels", "total_system_time_veh_s", "sensitivity"}, set(), "bundle")
    if value["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"bundle schema_version must equal {BUNDLE_SCHEMA_VERSION!r}")
    metadata = _object(value["metadata"], "bundle.metadata")
    layout_raw = _object(value["layout"], "bundle.layout")
    arrays = _object(value["arrays"], "bundle.arrays")
    layout = LayoutDefinition(
        tuple(layout_raw["vertex_ids"]), _float_tensor(layout_raw["vertex_xy"], "vertex_xy", ndim=2),
        torch.tensor(layout_raw["link_tail_index"], dtype=torch.long), torch.tensor(layout_raw["link_head_index"], dtype=torch.long),
        tuple(layout_raw["link_labels"]), torch.tensor(layout_raw["node_vertex_index"], dtype=torch.long),
        torch.tensor(layout_raw["source_vertex_index"], dtype=torch.long), torch.tensor(layout_raw["sink_vertex_index"], dtype=torch.long),
        tuple(layout_raw["selected_frames"]), tuple((str(item[0]), float(item[1]), float(item[2])) for item in layout_raw["comparison_limits"]),
    )
    sensitivity_raw = value["sensitivity"]
    sensitivity = None
    if sensitivity_raw is not None:
        sensitivity_raw = _object(sensitivity_raw, "bundle.sensitivity")
        sensitivity = SensitivitySummary(
            sensitivity_raw["objective_name"], sensitivity_raw["objective_unit"], sensitivity_raw["control_node_id"],
            _float_tensor(sensitivity_raw["physical_green"], "physical_green", ndim=1),
            _float_tensor(sensitivity_raw["direction"], "direction", ndim=1),
            float(sensitivity_raw["reverse_directional"]), float(sensitivity_raw["jvp_directional"]),
            bool(sensitivity_raw["reverse_jvp_agree"]), bool(sensitivity_raw["stable_scenario_passes"]), bool(sensitivity_raw["event_detected"]),
        )
    tensor = lambda name: _float_tensor(arrays[name], name)
    bundle = VisualizationBundle(
        *(tensor(name) for name in (
            "boundary_time_s", "occupancy_veh", "occupancy_ratio", "source_queue_veh", "cumulative_sink_veh",
            "conservation_residual_veh", "sending_veh", "receiving_veh", "link_inflow_veh", "link_outflow_veh", "node_total_flow_veh",
        )),
        tuple(_float_tensor(item, "movement_flow_veh", ndim=2) for item in value["movement_flow_veh"]),
        tuple(tuple(str(item) for item in row) for row in value["active_regime_labels"]),
        float(value["total_system_time_veh_s"]), layout,
        tuple(metadata["link_ids"]), tuple(metadata["node_ids"]), tuple(metadata["source_ids"]), tuple(metadata["sink_ids"]),
        tensor("link_capacity_step_veh"), tensor("link_storage_veh"),
        tuple(None if item is None else _float_tensor(item, "physical_green", ndim=1) for item in value["physical_green_by_node"]),
        sensitivity,
    )
    validate_visualization_bundle(bundle)
    return bundle


def save_bundle(bundle: VisualizationBundle, path: str | Path) -> None:
    Path(path).write_text(json.dumps(bundle_to_dict(bundle), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_bundle(path: str | Path) -> VisualizationBundle:
    return bundle_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
