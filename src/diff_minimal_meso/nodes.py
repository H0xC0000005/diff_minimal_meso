"""First-order Branch-T and restricted continuum-service node solvers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor

from .movements import (
    NodeMovementMap,
    aggregate_input_flow,
    aggregate_output_flow,
)


VALIDATION_ATOL = 1.0e-10


class NodeKind(str, Enum):
    ORDINARY_ORCA = "ordinary_orca"
    RESTRICTED_CONTINUUM_SIGNAL = "restricted_continuum_signal"


@dataclass(frozen=True, slots=True, order=True)
class ConstraintID:
    kind: str
    local_index_or_pair: int | tuple[int, int]


@dataclass(frozen=True, slots=True)
class NodeParameters:
    movement_map: NodeMovementMap
    input_capacity_rate: Tensor
    kind: NodeKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NodeKind):
            raise TypeError("kind must be a NodeKind")
        value = self.input_capacity_rate
        if not isinstance(value, Tensor) or value.shape != (self.movement_map.input_count,):
            raise ValueError("input_capacity_rate must have shape [I]")
        if value.dtype != torch.float64 or value.device.type != "cpu":
            raise TypeError("input_capacity_rate must be a CPU float64 tensor")
        if value.requires_grad:
            raise ValueError("input_capacity_rate is structural metadata")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0.0).any()):
            raise ValueError("input capacities must be finite and strictly positive")


@dataclass(frozen=True, slots=True)
class NodeFlows:
    movement_flow: Tensor
    input_outflow: Tensor
    output_inflow: Tensor
    binding_mask: Tensor
    tied_constraint_ids: tuple[ConstraintID, ...]
    selected_pivot_ids: tuple[ConstraintID, ...]


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validate_continuous(
    name: str, value: Tensor, count: int, *, device: torch.device | None = None
) -> None:
    if not isinstance(value, Tensor) or value.shape != (count,):
        raise ValueError(f"{name} must have shape [{count}]")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}")
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")


def _input_demand_and_validate_fifo(
    oriented_demand: Tensor, movement_map: NodeMovementMap
) -> Tensor:
    demand = aggregate_input_flow(oriented_demand, movement_map)
    input_index = movement_map.movement_input_index.to(oriented_demand.device)
    beta = movement_map.turning_fraction.to(oriented_demand.device)
    expected = beta * demand[input_index]
    if not bool(
        torch.isclose(
            oriented_demand, expected, rtol=0.0, atol=1.0e-9
        ).all()
    ):
        raise ValueError("oriented demand is inconsistent with the movement turning fractions")
    return demand


def _ordinary_constraint_ids(parameters: NodeParameters) -> tuple[ConstraintID, ...]:
    return tuple(
        [ConstraintID("demand", i) for i in range(parameters.movement_map.input_count)]
        + [
            ConstraintID("receiving", j)
            for j in range(parameters.movement_map.output_count)
        ]
    )


def solve_ordinary_orca(
    oriented_demand: Tensor,
    receiving: Tensor,
    parameters: NodeParameters,
    dt_s: float,
) -> NodeFlows:
    """Solve the published oriented-capacity-proportional Branch-T instance."""

    if parameters.kind is not NodeKind.ORDINARY_ORCA:
        raise ValueError("ordinary solver requires ORDINARY_ORCA parameters")
    dt = _positive_finite("dt_s", dt_s)
    movement_map = parameters.movement_map
    _validate_continuous("oriented_demand", oriented_demand, movement_map.movement_count)
    _validate_continuous(
        "receiving", receiving, movement_map.output_count, device=oriented_demand.device
    )
    demand = _input_demand_and_validate_fifo(oriented_demand, movement_map)
    beta = movement_map.turning_fraction.to(oriented_demand.device)
    m_input = movement_map.movement_input_index.to(oriented_demand.device)
    m_output = movement_map.movement_output_index.to(oriented_demand.device)
    capacity_step = parameters.input_capacity_rate.to(oriented_demand.device) * dt

    resolved: list[Tensor] = [oriented_demand.new_zeros(()) for _ in range(movement_map.input_count)]
    unresolved = {i for i in range(movement_map.input_count) if bool(demand[i] != 0.0)}
    tied_ids: list[ConstraintID] = []
    pivot_ids: list[ConstraintID] = []
    incumbent: int | None = None
    iterations = 0

    while unresolved:
        iterations += 1
        if iterations > 2 * (movement_map.input_count + movement_map.output_count):
            raise RuntimeError("ORCA active set failed its finite progress bound")
        levels: dict[int, Tensor] = {}
        for output in range(movement_map.output_count):
            residual = receiving[output]
            denominator = oriented_demand.new_zeros(())
            for movement in range(movement_map.movement_count):
                i = int(movement_map.movement_input_index[movement])
                j = int(movement_map.movement_output_index[movement])
                if j != output:
                    continue
                if i in unresolved:
                    denominator = denominator + beta[movement] * capacity_step[i]
                else:
                    residual = residual - beta[movement] * resolved[i]
            if bool(residual < -VALIDATION_ATOL):
                raise ValueError("resolved flow exceeds output receiving mass")
            if bool(denominator > 0.0):
                if bool(residual < 0.0):
                    raise ValueError("negative residual receiving mass is not clipped")
                levels[output] = residual / denominator

        if not levels:
            for i in tuple(unresolved):
                resolved[i] = demand[i]
                unresolved.remove(i)
            break

        level_values = torch.stack(tuple(levels.values()))
        minimum = torch.min(level_values)
        tied_outputs = tuple(j for j, value in levels.items() if bool(value == minimum))
        if len(tied_outputs) > 1:
            for output in tied_outputs:
                identifier = ConstraintID("receiving", output)
                if identifier not in tied_ids:
                    tied_ids.append(identifier)
        pivot = incumbent if incumbent in tied_outputs else min(tied_outputs)
        incumbent = pivot
        pivot_ids.append(ConstraintID("receiving", pivot))

        competitors = {
            i
            for i in unresolved
            if any(
                int(movement_map.movement_input_index[m]) == i
                and int(movement_map.movement_output_index[m]) == pivot
                and bool(beta[m] > 0.0)
                for m in range(movement_map.movement_count)
            )
        }
        low_demand = {
            i for i in competitors if bool(demand[i] <= minimum * capacity_step[i])
        }
        fixed = low_demand if low_demand else competitors
        if not fixed:
            raise RuntimeError("selected ORCA pivot has no unresolved competitor")
        for i in fixed:
            resolved[i] = demand[i] if low_demand else minimum * capacity_step[i]
            unresolved.remove(i)

    input_outflow = torch.stack(tuple(resolved))
    movement_flow = beta * input_outflow[m_input]
    output_inflow = aggregate_output_flow(movement_flow, movement_map)
    if bool((movement_flow < 0.0).any()) or bool(
        (output_inflow > receiving + VALIDATION_ATOL).any()
    ):
        raise ValueError("ORCA result violates nonnegativity or receiving feasibility")
    demand_binding = torch.isclose(
        input_outflow, demand, rtol=0.0, atol=VALIDATION_ATOL
    )
    receiving_binding = torch.isclose(
        output_inflow, receiving, rtol=0.0, atol=VALIDATION_ATOL
    )
    return NodeFlows(
        movement_flow=movement_flow,
        input_outflow=input_outflow,
        output_inflow=output_inflow,
        binding_mask=torch.cat((demand_binding, receiving_binding)),
        tied_constraint_ids=tuple(tied_ids),
        selected_pivot_ids=tuple(pivot_ids),
    )


def solve_restricted_continuum_signal(
    oriented_demand: Tensor,
    receiving: Tensor,
    parameters: NodeParameters,
    dt_s: float,
    input_exposure: Tensor,
    input_saturation_rate: Tensor,
) -> NodeFlows:
    """Solve the approved H1--H4 restricted continuum-service equations."""

    if parameters.kind is not NodeKind.RESTRICTED_CONTINUUM_SIGNAL:
        raise ValueError("signal solver requires RESTRICTED_CONTINUUM_SIGNAL parameters")
    dt = _positive_finite("dt_s", dt_s)
    movement_map = parameters.movement_map
    _validate_continuous("oriented_demand", oriented_demand, movement_map.movement_count)
    _validate_continuous(
        "receiving", receiving, movement_map.output_count, device=oriented_demand.device
    )
    _validate_continuous(
        "input_exposure", input_exposure, movement_map.input_count,
        device=oriented_demand.device,
    )
    if bool((input_exposure > 1.0).any()):
        raise ValueError("input exposure must be at most one")
    _validate_continuous(
        "input_saturation_rate", input_saturation_rate,
        movement_map.input_count, device=oriented_demand.device,
    )
    if bool((input_saturation_rate <= 0.0).any()):
        raise ValueError("input saturation rates must be strictly positive")
    demand = _input_demand_and_validate_fifo(oriented_demand, movement_map)
    beta = movement_map.turning_fraction.to(oriented_demand.device)
    m_input = movement_map.movement_input_index.to(oriented_demand.device)
    m_output = movement_map.movement_output_index.to(oriented_demand.device)

    for output in range(movement_map.output_count):
        participating = {
            int(movement_map.movement_input_index[m])
            for m in range(movement_map.movement_count)
            if int(movement_map.movement_output_index[m]) == output and bool(beta[m] > 0.0)
        }
        exposure_sum = torch.stack(tuple(input_exposure[i] for i in participating)).sum()
        if bool(exposure_sum > 1.0 + VALIDATION_ATOL):
            raise ValueError("H2 exposure shares exceed one for an output")

    input_values: list[Tensor] = []
    tied: list[ConstraintID] = []
    pivots: list[ConstraintID] = []
    binding_parts: list[Tensor] = []
    for i in range(movement_map.input_count):
        service = input_exposure[i] * input_saturation_rate[i] * dt
        ids = [ConstraintID("demand", i), ConstraintID("service", i)]
        candidates = [demand[i], service]
        for m in range(movement_map.movement_count):
            if int(movement_map.movement_input_index[m]) == i and bool(beta[m] > 0.0):
                bound = input_exposure[i] * receiving[m_output[m]] / beta[m]
                candidates.append(bound)
                ids.append(
                    ConstraintID(
                        "receiving_share",
                        (i, int(movement_map.movement_output_index[m])),
                    )
                )
        candidate_tensor = torch.stack(tuple(candidates))
        value = torch.min(candidate_tensor)
        active = candidate_tensor == value
        active_ids = tuple(identifier for identifier, flag in zip(ids, active, strict=True) if bool(flag))
        if len(active_ids) > 1:
            tied.extend(identifier for identifier in active_ids if identifier not in tied)
        pivots.append(active_ids[0])
        input_values.append(value)
        binding_parts.append(active)

    input_outflow = torch.stack(tuple(input_values))
    movement_flow = beta * input_outflow[m_input]
    output_inflow = aggregate_output_flow(movement_flow, movement_map)
    if bool((output_inflow > receiving + VALIDATION_ATOL).any()):
        raise ValueError("restricted signal result violates output receiving mass")
    # Binding order is input-major: demand, service, then each positive-turn
    # receiving-share constraint in canonical movement order for that input.
    return NodeFlows(
        movement_flow=movement_flow,
        input_outflow=input_outflow,
        output_inflow=output_inflow,
        binding_mask=torch.cat(tuple(binding_parts)),
        tied_constraint_ids=tuple(tied),
        selected_pivot_ids=tuple(pivots),
    )


def solve_node(
    oriented_demand: Tensor,
    receiving: Tensor,
    parameters: NodeParameters,
    dt_s: float,
    *,
    input_exposure: Tensor | None = None,
    input_saturation_rate: Tensor | None = None,
) -> NodeFlows:
    """Dispatch explicitly by the node's fixed structural regime."""

    if parameters.kind is NodeKind.ORDINARY_ORCA:
        if input_exposure is not None or input_saturation_rate is not None:
            raise ValueError("ordinary ORCA does not accept signal service inputs")
        return solve_ordinary_orca(oriented_demand, receiving, parameters, dt_s)
    if input_exposure is None or input_saturation_rate is None:
        raise ValueError("restricted signal nodes require exposure and saturation rate")
    return solve_restricted_continuum_signal(
        oriented_demand,
        receiving,
        parameters,
        dt_s,
        input_exposure,
        input_saturation_rate,
    )
