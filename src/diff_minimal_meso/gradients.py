"""Directional reverse/JVP/central-difference diagnostics for Milestone 0."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Hashable

import torch
from torch import Tensor

from .simulation import RolloutResult


STEP_RATIOS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
AD_ATOL = 1.0e-10
AD_RTOL = 1.0e-10
FD_ATOL = 1.0e-8
FD_RTOL = 1.0e-6


@dataclass(frozen=True, slots=True)
class StepScanRow:
    step_size: float
    feasible: bool
    objective_plus: Tensor | None
    objective_minus: Tensor | None
    finite_difference: Tensor | None
    absolute_error: Tensor | None
    baseline_regime: Hashable | None
    plus_regime: Hashable | None
    minus_regime: Hashable | None
    stable_regime: bool
    passes: bool


@dataclass(frozen=True, slots=True)
class DirectionalCheck:
    baseline_objective: Tensor
    direction: Tensor
    reverse_directional: Tensor
    jvp_directional: Tensor
    reverse_jvp_agree: bool
    rows: tuple[StepScanRow, ...]
    stable_adjacent_pass_count: int
    stable_scenario_passes: bool


def _validate_point_direction(point: Tensor, direction: Tensor) -> None:
    for name, value in (("point", point), ("direction", direction)):
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise ValueError(f"{name} must have shape [P]")
        if value.dtype != torch.float64:
            raise TypeError(f"{name} must use torch.float64")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if point.shape != direction.shape or point.numel() < 2:
        raise ValueError("point and direction must have equal phase dimension")
    if point.device != direction.device:
        raise ValueError("point and direction must use the same device")
    zero = point.new_zeros(())
    one = point.new_ones(())
    if not bool(torch.isclose(direction.sum(), zero, rtol=0.0, atol=1.0e-12)):
        raise ValueError("physical direction must sum to zero")
    if not bool(torch.isclose(torch.linalg.vector_norm(direction), one,
                              rtol=0.0, atol=1.0e-12)):
        raise ValueError("physical direction must have unit L2 norm")


def physical_step_scale(point: Tensor, direction: Tensor) -> float:
    """Return the approved simplex-feasible scan scale."""

    _validate_point_direction(point, direction)
    nonzero = direction != 0.0
    h_max = torch.min(point[nonzero] / torch.abs(direction[nonzero]))
    return min(1.0, float(h_max))


def node_active_signature(result: RolloutResult) -> tuple[Hashable, ...]:
    """Return stable node binding/tie/pivot metadata for regime comparison."""

    signature: list[Hashable] = []
    for step_index, step in enumerate(result.step_results):
        for node_index, record in enumerate(step.active_constraint_records):
            tied = tuple(
                (identifier.kind, identifier.local_index_or_pair)
                for identifier in record.tied_constraint_ids
            )
            pivots = tuple(
                (identifier.kind, identifier.local_index_or_pair)
                for identifier in record.selected_pivot_ids
            )
            signature.append(
                (step_index, node_index, tuple(record.binding_mask.tolist()), tied, pivots)
            )
    return tuple(signature)


def _close(left: Tensor, right: Tensor, atol: float, rtol: float) -> bool:
    tolerance = atol + rtol * torch.maximum(torch.abs(left), torch.abs(right))
    return bool(torch.abs(left - right) <= tolerance)


def _maximum_adjacent_true(values: list[bool]) -> int:
    maximum = current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def directional_check(
    function: Callable[[Tensor], Tensor],
    point: Tensor,
    direction: Tensor,
    *,
    regime_function: Callable[[Tensor], Hashable] | None = None,
) -> DirectionalCheck:
    """Apply the frozen reverse/JVP/central-FD rule on the physical simplex."""

    _validate_point_direction(point, direction)
    if bool((point < 0.0).any()):
        raise ValueError("physical point must be nonnegative")
    if not bool(torch.isclose(point.sum(), point.new_ones(()), rtol=0.0, atol=1.0e-9)):
        raise ValueError("physical point must sum to one")

    reverse_point = point.detach().clone().requires_grad_(True)
    baseline = function(reverse_point)
    if baseline.ndim != 0 or baseline.dtype != torch.float64:
        raise TypeError("diagnostic function must return a scalar float64 tensor")
    gradient = torch.autograd.grad(baseline, reverse_point)[0]
    reverse = torch.dot(gradient, direction)
    _, jvp = torch.func.jvp(function, (point,), (direction,))
    ad_agreement = _close(reverse, jvp, AD_ATOL, AD_RTOL)

    baseline_regime = regime_function(point) if regime_function is not None else None
    rows: list[StepScanRow] = []
    scale = physical_step_scale(point, direction)
    for ratio in STEP_RATIOS:
        h = scale * ratio
        plus_point = point + h * direction
        minus_point = point - h * direction
        feasible = bool((plus_point >= 0.0).all()) and bool((minus_point >= 0.0).all())
        if not feasible:
            rows.append(StepScanRow(h, False, None, None, None, None,
                                    baseline_regime, None, None, False, False))
            continue
        plus = function(plus_point)
        minus = function(minus_point)
        finite_difference = (plus - minus) / (2.0 * h)
        plus_regime = regime_function(plus_point) if regime_function is not None else None
        minus_regime = regime_function(minus_point) if regime_function is not None else None
        stable = (
            regime_function is None
            or (baseline_regime == plus_regime == minus_regime)
        )
        error = torch.abs(reverse - finite_difference)
        passes = stable and _close(reverse, finite_difference, FD_ATOL, FD_RTOL)
        rows.append(StepScanRow(h, True, plus, minus, finite_difference, error,
                                baseline_regime, plus_regime, minus_regime,
                                stable, passes))

    adjacent = _maximum_adjacent_true([row.passes for row in rows])
    return DirectionalCheck(
        baseline_objective=baseline,
        direction=direction,
        reverse_directional=reverse,
        jvp_directional=jvp,
        reverse_jvp_agree=ad_agreement,
        rows=tuple(rows),
        stable_adjacent_pass_count=adjacent,
        stable_scenario_passes=ad_agreement and adjacent >= 3,
    )


def two_phase_direction(*, device: torch.device | str | None = None) -> Tensor:
    return torch.tensor(
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        dtype=torch.float64,
        device=device,
    )
