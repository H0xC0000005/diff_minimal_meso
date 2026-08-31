"""Continuum fixed-time signal exposure and nominal service mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from .movements import NodeMovementMap


SIMPLEX_ATOL = 1.0e-9
HOMOGENEOUS_EXPOSURE_ATOL = 1.0e-9


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class FixedPhasePlan:
    """Fixed phase permissions and structural input saturation rates."""

    phase_ids: tuple[int, ...]
    movement_phase_matrix: Tensor
    input_saturation_rate: Tensor

    def __post_init__(self) -> None:
        if not self.phase_ids:
            raise ValueError("phase_ids must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.phase_ids):
            raise TypeError("phase_ids must contain integers")
        if len(set(self.phase_ids)) != len(self.phase_ids):
            raise ValueError("phase_ids must be unique")
        matrix = self.movement_phase_matrix
        if not isinstance(matrix, Tensor) or matrix.ndim != 2:
            raise ValueError("movement_phase_matrix must have shape [M, P]")
        if matrix.dtype != torch.bool or matrix.device.type != "cpu":
            raise TypeError("movement_phase_matrix must be a CPU bool tensor")
        if matrix.shape[1] != len(self.phase_ids):
            raise ValueError("movement_phase_matrix phase dimension must match phase_ids")
        saturation = self.input_saturation_rate
        if not isinstance(saturation, Tensor) or saturation.ndim != 1:
            raise ValueError("input_saturation_rate must have shape [I]")
        if saturation.dtype != torch.float64 or saturation.device.type != "cpu":
            raise TypeError("input_saturation_rate must be a CPU float64 tensor")
        if saturation.requires_grad:
            raise ValueError("input_saturation_rate is structural metadata")
        if not bool(torch.isfinite(saturation).all()) or bool((saturation <= 0.0).any()):
            raise ValueError("input saturation rates must be finite and strictly positive")


@dataclass(frozen=True, slots=True)
class ContinuumService:
    physical_green: Tensor
    movement_exposure: Tensor
    input_exposure: Tensor
    input_service_mass: Tensor


def validate_green_split(physical_green: Tensor, phase_count: int) -> None:
    """Validate the physical unit-simplex coordinate without modifying it."""

    if not isinstance(physical_green, Tensor) or physical_green.shape != (phase_count,):
        raise ValueError(f"physical_green must have shape [{phase_count}]")
    if physical_green.dtype != torch.float64:
        raise TypeError("physical_green must use torch.float64")
    if not bool(torch.isfinite(physical_green).all()) or bool(
        (physical_green < 0.0).any()
    ):
        raise ValueError("physical_green must be finite and nonnegative")
    if not bool(
        torch.isclose(
            physical_green.sum(), physical_green.new_tensor(1.0),
            rtol=0.0, atol=SIMPLEX_ATOL,
        )
    ):
        raise ValueError("physical_green must sum to one")


def phase_exposure(physical_green: Tensor, phase_plan: FixedPhasePlan) -> Tensor:
    """Return additive movement exposure ``A g`` on the green tensor's device."""

    validate_green_split(physical_green, len(phase_plan.phase_ids))
    matrix = phase_plan.movement_phase_matrix.to(device=physical_green.device)
    exposure = matrix.to(dtype=torch.float64) @ physical_green
    if bool((exposure < 0.0).any()) or bool((exposure > 1.0).any()):
        raise ValueError("overlapping phase exposure lies outside [0, 1]")
    return exposure


def continuum_service(
    physical_green: Tensor,
    phase_plan: FixedPhasePlan,
    movement_map: NodeMovementMap,
    dt_s: float,
) -> ContinuumService:
    """Map physical green to approved H1/H2 signal-service parameters."""

    dt = _positive_finite("dt_s", dt_s)
    if phase_plan.movement_phase_matrix.shape[0] != movement_map.movement_count:
        raise ValueError("phase plan movement dimension must match the movement map")
    if phase_plan.input_saturation_rate.shape != (movement_map.input_count,):
        raise ValueError("phase plan saturation dimension must match the movement map")
    movement_exposure = phase_exposure(physical_green, phase_plan)
    beta = movement_map.turning_fraction.to(physical_green.device)
    movement_input = movement_map.movement_input_index.to(physical_green.device)
    movement_output = movement_map.movement_output_index.to(physical_green.device)

    input_values: list[Tensor] = []
    for input_index in range(movement_map.input_count):
        mask = (movement_input == input_index) & (beta > 0.0)
        values = movement_exposure[mask]
        reference = values[0]
        if not bool(
            torch.isclose(
                values, reference, rtol=0.0, atol=HOMOGENEOUS_EXPOSURE_ATOL
            ).all()
        ):
            raise ValueError("H1 requires homogeneous exposure within each input")
        input_values.append(reference)
    input_exposure = torch.stack(tuple(input_values))

    for output_index in range(movement_map.output_count):
        participating_inputs = tuple(
            input_index
            for input_index in range(movement_map.input_count)
            if bool(
                (
                    (movement_output == output_index)
                    & (movement_input == input_index)
                    & (beta > 0.0)
                ).any()
            )
        )
        if not participating_inputs:
            raise ValueError("every declared output must have a positive-turn movement")
        exposure_sum = torch.stack(
            tuple(input_exposure[input_index] for input_index in participating_inputs)
        ).sum()
        if bool(exposure_sum > 1.0):
            raise ValueError("H2 exposure shares exceed one for an output")

    saturation = phase_plan.input_saturation_rate.to(physical_green.device)
    return ContinuumService(
        physical_green=physical_green,
        movement_exposure=movement_exposure,
        input_exposure=input_exposure,
        input_service_mass=input_exposure * saturation * dt,
    )


def two_phase_split(theta: Tensor) -> Tensor:
    """Return the diagnostic physical split ``(theta, 1-theta)``."""

    if not isinstance(theta, Tensor) or theta.ndim != 0:
        raise ValueError("theta must be a scalar tensor")
    if theta.dtype != torch.float64:
        raise TypeError("theta must use torch.float64")
    if not bool(torch.isfinite(theta)) or bool(theta < 0.0) or bool(theta > 1.0):
        raise ValueError("theta must lie in [0, 1]")
    return torch.stack((theta, 1.0 - theta))
