"""Validated fixed movement maps and differentiable movement accounting."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import Tensor


TURNING_FRACTION_ATOL = 1.0e-9


def _validate_link_ids(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise TypeError(f"{name} must contain integer link IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate link IDs")


def validate_turning_fractions(
    turning_fraction: Tensor,
    movement_input_index: Tensor,
    input_count: int,
    *,
    atol: float = TURNING_FRACTION_ATOL,
) -> None:
    """Validate fixed float64 turning fractions without repair or normalization."""

    if turning_fraction.ndim != 1 or turning_fraction.dtype != torch.float64:
        raise TypeError("turning_fraction must be a one-dimensional float64 tensor")
    if turning_fraction.device.type != "cpu":
        raise ValueError("structural turning_fraction must be stored on CPU")
    if turning_fraction.requires_grad:
        raise ValueError("turning_fraction is fixed structural metadata")
    if not bool(torch.isfinite(turning_fraction).all()):
        raise ValueError("turning fractions must be finite")
    if bool((turning_fraction < 0.0).any()):
        raise ValueError("turning fractions must be nonnegative")
    if movement_input_index.shape != turning_fraction.shape:
        raise ValueError("turning fractions and movement indices must have equal length")
    for input_index in range(input_count):
        row_sum = turning_fraction[movement_input_index == input_index].sum()
        if not bool(torch.isclose(row_sum, row_sum.new_tensor(1.0), rtol=0.0, atol=atol)):
            raise ValueError(
                f"turning fractions for local input {input_index} must sum to one"
            )


@dataclass(frozen=True, slots=True)
class NodeMovementMap:
    """Canonical sparse movement incidence for one node.

    Link-ID tuple order defines local input/output indices. Movement tensors are
    CPU structural metadata ordered lexicographically by local ``(input, output)``.
    """

    input_link_ids: tuple[int, ...]
    output_link_ids: tuple[int, ...]
    movement_input_index: Tensor
    movement_output_index: Tensor
    turning_fraction: Tensor

    def __post_init__(self) -> None:
        _validate_link_ids("input_link_ids", self.input_link_ids)
        _validate_link_ids("output_link_ids", self.output_link_ids)
        for name, index in (
            ("movement_input_index", self.movement_input_index),
            ("movement_output_index", self.movement_output_index),
        ):
            if not isinstance(index, Tensor) or index.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional tensor")
            if index.dtype != torch.long or index.device.type != "cpu":
                raise TypeError(f"{name} must be a CPU torch.long tensor")
        if self.movement_input_index.shape != self.movement_output_index.shape:
            raise ValueError("movement index tensors must have equal length")
        if self.movement_input_index.numel() == 0:
            raise ValueError("at least one allowed movement is required")
        if bool((self.movement_input_index < 0).any()) or bool(
            (self.movement_input_index >= self.input_count).any()
        ):
            raise ValueError("movement input index is out of bounds")
        if bool((self.movement_output_index < 0).any()) or bool(
            (self.movement_output_index >= self.output_count).any()
        ):
            raise ValueError("movement output index is out of bounds")

        pairs = list(
            zip(
                self.movement_input_index.tolist(),
                self.movement_output_index.tolist(),
                strict=True,
            )
        )
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate movement pair")
        if pairs != sorted(pairs):
            raise ValueError("movements must be in canonical lexicographic order")
        if set(self.movement_input_index.tolist()) != set(range(self.input_count)):
            raise ValueError("every declared input must have an allowed movement")
        if set(self.movement_output_index.tolist()) != set(range(self.output_count)):
            raise ValueError("every declared output must have an allowed movement")
        validate_turning_fractions(
            self.turning_fraction, self.movement_input_index, self.input_count
        )

    @property
    def input_count(self) -> int:
        return len(self.input_link_ids)

    @property
    def output_count(self) -> int:
        return len(self.output_link_ids)

    @property
    def movement_count(self) -> int:
        return self.movement_input_index.numel()


def build_movement_map(
    input_link_ids: Sequence[int],
    output_link_ids: Sequence[int],
    movements: Sequence[tuple[int, int, float]],
) -> NodeMovementMap:
    """Build a canonical map from ``(input_link_id, output_link_id, beta)`` rows."""

    inputs = tuple(input_link_ids)
    outputs = tuple(output_link_ids)
    _validate_link_ids("input_link_ids", inputs)
    _validate_link_ids("output_link_ids", outputs)
    input_local = {link_id: index for index, link_id in enumerate(inputs)}
    output_local = {link_id: index for index, link_id in enumerate(outputs)}
    local_rows: list[tuple[int, int, float]] = []
    for row in movements:
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError("each movement must be an (input ID, output ID, beta) tuple")
        input_id, output_id, beta = row
        if (
            isinstance(input_id, bool)
            or not isinstance(input_id, Integral)
            or isinstance(output_id, bool)
            or not isinstance(output_id, Integral)
        ):
            raise TypeError("movement link IDs must be integers")
        if input_id not in input_local or output_id not in output_local:
            raise ValueError("movement references an undeclared input or output link")
        if isinstance(beta, bool) or not isinstance(beta, Real):
            raise TypeError("turning fractions must be real numbers")
        local_rows.append((input_local[input_id], output_local[output_id], float(beta)))
    local_rows.sort(key=lambda value: (value[0], value[1]))
    return NodeMovementMap(
        input_link_ids=inputs,
        output_link_ids=outputs,
        movement_input_index=torch.tensor(
            [row[0] for row in local_rows], dtype=torch.long
        ),
        movement_output_index=torch.tensor(
            [row[1] for row in local_rows], dtype=torch.long
        ),
        turning_fraction=torch.tensor(
            [row[2] for row in local_rows], dtype=torch.float64
        ),
    )


def _validate_values(name: str, values: Tensor, expected_count: int) -> None:
    if not isinstance(values, Tensor) or values.ndim != 1 or values.shape[0] != expected_count:
        raise ValueError(f"{name} must have shape [{expected_count}]")
    if values.dtype != torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")


def project_oriented_demand(sending: Tensor, movement_map: NodeMovementMap) -> Tensor:
    """Return ``D_m = beta_m S_i`` on the sending tensor's device."""

    _validate_values("sending", sending, movement_map.input_count)
    indices = movement_map.movement_input_index.to(device=sending.device)
    beta = movement_map.turning_fraction.to(device=sending.device)
    return sending[indices] * beta


def _aggregate(
    movement_values: Tensor, indices: Tensor, group_count: int
) -> Tensor:
    device_indices = indices.to(device=movement_values.device)
    return torch.stack(
        tuple(
            movement_values[device_indices == group].sum()
            for group in range(group_count)
        )
    )


def aggregate_input_flow(
    movement_values: Tensor, movement_map: NodeMovementMap
) -> Tensor:
    """Aggregate movement mass to node-local input-link outflow."""

    _validate_values("movement_values", movement_values, movement_map.movement_count)
    return _aggregate(
        movement_values, movement_map.movement_input_index, movement_map.input_count
    )


def aggregate_output_flow(
    movement_values: Tensor, movement_map: NodeMovementMap
) -> Tensor:
    """Aggregate movement mass to node-local output-link inflow."""

    _validate_values("movement_values", movement_values, movement_map.movement_count)
    return _aggregate(
        movement_values, movement_map.movement_output_index, movement_map.output_count
    )
