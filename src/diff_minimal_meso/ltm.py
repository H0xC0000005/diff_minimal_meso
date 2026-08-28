"""Differentiable cumulative-count Link Transmission Model primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor

from .fd import LinkFDParameters


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class CumulativeLinkState:
    """Boundary-time cumulative counts with shape ``[H+1, A]``."""

    n_in: Tensor
    n_out: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.n_in, Tensor) or not isinstance(self.n_out, Tensor):
            raise TypeError("n_in and n_out must be torch tensors")
        if self.n_in.ndim != 2 or self.n_in.shape[0] < 1:
            raise ValueError("cumulative histories must have shape [H+1, A]")
        if self.n_out.shape != self.n_in.shape:
            raise ValueError("n_in and n_out must have identical shapes")
        if self.n_in.dtype != torch.float64 or self.n_out.dtype != torch.float64:
            raise TypeError("cumulative histories must use torch.float64")
        if self.n_in.device != self.n_out.device:
            raise ValueError("cumulative histories must be on the same device")
        if not bool(torch.isfinite(self.n_in).all()) or not bool(
            torch.isfinite(self.n_out).all()
        ):
            raise ValueError("cumulative histories must be finite")
        if bool((self.n_in < 0.0).any()) or bool((self.n_out < 0.0).any()):
            raise ValueError("cumulative histories must be nonnegative")
        if self.n_in.shape[0] > 1 and (
            bool((torch.diff(self.n_in, dim=0) < 0.0).any())
            or bool((torch.diff(self.n_out, dim=0) < 0.0).any())
        ):
            raise ValueError("cumulative histories must be nondecreasing")
        if bool((self.n_out > self.n_in).any()):
            raise ValueError("cumulative outflow cannot exceed cumulative inflow")

    @property
    def current_index(self) -> int:
        return self.n_in.shape[0] - 1

    @property
    def link_count(self) -> int:
        return self.n_in.shape[1]


@dataclass(frozen=True, slots=True)
class LinkStepResult:
    """Link boundary limits and active-constraint diagnostics for one interval."""

    sending: Tensor
    receiving: Tensor
    occupancy: Tensor
    sending_active: Tensor
    receiving_active: Tensor


def _query_history(
    history: Tensor, query_times_s: Sequence[float], dt_s: float, current_index: int
) -> Tensor:
    """Query each link's causal history with structural linear interpolation."""

    dt = _positive_finite("dt_s", dt_s)
    if len(query_times_s) != history.shape[1]:
        raise ValueError("one query time is required per link")
    if current_index < 0 or current_index >= history.shape[0]:
        raise IndexError("current_index is outside the supplied history")
    current_time = current_index * dt
    values: list[Tensor] = []
    for link, query in enumerate(query_times_s):
        q = float(query)
        if not math.isfinite(q):
            raise ValueError("history query times must be finite")
        if q > current_time:
            raise ValueError("history query cannot be after the current boundary")
        if q <= 0.0:
            values.append(history.new_zeros(()))
            continue
        ratio = q / dt
        left = math.floor(ratio)
        theta = ratio - left
        if theta == 0.0:
            values.append(history[left, link])
        else:
            values.append(
                (1.0 - theta) * history[left, link]
                + theta * history[left + 1, link]
            )
    return torch.stack(values)


def compute_link_step(
    state: CumulativeLinkState,
    parameters: Sequence[LinkFDParameters],
    dt_s: float,
) -> LinkStepResult:
    """Compute LTM sending/receiving masses for the next interval."""

    dt = _positive_finite("dt_s", dt_s)
    if len(parameters) != state.link_count:
        raise ValueError("one LinkFDParameters record is required per link")
    k = state.current_index
    end_time = (k + 1) * dt
    q_f = [end_time - p.free_flow_time_s for p in parameters]
    q_b = [end_time - p.backward_wave_time_s for p in parameters]
    delayed_in = _query_history(state.n_in, q_f, dt, k)
    delayed_out = _query_history(state.n_out, q_b, dt, k)
    capacity = state.n_in.new_tensor([p.capacity_veh_per_s for p in parameters])
    storage = state.n_in.new_tensor([p.jam_storage_veh for p in parameters])
    occupancy = state.n_in[k] - state.n_out[k]
    if bool((occupancy > storage).any()):
        raise ValueError("current occupancy exceeds jam storage")

    availability = torch.clamp_min(delayed_in - state.n_out[k], 0.0)
    storage_room = torch.clamp_min(delayed_out + storage - state.n_in[k], 0.0)
    step_capacity = capacity * dt
    sending = torch.minimum(availability, step_capacity)
    receiving = torch.minimum(storage_room, step_capacity)

    return LinkStepResult(
        sending=sending,
        receiving=receiving,
        occupancy=occupancy,
        sending_active=torch.stack(
            (sending == availability, sending == step_capacity), dim=1
        ),
        receiving_active=torch.stack(
            (receiving == storage_room, receiving == step_capacity), dim=1
        ),
    )


def advance_link_state(
    state: CumulativeLinkState,
    inflow: Tensor,
    outflow: Tensor,
    limits: LinkStepResult,
    parameters: Sequence[LinkFDParameters],
) -> CumulativeLinkState:
    """Append one accepted boundary-flow update without mutating history."""

    expected = (state.link_count,)
    if len(parameters) != state.link_count:
        raise ValueError("one LinkFDParameters record is required per link")
    if limits.sending.shape != expected or limits.receiving.shape != expected:
        raise ValueError("link limits must have shape [A]")
    for name, value in (("inflow", inflow), ("outflow", outflow)):
        if not isinstance(value, Tensor) or value.shape != expected:
            raise ValueError(f"{name} must have shape [A]")
        if value.dtype != torch.float64 or value.device != state.n_in.device:
            raise TypeError(f"{name} must match state dtype and device")
        if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
    if bool((inflow > limits.receiving).any()):
        raise ValueError("accepted inflow exceeds receiving mass")
    if bool((outflow > limits.sending).any()):
        raise ValueError("accepted outflow exceeds sending mass")
    next_in = state.n_in[-1] + inflow
    next_out = state.n_out[-1] + outflow
    storage = state.n_in.new_tensor([p.jam_storage_veh for p in parameters])
    if bool((next_out > next_in).any()) or bool((next_in - next_out > storage).any()):
        raise ValueError("accepted flows produce an inadmissible occupancy")
    return CumulativeLinkState(
        n_in=torch.cat((state.n_in, next_in.unsqueeze(0)), dim=0),
        n_out=torch.cat((state.n_out, next_out.unsqueeze(0)), dim=0),
    )
