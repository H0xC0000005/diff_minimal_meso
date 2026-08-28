"""Standard-library scalar oracle for the frozen cumulative-count LTM equations."""

from __future__ import annotations

import math


def history_value(history: list[float], query_s: float, dt_s: float, k: int) -> float:
    if query_s > k * dt_s:
        raise ValueError("noncausal query")
    if query_s <= 0.0:
        return 0.0
    ratio = query_s / dt_s
    left = math.floor(ratio)
    fraction = ratio - left
    if fraction == 0.0:
        return history[left]
    return (1.0 - fraction) * history[left] + fraction * history[left + 1]


def link_step(
    n_in: list[float],
    n_out: list[float],
    dt_s: float,
    capacity_veh_per_s: float,
    storage_veh: float,
    tau_f_s: float,
    tau_b_s: float,
) -> tuple[float, float, float]:
    k = len(n_in) - 1
    end_s = (k + 1) * dt_s
    availability = max(
        0.0, history_value(n_in, end_s - tau_f_s, dt_s, k) - n_out[k]
    )
    storage_room = max(
        0.0,
        history_value(n_out, end_s - tau_b_s, dt_s, k)
        + storage_veh
        - n_in[k],
    )
    step_capacity = capacity_veh_per_s * dt_s
    return (
        min(availability, step_capacity),
        min(storage_room, step_capacity),
        n_in[k] - n_out[k],
    )
