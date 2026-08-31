"""Milestone 0 diagnostic objective and non-lossy traffic metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .simulation import RolloutResult, Scenario, mass_balance_residual


@dataclass(frozen=True, slots=True)
class TrafficMetrics:
    throughput: Tensor
    terminal_source_queue: Tensor
    terminal_link_occupancy: Tensor
    terminal_mass: Tensor
    maximum_absolute_conservation_residual: Tensor


def total_system_time(result: RolloutResult, scenario: Scenario) -> Tensor:
    """Return left-interval diagnostic TST in continuous veh-eq seconds."""

    if len(result.step_results) != scenario.horizon_steps:
        raise ValueError("rollout length does not match scenario horizon")
    if result.cumulative_link_history.n_in.shape[0] != scenario.horizon_steps + 1:
        raise ValueError("rollout link history must contain T+1 boundaries")
    source_available = torch.stack(
        tuple(step.source_available.sum() for step in result.step_results)
    )
    occupancy = (
        result.cumulative_link_history.n_in[:-1]
        - result.cumulative_link_history.n_out[:-1]
    ).sum(dim=1)
    return scenario.dt_s * (source_available + occupancy).sum()


def traffic_metrics(result: RolloutResult, scenario: Scenario) -> TrafficMetrics:
    """Return differentiable scalar accounting metrics without detaching."""

    terminal_occupancy = (
        result.cumulative_link_history.n_in[-1]
        - result.cumulative_link_history.n_out[-1]
    ).sum()
    terminal_queue = result.source_queue_history[-1].sum()
    residual = mass_balance_residual(result, scenario)
    return TrafficMetrics(
        throughput=result.cumulative_sink_history[-1].sum(),
        terminal_source_queue=terminal_queue,
        terminal_link_occupancy=terminal_occupancy,
        terminal_mass=terminal_queue + terminal_occupancy,
        maximum_absolute_conservation_residual=torch.max(torch.abs(residual)),
    )
