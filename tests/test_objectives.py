"""Component 0.7 diagnostic objective and metrics checks."""

from __future__ import annotations

import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.objectives import total_system_time, traffic_metrics
from diff_minimal_meso.simulation import NetworkDefinition, Scenario, rollout


def one_link_network() -> NetworkDefinition:
    parameters = LinkFDParameters(0.1, 1.0, 100.0, 2.0, 2.0)
    return NetworkDefinition(
        (parameters,), (), (), (),
        torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long),
    )


def test_literal_left_interval_total_system_time() -> None:
    scenario = Scenario(
        1.0, 3,
        torch.tensor([[0.125], [0.0], [0.0]], dtype=torch.float64),
    )
    result = rollout(one_link_network(), scenario)

    # Left-interval terms are 0.125, 0.125, 0.125 veh-eq.
    objective = total_system_time(result, scenario)
    assert objective.dtype == torch.float64
    assert objective.item() == 0.375


def test_metrics_retain_graph_and_match_accounting() -> None:
    arrivals = torch.tensor(
        [[0.125], [0.0], [0.0]], dtype=torch.float64, requires_grad=True
    )
    scenario = Scenario(1.0, 3, arrivals)
    result = rollout(one_link_network(), scenario)
    metrics = traffic_metrics(result, scenario)

    assert metrics.throughput.item() == 0.125
    assert metrics.terminal_source_queue.item() == 0.0
    assert metrics.terminal_link_occupancy.item() == 0.0
    assert metrics.terminal_mass.item() == 0.0
    assert metrics.maximum_absolute_conservation_residual.item() == 0.0
    total_system_time(result, scenario).backward()
    assert arrivals.grad is not None
    assert arrivals.grad[0, 0].item() == 3.0
