"""Component 0.7 reverse/JVP/finite-difference harness checks."""

from __future__ import annotations

import math

import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.gradients import (
    directional_check,
    node_active_signature,
    physical_step_scale,
    two_phase_direction,
)
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.objectives import total_system_time
from diff_minimal_meso.signals import FixedPhasePlan
from diff_minimal_meso.simulation import (
    NetworkDefinition,
    Scenario,
    SignalControl,
    rollout,
)


def signal_network() -> NetworkDefinition:
    movement_map = build_movement_map(
        [0, 1], [2], [(0, 2, 1.0), (1, 2, 1.0)]
    )
    phase_plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    node = NodeParameters(
        movement_map, torch.tensor([10.0, 10.0], dtype=torch.float64),
        NodeKind.RESTRICTED_CONTINUUM_SIGNAL,
    )
    links = tuple(LinkFDParameters(0.1, 10.0, 100.0, 1.0, 1.0) for _ in range(3))
    return NetworkDefinition(
        links, (movement_map,), (node,), (phase_plan,),
        torch.tensor([0, 1], dtype=torch.long), torch.tensor([2], dtype=torch.long),
    )


def traffic_functions(arrival_pair=(1.0, 10.0)):
    scenario = Scenario(
        1.0, 4,
        torch.tensor(
            [arrival_pair, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            dtype=torch.float64,
        ),
    )
    network = signal_network()

    def run(green):
        return rollout(network, scenario, SignalControl((green,)))

    return (
        lambda green: total_system_time(run(green), scenario),
        lambda green: node_active_signature(run(green)),
    )


def test_linear_and_polynomial_toy_functions() -> None:
    point = torch.tensor([0.4, 0.6], dtype=torch.float64)
    direction = two_phase_direction()
    linear = directional_check(
        lambda value: torch.dot(
            value, torch.tensor([2.0, -3.0], dtype=torch.float64)
        ),
        point, direction,
    )
    expected = 5.0 / math.sqrt(2.0)
    assert linear.stable_scenario_passes
    assert linear.reverse_directional.item() == expected
    assert linear.jvp_directional.item() == expected
    assert all(row.passes for row in linear.rows)

    polynomial = directional_check(lambda value: torch.sum(value ** 3), point, direction)
    assert polynomial.reverse_jvp_agree
    assert polynomial.stable_adjacent_pass_count >= 3
    assert polynomial.stable_scenario_passes


def test_physical_step_scale_and_scan_grid_are_frozen() -> None:
    point = torch.tensor([0.4, 0.6], dtype=torch.float64)
    direction = two_phase_direction()
    scale = physical_step_scale(point, direction)
    torch.testing.assert_close(
        torch.tensor(scale), torch.tensor(0.4 * math.sqrt(2.0))
    )
    check = directional_check(lambda value: value[0] - value[1], point, direction)
    expected = tuple(scale * ratio for ratio in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6))
    assert tuple(row.step_size for row in check.rows) == expected


def test_stable_signalized_tst_reverse_jvp_and_fd_agree() -> None:
    function, regime = traffic_functions()
    check = directional_check(
        function, torch.tensor([0.4, 0.6], dtype=torch.float64),
        two_phase_direction(), regime_function=regime,
    )

    torch.testing.assert_close(
        check.reverse_directional,
        torch.tensor(10.0 / math.sqrt(2.0), dtype=torch.float64),
    )
    assert check.baseline_objective.item() == 37.0
    assert check.reverse_jvp_agree
    assert check.stable_adjacent_pass_count == 6
    assert check.stable_scenario_passes
    assert all(row.stable_regime and row.passes for row in check.rows)


def test_exact_event_boundary_is_exposed_not_misclassified_smooth() -> None:
    function, regime = traffic_functions(arrival_pair=(5.0, 0.0))
    check = directional_check(
        function, torch.tensor([0.5, 0.5], dtype=torch.float64),
        two_phase_direction(), regime_function=regime,
    )

    baseline_signature = check.rows[0].baseline_regime
    assert baseline_signature is not None
    assert any(record[3] for record in baseline_signature)  # tied constraint IDs
    assert any(not row.stable_regime for row in check.rows)
    assert not check.stable_scenario_passes
    assert all(
        row.objective_plus is not None and torch.isfinite(row.objective_plus)
        and row.objective_minus is not None and torch.isfinite(row.objective_minus)
        for row in check.rows
    )
