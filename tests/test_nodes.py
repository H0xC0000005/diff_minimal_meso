"""Component 0.4 Branch-T and restricted continuum node checks."""

from __future__ import annotations

import pytest
import torch

from diff_minimal_meso.movements import build_movement_map, project_oriented_demand
from diff_minimal_meso.nodes import (
    ConstraintID,
    NodeKind,
    NodeParameters,
    solve_node,
)
from references.node_scalar_reference import ordinary_orca, restricted_signal


DT = 1.0


def make_case(
    beta: list[list[float]],
    capacities: list[float],
    kind: NodeKind = NodeKind.ORDINARY_ORCA,
):
    inputs = list(range(100, 100 + len(beta)))
    outputs = list(range(200, 200 + len(beta[0])))
    movements = [
        (inputs[i], outputs[j], value)
        for i, row in enumerate(beta)
        for j, value in enumerate(row)
        if value > 0.0
    ]
    movement_map = build_movement_map(inputs, outputs, movements)
    parameters = NodeParameters(
        movement_map,
        torch.tensor(capacities, dtype=torch.float64),
        kind,
    )
    return movement_map, parameters


def demand_vector(movement_map, input_demand: list[float]) -> torch.Tensor:
    return project_oriented_demand(
        torch.tensor(input_demand, dtype=torch.float64), movement_map
    )


@pytest.mark.parametrize(
    ("demand", "receiving", "expected"),
    [(3.0, 5.0, 3.0), (8.0, 5.0, 5.0)],
)
def test_siso_demand_and_supply_cases(demand, receiving, expected) -> None:
    movement_map, parameters = make_case([[1.0]], [10.0])
    result = solve_node(
        demand_vector(movement_map, [demand]),
        torch.tensor([receiving], dtype=torch.float64), parameters, DT,
    )
    assert result.movement_flow.item() == pytest.approx(expected)
    assert result.input_outflow.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("demand", "receiving", "expected"),
    [
        (10.0, [3.0, 9.0], [3.0, 3.0]),
        (4.0, [10.0, 10.0], [2.0, 2.0]),
    ],
)
def test_two_simo_cases(demand, receiving, expected) -> None:
    movement_map, parameters = make_case([[0.5, 0.5]], [10.0])
    result = solve_node(
        demand_vector(movement_map, [demand]),
        torch.tensor(receiving, dtype=torch.float64), parameters, DT,
    )
    assert result.movement_flow.tolist() == pytest.approx(expected)
    # Full FIFO: both turns retain beta=(1/2,1/2).
    assert result.movement_flow[0] == result.movement_flow[1]


@pytest.mark.parametrize(
    ("demand", "receiving", "expected"),
    [
        ([6.0, 4.0], 5.0, [3.0, 2.0]),
        ([6.0, 1.0], 6.0, [5.0, 1.0]),
    ],
)
def test_two_miso_cases(demand, receiving, expected) -> None:
    movement_map, parameters = make_case([[1.0], [1.0]], [6.0, 4.0])
    result = solve_node(
        demand_vector(movement_map, demand),
        torch.tensor([receiving], dtype=torch.float64), parameters, DT,
    )
    assert result.input_outflow.tolist() == pytest.approx(expected)
    assert result.output_inflow.item() <= receiving + 1e-12


def test_required_rational_2x2_orca_trace() -> None:
    beta = [[0.5, 0.5], [0.25, 0.75]]
    movement_map, parameters = make_case(beta, [6.0, 4.0])
    result = solve_node(
        demand_vector(movement_map, [6.0, 8.0 / 5.0]),
        torch.tensor([2.0, 4.0], dtype=torch.float64), parameters, DT,
    )
    expected = torch.tensor([8 / 5, 8 / 5, 2 / 5, 6 / 5], dtype=torch.float64)
    torch.testing.assert_close(result.movement_flow, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        result.output_inflow, torch.tensor([2.0, 14 / 5], dtype=torch.float64),
        rtol=1e-10, atol=1e-12,
    )
    assert result.selected_pivot_ids == (
        ConstraintID("receiving", 0), ConstraintID("receiving", 0)
    )


def test_second_mimo_case_is_demand_limited() -> None:
    beta = [[0.7, 0.3], [0.2, 0.8]]
    movement_map, parameters = make_case(beta, [5.0, 7.0])
    demand = [2.0, 3.0]
    result = solve_node(
        demand_vector(movement_map, demand),
        torch.tensor([20.0, 20.0], dtype=torch.float64), parameters, DT,
    )
    assert result.input_outflow.tolist() == pytest.approx(demand)
    assert result.movement_flow.tolist() == pytest.approx([1.4, 0.6, 0.6, 2.4])


def test_dynamic_zero_demand_and_zero_supply() -> None:
    movement_map, parameters = make_case([[1.0], [1.0]], [6.0, 4.0])
    result = solve_node(
        demand_vector(movement_map, [0.0, 4.0]),
        torch.tensor([0.0], dtype=torch.float64), parameters, DT,
    )
    assert result.movement_flow.tolist() == [0.0, 0.0]
    assert result.output_inflow.item() == 0.0


def test_exact_tie_reports_all_outputs_and_permutation_preserves_flow() -> None:
    movement_map, parameters = make_case([[0.5, 0.5]], [10.0])
    result = solve_node(
        demand_vector(movement_map, [10.0]),
        torch.tensor([2.5, 2.5], dtype=torch.float64), parameters, DT,
    )
    assert result.movement_flow.tolist() == [2.5, 2.5]
    assert result.tied_constraint_ids == (
        ConstraintID("receiving", 0), ConstraintID("receiving", 1)
    )
    assert result.selected_pivot_ids == (ConstraintID("receiving", 0),)

    permuted_map = build_movement_map(
        [100], [201, 200], [(100, 201, 0.5), (100, 200, 0.5)]
    )
    permuted_parameters = NodeParameters(
        permuted_map, torch.tensor([10.0], dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    permuted = solve_node(
        demand_vector(permuted_map, [10.0]),
        torch.tensor([2.5, 2.5], dtype=torch.float64), permuted_parameters, DT,
    )
    by_output = dict(zip(permuted_map.output_link_ids, permuted.output_inflow.tolist(), strict=True))
    assert by_output == {201: 2.5, 200: 2.5}


def test_near_tie_has_only_one_selected_constraint() -> None:
    movement_map, parameters = make_case([[0.5, 0.5]], [10.0])
    result = solve_node(
        demand_vector(movement_map, [10.0]),
        torch.tensor([2.5, 2.500001], dtype=torch.float64), parameters, DT,
    )
    assert result.tied_constraint_ids == ()
    assert result.selected_pivot_ids == (ConstraintID("receiving", 0),)


def test_new_tie_retains_the_unique_incumbent_pivot() -> None:
    beta = [[0.25, 0.75], [0.25, 0.75], [0.5, 0.5]]
    movement_map, parameters = make_case(beta, [2.0, 2.0, 2.0])
    result = solve_node(
        demand_vector(movement_map, [0.25, 2.0, 2.0]),
        torch.tensor([7 / 16, 13 / 16], dtype=torch.float64), parameters, DT,
    )
    # Output 1 is uniquely restrictive initially. After input 0 is fixed at
    # demand, both residual restriction levels are exactly 1/4; output 1 stays
    # the internal pivot even though output 0 is canonical-first.
    assert result.selected_pivot_ids == (
        ConstraintID("receiving", 1), ConstraintID("receiving", 1)
    )
    assert result.tied_constraint_ids == (
        ConstraintID("receiving", 0), ConstraintID("receiving", 1)
    )
    assert result.input_outflow.tolist() == [0.25, 0.5, 0.5]


@pytest.mark.parametrize(
    ("beta", "capacity", "demand", "receiving"),
    [
        ([[1.0]], [10.0], [8.0], [5.0]),
        ([[0.5, 0.5]], [10.0], [10.0], [3.0, 9.0]),
        ([[1.0], [1.0]], [6.0, 4.0], [6.0, 1.0], [6.0]),
        ([[0.5, 0.5], [0.25, 0.75]], [6.0, 4.0], [6.0, 1.6], [2.0, 4.0]),
    ],
)
def test_ordinary_production_matches_independent_scalar(
    beta, capacity, demand, receiving
) -> None:
    movement_map, parameters = make_case(beta, capacity)
    production = solve_node(
        demand_vector(movement_map, demand),
        torch.tensor(receiving, dtype=torch.float64), parameters, DT,
    )
    accepted, matrix, _, _ = ordinary_orca(beta, demand, capacity, receiving)
    # Compare input/output forms to avoid relying on dense-zero movement storage.
    assert production.input_outflow.tolist() == pytest.approx(accepted)
    expected_output = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(receiving))]
    assert production.output_inflow.tolist() == pytest.approx(expected_output)


def test_stable_orca_branch_reverse_mode_matches_central_difference() -> None:
    movement_map, parameters = make_case([[1.0], [1.0]], [6.0, 4.0])
    supply = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    result = solve_node(
        demand_vector(movement_map, [6.0, 4.0]), supply.reshape(1), parameters, DT
    )
    value = result.input_outflow[0]
    (gradient,) = torch.autograd.grad(value, supply)

    def evaluate(r: float) -> float:
        return solve_node(
            demand_vector(movement_map, [6.0, 4.0]),
            torch.tensor([r], dtype=torch.float64), parameters, DT,
        ).input_outflow[0].item()

    h = 1e-5
    finite_difference = (evaluate(5.0 + h) - evaluate(5.0 - h)) / (2 * h)
    assert gradient.item() == pytest.approx(0.6, rel=1e-10, abs=1e-12)
    assert finite_difference == pytest.approx(gradient.item(), rel=1e-9, abs=1e-10)


def test_restricted_han_rational_case_leaves_share_unused() -> None:
    beta = [[1.0], [1.0]]
    movement_map, parameters = make_case(
        beta, [10.0, 10.0], NodeKind.RESTRICTED_CONTINUUM_SIGNAL
    )
    exposure = torch.tensor([0.4, 0.6], dtype=torch.float64)
    saturation = torch.tensor([100.0, 100.0], dtype=torch.float64)
    result = solve_node(
        demand_vector(movement_map, [1.0, 10.0]),
        torch.tensor([10.0], dtype=torch.float64), parameters, DT,
        input_exposure=exposure, input_saturation_rate=saturation,
    )
    assert result.input_outflow.tolist() == [1.0, 6.0]
    assert result.output_inflow.item() == 7.0
    assert 10.0 - result.output_inflow.item() == 3.0
    accepted, matrix = restricted_signal(
        beta, [1.0, 10.0], [10.0], [0.4, 0.6], [100.0, 100.0], DT
    )
    assert result.input_outflow.tolist() == pytest.approx(accepted)
    assert result.movement_flow.tolist() == pytest.approx([row[0] for row in matrix])


def test_restricted_signal_service_bound_and_gradient() -> None:
    movement_map, parameters = make_case(
        [[1.0]], [10.0], NodeKind.RESTRICTED_CONTINUUM_SIGNAL
    )
    exposure = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    result = solve_node(
        demand_vector(movement_map, [10.0]),
        torch.tensor([100.0], dtype=torch.float64), parameters, 2.0,
        input_exposure=exposure,
        input_saturation_rate=torch.tensor([3.0], dtype=torch.float64),
    )
    (gradient,) = torch.autograd.grad(result.input_outflow.sum(), exposure)
    assert result.input_outflow.item() == pytest.approx(2.4)
    assert gradient.item() == pytest.approx(6.0)


def test_restricted_signal_rejects_h2_violation() -> None:
    movement_map, parameters = make_case(
        [[1.0], [1.0]], [10.0, 10.0], NodeKind.RESTRICTED_CONTINUUM_SIGNAL
    )
    with pytest.raises(ValueError, match="H2"):
        solve_node(
            demand_vector(movement_map, [5.0, 5.0]),
            torch.tensor([10.0], dtype=torch.float64), parameters, DT,
            input_exposure=torch.tensor([0.6, 0.6], dtype=torch.float64),
            input_saturation_rate=torch.tensor([10.0, 10.0], dtype=torch.float64),
        )


def test_rejects_nonpositive_structural_capacity_and_bad_fifo_demand() -> None:
    movement_map, _ = make_case([[0.5, 0.5]], [10.0])
    with pytest.raises(ValueError, match="strictly positive"):
        NodeParameters(
            movement_map, torch.tensor([0.0], dtype=torch.float64),
            NodeKind.ORDINARY_ORCA,
        )
    parameters = NodeParameters(
        movement_map, torch.tensor([10.0], dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        solve_node(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([5.0, 5.0], dtype=torch.float64), parameters, DT,
        )


def test_explicit_regime_dispatch_has_no_fallback() -> None:
    movement_map, signal_parameters = make_case(
        [[1.0]], [10.0], NodeKind.RESTRICTED_CONTINUUM_SIGNAL
    )
    with pytest.raises(ValueError, match="require exposure"):
        solve_node(
            demand_vector(movement_map, [1.0]),
            torch.tensor([1.0], dtype=torch.float64), signal_parameters, DT,
        )
