"""Component 0.5 continuum fixed-time signal checks."""

from __future__ import annotations

import math

import pytest
import torch

from diff_minimal_meso.movements import build_movement_map, project_oriented_demand
from diff_minimal_meso.nodes import NodeKind, NodeParameters, solve_node
from diff_minimal_meso.signals import (
    FixedPhasePlan,
    continuum_service,
    phase_exposure,
    two_phase_split,
    validate_green_split,
)


def shared_output_map():
    return build_movement_map(
        [10, 11], [20], [(10, 20, 1.0), (11, 20, 1.0)]
    )


def shared_output_plan(saturation=(10.0, 20.0)) -> FixedPhasePlan:
    return FixedPhasePlan(
        (0, 1),
        torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        torch.tensor(saturation, dtype=torch.float64),
    )


def test_zero_partial_and_full_green_literal_values() -> None:
    movement_map = shared_output_map()
    plan = shared_output_plan()
    service = continuum_service(
        torch.tensor([0.0, 1.0], dtype=torch.float64), plan, movement_map, 2.0
    )
    assert service.movement_exposure.tolist() == [0.0, 1.0]
    assert service.input_exposure.tolist() == [0.0, 1.0]
    assert service.input_service_mass.tolist() == [0.0, 40.0]

    partial = continuum_service(
        torch.tensor([0.4, 0.6], dtype=torch.float64), plan, movement_map, 2.0
    )
    assert partial.movement_exposure.tolist() == [0.4, 0.6]
    assert partial.input_service_mass.tolist() == pytest.approx([8.0, 24.0])


def test_unserved_movement_has_exact_zero_exposure() -> None:
    movement_map = build_movement_map([10], [20], [(10, 20, 1.0)])
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[False, False]], dtype=torch.bool),
        torch.tensor([10.0], dtype=torch.float64),
    )
    service = continuum_service(
        torch.tensor([0.25, 0.75], dtype=torch.float64), plan, movement_map, 1.0
    )
    assert service.movement_exposure.item() == 0.0
    assert service.input_service_mass.item() == 0.0


def test_movement_can_use_multiple_nonoverlapping_phase_shares() -> None:
    movement_map = build_movement_map([10], [20], [(10, 20, 1.0)])
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, True]], dtype=torch.bool),
        torch.tensor([10.0], dtype=torch.float64),
    )
    service = continuum_service(
        torch.tensor([0.4, 0.6], dtype=torch.float64), plan, movement_map, 1.0
    )
    assert service.movement_exposure.item() == 1.0
    assert service.input_service_mass.item() == 10.0


@pytest.mark.parametrize(
    "green",
    [
        torch.tensor([-0.1, 1.1], dtype=torch.float64),
        torch.tensor([0.2, 0.7], dtype=torch.float64),
        torch.tensor([math.nan, math.nan], dtype=torch.float64),
    ],
)
def test_invalid_physical_green_fails_without_repair(green) -> None:
    with pytest.raises(ValueError):
        validate_green_split(green, 2)


def test_green_shape_and_dtype_are_strict() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_green_split(torch.ones(3, dtype=torch.float64) / 3, 2)
    with pytest.raises(TypeError, match="float64"):
        validate_green_split(torch.tensor([0.5, 0.5], dtype=torch.float32), 2)


def test_overlap_above_one_is_rejected_even_inside_simplex_tolerance() -> None:
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, True]], dtype=torch.bool),
        torch.tensor([1.0], dtype=torch.float64),
    )
    green = torch.tensor([0.5, 0.5000000005], dtype=torch.float64)
    with pytest.raises(ValueError, match="overlapping"):
        phase_exposure(green, plan)


def test_h1_rejects_heterogeneous_movements_of_one_input() -> None:
    movement_map = build_movement_map(
        [10], [20, 21], [(10, 20, 0.5), (10, 21, 0.5)]
    )
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        torch.tensor([10.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="H1"):
        continuum_service(
            torch.tensor([0.4, 0.6], dtype=torch.float64), plan, movement_map, 1.0
        )


def test_h2_rejects_overallocated_shared_output() -> None:
    movement_map = shared_output_map()
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, False], [True, False]], dtype=torch.bool),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="H2"):
        continuum_service(
            torch.tensor([1.0, 0.0], dtype=torch.float64), plan, movement_map, 1.0
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_rejects_invalid_saturation_rate(bad: float) -> None:
    with pytest.raises(ValueError, match="saturation"):
        FixedPhasePlan(
            (0,), torch.tensor([[True]], dtype=torch.bool),
            torch.tensor([bad], dtype=torch.float64),
        )


def test_rejects_invalid_phase_structure_and_component_shapes() -> None:
    with pytest.raises(ValueError, match="unique"):
        FixedPhasePlan(
            (0, 0), torch.ones((1, 2), dtype=torch.bool),
            torch.ones(1, dtype=torch.float64),
        )
    with pytest.raises(TypeError, match="bool"):
        FixedPhasePlan(
            (0,), torch.ones((1, 1), dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
        )
    movement_map = shared_output_map()
    bad_plan = FixedPhasePlan(
        (0,), torch.ones((1, 1), dtype=torch.bool),
        torch.ones(2, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="movement dimension"):
        continuum_service(
            torch.ones(1, dtype=torch.float64), bad_plan, movement_map, 1.0
        )


def test_movement_exposure_jacobian_equals_permission_matrix() -> None:
    plan = FixedPhasePlan(
        (0, 1),
        torch.tensor([[True, False], [False, True], [True, True]], dtype=torch.bool),
        torch.ones(1, dtype=torch.float64),
    )
    green = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda g: phase_exposure(g, plan), green)
    expected = plan.movement_phase_matrix.to(dtype=torch.float64)
    torch.testing.assert_close(jacobian, expected, rtol=0.0, atol=0.0)


def test_fixed_total_directional_derivative_matches_analytic_service_map() -> None:
    movement_map = shared_output_map()
    plan = shared_output_plan(saturation=(3.0, 5.0))
    green = torch.tensor([0.4, 0.6], dtype=torch.float64)
    direction = torch.tensor([1.0, -1.0], dtype=torch.float64) / math.sqrt(2.0)

    def service_mass(g: torch.Tensor) -> torch.Tensor:
        return continuum_service(g, plan, movement_map, 2.0).input_service_mass

    value, tangent = torch.func.jvp(service_mass, (green,), (direction,))
    assert value.tolist() == pytest.approx([2.4, 6.0])
    expected = torch.tensor([6.0, -10.0], dtype=torch.float64) / math.sqrt(2.0)
    torch.testing.assert_close(tangent, expected, rtol=1e-10, atol=1e-12)


def test_two_phase_split_boundaries_and_gradient() -> None:
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    split = two_phase_split(theta)
    assert split.tolist() == [0.4, 0.6]
    (gradient,) = torch.autograd.grad(split[0] - split[1], theta)
    assert gradient.item() == 2.0
    assert two_phase_split(torch.tensor(0.0, dtype=torch.float64)).tolist() == [0.0, 1.0]
    assert two_phase_split(torch.tensor(1.0, dtype=torch.float64)).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        two_phase_split(torch.tensor(1.1, dtype=torch.float64))


def test_rational_han_signal_node_coupling_and_stable_gradient() -> None:
    movement_map = shared_output_map()
    plan = shared_output_plan(saturation=(100.0, 100.0))
    green = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    service = continuum_service(green, plan, movement_map, 1.0)
    parameters = NodeParameters(
        movement_map, torch.tensor([10.0, 10.0], dtype=torch.float64),
        NodeKind.RESTRICTED_CONTINUUM_SIGNAL,
    )
    demand = project_oriented_demand(
        torch.tensor([1.0, 10.0], dtype=torch.float64), movement_map
    )
    flow = solve_node(
        demand, torch.tensor([10.0], dtype=torch.float64), parameters, 1.0,
        input_exposure=service.input_exposure,
        input_saturation_rate=plan.input_saturation_rate,
    )
    assert flow.input_outflow.tolist() == [1.0, 6.0]
    assert flow.output_inflow.item() == 7.0
    assert 10.0 - flow.output_inflow.item() == 3.0
    (gradient,) = torch.autograd.grad(flow.output_inflow, green)
    assert gradient.tolist() == [0.0, 10.0]
