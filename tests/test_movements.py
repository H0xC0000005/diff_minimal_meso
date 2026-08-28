"""Component 0.3 fixed movement-map and accounting checks."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from diff_minimal_meso.movements import (
    NodeMovementMap,
    aggregate_input_flow,
    aggregate_output_flow,
    build_movement_map,
    project_oriented_demand,
)


def simple_map() -> NodeMovementMap:
    return build_movement_map(
        [10], [20, 21], [(10, 21, 0.75), (10, 20, 0.25)]
    )


def sparse_map() -> NodeMovementMap:
    return build_movement_map(
        [10, 11],
        [20, 21],
        [(11, 21, 1.0), (10, 21, 0.6), (10, 20, 0.4)],
    )


def test_builder_preserves_link_order_and_canonicalizes_movements() -> None:
    movement_map = sparse_map()
    assert movement_map.input_link_ids == (10, 11)
    assert movement_map.output_link_ids == (20, 21)
    assert movement_map.movement_input_index.tolist() == [0, 0, 1]
    assert movement_map.movement_output_index.tolist() == [0, 1, 1]
    assert movement_map.turning_fraction.tolist() == [0.4, 0.6, 1.0]


def test_approved_one_input_two_output_hand_projection() -> None:
    demand = project_oriented_demand(
        torch.tensor([2.5], dtype=torch.float64), simple_map()
    )
    torch.testing.assert_close(
        demand,
        torch.tensor([0.625, 1.875], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_zero_sending_retains_movement_shape() -> None:
    demand = project_oriented_demand(
        torch.zeros(2, dtype=torch.float64), sparse_map()
    )
    assert demand.shape == (3,)
    assert demand.tolist() == [0.0, 0.0, 0.0]


def test_projection_and_aggregation_conserve_each_input() -> None:
    movement_map = sparse_map()
    sending = torch.tensor([2.5, 1.25], dtype=torch.float64)
    demand = project_oriented_demand(sending, movement_map)
    input_outflow = aggregate_input_flow(demand, movement_map)
    output_inflow = aggregate_output_flow(demand, movement_map)
    torch.testing.assert_close(input_outflow, sending, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        output_inflow, torch.tensor([1.0, 2.75], dtype=torch.float64)
    )
    assert demand.sum().item() == pytest.approx(sending.sum().item())


def test_turning_sum_tolerance_does_not_renormalize() -> None:
    movement_map = build_movement_map(
        [10], [20, 21], [(10, 20, 0.5), (10, 21, 0.5000000005)]
    )
    assert movement_map.turning_fraction.sum().item() == 1.0000000005
    demand = project_oriented_demand(
        torch.tensor([2.0], dtype=torch.float64), movement_map
    )
    assert demand.sum().item() == pytest.approx(2.000000001, rel=0.0, abs=1e-15)


def test_projection_jacobian_is_fixed_sparse_matrix() -> None:
    movement_map = sparse_map()

    def project(sending: torch.Tensor) -> torch.Tensor:
        return project_oriented_demand(sending, movement_map)

    sending = torch.tensor([2.5, 1.25], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(project, sending)
    expected = torch.tensor([[0.4, 0.0], [0.6, 0.0], [0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(jacobian, expected, rtol=0.0, atol=0.0)


def test_aggregation_preserves_gradient() -> None:
    movement_map = sparse_map()
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    objective = (
        aggregate_input_flow(values, movement_map)
        * torch.tensor([2.0, 5.0], dtype=torch.float64)
    ).sum()
    (gradient,) = torch.autograd.grad(objective, values)
    assert gradient.tolist() == [2.0, 2.0, 5.0]


def test_mapped_relabeling_gives_pair_identical_demands() -> None:
    original = sparse_map()
    relabeled = build_movement_map(
        [101, 100],
        [201, 200],
        [(100, 200, 0.4), (100, 201, 0.6), (101, 201, 1.0)],
    )
    original_demand = project_oriented_demand(
        torch.tensor([2.5, 1.25], dtype=torch.float64), original
    )
    relabeled_demand = project_oriented_demand(
        torch.tensor([1.25, 2.5], dtype=torch.float64), relabeled
    )

    # Use explicit ID maps rather than relying on enumeration order.
    original_pairs = {
        (original.input_link_ids[i], original.output_link_ids[j]): original_demand[m].item()
        for m, (i, j) in enumerate(
            zip(original.movement_input_index.tolist(), original.movement_output_index.tolist(), strict=True)
        )
    }
    relabel_to_original = {100: 10, 101: 11, 200: 20, 201: 21}
    relabeled_pairs = {
        (relabel_to_original[relabeled.input_link_ids[i]], relabel_to_original[relabeled.output_link_ids[j]]): relabeled_demand[m].item()
        for m, (i, j) in enumerate(
            zip(relabeled.movement_input_index.tolist(), relabeled.movement_output_index.tolist(), strict=True)
        )
    }
    assert original_pairs == relabeled_pairs


@pytest.mark.parametrize(
    "movements",
    [
        [(10, 20, 0.5), (10, 21, 0.49)],
        [(10, 20, -0.1), (10, 21, 1.1)],
        [(10, 20, math.nan), (10, 21, math.nan)],
        [(10, 20, math.inf), (10, 21, 0.0)],
    ],
)
def test_rejects_invalid_turning_fractions(movements) -> None:
    with pytest.raises(ValueError):
        build_movement_map([10], [20, 21], movements)


def test_rejects_duplicate_movements() -> None:
    with pytest.raises(ValueError, match="duplicate movement"):
        build_movement_map([10], [20], [(10, 20, 0.5), (10, 20, 0.5)])


@pytest.mark.parametrize(
    ("inputs", "outputs", "movements", "message"),
    [
        ([10, 11], [20], [(10, 20, 1.0)], "every declared input"),
        ([10], [20, 21], [(10, 20, 1.0)], "every declared output"),
        ([10], [20], [(11, 20, 1.0)], "undeclared"),
        ([10, 10], [20], [(10, 20, 1.0)], "duplicate"),
    ],
)
def test_rejects_incomplete_or_invalid_declared_incidence(
    inputs, outputs, movements, message
) -> None:
    with pytest.raises(ValueError, match=message):
        build_movement_map(inputs, outputs, movements)


def test_direct_record_rejects_noncanonical_order() -> None:
    with pytest.raises(ValueError, match="canonical"):
        NodeMovementMap(
            (10,),
            (20, 21),
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([1, 0], dtype=torch.long),
            torch.tensor([0.75, 0.25], dtype=torch.float64),
        )


def test_rejects_bad_continuous_shapes_dtype_and_values() -> None:
    movement_map = simple_map()
    with pytest.raises(ValueError, match="shape"):
        project_oriented_demand(torch.zeros(2, dtype=torch.float64), movement_map)
    with pytest.raises(TypeError, match="float64"):
        project_oriented_demand(torch.zeros(1, dtype=torch.float32), movement_map)
    with pytest.raises(ValueError, match="nonnegative"):
        aggregate_output_flow(torch.tensor([-1.0, 1.0], dtype=torch.float64), movement_map)


def test_map_record_is_frozen() -> None:
    movement_map = simple_map()
    with pytest.raises(FrozenInstanceError):
        movement_map.input_link_ids = (99,)  # type: ignore[misc]
