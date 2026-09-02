"""Component 1.2 immutable ordered-ledger state checks."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.ledger import LedgerEntry, OrderedLedger
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.routes import RouteDefinition, RouteTable, build_route_table
from diff_minimal_meso.simulation import NetworkDefinition


def scalar(value: float, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)


def route_table() -> RouteTable:
    fd = LinkFDParameters(0.1, 1.0, 100.0, 1.0, 1.0)
    movement_map = build_movement_map(
        [0], [1, 2], [(0, 1, 0.5), (0, 2, 0.5)]
    )
    node = NodeParameters(
        movement_map, torch.ones(1, dtype=torch.float64), NodeKind.ORDINARY_ORCA
    )
    network = NetworkDefinition(
        (fd, fd, fd),
        (movement_map,),
        (node,),
        (None,),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
    )
    return build_route_table(
        network,
        (RouteDefinition("R1", (0, 1)), RouteDefinition("R2", (0, 2))),
    )


def entry(
    route_index: int = 0,
    route_position: int = 0,
    mass: torch.Tensor | None = None,
    front: torch.Tensor | None = None,
    tail: torch.Tensor | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        route_index,
        route_position,
        scalar(0.125) if mass is None else mass,
        scalar(2.0) if front is None else front,
        scalar(2.5) if tail is None else tail,
    )


def test_preserves_declared_order_fields_and_fractional_total() -> None:
    table = route_table()
    first = entry()
    second = entry(1, mass=scalar(0.375), front=scalar(2.5), tail=scalar(3.5))
    ledger = OrderedLedger(0, (first, second), table)

    assert ledger.entries == (first, second)
    assert [item.route_index for item in ledger.entries] == [0, 1]
    assert torch.equal(ledger.total_mass(), scalar(0.5))
    assert ledger.device == torch.device("cpu")


def test_reverse_mode_retains_mass_and_timing_graphs() -> None:
    masses = (scalar(0.125, requires_grad=True), scalar(0.375, requires_grad=True))
    front = scalar(2.0, requires_grad=True)
    tail = scalar(2.5, requires_grad=True)
    ledger = OrderedLedger(
        0,
        (
            entry(mass=masses[0], front=front, tail=tail),
            entry(1, mass=masses[1], front=scalar(2.5), tail=scalar(3.5)),
        ),
        route_table(),
    )

    mass_grad = torch.autograd.grad(ledger.total_mass(), masses, retain_graph=True)
    assert torch.equal(torch.stack(mass_grad), torch.ones(2, dtype=torch.float64))

    timing_scalar = ledger.entries[0].mass * (tail - front)
    timing_grad = torch.autograd.grad(timing_scalar, (front, tail))
    expected = scalar(0.125) * scalar(1.0).new_tensor([-1.0, 1.0])
    assert torch.equal(torch.stack(timing_grad), expected)


def test_tensor_leaf_jvp_through_record_construction() -> None:
    table = route_table()

    def totals(
        first_mass: torch.Tensor,
        second_mass: torch.Tensor,
        front: torch.Tensor,
        tail: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ledger = OrderedLedger(
            0,
            (
                entry(mass=first_mass, front=front, tail=tail),
                entry(1, mass=second_mass, front=tail, tail=tail + 1.0),
            ),
            table,
        )
        split_compatible = first_mass * (tail - front)
        return ledger.total_mass(), split_compatible

    primal, tangent = torch.func.jvp(
        totals,
        (scalar(0.125), scalar(0.375), scalar(2.0), scalar(2.5)),
        (scalar(0.2), scalar(-0.1), scalar(0.3), scalar(0.7)),
    )
    assert torch.allclose(torch.stack(primal), scalar(1.0).new_tensor([0.5, 0.0625]))
    assert torch.allclose(torch.stack(tangent), scalar(1.0).new_tensor([0.1, 0.15]))


def test_empty_ledger_requires_explicit_tensor_context() -> None:
    ledger = OrderedLedger(0, (), route_table())
    assert ledger.device is None
    with pytest.raises(ValueError, match="explicit like"):
        ledger.total_mass()
    result = ledger.total_mass(like=scalar(7.0))
    assert result.shape == torch.Size([])
    assert result.dtype == torch.float64
    assert result.device == torch.device("cpu")
    assert result.item() == 0.0


@pytest.mark.parametrize("bad_mass", [0.0, -0.1, float("inf"), float("nan")])
def test_rejects_nonpositive_or_nonfinite_mass(bad_mass: float) -> None:
    with pytest.raises(AssertionError, match="mass"):
        entry(mass=scalar(bad_mass))


@pytest.mark.parametrize(
    ("front", "tail"),
    [(2.0, 2.0), (3.0, 2.0), (float("inf"), 3.0), (2.0, float("nan"))],
)
def test_rejects_invalid_eligibility(front: float, tail: float) -> None:
    with pytest.raises(AssertionError, match="eligible"):
        entry(front=scalar(front), tail=scalar(tail))


def test_rejects_bad_tensor_shape_dtype_and_device_consistency() -> None:
    with pytest.raises(ValueError, match="scalar"):
        entry(mass=torch.tensor([0.125], dtype=torch.float64))
    with pytest.raises(TypeError, match="float64"):
        entry(mass=torch.tensor(0.125, dtype=torch.float32))
    with pytest.raises(TypeError, match="tensor"):
        entry(mass=0.125)  # type: ignore[arg-type]
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="one device"):
            entry(tail=scalar(2.5).cuda())


@pytest.mark.parametrize(
    ("route_index", "route_position", "error"),
    [
        (True, 0, TypeError),
        (0, True, TypeError),
        (-1, 0, ValueError),
        (0, -1, ValueError),
    ],
)
def test_rejects_bad_entry_structural_metadata(
    route_index: int, route_position: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        entry(route_index, route_position)


def test_ledger_rejects_invalid_route_owner_and_container_metadata() -> None:
    table = route_table()
    with pytest.raises(ValueError, match="invalid route"):
        OrderedLedger(0, (entry(2),), table)
    with pytest.raises(ValueError, match="invalid route"):
        OrderedLedger(0, (entry(0, 2),), table)
    with pytest.raises(ValueError, match="does not match"):
        OrderedLedger(1, (entry(),), table)
    with pytest.raises(TypeError, match="integer"):
        OrderedLedger(True, (), table)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="out of range"):
        OrderedLedger(3, (), table)
    with pytest.raises(TypeError, match="tuple"):
        OrderedLedger(0, [entry()], table)  # type: ignore[arg-type]


def test_records_are_frozen_and_operations_leave_original_values_unchanged() -> None:
    original = entry()
    ledger = OrderedLedger(0, (original,), route_table())
    before = tuple(
        tensor.detach().clone()
        for tensor in (
            original.mass,
            original.eligible_front_s,
            original.eligible_tail_s,
        )
    )

    with pytest.raises(FrozenInstanceError):
        original.mass = scalar(1.0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ledger.entries = ()  # type: ignore[misc]
    _ = ledger.total_mass() + 1.0

    after = (original.mass, original.eligible_front_s, original.eligible_tail_s)
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))
