"""Component 1.3 route-to-current-movement projection checks."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.ledger import (
    LedgerEntry,
    MovementRun,
    OrderedLedger,
    movement_totals,
    project_movement_runs,
)
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.routes import RouteDefinition, RouteTable, build_route_table
from diff_minimal_meso.simulation import NetworkDefinition


def scalar(value: float, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)


def ordinary_node(inputs, outputs, movements):
    movement_map = build_movement_map(inputs, outputs, movements)
    parameters = NodeParameters(
        movement_map,
        torch.ones(len(inputs), dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    return movement_map, parameters


def projection_table() -> RouteTable:
    fd = LinkFDParameters(0.1, 1.0, 100.0, 1.0, 1.0)
    diverge_map, diverge = ordinary_node(
        [0], [1, 2], [(0, 1, 0.5), (0, 2, 0.5)]
    )
    downstream_map, downstream = ordinary_node([1], [3], [(1, 3, 1.0)])
    network = NetworkDefinition(
        (fd, fd, fd, fd),
        (diverge_map, downstream_map),
        (diverge, downstream),
        (None, None),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
    )
    return build_route_table(
        network,
        (
            RouteDefinition("R1", (0, 1, 3)),
            RouteDefinition("R2", (0, 2)),
            RouteDefinition("R3", (0, 1, 3)),
            RouteDefinition("R4", (0, 1, 3)),
        ),
    )


def entry(
    route_index: int,
    mass: torch.Tensor,
    front: float,
    tail: float,
    *,
    position: int = 0,
) -> LedgerEntry:
    return LedgerEntry(
        route_index, position, mass, scalar(front), scalar(tail)
    )


def literal_ledger(masses: tuple[torch.Tensor, ...] | None = None) -> OrderedLedger:
    table = projection_table()
    values = masses or (scalar(0.5), scalar(0.25), scalar(0.75), scalar(1.75))
    return OrderedLedger(
        0,
        (
            entry(0, values[0], 0.0, 1.0),
            entry(1, values[1], 1.0, 2.0),
            entry(2, values[2], 2.0, 3.0),
            entry(3, values[3], 3.0, 4.0),
        ),
        table,
    )


def test_literal_projection_run_length_encoding_and_totals() -> None:
    ledger = literal_ledger()
    runs = project_movement_runs(ledger, 0, ledger.route_table)

    assert [run.movement_index for run in runs] == [0, 1, 0]
    torch.testing.assert_close(
        torch.stack(tuple(run.mass for run in runs)),
        torch.tensor([0.5, 0.25, 2.5], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        movement_totals(runs, 2),
        torch.tensor([3.0, 0.25], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    assert torch.equal(sum((run.mass for run in runs), scalar(0.0)), ledger.total_mass())


def test_same_grouped_totals_can_retain_different_run_order() -> None:
    table = projection_table()
    alternate = OrderedLedger(
        0,
        (entry(1, scalar(0.25), 0.0, 1.0), entry(0, scalar(3.0), 1.0, 2.0)),
        table,
    )
    runs = project_movement_runs(alternate, 0, table)
    assert [run.movement_index for run in runs] == [1, 0]
    assert torch.equal(movement_totals(runs, 2), scalar(1.0).new_tensor([3.0, 0.25]))


def test_projection_is_reproducible_disposable_and_nonmutating() -> None:
    ledger = literal_ledger()
    entry_values = tuple(item.mass.detach().clone() for item in ledger.entries)
    first = project_movement_runs(ledger, 0, ledger.route_table)
    second = project_movement_runs(ledger, 0, ledger.route_table)

    assert tuple((run.movement_index, float(run.mass)) for run in first) == tuple(
        (run.movement_index, float(run.mass)) for run in second
    )
    assert all(not hasattr(run, "route_index") for run in first)
    assert all(not hasattr(run, "ledger") for run in first)
    assert all(
        torch.equal(before, after.mass)
        for before, after in zip(entry_values, ledger.entries, strict=True)
    )


def test_reverse_mode_matches_fixed_linear_projection() -> None:
    masses = tuple(scalar(value, requires_grad=True) for value in (0.5, 0.25, 0.75, 1.75))
    ledger = literal_ledger(masses)
    runs = project_movement_runs(ledger, 0, ledger.route_table)
    totals = movement_totals(runs, 2)
    objective = totals @ totals.new_tensor([2.0, 5.0])
    gradients = torch.autograd.grad(objective, masses)
    assert torch.equal(torch.stack(gradients), scalar(1.0).new_tensor([2.0, 5.0, 2.0, 2.0]))


def test_tensor_leaf_jvp_matches_hand_linear_map() -> None:
    table = projection_table()

    def project(*masses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ledger = OrderedLedger(
            0,
            tuple(
                entry(route, mass, float(route), float(route + 1))
                for route, mass in enumerate(masses)
            ),
            table,
        )
        runs = project_movement_runs(ledger, 0, table)
        return torch.stack(tuple(run.mass for run in runs)), movement_totals(runs, 2)

    _, tangent = torch.func.jvp(
        project,
        (scalar(0.5), scalar(0.25), scalar(0.75), scalar(1.75)),
        (scalar(0.1), scalar(0.2), scalar(0.3), scalar(0.4)),
    )
    assert torch.equal(tangent[0], scalar(1.0).new_tensor([0.1, 0.2, 0.7]))
    torch.testing.assert_close(
        tangent[1],
        scalar(1.0).new_tensor([0.8, 0.2]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_rejects_terminal_wrong_node_and_mismatched_route_table() -> None:
    table = projection_table()
    terminal = OrderedLedger(
        2, (entry(1, scalar(0.25), 0.0, 1.0, position=1),), table
    )
    with pytest.raises(ValueError, match="terminal"):
        project_movement_runs(terminal, 0, table)

    ledger = literal_ledger()
    with pytest.raises(ValueError, match="not an input"):
        project_movement_runs(ledger, 1, ledger.route_table)
    with pytest.raises(IndexError, match="node_index"):
        project_movement_runs(ledger, 2, ledger.route_table)
    with pytest.raises(TypeError, match="node_index"):
        project_movement_runs(ledger, True, ledger.route_table)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same route table"):
        project_movement_runs(ledger, 0, projection_table())


def test_empty_projection_and_totals_require_explicit_device_context() -> None:
    table = projection_table()
    empty = OrderedLedger(0, (), table)
    assert project_movement_runs(empty, 0, table) == ()
    with pytest.raises(ValueError, match="explicit like"):
        movement_totals((), 2)
    assert torch.equal(movement_totals((), 2, like=scalar(7.0)), torch.zeros(2, dtype=torch.float64))


def test_movement_run_and_totals_validation_and_immutability() -> None:
    run = MovementRun(0, scalar(0.5))
    with pytest.raises(FrozenInstanceError):
        run.mass = scalar(1.0)  # type: ignore[misc]
    with pytest.raises(TypeError, match="movement_index"):
        MovementRun(True, scalar(0.5))  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="positive"):
        MovementRun(0, scalar(0.0))
    with pytest.raises(IndexError, match="out of range"):
        movement_totals((MovementRun(2, scalar(0.5)),), 2)
    with pytest.raises(ValueError, match="positive"):
        movement_totals((), 0, like=scalar(0.0))
