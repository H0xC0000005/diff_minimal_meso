"""Component 1.4 pure ledger operation and transaction checks."""

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.ledger import (
    LedgerEntry,
    OrderedLedger,
    OutboundPackage,
    append_ordered_packages,
    assign_discharge_times,
    execute_node_transfer,
    extract_movement_quotas,
    merge_adjacent_exact,
    progress_transferred_entries,
    split_entry,
    take_ledger_prefix,
)
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.routes import RouteDefinition, RouteTable, build_route_table
from diff_minimal_meso.simulation import NetworkDefinition
from tests.references.ledger_scalar_reference import (
    discharge_intervals,
    exact_mergeable,
    split_mass,
)


RTOL = 1.0e-10
ATOL = 1.0e-12


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


def operation_table() -> RouteTable:
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
            RouteDefinition("T1", (0, 1, 3)),
            RouteDefinition("L", (0, 2)),
            RouteDefinition("T2", (0, 1, 3)),
        ),
    )


def entry(route, mass, front, tail, *, position=0):
    return LedgerEntry(
        route, position, scalar(mass), scalar(front), scalar(tail)
    )


def assert_entry_values(item, expected):
    actual = torch.stack((item.mass, item.eligible_front_s, item.eligible_tail_s))
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), rtol=RTOL, atol=ATOL
    )


def test_affine_split_matches_independent_scalar_oracle() -> None:
    original = entry(0, 2.0, 10.0, 12.0)
    head, tail = split_entry(original, scalar(0.75))
    expected_head, expected_tail = split_mass(2.0, 10.0, 12.0, 0.75)
    assert_entry_values(head, expected_head)
    assert_entry_values(tail, expected_tail)
    assert head.eligible_tail_s is tail.eligible_front_s
    assert_entry_values(original, (2.0, 10.0, 12.0))


def test_prefix_zero_partial_full_and_invalid_quotas() -> None:
    table = operation_table()
    ledger = OrderedLedger(
        0, (entry(0, 0.5, 0.0, 1.0), entry(1, 0.25, 1.0, 2.0)), table
    )
    zero = take_ledger_prefix(ledger, scalar(0.0))
    assert zero.residual is ledger
    assert zero.transferred.entries == ()

    partial = take_ledger_prefix(ledger, scalar(0.6))
    torch.testing.assert_close(
        torch.stack(tuple(item.mass for item in partial.transferred.entries)),
        scalar(1.0).new_tensor([0.5, 0.1]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        partial.residual.entries[0].mass, scalar(0.15), rtol=RTOL, atol=ATOL
    )
    assert partial.evidence.split_entry_indices == (1,)

    full = take_ledger_prefix(ledger, scalar(0.75))
    assert len(full.transferred.entries) == 2
    assert full.residual.entries == ()
    assert all(float(item.mass) > 0.0 for item in full.transferred.entries)

    with pytest.raises(AssertionError, match="nonnegative"):
        take_ledger_prefix(ledger, scalar(-0.1))
    with pytest.raises(AssertionError, match="exceeds"):
        take_ledger_prefix(ledger, scalar(0.8))


def test_stable_movement_quota_extraction_literal_case() -> None:
    table = operation_table()
    ledger = OrderedLedger(
        0,
        (
            entry(0, 0.5, 0.0, 0.5),
            entry(1, 0.25, 0.5, 0.75),
            entry(2, 0.75, 0.75, 1.5),
        ),
        table,
    )
    result = extract_movement_quotas(
        ledger, 0, scalar(1.5), scalar(1.0).new_tensor([0.8, 0.2]), table
    )
    torch.testing.assert_close(
        torch.stack(tuple(item.mass for item in result.transferred.entries)),
        scalar(1.0).new_tensor([0.5, 0.2, 0.3]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert [item.route_index for item in result.transferred.entries] == [0, 1, 2]
    torch.testing.assert_close(
        torch.stack(tuple(item.mass for item in result.residual.entries)),
        scalar(1.0).new_tensor([0.05, 0.45]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert [item.route_index for item in result.residual.entries] == [1, 2]
    assert result.evidence.selected_entry_indices == (0, 1, 2)
    assert result.evidence.split_entry_indices == (1, 2)
    total = result.residual.total_mass() + result.transferred.total_mass()
    torch.testing.assert_close(total, ledger.total_mass(), rtol=RTOL, atol=ATOL)


def test_quota_cannot_search_beyond_eligible_tranche() -> None:
    table = operation_table()
    ledger = OrderedLedger(
        0,
        (
            entry(0, 0.5, 0.0, 0.5),
            entry(1, 0.5, 0.5, 1.0),
            entry(0, 0.5, 1.0, 1.5),
        ),
        table,
    )
    with pytest.raises(AssertionError, match="cannot be realized"):
        extract_movement_quotas(
            ledger, 0, scalar(1.0), scalar(1.0).new_tensor([0.6, 0.0]), table
        )


def test_progresses_crossing_entries_and_completes_terminal_entries() -> None:
    table = operation_table()
    result = progress_transferred_entries(
        (entry(0, 0.5, 0.0, 1.0), entry(1, 0.25, 1.0, 2.0)), table
    )
    assert [item.route_position for item in result.progressed] == [1, 1]
    assert result.completed == ()

    terminal = entry(1, 0.25, 2.0, 3.0, position=1)
    completion = progress_transferred_entries((terminal,), table)
    assert completion.progressed == ()
    assert completion.completed == (terminal,)


def test_actual_discharge_rank_and_outbound_delay_match_oracle() -> None:
    selected = (entry(0, 0.5, 0.0, 1.0), entry(2, 1.5, 1.0, 2.0))
    result = assign_discharge_times(
        selected, scalar(2.0), scalar(4.0), scalar(5.0), scalar(3.0)
    )
    expected = discharge_intervals([0.5, 1.5], 2.0, 4.0, 5.0)
    for actual, reference in zip(result.actual_intervals_s, expected, strict=True):
        torch.testing.assert_close(
            torch.stack(actual),
            torch.tensor(reference, dtype=torch.float64),
            rtol=RTOL,
            atol=ATOL,
        )
    assert_entry_values(result.entries[0], (0.5, 7.0, 7.25))
    assert_entry_values(result.entries[1], (1.5, 7.25, 8.0))


def test_composed_node_transfer_applies_movement_specific_delays() -> None:
    table = operation_table()
    ledger = OrderedLedger(
        0,
        (
            entry(0, 0.5, 0.0, 0.5),
            entry(1, 0.25, 0.5, 0.75),
            entry(2, 0.75, 0.75, 1.5),
        ),
        table,
    )
    result = execute_node_transfer(
        ledger,
        0,
        scalar(1.5),
        scalar(1.0).new_tensor([0.8, 0.2]),
        scalar(4.0),
        scalar(5.0),
        scalar(1.0).new_tensor([3.0, 4.0]),
        table,
    )
    assert [item.route_position for item in result.transferred] == [1, 1, 1]
    assert [item.route_index for item in result.transferred] == [0, 1, 2]
    assert_entry_values(result.transferred[0], (0.5, 7.0, 7.5))
    assert_entry_values(result.transferred[1], (0.2, 8.5, 8.7))
    assert_entry_values(result.transferred[2], (0.3, 7.7, 8.0))
    transferred_mass = torch.stack(tuple(item.mass for item in result.transferred)).sum()
    torch.testing.assert_close(
        result.residual.total_mass() + transferred_mass,
        ledger.total_mass(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_whole_packages_order_by_front_then_input_priority() -> None:
    table = operation_table()
    base = OrderedLedger(1, (), table)
    later_front = OutboundPackage(0, (entry(0, 1.0, 7.5, 8.5, position=1),))
    early_front = OutboundPackage(1, (entry(2, 1.0, 7.0, 8.0, position=1),))
    ordered = append_ordered_packages(base, (later_front, early_front))
    assert ordered.ordering_permutation == (1, 0)
    assert [item.route_index for item in ordered.ledger.entries] == [2, 0]

    tied_low = OutboundPackage(0, (entry(0, 0.5, 9.0, 9.5, position=1),))
    tied_high = OutboundPackage(1, (entry(2, 0.5, 9.0, 9.5, position=1),))
    tied = append_ordered_packages(base, (tied_high, tied_low))
    assert tied.ordering_permutation == (1, 0)


def test_exact_merge_and_safe_nonmerge_reasons() -> None:
    table = operation_table()
    positive = OrderedLedger(
        0, (entry(0, 1.0, 0.0, 1.0), entry(0, 2.0, 1.0, 3.0)), table
    )
    assert exact_mergeable((0, 0, 1.0, 0.0, 1.0), (0, 0, 2.0, 1.0, 3.0))
    merged = merge_adjacent_exact(positive)
    assert len(merged.ledger.entries) == 1
    assert_entry_values(merged.ledger.entries[0], (3.0, 0.0, 3.0))
    assert merged.diagnostics.exact_merges == 1
    assert merged.diagnostics.adjacent_pairs_examined == 1

    cases = (
        ((entry(0, 1.0, 0.0, 1.0), entry(1, 1.0, 1.0, 2.0)), "route_mismatch"),
        ((entry(0, 1.0, 0.0, 1.0), entry(0, 1.0, 2.0, 3.0)), "time_gap"),
        ((entry(0, 1.0, 0.0, 1.0), entry(0, 1.0, 1.0, 3.0)), "rate_breakpoint"),
    )
    for entries, reason in cases:
        result = merge_adjacent_exact(OrderedLedger(0, entries, table))
        assert result.ledger.entries == entries
        assert result.diagnostics.safe_nonmerge_reasons == (reason,)


def test_long_split_progress_append_merge_chain_conserves_and_does_not_mutate() -> None:
    table = operation_table()
    original = entry(0, 2.0, 7.0, 9.0)
    before = torch.stack((original.mass, original.eligible_front_s, original.eligible_tail_s)).clone()
    head, tail = split_entry(original, scalar(0.75))
    progressed = progress_transferred_entries((head, tail), table).progressed
    base = OrderedLedger(1, (), table)
    appended = append_ordered_packages(base, (OutboundPackage(0, progressed),)).ledger
    result = merge_adjacent_exact(appended)
    assert len(result.ledger.entries) == 1
    assert_entry_values(result.ledger.entries[0], (2.0, 7.0, 9.0))
    assert torch.equal(
        torch.stack((original.mass, original.eligible_front_s, original.eligible_tail_s)),
        before,
    )


def test_split_reverse_jvp_and_central_difference_agree() -> None:
    def split_time(values: torch.Tensor) -> torch.Tensor:
        item = LedgerEntry(0, 0, values[0], values[2], values[3])
        return split_entry(item, values[1])[0].eligible_tail_s

    values = torch.tensor(
        [2.0, 0.75, 10.0, 12.0], dtype=torch.float64, requires_grad=True
    )
    output = split_time(values)
    (reverse,) = torch.autograd.grad(output, values)
    direction = torch.tensor([0.2, -0.1, 0.3, 0.4], dtype=torch.float64)
    _, jvp = torch.func.jvp(split_time, (values.detach(),), (direction,))
    directional_reverse = reverse @ direction
    step = 1.0e-6
    finite_difference = (
        split_time(values.detach() + step * direction)
        - split_time(values.detach() - step * direction)
    ) / (2.0 * step)
    torch.testing.assert_close(jvp, directional_reverse, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(jvp, finite_difference, rtol=1e-6, atol=1e-8)


def test_fixed_signature_extraction_reverse_jvp_and_fd_agree() -> None:
    table = operation_table()

    def selected_objective(quota: torch.Tensor) -> torch.Tensor:
        ledger = OrderedLedger(
            0,
            (
                entry(0, 0.5, 0.0, 0.5),
                entry(1, 0.25, 0.5, 0.75),
                entry(2, 0.75, 0.75, 1.5),
            ),
            table,
        )
        result = extract_movement_quotas(
            ledger, 0, scalar(1.5), quota, table
        )
        selected = torch.stack(
            tuple(item.mass for item in result.transferred.entries)
        )
        return selected @ selected.new_tensor([1.0, 2.0, 3.0])

    quota = torch.tensor([0.8, 0.2], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.1, -0.05], dtype=torch.float64)
    (reverse,) = torch.autograd.grad(selected_objective(quota), quota)
    _, jvp = torch.func.jvp(
        selected_objective, (quota.detach(),), (direction,)
    )
    step = 1.0e-6
    finite_difference = (
        selected_objective(quota.detach() + step * direction)
        - selected_objective(quota.detach() - step * direction)
    ) / (2.0 * step)
    assert torch.equal(reverse, scalar(1.0).new_tensor([3.0, 2.0]))
    torch.testing.assert_close(jvp, reverse @ direction, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(jvp, finite_difference, rtol=1e-6, atol=1e-8)


def test_invalid_timing_and_package_requests_fail() -> None:
    selected = (entry(0, 0.5, 0.0, 1.0),)
    with pytest.raises(AssertionError, match="equal"):
        assign_discharge_times(
            selected, scalar(0.6), scalar(4.0), scalar(5.0), scalar(3.0)
        )
    with pytest.raises(AssertionError, match="positive duration"):
        assign_discharge_times(
            selected, scalar(0.5), scalar(5.0), scalar(5.0), scalar(3.0)
        )
    with pytest.raises(ValueError, match="nonempty"):
        OutboundPackage(0, ())
