"""Component 1.6 ordered diagnostic schema and rendering checks."""

import json

import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.ledger import LedgerEntry, OrderedLedger
from diff_minimal_meso.ledger_diagnostics import (
    build_ledger_diagnostics,
    diagnostics_json,
    diagnostics_markdown,
)
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.route_simulation import MesoScenario, MesoState, initialize_meso_state
from diff_minimal_meso.routes import RouteDefinition, build_route_table
from diff_minimal_meso.simulation import NetworkDefinition


def fixture_state(order=(0, 1, 2)):
    fd = LinkFDParameters(0.1, 1.0, 100.0, 1.0, 1.0)
    movement_map = build_movement_map(
        [0], [1, 2], [(0, 1, 0.5), (0, 2, 0.5)]
    )
    node = NodeParameters(
        movement_map, torch.ones(1, dtype=torch.float64), NodeKind.ORDINARY_ORCA
    )
    network = NetworkDefinition(
        (fd, fd, fd), (movement_map,), (node,), (None,), torch.tensor([0]), torch.tensor([1, 2])
    )
    table = build_route_table(
        network,
        (
            RouteDefinition("R1", (0, 1)),
            RouteDefinition("R2", (0, 2)),
            RouteDefinition("R3", (0, 1)),
        ),
    )
    scenario = MesoScenario(
        1.0, 1, (((),),), table, source_entry_capacity=torch.ones((1, 1), dtype=torch.float64)
    )
    initial = initialize_meso_state(scenario)
    masses = (0.5, 0.25, 0.75)
    entries = tuple(
        LedgerEntry(
            route,
            0,
            torch.tensor(masses[route], dtype=torch.float64),
            torch.tensor(float(rank), dtype=torch.float64),
            torch.tensor(float(rank + 1), dtype=torch.float64),
        )
        for rank, route in enumerate(order)
    )
    state = MesoState(
        initial.macro_state,
        initial.source_ledgers,
        (OrderedLedger(0, entries, table), initial.link_ledgers[1], initial.link_ledgers[2]),
        torch.tensor([0.0, 0.125, 0.0], dtype=torch.float64),
    )
    return state, table


def test_literal_entry_view_and_separate_totals_round_trip() -> None:
    state, table = fixture_state()
    evidence = build_ledger_diagnostics(state, table)
    assert [row.route_id for row in evidence.ordered_entries] == ["R1", "R2", "R3"]
    assert [row.mass_veh for row in evidence.ordered_entries] == [0.5, 0.25, 0.75]
    assert [(row.movement_index, row.total_mass_veh) for row in evidence.movement_totals] == [(0, 1.25), (1, 0.25)]
    for row, entry in zip(evidence.ordered_entries, state.link_ledgers[0].entries, strict=True):
        assert row.route_index == entry.route_index
        assert row.route_position == entry.route_position
        assert row.mass_veh == float(entry.mass)


def test_same_totals_different_order_has_distinct_authoritative_view() -> None:
    first_state, table = fixture_state((0, 1, 2))
    second_state, _ = fixture_state((1, 0, 2))
    first = build_ledger_diagnostics(first_state, table)
    # The independently built table is structurally equal but intentionally not
    # the same validation authority, so rebuild the second state with `table`.
    second_state = MesoState(
        second_state.macro_state,
        tuple(OrderedLedger(x.owner_link_index, x.entries, table) for x in second_state.source_ledgers),
        tuple(OrderedLedger(x.owner_link_index, x.entries, table) for x in second_state.link_ledgers),
        second_state.completed_route_mass,
    )
    second = build_ledger_diagnostics(second_state, table)
    assert first.ordered_entries != second.ordered_entries
    assert first.movement_totals == second.movement_totals


def test_empty_terminal_source_and_completed_schemas_are_explicit() -> None:
    state, table = fixture_state()
    source_entry = LedgerEntry(
        0, 0, torch.tensor(0.1, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    )
    terminal_entry = LedgerEntry(
        0, 1, torch.tensor(0.2, dtype=torch.float64),
        torch.tensor(2.0, dtype=torch.float64), torch.tensor(3.0, dtype=torch.float64)
    )
    state = MesoState(
        state.macro_state,
        (OrderedLedger(0, (source_entry,), table),),
        (
            state.link_ledgers[0],
            OrderedLedger(1, (terminal_entry,), table),
            state.link_ledgers[2],
        ),
        state.completed_route_mass,
    )
    evidence = build_ledger_diagnostics(state, table)
    assert len(evidence.locations) == 4
    assert any(row.entry_count == 0 for row in evidence.locations)
    assert any(row.location_kind == "source_queue" for row in evidence.ordered_entries)
    assert any(row.terminal and row.movement_index is None for row in evidence.ordered_entries)
    assert len(evidence.completed_routes) == 3
    assert evidence.completed_routes[1].completed_mass_veh == 0.125
    data = diagnostics_json(evidence)
    assert set(data) == {"schema_version", "units", "locations", "ordered_entries", "movement_totals", "completed_routes"}


def test_json_and_markdown_are_deterministic_and_field_consistent() -> None:
    state, table = fixture_state()
    before = tuple(entry.mass.clone() for entry in state.link_ledgers[0].entries)
    evidence = build_ledger_diagnostics(state, table)
    first_json = diagnostics_json(evidence)
    second_json = diagnostics_json(build_ledger_diagnostics(state, table))
    assert json.dumps(first_json, sort_keys=True) == json.dumps(second_json, sort_keys=True)
    markdown = diagnostics_markdown(evidence)
    for row in first_json["ordered_entries"]:
        assert row["route_id"] in markdown
        assert str(row["mass_veh"]) in markdown
    for row in first_json["movement_totals"]:
        assert str(row["total_mass_veh"]) in markdown
    assert all(
        torch.equal(old, current.mass)
        for old, current in zip(before, state.link_ledgers[0].entries, strict=True)
    )
