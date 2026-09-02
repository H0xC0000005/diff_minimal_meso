"""Read-only ordered-ledger diagnostics for Milestone 1 Component 1.6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .ledger import movement_totals, project_movement_runs
from .route_simulation import MesoState
from .routes import RouteTable


@dataclass(frozen=True, slots=True)
class OrderedEntryRow:
    location_kind: str
    location_index: int
    link_index: int
    ordinal: int
    route_id: str
    route_index: int
    route_position: int
    movement_index: int | None
    terminal: bool
    mass_veh: float
    eligible_front_s: float
    eligible_tail_s: float


@dataclass(frozen=True, slots=True)
class MovementTotalRow:
    node_index: int
    input_link_index: int
    movement_index: int
    total_mass_veh: float


@dataclass(frozen=True, slots=True)
class LocationRow:
    location_kind: str
    location_index: int
    link_index: int
    entry_count: int


@dataclass(frozen=True, slots=True)
class CompletedRouteRow:
    route_id: str
    route_index: int
    completed_mass_veh: float


@dataclass(frozen=True, slots=True)
class LedgerDiagnosticEvidence:
    schema_version: str
    units: tuple[tuple[str, str], ...]
    locations: tuple[LocationRow, ...]
    ordered_entries: tuple[OrderedEntryRow, ...]
    movement_totals: tuple[MovementTotalRow, ...]
    completed_routes: tuple[CompletedRouteRow, ...]


def _number(value) -> float:
    """Copy one diagnostic scalar to host without altering production tensors."""

    return float(value.detach().cpu())


def build_ledger_diagnostics(
    state: MesoState, route_table: RouteTable
) -> LedgerDiagnosticEvidence:
    """Build entry-preserving and separately grouped read-only evidence."""

    if not isinstance(state, MesoState):
        raise TypeError("state must be a MesoState")
    if not isinstance(route_table, RouteTable):
        raise TypeError("route_table must be a RouteTable")
    if any(ledger.route_table is not route_table for ledger in state.link_ledgers):
        raise ValueError("link ledgers must use the supplied route table")
    if any(ledger.route_table is not route_table for ledger in state.source_ledgers):
        raise ValueError("source ledgers must use the supplied route table")

    locations: list[LocationRow] = []
    rows: list[OrderedEntryRow] = []
    location_sets = (
        ("source_queue", state.source_ledgers),
        ("link", state.link_ledgers),
    )
    for kind, ledgers in location_sets:
        for location_index, ledger in enumerate(ledgers):
            locations.append(
                LocationRow(
                    kind, location_index, ledger.owner_link_index, len(ledger.entries)
                )
            )
            for ordinal, entry in enumerate(ledger.entries):
                terminal = route_table.is_terminal(
                    entry.route_index, entry.route_position
                )
                movement = (
                    None
                    if terminal
                    else route_table.transition(
                        entry.route_index, entry.route_position
                    ).movement_index
                )
                rows.append(
                    OrderedEntryRow(
                        kind,
                        location_index,
                        ledger.owner_link_index,
                        ordinal,
                        route_table.route_at(entry.route_index).route_id,
                        entry.route_index,
                        entry.route_position,
                        movement,
                        terminal,
                        _number(entry.mass),
                        _number(entry.eligible_front_s),
                        _number(entry.eligible_tail_s),
                    )
                )

    totals: list[MovementTotalRow] = []
    template = state.completed_route_mass.new_zeros(())
    for node, movement_map in enumerate(route_table.network.node_movement_maps):
        for input_link in movement_map.input_link_ids:
            ledger = state.link_ledgers[input_link]
            runs = project_movement_runs(ledger, node, route_table)
            values = movement_totals(
                runs, movement_map.movement_count, like=template
            )
            local_input = movement_map.input_link_ids.index(input_link)
            for movement in range(movement_map.movement_count):
                if int(movement_map.movement_input_index[movement]) == local_input:
                    totals.append(
                        MovementTotalRow(
                            node, input_link, movement, _number(values[movement])
                        )
                    )

    completed = tuple(
        CompletedRouteRow(
            route.route_id, route_index, _number(state.completed_route_mass[route_index])
        )
        for route_index, route in enumerate(route_table.routes)
    )
    return LedgerDiagnosticEvidence(
        "milestone-1-ledger-diagnostics-v1",
        (("mass", "veh-eq"), ("time", "s")),
        tuple(locations),
        tuple(rows),
        tuple(totals),
        completed,
    )


def diagnostics_json(evidence: LedgerDiagnosticEvidence) -> dict[str, Any]:
    """Return a deterministic JSON-compatible object from one evidence object."""

    return {
        "schema_version": evidence.schema_version,
        "units": dict(evidence.units),
        "locations": [asdict(row) for row in evidence.locations],
        "ordered_entries": [asdict(row) for row in evidence.ordered_entries],
        "movement_totals": [asdict(row) for row in evidence.movement_totals],
        "completed_routes": [asdict(row) for row in evidence.completed_routes],
    }


def diagnostics_markdown(evidence: LedgerDiagnosticEvidence) -> str:
    """Render the same evidence as a deterministic human-readable summary."""

    lines = [
        "# Ordered ledger diagnostics",
        "",
        f"Schema: `{evidence.schema_version}`; mass: `veh-eq`; time: `s`.",
        "",
        "## Locations",
        "",
        "| kind | index | link | entries |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.location_kind} | {row.location_index} | {row.link_index} | {row.entry_count} |"
        for row in evidence.locations
    )
    lines.extend(("", "## Ordered entries", ""))
    if not evidence.ordered_entries:
        lines.append("No entries.")
    else:
        lines.extend((
            "| kind | location | link | ordinal | route | route index | position | movement | terminal | mass (veh-eq) | front (s) | tail (s) |",
            "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
        ))
        lines.extend(
            f"| {r.location_kind} | {r.location_index} | {r.link_index} | {r.ordinal} | {r.route_id} | {r.route_index} | {r.route_position} | {r.movement_index if r.movement_index is not None else 'terminal'} | {str(r.terminal).lower()} | {r.mass_veh} | {r.eligible_front_s} | {r.eligible_tail_s} |"
            for r in evidence.ordered_entries
        )
    lines.extend(("", "## Movement totals", ""))
    if not evidence.movement_totals:
        lines.append("No node movements.")
    else:
        lines.extend((
            "| node | input link | movement | mass (veh-eq) |",
            "|---:|---:|---:|---:|",
        ))
        lines.extend(
            f"| {r.node_index} | {r.input_link_index} | {r.movement_index} | {r.total_mass_veh} |"
            for r in evidence.movement_totals
        )
    lines.extend(("", "## Completed routes", "", "| route | index | mass (veh-eq) |", "|---|---:|---:|"))
    lines.extend(
        f"| {r.route_id} | {r.route_index} | {r.completed_mass_veh} |"
        for r in evidence.completed_routes
    )
    return "\n".join(lines) + "\n"
