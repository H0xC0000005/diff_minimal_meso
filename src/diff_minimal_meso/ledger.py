"""Immutable ordered route-mass ledger state for Milestone 1 Component 1.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import torch
from torch import Tensor

from .routes import RouteTable


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _scalar_float64(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != torch.Size([]):
        raise ValueError(f"{name} must be a scalar tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype torch.float64")
    torch._assert(torch.isfinite(value), f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One continuous mass block with an affine eligibility-time profile."""

    route_index: int
    route_position: int
    mass: Tensor
    eligible_front_s: Tensor
    eligible_tail_s: Tensor

    def __post_init__(self) -> None:
        route_index = _nonnegative_integer("route_index", self.route_index)
        route_position = _nonnegative_integer("route_position", self.route_position)
        mass = _scalar_float64("mass", self.mass)
        eligible_front = _scalar_float64(
            "eligible_front_s", self.eligible_front_s
        )
        eligible_tail = _scalar_float64("eligible_tail_s", self.eligible_tail_s)

        devices = {mass.device, eligible_front.device, eligible_tail.device}
        if len(devices) != 1:
            raise ValueError("entry tensors must be on one device")
        torch._assert(mass > 0.0, "mass must be positive")
        torch._assert(
            eligible_tail > eligible_front,
            "eligible_tail_s must be greater than eligible_front_s",
        )

        object.__setattr__(self, "route_index", route_index)
        object.__setattr__(self, "route_position", route_position)


@dataclass(frozen=True, slots=True)
class OrderedLedger:
    """An immutable ordered sequence of entries resident on one link.

    ``route_table`` is retained as structural validation context. It does not
    duplicate route data in entries and does not participate in tensor
    differentiation.
    """

    owner_link_index: int
    entries: tuple[LedgerEntry, ...]
    route_table: RouteTable = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.route_table, RouteTable):
            raise TypeError("route_table must be a RouteTable")
        owner = _nonnegative_integer("owner_link_index", self.owner_link_index)
        if owner >= self.route_table.network.link_count:
            raise IndexError("owner_link_index is out of range")
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple")
        if any(not isinstance(entry, LedgerEntry) for entry in self.entries):
            raise TypeError("entries must contain LedgerEntry records")

        entry_device: torch.device | None = None
        for entry in self.entries:
            try:
                current_link = self.route_table.current_link(
                    entry.route_index, entry.route_position
                )
            except (IndexError, TypeError) as error:
                raise ValueError("entry has invalid route metadata") from error
            if current_link != owner:
                raise ValueError("entry route position does not match ledger owner")
            if entry_device is None:
                entry_device = entry.mass.device
            elif entry.mass.device != entry_device:
                raise ValueError("all ledger entries must be on one device")

        object.__setattr__(self, "owner_link_index", owner)

    @property
    def device(self) -> torch.device | None:
        """Return the resident tensor device, or ``None`` for an empty ledger."""

        return self.entries[0].mass.device if self.entries else None

    def total_mass(self, *, like: Tensor | None = None) -> Tensor:
        """Return scalar resident mass without inventing an empty-ledger device."""

        if self.entries:
            if like is not None and like.device != self.entries[0].mass.device:
                raise ValueError("like tensor device does not match ledger device")
            return torch.stack(tuple(entry.mass for entry in self.entries)).sum()

        if like is None:
            raise ValueError("empty ledger total requires an explicit like tensor")
        _scalar_float64("like", like)
        return like.new_zeros(())


@dataclass(frozen=True, slots=True)
class MovementRun:
    """One adjacent run in the disposable node-facing movement shorthand."""

    movement_index: int
    mass: Tensor

    def __post_init__(self) -> None:
        movement_index = _nonnegative_integer("movement_index", self.movement_index)
        mass = _scalar_float64("mass", self.mass)
        torch._assert(mass > 0.0, "movement run mass must be positive")
        object.__setattr__(self, "movement_index", movement_index)


def _node_index(route_table: RouteTable, node_index: int) -> int:
    node = _nonnegative_integer("node_index", node_index)
    if node >= route_table.network.node_count:
        raise IndexError("node_index is out of range")
    return node


def project_movement_runs(
    ledger: OrderedLedger, node_index: int, route_table: RouteTable
) -> tuple[MovementRun, ...]:
    """Project resident route entries to adjacent movement/mass runs.

    The returned shorthand is disposable: it retains neither route metadata nor
    a backward pointer to the authoritative ledger.
    """

    if not isinstance(ledger, OrderedLedger):
        raise TypeError("ledger must be an OrderedLedger")
    if not isinstance(route_table, RouteTable):
        raise TypeError("route_table must be a RouteTable")
    if ledger.route_table is not route_table:
        raise ValueError("ledger and projection must use the same route table")
    node = _node_index(route_table, node_index)
    movement_map = route_table.network.node_movement_maps[node]

    transitions = []
    for entry in ledger.entries:
        try:
            transitions.append(
                route_table.transition(entry.route_index, entry.route_position)
            )
        except ValueError as error:
            raise ValueError("terminal ledger entry has no node movement") from error
    if ledger.owner_link_index not in movement_map.input_link_ids:
        raise ValueError("ledger owner is not an input of the selected node")

    runs: list[MovementRun] = []
    for entry, transition in zip(ledger.entries, transitions, strict=True):
        if transition.node_index != node:
            raise ValueError("ledger entry does not transition through selected node")
        if transition.movement_index >= movement_map.movement_count:
            raise ValueError("route transition references an undefined movement")

        if runs and runs[-1].movement_index == transition.movement_index:
            previous = runs[-1]
            runs[-1] = MovementRun(previous.movement_index, previous.mass + entry.mass)
        else:
            runs.append(MovementRun(transition.movement_index, entry.mass))
    return tuple(runs)


def movement_totals(
    runs: tuple[MovementRun, ...],
    movement_count: int,
    *,
    like: Tensor | None = None,
) -> Tensor:
    """Group disposable run mass by movement while preserving tensor graphs."""

    count = _nonnegative_integer("movement_count", movement_count)
    if count == 0:
        raise ValueError("movement_count must be positive")
    if not isinstance(runs, tuple):
        raise TypeError("runs must be a tuple")
    if any(not isinstance(run, MovementRun) for run in runs):
        raise TypeError("runs must contain MovementRun records")

    if runs:
        device = runs[0].mass.device
        if any(run.mass.device != device for run in runs):
            raise ValueError("all movement runs must be on one device")
        if like is not None and like.device != device:
            raise ValueError("like tensor device does not match run device")
        template = runs[0].mass
    else:
        if like is None:
            raise ValueError("empty movement totals require an explicit like tensor")
        template = _scalar_float64("like", like)

    totals = [template.new_zeros(()) for _ in range(count)]
    for run in runs:
        if run.movement_index >= count:
            raise IndexError("movement_index is out of range")
        totals[run.movement_index] = totals[run.movement_index] + run.mass
    return torch.stack(totals)


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    """Fixed-regime structural signature for one ledger selection."""

    selected_entry_indices: tuple[int, ...]
    split_entry_indices: tuple[int, ...]
    selected_movement_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LedgerSelectionResult:
    residual: OrderedLedger
    transferred: OrderedLedger
    evidence: SelectionEvidence


def _nonnegative_scalar(name: str, value: Tensor, device: torch.device) -> Tensor:
    result = _scalar_float64(name, value)
    if result.device != device:
        raise ValueError(f"{name} device does not match ledger device")
    torch._assert(result >= 0.0, f"{name} must be nonnegative")
    return result


def split_entry(entry: LedgerEntry, head_mass: Tensor) -> tuple[LedgerEntry, LedgerEntry]:
    """Split an entry at a strictly interior mass using affine rank timing."""

    if not isinstance(entry, LedgerEntry):
        raise TypeError("entry must be a LedgerEntry")
    amount = _nonnegative_scalar("head_mass", head_mass, entry.mass.device)
    torch._assert(amount > 0.0, "head_mass must be positive")
    torch._assert(amount < entry.mass, "head_mass must be less than entry mass")
    split_time = entry.eligible_front_s + (amount / entry.mass) * (
        entry.eligible_tail_s - entry.eligible_front_s
    )
    return (
        LedgerEntry(
            entry.route_index,
            entry.route_position,
            amount,
            entry.eligible_front_s,
            split_time,
        ),
        LedgerEntry(
            entry.route_index,
            entry.route_position,
            entry.mass - amount,
            split_time,
            entry.eligible_tail_s,
        ),
    )


def take_ledger_prefix(ledger: OrderedLedger, quota: Tensor) -> LedgerSelectionResult:
    """Take an exact FIFO mass prefix, omitting exact-zero public residuals."""

    if not isinstance(ledger, OrderedLedger):
        raise TypeError("ledger must be an OrderedLedger")
    if ledger.entries:
        device = ledger.entries[0].mass.device
    else:
        device = quota.device if isinstance(quota, Tensor) else torch.device("cpu")
    amount = _nonnegative_scalar("quota", quota, device)
    if not ledger.entries:
        torch._assert(amount == 0.0, "quota exceeds ledger mass")
        return LedgerSelectionResult(
            ledger,
            ledger,
            SelectionEvidence((), (), ()),
        )
    torch._assert(amount <= ledger.total_mass(), "quota exceeds ledger mass")
    if bool(amount == 0.0):
        return LedgerSelectionResult(
            ledger,
            OrderedLedger(ledger.owner_link_index, (), ledger.route_table),
            SelectionEvidence((), (), ()),
        )

    selected: list[LedgerEntry] = []
    residual: list[LedgerEntry] = []
    selected_indices: list[int] = []
    split_indices: list[int] = []
    remaining = amount
    for index, item in enumerate(ledger.entries):
        if bool(remaining == 0.0):
            residual.append(item)
        elif bool(remaining >= item.mass):
            selected.append(item)
            selected_indices.append(index)
            remaining = remaining - item.mass
        else:
            head, tail = split_entry(item, remaining)
            selected.append(head)
            residual.append(tail)
            selected_indices.append(index)
            split_indices.append(index)
            remaining = remaining - head.mass
    torch._assert(remaining == 0.0, "quota was not fully consumed")
    return LedgerSelectionResult(
        OrderedLedger(ledger.owner_link_index, tuple(residual), ledger.route_table),
        OrderedLedger(ledger.owner_link_index, tuple(selected), ledger.route_table),
        SelectionEvidence(tuple(selected_indices), tuple(split_indices), ()),
    )


def extract_movement_quotas(
    ledger: OrderedLedger,
    node_index: int,
    eligible_mass: Tensor,
    movement_quota: Tensor,
    route_table: RouteTable,
) -> LedgerSelectionResult:
    """Stably realize fixed movement quotas inside an exact eligible prefix."""

    if not ledger.entries:
        raise ValueError("movement extraction requires a nonempty ledger")
    if ledger.route_table is not route_table:
        raise ValueError("ledger and extraction must use the same route table")
    node = _node_index(route_table, node_index)
    movement_map = route_table.network.node_movement_maps[node]
    if ledger.owner_link_index not in movement_map.input_link_ids:
        raise ValueError("ledger owner is not an input of the selected node")
    eligible = _nonnegative_scalar(
        "eligible_mass", eligible_mass, ledger.entries[0].mass.device
    )
    if (
        not isinstance(movement_quota, Tensor)
        or movement_quota.shape != (movement_map.movement_count,)
    ):
        raise ValueError(
            f"movement_quota must have shape [{movement_map.movement_count}]"
        )
    if movement_quota.dtype != torch.float64:
        raise TypeError("movement_quota must have dtype torch.float64")
    if movement_quota.device != ledger.entries[0].mass.device:
        raise ValueError("movement_quota device does not match ledger device")
    torch._assert(torch.isfinite(movement_quota).all(), "movement_quota must be finite")
    torch._assert((movement_quota >= 0.0).all(), "movement_quota must be nonnegative")
    torch._assert(
        movement_quota.sum() <= eligible,
        "total movement quota exceeds eligible mass",
    )

    if bool((movement_quota == 0.0).all()):
        return LedgerSelectionResult(
            ledger,
            OrderedLedger(ledger.owner_link_index, (), route_table),
            SelectionEvidence((), (), ()),
        )

    tranche = take_ledger_prefix(ledger, eligible)
    remaining = list(movement_quota.unbind())
    transferred: list[LedgerEntry] = []
    tranche_residual: list[LedgerEntry] = []
    selected_indices: list[int] = []
    split_indices: list[int] = []
    selected_movements: list[int] = []

    for index, item in enumerate(tranche.transferred.entries):
        transition = route_table.transition(item.route_index, item.route_position)
        if transition.node_index != node:
            raise ValueError("eligible entry does not transition through selected node")
        movement = transition.movement_index
        available_quota = remaining[movement]
        if bool(available_quota == 0.0):
            tranche_residual.append(item)
        elif bool(available_quota >= item.mass):
            transferred.append(item)
            selected_indices.append(index)
            selected_movements.append(movement)
            remaining[movement] = available_quota - item.mass
        else:
            head, tail = split_entry(item, available_quota)
            transferred.append(head)
            tranche_residual.append(tail)
            selected_indices.append(index)
            split_indices.append(index)
            selected_movements.append(movement)
            remaining[movement] = available_quota - head.mass

    unresolved = torch.stack(remaining)
    torch._assert(
        (unresolved == 0.0).all(),
        "movement quota cannot be realized inside eligible tranche",
    )
    residual_entries = tuple(tranche_residual) + tranche.residual.entries
    return LedgerSelectionResult(
        OrderedLedger(ledger.owner_link_index, residual_entries, route_table),
        OrderedLedger(ledger.owner_link_index, tuple(transferred), route_table),
        SelectionEvidence(
            tuple(selected_indices),
            tuple(split_indices),
            tuple(selected_movements),
        ),
    )


@dataclass(frozen=True, slots=True)
class ProgressResult:
    progressed: tuple[LedgerEntry, ...]
    completed: tuple[LedgerEntry, ...]


def progress_transferred_entries(
    entries: tuple[LedgerEntry, ...], route_table: RouteTable
) -> ProgressResult:
    """Progress node transfers and route terminal transfers without overrun."""

    if not isinstance(entries, tuple) or any(
        not isinstance(entry, LedgerEntry) for entry in entries
    ):
        raise TypeError("entries must be a tuple of LedgerEntry records")
    progressed: list[LedgerEntry] = []
    completed: list[LedgerEntry] = []
    for item in entries:
        if route_table.is_terminal(item.route_index, item.route_position):
            completed.append(item)
        else:
            progressed.append(
                LedgerEntry(
                    item.route_index,
                    route_table.progressed_position(
                        item.route_index, item.route_position
                    ),
                    item.mass,
                    item.eligible_front_s,
                    item.eligible_tail_s,
                )
            )
    return ProgressResult(tuple(progressed), tuple(completed))


@dataclass(frozen=True, slots=True)
class DischargeTimingResult:
    entries: tuple[LedgerEntry, ...]
    actual_intervals_s: tuple[tuple[Tensor, Tensor], ...]


def assign_discharge_times(
    entries: tuple[LedgerEntry, ...],
    discharge_mass: Tensor,
    interval_front_s: Tensor,
    interval_tail_s: Tensor,
    outbound_free_flow_s: Tensor,
) -> DischargeTimingResult:
    """Map selected FIFO rank affinely to actual and outbound eligibility times."""

    if not entries:
        raise ValueError("discharge timing requires at least one entry")
    if any(not isinstance(item, LedgerEntry) for item in entries):
        raise TypeError("entries must contain LedgerEntry records")
    device = entries[0].mass.device
    flow = _nonnegative_scalar("discharge_mass", discharge_mass, device)
    front = _scalar_float64("interval_front_s", interval_front_s)
    tail = _scalar_float64("interval_tail_s", interval_tail_s)
    delay = _nonnegative_scalar("outbound_free_flow_s", outbound_free_flow_s, device)
    if front.device != device or tail.device != device:
        raise ValueError("discharge interval device does not match entry device")
    torch._assert(flow > 0.0, "discharge_mass must be positive")
    torch._assert(tail > front, "discharge interval must have positive duration")
    total = torch.stack(tuple(item.mass for item in entries)).sum()
    torch._assert(
        torch.isclose(total, flow, rtol=1.0e-10, atol=1.0e-12),
        "selected mass must equal discharge_mass",
    )

    cumulative = flow.new_zeros(())
    timed: list[LedgerEntry] = []
    actual: list[tuple[Tensor, Tensor]] = []
    duration = tail - front
    for item in entries:
        actual_front = front + (cumulative / flow) * duration
        cumulative = cumulative + item.mass
        actual_tail = front + (cumulative / flow) * duration
        actual.append((actual_front, actual_tail))
        timed.append(
            LedgerEntry(
                item.route_index,
                item.route_position,
                item.mass,
                actual_front + delay,
                actual_tail + delay,
            )
        )
    return DischargeTimingResult(tuple(timed), tuple(actual))


@dataclass(frozen=True, slots=True)
class OutboundPackage:
    input_priority: int
    entries: tuple[LedgerEntry, ...]

    def __post_init__(self) -> None:
        priority = _nonnegative_integer("input_priority", self.input_priority)
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("package entries must be a nonempty tuple")
        if any(not isinstance(item, LedgerEntry) for item in self.entries):
            raise TypeError("package entries must contain LedgerEntry records")
        object.__setattr__(self, "input_priority", priority)


@dataclass(frozen=True, slots=True)
class PackageOrderResult:
    ledger: OrderedLedger
    ordering_permutation: tuple[int, ...]


def append_ordered_packages(
    ledger: OrderedLedger, packages: tuple[OutboundPackage, ...]
) -> PackageOrderResult:
    """Append whole packages by leading time and exact-tie input priority."""

    if not isinstance(packages, tuple) or any(
        not isinstance(package, OutboundPackage) for package in packages
    ):
        raise TypeError("packages must be a tuple of OutboundPackage records")
    for package in packages:
        for item in package.entries:
            if (
                ledger.route_table.current_link(
                    item.route_index, item.route_position
                )
                != ledger.owner_link_index
            ):
                raise ValueError("package entry does not match output ledger owner")
    permutation = tuple(
        sorted(
            range(len(packages)),
            key=lambda index: (
                float(packages[index].entries[0].eligible_front_s),
                packages[index].input_priority,
            ),
        )
    )
    appended = ledger.entries + tuple(
        item for index in permutation for item in packages[index].entries
    )
    return PackageOrderResult(
        OrderedLedger(ledger.owner_link_index, appended, ledger.route_table),
        permutation,
    )


@dataclass(frozen=True, slots=True)
class NodeTransferResult:
    """Composed extraction, timing, and progression result for one input."""

    residual: OrderedLedger
    transferred: tuple[LedgerEntry, ...]
    actual_intervals_s: tuple[tuple[Tensor, Tensor], ...]
    evidence: SelectionEvidence


def execute_node_transfer(
    ledger: OrderedLedger,
    node_index: int,
    eligible_mass: Tensor,
    movement_quota: Tensor,
    interval_front_s: Tensor,
    interval_tail_s: Tensor,
    outbound_free_flow_s: Tensor,
    route_table: RouteTable,
) -> NodeTransferResult:
    """Compose fixed-quota replay, actual timing, and route progression."""

    selection = extract_movement_quotas(
        ledger, node_index, eligible_mass, movement_quota, route_table
    )
    if not selection.transferred.entries:
        return NodeTransferResult(
            selection.residual, (), (), selection.evidence
        )

    node = _node_index(route_table, node_index)
    movement_count = route_table.network.node_movement_maps[node].movement_count
    if (
        not isinstance(outbound_free_flow_s, Tensor)
        or outbound_free_flow_s.shape != (movement_count,)
    ):
        raise ValueError(
            f"outbound_free_flow_s must have shape [{movement_count}]"
        )
    if outbound_free_flow_s.dtype != torch.float64:
        raise TypeError("outbound_free_flow_s must have dtype torch.float64")
    device = selection.transferred.entries[0].mass.device
    if outbound_free_flow_s.device != device:
        raise ValueError("outbound_free_flow_s device does not match ledger device")
    torch._assert(
        torch.isfinite(outbound_free_flow_s).all(),
        "outbound_free_flow_s must be finite",
    )
    torch._assert(
        (outbound_free_flow_s >= 0.0).all(),
        "outbound_free_flow_s must be nonnegative",
    )

    flow = movement_quota.sum()
    zero_delay_timing = assign_discharge_times(
        selection.transferred.entries,
        flow,
        interval_front_s,
        interval_tail_s,
        flow.new_zeros(()),
    )
    delayed: list[LedgerEntry] = []
    for item, (actual_front, actual_tail) in zip(
        selection.transferred.entries,
        zero_delay_timing.actual_intervals_s,
        strict=True,
    ):
        transition = route_table.transition(item.route_index, item.route_position)
        delay = outbound_free_flow_s[transition.movement_index]
        delayed.append(
            LedgerEntry(
                item.route_index,
                item.route_position,
                item.mass,
                actual_front + delay,
                actual_tail + delay,
            )
        )
    progression = progress_transferred_entries(tuple(delayed), route_table)
    if progression.completed:
        raise ValueError("node transfer cannot complete a route before entering output")
    return NodeTransferResult(
        selection.residual,
        progression.progressed,
        zero_delay_timing.actual_intervals_s,
        selection.evidence,
    )


@dataclass(frozen=True, slots=True)
class MergeDiagnostics:
    adjacent_pairs_examined: int
    exact_merges: int
    before_count: int
    after_count: int
    safe_nonmerge_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MergeResult:
    ledger: OrderedLedger
    diagnostics: MergeDiagnostics


def _exact_merge_reason(left: LedgerEntry, right: LedgerEntry) -> str | None:
    if left.route_index != right.route_index:
        return "route_mismatch"
    if left.route_position != right.route_position:
        return "position_mismatch"
    if not bool(left.eligible_tail_s == right.eligible_front_s):
        return "time_gap"
    left_duration = left.eligible_tail_s - left.eligible_front_s
    right_duration = right.eligible_tail_s - right.eligible_front_s
    if not bool(left_duration * right.mass == right_duration * left.mass):
        return "rate_breakpoint"
    return None


def merge_adjacent_exact(ledger: OrderedLedger) -> MergeResult:
    """Explicitly merge only adjacent entries forming one exact affine segment."""

    if not isinstance(ledger, OrderedLedger):
        raise TypeError("ledger must be an OrderedLedger")
    if not ledger.entries:
        return MergeResult(ledger, MergeDiagnostics(0, 0, 0, 0, ()))
    output = [ledger.entries[0]]
    examined = 0
    merges = 0
    reasons: list[str] = []
    for right in ledger.entries[1:]:
        left = output[-1]
        examined += 1
        reason = _exact_merge_reason(left, right)
        if reason is not None:
            reasons.append(reason)
            output.append(right)
            continue
        output[-1] = LedgerEntry(
            left.route_index,
            left.route_position,
            left.mass + right.mass,
            left.eligible_front_s,
            right.eligible_tail_s,
        )
        merges += 1
    result = OrderedLedger(ledger.owner_link_index, tuple(output), ledger.route_table)
    return MergeResult(
        result,
        MergeDiagnostics(
            examined,
            merges,
            len(ledger.entries),
            len(result.entries),
            tuple(reasons),
        ),
    )
