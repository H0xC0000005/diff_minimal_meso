"""Immutable fixed-route tables for Milestone 1 Component 1.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

from .simulation import NetworkDefinition


def _structural_index(name: str, value: int, size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result >= size:
        raise IndexError(f"{name} is out of range")
    return result


@dataclass(frozen=True, slots=True)
class RouteDefinition:
    """One immutable, simple, connected route declaration."""

    route_id: str
    link_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str):
            raise TypeError("route_id must be a string")
        if not self.route_id:
            raise ValueError("route_id must not be empty")
        if not isinstance(self.link_ids, tuple):
            raise TypeError("link_ids must be a tuple")
        if not self.link_ids:
            raise ValueError("route must contain at least one link")
        for link_id in self.link_ids:
            if isinstance(link_id, bool) or not isinstance(link_id, Integral):
                raise TypeError("route link IDs must be integers")
        if len(set(self.link_ids)) != len(self.link_ids):
            raise ValueError("route must not repeat a link")


@dataclass(frozen=True, slots=True)
class RouteTransition:
    """Prevalidated node-local transition after one route position."""

    node_index: int
    movement_index: int
    input_local_index: int
    output_local_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("node_index", self.node_index),
            ("movement_index", self.movement_index),
            ("input_local_index", self.input_local_index),
            ("output_local_index", self.output_local_index),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class RouteTable:
    """Validated fixed routes and compact runtime lookup metadata."""

    network: NetworkDefinition = field(repr=False, compare=False)
    routes: tuple[RouteDefinition, ...]
    route_id_to_index: Mapping[str, int] = field(init=False)
    _transitions: tuple[tuple[RouteTransition, ...], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.network, NetworkDefinition):
            raise TypeError("network must be a NetworkDefinition")
        if not isinstance(self.routes, tuple):
            raise TypeError("routes must be a tuple")
        if not self.routes:
            raise ValueError("route table must contain at least one route")
        if any(not isinstance(route, RouteDefinition) for route in self.routes):
            raise TypeError("routes must contain RouteDefinition records")

        route_ids = [route.route_id for route in self.routes]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("route IDs must be unique")

        source_links = set(self.network.source_link_index.tolist())
        sink_links = set(self.network.sink_link_index.tolist())
        all_transitions: list[tuple[RouteTransition, ...]] = []
        for route in self.routes:
            for link_id in route.link_ids:
                if link_id < 0 or link_id >= self.network.link_count:
                    raise ValueError(
                        f"route {route.route_id!r} contains a link outside the network"
                    )
            if route.link_ids[0] not in source_links:
                raise ValueError(
                    f"route {route.route_id!r} first link is not source-owned"
                )
            if route.link_ids[-1] not in sink_links:
                raise ValueError(
                    f"route {route.route_id!r} final link is not sink-owned"
                )

            transitions = tuple(
                self._find_transition(route.route_id, input_link, output_link)
                for input_link, output_link in zip(
                    route.link_ids[:-1], route.link_ids[1:], strict=True
                )
            )
            all_transitions.append(transitions)

        object.__setattr__(
            self,
            "route_id_to_index",
            MappingProxyType(
                {route_id: index for index, route_id in enumerate(route_ids)}
            ),
        )
        object.__setattr__(self, "_transitions", tuple(all_transitions))

    def _find_transition(
        self, route_id: str, input_link: int, output_link: int
    ) -> RouteTransition:
        matches: list[RouteTransition] = []
        for node_index, movement_map in enumerate(self.network.node_movement_maps):
            try:
                input_local = movement_map.input_link_ids.index(input_link)
                output_local = movement_map.output_link_ids.index(output_link)
            except ValueError:
                continue
            for movement_index, (candidate_input, candidate_output) in enumerate(
                zip(
                    movement_map.movement_input_index.tolist(),
                    movement_map.movement_output_index.tolist(),
                    strict=True,
                )
            ):
                if candidate_input == input_local and candidate_output == output_local:
                    matches.append(
                        RouteTransition(
                            node_index,
                            movement_index,
                            input_local,
                            output_local,
                        )
                    )
        if len(matches) != 1:
            raise ValueError(
                f"route {route_id!r} adjacency ({input_link}, {output_link}) "
                "must match exactly one configured movement"
            )
        return matches[0]

    @property
    def route_count(self) -> int:
        return len(self.routes)

    def index_for_id(self, route_id: str) -> int:
        if not isinstance(route_id, str):
            raise TypeError("route_id must be a string")
        try:
            return self.route_id_to_index[route_id]
        except KeyError as error:
            raise KeyError(f"unknown route ID {route_id!r}") from error

    def route_at(self, route_index: int) -> RouteDefinition:
        return self.routes[
            _structural_index("route_index", route_index, self.route_count)
        ]

    def _validated_position(
        self, route_index: int, route_position: int
    ) -> tuple[int, int]:
        index = _structural_index("route_index", route_index, self.route_count)
        position = _structural_index(
            "route_position", route_position, len(self.routes[index].link_ids)
        )
        return index, position

    def current_link(self, route_index: int, route_position: int) -> int:
        index, position = self._validated_position(route_index, route_position)
        return self.routes[index].link_ids[position]

    def is_terminal(self, route_index: int, route_position: int) -> bool:
        index, position = self._validated_position(route_index, route_position)
        return position == len(self.routes[index].link_ids) - 1

    def next_link(self, route_index: int, route_position: int) -> int:
        index, position = self._validated_position(route_index, route_position)
        if position == len(self.routes[index].link_ids) - 1:
            raise ValueError("terminal route position has no next link")
        return self.routes[index].link_ids[position + 1]

    def transition(self, route_index: int, route_position: int) -> RouteTransition:
        index, position = self._validated_position(route_index, route_position)
        if position == len(self.routes[index].link_ids) - 1:
            raise ValueError("terminal route position has no movement transition")
        return self._transitions[index][position]

    def progressed_position(self, route_index: int, route_position: int) -> int:
        index, position = self._validated_position(route_index, route_position)
        if position == len(self.routes[index].link_ids) - 1:
            raise ValueError("cannot progress beyond a terminal route position")
        return position + 1


def build_route_table(
    network: NetworkDefinition, routes: tuple[RouteDefinition, ...]
) -> RouteTable:
    """Validate and build a compact immutable route table."""

    return RouteTable(network, routes)


def incoming_link_priority(
    network: NetworkDefinition, node_index: int, input_link_id: int
) -> int:
    """Return the existing node-local input tuple position used for exact ties."""

    if not isinstance(network, NetworkDefinition):
        raise TypeError("network must be a NetworkDefinition")
    node = _structural_index("node_index", node_index, network.node_count)
    if isinstance(input_link_id, bool) or not isinstance(input_link_id, Integral):
        raise TypeError("input_link_id must be an integer")
    try:
        return network.node_movement_maps[node].input_link_ids.index(int(input_link_id))
    except ValueError as error:
        raise ValueError("link is not an input of the selected node") from error
