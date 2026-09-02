"""Component 1.1 fixed-route table and progression checks."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.routes import (
    RouteDefinition,
    RouteTable,
    RouteTransition,
    build_route_table,
    incoming_link_priority,
)
from diff_minimal_meso.simulation import NetworkDefinition


def link() -> LinkFDParameters:
    return LinkFDParameters(0.1, 1.0, 100.0, 1.0, 1.0)


def ordinary_node(inputs, outputs, movements):
    movement_map = build_movement_map(inputs, outputs, movements)
    parameters = NodeParameters(
        movement_map,
        torch.ones(len(inputs), dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    return movement_map, parameters


def chain_with_one_link_route() -> NetworkDefinition:
    first_map, first_node = ordinary_node([0], [1], [(0, 1, 1.0)])
    second_map, second_node = ordinary_node([1], [2], [(1, 2, 1.0)])
    return NetworkDefinition(
        (link(), link(), link(), link()),
        (first_map, second_map),
        (first_node, second_node),
        (None, None),
        torch.tensor([0, 3], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
    )


def valid_table() -> RouteTable:
    return build_route_table(
        chain_with_one_link_route(),
        (
            RouteDefinition("R-main", (0, 1, 2)),
            RouteDefinition("R-one", (3,)),
        ),
    )


def test_valid_chain_lookup_progression_and_terminal() -> None:
    table = valid_table()
    main = table.index_for_id("R-main")
    assert main == 0
    assert table.route_at(main).route_id == "R-main"
    assert [table.current_link(main, position) for position in range(3)] == [0, 1, 2]
    assert table.next_link(main, 0) == 1
    assert table.next_link(main, 1) == 2
    assert table.progressed_position(main, 0) == 1
    assert table.progressed_position(main, 1) == 2
    assert not table.is_terminal(main, 1)
    assert table.is_terminal(main, 2)

    first = table.transition(main, 0)
    second = table.transition(main, 1)
    assert (first.node_index, first.movement_index) == (0, 0)
    assert (second.node_index, second.movement_index) == (1, 0)


def test_valid_one_link_route_has_no_transition() -> None:
    table = valid_table()
    route = table.index_for_id("R-one")
    assert table.current_link(route, 0) == 3
    assert table.is_terminal(route, 0)
    with pytest.raises(ValueError, match="no next link"):
        table.next_link(route, 0)
    with pytest.raises(ValueError, match="no movement"):
        table.transition(route, 0)
    with pytest.raises(ValueError, match="cannot progress"):
        table.progressed_position(route, 0)


def test_external_ids_are_immutable_and_distinct_from_compact_indices() -> None:
    table = valid_table()
    assert table.index_for_id("R-main") == 0
    assert table.index_for_id("R-one") == 1
    assert dict(table.route_id_to_index) == {"R-main": 0, "R-one": 1}
    with pytest.raises(TypeError):
        table.route_id_to_index["R-new"] = 2  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        table.routes = ()  # type: ignore[misc]
    with pytest.raises(KeyError, match="unknown route"):
        table.index_for_id("missing")


@pytest.mark.parametrize(
    ("route", "message"),
    [
        (RouteDefinition("bad-link", (0, 4)), "outside"),
        (RouteDefinition("missing-move", (0, 2)), "exactly one"),
        (RouteDefinition("wrong-source", (1, 2)), "source-owned"),
        (RouteDefinition("wrong-sink", (0, 1)), "sink-owned"),
    ],
)
def test_rejects_invalid_network_route(route, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_route_table(chain_with_one_link_route(), (route,))


def test_rejects_invalid_route_declarations() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RouteDefinition("empty", ())
    with pytest.raises(ValueError, match="repeat"):
        RouteDefinition("cycle", (0, 1, 0))
    with pytest.raises(TypeError, match="integers"):
        RouteDefinition("bool-link", (True,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string"):
        RouteDefinition(7, (0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        RouteDefinition("", (0,))
    with pytest.raises(TypeError, match="tuple"):
        RouteDefinition("list", [0])  # type: ignore[arg-type]


def test_rejects_bad_table_and_duplicate_route_ids() -> None:
    network = chain_with_one_link_route()
    with pytest.raises(ValueError, match="at least one"):
        build_route_table(network, ())
    with pytest.raises(ValueError, match="unique"):
        build_route_table(
            network,
            (RouteDefinition("R", (0, 1, 2)), RouteDefinition("R", (3,))),
        )
    with pytest.raises(TypeError, match="tuple"):
        build_route_table(network, [RouteDefinition("R", (3,))])  # type: ignore[arg-type]


def test_transition_record_rejects_invalid_structural_indices() -> None:
    with pytest.raises(TypeError, match="node_index"):
        RouteTransition(True, 0, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="movement_index"):
        RouteTransition(0, -1, 0, 0)


@pytest.mark.parametrize(
    ("route_index", "route_position", "error"),
    [
        (True, 0, TypeError),
        (0, True, TypeError),
        (-1, 0, IndexError),
        (2, 0, IndexError),
        (0, -1, IndexError),
        (0, 3, IndexError),
    ],
)
def test_rejects_invalid_runtime_indices(route_index, route_position, error) -> None:
    with pytest.raises(error):
        valid_table().current_link(route_index, route_position)


def merge_network(input_order=(0, 1)) -> NetworkDefinition:
    movement_map, node = ordinary_node(
        list(input_order),
        [2],
        [(input_order[0], 2, 1.0), (input_order[1], 2, 1.0)],
    )
    return NetworkDefinition(
        (link(), link(), link()),
        (movement_map,),
        (node,),
        (None,),
        torch.tensor(list(input_order), dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
    )


def test_input_tuple_position_is_the_recorded_tie_priority() -> None:
    original = merge_network((0, 1))
    assert incoming_link_priority(original, 0, 0) == 0
    assert incoming_link_priority(original, 0, 1) == 1

    relabeled = merge_network((1, 0))
    assert incoming_link_priority(relabeled, 0, 1) == 0
    assert incoming_link_priority(relabeled, 0, 0) == 1

    with pytest.raises(ValueError, match="not an input"):
        incoming_link_priority(original, 0, 2)
    with pytest.raises(TypeError, match="integer"):
        incoming_link_priority(original, 0, True)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="node_index"):
        incoming_link_priority(original, 1, 0)


def test_consistent_link_relabeling_preserves_route_queries() -> None:
    original = valid_table()
    link_map = {0: 2, 1: 0, 2: 1, 3: 3}

    first_map, first_node = ordinary_node([2], [0], [(2, 0, 1.0)])
    second_map, second_node = ordinary_node([0], [1], [(0, 1, 1.0)])
    relabeled_network = NetworkDefinition(
        (link(), link(), link(), link()),
        (first_map, second_map),
        (first_node, second_node),
        (None, None),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([1, 3], dtype=torch.long),
    )
    relabeled = build_route_table(
        relabeled_network,
        (
            RouteDefinition("R-main", (2, 0, 1)),
            RouteDefinition("R-one", (3,)),
        ),
    )

    for route_id in ("R-main", "R-one"):
        original_index = original.index_for_id(route_id)
        relabeled_index = relabeled.index_for_id(route_id)
        route_length = len(original.route_at(original_index).link_ids)
        for position in range(route_length):
            assert relabeled.current_link(relabeled_index, position) == link_map[
                original.current_link(original_index, position)
            ]
            assert relabeled.is_terminal(
                relabeled_index, position
            ) == original.is_terminal(original_index, position)
            if not original.is_terminal(original_index, position):
                assert relabeled.next_link(relabeled_index, position) == link_map[
                    original.next_link(original_index, position)
                ]
                assert relabeled.progressed_position(
                    relabeled_index, position
                ) == original.progressed_position(original_index, position)
                original_transition = original.transition(original_index, position)
                relabeled_transition = relabeled.transition(relabeled_index, position)
                assert (
                    relabeled_transition.node_index,
                    relabeled_transition.movement_index,
                ) == (
                    original_transition.node_index,
                    original_transition.movement_index,
                )
