"""Component 1.5 passive macro/ledger coupling checks."""

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.ledger import LedgerEntry, OrderedLedger
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.route_simulation import (
    MesoScenario,
    MesoState,
    SourceRouteBlock,
    initialize_meso_state,
    meso_rollout,
    meso_simulation_step,
)
from diff_minimal_meso.routes import RouteDefinition, build_route_table
from diff_minimal_meso.signals import FixedPhasePlan
from diff_minimal_meso.simulation import NetworkDefinition, SignalControl, rollout


def scalar(value: float, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)


def link(capacity=1.0, tau=1.0):
    return LinkFDParameters(0.1, capacity, 100.0, tau, tau)


def ordinary_node(inputs, outputs, movements):
    movement_map = build_movement_map(inputs, outputs, movements)
    node = NodeParameters(
        movement_map,
        torch.tensor([1.0] * len(inputs), dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    return movement_map, node


def one_link_setup():
    network = NetworkDefinition(
        (link(),), (), (), (), torch.tensor([0]), torch.tensor([0])
    )
    table = build_route_table(network, (RouteDefinition("R", (0,)),))
    return network, table


def chain_setup():
    movement_map, node = ordinary_node([0], [1], [(0, 1, 1.0)])
    network = NetworkDefinition(
        (link(), link()),
        (movement_map,),
        (node,),
        (None,),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    table = build_route_table(network, (RouteDefinition("R", (0, 1)),))
    return network, table


def block(route, mass, front, tail):
    return SourceRouteBlock(route, scalar(mass), scalar(front), scalar(tail))


def test_source_bucket_is_authoritative_and_preserves_block_order() -> None:
    _, table = one_link_setup()
    scenario = MesoScenario(
        1.0,
        2,
        ((((block(0, 0.125, -2.0, -1.0), block(0, 0.375, -1.0, 0.0))),), ((),)),
        table,
    )
    macro = scenario.to_macro_scenario()
    assert torch.equal(macro.arrivals, scalar(1.0).new_tensor([[0.5], [0.0]]))
    state = initialize_meso_state(scenario)
    state, result = meso_simulation_step(state, scenario, 0)
    source_transfer = next(item for item in result.transfers if item.kind == "source")
    assert [float(item.mass) for item in source_transfer.transferred] == [0.125, 0.375]
    assert float(state.link_ledgers[0].total_mass()) == 0.5


def test_blocked_source_uses_actual_admission_not_generation_timing() -> None:
    _, table = one_link_setup()
    scenario = MesoScenario(
        1.0,
        4,
        (((block(0, 0.125, -1.0, 0.0),),), ((),), ((),), ((),)),
        table,
        source_entry_capacity=scalar(1.0).new_tensor([[0.0], [1.0], [1.0], [1.0]]),
    )
    result = meso_rollout(scenario)
    admitted_entry = result.ledger_history[2].link_ledgers[0].entries[0]
    assert float(admitted_entry.eligible_front_s) == 2.0
    assert float(admitted_entry.eligible_tail_s) == 3.0
    assert float(result.ledger_history[1].source_ledgers[0].total_mass()) == 0.125
    transfer = next(
        item for item in result.step_results[1].transfers if item.kind == "source"
    )
    assert len(transfer.actual_intervals_s) == len(transfer.transferred) == 1
    assert [float(value) for value in transfer.actual_intervals_s[0]] == [1.0, 2.0]


def test_one_link_fractional_pulse_matches_macro_and_completes_route() -> None:
    network, table = one_link_setup()
    scenario = MesoScenario(
        1.0,
        3,
        (((block(0, 0.125, -1.0, 0.0),),), ((),), ((),)),
        table,
    )
    result = meso_rollout(scenario)
    independent = rollout(network, scenario.to_macro_scenario())
    assert torch.equal(
        result.macro_rollout_result.cumulative_link_history.n_in,
        independent.cumulative_link_history.n_in,
    )
    assert torch.equal(
        result.macro_rollout_result.cumulative_link_history.n_out,
        independent.cumulative_link_history.n_out,
    )
    assert float(result.ledger_history[-1].completed_route_mass[0]) == 0.125
    assert result.ledger_history[-1].link_ledgers[0].entries == ()


def test_two_link_chain_has_no_same_step_cascade_and_progresses_once() -> None:
    network, table = chain_setup()
    scenario = MesoScenario(
        1.0,
        4,
        (((block(0, 0.125, -1.0, 0.0),),), ((),), ((),), ((),)),
        table,
    )
    result = meso_rollout(scenario)
    independent = rollout(network, scenario.to_macro_scenario())
    assert torch.equal(
        result.macro_rollout_result.cumulative_link_history.n_in,
        independent.cumulative_link_history.n_in,
    )
    assert result.ledger_history[1].link_ledgers[0].entries[0].route_position == 0
    assert result.ledger_history[1].link_ledgers[1].entries == ()
    assert result.ledger_history[2].link_ledgers[0].entries == ()
    assert result.ledger_history[2].link_ledgers[1].entries[0].route_position == 1
    assert float(result.ledger_history[-1].completed_route_mass[0]) == 0.125
    node_transfer = next(
        item for item in result.step_results[1].transfers if item.kind == "node"
    )
    sink_transfer = next(
        item for item in result.step_results[2].transfers if item.kind == "sink"
    )
    assert [float(value) for value in node_transfer.actual_intervals_s[0]] == [1.0, 2.0]
    assert [float(value) for value in sink_transfer.actual_intervals_s[0]] == [2.0, 3.0]


def test_coupled_rollout_is_deterministic_and_link_relabeling_maps_exactly() -> None:
    _, table = chain_setup()
    arrivals = (((block(0, 0.125, -1.0, 0.0),),), ((),), ((),), ((),))
    scenario = MesoScenario(1.0, 4, arrivals, table)
    first = meso_rollout(scenario)
    repeated = meso_rollout(scenario)
    assert torch.equal(
        first.macro_rollout_result.cumulative_link_history.n_in,
        repeated.macro_rollout_result.cumulative_link_history.n_in,
    )
    assert torch.equal(
        first.ledger_history[-1].completed_route_mass,
        repeated.ledger_history[-1].completed_route_mass,
    )

    movement_map, node = ordinary_node([1], [0], [(1, 0, 1.0)])
    network = NetworkDefinition(
        (link(), link()),
        (movement_map,),
        (node,),
        (None,),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    mapped_table = build_route_table(
        network, (RouteDefinition("R", (1, 0)),)
    )
    mapped = meso_rollout(MesoScenario(1.0, 4, arrivals, mapped_table))
    assert torch.equal(
        mapped.macro_rollout_result.cumulative_link_history.n_in[:, [1, 0]],
        first.macro_rollout_result.cumulative_link_history.n_in,
    )
    assert torch.equal(
        mapped.ledger_history[-1].completed_route_mass,
        first.ledger_history[-1].completed_route_mass,
    )


def test_diverge_replays_macro_movement_quotas_and_conserves_routes() -> None:
    movement_map, node = ordinary_node(
        [0], [1, 2], [(0, 1, 0.5), (0, 2, 0.5)]
    )
    network = NetworkDefinition(
        (link(), link(), link()),
        (movement_map,),
        (node,),
        (None,),
        torch.tensor([0]),
        torch.tensor([1, 2]),
    )
    table = build_route_table(
        network, (RouteDefinition("T", (0, 1)), RouteDefinition("L", (0, 2)))
    )
    scenario = MesoScenario(
        1.0,
        4,
        (((block(0, 0.5, -2.0, -1.0), block(1, 0.5, -1.0, 0.0)),), ((),), ((),), ((),)),
        table,
    )
    result = meso_rollout(scenario)
    node_step = result.step_results[1]
    node_transfer = next(item for item in node_step.transfers if item.kind == "node")
    by_route = torch.zeros(2, dtype=torch.float64)
    for item in node_transfer.transferred:
        by_route[item.route_index] += item.mass
    assert torch.equal(by_route, node_step.macro_step_result.movement_flow[0])
    generated = scenario.to_macro_scenario().arrivals.sum()
    final = result.ledger_history[-1]
    resident = sum(
        (ledger.total_mass(like=generated) for ledger in final.link_ledgers),
        generated.new_zeros(()),
    )
    queued = sum(
        (ledger.total_mass(like=generated) for ledger in final.source_ledgers),
        generated.new_zeros(()),
    )
    torch.testing.assert_close(
        queued + resident + final.completed_route_mass.sum(),
        generated,
        rtol=1e-10,
        atol=1e-12,
    )


def test_invalid_bucket_route_source_and_state_mismatch_reject() -> None:
    _, table = one_link_setup()
    with pytest.raises(ValueError, match="outside"):
        MesoScenario(1.0, 1, (((block(0, 0.1, 0.0, 0.5),),),), table)

    scenario = MesoScenario(
        1.0, 2, (((block(0, 0.1, -1.0, 0.0),),), ((),)), table
    )
    state = initialize_meso_state(scenario)
    bad_macro, _ = meso_simulation_step(state, scenario, 0)
    corrupted = MesoState(
        bad_macro.macro_state,
        bad_macro.source_ledgers,
        (OrderedLedger(0, (), table),),
        bad_macro.completed_route_mass,
    )
    with pytest.raises(AssertionError, match="occupancy mismatch"):
        meso_simulation_step(corrupted, scenario, 1)


def test_boundary_route_conservation_rejects_offsetting_completion_swap() -> None:
    network = NetworkDefinition(
        (link(),), (), (), (), torch.tensor([0]), torch.tensor([0])
    )
    table = build_route_table(
        network, (RouteDefinition("A", (0,)), RouteDefinition("B", (0,)))
    )
    scenario = MesoScenario(
        1.0,
        3,
        (
            ((block(0, 0.2, -1.0, -0.5), block(1, 0.3, -0.5, 0.0)),),
            ((),),
            ((),),
        ),
        table,
    )
    state = initialize_meso_state(scenario)
    state, _ = meso_simulation_step(state, scenario, 0)
    state, _ = meso_simulation_step(state, scenario, 1)
    corrupted = MesoState(
        state.macro_state,
        state.source_ledgers,
        state.link_ledgers,
        scalar(1.0).new_tensor([0.25, 0.25]),
    )
    with pytest.raises(AssertionError, match="generated route mass"):
        meso_simulation_step(corrupted, scenario, 2)


def test_public_coupled_state_shape_mismatch_rejects_early() -> None:
    _, table = one_link_setup()
    scenario = MesoScenario(1.0, 1, (((block(0, 0.1, -1.0, 0.0),),),), table)
    state = initialize_meso_state(scenario)
    malformed_sources = MesoState(
        state.macro_state, (), state.link_ledgers, state.completed_route_mass
    )
    with pytest.raises(ValueError, match="source_ledgers"):
        meso_simulation_step(malformed_sources, scenario, 0)
    malformed_completed = MesoState(
        state.macro_state,
        state.source_ledgers,
        state.link_ledgers,
        scalar(0.0).new_zeros((2,)),
    )
    with pytest.raises(ValueError, match="completed_route_mass"):
        meso_simulation_step(malformed_completed, scenario, 0)


def test_wrong_source_and_wrong_sink_are_rejected_by_construction() -> None:
    network = NetworkDefinition(
        (link(), link()), (), (), (), torch.tensor([0, 1]), torch.tensor([0, 1])
    )
    table = build_route_table(
        network, (RouteDefinition("A", (0,)), RouteDefinition("B", (1,)))
    )
    with pytest.raises(ValueError, match="source link"):
        MesoScenario(
            1.0,
            1,
            (((), (block(0, 0.1, -1.0, 0.0),)),),
            table,
        )
    with pytest.raises(ValueError, match="ledger owner"):
        OrderedLedger(
            1,
            (
                LedgerEntry(0, 0, scalar(0.1), scalar(0.0), scalar(1.0)),
            ),
            table,
        )


def test_coupled_incompatible_movement_composition_rejects() -> None:
    movement_map, node = ordinary_node(
        [0], [1, 2], [(0, 1, 0.5), (0, 2, 0.5)]
    )
    network = NetworkDefinition(
        (link(), link(), link()),
        (movement_map,),
        (node,),
        (None,),
        torch.tensor([0]),
        torch.tensor([1, 2]),
    )
    table = build_route_table(
        network, (RouteDefinition("only-present", (0, 1)), RouteDefinition("absent", (0, 2)))
    )
    scenario = MesoScenario(
        1.0,
        2,
        (((block(0, 1.0, -1.0, 0.0),),), ((),)),
        table,
    )
    state = initialize_meso_state(scenario)
    state, _ = meso_simulation_step(state, scenario, 0)
    with pytest.raises(AssertionError, match="cannot be realized"):
        meso_simulation_step(state, scenario, 1)


def test_actual_transfer_interval_remains_graph_connected() -> None:
    network = NetworkDefinition(
        (link(capacity=2.0),), (), (), (), torch.tensor([0]), torch.tensor([0])
    )
    table = build_route_table(network, (RouteDefinition("R", (0,)),))
    first_mass = scalar(0.25, requires_grad=True)
    second_mass = scalar(0.75, requires_grad=True)
    scenario = MesoScenario(
        1.0,
        1,
        (((
            SourceRouteBlock(0, first_mass, scalar(-1.0), scalar(-0.5)),
            SourceRouteBlock(0, second_mass, scalar(-0.5), scalar(0.0)),
        ),),),
        table,
    )
    _, result = meso_simulation_step(initialize_meso_state(scenario), scenario, 0)
    transfer = next(item for item in result.transfers if item.kind == "source")
    split_actual_time = transfer.actual_intervals_s[0][1]
    assert split_actual_time.requires_grad
    gradients = torch.autograd.grad(split_actual_time, (first_mass, second_mass))
    torch.testing.assert_close(gradients[0], scalar(0.75), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gradients[1], scalar(-0.25), rtol=1e-10, atol=1e-12)


def test_actual_step_merge_telemetry_matches_installed_ledger() -> None:
    _, table = one_link_setup()
    scenario = MesoScenario(
        1.0,
        1,
        (((
            block(0, 0.25, -1.0, -0.5),
            block(0, 0.25, -0.5, 0.0),
        ),),),
        table,
    )
    state, result = meso_simulation_step(
        initialize_meso_state(scenario), scenario, 0
    )
    diagnostic = result.merge_diagnostics[0]
    assert diagnostic.before_count == 2
    assert diagnostic.after_count == 1
    assert diagnostic.exact_merges == 1
    assert len(state.link_ledgers[0].entries) == 1
    merged = state.link_ledgers[0].entries[0]
    torch.testing.assert_close(merged.mass, scalar(0.5), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        merged.eligible_front_s, scalar(1.0), rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        merged.eligible_tail_s, scalar(2.0), rtol=1e-10, atol=1e-12
    )


def test_signal_gradient_reverse_jvp_and_fd_match_unchanged_macro_path() -> None:
    movement_map = build_movement_map(
        [0, 1], [2], [(0, 2, 1.0), (1, 2, 1.0)]
    )
    plan = FixedPhasePlan(
        (0, 1),
        torch.tensor([[True, False], [False, True]]),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    node = NodeParameters(
        movement_map,
        torch.tensor([10.0, 10.0], dtype=torch.float64),
        NodeKind.RESTRICTED_CONTINUUM_SIGNAL,
    )
    network = NetworkDefinition(
        (link(10.0), link(10.0), link(10.0)),
        (movement_map,),
        (node,),
        (plan,),
        torch.tensor([0, 1]),
        torch.tensor([2]),
    )
    table = build_route_table(
        network, (RouteDefinition("A", (0, 2)), RouteDefinition("B", (1, 2)))
    )
    scenario = MesoScenario(
        1.0,
        3,
        (
            ((block(0, 5.0, -1.0, 0.0),), (block(1, 5.0, -1.0, 0.0),)),
            ((), ()),
            ((), ()),
        ),
        table,
    )

    def movement_a(green):
        result = meso_rollout(scenario, SignalControl((green,)))
        return result.step_results[1].macro_step_result.movement_flow[0][0]

    point = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    ordered = meso_rollout(scenario, SignalControl((point.detach(),)))
    assert [
        item.route_index for item in ordered.ledger_history[2].link_ledgers[2].entries
    ] == [0, 1]
    (reverse,) = torch.autograd.grad(movement_a(point), point)
    direction = torch.tensor([1.0, -1.0], dtype=torch.float64)
    _, jvp = torch.func.jvp(movement_a, (point.detach(),), (direction,))
    step = 1.0e-6
    finite_difference = (
        movement_a(point.detach() + step * direction)
        - movement_a(point.detach() - step * direction)
    ) / (2.0 * step)
    assert torch.equal(reverse, scalar(1.0).new_tensor([10.0, 0.0]))
    torch.testing.assert_close(jvp, reverse @ direction, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(jvp, finite_difference, rtol=1e-6, atol=1e-8)
