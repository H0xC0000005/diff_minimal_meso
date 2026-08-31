"""Milestone 0 end-to-end macro-stack integration checks."""

from __future__ import annotations

import math

import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.gradients import directional_check, node_active_signature, two_phase_direction
from diff_minimal_meso.movements import (
    aggregate_input_flow,
    aggregate_output_flow,
    build_movement_map,
)
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.objectives import total_system_time, traffic_metrics
from diff_minimal_meso.signals import FixedPhasePlan, continuum_service
from diff_minimal_meso.simulation import (
    NetworkDefinition,
    Scenario,
    SignalControl,
    mass_balance_residual,
    rollout,
)


ATOL = 1.0e-12


def link(capacity: float = 10.0) -> LinkFDParameters:
    return LinkFDParameters(0.1, capacity, 100.0, 1.0, 1.0)


def ordinary_chain(*, permuted: bool = False) -> NetworkDefinition:
    # Physical chain A -> B -> C. The alternate assembly maps old IDs to [2, 0, 1]
    # and reverses node-record order without changing physical behavior.
    ids = (2, 0, 1) if permuted else (0, 1, 2)
    first = build_movement_map([ids[0]], [ids[1]], [(ids[0], ids[1], 1.0)])
    second = build_movement_map([ids[1]], [ids[2]], [(ids[1], ids[2], 1.0)])
    first_node = NodeParameters(
        first, torch.tensor([10.0], dtype=torch.float64), NodeKind.ORDINARY_ORCA
    )
    second_node = NodeParameters(
        second, torch.tensor([10.0], dtype=torch.float64), NodeKind.ORDINARY_ORCA
    )
    maps = (second, first) if permuted else (first, second)
    nodes = (second_node, first_node) if permuted else (first_node, second_node)
    return NetworkDefinition(
        (link(), link(), link()), maps, nodes, (None, None),
        torch.tensor([ids[0]], dtype=torch.long),
        torch.tensor([ids[2]], dtype=torch.long),
    )


def signalized_merge() -> tuple[NetworkDefinition, FixedPhasePlan]:
    movement_map = build_movement_map(
        [0, 1], [2], [(0, 2, 1.0), (1, 2, 1.0)]
    )
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    node = NodeParameters(
        movement_map, torch.tensor([10.0, 10.0], dtype=torch.float64),
        NodeKind.RESTRICTED_CONTINUUM_SIGNAL,
    )
    network = NetworkDefinition(
        (link(), link(), link()), (movement_map,), (node,), (plan,),
        torch.tensor([0, 1], dtype=torch.long), torch.tensor([2], dtype=torch.long),
    )
    return network, plan


def assert_rollout_feasible(result, scenario, network) -> None:
    torch.testing.assert_close(
        mass_balance_residual(result, scenario),
        torch.zeros(scenario.horizon_steps + 1, dtype=torch.float64),
        rtol=0.0, atol=ATOL,
    )
    occupancy = (
        result.cumulative_link_history.n_in - result.cumulative_link_history.n_out
    )
    storage = torch.tensor(
        [parameters.jam_storage_veh for parameters in network.link_parameters],
        dtype=torch.float64,
    )
    assert bool((occupancy >= 0.0).all())
    assert bool((occupancy <= storage + ATOL).all())
    for step_index, step in enumerate(result.step_results):
        assert bool((step.link_inflow >= 0.0).all())
        assert bool((step.link_outflow >= 0.0).all())
        assert bool((step.link_inflow <= step.receiving + ATOL).all())
        assert bool((step.link_outflow <= step.sending + ATOL).all())
        torch.testing.assert_close(
            result.cumulative_link_history.n_in[step_index + 1]
            - result.cumulative_link_history.n_in[step_index],
            step.link_inflow, rtol=0.0, atol=ATOL,
        )
        torch.testing.assert_close(
            result.cumulative_link_history.n_out[step_index + 1]
            - result.cumulative_link_history.n_out[step_index],
            step.link_outflow, rtol=0.0, atol=ATOL,
        )
        for movement_map, demand, flow in zip(
            network.node_movement_maps,
            step.movement_demand,
            step.movement_flow,
            strict=True,
        ):
            assert bool((flow >= 0.0).all())
            assert bool((flow <= demand + ATOL).all())
            input_index = torch.tensor(movement_map.input_link_ids, dtype=torch.long)
            output_index = torch.tensor(movement_map.output_link_ids, dtype=torch.long)
            torch.testing.assert_close(
                aggregate_input_flow(flow, movement_map),
                step.link_outflow[input_index], rtol=0.0, atol=ATOL,
            )
            torch.testing.assert_close(
                aggregate_output_flow(flow, movement_map),
                step.link_inflow[output_index], rtol=0.0, atol=ATOL,
            )


def test_ordinary_two_node_chain_matches_boundary_trace_and_conserves() -> None:
    scenario = Scenario(
        1.0, 4,
        torch.tensor([[0.125], [0.0], [0.0], [0.0]], dtype=torch.float64),
    )
    network = ordinary_chain()
    result = rollout(network, scenario)

    expected_in = torch.tensor(
        [[0.0, 0.0, 0.0], [0.125, 0.0, 0.0],
         [0.125, 0.125, 0.0], [0.125, 0.125, 0.125],
         [0.125, 0.125, 0.125]], dtype=torch.float64,
    )
    expected_out = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [0.125, 0.0, 0.0], [0.125, 0.125, 0.0],
         [0.125, 0.125, 0.125]], dtype=torch.float64,
    )
    torch.testing.assert_close(result.cumulative_link_history.n_in, expected_in)
    torch.testing.assert_close(result.cumulative_link_history.n_out, expected_out)
    assert result.cumulative_sink_history[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.125]
    assert_rollout_feasible(result, scenario, network)


def test_global_link_and_node_record_permutation_is_behaviorally_identical() -> None:
    scenario = Scenario(
        1.0, 4,
        torch.tensor([[0.125], [0.0], [0.0], [0.0]], dtype=torch.float64),
    )
    canonical = rollout(ordinary_chain(), scenario)
    mapped = rollout(ordinary_chain(permuted=True), scenario)
    old_to_new = [2, 0, 1]

    torch.testing.assert_close(
        mapped.cumulative_link_history.n_in[:, old_to_new],
        canonical.cumulative_link_history.n_in,
    )
    torch.testing.assert_close(
        mapped.cumulative_link_history.n_out[:, old_to_new],
        canonical.cumulative_link_history.n_out,
    )
    assert torch.equal(mapped.cumulative_sink_history, canonical.cumulative_sink_history)
    assert torch.equal(mapped.source_queue_history, canonical.source_queue_history)


def signal_scenario(arrival_pair=(1.0, 10.0)) -> Scenario:
    return Scenario(
        1.0, 4,
        torch.tensor(
            [arrival_pair, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            dtype=torch.float64,
        ),
    )


def test_signalized_trace_competing_greens_and_service_constraints() -> None:
    network, plan = signalized_merge()
    scenario = signal_scenario()
    green_a = torch.tensor([0.4, 0.6], dtype=torch.float64)
    green_b = torch.tensor([0.45, 0.55], dtype=torch.float64)
    first = rollout(network, scenario, SignalControl((green_a,)))
    second = rollout(network, scenario, SignalControl((green_b,)))

    torch.testing.assert_close(
        first.step_results[1].movement_flow[0],
        torch.tensor([1.0, 6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        first.step_results[2].movement_flow[0],
        torch.tensor([0.0, 4.0], dtype=torch.float64),
    )
    assert first.cumulative_sink_history[:, 0].tolist() == [0.0, 0.0, 0.0, 7.0, 11.0]
    assert total_system_time(first, scenario).item() == 37.0
    assert total_system_time(second, scenario).item() == 37.5
    assert_rollout_feasible(first, scenario, network)
    assert_rollout_feasible(second, scenario, network)

    for green, result in ((green_a, first), (green_b, second)):
        service = continuum_service(
            green, plan, network.node_movement_maps[0], scenario.dt_s
        )
        for step in result.step_results:
            if step.movement_flow:
                input_flow = step.movement_flow[0]
                assert bool((input_flow <= service.input_service_mass + ATOL).all())


def test_signalized_differentiability_event_and_reproducibility() -> None:
    network, _ = signalized_merge()
    scenario = signal_scenario()

    def run(green):
        return rollout(network, scenario, SignalControl((green,)))

    function = lambda green: total_system_time(run(green), scenario)
    regime = lambda green: node_active_signature(run(green))
    point = torch.tensor([0.4, 0.6], dtype=torch.float64)
    check = directional_check(
        function, point, two_phase_direction(), regime_function=regime
    )
    torch.testing.assert_close(
        check.reverse_directional,
        torch.tensor(10.0 / math.sqrt(2.0), dtype=torch.float64),
    )
    assert check.reverse_jvp_agree
    assert check.stable_adjacent_pass_count == 6

    event_scenario = signal_scenario((5.0, 0.0))
    event_run = lambda green: rollout(
        network, event_scenario, SignalControl((green,))
    )
    event = directional_check(
        lambda green: total_system_time(event_run(green), event_scenario),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        two_phase_direction(),
        regime_function=lambda green: node_active_signature(event_run(green)),
    )
    assert all(not row.stable_regime and not row.passes for row in event.rows)

    repeated = run(point)
    original = run(point)
    assert torch.equal(
        repeated.cumulative_link_history.n_in,
        original.cumulative_link_history.n_in,
    )
    assert torch.equal(repeated.cumulative_sink_history, original.cumulative_sink_history)
    metrics = traffic_metrics(original, scenario)
    assert metrics.maximum_absolute_conservation_residual.item() == 0.0
