"""Component 0.6 deterministic rollout and accounting checks."""

from __future__ import annotations

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.signals import FixedPhasePlan
from diff_minimal_meso.simulation import (
    NetworkDefinition,
    Scenario,
    SignalControl,
    initialize_state,
    mass_balance_residual,
    rollout,
)


def link(*, capacity: float = 1.0, storage: float = 100.0, tau: float = 1.0):
    return LinkFDParameters(0.1, capacity, storage, tau, tau)


def boundary_network(parameters, *, sources, sinks):
    return NetworkDefinition(
        tuple(parameters), (), (), (),
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(sinks, dtype=torch.long),
    )


def chain_network(*, kind=NodeKind.ORDINARY_ORCA, capacities=(1.0, 1.0), plan=None):
    parameters = tuple(link(capacity=value) for value in capacities)
    movement_map = build_movement_map([0], [1], [(0, 1, 1.0)])
    node = NodeParameters(
        movement_map, torch.tensor([capacities[0]], dtype=torch.float64), kind
    )
    return NetworkDefinition(
        parameters, (movement_map,), (node,), (plan,),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )


def assert_conserved(result, scenario):
    torch.testing.assert_close(
        mass_balance_residual(result, scenario),
        torch.zeros(scenario.horizon_steps + 1, dtype=torch.float64,
                    device=scenario.arrivals.device),
        rtol=0.0, atol=1.0e-12,
    )


def test_one_link_positive_travel_time_and_full_histories() -> None:
    network = boundary_network([link(tau=2.0)], sources=[0], sinks=[0])
    scenario = Scenario(
        1.0, 4, torch.tensor([[0.125], [0.0], [0.0], [0.0]], dtype=torch.float64)
    )
    result = rollout(network, scenario)

    torch.testing.assert_close(
        result.cumulative_link_history.n_in[:, 0],
        torch.tensor([0.0, 0.125, 0.125, 0.125, 0.125], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.cumulative_link_history.n_out[:, 0],
        torch.tensor([0.0, 0.0, 0.0, 0.125, 0.125], dtype=torch.float64),
    )
    assert result.source_queue_history.shape == (5, 1)
    assert result.cumulative_sink_history.shape == (5, 1)
    assert result.step_results[0].sink_outflow.item() == 0.0
    assert result.step_results[1].sink_outflow.item() == 0.0
    assert result.step_results[2].sink_outflow.item() == 0.125
    assert_conserved(result, scenario)


def test_two_link_chain_does_not_cascade_through_second_link_same_step() -> None:
    network = chain_network(capacities=(1.0, 1.0))
    # Replace both link travel times by two steps while retaining the node assembly.
    network = NetworkDefinition(
        (link(tau=2.0), link(tau=2.0)), network.node_movement_maps,
        network.node_parameters, network.phase_plans,
        network.source_link_index, network.sink_link_index,
    )
    scenario = Scenario(
        1.0, 5,
        torch.tensor([[0.125], [0.0], [0.0], [0.0], [0.0]], dtype=torch.float64),
    )
    result = rollout(network, scenario)

    assert result.step_results[2].movement_flow[0].item() == 0.125
    assert result.step_results[2].sink_outflow.item() == 0.0
    assert result.step_results[3].sink_outflow.item() == 0.0
    assert result.step_results[4].sink_outflow.item() == 0.125
    assert_conserved(result, scenario)


def test_blocked_source_queues_and_later_admits_without_loss() -> None:
    network = boundary_network(
        [link(capacity=0.1, storage=1.0, tau=10.0)], sources=[0], sinks=[0]
    )
    scenario = Scenario(
        10.0, 3,
        torch.tensor([[1.0], [0.5], [0.0]], dtype=torch.float64),
        sink_receiving=torch.zeros((3, 1), dtype=torch.float64),
    )
    result = rollout(network, scenario)

    assert result.step_results[0].source_admitted.item() == 1.0
    assert result.step_results[1].source_admitted.item() == 0.0
    assert result.source_queue_history[2, 0].item() == 0.5
    assert result.cumulative_link_history.n_in[-1, 0].item() == 1.0
    assert result.cumulative_sink_history[-1, 0].item() == 0.0
    assert_conserved(result, scenario)


def test_ordinary_merge_is_composed_exactly_once() -> None:
    movement_map = build_movement_map(
        [0, 1], [2], [(0, 2, 1.0), (1, 2, 1.0)]
    )
    parameters = (link(capacity=6.0), link(capacity=4.0), link(capacity=5.0))
    node = NodeParameters(
        movement_map, torch.tensor([6.0, 4.0], dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    network = NetworkDefinition(
        parameters, (movement_map,), (node,), (None,),
        torch.tensor([0, 1], dtype=torch.long), torch.tensor([2], dtype=torch.long),
    )
    scenario = Scenario(
        1.0, 3,
        torch.tensor([[6.0, 4.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
    )
    result = rollout(network, scenario)

    torch.testing.assert_close(
        result.step_results[1].movement_flow[0],
        torch.tensor([3.0, 2.0], dtype=torch.float64),
    )
    assert result.cumulative_sink_history[-1, 0].item() == 5.0
    assert_conserved(result, scenario)


def restricted_merge_network():
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
    return NetworkDefinition(
        (link(capacity=10.0),) * 3, (movement_map,), (node,), (plan,),
        torch.tensor([0, 1], dtype=torch.long), torch.tensor([2], dtype=torch.long),
    )


def test_restricted_signal_rollout_preserves_green_gradient_path() -> None:
    green = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    arrivals = torch.tensor(
        [[1.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64, requires_grad=True,
    )
    scenario = Scenario(1.0, 3, arrivals)
    result = rollout(
        restricted_merge_network(), scenario, SignalControl((green,))
    )

    torch.testing.assert_close(
        result.step_results[1].movement_flow[0],
        torch.tensor([1.0, 6.0], dtype=torch.float64),
    )
    assert result.cumulative_sink_history[-1, 0].item() == 7.0
    result.cumulative_sink_history[-1].sum().backward()
    torch.testing.assert_close(green.grad, torch.tensor([0.0, 10.0], dtype=torch.float64))
    assert arrivals.grad is not None
    assert arrivals.grad[0, 0].item() == 1.0
    assert_conserved(result, scenario)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_restricted_signal_rollout_cuda_device_path() -> None:
    device = torch.device("cuda")
    green = torch.tensor(
        [0.4, 0.6], dtype=torch.float64, device=device, requires_grad=True
    )
    arrivals = torch.tensor(
        [[1.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64, device=device,
    )
    scenario = Scenario(1.0, 3, arrivals)
    result = rollout(
        restricted_merge_network(), scenario, SignalControl((green,))
    )

    assert result.cumulative_link_history.n_in.device.type == device.type
    assert result.cumulative_sink_history.device.type == device.type
    result.cumulative_sink_history[-1].sum().backward()
    torch.testing.assert_close(
        green.grad, torch.tensor([0.0, 10.0], dtype=torch.float64, device=device)
    )
    assert_conserved(result, scenario)


def test_rollout_is_deterministic_and_does_not_mutate_inputs() -> None:
    network = boundary_network([link()], sources=[0], sinks=[0])
    arrivals = torch.tensor([[0.25], [0.0]], dtype=torch.float64)
    original = arrivals.clone()
    scenario = Scenario(1.0, 2, arrivals)
    first = rollout(network, scenario)
    second = rollout(network, scenario)

    assert torch.equal(arrivals, original)
    assert torch.equal(first.cumulative_link_history.n_in, second.cumulative_link_history.n_in)
    assert torch.equal(first.cumulative_link_history.n_out, second.cumulative_link_history.n_out)
    assert torch.equal(first.source_queue_history, second.source_queue_history)
    assert torch.equal(first.cumulative_sink_history, second.cumulative_sink_history)
    assert first.initial_state.cumulative_links.current_index == 0


def test_structural_ownership_and_capacity_mismatch_are_rejected() -> None:
    movement_map = build_movement_map([0], [1], [(0, 1, 1.0)])
    wrong_capacity = NodeParameters(
        movement_map, torch.tensor([2.0], dtype=torch.float64), NodeKind.ORDINARY_ORCA
    )
    with pytest.raises(ValueError, match="capacity"):
        NetworkDefinition(
            (link(), link()), (movement_map,), (wrong_capacity,), (None,),
            torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long),
        )

    correct = NodeParameters(
        movement_map, torch.tensor([1.0], dtype=torch.float64), NodeKind.ORDINARY_ORCA
    )
    with pytest.raises(ValueError, match="receiving boundary"):
        NetworkDefinition(
            (link(), link()), (movement_map,), (correct,), (None,),
            torch.tensor([1], dtype=torch.long), torch.tensor([1], dtype=torch.long),
        )


def test_scenario_dimensions_and_required_signal_control_are_validated() -> None:
    network = boundary_network([link()], sources=[0], sinks=[0])
    scenario = Scenario(1.0, 2, torch.zeros((2, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="source dimension"):
        initialize_state(network, scenario)

    restricted = restricted_merge_network()
    valid = Scenario(1.0, 1, torch.zeros((1, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="signal control"):
        rollout(restricted, valid)
