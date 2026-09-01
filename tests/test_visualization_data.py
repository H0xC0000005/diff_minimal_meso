from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from diff_minimal_meso.simulation import rollout
from diff_minimal_meso.visualization_data import (
    BUNDLE_SCHEMA_VERSION,
    bundle_from_dict,
    bundle_to_dict,
    extract_visualization_bundle,
    load_scenario_config,
    parse_scenario_config,
    run_visualization_case,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "milestone_0_a1"


def load_raw(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_declarative_ids_are_mapped_deterministically() -> None:
    raw = load_raw("signalized_merge.json")
    raw["links"].reverse()
    case = parse_scenario_config(raw)
    assert case.link_ids == ("approach_a", "approach_b", "exit")
    assert case.node_ids == ("signal",)
    assert case.source_ids == ("approach_a", "approach_b")
    assert case.network.source_link_index.tolist() == [0, 1]
    assert case.layout.link_labels == ("Approach A", "Approach B", "Exit")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(extra=True), "unknown keys"),
        (lambda raw: raw["links"].append(deepcopy(raw["links"][0])), "IDs must be unique"),
        (lambda raw: raw["nodes"][0]["movements"][0].update(beta=0.8), "must sum to one"),
        (lambda raw: raw["layout"]["links"].pop(), "cover exactly"),
    ],
)
def test_schema_rejects_invalid_data_without_repair(change, message: str) -> None:
    raw = load_raw("ordinary_chain.json")
    change(raw)
    with pytest.raises((TypeError, ValueError), match=message):
        parse_scenario_config(raw)


def test_raw_direction_is_rejected_instead_of_normalized() -> None:
    raw = load_raw("signalized_merge.json")
    raw["sensitivity"]["direction"] = [1.0, -1.0]
    with pytest.raises(ValueError, match="unit L2 norm"):
        parse_scenario_config(raw)


def test_phase_matrix_rejects_numeric_booleans() -> None:
    raw = load_raw("signalized_merge.json")
    raw["nodes"][0]["phase_plan"]["movement_phase_matrix"] = [[1, 0], [0, 1]]
    with pytest.raises(TypeError, match="JSON booleans"):
        parse_scenario_config(raw)


def test_signal_sensitivity_uses_accepted_unit_direction() -> None:
    bundle = run_visualization_case(load_scenario_config(CONFIGS / "signalized_merge.json"))
    assert bundle.sensitivity is not None
    assert bundle.physical_green_by_node[0] is not None
    assert bundle.physical_green_by_node[0].tolist() == pytest.approx([0.4, 0.6])
    assert float(torch.linalg.vector_norm(bundle.sensitivity.direction)) == pytest.approx(1.0)
    assert bundle.sensitivity.direction.tolist() == pytest.approx(
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)]
    )
    assert bundle.sensitivity.reverse_directional == pytest.approx(7.071067811865475)
    assert bundle.sensitivity.jvp_directional == pytest.approx(
        bundle.sensitivity.reverse_directional, rel=1e-12, abs=1e-12
    )
    assert bundle.sensitivity.reverse_jvp_agree
    assert bundle.sensitivity.stable_scenario_passes
    assert not bundle.sensitivity.event_detected


def test_spillback_fixture_matches_frozen_authority() -> None:
    bundle = run_visualization_case(load_scenario_config(CONFIGS / "spillback_chain.json"))
    upstream = bundle.link_ids.index("upstream")
    downstream = bundle.link_ids.index("downstream")
    assert bundle.occupancy_veh[:, [upstream, downstream]].tolist() == [
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
        [2.0, 2.0], [2.0, 2.0], [2.0, 2.0],
    ]
    assert bundle.source_queue_veh[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    assert bundle.receiving_veh[:, upstream].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert bundle.receiving_veh[:, downstream].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert bundle.movement_flow_veh[0][:, 0].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert bundle.link_inflow_veh[:, upstream].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert torch.count_nonzero(bundle.cumulative_sink_veh) == 0
    assert torch.count_nonzero(bundle.conservation_residual_veh) == 0


def test_bundle_is_detached_non_aliasing_and_round_trips() -> None:
    case = load_scenario_config(CONFIGS / "ordinary_chain.json")
    result = rollout(case.network, case.scenario, case.control)
    bundle = extract_visualization_bundle(case, result)
    before = bundle.occupancy_veh.clone()
    result.cumulative_link_history.n_in[1, 0].add_(99.0)
    assert torch.equal(bundle.occupancy_veh, before)
    assert all(
        value.dtype == torch.float64 and value.device.type == "cpu" and not value.requires_grad
        for value in (
            bundle.occupancy_veh, bundle.sending_veh, bundle.source_queue_veh,
            *bundle.movement_flow_veh,
        )
    )
    encoded = bundle_to_dict(bundle)
    assert encoded["schema_version"] == BUNDLE_SCHEMA_VERSION
    restored = bundle_from_dict(json.loads(json.dumps(encoded)))
    assert bundle_to_dict(restored) == encoded


def _junction_config(input_count: int, output_count: int) -> dict:
    horizon = 2
    input_ids = [f"i{index}" for index in range(input_count)]
    output_ids = [f"o{index}" for index in range(output_count)]
    link_ids = input_ids + output_ids
    links = [
        {"id": value, "label": value, "critical_density_veh_per_m": 0.1,
         "capacity_veh_per_s": 1.0, "jam_storage_veh": 10.0,
         "free_flow_time_s": 1.0, "backward_wave_time_s": 1.0}
        for value in link_ids
    ]
    movements = [
        {"input": input_id, "output": output_id, "beta": 1.0 / output_count}
        for input_id in input_ids for output_id in output_ids
    ]
    vertices = [
        {"id": f"v_{value}", "x": float(index // max(1, input_count)), "y": float(index)}
        for index, value in enumerate(link_ids)
    ] + [{"id": "v_node", "x": 1.0, "y": 0.0}]
    edges = [
        {"link_id": value, "tail": f"v_{value}", "head": "v_node"}
        for value in input_ids
    ] + [
        {"link_id": value, "tail": "v_node", "head": f"v_{value}"}
        for value in output_ids
    ]
    return {
        "schema_version": "m0-a1-scenario-v1", "dt_s": 1.0,
        "horizon_steps": horizon, "links": links,
        "nodes": [{"id": "node", "kind": "ordinary_orca", "input_links": input_ids,
                   "output_links": output_ids, "movements": movements,
                   "input_capacity_rate": [1.0] * input_count}],
        "sources": [{"link_id": value, "arrivals": [1.0, 0.0]} for value in input_ids],
        "sinks": [{"link_id": value} for value in output_ids], "controls": {},
        "layout": {"vertices": vertices, "links": edges, "node_vertices": {"node": "v_node"},
                   "selected_frames": [0, 2], "comparison_limits": {}},
    }


@pytest.mark.parametrize(("inputs", "outputs"), [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_arbitrary_siso_simo_miso_mimo_topologies(inputs: int, outputs: int) -> None:
    case = parse_scenario_config(_junction_config(inputs, outputs))
    bundle = run_visualization_case(case)
    assert case.network.node_movement_maps[0].input_count == inputs
    assert case.network.node_movement_maps[0].output_count == outputs
    assert bundle.occupancy_veh.shape == (3, inputs + outputs)
    assert torch.max(torch.abs(bundle.conservation_residual_veh)) <= 1e-10


def test_grid_2x2_single_od_fixture_has_eight_selected_boundaries() -> None:
    case = load_scenario_config(CONFIGS / "grid_2x2_single_od.json")
    bundle = run_visualization_case(case)
    assert case.network.node_count == 4
    assert case.network.link_count == 6
    assert len(bundle.source_ids) == len(bundle.sink_ids) == 1
    assert bundle.layout.selected_frames == (0, 1, 2, 3, 4, 5, 7, 10)
    assert torch.max(torch.abs(bundle.conservation_residual_veh)) <= 1e-10
    assert bundle.cumulative_sink_veh[-1, 0] == pytest.approx(8.0)


def test_multi_od_grid_structurally_preserves_selected_8_to_4() -> None:
    case = load_scenario_config(CONFIGS / "grid_2x2_multi_od_selected_8_to_4.json")
    transitions: dict[int, set[int]] = {}
    for movement_map in case.network.node_movement_maps:
        for input_local, output_local in zip(
            movement_map.movement_input_index.tolist(),
            movement_map.movement_output_index.tolist(), strict=True,
        ):
            input_link = movement_map.input_link_ids[input_local]
            output_link = movement_map.output_link_ids[output_local]
            transitions.setdefault(input_link, set()).add(output_link)

    def reachable(start: int) -> set[int]:
        found = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for successor in transitions.get(current, set()):
                if successor not in found:
                    found.add(successor)
                    frontier.append(successor)
        return found

    link_index = {value: index for index, value in enumerate(case.link_ids)}
    selected_origin = link_index["leg8_in"]
    selected_destination = link_index["leg4_out"]
    assert selected_destination in reachable(selected_origin)
    selected_reachable_sinks = set(case.network.sink_link_index.tolist()) & reachable(selected_origin)
    assert selected_reachable_sinks == {selected_destination}
    other_sources = set(case.network.source_link_index.tolist()) - {selected_origin}
    assert all(selected_destination not in reachable(source) for source in other_sources)

    bundle = run_visualization_case(case)
    assert bundle.layout.selected_frames == (0, 4, 6, 9, 12, 14, 16, 18)
    assert torch.max(torch.abs(bundle.conservation_residual_veh)) <= 1e-10
    assert float(bundle.source_queue_veh.max()) > 0.0
    assert float(bundle.occupancy_ratio.max()) == pytest.approx(1.0)
