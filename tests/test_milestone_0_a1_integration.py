from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch

from diff_minimal_meso.visualization_data import load_scenario_config, run_visualization_case


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "milestone_0_a1"
RUNNER = ROOT / "scripts" / "run_milestone_0_a1_visualizer.py"


def _runner():
    spec = importlib.util.spec_from_file_location("m0a1_integration_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_a_and_b_share_literal_frame_renderer(tmp_path: Path) -> None:
    runner = _runner()
    config = CONFIGS / "spillback_chain.json"
    candidate_a = tmp_path / "candidate_a"
    candidate_b = tmp_path / "candidate_b"
    runner.generate_candidate_a(output_dir=candidate_a, config_path=config, command=["a"])
    runner.generate_candidate_b(output_dir=candidate_b, config_path=config, command=["b"])

    assert _checksum(candidate_a / "overview.png") == _checksum(candidate_b / "overview.png")
    for selected in ("frame_0000.png", "frame_0003.png", "frame_0004.png", "frame_0006_terminal.png"):
        assert _checksum(candidate_a / "selected_frames" / selected) == _checksum(
            candidate_b / "frames" / selected
        )


def test_paired_signal_cases_preserve_comparison_and_gradient_contract() -> None:
    first = run_visualization_case(load_scenario_config(CONFIGS / "signalized_merge.json"))
    second = run_visualization_case(
        load_scenario_config(CONFIGS / "signalized_merge_green_60_40.json")
    )

    assert first.layout.vertex_ids == second.layout.vertex_ids
    assert torch.equal(first.layout.vertex_xy, second.layout.vertex_xy)
    assert torch.equal(first.layout.link_tail_index, second.layout.link_tail_index)
    assert torch.equal(first.layout.link_head_index, second.layout.link_head_index)
    assert first.layout.comparison_limits == second.layout.comparison_limits
    assert torch.equal(first.boundary_time_s, second.boundary_time_s)
    assert torch.equal(first.link_capacity_step_veh, second.link_capacity_step_veh)
    assert torch.equal(first.link_storage_veh, second.link_storage_veh)
    assert first.sensitivity is not None and second.sensitivity is not None
    assert first.sensitivity.objective_name == second.sensitivity.objective_name == "TST"
    assert first.sensitivity.objective_unit == second.sensitivity.objective_unit == "veh-eq*s"
    assert first.sensitivity.control_node_id == second.sensitivity.control_node_id
    assert torch.equal(first.sensitivity.direction, second.sensitivity.direction)
