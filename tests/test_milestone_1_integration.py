"""Milestone 1 integration and immutable artifact checks."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_milestone_1_diagnostics.py"
CONFIG = ROOT / "configs/milestone_1/minimal_ordered_route.json"


def runner():
    spec = importlib.util.spec_from_file_location("m1_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_integration_evidence_meets_every_gate_and_is_deterministic():
    module = runner()
    text = CONFIG.read_text(encoding="utf-8")
    config = json.loads(text)
    kwargs = dict(config_text=text, command=["fixed"], timestamp="fixed")
    first, first_md = module.build_evidence(config, **kwargs)
    second, second_md = module.build_evidence(config, **kwargs)
    assert first == second
    assert first_md == second_md
    assert all(first["acceptance"].values())
    assert first["ordinary"]["generated_mass"] == first["ordinary"]["accounted_mass"]
    telemetry = first["ordinary"]["merge_telemetry"]
    assert telemetry["source"] == "actual_integration_orchestration"
    assert telemetry["calls"] > 0
    assert telemetry["adjacent_pairs_examined"] > 0
    assert telemetry["exact_merges"] >= 1
    assert telemetry["entries_before"] - telemetry["entries_after"] == telemetry["exact_merges"]
    assert first["ordinary"]["ordered_snapshot"]["ordered_entries"]
    assert "route_mismatch" in telemetry["safe_nonmerge_reasons"]
    assert first["signal"]["stable_adjacent_pass_count"] >= 3
    assert len(first["signal"]["step_scan"]) == 6
    assert [row["step_size"] for row in first["signal"]["step_scan"]] != config.get("fd_step_sizes")
    assert first["signal"]["effective_tolerances"] == {
        "ad_atol": 1e-10,
        "ad_rtol": 1e-10,
        "fd_atol": 1e-8,
        "fd_rtol": 1e-6,
    }
    assert first["macro_equivalence"]["basis"].startswith("meso_rollout fail-fast")


def test_runner_writes_both_views_and_refuses_overwrite(tmp_path):
    command = [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--output-root", str(tmp_path)]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "reference_cpu_float64_v3"
    assert (output / "evidence.json").is_file()
    assert (output / "summary.md").is_file()
    evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["artifacts"] == {
        "evidence_json": str(output / "evidence.json"),
        "summary_markdown": str(output / "summary.md"),
    }
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
