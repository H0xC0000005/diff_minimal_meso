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
    assert first["ordinary"]["merge_telemetry"]["exact_merges"] == 1
    assert first["ordinary"]["ordered_snapshot"]["ordered_entries"]
    assert set(first["ordinary"]["merge_telemetry"]["safe_nonmerge_reasons"]) == {"time_gap", "route_mismatch", "rate_breakpoint"}
    assert first["signal"]["stable_adjacent_pass_count"] >= 3


def test_runner_writes_both_views_and_refuses_overwrite(tmp_path):
    command = [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--output-root", str(tmp_path)]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "reference_cpu_float64_v2"
    assert (output / "evidence.json").is_file()
    assert (output / "summary.md").is_file()
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
