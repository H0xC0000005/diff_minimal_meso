"""Component 0.7 durable artifact contract checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_milestone_0_diagnostics.py"
CONFIG = ROOT / "configs" / "milestone_0" / "minimal_signalized.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("milestone_0_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_schema_and_values_are_deterministic() -> None:
    runner = load_runner()
    text = CONFIG.read_text(encoding="utf-8")
    config = json.loads(text)
    keywords = dict(config_text=text, command=["fixed-command"], timestamp="fixed-time")
    first = runner.build_diagnostics(config, **keywords)
    second = runner.build_diagnostics(config, **keywords)

    assert first == second
    assert first["schema_version"] == "milestone-0-diagnostics-v1"
    assert first["stable"]["objective"] == 37.0
    assert first["stable"]["stable_adjacent_pass_count"] == 6
    assert all(first["acceptance"].values())
    assert len(first["stable"]["step_scan"]) == 6


def test_summary_contains_the_acceptance_evidence() -> None:
    runner = load_runner()
    text = CONFIG.read_text(encoding="utf-8")
    data = runner.build_diagnostics(
        json.loads(text), config_text=text, command=["fixed"], timestamp="fixed"
    )
    summary = runner._summary(data)
    assert "Stable TST: `37.0`" in summary
    assert "All acceptance items: `True`" in summary


def test_runner_refuses_to_overwrite_nonempty_directory(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user result", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--config", str(CONFIG),
            "--output-dir", str(output),
        ],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode != 0
    assert "refusing to overwrite" in completed.stderr
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user result"
