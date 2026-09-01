from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_milestone_0_a1_visualizer.py"
CONFIG = ROOT / "configs" / "milestone_0_a1" / "signalized_merge.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("m0a1_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_a_run_and_saved_data_rerender_are_equivalent(tmp_path: Path) -> None:
    runner = load_runner()
    first = tmp_path / "first"
    second = tmp_path / "second"
    runner.generate_candidate_a(output_dir=first, config_path=CONFIG, command=["fixed"])
    runner.generate_candidate_a(
        output_dir=second, rerender_path=first / "manifest.json", command=["rerender"]
    )
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["bundle"] == second_manifest["bundle"]
    assert first_manifest["render"] == second_manifest["render"]
    first_artifacts = {item["path"]: item["sha256"] for item in first_manifest["artifacts"]}
    second_artifacts = {item["path"]: item["sha256"] for item in second_manifest["artifacts"]}
    assert first_artifacts == second_artifacts
    assert first_artifacts == {
        "overview.png": checksum(first / "overview.png"),
        "selected_frames/frame_0000.png": checksum(first / "selected_frames/frame_0000.png"),
        "selected_frames/frame_0001.png": checksum(first / "selected_frames/frame_0001.png"),
        "selected_frames/frame_0004_terminal.png": checksum(first / "selected_frames/frame_0004_terminal.png"),
    }
    with Image.open(first / "overview.png") as image:
        assert image.size == (1200, 800)


def test_runner_refuses_existing_path_without_changing_it(tmp_path: Path) -> None:
    runner = load_runner()
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing existing"):
        runner.generate_candidate_a(output_dir=output, config_path=CONFIG)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_cli_generates_candidate_a_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--output-dir", str(output)],
        cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": "src"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "manifest.json").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "overview.png").is_file()


def test_candidate_b_complete_sequence_is_ordered_recorded_and_rerenderable(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    first = tmp_path / "complete_first"
    second = tmp_path / "complete_second"
    runner.generate_candidate_b(output_dir=first, config_path=CONFIG, command=["fixed"])
    runner.generate_candidate_b(
        output_dir=second, rerender_path=first / "manifest.json", command=["rerender"]
    )

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    sequence = first_manifest["render"]["frame_sequence"]
    expected = [
        "frame_0000.png", "frame_0001.png", "frame_0002.png",
        "frame_0003.png", "frame_0004_terminal.png",
    ]
    assert first_manifest["render"]["profile"] == "candidate_b_complete_sequence"
    assert sequence["directory"] == "frames"
    assert sequence["count"] == 5
    assert sequence["ordered_files"] == expected
    assert sequence["checksums"] == [checksum(first / "frames" / name) for name in expected]
    assert sequence["total_bytes"] == sum(
        (first / "frames" / name).stat().st_size for name in expected
    )
    assert first_manifest["bundle"] == second_manifest["bundle"]
    assert first_manifest["render"] == second_manifest["render"]
    assert sequence["checksums"] == second_manifest["render"]["frame_sequence"]["checksums"]
    assert not (first / "selected_frames").exists()


def test_cli_generates_candidate_b_complete_sequence(tmp_path: Path) -> None:
    output = tmp_path / "complete_cli"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--config", str(CONFIG),
            "--output-dir", str(output), "--complete-frame-sequence",
        ],
        cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": "src"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((output / "frames").glob("*.png"))) == 5
    assert (output / "frames" / "frame_0004_terminal.png").is_file()
