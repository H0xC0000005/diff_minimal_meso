#!/usr/bin/env python3
"""Generate deterministic Candidate-A PNG artifacts from a scenario or bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import tempfile
from typing import Any

_CACHE_DIR = Path(tempfile.gettempdir()) / "diff_minimal_meso_matplotlib_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy
from PIL import Image, __version__ as PILLOW_VERSION
import torch

from diff_minimal_meso.visualization import DPI, FIGURE_SIZE_IN, save_frame
from diff_minimal_meso.visualization_data import (
    BUNDLE_SCHEMA_VERSION,
    VisualizationBundle,
    bundle_from_dict,
    bundle_to_dict,
    load_scenario_config,
    run_visualization_case,
)


MANIFEST_SCHEMA_VERSION = "m0-a1-manifest-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], check=False, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _frame_name(frame: int, terminal: int) -> str:
    suffix = "_terminal" if frame == terminal else ""
    return f"frame_{frame:04d}{suffix}.png"


def _source_bundle(path: Path) -> tuple[VisualizationBundle, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") == BUNDLE_SCHEMA_VERSION:
        return bundle_from_dict(raw), {"mode": "saved_bundle", "path": str(path)}
    if raw.get("schema_version") == MANIFEST_SCHEMA_VERSION:
        return bundle_from_dict(raw["bundle"]), {
            "mode": "saved_manifest", "path": str(path),
            "original_source": raw.get("source"),
        }
    raise ValueError("rerender input must be an M0_a1 bundle or manifest")


def _write_artifacts(
    bundle: VisualizationBundle,
    output_dir: Path,
    *,
    source: dict[str, Any],
    command: list[str],
    complete_frame_sequence: bool = False,
) -> None:
    terminal = bundle.boundary_time_s.numel() - 1
    overview_frame = min(terminal - 1, terminal // 2) if terminal > 0 else 0
    overview_path = output_dir / "overview.png"
    save_frame(bundle, overview_frame, overview_path)
    frame_dir = output_dir / ("frames" if complete_frame_sequence else "selected_frames")
    frame_dir.mkdir()
    frame_indices = range(terminal + 1) if complete_frame_sequence else bundle.layout.selected_frames
    frame_paths: list[Path] = []
    for frame in frame_indices:
        path = frame_dir / _frame_name(frame, terminal)
        save_frame(bundle, frame, path)
        frame_paths.append(path)

    png_paths = [overview_path, *frame_paths]
    for path in png_paths:
        with Image.open(path) as image:
            if image.size != (int(FIGURE_SIZE_IN[0] * DPI), int(FIGURE_SIZE_IN[1] * DPI)):
                raise RuntimeError("renderer produced an unexpected PNG size")
            image.verify()
    artifacts = [
        {"path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in png_paths
    ]
    sequence_record = {
        "directory": frame_dir.name,
        "count": len(frame_paths),
        "ordered_files": [path.name for path in frame_paths],
        "checksums": [_sha256(path) for path in frame_paths],
        "total_bytes": sum(path.stat().st_size for path in frame_paths),
    }
    render_record = {
        "figure_size_in": list(FIGURE_SIZE_IN), "dpi": DPI,
        "frame_convention": "state k plus interval k for k<T; terminal state only",
        "overview_frame": overview_frame,
    }
    if complete_frame_sequence:
        render_record.update(
            profile="candidate_b_complete_sequence", frame_sequence=sequence_record
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "command": command,
        "git": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "matplotlib": matplotlib.__version__, "numpy": numpy.__version__,
            "pillow": PILLOW_VERSION, "backend": matplotlib.get_backend(),
            "dtype": "torch.float64", "device": "cpu",
        },
        "render": render_record,
        "bundle": bundle_to_dict(bundle),
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sensitivity = bundle.sensitivity
    sensitivity_line = (
        "not requested"
        if sensitivity is None
        else f"{sensitivity.reverse_directional:.12g} veh-eq*s along {sensitivity.direction.tolist()}"
    )
    summary_title = (
        "# Milestone 0_a1 Candidate-B complete sequence\n\n"
        if complete_frame_sequence
        else "# Milestone 0_a1 Candidate-A visualization\n\n"
    )
    (output_dir / "summary.md").write_text(
        summary_title
        + f"- Links: {', '.join(bundle.link_ids)}\n"
        f"- Boundaries / intervals: {terminal + 1} / {terminal}\n"
        f"- TST: {bundle.total_system_time_veh_s:.12g} veh-eq*s\n"
        f"- Maximum conservation residual: {bundle.conservation_residual_veh.abs().max().item():.3e} veh-eq\n"
        f"- TST directional sensitivity: {sensitivity_line}\n"
        f"- Backend: {matplotlib.get_backend()}\n",
        encoding="utf-8",
    )


def generate_candidate_a(
    *,
    output_dir: Path,
    config_path: Path | None = None,
    rerender_path: Path | None = None,
    command: list[str] | None = None,
) -> None:
    """Generate artifacts atomically and refuse every existing output path."""

    _generate_artifacts(
        output_dir=output_dir, config_path=config_path, rerender_path=rerender_path,
        command=command, complete_frame_sequence=False,
    )


def generate_candidate_b(
    *,
    output_dir: Path,
    config_path: Path | None = None,
    rerender_path: Path | None = None,
    command: list[str] | None = None,
) -> None:
    """Generate the complete ordered boundary-frame sequence atomically."""

    _generate_artifacts(
        output_dir=output_dir, config_path=config_path, rerender_path=rerender_path,
        command=command, complete_frame_sequence=True,
    )


def _generate_artifacts(
    *,
    output_dir: Path,
    config_path: Path | None,
    rerender_path: Path | None,
    command: list[str] | None,
    complete_frame_sequence: bool,
) -> None:
    if (config_path is None) == (rerender_path is None):
        raise ValueError("provide exactly one config_path or rerender_path")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output path {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        config_text = config_path.read_text(encoding="utf-8")
        bundle = run_visualization_case(load_scenario_config(config_path))
        source = {
            "mode": "simulation", "path": str(config_path),
            "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            "config": json.loads(config_text),
        }
    else:
        assert rerender_path is not None
        bundle, source = _source_bundle(rerender_path)
    with TemporaryDirectory(prefix=".m0a1_candidate_a_", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        _write_artifacts(
            bundle, staging, source=source, command=command or [],
            complete_frame_sequence=complete_frame_sequence,
        )
        staging.replace(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--rerender-bundle", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--complete-frame-sequence", action="store_true",
        help="emit Candidate B: one ordered PNG for every boundary 0..T",
    )
    args = parser.parse_args()
    generator = generate_candidate_b if args.complete_frame_sequence else generate_candidate_a
    generator(
        output_dir=args.output_dir,
        config_path=args.config,
        rerender_path=args.rerender_bundle,
        command=[sys.executable, *sys.argv],
    )


if __name__ == "__main__":
    main()
