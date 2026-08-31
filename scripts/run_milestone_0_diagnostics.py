#!/usr/bin/env python3
"""Write immutable JSON/Markdown evidence for the tested Milestone 0 harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.gradients import directional_check, node_active_signature, two_phase_direction
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.objectives import total_system_time, traffic_metrics
from diff_minimal_meso.signals import FixedPhasePlan
from diff_minimal_meso.simulation import NetworkDefinition, Scenario, SignalControl, rollout


def _network() -> NetworkDefinition:
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
    links = tuple(LinkFDParameters(0.1, 10.0, 100.0, 1.0, 1.0) for _ in range(3))
    return NetworkDefinition(
        links, (movement_map,), (node,), (plan,),
        torch.tensor([0, 1], dtype=torch.long), torch.tensor([2], dtype=torch.long),
    )


def _scenario(config: dict[str, Any], key: str) -> Scenario:
    pair = config[key]["arrival_pair"]
    arrivals = torch.tensor(
        [pair] + [[0.0, 0.0]] * (config["horizon_steps"] - 1),
        dtype=torch.float64,
    )
    return Scenario(config["dt_s"], config["horizon_steps"], arrivals)


def _evaluate(config: dict[str, Any], key: str):
    network = _network()
    scenario = _scenario(config, key)

    def run(green):
        return rollout(network, scenario, SignalControl((green,)))

    point = torch.tensor(config[key]["green"], dtype=torch.float64)
    check = directional_check(
        lambda green: total_system_time(run(green), scenario),
        point,
        two_phase_direction(),
        regime_function=lambda green: node_active_signature(run(green)),
    )
    result = run(point)
    return check, traffic_metrics(result, scenario)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _check_record(check) -> dict[str, Any]:
    return {
        "objective": _jsonable(check.baseline_objective),
        "direction": _jsonable(check.direction),
        "reverse_directional": _jsonable(check.reverse_directional),
        "jvp_directional": _jsonable(check.jvp_directional),
        "reverse_jvp_agree": check.reverse_jvp_agree,
        "stable_adjacent_pass_count": check.stable_adjacent_pass_count,
        "stable_scenario_passes": check.stable_scenario_passes,
        "step_scan": [
            {
                "h": row.step_size,
                "feasible": row.feasible,
                "objective_plus": _jsonable(row.objective_plus),
                "objective_minus": _jsonable(row.objective_minus),
                "finite_difference": _jsonable(row.finite_difference),
                "absolute_error": _jsonable(row.absolute_error),
                "baseline_regime": _jsonable(row.baseline_regime),
                "plus_regime": _jsonable(row.plus_regime),
                "minus_regime": _jsonable(row.minus_regime),
                "stable_regime": row.stable_regime,
                "passes": row.passes,
            }
            for row in check.rows
        ],
    }


def build_diagnostics(
    config: dict[str, Any], *, config_text: str, command: list[str], timestamp: str
) -> dict[str, Any]:
    stable, stable_metrics = _evaluate(config, "stable")
    event, event_metrics = _evaluate(config, "event_boundary")
    metrics = lambda value: {
        "throughput": _jsonable(value.throughput),
        "terminal_source_queue": _jsonable(value.terminal_source_queue),
        "terminal_link_occupancy": _jsonable(value.terminal_link_occupancy),
        "terminal_mass": _jsonable(value.terminal_mass),
        "maximum_absolute_conservation_residual": _jsonable(
            value.maximum_absolute_conservation_residual
        ),
    }
    return {
        "schema_version": config["schema_version"],
        "timestamp_utc": timestamp,
        "command": command,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "dtype": "torch.float64",
            "device": "cpu",
        },
        "config": config,
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "stable": {**_check_record(stable), "metrics": metrics(stable_metrics)},
        "event_boundary": {**_check_record(event), "metrics": metrics(event_metrics)},
        "acceptance": {
            "stable_reverse_jvp": stable.reverse_jvp_agree,
            "stable_three_adjacent_fd": stable.stable_scenario_passes,
            "event_exposed": any(not row.stable_regime for row in event.rows),
            "conservation": bool(
                stable_metrics.maximum_absolute_conservation_residual == 0.0
                and event_metrics.maximum_absolute_conservation_residual == 0.0
            ),
        },
    }


def _summary(data: dict[str, Any]) -> str:
    stable = data["stable"]
    event = data["event_boundary"]
    acceptance = data["acceptance"]
    return "\n".join((
        "# Milestone 0 diagnostic summary",
        "",
        f"- Stable TST: `{stable['objective']}` veh-eq*s",
        f"- Stable reverse/JVP: `{stable['reverse_directional']}` / `{stable['jvp_directional']}`",
        f"- Adjacent passing stable FD rows: `{stable['stable_adjacent_pass_count']}`",
        f"- Event-boundary TST: `{event['objective']}` veh-eq*s",
        f"- Event regime change exposed: `{acceptance['event_exposed']}`",
        f"- Maximum conservation residual: `{stable['metrics']['maximum_absolute_conservation_residual']}`",
        f"- All acceptance items: `{all(acceptance.values())}`",
        "",
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    config_text = args.config.read_text(encoding="utf-8")
    config = json.loads(config_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    data = build_diagnostics(
        config, config_text=config_text, command=[sys.executable, *sys.argv],
        timestamp=timestamp,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.output_dir / "diagnostics.json"
    summary_path = args.output_dir / "summary.md"
    diagnostics_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(_summary(data), encoding="utf-8")


if __name__ == "__main__":
    main()
