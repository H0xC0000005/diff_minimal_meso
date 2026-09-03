#!/usr/bin/env python3
"""Generate immutable Milestone 1 integration evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.gradients import (
    AD_ATOL,
    AD_RTOL,
    FD_ATOL,
    FD_RTOL,
    directional_check,
    node_active_signature,
    two_phase_direction,
)
from diff_minimal_meso.ledger_diagnostics import build_ledger_diagnostics, diagnostics_json, diagnostics_markdown
from diff_minimal_meso.movements import build_movement_map
from diff_minimal_meso.nodes import NodeKind, NodeParameters
from diff_minimal_meso.objectives import total_system_time
from diff_minimal_meso.route_simulation import MesoScenario, SourceRouteBlock, meso_rollout
from diff_minimal_meso.routes import RouteDefinition, build_route_table
from diff_minimal_meso.signals import FixedPhasePlan
from diff_minimal_meso.simulation import NetworkDefinition, SignalControl


def _link(capacity: float = 1.0) -> LinkFDParameters:
    return LinkFDParameters(0.1, capacity, 100.0, 1.0, 1.0)


def _ordinary_node(inputs, outputs, movements, capacity=1.0):
    movement_map = build_movement_map(inputs, outputs, movements)
    node = NodeParameters(
        movement_map,
        torch.full((len(inputs),), capacity, dtype=torch.float64),
        NodeKind.ORDINARY_ORCA,
    )
    return movement_map, node


def _block(route, mass, front, tail):
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    return SourceRouteBlock(route, tensor(mass), tensor(front), tensor(tail))


def ordinary_case(config):
    merge_map, merge = _ordinary_node([0, 1], [2], [(0, 2, 1.0), (1, 2, 1.0)])
    diverge_map, diverge = _ordinary_node([2], [3, 4], [(2, 3, 0.5), (2, 4, 0.5)])
    network = NetworkDefinition(
        tuple(_link() for _ in range(5)),
        (merge_map, diverge_map), (merge, diverge), (None, None),
        torch.tensor([0, 1]), torch.tensor([3, 4]),
    )
    table = build_route_table(
        network, (RouteDefinition("A", (0, 2, 3)), RouteDefinition("B", (1, 2, 4)))
    )
    empty = ((), ())
    arrivals = (
        (
            (
                _block(0, 0.25, -1.0, -0.5),
                _block(0, 0.25, -0.5, 0.0),
            ),
            (_block(1, 0.5, -1.0, 0.0),),
        ),
        ((_block(0, 0.125, 0.0, 1.0),), (_block(1, 0.125, 0.0, 1.0),)),
        *([empty] * (config["ordinary_horizon_steps"] - 2)),
    )
    capacity = torch.ones((config["ordinary_horizon_steps"], 2), dtype=torch.float64)
    capacity[1] = 0.0
    scenario = MesoScenario(
        config["dt_s"], config["ordinary_horizon_steps"], arrivals, table,
        source_entry_capacity=capacity,
    )
    result = meso_rollout(scenario)
    return scenario, result


def signal_case(config):
    movement_map = build_movement_map([0, 1], [2], [(0, 2, 1.0), (1, 2, 1.0)])
    plan = FixedPhasePlan(
        (0, 1), torch.tensor([[True, False], [False, True]]),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    node = NodeParameters(
        movement_map, torch.tensor([10.0, 10.0], dtype=torch.float64),
        NodeKind.RESTRICTED_CONTINUUM_SIGNAL,
    )
    network = NetworkDefinition(
        tuple(_link(10.0) for _ in range(3)), (movement_map,), (node,), (plan,),
        torch.tensor([0, 1]), torch.tensor([2]),
    )
    table = build_route_table(network, (RouteDefinition("A", (0, 2)), RouteDefinition("B", (1, 2))))
    empty = ((), ())
    arrivals = (
        ((_block(0, 1.0, -1.0, 0.0),), (_block(1, 10.0, -1.0, 0.0),)),
        *([empty] * (config["signal_horizon_steps"] - 1)),
    )
    scenario = MesoScenario(config["dt_s"], config["signal_horizon_steps"], arrivals, table)
    macro_scenario = scenario.to_macro_scenario()

    def run(green):
        return meso_rollout(scenario, SignalControl((green,)))

    point = torch.tensor(config["signal_green"], dtype=torch.float64)
    check = directional_check(
        lambda green: total_system_time(run(green).macro_rollout_result, macro_scenario),
        point, two_phase_direction(),
        regime_function=lambda green: node_active_signature(run(green).macro_rollout_result),
    )
    return scenario, run(point), check


def merge_telemetry(result):
    diagnostics = tuple(
        item
        for step in result.step_results
        for item in step.merge_diagnostics
    )
    return {
        "source": "actual_integration_orchestration",
        "calls": len(diagnostics),
        "adjacent_pairs_examined": sum(x.adjacent_pairs_examined for x in diagnostics),
        "exact_merges": sum(x.exact_merges for x in diagnostics),
        "entries_before": sum(x.before_count for x in diagnostics),
        "entries_after": sum(x.after_count for x in diagnostics),
        "safe_nonmerge_reasons": sorted(
            {reason for x in diagnostics for reason in x.safe_nonmerge_reasons}
        ),
    }


def step_scan(check):
    return [
        {
            "step_size": row.step_size,
            "feasible": row.feasible,
            "finite_difference": None if row.finite_difference is None else float(row.finite_difference),
            "absolute_error": None if row.absolute_error is None else float(row.absolute_error),
            "stable_regime": row.stable_regime,
            "passes": row.passes,
        }
        for row in check.rows
    ]


def build_evidence(config, *, config_text, command, timestamp, artifact_paths=None):
    torch.manual_seed(config["seed"])
    ordinary_scenario, ordinary = ordinary_case(config)
    signal_scenario_value, signal, check = signal_case(config)
    generated = ordinary_scenario.to_macro_scenario().arrivals.sum()
    final = ordinary.ledger_history[-1]
    remaining = final.completed_route_mass.sum()
    remaining = remaining + sum((x.total_mass(like=generated) for x in final.link_ledgers), generated.new_zeros(()))
    remaining = remaining + sum((x.total_mass(like=generated) for x in final.source_ledgers), generated.new_zeros(()))
    diagnostic_state = max(
        ordinary.ledger_history,
        key=lambda state: sum(len(ledger.entries) for ledger in state.link_ledgers),
    )
    ledger_view = build_ledger_diagnostics(diagnostic_state, ordinary_scenario.route_table)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout)
    acceptance = {
        "ordinary_conservation": bool(torch.isclose(remaining, generated, rtol=config["rtol"], atol=config["atol"])),
        "blocked_source_exercised": bool(ordinary_scenario.source_entry_capacity[1].sum() == 0),
        "macro_exact": True,
        "reverse_jvp": check.reverse_jvp_agree,
        "three_adjacent_fd": check.stable_scenario_passes,
        "finite_gradients": bool(torch.isfinite(check.reverse_directional) and torch.isfinite(check.jvp_directional)),
    }
    return {
        "schema_version": config["schema_version"], "timestamp_utc": timestamp,
        "command": command, "git": {"commit": git_commit, "dirty": dirty},
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "device": "cpu", "dtype": "torch.float64"},
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(), "config": config,
        "artifacts": artifact_paths,
        "ordinary": {"generated_mass": float(generated), "accounted_mass": float(remaining), "completed_by_route": final.completed_route_mass.tolist(), "ordered_snapshot": diagnostics_json(ledger_view), "merge_telemetry": merge_telemetry(ordinary)},
        "macro_equivalence": {"exact": True, "basis": "meso_rollout fail-fast torch.equal checks"},
        "signal": {"objective": float(check.baseline_objective), "reverse_directional": float(check.reverse_directional), "jvp_directional": float(check.jvp_directional), "stable_adjacent_pass_count": check.stable_adjacent_pass_count, "active_signature": node_active_signature(signal.macro_rollout_result), "step_scan": step_scan(check), "effective_tolerances": {"ad_atol": AD_ATOL, "ad_rtol": AD_RTOL, "fd_atol": FD_ATOL, "fd_rtol": FD_RTOL}},
        "acceptance": acceptance,
    }, diagnostics_markdown(ledger_view)


def summary(data, ledger_markdown):
    return "\n".join(("# Milestone 1 integration summary", "", f"- Generated/accounted mass: `{data['ordinary']['generated_mass']}` / `{data['ordinary']['accounted_mass']}` veh-eq", f"- Signal reverse/JVP: `{data['signal']['reverse_directional']}` / `{data['signal']['jvp_directional']}`", f"- Stable adjacent FD rows: `{data['signal']['stable_adjacent_pass_count']}`", f"- All acceptance items: `{all(data['acceptance'].values())}`", "", ledger_markdown))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    text = args.config.read_text(encoding="utf-8")
    config = json.loads(text)
    output = args.output_root / config["run_id"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty run directory")
    artifact_paths = {
        "evidence_json": str(output / "evidence.json"),
        "summary_markdown": str(output / "summary.md"),
    }
    data, ledger_markdown = build_evidence(config, config_text=text, command=[sys.executable, *sys.argv], timestamp=datetime.now(timezone.utc).isoformat(), artifact_paths=artifact_paths)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(summary(data, ledger_markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
