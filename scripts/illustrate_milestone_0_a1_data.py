#!/usr/bin/env python3
"""Run the A1.1 data path and save a re-renderable bundle for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from diff_minimal_meso.visualization_data import (
    load_bundle,
    load_scenario_config,
    run_visualization_case,
    save_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    if args.bundle.exists():
        raise FileExistsError(f"refusing to overwrite {args.bundle}")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)

    case = load_scenario_config(args.config)
    bundle = run_visualization_case(case)
    save_bundle(bundle, args.bundle)
    restored = load_bundle(args.bundle)

    print(f"scenario: {args.config}")
    print(f"links: {', '.join(restored.link_ids)}")
    print(f"boundaries/intervals: {restored.boundary_time_s.numel()}/{restored.sending_veh.shape[0]}")
    print(f"terminal occupancy [veh-eq]: {restored.occupancy_veh[-1].tolist()}")
    print(f"terminal source queue [veh-eq]: {restored.source_queue_veh[-1].tolist()}")
    print(f"maximum conservation residual [veh-eq]: {restored.conservation_residual_veh.abs().max().item():.3e}")
    print(f"TST [veh-eq*s]: {restored.total_system_time_veh_s:.6g}")
    if restored.sensitivity is None:
        print("TST directional sensitivity: not requested")
    else:
        value = restored.sensitivity
        print(f"TST direction at {value.control_node_id}: {value.direction.tolist()}")
        print(f"TST directional sensitivity [veh-eq*s]: {value.reverse_directional:.12g}")
        print(f"reverse/JVP agree: {value.reverse_jvp_agree}; stable check passes: {value.stable_scenario_passes}; event: {value.event_detected}")
    print(f"saved and reloaded renderer-facing bundle: {args.bundle}")
    print("Matplotlib figures begin in Component A1.2 after this checkpoint.")


if __name__ == "__main__":
    main()
