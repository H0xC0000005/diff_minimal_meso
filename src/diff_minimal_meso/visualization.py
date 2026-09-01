"""Fixed headless Matplotlib renderer for Milestone 0_a1 Candidate A."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

_CACHE_DIR = Path(tempfile.gettempdir()) / "diff_minimal_meso_matplotlib_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))
(Path(os.environ["XDG_CACHE_HOME"])).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from .visualization_data import VisualizationBundle, validate_visualization_bundle


FIGURE_SIZE_IN = (12.0, 8.0)
DPI = 100
FONT_FAMILY = "DejaVu Sans"
PNG_METADATA = {"Software": "diff_minimal_meso M0_a1"}


@dataclass(frozen=True, slots=True)
class FramePlotData:
    frame_index: int
    time_s: float
    terminal: bool
    occupancy_ratio: tuple[float, ...]
    link_outflow_ratio: tuple[float, ...] | None
    source_queue_veh: tuple[float, ...]
    node_total_flow_veh: tuple[float, ...] | None


def frame_plot_data(bundle: VisualizationBundle, frame_index: int) -> FramePlotData:
    """Return literal artist inputs, preserving boundary/interval semantics."""

    validate_visualization_bundle(bundle)
    terminal_index = bundle.boundary_time_s.numel() - 1
    if frame_index < 0 or frame_index > terminal_index:
        raise IndexError("frame_index must be a boundary index in [0,T]")
    terminal = frame_index == terminal_index
    outflow_ratio = None
    node_flow = None
    if not terminal:
        outflow_ratio = tuple(
            (bundle.link_outflow_veh[frame_index] / bundle.link_capacity_step_veh).tolist()
        )
        node_flow = tuple(bundle.node_total_flow_veh[frame_index].tolist())
    return FramePlotData(
        frame_index,
        float(bundle.boundary_time_s[frame_index]),
        terminal,
        tuple(bundle.occupancy_ratio[frame_index].tolist()),
        outflow_ratio,
        tuple(bundle.source_queue_veh[frame_index].tolist()),
        node_flow,
    )


def _status_text(bundle: VisualizationBundle, data: FramePlotData) -> str:
    if data.terminal:
        interval = "terminal boundary — no interval flow overlay"
        active = "active constraints: N/A at terminal boundary"
    else:
        end = float(bundle.boundary_time_s[data.frame_index + 1])
        interval = f"interval {data.frame_index}: [{data.time_s:g}, {end:g}] s"
        labels = bundle.active_regime_labels[data.frame_index]
        active = "active constraints:\n" + ("\n".join(labels) if labels else "none")
    sensitivity = bundle.sensitivity
    if sensitivity is None:
        sensitivity_text = "d(TST)/dg · d: not requested"
    else:
        warning = "event/regime warning" if sensitivity.event_detected else "stable-regime check passed"
        sensitivity_text = (
            "d(TST)/dg · d\n"
            f"node={sensitivity.control_node_id}, g={sensitivity.physical_green.tolist()}\n"
            f"d={sensitivity.direction.tolist()}\n"
            f"reverse={sensitivity.reverse_directional:.6g}, JVP={sensitivity.jvp_directional:.6g}\n"
            f"{warning}"
        )
    return (
        f"boundary {data.frame_index} at t={data.time_s:g} s\n{interval}\n\n"
        f"TST={bundle.total_system_time_veh_s:.6g} veh-eq·s\n"
        f"max |mass residual|={bundle.conservation_residual_veh.abs().max().item():.3e} veh-eq\n\n"
        f"{active}\n\n{sensitivity_text}"
    )


def render_frame(bundle: VisualizationBundle, frame_index: int):
    """Build the fixed four-panel Candidate-A figure for one boundary."""

    data = frame_plot_data(bundle, frame_index)
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 9})
    figure = plt.figure(figsize=FIGURE_SIZE_IN, dpi=DPI, layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.2, 1.0), height_ratios=(1.0, 1.0))
    topology = figure.add_subplot(grid[0, 0])
    heatmap = figure.add_subplot(grid[1, 0])
    accounting = figure.add_subplot(grid[0, 1])
    status = figure.add_subplot(grid[1, 1])

    xy = bundle.layout.vertex_xy.numpy()
    segments = np.stack(
        (xy[bundle.layout.link_tail_index.numpy()], xy[bundle.layout.link_head_index.numpy()]),
        axis=1,
    )
    if data.link_outflow_ratio is None:
        widths = np.full(len(bundle.link_ids), 2.0)
    else:
        widths = 1.5 + 5.0 * np.clip(np.asarray(data.link_outflow_ratio), 0.0, 1.0)
    collection = LineCollection(
        segments,
        array=np.asarray(data.occupancy_ratio),
        cmap="viridis",
        norm=Normalize(0.0, 1.0),
        linewidths=widths,
        capstyle="round",
    )
    topology.add_collection(collection)
    topology.scatter(xy[:, 0], xy[:, 1], s=28, color="black", zorder=3)
    for index, label in enumerate(bundle.layout.link_labels):
        midpoint = segments[index].mean(axis=0)
        topology.text(midpoint[0], midpoint[1], label, fontsize=8, ha="center", va="bottom")
    for source, vertex in enumerate(bundle.layout.source_vertex_index.tolist()):
        topology.text(xy[vertex, 0], xy[vertex, 1], f"Q={data.source_queue_veh[source]:.2g}", color="tab:red", va="top")
    for node, vertex in enumerate(bundle.layout.node_vertex_index.tolist()):
        lines = []
        if data.node_total_flow_veh is not None:
            lines.append(f"F={data.node_total_flow_veh[node]:.2g}")
        green = bundle.physical_green_by_node[node]
        if green is not None:
            lines.append(f"g={[round(value, 3) for value in green.tolist()]}")
        if lines:
            topology.text(
                xy[vertex, 0], xy[vertex, 1], "\n".join(lines),
                color="tab:blue", ha="center", va="top", fontsize=8,
            )
    topology.autoscale()
    topology.margins(0.18)
    topology.set_aspect("equal", adjustable="datalim")
    topology.set_title("Network state: color=occupancy/storage; width=outflow/(C·dt)")
    topology.set_xlabel("layout x (drawing coordinate)")
    topology.set_ylabel("layout y (drawing coordinate)")
    figure.colorbar(collection, ax=topology, label="occupancy / storage", fraction=0.046)

    image = heatmap.imshow(
        bundle.occupancy_ratio.T.numpy(), aspect="auto", origin="lower",
        vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest",
    )
    heatmap.axvline(data.frame_index, color="white", linewidth=1.5)
    heatmap.set_yticks(range(len(bundle.link_ids)), labels=bundle.layout.link_labels)
    heatmap.set_xlabel("boundary index")
    heatmap.set_ylabel("link")
    heatmap.set_title("Occupancy/storage history")
    figure.colorbar(image, ax=heatmap, label="occupancy / storage", fraction=0.046)

    times = bundle.boundary_time_s.numpy()
    accounting.plot(times, bundle.source_queue_veh.sum(dim=1).numpy(), label="source queue")
    accounting.plot(times, bundle.cumulative_sink_veh.sum(dim=1).numpy(), label="cumulative sink exit")
    accounting.plot(times, bundle.occupancy_veh.sum(dim=1).numpy(), label="on-network occupancy")
    accounting.axvline(data.time_s, color="black", linestyle="--", linewidth=1)
    accounting.set_xlabel("time [s]")
    accounting.set_ylabel("traffic mass [veh-eq]")
    accounting.set_title("Traffic accounting")
    accounting.legend(loc="best", fontsize=8)
    residual_axis = accounting.twinx()
    residual_axis.plot(times, bundle.conservation_residual_veh.numpy(), color="tab:red", alpha=0.65, label="mass residual")
    residual_axis.set_ylabel("residual [veh-eq]", color="tab:red")

    status.axis("off")
    status.text(0.0, 1.0, _status_text(bundle, data), va="top", ha="left", family="monospace", fontsize=8)
    status.set_title("Frame status and TST sensitivity")
    qualifier = "terminal boundary" if data.terminal else f"state k={data.frame_index} + interval k={data.frame_index}"
    figure.suptitle(f"Milestone 0_a1 headless visualizer — {qualifier}", fontsize=14)
    return figure


def save_frame(bundle: VisualizationBundle, frame_index: int, path: str | Path) -> None:
    figure = render_frame(bundle, frame_index)
    try:
        figure.savefig(path, dpi=DPI, metadata=PNG_METADATA, facecolor="white")
    finally:
        plt.close(figure)
