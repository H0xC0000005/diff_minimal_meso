from __future__ import annotations

from pathlib import Path

import matplotlib
from PIL import Image
import pytest

from diff_minimal_meso.visualization import (
    DPI,
    FIGURE_SIZE_IN,
    frame_plot_data,
    render_frame,
    save_frame,
)
from diff_minimal_meso.visualization_data import load_scenario_config, run_visualization_case


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "milestone_0_a1"


@pytest.fixture(scope="module")
def spillback_bundle():
    return run_visualization_case(load_scenario_config(CONFIGS / "spillback_chain.json"))


def test_renderer_forces_headless_agg() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_frame_artist_inputs_equal_bundle_values(spillback_bundle) -> None:
    data = frame_plot_data(spillback_bundle, 3)
    assert data.occupancy_ratio == pytest.approx(spillback_bundle.occupancy_ratio[3].tolist())
    assert data.source_queue_veh == pytest.approx(spillback_bundle.source_queue_veh[3].tolist())
    assert data.link_outflow_ratio == pytest.approx(
        (spillback_bundle.link_outflow_veh[3] / spillback_bundle.link_capacity_step_veh).tolist()
    )
    assert not data.terminal


def test_terminal_frame_has_no_interval_overlay(spillback_bundle) -> None:
    data = frame_plot_data(spillback_bundle, 6)
    assert data.terminal
    assert data.link_outflow_ratio is None
    assert data.node_total_flow_veh is None
    figure = render_frame(spillback_bundle, 6)
    try:
        assert "terminal boundary" in figure._suptitle.get_text()
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_signal_figure_contains_node_flow_and_green() -> None:
    bundle = run_visualization_case(load_scenario_config(CONFIGS / "signalized_merge.json"))
    figure = render_frame(bundle, 1)
    try:
        topology_text = [value.get_text() for value in figure.axes[0].texts]
        assert any("F=" in value and "g=[0.4, 0.6]" in value for value in topology_text)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_saved_png_has_frozen_dimensions_and_is_readable(spillback_bundle, tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    save_frame(spillback_bundle, 4, path)
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.size == (int(FIGURE_SIZE_IN[0] * DPI), int(FIGURE_SIZE_IN[1] * DPI))
        assert image.mode == "RGBA"
        image.verify()


def test_frame_index_must_be_boundary(spillback_bundle) -> None:
    with pytest.raises(IndexError, match="boundary index"):
        frame_plot_data(spillback_bundle, 7)
