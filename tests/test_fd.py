"""Component 0.1 reference, unit, and rejection checks."""

from dataclasses import FrozenInstanceError
import math

import pytest

from diff_minimal_meso.fd import (
    LinkGeometry,
    TriangularFD,
    capacity_veh_per_s,
    critical_density,
    derive_link_fd,
    jam_storage_veh,
    travel_times_s,
)


RTOL = 1.0e-10
ATOL = 1.0e-12


@pytest.fixture
def hand_fd() -> TriangularFD:
    return TriangularFD(
        free_speed_mps=15.0,
        backward_wave_speed_mps=5.0,
        jam_density_veh_per_m=0.15,
    )


def test_approved_hand_anchor(hand_fd: TriangularFD) -> None:
    geometry = LinkGeometry(length_m=300.0)
    result = derive_link_fd(hand_fd, geometry, dt_s=10.0)

    assert result.critical_density_veh_per_m == pytest.approx(
        0.0375, rel=RTOL, abs=ATOL
    )
    assert result.capacity_veh_per_s == pytest.approx(
        0.5625, rel=RTOL, abs=ATOL
    )
    assert result.jam_storage_veh == pytest.approx(45.0, rel=RTOL, abs=ATOL)
    assert result.free_flow_time_s == pytest.approx(20.0, rel=RTOL, abs=ATOL)
    assert result.backward_wave_time_s == pytest.approx(
        60.0, rel=RTOL, abs=ATOL
    )
    assert result.capacity_veh_per_s * 10.0 == pytest.approx(
        5.625, rel=RTOL, abs=ATOL
    )


@pytest.mark.parametrize(
    ("v", "w", "jam_density"),
    [
        (15.0, 5.0, 0.15),
        (30.0, 3.0, 0.20),
        (7.5, 7.5, 0.08),
    ],
)
def test_triangular_fd_identities(v: float, w: float, jam_density: float) -> None:
    fd = TriangularFD(v, w, jam_density)
    k_c = critical_density(fd)
    capacity = capacity_veh_per_s(fd)

    assert 0.0 < k_c < jam_density
    assert capacity == pytest.approx(v * k_c, rel=RTOL, abs=ATOL)
    assert capacity == pytest.approx(
        w * (jam_density - k_c), rel=RTOL, abs=ATOL
    )


def test_geometry_storage_and_travel_time_units(hand_fd: TriangularFD) -> None:
    geometry = LinkGeometry(300.0)

    assert jam_storage_veh(geometry, hand_fd) == pytest.approx(45.0)
    assert travel_times_s(geometry, hand_fd) == pytest.approx((20.0, 60.0))


def test_causal_boundary_allows_time_step_equal_to_minimum_travel_time(
    hand_fd: TriangularFD,
) -> None:
    result = derive_link_fd(hand_fd, LinkGeometry(300.0), dt_s=20.0)
    assert result.capacity_veh_per_s * 20.0 <= result.jam_storage_veh


def test_reject_time_step_larger_than_free_flow_time(hand_fd: TriangularFD) -> None:
    with pytest.raises(ValueError, match="dt_s must not exceed"):
        derive_link_fd(hand_fd, LinkGeometry(300.0), dt_s=20.0001)


def test_reject_time_step_larger_than_backward_wave_time() -> None:
    fd = TriangularFD(
        free_speed_mps=5.0,
        backward_wave_speed_mps=15.0,
        jam_density_veh_per_m=0.15,
    )
    with pytest.raises(ValueError, match="dt_s must not exceed"):
        derive_link_fd(fd, LinkGeometry(300.0), dt_s=20.0001)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize(
    "field_name",
    ["free_speed_mps", "backward_wave_speed_mps", "jam_density_veh_per_m"],
)
def test_reject_invalid_fd_primitives(field_name: str, bad_value: float) -> None:
    values = {
        "free_speed_mps": 15.0,
        "backward_wave_speed_mps": 5.0,
        "jam_density_veh_per_m": 0.15,
    }
    values[field_name] = bad_value
    with pytest.raises(ValueError, match=field_name):
        TriangularFD(**values)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_reject_invalid_geometry(bad_value: float) -> None:
    with pytest.raises(ValueError, match="length_m"):
        LinkGeometry(bad_value)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_reject_invalid_time_step(hand_fd: TriangularFD, bad_value: float) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        derive_link_fd(hand_fd, LinkGeometry(300.0), bad_value)


@pytest.mark.parametrize(
    "constructor,args",
    [
        (TriangularFD, (True, 5.0, 0.15)),
        (LinkGeometry, (True,)),
    ],
)
def test_reject_boolean_as_structural_number(constructor, args) -> None:
    with pytest.raises(TypeError):
        constructor(*args)


def test_records_are_immutable(hand_fd: TriangularFD) -> None:
    geometry = LinkGeometry(300.0)
    result = derive_link_fd(hand_fd, geometry, 10.0)

    with pytest.raises(FrozenInstanceError):
        hand_fd.free_speed_mps = 20.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        geometry.length_m = 500.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.capacity_veh_per_s = 1.0  # type: ignore[misc]

