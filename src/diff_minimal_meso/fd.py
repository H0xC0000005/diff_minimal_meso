"""Traffic units and triangular fundamental-diagram identities.

All values in this module are immutable structural parameters.  Component 0.1
does not put fundamental-diagram parameters on a differentiation path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


def _positive_finite(name: str, value: float) -> float:
    """Return ``value`` as a float after strict structural validation."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class TriangularFD:
    """Primitive triangular-FD values.

    ``backward_wave_speed_mps`` is the positive magnitude of the backward wave
    speed, not a signed velocity.
    """

    free_speed_mps: float
    backward_wave_speed_mps: float
    jam_density_veh_per_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "free_speed_mps",
            _positive_finite("free_speed_mps", self.free_speed_mps),
        )
        object.__setattr__(
            self,
            "backward_wave_speed_mps",
            _positive_finite(
                "backward_wave_speed_mps", self.backward_wave_speed_mps
            ),
        )
        object.__setattr__(
            self,
            "jam_density_veh_per_m",
            _positive_finite("jam_density_veh_per_m", self.jam_density_veh_per_m),
        )


@dataclass(frozen=True, slots=True)
class LinkGeometry:
    """Structural geometry for one homogeneous LTM link."""

    length_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length_m", _positive_finite("length_m", self.length_m)
        )


@dataclass(frozen=True, slots=True)
class LinkFDParameters:
    """Derived, immutable FD quantities for one link."""

    critical_density_veh_per_m: float
    capacity_veh_per_s: float
    jam_storage_veh: float
    free_flow_time_s: float
    backward_wave_time_s: float


def critical_density(fd: TriangularFD) -> float:
    """Return triangular-FD critical density in veh-eq/m."""

    v = fd.free_speed_mps
    w = fd.backward_wave_speed_mps
    return (w / (v + w)) * fd.jam_density_veh_per_m


def capacity_veh_per_s(fd: TriangularFD) -> float:
    """Return the derived triangular-FD capacity in veh-eq/s."""

    return fd.free_speed_mps * critical_density(fd)


def jam_storage_veh(geometry: LinkGeometry, fd: TriangularFD) -> float:
    """Return link jam storage in continuous vehicle-equivalent mass."""

    return fd.jam_density_veh_per_m * geometry.length_m


def travel_times_s(geometry: LinkGeometry, fd: TriangularFD) -> tuple[float, float]:
    """Return free-flow and backward-wave link travel times in seconds."""

    return (
        geometry.length_m / fd.free_speed_mps,
        geometry.length_m / fd.backward_wave_speed_mps,
    )


def derive_link_fd(
    fd: TriangularFD, geometry: LinkGeometry, dt_s: float
) -> LinkFDParameters:
    """Derive link quantities and enforce the causal fixed-step condition.

    The approved scheduler evaluates end-of-interval shifted cumulative-count
    histories using only data available through the start of that interval.
    Therefore ``dt_s`` may not exceed either characteristic travel time.
    """

    dt = _positive_finite("dt_s", dt_s)
    tau_f, tau_b = travel_times_s(geometry, fd)
    minimum_travel_time = min(tau_f, tau_b)
    if dt > minimum_travel_time:
        raise ValueError(
            "dt_s must not exceed either link characteristic travel time: "
            f"dt_s={dt!r}, tau_f={tau_f!r}, tau_b={tau_b!r}"
        )

    k_c = critical_density(fd)
    capacity = capacity_veh_per_s(fd)
    storage = jam_storage_veh(geometry, fd)

    # This follows analytically from the triangular FD and causal time-step
    # condition.  Keep it as an internal consistency assertion, not a fallback.
    if capacity * dt > storage:
        raise AssertionError("derived step capacity exceeds link jam storage")

    return LinkFDParameters(
        critical_density_veh_per_m=k_c,
        capacity_veh_per_s=capacity,
        jam_storage_veh=storage,
        free_flow_time_s=tau_f,
        backward_wave_time_s=tau_b,
    )

