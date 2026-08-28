from __future__ import annotations

import math

import pytest
import torch

from diff_minimal_meso.fd import LinkFDParameters
from diff_minimal_meso.ltm import (
    CumulativeLinkState,
    advance_link_state,
    compute_link_step,
)
from references.ltm_scalar_reference import link_step as scalar_link_step


DT = 10.0


def parameter(
    *, capacity: float = 0.5625, storage: float = 45.0, tau_f: float = 20.0,
    tau_b: float = 40.0,
) -> LinkFDParameters:
    return LinkFDParameters(
        critical_density_veh_per_m=0.01875,
        capacity_veh_per_s=capacity,
        jam_storage_veh=storage,
        free_flow_time_s=tau_f,
        backward_wave_time_s=tau_b,
    )


def state(n_in: list[list[float]], n_out: list[list[float]]) -> CumulativeLinkState:
    return CumulativeLinkState(
        torch.tensor(n_in, dtype=torch.float64),
        torch.tensor(n_out, dtype=torch.float64),
    )


def test_empty_link_limits() -> None:
    # Also mirrors the compatible empty-link/capacity-boundary cases visible in
    # UNsim v0.12.0 commit 5c396357, without importing or executing UNsim.
    result = compute_link_step(state([[0.0]], [[0.0]]), [parameter()], DT)
    assert result.sending.item() == 0.0
    assert result.receiving.item() == pytest.approx(5.625)
    assert result.occupancy.item() == 0.0
    assert result.sending_active.tolist() == [[True, False]]
    assert result.receiving_active.tolist() == [[False, True]]


def test_pulse_waits_for_free_flow_travel_time_and_keeps_fractional_mass() -> None:
    p = parameter()
    before = compute_link_step(state([[0.0], [0.125]], [[0.0], [0.0]]), [p], DT)
    eligible = compute_link_step(
        state([[0.0], [0.125], [0.125]], [[0.0], [0.0], [0.0]]), [p], DT
    )
    assert before.sending.item() == 0.0
    assert eligible.sending.item() == pytest.approx(0.125)


def test_fractional_history_interpolation() -> None:
    p = parameter(capacity=10.0, tau_f=15.0)
    result = compute_link_step(
        state([[0.0], [2.0], [6.0]], [[0.0], [0.0], [0.0]]), [p], DT
    )
    # k=2: q_f=30-15=15, hence 0.5*N_in[1]+0.5*N_in[2].
    assert result.sending.item() == pytest.approx(4.0)


def test_capacity_and_storage_ties_are_diagnosed() -> None:
    p = parameter(capacity=0.5, storage=5.0, tau_f=10.0, tau_b=10.0)
    result = compute_link_step(
        state([[0.0], [5.0]], [[0.0], [0.0]]), [p], DT
    )
    assert result.sending.item() == 5.0
    assert result.receiving.item() == 0.0
    assert result.sending_active.tolist() == [[True, True]]
    assert result.receiving_active.tolist() == [[True, False]]


def test_downstream_departures_release_receiving_space_after_wave_time() -> None:
    p = parameter(capacity=10.0, storage=45.0, tau_b=15.0)
    result = compute_link_step(
        state([[0.0], [20.0], [44.0]], [[0.0], [1.0], [1.0]]), [p], DT
    )
    # q_b=15 interpolates N_out to 1, leaving 2 vehicle-equivalents of room.
    assert result.receiving.item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("n_in", "n_out", "p"),
    [
        ([0.0], [0.0], parameter()),
        ([0.0, 0.125, 0.125], [0.0, 0.0, 0.0], parameter()),
        ([0.0, 2.0, 6.0], [0.0, 0.0, 0.0], parameter(capacity=10.0, tau_f=15.0)),
        ([0.0, 20.0, 44.0], [0.0, 1.0, 1.0], parameter(capacity=10.0, tau_b=15.0)),
    ],
)
def test_matches_independent_scalar_reference(
    n_in: list[float], n_out: list[float], p: LinkFDParameters
) -> None:
    production = compute_link_step(
        state([[x] for x in n_in], [[x] for x in n_out]), [p], DT
    )
    expected = scalar_link_step(
        n_in, n_out, DT, p.capacity_veh_per_s, p.jam_storage_veh,
        p.free_flow_time_s, p.backward_wave_time_s,
    )
    torch.testing.assert_close(
        torch.stack((production.sending[0], production.receiving[0], production.occupancy[0])),
        torch.tensor(expected, dtype=torch.float64), rtol=1e-10, atol=1e-12,
    )


def test_advance_is_out_of_place_and_conserves_boundary_counts() -> None:
    original = state([[0.0]], [[0.0]])
    p = parameter()
    limits = compute_link_step(original, [p], DT)
    updated = advance_link_state(
        original, torch.tensor([0.125], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64), limits, [p],
    )
    assert original.n_in.shape == (1, 1)
    assert updated.n_in.tolist() == [[0.0], [0.125]]
    assert updated.n_out.tolist() == [[0.0], [0.0]]
    assert (updated.n_in[-1] - updated.n_out[-1]).item() == 0.125


def test_rejects_flow_limit_violations() -> None:
    original = state([[0.0]], [[0.0]])
    p = parameter()
    limits = compute_link_step(original, [p], DT)
    with pytest.raises(ValueError, match="sending"):
        advance_link_state(
            original, torch.zeros(1, dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64), limits, [p],
        )
    with pytest.raises(ValueError, match="receiving"):
        advance_link_state(
            original, torch.tensor([6.0], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), limits, [p],
        )


@pytest.mark.parametrize(
    ("n_in", "n_out", "message"),
    [
        ([[0.0], [-1.0]], [[0.0], [0.0]], "nonnegative"),
        ([[0.0], [1.0], [0.5]], [[0.0], [0.0], [0.0]], "nondecreasing"),
        ([[0.0], [1.0]], [[0.0], [2.0]], "cannot exceed"),
    ],
)
def test_rejects_inadmissible_histories(n_in, n_out, message) -> None:
    with pytest.raises(ValueError, match=message):
        state(n_in, n_out)


def test_rejects_noncausal_delay_query() -> None:
    with pytest.raises(ValueError, match="after the current boundary"):
        compute_link_step(state([[0.0]], [[0.0]]), [parameter(tau_f=1.0)], DT)


def test_stable_fractional_branch_reverse_mode_and_central_difference() -> None:
    x = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    n_in = torch.stack((torch.tensor(0.0), torch.tensor(2.0), x)).reshape(3, 1)
    n_in = n_in.to(dtype=torch.float64)
    s = CumulativeLinkState(n_in, torch.zeros((3, 1), dtype=torch.float64))
    p = parameter(capacity=10.0, tau_f=15.0)
    value = compute_link_step(s, [p], DT).sending[0]
    (gradient,) = torch.autograd.grad(value, x)

    def evaluate(v: float) -> float:
        return compute_link_step(
            state([[0.0], [2.0], [v]], [[0.0], [0.0], [0.0]]), [p], DT
        ).sending.item()

    h = 1e-5
    finite_difference = (evaluate(6.0 + h) - evaluate(6.0 - h)) / (2.0 * h)
    assert value.grad_fn is not None
    assert gradient.item() == pytest.approx(0.5, rel=1e-10, abs=1e-12)
    assert finite_difference == pytest.approx(gradient.item(), rel=1e-9, abs=1e-10)
    assert math.isfinite(gradient.item())


def test_batched_links_preserve_shape_dtype_and_device() -> None:
    s = state([[0.0, 0.0], [1.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]])
    result = compute_link_step(s, [parameter(tau_f=10.0), parameter(tau_f=10.0)], DT)
    assert result.sending.shape == (2,)
    assert result.sending.dtype == torch.float64
    assert result.sending.device == s.n_in.device
    assert result.sending.tolist() == [1.0, 2.0]
