"""Independent standard-library scalar oracle for ledger rank arithmetic."""

from __future__ import annotations


def split_mass(
    mass: float, front_s: float, tail_s: float, head_mass: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    split_s = front_s + (head_mass / mass) * (tail_s - front_s)
    return (
        (head_mass, front_s, split_s),
        (mass - head_mass, split_s, tail_s),
    )


def discharge_intervals(
    masses: list[float], flow: float, front_s: float, tail_s: float
) -> list[tuple[float, float]]:
    cumulative = 0.0
    result = []
    for mass in masses:
        start = front_s + cumulative / flow * (tail_s - front_s)
        cumulative += mass
        end = front_s + cumulative / flow * (tail_s - front_s)
        result.append((start, end))
    return result


def exact_mergeable(
    left: tuple[int, int, float, float, float],
    right: tuple[int, int, float, float, float],
) -> bool:
    lr, lp, lm, lf, lt = left
    rr, rp, rm, rf, rt = right
    return lr == rr and lp == rp and lt == rf and (lt - lf) * rm == (rt - rf) * lm
