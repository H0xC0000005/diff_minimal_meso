"""Independent scalar references for the approved Component 0.4 equations."""

from __future__ import annotations


def ordinary_orca(
    beta: list[list[float]],
    demand: list[float],
    capacity_step: list[float],
    receiving: list[float],
) -> tuple[list[float], list[list[float]], list[int], list[tuple[int, ...]]]:
    """Scalar Branch-T solver organized around rows rather than movement vectors."""

    input_count = len(beta)
    output_count = len(receiving)
    unresolved = {i for i, value in enumerate(demand) if value != 0.0}
    accepted = [0.0] * input_count
    pivots: list[int] = []
    tie_sets: list[tuple[int, ...]] = []
    incumbent: int | None = None
    while unresolved:
        restriction: dict[int, float] = {}
        for j in range(output_count):
            residual = receiving[j] - sum(
                beta[i][j] * accepted[i]
                for i in range(input_count)
                if i not in unresolved
            )
            denominator = sum(
                beta[i][j] * capacity_step[i]
                for i in unresolved
                if beta[i][j] > 0.0
            )
            if denominator > 0.0:
                restriction[j] = residual / denominator
        if not restriction:
            for i in tuple(unresolved):
                accepted[i] = demand[i]
                unresolved.remove(i)
            break
        minimum = min(restriction.values())
        tied = tuple(j for j, value in restriction.items() if value == minimum)
        if len(tied) > 1:
            tie_sets.append(tied)
        pivot = incumbent if incumbent in tied else min(tied)
        incumbent = pivot
        pivots.append(pivot)
        competitors = {i for i in unresolved if beta[i][pivot] > 0.0}
        low = {
            i
            for i in competitors
            if demand[i] <= minimum * capacity_step[i]
        }
        fixed = low or competitors
        for i in fixed:
            accepted[i] = demand[i] if low else minimum * capacity_step[i]
            unresolved.remove(i)
    movement = [
        [beta[i][j] * accepted[i] for j in range(output_count)]
        for i in range(input_count)
    ]
    return accepted, movement, pivots, tie_sets


def restricted_signal(
    beta: list[list[float]],
    demand: list[float],
    receiving: list[float],
    exposure: list[float],
    saturation_rate: list[float],
    dt_s: float,
) -> tuple[list[float], list[list[float]]]:
    accepted = []
    for i, row in enumerate(beta):
        bounds = [demand[i], exposure[i] * saturation_rate[i] * dt_s]
        bounds.extend(
            exposure[i] * receiving[j] / value
            for j, value in enumerate(row)
            if value > 0.0
        )
        accepted.append(min(bounds))
    movement = [
        [beta[i][j] * accepted[i] for j in range(len(receiving))]
        for i in range(len(beta))
    ]
    return accepted, movement
