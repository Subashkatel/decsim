"""Multi-shot sampling helpers."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Optional

from .wiring import build_and_run
from .metrics import LogicalErrorRate


def wilson_interval(errors: int, shots: int, confidence: float = 0.95):
    """Wilson score interval for a binomial error count."""
    if shots <= 0:
        return (0.0, 0.0, 0.0)
    z_score = NormalDist().inv_cdf((1 + confidence) / 2)
    estimate = errors / shots
    denominator = 1 + z_score * z_score / shots
    center = (estimate + z_score * z_score / (2 * shots)) / denominator
    half_width = (
        z_score
        * math.sqrt(estimate * (1 - estimate) / shots
                    + z_score * z_score / (4 * shots * shots))
        / denominator
    )
    return (estimate, max(0.0, center - half_width), min(1.0, center + half_width))


def _shot_verdicts(ops, device, build_kwargs: dict) -> tuple:
    """Run one shot and return its cluster plus logical-error verdicts."""
    result = build_and_run(ops=ops, device=device, verbose=False, **build_kwargs)
    cluster = result["cluster"]
    verdicts = LogicalErrorRate(cluster, device).verdicts()
    if not verdicts:
        raise ValueError(
            "no operation produced a logical value. The device must be a StimDevice "
            "(real syndromes) and the decoder a failure-reporting decoder (e.g. "
            "PyMatchingDecoder), not a timing-only decoder")
    return cluster, verdicts


def _resolve_score_op(current_score_op: Optional[int], verdicts: dict) -> int:
    """Choose which operation's logical error verdict to count."""
    if current_score_op is not None:
        if current_score_op not in verdicts:
            raise ValueError(
                f"score_op={current_score_op} produced no logical value this shot "
                f"(got {sorted(verdicts)})")
        return current_score_op

    if len(verdicts) != 1:
        raise ValueError(
            f"score_op is ambiguous: ops {sorted(verdicts)} each produced "
            "a logical value; pass score_op= to pick one")
    return next(iter(verdicts))


def _logical_error_rate_result(*, shots: int, errors: int,
                               confidence: float, score_op: int) -> dict:
    """Format the final logical-error-rate result."""
    estimate, low, high = wilson_interval(errors, shots, confidence)
    return {
        "shots": shots,
        "errors": errors,
        "ler": estimate,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "score_op": score_op,
    }


def logical_error_rate(ops, *, shots: int, device, score_op: Optional[int] = None,
                       confidence: float = 0.95,
                       on_shot: Optional[Callable[[int, object, object], None]] = None,
                       **build_kwargs) -> dict:
    """Run a memory experiment many times and return LER with a Wilson interval."""
    errors = 0
    resolved = score_op
    for shot_index in range(shots):
        cluster, verdicts = _shot_verdicts(ops, device, build_kwargs)
        resolved = _resolve_score_op(resolved, verdicts)
        errors += verdicts[resolved]["error"]
        if on_shot is not None:
            on_shot(shot_index, cluster, device)

    return _logical_error_rate_result(
        shots=shots, errors=errors, confidence=confidence, score_op=resolved)
