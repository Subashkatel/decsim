from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Optional

from .wiring import build_and_run
from .metrics import LogicalErrorRate

# ==================================================================================
# SAMPLING
# A logical error rate is a property of MANY shots, but one build_and_run() is ONE shot
# (a fresh engine drained once). So the rate lives in a multi-shot harness that runs the
# full engine once per shot and tallies the per-shot LogicalErrorRate verdicts -- with a
# proper confidence interval, because a naive errors/shots bar (sqrt(p(1-p)/n)) under-covers
# and can leave [0,1] when errors are few (sinter reports a real interval for the same reason).
# ==================================================================================


def wilson_interval(errors: int, shots: int, confidence: float = 0.95):
    """Wilson score interval for a Binomial error count. Correct coverage even when `errors`
    is small (where the normal-approx SEM fails); never leaves [0, 1]. Returns
    (p_hat, low, high)."""
    if shots <= 0:
        return (0.0, 0.0, 0.0)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = errors / shots
    denom = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denom
    half = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def logical_error_rate(ops, *, shots: int, device, score_op: Optional[int] = None,
                       confidence: float = 0.95,
                       on_shot: Optional[Callable[[int, object, object], None]] = None,
                       **build_kwargs) -> dict:
    """Run a memory experiment `shots` times through the FULL engine and return its logical
    error rate with a Wilson confidence interval.

    ops:        the operation list; the scored op carries a noisy stim circuit
                (e.g. from NoiseModel.circuit(...)).
    device:     a StimDevice -- the seeded sample stream, REUSED across all shots. It re-samples
                on each begin_operation, so successive runs draw successive shots from one
                deterministic stream: one device in -> one reproducible shard out (distinct
                seeds give mergeable shards, the experiments' shard-and-sum pattern).
    score_op:   which operation id to score (default: the single op that produced a logical
                value; raises if more than one did, so the choice is never silent).
    confidence: two-sided interval level (default 95%).
    on_shot:    optional callback(shot_index, cluster, device) after each run -- e.g. to also
                decode the same shot globally (device._dets[op]) and tally windowed-vs-global
                agreement, the seed-robust validation anchor.
    build_kwargs: forwarded to build_and_run (decoder, code, scheme, rounds_policy, num_units,
                d, ...). verbose is forced off.

    Returns {"shots", "errors", "ler", "ci_low", "ci_high", "confidence", "score_op"}.
    """
    errors = 0
    resolved = score_op
    for s in range(shots):
        res = build_and_run(ops=ops, device=device, verbose=False, **build_kwargs)
        cluster = res["cluster"]
        verdicts = LogicalErrorRate(cluster, device).verdicts()
        if not verdicts:
            raise ValueError(
                "no operation produced a logical value -- the device must be a StimDevice "
                "(real syndromes) and the decoder a failure-reporting decoder (e.g. "
                "PyMatchingDecoder), not a timing-only stub")
        if resolved is None:
            if len(verdicts) != 1:
                raise ValueError(f"score_op is ambiguous: ops {sorted(verdicts)} each produced "
                                 "a logical value; pass score_op= to pick one")
            resolved = next(iter(verdicts))
        if resolved not in verdicts:
            raise ValueError(f"score_op={resolved} produced no logical value this shot "
                             f"(got {sorted(verdicts)})")
        errors += verdicts[resolved]["error"]
        if on_shot is not None:
            on_shot(s, cluster, device)
    p, low, high = wilson_interval(errors, shots, confidence)
    return {"shots": shots, "errors": errors, "ler": p, "ci_low": low, "ci_high": high,
            "confidence": confidence, "score_op": resolved}
