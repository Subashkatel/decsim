"""Gate 7 P6b: BurstEscalationDetector focused tests.

Predeclaration: docs/validation/2026-07-04-gate7-p6-predeclaration.md.
Unit level: warmup gating, escalation math, quorum, latch, and
baseline self-poisoning exclusion.
"""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.metrics import BurstEscalationDetector

PATCHES = ["a", "b", "c", "d"]


def make(**kw):
    args = dict(z=6.0, baseline_bins=50, warmup_bins=10, patch_quorum=3)
    args.update(kw)
    return BurstEscalationDetector(PATCHES, **args)


def feed_flat(det, bins, value=50):
    for _ in range(bins):
        det.ingest_bin({p: value for p in PATCHES})


def test_no_escalation_during_warmup():
    det = make()
    for _ in range(9):                        # below warmup_bins=10
        det.ingest_bin({p: 50 for p in PATCHES})
    assert not det.ingest_bin({p: 10_000 for p in PATCHES})
    assert not det.fired                      # history too short to judge


def test_quorum_requires_simultaneous_patches():
    det = make()
    feed_flat(det, 20)
    # two hot patches: escalations recorded, but no fire (quorum 3)
    assert not det.ingest_bin({"a": 10_000, "b": 10_000, "c": 50, "d": 50})
    assert det.escalations and len(det.escalations[-1][1]) == 2
    assert not det.fired
    # three hot patches in ONE bin: fire
    assert det.ingest_bin({"a": 10_000, "b": 10_000, "c": 10_000, "d": 50})
    assert det.fired and set(det.fired_patches) == {"a", "b", "c"}


def test_fire_latches_at_first_bin():
    det = make()
    feed_flat(det, 20)
    det.ingest_bin({p: 10_000 for p in PATCHES})
    first = det.fired_bin
    det.ingest_bin({p: 10_000 for p in PATCHES})
    assert det.fired_bin == first


def test_constant_stream_never_fires():
    det = make()
    feed_flat(det, 300)
    assert not det.fired and not det.escalations


def test_slow_ramp_blind_spot_is_a_pinned_design_property():
    """Codex P6 review finding: with a NOISY baseline (sigma > 0), a
    sub-threshold slow ramp ratchets the trailing baseline and never
    fires. Pinned as the documented design property (abrupt-onset
    detector), NOT silently relied upon. (A zero-variance baseline
    makes ANY step escalate — hence the jittered history here.)"""
    det = make()
    for i in range(60):                        # mu 50, sigma 7 jitter
        det.ingest_bin({p: 50 + (7 if i % 2 else -7) for p in PATCHES})
    for step in range(30):                     # 55 -> 200, +5 per bin
        det.ingest_bin({p: 55 + 5 * step for p in PATCHES})
    assert not det.fired
    assert not det.escalations


def test_escalated_bins_do_not_poison_the_baseline():
    """A long burst must not raise its own reference: counts stay hot
    relative to the PRE-burst baseline for the full burst."""
    det = make(patch_quorum=4)
    feed_flat(det, 60)
    hot_bins = 0
    for _ in range(40):                       # sustained 4x elevation
        det.ingest_bin({p: 200 for p in PATCHES})
        hot_bins += bool(det.escalations
                         and det.escalations[-1][0] == det.bin_index)
    assert hot_bins == 40, "burst poisoned its own baseline"
    assert det.fired


@pytest.mark.parametrize("parameters, name", [
    ({"patches": []}, "patches"),
    ({"patches": ["a", "a"]}, "patches"),
    ({"baseline_bins": 0}, "baseline_bins"),
    ({"warmup_bins": 0}, "warmup_bins"),
    ({"patch_quorum": 0}, "patch_quorum"),
    ({"patch_quorum": len(PATCHES) + 1}, "patch_quorum"),
    ({"z": -0.1}, "z"),
    ({"z": float("nan")}, "z"),
    ({"z": float("inf")}, "z"),
])
def test_burst_detector_rejects_undefined_domains(parameters, name):
    parameters = dict(parameters)
    patches = parameters.pop("patches", PATCHES)

    with pytest.raises(ValueError, match=name):
        BurstEscalationDetector(patches, **parameters)
