"""Regression tests for static and round-driven window model slicing.

SWIPER (arXiv:2412.05115 Sections 2.4 and 5.1, Figure 9) motivates
constructing windows as rounds arrive. These tests compare both construction
paths exactly and include a fixed-shot decoding regression. The sampled
agreement test is supporting evidence, not a proof for arbitrary models.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.stimcircuits import NoiseModel
from decsim.schemes import SlidingWindowScheme, ParallelWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    LINKED_FAULT_MODELS_REQUIRED,
    WindowSlicer,
    build_window_error_models,
    decode_windowed,
)
from decsim.mwpm_decoder import matching_window_decoder

D = 3


def _same(a, b):
    a_faults = a.require_faults(FaultRepresentation.GRAPHLIKE)
    b_faults = b.require_faults(FaultRepresentation.GRAPHLIKE)
    return (
        a.detector_ids == b.detector_ids
        and np.array_equal(a_faults.check, b_faults.check)
        and np.array_equal(a_faults.observables, b_faults.observables)
        and np.array_equal(a_faults.owned, b_faults.owned)
        and a_faults.future_flips == b_faults.future_flips
        and np.allclose(a_faults.priors, b_faults.priors)
    )


def _incremental(
    circ, plan, folded, round_count,
    fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
):
    slicer = WindowSlicer(
        circ,
        round_count=round_count,
        detector_rounds=folded,
        fault_model_requirement=fault_model_requirement,
    )
    out = []
    for k, win in enumerate(plan):
        if len(win) == 4:
            bl, cl, ch, bh = win
        else:
            cl, ch, bh = win
            bl = cl
        out.append(slicer.slice_window(bl, cl, ch, bh, is_last=(k == len(plan) - 1)))
    return out


@pytest.mark.parametrize("scheme,R", [(SlidingWindowScheme(), 24), (ParallelWindowScheme(), 30)])
def test_slicer_identical_to_static_builder(scheme, R):
    """Window-for-window, the incremental slicer reproduces build_window_error_models exactly --
    pins the two against drift for both the sliding and the A/B (two-sided-buffer) schemes."""
    circ = NoiseModel.circuit_level(0.003).circuit(distance=D, rounds=R)
    folded = {det: min(int(c[-1]) + 1, R) for det, c in circ.get_detector_coordinates().items()}
    planned = scheme.plan_operation(
        0,
        R,
        commit_round_count=D,
        buffer_round_count=D,
    ).windows
    plan = [
        (
            (window.commit_lo, window.commit_hi, min(window.buffer_hi, R))
            if window.buffer_lo == window.commit_lo
            else (
                window.buffer_lo,
                window.commit_lo,
                window.commit_hi,
                min(window.buffer_hi, R),
            )
        )
        for window in planned
    ]
    ref = build_window_error_models(
        circ,
        plan,
        round_count=R,
        detector_rounds=folded,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    inc = _incremental(circ, plan, folded, R)
    assert len(inc) == len(ref)
    assert all(_same(a, b) for a, b in zip(ref, inc))


def test_incremental_sliding_matches_global_on_fixed_shot_sample():
    """Pin agreement on one seeded surface-code sample without generalizing it."""
    R = 24
    circ = NoiseModel.circuit_level(0.003).circuit(distance=D, rounds=R)
    folded = {det: min(int(c[-1]) + 1, R) for det, c in circ.get_detector_coordinates().items()}
    plan = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0,
            R,
            commit_round_count=D,
            buffer_round_count=D,
        ).windows
    ]
    inc = _incremental(circ, plan, folded, R)
    gm = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    inner = matching_window_decoder()
    shots = 1500
    dets, obs = circ.compile_detector_sampler(seed=5).sample(shots, separate_observables=True)
    agree = 0
    for s in range(shots):
        pw = int(
            decode_windowed(
                inc,
                dets[s],
                inner,
                selected_fault_representation=FaultRepresentation.GRAPHLIKE,
            )[0]
        )
        pg = int(gm.decode(dets[s])[0])
        agree += (pw == pg)
    assert agree == shots


def test_matching_window_decoder_cache_survives_id_reuse():
    """Replication-run finding: the matching cache was keyed by id() with no
    eviction, so a model allocated at a dead model's address received the
    dead model's matching (shape errors or silently wrong corrections)."""
    import gc
    import numpy as np
    from decsim.detector_error_model import build_single_window_error_model
    from decsim.mwpm_decoder import matching_window_decoder

    import pathlib
    data = pathlib.Path(__file__).resolve().parent / "data"
    circ = stim.Circuit.from_file(str(data / "rsc-d3-r6-p0.005.stim"))
    inner = matching_window_decoder()

    def one_model(buffer_rounds):
        entry = (1, 3, min(3 + buffer_rounds, 6))
        return build_single_window_error_model(
            circ,
            entry,
            round_count=6,
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        )

    a = one_model(0)
    a_faults = a.require_faults(FaultRepresentation.GRAPHLIKE)
    n_dets_a = a_faults.check.shape[0]
    inner(a, np.zeros(n_dets_a, dtype=np.uint8))
    target = id(a_faults)
    del a_faults
    del a
    gc.collect()
    for _ in range(500):                       # try to force an id() collision
        b = one_model(3)                       # DIFFERENT window shape
        b_faults = b.require_faults(FaultRepresentation.GRAPHLIKE)
        if id(b_faults) == target:
            break
        del b_faults
        del b
        gc.collect()
    else:
        pytest.skip("could not provoke an id() reuse on this platform")
    # with the stale cache this raised ValueError (wrong matching graph)
    inner(b, np.zeros(b_faults.check.shape[0], dtype=np.uint8))


def test_slicer_rejects_partial_explicit_ownership_and_boolean_exclusions():
    rounds = 6
    circuit = NoiseModel.circuit_level(0.003).circuit(
        distance=D, rounds=rounds)
    folded = {
        detector_id: min(int(coordinates[-1]) + 1, rounds)
        for detector_id, coordinates in circuit.get_detector_coordinates().items()
    }
    slicer = WindowSlicer(
        circuit,
        round_count=rounds,
        detector_rounds=folded,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )
    with pytest.raises(TypeError, match="num_observables"):
        WindowSlicer(
            circuit,
            True,
            round_count=rounds,
            detector_rounds=folded,
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        )

    owners = {FaultRepresentation.GRAPHLIKE: set()}
    with pytest.raises(ValueError, match="must be supplied together"):
        slicer.slice_window(
            1, 1, 3, 6, is_last=True,
            explicitly_owned_faults=owners,
        )
    wrong_maps = {
        FaultRepresentation.GRAPHLIKE: set(),
        FaultRepresentation.PHYSICAL: set(),
    }
    with pytest.raises(ValueError, match="exactly match"):
        slicer.slice_window(
            1, 1, 3, 6, is_last=True,
            explicitly_owned_faults=wrong_maps,
            explicitly_prior_faults=wrong_maps,
        )
    with pytest.raises(TypeError, match="built-in integer"):
        slicer.slice_window(
            1, 1, 3, 6, is_last=True,
            fault_exclusion_ranges=((True, 2),),
        )


def _complete_model_equal(left, right):
    if (
        left.detector_ids != right.detector_ids
        or left.detector_coordinates != right.detector_coordinates
        or left.commit_lo != right.commit_lo
        or left.commit_hi != right.commit_hi
        or left.buffer_lo != right.buffer_lo
    ):
        return False
    for representation in (
        FaultRepresentation.GRAPHLIKE,
        FaultRepresentation.PHYSICAL,
    ):
        left_faults = left.require_faults(representation)
        right_faults = right.require_faults(representation)
        if not (
            left_faults.representation is right_faults.representation
            and np.array_equal(left_faults.check, right_faults.check)
            and np.array_equal(left_faults.observables, right_faults.observables)
            and np.array_equal(left_faults.priors, right_faults.priors)
            and np.array_equal(left_faults.owned, right_faults.owned)
            and left_faults.source_fault_ids == right_faults.source_fault_ids
            and dict(left_faults.future_flips) == dict(right_faults.future_flips)
            and dict(left_faults.boundary_flips) == dict(right_faults.boundary_flips)
        ):
            return False
    return np.array_equal(
        left.physical_to_graphlike_detector_projection,
        right.physical_to_graphlike_detector_projection,
    )


def test_production_sliding_static_and_incremental_linked_models_are_identical():
    rounds = 12
    circuit = NoiseModel.circuit_level(0.003).circuit(
        distance=D, rounds=rounds)
    folded = {
        detector_id: min(int(coordinates[-1]) + 1, rounds)
        for detector_id, coordinates in circuit.get_detector_coordinates().items()
    }
    planned = SlidingWindowScheme().plan_operation(
        0,
        rounds,
        commit_round_count=D,
        buffer_round_count=D,
    ).windows
    plan = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in planned
    ]
    static = build_window_error_models(
        circuit,
        plan,
        round_count=rounds,
        detector_rounds=folded,
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
    )
    incremental = _incremental(
        circuit,
        plan,
        folded,
        rounds,
        LINKED_FAULT_MODELS_REQUIRED,
    )
    assert len(static) == len(incremental)
    assert all(
        _complete_model_equal(left, right)
        for left, right in zip(static, incremental)
    )

    graphlike = static[0].graphlike_faults
    with pytest.raises(TypeError):
        graphlike.future_flips[0] = (1,)
    with pytest.raises(TypeError):
        graphlike.boundary_flips[0] = (1,)
