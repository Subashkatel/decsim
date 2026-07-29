"""Continuous-stream support in StimDevice (3b/5-real foundation).

A continuous patch is one circuit sampled ONCE per shot; its operation segments each serve
their local rounds from the shared sample at the right GLOBAL round (stream_offset + r), so the
whole stream is one detection record with one observable. Standalone ops are unchanged.

Requires stim."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")

from decsim.message import Operation
from decsim.adapters.stim_device import StimDevice
from decsim.stimcircuits import NoiseModel

D = 3
R1, R2 = 12, 12
R = R1 + R2


def _single_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    assert len(payloads) == 1
    return payloads[0]


def _continuous_circuit():
    return NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R)


def test_stream_segments_cover_the_whole_record_once():
    circ = _continuous_circuit()
    segA = Operation(0, "segA", (0,), circuit=circ, stream_id="s", stream_offset=0)
    segB = Operation(1, "segB", (0,), circuit=circ, stream_id="s", stream_offset=R1)
    dev = StimDevice(seed=4)
    dev.begin_operation(segA, R1)
    dev.begin_operation(segB, R2)                  # must NOT re-sample the stream
    # concatenate every segment's per-round bits in global order == the full detection record
    bits = []
    for r in range(1, R1 + 1):
        bits += list(_single_payload(dev, segA, r).bits)
    for r in range(1, R2 + 1):
        bits += list(_single_payload(dev, segB, r).bits)
    full = dev._dets["s"]
    assert len(bits) == len(full)
    assert np.array_equal(np.asarray(bits, np.uint8), np.asarray(full, np.uint8))
    # one observable truth shared across the stream; mirrored onto each segment id
    assert dev._truth[0] is dev._truth["s"] and dev._truth[1] is dev._truth["s"]


def test_stream_segment_serves_its_global_rounds():
    circ = _continuous_circuit()
    segA = Operation(0, "segA", (0,), circuit=circ, stream_id="s", stream_offset=0)
    segB = Operation(1, "segB", (0,), circuit=circ, stream_id="s", stream_offset=R1)
    dev = StimDevice(seed=7)
    dev.begin_operation(segA, R1)
    dev.begin_operation(segB, R2)
    # segB local round r serves global round R1+r: compare against the same
    # stream sampled as one segment (same stream_id, so the same shot)
    ref = StimDevice(seed=7)
    whole = Operation(9, "whole", (0,), circuit=circ, stream_id="s",
                      stream_offset=0)                   # local == global
    ref.begin_operation(whole, R)
    for r in range(1, R2 + 1):
        got = np.asarray(_single_payload(dev, segB, r).bits, np.uint8)
        want = np.asarray(_single_payload(ref, whole, R1 + r).bits, np.uint8)
        assert np.array_equal(got, want), r


def test_standalone_op_unchanged():
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    op = Operation(3, "mem", (0,), circuit=circ)          # no stream_id
    dev = StimDevice(seed=2)
    dev.begin_operation(op, R1)
    total = sum(len(_single_payload(dev, op, r).bits) for r in range(1, R1 + 1))
    assert total == circ.num_detectors                   # every detector emitted exactly once


def test_standalone_detector_rounds_use_the_supplied_resolved_count():
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    op = Operation(3, "shortened", (0,), circuit=circ)
    device = StimDevice(seed=2)

    device.begin_operation(op, 3)

    emitted = sum(
        len(_single_payload(device, op, round_index).bits)
        for round_index in range(1, 4)
    )
    assert emitted == circ.num_detectors
    assert len(_single_payload(device, op, 4).bits) == 0


def test_distinct_operations_draw_independent_shots():
    """Standalone operations use distinct per-identity sampling substreams."""
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    dev = StimDevice(seed=1234)
    first = Operation(0, "memA", (0,), circuit=circ)
    second = Operation(1, "memB", (1,), circuit=circ)
    dev.begin_operation(first, R1)
    dev.begin_operation(second, R1)
    assert not np.array_equal(dev._dets[0], dev._dets[1])


def test_operation_shots_do_not_depend_on_the_other_operations_present():
    """An operation's draw is keyed on its own identity, so adding,
    removing or reordering unrelated operations leaves it untouched."""
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    alone = StimDevice(seed=77)
    target = Operation(5, "target", (0,), circuit=circ)
    alone.begin_operation(target, R1)
    expected = np.asarray(alone._dets[5], np.uint8)

    crowded = StimDevice(seed=77)
    for op_id in (9, 2):                       # decoded first, and out of order
        other = Operation(op_id, f"other{op_id}", (0,), circuit=circ)
        crowded.begin_operation(other, R1)
    crowded.begin_operation(
        Operation(5, "target", (0,), circuit=circ),
        R1,
    )
    assert np.array_equal(np.asarray(crowded._dets[5], np.uint8), expected)


def test_sample_substreams_distinguish_identity_types():
    device = StimDevice(seed=77)
    assert device._sample_seed(1) != device._sample_seed("1")
    with pytest.raises(TypeError):
        device._sample_seed(True)
    with pytest.raises(TypeError):
        device._sample_seed(("stream", 1))


@pytest.mark.parametrize("explicit_seed", [0, 7])
def test_run_seed_rejects_an_explicit_stim_seed_including_zero(explicit_seed):
    device = StimDevice(seed=explicit_seed)

    with pytest.raises(
        ValueError,
        match=r"StimDevice.*explicit seed.*run root",
    ):
        device.reserve_run_seed(23)


def test_stim_run_seed_reservation_cancels_or_commits_without_a_draw():
    cancelled_device = StimDevice()
    cancelled = cancelled_device.reserve_run_seed(19)
    assert (cancelled.proposed_seed_source, cancelled.proposed_seed) == ("derived", 19)
    cancelled_device.cancel_run_seed(cancelled)

    committed_device = StimDevice()
    committed = committed_device.reserve_run_seed(19)
    committed_device.commit_run_seed(committed)
    with pytest.raises(ValueError, match=r"StimDevice.*already claimed"):
        committed_device.reserve_run_seed(19)

    operation = Operation(
        5,
        "target",
        (0,),
        circuit=NoiseModel.circuit_level(0.01).circuit(
            distance=D,
            rounds=R1,
        ),
    )
    committed_device.begin_operation(operation, R1)

    reference = StimDevice(seed=19)
    reference.begin_operation(operation, R1)
    assert np.array_equal(
        committed_device._dets[operation.id],
        reference._dets[operation.id],
    )
    assert np.array_equal(
        committed_device._truth[operation.id],
        reference._truth[operation.id],
    )


def test_stim_direct_draw_prevents_later_run_seed_binding_without_reset():
    operation = Operation(
        6,
        "used",
        (0,),
        circuit=NoiseModel.circuit_level(0.01).circuit(
            distance=D,
            rounds=R1,
        ),
    )
    device = StimDevice(seed=None)
    device.begin_operation(operation, R1)
    detectors_before = np.array(device._dets[operation.id], copy=True)
    truth_before = np.array(device._truth[operation.id], copy=True)

    with pytest.raises(ValueError, match=r"StimDevice.*already used"):
        device.reserve_run_seed(23)

    assert np.array_equal(device._dets[operation.id], detectors_before)
    assert np.array_equal(device._truth[operation.id], truth_before)


@pytest.mark.parametrize(
    ("explicit_seed", "expected_source", "expected_seed"),
    [
        (None, "entropy", None),
        (0, "explicit_local", 0),
    ],
)
def test_none_run_root_preserves_stim_entropy_or_explicit_local_seed(
    explicit_seed,
    expected_source,
    expected_seed,
):
    device = StimDevice(seed=explicit_seed)
    reservation = device.reserve_run_seed(None)

    assert (reservation.proposed_seed_source, reservation.proposed_seed) == (
        expected_source,
        expected_seed,
    )


def test_invalid_identity_is_rejected_before_cached_sampler_lookup():
    """A bool must not alias an already-cached integer identity."""
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    device = StimDevice(seed=77)
    device.begin_operation(
        Operation(1, "integer", (0,), circuit=circ),
        R1,
    )

    with pytest.raises(TypeError, match="stream_id must be a stable"):
        Operation(
            2,
            "boolean",
            (0,),
            circuit=circ,
            stream_id=True,
        )


def test_terminal_finalizer_rejects_an_unsampled_stream():
    device = StimDevice(
        terminal_detector_ids={"s": (0,)},
        terminal_data_bits={"s": 1},
    )
    finalizer = Operation(
        9, "terminal", (0,), stream_id="s", stream_offset=0,
        finalizes_stream_round=True,
        syndrome_fragment_index=1, syndrome_fragment_count=2,
    )
    with pytest.raises(RuntimeError, match="sampled stream"):
        device.finalize_stream_round(finalizer)


def test_unseeded_sampling_keeps_stim_owned_identity_behavior():
    class RecordingSampler:
        def __init__(self):
            self.sample_calls = 0

        def sample(self, *, shots, separate_observables):
            self.sample_calls += 1
            assert shots == 1
            assert separate_observables is True
            return np.zeros((1, 0), dtype=np.uint8), \
                np.zeros((1, 1), dtype=np.uint8)

    class RecordingCircuit:
        def __init__(self):
            self.compile_kwargs = []
            self.sampler = RecordingSampler()

        def compile_detector_sampler(self, **kwargs):
            self.compile_kwargs.append(kwargs)
            return self.sampler

        def get_detector_coordinates(self):
            return {}

    circ = RecordingCircuit()
    legacy_key = ("stream", 1)
    device = StimDevice(seed=None)
    first_segment = Operation(
        2,
        "unseeded",
        (0,),
        circuit=circ,
        stream_id=legacy_key,
        stream_offset=0,
    )
    later_segment = Operation(
        3,
        "unseeded-later",
        (0,),
        circuit=circ,
        stream_id=legacy_key,
        stream_offset=1,
    )

    device.begin_operation(first_segment, 1)
    device.begin_operation(later_segment, 1)
    assert circ.compile_kwargs == [{}]
    assert circ.sampler.sample_calls == 1
    assert legacy_key in device._dets
    assert device._dets[2] is device._dets[legacy_key]
    assert device._dets[3] is device._dets[legacy_key]


@pytest.mark.parametrize("integral_seed", [np.int64(1), np.uint64(1)])
def test_root_seed_accepts_numpy_integral_scalars(integral_seed):
    """Scientific sweep scalars retain Stim's integer seed semantics."""
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    operation = Operation(4, "numpy-seed", (0,), circuit=circ)

    device = StimDevice(seed=integral_seed)
    device.begin_operation(operation, R1)

    builtin_reference = StimDevice(seed=1)
    builtin_reference.begin_operation(operation, R1)
    assert np.array_equal(device._dets[4], builtin_reference._dets[4])
    assert np.array_equal(device._truth[4], builtin_reference._truth[4])


@pytest.mark.parametrize("invalid_seed", [-1, 1 << 64, 1.0, "1"])
def test_invalid_root_seed_retains_stim_validation(invalid_seed):
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    device = StimDevice(seed=invalid_seed)
    operation = Operation(1, "mem", (0,), circuit=circ)

    with pytest.raises(ValueError, match="64-bit unsigned integer"):
        device.begin_operation(operation, R1)


def test_stream_segments_still_share_one_shot():
    """Substreams are keyed on the shot identity the device already uses,
    so a stream's segments keep sharing one draw."""
    circ = _continuous_circuit()
    dev = StimDevice(seed=4)
    segA = Operation(0, "segA", (0,), circuit=circ, stream_id="s", stream_offset=0)
    segB = Operation(1, "segB", (0,), circuit=circ, stream_id="s", stream_offset=R1)
    dev.begin_operation(segA, R1)
    dev.begin_operation(segB, R2)
    assert np.array_equal(np.asarray(dev._dets[0], np.uint8),
                          np.asarray(dev._dets["s"], np.uint8))
    assert np.array_equal(np.asarray(dev._dets[1], np.uint8),
                          np.asarray(dev._dets["s"], np.uint8))
