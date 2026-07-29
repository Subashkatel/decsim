"""Live stream feedback tests.

These tests cover the SWIPER-style timing case where feedback waits stretch the
same syndrome stream that produced the blocking operation.
"""

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.decoders import PresetLatencyDecoder
from decsim.detector_error_model import (
    FaultRepresentation,
    NO_FAULT_MODEL_REQUIRED,
)
from decsim.devices import TimingOnlyDevice
from decsim.message import DecodeResult, Operation
from decsim.layouts import UniformLayout
from decsim.planner import PerOpRounds
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate
from decsim.policies import from_mode


def _live_stream_pair():
    stream = Operation(0, "live-stream", (0,), clifford=True, patches=(0,))
    first = Operation(
        1,
        "A:T(q0)",
        (0,),
        clifford=False,
        consumes_magic_state=False,
        patches=(0,),
        stream_id=stream.id,
        has_successor=True,
    )
    second = Operation(
        2,
        "B:T(q0)",
        (0,),
        clifford=False,
        consumes_magic_state=False,
        patches=(0,),
        predecessors=(first.id,),
        stream_id=stream.id,
        blocked_by=first.id,
    )
    return stream, [first, second]


def _live_stream_pair_with_circuit(circuit):
    stream, operations = _live_stream_pair()
    stream.circuit = circuit
    for operation in operations:
        operation.circuit = circuit
    return stream, operations


class _RecordingDecoder:
    """Wrap a real decoder and record the syndrome sizes it decoded."""

    def __init__(self, inner):
        self.inner = inner
        self.fault_model_requirement = inner.fault_model_requirement
        self.rows_seen: list[tuple[int, int]] = []

    def latency(self, job):
        return self.inner.latency(job)

    def decode(self, job):
        payload_bits = sum(
            len(payload.bits)
            for payload in job.payloads
            if payload.bits is not None
        )
        model_rows = (
            job.dem.require_faults(FaultRepresentation.GRAPHLIKE).check.shape[0]
            if job.dem is not None else 0
        )
        self.rows_seen.append((payload_bits, model_rows))
        return self.inner.decode(job)


class _FunctionalDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def latency(self, job):
        return 0

    def decode(self, job):
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_observables=(1,),
        )


def test_feedback_idle_rounds_extend_the_live_stream():
    """Feedback idle rounds become real stream rounds in extend_stream mode."""
    stream, operations = _live_stream_pair()
    code = SurfaceCodeModel(d=3, commit_rounds_override=2, buffer_rounds_override=1)
    rounds = {operations[0].id: 2, operations[1].id: 2}

    result = simulate(RunSpec(
                 ops=operations,
                 dynamic_streams=[stream],
                 idle_policy=from_mode("extend_stream"),
                 device=TimingOnlyDevice(),
                 code=code,
                 rounds_policy=PerOpRounds(rounds),
                 scheme=SlidingWindowScheme(),
                 decoder=PresetLatencyDecoder(2.0),
                 num_units=1,
                 round_us=1.0,
             ), verbose=False)

    cluster = result.window_manager
    chip = result.chip
    caller_first, caller_second = operations
    offsets = result.result.stream_offsets()
    first_offset = offsets[caller_first.id]
    second_offset = offsets[caller_second.id]

    assert chip.done_bodies == {caller_first.id, caller_second.id}
    assert caller_first.stream_offset is None
    assert caller_second.stream_offset is None
    assert first_offset == 0
    assert second_offset is not None
    assert second_offset > rounds[caller_first.id]
    assert cluster.rounds_arrived[stream.id] == (
        second_offset + rounds[caller_second.id]
    )
    assert cluster.committed_stream_round_count(stream.id) == cluster.rounds_arrived[stream.id]
    assert cluster.window_count[stream.id] > 1
    assert len(cluster.committed_windows) == cluster.total_windows


def test_feedback_idle_rounds_use_frozen_patch_cadence():
    class FreezeDetectingLayout(UniformLayout):
        def __init__(self, code):
            super().__init__(code)
            self.patch_selection_calls = 0

        def code_for_patch(self, patch_id):
            self.patch_selection_calls += 1
            if self.patch_selection_calls > 1:
                raise AssertionError(
                    "runtime consulted the live patch selector"
                )
            return self.code

    stream, operations = _live_stream_pair()
    code = SurfaceCodeModel(
        d=3,
        round_us=1.0,
        commit_rounds_override=2,
        buffer_rounds_override=1,
    )
    layout = FreezeDetectingLayout(code)
    rounds = {operations[0].id: 2, operations[1].id: 2}

    result = simulate(RunSpec(
        ops=operations,
        dynamic_streams=[stream],
        idle_policy=from_mode("extend_stream"),
        device=TimingOnlyDevice(),
        layout=layout,
        rounds_policy=PerOpRounds(rounds),
        scheme=SlidingWindowScheme(),
        decoder=PresetLatencyDecoder(2.0),
        num_units=1,
    ))

    assert result.chip.idle_rounds_emitted > 0
    assert layout.patch_selection_calls == 1


def test_committed_stream_round_count_releases_blocked_operation_before_stream_result():
    """The blocked operation is released once its stream segment has committed."""
    stream, operations = _live_stream_pair()
    code = SurfaceCodeModel(d=3, commit_rounds_override=2, buffer_rounds_override=1)
    rounds = {operations[0].id: 2, operations[1].id: 2}

    result = simulate(RunSpec(
                 ops=operations,
                 dynamic_streams=[stream],
                 idle_policy=from_mode("extend_stream"),
                 device=TimingOnlyDevice(),
                 code=code,
                 rounds_policy=PerOpRounds(rounds),
                 scheme=SlidingWindowScheme(),
                 decoder=PresetLatencyDecoder(2.0),
                 num_units=1,
                 round_us=1.0,
             ), verbose=False)

    chip = result.chip
    first, second = operations

    assert second.id in chip.decode_release_time
    assert chip.decode_release_time[second.id] >= chip.body_done_time[first.id]
    assert chip.decode_release_time[second.id] < chip.body_done_time[second.id]


def test_exact_live_segment_publishes_functional_vector_and_effect():
    stream, operations = _live_stream_pair()
    code = SurfaceCodeModel(
        d=3,
        commit_rounds_override=2,
        buffer_rounds_override=1,
    )
    rounds = {
        operations[0].id: 2,
        operations[1].id: 2,
    }

    result = simulate(RunSpec(
        ops=operations,
        dynamic_streams=[stream],
        idle_policy=from_mode("extend_stream"),
        device=TimingOnlyDevice(),
        code=code,
        rounds_policy=PerOpRounds(rounds),
        scheme=SlidingWindowScheme(),
        decoder=_FunctionalDecoder(),
        num_units=1,
        round_us=1.0,
    ), verbose=False)

    first, second = operations
    assert result.window_manager.op_results[first.id] == (1,)
    assert result.chip.applied_basis[second.id] == "X"
    assert result.chip.applied_frame_delta[second.id] != (0, 0)


def test_functional_live_segment_rejects_contribution_boundary_crossing():
    stream, operations = _live_stream_pair()
    code = SurfaceCodeModel(
        d=3,
        commit_rounds_override=3,
        buffer_rounds_override=1,
    )
    rounds = {
        operations[0].id: 2,
        operations[1].id: 2,
    }

    with pytest.raises(RuntimeError, match="functional.*boundary"):
        simulate(RunSpec(
            ops=operations,
            dynamic_streams=[stream],
            idle_policy=from_mode("extend_stream"),
            device=TimingOnlyDevice(),
            code=code,
            rounds_policy=PerOpRounds(rounds),
            scheme=SlidingWindowScheme(),
            decoder=_FunctionalDecoder(),
            num_units=1,
            round_us=1.0,
        ), verbose=False)


def test_real_syndrome_feedback_idle_rounds_extend_the_live_stream():
    """Stim payloads also work when feedback idle rounds extend a live stream."""
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from decsim.adapters.stim_device import StimDevice
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.stimcircuits import NoiseModel

    code = SurfaceCodeModel(d=3, commit_rounds_override=2, buffer_rounds_override=1)
    timing_stream, timing_operations = _live_stream_pair()
    operation_rounds = {
        timing_operations[0].id: 2,
        timing_operations[1].id: 2,
    }
    timing_result = simulate(RunSpec(
                        ops=timing_operations,
                        dynamic_streams=[timing_stream],
                        idle_policy=from_mode("extend_stream"),
                        device=TimingOnlyDevice(),
                        code=code,
                        rounds_policy=PerOpRounds(operation_rounds),
                        scheme=SlidingWindowScheme(),
                        decoder=PresetLatencyDecoder(2.0),
                        num_units=1,
                        round_us=1.0,
                    ), verbose=False)
    stream_round_count = timing_result.window_manager.rounds_arrived[timing_stream.id]

    circuit = NoiseModel.circuit_level(0.003).circuit(
        distance=code.distance,
        rounds=stream_round_count,
    )
    stream, operations = _live_stream_pair_with_circuit(circuit)
    rounds = {
        stream.id: stream_round_count,
        operations[0].id: operation_rounds[operations[0].id],
        operations[1].id: operation_rounds[operations[1].id],
    }
    decoder = _RecordingDecoder(PyMatchingDecoder(PresetLatencyDecoder(2.0)))

    result = simulate(RunSpec(
                 ops=operations,
                 dynamic_streams=[stream],
                 idle_policy=from_mode("extend_stream"),
                 device=StimDevice(),
                 code=code,
                 rounds_policy=PerOpRounds(rounds),
                 scheme=SlidingWindowScheme(),
                 decoder=decoder,
                 num_units=1,
                 round_us=1.0,
                 seed=13,
             ), verbose=False)

    cluster = result.window_manager
    chip = result.chip
    caller_first, caller_second = operations
    second_offset = result.result.stream_offsets()[caller_second.id]

    assert chip.done_bodies == {caller_first.id, caller_second.id}
    assert caller_first.stream_offset is None
    assert caller_second.stream_offset is None
    assert second_offset is not None
    assert second_offset > rounds[caller_first.id]
    assert cluster.rounds_arrived[stream.id] == (
        second_offset + rounds[caller_second.id]
    )
    assert cluster.rounds_arrived[stream.id] == stream_round_count
    assert cluster.committed_stream_round_count(stream.id) == cluster.rounds_arrived[stream.id]
    assert stream.id in cluster.op_results
    assert decoder.rows_seen
    assert all(payload_bits == model_rows for payload_bits, model_rows in decoder.rows_seen)
    assert any(payload_bits > 0 for payload_bits, _model_rows in decoder.rows_seen)


def test_real_syndrome_live_stream_rejects_inexact_circuit_length():
    """Real-syndrome streams fail clearly when the Stim circuit is longer than the run."""
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from decsim.adapters.stim_device import StimDevice
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.stimcircuits import NoiseModel

    code = SurfaceCodeModel(d=3, commit_rounds_override=2, buffer_rounds_override=1)
    circuit = NoiseModel.circuit_level(0.003).circuit(distance=code.distance, rounds=20)
    stream, operations = _live_stream_pair_with_circuit(circuit)
    rounds = {
        stream.id: 20,
        operations[0].id: 2,
        operations[1].id: 2,
    }

    with pytest.raises(RuntimeError, match="exact finite circuit"):
        simulate(RunSpec(
            ops=operations,
            dynamic_streams=[stream],
            idle_policy=from_mode("extend_stream"),
            device=StimDevice(),
            code=code,
            rounds_policy=PerOpRounds(rounds),
            scheme=SlidingWindowScheme(),
            decoder=PyMatchingDecoder(PresetLatencyDecoder(2.0)),
            num_units=1,
            round_us=1.0,
            seed=17,
        ), verbose=False)
