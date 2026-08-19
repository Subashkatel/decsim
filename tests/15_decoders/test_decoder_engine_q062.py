"""Behavior tests for the Q-062(c) timed decoder."""

import pytest

from decsim.config import us
from decsim.decoders.decoder_engine import (
    ALGORITHM_STAGE,
    DecoderStage,
    DecoderTiming,
    DecoderEngine,
)
from decsim.decoders.decoders import PerRoundDecoder, PresetLatencyDecoder
from decsim.engine import Engine
from decsim.message import DecodeJob, DecodeResult, DecoderRequestKey, DecoderTier

MHZ = 250.0
CYCLE = us(1 / MHZ)


def _timing(before=(DecoderStage("fetch", cycles_per_round=1),),
            after=(DecoderStage("release", cycles_per_job=1),),
            frequency_mhz=MHZ):
    return DecoderTiming(tuple(before), tuple(after), frequency_mhz)


class _RecordingInner:
    """Timing-only decoder that records the tick at which decode() ran."""

    fault_model_requirement = PresetLatencyDecoder().fault_model_requirement

    def __init__(self, engine, latency_us=2.0):
        self.engine = engine
        self.latency_us = latency_us
        self.decode_ticks = []

    def latency(self, job):
        return us(self.latency_us)

    def decode(self, job):
        self.decode_ticks.append(self.engine.now)
        return DecodeResult(job.op_id, job.window_id)


def _job(window_id=0, n_rounds=3):
    return DecodeJob(op_id=1, window_id=window_id, n_rounds=n_rounds,
                     label=f"W{window_id}",
                     request_key=DecoderRequestKey(1, window_id, DecoderTier.WEAK,
                                                   window_id))


def _run(decoder, engine, job):
    seen = {}

    def on_result(result):
        seen["done_at"] = engine.now
        seen["result"] = result

    decoder.run(job, engine, on_result)
    engine.run()
    return seen


def test_stages_before_algorithm_after_in_order_with_ticks():
    engine = Engine(verbose=False)
    inner = _RecordingInner(engine)
    decoder = DecoderEngine(inner, _timing())

    seen = _run(decoder, engine, _job(n_rounds=3))

    records = decoder.stage_records_for(1, 0)
    assert [(r.stage, r.cycles, r.start_ticks, r.end_ticks) for r in records] == [
        ("fetch", 3, 0, 3 * CYCLE),
        (ALGORITHM_STAGE, None, 3 * CYCLE, 3 * CYCLE + us(2.0)),
        ("release", 1, 3 * CYCLE + us(2.0), 4 * CYCLE + us(2.0)),
    ]
    assert seen["done_at"] == 4 * CYCLE + us(2.0)
    assert seen["result"].window_id == 0
    assert decoder.latency(_job(n_rounds=3)) == seen["done_at"]


def test_the_result_is_produced_when_the_algorithm_time_ends():
    engine = Engine(verbose=False)
    inner = _RecordingInner(engine, latency_us=5.0)
    decoder = DecoderEngine(inner, _timing())

    _run(decoder, engine, _job())

    algorithm = decoder.stage_records_for(1, 0)[1]
    assert inner.decode_ticks == [algorithm.end_ticks]


def test_algorithm_time_is_the_wrapped_decoder_latency_only():
    inner = PerRoundDecoder(tau_us=0.5)
    decoder = DecoderEngine(inner, _timing(before=(), after=()))
    job = _job(n_rounds=4)
    assert decoder.latency(job) == inner.latency(job)


def test_hardware_stages_are_data_with_free_names():
    engine = Engine(verbose=False)
    timing = _timing(
        before=(DecoderStage("syndrome_ingest", cycles_per_round=2),
                DecoderStage("predecode", cycles_per_round=3)),
        after=(DecoderStage("correction_output", cycles_per_job=4),))
    decoder = DecoderEngine(_RecordingInner(engine, 0.0), timing)

    _run(decoder, engine, _job(n_rounds=2))

    assert [(r.stage, r.cycles) for r in decoder.stage_records_for(1, 0)] == [
        ("syndrome_ingest", 4), ("predecode", 6), (ALGORITHM_STAGE, None),
        ("correction_output", 4)]


def test_cancelled_job_still_holds_the_unit_but_skips_the_algorithm():
    engine = Engine(verbose=False)
    inner = _RecordingInner(engine)
    decoder = DecoderEngine(inner, _timing())
    job = _job()
    job.cancelled = True
    done = []
    decoder.run(job, engine, lambda result: done.append((engine.now, result)))
    engine.run()
    assert inner.decode_ticks == []
    assert done == [(decoder.latency(job), None)]


def test_result_reaches_the_callback():
    engine = Engine(verbose=False)
    decoder = DecoderEngine(_RecordingInner(engine), _timing())
    job = _job()
    seen = _run(decoder, engine, job)
    assert seen["result"].op_id == 1


def test_guards():
    with pytest.raises(ValueError):
        DecoderStage("x", cycles_per_job=-1)
    with pytest.raises(ValueError):
        DecoderTiming((), (), 0.0)


def test_end_to_end_stim_memory_run_through_the_timed_decoder():
    """A real rotated memory circuit decoded by PyMatching inside the timed
    unit: same logical answers as the bare decoder, three stages per window
    in the trace, no two windows overlapping on the single unit."""
    stim = pytest.importorskip("stim")
    from decsim.qpu.stim_device import StimDevice
    from decsim.message import Operation
    from decsim.decoders.mwpm.decoder import PyMatchingDecoder
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec

    def build(decoder):
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", rounds=6, distance=3,
            after_clifford_depolarization=0.005,
            before_measure_flip_probability=0.005,
            after_reset_flip_probability=0.005,
            before_round_data_depolarization=0.005)
        operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                              circuit=circuit)
        return RunSpec(ops=[operation], d=3, rounds_policy=FixedRounds(6),
                       device=StimDevice(), decoder=decoder, seed=11).build()

    bare = build(PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)))
    timed = DecoderEngine(PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)), _timing())
    staged = build(timed)

    assert staged.result.terminal_status == "complete"
    assert ([r.logical_observables for r in staged.result.operation_results]
            == [r.logical_observables for r in bare.result.operation_results])
    windows = sorted({(r.op_id, r.window_id) for r in timed.stage_records})
    assert windows
    for key in windows:
        names = [r.stage for r in timed.stage_records_for(*key)]
        assert names == ["fetch", ALGORITHM_STAGE, "release"]
    spans = sorted((timed.stage_records_for(*key)[0].start_ticks,
                    timed.stage_records_for(*key)[-1].end_ticks)
                   for key in windows)
    assert all(a_end <= b_start
               for (_, a_end), (b_start, _) in zip(spans, spans[1:]))
    assert any(ALGORITHM_STAGE in line for line in staged.engine.log_lines)


def test_measured_wall_clock_algorithm_holds_the_unit_for_the_real_call():
    """PyMatchingDecoder(latency_model=None) inside the engine: the real matching
    call runs at algorithm start, the unit stays busy for exactly the measured
    time, and the result is released only then."""
    stim = pytest.importorskip("stim")
    from decsim.qpu.stim_device import StimDevice
    from decsim.message import Operation
    from decsim.decoders.mwpm.decoder import PyMatchingDecoder
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=6, distance=3,
        after_clifford_depolarization=0.005, before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005, before_round_data_depolarization=0.005)
    operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    engine = DecoderEngine(PyMatchingDecoder(latency_model=None), _timing())
    completed = RunSpec(ops=[operation], d=3, rounds_policy=FixedRounds(6),
                        device=StimDevice(), decoder=engine, seed=3).build()
    assert completed.result.terminal_status == "complete"
    algorithm = [r for r in engine.stage_records if r.stage == ALGORITHM_STAGE]
    assert algorithm
    for record in algorithm:
        assert record.measured_ns is not None and record.measured_ns > 0
        assert record.end_ticks - record.start_ticks == us(record.measured_ns / 1000.0)
    with pytest.raises(RuntimeError, match="measured wall-clock"):
        PyMatchingDecoder(latency_model=None).latency(_job())


def test_cancel_stops_the_remaining_stages_and_never_calls_on_done():
    engine = Engine(verbose=False)
    inner = _RecordingInner(engine, latency_us=5.0)
    decoder = DecoderEngine(inner, _timing())
    job = _job()
    done = []
    decoder.run(job, engine, lambda result: done.append(engine.now))
    engine.schedule(1, lambda: decoder.cancel(job))          # during fetch
    engine.run()
    assert done == []
    assert inner.decode_ticks == []
    assert [r.stage for r in decoder.stage_records_for(1, 0)] == ["fetch"]


def test_a_hardware_stage_may_not_be_named_algorithm():
    with pytest.raises(ValueError, match="names the decoder itself"):
        DecoderTiming((DecoderStage(ALGORITHM_STAGE, cycles_per_job=1),), (), MHZ)


def test_stage_ticks_sum_to_the_whole_job_at_the_clock_for_any_partition():
    job = _job(n_rounds=3)
    one = DecoderTiming((DecoderStage("a", cycles_per_job=2),), (), 300.0)
    split = DecoderTiming((DecoderStage("a", cycles_per_job=1),
                           DecoderStage("b", cycles_per_job=1)), (), 300.0)
    assert sum(one.stage_ticks(job).values()) == sum(split.stage_ticks(job).values()) == us(2 / 300.0)
