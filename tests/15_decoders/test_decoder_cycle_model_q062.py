"""Behavior tests for the Q-062(c) staged weak decoder."""

import pytest

from decsim.config import us
from decsim.decoder_cycle_model import (
    CycleQuantityBasis,
    DecoderCycleModel,
    DecoderPipelineStage,
    StagedDecoder,
    StageCycleConfig,
)
from decsim.decoder_cycle_profiles import (
    delegated_latency_model,
    lookup_table_published_total_model,
)
from decsim.decoders import PerRoundDecoder, PresetLatencyDecoder
from decsim.engine import Engine
from decsim.message import DecodeJob, DecodeResult

STAGES = tuple(DecoderPipelineStage)


def _stage(cycles, basis=CycleQuantityBasis.PER_JOB):
    return StageCycleConfig(cycles, basis, "test card")


def _card(fetch=1, decode=1, execute=7, memory=1, writeback=1, *,
          fetch_basis=CycleQuantityBasis.PER_JOB, frequency_mhz=250.0,
          include_inner_latency=False):
    return DecoderCycleModel(
        fetch=_stage(fetch, fetch_basis), decode=_stage(decode),
        execute=_stage(execute), memory=_stage(memory),
        writeback=_stage(writeback), frequency_mhz=frequency_mhz,
        frequency_source="test clock", include_inner_latency=include_inner_latency,
        model_name="test")


class _RecordingInner:
    """Timing-only inner decoder that records when decode() ran."""

    fault_model_requirement = PresetLatencyDecoder().fault_model_requirement

    def __init__(self, latency_us=0.0):
        self.latency_us = latency_us
        self.decode_ticks = []

    def latency(self, job):
        return us(self.latency_us)

    def decode(self, job):
        self.decode_ticks.append(job.window_id)
        return DecodeResult(job.op_id, job.window_id)


def _job(window_id=0, n_rounds=3):
    return DecodeJob(op_id=1, window_id=window_id, n_rounds=n_rounds,
                     label=f"W{window_id}")


def _run(decoder, engine, jobs):
    done = []
    for job in jobs:
        decoder.run(job, engine, lambda j=job: done.append((j.window_id, engine.now)))
    engine._start_running()
    engine.run()
    return done


def test_one_job_walks_the_five_stages_in_order_with_stage_ticks():
    engine = Engine(verbose=False)
    inner = _RecordingInner()
    decoder = StagedDecoder(inner, _card())
    tick = us(1 / 250.0)                    # one cycle at 250 MHz

    done = _run(decoder, engine, [_job()])

    records = decoder.stage_records_for(1, 0)
    assert tuple(r.stage for r in records) == STAGES
    ends = [1, 2, 9, 10, 11]
    assert [(r.start_ticks, r.end_ticks) for r in records] == [
        ((e - c) * tick, e * tick)
        for e, c in zip(ends, (1, 1, 7, 1, 1))]
    assert done == [(0, 11 * tick)]
    assert decoder.latency(_job()) == 11 * tick


def test_execute_runs_the_real_decode_at_execute_entry_and_result_is_released():
    engine = Engine(verbose=False)
    inner = _RecordingInner()
    decoder = StagedDecoder(inner, _card())
    job = _job()
    seen = {}

    def on_done():
        seen["done_at"] = engine.now
        seen["result"] = decoder.decode(job)

    decoder.run(job, engine, on_done)
    engine._start_running()
    engine.run()

    execute = decoder.stage_records_for(1, 0)[2]
    assert execute.stage is DecoderPipelineStage.EXECUTE
    assert inner.decode_ticks == [0]
    assert seen["result"].window_id == 0
    assert seen["done_at"] == execute.end_ticks + 2 * us(1 / 250.0)
    with pytest.raises(RuntimeError, match="not been released"):
        decoder.decode(job)


def test_fetch_charges_per_round_when_the_card_says_so():
    engine = Engine(verbose=False)
    decoder = StagedDecoder(
        _RecordingInner(), _card(fetch=2, fetch_basis=CycleQuantityBasis.PER_ROUND))

    _run(decoder, engine, [_job(n_rounds=5)])

    fetch = decoder.stage_records_for(1, 0)[0]
    assert fetch.cycles == 10
    assert fetch.end_ticks - fetch.start_ticks == 10 * us(1 / 250.0)


def test_inner_latency_is_added_to_execute_only():
    inner = _RecordingInner(latency_us=3.0)
    decoder = StagedDecoder(inner, _card(include_inner_latency=True))
    engine = Engine(verbose=False)

    _run(decoder, engine, [_job()])

    records = decoder.stage_records_for(1, 0)
    tick = us(1 / 250.0)
    assert records[2].end_ticks - records[2].start_ticks == 7 * tick + us(3.0)
    assert records[4].end_ticks == 11 * tick + us(3.0)
    assert decoder.latency(_job()) == records[4].end_ticks


def test_two_jobs_on_the_same_unit_do_not_overlap_when_started_back_to_back():
    """One unit is one job at a time; the manager owns the unit count, so the
    unit itself simply reports both walks with their true ticks."""
    engine = Engine(verbose=False)
    decoder = StagedDecoder(_RecordingInner(), _card())
    first, second = _job(0), _job(1)
    engine.schedule(0, lambda: decoder.run(first, engine, lambda: None))
    engine.schedule(decoder.latency(first),
                    lambda: decoder.run(second, engine, lambda: None))
    engine._start_running()
    engine.run()

    a = decoder.stage_records_for(1, 0)
    b = decoder.stage_records_for(1, 1)
    assert a[-1].end_ticks <= b[0].start_ticks
    assert tuple(r.stage for r in b) == STAGES


def test_a_job_cannot_be_started_twice_while_running():
    engine = Engine(verbose=False)
    decoder = StagedDecoder(_RecordingInner(), _card())
    job = _job()
    decoder.run(job, engine, lambda: None)
    with pytest.raises(RuntimeError, match="already running"):
        decoder.run(job, engine, lambda: None)


def test_cancelled_job_skips_the_real_decode_but_still_occupies_the_unit():
    engine = Engine(verbose=False)
    inner = _RecordingInner()
    decoder = StagedDecoder(inner, _card())
    job = _job()
    job.cancelled = True

    done = _run(decoder, engine, [job])

    assert inner.decode_ticks == []
    assert done == [(0, 11 * us(1 / 250.0))]


def test_total_equals_sum_of_stage_cycles_at_the_clock_for_odd_clocks():
    card = _card(fetch=1, decode=1, execute=3, memory=1, writeback=1,
                 frequency_mhz=300.0)
    decoder = StagedDecoder(_RecordingInner(), card)
    engine = Engine(verbose=False)
    _run(decoder, engine, [_job()])
    records = decoder.stage_records_for(1, 0)
    assert records[-1].end_ticks == us(7 / 300.0)
    assert sum(r.end_ticks - r.start_ticks for r in records) == us(7 / 300.0)


def test_card_guards():
    with pytest.raises(ValueError):
        StageCycleConfig(-1, CycleQuantityBasis.PER_JOB, "x")
    with pytest.raises(ValueError):
        _card(frequency_mhz=0.0)
    with pytest.raises(ValueError):
        _card(frequency_mhz=1e12)


def test_profiles_delegated_card_is_exactly_the_wrapped_decoder_latency():
    inner = PerRoundDecoder(tau_us=0.5)
    decoder = StagedDecoder(inner, delegated_latency_model())
    job = _job(n_rounds=4)
    assert decoder.latency(job) == inner.latency(job)


def test_profiles_lilliput_card_is_seven_cycles_at_250_mhz():
    decoder = StagedDecoder(PresetLatencyDecoder(1.0),
                            lookup_table_published_total_model())
    assert decoder.latency(_job()) == us(7 / 250.0)


def test_end_to_end_stim_memory_run_through_the_staged_decoder():
    """A real rotated memory circuit decoded by PyMatching inside the staged
    unit: same logical answers as the bare decoder, five stages per window in
    the trace, and one unit never overlapping two windows."""
    stim = pytest.importorskip("stim")
    from decsim.adapters.stim_device import StimDevice
    from decsim.message import Operation
    from decsim.mwpm_decoder.decoder import PyMatchingDecoder
    from decsim.rounds import FixedRounds
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
    staged_decoder = StagedDecoder(
        PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)),
        _card(fetch=1, fetch_basis=CycleQuantityBasis.PER_ROUND,
              include_inner_latency=True))
    staged = build(staged_decoder)

    assert staged.result.terminal_status == "complete"
    assert ([r.logical_observables for r in staged.result.operation_results]
            == [r.logical_observables for r in bare.result.operation_results])
    records = staged_decoder.stage_records
    windows = sorted({(r.op_id, r.window_id) for r in records})
    assert windows
    for key in windows:
        assert tuple(r.stage for r in staged_decoder.stage_records_for(*key)) == STAGES
    fetches = [r for r in records if r.stage is DecoderPipelineStage.FETCH]
    assert all(f.rounds_read > 0 and f.cycles == f.rounds_read for f in fetches)
    spans = sorted((staged_decoder.stage_records_for(*key)[0].start_ticks,
                    staged_decoder.stage_records_for(*key)[-1].end_ticks)
                   for key in windows)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))
    assert any("EXECUTE" in line for line in staged.engine.log_lines)
