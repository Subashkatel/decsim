"""Every unit is a depth-1 decoupled access-execute machine: two input
slots, so the next window's SBD transfer overlaps the current compute
(Smith 1982 DAE; TI EDMA ping-pong SPRAAN4A; gem5-Aladdin ready bits,
Shao MICRO 2016). Compute is claimed separately from the slots: a job
whose boundary has not arrived waits in its slot, never on the unit
(Tomasulo's rule; gem5 O3 scheduleReadyInsts issues only ready work)."""

import pytest
import stim

from decsim.config import microseconds_to_ticks
from decsim.decoders.decoder_memory import (DecoderMemoryCapacityExhaustion,
                                            DecoderMemoryConfig)
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.decoders.weak_strong_switching import StrongOnly
from decsim.links.link_profiles import logical_reference_profile
from decsim.message import Operation
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import TanSandwichScheme

ROUNDS = 15


def _run(*, units=1, decode_us=5.0, capacity=None):
    p = 0.003
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=ROUNDS, distance=3,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    return RunSpec(
        ops=[Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                       circuit=circuit)],
        d=3, rounds_policy=FixedRounds(ROUNDS), round_us=1.0,
        device=StimDevice(),
        decoder=PyMatchingDecoder(PresetLatencyDecoder(decode_us)),
        num_units=units, scheme=TanSandwichScheme(),
        escalation_policy=StrongOnly(),
        decoder_memory=(None if capacity is None
                        else DecoderMemoryConfig({"default": capacity})),
        links=logical_reference_profile(), seed=11).build()


def _events(completed):
    """(sbd send tick, delivery tick) per window plus per-window stamps."""
    sends = {}
    for row in completed.result.link_traffic["transfers"]:
        if row["path"] == "strong_buffer_to_strong_decoder":
            sends[row["attribution"]["window_id"]] = (
                row["send_ticks"], row["delivery_ticks"])
    stamps = {k: (w.t_dispatch, w.t_done)
              for (_, k), w in completed.window_manager.windows.items()}
    return sends, stamps


def test_results_are_correct_and_every_window_decodes():
    completed = _run()
    assert completed.result.operation_results[0].logical_failure is False
    windows = completed.window_manager.windows.values()
    assert all(w.t_done is not None for w in windows)


def test_transfers_overlap_computes():
    completed = _run()
    sends, stamps = _events(completed)
    busy = sorted((sends[k][1], done) for k, (_, done) in stamps.items())
    overlapped = sum(
        1 for k, (send, _delivery) in sends.items()
        if any(start < send < end for start, end in busy))
    assert overlapped > 0, "no transfer overlapped a compute"


def test_saturated_cadence_is_max_of_transfer_and_compute():
    completed = _run()
    _, stamps = _events(completed)
    dones = sorted(done for _, done in stamps.values())
    gaps = [b - a for a, b in zip(dones, dones[1:])]
    saturated = [gap for gap in gaps if gap > 0]
    # sbd 2.0, decode 5.0: back-to-back completions tick at 5.0 us
    assert microseconds_to_ticks(5.0) in saturated, sorted(set(saturated))
    assert microseconds_to_ticks(7.0) not in saturated, sorted(set(saturated))


def test_unit_idles_until_a_late_landing():
    # decode 0.5 us against sbd 2.0: compute ends before the next DMA
    # lands, so the unit idles and service starts at the landing tick
    completed = _run(decode_us=0.5)
    sends, stamps = _events(completed)
    at_landing = sum(1 for k, (_, done) in stamps.items()
                     if done - microseconds_to_ticks(0.5) == sends[k][1])
    assert at_landing > 0


def test_tight_memory_degrades_to_serial_residency_where_pairs_do_not_fit():
    # a Tan type-1 core reads 9 rounds (s=3, b=3): capacity 12 holds a
    # core beside a small seam but never two cores, so core-to-core the
    # machine keeps serial residency and pays transfer + compute in the
    # cadence; it completes instead of exhausting the memory
    completed = _run(capacity=12)
    assert completed.result.operation_results[0].logical_failure is False
    _, stamps = _events(completed)
    dones = sorted(done for _, done in stamps.values())
    gaps = [b - a for a, b in zip(dones, dones[1:])]
    assert microseconds_to_ticks(7.0) in gaps, sorted(set(gaps))


def test_an_oversized_single_window_still_stops_loudly():
    with pytest.raises(DecoderMemoryCapacityExhaustion):
        _run(capacity=8)


def _fixed_input(engine, transfer_us):
    """A send_input that lands after a fixed delay, no link in the way."""
    delay_ticks = microseconds_to_ticks(transfer_us)

    def send_input(on_landed):
        engine.schedule(delay_ticks, on_landed)
        return delay_ticks

    return send_input


def _standalone_pool(units, transfer_us, compute_us, decoder=None):
    """A decoder manager fed window jobs directly: (engine, manager,
    submit(index, arrival), compute start ticks by window)."""
    from decsim.decoders.decoder_manager import DecoderManager
    from decsim.decoders.decoders import CodeRouter
    from decsim.decoders.schedulers import FifoScheduler
    from decsim.decoders.weak_strong_switching import Baseline
    from decsim.engine import Engine
    from decsim.message import (DecodeJob, DecoderRequestKey, DecoderTier,
                                RetainedSyndromeFragment)

    engine = Engine(verbose=False)
    manager = DecoderManager(
        engine, router=CodeRouter(decoder or PresetLatencyDecoder(compute_us)),
        scheduler=FifoScheduler(), num_units=units,
        escalation_policy=Baseline(), services=None,
        on_window_decoded=lambda job, result: None,
        on_strong_window_decoded=None)
    compute_start = {}
    original_begin = manager._begin_service

    def recording_begin(job, gated=True):
        compute_start[job.label] = engine.now
        original_begin(job, gated)

    manager._begin_service = recording_begin

    def submit(index, arrival_us):
        payload = RetainedSyndromeFragment(operation_id=1, patch_id="p", round_index=index,
                                           bits=(0, 1), size_bits=2, fragment_index=0)
        job = DecodeJob(op_id=1, window_id=index, n_rounds=1, payloads=[payload],
                        label=f"w{index}",
                        request_key=DecoderRequestKey(1, index, DecoderTier.WEAK, index))
        engine.schedule(microseconds_to_ticks(arrival_us),
                        lambda: manager.enqueue(job, _fixed_input(engine, transfer_us)))

    return engine, manager, submit, compute_start


def test_a_full_pool_stages_the_next_window_on_the_unit_that_frees_first():
    """Least work left (Harchol-Balter 2013, Ch. 24): with every unit
    computing, the next window's input goes to the unit whose compute
    ends first, counting a claimed unit's decode from its input's landing,
    so it starts when a central FIFO queue over the pool would."""
    engine, manager, submit, compute_start = _standalone_pool(2, 3.5, 4.0)
    for index, arrival in enumerate((2.0, 4.0, 8.5, 15.0, 16.5, 19.0)):
        submit(index, arrival)
    engine.run()
    manager.check_decode_work_settled()
    # unit 0: w0 5.5..9.5, w2 12.0..16.0, w4 20.0..24.0 (claimed at 16.5,
    # input lands 20.0); unit 1: w1 7.5..11.5, w3 18.5..22.5
    assert compute_start["w3"] == microseconds_to_ticks(18.5) and compute_start["w4"] == microseconds_to_ticks(20.0)
    # w5's input lands at 22.5: unit 1 frees at 22.5, unit 0 not before 24.0
    assert compute_start["w5"] == microseconds_to_ticks(22.5)


def test_a_job_without_input_waits_in_the_queue_for_free_compute():
    """submit_decode carries no syndrome data, so there is nothing to
    prefetch into a busy unit's second slot; the job waits centrally and
    takes the first unit that frees (G/D/k FIFO)."""
    from decsim.decoders.decoder_manager import DecoderManager
    from decsim.decoders.decoders import CodeRouter
    from decsim.decoders.schedulers import FifoScheduler
    from decsim.engine import Engine

    engine = Engine(verbose=False)
    manager = DecoderManager(
        engine, router=CodeRouter(PresetLatencyDecoder(4.0)),
        scheduler=FifoScheduler(), num_units=2,
        escalation_policy=None, services=None,
        on_window_decoded=None, on_strong_window_decoded=None)
    done = {}
    for label, arrival in (("a", 0.0), ("b", 1.0), ("c", 2.0)):
        engine.schedule(microseconds_to_ticks(arrival), lambda label=label: manager.submit_decode(
            1, lambda label=label: done.__setitem__(label, engine.now), label=label))
    engine.run()
    # a on unit 0 (0..4), b on unit 1 (1..5); c waits for unit 0 at 4
    assert done == {"a": microseconds_to_ticks(4.0), "b": microseconds_to_ticks(5.0), "c": microseconds_to_ticks(8.0)}


def test_a_pipelined_units_compute_returns_to_the_pool_once():
    """The intake goes back to the pool when the initiation interval ends;
    the decode's later completion must not return it a second time, or
    the pool counts free units it does not have and starts the next
    window while the intake is still busy (Hennessy and Patterson App. C:
    one issue per initiation interval)."""
    from decsim.decoders.decoders import PipelinedDecoder
    decoder = PipelinedDecoder(PresetLatencyDecoder(4.0), initiation_interval_us=0.5)
    engine, manager, submit, compute_start = _standalone_pool(1, 0.2, 4.0, decoder)
    # w0 starts at 0.2, its intake frees at 0.7, its result is out at 4.2;
    # w1 starts at 4.5 - 0.2 = 4.3 ... w2 and w3 arrive back to back after
    # the result of w0 has been out, when a double free would show
    for index, arrival in enumerate((0.0, 4.3, 4.31, 4.32)):
        submit(index, arrival)
    engine.run()
    manager.check_decode_work_settled()
    assert manager.pool_free["default"] == 1
    assert compute_start["w1"] == microseconds_to_ticks(4.5)
    # one start per initiation interval: w2 and w3 follow at 0.5 us steps
    assert compute_start["w2"] == microseconds_to_ticks(5.0) and compute_start["w3"] == microseconds_to_ticks(5.5)
