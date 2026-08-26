"""Every unit is a depth-1 decoupled access-execute machine: two input
slots, so the next window's SBD transfer overlaps the current compute
(Smith 1982 DAE; TI EDMA ping-pong SPRAAN4A; gem5-Aladdin ready bits,
Shao MICRO 2016). Compute is claimed separately from the slots: a job
whose boundary has not arrived waits in its slot, never on the unit
(Tomasulo's rule; gem5 O3 scheduleReadyInsts issues only ready work)."""

import pytest
import stim

from decsim.config import us
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
        if row["path"] == "sbd":
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
    assert us(5.0) in saturated, sorted(set(saturated))
    assert us(7.0) not in saturated, sorted(set(saturated))


def test_unit_idles_until_a_late_landing():
    # decode 0.5 us against sbd 2.0: compute ends before the next DMA
    # lands, so the unit idles and service starts at the landing tick
    completed = _run(decode_us=0.5)
    sends, stamps = _events(completed)
    at_landing = sum(1 for k, (_, done) in stamps.items()
                     if done - us(0.5) == sends[k][1])
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
    assert us(7.0) in gaps, sorted(set(gaps))


def test_an_oversized_single_window_still_stops_loudly():
    with pytest.raises(DecoderMemoryCapacityExhaustion):
        _run(capacity=8)
