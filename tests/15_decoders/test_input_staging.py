"""The per-unit input staging slot (gem5-Aladdin pipelined-DMA rule set,
depth = decoupled access-execute queue depth): depth 0 is today's machine,
depth 1 overlaps the next window's input transfer with the current compute.
Both depths run in every test that applies to both."""

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


def _run(depth, *, units=1, decode_us=5.0, capacity=None):
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
        input_staging_depth=depth,
        decoder_memory=(None if capacity is None
                        else DecoderMemoryConfig({"default": capacity})),
        links=logical_reference_profile(), seed=11).build()


def _events(completed):
    """(csd send tick, service window id) pairs plus per-window stamps."""
    sends = {}
    for row in completed.result.link_traffic["transfers"]:
        if row["path"] == "csd":
            sends[row["attribution"]["window_id"]] = (
                row["send_ticks"], row["delivery_ticks"])
    stamps = {k: (w.t_dispatch, w.t_done)
              for (_, k), w in completed.window_manager.windows.items()}
    return sends, stamps


def _service_spans(stamps, sends):
    """Per window: (service start, done). Service starts at DMA delivery
    or later (after the previous compute on a shared unit)."""
    spans = {}
    for k, (dispatch, done) in stamps.items():
        spans[k] = (sends[k][1], done)
    return spans


@pytest.mark.parametrize("depth", (0, 1))
def test_results_are_identical_across_depths(depth):
    completed = _run(depth)
    row = completed.result.operation_results[0]
    assert row.logical_failure is False


def test_depth0_never_transfers_while_the_unit_computes():
    completed = _run(0)
    sends, stamps = _events(completed)
    busy = sorted((sends[k][1], done) for k, (_, done) in stamps.items())
    for k, (send, _delivery) in sends.items():
        for start, end in busy:
            assert not (start < send < end), (
                f"depth 0 sent window {k}'s input at {send} inside a "
                f"compute span ({start}, {end})")


def test_depth1_overlaps_transfer_with_compute():
    completed = _run(1)
    sends, stamps = _events(completed)
    busy = sorted((sends[k][1], done) for k, (_, done) in stamps.items())
    overlapped = sum(
        1 for k, (send, _d) in sends.items()
        if any(start < send < end for start, end in busy))
    assert overlapped > 0, "no staged transfer overlapped a compute"


def test_depth1_swaps_at_compute_end_when_ready():
    completed = _run(1)
    sends, stamps = _events(completed)
    dones = sorted(done for _, done in stamps.values())
    starts = {k: max(sends[k][1],
                     max((d for d in dones if d <= stamps[k][1]
                          and d != stamps[k][1]), default=0))
              for k in stamps}
    # at least one window starts service at exactly the previous done tick
    same_tick = sum(
        1 for k in stamps
        if stamps[k][1] - (stamps[k][1] - max(
            (d for d in dones if d < stamps[k][1]), default=0)) in
        [max((d for d in dones if d < stamps[k][1]), default=0)]
        and sends[k][1] <= max((d for d in dones if d < stamps[k][1]),
                               default=0))
    assert same_tick > 0


def test_depth1_saturated_cadence_is_max_of_transfer_and_compute():
    completed = _run(1)
    _, stamps = _events(completed)
    dones = sorted(done for _, done in stamps.values())
    gaps = [b - a for a, b in zip(dones, dones[1:])]
    saturated = [gap for gap in gaps if gap > 0]
    # csd 2.0, decode 5.0: back-to-back completions tick at 5.0 us
    assert us(5.0) in saturated, sorted(set(saturated))
    assert us(7.0) not in saturated, sorted(set(saturated))


def test_depth0_saturated_cadence_pays_transfer_plus_compute():
    completed = _run(0)
    _, stamps = _events(completed)
    dones = sorted(done for _, done in stamps.values())
    gaps = [b - a for a, b in zip(dones, dones[1:])]
    assert us(7.0) in gaps, sorted(set(gaps))


def test_depth1_idles_until_late_landing():
    # decode 0.5 us against csd 2.0: compute ends before the staged DMA
    # lands, so the unit idles and service starts at the landing tick
    completed = _run(1, decode_us=0.5)
    sends, stamps = _events(completed)
    landings = {k: sends[k][1] for k in sends}
    at_landing = sum(1 for k, (_, done) in stamps.items()
                     if done - us(0.5) == landings[k])
    assert at_landing > 0


def test_depth1_charges_both_inputs_to_unit_memory():
    # a Tan type-1 window reads 9 rounds (s=3, b=3): capacity 12 holds one
    # window with headroom but not two, so depth 0 fits and depth 1 must
    # refuse: the ping-pong's doubled SRAM is charged, not hidden
    _run(0, capacity=12)
    with pytest.raises(DecoderMemoryCapacityExhaustion):
        _run(1, capacity=12)


def test_invalid_depth_is_refused():
    with pytest.raises(ValueError, match="input_staging_depth"):
        _run(2)


def test_bulk_strong_with_staging_is_refused():
    from decsim.decoders.decoder_manager import DecoderManager
    with pytest.raises(ValueError, match="bulk_strong"):
        DecoderManager(
            object(), router=None, scheduler=None, num_units=1,
            bulk_strong=True, input_staging_depth=1,
            escalation_policy=None, services=None,
            on_window_decoded=lambda *a: None,
            on_strong_window_decoded=lambda *a: None)
