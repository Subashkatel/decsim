"""Decoder-side boundary application, the machine's one rule: raw rounds
ship at data-complete, the predecessor's boundary mask is XORed into the
landed input when the decode starts (qLDPC net_error, cudaq-x
syndrome_mods, LILLIPUT's state register, Skoric's artificial defects).
A boundary-blocked job waits in its input slot, never on the unit."""

import stim

from decsim.config import us
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.decoders.weak_strong_switching import StrongOnly
from decsim.links.link_profiles import logical_reference_profile
from decsim.message import Operation
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import (SlidingTerminalPolicy,
                                              SlidingWindowScheme,
                                              TanSandwichScheme)

ROUNDS = 27


def _sliding():
    return SlidingWindowScheme(
        terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD)


def _stim_run(*, round_us=1.0, scheme=None, units=1, policy=None):
    p = 0.003
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=ROUNDS, distance=3,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    return RunSpec(
        ops=[Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                       circuit=circuit)],
        d=3, rounds_policy=FixedRounds(ROUNDS), round_us=round_us,
        device=StimDevice(),
        decoder=PyMatchingDecoder(PresetLatencyDecoder(5.0)), num_units=units,
        scheme=(scheme if scheme is not None else _sliding()),
        escalation_policy=(policy if policy is not None else StrongOnly()),
        links=logical_reference_profile(), seed=3).build()


def _gaps(completed):
    dones = sorted(w.t_done for _, w in
                   completed.window_manager.windows.items())
    return sorted({b - a for a, b in zip(dones, dones[1:])})


def test_saturated_chain_is_dd_plus_max_of_transfer_and_decode():
    """The reference cadence: with the raw input shipped under the previous
    decode, each window costs dd (0.5) + max(sbd 2.0, decode 5.0) = 5.5 us,
    never the serial dd + sbd + decode = 7.5 us."""
    assert _gaps(_stim_run()) == [us(5.5)]


def test_parked_decode_starts_at_the_boundary_arrival():
    """The landed input waits parked; the decode begins the tick the last
    boundary lands (predecessor done + dd 0.5)."""
    completed = _stim_run()
    windows = dict(sorted(completed.window_manager.windows.items()))
    dones = {k: w.t_done for (_, k), w in windows.items()}
    for (_, k), window in windows.items():
        if k == 0 or window.t_done is None:
            continue
        boundary_arrival = dones[k - 1] + us(0.5)
        assert window.t_done - us(5.0) >= boundary_arrival

def test_relaxed_stream_parks_only_the_clamped_terminal_window():
    """With rounds every 3.0 us the chain drains between arrivals: no
    window but the terminal pair's dependent ever waits parked past its
    data-complete plus the transfer."""
    completed = _stim_run(round_us=3.0)
    windows = dict(sorted(completed.window_manager.windows.items()))
    late = [k for (_, k), w in windows.items()
            if w.t_done is not None
            and w.t_done - us(5.0) - us(2.0) > w.t_data_complete + us(0.6)]
    last = max(k for (_, k) in windows)
    assert late in ([], [last]), late


def test_tan_seams_never_deadlock_the_unit():
    """A Tan seam fills before its neighbor cores and reads both of their
    boundaries. It must wait in its slot, not on the unit: every window
    decodes and the answer is right, on a single unit."""
    completed = _stim_run(scheme=TanSandwichScheme(), units=1)
    windows = completed.window_manager.windows.values()
    assert all(w.t_done is not None for w in windows)
    assert completed.result.operation_results[0].logical_failure is False
