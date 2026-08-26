"""The boundary application axis: ASSEMBLY folds the predecessor's boundary
into the payloads host-side (the original model); DECODER ships raw rounds
at data-complete and XORs the mask into the landed input when the boundary
arrives (qLDPC net_error, cudaq-x syndrome_mods, LILLIPUT's state register,
Skoric's artificial defects). Results must be bit-identical across modes;
only timing may differ."""

import pytest
import stim

from decsim.config import us
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.decoders.weak_strong_switching import StrongOnly, Switching
from decsim.decoders.decoders import SAMPLED_CONFIDENCE_SOURCE
from decsim.links.link_profiles import logical_reference_profile
from decsim.message import BoundaryApplication, Operation
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import (SlidingTerminalPolicy,
                                              SlidingWindowScheme)

ROUNDS = 27


def _sliding():
    return SlidingWindowScheme(
        terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD)


def _stim_run(mode, depth, round_us=1.0):
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
        decoder=PyMatchingDecoder(PresetLatencyDecoder(5.0)), num_units=1,
        scheme=_sliding(), escalation_policy=StrongOnly(),
        input_staging_depth=depth, boundary_application=mode,
        links=logical_reference_profile(), seed=3).build()


def _gaps(completed):
    dones = sorted(w.t_done for _, w in
                   completed.window_manager.windows.items())
    return sorted({b - a for a, b in zip(dones, dones[1:])})


def test_results_are_bit_identical_across_modes():
    assembly = _stim_run(None, 0)
    decoder_mode = _stim_run(BoundaryApplication.DECODER, 1)
    row_a = assembly.result.operation_results[0]
    row_d = decoder_mode.result.operation_results[0]
    assert row_a.logical_observables == row_d.logical_observables
    assert row_a.logical_failure is False and row_d.logical_failure is False
    ledger_a = {key: contribution.logical_observables for key, contribution
                in assembly.window_manager.ledger.contributions.items()}
    ledger_d = {key: contribution.logical_observables for key, contribution
                in decoder_mode.window_manager.ledger.contributions.items()}
    assert ledger_a == ledger_d


def test_decoder_mode_with_staging_reaches_the_reference_cadence():
    # dd 0.5 + max(csd 2.0, decode 5.0) = 5.5 per chain link
    assert _gaps(_stim_run(BoundaryApplication.DECODER, 1)) == [us(5.5)]


def test_assembly_mode_pays_the_full_chain():
    # dd 0.5 + csd 2.0 + decode 5.0 = 7.5 per chain link
    assert _gaps(_stim_run(None, 0)) == [us(7.5)]


def test_parked_decode_starts_at_the_boundary_arrival():
    # saturated stream: the landed input always waits for the boundary,
    # so every service start equals the predecessor's done + dd; that is
    # exactly the 5.5 cadence asserted above, plus the park lines in the log
    completed = _stim_run(BoundaryApplication.DECODER, 1)
    parked_lines = [line for line in completed.engine.log_lines
                    if "PARK DECODE" in line]
    assert parked_lines, "no decode ever parked under a saturated chain"


def test_relaxed_stream_parks_only_the_clamped_terminal_window():
    # rounds every 3 us: the boundary precedes the landing for every
    # interior window. The terminal window is the one honest exception:
    # its clamped read range data-completes on the same tick as its
    # predecessor's (both need the last round), so it waits for that
    # boundary just like the references' final window does
    completed = _stim_run(BoundaryApplication.DECODER, 1, round_us=3.0)
    parked_lines = [line for line in completed.engine.log_lines
                    if "PARK DECODE" in line]
    assert all("W8" in line for line in parked_lines), parked_lines
    row = completed.result.operation_results[0]
    assert row.logical_failure is False


def test_decoder_mode_with_an_escalating_policy_is_refused():
    weak = PresetLatencyDecoder(0.1)
    with pytest.raises(ValueError, match="DECODER"):
        RunSpec(ops=[Operation(0, "memory", (0,), patches=(0,))], d=3,
                rounds_policy=FixedRounds(9), round_us=1.0,
                decoder=weak, num_units=1, scheme=_sliding(),
                escalation_policy=Switching(
                    expected_source=SAMPLED_CONFIDENCE_SOURCE,
                    confidence_threshold=0.5),
                boundary_application=BoundaryApplication.DECODER,
                seed=1).build()
