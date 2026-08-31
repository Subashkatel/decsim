"""DEC-003 characterization: unit occupancy equals service latency.

Executable evidence for the traceability row DEC-003 (PARTIAL): today a
decoder unit's compute is claimed from service start to decode end
(decoder_manager.py _begin_service through _on_decode_done), so the
per-unit initiation interval equals the service latency and per-unit
throughput is 1/latency. The two-slot memory model lets exactly one
waiting window land its input DMA early (decoder_manager.py
_eligible_unit: at most two residents per slot), which overlaps transfer
with the previous compute but never overlaps two computes.

The scenario: four independent patches, every window's data complete at
12 us, one default unit, declared weak latency 100 us. A pipelined
decoder with a 1 us initiation interval would finish them at 117, 118,
119, 120; the current model finishes them at 117, 217, 317, 417.

The passing tests pin today's behavior; the xfail states the pipelined
behavior DEC-003 requires. Production code must not change here: the
xfail flips only when the initiation-interval model lands (see
credibility/decoder_pipeline_requirements.md, outside the repo).
"""

import pytest

from decsim.config import us
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec

SERVICE_US = 100.0
DESIRED_INITIATION_US = 1.0


def _one_unit_backlog_run(fabric):
    """Four parallel one-window ops, one default unit, 100 us decoder."""
    spec = RunSpec(
        ops=[fabric["memory_op"](op_id) for op_id in (1, 2, 3, 4)],
        d=3, rounds_policy=FixedRounds(3),
        decoder=PresetLatencyDecoder(SERVICE_US),
        links=fabric["declared_profile"](cwb=True, csb=False),
        timing=fabric["declared_timing"](),
        pauli_frame=PauliFrameConfig(commit_us=fabric["DECLARED_US"]["frame"]),
        seed=0)
    return spec.build()


def _windows(completed):
    return [completed.window_manager.windows[(op_id, 0)]
            for op_id in (1, 2, 3, 4)]


def test_unit_occupancy_equals_service_latency(fabric):
    """All four windows are ready at 12 us, yet completions are spaced by
    exactly the 100 us service latency: the unit is occupied for its whole
    service time, so per-unit initiation interval == service latency."""
    windows = _windows(_one_unit_backlog_run(fabric))
    wbd = fabric["DECLARED_US"]["wbd"]

    assert [window.t_data_complete for window in windows] == [us(12)] * 4
    assert [window.t_done for window in windows] == \
        [us(12 + wbd + SERVICE_US), us(12 + wbd + 2 * SERVICE_US),
         us(12 + wbd + 3 * SERVICE_US), us(12 + wbd + 4 * SERVICE_US)]
    completion_gaps = [later.t_done - earlier.t_done
                       for earlier, later in zip(windows, windows[1:])]
    assert completion_gaps == [us(SERVICE_US)] * 3


def test_input_slot_gives_one_window_of_lookahead_and_no_more(fabric):
    """The two-slot unit accepts exactly one waiting window's input DMA
    while computing (dispatch at 12 us) but parks its compute; the rest
    dispatch only when the previous compute starts and frees the slot."""
    windows = _windows(_one_unit_backlog_run(fabric))

    assert windows[0].t_dispatch == us(12)
    assert windows[1].t_dispatch == us(12)                  # DMA overlap only
    assert windows[1].t_done - windows[0].t_done == us(SERVICE_US)
    assert windows[2].t_dispatch == windows[0].t_done       # slot frees late
    assert windows[3].t_dispatch == windows[1].t_done


@pytest.mark.xfail(
    strict=True,
    reason="DEC-003: no initiation-interval model; unit occupancy equals "
           "service latency, so a 1 us cadence on one unit is impossible")
def test_pipelined_unit_would_finish_back_to_back_windows_each_round(fabric):
    """What DEC-003 requires and today's model cannot do: with service
    latency 100 us and initiation interval 1 us, four in-flight windows
    finish at 117, 118, 119, 120 us on ONE unit."""
    windows = _windows(_one_unit_backlog_run(fabric))
    wbd = fabric["DECLARED_US"]["wbd"]

    assert [window.t_done for window in windows] == [
        us(12 + wbd + SERVICE_US + k * DESIRED_INITIATION_US)
        for k in range(4)]
