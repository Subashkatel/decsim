"""DEC-003: decoder throughput independent of response latency.

Two models coexist and both are pinned here.

The plain (non-pipelined) models keep occupancy == service latency: a
unit's compute is claimed from service start to decode end
(decoder_manager.py _begin_service through _on_decode_done), so the
per-unit initiation interval equals the service latency. That is the
correct model for a simple decoder and stays the default.

PipelinedDecoder adds the second number: results still return after the
service latency while a new decode may START on the same unit every
initiation interval, with at most pipeline_depth in flight. The
acceptance gate for DEC-003 is credibility/acceptance_criteria.md
(criteria A to H, outside the repo); the tests below are its evidence:
A/C/D/F the cadence-and-completion test, B/E the depth test, G the
characterization tests plus the full suite and frozen gate, H the
determinism test (same-tick completion and initiation events resolve by
the engine's fixed order).

The scenario throughout: four (or six) independent patches, every
window's data complete at 12 us, one default unit, service 100 us.
"""

import pytest

from decsim.config import us
from decsim.decoders.decoders import PipelinedDecoder, PresetLatencyDecoder
from decsim.decoders.weak_strong_switching import StrongOnly
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec

SERVICE_US = 100.0
INITIATION_US = 1.0


def _backlog_run(fabric, *, decoder, n_ops=4, csb=False, escalation_policy=None):
    """n_ops parallel one-window ops, one default unit."""
    spec = RunSpec(
        ops=[fabric["memory_op"](op_id) for op_id in range(1, n_ops + 1)],
        d=3, rounds_policy=FixedRounds(3),
        decoder=decoder,
        escalation_policy=escalation_policy,
        links=fabric["declared_profile"](cwb=True, csb=csb),
        timing=fabric["declared_timing"](),
        pauli_frame=PauliFrameConfig(commit_us=fabric["DECLARED_US"]["frame"]),
        seed=0)
    return spec.build()


def _windows(completed, n_ops=4):
    return [completed.window_manager.windows[(op_id, 0)]
            for op_id in range(1, n_ops + 1)]


# ---- the plain model, characterized (and the DEC-003 default, criterion G)


def test_unit_occupancy_equals_service_latency(fabric):
    """Non-pipelined: all four windows ready at 12 us, completions spaced
    by exactly the 100 us service latency; occupancy == latency is the
    plain model's meaning, not a bug."""
    windows = _windows(_backlog_run(
        fabric, decoder=PresetLatencyDecoder(SERVICE_US)))
    wbd = fabric["DECLARED_US"]["wbd"]

    assert [window.t_data_complete for window in windows] == [us(12)] * 4
    assert [window.t_done for window in windows] == \
        [us(12 + wbd + SERVICE_US), us(12 + wbd + 2 * SERVICE_US),
         us(12 + wbd + 3 * SERVICE_US), us(12 + wbd + 4 * SERVICE_US)]


def test_input_slot_gives_one_window_of_lookahead_and_no_more(fabric):
    """Non-pipelined: the two-slot unit accepts exactly one waiting
    window's input DMA while computing but never overlaps two computes."""
    windows = _windows(_backlog_run(
        fabric, decoder=PresetLatencyDecoder(SERVICE_US)))

    assert windows[0].t_dispatch == us(12)
    assert windows[1].t_dispatch == us(12)                  # DMA overlap only
    assert windows[2].t_dispatch == windows[0].t_done       # slot frees late
    assert windows[3].t_dispatch == windows[1].t_done


# ---- the pipelined model (criteria A, C, D, F)


def test_pipelined_unit_starts_every_initiation_interval(fabric):
    """Latency 100 us, initiation interval 1 us, ONE unit: six windows
    finish at 117..122 us. Starts respect the 1 us cadence (D), every
    completion is its start + exactly the declared latency (C, F), and
    the cadence beats latency (A)."""
    decoder = PipelinedDecoder(PresetLatencyDecoder(SERVICE_US), INITIATION_US)
    completed = _backlog_run(fabric, decoder=decoder, n_ops=6)
    windows = _windows(completed, n_ops=6)
    wbd = fabric["DECLARED_US"]["wbd"]
    landing = us(12 + wbd)                                  # inputs land at 17

    assert [window.t_dispatch for window in windows] == [us(12)] * 6
    assert [window.t_done for window in windows] == [
        landing + us(k * INITIATION_US + SERVICE_US) for k in range(6)]
    service_starts = [window.t_done - us(SERVICE_US) for window in windows]
    start_gaps = {later - earlier
                  for earlier, later in zip(service_starts, service_starts[1:])}
    assert start_gaps == {us(INITIATION_US)}                # D: exact cadence
    assert all(start >= landing for start in service_starts)  # F: never early


def test_pipeline_depth_bounds_in_flight_work(fabric):
    """Depth 2 (B): the third start waits for the first COMPLETION even
    though the intake stage is free, and the fourth window cannot even
    become resident until the first result leaves the unit's memory (the
    SRAM price of in-flight work stays visible, never impossible
    parallelism). Overload therefore queues (E) instead of exceeding the
    declared capacity."""
    decoder = PipelinedDecoder(PresetLatencyDecoder(SERVICE_US), INITIATION_US,
                               pipeline_depth=2)
    windows = _windows(_backlog_run(fabric, decoder=decoder))
    wbd = fabric["DECLARED_US"]["wbd"]

    assert [window.t_done for window in windows] == \
        [us(117), us(118), us(217), us(222)]
    assert windows[2].t_done - us(SERVICE_US) == windows[0].t_done  # 3rd start = 1st done
    assert windows[3].t_dispatch == windows[0].t_done       # residency freed late
    assert windows[3].t_done == windows[3].t_dispatch + us(wbd + SERVICE_US)


def test_pipelined_timeline_is_deterministic(fabric):
    """H: the depth-2 run has a completion and an initiation release on
    the same tick (118 us); two independent builds resolve them into
    identical full timelines through the engine's fixed order."""
    def timeline():
        decoder = PipelinedDecoder(PresetLatencyDecoder(SERVICE_US),
                                   INITIATION_US, pipeline_depth=2)
        windows = _windows(_backlog_run(fabric, decoder=decoder))
        return [(w.t_queued, w.t_dispatch, w.t_done) for w in windows]

    assert timeline() == timeline()


# ---- refusals and parameter validation


def test_pipelined_strong_primary_is_the_plain_path_and_works(fabric):
    """Strong-primary (StrongOnly) window jobs are plain decodes, so the
    pipelined model serves them: one window, sbd 6 lands at 21 us, done
    at 121 us."""
    decoder = PipelinedDecoder(PresetLatencyDecoder(SERVICE_US), INITIATION_US)
    completed = _backlog_run(fabric, decoder=decoder, n_ops=1, csb=True,
                             escalation_policy=StrongOnly())
    window = completed.window_manager.windows[(1, 0)]
    assert window.t_done == us(21 + SERVICE_US)


def test_pipelined_escalation_route_refuses(fabric):
    """A switching ESCALATION request carries strong_decode_for; routing
    it to a PipelinedDecoder refuses loudly instead of silently
    serializing (the strong escalation tier is not pipelined yet)."""
    from decsim.controller.policies import Held
    from decsim.decoders.decoders import (SAMPLED_CONFIDENCE_SOURCE,
                                          SampledConfidenceDecoder,
                                          SwitchingRouter)
    from decsim.decoders.weak_strong_switching import Switching
    from decsim.windows.windowing_schemes import (SlidingTerminalPolicy,
                                                  SlidingWindowScheme)

    weak = SampledConfidenceDecoder(
        PresetLatencyDecoder(fabric["DECLARED_US"]["weak"]), 1.0)
    router = SwitchingRouter(
        weak=weak,
        strong=PipelinedDecoder(PresetLatencyDecoder(SERVICE_US), INITIATION_US))
    spec = RunSpec(
        ops=[fabric["memory_op"](1)], d=3, rounds_policy=FixedRounds(6),
        scheme=SlidingWindowScheme(
            terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD),
        router=router,
        escalation_policy=Switching(0.5, SAMPLED_CONFIDENCE_SOURCE),
        boundary_policy=Held(),
        unit_pools={"default": 1, "strong": 1},
        links=fabric["declared_profile"](cwb=True, csb=True),
        timing=fabric["declared_timing"](),
        pauli_frame=PauliFrameConfig(commit_us=fabric["DECLARED_US"]["frame"]),
        seed=0)
    with pytest.raises(RuntimeError, match="not pipelined yet"):
        spec.build()


def test_pipelined_decoder_validates_its_parameters():
    inner = PresetLatencyDecoder(SERVICE_US)
    with pytest.raises(ValueError, match="positive"):
        PipelinedDecoder(inner, 0.0)
    with pytest.raises(ValueError, match="rounds to zero"):
        PipelinedDecoder(inner, 1e-9)
    with pytest.raises(ValueError, match="at least 1"):
        PipelinedDecoder(inner, 1.0, pipeline_depth=0)


def test_pipelined_unit_refuses_mixed_inflight_latencies(fabric):
    """In-order completion (criterion H's other half): a pipelined unit
    takes one latency; a job declaring a different latency while another
    is in flight refuses loudly instead of completing out of order."""
    from decsim.decoders.decoders import FunctionLatencyDecoder

    per_op_latency = {1: 100.0, 2: 50.0}
    decoder = PipelinedDecoder(
        FunctionLatencyDecoder(lambda job: per_op_latency[job.op_id]),
        INITIATION_US)
    with pytest.raises(RuntimeError, match="completes in order"):
        _backlog_run(fabric, decoder=decoder, n_ops=2)
