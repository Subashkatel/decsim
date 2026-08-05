from typing import get_args, get_type_hints

import pytest

from decsim.config import us
from decsim.decoders import (PerRoundDecoder, PresetLatencyDecoder,
                             SAMPLED_CONFIDENCE_SOURCE,
                             SampledConfidenceDecoder, SwitchingRouter)
from decsim.frontends.circuit import cnot_plus_two_t_circuit, three_cnot_circuit
from decsim.links import (BoundaryTransferRelation, RequestTransferRelation,
                          TrafficAttribution)
from decsim.message import (
    DecodeResult,
    DecoderRequestKey,
    DecoderServiceKey,
    DecoderTier,
    Operation,
    StrongDecodeCompletion,
)
from decsim.metrics import BacklogTrajectory, WindowLatencyBreakdown
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.views import FinalWindowRow, SwitchingRecordsView


def _baseline_recording_run(record):
    return simulate(RunSpec(
        ops=three_cnot_circuit(), num_units=1, d=3,
        rounds_policy=FixedRounds(11), decoder=PresetLatencyDecoder(1.0),
        record_switching_windows=record))


def test_decoder_observability_keys_are_exact_immutable_carriers():
    weak_key = DecoderRequestKey(3, 0, DecoderTier.WEAK, 7)
    service_key = DecoderServiceKey(7)
    completion = StrongDecodeCompletion(
        DecoderRequestKey(3, 0, DecoderTier.STRONG, 8),
        DecodeResult(3, 0),
    )

    assert weak_key.operation_id == 3
    assert service_key.run_sequence == 7
    assert completion.request_key.tier is DecoderTier.STRONG


def test_switching_observability_types_are_resolvable_on_supported_python():
    assert get_type_hints(FinalWindowRow)["selected_request_key"] is not None
    assert get_type_hints(SwitchingRecordsView)["requests"] is not None
    relation = get_type_hints(TrafficAttribution)["relation"]
    assert set(get_args(relation)) == {
        RequestTransferRelation,
        BoundaryTransferRelation,
        type(None),
    }


def test_decoder_observability_keys_reject_ambiguous_identity_values():
    with pytest.raises(ValueError, match="identity"):
        StrongDecodeCompletion(DecoderRequestKey(3, 0, DecoderTier.STRONG, 0), DecodeResult(4, 0))
    with pytest.raises(TypeError):
        DecoderRequestKey(True, 0, DecoderTier.WEAK, 0)
    with pytest.raises(TypeError):
        DecoderRequestKey(3, True, DecoderTier.WEAK, 0)
    with pytest.raises(TypeError):
        DecoderRequestKey(3, 0, "weak", 0)
    with pytest.raises(ValueError):
        DecoderRequestKey(3, -1, DecoderTier.WEAK, 0)
    with pytest.raises(ValueError):
        DecoderServiceKey(-1)


def test_window_switching_records_capture_terminal_weak_requests_and_services():
    completed = _baseline_recording_run(True)

    records = completed.result.metric_values()["window_switching_records"]
    assert records["identity_scope"] == "single_primary_run"
    assert records["tick_unit"] == "ticks"
    assert len(records["windows"]) == 12
    assert len(records["requests"]) == len({row["request_key"]["run_sequence"] for row in records["requests"]}) == 12
    assert len(records["services"]) == len({row["service_key"]["run_sequence"] for row in records["services"]}) == 12
    assert {row["terminal_processing_outcome"] for row in records["requests"]} == {
        "weak_forwarded_for_delivery"
    }
    assert all(row["syndrome_weight"] is None for row in records["requests"])
    assert all(row["selected_request_key"] is not None
               for row in records["windows"])


def test_switching_record_capture_off_allocates_no_terminal_history():
    completed = _baseline_recording_run(False)

    assert completed.decoder_manager._terminal_request_records is None
    assert completed.decoder_manager._terminal_service_records is None
    assert completed.window_manager._selected_request_keys is None
    assert "window_switching_records" not in completed.result.metric_values()


def test_switching_capture_is_same_seed_execution_and_traffic_neutral():
    def run(record):
        weak = SampledConfidenceDecoder(PerRoundDecoder(0.1), 0.0)
        return simulate(RunSpec(
            ops=[Operation(0, "memory", (0,))], d=3, seed=17,
            rounds_policy=FixedRounds(12), scheme=SlidingWindowScheme(),
            strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE,
                               confidence_threshold=0.5, run_both_at_once=True),
            router=SwitchingRouter(weak, PerRoundDecoder(1.0)),
            unit_pools={"default": 1, "strong": 1},
            record_switching_windows=record))

    bare, captured = run(False), run(True)
    assert bare.engine.log_lines == captured.engine.log_lines
    assert bare.result.link_traffic == captured.result.link_traffic
    assert bare.window_manager.op_results == captured.window_manager.op_results
    assert bare.engine.now == captured.engine.now
    assert (bare.decoder_manager.strong_needed,
            bare.decoder_manager.strong_cancelled,
            bare.decoder_manager.queue_log) == (
                captured.decoder_manager.strong_needed,
                captured.decoder_manager.strong_cancelled,
                captured.decoder_manager.queue_log)


def test_window_latency_breakdown_stages():
    r = simulate(RunSpec(
            ops=three_cnot_circuit(),
            num_units=1,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(1.0),
            make_metrics=lambda e, wm, dm, ch, f: [WindowLatencyBreakdown(wm)],
        ), verbose=False)
    breakdown = r.result.metric_values()["window_latency"]
    rows = WindowLatencyBreakdown(r.window_manager).rows()
    assert breakdown["total"]["n"] == 12 and len(rows) == 12
    for row in rows:
        assert row["buffer_fill"] >= 0 and row["dep_block"] >= 0
        assert row["queue_wait"] >= 0
        assert row["service"] == us(1.0)       # PresetLatencyDecoder's constant
        assert (row["buffer_fill"] + row["dep_block"] + row["queue_wait"]
                + row["service"]) == row["total"]
    assert breakdown["dep_block"]["max"] > 0


def test_breakdown_separates_queue_wait_from_dep_block():
    r = simulate(RunSpec(
            ops=three_cnot_circuit(),
            num_units=1,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(5.0),
            make_metrics=lambda e, wm, dm, ch, f: [WindowLatencyBreakdown(wm)],
        ), verbose=False)
    assert r.result.metric_values()["window_latency"]["queue_wait"]["max"] > 0


# ---- per-gate backlog (the r_i of arXiv:2510.25222) -------------------------------------

def test_backlog_trajectory_measures_the_blocked_t_gate():
    r = simulate(RunSpec(
            ops=cnot_plus_two_t_circuit(),
            num_units=2,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(1.0),
            make_metrics=lambda e, wm, dm, ch, f: [BacklogTrajectory(ch)],
        ), verbose=False)
    res = r.result.metric_values()["backlog_trajectory"]
    rows = BacklogTrajectory(r.chip).rows()
    assert res["n"] == 1 and len(rows) == 1
    assert rows[0]["wait"] > 0                              # reaction is never free
    assert rows[0]["backlog_rounds"] == rows[0]["wait"] / r.chip.round_ticks + 11


def test_backlog_trajectory_registration_changes_nothing():
    bare = simulate(RunSpec(
               ops=cnot_plus_two_t_circuit(),
               num_units=2,
               d=3,
               rounds_policy=FixedRounds(11),
               decoder=PresetLatencyDecoder(1.0),
           ), verbose=False)
    metered = simulate(RunSpec(
                  ops=cnot_plus_two_t_circuit(),
                  num_units=2,
                  d=3,
                  rounds_policy=FixedRounds(11),
                  decoder=PresetLatencyDecoder(1.0),
                  make_metrics=lambda e, wm, dm, ch, f: [BacklogTrajectory(ch)],
              ), verbose=False)
    assert bare.engine.log_lines == metered.engine.log_lines


def test_backlog_grows_when_the_decoder_is_too_slow():
    waits = {}
    for lat in (1.0, 50.0):
        r = simulate(RunSpec(
                ops=cnot_plus_two_t_circuit(),
                num_units=2,
                d=3,
                rounds_policy=FixedRounds(11),
                decoder=PresetLatencyDecoder(lat),
                make_metrics=lambda e, wm, dm, ch, f: [BacklogTrajectory(ch)],
            ), verbose=False)
        waits[lat] = r.result.metric_values()["backlog_trajectory"]["max_wait"]
    assert waits[50.0] > waits[1.0]


def test_backlog_rows_cover_fan_out_gating():
    from decsim.frontends.circuit import CircuitFrontend
    from decsim.message import Operation
    ops = CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q1)", (1,), clifford=False, blocked_by=0),
        Operation(2, "C:T(q2)", (2,), clifford=False, blocked_by=0),
    ]).build()
    r = simulate(RunSpec(
            ops=ops,
            num_units=2,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(1.0),
            make_metrics=lambda e, wm, dm, ch, f: [BacklogTrajectory(ch)],
        ), verbose=False)
    rows = BacklogTrajectory(r.chip).rows()
    assert len(rows) == 2
    assert rows[0]["wait"] == rows[1]["wait"] > 0


def test_backlog_rounds_use_the_ops_own_cadence():
    from decsim.codes import SurfaceCodeModel
    from decsim.frontends.circuit import CircuitFrontend
    from decsim.message import Operation
    fast = SurfaceCodeModel(d=3, round_us=0.5)             # != the global 1.1 us
    ops = CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()
    r = simulate(RunSpec(
            ops=ops,
            num_units=2,
            rounds_policy=FixedRounds(11),
            code=fast,
            decoder=PresetLatencyDecoder(1.0),
            make_metrics=lambda e, wm, dm, ch, f: [BacklogTrajectory(ch)],
        ), verbose=False)
    row = BacklogTrajectory(r.chip).rows()[0]
    assert row["backlog_rounds"] == row["wait"] / us(0.5) + 11
