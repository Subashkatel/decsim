"""ConditionalReactionTime reports SWIPER-style reaction waits."""

from decsim.config import us
from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import DecodeResult, Operation
from decsim.metrics import BacklogTrajectory, ConditionalReactionTime
from decsim.schemes import NaiveOnlineScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


class _FixedLatency:
    """Timing decoder with a fixed service time."""

    def __init__(self, latency_us):
        self.latency_ticks = us(latency_us)

    def latency(self, job):
        return self.latency_ticks

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id)


def _two_conditional_ops():
    return CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q1)", (1,), clifford=False, blocked_by=0),
        Operation(2, "C:T(q2)", (2,), clifford=False, blocked_by=0),
    ]).build()


def test_reaction_time_is_pure_wait_in_rounds():
    """The new metric does not add the blocked operation's own rounds."""
    ops = CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()
    result = simulate(RunSpec(
                 ops=ops,
                 num_units=2,
                 d=3,
                 rounds_policy=FixedRounds(11),
                 decoder=PresetLatencyDecoder(1.0),
                 make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip),
            BacklogTrajectory(chip),
        ],
             ), verbose=False)

    reaction = result["metrics"]["conditional_reaction_time"]
    backlog_row = BacklogTrajectory(result["chip"]).rows()[0]
    wait_rounds = backlog_row["wait"] / result["chip"].round_ticks

    assert reaction["total_conditionals"] == 1
    assert reaction["released_conditionals"] == 1
    assert reaction["avg_conditioned_decode_wait_time"] == wait_rounds
    assert reaction["avg_conditioned_decode_wait_time"] + 11 == backlog_row["backlog_rounds"]


def test_reaction_time_uses_all_conditionals_as_denominator():
    """SWIPER averages over the number of conditional operations."""
    result = simulate(RunSpec(
                 ops=_two_conditional_ops(),
                 num_units=2,
                 d=3,
                 rounds_policy=FixedRounds(11),
                 decoder=PresetLatencyDecoder(1.0),
                 make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip)
        ],
             ), verbose=False)

    reaction = result["metrics"]["conditional_reaction_time"]
    waits = reaction["conditioned_decode_wait_times"]

    assert reaction["success"] is True
    assert reaction["total_conditionals"] == 2
    assert reaction["released_conditionals"] == 2
    assert reaction["avg_conditioned_decode_wait_time"] == sum(waits.values()) / 2


def test_reaction_time_marks_threshold_divergence():
    """A configured threshold marks off-scale reaction waits as divergent."""
    result = simulate(RunSpec(
                 ops=_two_conditional_ops(),
                 num_units=2,
                 d=3,
                 rounds_policy=FixedRounds(11),
                 decoder=PresetLatencyDecoder(1.0),
                 make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip, divergence_threshold_rounds=0.5)
        ],
             ), verbose=False)

    reaction = result["metrics"]["conditional_reaction_time"]

    assert reaction["success"] is False
    assert reaction["failed"] is True
    assert reaction["diverged"] is True
    assert "conditioned wait exceeded" in reaction["failure_reason"]


def test_reaction_time_marks_idle_cap_failure():
    """If the chip safety cap fires, the reaction-time run is not trustworthy."""
    ops = CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False, consumes_magic_state=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                  consumes_magic_state=False),
    ]).build()
    result = simulate(RunSpec(
                 ops=ops,
                 num_units=1,
                 d=3,
                 rounds_policy=FixedRounds(3),
                 round_us=1.0,
                 scheme=NaiveOnlineScheme(),
                 decoder=_FixedLatency(50.0),
                 max_idle_rounds=1,
                 make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip)
        ],
             ), verbose=False)

    reaction = result["metrics"]["conditional_reaction_time"]

    assert result["chip"].idle_cap_hits
    assert reaction["success"] is False
    assert reaction["failed"] is True
    assert reaction["failure_reason"] == "idle-round cap reached"


def test_final_non_clifford_decode_does_not_return_to_chip_by_default():
    """A final T result can stay in the orchestrator unless the caller asks otherwise."""
    ops = CircuitFrontend([
        Operation(0, "T0", (0,), clifford=False, consumes_magic_state=False),
    ]).build()
    result = simulate(RunSpec(
                 ops=ops,
                 num_units=1,
                 d=3,
                 rounds_policy=FixedRounds(5),
                 round_us=1.0,
                 decoder=PresetLatencyDecoder(2.0),
             ), verbose=False)

    assert result["cluster"].windows[(0, 0)].t_done is not None
    assert result["chip"].result_return_time_by_operation == {}
    assert not any("DISPATCH result return" in line
                   for line in result["engine"].log_lines)


def test_explicit_result_return_to_chip_uses_feedback_links():
    """A marked final result returns through decoder->orchestrator->controller->chip."""
    ops = CircuitFrontend([
        Operation(
            0,
            "T0",
            (0,),
            clifford=False,
            consumes_magic_state=False,
            requires_result_return_to_chip=True,
        ),
    ]).build()
    result = simulate(RunSpec(
                 ops=ops,
                 num_units=1,
                 d=3,
                 rounds_policy=FixedRounds(5),
                 round_us=1.0,
                 decoder=PresetLatencyDecoder(2.0),
             ), verbose=False)

    controller = result["controller"]
    window_done = max(
        window.t_done
        for key, window in result["cluster"].windows.items()
        if key[0] == 0
    )
    expected_return = (
        window_done
        + controller.links.do.cost()
        + controller.links.oc.cost()
        + controller.links.cq.cost()
    )

    assert result["chip"].decode_release_time == {}
    assert result["chip"].result_return_time_by_operation[0] == expected_return
    assert result["fully_done"] == expected_return
    assert any("DISPATCH result return for op#0" in line
               for line in result["engine"].log_lines)
    assert any("received result return for T0" in line
               for line in result["engine"].log_lines)


def test_naive_batch_decode_label_shows_absorbed_idle_rounds():
    """The decode label reports the enlarged feedback-to-feedback batch."""
    ops = CircuitFrontend([
        Operation(0, "T0", (0,), clifford=False, consumes_magic_state=False),
        Operation(1, "T1", (0,), clifford=False, blocked_by=0,
                  consumes_magic_state=False),
    ]).build()
    result = simulate(RunSpec(
                 ops=ops,
                 num_units=1,
                 d=3,
                 rounds_policy=FixedRounds(27),
                 round_us=1.0,
                 scheme=NaiveOnlineScheme(),
                 decoder=_FixedLatency(20.0),
                 max_idle_rounds=1000,
             ), verbose=False)

    assert result["cluster"].windows[(1, 0)].n_rounds > 27
    assert any("T1 [whole op," in line and "idle + 27 body" in line
               for line in result["engine"].log_lines)
