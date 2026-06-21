"""ConditionalReactionTime reports SWIPER-style reaction waits."""

from decsim.config import us
from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import DecodeResult, Operation
from decsim.metrics import BacklogTrajectory, ConditionalReactionTime
from decsim.schemes import NaiveOnlineScheme
from decsim.wiring import build_and_run


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
    result = build_and_run(
        ops,
        num_units=2,
        d=3,
        rounds_per_op=11,
        decoder=PresetLatencyDecoder(1.0),
        make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip),
            BacklogTrajectory(chip),
        ],
        verbose=False,
    )

    reaction = result["metrics"]["conditional_reaction_time"]
    backlog_row = BacklogTrajectory(result["chip"]).rows()[0]
    wait_rounds = backlog_row["wait"] / result["chip"].round_ticks

    assert reaction["total_conditionals"] == 1
    assert reaction["released_conditionals"] == 1
    assert reaction["avg_conditioned_decode_wait_time"] == wait_rounds
    assert reaction["avg_conditioned_decode_wait_time"] + 11 == backlog_row["backlog_rounds"]


def test_reaction_time_uses_all_conditionals_as_denominator():
    """SWIPER averages over the number of conditional operations."""
    result = build_and_run(
        _two_conditional_ops(),
        num_units=2,
        d=3,
        rounds_per_op=11,
        decoder=PresetLatencyDecoder(1.0),
        make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip)
        ],
        verbose=False,
    )

    reaction = result["metrics"]["conditional_reaction_time"]
    waits = reaction["conditioned_decode_wait_times"]

    assert reaction["success"] is True
    assert reaction["total_conditionals"] == 2
    assert reaction["released_conditionals"] == 2
    assert reaction["avg_conditioned_decode_wait_time"] == sum(waits.values()) / 2


def test_reaction_time_marks_threshold_divergence():
    """A configured threshold marks off-scale reaction waits as divergent."""
    result = build_and_run(
        _two_conditional_ops(),
        num_units=2,
        d=3,
        rounds_per_op=11,
        decoder=PresetLatencyDecoder(1.0),
        make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip, divergence_threshold_rounds=0.5)
        ],
        verbose=False,
    )

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
    result = build_and_run(
        ops,
        num_units=1,
        d=3,
        rounds_per_op=3,
        round_us=1.0,
        scheme=NaiveOnlineScheme(),
        decoder=_FixedLatency(50.0),
        max_idle_rounds=1,
        make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip)
        ],
        verbose=False,
    )

    reaction = result["metrics"]["conditional_reaction_time"]

    assert result["chip"].idle_cap_hits
    assert reaction["success"] is False
    assert reaction["failed"] is True
    assert reaction["failure_reason"] == "idle-round cap reached"
