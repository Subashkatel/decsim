"""End-to-end simulate() smoke tests (real engine, real parts, fake decoder)."""
import pytest

from decsim.decoders import PerRoundDecoder
from decsim.run_spec import simulate
from decsim.message import Operation
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec


def _two_op_clifford():
    a = Operation(0, "A:CNOT(q0,q1)", (0, 1), clifford=True, has_successor=True)
    b = Operation(1, "B:CNOT(q0,q1)", (0, 1), clifford=True, predecessors=(0,))
    return [a, b]


def test_simulate_two_clifford_ops_end_to_end():
    res = simulate(RunSpec(ops=_two_op_clifford(),
                           decoder=PerRoundDecoder(0.5),
                           rounds_policy=FixedRounds(11), num_units=2))
    assert res["chip_done"] == 22 * 1_100_000            # 2 ops x 11 rounds, serial
    assert res["fully_done"] > res["chip_done"]          # decode + delivery tail
    assert set(res["cluster"].op_results) <= {0, 1}
    assert res["chip"].body_done_time[1] == res["chip_done"]


def test_simulate_blocked_t_reaction_path():
    ops = [Operation(0, "A:T(q0)", (0,), clifford=False, has_successor=True),
           Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                     predecessors=(0,))]
    res = simulate(RunSpec(ops=ops, decoder=PerRoundDecoder(0.5),
                           rounds_policy=FixedRounds(11)))
    gate = res["chip"]
    assert 1 in gate.decode_released                     # Decision released B
    assert gate.decode_release_time[1] > gate.body_done_time[0]
    assert res["cluster"].memory_rounds_total > 0        # idle rounds while blocked


def test_runspec_validation():
    with pytest.raises(ValueError, match="exactly one"):
        RunSpec().validate()
    with pytest.raises(ValueError, match="decoder"):
        RunSpec(ops=_two_op_clifford()).build()
    spec = RunSpec(ops=_two_op_clifford(), decoder=PerRoundDecoder(0.5))
    spec.validate()                                      # defaults are coherent


def test_build_runs_validate_first():
    """build() enforces validate() (Codex review finding): an invalid
    feedback_boundary_mode must raise at build, not silently misbehave."""
    import pytest as _pytest
    from decsim.planner import FixedRounds
    from decsim.run_spec import RunSpec
    with _pytest.raises(ValueError, match="feedback_boundary_mode"):
        RunSpec(ops=[], rounds_policy=FixedRounds(11),
                feedback_boundary_mode="bogus").build()


def test_validate_rejects_ambiguous_stream_configs():
    """Codex API review: an op in decode_ops AND dynamic_streams raised a
    duplicate-lease crash deep in PayloadStore; now a clear validate error."""
    import pytest as _pytest
    from decsim.planner import FixedRounds
    from decsim.run_spec import RunSpec
    from decsim.message import Operation as Op
    stream = Op(0, "S(q0)", (0,), clifford=True)
    with _pytest.raises(ValueError, match="decode_ops and dynamic_streams"):
        RunSpec(ops=[], rounds_policy=FixedRounds(11),
                decode_ops=[stream], dynamic_streams=[stream]).build()


def test_rounds_policy_below_one_raises():
    """A rounds policy that yields < 1 round surfaces a clear validation
    error rather than running an empty operation."""
    with pytest.raises(ValueError, match="FixedRounds must give >= 1"):
        simulate(RunSpec(
            ops=[],
            rounds_policy=FixedRounds(0),
            decoder=PerRoundDecoder(tau_us=1.0),
        ), verbose=False)
