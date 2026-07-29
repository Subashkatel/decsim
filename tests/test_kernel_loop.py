"""End-to-end simulate() smoke tests (real engine, real parts, fake decoder)."""

import pytest

from decsim.decoders import PerRoundDecoder
from decsim.run_spec import CompletedRun, simulate
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
    assert isinstance(res, CompletedRun)
    assert res.result.chip_done_ticks == 22 * 1_100_000            # 2 ops x 11 rounds, serial
    assert res.result.fully_done_ticks > res.result.chip_done_ticks          # decode + delivery tail
    assert set(res.window_manager.op_results) <= {0, 1}
    assert res.chip.body_done_time[1] == res.result.chip_done_ticks
    assert res.result.terminal_status == "complete"
    assert res.result.event_queue_empty
    assert res.result.decode_work_settled
    assert res.result.chip_workload_complete
    with pytest.raises(RuntimeError, match="completed"):
        res.engine.run()
    with pytest.raises(RuntimeError, match="completed"):
        res.engine.schedule(0, lambda: None)


def test_completed_run_replaces_the_ambiguous_world_name():
    import decsim.run_spec as run_spec_module

    assert not hasattr(run_spec_module, "World")


def test_completed_run_is_replayable():
    def run():
        return simulate(RunSpec(
            ops=_two_op_clifford(),
            decoder=PerRoundDecoder(0.5),
            rounds_policy=FixedRounds(11),
            num_units=2,
            seed=17,
        ))

    first = run()
    second = run()

    assert first.result == second.result


def test_simulate_blocked_t_reaction_path():
    ops = [Operation(0, "A:T(q0)", (0,), clifford=False, has_successor=True),
           Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                     predecessors=(0,))]
    res = simulate(RunSpec(ops=ops, decoder=PerRoundDecoder(0.5),
                           rounds_policy=FixedRounds(11)))
    gate = res.chip
    assert 1 in gate.decode_released                     # Decision released B
    assert gate.decode_release_time[1] > gate.body_done_time[0]
    assert res.window_manager.memory_rounds_total > 0        # idle rounds while blocked


def test_runspec_build_contracts():
    with pytest.raises(ValueError, match="exactly one"):
        RunSpec().build()
    with pytest.raises(ValueError, match="decoder"):
        RunSpec(ops=_two_op_clifford()).build()
    RunSpec(ops=[], decoder=PerRoundDecoder(0.5)).build()


def test_build_rejects_invalid_feedback_boundary_mode():
    import pytest as _pytest
    from decsim.planner import FixedRounds
    from decsim.run_spec import RunSpec
    with _pytest.raises(ValueError, match="feedback_boundary_mode"):
        RunSpec(ops=[], rounds_policy=FixedRounds(11),
                feedback_boundary_mode="bogus").build()


def test_build_rejects_ambiguous_stream_configs():
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
