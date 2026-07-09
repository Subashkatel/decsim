"""The idle-round emitter's safety cap (chip.max_idle_rounds).

Regression for a silent-distortion hazard: the emitter used to stop at a hard-coded
100*d rounds with NO trace of having done so -- a long-reaction (backlog/divergence)
study would read artificially stable numbers. The cap is now a constructor knob and
the chip logs a loud WARNING when it fires while a blocked successor is still waiting."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import Operation
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


def _t_then_blocked_t():
    """A non-Clifford op whose successor is blocked on its decode, making
    the patch idle in storage and emit memory rounds until the correction returns."""
    return CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()


def test_cap_fires_loudly_and_stops_emission():
    """With a reaction wait far longer than the cap: exactly `cap` memory rounds are
    emitted, the WARNING names the knob, and the workload still completes."""
    r = simulate(RunSpec(
            ops=_t_then_blocked_t(),
            num_units=1,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(200.0),
            max_idle_rounds=10,
        ), verbose=False)
    lines = r["engine"].log_lines
    assert sum("memory round for A:T(q0)" in l for l in lines) == 10
    assert any("hit the idle-round cap" in l and "max_idle_rounds=10" in l
               for l in lines)
    assert any("START B:T(q0)" in l for l in lines)          # run completed anyway


def test_default_cap_is_unchanged_and_silent():
    """No knob: the cap stays 100*d and a normal run never logs the warning."""
    r = simulate(RunSpec(
            ops=_t_then_blocked_t(),
            num_units=1,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(1.0),
        ), verbose=False)
    assert r["chip"].max_idle_rounds == 100 * 3
    assert not any("hit the idle-round cap" in l for l in r["engine"].log_lines)
