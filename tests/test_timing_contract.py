import time

from conftest import fixed_latency_link_config, trace_time

from decsim.config import TICKS_PER_US, us
from decsim.controllers import ModularController
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import DecodeResult, Operation
from decsim.metrics import WindowLatencyBreakdown
from decsim.schemes import NaiveOnlineScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


class BlockingDecoder:
    """A real-ish decoder whose Python work is deliberately slower than its model."""
    # It takes one microsecond of simulated time to decode, but one second of wall clock time. 
    # this makes it easy to verify that the simulated time is working as intended
    def __init__(self, latency_us=1.0, sleep_s=1.0):
        self.latency_ticks = us(latency_us) # uses the us function to convert microseconds to ticks
        self.sleep_s = sleep_s
        self.calls = 0

    def latency(self, job):
        return self.latency_ticks

    def decode(self, job):
        self.calls += 1
        time.sleep(self.sleep_s)
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(0,))


def _zero_link_controller(engine, links):
    return ModularController(engine, links=links, log_syndromes=False)


def test_wall_clock_decode_work_does_not_advance_simulated_service_time():
    """decode() may block Python; simulated service time still comes from latency()."""
    decoder = BlockingDecoder(latency_us=1.0, sleep_s=1.0)
    op = Operation(0, "M(q0)", (0,), clifford=True, patches=(0,))

    t0 = time.perf_counter()
    res = simulate(RunSpec(
              ops=[op],
              num_units=1,
              d=3,
              rounds_policy=FixedRounds(3),
              round_us=1.0,
              decoder=decoder,
              links=fixed_latency_link_config(),
              make_controller=_zero_link_controller,
              make_metrics=lambda e, cl, ch, f: [WindowLatencyBreakdown(cl)],
          ), verbose=False)
    elapsed = time.perf_counter() - t0

    row = WindowLatencyBreakdown(res.cluster).rows()[0]
    assert decoder.calls == 1
    assert elapsed >= decoder.sleep_s
    assert row["service"] == us(1.0)


def test_blocked_operation_waits_for_modeled_decode_time_not_wall_clock_runtime():
    """A feedback-blocked op releases on simulated decode completion, not wall-clock."""
    decoder = BlockingDecoder(latency_us=1.0, sleep_s=1.0)
    ops = CircuitFrontend([
        Operation(0, "T0", (0,), clifford=False, consumes_magic_state=False),
        Operation(1, "T1", (0,), clifford=False, blocked_by=0,
                  consumes_magic_state=False),
    ]).build()

    res = simulate(RunSpec(
              ops=ops,
              num_units=1,
              d=3,
              rounds_policy=FixedRounds(3),
              round_us=1.0,
              decoder=decoder,
              scheme=NaiveOnlineScheme(),
              links=fixed_latency_link_config(),
              make_controller=_zero_link_controller,
          ), verbose=False)

    first_window = res.cluster.windows[(0, 0)]
    assert first_window.t_done - first_window.t_dispatch == us(1.0)
    assert res.chip.decode_release_time[1] == first_window.t_done
    assert trace_time(res.engine.log_lines, "START T1") == (
        first_window.t_done / TICKS_PER_US)
