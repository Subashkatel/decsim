"""The utilization metric is a listener, and its numbers are identities.

DecoderUtilization hears the decoder pool's unit_busy and unit_freed,
and is checked against a fact it never sees: the busy integral against
the decode service's own dispatch-to-finish spans. Gate point 1 is the
frozen suite's first strict point (weak_decoder_baseline d 3 p 0.003
seed 0). A time average runs from the run's start to the tick it is
read, as gem5's AvgStor integrates to curTick()
(src/base/stats/storage.hh:130-213).
"""

import types

import decsim.machine as machine_module
import decsim.observe.metrics as metrics
import tests.declared_run as declared_run
import tests.observe.gate_point as gate_point


def test_the_busy_integral_equals_the_services_own_spans():
    """A unit's compute is busy from its job's dispatch to its decode end.

    An identity, not a golden number: the integral the listener stepped
    at the pool's claims and returns equals the sum of the point's nine
    decodes' spans, dispatch to finish, which the decode service reports
    and the listener never sees.
    """
    point = gate_point.settings()
    machine = machine_module.Machine.build(point, gate_point.SEED)
    decodes = declared_run.FinishedDecodes()
    decodes.attach(machine)
    machine.run()

    utilization = machine.observation.decoder_utilization.result()
    spans = [
        decode.finish_ticks - decode.dispatch_ticks
        for decode in decodes.finished
    ]
    busy_ticks = sum(spans)

    assert len(decodes.finished) == 9
    assert utilization["busy_unit_ticks"] == busy_ticks
    assert utilization["aggregate_total_units"] == 1
    span_ticks = utilization["observation_span_ticks"]
    fraction = busy_ticks / span_ticks
    assert utilization["aggregate_busy_fraction"] == fraction


def test_a_unit_busy_a_tenth_of_the_run_is_a_tenth_busy():
    """One unit busy from 10 us to 20 us of a 100 us run: 0.1 busy.

    The average is the integral over the run, from tick 0 to the tick
    it is read (gem5 AvgStor: lastReset at 0, result at curTick), not
    over the span the unit was in use, which would read fully busy.
    """
    engine = types.SimpleNamespace(now=0)
    utilization = metrics.DecoderUtilization(engine, {"weak": 1})
    unit = types.SimpleNamespace(pool="weak")
    engine.now = 10_000_000
    utilization.unit_busy(unit)
    engine.now = 20_000_000
    utilization.unit_freed(unit)
    engine.now = 100_000_000

    result = utilization.result()

    assert result["busy_unit_ticks"] == 10_000_000
    assert result["aggregate_busy_fraction"] == 0.1
