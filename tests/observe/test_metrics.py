"""The two sweep metrics are listeners, and their numbers are identities.

DecoderUtilization hears the decoder pool's unit_busy and unit_freed;
DecoderMemoryOccupancy hears every unit memory's deposited and taken.
Each is checked against a fact it never sees: the busy integral against
the spans the decode records report, the held-round count against the
memory's own dictionary of inputs. Gate point 1 is the frozen suite's
first strict point (weak_decoder_baseline d 3 p 0.003 seed 0).
"""

import decsim.machine as machine_module
import tests.observe.gate_point as gate_point

pytestmark = gate_point.needs_the_frozen_suite


class _MemoryWatcher:
    """The memory's own held count, read at every event it fires."""

    def __init__(self, occupancy, unit) -> None:
        self.occupancy = occupancy
        self.unit = unit
        self.samples: list = []

    def changed(self, _job, _decoder_input) -> None:
        """One deposit or take: the listener's count beside the memory's."""
        counted = self.occupancy.by_unit[self.unit.name].held_rounds
        held = self.unit.memory.occupied_rounds
        self.samples.append((counted, held))


def test_the_busy_integral_equals_the_services_own_spans():
    """A unit's compute is busy from its job's dispatch to its decode end.

    An identity, not a golden number: the integral the listener stepped
    at the pool's claims and returns equals the sum of the spans the
    decode records report for the point's nine services, which the
    listener never sees.
    """
    machine, _result = gate_point.run(
        decoder_utilization=True, record_switching_windows=True
    )

    utilization = machine.observation.decoder_utilization.result()
    services = machine.observation.decode_records.services
    spans = []
    for service in services:
        span = service.terminal_ticks - service.dispatch_ticks
        spans.append(span)
    busy_ticks = sum(spans)

    assert len(services) == 9
    assert utilization["busy_unit_ticks"] == busy_ticks
    assert utilization["aggregate_total_units"] == 1
    span_ticks = utilization["observation_span_ticks"]
    fraction = busy_ticks / span_ticks
    assert utilization["aggregate_busy_fraction"] == fraction


def test_the_memory_occupancy_is_the_memorys_own_count_at_every_change():
    """The listener's steps and the memory's own dictionary agree.

    The point runs one unit, so the watcher connects to the one memory
    after the listener and reads the memory's occupied_rounds at every
    deposit and take: nine of each, and no disagreement.
    """
    point = gate_point.settings(decoder_memory_occupancy=True)
    machine = machine_module.Machine.build(point, gate_point.SEED)
    occupancy = machine.observation.decoder_memory_occupancy
    (unit,) = machine.decoder_manager.pool.units()
    watcher = _MemoryWatcher(occupancy, unit)
    unit.memory.deposited.connect(watcher.changed)
    unit.memory.taken.connect(watcher.changed)

    machine.run()

    assert len(watcher.samples) == 18
    disagreements = [pair for pair in watcher.samples if pair[0] != pair[1]]
    assert disagreements == []
    rows = occupancy.rows()
    snapshot = unit.memory.snapshot()
    assert rows == [
        {
            "unit": "default#0",
            "capacity_rounds": snapshot.capacity_rounds,
            "occupied_rounds": snapshot.occupied_rounds,
            "peak_occupied_rounds": snapshot.peak_occupied_rounds,
            "admissions": snapshot.admissions,
        }
    ]
