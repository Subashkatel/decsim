"""The two sweep metrics are listeners, and their numbers are identities.

DecoderUtilization hears the decoder pool's unit_busy and unit_freed;
DecoderMemoryOccupancy hears every unit memory's deposited and taken.
Each is checked against a fact it never sees: the busy integral against
the decode service's own dispatch-to-finish spans, the held-round count
against the memory's own dictionary of inputs. Gate point 1 is the
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


class _MemoryWatcher:
    """The memory's own held count, read at every event it fires."""

    def __init__(self, occupancy, unit) -> None:
        self.occupancy = occupancy
        self.unit = unit
        self.samples: list = []
        self.deposited_inputs: list = []

    def deposited(self, job, decoder_input) -> None:
        """One deposit: the input that landed, then the two counts."""
        self.deposited_inputs.append(decoder_input)
        self.changed(job, decoder_input)

    def changed(self, _job, _decoder_input) -> None:
        """One deposit or take: the listener's count beside the memory's."""
        counted = self.occupancy.by_unit[self.unit.name].held_bits
        held = self.unit.memory.occupied_bits
        self.samples.append((counted, held))


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


def test_the_memory_occupancy_is_the_memorys_own_count_at_every_change():
    """The listener's steps and the memory's own dictionary agree.

    The point runs one unit, so the watcher connects to the one memory
    after the listener and reads the memory's occupied_bits at every
    deposit and take: nine of each, and no disagreement.
    """
    point = gate_point.settings(decoder_memory_occupancy=True)
    machine = machine_module.Machine.build(point, gate_point.SEED)
    occupancy = machine.observation.decoder_memory_occupancy
    (unit,) = machine.decoders.decoder_manager.pool.units
    watcher = _MemoryWatcher(occupancy, unit)
    unit.memory.trace.deposited.connect(watcher.deposited)
    unit.memory.trace.taken.connect(watcher.changed)

    machine.run()

    assert len(watcher.samples) == 18
    listener_bits = [listener for listener, _memory in watcher.samples]
    memory_bits = [memory for _listener, memory in watcher.samples]
    assert listener_bits == memory_bits
    rows = occupancy.rows()
    input_sizes = [
        decoder_input.size_bits() for decoder_input in watcher.deposited_inputs
    ]
    assert rows == [
        {
            "unit": "default#0",
            "capacity_bits": unit.memory.capacity_bits,
            "occupied_bits": unit.memory.occupied_bits,
            "peak_occupied_bits": max(memory_bits),
            "admissions": len(watcher.deposited_inputs),
            "unsized_admission_count": input_sizes.count(None),
        }
    ]


class _HeldInput:
    """A decoder input of a fixed size, as a unit memory hands it on."""

    def __init__(self, bits: int) -> None:
        self.bits = bits

    def held_bits(self) -> int:
        return self.bits

    def size_bits(self) -> int:
        return self.bits


class _UnsizedInput:
    """A decoder input whose rounds state no size, so it holds no bits."""

    def held_bits(self) -> int:
        return 0

    def size_bits(self) -> None:
        return None


def test_an_input_of_no_stated_size_is_admitted_and_counted_apart():
    """An unbounded memory admits rounds of no size and holds no bits.

    They are counted apart, as the links count a transfer of unknown
    width (decsim/observe/link_traffic.py).
    """
    engine = types.SimpleNamespace(now=0)
    occupancy = metrics.DecoderMemoryOccupancy(engine, {"unit": None})
    unsized = _UnsizedInput()
    occupancy.deposited("unit", None, unsized)

    (row,) = occupancy.rows()

    assert row["occupied_bits"] == 0
    assert row["admissions"] == 1
    assert row["unsized_admission_count"] == 1


def test_a_memory_used_a_tenth_of_the_run_is_a_tenth_occupied():
    """100 bits held from 10 us to 20 us of a 100 us run: 10 bits on average.

    The average is the integral over the run, from tick 0 to the tick
    it is read (gem5 AvgStor: lastReset at 0, result at curTick), not
    over the span the memory was in use, which would read full.
    """
    engine = types.SimpleNamespace(now=0)
    occupancy = metrics.DecoderMemoryOccupancy(engine, {"unit": 100})
    held = _HeldInput(100)
    engine.now = 10_000_000
    occupancy.deposited("unit", None, held)
    engine.now = 20_000_000
    occupancy.taken("unit", None, held)
    engine.now = 100_000_000

    result = occupancy.result()

    unit = result["per_unit"]["unit"]
    assert unit["time_avg_occupied_bits"] == 10.0
    assert unit["time_avg_occupied_fraction"] == 0.1
