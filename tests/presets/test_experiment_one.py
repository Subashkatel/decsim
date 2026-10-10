"""Experiment 1's machine resolves, with its fiber loop where it runs.

The referent is the record each machine resolves to (topology/legacy.py)
and the fiber network the preset cites: a round goes up in 40 cycles of
the 250 MHz chip clock (2603.16203 lines 895-897), the correction comes
back in 41 (lines 903-904), on four lanes of 38.79 bits a cycle (lines
971-975), and the frame is read in one cycle; the strong side is
Backline's CPU echo over RoCE v2 (2609.09270 Table III), each leg half
of 2.305 us less the 16-byte echo's 1.28 ns on the 100 Gb/s cable.
"""

import fractions

import decsim.presets.experiment_one as experiment_one
import decsim.presets.machines as machines
import decsim.topology.fabrics as fabrics
import decsim.topology.legacy as legacy

# 40 and 41 cycles of 4000 ticks
UPLINK_TICKS = 160_000
DOWNLINK_TICKS = 164_000
FRAME_READ_TICKS = 4000
# four lanes of 38.79 bits a 4 ns cycle
FIBER_BITS_PER_MICROSECOND = fractions.Fraction(38790)
# 2.305 / 2 us less 128 bits at 100000 bits per us
CABLE_LEG_TICKS = 1_151_220


def test_the_weak_machine_resolves_with_every_port_bound():
    machine = experiment_one.weak_alone(3, 0.001)

    settings = legacy.resolved(machine)

    assert machine.unbound_ports() == []
    assert settings.weak_decoder == experiment_one.weak_decoder()
    assert settings.strong_decoder is None
    assert settings.switching is None


def test_the_strong_machine_drops_the_weak_pool_and_keeps_its_store():
    strong_pool = experiment_one.strong_decoder_pool()
    machine = experiment_one.strong_alone(3, 0.001, strong_pool)

    settings = legacy.resolved(machine)

    part_names = [part.name for part in machine.parts]
    assert machine.unbound_ports() == []
    assert machines.WEAK_POOL not in part_names
    assert machines.WEAK_STORE_WRITE in machine.wires
    assert settings.weak_decoder is None
    assert settings.strong_decoder == strong_pool


def test_the_switching_machine_escalates_union_find_to_relay_belief():
    machine = experiment_one.switching(3, 0.001)

    settings = legacy.resolved(machine)

    threshold = settings.switching.threshold
    assert machine.unbound_ports() == []
    assert machines.ESCALATION in machine.wires
    assert settings.weak_decoder == experiment_one.weak_decoder()
    assert settings.strong_decoder == experiment_one.strong_decoder_pool()
    assert threshold.threshold_decibels == 20.0


def test_the_pools_sit_on_the_chip_and_the_host():
    machine = experiment_one.switching(3, 0.001)

    weak_pool = machine.part(machines.WEAK_POOL)
    strong_pool = machine.part(machines.STRONG_POOL)

    assert weak_pool.placed == fabrics.FRIDGE
    assert strong_pool.placed == fabrics.HOST


def test_the_fiber_loop_carries_the_weak_tiers_write_read_and_correction():
    machine = experiment_one.weak_alone(3, 0.001)

    settings = legacy.resolved(machine)
    links = settings.links

    uplink = links.controller_to_weak_buffer
    frame_read = links.weak_buffer_to_weak_decoder
    downlink = links.weak_decoder_to_frame
    uplink_rate = uplink.channel.capacity.input_bits_per_microsecond
    downlink_rate = downlink.channel.capacity.input_bits_per_microsecond
    assert uplink.channel.propagation_latency_ticks == UPLINK_TICKS
    assert frame_read.channel.propagation_latency_ticks == FRAME_READ_TICKS
    assert downlink.channel.propagation_latency_ticks == DOWNLINK_TICKS
    assert uplink_rate == FIBER_BITS_PER_MICROSECOND
    assert downlink_rate == FIBER_BITS_PER_MICROSECOND
    assert frame_read.channel.capacity is None


def test_only_the_uplink_times_the_wire_alone():
    """The uplink leaves out the root's 20 ns (2603.16203 lines 897-899)."""
    machine = experiment_one.weak_alone(3, 0.001)

    settings = legacy.resolved(machine)
    links = settings.links

    assert links.controller_to_weak_buffer.excludes_receiver_processing
    assert not links.weak_buffer_to_weak_decoder.excludes_receiver_processing
    assert not links.weak_decoder_to_frame.excludes_receiver_processing


def test_the_strong_side_rides_backlines_cpu_echo():
    machine = experiment_one.switching(3, 0.001)

    settings = legacy.resolved(machine)
    links = settings.links

    escalation = links.weak_decoder_to_strong_decoder.channel
    reply = links.strong_decoder_to_frame.channel
    poll = links.strong_buffer_to_strong_decoder.channel
    assert escalation.propagation_latency_ticks == CABLE_LEG_TICKS
    assert reply.propagation_latency_ticks == CABLE_LEG_TICKS
    assert escalation.capacity.input_bits_per_microsecond == 100_000
    assert poll.propagation_latency_ticks == 0
