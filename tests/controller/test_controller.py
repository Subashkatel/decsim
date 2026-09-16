"""The controller's intake: a readout becomes a fragment after the crossing.

The model carries no analog waveform; the readout's bits cross
qpu_to_controller as classified bits and become a normalized fragment for
the assembler after the readout-to-bits delay (Fermilab 2406.18807: 40 ns
in-FPGA discrimination; Yang 2605.04892: 20 ns to a syndrome). The link
law is the channel's (tests/links/test_channel.py); here the reference
card prices the hop and the controller adds 3 us on top, and the delay
is charged from the controller clock's next edge (gem5's
src/sim/clocked_object.hh lines 174-186). The instant the readout left
the QPU is the QPU's own event, and the law for it is
tests/qpu/test_cycle_clock.py.
"""

import decsim.config as config
import decsim.controller.controller as controller_module
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records

# a 1 GHz controller, so its period is 1000 ticks and the reference
# card's 0.15 us crossing already lands on one of its edges
CLOCK = config.Clock(1000)
READOUT_TICKS = 3_000_000
SETTINGS = controller_settings.ControllerSettings(
    clock=CLOCK, readout_to_bits_cycles=3000
)
FREE_SETTINGS = controller_settings.ControllerSettings()
# a 1 MHz controller: the same 0.15 us crossing lands mid-cycle
SLOW_CLOCK = config.Clock(1_000_000)
SLOW_SETTINGS = controller_settings.ControllerSettings(
    clock=SLOW_CLOCK, readout_to_bits_cycles=3
)


class RecordingAssembler:
    def __init__(self, engine):
        self.engine = engine
        self.added = []

    def add(self, fragment, fragment_count, route):
        self.added.append((self.engine.now, fragment, fragment_count, route))


def controller_with(engine, links, assembler, settings=SETTINGS):
    return controller_module.Controller(engine, links, settings, assembler)


def test_a_readout_reaches_the_assembler_after_the_crossing_and_the_delay():
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(reference, engine)
    assembler = RecordingAssembler(engine)
    controller = controller_with(engine, links, assembler)
    readout = round_records.QPUReadout(
        7, "patch-a", 4, bits=[True, False, 1, 0], size_bits=4
    )
    crossing_ticks = links.expected_delay_ticks(
        transfer_records.LinkPath.QPU_TO_CONTROLLER, 4, 0
    )

    controller.accept_qpu_readout(readout, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    (added,) = assembler.added
    tick, fragment, fragment_count, route = added
    assert tick == crossing_ticks + READOUT_TICKS
    assert fragment.bits == (1, 0, 1, 0)
    assert fragment.size_bits == 4
    assert fragment_count == 1
    assert route is round_records.WINDOW_INPUT_ROUTE


def test_a_readout_landing_mid_cycle_is_charged_from_the_next_edge():
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(reference, engine)
    assembler = RecordingAssembler(engine)
    controller = controller_with(
        engine, links, assembler, settings=SLOW_SETTINGS
    )
    readout = round_records.QPUReadout(
        7, "patch-a", 4, bits=[True, False, 1, 0], size_bits=4
    )

    controller.accept_qpu_readout(readout, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    (added,) = assembler.added
    # the crossing lands at 150_000, a quarter of the way into the first
    # microsecond, so the three cycles run from the edge at 1_000_000
    assert added[0] == 4_000_000


def test_a_readout_with_no_delay_reaches_the_assembler_at_the_crossing():
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(reference, engine)
    assembler = RecordingAssembler(engine)
    controller = controller_with(
        engine, links, assembler, settings=FREE_SETTINGS
    )
    readout = round_records.QPUReadout(7, "patch-a", 4, bits=[1], size_bits=1)
    crossing_ticks = links.expected_delay_ticks(
        transfer_records.LinkPath.QPU_TO_CONTROLLER, 1, 0
    )

    controller.accept_qpu_readout(readout, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    (added,) = assembler.added
    assert added[0] == crossing_ticks
