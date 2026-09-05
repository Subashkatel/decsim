"""The controller's intake: a readout becomes a fragment after the crossing.

The model carries no analog waveform; the readout's bits cross
qpu_to_controller as classified bits and become a normalized fragment for
the assembler after the readout-to-bits delay (Fermilab 2406.18807: 40 ns
in-FPGA discrimination; Yang 2605.04892: 20 ns to a syndrome). The link
law is the channel's (tests/links/test_channel.py); here the reference
card prices the hop and the controller adds 3 us on top.
"""

import decsim.config as config
import decsim.controller.controller as controller_module
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.message as message
import decsim.observe.round_events as round_events

READOUT_TICKS = config.microseconds_to_ticks(3.0)


class RecordingAssembler:
    def __init__(self, engine):
        self.engine = engine
        self.added = []

    def add(self, fragment, fragment_count, route):
        self.added.append((self.engine.now, fragment, fragment_count, route))


def controller_with(engine, links, assembler, recorder):
    return controller_module.Controller(
        engine,
        qpu=None,
        window_manager=None,
        assembler=assembler,
        recorder=recorder,
        measurement_signal_to_classical_bits_ticks=READOUT_TICKS,
        links=links,
        resolved_operations=(),
        resolved_patches=(),
        idle_policy=None,
        feedback_streams=None,
    )


def test_a_readout_reaches_the_assembler_after_the_crossing_and_the_delay():
    engine = engine_module.Engine(verbose=False)
    reference = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(reference, engine)
    assembler = RecordingAssembler(engine)
    recorder = round_events.RoundEventRecorder(engine)
    controller = controller_with(engine, links, assembler, recorder)
    readout = message.QPUReadout(
        7, "patch-a", 4, bits=[True, False, 1, 0], size_bits=4
    )
    crossing_ticks = links.expected_delay_ticks(
        message.LinkPath.QPU_TO_CONTROLLER, 4, 0
    )

    controller.accept_qpu_readout(readout, message.WINDOW_INPUT_ROUTE)
    engine.run()

    (added,) = assembler.added
    tick, fragment, fragment_count, route = added
    assert tick == crossing_ticks + READOUT_TICKS
    assert fragment.bits == (1, 0, 1, 0)
    assert fragment.size_bits == 4
    assert fragment_count == 1
    assert route is message.WINDOW_INPUT_ROUTE
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("EMITTED", 0)]


def test_a_readout_with_no_delay_reaches_the_assembler_at_the_crossing():
    engine = engine_module.Engine(verbose=False)
    reference = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(reference, engine)
    assembler = RecordingAssembler(engine)
    recorder = round_events.NoRoundEvents()
    controller = controller_with(engine, links, assembler, recorder)
    controller.measurement_signal_to_classical_bits_ticks = 0
    readout = message.QPUReadout(7, "patch-a", 4, bits=[1], size_bits=1)
    crossing_ticks = links.expected_delay_ticks(
        message.LinkPath.QPU_TO_CONTROLLER, 1, 0
    )

    controller.accept_qpu_readout(readout, message.WINDOW_INPUT_ROUTE)
    engine.run()

    (added,) = assembler.added
    assert added[0] == crossing_ticks
