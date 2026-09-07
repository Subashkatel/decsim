"""The controller's intake: a QPU readout becomes a fragment for the assembler.

The model carries no analog waveform. A readout's classified bits cross
qpu_to_controller, pay the readout-to-bits cost (40 ns in-FPGA
discrimination, Fermilab 2406.18807; 20 ns to a syndrome, Yang
2605.04892), and reach the round assembler as one fragment. The rest of
the controller is its own components: the issuer (operation_issue.py)
turns admitted operations into commands, the output
(instruction_output.py) carries commands and decisions to the QPU, the
idle accounting (idle_rounds.py) routes and charges idle rounds, and the
feedback streams (feedback_streams.py) keep the protected regions.
"""

import decsim.controller.settings as controller_settings
import decsim.message as message
import decsim.observe.trace_source as trace_source
import decsim.records.rounds as round_records


class Controller:
    """The readout intake; implements the ReadoutReceiver port.

    Trace sources: round_event(RoundEvent) with kind EMITTED as a readout
    leaves the QPU; copy_made(round_key, bits, "readout", "controller
    intake") for the intake's copy of the bits (data_path.md hop 1).
    """

    def __init__(
        self,
        engine,
        link,
        settings: controller_settings.ControllerSettings,
        assembler,
    ) -> None:
        self.engine = engine
        self.link = link
        self.settings = settings
        self.assembler = assembler
        self.round_event = trace_source.TraceSource()
        self.copy_made = trace_source.TraceSource()

    def accept_qpu_readout(
        self,
        readout: round_records.QPUReadout,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        """One readout crosses qpu_to_controller to the assembler.

        The fragment reaches the assembler after the readout delay.
        """
        fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
        fragment_count = readout.n_fragments
        round_key = (fragment.operation_id, fragment.round_index)
        self.copy_made.fire(
            round_key, readout.size_bits, "readout", "controller intake"
        )
        emitted = round_records.RoundEvent.of(
            "EMITTED",
            self.engine.now,
            fragment.operation_id,
            fragment.round_index,
            route,
            fragment.patch_id,
        )
        self.round_event.fire(emitted)
        attribution = message.TransferAttribution.for_round(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index
        )
        readout_ticks = self.settings.readout_to_bits_ticks()

        def receive():
            self.assembler.add(fragment, fragment_count, route)

        def at_controller(_transfer):
            if readout_ticks == 0:
                receive()
                return
            self.engine.schedule(
                readout_ticks, receive, label="controller-binary-availability"
            )

        self.link.send(
            message.LinkPath.QPU_TO_CONTROLLER,
            readout.size_bits,
            self.engine.now,
            attribution,
            at_controller,
        )
