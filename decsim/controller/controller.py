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

import dataclasses

import decsim.controller.settings as controller_settings
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class Controller:
    """The readout intake; implements the ReadoutReceiver port.

    Trace source: copy_made(round_key, bits, "readout", "controller
    intake") for the intake's copy of the bits (data_path.md hop 1). The
    instant the readout left is the QPU's own event (qpu/cycle_clock.py).
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
        self.trace = _TraceSources()

    def accept_qpu_readout(
        self,
        readout: round_records.QPUReadout,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        """One readout crosses qpu_to_controller to the assembler.

        The fragment reaches the assembler after the readout delay.
        """
        fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
        fragment_count = readout.fragment_count
        round_key = (fragment.operation_id, fragment.round_index)
        self.trace.copy_made.fire(
            round_key, readout.size_bits, "readout", "controller intake"
        )
        attribution = transfer_records.TransferAttribution.for_round(
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
            transfer_records.LinkPath.QPU_TO_CONTROLLER,
            readout.size_bits,
            self.engine.now,
            attribution,
            at_controller,
        )


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the controller reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
