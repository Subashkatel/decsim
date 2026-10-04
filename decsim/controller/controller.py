"""The controller's intake: a QPU readout becomes a fragment for the assembler.

The model carries no analog waveform. A readout's classified bits cross
qpu_to_controller, pay the readout-to-bits cost (40 ns in-FPGA
discrimination, Fermilab 2406.18807; 32 ns IQ demodulation plus 4 ns
classification, Yang 2605.04892) and reach the assembler as one
fragment.
"""

import dataclasses

import decsim.controller.round_assembly as round_assembly
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class Controller:
    """The readout intake; implements the ReadoutReceiver port.

    copy_made fires for the intake's copy of the bits (data_path.md hop 1).
    """

    link = ports.Port(ports.Link)
    assembler = ports.Port(round_assembly.RoundAssembler)

    def __init__(
        self,
        engine: engine_module.Engine,
        settings: controller_settings.ControllerSettings,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.trace = _TraceSources()

    def accept_qpu_readout(
        self,
        readout: round_records.QPUReadout,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        """Send one readout over qpu_to_controller; it then pays the delay."""
        fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
        self.assembler.expect_round(fragment, readout.fragment_count, route)
        round_key = (fragment.operation_id, fragment.round_index)
        self.trace.copy_made.fire(
            round_key, readout.size_bits, "readout", "controller intake"
        )
        attribution = transfer_records.TransferAttribution.for_round(
            fragment.operation_id, fragment.patch_ids, fragment.round_index
        )
        readout_cycles = self.settings.readout_to_bits_cycles

        def receive():
            self.assembler.add(fragment, route)

        def at_controller(_transfer):
            if readout_cycles == 0:
                receive()
                return
            now = self.engine.now
            delay = self.settings.clock.ticks_to_edge(readout_cycles, now)
            self.engine.schedule(
                delay, receive, label="controller-binary-availability"
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
    """Every event the controller reports, as one member."""

    copy_made: trace_source.TraceSource = trace_source.new_source()
