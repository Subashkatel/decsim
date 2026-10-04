"""The controller's output: commands and decisions to the QPU.

A decision lands here over frame_to_controller. A release starts its
operation's command; a result return is reported to the QPU. Either then
pays the decision-to-pulse cost of the control processor's issue
pipeline (QubiC, Fruitwala 2404.15260 Sec. III and IV; QICK 2110.00557
measures 16 clocks for the conditional and jump and 20 for the next
pulse), crosses controller_to_qpu, and starts on the QPU's next cycle
boundary. A preloaded command skips the output path: it was prepared
before the simulated interval.
"""

import dataclasses
import functools
from collections.abc import Callable

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class InstructionOutput:
    """The digital-to-QPU path; implements the InstructionReceiver port.

    Trace source: output_event(ControllerOutputEvent) with kinds
    PRELOADED_COMMAND, DECISION_AVAILABLE, CONTROL_DECISION_ISSUED and
    CONTROL_PULSE_COMMAND_ISSUED, each carrying its payload.
    """

    link = ports.Port(ports.Link)
    qpu = ports.Port(ports.Qpu)

    def __init__(
        self,
        engine: engine_module.Engine,
        clock: config.Clock,
        pulse_cycles: int,
    ) -> None:
        self.engine = engine
        self.clock = clock
        self.pulse_cycles = pulse_cycles
        self.trace = _TraceSources()

    def start_preloaded(
        self,
        command: program_records.RunOperationBody,
        on_started: Callable[[int], None],
    ) -> None:
        """A command prepared before the run starts at the next boundary."""
        operation_id = command.operation.id
        self._fire("PRELOADED_COMMAND", operation_id, command)
        self._start(command, on_started)

    def send_command(
        self,
        command: program_records.RunOperationBody,
        on_started: Callable[[int], None],
    ) -> None:
        """A feedback-selected command pays the pulse cost and the crossing."""
        operation_id = command.operation.id
        deliver = functools.partial(self._start, on_started=on_started)
        self._send(
            command, operation_id, "CONTROL_PULSE_COMMAND_ISSUED", deliver
        )

    def relay_instruction(
        self,
        decision: program_records.Decision,
        deliver: Callable[[program_records.Decision], None],
    ) -> None:
        """Take a Pauli-frame decision at the landing of frame_to_controller.

        A release is consumed here and its command crosses the output path in
        send_command. A result return crosses the output path to the execution
        runtime, the classical program that branches on the outcome, since the
        QPU device holds no register an outcome lands in (QubiC branches on the
        control processor, Fruitwala et al. 2404.15260).
        """
        self._fire("DECISION_AVAILABLE", decision.target_operation_id, decision)
        if decision.releases_operation:
            deliver(decision)
            return
        self._send(
            decision,
            decision.target_operation_id,
            "CONTROL_DECISION_ISSUED",
            deliver,
        )

    def finish(self) -> None:
        """The program is complete: the QPU's idle patches stop."""
        self.qpu.finish()

    def _start(
        self,
        command: program_records.RunOperationBody,
        on_started: Callable[[int], None],
    ) -> None:
        """The command is at the QPU: it starts on the next cycle boundary."""
        self.qpu.issue(command)
        boundary = self.qpu.next_boundary()
        on_started(boundary)

    def _send(self, payload, operation_id, event_kind: str, deliver) -> None:
        """Pay the pulse cost, then cross controller_to_qpu to deliver."""
        attribution = transfer_records.TransferAttribution(
            operation_id=operation_id,
            patch_ids=(),
            window_id=None,
            first_round=None,
            last_round=None,
        )

        def delivered(_transfer):
            deliver(payload)

        def output_ready():
            self._fire(event_kind, operation_id, payload)
            self.link.send(
                transfer_records.LinkPath.CONTROLLER_TO_QPU,
                None,
                self.engine.now,
                attribution,
                delivered,
            )

        delay = self._pulse_delay()
        self.engine.schedule(
            delay, output_ready, label="controller-output-ready"
        )

    def _pulse_delay(self) -> int:
        """The ticks to the edge the pulse processing ends on; zero is free."""
        if self.pulse_cycles == 0:
            return 0
        now = self.engine.now
        edge = self.clock.edge(self.pulse_cycles, now)
        return edge - now

    def _fire(self, kind: str, operation_id, payload) -> None:
        event = round_records.ControllerOutputEvent(
            kind, self.engine.now, operation_id, payload
        )
        self.trace.output_event.fire(event)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the instruction output reports, as one member."""

    output_event: trace_source.TraceSource = trace_source.new_source()
