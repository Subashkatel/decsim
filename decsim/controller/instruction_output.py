"""The controller's output: commands and decisions to the QPU.

Every command is prepared ahead and queued for its start boundary: a
control processor runs ahead of the qubits' timeline and plays each
queued pulse at its time (QICK 2110.00557 Sec. III.A; eQASM 1808.02449
Sec. 3.1). A decision lands here over frame_to_controller, pays the
decision-to-pulse cost of the control processor's branch once (QICK
measures 16 clocks for the conditional and jump and 20 for the next
pulse, Sec. IV.A; QubiC, Fruitwala 2404.15260 Sec. III and IV), and
crosses controller_to_qpu. A release then lets its queued command fire
on the next cycle boundary, so a release that lands before the
operation is otherwise ready costs it nothing: the condition only gates
a pulse already on the timeline (eQASM Sec. 3.5, fast conditional
execution).
"""

import dataclasses
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
    PRELOADED_COMMAND, DECISION_AVAILABLE and CONTROL_DECISION_ISSUED,
    each carrying its payload.
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
        """A queued command starts at the next boundary."""
        operation_id = command.operation.id
        self._fire("PRELOADED_COMMAND", operation_id, command)
        self._start(command, on_started)

    def relay_instruction(
        self,
        decision: program_records.Decision,
        deliver: Callable[[program_records.Decision], None],
    ) -> None:
        """Take a Pauli-frame decision at the landing of frame_to_controller.

        Either kind crosses the output path to the execution runtime, the
        classical program that branches on the outcome, since the QPU device
        holds no register an outcome lands in (QubiC branches on the control
        processor, Fruitwala et al. 2404.15260). A release there lets its
        operation's queued command go.
        """
        self._fire("DECISION_AVAILABLE", decision.target_operation_id, decision)
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
            owner_id=operation_id,
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
                transfer_records.CONTROLLER_TO_QPU,
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
