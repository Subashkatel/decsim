"""The controller's output: commands and decisions to the QPU.

A decision from the Pauli frame lands here over frame_to_controller,
which the frame side executes, and is available at the controller the
instant it lands; a release starts its operation's command,
a result return is reported to the QPU. Either payload then pays the
decision-to-pulse cost (a 42 ns conditional jump and a 52 ns next pulse
on QICK, 2110.00557 Table II) and crosses controller_to_qpu, and the
QPU starts a command on its next cycle boundary. A preloaded command
(a program root, an ordinary successor) skips the output path: its
controller preparation happened before the simulated interval.
"""

import dataclasses
import functools
from typing import Callable

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

    def __init__(self, engine, link, qpu, pulse_ticks: int) -> None:
        self.engine = engine
        self.link = link
        self.qpu = qpu
        self.pulse_ticks = pulse_ticks
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

        The frame side executes that crossing
        (pauli_frame/decision_dispatch.py); the decision is available at
        the controller the instant this runs. A release is consumed
        here, and the command it releases crosses the output path in
        send_command. A result return crosses the output path before it
        is available at the QPU, and what receives it there is the
        execution runtime, the classical program that branches on the
        outcome, because the QPU device models the cycle cadence and the
        pulses and holds no register an outcome lands in (QubiC runs the
        branch on the control processor beside the qubit, Fruitwala et
        al. 2404.15260 Sec. III and IV).
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
        """Pay the pulse cost, then cross controller_to_qpu.

        The send is made at the output tick, when the pulse processing is
        done; deliver(payload) runs when the QPU has it.
        """
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
            if self.link is None:
                deliver(payload)
                return
            self.link.send(
                transfer_records.LinkPath.CONTROLLER_TO_QPU,
                None,
                self.engine.now,
                attribution,
                delivered,
            )

        self.engine.schedule(
            self.pulse_ticks, output_ready, label="controller-output-ready"
        )

    def _fire(self, kind: str, operation_id, payload) -> None:
        event = round_records.ControllerOutputEvent(
            kind, self.engine.now, operation_id, payload
        )
        self.trace.output_event.fire(event)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the instruction output reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    output_event: trace_source.TraceSource = trace_source.new_source()
