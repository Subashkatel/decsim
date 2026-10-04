"""The control part: the program's execution and the path back to the QPU.

The execution runtime admits each operation when its predecessors and
its resources allow, the issuer turns it into a command, and the
instruction output drives the QPU. A decoded result reaches the Pauli
frame, the conditional release lets what waited on it go, and the
decision travels back through the controller as an instruction.
"""

import dataclasses
from typing import Optional, Union

import decsim.build.plan as plan_build
import decsim.config as config
import decsim.controller.conditional_release as conditional_release_module
import decsim.controller.feedback_streams as feedback_streams
import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.instruction_output as instruction_output_module
import decsim.controller.operation_issue as operation_issue
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.pauli_frame.decision_dispatch as decision_dispatch_module
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.ports as ports


@dataclasses.dataclass(frozen=True)
class Control:
    """The components that run the program's feedback loop with the QPU.

    pauli_frame is None when the run commits into no frame.
    """

    execution_runtime: execution_runtime_module.ExecutionRuntime
    issuer: operation_issue.OperationIssuer
    streams: Union[
        feedback_streams.NoFeedbackStreams, feedback_streams.FeedbackStreams
    ]
    idle_rounds: idle_rounds_module.IdleRoundAccounting
    instruction_output: instruction_output_module.InstructionOutput
    conditional_release: conditional_release_module.ConditionalRelease
    pauli_frame: Optional[ports.Frame]
    decision_dispatch: decision_dispatch_module.DecisionDispatch

    @classmethod
    def build(
        cls,
        controller: controller_settings.ControllerSettings,
        frame: Optional[pauli_frame_module.PauliFrameConfig],
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        plan: plan_build.Plan,
        links: ports.Link,
    ) -> "Control":
        """Every component of the control side, wired to one another."""
        clocked_controller = config.with_machine_clock(
            controller, machine_clock
        )
        run_plan = plan.run_plan
        execution_runtime = execution_runtime_module.ExecutionRuntime(
            engine, plan.resource_claims
        )
        issuer = operation_issue.OperationIssuer(
            engine, run_plan.resolved_operations
        )
        streams = _feedback_streams(engine, plan)
        patch_by_identity = _resolved_patches_by_identity(plan)
        idle_rounds = idle_rounds_module.IdleRoundAccounting(
            plan.idle_policy, patch_by_identity
        )
        instruction_output = instruction_output_module.InstructionOutput(
            engine,
            clocked_controller.clock,
            clocked_controller.decision_to_pulse_cycles,
        )
        conditional_release = conditional_release_module.ConditionalRelease()
        pauli_frame = None
        if frame is not None:
            clocked_frame = config.with_machine_clock(frame, machine_clock)
            pauli_frame = clocked_frame.build(engine)
        decision_dispatch = decision_dispatch_module.DecisionDispatch(engine)
        control = cls(
            execution_runtime=execution_runtime,
            issuer=issuer,
            streams=streams,
            idle_rounds=idle_rounds,
            instruction_output=instruction_output,
            conditional_release=conditional_release,
            pauli_frame=pauli_frame,
            decision_dispatch=decision_dispatch,
        )
        control._wire_inside(links)
        return control

    def connect(
        self,
        qpu: ports.Qpu,
        factory: ports.MagicStateFactory,
        windows: ports.WindowInput,
        decode_queue: ports.DecodeQueue,
    ) -> None:
        """Drive the QPU and hand the window side each stream and idle."""
        self.execution_runtime.factory = factory
        self.instruction_output.qpu = qpu
        self.streams.qpu = qpu
        self.streams.windows = windows
        self.idle_rounds.qpu = qpu
        self.idle_rounds.windows = windows
        self.idle_rounds.decode_queue = decode_queue

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes."""
        return (
            ("conditional_release", self.conditional_release),
            ("execution_runtime", self.execution_runtime),
            ("pauli_frame", self.pauli_frame),
        )

    def _wire_inside(self, links: ports.Link) -> None:
        self.execution_runtime.issuer = self.issuer
        self.issuer.streams = self.streams
        self.issuer.idle_rounds = self.idle_rounds
        self.issuer.output = self.instruction_output
        self.streams.runtime = self.execution_runtime
        self.idle_rounds.streams = self.streams
        self.instruction_output.link = links
        self.conditional_release.dispatch = self.decision_dispatch
        self.conditional_release.runtime = self.execution_runtime
        self.decision_dispatch.instruction_output = self.instruction_output
        self.decision_dispatch.link = links


def _feedback_streams(
    engine: engine_module.Engine, plan: plan_build.Plan
) -> Union[
    feedback_streams.NoFeedbackStreams, feedback_streams.FeedbackStreams
]:
    """The stream bookkeeping, only when the workload has feedback."""
    if not _has_feedback(plan):
        return feedback_streams.NoFeedbackStreams()
    run_plan = plan.run_plan
    return feedback_streams.FeedbackStreams(
        engine,
        regions=plan.protected_regions,
        resolved_operations=run_plan.resolved_operations,
        resolved_patches=run_plan.resolved_patches,
    )


def _has_feedback(plan: plan_build.Plan) -> bool:
    """Whether any operation shares a stream or declares a region."""
    if plan.protected_regions:
        return True
    for operation in plan.all_operations:
        if operation.stream_id is not None:
            return True
    return False


def _resolved_patches_by_identity(plan: plan_build.Plan) -> dict:
    """The resolved patches by identity, for the idle accounting."""
    patch_by_identity = {}
    for patch in plan.run_plan.resolved_patches:
        patch_by_identity[patch.patch_identity] = patch
    return patch_by_identity
