"""The workload settings: what the machine runs, and for how many rounds."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.ports as ports
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import decsim.windows.built_window_models as built_window_models

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")


@dataclasses.dataclass(frozen=True)
class RoundsPerShot:
    """How many rounds one memory shot runs: a count, or per distance.

    "10d" is ten rounds per unit of code distance, the 10 d-round memory
    experiment of Toshio 2510.25222 Fig. 4.
    """

    fixed: Optional[int] = None
    per_distance: Optional[int] = None

    @classmethod
    def from_yaml(cls, value) -> "RoundsPerShot":
        """`rounds_per_shot`: an int, or "<n>d"."""
        if isinstance(value, int):
            return cls(fixed=value)
        if _is_per_distance_text(value):
            per_distance = int(value[:-1])
            return cls(per_distance=per_distance)
        raise ValueError(
            "workload.rounds_per_shot is a round count or '<n>d' (rounds "
            f"per unit of distance), got {value!r}"
        )

    def rounds_for(self, distance: int) -> int:
        """The rounds one shot runs at this distance."""
        if self.fixed is not None:
            return self.fixed
        return self.per_distance * distance

    def __str__(self) -> str:
        if self.fixed is not None:
            return str(self.fixed)
        return f"{self.per_distance}d"


@dataclasses.dataclass(frozen=True)
class WorkloadSettings:
    """The yaml's `workload` section.

    Table rows (decsim/machine.py): memory_circuit (Stim's generated
    memory circuit for the code task, one physical error probability on
    all four of Stim's noise channels, rounds_per_shot rounds),
    circuit_list (a Python-built operation list), surgery_ir (the
    line-based text IR), qlx (a lowered QLX program). The other fields
    are Python-only: the decode owners, the dynamic streams and
    protected regions of a feedback workload, the round policy
    (GateRounds by default; the memory circuit fixes its rounds), the
    feedback boundary mode every operation takes unless it names its
    own, and the window error models a task built once for all of its
    shots (built_window_models; a Machine built alone gets none and
    builds its own).
    """

    kind: str = "circuit_list"
    code_task: str = "surface_code:rotated_memory_z"
    rounds_per_shot: RoundsPerShot = RoundsPerShot(fixed=15)
    physical_error_probability: Optional[float] = None
    operations: tuple = ()
    text: str = ""
    program: Optional[ports.Workload] = None
    qubit_to_patch: Optional[dict] = None
    decode_operations: Optional[tuple] = None
    dynamic_streams: tuple = ()
    protected_regions: tuple = ()
    rounds_policy: Optional[round_policies.RoundsPolicy] = None
    feedback_boundary_mode: str = "trailing_buffer"
    built_models: Optional[built_window_models.BuiltWindowModels] = None

    def __post_init__(self) -> None:
        if self.feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
            raise ValueError(
                "workload.feedback_boundary_mode must be one of "
                f"{FEEDBACK_BOUNDARY_MODES}, got "
                f"{self.feedback_boundary_mode!r}"
            )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "WorkloadSettings":
        """The `workload` section: a memory circuit and its rounds."""
        kind = section.get("kind", "memory_circuit")
        if kind != "memory_circuit":
            raise ValueError(
                "a yaml workload is a memory_circuit; the other rows "
                "(circuit_list, surgery_ir, qlx) are built in Python"
            )
        rounds_per_shot = RoundsPerShot.from_yaml(section["rounds_per_shot"])
        return cls(
            kind=kind,
            code_task=section["code_task"],
            rounds_per_shot=rounds_per_shot,
        )


def memory_circuit(
    code_task: str, rounds: int, distance: int, probability: float
) -> stim.Circuit:
    """Stim's generated memory circuit.

    One probability on all four noise channels, as Stim's guide does.
    """
    return stim.Circuit.generated(
        code_task,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=probability,
        before_round_data_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
    )


def memory_circuit_operations(settings: WorkloadSettings, code) -> tuple:
    """The memory_circuit row: one operation, its rounds fixed."""
    if settings.physical_error_probability is None:
        raise ValueError(
            "a memory_circuit workload needs physical_error_probability; "
            "the sweep sets it per point"
        )
    rounds = settings.rounds_per_shot.rounds_for(code.distance)
    circuit = memory_circuit(
        settings.code_task,
        rounds,
        code.distance,
        settings.physical_error_probability,
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    return (operation,), round_policies.FixedRounds(rounds)


def circuit_list_operations(settings: WorkloadSettings, code) -> tuple:
    """The circuit_list row: the operations as given."""
    del code
    return tuple(settings.operations), None


def surgery_ir_operations(settings: WorkloadSettings, code) -> tuple:
    """The surgery_ir row: the text IR parsed and wired."""
    del code
    frontend = circuit_frontend.SurgeryIRFrontend(
        settings.text, settings.qubit_to_patch
    )
    operations = frontend.build()
    return tuple(operations), None


def qlx_operations(settings: WorkloadSettings, code) -> tuple:
    """The qlx row: the lowered program's operations."""
    del code
    if settings.program is None:
        raise ValueError("a qlx workload needs its lowered program")
    operations = settings.program.build()
    return tuple(operations), None


def _is_per_distance_text(value) -> bool:
    """True for "<n>d": digits then a d."""
    if not isinstance(value, str):
        return False
    if not value.endswith("d"):
        return False
    digits = value[:-1]
    return digits.isdigit()


# workload.kind names one of these rows: a row turns its settings and
# the run's code into the operations and, when the row fixes them, the
# rounds policy.
WORKLOADS = {
    "memory_circuit": memory_circuit_operations,
    "circuit_list": circuit_list_operations,
    "surgery_ir": surgery_ir_operations,
    "qlx": qlx_operations,
}
