"""The workload settings: what the machine runs, and for how many rounds."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.ports as ports
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import decsim.tables as tables
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

    Table rows (WORKLOADS, below): memory_circuit (Stim's generated
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
    rounds_policy: Optional[ports.RoundsPolicy] = None
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
        """The `workload` section: the kind's row reads its own keys."""
        kind = section.get("kind", "memory_circuit")
        row = tables.row(WORKLOADS, "workload.kind", kind)
        fields = row.from_yaml(section)
        return cls(kind=kind, **fields)


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


class MemoryCircuitWorkload:
    """The memory_circuit row: Stim's generated memory circuit.

    One operation for the whole shot, its rounds fixed by
    rounds_per_shot, so the run has no operation chain in front of it.
    """

    has_frontend = False

    @staticmethod
    def from_yaml(section: Mapping) -> dict:
        """The `workload` keys this row reads, as settings fields."""
        rounds_per_shot = RoundsPerShot.from_yaml(section["rounds_per_shot"])
        return {
            "code_task": section["code_task"],
            "rounds_per_shot": rounds_per_shot,
        }

    @staticmethod
    def operations(settings: "WorkloadSettings", code) -> tuple:
        """One operation, its rounds fixed."""
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


class CircuitListWorkload:
    """The circuit_list row: the operations as the caller built them."""

    has_frontend = False

    @staticmethod
    def from_yaml(section: Mapping) -> dict:
        """Refused: this row's operations are Operation records."""
        del section
        raise ValueError(
            "workload.kind circuit_list takes a list of Operation records "
            "with their Stim circuits, which a yaml scalar cannot carry; "
            "build it in Python (WorkloadSettings(operations=...))"
        )

    @staticmethod
    def operations(settings: "WorkloadSettings", code) -> tuple:
        """The operations as given."""
        del code
        return tuple(settings.operations), None


class SurgeryIRWorkload:
    """The surgery_ir row: the line-based text IR parsed and wired."""

    has_frontend = True

    @staticmethod
    def from_yaml(section: Mapping) -> dict:
        """Refused: this row needs the caller's qubit-to-patch mapping."""
        del section
        raise ValueError(
            "workload.kind surgery_ir takes the IR text and the "
            "qubit_to_patch mapping the caller allocated its patches "
            "with, which no yaml key carries today; build it in Python "
            "(WorkloadSettings(text=..., qubit_to_patch=...))"
        )

    @staticmethod
    def operations(settings: "WorkloadSettings", code) -> tuple:
        """The text IR parsed and wired."""
        del code
        frontend = circuit_frontend.SurgeryIRFrontend(
            settings.text, settings.qubit_to_patch
        )
        operations = frontend.build()
        return tuple(operations), None


class QlxWorkload:
    """The qlx row: a lowered QLX program's operations."""

    has_frontend = True

    @staticmethod
    def from_yaml(section: Mapping) -> dict:
        """Refused: this row takes a lowered program object."""
        del section
        raise ValueError(
            "workload.kind qlx takes a lowered QLX program object, which "
            "a yaml cannot carry; lower it and build it in Python "
            "(WorkloadSettings(program=...))"
        )

    @staticmethod
    def operations(settings: "WorkloadSettings", code) -> tuple:
        """The lowered program's operations."""
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


# workload.kind names one of these rows: a row reads its own section
# keys at the yaml boundary, declares whether an operation chain is
# built in front of the run (has_frontend), and turns its settings and
# the run's code into the operations and, when the row fixes them, the
# rounds policy.
WORKLOADS = {
    "memory_circuit": MemoryCircuitWorkload,
    "circuit_list": CircuitListWorkload,
    "surgery_ir": SurgeryIRWorkload,
    "qlx": QlxWorkload,
}
