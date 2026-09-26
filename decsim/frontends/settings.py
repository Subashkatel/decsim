"""The workload settings: what the machine runs, and for how many rounds."""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.ports as ports
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import decsim.tables as tables
import decsim.windows.built_window_models as built_window_models

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")
# The keys every row of the workload section shares; any other key is the
# row's own (its Settings, decsim/tables.py row_settings).
WORKLOAD_KEYS = ("kind",)


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
        """`rounds_per_shot`: a count, or "<n>d", each at least one round.

        Stim's generator refuses fewer: "Need rounds >= 1."
        """
        is_count = isinstance(value, int) and not isinstance(value, bool)
        if is_count and value >= 1:
            return cls(fixed=value)
        if _is_per_distance_text(value):
            per_distance = int(value[:-1])
            return cls(per_distance=per_distance)
        raise ValueError(
            "workload.rounds_per_shot is a round count of at least 1 or "
            f"'<n>d' (n rounds per unit of distance, n at least 1), got "
            f"{value!r}"
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
    all four of Stim's noise channels, rounds_per_shot rounds, both keys
    its own Settings), circuit_list (a Python-built operation list),
    surgery_ir (the line-based text IR).
    The other fields are Python-only: the decode owners, the dynamic
    streams and protected regions of a feedback workload, the round policy
    (GateRounds by default; the memory circuit fixes its rounds and
    refuses one), the feedback boundary mode every operation takes
    unless it names its own, and the window error models a task built
    once for all of its shots (built_window_models; a Machine built
    alone gets none and builds its own). row_settings is the row's own
    Settings, read from the section's keys beside kind, or None for a
    row that declares none.
    """

    kind: str = "circuit_list"
    physical_error_probability: Optional[float] = None
    operations: tuple = ()
    text: str = ""
    qubit_to_patch: Optional[dict] = None
    decode_operations: Optional[tuple] = None
    dynamic_streams: tuple = ()
    protected_regions: tuple = ()
    rounds_policy: Optional[ports.RoundsPolicy] = None
    feedback_boundary_mode: str = "trailing_buffer"
    built_models: Optional[built_window_models.BuiltWindowModels] = None
    # the row's own Settings record, opaque to the section
    row_settings: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
            raise ValueError(
                "workload.feedback_boundary_mode must be one of "
                f"{FEEDBACK_BOUNDARY_MODES}, got "
                f"{self.feedback_boundary_mode!r}"
            )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "WorkloadSettings":
        """The `workload` section: the kind's row reads its own keys.

        The yaml names its kind: the dataclass's default, circuit_list, is
        for Python-built runs, and a section without one is refused.
        """
        kind = section.get("kind")
        row = tables.row(WORKLOADS, "workload.kind", kind)
        row_settings = tables.row_settings(
            row, "workload", section, WORKLOAD_KEYS
        )
        return cls(kind=kind, row_settings=row_settings)


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

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row's own keys: Stim's code task and the rounds a shot runs."""

        code_task: str
        rounds_per_shot: RoundsPerShot

        @classmethod
        def from_yaml(
            cls, section: Mapping
        ) -> "MemoryCircuitWorkload.Settings":
            """Both keys; a missing one is refused rather than defaulted."""
            missing = []
            for field in dataclasses.fields(cls):
                if field.name not in section:
                    missing.append(field.name)
            if missing:
                raise ValueError(
                    f"workload.kind memory_circuit needs {missing} beside kind"
                )
            rounds_per_shot = RoundsPerShot.from_yaml(
                section["rounds_per_shot"]
            )
            return cls(
                code_task=section["code_task"], rounds_per_shot=rounds_per_shot
            )

    @staticmethod
    def operations(settings: "WorkloadSettings", code) -> tuple:
        """One operation, its rounds fixed."""
        if settings.physical_error_probability is None:
            raise ValueError(
                "a memory_circuit workload needs physical_error_probability; "
                "the sweep sets it per point"
            )
        row_settings = settings.row_settings
        rounds_per_shot = row_settings.rounds_per_shot
        rounds = rounds_per_shot.rounds_for(code.distance)
        circuit = memory_circuit(
            row_settings.code_task,
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

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """No key: the yaml cannot name this row at all."""

        @classmethod
        def from_yaml(cls, section: Mapping):
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

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """No key: the yaml cannot name this row at all."""

        @classmethod
        def from_yaml(cls, section: Mapping):
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


def _is_per_distance_text(value) -> bool:
    """True for "<n>d": digits naming at least one round, then a d."""
    if not isinstance(value, str):
        return False
    if not value.endswith("d"):
        return False
    digits = value[:-1]
    if not digits.isdigit():
        return False
    return int(digits) >= 1


# workload.kind names one of these rows: a row reads its own section
# keys at the yaml boundary through its Settings, declares whether an
# operation chain is built in front of the run (has_frontend), and turns
# its settings and the run's code into the operations and, when the row
# fixes them, the rounds policy.
WORKLOADS = {
    "memory_circuit": MemoryCircuitWorkload,
    "circuit_list": CircuitListWorkload,
    "surgery_ir": SurgeryIRWorkload,
}
