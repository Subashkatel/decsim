"""The workload settings: what the machine runs, and for how many rounds.

A yaml names what makes its workload (WORKLOADS, below): a maker
function with its arguments (producer), or the maker's two outputs read
from disk (files). A maker runs once per sweep point (made) and its
records.workload.Workload is lowered into the operations the machine
issues (circuit_frontend.lowered); a Python caller hands the same
fields in directly.
"""

import dataclasses
import pathlib
import pkgutil
from collections.abc import Mapping
from typing import Any, Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.frontends.workload_files as workload_files
import decsim.ports as ports
import decsim.records.workload as workload_records
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
            "rounds_per_shot is a round count of at least 1 or "
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
    """The yaml's `workload` section, and the program it lowers to.

    kind names the row that makes the workload (WORKLOADS, below); None
    is a Python-built workload, its fields handed in directly. A row's
    workload is made at each sweep point and lowered into the fields
    below. decode_operations, feedback_boundary_mode and built_models
    are Python-only; a Machine built alone builds its own window models.
    """

    kind: Optional[str] = None
    operations: tuple = ()
    decode_operations: Optional[tuple] = None
    dynamic_streams: tuple = ()
    protected_regions: tuple = ()
    rounds_policy: Optional[ports.RoundsPolicy] = None
    physical_circuits: Mapping = dataclasses.field(default_factory=dict)
    feedback_boundary_mode: str = "trailing_buffer"
    built_models: Optional[built_window_models.BuiltWindowModels] = None
    # the record the fields above were lowered from, which a run folder
    # writes to its inputs (experiments/run_folder.py record_point)
    workload_record: Optional[workload_records.Workload] = None
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
    def from_yaml(
        cls, section: Mapping, base_directory: Optional[pathlib.Path]
    ) -> "WorkloadSettings":
        """The `workload` section; relative paths are from base_directory."""
        kind = section.get("kind")
        row = tables.row(WORKLOADS, "workload.kind", kind)
        row_settings = tables.row_settings(
            row, "workload", section, WORKLOAD_KEYS, base_directory
        )
        return cls(kind=kind, row_settings=row_settings)

    def made(self) -> "WorkloadSettings":
        """The section running the workload its row makes, once per point."""
        row = tables.row(WORKLOADS, "workload.kind", self.kind)
        workload = row.workload(self.row_settings)
        return self.running(workload)

    def running(
        self, workload: workload_records.Workload
    ) -> "WorkloadSettings":
        """This section running a maker's workload, lowered for the machine."""
        program = circuit_frontend.lowered(workload)
        return dataclasses.replace(
            self,
            operations=program.operations,
            dynamic_streams=program.dynamic_streams,
            protected_regions=program.protected_regions,
            rounds_policy=program.rounds_policy,
            physical_circuits=program.physical_circuits,
            workload_record=workload,
        )


class ProducerWorkload:
    """The producer row: a maker function and the arguments it is called with.

    function is module:function, resolved by pkgutil.resolve_name as
    Python's entry points are; Hydra's instantiate calls a named target
    with its keyword arguments the same way
    (hydra/_internal/instantiate/_instantiate2.py:76-82). The maker is
    called once per sweep point with its arguments as that point
    resolves them: an axis may set one, and a reference such as
    `distance: ${qpu.distance}` shares a value the machine reads too.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The maker's name and its own arguments."""

        function: str
        arguments: Mapping

        @classmethod
        def from_yaml(
            cls, section: Mapping, base_directory: Optional[pathlib.Path]
        ) -> "ProducerWorkload.Settings":
            """The function and its arguments, as the yaml writes them."""
            del base_directory
            arguments = section.get("arguments", {})
            return cls(function=section["function"], arguments=arguments)

    @staticmethod
    def workload(
        settings: "ProducerWorkload.Settings",
    ) -> workload_records.Workload:
        """The maker's workload at one sweep point."""
        maker = _maker(settings.function)
        made = maker(**settings.arguments)
        if isinstance(made, workload_records.Workload):
            return made
        made_type = type(made)
        raise ValueError(
            f"workload.function {settings.function} returned a "
            f"{made_type.__name__}; a maker returns a "
            "decsim.records.workload.Workload"
        )


class FilesWorkload:
    """The files row: a maker's two outputs read from disk.

    operations is a decsim.ops/1 json; the physical circuit is circuit,
    a finite .stim, with measurement_rounds, or fragments, a folder of
    the four live fragments, or neither for a run on the code card's
    timing alone.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The workload's files, their paths resolved."""

        operations: pathlib.Path
        circuit: Optional[pathlib.Path] = None
        measurement_rounds: Optional[pathlib.Path] = None
        fragments: Optional[pathlib.Path] = None

        @classmethod
        def from_yaml(
            cls, section: Mapping, base_directory: pathlib.Path
        ) -> "FilesWorkload.Settings":
            """The paths, checked to name one physical circuit at most."""
            paths = {}
            for key, value in section.items():
                paths[key] = pathlib.Path(base_directory, value)
            settings = cls(**paths)
            _check_one_physical_circuit(settings)
            return settings

    @staticmethod
    def workload(
        settings: "FilesWorkload.Settings",
    ) -> workload_records.Workload:
        """The workload the files hold."""
        return workload_files.read_workload(
            settings.operations,
            settings.circuit,
            settings.measurement_rounds,
            settings.fragments,
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


def _maker(function: str) -> Any:
    """The callable module:function names; one not there is refused."""
    try:
        return pkgutil.resolve_name(function)
    except (ImportError, AttributeError) as missing:
        raise ValueError(
            f"workload.function {function} names no maker: {missing}"
        ) from missing


def _check_one_physical_circuit(settings: "FilesWorkload.Settings") -> None:
    """A finite circuit or the fragments; with both, one would go unread."""
    if settings.circuit is not None and settings.fragments is not None:
        raise ValueError(
            "workload.circuit and workload.fragments are both set; a "
            "workload carries one physical circuit"
        )


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


# workload.kind names one of these rows; each fills the WorkloadRow port
# (decsim/ports.py).
WORKLOADS = {
    "producer": ProducerWorkload,
    "files": FilesWorkload,
}
