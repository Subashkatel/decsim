"""The observation settings: what a run records beyond its results."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

LOG_MODES = ("off", "print", "file", "both")
WINDOW_CHECKS = ("none", "tesseract")
# the trace's own word for "name the file yourself", so a study asks for
# a Chrome trace without choosing a path
CHROME_TRACE = "chrome"


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """The yaml's `observation` section.

    log is the engine narrator: print shows it live, file writes each
    shot's full line record next to the results, both does both. trace
    is the Chrome trace of the data path: off, chrome (the front names
    the file next to the results), or a path of its own; the front
    writes it for the shots trace_shots names.
    log_component_io adds component I/O lines (what each store and unit
    received, holds and emitted). check_windows_with tesseract re-decodes
    every window with the Tesseract referee and counts disagreements,
    never priced. record_switching_windows keeps every request and
    service record for the switching views; round_store_occupancy builds
    the L5 listener on Buffer 0; backlog_trace builds the decode backlog
    sampler the D7 harness reads; decoder_utilization and
    decoder_memory_occupancy build the unit-count and memory sweeps'
    samplers; data_movement builds the copy, reference and move
    counters the RunResult carries (record_switching_windows,
    round_store_occupancy, backlog_trace, decoder_utilization,
    decoder_memory_occupancy and data_movement are Python-only knobs
    until a study asks for them in the yaml).
    """

    log: str = "off"
    log_component_io: bool = False
    check_windows_with: str = "none"
    record_switching_windows: bool = False
    round_store_occupancy: bool = False
    backlog_trace: bool = False
    decoder_utilization: bool = False
    decoder_memory_occupancy: bool = False
    trace: str = "off"
    trace_shots: tuple = (0,)
    data_movement: bool = False

    @classmethod
    def from_yaml(cls, section: Mapping) -> "ObservationSettings":
        """The `observation` section, every key optional."""
        log = section.get("log", "off")
        if log is False:
            log = "off"  # yaml 1.1 reads a bare `off` as boolean False
        if log not in LOG_MODES:
            raise ValueError(
                f"observation.log must be one of {LOG_MODES}, got {log!r}"
            )
        trace = section.get("trace", "off")
        if trace is False:
            trace = "off"
        log_component_io = section.get("log_component_io", False)
        if log_component_io not in (True, False):
            raise ValueError(
                "observation.log_component_io must be true or false, got "
                f"{log_component_io!r}"
            )
        check_windows_with = section.get("check_windows_with", "none")
        if check_windows_with not in WINDOW_CHECKS:
            raise ValueError(
                "observation.check_windows_with must be one of "
                f"{WINDOW_CHECKS}, got {check_windows_with!r}"
            )
        trace_shots = section.get("trace_shots", (0,))
        return cls(
            log=log,
            trace=trace,
            trace_shots=tuple(trace_shots),
            log_component_io=log_component_io,
            check_windows_with=check_windows_with,
        )

    @property
    def prints_log(self) -> bool:
        """Whether the engine prints every log line as it is written."""
        return self.log in ("print", "both")

    @property
    def writes_log(self) -> bool:
        """Whether the front writes each shot's log next to the results."""
        return self.log in ("file", "both")

    @property
    def writes_trace(self) -> bool:
        """Whether the run builds the Chrome trace writer at all."""
        return self.trace != "off"

    @property
    def trace_path(self) -> Optional[str]:
        """The path the trace names, None when the front names it."""
        if self.trace in ("off", CHROME_TRACE):
            return None
        return self.trace
