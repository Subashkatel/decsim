"""The observation settings: what a run records beyond its results."""

import dataclasses
from collections.abc import Mapping

TRACE_MODES = ("off", "print", "file", "both")
WINDOW_CHECKS = ("none", "tesseract")


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """The yaml's `observation` section.

    trace is the engine narrator: print shows it live, file writes each
    shot's full line record next to the results, both does both.
    log_component_io adds component I/O lines (what each store and unit
    received, holds and emitted). check_windows_with tesseract re-decodes
    every window with the Tesseract referee and counts disagreements,
    never priced. record_switching_windows keeps every request and
    service record for the switching views; round_store_occupancy builds
    the L5 listener on Buffer 0 (both Python-only knobs until slice 10's
    surface commit names them in the yaml).
    """

    trace: str = "off"
    log_component_io: bool = False
    check_windows_with: str = "none"
    record_switching_windows: bool = False
    round_store_occupancy: bool = False

    @classmethod
    def from_yaml(cls, section: Mapping) -> "ObservationSettings":
        """The `observation` section, every key optional."""
        trace = section.get("trace", "off")
        if trace is False:
            trace = "off"  # yaml 1.1 reads a bare `off` as boolean False
        if trace not in TRACE_MODES:
            raise ValueError(
                f"observation.trace must be one of {TRACE_MODES}, got {trace!r}"
            )
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
        return cls(
            trace=trace,
            log_component_io=log_component_io,
            check_windows_with=check_windows_with,
        )

    @property
    def prints_trace(self) -> bool:
        """Whether the engine prints every log line as it is written."""
        return self.trace in ("print", "both")

    @property
    def writes_trace(self) -> bool:
        """Whether the front writes each shot's log next to the results."""
        return self.trace in ("file", "both")
