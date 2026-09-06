"""The observation settings: what a run records beyond its results."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

LOG_MODES = ("off", "print", "file", "both")
WINDOW_CHECKS = ("none", "tesseract")
# the trace's own word for "name the file yourself", so a study asks for
# a Chrome trace without choosing a path
CHROME_TRACE = "chrome"
# the narrator's words, which observation.trace carried before the trace
# took the key
NARRATOR_MODES = ("print", "file", "both")
OBSERVATION_KEYS = (
    "log",
    "log_component_io",
    "check_windows_with",
    "record_switching_windows",
    "round_store_occupancy",
    "backlog_trace",
    "decoder_utilization",
    "decoder_memory_occupancy",
    "trace",
    "trace_shots",
    "data_movement",
)


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
    samplers; data_movement builds the copy, reference and move counters
    the RunResult carries.
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
        _refuse_an_unknown_key(section)
        log = _log_mode(section)
        trace = _trace_word_or_path(section)
        trace_shots = _trace_shots(section)
        check_windows_with = _window_check(section)
        log_component_io = _boolean(section, "log_component_io")
        record_switching_windows = _boolean(section, "record_switching_windows")
        round_store_occupancy = _boolean(section, "round_store_occupancy")
        backlog_trace = _boolean(section, "backlog_trace")
        decoder_utilization = _boolean(section, "decoder_utilization")
        decoder_memory_occupancy = _boolean(section, "decoder_memory_occupancy")
        data_movement = _boolean(section, "data_movement")
        return cls(
            log=log,
            trace=trace,
            trace_shots=trace_shots,
            log_component_io=log_component_io,
            check_windows_with=check_windows_with,
            record_switching_windows=record_switching_windows,
            round_store_occupancy=round_store_occupancy,
            backlog_trace=backlog_trace,
            decoder_utilization=decoder_utilization,
            decoder_memory_occupancy=decoder_memory_occupancy,
            data_movement=data_movement,
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


def _refuse_an_unknown_key(section: Mapping) -> None:
    """A key the section does not have is a stale or misspelled knob."""
    for key in section:
        if key not in OBSERVATION_KEYS:
            raise ValueError(
                f"observation.{key} is not an observation key; the "
                f"observation section takes {OBSERVATION_KEYS}"
            )


def _log_mode(section: Mapping) -> str:
    """The engine narrator's word."""
    log = section.get("log", "off")
    if log is False:
        log = "off"  # yaml 1.1 reads a bare `off` as boolean False
    if log not in LOG_MODES:
        raise ValueError(
            f"observation.log must be one of {LOG_MODES}, got {log!r}"
        )
    return log


def _window_check(section: Mapping) -> str:
    """The referee that re-decodes every window, or none."""
    check_windows_with = section.get("check_windows_with", "none")
    if check_windows_with not in WINDOW_CHECKS:
        raise ValueError(
            "observation.check_windows_with must be one of "
            f"{WINDOW_CHECKS}, got {check_windows_with!r}"
        )
    return check_windows_with


def _trace_word_or_path(section: Mapping) -> str:
    """off, chrome, or the path the Chrome trace is written to."""
    trace = section.get("trace", "off")
    if trace is False:
        trace = "off"  # yaml 1.1 reads a bare `off` as boolean False
    if not isinstance(trace, str):
        raise ValueError(
            "observation.trace must be off, chrome, or a path, got "
            f"{trace!r}; yaml reads a bare `on` as true, so quote a path "
            "that looks like a word"
        )
    if trace in NARRATOR_MODES:
        raise ValueError(
            f"observation.trace no longer names the engine narrator; "
            f"write `log: {trace}` for that, and leave `trace: off` or "
            "name a Chrome trace with chrome or a path"
        )
    return trace


def _trace_shots(section: Mapping) -> tuple:
    """The shots of a sweep point the trace is written for."""
    shots = section.get("trace_shots", (0,))
    if not isinstance(shots, (list, tuple)):
        raise ValueError(
            "observation.trace_shots must be a list of shot numbers, got "
            f"{shots!r}"
        )
    for shot in shots:
        _refuse_a_shot_that_is_not_a_count(shot)
    return tuple(shots)


def _refuse_a_shot_that_is_not_a_count(shot) -> None:
    """A shot is named by its number in the sweep point, counting from 0."""
    is_a_count = isinstance(shot, int)
    if isinstance(shot, bool):
        is_a_count = False
    if is_a_count and shot >= 0:
        return
    raise ValueError(
        "observation.trace_shots must be a list of non-negative whole "
        f"numbers, got {shot!r}"
    )


def _boolean(section: Mapping, key: str) -> bool:
    """One of the section's on-or-off knobs, off when the yaml is silent."""
    value = section.get(key, False)
    if value not in (True, False):
        raise ValueError(
            f"observation.{key} must be true or false, got {value!r}"
        )
    return value
