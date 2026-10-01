"""The observation settings: what a run records beyond its results."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.decoders.verify_windows as verify_windows

LOG_MODES = ("off", "print", "file", "both")
# observation.check_windows_with names one of these rows: the referee
# that re-decodes every window, or none.
WINDOW_CHECKS = {
    "none": None,
    "tesseract": verify_windows.TesseractCheckedDecoder,
}
# the trace's own word for "name the file yourself", so a study asks for
# a Chrome trace without choosing a path
CHROME_TRACE = "chrome"
# the narrator's words: they name the log, never a trace file
NARRATOR_MODES = ("print", "file", "both")
OBSERVATION_KEYS = (
    "log",
    "log_component_io",
    "check_windows_with",
    "record_switching_windows",
    "backlog_trace",
    "decoder_memory_occupancy",
    "trace",
    "trace_shots",
    "data_movement",
    "confidence_shot_count",
)
# observation.confidence_shot_count's word for every shot of a point
EVERY_SHOT = "all"


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """The yaml's `observation` section.

    log is the engine narrator: print shows it live, file writes each shot's
    full line record next to the results, both does both. trace is the
    Chrome trace of the data path: off, chrome (the experiments layer names
    the file next to the results), or a path of its own; the experiments
    layer writes it for the shots trace_shots names. log_component_io adds
    component I/O lines (what each store and unit received, holds and
    emitted). check_windows_with tesseract re-decodes every window with the
    Tesseract referee and counts disagreements, never priced.
    record_switching_windows keeps every request record for the switching
    study; backlog_trace builds the sampler of the rounds waiting to be
    decoded; decoder_memory_occupancy builds the memory sweep's sampler (the
    decoder utilization is always integrated, every run's pool columns read
    it); data_movement builds the copy, reference and move counters the
    RunResult carries. confidence_shot_count is how many shots of each
    point, from seed 0, write their windows' confidence gaps to
    window_confidence.csv when a confidence signal decides the escalation;
    None writes every scored shot's.

    The keys that only record the run, the log, the trace, the memory
    occupancy listener and confidence_shot_count, are labels
    (compare=False) and no part of a point's id, as sinter keeps its
    output options out of a task's strong id
    (sinter/_data/_task.py:167-204): each writer and listener schedules
    nothing and calls no component (observe/trace_writer.py), so the
    shots' rows are the same with them or without. The others
    stay in the id because they change a shot's row: the referee fills
    the referee columns, record_switching_windows and backlog_trace add
    the wait and backlog columns (experiments/measure.py), and
    data_movement adds the shot_data_movement rows.
    """

    log: str = dataclasses.field(compare=False, default="off")
    log_component_io: bool = dataclasses.field(compare=False, default=False)
    check_windows_with: str = "none"
    record_switching_windows: bool = False
    backlog_trace: bool = False
    decoder_memory_occupancy: bool = dataclasses.field(
        compare=False, default=False
    )
    trace: str = dataclasses.field(compare=False, default="off")
    trace_shots: tuple = dataclasses.field(compare=False, default=(0,))
    data_movement: bool = False
    confidence_shot_count: Optional[int] = dataclasses.field(
        compare=False, default=100
    )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "ObservationSettings":
        """The `observation` section, every key optional."""
        _refuse_a_section_that_is_not_a_block(section)
        _refuse_an_unknown_key(section)
        log = _log_mode(section)
        trace = _trace_word_or_path(section)
        trace_shots = _trace_shots(section)
        check_windows_with = _window_check(section)
        log_component_io = config.boolean(
            section, "observation", "log_component_io"
        )
        record_switching_windows = config.boolean(
            section, "observation", "record_switching_windows"
        )
        backlog_trace = config.boolean(section, "observation", "backlog_trace")
        decoder_memory_occupancy = config.boolean(
            section, "observation", "decoder_memory_occupancy"
        )
        data_movement = config.boolean(section, "observation", "data_movement")
        confidence_shot_count = _confidence_shot_count(section)
        return cls(
            log=log,
            trace=trace,
            trace_shots=trace_shots,
            log_component_io=log_component_io,
            check_windows_with=check_windows_with,
            record_switching_windows=record_switching_windows,
            backlog_trace=backlog_trace,
            decoder_memory_occupancy=decoder_memory_occupancy,
            data_movement=data_movement,
            confidence_shot_count=confidence_shot_count,
        )

    @property
    def prints_log(self) -> bool:
        """Whether the engine prints every log line as it is written."""
        return self.log in ("print", "both")

    @property
    def writes_log(self) -> bool:
        """Whether the experiments layer writes each shot's log."""
        return self.log in ("file", "both")

    @property
    def writes_trace(self) -> bool:
        """Whether the run builds the Chrome trace writer at all."""
        return self.trace != "off"

    def samples_confidence_of(self, seed: int) -> bool:
        """Whether this seed's windows go into window_confidence.csv."""
        if self.confidence_shot_count is None:
            return True
        return seed < self.confidence_shot_count

    @property
    def trace_path(self) -> Optional[str]:
        """The path the trace names, None when decsim.experiments names it."""
        if self.trace in ("off", CHROME_TRACE):
            return None
        return self.trace


def _refuse_a_section_that_is_not_a_block(section) -> None:
    """The section is a block of keys; a bare value names none of them."""
    if isinstance(section, Mapping):
        return
    raise ValueError(
        f"the observation section must be a block of keys, got {section!r}; "
        f"the observation keys are {OBSERVATION_KEYS}"
    )


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
    # a list of names, not the table: a yaml list or block is unhashable
    # and a dictionary lookup would raise TypeError before the sentence
    rows = sorted(WINDOW_CHECKS)
    if check_windows_with not in rows:
        raise ValueError(
            "observation.check_windows_with must be one of "
            f"{rows}, got {check_windows_with!r}"
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
    if not trace:
        raise ValueError(
            "observation.trace is empty, which names no file; write off, "
            "chrome, or a path"
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


def _confidence_shot_count(section: Mapping) -> Optional[int]:
    """How many shots of a point write their windows' gaps; None, all."""
    shot_count = section.get("confidence_shot_count", 100)
    if shot_count == EVERY_SHOT:
        return None
    is_a_count = isinstance(shot_count, int) and not isinstance(
        shot_count, bool
    )
    if is_a_count and shot_count >= 0:
        return shot_count
    raise ValueError(
        "observation.confidence_shot_count must be a non-negative whole "
        f"number of shots or {EVERY_SHOT}, got {shot_count!r}"
    )


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
