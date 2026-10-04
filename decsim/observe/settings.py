"""The observation settings: what a run records beyond its results."""

import dataclasses
from typing import Optional

import decsim.config as config

LOG_MODES = ("off", "print", "file", "both")
# the trace's own word for "name the file yourself", so a study asks for
# a Chrome trace without choosing a path
CHROME_TRACE = "chrome"
# the narrator's words: they name the log, never a trace file
NARRATOR_MODES = ("print", "file", "both")


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """What a run records beside its results.

    log is the engine narrator: print shows it live, file writes each shot's
    full line record next to the results, both does both. trace is the
    Chrome trace of the data path: off, chrome (the experiments layer names
    the file next to the results), or a path of its own; the experiments
    layer writes it for the shots trace_shots names. log_component_io adds
    component I/O lines (what each store and unit received, holds and
    emitted). record_switching_windows keeps every request record for the
    switching study; backlog_trace builds the sampler of the rounds waiting
    to be decoded (the decoder utilization is always integrated, every run's
    pool columns read it); data_movement builds the copy, reference and move
    counters the RunResult carries.

    The fields that only record the run, the log and the trace, are
    labels (compare=False) and no part of a
    point's id, as sinter keeps its output options out of a task's strong id
    (sinter/_data/_task.py:167-204): each writer schedules nothing and
    calls no component (observe/trace_writer.py), so the shots'
    rows are the same with them or without. The others stay in the id
    because they change a shot's row: record_switching_windows and
    backlog_trace add the wait and backlog columns
    (experiments/measure.py), and data_movement adds the shot_data_movement
    rows.
    """

    log: str = dataclasses.field(compare=False, default="off")
    log_component_io: bool = dataclasses.field(compare=False, default=False)
    record_switching_windows: bool = False
    backlog_trace: bool = False
    trace: str = dataclasses.field(compare=False, default="off")
    trace_shots: tuple = dataclasses.field(compare=False, default=(0,))
    data_movement: bool = False

    def __post_init__(self) -> None:
        """Every value checked, with the sentence the caller reads."""
        _check_log(self.log)
        _check_trace(self.trace)
        _check_trace_shots(self.trace_shots)
        config.check_boolean(
            "observation.log_component_io", self.log_component_io
        )
        config.check_boolean(
            "observation.record_switching_windows",
            self.record_switching_windows,
        )
        config.check_boolean("observation.backlog_trace", self.backlog_trace)
        config.check_boolean("observation.data_movement", self.data_movement)

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

    @property
    def trace_path(self) -> Optional[str]:
        """The path the trace names, None when decsim.experiments names it."""
        if self.trace in ("off", CHROME_TRACE):
            return None
        return self.trace


def _check_log(log) -> None:
    """The engine narrator's word is one of LOG_MODES."""
    # a tuple of words, not a set, so an unhashable value meets the
    # sentence below rather than a TypeError
    if log in LOG_MODES:
        return
    raise ValueError(f"observation.log must be one of {LOG_MODES}, got {log!r}")


def _check_trace(trace) -> None:
    """The trace is off, chrome, or the path the Chrome trace is written to."""
    if not isinstance(trace, str):
        raise ValueError(
            f"observation.trace must be off, chrome, or a path, got {trace!r}"
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


def _check_trace_shots(shots) -> None:
    """The shots of a sweep point the trace is written for."""
    if not isinstance(shots, tuple):
        raise ValueError(
            "observation.trace_shots must be a list of shot numbers, a "
            f"tuple in Python, got {shots!r}"
        )
    for shot in shots:
        _refuse_a_shot_that_is_not_a_count(shot)


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
