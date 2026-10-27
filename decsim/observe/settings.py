"""The observation settings: what a run records beyond its results."""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.escalation.threshold_sources as threshold_sources

LOG_MODES = ("off", "print", "file", "both")
# the trace's own word for "name the file yourself", so a study asks for
# a Chrome trace without choosing a path
CHROME_TRACE = "chrome"
# the narrator's words: they name the log, never a trace file
NARRATOR_MODES = ("print", "file", "both")
# the chains an answer check keeps (observe/answer_checks.py): every weak
# answer, every strong answer, or the one of each window that passes
ANSWER_CHECK_PATHS = ("weak", "strong", "best_of_two")


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """What a run records beside its results.

    log is the engine narrator: print, file (each shot's lines next to the
    results) or both. trace is the Chrome trace: off, chrome, or a path;
    trace_shots are the seeds whose shots a run traces. log_component_io
    adds each component's I/O lines.

    first_recorded_round and last_recorded_round are the measurement
    interval, both ends included, rounds counted from 1 within each
    operation's stream; None as the last runs it to the shot's end, so
    the defaults record everything. The latency samples keep only the
    windows whose first committed round (commit_lo) lies in it and the
    rounds in it. Commit regions tile a stream, so the interval picks
    each round's window once, where a window's first read round would
    pick a leading buffer's rounds twice. The machine is not told: every
    window runs and finishes, and only what is recorded is filtered, as
    OMNeT++'s warmup-period filters recorded values and ns-3's
    FlowMonitor counts a packet by its first send and lets it finish
    after the stop. A long shot starts with empty queues and ends by
    draining them, and the interval leaves both ends out. The user picks
    it, for instance from a pilot shot's samples in window order.

    The interval cuts the latency samples and what is made from them:
    the means and maxes, the windows' tiers, the reaction growth rates
    and deadline misses, load and parallel_processes_needed, but for the
    end-of-stream reaction, which is the drain itself. The logical
    failure and the predictions stay whole, since a logical error rate
    is a rate over whole shots, and so do the shot's counts and rates
    (decoded_windows, escalated_windows, throughput, the queue peaks,
    busy fractions, switching and backlog columns, the link totals): a
    report divides one count by another, escalated over decoded
    windows, and a cut on one side would bias the ratio.

    reaction_deadline_microseconds is a budget on each recorded window's
    reaction time (buffer0_ready_to_frame); None, the default, sets
    none. A shot then counts the windows over it and the shot-clock
    time the first of them arrived, which says when an overloaded tier,
    whose wait grows without bound, first breaks the budget over the
    run length chosen.

    warmup_rounds is the rounds a region of interest leaves out: at the
    first round past them the run's statistics restart, gem5's
    m5.stats.reset() (observe/region_of_interest.py). It cuts the
    statistic rows and what is read from them, the queue peaks, busy
    fractions, offered loads and backlog peak; the latency samples keep
    the interval above. 0, the default, measures the whole run.

    folds_windows is a stream's: the run keeps no run-long ledger of
    windows, frame writes, stages, round events, transfers, decode
    requests, confidence records or log lines,
    and a stream's measures (experiments/stream_measures.py) read each
    window once it is final and let it go, as ns-3's FlowMonitor keeps a
    running record per flow rather than its packets. The machine lets a
    stream's final windows go a few behind (windows/window_release.py),
    so a stream's memory stays flat however long it runs. measure_shot
    then has nothing to read. False, the default, keeps everything.

    record_window_outcomes keeps each window's committed answer, and
    its true label when the source keeps its errors
    (ErrorModelStimDevice), for window_outcomes.csv
    (observe/window_outcomes.py).

    checkpoint_every_rounds closes a checkpoint of each stream every
    that many rounds and at its last round: the logical outcome had the
    stream been read out there, a source copy read out without noise and
    decoded to the end on the machine's own path at no simulated time
    (Tan et al. 2209.09219 lines 1003-1037; observe/checkpoints.py). Its
    rows give the logical error rate of runs of every length from one
    run (experiments/failure_statistics.py). None, the default, closes
    none.

    answer_check_paths checks each window's weak and strong answers
    against the truth, each at the checkpoint after the window, and
    keeps one chain of answers a path named: weak, strong or best_of_two
    (observe/answer_checks.py). One switching run then gives each
    decoder alone and the per-window best-of-two line on the same shots.
    Empty, the default, checks nothing.

    answer_check_thresholds keeps one more chain a threshold named, in
    decibels: a static chain, which keeps the weak answer unless the
    machine's own verdict row escalates it at that fixed threshold, and
    then keeps the strong one. Only its outcome and its escalated and
    decided windows are kept, so one switching run gives the failure
    rate and escalated share of every static threshold on the same
    shots. Empty, the default, keeps none.

    The log and the trace are labels (compare=False) and no part of a
    task's id, as sinter keeps output options out of a task's strong id
    (sinter/_data/_task.py:167-204): the writers schedule nothing. The
    others stay in the id because they add or shape a shot's columns:
    record_switching_windows and backlog_trace the wait and backlog
    columns, data_movement the shot_data_movement rows,
    record_window_outcomes the window_outcomes rows, the interval the
    latency columns, the deadline the deadline columns.
    """

    log: str = dataclasses.field(compare=False, default="off")
    log_component_io: bool = dataclasses.field(compare=False, default=False)
    record_switching_windows: bool = False
    backlog_trace: bool = False
    trace: str = dataclasses.field(compare=False, default="off")
    trace_shots: tuple = dataclasses.field(compare=False, default=(0,))
    data_movement: bool = False
    record_window_outcomes: bool = False
    first_recorded_round: int = 1
    last_recorded_round: Optional[int] = None
    reaction_deadline_microseconds: Optional[float] = None
    warmup_rounds: int = config.added_field(0)
    folds_windows: bool = config.added_field(False)
    checkpoint_every_rounds: Optional[int] = config.added_field(None)
    answer_check_paths: tuple = config.added_field(())
    answer_check_thresholds: tuple = config.added_field(())

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
        config.check_boolean(
            "observation.record_window_outcomes", self.record_window_outcomes
        )
        config.check_whole_count(
            "observation.first_recorded_round",
            self.first_recorded_round,
            "rounds",
        )
        if self.last_recorded_round is not None:
            config.check_whole_count(
                "observation.last_recorded_round",
                self.last_recorded_round,
                "rounds",
                minimum=self.first_recorded_round,
            )
        config.check_whole_count(
            "observation.warmup_rounds", self.warmup_rounds, "rounds", 0
        )
        config.check_boolean("observation.folds_windows", self.folds_windows)
        _check_folding_keeps_no_outcomes(
            self.folds_windows, self.record_window_outcomes
        )
        if self.reaction_deadline_microseconds is not None:
            config.check_microseconds(
                "observation.reaction_deadline_microseconds",
                self.reaction_deadline_microseconds,
            )
        if self.checkpoint_every_rounds is not None:
            config.check_whole_count(
                "observation.checkpoint_every_rounds",
                self.checkpoint_every_rounds,
                "rounds",
            )
        _check_answer_check_paths(self.answer_check_paths)
        _check_answer_check_thresholds(self.answer_check_thresholds)

    def records_round(self, round_index: int) -> bool:
        """Whether the latency statistics take a sample of this round."""
        if round_index < self.first_recorded_round:
            return False
        if self.last_recorded_round is None:
            return True
        return round_index <= self.last_recorded_round

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


def _check_folding_keeps_no_outcomes(
    folds_windows: bool, record_window_outcomes: bool
) -> None:
    """A stream that lets its windows go keeps no window's outcome."""
    if folds_windows and record_window_outcomes:
        raise ValueError(
            "not implemented yet: observation.record_window_outcomes keeps "
            "every window's answer, and observation.folds_windows lets each "
            "window go once it is final"
        )


def _check_answer_check_paths(paths) -> None:
    """A tuple of distinct paths, each one of ANSWER_CHECK_PATHS."""
    is_tuple = isinstance(paths, tuple)
    known = is_tuple and all(path in ANSWER_CHECK_PATHS for path in paths)
    distinct = known and len(set(paths)) == len(paths)
    if distinct:
        return
    raise ValueError(
        "observation.answer_check_paths must be a tuple of distinct paths "
        f"from {ANSWER_CHECK_PATHS}, got {paths!r}"
    )


def _check_answer_check_thresholds(thresholds) -> None:
    """A tuple of distinct thresholds, each a keep threshold in decibels."""
    name = "observation.answer_check_thresholds"
    distinct = isinstance(thresholds, tuple)
    distinct = distinct and len(set(thresholds)) == len(thresholds)
    if not distinct:
        raise ValueError(
            f"{name} must be a tuple of distinct thresholds in decibels, "
            f"got {thresholds!r}"
        )
    for threshold in thresholds:
        threshold_sources.checked_decibels(threshold, name)


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
    """The shots of a task the trace is written for."""
    if not isinstance(shots, tuple):
        raise ValueError(
            "observation.trace_shots must be a list of shot numbers, a "
            f"tuple in Python, got {shots!r}"
        )
    for shot in shots:
        _refuse_a_shot_that_is_not_a_count(shot)


def _refuse_a_shot_that_is_not_a_count(shot) -> None:
    """A shot is named by its number in the task, counting from 0."""
    is_a_count = isinstance(shot, int)
    if isinstance(shot, bool):
        is_a_count = False
    if is_a_count and shot >= 0:
        return
    raise ValueError(
        "observation.trace_shots must be a list of non-negative whole "
        f"numbers, got {shot!r}"
    )
