"""The switching-per-window run's tables, from its folder.

`python tables.py <run folder> <results folder>` reads the folded
shots.csv and window_outcomes.csv and each task's record, and writes
tidy tables of counts, so every figure and the final comparison table
can be drawn from them:

- shot_summary.csv: per configuration and setting, the shots, the
  failed shots with the rate's 95% Wilson interval, the windows
  decoded, and the shots with at least one wrong window.
- window_summary.csv: per configuration and setting, the windows by
  their final answer's class (ANSWER_CLASSES).
- switching_summary.csv: per setting, switching's union-find windows
  by decision (escalated or kept) and by the class of union-find's
  answer on the window's own rounds; and the escalated windows by
  Relay-BP-5's outcome against union-find alone on the same rounds,
  once with seam pairs counted right (fixed, broke, ...) and once
  with every wrong label counted wrong (label_fixed, label_broke, ...).
- union_find_by_gap.csv: switching's union-find windows by their gap
  in 1 dB bins and by the class of union-find's answer.
- kept_by_detection_events.csv: switching's kept windows by their
  detection-event count and by the class of union-find's answer.

A window is right when its answer equals its label, the parity of the
true errors it owns (2509.03815 Eq. 1). Two wrong windows that share a
seam flip the shot twice and cancel: one error near the seam, owned by
one window and corrected by the other, gives such a pair. Pairs are
taken left to right in time and do not overlap; a wrong window in no
pair is wrong alone.
"""

import collections
import csv
import json
import math
import pathlib
import sys

import decsim.escalation.threshold_sources as threshold_sources

UNION_FIND_ALONE = "union_find_alone"
SWITCHING = "switching"
# a rate is shown only with this many failures (LOG.md, stop rule)
SHOWN_FAILURES = 20
# the normal quantile of a two-sided 95% interval
NORMAL_QUANTILE_95 = 1.959963984540054
GAP_BIN_DECIBELS = 1.0
# An escalated window commits its own rounds and the next two windows',
# which union-find never answers. A wrong union-find window beside
# those rounds in no pair is wrong_partner_unseen, as its seam partner
# may lie there: an escalated window, or the window after one.
ANSWER_CLASSES = (
    "right",
    "wrong_alone",
    "wrong_paired",
    "wrong_partner_unseen",
)
DECISIONS = ("escalated", "kept")
# Relay-BP-5 against union-find alone on an escalated window's rounds,
# keyed (Relay-BP-5 right, union-find right); a window Relay-BP-5 did
# not converge on is counted apart as not_converged
STRONG_OUTCOME_OF = {
    (True, False): "fixed",
    (False, True): "broke",
    (True, True): "both_right",
    (False, False): "both_wrong",
}
STRONG_OUTCOMES = (
    "fixed",
    "broke",
    "both_right",
    "both_wrong",
    "not_converged",
)
LABEL_PREFIX = "label_"

csv.field_size_limit(sys.maxsize)


def main(run_dir: pathlib.Path, results_dir: pathlib.Path) -> None:
    """Every table written from one run folder into results_dir."""
    tasks = task_settings(run_dir)
    shots_path = run_dir / "shots.csv"
    outcomes_path = run_dir / "window_outcomes.csv"
    shots = shot_counts(shots_path, tasks)
    windows = window_tables(outcomes_path, tasks)
    union_find_answers = union_find_answers_on(
        outcomes_path, tasks, windows.escalated_rounds
    )
    strong = strong_on_escalated(windows.escalated, union_find_answers)
    tables = {
        "shot_summary.csv": windows.shot_rows(shots),
        "window_summary.csv": windows.window_rows(),
        "switching_summary.csv": windows.switching_rows(strong),
        "union_find_by_gap.csv": windows.gap_rows(),
        "kept_by_detection_events.csv": windows.event_rows(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    for file_name, rows in tables.items():
        table_path = results_dir / file_name
        write_rows(table_path, rows)


def task_settings(run_dir: pathlib.Path) -> dict:
    """Each task id's configuration, distance and physical error rate.

    A task's folder is named configuration_d<distance>_p<rate>, as
    run.py names it, and its machine.json holds its id.
    """
    record_paths = run_dir.glob("tasks/*/machine.json")
    tasks = {}
    for record_path in sorted(record_paths):
        with record_path.open() as record_file:
            record = json.load(record_file)
        tasks[record["id"]] = _parts_of(record["name"])
    return tasks


def shot_counts(shots_path: pathlib.Path, tasks: dict) -> dict:
    """Each setting's shots, failed shots and windows decoded."""
    counts = collections.defaultdict(collections.Counter)
    for row in _rows(shots_path):
        setting_counts = counts[tasks[row["task_id"]]]
        setting_counts["shots"] += 1
        setting_counts["failed_shots"] += row["logical_failure"] == "True"
        setting_counts["decoded_windows"] += int(row["decoded_windows"])
    return counts


class WindowTables:
    """The counts one pass over window_outcomes.csv gives.

    escalated holds each escalated window's (distance, rate, seed, first
    and last commit round, class of Relay-BP-5's answer, converged);
    escalated_rounds the same spans by (distance, rate, seed), for the
    join with union-find alone.
    """

    def __init__(self) -> None:
        self.answers = collections.Counter()
        self.shots_with_wrong = collections.Counter()
        self.failed_by_parity = collections.Counter()
        self.decisions = collections.Counter()
        self.gaps = collections.Counter()
        self.events = collections.Counter()
        self.escalated = []
        self.escalated_rounds = collections.defaultdict(set)

    def shot_rows(self, shots: dict) -> list:
        """shot_summary.csv, checking each failure against the windows."""
        rows = []
        for setting in sorted(shots):
            counts = shots[setting]
            failed = counts["failed_shots"]
            assert failed == self.failed_by_parity[setting], (
                f"window answers do not give the failed shots of {setting}"
            )
            row = _failure_row(setting, counts["shots"], failed)
            row["decoded_windows"] = counts["decoded_windows"]
            row["shots_with_a_wrong_window"] = self.shots_with_wrong[setting]
            rows.append(row)
        return rows

    def window_rows(self) -> list:
        """window_summary.csv: final answers by class per setting."""
        settings = {key[:3] for key in self.answers}
        rows = []
        for setting in sorted(settings):
            row = _setting_row(setting)
            for answer_class in ANSWER_CLASSES:
                row[answer_class] = self.answers[(*setting, answer_class)]
            rows.append(row)
        return rows

    def switching_rows(self, strong: collections.Counter) -> list:
        """switching_summary.csv: decisions, answers and strong outcomes."""
        settings = {key[:2] for key in self.decisions}
        outcome_columns = _strong_columns()
        rows = []
        for setting in sorted(settings):
            row = _setting_row(setting)
            decision_columns = _decision_columns(self.decisions, setting)
            row.update(decision_columns)
            for column in outcome_columns:
                row[column] = strong[(*setting, column)]
            rows.append(row)
        return rows

    def gap_rows(self) -> list:
        """union_find_by_gap.csv: windows by gap bin and answer class."""
        rows = []
        for key in sorted(self.gaps):
            distance, rate, gap_bin, answer_class = key
            row = _setting_row((distance, rate))
            row["gap_low_db"] = gap_bin * GAP_BIN_DECIBELS
            row["gap_high_db"] = (gap_bin + 1) * GAP_BIN_DECIBELS
            row["union_find_answer"] = answer_class
            row["windows"] = self.gaps[key]
            rows.append(row)
        return rows

    def event_rows(self) -> list:
        """kept_by_detection_events.csv: kept windows by event count."""
        rows = []
        for key in sorted(self.events):
            distance, rate, events, answer_class = key
            row = _setting_row((distance, rate))
            row["detection_events"] = events
            row["union_find_answer"] = answer_class
            row["windows"] = self.events[key]
            rows.append(row)
        return rows


def window_tables(outcomes_path: pathlib.Path, tasks: dict) -> WindowTables:
    """Every count but the union-find join, in one pass.

    A shot's rows are consecutive in the folded file, so each shot is
    counted when its last row has passed.
    """
    tables = WindowTables()
    shot = _ShotTally()
    for row in _rows(outcomes_path):
        setting = tasks[row["task_id"]]
        shot_key = (setting, row["seed"])
        if shot_key != shot.key:
            shot.close_into(tables)
            shot = _ShotTally(shot_key)
        shot.add(row)
    shot.close_into(tables)
    return tables


def union_find_answers_on(
    outcomes_path: pathlib.Path, tasks: dict, escalated_rounds: dict
) -> dict:
    """Union-find alone's answer on each escalated span, and its edges.

    Keyed (distance, rate, seed, commit_lo, commit_hi): the XOR of the
    answers and of the labels of union-find alone's windows inside the
    span, the rounds they cover, which must be the whole span, and
    whether a window just outside either edge is wrong, the span's
    possible seam partner.
    """
    spans = {}
    for row in _rows(outcomes_path):
        configuration, distance, rate = tasks[row["task_id"]]
        if configuration != UNION_FIND_ALONE:
            continue
        shot_key = (distance, rate, row["seed"])
        shot_spans = escalated_rounds.get(shot_key, ())
        for commit_lo, commit_hi in shot_spans:
            span_key = (*shot_key, commit_lo, commit_hi)
            _add_to_span(spans, span_key, row)
    return spans


def strong_on_escalated(
    escalated: list, union_find_answers: dict
) -> collections.Counter:
    """Escalated windows by Relay-BP-5's outcome, per setting.

    Counted twice: with a wrong window in a seam pair counted right,
    and with every wrong label counted wrong (the LABEL_PREFIX columns).
    """
    counts = collections.Counter()
    for window in escalated:
        distance, rate, seed, commit_lo, commit_hi, strong_class, converged = (
            window
        )
        span_key = (distance, rate, seed, commit_lo, commit_hi)
        span = union_find_answers[span_key]
        union_find_class = _span_class(span, span_key)
        outcome = _strong_outcome(strong_class, union_find_class, converged)
        label_outcome = _label_outcome(
            strong_class, union_find_class, converged
        )
        counts[(distance, rate, outcome)] += 1
        counts[(distance, rate, LABEL_PREFIX + label_outcome)] += 1
    return counts


def paired_positions(spans: list) -> set:
    """The positions of wrong windows that pair with a wrong neighbour.

    spans holds (commit_lo, commit_hi, is_wrong) in time order; a
    commit_hi of None never pairs with the window after it.
    """
    paired = set()
    open_position = None
    for position, (commit_lo, _commit_hi, is_wrong) in enumerate(spans):
        if not is_wrong:
            open_position = None
            continue
        if _is_after(spans, open_position, commit_lo):
            paired.update((open_position, position))
            open_position = None
            continue
        open_position = position
    return paired


def wilson_interval(failures: int, shots: int) -> tuple:
    """The 95% Wilson score interval of failures / shots.

    Wilson, J. Am. Stat. Assoc. 22 (1927); Brown, Cai and DasGupta,
    Stat. Sci. 16 (2001) Eq. 4, which recommend it over the Wald
    interval at small counts.
    """
    z_squared = NORMAL_QUANTILE_95**2
    rate = failures / shots
    denominator = 1 + z_squared / shots
    centre = (rate + z_squared / (2 * shots)) / denominator
    spread_squared = rate * (1 - rate) / shots + z_squared / (4 * shots**2)
    spread = math.sqrt(spread_squared)
    half_width = NORMAL_QUANTILE_95 * spread / denominator
    # at zero failures the bound is 0 up to rounding
    rounded_low = centre - half_width
    low = max(0.0, rounded_low)
    return low, centre + half_width


def write_rows(path: pathlib.Path, rows: list) -> None:
    """Rows of dicts as a csv file, the first row's keys its header."""
    first_row = rows[0]
    with path.open("w", newline="") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=list(first_row))
        writer.writeheader()
        writer.writerows(rows)


class _ShotTally:
    """One shot's window rows, counted when the shot ends."""

    def __init__(self, key: tuple = None) -> None:
        self.key = key
        self.rows = []

    def add(self, row: dict) -> None:
        self.rows.append(row)

    def close_into(self, tables: WindowTables) -> None:
        """The shot's final answers, and switching's decisions, counted."""
        if self.key is None:
            return
        setting, seed = self.key
        self.rows.sort(key=_commit_lo_of)
        final_classes = _count_answers(tables, setting, self.rows)
        if setting[0] == SWITCHING:
            shot = (setting, seed, self.rows, final_classes)
            _count_switching(tables, shot)


def _count_answers(tables: WindowTables, setting: tuple, rows: list) -> list:
    """Final answers by class, and the shot failed when its parities differ.

    The shot fails exactly when its answers' parity differs from its
    labels' parity (2509.03815 Eq. 2). Returns each row's class.
    """
    spans = []
    answer_parity = 0
    label_parity = 0
    for row in rows:
        is_wrong = row["is_right"] == "False"
        spans.append((int(row["commit_lo"]), int(row["commit_hi"]), is_wrong))
        answer_parity ^= _bit_of(row["answer"])
        label_parity ^= _bit_of(row["label"])
    classes = _answer_classes(spans)
    for answer_class in classes:
        tables.answers[(*setting, answer_class)] += 1
    wrong_flags = [span[2] for span in spans]
    tables.shots_with_wrong[setting] += any(wrong_flags)
    tables.failed_by_parity[setting] += answer_parity != label_parity
    return classes


def _count_switching(tables: WindowTables, shot: tuple) -> None:
    """Each union-find window's decision, answer class, gap and events.

    A row with no gap holds no union-find decision and is skipped.
    """
    setting, seed, rows, final_classes = shot
    decided = []
    for row, final_class in zip(rows, final_classes, strict=True):
        if row["gap_nats"] != "":
            decided.append((row, final_class))
    spans = [_union_find_span(row) for row, _final_class in decided]
    classes = _answer_classes(spans)
    _configuration, distance, rate = setting
    for (row, final_class), answer_class in zip(decided, classes, strict=True):
        decision = DECISIONS[row["is_escalated"] != "True"]
        tables.decisions[(distance, rate, decision, answer_class)] += 1
        gap_bin = _gap_bin_of(row)
        tables.gaps[(distance, rate, gap_bin, answer_class)] += 1
        if decision == DECISIONS[0]:
            window = (distance, rate, seed, row, final_class)
            _keep_escalated(tables, window)
            continue
        events = int(row["detection_events"])
        tables.events[(distance, rate, events, answer_class)] += 1


def _union_find_span(row: dict) -> tuple:
    """Union-find's answer on a window's own rounds, as a span to pair.

    An escalated window's own rounds end where the two windows it takes
    in begin, so no window after it is its neighbour.
    """
    commit_lo = int(row["commit_lo"])
    if row["is_escalated"] == "True":
        is_wrong = row["weak_answer"] != row["weak_label"]
        return commit_lo, None, is_wrong
    is_wrong = row["is_right"] == "False"
    return commit_lo, int(row["commit_hi"]), is_wrong


def _answer_classes(spans: list) -> list:
    """Each span's class in ANSWER_CLASSES.

    A wrong window's partner is unseen when it sits beside rounds
    union-find never answered: an escalated window, or the window after
    one.
    """
    paired = paired_positions(spans)
    classes = []
    previous_hi = 0
    for position, (_commit_lo, commit_hi, is_wrong) in enumerate(spans):
        is_beside_unseen = commit_hi is None or previous_hi is None
        is_paired = position in paired
        answer_class = _answer_class(is_wrong, is_paired, is_beside_unseen)
        classes.append(answer_class)
        previous_hi = commit_hi
    return classes


def _answer_class(
    is_wrong: bool, is_paired: bool, is_beside_unseen: bool
) -> str:
    if not is_wrong:
        return "right"
    if is_paired:
        return "wrong_paired"
    if is_beside_unseen:
        return "wrong_partner_unseen"
    return "wrong_alone"


def _is_after(spans: list, open_position, commit_lo: int) -> bool:
    """Whether the open wrong window ends just before commit_lo."""
    if open_position is None:
        return False
    open_hi = spans[open_position][1]
    return open_hi is not None and open_hi + 1 == commit_lo


def _keep_escalated(tables: WindowTables, window: tuple) -> None:
    """An escalated window's span and Relay-BP-5 outcome, for the join."""
    distance, rate, seed, row, final_class = window
    commit_lo = int(row["commit_lo"])
    commit_hi = int(row["commit_hi"])
    converged = row["decode_status"] == ""
    escalated = (
        distance,
        rate,
        seed,
        commit_lo,
        commit_hi,
        final_class,
        converged,
    )
    tables.escalated.append(escalated)
    shot_spans = tables.escalated_rounds[(distance, rate, seed)]
    shot_spans.add((commit_lo, commit_hi))


def _decision_columns(decisions: collections.Counter, setting: tuple) -> dict:
    """decision_answer columns, e.g. escalated_wrong_alone, per setting."""
    columns = {}
    for decision in DECISIONS:
        for answer_class in ANSWER_CLASSES:
            column = f"{decision}_{answer_class}"
            columns[column] = decisions[(*setting, decision, answer_class)]
    return columns


def _strong_columns() -> list:
    """The strong outcomes, seam-corrected first, then by label."""
    columns = list(STRONG_OUTCOMES)
    for outcome in STRONG_OUTCOMES:
        column = LABEL_PREFIX + outcome
        columns.append(column)
    return columns


def _gap_bin_of(row: dict) -> int:
    gap_nats = float(row["gap_nats"])
    gap_decibels = threshold_sources.nats_to_decibels(gap_nats)
    gap_bins = gap_decibels / GAP_BIN_DECIBELS
    return math.floor(gap_bins)


def _add_to_span(spans: dict, span_key: tuple, row: dict) -> None:
    """A union-find-alone window added to a span inside or beside it."""
    _distance, _rate, _seed, commit_lo, commit_hi = span_key
    window_lo = int(row["commit_lo"])
    window_hi = int(row["commit_hi"])
    span = spans.setdefault(span_key, [0, 0, 0, False])
    is_edge = window_hi + 1 == commit_lo or window_lo == commit_hi + 1
    if is_edge:
        span[3] = span[3] or row["is_right"] == "False"
        return
    if window_lo < commit_lo or window_hi > commit_hi:
        return
    span[0] ^= _bit_of(row["answer"])
    span[1] ^= _bit_of(row["label"])
    span[2] += window_hi - window_lo + 1


def _span_class(span: list, span_key: tuple) -> str:
    """Union-find alone's answer class on the span.

    A wrong span with a wrong window just outside an edge is counted as
    a seam pair.
    """
    answer, label, rounds, is_edge_wrong = span
    _distance, _rate, _seed, commit_lo, commit_hi = span_key
    span_rounds = commit_hi - commit_lo + 1
    assert rounds == span_rounds, (
        f"union-find alone's windows do not tile the span {span_key}"
    )
    is_wrong = answer != label
    return _answer_class(is_wrong, is_edge_wrong, False)


def _strong_outcome(
    strong_class: str, union_find_class: str, converged: bool
) -> str:
    """The outcome with a wrong window in a seam pair counted right."""
    if not converged:
        return "not_converged"
    strong_right = strong_class in ("right", "wrong_paired")
    union_find_right = union_find_class in ("right", "wrong_paired")
    return STRONG_OUTCOME_OF[(strong_right, union_find_right)]


def _label_outcome(
    strong_class: str, union_find_class: str, converged: bool
) -> str:
    """The outcome with every wrong label counted wrong."""
    if not converged:
        return "not_converged"
    strong_right = strong_class == "right"
    union_find_right = union_find_class == "right"
    return STRONG_OUTCOME_OF[(strong_right, union_find_right)]


def _failure_row(setting: tuple, shots: int, failed: int) -> dict:
    low, high = wilson_interval(failed, shots)
    row = _setting_row(setting)
    row["shots"] = shots
    row["failed_shots"] = failed
    row["failure_rate"] = failed / shots
    row["failure_rate_low"] = low
    row["failure_rate_high"] = high
    row["is_shown"] = failed >= SHOWN_FAILURES
    return row


def _setting_row(setting: tuple) -> dict:
    """The first columns of a row: configuration if any, d and rate."""
    names = ("distance", "physical_error_rate")
    if len(setting) == 3:
        names = ("configuration", *names)
    return dict(zip(names, setting, strict=True))


def _commit_lo_of(row: dict) -> int:
    return int(row["commit_lo"])


def _parts_of(task_name: str) -> tuple:
    """configuration_d<distance>_p<rate> as its three parts."""
    configuration, distance_part, rate_part = task_name.rsplit("_", 2)
    distance_text = distance_part.removeprefix("d")
    rate_text = rate_part.removeprefix("p")
    return configuration, int(distance_text), float(rate_text)


def _bit_of(cell: str) -> int:
    """A one-observable bit cell, json text such as "1", as 0 or 1."""
    bit_text = json.loads(cell)
    return int(bit_text)


def _rows(path: pathlib.Path):
    with path.open(newline="") as table_file:
        yield from csv.DictReader(table_file)


if __name__ == "__main__":
    run_folder = pathlib.Path(sys.argv[1])
    results_folder = pathlib.Path(sys.argv[2])
    main(run_folder, results_folder)
