"""The switching-per-window run's tables, one per figure, from its folder.

`python tables.py <run folder> <results folder>` reads the folded
shots.csv and window_outcomes.csv and each task's record, and writes
five tidy tables, every count kept so any figure can be redrawn:

- shot_failures.csv: each configuration's failed shots per setting, the
  rate and its 95% Wilson interval, and whether it has the 20 failures
  a shown rate needs (LOG.md, stop rule).
- union_find_by_gap.csv: switching's union-find windows by their gap in
  1 dB bins, and how many union-find got wrong: P_weak(error | gap),
  Toshio et al. 2510.25222 Eq. 2 and Fig. 5.
- switching_decisions.csv: each union-find window of switching as a
  correct escalation (escalated, union-find wrong), a false positive
  (escalated, right), a false negative (kept, wrong) or a correct keep.
- strong_on_escalated.csv: each escalated window's Relay-BP-5 answer on
  the rounds its double window commits, beside union-find alone's
  answer on the same rounds and shot, and whether Relay-BP-5 converged.
- wrong_windows.csv: shots by configuration, setting, failed or not,
  and their number of wrong windows.

A window is right when its answer equals its label, the parity of the
true errors it owns (2509.03815 Eq. 1). A kept window's union-find
answer is its row's answer; an escalated window's is its weak_answer,
labelled on its own rounds. Union-find alone's windows tile the same
rounds as switching's (the same sliding windows and seeds), so its
answer on a double window's rounds is the XOR of the windows inside.
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
# switching's four outcomes, keyed (escalated, union-find right)
DECISION_OF = {
    (True, False): "correct_escalation",
    (True, True): "false_positive",
    (False, False): "false_negative",
    (False, True): "correct_keep",
}
DECISIONS = (
    "correct_escalation",
    "false_positive",
    "false_negative",
    "correct_keep",
)
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

csv.field_size_limit(sys.maxsize)


def main(run_dir: pathlib.Path, results_dir: pathlib.Path) -> None:
    """Every table written from one run folder into results_dir."""
    tasks = task_settings(run_dir)
    shots_path = run_dir / "shots.csv"
    outcomes_path = run_dir / "window_outcomes.csv"
    failures = shot_failures(shots_path, tasks)
    windows = window_tables(outcomes_path, tasks)
    escalated_rounds = windows.escalated_rounds
    union_find_answers = union_find_answers_on(
        outcomes_path, tasks, escalated_rounds
    )
    strong = strong_on_escalated(windows.escalated, union_find_answers)
    tables = {
        "shot_failures.csv": failures,
        "union_find_by_gap.csv": windows.gap_rows(),
        "switching_decisions.csv": windows.decision_rows(),
        "strong_on_escalated.csv": strong,
        "wrong_windows.csv": windows.wrong_window_rows(),
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


def shot_failures(shots_path: pathlib.Path, tasks: dict) -> list:
    """Each task's shots and failed shots, with the rate's interval."""
    shots = collections.Counter()
    failed = collections.Counter()
    for row in _rows(shots_path):
        setting = tasks[row["task_id"]]
        shots[setting] += 1
        failed[setting] += row["logical_failure"] == "True"
    rows = []
    for setting in sorted(shots):
        row = _failure_row(setting, shots[setting], failed[setting])
        rows.append(row)
    return rows


class WindowTables:
    """The counts one pass over window_outcomes.csv gives.

    escalated holds each escalated window's (distance, rate, seed, first
    and last commit round, Relay-BP-5 right, converged); escalated_rounds
    the same spans by (distance, rate, seed), for the join with
    union-find alone.
    """

    def __init__(self) -> None:
        self.gap_windows = collections.Counter()
        self.gap_wrong = collections.Counter()
        self.decision_counts = collections.Counter()
        self.shot_wrong_counts = collections.Counter()
        self.escalated = []
        self.escalated_rounds = collections.defaultdict(set)

    def gap_rows(self) -> list:
        """union_find_by_gap.csv: windows and wrong ones per gap bin."""
        rows = []
        for key in sorted(self.gap_windows):
            distance, rate, gap_bin = key
            row = {
                "distance": distance,
                "physical_error_rate": rate,
                "gap_low_db": gap_bin * GAP_BIN_DECIBELS,
                "gap_high_db": (gap_bin + 1) * GAP_BIN_DECIBELS,
                "windows": self.gap_windows[key],
                "union_find_wrong": self.gap_wrong[key],
            }
            rows.append(row)
        return rows

    def decision_rows(self) -> list:
        """switching_decisions.csv: the four outcomes per setting."""
        return _setting_rows(self.decision_counts, DECISIONS)

    def wrong_window_rows(self) -> list:
        """wrong_windows.csv: shots by failed and wrong-window count."""
        rows = []
        for key in sorted(self.shot_wrong_counts):
            configuration, distance, rate, failed, wrong = key
            row = {
                "configuration": configuration,
                "distance": distance,
                "physical_error_rate": rate,
                "shot_failed": failed,
                "wrong_windows": wrong,
                "shots": self.shot_wrong_counts[key],
            }
            rows.append(row)
        return rows


def window_tables(outcomes_path: pathlib.Path, tasks: dict) -> WindowTables:
    """Every count but the union-find join, in one pass.

    A shot's rows are consecutive in the folded file, so its wrong
    windows and its parities are counted as its rows pass.
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
        if setting[0] == SWITCHING:
            _count_switching_window(tables, setting, row)
    shot.close_into(tables)
    return tables


def union_find_answers_on(
    outcomes_path: pathlib.Path, tasks: dict, escalated_rounds: dict
) -> dict:
    """Union-find alone's answer and label parity on each escalated span.

    Keyed (distance, rate, seed, commit_lo, commit_hi): the XOR of the
    answers and of the labels of union-find alone's windows inside the
    span, and the rounds they cover, which must be the whole span.
    """
    spans = {}
    for row in _rows(outcomes_path):
        configuration, distance, rate = tasks[row["task_id"]]
        if configuration != UNION_FIND_ALONE:
            continue
        shot_key = (distance, rate, row["seed"])
        shot_spans = escalated_rounds.get(shot_key, ())
        for commit_lo, commit_hi in shot_spans:
            _add_inside(spans, shot_key, commit_lo, commit_hi, row)
    return spans


def strong_on_escalated(escalated: list, union_find_answers: dict) -> list:
    """strong_on_escalated.csv: escalated windows by both decoders' outcome."""
    counts = collections.Counter()
    for window in escalated:
        distance, rate, seed, commit_lo, commit_hi, strong_right, converged = (
            window
        )
        span_key = (distance, rate, seed, commit_lo, commit_hi)
        span = union_find_answers[span_key]
        union_find_right = _span_right(span, span_key)
        outcome = _strong_outcome(strong_right, union_find_right, converged)
        counts[(distance, rate, outcome)] += 1
    return _setting_rows(counts, STRONG_OUTCOMES)


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
    """One shot's wrong windows and the parity of its answers and labels."""

    def __init__(self, key: tuple = None) -> None:
        self.key = key
        self.wrong = 0
        self.answer_parity = 0
        self.label_parity = 0

    def add(self, row: dict) -> None:
        self.wrong += row["is_right"] == "False"
        self.answer_parity ^= _bit_of(row["answer"])
        self.label_parity ^= _bit_of(row["label"])

    def close_into(self, tables: WindowTables) -> None:
        """The shot counted, failed when its parities differ (Eq. 2)."""
        if self.key is None:
            return
        setting, _seed = self.key
        failed = self.answer_parity != self.label_parity
        tables.shot_wrong_counts[(*setting, failed, self.wrong)] += 1


def _count_switching_window(
    tables: WindowTables, setting: tuple, row: dict
) -> None:
    """A switching window's gap bin, decision and, escalated, its span.

    A window with no gap was absorbed into a double window before
    union-find answered it, so switching made no decision on it.
    """
    if row["gap_nats"] == "":
        return
    _configuration, distance, rate = setting
    is_escalated = row["is_escalated"] == "True"
    union_find_right = _union_find_right(row, is_escalated)
    gap_nats = float(row["gap_nats"])
    gap_decibels = threshold_sources.nats_to_decibels(gap_nats)
    gap_bins = gap_decibels / GAP_BIN_DECIBELS
    gap_bin = math.floor(gap_bins)
    tables.gap_windows[(distance, rate, gap_bin)] += 1
    tables.gap_wrong[(distance, rate, gap_bin)] += not union_find_right
    decision = DECISION_OF[(is_escalated, union_find_right)]
    tables.decision_counts[(distance, rate, decision)] += 1
    if is_escalated:
        _keep_escalated(tables, distance, rate, row)


def _keep_escalated(
    tables: WindowTables, distance: int, rate: float, row: dict
) -> None:
    """An escalated window's span and Relay-BP-5 outcome, for the join."""
    seed = row["seed"]
    commit_lo = int(row["commit_lo"])
    commit_hi = int(row["commit_hi"])
    strong_right = row["is_right"] == "True"
    converged = row["decode_status"] == ""
    window = (
        distance,
        rate,
        seed,
        commit_lo,
        commit_hi,
        strong_right,
        converged,
    )
    tables.escalated.append(window)
    shot_spans = tables.escalated_rounds[(distance, rate, seed)]
    shot_spans.add((commit_lo, commit_hi))


def _union_find_right(row: dict, is_escalated: bool) -> bool:
    """Union-find's answer on the window's own rounds against its label."""
    if is_escalated:
        return row["weak_answer"] == row["weak_label"]
    return row["is_right"] == "True"


def _add_inside(
    spans: dict, shot_key: tuple, commit_lo: int, commit_hi: int, row: dict
) -> None:
    """A union-find-alone window added to a span it lies inside."""
    window_lo = int(row["commit_lo"])
    window_hi = int(row["commit_hi"])
    if window_lo < commit_lo or window_hi > commit_hi:
        return
    span_key = (*shot_key, commit_lo, commit_hi)
    answer, label, rounds = spans.get(span_key, (0, 0, 0))
    answer ^= _bit_of(row["answer"])
    label ^= _bit_of(row["label"])
    rounds += window_hi - window_lo + 1
    spans[span_key] = (answer, label, rounds)


def _span_right(span: tuple, span_key: tuple) -> bool:
    """Union-find alone's answers on the span against its labels."""
    answer, label, rounds = span
    _distance, _rate, _seed, commit_lo, commit_hi = span_key
    span_rounds = commit_hi - commit_lo + 1
    assert rounds == span_rounds, (
        f"union-find alone's windows do not tile the span {span_key}"
    )
    return answer == label


def _strong_outcome(
    strong_right: bool, union_find_right: bool, converged: bool
) -> str:
    if not converged:
        return "not_converged"
    return STRONG_OUTCOME_OF[(strong_right, union_find_right)]


def _setting_rows(counts: collections.Counter, outcomes: tuple) -> list:
    """One row per setting, a column per outcome, from (d, rate, outcome)."""
    settings = {key[:2] for key in counts}
    rows = []
    for distance, rate in sorted(settings):
        row = {"distance": distance, "physical_error_rate": rate}
        for outcome in outcomes:
            row[outcome] = counts[(distance, rate, outcome)]
        rows.append(row)
    return rows


def _failure_row(setting: tuple, shots: int, failed: int) -> dict:
    configuration, distance, rate = setting
    low, high = wilson_interval(failed, shots)
    return {
        "configuration": configuration,
        "distance": distance,
        "physical_error_rate": rate,
        "shots": shots,
        "failed_shots": failed,
        "failure_rate": failed / shots,
        "failure_rate_low": low,
        "failure_rate_high": high,
        "is_shown": failed >= SHOWN_FAILURES,
    }


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
