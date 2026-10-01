"""decsim.results and `decsim diff`: run folders read back and compared.

Two runs of one yaml whose decoder is priced by a card are the same
run, number for number; a run at another shot length differs in one
setting and in the circuit it ran. The folders come from `decsim
collect` itself, so the reader is pinned against what the writer wrote.
The paired comparison is checked on shots rewritten by hand against
Robbins' beta-binomial mixture, computed with scipy.special.betaln, and
Howard et al. (arXiv 1810.08240) eq. (24), computed here from the
formula.
"""

import csv
import json
import math
import pathlib
import shutil

import numpy
import pytest
import scipy.special
import scipy.stats

import decsim.experiments.command as command
import decsim.results as results
import tests.experiments.yaml_configs as yaml_configs

TWO_POINT_SWEEP = [
    {
        "axes": {
            "workload.arguments.physical_error_probability": [0.003, 0.01],
            "qpu.distance": [3],
            "qpu.round_period_microseconds": [1.0],
        },
        "collection": {"max_shots": 20},
    }
]

# The shots of a paired folder's first point: room for 25 shots that
# only the first folder fails beside 5 that only the second fails.
PAIRED_SHOTS = 200
# The seeds both paired folders fail.
SHARED_FAILING_SEEDS = (0, 1, 2, 3, 4)
# The paired line of a point whose 20 shots failed alike in both runs.
SAME_SHOTS_TEXT = (
    " paired on the 20 shots both runs scored of 20 shared shots (0 "
    "unscored in either run, left out): 0 failed in the first only, 0 in "
    "the second only; no difference shown by the mixture test;"
)

# The swept path that tells the two points apart, a column of sweep.csv.
PROBABILITY_AXIS = "workload.arguments.physical_error_probability"
ROUNDS_COLUMN = "settings.workload.row_settings.arguments.rounds_per_shot"
PROBABILITY_COLUMN = (
    "settings.workload.row_settings.arguments.physical_error_probability"
)


def _point_text(probability: float) -> str:
    """How diff names a point: its metadata's json text."""
    metadata = {
        "qpu.distance": 3,
        "workload.arguments.physical_error_probability": probability,
        "qpu.round_period_microseconds": 1.0,
    }
    text = json.dumps(metadata, sort_keys=True)
    return f"{text}:"


def _collected(folder, overrides: dict):
    """One `decsim collect` of the minimal config into its own folder."""
    folder.mkdir()
    sweep = {"sweep": TWO_POINT_SWEEP}
    sweep.update(overrides)
    config_path = yaml_configs.write_config(folder, sweep)
    run_dir = folder / "run"
    command.main(["collect", str(config_path), "--out", str(run_dir)])
    return yaml_configs.run_folder_of(run_dir)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two runs of one yaml, and one at another shot length."""
    root = tmp_path_factory.mktemp("runs")
    first_folder = root / "first"
    second_folder = root / "second"
    shorter_folder = root / "shorter"
    shorter_workload = yaml_configs.memory_workload(12)
    shorter_overrides = {"workload": shorter_workload}
    first = _collected(first_folder, {})
    second = _collected(second_folder, {})
    shorter = _collected(shorter_folder, shorter_overrides)
    return {"first": first, "second": second, "shorter": shorter}


def _diff_printed(capsys, first, second) -> list:
    command.main(["diff", str(first), str(second)])
    printed = capsys.readouterr()
    return printed.out.splitlines()


def _csv_rows(path: pathlib.Path) -> list:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv_rows(path: pathlib.Path, rows: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rewritten(run_dir, tmp_path, name: str, values: dict):
    """A copy of a run folder whose first sweep.csv row holds values."""
    destination = tmp_path / name
    shutil.copytree(run_dir, destination)
    sweep_path = destination / "sweep.csv"
    rows = _csv_rows(sweep_path)
    rows[0].update(values)
    _write_csv_rows(sweep_path, rows)
    return destination


def test_two_runs_of_one_yaml_are_the_same(runs, capsys):
    """Nothing differs, and every shared point's shots fail alike."""
    lines = _diff_printed(capsys, runs["first"], runs["second"])

    low_point = _point_text(0.003)
    high_point = _point_text(0.01)
    low_line = _line_with(lines, low_point)
    high_line = _line_with(lines, high_point)
    assert lines[:5] == [
        "settings:",
        "  the same",
        "inputs:",
        "  the same",
        "results:",
    ]
    assert len(lines) == 7
    assert SAME_SHOTS_TEXT in low_line
    assert SAME_SHOTS_TEXT in high_line


def test_diff_names_the_setting_and_the_input_that_changed(runs, capsys):
    lines = _diff_printed(capsys, runs["first"], runs["shorter"])

    point = _point_text(0.003)
    assert (
        f"  {point} settings.workload.row_settings.arguments."
        "rounds_per_shot: 15 -> 12"
    ) in lines
    assert f"  {point} operation_1.stim: sha256 differs" in lines
    assert f"  {point} operations.json: sha256 differs" in lines


def test_diff_judges_a_logical_error_rate_by_its_exact_interval(
    runs, tmp_path, capsys
):
    """Overlapping limits agree; limits that do not overlap do not."""
    near = {
        "logical_error_rate_estimate": 0.1,
        "logical_error_rate_low": 0.05,
        "logical_error_rate_high": 0.2,
    }
    known = _rewritten(runs["first"], tmp_path, "known", near)
    inside = _rewritten(
        runs["first"],
        tmp_path,
        "inside",
        {**near, "logical_error_rate_estimate": 0.12},
    )
    apart = _rewritten(
        runs["first"],
        tmp_path,
        "apart",
        {
            "logical_error_rate_estimate": 0.9,
            "logical_error_rate_low": 0.85,
            "logical_error_rate_high": 0.95,
        },
    )

    inside_lines = _diff_printed(capsys, known, inside)
    apart_lines = _diff_printed(capsys, known, apart)

    inside_rate = _line_with(inside_lines, " logical_error_rate_estimate: ")
    apart_rate = _line_with(apart_lines, " logical_error_rate_estimate: ")
    assert inside_rate.endswith(", within error bars")
    assert apart_rate.endswith(", beyond error bars")
    # the rate's line and each of the two points' paired lines
    assert len(apart_lines) == 8


def test_diff_makes_no_comparison_of_a_rate_with_no_interval(
    runs, tmp_path, capsys
):
    """No scored shot gives no interval, and so nothing to compare by.

    A point whose every shot went unscored has no rate and no limits; a
    verdict read off them would say the two runs agree on nothing.
    """
    unscored = _rewritten(
        runs["first"],
        tmp_path,
        "unscored",
        {
            "logical_error_rate_estimate": 0.5,
            "logical_error_rate_low": "",
            "logical_error_rate_high": "",
        },
    )

    lines = _diff_printed(capsys, runs["first"], unscored)

    rate_line = _line_with(lines, " logical_error_rate_estimate: ")
    assert rate_line.endswith(", no statistical comparison possible")


def _line_with(lines: list, text: str) -> str:
    """The one printed line that holds this text."""
    (line,) = [line for line in lines if text in line]
    return line


def _spread_shots(run_dir, column: str, half_width: float) -> None:
    """The first point's shots of a column pushed apart, their mean kept.

    Alternate shots move up and down by half_width, so the column's
    standard error over the point's 20 shots is half_width times
    sqrt(20 / 19) / sqrt(20), about 0.23 half_width.
    """
    shots_path = run_dir / "shots.csv"
    rows = _csv_rows(shots_path)
    first_point = rows[0]["point_id"]
    sign = 1
    for row in rows:
        if row["point_id"] != first_point:
            continue
        value = float(row[column])
        row[column] = value + sign * half_width
        sign = -sign
    _write_csv_rows(shots_path, rows)


@pytest.mark.parametrize(
    "column, shot_column",
    [("load", "load"), ("windows_per_shot", "decoded_windows")],
)
def test_diff_judges_a_mean_by_the_standard_error_of_its_shots(
    runs, tmp_path, capsys, column, shot_column
):
    """A mean moved by 1 standard error agrees; one moved by 10 does not.

    windows_per_shot is the mean of shots.csv's windows column.
    """
    first_rows = results.load(runs["first"])
    mean = first_rows[0][column]
    standard_error = 0.01 * (20 / 19) ** 0.5 / 20**0.5
    near_mean = mean + standard_error
    far_mean = mean + 10 * standard_error
    near = _rewritten(runs["first"], tmp_path, "near", {column: near_mean})
    far = _rewritten(runs["first"], tmp_path, "far", {column: far_mean})
    _spread_shots(near, shot_column, 0.01)
    _spread_shots(far, shot_column, 0.01)

    near_lines = _diff_printed(capsys, runs["first"], near)
    far_lines = _diff_printed(capsys, runs["first"], far)

    near_line = _line_with(near_lines, f" {column}: ")
    far_line = _line_with(far_lines, f" {column}: ")
    assert near_line.endswith(", within error bars")
    assert far_line.endswith(", beyond error bars")


def test_diff_compares_a_column_with_no_error_bar_exactly(
    runs, tmp_path, capsys
):
    """The largest over shots has no standard error, though shots hold it."""
    changed = _rewritten(
        runs["first"], tmp_path, "changed", {"max_queued_windows": 999}
    )

    lines = _diff_printed(capsys, runs["first"], changed)

    changed_line = _line_with(lines, " max_queued_windows: ")
    assert changed_line.endswith(", no error bar: compared exactly")


def _kept_shots(run_dir, keep) -> None:
    """The run's shots.csv rows that keep(row) holds, the rest dropped."""
    shots_path = run_dir / "shots.csv"
    rows = _csv_rows(shots_path)
    kept = []
    for row in rows:
        if keep(row):
            kept.append(row)
    _write_csv_rows(shots_path, kept)


def _first_shot_only(runs, tmp_path, name: str, values: dict):
    """A copy whose first point kept one shot and whose row holds values."""
    changed = _rewritten(runs["first"], tmp_path, name, values)
    first_rows = results.load(runs["first"])
    first_point = first_rows[0]["point_id"]
    seen = []

    def keep(row) -> bool:
        if row["point_id"] != first_point:
            return True
        seen.append(row)
        return len(seen) == 1

    _kept_shots(changed, keep)
    return changed


def test_diff_gives_a_mean_over_one_shot_no_error_bar(runs, tmp_path, capsys):
    """A standard error needs two shots; one shot's mean is compared exactly."""
    first_rows = results.load(runs["first"])
    moved = first_rows[0]["load"] + 1.0
    changed = _first_shot_only(runs, tmp_path, "one_shot", {"load": moved})

    lines = _diff_printed(capsys, runs["first"], changed)

    load_line = _line_with(lines, " load: ")
    assert load_line.endswith(", no error bar: compared exactly")


def _blank_shot_column(run_dir, column: str) -> None:
    """The run's shots.csv with one column's values left empty."""
    shots_path = run_dir / "shots.csv"
    rows = _csv_rows(shots_path)
    for row in rows:
        row[column] = ""
    _write_csv_rows(shots_path, rows)


def test_diff_gives_a_mean_its_shots_do_not_hold_no_error_bar(
    runs, tmp_path, capsys
):
    """A run that kept no per-shot values of a mean has no error bar for it."""
    first_rows = results.load(runs["first"])
    moved = first_rows[0]["load"] + 1.0
    changed = _rewritten(runs["first"], tmp_path, "unkept", {"load": moved})
    _blank_shot_column(changed, "load")

    lines = _diff_printed(capsys, runs["first"], changed)

    load_line = _line_with(lines, " load: ")
    assert load_line.endswith(", no error bar: compared exactly")


def test_diff_names_a_column_only_the_second_folder_holds(
    runs, tmp_path, capsys
):
    changed = _rewritten(runs["first"], tmp_path, "wider", {"extra": 5})

    lines = _diff_printed(capsys, runs["first"], changed)

    point = _point_text(0.003)
    assert (
        f"  {point} extra: None -> 5, no error bar: compared exactly" in lines
    )


def test_diff_names_a_point_only_one_folder_holds(runs, tmp_path, capsys):
    fewer = tmp_path / "fewer"
    shutil.copytree(runs["first"], fewer)
    sweep_path = fewer / "sweep.csv"
    sweep_text = sweep_path.read_text()
    sweep_lines = sweep_text.splitlines()
    kept_text = "\n".join(sweep_lines[:2]) + "\n"
    sweep_path.write_text(kept_text)

    first_lines = _diff_printed(capsys, runs["first"], fewer)
    second_lines = _diff_printed(capsys, fewer, runs["first"])

    point = _point_text(0.01)
    assert f"  {point} only in the first folder" in first_lines
    assert f"  {point} only in the second folder" in second_lines


def test_diff_refuses_a_folder_without_its_resolved_settings(
    runs, tmp_path, capsys
):
    """Without resolved/ the settings would read as never differing."""
    bare = tmp_path / "bare"
    shutil.copytree(runs["first"], bare)
    resolved_dir = bare / "resolved"
    shutil.rmtree(resolved_dir)

    with pytest.raises(SystemExit) as stopped:
        command.main(["diff", str(runs["shorter"]), str(bare)])
    printed = capsys.readouterr()

    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "holds no resolved/ folder" in printed.err


@pytest.mark.parametrize(
    ("first_only", "second_only", "word"),
    [(25, 5, "differ"), (12, 2, "no difference shown")],
)
def test_the_paired_verdict_is_the_beta_binomial_mixture_against_twenty(
    runs, tmp_path, capsys, first_only, second_only, word
):
    """The verdict is whether log B(a + 1, b + 1) + (a + b) log 2 >= log 20.

    That sum is log M, Robbins' mixture with a uniform prior over the
    chance that the first point is the one failing on a discordant shot
    (Howard et al. Proposition 7): 25 against 5 crosses 20, 12 against 2
    does not.
    """
    first_failing, second_failing = _failing_seeds(first_only, second_only)
    first = _paired_folder(runs["first"], tmp_path, "first", first_failing)
    second = _paired_folder(runs["first"], tmp_path, "second", second_failing)

    lines, row = _paired_diff(capsys, tmp_path, first, second)

    first_shape = first_only + 1
    second_shape = second_only + 1
    discordant_count = first_only + second_only
    log_beta = scipy.special.betaln(first_shape, second_shape)
    log_evidence = log_beta + discordant_count * math.log(2)
    is_difference = log_evidence >= math.log(20)
    paired_line = _line_with(lines, " paired on ")
    assert row["first_only_failures"] == str(first_only)
    assert row["second_only_failures"] == str(second_only)
    assert row["is_mixture_difference"] == str(is_difference)
    assert f"; {word} by the mixture test;" in paired_line


def test_the_paired_interval_is_equation_24_on_the_shifted_differences(
    runs, tmp_path, capsys
):
    first_failing, second_failing = _failing_seeds(25, 5)
    first = _paired_folder(runs["first"], tmp_path, "first", first_failing)
    second = _paired_folder(runs["first"], tmp_path, "second", second_failing)

    _lines, row = _paired_diff(capsys, tmp_path, first, second)

    expected_low, expected_high = _equation_24_difference(
        first_failing, second_failing
    )
    low = float(row["difference_low"])
    high = float(row["difference_high"])
    assert low == pytest.approx(expected_low, rel=1e-12, abs=0)
    assert high == pytest.approx(expected_high, rel=1e-12, abs=0)
    assert row["scored_pairs"] == str(PAIRED_SHOTS)


def test_a_shared_shot_drawn_differently_refuses_the_pairing(
    runs, tmp_path, capsys
):
    """One seed's sample_digest differs: no pairs, the interval line kept."""
    first_failing, second_failing = _failing_seeds(25, 5)
    first = _paired_folder(runs["first"], tmp_path, "first", first_failing)
    second = _paired_folder(
        runs["first"],
        tmp_path,
        "second",
        second_failing,
        redrawn_seeds=(7,),
    )
    _set_limits(first, 0.1, 0.2)
    _set_limits(second, 0.02, 0.09)

    lines, row = _paired_diff(capsys, tmp_path, first, second)

    paired_line = _line_with(lines, " not paired: ")
    rate_line = _line_with(lines, " logical_error_rate_estimate: ")
    assert row["digest_mismatches"] == "1"
    assert row["scored_pairs"] == ""
    assert row["is_mixture_difference"] == ""
    assert paired_line.endswith(
        " not paired: 1 of 200 shared shots hold a different "
        "sample_digest, so the two points did not decode the same shots"
    )
    assert rate_line.endswith(", beyond error bars")


def test_a_shot_either_run_left_unscored_is_counted_and_left_out(
    runs, tmp_path, capsys
):
    """Seed 30 fails only in the second; unscored in the first, no pair."""
    first_failing, second_failing = _failing_seeds(25, 5)
    first = _paired_folder(
        runs["first"],
        tmp_path,
        "first",
        first_failing,
        unscored_seeds=(30,),
    )
    second = _paired_folder(runs["first"], tmp_path, "second", second_failing)

    lines, row = _paired_diff(capsys, tmp_path, first, second)

    paired_line = _line_with(lines, " paired on ")
    assert row["unscored_shots"] == "1"
    assert row["scored_pairs"] == "199"
    assert row["second_only_failures"] == "4"
    assert (
        " paired on the 199 shots both runs scored of 200 shared shots "
        "(1 unscored in either run, left out): "
    ) in paired_line
    assert "; on these pairs, not the whole runs, the failure rate" in (
        paired_line
    )


def test_the_pairs_end_where_the_shorter_prefix_stopped(runs, tmp_path, capsys):
    """The second stopped at its 150th shot; its later shots are no pairs.

    A folder holds shots past its stop when a piece ran on beyond it, so
    the first's failures at seeds 160 to 169 find shots in the second,
    and still pair with nothing.
    """
    first_failing, second_failing = _failing_seeds(25, 5)
    late_failures = tuple(range(160, 170))
    first_and_late = first_failing + late_failures
    first = _paired_folder(runs["first"], tmp_path, "first", first_and_late)
    second = _paired_folder(runs["first"], tmp_path, "second", second_failing)
    _set_prefix_shots(second, 150)

    _lines, row = _paired_diff(capsys, tmp_path, first, second)

    assert row["shared_shots"] == "150"
    assert row["first_only_failures"] == "25"


def test_equal_rates_still_print_the_paired_difference(runs, tmp_path, capsys):
    """25 of 100 against 50 of 200: one rate, yet 25 against 5 paired.

    The second's 5 failures among the 100 shared seeds miss the first's
    25, and its other 45 come after the first stopped, so the two rates
    print alike and only the paired line shows the difference.
    """
    first_failing = tuple(range(5, 30))
    second_shared = tuple(range(30, 35))
    second_later = tuple(range(100, 145))
    second_failing = second_shared + second_later
    first = _paired_folder(runs["first"], tmp_path, "first", first_failing)
    second = _paired_folder(runs["first"], tmp_path, "second", second_failing)
    _set_prefix_shots(first, 100)
    _set_rate(first, 0.25)

    lines, row = _paired_diff(capsys, tmp_path, first, second)

    paired_line = _line_with(lines, " paired on ")
    rate_lines = [line for line in lines if "logical_error_rate_est" in line]
    assert rate_lines == []
    assert row["first_only_failures"] == "25"
    assert row["second_only_failures"] == "5"
    assert "; differ by the mixture test;" in paired_line


def _failing_seeds(first_only: int, second_only: int) -> tuple:
    """Both fail the shared seeds, then the first its own, then the second."""
    first_start = len(SHARED_FAILING_SEEDS)
    second_start = first_start + first_only
    second_end = second_start + second_only
    first_own = range(first_start, second_start)
    second_own = range(second_start, second_end)
    first_failing = SHARED_FAILING_SEEDS + tuple(first_own)
    second_failing = SHARED_FAILING_SEEDS + tuple(second_own)
    return first_failing, second_failing


def _paired_folder(
    run_dir,
    tmp_path,
    name: str,
    failing_seeds: tuple,
    unscored_seeds: tuple = (),
    redrawn_seeds: tuple = (),
):
    """A copy whose first point holds PAIRED_SHOTS shots written here.

    Each is the point's first shot but for its seed, its failure, its
    scoring and its sample_digest, which names the seed and whether it
    was drawn again. The sweep row's prefix and rate follow the shots.
    """
    destination = tmp_path / name
    shutil.copytree(run_dir, destination)
    sweep_path = destination / "sweep.csv"
    sweep_rows = _csv_rows(sweep_path)
    point_id = sweep_rows[0]["point_id"]
    shots_path = destination / "shots.csv"
    shot_rows = _csv_rows(shots_path)
    point_rows = [row for row in shot_rows if row["point_id"] == point_id]
    other_rows = [row for row in shot_rows if row["point_id"] != point_id]
    template = point_rows[0]
    written = [
        _shot_row(template, seed, failing_seeds, unscored_seeds, redrawn_seeds)
        for seed in range(PAIRED_SHOTS)
    ]
    every_row = written + other_rows
    _write_csv_rows(shots_path, every_row)
    sweep_rows[0]["prefix_shots"] = PAIRED_SHOTS
    sweep_rows[0]["logical_error_rate_estimate"] = (
        len(failing_seeds) / PAIRED_SHOTS
    )
    _write_csv_rows(sweep_path, sweep_rows)
    return destination


def _set_limits(run_dir, low: float, high: float) -> None:
    """The first sweep row's exact interval set by hand."""
    sweep_path = run_dir / "sweep.csv"
    sweep_rows = _csv_rows(sweep_path)
    sweep_rows[0]["logical_error_rate_low"] = low
    sweep_rows[0]["logical_error_rate_high"] = high
    _write_csv_rows(sweep_path, sweep_rows)


def _set_prefix_shots(run_dir, prefix_shots: int) -> None:
    """The first sweep row's prefix stopped by hand at prefix_shots."""
    sweep_path = run_dir / "sweep.csv"
    sweep_rows = _csv_rows(sweep_path)
    sweep_rows[0]["prefix_shots"] = prefix_shots
    _write_csv_rows(sweep_path, sweep_rows)


def _set_rate(run_dir, rate: float) -> None:
    """The first sweep row's shot failure rate set by hand."""
    sweep_path = run_dir / "sweep.csv"
    sweep_rows = _csv_rows(sweep_path)
    sweep_rows[0]["logical_error_rate_estimate"] = rate
    _write_csv_rows(sweep_path, sweep_rows)


def _shot_row(
    template: dict,
    seed: int,
    failing_seeds: tuple,
    unscored_seeds: tuple,
    redrawn_seeds: tuple,
) -> dict:
    row = dict(template)
    row["seed"] = seed
    row["logical_failure"] = seed in failing_seeds
    row["is_scored"] = seed not in unscored_seeds
    row["sample_digest"] = f"seed {seed} drawn again {seed in redrawn_seeds}"
    return row


def _paired_diff(capsys, tmp_path, first, second) -> tuple:
    """The lines diff prints of the point written here, and its csv row."""
    out = tmp_path / "paired.csv"
    command.main(["diff", str(first), str(second), "--out", str(out)])
    printed = capsys.readouterr()
    lines = printed.out.splitlines()
    sweep_path = first / "sweep.csv"
    sweep_rows = _csv_rows(sweep_path)
    point_id = sweep_rows[0]["point_id"]
    probability = float(sweep_rows[0][PROBABILITY_AXIS])
    point = _point_text(probability)
    point_lines = [line for line in lines if point in line]
    rows = _csv_rows(out)
    (row,) = [row for row in rows if row["first_point_id"] == point_id]
    return point_lines, row


def _equation_24_difference(first_failing: tuple, second_failing: tuple):
    """Howard et al. eq. (24) at the last of PAIRED_SHOTS pairs.

    On y = (fail_first - fail_second + 1) / 2 in [0, 1]: each value's
    prediction is the mean of the values before it, zero for the first;
    V is the squared misses summed, floored at one; the radius is
    [1.7 sqrt(V (log log 2V + 3.8)) + 3.4 log log 2V + 13] / t; and an
    end y maps back to the difference as 2y - 1.
    """
    seeds = numpy.arange(PAIRED_SHOTS)
    first = numpy.isin(seeds, first_failing)
    second = numpy.isin(seeds, second_failing)
    differences = first.astype(float) - second
    values = (differences + 1) / 2
    positions = seeds + 1
    means = numpy.cumsum(values) / positions
    predictions = numpy.concatenate(([0.0], means[:-1]))
    squared_misses = (values - predictions) ** 2
    miss_sum = numpy.sum(squared_misses)
    variance = max(miss_sum, 1.0)
    doubled_variance = 2 * variance
    log_doubled = math.log(doubled_variance)
    iterated_log = math.log(log_doubled)
    root_argument = variance * (iterated_log + 3.8)
    root = math.sqrt(root_argument)
    radius = (1.7 * root + 3.4 * iterated_log + 13) / PAIRED_SHOTS
    low = 2 * (means[-1] - radius) - 1
    high = 2 * (means[-1] + radius) - 1
    return low, high


def test_load_gives_a_row_per_point_with_its_results_and_settings(runs):
    rows = results.load(runs["first"], runs["shorter"])

    rounds = [row[ROUNDS_COLUMN] for row in rows]
    assert len(rows) == 4
    assert rounds == [15, 15, 12, 12]
    assert rows[0]["run_dir"] == str(runs["first"])
    assert rows[0]["shots"] == 20


def test_a_loaded_row_holds_what_an_error_rate_figure_is_drawn_from(runs):
    """The estimate, its exact limits and counts, and the point's values.

    decsim draws no error rate figure; a reader draws one from these
    rows, the numbers sinter's plot_error_rate reads off its csv. The
    point stopped at its shot cap, so its limits are Clopper and
    Pearson's, which scipy's exact binomial interval is.
    """
    rows = results.load(runs["first"])

    noisier = rows[1]
    failures = noisier["prefix_failures"]
    scored_shots = noisier["prefix_scored_shots"]
    test = scipy.stats.binomtest(failures, scored_shots)
    interval = test.proportion_ci(method="exact")
    assert noisier["state"] == "cap"
    assert failures > 0
    assert noisier["logical_error_rate_estimate"] == failures / scored_shots
    assert noisier["logical_error_rate_low"] == pytest.approx(
        interval.low, rel=1e-12
    )
    assert noisier["logical_error_rate_high"] == pytest.approx(
        interval.high, rel=1e-12
    )
    assert noisier[PROBABILITY_COLUMN] == 0.01
