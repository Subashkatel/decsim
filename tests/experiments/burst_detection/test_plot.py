"""The burst detection plot script against small rows worked by hand."""

import importlib.util
import itertools
import pathlib
import sys

import pytest

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT_FOLDER = REPOSITORY_ROOT / "experiments" / "burst_detection"
EXPECTED_POINT_PLOTS = {
    "caught_share_d5_p0.003.png",
    "median_delay_d5_p0.003.png",
    "catch_against_false_alarms_d5_p0.003.png",
    "false_alarm_check_d5_p0.003.png",
    "traces_1_d5_p0.003.png",
    "traces_0.1_d5_p0.003.png",
    "traces_0.03_d5_p0.003.png",
}
EXPECTED_CROSS_PLOTS = {
    "d5_p0.003",
    "false_alarm_check.png",
    "caught_share_against_d_1.png",
    "caught_share_against_d_0.1.png",
    "caught_share_against_d_0.03.png",
    "caught_share_against_p_1.png",
    "caught_share_against_p_0.1.png",
    "caught_share_against_p_0.03.png",
    "median_delay_against_d_1.png",
    "median_delay_against_d_0.1.png",
    "median_delay_against_d_0.03.png",
}


def test_a_class_counts_catches_by_300_rounds_and_its_median_delay(
    monkeypatch,
):
    """Alarms 0, 300 and 301 rounds on, and one never: two in time."""
    plot = script_module(monkeypatch)
    trials = [
        trial_row(0, 1000),
        trial_row(1, 1300),
        trial_row(2, 1301),
        trial_row(3, None),
    ]

    (summary,) = plot.class_summaries(trials)

    assert summary["bursts"] == 4
    assert summary["caught"] == 2
    assert summary["caught_share"] == 0.5
    assert summary["median_delay"] == 300.0


def test_a_classs_catches_sit_at_its_lines_asked_rate(monkeypatch):
    """No quiet row is read, so a line that saw no quiet alarm still shows."""
    plot = script_module(monkeypatch)
    trials = [trial_row(0, 1000), trial_row(1, None)]
    (summary,) = plot.class_summaries(trials)

    (stat,) = plot.catch_stats([summary])

    assert stat.json_metadata["false alarms per s asked"] == 0.1
    assert stat.custom_counts == {"caught": 1, "bursts": 2}


def test_the_quiet_parts_sum_to_one_row_per_line(monkeypatch):
    plot = script_module(monkeypatch)
    parts = [
        {"d": 5, "p": 0.003, "part": 0, "alarm_line": 0.1},
        {"d": 5, "p": 0.003, "part": 1, "alarm_line": 0.1},
    ]
    parts[0] |= {"quiet_seconds": 25.0, "alarms": 2}
    parts[1] |= {"quiet_seconds": 25.5, "alarms": 3}

    (total,) = plot.quiet_totals(parts)

    assert total == {
        "d": 5,
        "p": 0.003,
        "alarm_line": 0.1,
        "quiet_seconds": 50.5,
        "alarms": 5,
    }


def test_each_rate_gets_a_folder_of_figures_beside_the_cross_ones(
    tmp_path, monkeypatch
):
    plot = script_module(monkeypatch)
    write_rows(tmp_path, plot.run)

    plot.main(tmp_path)

    plots_folder = tmp_path / "plots"
    point_folder = plots_folder / "d5_p0.003"
    written = {path.name for path in plots_folder.iterdir()}
    point_written = {path.name for path in point_folder.iterdir()}
    assert written == EXPECTED_CROSS_PLOTS
    assert point_written == EXPECTED_POINT_PLOTS


def trial_row(trial: int, first_alarm_b) -> dict:
    return {
        "d": 5,
        "p": 0.003,
        "size": "small",
        "strength": "weak",
        "trial": trial,
        "alarm_line": 0.1,
        "first_alarm_a": None,
        "first_alarm_b": first_alarm_b,
    }


def write_rows(folder: pathlib.Path, run) -> None:
    """Two trials a class and line, a quiet row a line, two trace rounds."""
    trials = ["d,p,size,strength,trial,alarm_line,first_alarm_a,first_alarm_b"]
    traces = ["d,p,size,strength,copy,round,alarm_line,score_over_level"]
    quiet = ["d,p,part,alarm_line,quiet_seconds,alarms"]
    classes = itertools.product(
        run.RADIUS_BY_SIZE, run.FIRING_MULTIPLE_BY_STRENGTH
    )
    for (size, strength), line in itertools.product(
        classes, run.FALSE_ALARMS_PER_SECOND
    ):
        trials.append(f"5,0.003,{size},{strength},0,{line},,1004")
        trials.append(f"5,0.003,{size},{strength},1,{line},1500,")
        traces.append(f"5,0.003,{size},{strength},a,1000,{line},0.2")
        traces.append(f"5,0.003,{size},{strength},b,1000,{line},1.3")
    for line in run.FALSE_ALARMS_PER_SECOND:
        expected_alarms = line * 100
        alarms = round(expected_alarms)
        quiet.append(f"5,0.003,0,{line},50.0,{alarms}")
        quiet.append(f"5,0.003,1,{line},50.0,0")
    write_lines(folder, "trials.csv", trials)
    write_lines(folder, "traces.csv", traces)
    write_lines(folder, "quiet.csv", quiet)


def write_lines(folder: pathlib.Path, name: str, lines: list) -> None:
    text = "\n".join(lines)
    path = folder / name
    path.write_text(f"{text}\n")


def script_module(monkeypatch: pytest.MonkeyPatch):
    """experiments/burst_detection/plot.py, its main not run.

    plot.py imports its run.py by the name run, which another
    experiment's test may have imported first, so that name is cleared.
    """
    monkeypatch.syspath_prepend(str(SCRIPT_FOLDER))
    monkeypatch.delitem(sys.modules, "run", raising=False)
    path = SCRIPT_FOLDER / "plot.py"
    spec = importlib.util.spec_from_file_location("burst_detection_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
