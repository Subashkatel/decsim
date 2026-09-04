"""The front against its referents: sinter's collect, and the old runner.

Referent one is the runner before decsim.collect existed (rewrite-sample
10ad11f): reference.yaml's sweep run through it, its sweep.csv and
links.csv kept in data/. The weak decoder of reference.yaml is
pymatching, which prices its measured wall clock, so the columns that
carry decode time (algorithm, service, queue wait, the four totals,
load, throughput, the queue peak and the wall seconds) vary between two
runs of the same code; they are left out of the comparison, and the
columns kept are exactly the ones two runs of the old runner agreed on.
Referent two is sinter (sinter/_collection/_collection.py collect,
sinter/_data/_task.py strong_id): a task named twice runs once, and a
decoder off the table is refused by name (sinter/_decoding/_decoding.py
"Unrecognized decoder").
"""

import csv
import dataclasses
import pathlib

import pytest
import yaml

import decsim.collect as collect
import decsim.decoders.settings as decoder_settings
import experiments.experiment_config as experiment_config
import experiments.measure_shot as measure_shot
import experiments.run as run
import experiments.sweep_report as sweep_report

THIS_FILE = pathlib.Path(__file__)
DATA = THIS_FILE.parent / "data"
CONFIGS = THIS_FILE.parents[2] / "experiments" / "configs"
REFERENCE_YAML = CONFIGS / "reference.yaml"
# The sweep.csv columns that carry the decoder's measured wall clock.
WALL_CLOCK_POINTS = (
    "algorithm",
    "queue_wait",
    "service",
    "buffer0_ready_to_frame",
    "buffer0_first_round_to_frame",
    "qpu_last_round_to_frame",
    "qpu_first_round_to_frame",
)
WALL_CLOCK_COLUMNS = (
    "load",
    "max_queued_windows",
    "sim_wall_seconds_per_shot",
    "throughput_rounds_per_us",
    "throughput_windows_per_us",
)


def _is_wall_clock_column(column: str) -> bool:
    if column in WALL_CLOCK_COLUMNS:
        return True
    for point in WALL_CLOCK_POINTS:
        if column.startswith(f"{point}_"):
            return True
    return False


def _csv_rows(path: pathlib.Path) -> list:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _written_rows(rows: list, tmp_path: pathlib.Path, name: str) -> list:
    """The rows as the runner writes them, read back as csv text."""
    path = tmp_path / name
    sweep_report.write_csv(rows, path)
    return _csv_rows(path)


def _reference_yaml() -> dict:
    text = REFERENCE_YAML.read_text()
    return yaml.safe_load(text)


def _written_yaml(raw: dict, path: pathlib.Path) -> pathlib.Path:
    text = yaml.safe_dump(raw)
    path.write_text(text)
    return path


def _stable_columns(row: dict) -> dict:
    stable = {}
    for column, value in row.items():
        if not _is_wall_clock_column(column):
            stable[column] = value
    return stable


def test_reference_yaml_rows_equal_the_old_runners_sweep_and_links(tmp_path):
    config = experiment_config.load_experiment(REFERENCE_YAML)
    measurements = run.run_sweep(config, None)
    summary_rows = sweep_report.summarize(measurements)
    link_rows = sweep_report.link_rows(measurements)
    sweep_now = _written_rows(summary_rows, tmp_path, "sweep.csv")
    links_now = _written_rows(link_rows, tmp_path, "links.csv")
    sweep_path = DATA / "reference_sweep.csv"
    links_path = DATA / "reference_links.csv"
    sweep_before = _csv_rows(sweep_path)
    links_before = _csv_rows(links_path)
    assert len(sweep_now) == 1
    assert _stable_columns(sweep_now[0]) == _stable_columns(sweep_before[0])
    assert links_now == links_before


def test_a_task_named_by_two_blocks_runs_once(tmp_path):
    raw = _reference_yaml()
    raw["sweep"] = [raw["sweep"][0], dict(raw["sweep"][0])]
    twice_path = tmp_path / "twice.yaml"
    twice = _written_yaml(raw, twice_path)
    config = experiment_config.load_experiment(twice)
    tasks = config.tasks()
    assert len(tasks) == 2
    seeds = []

    def record_seed(shot: collect.Shot) -> int:
        seeds.append(shot.seed)
        return shot.seed

    collect.collect(tasks, record_seed)
    assert seeds == [0, 1]


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    config = experiment_config.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        physical_error_probability=0.001,
        distance=3,
        round_period_us=1.0,
        shots=1,
    )
    weak_decoder = decoder_settings.DecoderSettings(kind="lookup_table")
    settings = dataclasses.replace(task.settings, weak_decoder=weak_decoder)
    unknown = dataclasses.replace(task, settings=settings)
    with pytest.raises(
        ValueError,
        match="weak_decoder.kind 'lookup_table' is not a row of its table; "
        r"the rows are \['belief_matching', 'bposd', 'pymatching', "
        r"'relay_bp', 'tesseract', 'union_find', 'unweighted_pymatching'\]",
    ):
        collect.collect([unknown], measure_shot.measure_shot)


def test_every_shot_of_a_point_shares_the_tasks_calibrator(tmp_path):
    """threshold_source online: one calibrator per point, on the task."""
    raw = _reference_yaml()
    raw["escalation"] = {
        "kind": "switching",
        "gap_threshold_db": 15.0,
        "threshold_source": "online",
    }
    online_path = tmp_path / "online.yaml"
    online = _written_yaml(raw, online_path)
    config = experiment_config.load_experiment(online)
    task = config.point_task(
        physical_error_probability=0.001,
        distance=3,
        round_period_us=1.0,
        shots=2,
    )
    assert task.threshold_calibrator is not None
    shot_settings = task.shot_settings()
    escalation = shot_settings.escalation
    assert escalation.threshold_calibrator is task.threshold_calibrator
