"""The experiments layer's referents: sinter's collect, and a recorded sweep.

Referent one is a sweep of reference.yaml recorded before decsim.collect
existed, its sweep.csv and links.csv kept in data/. Ten of its numbers
were amended when the controller-to-store hop stopped being priced at
the raw readout width: the recorded run's own buffer fill, its
per-syndrome buffer hop and that hop's link row. Eight zero columns were
added to it when the escalation points were added, the weak attempt
that did not commit and the escalation hop, which a weak-only sweep
never reaches, and eight more when the two per-round back-pressure
points were added, the wait a full the weak syndrome buffer or a full the
strong syndrome buffer puts on the writer, which a sweep with room in both
stores never sees, and four more when the park was split by cause: this sweep's
one unit is always free by the time a window's boundary is in, so its
compute_wait is zero on every window of both recorded runs. Four more came with
the confidence step, which a weak-baseline sweep answers on its decode and
never spends. Its first four columns became the point's metadata and
algorithm when a point came to be named by its id: the id hashes the
tree's settings records, not a measured value, so it is left out of the
comparison. Five zero columns came with the window decode statuses,
one per status besides success, which no window of the recorded sweep
carried, and three columns with the unscored shots, both its shots
scored, none unscored and a zero rate counting them as failures, and
one zero count of replaced provisional decodes
without a correction, which a weak-only sweep never makes. The weak
decoder of
reference.yaml is pymatching, which prices its measured wall clock, so the
columns that carry decode time (algorithm, service, queue wait, the park
before the compute, the four totals, load, throughput, the queue peak and
the wall seconds) vary between two runs of the same code; they are left
out of the comparison, and the columns kept are exactly the ones two
recorded runs agreed on. Its two columns of the per-shot whole-circuit
PyMatching reference went when that reference did. Referent two
is sinter (sinter/_collection/_collection.py collect, sinter/_data/_task.py
strong_id): a task named twice runs once, and a decoder off the table is
refused by name (sinter/_decoding/_decoding.py "Unrecognized decoder"),
and a shot its decoder could not answer is a discard, counted apart and
never an error (sinter/_decoding/_decoding.py:123-125). Referent three
is that retired reference's own columns, frozen by seed before it went,
which a full-history point beside a sliding one reproduces.
"""

import csv
import dataclasses
import decimal
import fractions
import json
import pathlib
import re
import shutil

import numpy
import pytest
import stim
import yaml

import decsim.collect as collect
import decsim.controller.settings as controller_settings
import decsim.decoders.relay_belief_propagation.window_decoder as relay_window
import decsim.decoders.settings as decoder_settings
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.experiments.collect_command as run
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure_shot
import decsim.experiments.report as sweep_report
import decsim.frontends.settings as workload_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.windows.built_window_models as built_window_models
import tests.experiments.yaml_configs as yaml_configs

THIS_FILE = pathlib.Path(__file__)
DATA = THIS_FILE.parent / "data"
CONFIGS = THIS_FILE.parents[2] / "configs"
REFERENCE_YAML = CONFIGS / "reference.yaml"
# The sweep.csv columns that carry the decoder's measured wall clock, and
# the release stage that starts from it and waits for a clock edge.
WALL_CLOCK_POINTS = (
    "algorithm",
    "release",
    "dep_block",
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
    "weak_queue_max",
    "strong_queue_max",
    "weak_busy_fraction",
    "strong_busy_fraction",
    "strong_service_mean_us",
    "parallel_processes_needed",
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


def _without_point_id(rows: list) -> list:
    kept = []
    for row in rows:
        without = dict(row)
        del without["point_id"]
        kept.append(without)
    return kept


def test_reference_yaml_rows_equal_the_recorded_sweep_and_links(tmp_path):
    config = experiment.load_experiment(REFERENCE_YAML)
    tasks = config.tasks()
    measurements = run.run_sweep(tasks, 2)
    record = sweep_report.record_of(measurements)
    summary_rows = sweep_report.summarize(record.shots, record.window_samples)
    link_rows = sweep_report.link_rows(record.shot_links)
    sweep_now = _written_rows(summary_rows, tmp_path, "sweep.csv")
    links_now = _written_rows(link_rows, tmp_path, "links.csv")
    sweep_path = DATA / "reference_sweep.csv"
    links_path = DATA / "reference_links.csv"
    sweep_before = _csv_rows(sweep_path)
    links_before = _csv_rows(links_path)
    sweep_measured = _without_point_id(sweep_now)
    links_measured = _without_point_id(links_now)
    assert len(sweep_measured) == 1
    stable_now = _stable_columns(sweep_measured[0])
    assert stable_now == _stable_columns(sweep_before[0])
    assert links_measured == links_before


# the sliding pymatching point's logical_failure, direct_failure and
# direct_mismatch for seeds 0 to 239, written at commit 9a4334b2, the
# last commit that measured the whole-circuit reference on every shot
DIRECT_REFERENCE = DATA / "direct_reference.csv"
FULL_HISTORY_PAIR = {
    "weak_decoder": {
        **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
        "kind": "pymatching",
    },
    "sweep": [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.015],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
                "windows.commit_rounds": [None, 16],
            },
            "collection": {"max_shots": 240},
        }
    ],
}


def test_a_full_history_point_reproduces_the_retired_direct_columns(
    tmp_path,
):
    """The per-shot whole-circuit reference, from a pair of points by seed.

    One window of 16 rounds decodes the shot's whole 15-round history,
    which is whole-circuit PyMatching on the same events, so seed by
    seed its failure is the retired direct_failure, and the sliding and
    full-history predictions differ exactly where direct_mismatch said
    (NOTE section 9 item 7). The frozen seeds hold two mismatches. The
    referent is one operation with one observable, so it shows the
    equivalence there only; with several observables or operations two
    failures can be two different answers, which is why the pair is
    compared by predictions.
    """
    config_path = yaml_configs.write_config(tmp_path, FULL_HISTORY_PAIR)
    out_dir = tmp_path / "out"

    run_dir, _rows = run.run_experiment(config_path, out_dir)

    shots_path = run_dir / "shots.csv"
    shots = _csv_rows(shots_path)
    sliding = _shots_by_seed(shots, "null")
    full_history = _shots_by_seed(shots, "16")
    reference = _csv_rows(DIRECT_REFERENCE)
    seeds = [int(row["seed"]) for row in reference]
    expected = [_frozen_columns(row) for row in reference]
    reproduced = [
        _retired_columns(sliding[seed], full_history[seed]) for seed in seeds
    ]
    sliding_digests = [sliding[seed]["sample_digest"] for seed in seeds]
    full_digests = [full_history[seed]["sample_digest"] for seed in seeds]
    mismatches = [columns[2] for columns in expected]
    assert reproduced == expected
    assert sliding_digests == full_digests
    assert sum(mismatches) == 2


def _columns_of_every_csv(run_dir) -> set:
    """Every column name any csv file under the folder holds."""
    columns = set()
    for path in run_dir.rglob("*.csv"):
        header = fold.header_of(path)
        columns.update(header)
    return columns


def test_a_swept_section_keeps_its_column_beside_every_measured_one(
    tmp_path,
):
    """A whole yaml section is an axis, and no measured column shares it.

    A row takes its swept values first and its measured values after
    (report._with_swept_values), so a measured column named as a section
    would overwrite the swept value in its cell. The swept windows cell
    is the section's compact json (run_folder.swept_values), and no csv
    column the run wrote is named as another yaml section.
    """
    windows = yaml_configs.MINIMAL_CONFIG["windows"]
    block = dict(yaml_configs.MINIMAL_CONFIG["sweep"][0])
    block["axes"] = dict(block["axes"], windows=[windows])
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block]})
    run_dir, _ = run.run_experiment(config_path)
    shots_path = run_dir / "shots.csv"
    shot, *_ = _csv_rows(shots_path)
    columns = _columns_of_every_csv(run_dir)
    sections_written = columns & set(machine_settings.SECTIONS)

    assert json.loads(shot["windows"]) == windows
    assert int(shot["decoded_windows"]) > 0
    assert sections_written == {"windows"}


def test_a_task_named_by_two_blocks_runs_once(tmp_path):
    raw = _reference_yaml()
    raw["sweep"] = [raw["sweep"][0], dict(raw["sweep"][0])]
    twice_path = tmp_path / "twice.yaml"
    twice = _written_yaml(raw, twice_path)
    config = experiment.load_experiment(twice)
    tasks = config.tasks()
    assert len(tasks) == 2
    seeds = []

    def record_seed(shot: collect.Shot) -> int:
        seeds.append(shot.seed)
        return shot.seed

    collect.collect(tasks, 2, record_seed)
    assert seeds == [0, 1]


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    weak_decoder = decoder_settings.DecoderSettings(kind="lookup_table")
    settings = dataclasses.replace(task.settings, weak_decoder=weak_decoder)
    unknown = dataclasses.replace(task, settings=settings)
    rows = sorted(decoder_settings.DECODERS)
    sentence = (
        "weak_decoder.kind 'lookup_table' is not a row of its table; "
        "the rows are " + re.escape(repr(rows))
    )
    with pytest.raises(ValueError, match=sentence):
        collect.collect([unknown], 1, measure_shot.measure_shot)


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
    config = experiment.load_experiment(online)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    assert task.online_threshold is not None
    shot_settings = task.shot_settings()
    escalation = shot_settings.escalation
    assert escalation.online_threshold is task.online_threshold


def _decoded(shot: collect.Shot) -> tuple:
    """What one shot drew, and what the loop and the whole circuit decoded.

    The whole-circuit PyMatching decode is built here, apart from the
    machine's models, so models the task shares cannot move it.
    """
    measurement = measure_shot.measure_shot(shot)
    return (
        shot.seed,
        measurement.decoded_windows,
        _drawn(shot),
        yaml_configs.loop_predictions(shot),
        yaml_configs.whole_circuit_predictions(shot),
    )


def _drawn(shot: collect.Shot) -> list:
    """Each operation's sampled detection events and observable truth."""
    sampled = shot.machine.observation.sampled_shots.shots_by_operation
    drawn = []
    for operation_result in shot.result.operation_results:
        sampled_shot = sampled[operation_result.operation_id]
        events = tuple(sampled_shot.detection_events)
        truth = tuple(operation_result.observable_truth)
        drawn.append((events, truth))
    return drawn


def _decoded_alone(task, seed: int) -> tuple:
    """One shot of the task, run on its own models and decoded."""
    shot = collect.run_shot(task, seed)
    return _decoded(shot)


def _run_every_shot(task, shots: int, built_models) -> None:
    for seed in range(shots):
        collect.run_shot(task, seed, built_models=built_models)


def test_a_tasks_shots_decode_the_same_with_the_models_built_once():
    """The task's own referent: sinter compiles once per task."""
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )

    whole_task = collect.Unit(task, 0, 4)
    shared, _ran = collect.run_unit(whole_task, _decoded)
    alone = [_decoded_alone(task, seed) for seed in range(4)]

    assert shared == alone


def test_the_first_shot_builds_the_models_and_the_rest_read_them():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    built = built_window_models.BuiltWindowModels()

    _run_every_shot(task, 3, built)

    assert built.builds == 1
    assert built.reuses == 2


def test_a_machine_built_alone_builds_its_own_models():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = task.shot_settings()

    assert settings.workload.built_models is None


def test_a_task_is_one_unit_of_all_its_seeds():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )

    units = collect.work_units([task], 5)

    assert units == [collect.Unit(task, 0, 5)]


def test_a_point_with_an_online_threshold_is_one_piece(tmp_path):
    """The calibrator learns over the point's shots in order.

    Pieces of one round would cut any other point into one piece a shot.
    """
    raw = _reference_yaml()
    raw["escalation"] = {
        "kind": "switching",
        "gap_threshold_db": 15.0,
        "threshold_source": "online",
    }
    raw["collection"] = {"piece_rounds": 1, "max_shots": 2}
    online_path = tmp_path / "online.yaml"
    online = _written_yaml(raw, online_path)
    out_dir = tmp_path / "out"

    run.run_experiment(online, out_dir)

    (piece_folder,) = out_dir.glob("pieces/*/*")
    assert piece_folder.name == "0-1"


def test_two_points_under_one_cache_do_not_share_models():
    """The key carries the circuit text, which is where d and p are.

    Two distances also differ in their window spans, so the noisier
    point at one distance is what pins the circuit text itself.
    """
    config = experiment.load_experiment(REFERENCE_YAML)
    built = built_window_models.BuiltWindowModels()
    at_three = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    at_five = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 5,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    noisier_at_three = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )

    collect.run_shot(at_three, 0, built_models=built)
    collect.run_shot(at_five, 0, built_models=built)
    collect.run_shot(noisier_at_three, 0, built_models=built)

    assert built.builds == 3
    assert built.reuses == 0


def test_the_summary_off_the_written_files_is_the_summary_of_the_shots(
    tmp_path,
):
    """The precondition sinter meets: a folder's rows rebuild its rows.

    sinter/_data/_task_stats.py keeps only additive fields, sums them in
    __add__ and derives every rate from the summed row. A decsim run
    folder records the same way, so reading its files back and
    summarizing them returns the rows the run wrote.
    """
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    whole_task = collect.Unit(task, 0, 3)
    measurements, _ran = collect.run_unit(whole_task, measure_shot.measure_shot)
    record = sweep_report.record_of(measurements)
    swept = {task.strong_id(): task.metadata}
    sweep_report.write_record(record, tmp_path, swept)
    shots_path = tmp_path / "shots.csv"
    samples_path = tmp_path / "window_samples.csv"
    shot_links_path = tmp_path / "shot_links.csv"

    read_shots = sweep_report.read_rows(shots_path)
    read_samples = sweep_report.read_rows(samples_path)
    read_shot_links = sweep_report.read_rows(shot_links_path)

    measured_summary = sweep_report.summarize(
        record.shots, record.window_samples
    )
    read_summary = sweep_report.summarize(read_shots, read_samples)
    assert read_summary == measured_summary
    measured_links = sweep_report.link_rows(record.shot_links)
    read_links = sweep_report.link_rows(read_shot_links)
    assert read_links == measured_links


# ---- an escalation row written outside decsim


class _OutsideEscalation:
    """An escalation row written outside decsim, delegating to Baseline.

    It subclasses no shipped row: every fact the experiments layer and
    the machine read off a row is declared here and every call is
    forwarded, which is the shape the P8 plug-in probe used. What the
    experiments layer reads to measure a shot is primary_tier.
    """

    decides_on_a_confidence = False
    requires_strong_context = False
    primary_tier = window_records.DecoderTier.WEAK
    default_boundary_policy = "eager"

    def __init__(self, collaborators) -> None:
        self.delegate = escalation_policies.Baseline(collaborators)

    def check_plan(self, plan) -> None:
        self.delegate.check_plan(plan)

    def tiers_for_ready_window(self, window) -> tuple:
        return self.delegate.tiers_for_ready_window(window)

    def verdict_for_weak_result(self, job, result):
        return self.delegate.verdict_for_weak_result(job, result)

    def learn_from_strong_result(self, window_key, result) -> None:
        self.delegate.learn_from_strong_result(window_key, result)


def _measured_shot(tmp_path, escalation_kind: str):
    """One seeded shot of the minimal config under that escalation row."""
    card = {"escalation": {"kind": escalation_kind}}
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    return yaml_configs.measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )


def test_an_outside_escalation_row_is_measured_over_its_tiers_links(
    tmp_path, monkeypatch
):
    """The tier is read off the row, so a row off the table measures.

    The experiments layer reads the tier the row declares, never the
    escalation's name, so `decsim collect` measures any row the machine
    runs.
    """
    shipped_directory = tmp_path / "shipped"
    shipped_directory.mkdir()
    shipped = _measured_shot(shipped_directory, "weak_baseline")
    monkeypatch.setitem(
        escalation_settings.ESCALATIONS, "outside_baseline", _OutsideEscalation
    )
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()

    outside = _measured_shot(outside_directory, "outside_baseline")

    assert outside.samples == shipped.samples
    assert outside.means == shipped.means
    assert outside.logical_failure == shipped.logical_failure


def _memory_task(rounds: int) -> collect.Task:
    """A one-shot task running Stim's d=3 memory circuit of that length."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=rounds, distance=3
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    workload = workload_settings.WorkloadSettings(operations=(operation,))
    settings = machine_settings.MachineSettings(workload=workload)
    return collect.Task(settings, {"point": 1})


def test_two_tasks_that_run_different_circuits_are_two_tasks():
    """The strong id carries the circuit text, as sinter's does."""
    three_rounds = _memory_task(3)
    five_rounds = _memory_task(5)

    unique = collect.unique_tasks([three_rounds, five_rounds])

    assert len(unique) == 2


def test_two_tasks_whose_controllers_stall_or_drop_are_two_tasks():
    """An enum setting enters the strong id as its member's name."""
    settings = machine_settings.MachineSettings()
    drop_round = controller_settings.PackingOverflowPolicy.DROP_ROUND
    dropping_controller = dataclasses.replace(
        settings.controller, packing_overflow=drop_round
    )
    dropping = dataclasses.replace(settings, controller=dropping_controller)
    stalling_task = collect.Task(settings, {"point": 1})
    dropping_task = collect.Task(dropping, {"point": 1})

    unique = collect.unique_tasks([stalling_task, dropping_task])

    assert len(unique) == 2


def test_two_tasks_whose_round_policies_differ_in_count_are_two_tasks():
    """A round policy enters the strong id by its arguments."""
    three_rounds = round_policies.FixedRounds(3)
    five_rounds = round_policies.FixedRounds(5)
    three_workload = workload_settings.WorkloadSettings(
        rounds_policy=three_rounds
    )
    five_workload = workload_settings.WorkloadSettings(
        rounds_policy=five_rounds
    )
    three_settings = machine_settings.MachineSettings(workload=three_workload)
    five_settings = machine_settings.MachineSettings(workload=five_workload)
    three_task = collect.Task(three_settings, {"point": 1})
    five_task = collect.Task(five_settings, {"point": 1})

    unique = collect.unique_tasks([three_task, five_task])

    assert len(unique) == 2


class _SlottedRounds:
    """A user's round policy that keeps its count in __slots__."""

    __slots__ = ("round_count",)

    def __init__(self, round_count: int) -> None:
        self.round_count = round_count

    def rounds_for(self, operation, code) -> int:
        del operation, code
        return self.round_count


def _slotted_rounds_task(round_count: int) -> collect.Task:
    rounds_policy = _SlottedRounds(round_count)
    workload = workload_settings.WorkloadSettings(rounds_policy=rounds_policy)
    settings = machine_settings.MachineSettings(workload=workload)
    return collect.Task(settings, {"point": 1})


def test_two_slotted_round_policies_that_differ_in_count_are_two_tasks():
    """A component's __slots__ enter the strong id as its __dict__ does."""
    three_rounds = _slotted_rounds_task(3)
    six_rounds = _slotted_rounds_task(6)

    unique = collect.unique_tasks([three_rounds, six_rounds])

    assert len(unique) == 2


def _device_task(round_count: int) -> collect.Task:
    """A task whose settings hold a Python-built finite source."""
    device = stim_device.StimDevice(measurement_rounds={1: {0: round_count}})
    qpu = qpu_settings.QpuSettings(device=device)
    settings = machine_settings.MachineSettings(qpu=qpu)
    return collect.Task(settings, {"point": 1})


def test_two_python_built_sources_that_hold_different_values_are_two_tasks():
    """A component with no record form enters the id by its content."""
    first = _device_task(1)
    second = _device_task(2)

    unique = collect.unique_tasks([first, second])

    assert len(unique) == 2


def _recorded_device_task(flip: int) -> collect.Task:
    """A task whose settings hold a recorded shot, its last bit flip."""
    measurements = numpy.array([[0, flip]], dtype=bool)
    device = stim_device.RecordedStimDevice(measurements, 0)
    qpu = qpu_settings.QpuSettings(device=device)
    settings = machine_settings.MachineSettings(qpu=qpu)
    return collect.Task(settings, {"point": 1})


def test_two_recorded_sources_that_replay_different_shots_are_two_tasks():
    """An array a component holds enters the id by its values."""
    first = _recorded_device_task(0)
    second = _recorded_device_task(1)

    unique = collect.unique_tasks([first, second])

    assert len(unique) == 2


def test_two_python_built_sources_that_hold_the_same_values_are_one_task():
    first = _device_task(1)
    same = _device_task(1)

    unique = collect.unique_tasks([first, same])

    assert len(unique) == 1


def _bandwidth_tasks(tmp_path, widths: tuple) -> list:
    """data_movement_switching.yaml's task with its readout hop at each width.

    One file is edited in place between loads, so the tasks differ in
    the hop's rate and in nothing else, the path included.
    """
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    path = configs / "data_movement_switching.yaml"
    text = path.read_text()
    unpriced = "clock: fridge, bits_per_cycle: null}"
    readout_hop = f"qpu_to_controller:  {{latency_cycles: 1, {unpriced}"
    assert readout_hop in text
    tasks = []
    for bits_per_cycle in widths:
        priced = f"clock: fridge, bits_per_cycle: {bits_per_cycle}}}"
        priced_hop = readout_hop.replace(unpriced, priced)
        priced_text = text.replace(readout_hop, priced_hop)
        path.write_text(priced_text)
        config = experiment.load_experiment(path)
        task = config.point_task(
            {
                "workload.arguments.physical_error_probability": 0.001,
                "qpu.distance": 3,
                "qpu.round_period_microseconds": 1.0,
            },
        )
        tasks.append(task)
    return tasks


def _readout_rate(shot) -> str:
    """The readout hop's rate, the one setting two bandwidth tasks differ in."""
    capacity = shot.task.settings.links.qpu_to_controller.channel.capacity
    return str(capacity.input_bits_per_microsecond)


def test_two_tasks_that_differ_only_in_bandwidth_are_two_tasks(tmp_path):
    """A yaml link rate is an exact Fraction and enters the id as its text.

    sinter's strong id is the sha256 of every value's json text
    (sinter/_data/_task.py strong_id_value), so two values give two ids.
    """
    narrow, wide = _bandwidth_tasks(tmp_path, (8, 64))

    unique = collect.unique_tasks([narrow, wide])
    rows = collect.collect([narrow, wide], 1, _readout_rate)

    assert narrow.strong_id() != wide.strong_id()
    assert len(unique) == 2
    assert rows == ["2000", "16000"]
    assert collect.json_value(narrow.settings) != collect.json_value(
        wide.settings
    )


@pytest.mark.parametrize(
    ("metadata", "sentence_start"),
    [
        ({1: "value"}, "metadata holds the key 1,"),
        ({"nested": {1: "value"}}, "metadata.nested holds the key 1,"),
        ({"listed": [{1: "value"}]}, "metadata.listed holds the key 1,"),
    ],
)
def test_a_metadata_key_that_is_not_text_is_refused_at_any_depth(
    metadata, sentence_start
):
    """A Python caller's metadata, where a yaml reader cannot refuse it."""
    config = experiment.load_experiment(REFERENCE_YAML)
    first_point = config.first_point_task()
    sentence = (
        f"{sentence_start} which is not text; a point's id is the json of "
        "its metadata, whose keys are text, so 1 and '1' would name one point"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        collect.Task(first_point.settings, metadata)


# Each number beside the json value it enters the strong id as.
EXACT_NUMBERS = [
    (fractions.Fraction(80, 11), "80/11"),
    (fractions.Fraction(16000), "16000"),
    (decimal.Decimal("1.10"), "1.10"),
    (numpy.int64(7), 7),
    (numpy.float32(0.5), 0.5),
    (numpy.bool_(True), True),
    (True, True),
    (7, 7),
    (0.001, 0.001),
]


@pytest.mark.parametrize(("value", "written"), EXACT_NUMBERS)
def test_a_number_enters_the_json_exactly(value, written):
    """A json number stays as written; any other number is its exact text."""
    json_number = collect.json_value(value)

    assert json_number == written
    assert type(json_number) is type(written)


def test_a_number_json_cannot_hold_reads_back_to_itself():
    """The written text is what Fraction and Decimal read back exactly."""
    rate = fractions.Fraction(80, 11)
    amount = decimal.Decimal("1.10")

    rate_text = collect.json_value(rate)
    amount_text = collect.json_value(amount)

    assert fractions.Fraction(rate_text) == rate
    assert decimal.Decimal(amount_text) == amount


class _CrashingRelayDecoder:
    """A relay-bp decoder whose every decode raises, a crashed backend."""

    def __init__(self, *arguments, **keywords) -> None:
        del arguments, keywords

    def decode_detailed(self, syndrome):
        del syndrome
        raise RuntimeError("the relay-bp backend crashed")


def _crashing_relay_type():
    return _CrashingRelayDecoder


def test_a_crashed_backend_leaves_unscored_shots_and_the_task_completes(
    tmp_path, monkeypatch
):
    """Every decode raises, and every shot is a row saying so.

    The crash is relay-bp's own decode_detailed raising, which the row
    turns into BACKEND_ERROR with no correction; each shot is unscored
    with that reason, counts its windows under backend_error_windows,
    and is not a failure; the point has no failure fraction and no
    interval, and counting its unscored shots as failures gives 1.
    """
    monkeypatch.setattr(
        relay_window, "_load_relay_decoder_type", _crashing_relay_type
    )
    weak_decoder = dict(
        yaml_configs.MINIMAL_CONFIG["weak_decoder"], kind="relay_bp"
    )
    sweep = [
        dict(
            yaml_configs.MINIMAL_CONFIG["sweep"][0], collection={"max_shots": 2}
        )
    ]
    config_path = yaml_configs.write_config(
        tmp_path, {"weak_decoder": weak_decoder, "sweep": sweep}
    )
    run_dir, rows = run.run_experiment(config_path)
    lines = sweep_report.terminal_lines(rows, run_dir)
    shots_path = run_dir / "shots.csv"
    first, second = _csv_rows(shots_path)

    assert [first["seed"], second["seed"]] == ["0", "1"]
    assert first["is_scored"] == "False"
    assert first["unscored_reason"] == "upstream_exception"
    assert first["backend_error_windows"] == first["decoded_windows"]
    assert first["logical_failure"] == "False"
    assert second["is_scored"] == "False"
    assert rows[0]["shots"] == 2
    assert rows[0]["scored_shots"] == 0
    assert rows[0]["unscored_shots"] == 2
    assert rows[0]["logical_failures"] == 0
    assert rows[0]["state"] == "cap"
    assert rows[0]["logical_error_rate_estimate"] is None
    assert rows[0]["logical_error_rate_low"] is None
    assert rows[0]["logical_error_rate_high"] is None
    assert rows[0]["logical_error_rate_unscored_as_failures"] == 1.0
    assert "logical failures: 0 of 0 scored shots" in lines
    assert "logical error rate among scored shots: none (cap)" in lines
    assert "unscored shots: 2 of 2 (1)" in lines


def _shots_by_seed(shots: list, commit_rounds_cell: str) -> dict:
    """One point's shots.csv rows by seed, the point named by its window."""
    by_seed = {}
    for row in shots:
        if row["windows.commit_rounds"] == commit_rounds_cell:
            by_seed[int(row["seed"])] = row
    return by_seed


def _frozen_columns(row: dict) -> tuple:
    """A frozen row's logical_failure, direct_failure, direct_mismatch."""
    logical_failure = int(row["logical_failure"])
    direct_failure = int(row["direct_failure"])
    direct_mismatch = int(row["direct_mismatch"])
    return (logical_failure, direct_failure, direct_mismatch)


def _retired_columns(sliding_row: dict, full_history_row: dict) -> tuple:
    """The three columns as the pair of points gives them for one seed.

    The mismatch compares the two points' predictions, as the retired
    column compared the loop's with the reference's, not their failures.
    """
    sliding_failure = sliding_row["logical_failure"] == "True"
    full_history_failure = full_history_row["logical_failure"] == "True"
    is_mismatch = sliding_row["predictions"] != full_history_row["predictions"]
    return (int(sliding_failure), int(full_history_failure), int(is_mismatch))
