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
comparison. The weak
decoder of
reference.yaml is pymatching, which prices its measured wall clock, so the
columns that carry decode time (algorithm, service, queue wait, the park
before the compute, the four totals, load, throughput, the queue peak and
the wall seconds) vary between two runs of the same code; they are left
out of the comparison, and the columns kept are exactly the ones two
recorded runs agreed on. Referent two
is sinter (sinter/_collection/_collection.py collect, sinter/_data/_task.py
strong_id): a task named twice runs once, and a decoder off the table is
refused by name (sinter/_decoding/_decoding.py "Unrecognized decoder").
"""

import csv
import dataclasses
import decimal
import fractions
import pathlib
import re
import shutil

import numpy
import pytest
import stim
import yaml

import decsim.collect as collect
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.experiments.collect_command as run
import decsim.experiments.experiment as experiment
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
    measurements = run.run_sweep(tasks, None)
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

    collect.collect(tasks, record_seed)
    assert seeds == [0, 1]


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        1,
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
    config = experiment.load_experiment(online)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        2,
    )
    assert task.online_threshold is not None
    shot_settings = task.shot_settings()
    escalation = shot_settings.escalation
    assert escalation.online_threshold is task.online_threshold


def _predictions_of(rows) -> list:
    """What each shot decoded, the fields no timing knob can move."""
    decoded = []
    for row in rows:
        decoded.append(
            (
                row.seed,
                row.windows,
                row.logical_failure,
                row.direct_failure,
                row.direct_mismatch,
            )
        )
    return decoded


def _measured_alone(task, seed: int):
    """One shot of the task, run and measured on its own."""
    shot = collect.run_shot(task, seed)
    return measure_shot.measure_shot(shot)


def _run_every_shot(task, built_models) -> None:
    for seed in range(task.shots):
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
        4,
    )

    whole_task = collect.Unit(task, 0, task.shots)
    shared, _ran = collect.run_unit(whole_task, measure_shot.measure_shot)
    alone = [_measured_alone(task, seed) for seed in range(task.shots)]

    assert _predictions_of(shared) == _predictions_of(alone)


def test_the_first_shot_builds_the_models_and_the_rest_read_them():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        3,
    )
    built = built_window_models.BuiltWindowModels()

    _run_every_shot(task, built)

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
        1,
    )
    settings = task.shot_settings()

    assert settings.workload.built_models is None


def test_a_task_is_one_unit_until_a_unit_size_splits_it():
    config = experiment.load_experiment(REFERENCE_YAML)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        5,
    )

    whole = collect.work_units([task])
    split = collect.work_units([task], 2)

    assert whole == [collect.Unit(task, 0, 5)]
    assert split == [
        collect.Unit(task, 0, 2),
        collect.Unit(task, 2, 2),
        collect.Unit(task, 4, 1),
    ]


def test_a_point_with_an_online_threshold_stays_one_unit(tmp_path):
    """The calibrator learns over the point's shots in order."""
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
        5,
    )

    units = collect.work_units([task], 1)

    assert task.online_threshold is not None
    assert units == [collect.Unit(task, 0, 5)]


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
        1,
    )
    at_five = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 5,
            "qpu.round_period_microseconds": 1.0,
        },
        1,
    )
    noisier_at_three = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        1,
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
        3,
    )
    whole_task = collect.Unit(task, 0, task.shots)
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
    return collect.Task(settings, 1, {"point": 1})


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
    stalling_task = collect.Task(settings, 1, {"point": 1})
    dropping_task = collect.Task(dropping, 1, {"point": 1})

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
    three_task = collect.Task(three_settings, 1, {"point": 1})
    five_task = collect.Task(five_settings, 1, {"point": 1})

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
    return collect.Task(settings, 1, {"point": 1})


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
    return collect.Task(settings, 1, {"point": 1})


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
    return collect.Task(settings, 1, {"point": 1})


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
            1,
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
    rows = collect.collect([narrow, wide], _readout_rate)

    assert narrow.strong_id() != wide.strong_id()
    assert len(unique) == 2
    assert rows == ["2000", "16000"]
    assert collect.json_value(narrow.settings) != collect.json_value(
        wide.settings
    )


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
