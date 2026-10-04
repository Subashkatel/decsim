"""The experiments layer's referents: sinter's collect, and a recorded sweep.

Referent one is a sweep of the reference machine recorded before
decsim.collect existed, rerun on its machine (run_files.REFERENCE), its
sweep.csv and its per-link means kept in data/, each mean
the one shot_links.csv gives a link over the point's shots. Ten of its
numbers
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
the reference machine is pymatching, which prices its measured wall clock,
so the
columns that carry decode time (algorithm, service, queue wait, the park
before the compute, the four totals, load, throughput, the queue peak and
the wall seconds) vary between two runs of the same code; they are left
out of the comparison, and the columns kept are exactly the ones two
recorded runs agreed on. Its two columns of the per-shot whole-circuit
PyMatching reference went when that reference did. Referent two
is sinter (sinter/_collection/_collection.py collect, sinter/_data/_task.py
strong_id): two tasks that run differently have two ids, a decoder
off the table is
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
import functools
import json
import pathlib
import re
import statistics
import time

import numpy
import pytest
import stim

import decsim.collect as collect
import decsim.decoders.relay_belief_propagation.window_decoder as relay_window
import decsim.experiments.collect_command as run
import decsim.experiments.measure as measure_shot
import decsim.experiments.report as sweep_report
import decsim.experiments.run_folder as run_folder
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.settings as machine_settings
import decsim.windows.built_window_models as built_window_models
import tests.declared_run as declared_run
import tests.experiments.run_files as run_files

THIS_FILE = pathlib.Path(__file__)
DATA = THIS_FILE.parent / "data"
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
    "strong_service_sum_us",
    "strong_service_mean_us",
    "parallel_processes_needed",
    "sim_wall_seconds_per_shot",
    "throughput_rounds_per_us",
    "throughput_windows_per_us",
)

# the sliding pymatching point's logical_failure, direct_failure and
# direct_mismatch for seeds 0 to 239, written at commit 9a4334b2, the
# last commit that measured the whole-circuit reference on every shot
DIRECT_REFERENCE = DATA / "direct_reference.csv"

FULL_HISTORY_PAIR = {
    "axes": {
        run_files.DISTANCE_PATH: (3,),
        run_files.ROUND_PERIOD_PATH: (1.0,),
        run_files.COMMIT_ROUNDS_PATH: (None, 16),
        run_files.ERROR_RATE_PATH: (0.015,),
    },
    "machine_arguments": {"weak_decoder": "pymatching"},
    "collection": {"max_shots": 240},
}

# how long the first unit waits for its release before it gives up
RELEASE_WAIT_SECONDS = 20

# the ledger counters shot_links.csv carries per shot per link
LINK_COUNTERS = (
    "transfers",
    "payload_bits",
    "unknown_payload_transfers",
    "queue_wait_us",
    "serialization_us",
    "propagation_us",
)

# the observation fields and record options that record a run without
# changing a shot's row
RECORDING_ONLY = {
    "observation": {
        "trace": "chrome",
        "trace_shots": (0, 1),
        "log": "print",
        "log_component_io": True,
    },
    "record_options": {"confidence_shot_count": 7},
}

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


def test_reference_rows_equal_the_recorded_sweep_and_links(tmp_path):
    study = run_files.sweep(**run_files.REFERENCE)
    tasks = run_files.tasks_of(study)
    measurements = run_files.run_sweep(tasks, 2)
    _rows, run_dir = run_files.folded_run(tmp_path, measurements)
    folded_sweep_path = run_dir / "sweep.csv"
    shot_links_path = run_dir / "shot_links.csv"
    sweep_now = _csv_rows(folded_sweep_path)
    shot_links = _csv_rows(shot_links_path)
    sweep_path = DATA / "reference_sweep.csv"
    links_path = DATA / "reference_links.csv"
    sweep_before = _csv_rows(sweep_path)
    links_before = _csv_rows(links_path)
    sweep_measured = _without_point_id(sweep_now)
    assert len(sweep_measured) == 1
    stable_now = _stable_columns(sweep_measured[0])
    assert stable_now == _stable_columns(sweep_before[0])
    assert _link_means(shot_links) == links_before


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
    run_file = run_files.write_run_file(tmp_path, **FULL_HISTORY_PAIR)
    out_dir = tmp_path / "out"

    run_dir, _rows = run.run_experiment(run_file, out_dir)

    shots_path = run_dir / "shots.csv"
    shots = _csv_rows(shots_path)
    sliding = _shots_by_seed(shots, "null")
    full_history = _shots_by_seed(shots, "16")
    reference = _csv_rows(DIRECT_REFERENCE)
    seeds = _integer_column(reference, "seed")
    expected = _frozen_columns(reference)
    reproduced = _retired_columns_by_seed(sliding, full_history, seeds)
    sliding_digests = _sample_digests(sliding, seeds)
    full_digests = _sample_digests(full_history, seeds)
    mismatches = _integer_column(reference, "direct_mismatch")
    assert reproduced == expected
    assert sliding_digests == full_digests
    assert sum(mismatches) == 2


def test_every_shot_of_a_point_decides_on_the_tasks_calibrator():
    """An online threshold: one calibrator per point, on the task."""
    task = run_files.first_task(
        machine="switching", machine_arguments={"online": {}}
    )
    first = collect.run_shot(task, 0)
    second = collect.run_shot(task, 1)

    assert task.online_threshold is not None
    assert first.machine.switching.policy.threshold is task.online_threshold
    assert second.machine.switching.policy.threshold is task.online_threshold


def test_a_tasks_shots_decode_the_same_with_the_models_built_once():
    """The task's own referent: sinter compiles once per task."""
    task = _reference_task()

    whole_task = collect.Unit(task, 0, 4)
    shared_outcome = collect.run_unit(whole_task, _decoded)
    shared = shared_outcome.rows
    alone = [_decoded_alone(task, seed) for seed in range(4)]

    assert shared == alone


def test_the_first_shot_builds_the_models_and_the_rest_read_them():
    task = _reference_task()
    built = built_window_models.BuiltWindowModels()

    _run_every_shot(task, 3, built)

    assert built.builds == 1
    assert built.reuses == 2


def test_a_machine_built_alone_builds_its_own_models():
    task = _reference_task()
    settings = task.settings

    first = machine_module.Machine.build(settings, 0)
    second = machine_module.Machine.build(settings, 1)

    first_models = first.windows.models.built_models
    second_models = second.windows.models.built_models
    assert first_models is not second_models


def test_two_points_under_one_cache_do_not_share_models():
    """The key carries the circuit text, which is where d and p are.

    Two distances also differ in their window spans, so the noisier
    point at one distance is what pins the circuit text itself.
    """
    built = built_window_models.BuiltWindowModels()
    at_three = _reference_task()
    at_five = _reference_task(distance=5)
    noisier_at_three = _reference_task(probability=0.003)

    collect.run_shot(at_three, 0, built_models=built)
    collect.run_shot(at_five, 0, built_models=built)
    collect.run_shot(noisier_at_three, 0, built_models=built)

    assert built.builds == 3
    assert built.reuses == 0


def test_a_unit_that_ended_is_handed_on_while_one_before_it_runs(tmp_path):
    """A pool hands each unit on the moment it ends, not in list order.

    The first unit waits until the two after it have been handed on, as
    a slow piece runs beside short ones. Handed on in list order, the
    short ones would wait behind it unsaved, and a job killed at its
    time limit would lose them.
    """
    task = _reference_task()
    units = [
        collect.Unit(task, 0, 1),
        collect.Unit(task, 1, 1),
        collect.Unit(task, 2, 1),
    ]
    release_path = tmp_path / "release"
    measure = functools.partial(_seed_after_the_release, release_path)
    handed_on = []
    on_unit_done = functools.partial(_hand_on, handed_on, release_path)

    collect.run_units(units, measure, on_unit_done=on_unit_done, processes=2)

    assert handed_on == [1, 2, 0]


def test_two_tasks_that_run_different_circuits_are_two_tasks():
    """The strong id carries the circuit text, as sinter's does."""
    three_rounds = _memory_task(3)
    five_rounds = _memory_task(5)

    assert three_rounds.strong_id() != five_rounds.strong_id()


def test_two_tasks_whose_operations_differ_in_kind_are_two_tasks():
    """An enum setting enters the strong id as its member's name."""
    memory = program_records.Operation(
        1, "op", (1,), patches=(1,), kind=program_records.OpKind.MEMORY
    )
    merge = dataclasses.replace(memory, kind=program_records.OpKind.MERGE)
    memory_workload = workload_settings.WorkloadSettings(operations=[memory])
    merge_workload = workload_settings.WorkloadSettings(operations=[merge])
    memory_settings = machine_settings.MachineSettings(workload=memory_workload)
    merge_settings = machine_settings.MachineSettings(workload=merge_workload)
    memory_task = collect.Task(memory_settings, {"point": 1})
    merge_task = collect.Task(merge_settings, {"point": 1})

    assert memory_task.strong_id() != merge_task.strong_id()


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

    assert three_task.strong_id() != five_task.strong_id()


def test_two_records_with_the_same_fields_are_two_tasks():
    """A record enters the id by its class beside its fields.

    Two rows that take no settings have records with no fields, so only
    their classes tell the points apart, as gem5's config.json writes
    each object's type.
    """

    @dataclasses.dataclass(frozen=True)
    class EveryRound:
        pass

    @dataclasses.dataclass(frozen=True)
    class NoRound:
        pass

    every_round = EveryRound()
    no_round = NoRound()
    every_round_task = _rounds_task(every_round)
    no_round_task = _rounds_task(no_round)

    assert every_round_task.strong_id() != no_round_task.strong_id()


def test_two_slotted_round_policies_that_differ_in_count_are_two_tasks():
    """A component's __slots__ enter the strong id as its __dict__ does."""
    three_rounds = _slotted_rounds_task(3)
    six_rounds = _slotted_rounds_task(6)

    assert three_rounds.strong_id() != six_rounds.strong_id()


def test_two_python_built_sources_that_hold_different_values_are_two_tasks():
    """A component with no record form enters the id by its content."""
    first = _device_task(1)
    second = _device_task(2)

    assert first.strong_id() != second.strong_id()


def test_two_recorded_sources_that_replay_different_shots_are_two_tasks():
    """An array a component holds enters the id by its values."""
    first = _recorded_device_task(0)
    second = _recorded_device_task(1)

    assert first.strong_id() != second.strong_id()


def test_two_python_built_sources_that_hold_the_same_values_are_one_task():
    first = _device_task(1)
    same = _device_task(1)

    assert first.strong_id() == same.strong_id()


def test_a_class_enters_the_id_by_its_name_whatever_it_holds():
    """A class in the settings is its module and qualified name.

    That is the identity pickle writes for a class (Lib/pickle.py
    save_global), so an attribute the class gains later, as Python adds
    __annotations__ on first read, leaves the id alone.
    """

    class Rule:
        pass

    before = collect.json_value(Rule)
    Rule.touched = True
    after = collect.json_value(Rule)

    assert before == f"{Rule.__module__}.{Rule.__qualname__}"
    assert after == before


def test_a_points_record_keeps_the_labels_its_id_leaves_out(tmp_path):
    """machine.json is where a label is read, so it keeps them all."""
    task = _reference_task()

    run_folder.record_point(tmp_path, "point", task)

    record_path = tmp_path / "points" / "point" / "machine.json"
    record_text = record_path.read_text()
    record = json.loads(record_text)
    links = record["settings"]["links"]
    channel = links["qpu_to_controller"]["channel"]
    assert links["profile_name"] == "logical_reference"
    assert channel["configuration_source"] == run_files.REFERENCE_SOURCE


def test_settings_that_differ_only_in_labels_are_one_point():
    """A label changes no id; a latency, which changes the run, does."""
    task = _reference_task()
    links = task.settings.links
    relabelled = dataclasses.replace(links, profile_name="other")
    channel = links.qpu_to_controller.channel
    slower_ticks = channel.propagation_latency_ticks + 1
    slower_channel = dataclasses.replace(
        channel, propagation_latency_ticks=slower_ticks
    )
    slower_path = dataclasses.replace(
        links.qpu_to_controller, channel=slower_channel
    )
    slower = dataclasses.replace(links, qpu_to_controller=slower_path)

    relabelled_task = _task_with_links(task, relabelled)
    slower_task = _task_with_links(task, slower)

    assert relabelled_task.strong_id() == task.strong_id()
    assert relabelled_task.settings == task.settings
    assert slower_task.strong_id() != task.strong_id()


def test_a_point_traced_and_logged_is_the_point_run_plain(tmp_path, capsys):
    """Recording is no part of a point: one id, one row, either way.

    sinter keeps its output options out of a task's strong id
    (sinter/_data/_task.py:167-204). The point with every recording-only
    observation key on has the plain point's id and, shot for shot, its
    rows but for the wall clock, and its resolved record says what it
    recorded.
    """
    plain = run_files.first_task()
    recorded = run_files.first_task(**RECORDING_ONLY)

    plain_rows = _shot_rows_without_wall_clock(plain)
    recorded_rows = _shot_rows_without_wall_clock(recorded)
    printed = capsys.readouterr()
    run_folder.record_point(tmp_path, "point", recorded)

    record_path = tmp_path / "points" / "point" / "machine.json"
    record_text = record_path.read_text()
    record = json.loads(record_text)
    recorded_observation = record["settings"]["observation"]
    assert recorded.strong_id() == plain.strong_id()
    assert recorded_rows == plain_rows
    assert recorded_observation["trace"] == "chrome"
    assert recorded_observation["log_component_io"] is True
    assert "PauliFrame: committed window" in printed.out


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
    """A point's metadata is checked where its id is made."""
    first_point = run_files.first_task()
    sentence = (
        f"{sentence_start} which is not text; a point's id is the json of "
        "its metadata, whose keys are text, so 1 and '1' would name one point"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        collect.Task(first_point.settings, metadata)


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
    run_file = run_files.write_run_file(
        tmp_path,
        machine_arguments={"weak_decoder": "relay_bp"},
        collection={"max_shots": 2},
    )
    out_dir = tmp_path / "out"
    run_dir, rows = run.run_experiment(run_file, out_dir)
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


def _is_wall_clock_column(column: str) -> bool:
    if column in WALL_CLOCK_COLUMNS:
        return True
    for point in WALL_CLOCK_POINTS:
        if column.startswith(f"{point}_"):
            return True
    return False


def _reference_task(
    distance: int = 3, probability: float = 0.001
) -> collect.Task:
    """The reference machine's task at one distance and error probability."""
    axes = {
        run_files.ERROR_RATE_PATH: (probability,),
        run_files.DISTANCE_PATH: (distance,),
        run_files.ROUND_PERIOD_PATH: (1.0,),
    }
    return run_files.first_task(axes=axes, machine="reference")


def _csv_rows(path: pathlib.Path) -> list:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _stable_columns(row: dict) -> dict:
    stable = {}
    for column, value in row.items():
        if not _is_wall_clock_column(column):
            stable[column] = value
    return stable


def _link_means(shot_link_rows: list) -> list:
    """Each link's counters averaged over one point's shots, as text."""
    rows_by_link = {}
    for row in shot_link_rows:
        link_rows = rows_by_link.setdefault(row["link"], [])
        link_rows.append(row)
    mean_rows = []
    for link in sorted(rows_by_link):
        rows = rows_by_link[link]
        mean_row = _means_of_one_link(link, rows)
        mean_rows.append(mean_row)
    return mean_rows


def _means_of_one_link(link: str, rows: list) -> dict:
    """One link's counters averaged over its shots' rows, as text.

    A mean is statistics.fmean, and bits_per_transfer is the mean
    payload over the mean transfers, zero on a link that sent nothing.
    """
    means = {}
    for counter in LINK_COUNTERS:
        values = [float(row[counter]) for row in rows]
        means[counter] = statistics.fmean(values)
    bits_per_transfer = 0.0
    if means["transfers"]:
        bits_per_transfer = means["payload_bits"] / means["transfers"]
    mean_row = {"algorithm": rows[0]["algorithm"], "link": link}
    for counter in LINK_COUNTERS:
        mean_row[f"{counter}_per_shot"] = str(means[counter])
    mean_row["bits_per_transfer"] = str(bits_per_transfer)
    return mean_row


def _without_point_id(rows: list) -> list:
    kept = []
    for row in rows:
        without = dict(row)
        del without["point_id"]
        kept.append(without)
    return kept


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
        run_files.loop_predictions(shot),
        run_files.whole_circuit_predictions(shot),
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


def _seed_after_the_release(release_path: pathlib.Path, shot) -> int:
    """The shot's seed; seed 0 once release_path exists, or the wait ends."""
    deadline = time.monotonic() + RELEASE_WAIT_SECONDS
    while shot.seed == 0 and not release_path.exists():
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)
    return shot.seed


def _hand_on(handed_on: list, release_path: pathlib.Path, unit, _outcome):
    """Each unit's first seed as it is handed on; the release after two."""
    handed_on.append(unit.first_seed)
    if len(handed_on) == 2:
        release_path.touch()


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
    return _rounds_task(rounds_policy)


def _rounds_task(rounds_policy) -> collect.Task:
    workload = workload_settings.WorkloadSettings(rounds_policy=rounds_policy)
    settings = machine_settings.MachineSettings(workload=workload)
    return collect.Task(settings, {"point": 1})


def _device_task(round_count: int) -> collect.Task:
    """A task whose settings hold a Python-built finite source."""
    device = stim_device.StimDevice(measurement_rounds={1: {0: round_count}})
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(source=source)
    settings = machine_settings.MachineSettings(qpu=qpu)
    return collect.Task(settings, {"point": 1})


def _recorded_device_task(flip: int) -> collect.Task:
    """A task whose settings hold a recorded shot, its last bit flip."""
    measurements = numpy.array([[0, flip]], dtype=bool)
    device = stim_device.RecordedStimDevice(measurements, 0)
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(source=source)
    settings = machine_settings.MachineSettings(qpu=qpu)
    return collect.Task(settings, {"point": 1})


def _shots_by_seed(shots: list, commit_rounds_cell: str) -> dict:
    """One point's shots.csv rows by seed, the point named by its window."""
    by_seed = {}
    for row in shots:
        if row["windows.commit_rounds"] == commit_rounds_cell:
            by_seed[int(row["seed"])] = row
    return by_seed


def _integer_column(rows: list, name: str) -> list:
    """One column of csv rows, each cell read as an integer."""
    return [int(row[name]) for row in rows]


def _frozen_columns(reference: list) -> list:
    """Each frozen row's logical_failure, direct_failure, direct_mismatch."""
    logical_failures = _integer_column(reference, "logical_failure")
    direct_failures = _integer_column(reference, "direct_failure")
    direct_mismatches = _integer_column(reference, "direct_mismatch")
    columns = zip(
        logical_failures, direct_failures, direct_mismatches, strict=True
    )
    return list(columns)


def _sample_digests(shots_by_seed: dict, seeds: list) -> list:
    """One point's sample digests, seed by seed."""
    return [shots_by_seed[seed]["sample_digest"] for seed in seeds]


def _retired_columns_by_seed(
    sliding: dict, full_history: dict, seeds: list
) -> list:
    """The three columns as the pair of points gives them, seed by seed."""
    return [
        _retired_columns(sliding[seed], full_history[seed]) for seed in seeds
    ]


def _retired_columns(sliding_row: dict, full_history_row: dict) -> tuple:
    """The three columns as the pair of points gives them for one seed.

    The mismatch compares the two points' predictions, as the retired
    column compared the loop's with the reference's, not their failures.
    """
    sliding_failure = sliding_row["logical_failure"] == "True"
    full_history_failure = full_history_row["logical_failure"] == "True"
    is_mismatch = sliding_row["predictions"] != full_history_row["predictions"]
    return (int(sliding_failure), int(full_history_failure), int(is_mismatch))


def _shot_rows_without_wall_clock(task: collect.Task) -> list:
    """Two shots' rows, the host's time to simulate each left out."""
    measurements = run_files.run_sweep([task], 2)
    record = sweep_report.record_of(measurements)
    rows = []
    for row in record.shots:
        stable = _stable_columns(row)
        del stable["sim_wall_seconds"]
        rows.append(stable)
    return rows


def _task_with_links(task: collect.Task, links) -> collect.Task:
    settings = dataclasses.replace(task.settings, links=links)
    return collect.Task(settings, task.metadata)


class _CrashingRelayDecoder:
    """A relay-bp decoder whose every decode raises, a crashed backend."""

    def __init__(self, *arguments, **keywords) -> None:
        del arguments, keywords

    def decode_detailed(self, syndrome):
        del syndrome
        raise RuntimeError("the relay-bp backend crashed")


def _crashing_relay_type():
    return _CrashingRelayDecoder
