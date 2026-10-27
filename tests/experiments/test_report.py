"""The rows report.py makes of measured shots, and the files a run writes.

The contract is sinter's additive shape (sinter/_data/_task_stats.py:
only fields that add are recorded and every summary is derived from
them). shot_data_movement.csv holds one row per shot per path off that
shot's own counters, with the memory class the path crosses, so a
folder folded from shards carries the rows one unsharded run would have
written. shot_stats.csv holds one row per shot per part, statistic and
field, the shot's registry rows as they are, and stat_values.csv each
task's count of every value, added over its shots. shot_pools.csv holds
one row per shot per pool, which pool_summary.csv folds per task and
pool as sweep.csv folds shots.csv. latency_samples.csv
holds
one measured algorithm wall clock per decoded window, recorded only for
a decoder named by a table row, each beside its window's inter-arrival,
the deadline a decode must beat. The confidence files leave out an
unscored shot, as sinter takes a discard out of every failure-conditioned
count (sinter/_decoding/_decoding.py:120-128).
"""

import collections
import csv
import math
import types

import pytest

import decsim.experiments.collect as collect
import decsim.experiments.collect_command as collect_command
import decsim.experiments.command as command
import decsim.experiments.measure as measure
import decsim.experiments.report as report
import decsim.records.decoding as decoding_records
import tests.experiments.run_files as run_files
import tests.patch_pools as patch_pools

# one task, every copy and move counted, two shots
COUNTING = {
    "observation": {"data_movement": True},
    "collection": {"max_shots": 2},
}


def measured_shots(count: int) -> list:
    """The counting task, one measured shot per seed."""
    measurements = []
    for seed in range(count):
        measurement = run_files.measured_shot(seed, **COUNTING)
        measurements.append(measurement)
    return measurements


def _measured_patch_shots(count: int) -> list:
    """The per-patch machine of tests/patch_pools.py, one shot a seed."""
    machine = patch_pools.shared_strong_per_patch()
    task = collect.Task("patches", machine, {})
    measurements = []
    for seed in range(count):
        shot = collect.run_shot(task, seed)
        measurement = measure.measure_shot(shot)
        measurements.append(measurement)
    return measurements


# the counters a shot splits over its paths, one share a row
PATH_COLUMNS = (
    "copies",
    "copied_rounds",
    "copy_bits",
    "moves",
    "moved_rounds",
    "move_bits",
)


def _column_sums(rows: list, columns: tuple) -> dict:
    sums = {column: 0 for column in columns}
    for row in rows:
        for column in columns:
            sums[column] += row[column]
    return sums


def test_a_shots_statistics_rows_are_its_registrys_rows():
    measurements = measured_shots(2)
    rows = report.shot_statistics_rows(measurements)

    for measurement in measurements:
        own_rows = [row for row in rows if row["seed"] == measurement.seed]
        values = {
            (row["path"], row["statistic"], row["field"]): row["value"]
            for row in own_rows
        }
        recorded = {
            (row.path, row.statistic, row.field): row.value
            for row in measurement.statistics
        }
        assert len(own_rows) == len(measurement.statistics)
        assert values == recorded
        assert ("weak.units", "busy_units", "area") in values


def test_a_tasks_value_counts_add_its_shots_counts():
    measurements = measured_shots(2)
    rows = report.statistic_value_rows(measurements)
    added = collections.Counter()
    for measurement in measurements:
        for value_count in measurement.statistic_values:
            key = (value_count.path, value_count.statistic, value_count.value)
            added[key] += value_count.count

    counted = {
        (row["path"], row["statistic"], row["value"]): row["count"]
        for row in rows
    }

    assert counted == dict(added)
    assert ("weak.decodes", "service") in {key[:2] for key in counted}


def _summary_values(rows: list, path: str, statistic: str) -> dict:
    """One statistic's measures off stat_summary.csv rows."""
    found = {}
    for row in rows:
        if (row["path"], row["statistic"]) == (path, statistic):
            found[row["measure"]] = row["value"]
    return found


def test_the_summary_is_derived_from_the_shots_own_statistics(tmp_path):
    """Means over the shots' added fields; a peak over every shot."""
    measurements = measured_shots(2)
    _rows, run_dir = run_files.folded_run(tmp_path, measurements)
    summary_path = run_dir / "stat_summary.csv"
    summary = report.read_rows(summary_path)
    fields = collections.defaultdict(list)
    for measurement in measurements:
        for row in measurement.statistics:
            fields[(row.path, row.statistic, row.field)].append(row.value)
    service_count = sum(fields[("weak.decodes", "service", "count")])
    service_total = sum(fields[("weak.decodes", "service", "total")])
    busy_area = sum(fields[("weak.units", "busy_units", "area")])
    busy_span = sum(fields[("weak.units", "busy_units", "span")])
    busy_peak = max(fields[("weak.units", "busy_units", "peak")])

    service = _summary_values(summary, "weak.decodes", "service")
    busy = _summary_values(summary, "weak.units", "busy_units")
    units = _summary_values(summary, "weak.units", "units")
    assert service["count"] == service_count
    service_mean = service_total / service_count
    busy_mean = busy_area / busy_span
    assert service["mean"] == pytest.approx(service_mean)
    assert service["minimum"] <= service["p50"] <= service["maximum"]
    assert busy["time_mean"] == pytest.approx(busy_mean)
    assert busy["peak"] == busy_peak
    assert units["per_shot"] == 1


def test_a_shots_pool_rows_are_its_measured_pools():
    measurements = _measured_patch_shots(1)
    (measurement,) = measurements

    rows = report.shot_pool_rows(measurements)

    pools = [row["pool"] for row in rows]
    loads = [row["offered_load"] for row in rows]
    measured_loads = [pool.offered_load for pool in measurement.pools]
    assert pools == ["weak", "w1", "w2", "strong"]
    assert loads == measured_loads


def test_a_tasks_pool_summary_folds_each_pools_rows_over_its_shots(tmp_path):
    """A mean load, the largest queue, the windows added, pool by pool."""
    first, second = _measured_patch_shots(2)
    _rows, run_dir = run_files.folded_run(tmp_path, [first, second])
    summary_path = run_dir / "pool_summary.csv"

    summary = report.read_rows(summary_path)

    w1 = summary[1]
    first_w1 = first.pools[1]
    second_w1 = second.pools[1]
    mean_load = (first_w1.offered_load + second_w1.offered_load) / 2
    largest_queue = max(first_w1.queue_max, second_w1.queue_max)
    committed = first_w1.committed_windows + second_w1.committed_windows
    escalated = first_w1.escalated_windows + second_w1.escalated_windows
    assert [row["pool"] for row in summary] == ["weak", "w1", "w2", "strong"]
    assert w1["shots"] == 2
    assert w1["offered_load"] == pytest.approx(mean_load)
    assert w1["queue_max"] == largest_queue
    assert w1["committed_windows"] == committed
    assert w1["escalated_windows"] == escalated


# statistics written as each piece's sums, a row per task and key
SUMMED = collect.RecordOptions(is_statistics_per_shot=False)


def _summary_by_key(run_dir) -> dict:
    """stat_summary.csv's values by task, path, statistic and measure."""
    summary_path = run_dir / "stat_summary.csv"
    by_key = {}
    for row in report.read_rows(summary_path):
        key = (row["task_id"], row["path"], row["statistic"], row["measure"])
        by_key[key] = row["value"]
    return by_key


def test_pieces_summed_statistics_fold_to_the_shots_own_summary(tmp_path):
    """Three shots: per shot in one piece, or summed in pieces of two.

    A peak keeps the largest and every other field adds, both laws
    associative, so summing first changes no summary, only how many
    rows shot_stats.csv holds: one per piece per key.
    """
    measurements = measured_shots(3)
    _rows, per_shot_dir = run_files.folded_run(tmp_path, measurements)
    _rows, summed_dir = run_files.folded_run(
        tmp_path, measurements, SUMMED, piece_shots=2
    )
    per_shot_summary = _summary_by_key(per_shot_dir)
    summed_summary = _summary_by_key(summed_dir)
    statistics_path = summed_dir / "shot_stats.csv"
    summed_statistics = report.read_rows(statistics_path)
    expected_rows = collections.Counter()
    for first_seed, piece in ((0, measurements[:2]), (2, measurements[2:])):
        keys = _statistic_keys(piece)
        expected_rows[(first_seed, len(piece))] = len(keys)
    written_rows = collections.Counter()
    for row in summed_statistics:
        written_rows[(row["seed"], row["shot_count"])] += 1

    assert summed_summary.keys() == per_shot_summary.keys()
    for key, value in per_shot_summary.items():
        assert summed_summary[key] == pytest.approx(value), key
    assert written_rows == expected_rows


def _statistic_keys(measurements: list) -> set:
    """Every part, statistic and field the shots recorded."""
    keys = set()
    for measurement in measurements:
        for row in measurement.statistics:
            keys.add((row.path, row.statistic, row.field))
    return keys


def test_pieces_summed_links_add_the_shots_own_counters(tmp_path):
    """Each link's counters over three shots, two pieces' rows added."""
    measurements = measured_shots(3)
    _rows, summed_dir = run_files.folded_run(
        tmp_path, measurements, SUMMED, piece_shots=2
    )
    links_path = summed_dir / "shot_links.csv"
    summed_links = report.read_rows(links_path)
    measured = []
    for measurement in measurements:
        link_counters = measurement.link_totals.items()
        measured.extend(link_counters)
    written = []
    for row in summed_links:
        counters = {name: row[name] for name in LINK_COUNTERS}
        written.append((row["link"], counters))
    measured_sums = _counter_sums(measured)
    written_sums = _counter_sums(written)
    link_paths = {path for path, _counters in measured}

    assert written_sums == pytest.approx(measured_sums)
    assert len(summed_links) == 2 * len(link_paths)


# the counters a link row holds (measure.link_totals)
LINK_COUNTERS = (
    "transfers",
    "payload_bits",
    "unknown_payload_transfers",
    "queue_wait_us",
    "serialization_us",
    "propagation_us",
)


def _counter_sums(counters_by_link: list) -> dict:
    """Each link's counters added over (link, counters) pairs."""
    sums = collections.Counter()
    for path, counters in counters_by_link:
        for name, value in counters.items():
            sums[(path, name)] += value
    return dict(sums)


def test_a_shots_rows_add_up_to_that_shots_own_counters():
    measurements = measured_shots(1)
    rows = report.shot_data_movement_rows(measurements)
    counted = measurements[0].data_movement
    summed = _column_sums(rows, PATH_COLUMNS)
    references = {row["references"] for row in rows}
    referenced_rounds = {row["referenced_rounds"] for row in rows}

    assert summed == {column: counted[column] for column in PATH_COLUMNS}
    assert references == {counted["references"]}
    assert referenced_rounds == {counted["referenced_rounds"]}


def test_each_rows_memory_class_is_the_one_the_counters_placed():
    measurements = measured_shots(1)
    rows = report.shot_data_movement_rows(measurements)
    class_by_path = {row["path"]: row["memory_class"] for row in rows}

    assert class_by_path["qpu_to_controller"] == "off_board"
    assert class_by_path["controller_to_weak_buffer"] == "on_board"
    assert class_by_path["readout -> controller intake"] == "on_chip"
    path = "controller assembler -> weak_buffer"
    assert class_by_path[path] == "on_board"


def test_a_run_that_counted_no_movement_writes_no_rows(tmp_path, monkeypatch):
    """observation.data_movement off means no counts, which is not zero."""
    monkeypatch.chdir(tmp_path)
    silent_path = run_files.write_run_file(tmp_path)
    run_dir, _rows = collect_command.run_experiment(silent_path)
    movement_path = run_dir / "shot_data_movement.csv"

    assert not movement_path.exists()


def _memory_classes_with(rows: list, column: str) -> set:
    """The memory classes of the rows whose column holds a count."""
    classes = set()
    for row in rows:
        if row[column]:
            classes.add(row["memory_class"])
    return classes


def test_a_counting_run_writes_the_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_path = run_files.write_run_file(tmp_path, **COUNTING)
    run_dir, _rows = collect_command.run_experiment(run_path)
    movement_path = run_dir / "shot_data_movement.csv"
    shot_rows = report.read_rows(movement_path)
    copied = _memory_classes_with(shot_rows, "copy_bits")
    moved = _memory_classes_with(shot_rows, "move_bits")

    assert "on_chip" in copied
    assert "off_board" in moved


def test_pieces_of_one_shot_fold_to_the_whole_runs_movement_rows(tmp_path):
    """The fold's precondition: the file adds, so pieces equal one run."""
    whole_path = run_files.write_run_file(tmp_path, **COUNTING)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_collection = {"max_shots": 2, "piece_rounds": 1}
    cut_path = run_files.write_run_file(
        cut_folder, **{**COUNTING, "collection": cut_collection}
    )
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["run", str(whole_path), "--out", str(whole_dir)])
    command.main(["run", str(cut_path), "--out", str(cut_dir)])
    whole_movement_path = whole_dir / "shot_data_movement.csv"
    folded_movement_path = cut_dir / "shot_data_movement.csv"
    whole_rows = report.read_rows(whole_movement_path)
    folded_rows = report.read_rows(folded_movement_path)
    cut_pieces = cut_dir.glob("pieces/*/*")

    assert len(list(cut_pieces)) == 2
    assert folded_rows == whole_rows


# PyMatching's measured wall clock on every window, two shots a task
WALL_CLOCK = {
    "machine_arguments": {"weak_decoder": "pymatching"},
    "collection": {"max_shots": 2},
}


def _counts_by_tier(rows: list, name: str) -> collections.Counter:
    """One latency point's sample count under each tier."""
    counted = collections.Counter()
    for row in rows:
        if row["name"] == name:
            counted[row["tier"]] += row["count"]
    return counted


def wall_clock_axes(distances: tuple) -> dict:
    return {
        run_files.DISTANCE_PATH: distances,
        run_files.ROUND_PERIOD_PATH: (1.0,),
        run_files.ERROR_RATE_PATH: (0.001,),
    }


def test_every_decoded_window_contributes_one_latency_sample():
    axes = wall_clock_axes((3,))
    measurement = run_files.measured_shot(axes=axes, **WALL_CLOCK)
    samples = measurement.samples["algorithm"]
    assert len(samples) == measurement.decoded_windows
    assert all(sample > 0 for sample in samples)


def test_the_end_of_stream_reaction_is_counted_once_a_shot_under_no_tier():
    """Its median and p99 in sweep.csv are over the task's shots."""
    measurements = measured_shots(2)
    rows = report.window_sample_rows(measurements)
    counted = _counts_by_tier(rows, "end_of_stream_reaction")

    assert counted == {"": 2}


def test_a_latency_sample_names_the_tier_that_decoded_its_window():
    """A weak-only run's windows are all the weak tier's."""
    axes = wall_clock_axes((3,))
    measurement = run_files.measured_shot(axes=axes, **WALL_CLOCK)
    rows = report.latency_sample_rows([measurement])
    tiers = [row["tier"] for row in rows]
    assert tiers == ["weak"] * measurement.decoded_windows


def test_a_wall_clock_run_records_every_windows_sample_and_deadline(
    tmp_path, monkeypatch
):
    """A sample per decoded window, beside its task and its deadline.

    Each task holds as many samples as its shots decoded windows. The
    deadline is the window's inter-arrival, commit rounds (d when null)
    times the round period: 3 us at d 3 and 5 us at d 5.
    """
    monkeypatch.chdir(tmp_path)
    axes = wall_clock_axes((3, 5))
    run_path = run_files.write_run_file(tmp_path, axes=axes, **WALL_CLOCK)
    run_dir, rows = collect_command.run_experiment(run_path)
    samples_path = run_dir / "latency_samples.csv"
    with open(samples_path, newline="") as handle:
        reader = csv.DictReader(handle)
        samples = list(reader)
    shots_path = run_dir / "shots.csv"
    with open(shots_path, newline="") as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)

    task_ids = [row["task_id"] for row in samples]
    sample_counts = collections.Counter(task_ids)
    deadlines = {
        (row["task_id"], row["qpu.distance"], row["window_period_us"])
        for row in samples
    }
    task_ids_by_distance = {
        row["qpu.distance"]: row["task_id"] for row in shots
    }
    assert sample_counts == _windows_by_task(shots)
    assert deadlines == {
        (task_ids_by_distance["3"], "3", "3.0"),
        (task_ids_by_distance["5"], "5", "5.0"),
    }


def test_a_latency_card_run_records_no_samples(tmp_path, monkeypatch):
    """The fixed-latency card is flat by construction: no raw samples."""
    monkeypatch.chdir(tmp_path)
    axes = wall_clock_axes((3, 5))
    card_path = run_files.write_run_file(tmp_path, axes=axes)
    card_run_dir, rows = collect_command.run_experiment(card_path)
    assert not (card_run_dir / "latency_samples.csv").exists()


def _windows_by_task(shots: list) -> collections.Counter:
    """The windows every task's shots decoded, off shots.csv."""
    windows = collections.Counter()
    for row in shots:
        windows[row["task_id"]] += int(row["decoded_windows"])
    return windows


def test_a_gaps_bin_is_its_tenth_of_a_decibel_below():
    ten_decibels_in_nats = 2.302585092994046

    assert report.gap_bin_low_decibels(ten_decibels_in_nats) == 10.0
    assert report.gap_bin_low_decibels(0.0) == 0.0
    assert report.gap_bin_low_decibels(math.inf) == math.inf


def test_a_window_with_no_gap_is_counted_in_the_empty_bin():
    """A gapless window escalates as the least confident: its shot's too."""
    ten_decibels_in_nats = 2.302585092994046
    windows = (
        decoding_records.WindowConfidence(
            (1, 0), ten_decibels_in_nats, False, None
        ),
        decoding_records.WindowConfidence((1, 1), None, True, None),
    )
    confidence = measure.ShotConfidence("complementary_gap", windows, True, 100)
    shot = types.SimpleNamespace(
        task_id="p",
        algorithm="pymatching",
        confidence=confidence,
        logical_failure=False,
        is_scored=True,
    )

    rows = report.confidence_histogram_rows([shot])

    assert rows == [
        _one_count("window", None, True),
        _one_count("window", 10.0, False),
        _one_count("shot_minimum", None, None),
    ]


def test_an_unscored_shot_is_in_neither_confidence_file():
    """An unscored shot is sinter's discard, and counts in no P(e|g).

    Its logical_failure reads False, so a row of it would count as a
    success beside every gap. sinter takes a discard out of every
    failure-conditioned count (sinter/_decoding/_decoding.py:120-128),
    and every row of both files carries shot_failed.
    """
    windows = (decoding_records.WindowConfidence((1, 0), 0.1, True, None),)
    confidence = measure.ShotConfidence("complementary_gap", windows, True, 100)
    shot = types.SimpleNamespace(
        task_id="p",
        algorithm="pymatching",
        seed=0,
        confidence=confidence,
        logical_failure=False,
        is_scored=False,
    )

    assert report.window_confidence_rows([shot]) == []
    assert report.confidence_histogram_rows([shot]) == []


def _one_count(histogram, gap_low_decibels, escalated) -> dict:
    """A histogram row of one window or shot of that no-failure shot."""
    return {
        "task_id": "p",
        "algorithm": "pymatching",
        "signal": "complementary_gap",
        "histogram": histogram,
        "gap_low_decibels": gap_low_decibels,
        "escalated": escalated,
        "shot_failed": False,
        "count": 1,
    }
