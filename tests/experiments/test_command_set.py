"""The `decsim` command set: the verbs, and what each one is equal to.

The shape is sinter's (sinter/_command/_main.py: one command, a verb per
first word, each verb's module imported lazily). Each test
here pins a verb against what the same work done in Python returns, so a
command line is never the only record of a number.
"""

import csv
import dataclasses
import datetime
import functools
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import types

import numpy
import pytest
import scipy.stats
import sinter
import stim

import decsim.collect as collect
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.experiments.collect_command as collect_command
import decsim.experiments.collection as collection_module
import decsim.experiments.command as command
import decsim.experiments.experiment as experiment
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.fold as fold
import decsim.experiments.pieces as pieces
import decsim.experiments.plan_command as plan_command
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.producers as producers
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.program as program_records
import decsim.settings as machine_settings
import tests.experiments.run_files as run_files
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

THIS_FILE = pathlib.Path(__file__)
THIS_PATH = THIS_FILE.resolve()
REPOSITORY = THIS_PATH.parents[2]
# The run files examples/ ships, each run with `decsim run`.
EXAMPLES_DIR = REPOSITORY / "examples"
EXAMPLES = (
    "my_first_sweep.py",
    "priced_cards_example.py",
    "recipes.py",
    "two_tiers.py",
)
FOUR_POINTS = {
    "axes": run_files.FOUR_POINT_AXES,
    "collection": {"max_shots": 2},
}
# the four points in pieces of one shot each
FOUR_POINTS_CUT = {
    "axes": run_files.FOUR_POINT_AXES,
    "collection": {"max_shots": 2, "piece_rounds": 1},
}
# a switching point whose threshold learns online
ONLINE = {"machine": "switching", "machine_arguments": {"online": {}}}
# one noisy point, which fails often enough for a stop rule to stop it
NOISY_AXES = {
    run_files.DISTANCE_PATH: (3,),
    run_files.ROUND_PERIOD_PATH: (1.0,),
    run_files.ERROR_RATE_PATH: (0.02,),
}


def _run_one_shot(config_path, seed=0, out_dir=None, *, trace=False):
    """`decsim run <file> --seed S`: the first point's one narrated shot."""
    run_path = pathlib.Path(config_path)
    study = experiment.load_one_point(run_path)
    return collect_command.run_one_shot(
        study, run_path, seed, out_dir, trace=trace
    )


# The suite's container ships no git, so the tests that need it skip.
_GIT_MISSING = shutil.which("git") is None
requires_git = pytest.mark.skipif(
    _GIT_MISSING, reason="git is not installed where the suite runs"
)


EVERY_FILE = (
    "sweep.csv",
    "shots.csv",
    "shot_links.csv",
    "window_samples.csv",
)


def _rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _rows_without_wall_clock(run_dir, name):
    path = run_dir / name
    rows = _rows(path)
    stripped = []
    for row in rows:
        row.pop("sim_wall_seconds", None)
        row.pop("sim_wall_seconds_per_shot", None)
        stripped.append(row)
    return stripped


def _rows_of_every_file(run_dir) -> dict:
    """Each file's rows without the wall clock, by file name."""
    return {
        name: _rows_without_wall_clock(run_dir, name) for name in EVERY_FILE
    }


@pytest.fixture(autouse=True)
def one_tree_reading_per_test(monkeypatch):
    """Every test here takes its own reading of the tree.

    run_folder reads the tree once per process, which is what a cluster
    task is. The suite is one process running many runs, and one of
    these tests answers for git itself, so the reading is dropped
    around each test rather than carried between them. A batch task
    refuses a tree git does not vouch for, which tests/test_tools.py
    holds; here a task runs on the tree as it stands, and the reading
    it exports goes when the test ends.
    """
    monkeypatch.setenv(run_folder.ALLOW_DIRTY_VARIABLE, "1")
    monkeypatch.delenv(run_folder.TREE_DIRTY_VARIABLE, raising=False)
    monkeypatch.delenv(run_folder.TREE_PATCH_VARIABLE, raising=False)
    run_folder._tree_reading.cache_clear()
    yield
    run_folder._tree_reading.cache_clear()


def _manifest_of(run_dir):
    path = run_dir / "run.json"
    text = path.read_text()
    return json.loads(text)


def _commit_of_this_tree():
    """This test file's own checkout at HEAD, read without git.

    Walked up from this file rather than from the module under test, so
    a manifest that named some other tree would fail here. The container
    the suite runs in has no git binary, which is why the git files are
    read directly; the reader knows a worktree's .git file and the refs
    it shares with the repo.
    """
    this_file = pathlib.Path(__file__)
    here = this_file.resolve()
    checkout = here
    while not (checkout / ".git").exists():
        checkout = checkout.parent
    return run_folder._commit_from_git_files(checkout)


def test_an_unknown_verb_prints_the_verbs_and_fails():
    with pytest.raises(SystemExit):
        command.main(["decode-everything"])


def test_help_prints_the_verbs_without_failing(capsys):
    command.main(["--help"])

    printed = capsys.readouterr()
    assert printed.err.strip() == command.usage()


def test_run_builds_a_point_whose_threshold_learns_online(tmp_path):
    """The one shot runs its point's threshold, as a collect's shot does."""
    run_file = run_files.write_run_file(tmp_path, **ONLINE)
    out_dir = tmp_path / "run"

    _run_one_shot(run_file, out_dir=out_dir)

    assert (out_dir / "result.json").exists()


def test_run_prints_the_result_fields_the_gate_hashes(tmp_path):
    run_file = run_files.write_run_file(tmp_path)
    lines = _run_one_shot(run_file, seed=0, out_dir=tmp_path)
    point = run_files.first_task()
    settings = point.settings
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    row = result.operation_results[0]
    text = "\n".join(lines)
    assert f"terminal status: {result.terminal_status}" in text
    assert f"execution done: {result.execution_done_ticks} ticks" in text
    assert f"fully done: {result.fully_done_ticks} ticks" in text
    assert f"observables {row.logical_observables}" in text
    assert f"truth {row.observable_truth}" in text


def test_run_with_trace_writes_the_shots_trace_file(tmp_path):
    run_file = run_files.write_run_file(tmp_path)
    _run_one_shot(run_file, seed=0, out_dir=tmp_path, trace=True)
    trace_dir = tmp_path / "trace"
    entries = trace_dir.iterdir()
    written = sorted(entries)
    assert len(written) == 1


def _four_point_names(tmp_path) -> tuple:
    """The four-point sweep's run file, and its point names in order."""
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    study = experiment.load(config_path)
    names = [point.name for point in study.points]
    return config_path, names


def test_run_list_prints_the_point_names_in_order(tmp_path, capsys):
    config_path, names = _four_point_names(tmp_path)

    command.main(["run", str(config_path), "--list"])

    printed = capsys.readouterr()
    assert printed.out.splitlines() == names


def test_run_only_collects_the_one_point_it_names(tmp_path):
    config_path, names = _four_point_names(tmp_path)
    out_dir = tmp_path / "out"

    command.main(
        ["run", str(config_path), "--out", str(out_dir), "--only", names[1]]
    )

    points_dir = out_dir / "points"
    recorded = sorted(path.name for path in points_dir.iterdir())
    assert recorded == [names[1]]
    manifest = _manifest_of(out_dir)
    assert len(manifest["points"]) == 4
    sweep_path = out_dir / "sweep.csv"
    sweep_rows = _rows(sweep_path)
    assert len(sweep_rows) == 1


def test_run_shots_gives_the_first_shots_of_the_full_collection(tmp_path):
    config_path, names = _four_point_names(tmp_path)
    full_dir = tmp_path / "full"
    first_dir = tmp_path / "first"

    command.main(["run", str(config_path), "--out", str(full_dir)])
    command.main(
        ["run", str(config_path), "--out", str(first_dir), "--shots", "1"]
    )

    full_rows = _rows_without_wall_clock(full_dir, "shots.csv")
    first_rows = _rows_without_wall_clock(first_dir, "shots.csv")
    seed_zero_rows = [row for row in full_rows if row["seed"] == "0"]
    assert len(full_rows) == 8
    assert first_rows == seed_zero_rows


@pytest.mark.parametrize("name", EXAMPLES)
def test_run_collects_a_first_shot_of_every_point_of_an_example(tmp_path, name):
    example_path = EXAMPLES_DIR / name
    study = experiment.load(example_path)
    out_dir = tmp_path / "out"

    command.main(
        ["run", str(example_path), "--out", str(out_dir), "--shots", "1"]
    )

    shot_rows = _rows_without_wall_clock(out_dir, "shots.csv")
    point_ids = {row["point_id"] for row in shot_rows}
    seeds = {row["seed"] for row in shot_rows}
    assert len(shot_rows) == len(study.points)
    assert len(point_ids) == len(study.points)
    assert seeds == {"0"}


def test_run_seed_narrates_the_shot_of_the_point_only_names(tmp_path):
    config_path, names = _four_point_names(tmp_path)
    out_dir = tmp_path / "out"

    command.main(
        [
            "run",
            str(config_path),
            "--seed",
            "1",
            "--only",
            names[2],
            "--out",
            str(out_dir),
        ]
    )

    record_path = out_dir / "points" / names[2] / "machine.json"
    record = run_folder.read_json(record_path)
    assert record["seeds"] == [[1, 1]]
    assert (out_dir / "result.json").exists()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--seed", "0", "--shots", "2"], "--seed runs one narrated shot"),
        (["--trace"], "--log and --trace narrate the one shot --seed runs"),
        (["--shots", "0"], "a shot count is at least 1, got 0"),
    ],
)
def test_run_refuses_options_that_ask_two_things(
    tmp_path, capsys, arguments, message
):
    config_path = run_files.write_run_file(tmp_path)

    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(config_path), *arguments])

    printed = capsys.readouterr()
    assert stopped.value.code == 2
    assert message in printed.err


def _csv_rows(path: pathlib.Path) -> list:
    """One csv file's rows, each value as its text."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _one_file(folder: pathlib.Path, pattern: str) -> pathlib.Path:
    found = folder.glob(pattern)
    matches = sorted(found)
    assert len(matches) == 1
    return matches[0]


def _sha256_of(path: pathlib.Path) -> str:
    contents = path.read_bytes()
    digest = hashlib.sha256(contents)
    return digest.hexdigest()


def _hashes_of(folder: pathlib.Path, *names: str) -> dict:
    """Each named file of the folder, with its sha256."""
    hashes = {}
    for name in names:
        path = folder / name
        hashes[name] = _sha256_of(path)
    return hashes


def test_a_run_folder_holds_the_points_values_and_workload(tmp_path):
    """What the run ran, every value of it, and the workload as files."""
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    lines = _run_one_shot(config_path, seed=0, out_dir=out_dir)
    resolved_dir = out_dir / "points"
    resolved_path = _one_file(resolved_dir, "*/machine.json")
    resolved_text = resolved_path.read_text()
    resolved = json.loads(resolved_text)
    inputs_dir = resolved_path.parent / "inputs"
    hashes_text = (inputs_dir / "hashes.json").read_text()
    hashes = json.loads(hashes_text)

    assert "terminal status: complete" in lines[2]
    assert resolved["seeds"] == [[0, 1]]
    assert resolved["built"]["commit_rounds"] == 3
    assert resolved["settings"]["qpu"]["distance"] == 3
    assert hashes == _hashes_of(
        inputs_dir, "operation_1.stim", "operations.json"
    )
    assert (out_dir / "result.json").exists()


def test_a_shots_rounds_add_up_every_patchs_rounds():
    """Three patches of fifteen rounds are forty-five patch-rounds.

    Tesseract 2503.10988 lines 287-290 count the rounds across every
    code in the shot, so a per-round rate compares with one memory.
    """
    task = three_patch_task()

    record = run_folder.point_record("point", task)

    assert record["rounds_per_shot"] == 45


def three_patch_task() -> collect.Task:
    """The minimal machine's point running three 15-round memory patches."""
    cells = {
        run_files.ERROR_RATE_PATH: 0.001,
        run_files.DISTANCE_PATH: 3,
        run_files.ROUND_PERIOD_PATH: 1.0,
    }
    base = run_files.minimal_machine(cells)
    patches = producers.memory_patches(run_files.CODE_TASK, 15, 3, 3, 0.001)
    workload = workload_settings.WorkloadSettings.running(patches)
    settings = dataclasses.replace(base, workload=workload)
    return collect.Task(settings, cells)


def test_a_streams_rounds_are_its_segments_counted_once():
    """A three-round segment of a three-round stream is three rounds.

    The QPU fires the workload's operations (qpu/cycle_clock.py
    _emit_operation_rounds), and the stream's owner is how the decoder
    reads them, so its rounds are the segment's and not more.
    """
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    segment = dataclasses.replace(
        owner,
        id=1,
        stream_id=100,
        stream_offset=0,
        syndrome_fragment_index=0,
        syndrome_fragment_count=1,
    )
    rounds = round_policies.PerOperationRounds(((100, 3), (1, 3)))
    workload = workload_settings.WorkloadSettings(
        operations=(segment,), decode_operations=(owner,), rounds_policy=rounds
    )
    timing_only = syndrome_devices.TimingOnlyDevice.Settings()
    qpu = qpu_settings.QpuSettings(distance=3, source=timing_only)
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )
    task = collect.Task(settings, {})

    record = run_folder.point_record("point", task)

    assert record["rounds_per_shot"] == 3


def test_a_point_recorded_again_hashes_only_its_inputs(tmp_path):
    """A retried folder records its points again over the first record."""
    task = run_files.first_task()
    run_dir = tmp_path / "run"

    run_folder.record_point(run_dir, "point", task)
    run_folder.record_point(run_dir, "point", task)

    inputs_dir = run_dir / "points" / "point" / "inputs"
    hashes_path = inputs_dir / "hashes.json"
    hashes_text = hashes_path.read_text()
    hashes = json.loads(hashes_text)
    assert sorted(hashes) == ["operation_1.stim", "operations.json"]


def test_a_collect_run_again_into_its_folder_reruns_no_saved_piece(tmp_path):
    """A killed or finished collect run again skips every piece it saved.

    A piece's folder is written whole or not at all, so one that exists
    holds its shots, and the fold of the second run is the first's.
    """
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    first_status = piece_path.stat()
    report_dir = out_dir
    first_rows = _rows_without_wall_clock(report_dir, "sweep.csv")

    command.main(["run", str(config_path), "--out", str(out_dir)])

    second_status = piece_path.stat()
    second_rows = _rows_without_wall_clock(report_dir, "sweep.csv")
    assert second_status.st_mtime_ns == first_status.st_mtime_ns
    assert second_rows == first_rows


def test_a_piece_records_the_peak_memory_of_the_process_that_ran_it(
    tmp_path,
):
    """The referent is getrusage in the process that ran the piece.

    A serial collect runs its piece here, and a process's peak never
    falls, so the piece's peak is above zero and at most this process's
    peak read after it (ru_maxrss in kilobytes on Linux).
    """
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"

    command.main(["run", str(config_path), "--out", str(out_dir)])

    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    piece_text = piece_path.read_text()
    piece = json.loads(piece_text)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_after_mb = usage.ru_maxrss / 1024
    assert 0 < piece["peak_memory_mb"] <= peak_after_mb


def test_a_piece_records_the_interpreter_packages_and_slurm_task(
    tmp_path, monkeypatch
):
    """The referents are the interpreter's and the packages' own versions.

    A rerun needs what sampled and decoded beside the commit, and the
    array job and task name the Slurm task when one ran the piece.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "14700001")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "14700000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"

    command.main(["run", str(config_path), "--out", str(out_dir)])

    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    piece_text = piece_path.read_text()
    piece = json.loads(piece_text)
    assert piece["python"] == platform.python_version()
    assert piece["packages"]["stim"] == stim.__version__
    assert piece["packages"]["numpy"] == numpy.__version__
    assert piece["slurm_job_id"] == "14700001"
    assert piece["slurm_array_job_id"] == "14700000"
    assert piece["slurm_array_task_id"] == "7"


def test_a_pooled_piece_names_the_decoder_package_its_worker_loaded(
    tmp_path, monkeypatch
):
    """relay_bp loads in the worker that decodes, not in the saving parent.

    The referent is the version relay-bp's installed distribution states,
    since the module states none of its own.
    """
    monkeypatch.delitem(sys.modules, "relay_bp", raising=False)
    config_path = run_files.write_run_file(
        tmp_path, machine_arguments={"weak_decoder": "relay_bp"}
    )
    out_dir = tmp_path / "out"
    pooled = ["run", str(config_path), "--out", str(out_dir)]

    command.main([*pooled, "--processes", "2"])

    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    piece_text = piece_path.read_text()
    piece = json.loads(piece_text)
    distribution_version = importlib.metadata.version("relay-bp")
    assert "relay_bp" not in sys.modules
    assert piece["packages"]["relay_bp"] == distribution_version


def test_a_module_that_states_its_version_is_named_by_it(monkeypatch):
    stated = types.ModuleType("example_package")
    stated.__version__ = "3.1.4"
    monkeypatch.setitem(sys.modules, "example_package", stated)

    versions = collect.imported_module_versions()

    assert versions["example_package"] == "3.1.4"


def test_the_standard_library_and_submodules_are_left_out():
    versions = collect.imported_module_versions()

    assert "json" not in versions
    assert "stim" in versions
    assert "decsim.collect" not in versions


def test_a_piece_names_the_processor_linux_lists(tmp_path, monkeypatch):
    processor_info = tmp_path / "cpuinfo"
    processor_info.write_text(
        "processor\t: 0\nmodel name\t: Example CPU 9000 @ 2.00GHz\n"
    )
    monkeypatch.setattr(run_folder, "PROCESSOR_INFO_FILE", processor_info)

    identity = run_folder.piece_identity()

    assert identity["processor_model"] == "Example CPU 9000 @ 2.00GHz"


def test_a_piece_with_no_processor_list_names_the_platforms_processor(
    tmp_path, monkeypatch
):
    missing = tmp_path / "no_cpuinfo"
    monkeypatch.setattr(run_folder, "PROCESSOR_INFO_FILE", missing)

    identity = run_folder.piece_identity()

    assert identity["processor_model"] == platform.processor()


def test_a_cut_run_with_a_deleted_piece_run_again_is_the_uncut_run(tmp_path):
    """The uncut collect is the oracle for a cut one that lost a piece.

    Pieces of one shot each, one of them deleted as a killed job leaves
    it missing, then the same collect again: it runs that piece alone
    and every folded file is the uncut run's.
    """
    whole_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_path = run_files.write_run_file(cut_folder, **FOUR_POINTS_CUT)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["run", str(whole_path), "--out", str(whole_dir)])
    command.main(["run", str(cut_path), "--out", str(cut_dir)])
    second_pieces = cut_dir.glob("pieces/*/1-1")
    cut_pieces = sorted(second_pieces)
    lost_piece = cut_pieces[0]
    kept_piece = cut_pieces[1]
    kept_status = (kept_piece / "piece.json").stat()
    shutil.rmtree(lost_piece)

    command.main(["run", str(cut_path), "--out", str(cut_dir)])

    reissued_status = (kept_piece / "piece.json").stat()
    whole_run_dir = whole_dir
    whole_rows = _rows_of_every_file(whole_run_dir)
    cut_run_dir = cut_dir
    cut_rows = _rows_of_every_file(cut_run_dir)
    assert lost_piece.is_dir()
    assert reissued_status.st_mtime_ns == kept_status.st_mtime_ns
    assert cut_rows == whole_rows


def test_a_staging_folder_a_killed_run_left_is_no_piece(tmp_path):
    """A killed writer's staging folder: the piece runs, the fold skips it."""
    config_path = run_files.write_run_file(tmp_path)
    whole_dir = tmp_path / "whole"
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(whole_dir)])
    pieces_dir = whole_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    point_dir = piece_path.parent.parent
    partial = (
        out_dir
        / "pieces"
        / point_dir.name
        / f".{piece_path.parent.name}.0123abcd.partial"
    )
    partial.mkdir(parents=True)
    (partial / "shots.csv").write_text("half a file")

    command.main(["run", str(config_path), "--out", str(out_dir)])

    written = out_dir / "pieces" / point_dir.name / piece_path.parent.name
    whole_run_dir = whole_dir
    whole_rows = _rows_of_every_file(whole_run_dir)
    out_run_dir = out_dir
    out_rows = _rows_of_every_file(out_run_dir)
    assert (written / "piece.json").exists()
    assert out_rows == whole_rows


def test_two_writers_of_one_piece_both_leave_it_whole(tmp_path, monkeypatch):
    """Two tasks handed the same piece write it at once and neither fails.

    The second writer runs whole while the first is between its files
    and its piece.json. Each ends with the piece in place and no staging
    folder left behind.
    """
    task = run_files.first_task()
    measurements = run_files.run_sweep([task], 1)
    experiment_dir = tmp_path / "experiment"
    point_id = task.strong_id()
    write_json = run_folder.write_json

    def second_writer_first(path, value):
        monkeypatch.setattr(run_folder, "write_json", write_json)
        pieces.write(experiment_dir, point_id, 0, measurements, {})
        write_json(path, value)

    monkeypatch.setattr(run_folder, "write_json", second_writer_first)
    folder = pieces.write(experiment_dir, point_id, 0, measurements, {})

    beside = folder.parent.iterdir()
    names = sorted(entry.name for entry in beside)
    assert names == ["0-0"]
    assert (folder / "piece.json").is_file()
    assert (folder / "shots.csv").is_file()


def _shots_to_the_target(shots_path, target: int) -> int:
    """The shots of a prefix up to the one whose failure reaches the target."""
    failures = 0
    for row in fold.row_stream(shots_path):
        failures += row["logical_failure"] == "True"
        if failures == target:
            return int(row["seed"]) + 1
    raise AssertionError("the uncut run never reached the target")


def _shot_count_of(run_dir) -> int:
    sweep_path = run_dir / "sweep.csv"
    (row,) = fold.row_stream(sweep_path)
    return int(row["shots"])


def test_a_collect_stops_on_the_shot_its_target_is_reached(tmp_path):
    """The rule's referent is the uncut run's own failures, seed by seed.

    One collect runs thirty shots with no target. Another cuts the same
    point into pieces of one shot with a target of three failures: it
    starts no piece past the shot of the third failure, so it holds the
    uncut run's shots up to that one and no more.
    """
    whole_card = {"axes": NOISY_AXES, "collection": {"max_shots": 30}}
    whole_path = run_files.write_run_file(tmp_path, **whole_card)
    target_folder = tmp_path / "target_config"
    target_folder.mkdir()
    collection = {"max_shots": 30, "max_failures": 3, "piece_rounds": 1}
    target_path = run_files.write_run_file(
        target_folder, axes=NOISY_AXES, collection=collection
    )
    whole_dir = tmp_path / "whole"
    target_dir = tmp_path / "target"
    command.main(["run", str(whole_path), "--out", str(whole_dir)])
    whole_run_dir = whole_dir
    whole_shots_path = whole_run_dir / "shots.csv"
    expected_shots = _shots_to_the_target(whole_shots_path, 3)

    command.main(["run", str(target_path), "--out", str(target_dir)])

    target_run_dir = target_dir
    target_pieces = target_dir.glob("pieces/*/*")
    assert expected_shots < 30
    assert len(list(target_pieces)) == expected_shots
    assert _shot_count_of(target_run_dir) == expected_shots


# a shot of NOISY_AXES runs 15 rounds, so a piece holds two shots
TWO_SHOT_PIECE_ROUNDS = 30


def test_a_raised_shot_cap_runs_on_from_the_saved_pieces(tmp_path, monkeypatch):
    """The referent is one collect run to the raised cap from the start.

    A cap of 3 in pieces of two ends on a piece of one shot, 2-2. The
    cap raised to 4 runs seed 3 alone beside it, so no seed is run
    twice, and the fold is the uncut run's. The seeds the second
    collect runs are noted as it runs them, since a saved piece's
    folder is never written twice and so cannot show a seed run again.
    """
    whole_dir = tmp_path / "whole"
    raised_dir = tmp_path / "raised"
    whole_run_dir = _collected_noisy_point(
        tmp_path, "whole_config", {"max_shots": 4}, whole_dir
    )
    _collected_noisy_point(tmp_path, "first", {"max_shots": 3}, raised_dir)
    ran_seeds = []
    noting = functools.partial(
        _run_the_unit_and_note, ran_seeds, collect.run_unit
    )
    monkeypatch.setattr(collect, "run_unit", noting)

    raised_run_dir = _collected_noisy_point(
        tmp_path, "second", {"max_shots": 4}, raised_dir
    )

    whole_rows = _rows_of_every_file(whole_run_dir)
    raised_rows = _rows_of_every_file(raised_run_dir)
    assert ran_seeds == [3]
    assert _piece_names(raised_dir) == ["0-1", "2-2", "3-3"]
    assert raised_rows == whole_rows


def test_a_raised_failure_target_runs_on_from_the_saved_pieces(tmp_path):
    """The referent is one collect run to the raised target from the start.

    A point stopped at one failure is collected again for three, as a
    pilot's target is raised for the final run: it runs on from its
    saved pieces and folds to what the three-failure run folds to.
    """
    whole_dir = tmp_path / "whole"
    raised_dir = tmp_path / "raised"
    three = {"max_shots": 30, "max_failures": 3}
    one = {"max_shots": 30, "max_failures": 1}
    whole_run_dir = _collected_noisy_point(
        tmp_path, "whole_config", three, whole_dir
    )
    _collected_noisy_point(tmp_path, "first", one, raised_dir)
    first_pieces = _piece_names(raised_dir)

    raised_run_dir = _collected_noisy_point(
        tmp_path, "second", three, raised_dir
    )

    whole_rows = _rows_of_every_file(whole_run_dir)
    raised_rows = _rows_of_every_file(raised_run_dir)
    raised_pieces = _piece_names(raised_dir)
    assert len(first_pieces) < len(raised_pieces)
    assert raised_rows == whole_rows


def _collected_noisy_point(tmp_path, name: str, collection: dict, out_dir):
    """The NOISY_AXES point collected into out_dir under this collection."""
    folder = tmp_path / name
    folder.mkdir()
    keys = {**collection, "piece_rounds": TWO_SHOT_PIECE_ROUNDS}
    config_path = run_files.write_run_file(
        folder, axes=NOISY_AXES, collection=keys
    )
    command.main(["run", str(config_path), "--out", str(out_dir)])
    return out_dir


def _piece_names(experiment_dir) -> list:
    folders = experiment_dir.glob("pieces/*/*")
    return sorted(folder.name for folder in folders)


def _sweep_row_of(run_dir) -> dict:
    """The one point's sweep.csv row, its cells read as numbers."""
    sweep_path = run_dir / "sweep.csv"
    (row,) = fold.row_stream(sweep_path)
    return fold.typed_row(row)


def test_a_target_stop_reports_its_exact_limits_and_unbiased_estimate(
    tmp_path,
):
    """The target row's referents: scipy's beta, GMS, and sinter.

    At a target stop with r failures in n scored shots the limits are
    the beta quantiles B(0.025; r, n - r + 1) and B(0.975; r, n - r)
    (Jennison and Turnbull, Technometrics 25, 1983), the unbiased
    estimate is (r - 1)/(n - 1) (Girshick, Mosteller and Savage 1946,
    Theorem 3), and a round's rate is sinter's
    shot_error_rate_to_piece_error_rate of the shot's.
    """
    collection = {"max_shots": 30, "max_failures": 3, "piece_rounds": 1}
    config_path = run_files.write_run_file(
        tmp_path, axes=NOISY_AXES, collection=collection
    )
    out_dir = tmp_path / "out"

    command.main(["run", str(config_path), "--out", str(out_dir)])

    run_dir = out_dir
    row = _sweep_row_of(run_dir)
    shots = row["prefix_scored_shots"]
    low_second_shape = shots - 2
    high_second_shape = shots - 3
    low = scipy.stats.beta.ppf(0.025, 3, low_second_shape)
    high = scipy.stats.beta.ppf(0.975, 3, high_second_shape)
    rate = 3 / shots
    per_round = sinter.shot_error_rate_to_piece_error_rate(rate, pieces=15)
    assert row["state"] == "target"
    assert row["prefix_failures"] == 3
    assert row["logical_error_rate_estimate"] == rate
    assert row["logical_error_rate_low"] == pytest.approx(low, rel=1e-12)
    assert row["logical_error_rate_high"] == pytest.approx(high, rel=1e-12)
    assert row["logical_error_rate_plan_unbiased"] == 2 / (shots - 1)
    assert row["logical_error_rate_per_round"] == pytest.approx(
        per_round, rel=1e-9
    )
    assert row["is_shot_rate_above_half"] is False


def test_pieces_past_the_stop_leave_the_estimate_as_the_serial_run_has_it(
    tmp_path,
):
    """A pool may end pieces past the stop; the estimate reads the prefix.

    The serial collect starts no piece past its stop, so its row is the
    referent: the pooled one counts its extra shots in shots and
    nowhere in the prefix, the estimate or the limits.
    """
    collection = {"max_shots": 30, "max_failures": 2, "piece_rounds": 1}
    config_path = run_files.write_run_file(
        tmp_path, axes=NOISY_AXES, collection=collection
    )
    serial_dir = tmp_path / "serial"
    pooled_dir = tmp_path / "pooled"
    command.main(["run", str(config_path), "--out", str(serial_dir)])
    pooled = ["run", str(config_path), "--out", str(pooled_dir)]

    command.main([*pooled, "--processes", "8"])

    serial_run_dir = serial_dir
    pooled_run_dir = pooled_dir
    serial = _sweep_row_of(serial_run_dir)
    pooled_row = _sweep_row_of(pooled_run_dir)
    assert pooled_row["shots"] > serial["shots"]
    assert pooled_row["prefix_shots"] == serial["prefix_shots"]
    assert pooled_row["state"] == serial["state"]
    estimate_columns = (
        "logical_error_rate_estimate",
        "logical_error_rate_low",
        "logical_error_rate_high",
    )
    assert _values_of(pooled_row, estimate_columns) == _values_of(
        serial, estimate_columns
    )


def _values_of(row: dict, columns: tuple) -> tuple:
    values = []
    for column in columns:
        values.append(row[column])
    return tuple(values)


def test_a_point_that_saved_its_stop_starts_no_piece_when_run_again(
    tmp_path, capsys
):
    """A rerun counts the saved pieces, finds the stop, and runs nothing."""
    collection = {"max_shots": 30, "max_failures": 2, "piece_rounds": 1}
    config_path = run_files.write_run_file(
        tmp_path, axes=NOISY_AXES, collection=collection
    )
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    first_pieces = out_dir.glob("pieces/*/*")
    first_names = sorted(first_pieces)
    capsys.readouterr()

    command.main(["run", str(config_path), "--out", str(out_dir)])

    printed = capsys.readouterr()
    second_pieces = out_dir.glob("pieces/*/*")
    second_names = sorted(second_pieces)
    stop_line = f": {len(first_names)} shots done (target)"
    assert second_names == first_names
    assert stop_line in printed.err


def test_a_gap_in_the_saved_pieces_holds_the_stop(tmp_path):
    """The rule reads the contiguous prefix: a piece past a gap waits.

    Seeds 0 and 2 are saved with a failure each and seed 1 is missing.
    Their failures would reach a target of two, but the prefix is seed 0
    alone, so the point runs seed 1 and does not stop.
    """
    task = run_files.first_task()
    point_id = task.strong_id()
    _write_a_saved_piece(tmp_path, point_id, 0, [True])
    _write_a_saved_piece(tmp_path, point_id, 2, [True])
    settings = collection_module.CollectionSettings(max_shots=3, max_failures=2)
    point_folders = pieces.folders_of(tmp_path, [point_id])
    saved = pieces.saved_counts(point_folders)
    point = collect_command.PointCollection("d3", task, settings, 1, 15, saved)

    units = point.next_units(tmp_path, 1)

    assert units == [collect.Unit(task, 1, 1)]
    assert point.tracker.stop_kind is None
    assert point.tracker.counts.failures == 1


def test_a_new_piece_ends_where_the_next_saved_piece_starts(tmp_path):
    """A piece handed out before a saved one stops at that one's first seed.

    Seeds 0 and 2 are saved in pieces of one shot and a piece is three
    shots, so seed 1's piece is one shot, and a piece of three would run
    seed 2 again; the piece after seed 2 is whole.
    """
    task = run_files.first_task()
    point_id = task.strong_id()
    _write_a_saved_piece(tmp_path, point_id, 0, [False])
    _write_a_saved_piece(tmp_path, point_id, 2, [False])
    settings = collection_module.CollectionSettings(max_shots=6)
    point_folders = pieces.folders_of(tmp_path, [point_id])
    saved = pieces.saved_counts(point_folders)
    point = collect_command.PointCollection("d3", task, settings, 3, 15, saved)

    units = point.next_units(tmp_path, 2)

    assert units == [collect.Unit(task, 1, 1), collect.Unit(task, 3, 3)]


def test_a_point_stops_on_the_shot_its_rule_stops_on_inside_a_piece(
    tmp_path, capsys
):
    """The collector stops where the report's prefix does: on a shot.

    One saved piece of four scored shots fails on its second. A target
    of one failure behind a minimum of two scored shots stops on shot
    two, a minimum stop, which is what the report reads off the same
    rows (collection.PrefixTracker); the piece's other two shots ran
    past the stop, and the progress line says so.
    """
    task = run_files.first_task()
    point_id = task.strong_id()
    failed = [False, True, False, False]
    _write_a_saved_piece(tmp_path, point_id, 0, failed)
    settings = collection_module.CollectionSettings(
        max_shots=10, max_failures=1, min_shots=2
    )
    point_folders = pieces.folders_of(tmp_path, [point_id])
    saved = pieces.saved_counts(point_folders)
    point = collect_command.PointCollection("d3", task, settings, 4, 15, saved)

    units = point.next_units(tmp_path, 1)

    printed = capsys.readouterr()
    assert units == []
    assert point.tracker.stop_kind is failure_statistics.StopKind.MINIMUM
    assert point.tracker.counts.shots == 2
    assert (
        printed.err == "d3: 2 shots done (minimum); 2 more ran past the stop\n"
    )


def test_a_stopped_point_says_so_once(tmp_path, capsys):
    """The collector counts every point after each share; one line a stop."""
    task = run_files.first_task()
    point_id = task.strong_id()
    _write_a_saved_piece(tmp_path, point_id, 0, [True, False])
    settings = collection_module.CollectionSettings(max_shots=2)
    point_folders = pieces.folders_of(tmp_path, [point_id])
    saved = pieces.saved_counts(point_folders)
    point = collect_command.PointCollection("d3", task, settings, 2, 15, saved)
    point.next_units(tmp_path, 1)
    capsys.readouterr()

    point.count_the_pending(tmp_path)

    printed = capsys.readouterr()
    assert printed.err == ""


def _write_a_saved_piece(
    experiment_dir, point_id: str, first_seed: int, failed: list
):
    """A saved piece of scored shots, failed or not, as pieces.write lays it."""
    count = len(failed)
    folder = pieces.piece_dir(experiment_dir, point_id, first_seed, count)
    folder.mkdir(parents=True)
    lines = [
        "seed,is_scored,logical_failure,sim_wall_seconds,scored_outputs,"
        "rounds_per_output"
    ]
    for offset, is_failure in enumerate(failed):
        seed = first_seed + offset
        lines.append(f"{seed},True,{is_failure},0.5,1,15")
    shots_text = "\n".join(lines) + "\n"
    (folder / "shots.csv").write_text(shots_text)
    (folder / pieces.PIECE_FILE).write_text("{}")


def test_a_collect_of_no_processes_is_refused_before_its_folder(
    tmp_path, capsys
):
    """Zero processes would deal no piece and finish having run nothing."""
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit):
        command.main(
            [
                "run",
                str(config_path),
                "--out",
                str(out_dir),
                "--processes=0",
            ]
        )

    printed = capsys.readouterr()
    assert "processes must be a whole number of at least 1, got 0" in (
        printed.err
    )
    assert not out_dir.exists()


def test_an_online_point_given_a_target_is_refused(tmp_path, capsys):
    """An online point's shots are not independent, so no target stops it."""
    collection = {"max_shots": 1, "max_failures": 5}
    config_path = run_files.write_run_file(
        tmp_path, collection=collection, **ONLINE
    )
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit):
        command.main(["run", str(config_path), "--out", str(out_dir)])

    printed = capsys.readouterr()
    lines = printed.err.splitlines()
    assert "calibrates its threshold online" in lines[-1]
    saved_pieces = out_dir.glob("pieces/*/*")
    assert not list(saved_pieces)


def test_an_online_point_cut_and_resumed_is_the_uncut_point(tmp_path):
    """The referent is the uncut collect: one piece of all four shots.

    Cut into pieces of one shot, each piece starts from the calibrator
    state the last one saved. Two pieces are lost, as a killed job
    leaves them, and the collect run again starts from the state of
    the last saved piece: the threshold's trajectory and every shot's
    decisions and outcome are the uncut run's. The pymatching units
    price their latency from measured wall clock, so the timing columns
    vary between any two runs and are not compared.
    """
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    whole_config = _online_config(tmp_path, 60)
    command.main(["run", str(whole_config), "--out", str(whole_dir)])
    cut_config = _online_config(tmp_path, 15)
    command.main(["run", str(cut_config), "--out", str(cut_dir)])
    _lose_the_piece(cut_dir, "2-2")
    _lose_the_piece(cut_dir, "3-3")

    command.main(["run", str(cut_config), "--out", str(cut_dir)])

    whole_run_dir = whole_dir
    cut_run_dir = cut_dir
    whole_decisions = _shot_decisions(whole_run_dir)
    cut_decisions = _shot_decisions(cut_run_dir)
    whole_trajectory = _online_trajectory_rows(whole_run_dir)
    cut_trajectory = _online_trajectory_rows(cut_run_dir)
    assert _piece_names(cut_dir) == ["0-0", "1-1", "2-2", "3-3"]
    assert cut_decisions == whole_decisions
    assert whole_trajectory[-1]["window_count"] == "20"
    assert cut_trajectory == whole_trajectory


def test_an_online_points_threshold_summary_is_its_end_counters(tmp_path):
    """threshold_summary.csv holds the counters the stderr line prints.

    The referents are files the run writes apart from the calibrator's
    summary: the machine counts every weak decision as a decoded window
    and every audit as an escalated one, since an audit sends a kept
    window to the strong decoder, and the trajectory records each
    audit's label and the final threshold.
    """
    config_path = _online_config(tmp_path, 60)
    out_dir = tmp_path / "out"

    command.main(["run", str(config_path), "--out", str(out_dir)])

    summary_path = out_dir / "threshold_summary.csv"
    sweep_path = out_dir / "sweep.csv"
    shots_path = out_dir / "shots.csv"
    (summary,) = _csv_rows(summary_path)
    (sweep_row,) = _csv_rows(sweep_path)
    shots = _csv_rows(shots_path)
    trajectory = _online_trajectory_rows(out_dir)
    decoded = sum(int(shot["decoded_windows"]) for shot in shots)
    escalated = sum(int(shot["escalated_windows"]) for shot in shots)
    events = [row["event"] for row in trajectory]
    labeled = events.count("audit_clean") + events.count("audit_bad")
    audits_started = int(summary["audited"]) + int(summary["pending_audits"])
    columns = list(summary)
    sweep_columns = list(sweep_row)
    assert columns[:5] == sweep_columns[:5]
    assert summary["point_id"] == sweep_row["point_id"]
    assert int(summary["windows"]) == decoded == 20
    assert int(summary["escalated"]) + audits_started == escalated
    assert int(summary["audited"]) == labeled == 1
    assert int(summary["audited_bad"]) == events.count("audit_bad")
    assert summary["threshold_db"] == trajectory[-1]["threshold_db"]


def test_a_saved_calibrator_whose_bytes_changed_is_refused(tmp_path):
    """A state that no longer hashes to its record is never unpickled.

    The piece's calibrator reads back as saved; one byte appended to its
    file, as a damaged copy on a shared disk would hold, and the read
    refuses it by name.
    """
    task = run_files.first_task(**ONLINE)
    measurements = run_files.run_sweep([task], 1)
    experiment_dir = tmp_path / "experiment"
    point_id = task.strong_id()
    calibrator = task.online_threshold
    folder = pieces.write(
        experiment_dir, point_id, 0, measurements, {}, calibrator
    )
    saved = pieces.read_state(folder)
    state_path = folder / pieces.STATE_FILE
    damaged_bytes = state_path.read_bytes() + b"\x00"
    state_path.write_bytes(damaged_bytes)

    with pytest.raises(refusal.RefusalError) as refused:
        pieces.read_state(folder)

    message = str(refused.value)
    assert saved.summary() == calibrator.summary()
    assert message.startswith(f"{state_path} does not hash")


def _lose_the_piece(experiment_dir, name: str) -> None:
    """The piece gone, as a killed job leaves it missing."""
    (lost_piece,) = experiment_dir.glob(f"pieces/*/{name}")
    shutil.rmtree(lost_piece)


def _online_config(tmp_path, piece_rounds: int):
    """The noisy point with an online threshold that audits often.

    Written over the same file each time; the collect that read it has
    run by then.
    """
    online = {
        "audit_rate": 0.3,
        "target_escalation_rate": 0.2,
        "max_escalation_rate": 0.5,
    }
    collection = {"max_shots": 4, "piece_rounds": piece_rounds}
    return run_files.write_run_file(
        tmp_path,
        axes=NOISY_AXES,
        collection=collection,
        machine="switching",
        machine_arguments={"online": online},
    )


# the shot columns no wall clock prices
DECISION_COLUMNS = (
    "seed",
    "sample_digest",
    "decoded_windows",
    "escalated_windows",
    "strong_decoded_rounds",
    "is_scored",
    "logical_failure",
)


def _shot_decisions(run_dir) -> list:
    shots_path = run_dir / "shots.csv"
    decisions = []
    for row in _csv_rows(shots_path):
        values = _values_of(row, DECISION_COLUMNS)
        decisions.append(values)
    return decisions


def _online_trajectory_rows(run_dir) -> list:
    (trajectory_path,) = run_dir.glob("online_threshold_*.csv")
    return _csv_rows(trajectory_path)


def test_every_file_names_a_point_by_its_id_then_its_swept_values(tmp_path):
    """The values the design fixed first, then what was measured.

    Wickham's tidy order (Tidy Data, J. Stat. Softw. 59(10), 2014,
    section 2.3): one column per path the sweep sets, right after
    the point id, in every file a point's rows are in.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "out"

    command.main(["run", str(config_path), "--out", str(out_dir)])

    run_dir = out_dir
    sweep_path = run_dir / "sweep.csv"
    shots_path = run_dir / "shots.csv"
    links_path = run_dir / "shot_links.csv"
    sweep_header = fold.header_of(sweep_path)
    shots_header = fold.header_of(shots_path)
    links_header = fold.header_of(links_path)
    sweep_rows = _csv_rows(sweep_path)
    first_columns = [
        "point_id",
        "qpu.distance",
        "qpu.round_period_microseconds",
        "workload.arguments.physical_error_probability",
        "algorithm",
    ]
    error_rate = "workload.arguments.physical_error_probability"
    points = {(row["qpu.distance"], row[error_rate]) for row in sweep_rows}
    assert sweep_header[:5] == first_columns
    assert shots_header[:5] == first_columns
    assert links_header[:5] == first_columns
    assert points == {
        ("3", "0.001"),
        ("5", "0.001"),
        ("3", "0.003"),
        ("5", "0.003"),
    }


def test_a_manifest_names_every_installed_package(tmp_path):
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    _run_one_shot(config_path, seed=0, out_dir=out_dir)
    manifest = _manifest_of(out_dir)
    packages = manifest["versions"]["packages"]

    assert packages["stim"] == stim.__version__
    assert "numpy" in packages


def test_a_manifest_names_the_library_a_loader_loads(
    tmp_path, monkeypatch, compiled_union_find_library
):
    """A library is built, not tracked, so the commit does not name it.

    The referent is the file the suite built, hashed here, keyed by its
    absolute path.
    """
    monkeypatch.delenv(compiled_decoder.LIBRARY_VARIABLE, raising=False)
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    _run_one_shot(config_path, seed=0, out_dir=out_dir)
    manifest = _manifest_of(out_dir)
    union_find_path = compiled_union_find_library.resolve()
    expected = {str(union_find_path): _sha256_of(union_find_path)}

    assert manifest["compiled_libraries"] == expected


@pytest.mark.parametrize(
    ("library_bytes", "expected_digest"),
    [
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (None, None),
    ],
)
def test_a_manifest_names_a_library_loaded_from_outside_the_package(
    tmp_path, monkeypatch, library_bytes, expected_digest
):
    """The bytes the loader's own environment names, and none when unbuilt.

    The stand-in library's bytes are "abc", whose sha256 is FIPS 180-2's
    first example.
    """
    library_path = tmp_path / "elsewhere" / "union_find.so"
    _write_library(library_path, library_bytes)
    monkeypatch.setenv(compiled_decoder.LIBRARY_VARIABLE, str(library_path))
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"

    _run_one_shot(config_path, seed=0, out_dir=out_dir)

    manifest = _manifest_of(out_dir)
    expected = _named_library(library_path, expected_digest)
    assert manifest["compiled_libraries"] == expected


def test_a_run_file_with_other_points_is_refused_in_the_folder(tmp_path):
    """A folder's copy of its run file is the one that made its rows.

    A grid split one file a distance names other points in each file,
    so the second file into the first one's folder is refused and the
    folder keeps its copy.
    """
    three_path = _one_distance_config(tmp_path, 3)
    five_path = _one_distance_config(tmp_path, 5)
    out_dir = tmp_path / "out"
    collect_command.run_experiment(three_path, out_dir)
    copy_path = out_dir / three_path.name
    copy_text = copy_path.read_text()

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(five_path, out_dir)

    assert "holds another unit_test_config.py" in str(refused.value)
    assert copy_path.read_text() == copy_text


def _one_distance_config(tmp_path, distance: int) -> pathlib.Path:
    """The four-point sweep at one distance, in a folder of its own."""
    folder = tmp_path / f"d{distance}"
    folder.mkdir()
    axes = {**run_files.FOUR_POINT_AXES, run_files.DISTANCE_PATH: (distance,)}
    collection = {"max_shots": 2, "piece_rounds": 1}
    return run_files.write_run_file(folder, axes=axes, collection=collection)


def test_a_raised_stop_rule_replaces_the_folders_run_file_copy(tmp_path):
    """The same points run further: the copy is the file that ran last."""
    out_dir = tmp_path / "out"
    _collected_noisy_point(tmp_path, "first", {"max_shots": 2}, out_dir)

    _collected_noisy_point(tmp_path, "second", {"max_shots": 3}, out_dir)

    second_path = tmp_path / "second" / "unit_test_config.py"
    copy_path = out_dir / "unit_test_config.py"
    assert copy_path.read_text() == second_path.read_text()


def test_a_shot_replayed_into_a_collection_keeps_its_records(tmp_path):
    """The collection's machine.json and run.json points stay as written.

    The collection's record holds what its fold needs (experiment), which
    a one-shot record does not, so a replay must not write over it.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "out"
    collect_command.run_experiment(config_path, out_dir)
    point_records = out_dir.glob("points/*/machine.json")
    record_paths = sorted(point_records)
    records_before = [path.read_bytes() for path in record_paths]
    manifest_before = _manifest_of(out_dir)
    points_before = manifest_before["points"]

    _run_one_shot(config_path, seed=0, out_dir=out_dir)

    records_after = [path.read_bytes() for path in record_paths]
    assert records_after == records_before
    manifest_after = _manifest_of(out_dir)
    assert manifest_after["points"] == points_before
    assert (out_dir / "result.json").exists()


def test_a_replay_of_a_point_recorded_with_other_settings_is_refused(
    tmp_path,
):
    """A saved record of another id under the name is another machine."""
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "out"
    collect_command.run_experiment(config_path, out_dir)
    study = experiment.load_one_point(config_path)
    (point,) = study.points
    record_path = out_dir / "points" / point.name / "machine.json"
    record = run_folder.read_json(record_path)
    record["id"] = "0" * 64
    record_text = json.dumps(record)
    record_path.write_text(record_text)

    with pytest.raises(refusal.RefusalError) as refused:
        _run_one_shot(config_path, seed=0, out_dir=out_dir)

    assert "with other settings or metadata" in str(refused.value)
    assert not (out_dir / "result.json").exists()


def test_a_record_class_names_settings_and_not_results(tmp_path):
    """machine.json names each settings record's class; result.json none.

    The class tells two records with the same fields apart, which is a
    setting's identity; a result's fields are all it holds.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "out"

    _run_one_shot(config_path, seed=0, out_dir=out_dir)

    (record_path,) = out_dir.glob("points/*/machine.json")
    record = run_folder.read_json(record_path)
    result_text = (out_dir / "result.json").read_text()
    assert record["settings"]["class"] == "decsim.settings.MachineSettings"
    assert '"class"' not in result_text


def test_a_run_into_a_folder_of_another_commit_is_refused(tmp_path, capsys):
    """A folder's rows pool every run into it, so they ran one tree.

    run.json is rewritten as a run at another commit leaves it, and the
    next run into the folder is refused before it writes anything,
    naming the commit the folder holds.
    """
    run_file = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    command.main(["run", str(run_file), "--out", str(out_dir)])
    record = _manifest_of(out_dir)
    other_commit = "b" * 40
    record["git"]["commit"] = other_commit
    run_path = out_dir / run_folder.RUN_FILE
    record_text = json.dumps(record)
    run_path.write_text(record_text)
    capsys.readouterr()

    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(run_file), "--out", str(out_dir)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert f"holds a run of commit {other_commit}" in printed.err
    assert run_path.read_text() == record_text


def test_two_dirty_trees_of_one_commit_are_two_trees(
    tmp_path, capsys, monkeypatch
):
    """A dirty tree is named by its commit and its patch's sha256.

    One run file runs from two trees at one commit with other
    uncommitted changes, each into its own folder, whose run.json names
    the sha256 of its code_state.patch. The second tree's run into the
    first folder is refused, and so is the fold of a point whose pieces
    the two trees ran, as a piece copied in by hand leaves it.
    """
    config_path = _capped_noisy_config(tmp_path, 2, 15)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "1")
    _as_a_tree_with_the_patch(monkeypatch, "first edit\n")
    command.main(["run", str(config_path), "--out", str(first_dir)])
    _as_a_tree_with_the_patch(monkeypatch, "second edit\n")
    command.main(["run", str(config_path), "--out", str(second_dir)])
    capsys.readouterr()

    with pytest.raises(SystemExit):
        command.main(["run", str(config_path), "--out", str(first_dir)])
    run_captured = capsys.readouterr()
    run_refusal = run_captured.err
    (first_piece,) = first_dir.glob("pieces/*/1-1")
    (second_piece,) = second_dir.glob("pieces/*/1-1")
    shutil.rmtree(first_piece)
    shutil.copytree(second_piece, first_piece)
    with pytest.raises(SystemExit):
        command.main(["run", "--fold", "--out", str(first_dir)])
    fold_captured = capsys.readouterr()
    fold_refusal = fold_captured.err

    first_manifest = _manifest_of(first_dir)
    second_manifest = _manifest_of(second_dir)
    first_git = first_manifest["git"]
    second_git = second_manifest["git"]
    first_patch = (first_dir / "code_state.patch").read_bytes()
    first_patch_hash = hashlib.sha256(first_patch)
    first_patch_sha256 = first_patch_hash.hexdigest()
    assert first_git["patch_sha256"] == first_patch_sha256
    assert first_git["commit"] == second_git["commit"]
    assert first_git["patch_sha256"] != second_git["patch_sha256"]
    assert f"patch {first_patch_sha256}" in run_refusal
    assert f"{first_piece} ran commit" in fold_refusal


def test_a_run_onto_pieces_another_tree_saved_is_refused_before_a_shot(
    tmp_path, capsys
):
    """A capped point whose saved pieces all ran another tree.

    They reach its cap, so a run again would skip them as done and fold
    them under this folder's run.json as this tree's. The run is refused
    before it writes anything, and the folder is as the first run left
    it.
    """
    config_path = _capped_noisy_config(tmp_path, 2, 15)
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    other_commit = "b" * 40
    piece_paths = out_dir.glob("pieces/*/*/piece.json")
    _as_pieces_run_at_commit(piece_paths, other_commit)
    run_path = out_dir / run_folder.RUN_FILE
    record_bytes = run_path.read_bytes()
    before = _run_folder_bytes(out_dir)
    capsys.readouterr()

    with pytest.raises(SystemExit):
        command.main(["run", str(config_path), "--out", str(out_dir)])

    printed = capsys.readouterr()
    assert f"ran commit {other_commit}" in printed.err
    assert run_path.read_bytes() == record_bytes
    assert _run_folder_bytes(out_dir) == before


def test_a_fold_of_two_points_saved_by_two_trees_is_refused(tmp_path, capsys):
    """Every piece of a folder ran the tree its run.json names.

    One point's pieces all ran another tree, which agrees with itself
    point by point, and the fold still refuses, publishing nothing.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    point_dirs = out_dir.glob("pieces/*")
    first_point_dir = min(point_dirs)
    other_commit = "b" * 40
    piece_paths = first_point_dir.glob("*/piece.json")
    _as_pieces_run_at_commit(piece_paths, other_commit)
    before = _run_folder_bytes(out_dir)
    capsys.readouterr()

    with pytest.raises(SystemExit):
        command.main(["run", "--fold", "--out", str(out_dir)])

    printed = capsys.readouterr()
    assert f"{first_point_dir}" in printed.err
    assert f"ran commit {other_commit}" in printed.err
    assert _run_folder_bytes(out_dir) == before


@pytest.mark.parametrize("record", ["run.json", "piece.json"])
def test_a_record_with_no_patch_hash_stops_a_run_before_a_shot(
    tmp_path, record
):
    """A record from before the patch hash cannot name its tree.

    Two shots are saved and the record loses its patch_sha256, as a
    record an older tree wrote has none; the run again under a raised
    cap stops on the missing key and runs no shot.
    """
    out_dir = tmp_path / "out"
    first_config = _capped_noisy_config(tmp_path, 2, 15)
    command.main(["run", str(first_config), "--out", str(out_dir)])
    piece_paths = out_dir.glob("pieces/*/*/piece.json")
    records = {
        "run.json": [out_dir / run_folder.RUN_FILE],
        "piece.json": list(piece_paths),
    }
    for record_path in records[record]:
        written = run_folder.read_json(record_path)
        identity = written.get("git", written)
        del identity["patch_sha256"]
        run_folder.write_json(record_path, written)
    raised_config = _capped_noisy_config(tmp_path, 3, 15)

    with pytest.raises(KeyError, match="patch_sha256"):
        command.main(["run", str(raised_config), "--out", str(out_dir)])

    assert _piece_names(out_dir) == ["0-0", "1-1"]


def test_a_fold_of_a_folder_whose_run_json_has_no_patch_hash_stops(
    tmp_path,
):
    """With no piece to compare, an older run.json still stops a fold.

    The pieces are gone, as before a first piece is saved, and run.json
    loses its patch_sha256, as one an older tree wrote has none. The
    fold stops on the missing key and publishes nothing.
    """
    out_dir = tmp_path / "out"
    config_path = _capped_noisy_config(tmp_path, 2, 15)
    command.main(["run", str(config_path), "--out", str(out_dir)])
    pieces_dir = out_dir / pieces.PIECES_FOLDER
    shutil.rmtree(pieces_dir)
    run_path = out_dir / run_folder.RUN_FILE
    record = run_folder.read_json(run_path)
    del record["git"]["patch_sha256"]
    run_folder.write_json(run_path, record)
    before = _run_folder_bytes(out_dir)

    with pytest.raises(KeyError, match="patch_sha256"):
        command.main(["run", "--fold", "--out", str(out_dir)])

    assert _run_folder_bytes(out_dir) == before


def _as_pieces_run_at_commit(piece_paths, commit: str) -> None:
    """The pieces as a process at another commit would have saved them."""
    for piece_path in piece_paths:
        piece = run_folder.read_json(piece_path)
        piece["commit"] = commit
        run_folder.write_json(piece_path, piece)


def _as_a_tree_with_the_patch(monkeypatch, patch_text: str) -> None:
    """The tree read again, as one whose uncommitted code is patch_text."""
    monkeypatch.setattr(
        run_folder, "_code_state_patch", lambda _checkout: patch_text
    )
    run_folder._tree_reading.cache_clear()


def test_a_new_folder_from_a_tree_with_no_commit_is_refused(
    tmp_path, monkeypatch
):
    """A run whose commit cannot be read could not say what ran."""
    monkeypatch.delenv(run_folder.ALLOW_DIRTY_VARIABLE)
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: None)
    monkeypatch.setattr(run_folder, "_commit_from_git_files", lambda _: None)
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    refused_dir = tmp_path / "refused"
    allowed_dir = tmp_path / "allowed"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, refused_dir)
    monkeypatch.setenv(run_folder.ALLOW_DIRTY_VARIABLE, "1")
    collect_command.run_experiment(config_path, allowed_dir)

    assert "cannot be read" in str(refused.value)
    assert not (refused_dir / "run.json").exists()
    assert (allowed_dir / "run.json").exists()


def test_a_run_where_git_cannot_answer_records_no_patch(tmp_path, monkeypatch):
    """The container the suite runs in ships no git."""
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: None)
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run_folder.snapshot_code_state(config_path, out_dir)

    assert not (out_dir / "code_state.patch").exists()


def _git_tree_with_maker(folder: pathlib.Path) -> pathlib.Path:
    """A new git tree holding an uncommitted maker.py."""
    folder.mkdir()
    subprocess.run(["git", "init", "-q", str(folder)], check=True)
    (folder / "maker.py").write_text("VALUE = 1\n")
    return folder


@requires_git
def test_an_uncommitted_edit_is_in_the_code_patch(tmp_path, monkeypatch):
    """The patch carries the tracked file's edit beside the untracked file."""
    tree_folder = tmp_path / "tree"
    tree = _git_tree_with_maker(tree_folder)
    replay_folder = tmp_path / "replay"
    replay = _git_tree_with_maker(replay_folder)
    commit = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit"]
    subprocess.run(["git", "-C", str(tree), "add", "maker.py"], check=True)
    subprocess.run([*commit, "-q", "-m", "one"], cwd=tree, check=True)
    (tree / "maker.py").write_text("VALUE = 2\n")
    monkeypatch.setattr(run_folder, "_checkout", lambda: tree)
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run_folder.snapshot_code_state(config_path, out_dir)

    patch_path = out_dir / "code_state.patch"
    apply = ["git", "-C", str(replay), "apply", str(patch_path)]
    subprocess.run(apply, check=True)
    assert (replay / "maker.py").read_text() == "VALUE = 2\n"


@requires_git
def test_an_untracked_file_is_in_the_code_patch(tmp_path, monkeypatch):
    """The patch creates the file, so commit plus patch is the code that ran."""
    tree = tmp_path / "tree"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    new_module = tree / "maker.py"
    new_module.write_text("VALUE = 1\n")
    monkeypatch.setattr(run_folder, "_checkout", lambda: tree)
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_folder.snapshot_code_state(config_path, out_dir)
    replay = tmp_path / "replay"
    subprocess.run(["git", "init", "-q", str(replay)], check=True)
    patch_path = out_dir / "code_state.patch"
    apply = ["git", "-C", str(replay), "apply", str(patch_path)]
    subprocess.run(apply, check=True)

    assert (replay / "maker.py").read_text() == "VALUE = 1\n"


@requires_git
def test_a_results_folder_in_the_checkout_leaves_the_tree_clean(
    tmp_path, monkeypatch
):
    """A run's own run.json and run file copy are results, not code.

    A planned batch writes them inside the checkout before its tasks
    start, and every task refuses a dirty tree, so they must not read
    as uncommitted code; an edit to the code still does.
    """
    tree_folder = tmp_path / "tree"
    tree = _git_tree_with_maker(tree_folder)
    commit = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit"]
    subprocess.run(["git", "-C", str(tree), "add", "maker.py"], check=True)
    subprocess.run([*commit, "-q", "-m", "one"], cwd=tree, check=True)
    results_folder = tree / "results" / "2026-10-02_sweep"
    results_folder.mkdir(parents=True)
    (results_folder / "run.json").write_text("{}\n")
    monkeypatch.setattr(run_folder, "_checkout", lambda: tree)

    _checkout, _commit, with_results = run_folder.fresh_tree_reading()
    (tree / "maker.py").write_text("VALUE = 2\n")
    _checkout, _commit, with_an_edit = run_folder.fresh_tree_reading()

    assert with_results is False
    assert with_an_edit is True


def test_a_pooled_collect_writes_the_serial_collects_rows(tmp_path):
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    serial_dir = tmp_path / "serial"
    pooled_dir = tmp_path / "pooled"
    command.main(["run", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "run",
            str(config_path),
            "--out",
            str(pooled_dir),
            "--processes",
            "4",
        ]
    )
    serial_run_dir = serial_dir
    serial_rows = _rows_of_every_file(serial_run_dir)
    pooled_run_dir = pooled_dir
    pooled_rows = _rows_of_every_file(pooled_run_dir)
    assert pooled_rows == serial_rows


def test_pieces_of_one_shot_fold_to_the_rows_of_one_piece_a_point(tmp_path):
    """The additive record's whole point: a folded point is the point.

    One collect saves each point as one piece; another cuts each point
    into pieces of one shot, serially and in a pool. Every file the
    three fold to is the same, row for row, but for the wall clock.
    """
    whole_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_path = run_files.write_run_file(cut_folder, **FOUR_POINTS_CUT)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    pooled_dir = tmp_path / "pooled"
    command.main(["run", str(whole_path), "--out", str(whole_dir)])
    command.main(["run", str(cut_path), "--out", str(cut_dir)])
    pooled = ["run", str(cut_path), "--out", str(pooled_dir)]
    command.main([*pooled, "--processes", "4"])

    whole_pieces = whole_dir.glob("pieces/*/*")
    cut_pieces = cut_dir.glob("pieces/*/*")
    whole_run_dir = whole_dir
    whole_rows = _rows_of_every_file(whole_run_dir)
    cut_run_dir = cut_dir
    cut_rows = _rows_of_every_file(cut_run_dir)
    pooled_run_dir = pooled_dir
    pooled_rows = _rows_of_every_file(pooled_run_dir)
    assert len(list(whole_pieces)) == 4
    assert len(list(cut_pieces)) == 8
    assert cut_rows == whole_rows
    assert pooled_rows == whole_rows


def test_a_manifest_names_the_commit_of_the_tree_it_imported(
    tmp_path, monkeypatch
):
    """A folder names its code, whatever folder the job ran from.

    A cluster task starts in the folder its job was submitted from and
    may import a checkout pinned somewhere else, so the manifest reads
    the tree decsim came from. Here the run is made from a directory
    that is no checkout at all, and the commit is still the one this
    test's own tree is at.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "run"
    monkeypatch.chdir(tmp_path)
    command.main(["run", str(config_path), "--out", str(out_dir)])
    run_dir = out_dir
    manifest = _manifest_of(run_dir)
    recorded = manifest["git"]

    assert recorded["commit"] == _commit_of_this_tree()
    assert "dirty" in recorded


def test_a_manifest_takes_the_dirty_flag_from_the_launcher_that_looked(
    tmp_path, monkeypatch
):
    """The interpreter may have no git; the launcher did.

    A job script looks at the tree as the job starts and exports what it
    saw, because the container image this runs in ships no git binary
    and a folder that cannot say whether its code was committed must say
    that rather than say clean.

    A process reads the tree once, so the two runs here take a reading
    each on purpose: one process launched by two different launchers is
    the test bench's own shape and never a cluster job's.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    dirty_dir = tmp_path / "dirty"
    clean_dir = tmp_path / "clean"
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "1")
    run_folder._tree_reading.cache_clear()
    command.main(["run", str(config_path), "--out", str(dirty_dir)])
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "0")
    run_folder._tree_reading.cache_clear()
    command.main(["run", str(config_path), "--out", str(clean_dir)])

    dirty_run_dir = dirty_dir
    clean_run_dir = clean_dir
    dirty_manifest = _manifest_of(dirty_run_dir)
    clean_manifest = _manifest_of(clean_run_dir)
    dirty = dirty_manifest["git"]
    clean = clean_manifest["git"]
    assert dirty["dirty"] is True
    assert clean["dirty"] is False


def test_both_manifests_of_a_run_name_the_tree_it_started_on(
    tmp_path, monkeypatch
):
    """A run writes its manifest twice and both name one reading.

    A tree committed to while an array runs would give a task's second
    reading another commit than the code it imported. Here git answers
    one commit for the first write and another for the second, and both
    manifests name the first, which is the code the run imported.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    out_dir = tmp_path / "run"
    started_utc = run_folder.utc_now()
    out_dir.mkdir()
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "0")
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: "aaaaaaa")
    run_folder._tree_reading.cache_clear()

    run_folder.write_run_record(out_dir, config_path, [], started_utc)
    at_the_start = _manifest_of(out_dir)
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "1")
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: "bbbbbbb")
    finished_utc = run_folder.utc_now()
    run_folder.write_run_record(
        out_dir, config_path, [], started_utc, finished_utc=finished_utc
    )
    at_the_end = _manifest_of(out_dir)

    assert at_the_start["git"]["commit"] == "aaaaaaa"
    assert at_the_start["git"]["dirty"] is False
    assert at_the_end["git"] == at_the_start["git"]
    assert at_the_end["finished_utc"] is not None


def test_two_pieces_that_hold_the_same_shot_are_refused(tmp_path, capsys):
    """A piece copied in under another range would count its shots twice."""
    config_path = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    piece_dir = piece_path.parent
    copied_dir = piece_dir.with_name("5-5")
    shutil.copytree(piece_dir, copied_dir)
    capsys.readouterr()

    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(config_path), "--out", str(out_dir)])

    printed = capsys.readouterr()
    lines = printed.err.splitlines()
    assert stopped.value.code == 1
    assert "is in more than one of" in lines[-1]


def test_run_refuses_a_run_file_that_is_not_there(tmp_path, capsys):
    missing = tmp_path / "not_a_run_file.py"
    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(missing)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(f"decsim: {missing} is not a file")
    assert "the shipped experiments are" in printed.err


def test_a_run_with_no_folder_writes_a_new_dated_one(tmp_path, monkeypatch):
    """results/<date>_<experiment>/, and a second run that day gets _2."""
    run_file = run_files.write_run_file(tmp_path, **run_files.REFERENCE)
    monkeypatch.chdir(tmp_path)

    first_dir, _rows = collect_command.run_experiment(run_file)
    second_dir, _rows = collect_command.run_experiment(run_file)

    today = datetime.date.today()
    day = today.isoformat()
    expected = pathlib.Path("results") / f"{day}_reference"
    assert first_dir == expected
    assert second_dir == expected.with_name(f"{expected.name}_2")
    assert (first_dir / "sweep.csv").is_file()


@pytest.mark.parametrize("shots", [0, -1, 1.5, "many", True])
def test_a_shot_cap_that_is_not_a_whole_number_of_one_or_more_is_refused(
    shots,
):
    refusal_text = "collection max_shots must be a whole number of at least 1"

    with pytest.raises(ValueError, match=refusal_text):
        collection_module.CollectionSettings(max_shots=shots)


def test_trace_refuses_an_action_it_does_not_have(tmp_path, capsys):
    trace_path = tmp_path / "shot.json"
    with pytest.raises(SystemExit) as stopped:
        command.main(["trace", "summarise", str(trace_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: decsim trace has no action")


def _write_library(path: pathlib.Path, contents) -> None:
    """A stand-in library holding the bytes, or no file for None."""
    path.parent.mkdir(parents=True)
    if contents is not None:
        path.write_bytes(contents)


def _named_library(path: pathlib.Path, digest) -> dict:
    """The manifest's entry for one library, or none for no digest."""
    if digest is None:
        return {}
    absolute = path.resolve()
    return {str(absolute): digest}


@pytest.mark.parametrize(
    "flag, value", [("--cores", "0"), ("--hours", "0"), ("--memory-mb", "-1")]
)
def test_a_slurm_task_asking_for_no_core_hour_or_memory_is_refused(
    tmp_path, capsys, flag, value
):
    """A task has a core, an hour and some memory, or nothing is written."""
    run_file = run_files.write_run_file(tmp_path)
    out_dir = tmp_path / "out"
    arguments = [str(run_file), "--out", str(out_dir), "--slurm", "--dry-run"]

    with pytest.raises(SystemExit):
        command.main(["run", *arguments, flag, value])

    printed = capsys.readouterr()
    assert f"{flag} must be at least 1, got {value}" in printed.err
    assert not out_dir.exists()


def test_array_tasks_then_the_fold_write_the_local_runs_rows(tmp_path):
    """The referent is one local run of the same four points.

    The launcher records the points; each array task then runs the line
    run.sbatch holds, in whatever order the array runs them, and the fold
    job's fold writes every file the local run wrote, row for row but the
    wall clock. The run file reads a file beside it, as a threshold table
    is read, and every task finds it.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    _read_a_file_beside(config_path)
    local_dir = tmp_path / "local"
    split_dir = tmp_path / "split"
    command.main(["run", str(config_path), "--out", str(local_dir)])
    job = plan_command.JobShape(cores=1, hours=1, memory_mb=1024)
    plan_command.launch(config_path, split_dir, job, dry_run=True)

    for index in ("3", "1", "0", "2"):
        task = _task_arguments(split_dir, index)
        command.main(task)
    command.main(["run", "--fold", "--out", str(split_dir)])

    assert _rows_of_every_file(split_dir) == _rows_of_every_file(local_dir)


def test_an_array_task_refuses_a_run_file_edited_since_the_launch(
    tmp_path, capsys
):
    """Task i of an edited run file may name another point, so it stops."""
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    split_dir = tmp_path / "split"
    job = plan_command.JobShape(cores=1, hours=1, memory_mb=1024)
    plan_command.launch(config_path, split_dir, job, dry_run=True)
    with config_path.open("a") as run_file:
        run_file.write("# edited after the launch\n")
    task = _task_arguments(split_dir, "0")

    with pytest.raises(SystemExit):
        command.main(task)

    printed = capsys.readouterr()
    saved_pieces = split_dir.glob("pieces/*")
    assert f"holds another {config_path.name} than" in printed.err
    assert not any(saved_pieces)


def _read_a_file_beside(run_path: pathlib.Path) -> None:
    """The run file made to read a file written beside it."""
    beside = run_path.with_name("beside.txt")
    beside.write_text("read by the run file\n")
    with run_path.open("a") as run_file:
        run_file.write(
            "import pathlib\n"
            'pathlib.Path(__file__).with_name("beside.txt").read_text()\n'
        )


def _task_arguments(run_dir: pathlib.Path, index: str) -> list:
    """The decsim arguments in run.sbatch's task line, at one task index."""
    run_script = run_dir / plan_command.RUN_SCRIPT
    script_text = run_script.read_text()
    script_lines = script_text.splitlines()
    task_line = script_lines[-1]
    indexed_line = task_line.replace("$SLURM_ARRAY_TASK_ID", index)
    words = shlex.split(indexed_line)
    return words[3:]


def _typed_sweep_rows(experiment_dir: pathlib.Path) -> list:
    sweep_path = experiment_dir / "sweep.csv"
    return report.read_rows(sweep_path)


def _estimate_by_the_statistics(row: dict) -> tuple:
    """failure_statistics on a sweep row's own prefix counts and stop."""
    stop_kind = failure_statistics.StopKind(row["state"])
    estimate = failure_statistics.estimate(
        row["prefix_failures"], row["prefix_scored_shots"], stop_kind
    )
    return (estimate.rate, estimate.low, estimate.high)


def _estimate_of(row: dict) -> tuple:
    """A sweep row's estimate and limits, an empty cell as None."""
    columns = (
        "logical_error_rate_estimate",
        "logical_error_rate_low",
        "logical_error_rate_high",
    )
    values = []
    for column in columns:
        value = row[column]
        if value == "":
            value = None
        values.append(value)
    return tuple(values)


def test_a_sweep_rows_estimate_is_failure_statistics_on_its_counts(tmp_path):
    """The referent is failure_statistics on the row's own counts and stop.

    One noisy point with a target of three failures runs until it
    stops; its row has the target state, and its estimate and exact
    limits are the ones the statistics module gives the same failures,
    scored shots and stop.
    """
    collection = {"max_shots": 60, "max_failures": 3, "piece_rounds": 45}
    config_path = run_files.write_run_file(
        tmp_path, axes=NOISY_AXES, collection=collection
    )
    out_dir = tmp_path / "out"

    command.main(["run", str(config_path), "--out", str(out_dir)])

    (row,) = _typed_sweep_rows(out_dir)
    assert row["state"] == "target"
    assert row["prefix_failures"] == 3
    assert _estimate_of(row) == _estimate_by_the_statistics(row)


def test_a_fold_keeps_a_point_its_run_file_no_longer_sweeps(tmp_path):
    """A fold folds what was recorded, not what the run file makes now.

    Four points are collected; the run file then drops one error rate. The
    two points it no longer makes keep their pieces, and a fold still
    counts them, every saved shot once.
    """
    config_path = run_files.write_run_file(tmp_path, **FOUR_POINTS)
    experiment_dir = tmp_path / "experiment"
    command.main(["run", str(config_path), "--out", str(experiment_dir)])
    narrower_axes = {
        **run_files.FOUR_POINT_AXES,
        run_files.ERROR_RATE_PATH: (0.001,),
    }
    run_files.write_run_file(
        tmp_path, axes=narrower_axes, collection={"max_shots": 2}
    )

    command.main(["run", "--fold", "--out", str(experiment_dir)])

    rows = _typed_sweep_rows(experiment_dir)
    assert len(rows) == 4
    assert _total_shots(rows) == 8


def _total_shots(rows: list) -> int:
    total = 0
    for row in rows:
        total += row["shots"]
    return total


def _capped_noisy_config(tmp_path, max_shots: int, piece_rounds: int):
    """The noisy point with a shot cap and no target, written in place."""
    collection = {"max_shots": max_shots, "piece_rounds": piece_rounds}
    return run_files.write_run_file(
        tmp_path, axes=NOISY_AXES, collection=collection
    )


def _write_csv_rows(path: pathlib.Path, rows: list) -> None:
    """Rows written over a csv file, their keys the header."""
    first_row = rows[0]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first_row))
        writer.writeheader()
        writer.writerows(rows)


def test_a_refused_fold_leaves_the_last_run_folder_as_it_was(tmp_path):
    """A fold that is refused publishes nothing, so the last one stands.

    Ten shots in two pieces are folded by the run. The second piece then
    loses a column of its shots.csv, as a piece measured by other code
    would, and a second fold is refused. Every file of the run folder is
    byte for byte what the first fold wrote.
    """
    config_path = _capped_noisy_config(tmp_path, 10, 75)
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    before = _run_folder_bytes(out_dir)
    piece_folders = out_dir.glob("pieces/*/*")
    second_piece = max(piece_folders, key=lambda folder: folder.name)
    shots_path = second_piece / "shots.csv"
    shot_rows = _csv_rows(shots_path)
    for row in shot_rows:
        del row["queue_wait_mean_us"]
    _write_csv_rows(shots_path, shot_rows)

    with pytest.raises(SystemExit):
        command.main(["run", "--fold", "--out", str(out_dir)])

    assert _run_folder_bytes(out_dir) == before


def test_a_fold_that_fails_part_way_leaves_the_last_run_folder_as_it_was(
    tmp_path,
):
    """A fold that raises after writing some files publishes none of them.

    Every piece is made to name a latency point this tree does not
    measure, as a tree that called service something else would write
    it, so the fold's sort of the window samples raises after sweep.csv
    and shots.csv are written. Those went into the staging folder, so
    every file of the run folder is what the first fold wrote.
    """
    config_path = _capped_noisy_config(tmp_path, 2, 15)
    out_dir = tmp_path / "out"
    command.main(["run", str(config_path), "--out", str(out_dir)])
    before = _run_folder_bytes(out_dir)
    for piece_folder in out_dir.glob("pieces/*/*"):
        _with_a_renamed_point(piece_folder, "service", "park")

    with pytest.raises(ValueError, match="x not in tuple"):
        command.main(["run", "--fold", "--out", str(out_dir)])

    assert _run_folder_bytes(out_dir) == before


def _with_a_renamed_point(piece_folder, name: str, renamed: str) -> None:
    """The piece as a tree that called that latency point renamed writes it."""
    shots_path = piece_folder / "shots.csv"
    shot_rows = _csv_rows(shots_path)
    for row in shot_rows:
        row[f"{renamed}_mean_us"] = row.pop(f"{name}_mean_us")
        row[f"{renamed}_max_us"] = row.pop(f"{name}_max_us")
    _write_csv_rows(shots_path, shot_rows)
    samples_path = piece_folder / "window_samples.csv"
    samples = _csv_rows(samples_path)
    for row in samples:
        if row["name"] == name:
            row["name"] = renamed
    _write_csv_rows(samples_path, samples)


def _run_folder_bytes(experiment_dir: pathlib.Path) -> dict:
    """Each file of the fold and the records, and its bytes."""
    csv_paths = experiment_dir.glob("*.csv")
    record_paths = experiment_dir.glob("points/*/machine.json")
    paths = [*csv_paths, *record_paths]
    return {str(path): path.read_bytes() for path in paths}


def _run_the_unit_and_note(ran_seeds: list, run_unit, unit, measure):
    """run_unit, collect's own, noting the seeds it runs."""
    end_seed = unit.first_seed + unit.seeds
    ran_seeds.extend(range(unit.first_seed, end_seed))
    return run_unit(unit, measure)
