"""The `decsim` command set: the verbs, and what each one is equal to.

The shape is sinter's (sinter/_command/_main.py: one command, a verb per
first word, each verb's module imported lazily). Each test
here pins a verb against what the same work done in Python returns, so a
command line is never the only record of a number.
"""

import csv
import functools
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest
import stim
import yaml

import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.experiments.command as command
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.run_command as run_command
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import tests.experiments.yaml_configs as yaml_configs
import tests.observe.gate_point as gate_point

CONFIGS_DIR = yaml_configs.CONFIGS_DIR
FOUR_POINT_SWEEP = {
    "sweep": [
        {
            "physical_error_probability": [0.001, 0.003],
            "distance": [3, 5],
            "round_period_microseconds": [1.0],
            "shots": 2,
        }
    ]
}


# The suite's container ships no git, so the tests that need it skip.
_GIT_MISSING = shutil.which("git") is None
requires_git = pytest.mark.skipif(
    _GIT_MISSING, reason="git is not installed where the suite runs"
)


EVERY_FILE = (
    "sweep.csv",
    "links.csv",
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


def _point_and_seed_of_every_row(run_dir, name):
    path = run_dir / name
    rows = _rows(path)
    order = []
    for row in rows:
        point_and_seed = (row["point_id"], row["seed"])
        order.append(point_and_seed)
    return order


def _collect_one_shard(config_path, out_dir, shard, shots_per_unit):
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(out_dir),
            "--shard",
            shard,
            "--shots-per-unit",
            str(shots_per_unit),
        ]
    )


def _rows_of_every_file(run_dir) -> dict:
    """Each file's rows without the wall clock, by file name."""
    return {
        name: _rows_without_wall_clock(run_dir, name) for name in EVERY_FILE
    }


def _bytes_of_files(run_dir, names) -> dict:
    contents = {}
    for name in names:
        path = run_dir / name
        contents[name] = path.read_bytes()
    return contents


def _collect_every_shard(config_path, shard_dirs: list) -> None:
    """Shard i of n into shard_dirs[i], one shot to a unit."""
    shard_count = len(shard_dirs)
    for index, shard_dir in enumerate(shard_dirs):
        _collect_one_shard(config_path, shard_dir, f"{index}/{shard_count}", 1)


def _combined(first_dir, second_dir, out_dir):
    command.main(
        ["combine", str(first_dir), str(second_dir), "--out", str(out_dir)]
    )


@pytest.fixture(autouse=True)
def one_tree_reading_per_test():
    """Every test here takes its own reading of the tree.

    run_folder reads the tree once per process, which is what a cluster
    task is. The suite is one process running many runs, and one of
    these tests answers for git itself, so the reading is dropped
    around each test rather than carried between them.
    """
    run_folder._tree_reading.cache_clear()
    yield
    run_folder._tree_reading.cache_clear()


def _manifest_of(run_dir):
    path = run_dir / "manifest.json"
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


def _seeds_of_every_shot(run_dir):
    path = run_dir / "shots.csv"
    rows = _rows(path)
    return [row["seed"] for row in rows]


def test_an_unknown_verb_prints_the_verbs_and_fails():
    with pytest.raises(SystemExit):
        command.main(["decode-everything"])


def test_help_prints_the_verbs_without_failing(capsys):
    command.main(["--help"])

    printed = capsys.readouterr()
    assert printed.err.strip() == command.usage()


@pytest.mark.parametrize("name", yaml_configs.SHIPPED_CONFIGS)
def test_show_lists_every_sections_kind_of_every_shipped_config(name):
    config_path = CONFIGS_DIR / name
    config = experiment.load_experiment(config_path)
    lines = experiment.resolved_description(config)
    text = "\n".join(lines)
    assert f"config: {config_path}" in lines[0]
    assert "qpu: kind stim_device" in text
    assert "sweep block 1:" in text
    assert "log: " in text
    assert "trace: " in text


@pytest.mark.parametrize(
    "qpu_kind, sentence",
    [
        ("recorded_stim", "required positional arguments: 'measurements'"),
        ("streaming_stim", "required positional argument: 'programs'"),
    ],
)
@pytest.mark.parametrize("verb", ["show", "run"])
def test_show_and_run_stop_at_a_source_the_yaml_cannot_build(
    tmp_path, verb, qpu_kind, sentence
):
    """The first point's machine is built, so show stops where run would."""
    qpu = {"qpu": {"kind": qpu_kind}}
    config_path = yaml_configs.write_config(tmp_path, qpu)

    with pytest.raises(TypeError, match=sentence):
        command.main([verb, str(config_path)])


@pytest.mark.parametrize("verb", ["show", "run"])
def test_show_and_run_refuse_a_maker_that_is_not_there_in_one_line(
    tmp_path, capsys, verb
):
    """The first point makes the workload, so show refuses what run would."""
    workload = {
        "kind": "producer",
        "function": "decsim.producers:no_such_maker",
        "arguments": {},
    }
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})

    with pytest.raises(SystemExit) as stopped:
        command.main([verb, str(config_path)])
    printed = capsys.readouterr()

    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "names no maker" in printed.err


def test_show_builds_a_point_whose_threshold_learns_online(tmp_path, capsys):
    """The first point's threshold is built per point, as the run builds it."""
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)

    command.main(["show", str(config_path)])
    printed = capsys.readouterr()

    assert "escalation: kind switching" in printed.out


def test_run_builds_a_point_whose_threshold_learns_online(tmp_path):
    """The one shot runs its point's threshold, as a collect's shot does."""
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    out_dir = tmp_path / "run"

    run_command.run_one_shot(config_path, out_dir=out_dir)

    assert (out_dir / "finished").exists()


def test_show_names_the_fabric_card_the_run_resolved_to():
    """The card is one line, because every hop on it is priced.

    A null card is the reference profile's numbers for that path
    (decsim/links/link_profiles.py logical_reference_profile), so the
    reader needs the card's name and nothing else.
    """
    config_path = CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(config_path)
    lines = experiment.resolved_description(config)
    text = "\n".join(lines)
    assert "links: card reference.yaml" in text


def _line_number_of(path, text: str) -> int:
    file_text = path.read_text()
    lines = file_text.splitlines()
    return lines.index(text) + 1


def test_show_names_each_values_layer_and_the_line_that_set_it(tmp_path):
    """Every value, the layer that set it and the yaml line it came from.

    A value your file writes, one a preset it extends writes, a swept
    one and one no file writes (its line is the one reference.yaml
    documents it on).
    """
    reference_path = CONFIGS_DIR / "reference.yaml"
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        f"extends: {reference_path}\n"
        "controller:\n"
        "  clock: fridge\n"
        "  readout_to_bits_cycles: 0\n"
        "  packing_cycles_per_round: 0\n"
        "  decision_to_pulse_cycles: 3\n"
    )
    config_path = tmp_path / "mine.yaml"
    config_path.write_text(
        "extends: preset.yaml\n"
        "pauli_frame:\n"
        "  kind: logical_register\n"
        "  clock: fridge\n"
        "  write_cycles: 5\n"
    )
    config = experiment.load_experiment(config_path)

    lines = experiment.value_lines(config)

    sweep_line = _line_number_of(reference_path, "sweep:")
    last_shots_line = _line_number_of(reference_path, "    shots: 2")
    overflow_line = _line_number_of(
        reference_path,
        "  packing_overflow: stall          # stall | drop_round: what the "
        "controller does",
    )
    assert (
        f"controller.decision_to_pulse_cycles = 3  [preset preset.yaml, "
        f"{preset_path}:6]"
    ) in lines
    assert (
        f"pauli_frame.write_cycles = 5  [your file, {config_path}:5]" in lines
    )
    assert (
        f"qpu.distance = [3]  [sweep, {reference_path}:{sweep_line}-"
        f"{last_shots_line}]"
    ) in lines
    assert (
        f'controller.packing_overflow = "STALL"  '
        f"[default, configs/reference.yaml:{overflow_line}]"
    ) in lines


def test_show_names_no_line_for_a_value_no_key_names_alone(tmp_path):
    """A derived or renamed value prints no source, never a guessed one.

    A channel's ticks come from its card's latency and clock, and a
    section's clock period from the domain it names and that domain's
    frequency; neither is one yaml key's value. unit_memory.bits and
    clocks.fridge are their own keys'.
    """
    reference_path = CONFIGS_DIR / "reference.yaml"
    config_path = tmp_path / "mine.yaml"
    config_path.write_text(
        f"extends: {reference_path}\n"
        "links:\n"
        "  qpu_to_controller:\n"
        "    clock: fridge\n"
        "    bits_per_cycle: null\n"
        "    latency_cycles: 7\n"
    )
    config = experiment.load_experiment(config_path)

    lines = experiment.value_lines(config)

    bits_line = _first_line_starting(reference_path, "    bits: null")
    fridge_line = _first_line_starting(reference_path, "  fridge:")
    assert (
        "links.qpu_to_controller.channel.propagation_latency_ticks = 28000"
    ) in lines
    assert "controller.clock.period_ticks = 4000" in lines
    assert (
        "clocks.fridge = 250.0  "
        f"[preset reference.yaml, {reference_path}:{fridge_line}]"
    ) in lines
    assert (
        "weak_decoder.unit_memory.bits = null  "
        f"[preset reference.yaml, {reference_path}:{bits_line}]"
    ) in lines


@functools.cache
def _key_paths_by_line(path) -> dict:
    """Each line of a yaml file that writes a mapping key, and that key."""
    with open(path) as handle:
        root = yaml.compose(handle)
    by_line = {}
    pending = [((), root)]
    while pending:
        prefix, node = pending.pop()
        if node.id != "mapping":
            continue
        for key_node, value_node in node.value:
            key = prefix + (key_node.value,)
            line = key_node.start_mark.line + 1
            by_line[line] = key
            pending.append((key, value_node))
    return by_line


def _cited_line(value_line: str) -> tuple:
    """The file and the first line a value line's bracket cites."""
    _, _, bracket = value_line.rpartition("  [")
    origin = bracket.removesuffix("]")
    _, _, source = origin.rpartition(", ")
    file_text, _, lines = source.rpartition(":")
    first_line, _, _ = lines.partition("-")
    return file_text, int(first_line)


def _value_key(value_line: str) -> tuple:
    """The yaml key a value line's value is read from."""
    dotted, _, _ = value_line.partition(" = ")
    names = dotted.split(".")
    path = tuple(names)
    key = experiment._yaml_key(path)
    if key in experiment.SWEEP_PATHS:
        return ("sweep",)
    return key


def _cited_lines(lines: list) -> list:
    cited = []
    for line in lines:
        if "  [" in line:
            cited.append(line)
    return cited


def _miscited_lines(cited: list) -> list:
    miscited = []
    for line in cited:
        if not _cites_its_own_key(line):
            miscited.append(line)
    return miscited


def _cites_its_own_key(value_line: str) -> bool:
    """Whether the line a value line cites writes that value's key."""
    file_text, first_line = _cited_line(value_line)
    cited_path = CONFIGS_DIR.parent / file_text
    key_paths = _key_paths_by_line(cited_path)
    return key_paths[first_line] == _value_key(value_line)


@pytest.mark.parametrize("name", yaml_configs.SHIPPED_CONFIGS)
def test_every_line_show_cites_holds_the_key_of_its_value(name):
    """Run on a shipped config, show cites only its values' own keys."""
    config_path = CONFIGS_DIR / name
    config = experiment.load_experiment(config_path)
    lines = experiment.value_lines(config)
    cited = _cited_lines(lines)

    miscited = _miscited_lines(cited)

    assert cited
    assert miscited == []


def _first_line_starting(path, start: str) -> int:
    text = path.read_text()
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if line.startswith(start):
            return number
    raise AssertionError(f"{path} has no line starting {start!r}")


def test_show_prints_every_value_after_the_sections(capsys):
    config_path = CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(config_path)
    value_lines = experiment.value_lines(config)

    command.main(["show", str(config_path)])
    printed = capsys.readouterr()

    lines = printed.out.splitlines()
    values_at = lines.index("values:")
    assert lines[values_at + 1 :] == value_lines


def test_run_prints_the_result_fields_the_gate_hashes(tmp_path):
    config_path = gate_point.CONFIG_PATH
    lines = run_command.run_one_shot(config_path, seed=0, out_dir=tmp_path)
    config = experiment.load_experiment(config_path)
    block = config.sweep[0]
    settings = config.point_settings(
        physical_error_probability=block.physical_error_probabilities[0],
        distance=block.distances[0],
        round_period_microseconds=block.round_periods_microseconds[0],
    )
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
    config_path = gate_point.CONFIG_PATH
    run_command.run_one_shot(config_path, seed=0, out_dir=tmp_path, trace=True)
    trace_dir = tmp_path / "trace"
    entries = trace_dir.iterdir()
    written = sorted(entries)
    assert len(written) == 1


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


def test_a_run_folder_holds_the_points_values_workload_and_maker(tmp_path):
    """What the run ran, every value of it, and the workload as files."""
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    lines = run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)
    resolved_dir = out_dir / "resolved"
    resolved_path = _one_file(resolved_dir, "*.json")
    resolved_text = resolved_path.read_text()
    resolved = json.loads(resolved_text)
    inputs_dir = out_dir / "inputs" / resolved_path.stem
    hashes_text = (inputs_dir / "hashes.json").read_text()
    hashes = json.loads(hashes_text)
    producer_text = (out_dir / "producer.json").read_text()
    producer = json.loads(producer_text)

    assert "terminal status: complete" in lines[2]
    assert resolved["seeds"] == [[0, 1]]
    assert resolved["built"]["commit_rounds"] == 3
    assert resolved["settings"]["qpu"]["distance"] == 3
    assert hashes == _hashes_of(
        inputs_dir, "operation_1.stim", "operations.json"
    )
    assert producer["function"] == "decsim.producers:memory_circuit"
    assert producer["arguments"]["rounds_per_shot"] == 15
    assert (out_dir / "result.json").exists()
    assert (out_dir / "finished").exists()


# A maker that answers a longer memory each time it is called, so a
# point made twice records one workload and runs another.
GROWING_MAKER = """
import decsim.producers as producers

CALLS = []


def growing_memory(distance, physical_error_probability):
    CALLS.append(distance)
    rounds = 15 + 3 * (len(CALLS) - 1)
    return producers.memory_circuit(
        "surface_code:rotated_memory_z",
        rounds,
        distance,
        physical_error_probability,
    )
"""


def test_a_collect_makes_each_points_workload_once(tmp_path, monkeypatch):
    """The workload recorded in inputs/ is the one the shots ran."""
    maker_path = tmp_path / "growing_maker.py"
    maker_path.write_text(GROWING_MAKER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "growing_maker", raising=False)
    workload = {"kind": "producer", "function": "growing_maker:growing_memory"}
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    run_dir = tmp_path / "run"

    command.main(["collect", str(config_path), "--out", str(run_dir)])

    growing_maker = sys.modules["growing_maker"]
    assert growing_maker.CALLS == [3]


def test_a_run_folders_inputs_rerun_the_point_without_the_maker(tmp_path):
    """The files row reads inputs/ back, and the shot is the same shot."""
    config_path = yaml_configs.write_config(tmp_path, {})
    first_dir = tmp_path / "first"
    run_command.run_one_shot(config_path, seed=0, out_dir=first_dir)
    resolved_dir = first_dir / "resolved"
    resolved_path = _one_file(resolved_dir, "*.json")
    inputs_dir = first_dir / "inputs" / resolved_path.stem
    operations_path = inputs_dir / "operations.json"
    files = {"kind": "files", "operations": str(operations_path)}
    rerun_folder = tmp_path / "rerun"
    rerun_folder.mkdir()
    rerun_path = yaml_configs.write_config(rerun_folder, {"workload": files})
    second_dir = tmp_path / "second"
    run_command.run_one_shot(rerun_path, seed=0, out_dir=second_dir)
    first_result = (first_dir / "result.json").read_text()
    second_result = (second_dir / "result.json").read_text()

    assert second_result == first_result


@pytest.mark.parametrize(
    "function",
    ["decsim.producers:memory_circuit", "decsim.producers.memory_circuit"],
)
def test_producer_json_names_the_maker_in_either_of_its_forms(
    tmp_path, function
):
    """pkgutil.resolve_name reads both forms, so the run records both."""
    workload = yaml_configs.memory_workload(15)
    workload["function"] = function
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    out_dir = tmp_path / "out"

    run_command.run_one_shot(config_path, out_dir=out_dir)

    producer_text = (out_dir / "producer.json").read_text()
    producer = json.loads(producer_text)
    assert producer["function"] == function
    assert (out_dir / "finished").exists()


def test_a_point_recorded_again_hashes_only_its_inputs(tmp_path):
    """A retried folder records its points again over the first record."""
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    run_dir = tmp_path / "run"

    point_id = run_folder.record_point(run_dir, task)
    run_folder.record_point(run_dir, task)

    inputs_dir = run_dir / "inputs" / point_id
    hashes_path = inputs_dir / "hashes.json"
    hashes_text = hashes_path.read_text()
    hashes = json.loads(hashes_text)
    assert sorted(hashes) == ["operation_1.stim", "operations.json"]


def test_a_sweep_rerun_into_a_finished_folder_leaves_it_as_it_is(
    tmp_path, capsys
):
    """A Slurm array rerun runs again only the shards that did not finish."""
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    sweep_path = out_dir / "sweep.csv"
    first_status = sweep_path.stat()
    capsys.readouterr()

    command.main(["collect", str(config_path), "--out", str(out_dir)])
    printed = capsys.readouterr()

    assert (out_dir / "finished").exists()
    second_status = sweep_path.stat()
    assert second_status.st_mtime_ns == first_status.st_mtime_ns
    assert "holds a finished run" in printed.err


def test_a_combined_folder_holds_every_shards_points(tmp_path):
    """A point's records are named by content, so the union is every point."""
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    combined_dir = tmp_path / "combined"
    _collect_one_shard(config_path, first_dir, "0/2", 1)
    _collect_one_shard(config_path, second_dir, "1/2", 1)
    _combined(first_dir, second_dir, combined_dir)
    resolved_dir = combined_dir / "resolved"
    resolved_files = resolved_dir.glob("*.json")
    resolved = list(resolved_files)

    assert len(resolved) == 4
    assert (combined_dir / "producer.json").exists()
    assert (combined_dir / "finished").exists()


def _seeds_of_the_one_point(run_dir):
    resolved_dir = run_dir / "resolved"
    resolved_path = _one_file(resolved_dir, "*.json")
    resolved_text = resolved_path.read_text()
    resolved = json.loads(resolved_text)
    return resolved["seeds"]


def test_a_shards_record_holds_the_seeds_it_ran(tmp_path):
    """Each shard records its own seeds, and the fold records all of them."""
    four_shots = {"sweep": [dict(yaml_configs.MINIMAL_CONFIG["sweep"][0])]}
    four_shots["sweep"][0]["shots"] = 4
    config_path = yaml_configs.write_config(tmp_path, four_shots)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    combined_dir = tmp_path / "combined"
    _collect_one_shard(config_path, first_dir, "0/2", 2)
    _collect_one_shard(config_path, second_dir, "1/2", 2)
    _combined(first_dir, second_dir, combined_dir)

    assert _seeds_of_every_shot(second_dir) == ["2", "3"]
    assert _seeds_of_the_one_point(first_dir) == [[0, 2]]
    assert _seeds_of_the_one_point(second_dir) == [[2, 2]]
    assert _seeds_of_the_one_point(combined_dir) == [[0, 4]]


def test_a_second_combine_into_the_same_folder_keeps_its_seeds(tmp_path):
    four_shots = {"sweep": [dict(yaml_configs.MINIMAL_CONFIG["sweep"][0])]}
    four_shots["sweep"][0]["shots"] = 4
    config_path = yaml_configs.write_config(tmp_path, four_shots)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    combined_dir = tmp_path / "combined"
    _collect_one_shard(config_path, first_dir, "0/2", 2)
    _collect_one_shard(config_path, second_dir, "1/2", 2)
    _combined(first_dir, second_dir, combined_dir)
    _combined(first_dir, second_dir, combined_dir)

    assert _seeds_of_the_one_point(combined_dir) == [[0, 4]]


def test_a_manifest_names_every_installed_package(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)
    manifest = _manifest_of(out_dir)
    packages = manifest["versions"]["packages"]

    assert packages["stim"] == stim.__version__
    assert "numpy" in packages


def test_a_manifest_names_the_union_find_library_it_loads(
    tmp_path, compiled_union_find_library
):
    """The library is built, not tracked, so the commit does not name it."""
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)
    manifest = _manifest_of(out_dir)
    library_hash = _sha256_of(compiled_union_find_library)

    assert manifest["union_find_library_sha256"] == library_hash


def test_a_manifest_without_the_union_find_library_names_none(
    tmp_path, monkeypatch
):
    """A run that decodes with PyMatching needs no library built."""
    absent = tmp_path / "absent.so"
    monkeypatch.setattr(compiled_decoder, "library_path", lambda: absent)
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"

    run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)

    manifest = _manifest_of(out_dir)
    assert manifest["union_find_library_sha256"] is None


def test_a_run_where_git_cannot_answer_records_no_patch(tmp_path, monkeypatch):
    """The container the suite runs in ships no git."""
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: None)
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run_folder.snapshot_code_state(config, out_dir)

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
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run_folder.snapshot_code_state(config, out_dir)

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
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_folder.snapshot_code_state(config, out_dir)
    replay = tmp_path / "replay"
    subprocess.run(["git", "init", "-q", str(replay)], check=True)
    patch_path = out_dir / "code_state.patch"
    apply = ["git", "-C", str(replay), "apply", str(patch_path)]
    subprocess.run(apply, check=True)

    assert (replay / "maker.py").read_text() == "VALUE = 1\n"


def test_a_pooled_collect_writes_the_serial_collects_rows(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    pooled_dir = tmp_path / "pooled"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(pooled_dir),
            "--processes",
            "4",
        ]
    )
    serial_rows = _rows_of_every_file(serial_dir)
    pooled_rows = _rows_of_every_file(pooled_dir)
    assert pooled_rows == serial_rows


def test_two_shards_combined_are_the_unsharded_collects_rows(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    whole_dir = tmp_path / "whole"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(whole_dir)])
    command.main(
        ["collect", str(config_path), "--out", str(first_dir), "--shard", "0/2"]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shard",
            "1/2",
        ]
    )
    command.main(
        [
            "combine",
            str(first_dir),
            str(second_dir),
            "--out",
            str(combined_dir),
        ]
    )
    whole_rows = _rows_of_every_file(whole_dir)
    combined_rows = _rows_of_every_file(combined_dir)
    assert combined_rows == whole_rows


def test_combine_writes_the_same_rows_whichever_order_the_shards_come_in(
    tmp_path,
):
    """The shards split every point's seeds, so every file is folded."""
    wall_clock_unit = {
        **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
        "kind": "pymatching",
    }
    config_path = yaml_configs.write_config(
        tmp_path, {**FOUR_POINT_SWEEP, "weak_decoder": wall_clock_unit}
    )
    serial_dir = tmp_path / "serial"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    forwards_dir = tmp_path / "forwards"
    backwards_dir = tmp_path / "backwards"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(first_dir),
            "--shots-per-unit",
            "1",
            "--shard",
            "0/2",
        ]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shots-per-unit",
            "1",
            "--shard",
            "1/2",
        ]
    )
    command.main(
        [
            "combine",
            str(first_dir),
            str(second_dir),
            "--out",
            str(forwards_dir),
        ]
    )
    command.main(
        [
            "combine",
            str(second_dir),
            str(first_dir),
            "--out",
            str(backwards_dir),
        ]
    )

    every_file = EVERY_FILE + ("latency_samples.csv",)
    forwards_bytes = _bytes_of_files(forwards_dir, every_file)
    backwards_bytes = _bytes_of_files(backwards_dir, every_file)
    serial_shots = _point_and_seed_of_every_row(serial_dir, "shots.csv")
    combined_shots = _point_and_seed_of_every_row(forwards_dir, "shots.csv")
    serial_samples = _point_and_seed_of_every_row(
        serial_dir, "latency_samples.csv"
    )
    combined_samples = _point_and_seed_of_every_row(
        forwards_dir, "latency_samples.csv"
    )
    assert forwards_bytes == backwards_bytes
    assert combined_shots == serial_shots
    assert combined_samples == serial_samples


def test_combining_folders_of_two_different_sweeps_is_refused(tmp_path, capsys):
    first_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    second_path = tmp_path / "other.yaml"
    other_text = first_path.read_text()
    more_shots = other_text.replace("shots: 2", "shots: 3")
    second_path.write_text(more_shots)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    combined_dir = tmp_path / "combined"
    command.main(
        ["collect", str(first_path), "--out", str(first_dir), "--shard", "0/2"]
    )
    command.main(
        [
            "collect",
            str(second_path),
            "--out",
            str(second_dir),
            "--shard",
            "1/2",
        ]
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "combine",
                str(first_dir),
                str(second_dir),
                "--out",
                str(combined_dir),
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "ran a different experiment from" in printed.err


def test_a_unit_size_that_splits_a_point_writes_the_serial_runs_rows(
    tmp_path,
):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    split_dir = tmp_path / "split"
    pooled_dir = tmp_path / "pooled"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(split_dir),
            "--shots-per-unit",
            "1",
        ]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(pooled_dir),
            "--shots-per-unit",
            "1",
            "--processes",
            "4",
        ]
    )

    serial_rows = _rows_of_every_file(serial_dir)
    split_rows = _rows_of_every_file(split_dir)
    pooled_rows = _rows_of_every_file(pooled_dir)
    assert split_rows == serial_rows
    assert pooled_rows == serial_rows


def test_shards_that_split_every_points_seeds_fold_to_the_serial_rows(
    tmp_path,
):
    """The additive record's whole point: a folded point is the point.

    Every point of the sweep is split, seed 0 to one shard and seed 1
    to the other, so no shard holds a whole point and every summary
    column has to come out of the additive files.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    forwards_dir = tmp_path / "forwards"
    backwards_dir = tmp_path / "backwards"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(first_dir),
            "--shots-per-unit",
            "1",
            "--shard",
            "0/2",
        ]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shots-per-unit",
            "1",
            "--shard",
            "1/2",
        ]
    )
    command.main(
        [
            "combine",
            str(first_dir),
            str(second_dir),
            "--out",
            str(forwards_dir),
        ]
    )
    command.main(
        [
            "combine",
            str(second_dir),
            str(first_dir),
            "--out",
            str(backwards_dir),
        ]
    )

    serial_rows = _rows_of_every_file(serial_dir)
    folded_rows = _rows_of_every_file(forwards_dir)
    forwards_bytes = _bytes_of_files(forwards_dir, EVERY_FILE)
    backwards_bytes = _bytes_of_files(backwards_dir, EVERY_FILE)
    assert folded_rows == serial_rows
    assert forwards_bytes == backwards_bytes


def test_a_combined_folder_folds_again_with_a_later_shard(tmp_path):
    """A Slurm array finishing in waves folds each wave as it lands.

    So a combined folder is a run folder: the additive files plus a
    manifest recording the sweep, and folding it with the last shard
    gives the rows of the run that never sharded at all.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    shard_dirs = [tmp_path / f"shard{index}" for index in (0, 1, 2)]
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    _collect_every_shard(config_path, shard_dirs)
    first_two_dir = tmp_path / "first_two"
    combined_dir = tmp_path / "combined"
    _combined(shard_dirs[0], shard_dirs[1], first_two_dir)
    _combined(first_two_dir, shard_dirs[2], combined_dir)
    serial_rows = _rows_of_every_file(serial_dir)
    combined_rows = _rows_of_every_file(combined_dir)
    assert combined_rows == serial_rows


def test_a_manifest_records_the_shard_and_the_unit_size_it_ran(tmp_path):
    """How a folder ran, beside what it ran: the array's own bookkeeping.

    Nothing reads these to fold the rows, which come back in the order
    the recorded sweep gives; they say what one of a hundred folders an
    array left behind holds.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    whole_dir = tmp_path / "whole"
    shard_dir = tmp_path / "shard"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(whole_dir)])
    _collect_one_shard(config_path, shard_dir, "0/2", 1)
    command.main(["combine", str(whole_dir), "--out", str(combined_dir)])
    whole = _manifest_of(whole_dir)
    sharded = _manifest_of(shard_dir)
    combined = _manifest_of(combined_dir)
    assert whole["shard"] is None
    assert whole["shots_per_unit"] is None
    assert sharded["shard"] == "0/2"
    assert sharded["shots_per_unit"] == 1
    assert combined["folded"] == [str(whole_dir)]
    assert combined["resolved_config"] == whole["resolved_config"]


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
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "run"
    monkeypatch.chdir(tmp_path)
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    manifest = _manifest_of(out_dir)
    recorded = manifest["git"]

    assert recorded["commit"] == _commit_of_this_tree()
    assert "dirty" in recorded


def test_a_manifest_takes_the_dirty_flag_from_the_launcher_that_looked(
    tmp_path, monkeypatch
):
    """The interpreter may have no git; the launcher did.

    slurm/slurm_run.sh looks at the tree as the job starts and exports
    what it saw, because the container image this runs in ships no git
    binary and a folder that cannot say whether its code was committed
    must say that rather than say clean.

    A process reads the tree once, so the two runs here take a reading
    each on purpose: one process launched by two different launchers is
    the test bench's own shape and never a cluster job's.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    dirty_dir = tmp_path / "dirty"
    clean_dir = tmp_path / "clean"
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "1")
    run_folder._tree_reading.cache_clear()
    command.main(["collect", str(config_path), "--out", str(dirty_dir)])
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "0")
    run_folder._tree_reading.cache_clear()
    command.main(["collect", str(config_path), "--out", str(clean_dir)])

    dirty_manifest = _manifest_of(dirty_dir)
    clean_manifest = _manifest_of(clean_dir)
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
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "run"
    started_utc = run_folder.utc_now()
    config = experiment.load_experiment(config_path)
    out_dir.mkdir()
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "0")
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: "aaaaaaa")
    run_folder._tree_reading.cache_clear()

    run_folder.write_manifest(config, out_dir, [], started_utc)
    at_the_start = _manifest_of(out_dir)
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "1")
    monkeypatch.setattr(run_folder, "_git_output", lambda *_: "bbbbbbb")
    finished_utc = run_folder.utc_now()
    run_folder.write_manifest(
        config, out_dir, [], started_utc, finished_utc=finished_utc
    )
    at_the_end = _manifest_of(out_dir)

    assert at_the_start["git"] == {"commit": "aaaaaaa", "dirty": False}
    assert at_the_end["git"] == at_the_start["git"]
    assert at_the_end["finished_utc"] is not None


def test_a_combined_folders_manifest_names_the_tree_the_fold_ran_on(
    tmp_path, monkeypatch
):
    """A fold writes one manifest, at the end, and it names the start.

    The fold of one weak_ler experiment's 500 shard folders took 71
    minutes and a commit landed five minutes into it, so the combined
    folder named a tree whose code no part of the fold read. Here git
    answers one commit until the first folder is opened and another
    after, and the manifest names the first.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    combined_dir = tmp_path / "combined"
    _collect_one_shard(config_path, first_dir, "0/2", 1)
    _collect_one_shard(config_path, second_dir, "1/2", 1)
    reading = ["aaaaaaa"]
    opened_folders = []
    rows_of_a_folder = fold.row_stream

    def moving_tree(*_):
        return reading[0]

    def commit_while_the_fold_reads(path):
        opened_folders.append(path)
        reading[0] = "bbbbbbb"
        return rows_of_a_folder(path)

    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "0")
    monkeypatch.setattr(run_folder, "_git_output", moving_tree)
    monkeypatch.setattr(fold, "row_stream", commit_while_the_fold_reads)
    run_folder._tree_reading.cache_clear()

    _combined(first_dir, second_dir, combined_dir)

    manifest = _manifest_of(combined_dir)
    assert opened_folders
    assert manifest["git"] == {"commit": "aaaaaaa", "dirty": False}


def test_a_shard_with_no_work_unit_says_so_and_writes_no_rows(tmp_path, capsys):
    """A Slurm array wider than the sweep's units is a shape, not a fault."""
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "empty"
    capsys.readouterr()
    command.main(
        ["collect", str(config_path), "--out", str(out_dir), "--shard", "1/2"]
    )

    printed = capsys.readouterr()
    manifest_path = out_dir / "manifest.json"
    sweep_path = out_dir / "sweep.csv"
    assert manifest_path.is_file()
    assert not sweep_path.exists()
    assert "wrote no rows beyond its manifest" in printed.err


def test_combine_skips_a_folder_that_ran_no_shot(tmp_path, capsys):
    config_path = yaml_configs.write_config(tmp_path, {})
    whole_dir = tmp_path / "whole"
    empty_dir = tmp_path / "empty"
    combined_dir = tmp_path / "combined"
    command.main(
        ["collect", str(config_path), "--out", str(whole_dir), "--shard", "0/2"]
    )
    command.main(
        ["collect", str(config_path), "--out", str(empty_dir), "--shard", "1/2"]
    )
    capsys.readouterr()
    command.main(
        [
            "combine",
            str(whole_dir),
            str(empty_dir),
            "--out",
            str(combined_dir),
        ]
    )

    printed = capsys.readouterr()
    assert "has no shots.csv, so combine skips it" in printed.err
    whole = _rows_without_wall_clock(whole_dir, "sweep.csv")
    combined = _rows_without_wall_clock(combined_dir, "sweep.csv")
    assert combined == whole


def test_every_shard_of_a_sweep_runs_a_share_of_its_points(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    command.main(
        ["collect", str(config_path), "--out", str(first_dir), "--shard", "0/2"]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shard",
            "1/2",
        ]
    )
    first_path = first_dir / "sweep.csv"
    second_path = second_dir / "sweep.csv"
    first_rows = _rows(first_path)
    second_rows = _rows(second_path)
    assert len(first_rows) == 2
    assert len(second_rows) == 2


def test_one_points_seeds_divide_across_two_shards(tmp_path):
    """The unit, not the point, is what a shard selects.

    This is the whole reason --shots-per-unit exists: a sweep of one
    point can still fill a Slurm array. Its law is that shard i of n
    runs the units whose position modulo n is i, so at one shot to a
    unit the even seeds go to shard 0 of 2 and the odd seeds to shard 1.
    """
    one_point = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_microseconds": [1.0],
                "shots": 4,
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, one_point)
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    _collect_every_shard(config_path, [first_dir, second_dir])
    first_seeds = _seeds_of_every_shot(first_dir)
    second_seeds = _seeds_of_every_shot(second_dir)
    assert first_seeds == ["0", "2"]
    assert second_seeds == ["1", "3"]


def test_a_shard_outside_its_count_is_refused(tmp_path, capsys):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "collect",
                str(config_path),
                "--out",
                str(out_dir),
                "--shard",
                "5/2",
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: --shard 5/2 is not a shard")
    assert not out_dir.exists()


@pytest.mark.parametrize("given", ["0", "-1"])
def test_a_unit_size_below_one_is_refused(tmp_path, capsys, given):
    """Zero steps `range` by nothing; -1 would run no unit at all."""
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "collect",
                str(config_path),
                "--out",
                str(out_dir),
                "--shots-per-unit",
                given,
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(
        f"decsim: --shots-per-unit {given} is not a unit size"
    )
    assert not out_dir.exists()


def test_combining_two_folders_that_hold_the_same_shot_is_refused(
    tmp_path, capsys
):
    config_path = yaml_configs.write_config(tmp_path, {})
    run_dir = tmp_path / "one"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(run_dir)])
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "combine",
                str(run_dir),
                str(run_dir),
                "--out",
                str(combined_dir),
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "is in more than one of" in printed.err


def test_run_refuses_a_yaml_that_is_not_there(tmp_path, capsys):
    missing = tmp_path / "not_a_config.yaml"
    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(missing)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(f"decsim: {missing} is not a file")
    assert "reference" in printed.err


def test_show_refuses_a_sweep_axis_the_yaml_layer_does_not_have(
    tmp_path, capsys
):
    unknown_axis = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_microseconds": [1.0],
                "algorithm": ["pymatching"],
                "shots": 1,
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, unknown_axis)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "sweep block 1 does not know ['algorithm']" in printed.err


@pytest.mark.parametrize("shots", [0, -1, 1.5, "many", True])
def test_show_refuses_a_shot_count_that_is_not_a_whole_number_of_one_or_more(
    tmp_path, capsys, shots
):
    bad_count = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_microseconds": [1.0],
                "shots": shots,
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, bad_count)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert "sweep block 1 shots must be a whole number of at least 1" in (
        printed.err
    )


def test_show_refuses_a_sweep_axis_written_as_one_value_not_a_list(
    tmp_path, capsys
):
    scalar_axis = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": 3,
                "round_period_microseconds": [1.0],
                "shots": 1,
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, scalar_axis)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert "sweep block 1 distance must be a list of at least one value" in (
        printed.err
    )


def test_plot_refuses_a_figure_it_does_not_draw(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        command.main(["plot", str(tmp_path), "--figure", "everything"])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: no figure named everything")


def test_plot_refuses_a_figure_against_no_named_setting(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        command.main(["plot", str(tmp_path), "--figure", "ler"])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.startswith(
        "decsim: the ler figure is drawn against a swept setting"
    )


def test_plot_refuses_a_where_that_names_no_value(tmp_path, capsys):
    arguments = [
        "plot",
        str(tmp_path),
        "--figure",
        "ler",
        "--x",
        "qpu.distance",
    ]
    arguments += ["--where", "qpu.distance"]
    with pytest.raises(SystemExit) as stopped:
        command.main(arguments)

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.startswith("decsim: --where qpu.distance names no value")


def test_trace_refuses_an_action_it_does_not_have(tmp_path, capsys):
    trace_path = tmp_path / "shot.json"
    with pytest.raises(SystemExit) as stopped:
        command.main(["trace", "summarise", str(trace_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: decsim trace has no action")


def test_a_refused_combine_leaves_no_folder(tmp_path):
    """The out folder is made once the fold is accepted, not before."""
    missing = tmp_path / "never_ran"
    out_dir = tmp_path / "combined"
    with pytest.raises(SystemExit) as exit_info:
        command.main(["combine", str(missing), "--out", str(out_dir)])
    assert exit_info.value.code == 1
    assert not out_dir.exists()


def test_a_build_refusal_under_run_is_one_line(tmp_path, capsys):
    """A yaml that loads but that a row refuses at build, as a sentence.

    burst_detector kind event_count sends a burst's windows to the
    strong decoder, which only escalation kind switching has, so the
    weak baseline is refused when the machine is built
    (decsim/build/escalation.py).
    """
    base_path = CONFIGS_DIR / "weak_decoder_baseline.yaml"
    config_path = tmp_path / "burst.yaml"
    config_path.write_text(
        f"extends: {base_path}\nburst_detector:\n  kind: event_count\n"
    )

    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(
        f"decsim: {config_path}: burst_detector.kind event_count"
    )
