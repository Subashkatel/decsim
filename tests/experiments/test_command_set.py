"""The `decsim` command set: the verbs, and what each one is equal to.

The shape is sinter's (sinter/_command/_main.py: one command, a verb per
first word, each verb's module imported lazily). Each test
here pins a verb against what the same work done in Python returns, so a
command line is never the only record of a number.
"""

import csv
import dataclasses
import functools
import hashlib
import json
import math
import pathlib
import resource
import shutil
import subprocess
import sys

import pytest
import scipy.stats
import sinter
import stim
import yaml

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
import decsim.experiments.run_command as run_command
import decsim.experiments.run_folder as run_folder
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.settings as machine_settings
import tests.experiments.yaml_configs as yaml_configs
import tests.observe.gate_point as gate_point

CONFIGS_DIR = yaml_configs.CONFIGS_DIR
FOUR_POINT_SWEEP = {
    "sweep": [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.001, 0.003],
                "qpu.distance": [3, 5],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": 2},
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


def _rows_of_every_file(run_dir) -> dict:
    """Each file's rows without the wall clock, by file name."""
    return {
        name: _rows_without_wall_clock(run_dir, name) for name in EVERY_FILE
    }


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
    first_point = config.first_point_task()
    lines = experiment.resolved_description(config, first_point.settings)
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

    assert (out_dir / "result.json").exists()


def test_show_names_the_fabric_card_the_run_resolved_to():
    """The card is one line, because every hop on it is priced.

    A null card is the reference profile's numbers for that path
    (decsim/links/link_profiles.py logical_reference_profile), so the
    reader needs the card's name and nothing else.
    """
    config_path = CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(config_path)
    first_point = config.first_point_task()
    lines = experiment.resolved_description(config, first_point.settings)
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

    first_point = config.first_point_task()
    lines = experiment.value_lines(config, first_point.settings)

    sweep_line = _line_number_of(reference_path, "sweep:")
    last_sweep_line = _line_number_of(
        reference_path, "      qpu.round_period_microseconds: [1.0]"
    )
    bound_line = _line_number_of(
        reference_path,
        "  packing_rounds_in_flight: null   # packing assembly workspace: "
        "rounds in flight",
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
        f"{last_sweep_line}]"
    ) in lines
    assert (
        "controller.packing_rounds_in_flight = null  "
        f"[default, configs/reference.yaml:{bound_line}]"
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

    first_point = config.first_point_task()
    lines = experiment.value_lines(config, first_point.settings)

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
    """The yaml key a value line's value is read from, a swept one's sweep."""
    dotted, _, rest = value_line.partition(" = ")
    if "  [sweep, " in rest:
        return ("sweep",)
    names = dotted.split(".")
    path = tuple(names)
    return experiment._yaml_key(path)


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
    point = config.first_point_task()
    lines = experiment.value_lines(config, point.settings)
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
    first_point = config.first_point_task()
    value_lines = experiment.value_lines(config, first_point.settings)

    command.main(["show", str(config_path)])
    printed = capsys.readouterr()

    lines = printed.out.splitlines()
    values_at = lines.index("values:")
    assert lines[values_at + 1 :] == value_lines


def test_run_prints_the_result_fields_the_gate_hashes(tmp_path):
    config_path = gate_point.CONFIG_PATH
    lines = run_command.run_one_shot(config_path, seed=0, out_dir=tmp_path)
    config = experiment.load_experiment(config_path)
    point = config.first_point_task()
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
    config_path = gate_point.CONFIG_PATH
    run_command.run_one_shot(config_path, seed=0, out_dir=tmp_path, trace=True)
    trace_dir = tmp_path / "trace"
    entries = trace_dir.iterdir()
    written = sorted(entries)
    assert len(written) == 1


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
    maker = resolved["maker"]

    assert "terminal status: complete" in lines[2]
    assert resolved["seeds"] == [[0, 1]]
    assert resolved["built"]["commit_rounds"] == 3
    assert resolved["settings"]["qpu"]["distance"] == 3
    assert hashes == _hashes_of(
        inputs_dir, "operation_1.stim", "operations.json"
    )
    assert maker["function"] == "decsim.producers:memory_circuit"
    assert maker["arguments"]["rounds_per_shot"] == 15
    assert (out_dir / "result.json").exists()


def test_a_shots_rounds_add_up_every_patchs_rounds(tmp_path):
    """Three patches of fifteen rounds are forty-five patch-rounds.

    Tesseract 2503.10988 lines 287-290 count the rounds across every
    code in the shot, so a per-round rate compares with one memory.
    """
    workload = yaml_configs.memory_workload(15)
    workload["function"] = "decsim.producers:memory_patches"
    workload["arguments"]["patch_count"] = 3
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    config = experiment.load_experiment(config_path)
    task = config.point_task(
        {
            yaml_configs.ERROR_RATE_PATH: 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        }
    )

    record = run_folder.point_record(task)

    assert record["rounds_per_shot"] == 45


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
    rounds = round_policies.PerOperationRounds({100: 3, 1: 3})
    workload = workload_settings.WorkloadSettings(
        operations=(segment,), decode_operations=(owner,), rounds_policy=rounds
    )
    qpu = qpu_settings.QpuSettings(distance=3, kind="timing_only")
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    weak_decoder = decoder_settings.DecoderSettings(kind=0.1, engine=engine)
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )
    task = collect.Task(settings, {})

    record = run_folder.point_record(task)

    assert record["rounds_per_shot"] == 3


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
    workload = {
        "kind": "producer",
        "function": "growing_maker:growing_memory",
        "arguments": {"distance": "${qpu.distance}"},
    }
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    run_dir = tmp_path / "run"

    command.main(["collect", str(config_path), "--out", str(run_dir)])

    growing_maker = sys.modules["growing_maker"]
    assert growing_maker.CALLS == [3]


WAITING_MAKER = """
import decsim.records.program as program
import decsim.records.workload as workload


def wait(**_arguments):
    operation = program.Operation(
        1, "wait", (0,), patches=(0,), emits_detector_data=False
    )
    return workload.Workload((operation,), {1: 3})
"""


def test_a_collect_of_a_workload_with_no_detector_rounds_is_refused(
    tmp_path, monkeypatch, capsys
):
    """A shot that sends no detector data has no rounds to size or score.

    Its rounds per shot are zero, so collect refuses the point before it
    sizes a piece, as its scoring would refuse the first shot.
    """
    maker_path = tmp_path / "waiting_maker.py"
    maker_path.write_text(WAITING_MAKER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "waiting_maker", raising=False)
    workload = {"kind": "producer", "function": "waiting_maker:wait"}
    card = {
        "workload": workload,
        "qpu": {"kind": "timing_only"},
        "sweep": yaml_configs.QPU_ONLY_SWEEP,
    }
    config_path = yaml_configs.write_config(tmp_path, card)
    run_dir = tmp_path / "run"

    with pytest.raises(SystemExit) as stopped:
        command.main(["collect", str(config_path), "--out", str(run_dir)])
    printed = capsys.readouterr()

    assert stopped.value.code == 1
    assert "sends detector data" in printed.err


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
    rerun_card = {"workload": files, "sweep": yaml_configs.QPU_ONLY_SWEEP}
    rerun_path = yaml_configs.write_config(rerun_folder, rerun_card)
    second_dir = tmp_path / "second"
    run_command.run_one_shot(rerun_path, seed=0, out_dir=second_dir)
    first_result = (first_dir / "result.json").read_text()
    second_result = (second_dir / "result.json").read_text()

    assert second_result == first_result


@pytest.mark.parametrize(
    "function",
    ["decsim.producers:memory_circuit", "decsim.producers.memory_circuit"],
)
def test_a_points_record_names_the_maker_in_either_of_its_forms(
    tmp_path, function
):
    """pkgutil.resolve_name reads both forms, so the run records both."""
    workload = yaml_configs.memory_workload(15)
    workload["function"] = function
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    out_dir = tmp_path / "out"

    run_command.run_one_shot(config_path, out_dir=out_dir)

    resolved_dir = out_dir / "resolved"
    resolved_path = _one_file(resolved_dir, "*.json")
    resolved_text = resolved_path.read_text()
    resolved = json.loads(resolved_text)
    assert resolved["maker"]["function"] == function


def test_a_points_record_names_the_maker_its_row_answers(tmp_path, monkeypatch):
    """The row says what made its workload; the runner names no row.

    The producer row under a second name answers the same maker, so the
    record holds it whatever the row is called.
    """
    monkeypatch.setitem(
        workload_settings.WORKLOADS,
        "made_elsewhere",
        workload_settings.ProducerWorkload,
    )
    workload = yaml_configs.memory_workload(15)
    workload["kind"] = "made_elsewhere"
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    out_dir = tmp_path / "out"

    run_command.run_one_shot(config_path, out_dir=out_dir)

    resolved_dir = out_dir / "resolved"
    resolved_path = _one_file(resolved_dir, "*.json")
    resolved_text = resolved_path.read_text()
    resolved = json.loads(resolved_text)
    maker = resolved["maker"]
    assert maker["function"] == "decsim.producers:memory_circuit"
    assert maker["arguments"]["rounds_per_shot"] == 15


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


def test_a_collect_run_again_into_its_folder_reruns_no_saved_piece(tmp_path):
    """A killed or finished collect run again skips every piece it saved.

    A piece's folder is written whole or not at all, so one that exists
    holds its shots, and the fold of the second run is the first's.
    """
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    first_status = piece_path.stat()
    report_dir = yaml_configs.run_folder_of(out_dir)
    first_rows = _rows_without_wall_clock(report_dir, "sweep.csv")

    command.main(["collect", str(config_path), "--out", str(out_dir)])

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
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    piece_text = piece_path.read_text()
    piece = json.loads(piece_text)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_after_mb = usage.ru_maxrss / 1024
    assert 0 < piece["peak_memory_mb"] <= peak_after_mb


def test_a_cut_run_with_a_deleted_piece_run_again_is_the_uncut_run(tmp_path):
    """The uncut collect is the oracle for a cut one that lost a piece.

    Pieces of one shot each, one of them deleted as a killed job leaves
    it missing, then the same collect again: it runs that piece alone
    and every folded file is the uncut run's.
    """
    whole_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_card = {**FOUR_POINT_SWEEP, "collection": {"piece_rounds": 1}}
    cut_path = yaml_configs.write_config(cut_folder, cut_card)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["collect", str(whole_path), "--out", str(whole_dir)])
    command.main(["collect", str(cut_path), "--out", str(cut_dir)])
    second_pieces = cut_dir.glob("pieces/*/1-1")
    cut_pieces = sorted(second_pieces)
    lost_piece = cut_pieces[0]
    kept_piece = cut_pieces[1]
    kept_status = (kept_piece / "piece.json").stat()
    shutil.rmtree(lost_piece)

    command.main(["collect", str(cut_path), "--out", str(cut_dir)])

    reissued_status = (kept_piece / "piece.json").stat()
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_rows = _rows_of_every_file(whole_run_dir)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    cut_rows = _rows_of_every_file(cut_run_dir)
    assert lost_piece.is_dir()
    assert reissued_status.st_mtime_ns == kept_status.st_mtime_ns
    assert cut_rows == whole_rows


def test_a_staging_folder_a_killed_run_left_is_no_piece(tmp_path):
    """A killed writer's staging folder: the piece runs, the fold skips it."""
    config_path = yaml_configs.write_config(tmp_path, {})
    whole_dir = tmp_path / "whole"
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(whole_dir)])
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

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    written = out_dir / "pieces" / point_dir.name / piece_path.parent.name
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_rows = _rows_of_every_file(whole_run_dir)
    out_run_dir = yaml_configs.run_folder_of(out_dir)
    out_rows = _rows_of_every_file(out_run_dir)
    assert (written / "piece.json").exists()
    assert out_rows == whole_rows


def test_two_writers_of_one_piece_both_leave_it_whole(tmp_path, monkeypatch):
    """Two tasks handed the same piece write it at once and neither fails.

    The second writer runs whole while the first is between its files
    and its piece.json. Each ends with the piece in place and no staging
    folder left behind.
    """
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    ((task, _collection),) = config.point_tasks()
    measurements = yaml_configs.run_sweep([task], 1)
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


NOISY_AXES = {
    "workload.arguments.physical_error_probability": [0.02],
    "qpu.distance": [3],
    "qpu.round_period_microseconds": [1.0],
}


def _piece_counts_read_by_csv(folder: pathlib.Path) -> dict:
    """A piece's counts summed straight from its own shots.csv."""
    counts_names = (
        "is_scored",
        "logical_failure",
        "sim_wall_seconds",
        *report.STATUS_SUMS,
    )
    sums = dict.fromkeys(counts_names, 0)
    shots_path = folder / "shots.csv"
    shot_rows = _csv_rows(shots_path)
    for row in shot_rows:
        typed = fold.typed_row(row)
        for name in counts_names:
            sums[name] += typed[name]
    shots = len(shot_rows)
    scored_shots = sums.pop("is_scored")
    core_seconds = sums.pop("sim_wall_seconds")
    counts = {
        "count": shots,
        "scored_shots": scored_shots,
        "failures": sums.pop("logical_failure"),
        "unscored_shots": shots - scored_shots,
        "core_seconds": pytest.approx(core_seconds),
    }
    return {**counts, **sums}


def _piece_facts_and_references(experiment_dir: pathlib.Path) -> list:
    """Each piece's piece.json lines, beside what its own files give."""
    configurations_path = experiment_dir / "configurations.csv"
    (configuration,) = _csv_rows(configurations_path)
    records = run_folder.resolved_by_point(experiment_dir)
    pairs = []
    folders = experiment_dir.glob("pieces/*/*")
    for folder in sorted(folders):
        piece = pieces.read_piece(folder)
        reference = _piece_counts_read_by_csv(folder)
        rounds_per_shot = records[folder.parent.name]["rounds_per_shot"]
        reference["configuration_id"] = configuration["configuration_id"]
        reference["rounds"] = rounds_per_shot * reference["count"]
        written = {name: piece[name] for name in reference}
        pairs.append((written, reference))
    return pairs


def test_a_configuration_line_is_staged_under_its_writers_own_name(tmp_path):
    """Two collects of one experiment may record their lines at once.

    Each stages configurations.csv under a name no other writer uses,
    so a copy another writer is staging beside it is left as it was.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    config = experiment.load_experiment(config_path)
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    others_copy = experiment_dir / ".configurations.csv.partial"
    others_copy.write_text("another writer's copy\n")

    run_folder.record_configuration(experiment_dir, config)

    configurations = run_folder.recorded_configurations(experiment_dir)
    assert others_copy.read_text() == "another writer's copy\n"
    assert list(configurations) == [run_folder.configuration_id(config)]


def test_a_configuration_path_with_a_comma_quote_and_semicolon_reads_back(
    tmp_path,
):
    """Python's csv module reads the row decsim wrote, and reopens its yaml."""
    folder = tmp_path / "semi;colon"
    folder.mkdir()
    written_path = yaml_configs.write_config(folder, FOUR_POINT_SWEEP)
    config_path = folder / 'my,"quoted".yaml'
    written_path.rename(config_path)
    config = experiment.load_experiment(config_path)
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()

    run_folder.record_configuration(experiment_dir, config)

    configurations_path = experiment_dir / "configurations.csv"
    (row,) = _csv_rows(configurations_path)
    configurations = run_folder.recorded_configurations(experiment_dir)
    (reopened,) = configurations[row["configuration_id"]]
    assert row["name"] == 'my,"quoted"'
    assert reopened.config_files[0] == config.config_files[0]


def test_a_piece_records_its_counts_rounds_and_configuration(tmp_path):
    """piece.json's lines against the piece's own shots.csv, summed by csv.

    A planner or a status reads piece.json without the shot rows, so
    each count there is the sum of its column over the piece's shots,
    its rounds are its shots times the point's rounds per shot, and its
    configuration is the one configurations.csv names.
    """
    collection = {"max_shots": 12, "piece_rounds": 45}
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    pairs = _piece_facts_and_references(out_dir)
    written = [pair[0] for pair in pairs]
    references = [pair[1] for pair in pairs]
    assert len(pairs) > 1
    assert written == references


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
    whole_card = {
        "sweep": [{"axes": NOISY_AXES, "collection": {"max_shots": 30}}]
    }
    whole_path = yaml_configs.write_config(tmp_path, whole_card)
    target_folder = tmp_path / "target_config"
    target_folder.mkdir()
    collection = {"max_shots": 30, "max_failures": 3, "piece_rounds": 1}
    target_card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    target_path = yaml_configs.write_config(target_folder, target_card)
    whole_dir = tmp_path / "whole"
    target_dir = tmp_path / "target"
    command.main(["collect", str(whole_path), "--out", str(whole_dir)])
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_shots_path = whole_run_dir / "shots.csv"
    expected_shots = _shots_to_the_target(whole_shots_path, 3)

    command.main(["collect", str(target_path), "--out", str(target_dir)])

    target_run_dir = yaml_configs.run_folder_of(target_dir)
    target_pieces = target_dir.glob("pieces/*/*")
    assert expected_shots < 30
    assert len(list(target_pieces)) == expected_shots
    assert _shot_count_of(target_run_dir) == expected_shots


# a shot of NOISY_AXES runs 15 rounds, so a piece holds two shots
TWO_SHOT_PIECE_ROUNDS = 30


def test_a_raised_shot_cap_runs_on_from_the_saved_pieces(tmp_path):
    """The referent is one collect run to the raised cap from the start.

    A cap of 3 in pieces of two ends on a piece of one shot, 2-2. The
    cap raised to 4 runs seed 3 alone beside it, so no seed is run
    twice, and the fold is the uncut run's.
    """
    whole_dir = tmp_path / "whole"
    raised_dir = tmp_path / "raised"
    whole_run_dir = _collected_noisy_point(
        tmp_path, "whole_config", {"max_shots": 4}, whole_dir
    )
    _collected_noisy_point(tmp_path, "first", {"max_shots": 3}, raised_dir)

    raised_run_dir = _collected_noisy_point(
        tmp_path, "second", {"max_shots": 4}, raised_dir
    )

    whole_rows = _rows_of_every_file(whole_run_dir)
    raised_rows = _rows_of_every_file(raised_run_dir)
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
    card = {"sweep": [{"axes": NOISY_AXES, "collection": keys}]}
    config_path = yaml_configs.write_config(folder, card)
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    return yaml_configs.run_folder_of(out_dir)


def _piece_names(experiment_dir) -> list:
    folders = experiment_dir.glob("pieces/*/*")
    return sorted(folder.name for folder in folders)


def test_a_resumed_collect_keeps_every_saved_shots_residence_rows(tmp_path):
    """The uncut collect's residence.csv is the referent of a resumed one's.

    Three traced shots in pieces of one; the first piece is lost, as a
    killed job leaves it, and the collect run again runs it alone. Its
    residence table still holds the rows of all three traced shots.
    """
    card = {
        "observation": {"trace": "chrome", "trace_shots": [0, 1, 2]},
        "sweep": [
            {
                "axes": NOISY_AXES,
                "collection": {"max_shots": 3, "piece_rounds": 15},
            }
        ],
    }
    config_path = yaml_configs.write_config(tmp_path, card)
    whole_dir = tmp_path / "whole"
    resumed_dir = tmp_path / "resumed"
    command.main(["collect", str(config_path), "--out", str(whole_dir)])
    command.main(["collect", str(config_path), "--out", str(resumed_dir)])
    (first_piece,) = resumed_dir.glob("pieces/*/0-0")
    shutil.rmtree(first_piece)

    command.main(["collect", str(config_path), "--out", str(resumed_dir)])

    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_path = whole_run_dir / "residence.csv"
    whole_rows = _csv_rows(whole_path)
    resumed_run_dir = yaml_configs.run_folder_of(resumed_dir)
    resumed_path = resumed_run_dir / "residence.csv"
    resumed_rows = _csv_rows(resumed_path)
    seeds = {row["seed"] for row in resumed_rows}
    assert seeds == {"0", "1", "2"}
    assert resumed_rows == whole_rows


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
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    run_dir = yaml_configs.run_folder_of(out_dir)
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
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    serial_dir = tmp_path / "serial"
    pooled_dir = tmp_path / "pooled"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    pooled = ["collect", str(config_path), "--out", str(pooled_dir)]

    command.main([*pooled, "--processes", "8"])

    serial_run_dir = yaml_configs.run_folder_of(serial_dir)
    pooled_run_dir = yaml_configs.run_folder_of(pooled_dir)
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
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    first_pieces = out_dir.glob("pieces/*/*")
    first_names = sorted(first_pieces)
    capsys.readouterr()

    command.main(["collect", str(config_path), "--out", str(out_dir)])

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
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    point_id = task.strong_id()
    _write_a_saved_piece(tmp_path, point_id, 0, [True])
    _write_a_saved_piece(tmp_path, point_id, 2, [True])
    settings = collection_module.CollectionSettings(max_shots=3, max_failures=2)
    point_folders = pieces.folders_of(tmp_path, [point_id])
    saved = pieces.saved_counts(point_folders)
    point = collect_command.PointCollection(task, settings, 1, 15, saved)

    units = point.next_units(tmp_path, 1)

    assert units == [collect.Unit(task, 1, 1)]
    assert point.tracker.stop_kind is None
    assert point.tracker.counts.failures == 1


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
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    point_id = task.strong_id()
    failed = [False, True, False, False]
    _write_a_saved_piece(tmp_path, point_id, 0, failed)
    settings = collection_module.CollectionSettings(
        max_shots=10, max_failures=1, min_shots=2
    )
    point_folders = pieces.folders_of(tmp_path, [point_id])
    saved = pieces.saved_counts(point_folders)
    point = collect_command.PointCollection(task, settings, 4, 15, saved)

    units = point.next_units(tmp_path, 1)

    printed = capsys.readouterr()
    assert units == []
    assert point.tracker.stop_kind is failure_statistics.StopKind.MINIMUM
    assert point.tracker.counts.shots == 2
    assert printed.err.endswith(
        ": 2 shots done (minimum); 2 more ran past the stop\n"
    )


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
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit):
        command.main(
            [
                "collect",
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


def test_a_sweep_block_that_says_shots_is_refused(tmp_path, capsys):
    """A block's shots are its collection's max_shots, and nothing else."""
    card = {"sweep": [{"axes": NOISY_AXES, "shots": 2}]}
    config_path = yaml_configs.write_config(tmp_path, card)

    with pytest.raises(SystemExit):
        command.main(["collect", str(config_path)])

    printed = capsys.readouterr()
    assert "sweep block 1 is" in printed.err
    assert "a collection of its own, where max_shots is" in printed.err


def test_a_point_two_blocks_collect_two_ways_is_refused(tmp_path, capsys):
    first = {"axes": NOISY_AXES, "collection": {"max_shots": 2}}
    second = {"axes": NOISY_AXES, "collection": {"max_shots": 3}}
    card = {"sweep": [first, second]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit):
        command.main(["collect", str(config_path), "--out", str(out_dir)])

    printed = capsys.readouterr()
    lines = printed.err.splitlines()
    assert "is in two sweep blocks that collect it two ways" in lines[-1]
    assert "max_shots 2" in lines[-1]
    assert "max_shots 3" in lines[-1]


def test_an_online_point_given_a_target_is_refused(tmp_path, capsys):
    """An online point's shots are not independent, so no target stops it."""
    overrides = yaml_configs.online_threshold()
    overrides["collection"] = {"max_failures": 5}
    config_path = yaml_configs.write_config(tmp_path, overrides)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit):
        command.main(["collect", str(config_path), "--out", str(out_dir)])

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
    command.main(["collect", str(whole_config), "--out", str(whole_dir)])
    cut_config = _online_config(tmp_path, 15)
    command.main(["collect", str(cut_config), "--out", str(cut_dir)])
    _lose_the_piece(cut_dir, "2-2")
    _lose_the_piece(cut_dir, "3-3")

    command.main(["collect", str(cut_config), "--out", str(cut_dir)])

    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    whole_decisions = _shot_decisions(whole_run_dir)
    cut_decisions = _shot_decisions(cut_run_dir)
    whole_trajectory = _online_trajectory_rows(whole_run_dir)
    cut_trajectory = _online_trajectory_rows(cut_run_dir)
    assert _piece_names(cut_dir) == ["0-0", "1-1", "2-2", "3-3"]
    assert cut_decisions == whole_decisions
    assert whole_trajectory[-1]["window_count"] == "20"
    assert cut_trajectory == whole_trajectory


def test_a_saved_calibrator_whose_bytes_changed_is_refused(tmp_path):
    """A state that no longer hashes to its record is never unpickled.

    The piece's calibrator reads back as saved; one byte appended to its
    file, as a damaged copy on a shared disk would hold, and the read
    refuses it by name.
    """
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    measurements = yaml_configs.run_sweep([task], 1)
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
    overrides = yaml_configs.online_threshold()
    overrides["escalation"]["online"] = {
        "audit_rate": 0.3,
        "target_escalation_rate": 0.2,
        "max_escalation_rate": 0.5,
    }
    collection = {"max_shots": 4, "piece_rounds": piece_rounds}
    overrides["sweep"] = [{"axes": NOISY_AXES, "collection": collection}]
    return yaml_configs.write_config(tmp_path, overrides)


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
    section 2.3): one column per yaml path the sweep sets, right after
    the point id, in every file a point's rows are in.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    run_dir = yaml_configs.run_folder_of(out_dir)
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


def test_a_point_holds_its_own_value_at_a_path_another_block_sets(tmp_path):
    """Each point has a value at every swept path, so no cell is missing.

    The first block leaves the window and the frame as the file writes
    them, and the second block leaves the distance; a mapping is one
    cell of compact json, as sinter writes json_metadata
    (sinter/_data/_csv_out.py:35-37).
    """
    frame = {"clock": "fridge", "write_cycles": 2}
    card = {
        "qpu": {"kind": "stim_device", "distance": 3},
        "workload": yaml_configs.memory_workload(15),
        "sweep": [
            {"axes": {"qpu.distance": [5]}, "collection": {"max_shots": 1}},
            {
                "axes": {"windows.commit_rounds": [2], "pauli_frame": [frame]},
                "collection": {"max_shots": 1},
            },
        ],
    }
    card["workload"]["arguments"]["physical_error_probability"] = 0.001
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    sweep_path = yaml_configs.run_folder_of(out_dir) / "sweep.csv"
    first, second = _csv_rows(sweep_path)
    assert first["qpu.distance"] == "5"
    assert first["windows.commit_rounds"] == "null"
    assert first["pauli_frame"] == '{"clock":"fridge","write_cycles":1}'
    assert second["qpu.distance"] == "3"
    assert second["windows.commit_rounds"] == "2"
    assert second["pauli_frame"] == '{"clock":"fridge","write_cycles":2}'


def test_a_swept_reference_is_written_as_the_value_it_resolved_to(tmp_path):
    """A table is built from the value a point ran, not the text it wrote."""
    card = {
        "qpu": {"kind": "stim_device", "distance": 7},
        "workload": yaml_configs.memory_workload(15),
        "sweep": [
            {
                "axes": {
                    "qpu.distance": ["${windows.commit_rounds}"],
                    "windows.commit_rounds": [3, 5],
                },
                "collection": {"max_shots": 1},
            }
        ],
    }
    card["workload"]["arguments"]["physical_error_probability"] = 0.001
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    sweep_path = yaml_configs.run_folder_of(out_dir) / "sweep.csv"
    rows = _csv_rows(sweep_path)
    distances = [row["qpu.distance"] for row in rows]
    assert distances == ["3", "5"]


def test_a_child_axis_under_a_swept_mapping_shows_in_the_mappings_cell(
    tmp_path,
):
    """The mapping is placed first, then its child: the point ran both."""
    frame = {"clock": "fridge", "write_cycles": 1}
    card = {
        "qpu": {"kind": "stim_device", "distance": 3},
        "workload": yaml_configs.memory_workload(15),
        "sweep": [
            {
                "axes": {
                    "pauli_frame": [frame],
                    "pauli_frame.write_cycles": [2],
                },
                "collection": {"max_shots": 1},
            }
        ],
    }
    card["workload"]["arguments"]["physical_error_probability"] = 0.001
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    sweep_path = yaml_configs.run_folder_of(out_dir) / "sweep.csv"
    (row,) = _csv_rows(sweep_path)
    assert row["pauli_frame"] == '{"clock":"fridge","write_cycles":2}'
    assert row["pauli_frame.write_cycles"] == "2"


def test_every_point_records_its_own_makers_arguments(tmp_path):
    """Each point's maker was called with its own arguments and version."""
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"

    command.main(["collect", str(config_path), "--out", str(out_dir)])

    records = run_folder.resolved_by_point(out_dir)
    arguments = [record["maker"]["arguments"] for record in records.values()]
    made_at = {
        (made["physical_error_probability"], made["distance"])
        for made in arguments
    }
    assert made_at == {(0.001, 3), (0.001, 5), (0.003, 3), (0.003, 5)}


def test_a_manifest_names_every_installed_package(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)
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
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)
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
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"

    run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)

    manifest = _manifest_of(out_dir)
    expected = _named_library(library_path, expected_digest)
    assert manifest["compiled_libraries"] == expected


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
    serial_run_dir = yaml_configs.run_folder_of(serial_dir)
    serial_rows = _rows_of_every_file(serial_run_dir)
    pooled_run_dir = yaml_configs.run_folder_of(pooled_dir)
    pooled_rows = _rows_of_every_file(pooled_run_dir)
    assert pooled_rows == serial_rows


def test_pieces_of_one_shot_fold_to_the_rows_of_one_piece_a_point(tmp_path):
    """The additive record's whole point: a folded point is the point.

    One collect saves each point as one piece; another cuts each point
    into pieces of one shot, serially and in a pool. Every file the
    three fold to is the same, row for row, but for the wall clock.
    """
    whole_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_card = {**FOUR_POINT_SWEEP, "collection": {"piece_rounds": 1}}
    cut_path = yaml_configs.write_config(cut_folder, cut_card)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    pooled_dir = tmp_path / "pooled"
    command.main(["collect", str(whole_path), "--out", str(whole_dir)])
    command.main(["collect", str(cut_path), "--out", str(cut_dir)])
    pooled = ["collect", str(cut_path), "--out", str(pooled_dir)]
    command.main([*pooled, "--processes", "4"])

    whole_pieces = whole_dir.glob("pieces/*/*")
    cut_pieces = cut_dir.glob("pieces/*/*")
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_rows = _rows_of_every_file(whole_run_dir)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    cut_rows = _rows_of_every_file(cut_run_dir)
    pooled_run_dir = yaml_configs.run_folder_of(pooled_dir)
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
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "run"
    monkeypatch.chdir(tmp_path)
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    run_dir = yaml_configs.run_folder_of(out_dir)
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
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    dirty_dir = tmp_path / "dirty"
    clean_dir = tmp_path / "clean"
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "1")
    run_folder._tree_reading.cache_clear()
    command.main(["collect", str(config_path), "--out", str(dirty_dir)])
    monkeypatch.setenv(run_folder.TREE_DIRTY_VARIABLE, "0")
    run_folder._tree_reading.cache_clear()
    command.main(["collect", str(config_path), "--out", str(clean_dir)])

    dirty_run_dir = yaml_configs.run_folder_of(dirty_dir)
    clean_run_dir = yaml_configs.run_folder_of(clean_dir)
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


def test_two_pieces_that_hold_the_same_shot_are_refused(tmp_path, capsys):
    """A piece copied in under another range would count its shots twice."""
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    pieces_dir = out_dir / "pieces"
    piece_path = _one_file(pieces_dir, "*/*/piece.json")
    piece_dir = piece_path.parent
    copied_dir = piece_dir.with_name("5-5")
    shutil.copytree(piece_dir, copied_dir)
    capsys.readouterr()

    with pytest.raises(SystemExit) as stopped:
        command.main(["collect", str(config_path), "--out", str(out_dir)])

    printed = capsys.readouterr()
    lines = printed.err.splitlines()
    assert stopped.value.code == 1
    assert "is in more than one of" in lines[-1]


def test_run_refuses_a_yaml_that_is_not_there(tmp_path, capsys):
    missing = tmp_path / "not_a_config.yaml"
    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(missing)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(f"decsim: {missing} is not a file")
    assert "reference.yaml" in printed.err
    assert "examples/two_tiers.yaml" in printed.err
    assert "bases/" not in printed.err


def test_show_refuses_a_sweep_axis_the_yaml_layer_does_not_have(
    tmp_path, capsys
):
    unknown_axis = {
        "sweep": [
            {
                "axes": {
                    "workload.arguments.physical_error_probability": [0.001],
                    "qpu.distance": [3],
                    "qpu.round_period_microseconds": [1.0],
                },
                "algorithm": ["pymatching"],
                "collection": {"max_shots": 1},
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, unknown_axis)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "sweep block 1 is " in printed.err
    assert "a block is axes" in printed.err


@pytest.mark.parametrize("shots", [0, -1, 1.5, "many", True])
def test_show_refuses_a_shot_cap_that_is_not_a_whole_number_of_one_or_more(
    tmp_path, capsys, shots
):
    bad_count = {
        "sweep": [
            {
                "axes": {
                    "workload.arguments.physical_error_probability": [0.001],
                    "qpu.distance": [3],
                    "qpu.round_period_microseconds": [1.0],
                },
                "collection": {"max_shots": shots},
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, bad_count)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    refusal_text = "sweep block 1 collection max_shots must be a whole number"
    assert refusal_text in printed.err


def test_show_refuses_a_sweep_axis_written_as_one_value_not_a_list(
    tmp_path, capsys
):
    scalar_axis = {
        "sweep": [
            {
                "axes": {
                    "workload.arguments.physical_error_probability": [0.001],
                    "qpu.distance": 3,
                    "qpu.round_period_microseconds": [1.0],
                },
                "collection": {"max_shots": 1},
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, scalar_axis)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    sentence = "sweep block 1 axis qpu.distance must be a list of at least"
    assert sentence in printed.err


def test_plot_refuses_a_figure_it_does_not_draw(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        command.main(["plot", str(tmp_path), "--figure", "everything"])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: no figure named everything")


def test_trace_refuses_an_action_it_does_not_have(tmp_path, capsys):
    trace_path = tmp_path / "shot.json"
    with pytest.raises(SystemExit) as stopped:
        command.main(["trace", "summarise", str(trace_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: decsim trace has no action")


def test_a_build_refusal_under_run_is_one_line(tmp_path, capsys):
    """A yaml that loads but that a row refuses at build, as a sentence.

    burst_detector kind event_count sends a burst's windows to the
    strong decoder, which only escalation kind switching has, so the
    weak baseline is refused when the machine is built
    (decsim/build/escalation.py).
    """
    base_path = CONFIGS_DIR / "bases/weak_decoder_baseline.yaml"
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


def _plan(config_paths: list, experiment_dir: pathlib.Path, tasks: int):
    """The yamls planned by `decsim plan`; the new round's folder or None."""
    arguments = [str(path) for path in config_paths]
    before = pieces.round_dirs(experiment_dir)
    out = ["--out", str(experiment_dir), "--tasks", str(tasks)]
    command.main(["plan", *arguments, *out])
    after = pieces.round_dirs(experiment_dir)
    if len(after) == len(before):
        return None
    return after[-1]


def _run_the_round(round_dir: pathlib.Path, task_order=None) -> None:
    """Every task of a round's plan, each as its own collect --plan."""
    tasks_path = round_dir / "tasks.csv"
    task_numbers = [row["task"] for row in _csv_rows(tasks_path)]
    if task_order is not None:
        task_numbers = task_order
    plan_path = round_dir / "plan.csv"
    for task_number in task_numbers:
        command.main(
            ["collect", "--plan", str(plan_path), "--task", str(task_number)]
        )


def _plan_and_run_until_stopped(config_paths: list, experiment_dir) -> int:
    """Rounds planned and run until every point has stopped; how many."""
    rounds = 0
    while True:
        round_dir = _plan(config_paths, experiment_dir, 3)
        if round_dir is None:
            return rounds
        _run_the_round(round_dir)
        rounds += 1


def _planned_ranges(round_dir: pathlib.Path) -> list:
    """A round's pieces as (point id, first seed, count)."""
    plan_path = round_dir / "plan.csv"
    ranges = []
    for _task, piece in pieces.read_plan(plan_path):
        ranges.append((piece.point_id, piece.first_seed, piece.count))
    return ranges


def _cut_four_point_sweep(tmp_path) -> pathlib.Path:
    """The four-point sweep, its pieces one shot each, in its own folder."""
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_card = {**FOUR_POINT_SWEEP, "collection": {"piece_rounds": 1}}
    return yaml_configs.write_config(cut_folder, cut_card)


def test_a_planned_run_with_a_lost_piece_planned_again_is_the_uncut_run(
    tmp_path,
):
    """The uncut collect is the oracle for rounds that lost a piece.

    Round one cuts every point into pieces of one shot over three tasks;
    one piece is then deleted, as a task killed before its rename leaves
    it missing. Round two plans that piece alone, with its own seeds,
    and the status fold of both rounds is the uncut run's, file by file;
    status writes points in its own order, so the rows are compared in
    one order.
    """
    whole_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    cut_path = _cut_four_point_sweep(tmp_path)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["collect", str(whole_path), "--out", str(whole_dir)])
    first_round = _plan([cut_path], cut_dir, 3)
    _run_the_round(first_round)
    second_pieces = cut_dir.glob("pieces/*/1-1")
    lost_piece, *_kept = sorted(second_pieces)
    shutil.rmtree(lost_piece)

    second_round = _plan([cut_path], cut_dir, 3)
    _run_the_round(second_round)
    command.main(["status", str(cut_dir)])

    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_rows = _sorted_rows_of_every_file(whole_run_dir)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    cut_rows = _sorted_rows_of_every_file(cut_run_dir)
    lost_point = lost_piece.parent.name
    assert _planned_ranges(second_round) == [(lost_point, 1, 1)]
    assert _plan([cut_path], cut_dir, 3) is None
    assert cut_rows == whole_rows


def _rounds_planned_without_running(
    config_path: pathlib.Path, experiment_dir: pathlib.Path, rounds: int
) -> list:
    """Each round's sorted pieces, for rounds planned with none run."""
    planned_rounds = []
    for _round in range(rounds):
        round_dir = _plan([config_path], experiment_dir, 2)
        round_ranges = _planned_ranges(round_dir)
        planned_rounds.append(sorted(round_ranges))
    return planned_rounds


def _rounds_of_saved_pieces(experiment_dir: pathlib.Path) -> set:
    """The rounds whose tasks saved the experiment's pieces."""
    rounds = set()
    for folder in experiment_dir.glob("pieces/*/*"):
        piece = pieces.read_piece(folder)
        rounds.add(piece["round"])
    return rounds


def test_a_piece_no_round_ran_is_planned_once_a_round(tmp_path):
    """A piece every earlier round listed and none ran is planned once.

    Rounds whose arrays never started leave each of their pieces listed
    in several plans; the next plan still names each seed range once,
    and the same pieces every round.
    """
    config_path = _cut_four_point_sweep(tmp_path)
    experiment_dir = tmp_path / "experiment"

    planned_rounds = _rounds_planned_without_running(
        config_path, experiment_dir, 4
    )

    first_round = planned_rounds[0]
    assert len(set(first_round)) == len(first_round)
    assert planned_rounds == [first_round] * 4


def _targeted_sweep(tmp_path) -> pathlib.Path:
    """The four-point sweep with a failure target, in one-shot pieces.

    The target is out of reach, so every round extends each point.
    """
    (block,) = FOUR_POINT_SWEEP["sweep"]
    targeted_block = {
        **block,
        "collection": {"max_failures": 1000, "max_shots": 8},
    }
    card = {"sweep": [targeted_block], "collection": {"piece_rounds": 1}}
    return yaml_configs.write_config(tmp_path, card)


def test_a_piece_saved_by_an_earlier_round_is_not_run_or_planned_again(
    tmp_path,
):
    """A piece planned again, then saved by its first round, is done.

    Round two plans round one's pieces again while none is saved; once
    round one saves them, round two's tasks skip them and round three
    plans only new seeds.
    """
    config_path = _targeted_sweep(tmp_path)
    experiment_dir = tmp_path / "experiment"
    first_round = _plan([config_path], experiment_dir, 2)
    second_round = _plan([config_path], experiment_dir, 2)

    _run_the_round(first_round)
    _run_the_round(second_round)
    third_round = _plan([config_path], experiment_dir, 2)

    first_ranges = _planned_ranges(first_round)
    second_ranges = _planned_ranges(second_round)
    third_ranges = _planned_ranges(third_round)
    assert second_ranges == first_ranges
    assert _rounds_of_saved_pieces(experiment_dir) == {1}
    assert not set(first_ranges) & set(third_ranges)


def _one_distance_config(tmp_path, distance: int) -> pathlib.Path:
    """The four-point sweep at one distance, in a folder of its own."""
    folder = tmp_path / f"d{distance}"
    folder.mkdir()
    (block,) = FOUR_POINT_SWEEP["sweep"]
    axes = {**block["axes"], "qpu.distance": [distance]}
    card = {
        "sweep": [{**block, "axes": axes}],
        "collection": {"piece_rounds": 1},
    }
    return yaml_configs.write_config(folder, card)


def _sorted_rows_of_every_file(run_dir) -> dict:
    """Each file's rows without the wall clock, in one order for all."""
    rows = _rows_of_every_file(run_dir)
    return {name: sorted(rows[name], key=_row_key) for name in rows}


def _row_key(row: dict) -> list:
    cells = row.items()
    return sorted(cells)


def test_pieces_of_two_distances_fold_to_one_run(tmp_path):
    """Two yamls, one a distance, of one configuration are one run.

    The sweep and the collection are no part of a configuration's id, so
    a grid split one file a distance is one configuration. Planned
    together, run as two tasks in reverse order, and folded by status,
    its rows are the uncut two-distance run's; the fold writes points in
    its own order, so the rows are compared in one order.
    """
    whole_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    split_paths = [
        _one_distance_config(tmp_path, 3),
        _one_distance_config(tmp_path, 5),
    ]
    whole_dir = tmp_path / "whole"
    split_dir = tmp_path / "split"
    command.main(["collect", str(whole_path), "--out", str(whole_dir)])
    round_dir = _plan(split_paths, split_dir, 2)

    _run_the_round(round_dir, task_order=[1, 0])
    command.main(["status", str(split_dir)])

    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    whole_rows = _sorted_rows_of_every_file(whole_run_dir)
    split_run_dir = yaml_configs.run_folder_of(split_dir)
    split_rows = _sorted_rows_of_every_file(split_run_dir)
    assert split_rows == whole_rows


def _typed_status_rows(experiment_dir: pathlib.Path) -> list:
    status_path = experiment_dir / "status.csv"
    return report.read_rows(status_path)


def _estimate_by_the_statistics(row: dict) -> tuple:
    """failure_statistics on a status row's own prefix counts and stop."""
    stop_kind = failure_statistics.StopKind(row["state"])
    estimate = failure_statistics.estimate(
        row["prefix_failures"], row["prefix_scored_shots"], stop_kind
    )
    return (estimate.rate, estimate.low, estimate.high)


def _estimate_of(row: dict) -> tuple:
    """A status row's estimate and limits, an empty cell as None."""
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


def test_a_status_rows_estimate_is_failure_statistics_on_its_counts(tmp_path):
    """The referent is failure_statistics on the row's own counts and stop.

    One noisy point with a target of three failures runs round after
    round until the plan has nothing left; status gives it the target
    state, and its estimate and exact limits are the ones the
    statistics module gives the same failures, scored shots and stop.
    """
    collection = {"max_shots": 60, "max_failures": 3, "piece_rounds": 45}
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"

    rounds = _plan_and_run_until_stopped([config_path], out_dir)
    command.main(["status", str(out_dir)])

    (row,) = _typed_status_rows(out_dir)
    assert rounds > 1
    assert row["state"] == "target"
    assert row["prefix_failures"] == 3
    assert _estimate_of(row) == _estimate_by_the_statistics(row)


def test_an_online_points_planned_pieces_run_in_one_task_as_the_uncut_point(
    tmp_path,
):
    """The referent is the uncut collect: one piece of all four shots.

    Planned over three tasks, the online point's four one-shot pieces go
    to one task in seed order, each starting from the calibrator the one
    before it saved; a collect then finds them all saved and folds them.
    Its decisions and threshold trajectory are the uncut run's.
    """
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    whole_config = _online_config(tmp_path, 60)
    command.main(["collect", str(whole_config), "--out", str(whole_dir)])
    cut_config = _online_config(tmp_path, 15)

    round_dir = _plan([cut_config], cut_dir, 3)
    _run_the_round(round_dir)
    command.main(["collect", str(cut_config), "--out", str(cut_dir)])

    plan_path = round_dir / "plan.csv"
    plan_rows = _csv_rows(plan_path)
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    whole_trajectory = _online_trajectory_rows(whole_run_dir)
    cut_trajectory = _online_trajectory_rows(cut_run_dir)
    assert _values_of_rows(plan_rows, ("task", "first_seed")) == [
        ("0", "0"),
        ("0", "1"),
        ("0", "2"),
        ("0", "3"),
    ]
    assert _shot_decisions(cut_run_dir) == _shot_decisions(whole_run_dir)
    assert cut_trajectory == whole_trajectory


def _values_of_rows(rows: list, columns: tuple) -> list:
    values = []
    for row in rows:
        row_values = _values_of(row, columns)
        values.append(row_values)
    return values


def test_a_rounds_extension_is_openmcs_ratio_of_the_target(tmp_path):
    """Round two plans shots x (target / failures) in all, less round one's.

    OpenMC extends a run by the ratio of the uncertainty it has to the
    one it wants, squared (trigger.cpp); for a failure count the squared
    ratio is the target over the failures seen. Round one plans one
    piece, having nothing measured.
    """
    collection = {"max_shots": 1000, "max_failures": 50, "piece_rounds": 90}
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"
    first_round = _plan([config_path], out_dir, 3)
    _run_the_round(first_round)
    (first_piece,) = out_dir.glob("pieces/*/*")
    counts = pieces.read_piece(first_piece)
    needed = counts["count"] * 50 / counts["failures"]
    wanted = math.ceil(needed)

    second_round = _plan([config_path], out_dir, 3)

    first_ranges = _planned_ranges(first_round)
    second_ranges = _planned_ranges(second_round)
    second_shots = sum(piece[2] for piece in second_ranges)
    assert first_ranges[0][1:] == (0, 6)
    assert counts["failures"] > 0
    assert second_shots == wanted - counts["count"]


def test_a_task_whose_point_its_yaml_no_longer_makes_is_refused(
    tmp_path, capsys
):
    """A plan names points by id, so a yaml changed after it is refused.

    The task's yaml now makes other points, so the plan's point is none
    of them, and the task says so and asks for a new plan rather than
    running something the plan did not name.
    """
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    round_dir = _plan([config_path], out_dir, 1)
    changed_card = {
        "sweep": [{"axes": NOISY_AXES, "collection": {"max_shots": 1}}]
    }
    yaml_configs.write_config(tmp_path, changed_card)
    plan_path = round_dir / "plan.csv"

    with pytest.raises(SystemExit):
        command.main(["collect", "--plan", str(plan_path), "--task", "0"])

    printed = capsys.readouterr()
    assert "no point of" in printed.err
    assert "plan again" in printed.err


def test_a_point_two_configurations_reach_is_planned_once(tmp_path):
    """One point, two configurations: its seeds are planned once.

    The yamls differ in a base distance their sweeps both set, so their
    configuration ids differ and their four points are the same four.
    """
    first_path = _base_distance_config(tmp_path, 5, FOUR_POINT_SWEEP)
    second_path = _base_distance_config(tmp_path, 7, FOUR_POINT_SWEEP)
    experiment_dir = tmp_path / "experiment"

    round_dir = _plan([first_path, second_path], experiment_dir, 3)

    planned = _planned_ranges(round_dir)
    point_ids = {point_id for point_id, _first, _count in planned}
    assert len(planned) == 4
    assert len(point_ids) == 4


def test_a_point_two_configurations_collect_two_ways_is_refused(
    tmp_path, capsys
):
    """A point stops by one rule, so two collections for it are refused.

    The first configuration has collected its points. A plan of both is
    refused before it writes anything, so every record keeps the owner
    and the stopping rule it had, byte for byte.
    """
    first_path = _base_distance_config(tmp_path, 5, FOUR_POINT_SWEEP)
    (block,) = FOUR_POINT_SWEEP["sweep"]
    other_block = {**block, "collection": {"max_shots": 3}}
    other_sweep = {"sweep": [other_block]}
    second_path = _base_distance_config(tmp_path, 7, other_sweep)
    experiment_dir = tmp_path / "experiment"
    command.main(["collect", str(first_path), "--out", str(experiment_dir)])
    records_before = _record_bytes(experiment_dir)
    arguments = [str(first_path), str(second_path)]
    out = ["--out", str(experiment_dir), "--tasks", "3"]

    with pytest.raises(SystemExit):
        command.main(["plan", *arguments, *out])

    printed = capsys.readouterr()
    round_dirs = pieces.round_dirs(experiment_dir)
    assert "two configurations" in printed.err
    assert round_dirs == []
    assert _record_bytes(experiment_dir) == records_before


def _record_bytes(experiment_dir: pathlib.Path) -> dict:
    """Each resolved/ record's name and bytes, and configurations.csv's."""
    paths = experiment_dir.glob("resolved/*.json")
    recorded = {path.name: path.read_bytes() for path in paths}
    configurations_path = experiment_dir / "configurations.csv"
    recorded["configurations.csv"] = configurations_path.read_bytes()
    return recorded


def _base_distance_config(tmp_path, distance: int, card: dict):
    """The card over a base qpu distance, in a folder of its own."""
    folder = tmp_path / f"base_distance_{distance}"
    folder.mkdir()
    qpu = {"kind": "stim_device", "distance": distance}
    return yaml_configs.write_config(folder, {**card, "qpu": qpu})


def test_status_keeps_a_point_its_yaml_no_longer_sweeps(tmp_path):
    """Status folds what was recorded, not what the yamls make now.

    Four points are collected; the yaml then drops one error rate. The
    two points it no longer makes keep their pieces, and status still
    counts them, every saved shot once.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    experiment_dir = tmp_path / "experiment"
    command.main(["collect", str(config_path), "--out", str(experiment_dir)])
    (block,) = FOUR_POINT_SWEEP["sweep"]
    narrower_axes = {**block["axes"], ERROR_RATE_AXIS: [0.001]}
    narrower_card = {"sweep": [{**block, "axes": narrower_axes}]}
    yaml_configs.write_config(tmp_path, narrower_card)

    command.main(["status", str(experiment_dir)])

    rows = _typed_status_rows(experiment_dir)
    assert len(rows) == 4
    assert _total_shots(rows) == 8


def test_status_counts_a_point_two_configurations_recorded_once(tmp_path):
    """A base setting changed between collects makes a second configuration.

    Its sweep sets the changed setting, so its points are the first
    configuration's, already saved: status counts each once, under the
    configuration that recorded it last.
    """
    first_path = _base_distance_config(tmp_path, 5, FOUR_POINT_SWEEP)
    experiment_dir = tmp_path / "experiment"
    command.main(["collect", str(first_path), "--out", str(experiment_dir)])
    qpu = {"kind": "stim_device", "distance": 7}
    changed_card = {**FOUR_POINT_SWEEP, "qpu": qpu}
    first_folder = first_path.parent
    second_path = yaml_configs.write_config(first_folder, changed_card)
    command.main(["collect", str(second_path), "--out", str(experiment_dir)])

    command.main(["status", str(experiment_dir)])

    rows = _typed_status_rows(experiment_dir)
    second_config = experiment.load_experiment(second_path)
    second_id = run_folder.configuration_id(second_config)
    configuration_ids = {row["configuration_id"] for row in rows}
    assert len(rows) == 4
    assert _total_shots(rows) == 8
    assert configuration_ids == {second_id}


# the sweep axis of a memory maker's physical error rate
ERROR_RATE_AXIS = "workload.arguments.physical_error_probability"


def _total_shots(rows: list) -> int:
    total = 0
    for row in rows:
        total += row["shots"]
    return total


def test_a_status_row_is_its_points_sweep_row_and_its_rounds(tmp_path):
    """status.csv carries every column the fold gives a point, and more.

    Each row holds its point's sweep.csv row whole, the per-round and
    plan-unbiased estimates among them, then the rounds and core seconds
    of all its pieces: two shots of fifteen rounds each here.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    experiment_dir = tmp_path / "experiment"
    command.main(["collect", str(config_path), "--out", str(experiment_dir)])

    command.main(["status", str(experiment_dir)])

    status_path = experiment_dir / "status.csv"
    status_rows = _csv_rows(status_path)
    run_dir = yaml_configs.run_folder_of(experiment_dir)
    sweep_path = run_dir / "sweep.csv"
    sweep_rows = _csv_rows(sweep_path)
    typed_rows = _typed_status_rows(experiment_dir)
    rounds = {row["rounds"] for row in typed_rows}
    assert _sweep_cells_status_misses(status_rows, sweep_rows) == []
    assert rounds == {30}


def _sweep_cells_status_misses(status_rows: list, sweep_rows: list) -> list:
    """Each (point id, column) of sweep.csv its status row lacks or changes."""
    status_by_point = {row["point_id"]: row for row in status_rows}
    missed = []
    for sweep_row in sweep_rows:
        point_id = sweep_row["point_id"]
        status_row = status_by_point[point_id]
        point_missed = _cells_missed(point_id, status_row, sweep_row)
        missed.extend(point_missed)
    return missed


def _cells_missed(point_id: str, status_row: dict, sweep_row: dict) -> list:
    missed = []
    for column, cell in sweep_row.items():
        if status_row.get(column) != cell:
            missed.append((point_id, column))
    return missed


def test_a_rounds_last_piece_is_cut_to_what_the_time_cap_allows(tmp_path):
    """The time cap bounds a round's shots below one whole piece.

    One piece of six shots is saved; the cap is then set to its core
    seconds and one hundredth more, which at its seconds a shot allows
    seven shots in all. The next round plans one shot, not a whole
    piece past the cap.
    """
    collection = {"max_shots": 1000, "max_failures": 1000, "piece_rounds": 90}
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"
    first_round = _plan([config_path], out_dir, 1)
    _run_the_round(first_round)
    (first_piece,) = out_dir.glob("pieces/*/*")
    saved = pieces.read_piece(first_piece)
    capped = {**collection, "max_core_seconds": saved["core_seconds"] * 1.01}
    capped_card = {"sweep": [{"axes": NOISY_AXES, "collection": capped}]}
    yaml_configs.write_config(tmp_path, capped_card)

    second_round = _plan([config_path], out_dir, 1)

    point_id = first_piece.parent.name
    assert saved["count"] == 6
    assert _planned_ranges(second_round) == [(point_id, 6, 1)]


@pytest.mark.parametrize(
    "flag, shape_arguments",
    [
        ("--tasks", ["--tasks", "0"]),
        ("--cores", ["--tasks", "1", "--cores", "0"]),
        ("--hours", ["--tasks", "1", "--hours", "-1"]),
        ("--memory-mb", ["--tasks", "1", "--memory-mb", "-1"]),
    ],
)
def test_a_plan_asking_for_no_task_core_hour_or_memory_is_refused(
    tmp_path, capsys, flag, shape_arguments
):
    """A round has a task, and a task a core, an hour and some memory."""
    config_path = yaml_configs.write_config(tmp_path, {})
    experiment_dir = tmp_path / "experiment"
    arguments = [str(config_path), "--out", str(experiment_dir)]

    with pytest.raises(SystemExit):
        command.main(["plan", *arguments, *shape_arguments])

    printed = capsys.readouterr()
    assert f"{flag} must be at least 1" in printed.err
    assert not experiment_dir.exists()


@pytest.mark.parametrize(
    "flag, task_count, shape",
    [
        ("--tasks", 0, (1, 1, 1)),
        ("--cores", 1, (0, 1, 1)),
        ("--hours", 1, (1, -1, 1)),
        ("--memory-mb", 1, (1, 1, -1)),
    ],
)
def test_plan_round_refuses_no_task_core_hour_or_memory_before_writing(
    tmp_path, flag, task_count, shape
):
    """plan_round itself refuses a shape below one, whoever calls it.

    shape is the task's cores, hours and memory in MB.
    """
    job = plan_command.JobShape(*shape)
    config_path = yaml_configs.write_config(tmp_path, {})
    experiment_dir = tmp_path / "experiment"

    with pytest.raises(refusal.RefusalError) as refused:
        plan_command.plan_round([config_path], experiment_dir, task_count, job)

    assert f"{flag} must be at least 1" in str(refused.value)
    assert not experiment_dir.exists()


def test_an_online_point_is_one_tasks_in_every_round_with_the_same_seeds(
    tmp_path,
):
    """Two rounds that may run at once hold one online point's same pieces.

    Round two is planned before round one runs, as when round one's
    array is still queued: it deals the point's four pieces again to one
    task, in seed order, cut where round one cut them. Run in either
    order, the rounds save each piece once and the fold is the uncut
    run's.
    """
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    whole_config = _online_config(tmp_path, 60)
    command.main(["collect", str(whole_config), "--out", str(whole_dir)])
    cut_config = _online_config(tmp_path, 15)
    first_round = _plan([cut_config], cut_dir, 3)
    second_round = _plan([cut_config], cut_dir, 3)

    _run_the_round(second_round)
    _run_the_round(first_round)
    command.main(["status", str(cut_dir)])

    first_plan_path = first_round / "plan.csv"
    second_plan_path = second_round / "plan.csv"
    first_plan = _csv_rows(first_plan_path)
    second_plan = _csv_rows(second_plan_path)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    assert _values_of_rows(second_plan, ("task", "first_seed", "count")) == [
        ("0", "0", "1"),
        ("0", "1", "1"),
        ("0", "2", "1"),
        ("0", "3", "1"),
    ]
    assert second_plan == first_plan
    assert _piece_names(cut_dir) == ["0-0", "1-1", "2-2", "3-3"]
    assert _shot_decisions(cut_run_dir) == _shot_decisions(whole_run_dir)


def test_a_point_that_changed_configuration_is_in_one_run_folder(tmp_path):
    """Status rebuilds every configuration's run folder, emptied ones too.

    A base setting changed between collects moves every point to the
    second configuration. After status, the first configuration's run
    folder holds none of them, so reading every run folder counts each
    saved shot once; the pieces themselves are untouched.
    """
    first_path = _base_distance_config(tmp_path, 5, FOUR_POINT_SWEEP)
    experiment_dir = tmp_path / "experiment"
    command.main(["collect", str(first_path), "--out", str(experiment_dir)])
    qpu = {"kind": "stim_device", "distance": 7}
    changed_card = {**FOUR_POINT_SWEEP, "qpu": qpu}
    first_folder = first_path.parent
    second_path = yaml_configs.write_config(first_folder, changed_card)
    command.main(["collect", str(second_path), "--out", str(experiment_dir)])
    pieces_before = _piece_names(experiment_dir)

    command.main(["status", str(experiment_dir)])

    sweep_rows = _rows_of_every_run_folder(experiment_dir)
    assert _total_shots(sweep_rows) == 8
    assert len(sweep_rows) == 4
    assert _piece_names(experiment_dir) == pieces_before


def _rows_of_every_run_folder(experiment_dir: pathlib.Path) -> list:
    """Every combined/*/sweep.csv's rows, typed."""
    rows = []
    for sweep_path in experiment_dir.glob("combined/*/sweep.csv"):
        folder_rows = report.read_rows(sweep_path)
        rows.extend(folder_rows)
    return rows


def test_a_plain_collect_and_an_older_planned_task_run_each_seed_once(
    tmp_path,
):
    """A plain collect cuts its pieces where a planned piece begins or ends.

    Round one plans seeds 0 to 5 and seed 6; only the first task runs.
    The cap is raised to ten and a plain collect runs on: it saves seed
    6 as the plan cut it, then seeds 7 to 9. The planned task run later
    finds seed 6 saved and runs nothing, so every seed is saved once.
    """
    config_path = _capped_noisy_config(tmp_path, 7, 90)
    out_dir = tmp_path / "out"
    round_dir = _plan([config_path], out_dir, 2)
    plan_path = round_dir / "plan.csv"
    command.main(["collect", "--plan", str(plan_path), "--task", "0"])
    _capped_noisy_config(tmp_path, 10, 90)
    command.main(["collect", str(config_path), "--out", str(out_dir)])

    command.main(["collect", "--plan", str(plan_path), "--task", "1"])
    command.main(["status", str(out_dir)])

    (row,) = _typed_status_rows(out_dir)
    (_first_piece, second_piece) = _planned_ranges(round_dir)
    assert second_piece[1:] == (6, 1)
    assert _piece_names(out_dir) == ["0-5", "6-6", "7-9"]
    assert row["shots"] == 10


def test_a_planned_piece_saved_under_another_cut_is_not_run_again(tmp_path):
    """A planned task runs only the seeds no saved piece holds.

    Round one plans seeds 0 to 5 and seed 6. Before it runs, a plain
    collect with pieces of two shots saves seeds 0 to 6 in four pieces.
    The round's tasks then find their seeds saved and run nothing.
    """
    config_path = _capped_noisy_config(tmp_path, 7, 90)
    out_dir = tmp_path / "out"
    round_dir = _plan([config_path], out_dir, 2)
    _capped_noisy_config(tmp_path, 7, 30)
    command.main(["collect", str(config_path), "--out", str(out_dir)])

    _run_the_round(round_dir)
    command.main(["status", str(out_dir)])

    (row,) = _typed_status_rows(out_dir)
    assert _piece_names(out_dir) == ["0-1", "2-3", "4-5", "6-6"]
    assert row["shots"] == 7
    assert _plan([config_path], out_dir, 2) is None


def _capped_noisy_config(tmp_path, max_shots: int, piece_rounds: int):
    """The noisy point with a shot cap and no target, written in place."""
    collection = {"max_shots": max_shots, "piece_rounds": piece_rounds}
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    return yaml_configs.write_config(tmp_path, card)


def test_a_status_row_counts_and_costs_one_reading_of_the_pieces(
    tmp_path, monkeypatch
):
    """A round that ends while status runs is in all of a row or in none.

    Round one's piece of six shots is saved and round two's is planned;
    round two ends right after status folds. The row's rounds are its
    shots' rounds, fifteen a shot, not the rounds of pieces the fold
    never read.
    """
    collection = {"max_shots": 60, "max_failures": 60, "piece_rounds": 90}
    card = {"sweep": [{"axes": NOISY_AXES, "collection": collection}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    out_dir = tmp_path / "out"
    first_round = _plan([config_path], out_dir, 1)
    _run_the_round(first_round)
    second_round = _plan([config_path], out_dir, 1)
    fold = collect_command.write_the_run_folder
    fold_then_run = functools.partial(_fold_then_run, fold, second_round)
    monkeypatch.setattr(collect_command, "write_the_run_folder", fold_then_run)

    command.main(["status", str(out_dir)])

    (row,) = _typed_status_rows(out_dir)
    assert row["shots"] == 6
    assert row["rounds"] == 6 * 15


def _fold_then_run(fold, round_dir: pathlib.Path, *arguments) -> list:
    """The fold, then a round's tasks, as a round that ends mid-status."""
    rows = fold(*arguments)
    _run_the_round(round_dir)
    return rows


def test_an_online_piece_with_no_saved_piece_before_it_is_refused(
    tmp_path, capsys
):
    """An online piece starts from the calibrator the piece before saved.

    A plan edited to hold only the point's third piece, seed 2, runs
    nothing: no piece ends at seed 2, so there is no calibrator to start
    from, and the collect says so and asks for a new plan.
    """
    config_path = _online_config(tmp_path, 15)
    out_dir = tmp_path / "out"
    round_dir = _plan([config_path], out_dir, 1)
    plan_path = round_dir / "plan.csv"
    plan_rows = _csv_rows(plan_path)
    third_rows = [row for row in plan_rows if row["first_seed"] == "2"]
    _write_csv_rows(plan_path, third_rows)
    point_id = third_rows[0]["point_id"]

    with pytest.raises(SystemExit):
        command.main(["collect", "--plan", str(plan_path), "--task", "0"])

    printed = capsys.readouterr()
    point_folders = pieces.folders_of(out_dir, [point_id])
    assert "has no saved piece ending at seed 2" in printed.err
    assert point_folders == []


def _write_csv_rows(path: pathlib.Path, rows: list) -> None:
    """Rows written over a csv file, their keys the header."""
    first_row = rows[0]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first_row))
        writer.writeheader()
        writer.writerows(rows)


def test_a_refused_status_leaves_the_last_run_folder_as_it_was(tmp_path):
    """A fold that is refused publishes nothing, so the last one stands.

    Ten shots in two pieces are folded by a status. The second piece
    then loses a column of its shots.csv, as a piece measured by other
    code would, and a second status is refused. Every file of the run
    folder and status.csv are byte for byte what the first status wrote.
    """
    config_path = _capped_noisy_config(tmp_path, 10, 75)
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    command.main(["status", str(out_dir)])
    before = _run_folder_bytes(out_dir)
    piece_folders = out_dir.glob("pieces/*/*")
    second_piece = max(piece_folders, key=lambda folder: folder.name)
    shots_path = second_piece / "shots.csv"
    shot_rows = _csv_rows(shots_path)
    for row in shot_rows:
        del row["queue_wait_mean_us"]
    _write_csv_rows(shots_path, shot_rows)

    with pytest.raises(SystemExit):
        command.main(["status", str(out_dir)])

    assert _run_folder_bytes(out_dir) == before


def _run_folder_bytes(experiment_dir: pathlib.Path) -> dict:
    """Each file of combined/ and status.csv, by its path, and its bytes."""
    paths = [
        path for path in experiment_dir.glob("combined/**/*") if path.is_file()
    ]
    status_path = experiment_dir / "status.csv"
    paths.append(status_path)
    return {str(path): path.read_bytes() for path in paths}


def test_rounds_planned_again_before_the_last_ran_run_each_seed_once(
    tmp_path, monkeypatch
):
    """The seeds planned by every round are one set, less the saved ones.

    Round one plans seeds 0 to 9 in one piece; a plain collect capped at
    five shots saves 0 to 4. Round two plans the rest, 5 to 9, and round
    three is planned before round two runs. Round three lists seeds 5 to
    9 once, and its task runs each of them once.
    """
    out_dir = tmp_path / "out"
    config_path = _capped_noisy_config(tmp_path, 10, 150)
    _plan([config_path], out_dir, 1)
    _capped_noisy_config(tmp_path, 5, 150)
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    _capped_noisy_config(tmp_path, 10, 150)
    _plan([config_path], out_dir, 1)
    third_round = _plan([config_path], out_dir, 1)
    ran_seeds = []
    run_unit = functools.partial(
        _run_the_unit_and_note, ran_seeds, collect.run_unit
    )
    monkeypatch.setattr(collect, "run_unit", run_unit)

    _run_the_round(third_round)

    ((_point_id, first_seed, count),) = _planned_ranges(third_round)
    assert (first_seed, count) == (5, 5)
    assert ran_seeds == [5, 6, 7, 8, 9]


def _run_the_unit_and_note(ran_seeds: list, run_unit, unit, measure):
    """run_unit, collect's own, noting the seeds it runs."""
    end_seed = unit.first_seed + unit.seeds
    ran_seeds.extend(range(unit.first_seed, end_seed))
    return run_unit(unit, measure)


def test_two_yaml_names_of_one_configuration_fold_into_one_run_folder(
    tmp_path,
):
    """A configuration's run folder is named by the first yaml recorded.

    t2.yaml and t6.yaml say the same thing, so they are one
    configuration id. Collected in that order, then folded by status,
    they write one run folder, t2's, holding the point once.
    """
    config_path = _capped_noisy_config(tmp_path, 1, 15)
    out_dir = tmp_path / "out"
    first_path = tmp_path / "t2.yaml"
    second_path = tmp_path / "t6.yaml"
    shutil.copyfile(config_path, first_path)
    shutil.copyfile(config_path, second_path)
    command.main(["collect", str(first_path), "--out", str(out_dir)])
    command.main(["collect", str(second_path), "--out", str(out_dir)])

    command.main(["status", str(out_dir)])

    run_folder_dir = yaml_configs.run_folder_of(out_dir)
    swept_rows = _rows_of_every_run_folder(out_dir)
    assert run_folder_dir.name.startswith("t2-")
    assert [row["shots"] for row in swept_rows] == [1]


def test_status_retires_the_fold_of_a_second_folder_of_one_configuration(
    tmp_path,
):
    """A folder an older collect named by another yaml loses its fold.

    The run folder is copied to a second name with the same id, as a
    collect of a second yaml name wrote it before one folder per id.
    Status leaves the point in one sweep.csv, and the second folder keeps
    what is not a fold's, its manifest.
    """
    config_path = _capped_noisy_config(tmp_path, 1, 15)
    out_dir = tmp_path / "out"
    command.main(["collect", str(config_path), "--out", str(out_dir)])
    run_folder_dir = yaml_configs.run_folder_of(out_dir)
    _name, identity8 = run_folder_dir.name.rsplit("-", 1)
    second_dir = run_folder_dir.with_name(f"t6-{identity8}")
    shutil.copytree(run_folder_dir, second_dir)

    command.main(["status", str(out_dir)])

    swept_rows = _rows_of_every_run_folder(out_dir)
    second_files = sorted(path.name for path in second_dir.iterdir())
    assert [row["shots"] for row in swept_rows] == [1]
    assert "manifest.json" in second_files
    assert not {"sweep.csv", "resolved", "inputs"} & set(second_files)
