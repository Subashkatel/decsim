"""Small Python experiments for the run-side tests, and their run files.

minimal_machine is a complete machine small enough for a functional
test: the weak decoder baseline on a 15-round memory shot, its readout
one fridge cycle, its weak decoder charged a 28 ns decode on an engine
that releases in one cycle. switching_machine adds a PyMatching strong
tier behind a complementary-gap threshold.

sweep builds an experiment from literal arguments, one point per
combination of its axes, so a test that runs `decsim run` writes a run
file of a few lines (write_run_file) that calls it, and the results
folder keeps that copy. An axis is named by the path its value sets,
as the shipped run files name their metadata, so a point's columns are
the shipped ones. The shot helpers run, measure and fold a point's
shots in the test's own process, and decode a shot's whole circuit as
the reference its windowed loop is checked against.
"""

import dataclasses
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Optional

import numpy
import pymatching

import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.collection as collection_module
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure
import decsim.experiments.pieces as pieces
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.producers as producers
import decsim.settings as machine_settings
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)
from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)
from decsim.decoders.union_find import cycle_count
from decsim.decoders.union_find import decoder as union_find

NAME = "unit_test_config"
DISTANCE_PATH = "qpu.distance"
ERROR_RATE_PATH = "workload.arguments.physical_error_probability"
ROUND_PERIOD_PATH = "qpu.round_period_microseconds"
COMMIT_ROUNDS_PATH = "windows.commit_rounds"
CODE_TASK_PATH = "workload.arguments.code_task"
# the minimal machine's one point
ONE_POINT_AXES = {
    DISTANCE_PATH: (3,),
    ROUND_PERIOD_PATH: (1.0,),
    ERROR_RATE_PATH: (0.001,),
}
FOUR_POINT_AXES = {
    DISTANCE_PATH: (3, 5),
    ROUND_PERIOD_PATH: (1.0,),
    ERROR_RATE_PATH: (0.001, 0.003),
}
DEFAULT_COLLECTION = {"max_shots": 1}
# the QPU's own round period, where a sweep sets none (QpuSettings)
UNSWEPT_ROUND_PERIOD = 1.1
ROUNDS_PER_SHOT = 15
CODE_TASK = "surface_code:rotated_memory_z"
# LILLIPUT's 28 ns, the weak base's charged decode
WEAK_DECODE_MICROSECONDS = 0.028
READOUT_SOURCE = "a run-side test's readout: one fridge cycle"
# The decoders a test names in a run file, each run for real.
PYMATCHING = minimum_weight_perfect_matching.PyMatchingDecoder
RELAY_BP = relay_belief_propagation.RelayBeliefPropagationDecoder
UNION_FIND = union_find.UnionFindDecoder
HOST_TIME = cycle_count.HostMeasuredTime()
ALGORITHMS = {
    "pymatching": PYMATCHING.Settings(),
    "relay_bp": RELAY_BP.Settings(),
    "union_find": UNION_FIND.Settings(timing=HOST_TIME),
}
# reference_machine's hops: (path, latency cycles, bits a cycle a lane,
# lanes), each on the fridge clock
REFERENCE_HOPS = (
    ("qpu_to_controller", 1, None, 1),
    ("controller_to_weak_buffer", 1, 1.0, 8),
    ("weak_buffer_to_weak_decoder", 1, None, 1),
    ("decoder_to_decoder", 1, None, 1),
    ("weak_decoder_to_frame", 1, None, 1),
    ("frame_to_controller", 0, 32, 1),
    ("controller_to_qpu", 22, 128, 1),
)
REFERENCE_SOURCE = "the reference machine's hops, in fridge cycles"
# the control processor's issue pipeline, the reference machine's count
REFERENCE_ISSUE_CYCLES = 8
# the reference sweep: one point, its axes in written order
REFERENCE = {
    "axes": {
        ERROR_RATE_PATH: (0.001,),
        DISTANCE_PATH: (3,),
        ROUND_PERIOD_PATH: (1.0,),
    },
    "machine": "reference",
    "collection": {"max_shots": 2},
    "name": "reference",
}
# Same-region gap below 20 dB redoes the window on the strong tier.
FIXED_THRESHOLD_DECIBELS = 20.0
# Where an online threshold starts before it learns.
ONLINE_THRESHOLD_DECIBELS = 15.0


def minimal_machine(
    cells: Optional[Mapping] = None,
    *,
    weak_decoder: Optional[str] = None,
    rounds_per_shot: int = ROUNDS_PER_SHOT,
) -> machine_settings.MachineSettings:
    """The minimal machine at one point's cells.

    cells maps an axis path to the point's value there; a path it does
    not name takes distance 3, physical error probability 0.001, the
    QPU's own 1.1 us round, the rotated Z memory and the scheme's own
    commit. weak_decoder names one of ALGORITHMS run for real, in place
    of the charged decode.
    """
    cells = cells or {}
    distance = cells.get(DISTANCE_PATH, 3)
    probability = cells.get(ERROR_RATE_PATH, 0.001)
    round_period = cells.get(ROUND_PERIOD_PATH, UNSWEPT_ROUND_PERIOD)
    base = machine_settings.weak_decoder_baseline(
        distance, probability, round_period
    )
    reference = link_profiles.logical_reference_profile()
    readout = link_profiles.path_card(
        reference,
        "qpu_to_controller",
        clock=machine_settings.FRIDGE_CLOCK,
        latency_cycles=1,
        bits_per_cycle=None,
        source=READOUT_SOURCE,
    )
    links = dataclasses.replace(reference, qpu_to_controller=readout)
    charged = PYMATCHING.Settings(
        preset_latency_microseconds=WEAK_DECODE_MICROSECONDS
    )
    algorithm = charged
    if weak_decoder is not None:
        algorithm = ALGORITHMS[weak_decoder]
    pool = decoder_pool(algorithm, machine_settings.FRIDGE_CLOCK)
    code_task = cells.get(CODE_TASK_PATH, CODE_TASK)
    made = producers.memory_circuit(
        code_task, rounds_per_shot, distance, probability
    )
    workload = workload_settings.WorkloadSettings.running(made)
    windows = base.windows
    commit_rounds = cells.get(COMMIT_ROUNDS_PATH)
    if commit_rounds is not None:
        windows = _windows_committing(windows, commit_rounds)
    return dataclasses.replace(
        base,
        links=links,
        weak_decoder=pool,
        workload=workload,
        windows=windows,
    )


def switching_machine(
    cells: Optional[Mapping] = None,
    *,
    online: Optional[Mapping] = None,
    rounds_per_shot: int = ROUNDS_PER_SHOT,
) -> machine_settings.MachineSettings:
    """The minimal machine switching from PyMatching to PyMatching.

    A window whose complementary gap is below the threshold is redone on
    the strong tier, on the room clock. online, when given, makes the
    threshold learn from 15 dB with these OnlineThreshold settings over
    its defaults; else it is a fixed 20 dB.
    """
    base = minimal_machine(
        cells, weak_decoder="pymatching", rounds_per_shot=rounds_per_shot
    )
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=FIXED_THRESHOLD_DECIBELS
    )
    if online is not None:
        threshold = threshold_sources.OnlineThreshold.Settings(
            threshold_decibels=ONLINE_THRESHOLD_DECIBELS, **online
        )
    confidence = complementary.ComplementaryGap.Settings()
    strong_window = strong_window_shapes.RedoWindow.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence,
        threshold=threshold,
        strong_window=strong_window,
    )
    windows = window_settings.switching_windows(
        base.windows, switching.strong_window
    )
    strong_decoder = decoder_pool(
        ALGORITHMS["pymatching"], machine_settings.ROOM_CLOCK
    )
    return dataclasses.replace(
        base,
        windows=windows,
        strong_decoder=strong_decoder,
        switching=switching,
    )


def reference_machine(
    cells: Optional[Mapping] = None,
) -> machine_settings.MachineSettings:
    """The reference machine, the minimal one on priced hops.

    PyMatching decodes for real; every hop is priced in fridge cycles
    (REFERENCE_HOPS); the controller issues in eight cycles; and the
    weak store, the windows and the decoder manager count on the fridge
    clock. Its first shot writes a Chrome trace.
    """
    base = minimal_machine(cells, weak_decoder="pymatching")
    fridge = machine_settings.FRIDGE_CLOCK
    cards = {}
    for path_name, latency_cycles, bits_per_cycle, lane_count in REFERENCE_HOPS:
        cards[path_name] = link_profiles.path_card(
            base.links,
            path_name,
            clock=fridge,
            latency_cycles=latency_cycles,
            bits_per_cycle=bits_per_cycle,
            source=REFERENCE_SOURCE,
            lane_count=lane_count,
        )
    links = dataclasses.replace(base.links, **cards)
    controller = dataclasses.replace(
        base.controller, decision_to_pulse_cycles=REFERENCE_ISSUE_CYCLES
    )
    weak_store = dataclasses.replace(base.weak_syndrome_buffer, clock=fridge)
    windows = dataclasses.replace(base.windows, clock=fridge)
    manager = dataclasses.replace(base.decoder_manager, clock=fridge)
    observation = dataclasses.replace(base.observation, trace="chrome")
    return dataclasses.replace(
        base,
        links=links,
        controller=controller,
        weak_syndrome_buffer=weak_store,
        windows=windows,
        decoder_manager=manager,
        observation=observation,
    )


MACHINES = {
    "minimal": minimal_machine,
    "switching": switching_machine,
    "reference": reference_machine,
}


def decoder_pool(algorithm, clock) -> decoder_settings.DecoderPoolSettings:
    """One unit of the algorithm on an engine of the clock, memory unbounded.

    The engine fetches a round a cycle and releases a job in one cycle.
    """
    engine = decoder_settings.EngineSettings(clock=clock)
    unit_memory = decoder_settings.UnitMemorySettings(bits=None, word_bits=None)
    return decoder_settings.DecoderPoolSettings(
        algorithm=algorithm,
        unit_count=1,
        engine=engine,
        unit_memory=unit_memory,
        copies_input=True,
        copies_boundary_fold=True,
        result_blocks_unit=False,
    )


def sweep(
    axes: Optional[Mapping] = None,
    *,
    collection: Optional[Mapping] = None,
    machine: str = "minimal",
    machine_arguments: Optional[Mapping] = None,
    observation: Optional[Mapping] = None,
    record_options: Optional[Mapping] = None,
    name: str = NAME,
) -> experiment.Experiment:
    """One point per combination of the axes, each collected alike.

    The axes run in the order given, the last fastest.
    machine names a MACHINES builder and machine_arguments its keywords;
    observation replaces fields of every point's observation, and
    record_options are collect.RecordOptions fields. A point is named by
    its id's first twelve characters.
    """
    axes = axes or ONE_POINT_AXES
    collection = collection or DEFAULT_COLLECTION
    machine_arguments = machine_arguments or {}
    settings = collection_module.CollectionSettings(**collection)
    record_options = record_options or {}
    options = collect.RecordOptions(**record_options)
    build = MACHINES[machine]
    points = []
    for cells in experiment.grid(**axes):
        settings_at_point = build(cells, **machine_arguments)
        if observation is not None:
            watched = dataclasses.replace(
                settings_at_point.observation, **observation
            )
            settings_at_point = dataclasses.replace(
                settings_at_point, observation=watched
            )
        point = _named_point(settings_at_point, cells, settings, options)
        points.append(point)
    return experiment.Experiment(name, points)


def write_run_file(directory: pathlib.Path, **arguments) -> pathlib.Path:
    """A run file whose experiment is sweep(**arguments).

    The arguments are literals, written as Python, so the file is the
    whole record of what the test ran.
    """
    path = pathlib.Path(directory) / f"{NAME}.py"
    text = (
        '"""A run-side test\'s experiment (tests/experiments/run_files.py)."""'
        "\n\nimport tests.experiments.run_files as run_files\n\n"
        f"experiment = run_files.sweep(**{arguments!r})\n"
    )
    path.write_text(text)
    return path


def first_task(**arguments) -> collect.Task:
    """The task of sweep(**arguments)'s first point."""
    study = sweep(**arguments)
    first_point = study.points[0]
    return experiment.task_of(first_point)


def run_one_shot(seed: int = 0, **arguments) -> collect.Shot:
    """One seeded shot of sweep(**arguments)'s first point, run."""
    task = first_task(**arguments)
    return collect.run_shot(task, seed)


def measured_shot(seed: int = 0, **arguments) -> measure.ShotMeasurement:
    """One seeded shot of sweep(**arguments)'s first point, measured."""
    shot = run_one_shot(seed, **arguments)
    return measure.measure_shot(shot)


def run_sweep(tasks: list, shots: int) -> list:
    """Seeds 0 to shots - 1 of every task, measured and not saved."""
    measurements = []
    for task in tasks:
        unit = collect.Unit(task, 0, shots)
        outcome = collect.run_unit(unit, measure.measure_shot)
        measurements.extend(outcome.rows)
    return measurements


def tasks_of(study: experiment.Experiment) -> list:
    """Every point's task, in the experiment's order."""
    return [experiment.task_of(point) for point in study.points]


def folded_run(tmp_path, measurements: list) -> tuple:
    """The shots folded into a run folder as a collect folds its pieces.

    Each point's shots are one piece, in the order they are given, and
    the point's record names no swept path, so the run folder's
    files hold only what the shots measured. Returns the sweep rows the
    fold computed, before csv turns an empty cell into text, and the
    run folder, whose shot_links.csv and shot_data_movement.csv the
    fold wrote.
    """
    temporary_dir = tempfile.mkdtemp(dir=tmp_path)
    experiment_dir = pathlib.Path(temporary_dir)
    measurements_by_point = {}
    for measurement in measurements:
        at_point = measurements_by_point.setdefault(measurement.point_id, [])
        at_point.append(measurement)
    folders = []
    for point_id, at_point in measurements_by_point.items():
        _write_a_bare_point_record(experiment_dir, point_id)
        first_seed = at_point[0].seed
        folder = pieces.write(
            experiment_dir, point_id, first_seed, at_point, {}
        )
        folders.append(folder)
    point_ids = list(measurements_by_point)
    run_dir = experiment_dir / "run"
    rows = report.fold_pieces(experiment_dir, folders, point_ids, run_dir)
    return rows, run_dir


def whole_circuit_predictions(shot: collect.Shot) -> list:
    """Each operation's observables, PyMatching on its whole circuit.

    The reference a windowed loop is checked against: the decomposed
    detector error model of the circuit the source sampled, decoded in
    one piece on the events it drew, operation by operation in the
    result's order.
    """
    sampled = shot.machine.observation.sampled_shots.shots_by_operation
    predictions = []
    for operation_result in shot.result.operation_results:
        sampled_shot = sampled[operation_result.operation_id]
        model = sampled_shot.circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        events = numpy.asarray(sampled_shot.detection_events, dtype=bool)
        predicted = matching.decode(events)
        prediction = tuple(int(bit) for bit in predicted)
        predictions.append(prediction)
    return predictions


def loop_predictions(shot: collect.Shot) -> list:
    """Each operation's observables as the machine's loop decoded them."""
    return [
        tuple(operation_result.logical_observables)
        for operation_result in shot.result.operation_results
    ]


def _write_a_bare_point_record(experiment_dir, point_id: str) -> None:
    """A record holding only what the fold reads of a point."""
    point_dir = experiment_dir / run_folder.POINTS_FOLDER / point_id
    point_dir.mkdir(parents=True)
    record = {"id": point_id, "name": point_id, "metadata": {}}
    record_path = point_dir / run_folder.RECORD_FILE
    run_folder.write_json(record_path, record)


def _named_point(
    settings: machine_settings.MachineSettings,
    cells: Mapping,
    collection: collection_module.CollectionSettings,
    record_options: collect.RecordOptions,
) -> experiment.Point:
    """The point of these settings and cells, named by its id."""
    metadata = dict(cells)
    task = collect.Task(settings, metadata, record_options=record_options)
    point_id = task.strong_id()
    name = point_id[:12]
    return experiment.Point(
        name, settings, metadata, collection, record_options=record_options
    )


def _windows_committing(
    windows: window_settings.WindowSettings, commit_rounds: int
) -> window_settings.WindowSettings:
    """The sliding windows committing commit_rounds rounds each."""
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=commit_rounds, buffer_rounds=None
    )
    return dataclasses.replace(windows, scheme=scheme)
