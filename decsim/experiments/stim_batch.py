"""Batch sampling: a point's shots drawn in blocks and decoded with no machine.

The yaml's top-level `sampling: stim_batch` selects it
(collect_command.SAMPLINGS). It is decsim's atomic mode beside the
machine's timing mode, the split gem5 keeps between atomic accesses for
fast forwarding and timing accesses that model queuing (gem5 memory
system documentation, "Access Types"). The loop is sinter's
(sinter/_decoding/_decoding.py:265-306): sample a block with
separate_observables, decode it, count the mistakes. Each shot goes
through the point's own weak decoder row, by its Decoder port, on the
one window naive_online lays over the operation, built by the machine's
own build, so a batch point answers every shot as the machine does.
What differs is how the shots are drawn, and that a piece keeps counts
only, as sinter's AnonTaskStats does (sinter/_data/_anon_task_stats.py:
27-31): no latency, no per-shot rows. decsim's offline lane, removed at
408f22c5, decoded window by window one shot at a time and ran within
3x of the machine; this mode samples whole blocks and decodes the
whole circuit.
"""

import hashlib
import pathlib
import time

import numpy
import stim

import decsim.build.decoders as build_decoders
import decsim.collect as collect
import decsim.experiments.collection as collection_module
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.results as result_records
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
import decsim.windows.built_window_models as built_window_models

SAMPLING = "stim_batch"
# Stim repeats a seed's samples only for the same calls, the same
# version, the same SIMD width and the same shot count (Stim API,
# compile_detector_sampler). So every block is drawn whole, BLOCK_SHOTS
# shots in one call, and a seed's shot is the same in every piece cut,
# every process and for every decoder of the point.
BLOCK_SHOTS = 1024
# The machine's seed path to its weak decoder row: assembly.SEED_ROOTS
# names the router decoder_router, the CodeRouter its default row, and
# the staged unit its algorithm decoder. A row bound under this path to
# root seed s draws what the machine's shot s draws.
ROW_SEED_PATH = (
    seed_records.RunSeedPathSegment("field", "decoder_router"),
    seed_records.RunSeedPathSegment("field", "default"),
    seed_records.RunSeedPathSegment("field", "decoder"),
)
REFUSAL_OPENING = (
    "sampling stim_batch decodes one window over the whole circuit on the "
    "weak tier alone, from the circuit's own samples, which is the "
    "machine's decode only under"
)


class WholeCircuitWindow:
    """The one window naive_online lays over a point's operation.

    Machine.build lays the operation's windows and builds their models
    into the task's model cache, as collect.run_unit's first shot does
    (built_window_models), so the model and its row order are the
    machine's own.
    """

    def __init__(self, task: collect.Task) -> None:
        built_models = built_window_models.BuiltWindowModels()
        settings = task.shot_settings(built_models)
        machine_module.Machine.build(settings, 0)
        held = built_models.models_by_key.values()
        (models,) = held
        (model,) = models
        (operation,) = task.settings.workload.operations
        self.operation = operation
        self.model = model
        self.detector_rows = numpy.asarray(model.detector_ids)

    def result_of(
        self, row, detection_events: numpy.ndarray
    ) -> decoding_records.DecodeResult:
        """One shot's result from the row's own Decoder port.

        The job is the one the machine's decoder manager hands the row:
        the window's detection events in its row order, as one fragment,
        the shape tests/decoders/windows.py builds for the referent
        tests. A job's round count prices its timing, which no row's
        decode reads and a batch keeps none of, so it is 1, as there.
        """
        events = detection_events[self.detector_rows]
        event_bits = events.astype(numpy.uint8)
        bit_list = event_bits.tolist()
        bits = tuple(bit_list)
        payload = round_records.RetainedSyndromeFragment(
            operation_id=self.operation.id,
            patch_ids=self.operation.patches,
            round_index=1,
            bits=bits,
            size_bits=len(bits),
            fragment_index=0,
        )
        job = decoding_records.DecodeJob(
            operation_id=self.operation.id,
            window_id=0,
            round_count=1,
            detector_error_model=self.model,
            payloads=[payload],
            label="whole circuit",
        )
        return row.decode(job)


def check_task(task: collect.Task) -> None:
    """A point whose whole-circuit decode is the machine's, or refused.

    A whole-circuit decode is the machine's decode only for one window
    per operation (windows/schemes/naive_online.py) on the weak tier
    alone, and the shots are the circuit's own only from stim_device,
    whose sampled circuit is the operation's (a burst device raises the
    noise, a recorded one replays), and for one operation a shot.
    """
    settings = task.settings
    required_kinds = (
        ("windows.kind", settings.windows.kind, "naive_online"),
        ("escalation.kind", settings.escalation.kind, "weak_baseline"),
        ("qpu.kind", settings.qpu.kind, "stim_device"),
    )
    for key, kind, wanted in required_kinds:
        if kind != wanted:
            raise refusal.RefusalError(
                f"{REFUSAL_OPENING} {key} {wanted}; this point sets {key} "
                f"{kind!r}"
            )
    operations = settings.workload.operations
    if len(operations) != 1:
        raise refusal.RefusalError(
            "sampling stim_batch decodes one operation's circuit a shot, and "
            f"this point's workload has {len(operations)} operations"
        )


def circuit_of(task: collect.Task) -> stim.Circuit:
    """The circuit of the point's one operation, which every shot samples."""
    (operation,) = task.settings.workload.operations
    return operation.circuit


def block_seed(circuit: stim.Circuit, block: int) -> int:
    """The seed of one block of a circuit's shots.

    It is named by the block and the circuit, never by the decoder, so
    every decoder of a point decodes the same samples.
    """
    circuit_text = str(circuit)
    encoded = circuit_text.encode("utf8")
    digest = hashlib.sha256(encoded)
    circuit_sha256 = digest.hexdigest()
    return seeding.substream_seed(block, (SAMPLING, circuit_sha256))


def block_samples(circuit: stim.Circuit, block: int) -> tuple:
    """One block's detection events and observables, BLOCK_SHOTS shots."""
    seed = block_seed(circuit, block)
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots=BLOCK_SHOTS, separate_observables=True)


def samples_of_seeds(
    circuit: stim.Circuit, first_seed: int, count: int
) -> tuple:
    """The detection events and observables of count seeds from first_seed."""
    event_parts = []
    observable_parts = []
    for block, low, high in _block_spans(first_seed, count):
        events, observables = block_samples(circuit, block)
        event_parts.append(events[low:high])
        observable_parts.append(observables[low:high])
    events = numpy.concatenate(event_parts)
    observables = numpy.concatenate(observable_parts)
    return events, observables


def bound_row(task: collect.Task, root_seed: int):
    """The point's weak decoder row, bound to root_seed at the machine's path.

    The row is the machine's own algorithm, built by the machine's own
    builder (build/decoders.py algorithm_of).
    """
    tier = task.settings.weak_decoder
    row = build_decoders.algorithm_of(tier, "weak")
    roots = ((ROW_SEED_PATH, row),)
    seeding.bind_run_seed(root_seed, roots)
    return row


def run_unit(unit: collect.Unit, measure) -> result_records.UnitOutcome:
    """The unit's seeds drawn and decoded block by block; its counts, one row.

    A row that draws from the run seed (the Relay-BP gamma table, a
    Tesseract with no fixed order seed) is built afresh for each block
    and bound to the block's seed, so a shot's answer is the same in
    every piece cut; another row serves the whole unit, compiled once,
    as sinter compiles one decoder per task. core_seconds is this
    process's time over the whole unit, the build included, sinter's
    seconds. measure is the machine's per-shot measure, which a unit
    with no machine shot has no use for.
    """
    del measure
    started_seconds = time.process_time()
    task = unit.task
    circuit = circuit_of(task)
    window = WholeCircuitWindow(task)
    tally = _Tally(circuit.num_observables)
    row = None
    for block, low, high in _block_spans(unit.first_seed, unit.seeds):
        root_seed = block_seed(circuit, block)
        row = _row_for_the_block(task, row, root_seed)
        events, observables = block_samples(circuit, block)
        block_events = events[low:high]
        block_observables = observables[low:high]
        first_seed = block * BLOCK_SHOTS + low
        tally.add_samples(block_events, block_observables)
        _decode_the_block(
            window, row, tally, first_seed, block_events, block_observables
        )
    ended_seconds = time.process_time()
    core_seconds = ended_seconds - started_seconds
    counts = tally.counts(core_seconds)
    peak_memory = collect.peak_memory_mb()
    return result_records.UnitOutcome([counts], task, peak_memory)


def write_piece(
    experiment_dir: pathlib.Path,
    facts: dict,
    rounds_by_point: dict,
    unit: collect.Unit,
    outcome: result_records.UnitOutcome,
) -> pathlib.Path:
    """A unit's counts saved as its piece, piece.json alone.

    The folder is staged and renamed whole, as pieces.write saves a
    machine piece, so a piece folder exists only whole.
    """
    point_id = unit.task.strong_id()
    (counts,) = outcome.rows
    rounds_per_shot = rounds_by_point[point_id]
    identity = run_folder.piece_identity()
    piece = {
        "point_id": point_id,
        "first_seed": unit.first_seed,
        **counts,
        **facts,
        "rounds": rounds_per_shot * unit.seeds,
        "peak_memory_mb": outcome.peak_memory_mb,
        **identity,
    }
    folder = pieces.piece_dir(
        experiment_dir, point_id, unit.first_seed, unit.seeds
    )
    staging = run_folder.staging_path(folder)
    staging.mkdir(parents=True)
    piece_path = staging / pieces.PIECE_FILE
    run_folder.write_json(piece_path, piece)
    pieces.publish(staging, folder)
    return folder


def shot_rows(folder: pathlib.Path):
    """A piece's shots as the rows a prefix tracker reads, in seed order.

    A shot failed or went unscored when piece.json names its seed. The
    piece's seconds ride on its last shot, so a time cap stops a point
    at a piece's end; a target stops it on the exact shot of its last
    wanted failure.
    """
    piece = pieces.read_piece(folder)
    failure_seeds = set(piece["failure_seeds"])
    unscored_seeds = set(piece["unscored_seeds"])
    first_seed = piece["first_seed"]
    end_seed = first_seed + piece["count"]
    last_seed = end_seed - 1
    for seed in range(first_seed, end_seed):
        seconds = 0.0
        if seed == last_seed:
            seconds = piece["core_seconds"]
        yield {
            "point_id": piece["point_id"],
            "seed": seed,
            "is_scored": seed not in unscored_seeds,
            "logical_failure": seed in failure_seeds,
            "sim_wall_seconds": seconds,
        }


def fold_into_the_staging(
    experiment_dir: pathlib.Path,
    folders: list,
    point_ids: list,
    staging: pathlib.Path,
) -> list:
    """The points' pieces folded into sweep.csv, and their records; the rows.

    A row per point with a piece: its counts over every shot, then its
    prefix's state, estimate and exact limits, read by its collection
    as the machine's fold reads them (report.summarize_point), and no
    latency column.
    """
    records = run_folder.resolved_by_point(experiment_dir)
    seeds_by_point = pieces.seed_ranges_of(folders)
    run_folder.copy_points_of(
        experiment_dir, point_ids, seeds_by_point, staging
    )
    swept = run_folder.swept_values(staging, point_ids)
    rows = []
    for point_id in point_ids:
        point_folders = pieces.point_folders(folders, point_id)
        if not point_folders:
            continue
        record = records[point_id]
        row = _point_row(record, point_folders)
        rows.append(row)
    sweep_path = staging / "sweep.csv"
    report.write_csv(rows, sweep_path, swept)
    return rows


class _Tally:
    """A unit's counts, shot by shot in seed order, and a hash of its samples.

    The hash lets a status compare the samples of paired points block by
    block, since Stim does not promise the same samples across SIMD
    widths.
    """

    def __init__(self, observable_count: int) -> None:
        self.shot_count = 0
        self.failure_seeds = []
        self.unscored_seeds = []
        self.failures_by_observable = numpy.zeros(
            observable_count, dtype=numpy.int64
        )
        self.samples_hash = hashlib.sha256()

    def add_samples(self, events, observables) -> None:
        """One block's slice of samples into the hash."""
        event_bytes = events.tobytes()
        observable_bytes = observables.tobytes()
        self.samples_hash.update(event_bytes)
        self.samples_hash.update(observable_bytes)

    def add_shot(
        self, seed: int, result: decoding_records.DecodeResult, observables
    ) -> None:
        """One decoded shot: unscored, failed, or right.

        A shot whose decode committed no correction is unscored, as the
        machine's measure unscores it (measure._unscored_reason); a
        scored shot fails when any observable differs from the truth.
        """
        self.shot_count += 1
        if result.no_correction_reason is not None:
            self.unscored_seeds.append(seed)
            return
        predicted = numpy.asarray(result.logical_observables, dtype=bool)
        missed = predicted != observables
        if not missed.any():
            return
        self.failure_seeds.append(seed)
        self.failures_by_observable += missed

    def counts(self, core_seconds: float) -> dict:
        """The unit's lines of piece.json.

        Every failure seed is kept, not only the first max_failures, so
        a raised target reads its prefix off the saved pieces.
        """
        unscored_shots = len(self.unscored_seeds)
        by_observable = self.failures_by_observable.tolist()
        return {
            "count": self.shot_count,
            "scored_shots": self.shot_count - unscored_shots,
            "failures": len(self.failure_seeds),
            "failures_by_observable": by_observable,
            "unscored_shots": unscored_shots,
            "core_seconds": core_seconds,
            "failure_seeds": self.failure_seeds,
            "unscored_seeds": self.unscored_seeds,
            "sampling": SAMPLING,
            "stim_version": stim.__version__,
            "sample_sha256": self.samples_hash.hexdigest(),
        }


def _block_spans(first_seed: int, count: int) -> list:
    """(block, low, high) of every block the seeds touch, their slice of it."""
    end_seed = first_seed + count
    first_block = first_seed // BLOCK_SHOTS
    last_block = (end_seed - 1) // BLOCK_SHOTS
    spans = []
    past_the_last_block = last_block + 1
    for block in range(first_block, past_the_last_block):
        block_start = block * BLOCK_SHOTS
        low_offset = first_seed - block_start
        high_offset = end_seed - block_start
        low = max(low_offset, 0)
        high = min(high_offset, BLOCK_SHOTS)
        spans.append((block, low, high))
    return spans


def _row_for_the_block(task: collect.Task, row, root_seed: int):
    """The unit's row for a block: kept, unless it draws from the run seed."""
    if row is not None and not _reads_the_run_seed(row):
        return row
    return bound_row(task, root_seed)


def _reads_the_run_seed(component) -> bool:
    """Whether a seed owner sits under the component.

    A child that names no children of its own is a seed owner; a latency
    model counted so only costs a rebuild.
    """
    for child in component.run_seed_children():
        owner = child.child
        if owner is None:
            continue
        if not isinstance(owner, seeding.RunSeedComposite):
            return True
        if _reads_the_run_seed(owner):
            return True
    return False


def _decode_the_block(
    window: WholeCircuitWindow,
    row,
    tally: _Tally,
    first_seed: int,
    block_events: numpy.ndarray,
    block_observables: numpy.ndarray,
) -> None:
    """Each shot of a block's slice decoded and counted, in seed order."""
    for offset, shot_events in enumerate(block_events):
        result = window.result_of(row, shot_events)
        seed = first_seed + offset
        tally.add_shot(seed, result, block_observables[offset])


def _point_row(record: dict, point_folders: list) -> dict:
    """One point's sweep row from its pieces' counts and its prefix."""
    tracker = _prefix_of(record, point_folders)
    totals = _summed_counts(point_folders)
    row = {
        "point_id": record["id"],
        "algorithm": record["experiment"]["algorithm"],
        "sampling": SAMPLING,
        "shots": totals["count"],
        "logical_failures": totals["failures"],
        "scored_shots": totals["scored_shots"],
        "unscored_shots": totals["unscored_shots"],
    }
    _add_the_prefix_columns(row, tracker)
    row["core_seconds_per_shot"] = totals["core_seconds"] / totals["count"]
    return row


def _prefix_of(
    record: dict, point_folders: list
) -> collection_module.PrefixTracker:
    """The point's contiguous prefix, read by the collection it recorded."""
    facts = record["experiment"]
    settings = collection_module.CollectionSettings(**facts["collection"])
    rule = collection_module.PointRule(
        settings, facts["adaptive"], record["rounds_per_shot"]
    )
    tracker = collection_module.PrefixTracker(rule)
    for folder in point_folders:
        for row in shot_rows(folder):
            tracker.add(row)
    return tracker


def _summed_counts(point_folders: list) -> dict:
    """A point's piece.json counts summed over its pieces."""
    names = ("count", "failures", "scored_shots", "unscored_shots")
    totals = dict.fromkeys(names, 0)
    totals["core_seconds"] = 0.0
    for folder in point_folders:
        piece = pieces.read_piece(folder)
        for name in totals:
            totals[name] += piece[name]
    return totals


def _add_the_prefix_columns(
    row: dict, tracker: collection_module.PrefixTracker
) -> None:
    """The prefix's state, counts, estimate and exact limits, as a fold's."""
    counts = tracker.counts
    stop_kind = tracker.stop_kind_for_limits()
    estimate = failure_statistics.estimate(
        counts.failures, counts.scored_shots, stop_kind
    )
    rounds = tracker.rule.rounds_per_shot
    row["state"] = tracker.state()
    row["logical_error_rate_estimate"] = estimate.rate
    row["logical_error_rate_low"] = estimate.low
    row["logical_error_rate_high"] = estimate.high
    row["logical_error_rate_per_round"] = _per_round(estimate.rate, rounds)
    row["logical_error_rate_per_round_low"] = _per_round(estimate.low, rounds)
    row["logical_error_rate_per_round_high"] = _per_round(estimate.high, rounds)
    row["prefix_shots"] = counts.shots
    row["prefix_scored_shots"] = counts.scored_shots
    row["prefix_failures"] = counts.failures


def _per_round(shot_rate, rounds: int):
    """A shot's rate as a round's, when there is a rate."""
    if shot_rate is None:
        return None
    return failure_statistics.per_round_rate(shot_rate, rounds)
