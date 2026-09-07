"""The offline LER lane: the closed-loop decode without the event engine.

LER depends on circuit, noise and corrections, never on timing, so this
lane runs the exact windowed decode the simulator performs (same window
models, same boundary XOR between windows, same owned-fault ledger, same
per-seed sampling) as a plain loop. Window models build once per sweep
point instead of once per shot, which is where the closed-loop wall time
goes (2.6 of 3.0 s at d=7).

Three verbs, one slurm-array pipeline:

    python -m experiments.offline_run plan  configs/<name>.yaml \
        [seeds_per_shard]
    python -m experiments.offline_run shard <run_dir> <shard_number>
    python -m experiments.offline_run merge <run_dir>

`plan` creates the timestamped run dir (manifest, config copies) and
shards.tsv, one line per (distance, p, seed range); each array task runs
one line; `merge` folds shards/ into shots.csv, ler.csv and ler.png.

Equivalence to the closed-loop runner is enforced two ways: the engine's
own detector-row layout check runs on every job here too, and
tests/test_offline_run.py pins shot-for-shot identical predictions and
truth against measure_shot. The lane covers the shipped modes
(weak_baseline, strong_only) on sliding windows; a switching mode can
replace committed predictions and would not be equivalent.
"""

import csv
import dataclasses
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

import numpy

import decsim.decoders.decoder_memory as decoder_memory
import decsim.frontends.settings as frontend_settings
import decsim.machine as machine_module
import decsim.message as message
import decsim.qpu.stim_device as stim_device
import decsim.seeding as seeding
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.window_interactions as window_interactions
import decsim.windows.windowing_schemes as windowing_schemes
import experiments.experiment_config as experiment_config
import experiments.plots as plots
import experiments.run as run_module
import experiments.sweep_report as sweep_report

OPERATION_ID = 1
USAGE = (
    "usage: python -m experiments.offline_run "
    "plan <config.yaml> [seeds_per_shard] "
    "| shard <run_dir> <n> | merge <run_dir>"
)
# papers report LER per d rounds (Toshio 2510.25222 Fig. 7); a 10d-round
# shot is ten d-round chunks, the production figure's convention
D_ROUND_CHUNKS_PER_SHOT = 0.1


@dataclasses.dataclass(frozen=True)
class OfflineShot:
    """One seed decoded offline: its windows, its verdict, its wall time."""

    seed: int
    windows: int
    logical_failure: bool
    wall_seconds: float


class SweepPointDecoder:
    """One sweep point's immutable decode state, built once per point.

    Reused for every seed: the circuit, the window chain, the window
    error models and a long-lived algorithm decoder (PyMatching caches
    its matching graph per window model; belief matching caches its BP
    decoder).
    """

    def __init__(
        self,
        config,
        *,
        physical_error_probability: float,
        distance: int,
    ):
        workload = config.settings.workload
        self.rounds = workload.rounds_per_shot.rounds_for(distance)
        self.circuit = frontend_settings.memory_circuit(
            workload.code_task,
            self.rounds,
            distance,
            physical_error_probability,
        )
        self.operation = message.Operation(
            id=OPERATION_ID,
            name="memory",
            qubits=(0,),
            patches=(0,),
            circuit=self.circuit,
        )
        self.windows = self._planned_windows(config, distance)
        tier = config.settings.escalation.decodes_on
        unit = machine_module.build_decoder_unit(config.settings, tier)
        self.algorithm = unit.decoder
        self.models = self._window_models()
        self.interaction = window_interactions.DefaultWindowInteraction()
        self.detectors_by_round = self._detectors_by_round()

    def decode_shot(self, seed: int) -> OfflineShot:
        """One shot, bit-identical to the closed-loop runner at that seed.

        Same device seed derivation, same sampled events and truth, same
        window decode chain.
        """
        wall_start = time.perf_counter()
        events, truth = self._sampled_shot(seed)
        ledger = committed_rounds.LogicalLedger()
        residual = None
        rounds = zip(self.windows, self.models, self.detectors_by_round)
        for window, model, by_round in rounds:
            job = self._window_job(window, model, by_round, events, residual)
            result = self.algorithm.decode(job)
            residual = self.interaction.boundary_from_result(result, None)
            self._install_contribution(ledger, window, result)
        predicted = self._committed_observables(ledger)
        wall_end = time.perf_counter()
        is_logical_failure = tuple(predicted) != truth
        wall_seconds = wall_end - wall_start
        return OfflineShot(
            seed=seed,
            windows=len(self.windows),
            logical_failure=is_logical_failure,
            wall_seconds=wall_seconds,
        )

    def _window_models(self) -> tuple:
        """Each window's detector error model, from the stim device."""
        device = stim_device.StimDevice()
        requirement = self.algorithm.fault_model_requirement
        return device.window_models_for_operation(
            self.operation,
            self.windows,
            self.rounds,
            fault_model_requirement=requirement,
            fault_exclusion_ranges=(),
            window_protocol=message.WindowProtocol.GENERIC,
        )

    def _detectors_by_round(self) -> list:
        """Per window: round -> the model's detector ids of that round.

        In id order, so the concatenation over ascending rounds is
        exactly model.detector_ids, the layout the engine's row check
        enforces.
        """
        per_window = []
        for model in self.models:
            by_round = _detector_ids_by_round(model)
            per_window.append(by_round)
        return per_window

    def _planned_windows(self, config, distance: int) -> list:
        """The scheme's window chain, materialized as the planner does.

        frontends/planner.py: geometry fields, dependency edges,
        dependency counters.
        """
        qpu = dataclasses.replace(config.settings.qpu, distance=distance)
        code, _layout = qpu.build_code(config.settings.windows)
        scheme = config.settings.windows.kind
        if scheme != "sliding":
            raise ValueError(
                f"the offline lane supports sliding windows, not {scheme}"
            )
        plan = self._sliding_plan(code)
        windows = []
        for index, geometry in enumerate(plan.windows):
            window = _window_from_geometry(index, geometry)
            windows.append(window)
        _link_dependencies(windows, plan.internal_dependencies)
        return windows

    def _sliding_plan(self, code):
        """The sliding scheme's plan for this point's round count."""
        scheme = windowing_schemes.SlidingWindowScheme()
        commit_round_count = code.commit_rounds()
        buffer_round_count = code.buffer_rounds()
        return scheme.plan_operation(
            OPERATION_ID,
            self.rounds,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )

    def _sampled_shot(self, seed: int) -> tuple:
        """The seed's detection events and its logical observable truth."""
        segment = message.RunSeedPathSegment("field", "device")
        device_seed = seeding.derive_component_seed(seed, (segment,))
        device = stim_device.StimDevice(seed=device_seed)
        device.begin_operation(self.operation, self.rounds, self.rounds)
        sampled = device.sampled_detection_events(OPERATION_ID)
        events = numpy.asarray(sampled, dtype=numpy.uint8)
        observables = device.logical_observable_truth(OPERATION_ID)
        return events, tuple(observables)

    def _install_contribution(self, ledger, window, result) -> None:
        """The window's owned observable contribution, on the ledger."""
        contribution = message.LogicalContribution(
            owner_key=window.key,
            commit_lo=window.commit_lo,
            commit_hi=window.commit_hi,
            ownership_kind="ordinary_window",
            logical_observables=result.logical_observables,
        )
        ledger.install(contribution)

    def _committed_observables(self, ledger) -> tuple:
        """The whole shot's observables, read out of the ledger."""
        commit_los = []
        commit_his = []
        for window in self.windows:
            commit_los.append(window.commit_lo)
            commit_his.append(window.commit_hi)
        return ledger.observables_for_interval(
            OPERATION_ID,
            min(commit_los),
            max(commit_his),
            boundary_policy="strict",
        )

    def _window_job(
        self, window, model, by_round: dict, events, residual
    ) -> message.DecodeJob:
        """The decode job the engine would form for one window.

        The window's rounds as fragments, the previous window's
        committed boundary XORed into them
        (window_interactions.merge_boundary + apply_boundary, the same
        two functions the run calls).
        """
        plain_info = message.WindowInfo.from_window(window)
        state = self.interaction.initial_boundary_state(plain_info)
        if residual is not None:
            state = self._merged_boundary(window, model, state, residual)
        fragments = self._boundary_fragments(
            window, state, plain_info, events, by_round
        )
        job = message.DecodeJob(
            op_id=OPERATION_ID,
            window_id=window.k,
            n_rounds=window.n_rounds,
            dem=model,
            payloads=fragments,
            window=window,
            label=f"W{window.k}",
        )
        # the engine's detector-row layout check; a slicing mistake
        # raises here instead of silently decoding the wrong syndrome
        decoder_input = decoder_memory.materialize_decoder_input(job)
        job.payloads = _fragments_of(decoder_input)
        return job

    def _merged_boundary(self, window, model, state, residual):
        """The previous window's committed boundary merged into the state."""
        source_window_id = window.k - 1
        delivery = message.BoundaryDelivery(
            source_key=(OPERATION_ID, source_window_id),
            destination_key=window.key,
            source_revision=1,
            delivery_revision=1,
            latest_source_revision=1,
            latest_delivery_revision=1,
            source_operation_round_count=self.rounds,
            dependency_released=False,
            payload=residual,
        )
        model_info = message.WindowInfo.from_window(
            window, detector_positions=model.defect_positions
        )
        merged = self.interaction.merge_boundary(delivery, model_info, state)
        return merged.state

    def _boundary_fragments(
        self, window, state, plain_info, events, by_round: dict
    ) -> list:
        """The window's rounds as fragments, the boundary applied to each."""
        fragments = []
        last_read_round = min(window.buffer_hi, self.rounds)
        past_last_read = last_read_round + 1
        for round_index in range(window.start_round, past_last_read):
            fragment = _round_fragment(events, by_round, round_index)
            applied = self.interaction.apply_boundary(
                state, plain_info, fragment, round_index
            )
            fragments.append(applied)
        return fragments


def sweep_points(config) -> list:
    """(distance, p, shots) per point, sorted.

    The offline lane ignores the round period axis (no timing here), so
    a block must sweep exactly one.
    """
    points = {}
    for block in config.sweep:
        _refuse_timing_sweep(block)
        _record_block_points(points, block)
    items = points.items()
    ordered = sorted(items)
    rows = []
    for key, shots in ordered:
        distance, probability = key
        rows.append((distance, probability, shots))
    return rows


def plan(config_path: str, seeds_per_shard: int = None) -> Path:
    """The run dir and its shard table.

    Every shard is one line of shards.tsv: distance, p, seed_start,
    seed_count; seeds 0..shots-1 per point, split into seeds_per_shard
    chunks (default: one shard per point).
    """
    config = experiment_config.load_experiment(config_path)
    points = sweep_points(config)  # refuse bad sweeps before writing
    run_dir = run_module.new_run_dir(config)
    run_module.snapshot_code_state(config, run_dir)
    # the run submits its own copy, so the run dir records exactly how
    # it ran even after the script in experiments/ moves on
    this_file = Path(__file__)
    submit_script = this_file.parent / "slurm_offline.sh"
    submit_copy = run_dir / "slurm_offline.sh"
    shutil.copy2(submit_script, submit_copy)
    started_utc = _now_utc()
    run_module.write_manifest(config, run_dir, started_utc=started_utc)
    (run_dir / "shards").mkdir()
    (run_dir / "logs").mkdir()
    lines = _shard_lines(points, seeds_per_shard)
    _write_shard_table(run_dir, lines)
    _print_submit_instructions(run_dir, len(lines))
    return run_dir


def decode_shard(run_dir: str, shard_number: int) -> Path:
    """One shards.tsv line decoded to shards/shard_<n>.csv.

    The line number is 1-based, matching SLURM_ARRAY_TASK_ID.
    """
    run_dir = Path(run_dir)
    config = _run_config(run_dir)
    distance, probability, seed_start, seed_count = _shard_line(
        run_dir, shard_number
    )
    point = SweepPointDecoder(
        config, distance=distance, physical_error_probability=probability
    )
    rows = []
    algorithm = config.active_decoder.kind
    past_last_seed = seed_start + seed_count
    for seed in range(seed_start, past_last_seed):
        shot = point.decode_shot(seed)
        row = _shard_row(shot, distance, probability, algorithm)
        rows.append(row)
    shard_path = run_dir / "shards" / f"shard_{shard_number}.csv"
    sweep_report.write_csv(rows, shard_path)
    failures = _count_failures(rows)
    print(f"{shard_path}: {len(rows)} shots, {failures} failures")
    return shard_path


def merge(run_dir: str) -> list:
    """shards/ -> shots.csv, ler.csv, ler.png; stamps finished_utc.

    Shards stream one at a time in shards.tsv order, which is sorted by
    point and then seed range, so shots.csv comes out sorted without
    holding the run in memory (the big sweeps are 10^7 rows). A missing
    shard raises before anything is written.
    """
    run_dir = Path(run_dir)
    shard_paths = _shard_paths(run_dir)
    _refuse_missing_shards(shard_paths)
    by_point = _stream_shots(run_dir, shard_paths)
    ler_rows = _ler_rows(by_point)
    ler_path = run_dir / "ler.csv"
    sweep_report.write_csv(ler_rows, ler_path)
    config = _run_config(run_dir)
    title = plots.decoder_title(config)
    figure_path = run_dir / "ler.png"
    plots.ler_plot(ler_rows, figure_path, title=title)
    _restamp_manifest(config, run_dir)
    for row in ler_rows:
        line = _ler_line(row)
        print(line)
    return ler_rows


def main(argv) -> None:
    """The command line: plan, shard or merge."""
    if len(argv) < 3:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    verb = argv[1]
    if verb == "plan":
        chunk = None
        if len(argv) > 3:
            chunk = int(argv[3])
        plan(argv[2], seeds_per_shard=chunk)
        return
    if verb == "shard":
        decode_shard(argv[2], int(argv[3]))
        return
    if verb == "merge":
        merge(argv[2])
        return
    print(USAGE, file=sys.stderr)
    raise SystemExit(2)


def _detector_ids_by_round(model) -> dict:
    """Round -> the model's detector ids of that round, in id order."""
    ids_by_round = {}
    for detector_id in model.detector_ids:
        position = model.defect_positions[detector_id]
        round_index = position[0]
        round_ids = ids_by_round.setdefault(round_index, [])
        round_ids.append(detector_id)
    arrays = {}
    for round_index, ids in ids_by_round.items():
        arrays[round_index] = numpy.asarray(sorted(ids))
    return arrays


def _window_from_geometry(index: int, geometry):
    """One planned window from the scheme's geometry record."""
    return message.Window(
        op_id=OPERATION_ID,
        k=index,
        commit_lo=geometry.commit_lo,
        commit_hi=geometry.commit_hi,
        buffer_hi=geometry.buffer_hi,
        n_rounds=geometry.round_count,
        buffer_lo=geometry.buffer_lo,
        closed_temporal_boundaries=geometry.closed_temporal_boundaries,
    )


def _link_dependencies(windows: list, internal_dependencies) -> None:
    """The plan's dependency edges and each window's remaining count."""
    for source_index, destination_index in internal_dependencies:
        windows[destination_index].deps.append((OPERATION_ID, source_index))
        windows[source_index].dependents.append(
            (OPERATION_ID, destination_index)
        )
    for window in windows:
        window.deps_remaining = len(window.deps)


def _round_fragment(events, by_round: dict, round_index: int):
    """One round's detection events as a retained syndrome fragment."""
    round_events = events[by_round[round_index]]
    bits = []
    for bit in round_events:
        bits.append(int(bit))
    return message.RetainedSyndromeFragment(
        operation_id=OPERATION_ID,
        patch_id=0,
        round_index=round_index,
        bits=tuple(bits),
        size_bits=len(bits),
        fragment_index=0,
    )


def _fragments_of(decoder_input) -> list:
    """Every fragment of a materialized decoder input, in round order."""
    fragments = []
    for round_input in decoder_input.rounds:
        fragments.extend(round_input.fragments)
    return fragments


def _refuse_timing_sweep(block) -> None:
    """A block that sweeps the round period belongs to the closed loop."""
    if len(block.round_periods_microseconds) != 1:
        raise ValueError(
            "offline configs sweep one round period (timing sweeps "
            "belong to the closed-loop runner)"
        )


def _record_block_points(points: dict, block) -> None:
    """One block's (distance, p) points, each keeping its largest shots."""
    for distance in block.distances:
        for probability in block.physical_error_probabilities:
            key = (distance, probability)
            earlier = points.get(key, 0)
            points[key] = max(earlier, block.shots)


def _shard_lines(points: list, seeds_per_shard) -> list:
    """One tab-separated line per shard: distance, p, seed start, count."""
    lines = []
    for distance, probability, shots in points:
        chunk = seeds_per_shard or shots
        for seed_start in range(0, shots, chunk):
            remaining = shots - seed_start
            seed_count = min(chunk, remaining)
            lines.append(
                f"{distance}\t{probability}\t{seed_start}\t{seed_count}"
            )
    return lines


def _write_shard_table(run_dir: Path, lines: list) -> None:
    """shards.tsv, one line per shard, newline-terminated."""
    joined = "\n".join(lines)
    table_text = joined + "\n"
    table_path = run_dir / "shards.tsv"
    table_path.write_text(table_text)


def _print_submit_instructions(run_dir: Path, shard_count: int) -> None:
    """How to submit the array and how to merge it afterwards.

    --output at submit time sends every task's log into the run dir; the
    #SBATCH lines in the script cannot name the run dir themselves.
    """
    print(
        f"{run_dir}: {shard_count} shards\n"
        f"submit: sbatch --array=1-{shard_count} "
        f"--output={run_dir}/logs/slurm-%A_%a.out "
        f"{run_dir}/slurm_offline.sh {run_dir}\n"
        f"then:   python -m experiments.offline_run merge {run_dir}"
    )


def _shard_line(run_dir: Path, shard_number: int) -> tuple:
    """One shards.tsv line's distance, p, first seed and seed count."""
    table_path = run_dir / "shards.tsv"
    table_text = table_path.read_text()
    lines = table_text.splitlines()
    line = lines[shard_number - 1]
    distance, probability, seed_start, seed_count = line.split("\t")
    return (
        int(distance),
        float(probability),
        int(seed_start),
        int(seed_count),
    )


def _shard_row(
    shot: OfflineShot, distance: int, probability: float, algorithm
) -> dict:
    """One offline shot's row of a shard csv."""
    return {
        "distance": distance,
        "physical_error_probability": probability,
        "algorithm": algorithm,
        "seed": shot.seed,
        "windows": shot.windows,
        "logical_failure": shot.logical_failure,
        "wall_seconds": round(shot.wall_seconds, 6),
    }


def _count_failures(rows: list) -> int:
    """How many of a shard's rows failed."""
    failures = 0
    for row in rows:
        failures += row["logical_failure"]
    return failures


def _shard_paths(run_dir: Path) -> list:
    """Every shard csv the run's shard table asks for, in table order."""
    table_path = run_dir / "shards.tsv"
    table_text = table_path.read_text()
    lines = table_text.splitlines()
    past_last_shard = len(lines) + 1
    shard_dir = run_dir / "shards"
    paths = []
    for number in range(1, past_last_shard):
        shard_path = shard_dir / f"shard_{number}.csv"
        paths.append(shard_path)
    return paths


def _refuse_missing_shards(shard_paths: list) -> None:
    """A merge before every shard has decoded writes nothing."""
    missing = []
    for path in shard_paths:
        if not path.exists():
            missing.append(path.name)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} shards not decoded yet, first: {missing[0]}"
        )


def _stream_shots(run_dir: Path, shard_paths: list) -> dict:
    """Every shard's rows into shots.csv, totalled per sweep point."""
    by_point = {}
    shots_path = run_dir / "shots.csv"
    with open(shots_path, "w", newline="") as out:
        writer = None
        for shard_path in shard_paths:
            writer = _stream_shard(out, writer, shard_path, by_point)
    return by_point


def _stream_shard(out, writer, shard_path: Path, by_point: dict):
    """One shard's rows written through, and folded into the totals."""
    with open(shard_path) as handle:
        for row in csv.DictReader(handle):
            writer = _write_shot(out, writer, row, by_point)
    return writer


def _write_shot(out, writer, row: dict, by_point: dict):
    """One shot row into shots.csv and into its sweep point's totals."""
    if writer is None:
        writer = _shots_writer(out, row)
    writer.writerow(row)
    _add_shot_to_point(by_point, row)
    return writer


def _shots_writer(out, row: dict):
    """The shots.csv writer, its header taken from the first row."""
    field_names = list(row)
    writer = csv.DictWriter(out, fieldnames=field_names)
    writer.writeheader()
    return writer


def _add_shot_to_point(by_point: dict, row: dict) -> None:
    """One shard row folded into its sweep point's totals."""
    distance = int(row["distance"])
    probability = float(row["physical_error_probability"])
    key = (distance, probability)
    fresh = {
        "shots": 0,
        "failures": 0,
        "wall": 0.0,
        "algorithm": row["algorithm"],
    }
    point = by_point.setdefault(key, fresh)
    point["shots"] += 1
    point["failures"] += row["logical_failure"] == "True"
    point["wall"] += float(row["wall_seconds"])


def _ler_rows(by_point: dict) -> list:
    """One ler.csv row per sweep point, in point order."""
    items = by_point.items()
    ordered = sorted(items)
    rows = []
    for key, point in ordered:
        distance, probability = key
        row = _ler_row(distance, probability, point)
        rows.append(row)
    return rows


def _ler_row(distance: int, probability: float, point: dict) -> dict:
    """One sweep point's logical error rate row."""
    low, high = sweep_report.wilson_interval(point["failures"], point["shots"])
    shot_rate = point["failures"] / point["shots"]
    survival = 1.0 - shot_rate
    per_d_rounds = 1.0 - survival**D_ROUND_CHUNKS_PER_SHOT
    wall_per_shot = point["wall"] / point["shots"]
    return {
        "distance": distance,
        "physical_error_probability": probability,
        "algorithm": point["algorithm"],
        "shots": point["shots"],
        "failures": point["failures"],
        "logical_error_rate": shot_rate,
        "ler_per_d_rounds": per_d_rounds,
        "ler_wilson_low": low,
        "ler_wilson_high": high,
        "wall_seconds_per_shot": round(wall_per_shot, 6),
        # ler_plot groups by (distance, round period); one lane, one
        # period
        "round_period_us": 0.0,
    }


def _ler_line(row: dict) -> str:
    """One sweep point's terminal line after a merge."""
    return (
        f"d={row['distance']} p={row['physical_error_probability']:g}: "
        f"{row['failures']}/{row['shots']} failures, "
        f"LER {row['logical_error_rate']:.2e}"
    )


def _restamp_manifest(config, run_dir: Path) -> None:
    """The manifest again, with the merge's finish time."""
    manifest = _manifest(run_dir)
    finished_utc = _now_utc()
    run_module.write_manifest(
        config,
        run_dir,
        started_utc=manifest["started_utc"],
        finished_utc=finished_utc,
    )


def _run_config(run_dir: Path):
    """The run's own config copy, so a shard decodes what plan planned.

    Even after configs/ has moved on.
    """
    manifest = _manifest(run_dir)
    # config_files lists the loaded file first, then its extends chain
    first_file = Path(manifest["config_files"][0])
    top_of_chain = first_file.name
    config_path = run_dir / "config" / top_of_chain
    return experiment_config.load_experiment(config_path)


def _manifest(run_dir: Path) -> dict:
    """The run folder's manifest.json."""
    manifest_path = run_dir / "manifest.json"
    manifest_text = manifest_path.read_text()
    return json.loads(manifest_text)


def _now_utc() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


if __name__ == "__main__":
    main(sys.argv)
