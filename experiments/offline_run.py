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

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decsim.message import (BoundaryDelivery, DecodeJob, LogicalContribution,
                            Operation, RetainedSyndromeFragment,
                            RunSeedPathSegment, Window, WindowInfo,
                            WindowProtocol)
from decsim.decoders.decoder_memory import materialize_decoder_input
from decsim.qpu.stim_device import StimDevice
from decsim.seeding import derive_component_seed
from decsim.windows.committed_rounds import LogicalLedger
from decsim.windows.window_interactions import DefaultWindowInteraction

from experiments.build_run import code_model, decoder_engine, memory_circuit
from experiments.experiment_config import ExperimentConfig, load_experiment
from experiments.plots import ler_plot
from experiments.run import new_run_dir, snapshot_code_state, write_manifest
from experiments.sweep_report import wilson_interval, write_csv

OPERATION_ID = 1


@dataclass(frozen=True)
class OfflineShot:
    seed: int
    windows: int
    logical_failure: bool
    wall_seconds: float


class SweepPointDecoder:
    """One sweep point's immutable decode state, built once, reused for
    every seed: the circuit, the window chain, the window error models and
    a long-lived algorithm decoder (PyMatching caches its matching graph
    per window model; belief matching caches its BP decoder)."""

    def __init__(self, config: ExperimentConfig, *,
                 physical_error_probability: float, distance: int):
        self.circuit = memory_circuit(config, physical_error_probability,
                                      distance)
        self.rounds = config.rounds_per_shot.rounds_for(distance)
        self.operation = Operation(id=OPERATION_ID, name="memory",
                                   qubits=(0,), patches=(0,),
                                   circuit=self.circuit)
        self.windows = self._planned_windows(config, distance)
        self.algorithm = decoder_engine(config).decoder
        self.models = StimDevice().window_models_for_operation(
            self.operation, self.windows, self.rounds,
            fault_model_requirement=self.algorithm.fault_model_requirement,
            fault_exclusion_ranges=(),
            window_protocol=WindowProtocol.GENERIC)
        self.interaction = DefaultWindowInteraction()
        # round -> the model's detector ids of that round, in id order; the
        # concatenation over ascending rounds is exactly model.detector_ids,
        # the layout the engine's row check enforces
        self.detectors_by_round = []
        for model in self.models:
            by_round = {}
            for detector_id in model.detector_ids:
                round_index = model.defect_positions[detector_id][0]
                by_round.setdefault(round_index, []).append(detector_id)
            self.detectors_by_round.append(
                {round_index: np.asarray(sorted(ids))
                 for round_index, ids in by_round.items()})

    def _planned_windows(self, config: ExperimentConfig,
                         distance: int) -> list:
        """The scheme's window chain, materialized exactly as the planner
        does (frontends/planner.py): geometry fields, dependency edges,
        dependency counters."""
        code = code_model(config, distance)
        scheme = config.windowing.scheme
        if scheme != "sliding":
            raise ValueError(
                f"the offline lane supports sliding windows, not {scheme}")
        from decsim.windows.windowing_schemes import SlidingWindowScheme
        plan = SlidingWindowScheme().plan_operation(
            OPERATION_ID, self.rounds,
            commit_round_count=code.commit_rounds(),
            buffer_round_count=code.buffer_rounds())
        windows = [
            Window(op_id=OPERATION_ID, k=index,
                   commit_lo=geometry.commit_lo,
                   commit_hi=geometry.commit_hi,
                   buffer_hi=geometry.buffer_hi,
                   n_rounds=geometry.round_count,
                   buffer_lo=geometry.buffer_lo,
                   closed_temporal_boundaries=(
                       geometry.closed_temporal_boundaries))
            for index, geometry in enumerate(plan.windows)]
        for source_index, destination_index in plan.internal_dependencies:
            windows[destination_index].deps.append(
                (OPERATION_ID, source_index))
            windows[source_index].dependents.append(
                (OPERATION_ID, destination_index))
        for window in windows:
            window.deps_remaining = len(window.deps)
        return windows

    def decode_shot(self, seed: int) -> OfflineShot:
        """One shot, bit-identical to the closed-loop runner at the same
        seed: same device seed derivation, same sampled events and truth,
        same window decode chain."""
        wall_start = time.perf_counter()
        device_seed = derive_component_seed(
            seed, (RunSeedPathSegment("field", "device"),))
        device = StimDevice(seed=device_seed)
        device.begin_operation(self.operation, self.rounds, self.rounds)
        events = np.asarray(device.sampled_detection_events(OPERATION_ID),
                            dtype=np.uint8)
        truth = tuple(device.logical_observable_truth(OPERATION_ID))

        ledger = LogicalLedger()
        residual = None
        for window, model, by_round in zip(self.windows, self.models,
                                           self.detectors_by_round):
            job = self._window_job(window, model, by_round, events, residual)
            result = self.algorithm.decode(job)
            residual = self.interaction.boundary_from_result(result, None)
            ledger.install(LogicalContribution(
                owner_key=window.key,
                commit_lo=window.commit_lo, commit_hi=window.commit_hi,
                ownership_kind="ordinary_window",
                logical_observables=result.logical_observables))
        predicted = ledger.observables_for_interval(
            OPERATION_ID,
            min(window.commit_lo for window in self.windows),
            max(window.commit_hi for window in self.windows),
            boundary_policy="strict")
        return OfflineShot(
            seed=seed, windows=len(self.windows),
            logical_failure=tuple(predicted) != truth,
            wall_seconds=time.perf_counter() - wall_start)

    def _window_job(self, window: Window, model, by_round: dict,
                    events: np.ndarray, residual) -> DecodeJob:
        """The decode job the engine would form: the window's rounds as
        fragments, the previous window's committed boundary XORed into
        them (window_interactions.merge_boundary + apply_boundary, the
        same two functions the run calls)."""
        plain_info = WindowInfo.from_window(window)
        state = self.interaction.initial_boundary_state(plain_info)
        if residual is not None:
            delivery = BoundaryDelivery(
                source_key=(OPERATION_ID, window.k - 1),
                destination_key=window.key,
                source_revision=1, delivery_revision=1,
                latest_source_revision=1, latest_delivery_revision=1,
                source_operation_round_count=self.rounds,
                dependency_released=False, payload=residual)
            model_info = WindowInfo.from_window(
                window, detector_positions=model.defect_positions)
            state = self.interaction.merge_boundary(
                delivery, model_info, state).state
        fragments = []
        last_read_round = min(window.buffer_hi, self.rounds)
        for round_index in range(window.start_round, last_read_round + 1):
            bits = tuple(int(bit) for bit in events[by_round[round_index]])
            fragment = RetainedSyndromeFragment(
                operation_id=OPERATION_ID, patch_id=0,
                round_index=round_index, bits=bits, size_bits=len(bits),
                fragment_index=0)
            fragments.append(self.interaction.apply_boundary(
                state, plain_info, fragment, round_index))
        job = DecodeJob(op_id=OPERATION_ID, window_id=window.k,
                        n_rounds=window.n_rounds, dem=model,
                        payloads=fragments, window=window,
                        label=f"W{window.k}")
        # the engine's detector-row layout check; a slicing mistake raises
        # here instead of silently decoding the wrong syndrome
        decoder_input = materialize_decoder_input(job)
        job.payloads = [fragment for round_input in decoder_input.rounds
                        for fragment in round_input.fragments]
        return job


def sweep_points(config: ExperimentConfig) -> list:
    """(distance, p, shots) per point; the offline lane ignores the round
    period axis (no timing here), so a block must sweep exactly one."""
    points = {}
    for block in config.sweep:
        if len(block.round_periods_us) != 1:
            raise ValueError(
                "offline configs sweep one round period (timing sweeps "
                "belong to the closed-loop runner)")
        for distance in block.distances:
            for probability in block.physical_error_probabilities:
                key = (distance, probability)
                points[key] = max(points.get(key, 0), block.shots)
    return [(distance, probability, shots)
            for (distance, probability), shots in sorted(points.items())]


def plan(config_path: str, seeds_per_shard: int = None) -> Path:
    """The run dir and its shard table. Every shard is one line of
    shards.tsv: distance, p, seed_start, seed_count; seeds 0..shots-1
    per point, split into seeds_per_shard chunks (default: one shard
    per point)."""
    import shutil
    config = load_experiment(config_path)
    points = sweep_points(config)   # refuse bad sweeps before writing
    run_dir = new_run_dir(config)
    snapshot_code_state(config, run_dir)
    # the run submits its own copy, so the run dir records exactly how
    # it ran even after the script in experiments/ moves on
    shutil.copy2(Path(__file__).parent / "slurm_offline.sh",
                 run_dir / "slurm_offline.sh")
    write_manifest(config, run_dir, started_utc=_now_utc())
    (run_dir / "shards").mkdir()
    lines = []
    for distance, probability, shots in points:
        chunk = seeds_per_shard or shots
        for seed_start in range(0, shots, chunk):
            seed_count = min(chunk, shots - seed_start)
            lines.append(f"{distance}\t{probability}"
                         f"\t{seed_start}\t{seed_count}")
    (run_dir / "shards.tsv").write_text("\n".join(lines) + "\n")
    print(f"{run_dir}: {len(lines)} shards\n"
          f"submit: sbatch --array=1-{len(lines)} "
          f"{run_dir}/slurm_offline.sh {run_dir}\n"
          f"then:   python -m experiments.offline_run merge {run_dir}")
    return run_dir


def decode_shard(run_dir: str, shard_number: int) -> Path:
    """One shards.tsv line (1-based, matching SLURM_ARRAY_TASK_ID) decoded
    to shards/shard_<n>.csv."""
    run_dir = Path(run_dir)
    config = _run_config(run_dir)
    line = run_dir.joinpath("shards.tsv").read_text().splitlines()[
        shard_number - 1]
    distance, probability, seed_start, seed_count = line.split("\t")
    distance, probability = int(distance), float(probability)
    point = SweepPointDecoder(config, distance=distance,
                              physical_error_probability=probability)
    rows = []
    for seed in range(int(seed_start), int(seed_start) + int(seed_count)):
        shot = point.decode_shot(seed)
        rows.append({"distance": distance,
                     "physical_error_probability": probability,
                     "algorithm": config.active_decoder.algorithm,
                     "seed": shot.seed, "windows": shot.windows,
                     "logical_failure": shot.logical_failure,
                     "wall_seconds": round(shot.wall_seconds, 6)})
    shard_path = run_dir / "shards" / f"shard_{shard_number}.csv"
    write_csv(rows, shard_path)
    print(f"{shard_path}: {len(rows)} shots, "
          f"{sum(row['logical_failure'] for row in rows)} failures")
    return shard_path


def merge(run_dir: str) -> list:
    """shards/ -> shots.csv, ler.csv, ler.png; stamps finished_utc."""
    run_dir = Path(run_dir)
    shots = []
    for shard_path in sorted(run_dir.glob("shards/shard_*.csv")):
        with open(shard_path) as handle:
            shots.extend(csv.DictReader(handle))
    shots.sort(key=lambda row: (int(row["distance"]),
                                float(row["physical_error_probability"]),
                                int(row["seed"])))
    write_csv(shots, run_dir / "shots.csv")

    by_point = {}
    for row in shots:
        key = (int(row["distance"]),
               float(row["physical_error_probability"]))
        by_point.setdefault(key, []).append(row)
    ler_rows = []
    for (distance, probability), group in sorted(by_point.items()):
        failures = sum(row["logical_failure"] == "True" for row in group)
        low, high = wilson_interval(failures, len(group))
        ler_rows.append({
            "distance": distance,
            "physical_error_probability": probability,
            "algorithm": group[0]["algorithm"],
            "shots": len(group), "failures": failures,
            "logical_error_rate": failures / len(group),
            "ler_wilson_low": low, "ler_wilson_high": high,
            "wall_seconds_per_shot": round(
                sum(float(row["wall_seconds"]) for row in group)
                / len(group), 6),
            # ler_plot groups by (distance, round period); one lane,
            # one period
            "round_period_us": 0.0})
    write_csv(ler_rows, run_dir / "ler.csv")
    ler_plot(ler_rows, run_dir / "ler.png")

    config = _run_config(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    write_manifest(config, run_dir, started_utc=manifest["started_utc"],
                   finished_utc=_now_utc())
    for row in ler_rows:
        print(f"d={row['distance']} p={row['physical_error_probability']:g}: "
              f"{row['failures']}/{row['shots']} failures, "
              f"LER {row['logical_error_rate']:.2e}")
    return ler_rows


def _run_config(run_dir: Path) -> ExperimentConfig:
    """The run's own config copy, so a shard decodes what plan planned
    even if configs/ has moved on."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # config_files lists the loaded file first, then its extends chain
    top_of_chain = Path(manifest["config_files"][0]).name
    return load_experiment(run_dir / "config" / top_of_chain)


def _now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main(argv) -> None:
    usage = ("usage: python -m experiments.offline_run "
             "plan <config.yaml> [seeds_per_shard] "
             "| shard <run_dir> <n> | merge <run_dir>")
    if len(argv) < 3:
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    verb = argv[1]
    if verb == "plan":
        chunk = int(argv[3]) if len(argv) > 3 else None
        plan(argv[2], seeds_per_shard=chunk)
    elif verb == "shard":
        decode_shard(argv[2], int(argv[3]))
    elif verb == "merge":
        merge(argv[2])
    else:
        print(usage, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv)
