"""The decoder baseline, offline: Stim circuits, sinter, each decoder's adapter.

Rotated surface code memory in X and Z, d 5 to 15, six error rates, 100
rounds, four decoders; a point stops at 100 errors. The Slurm time limit
is the budget: a task stopped by it keeps what it saved, and the same
submission run again goes on from there.

The command line is gem5 MultiSim's (gem5 v24.0 RELEASE-NOTES.md, "gem5
MultiSim"): `python run.py --list` prints the points, `python run.py <id>
--out DIR` runs one, `python run.py` runs them all and combines, and
`python run.py combine --out DIR` folds the saved points into stats.csv.
Each point is one sinter.collect into its own resume CSV, sinter's
save_resume_filepath (sinter/_collection/_collection.py), so array tasks
running at once never write one file and a resubmitted task goes on
where it stopped; stats.csv is sinter's combine, read_stats_from_csv_files
under its CSV_HEADER (sinter/_command/_main_combine.py). The folder keeps
run.json, the code state and a copy of this script, as every decsim
results folder does (docs/how-to/run_an_experiment.md).
"""

import argparse
import dataclasses
import itertools
import pathlib
import sys
from typing import Optional

import sinter
import stim
import tesseract_decoder

import decsim.decoders.relay_belief_propagation.decoder as relay_decoder
import decsim.experiments.run_folder as run_folder
import decsim.sinter_adapters.relay_bp as relay_bp_adapter
import decsim.sinter_adapters.union_find as union_find_adapter

NAME = "decoder_baseline"
_SCRIPT_AS_GIVEN = pathlib.Path(__file__)
SCRIPT = _SCRIPT_AS_GIVEN.resolve()
DISTANCES = [5, 7, 9, 11, 13, 15]
ERROR_RATES = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
BASES = ["x", "z"]
ROUNDS = 100
MAX_ERRORS = 100
# sinter needs a shot limit to stop a point that never reaches its
# errors; one far past what a time limit buys leaves the stop to Slurm.
MAX_SHOTS = 1_000_000_000
# the package's own profiles (src/tesseract_sinter_compat.pybind.h:466-472)
TESSERACT_PROFILES = tesseract_decoder.make_tesseract_sinter_decoders_dict()

# XYZ-Relay-BP-5 with the surface code values of Mueller et al.
# 2506.01779 (lines 307, 332, 343), the variant the paper finds
# comparable to matching on the surface code (lines 356-358), as the
# machine's relay_bp row's own record.
RELAY_BP = "xyz-relay-bp-5"
RELAY_BP_SETTINGS = relay_decoder.RelayBeliefPropagationDecoder.Settings(
    gamma0=0.35,
    gamma_interval=(-0.254, 0.985),
    pre_iterations=80,
    relay_set_count=600,
    iterations_per_set=60,
    converged_solution_count=5,
    bases="together",
)
# the seed each point's gamma table is drawn from, so a rerun repeats it
RELAY_BP_SEED = 20260927
# a point's decoders in the order of their ids
DECODERS = ["union-find", "pymatching", RELAY_BP, "tesseract-short-beam"]
# The decoders one object serves at every point; None is sinter's
# built-in. Relay-BP is built per point from its circuit
# (decsim/sinter_adapters/relay_bp.py says why).
SHARED_DECODERS = {
    "union-find": union_find_adapter.UnionFindDecoder(),
    "pymatching": None,
    "tesseract-short-beam": TESSERACT_PROFILES["tesseract-short-beam"],
}
# `run.py combine` folds every saved point into stats.csv
COMBINE = "combine"
POINTS_FOLDER = "points"
STATS_FILE = "stats.csv"


@dataclasses.dataclass(frozen=True)
class BaselinePoint:
    """One sinter task and the decoder object that decodes it.

    decoder is None for sinter's built-in of the task's decoder name. It
    is kept per point because a decoder may be built from its point's
    circuit, which sinter's compile step never sees (sinter.Decoder,
    compile_decoder_for_dem takes the model only).
    """

    task: sinter.Task
    decoder: Optional[sinter.Decoder]


def circuit(basis: str, distance: int, error_rate: float) -> stim.Circuit:
    """Stim's rotated memory circuit, one rate on all four noise channels.

    It is the circuit decsim.producers.memory_circuit builds, so this
    baseline and the machine's sample the same shots' distribution.
    """
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=ROUNDS,
        after_clifford_depolarization=error_rate,
        before_round_data_depolarization=error_rate,
        before_measure_flip_probability=error_rate,
        after_reset_flip_probability=error_rate,
    )


def point_decoder(
    decoder: str, point_circuit: stim.Circuit
) -> Optional[sinter.Decoder]:
    """The decoder object for one point, or None for sinter's built-in."""
    if decoder == RELAY_BP:
        return relay_bp_adapter.RelayBeliefPropagationDecoder(
            point_circuit, RELAY_BP_SETTINGS, RELAY_BP_SEED
        )
    return SHARED_DECODERS[decoder]


def baseline() -> list:
    """Every point, basis by distance by rate by decoder.

    The labels go into sinter's json_metadata, so stats.csv carries them
    beside every point's counts.
    """
    options = sinter.CollectionOptions(
        max_shots=MAX_SHOTS, max_errors=MAX_ERRORS
    )
    points = []
    grid = itertools.product(BASES, DISTANCES, ERROR_RATES, DECODERS)
    for basis, distance, error_rate, decoder in grid:
        point_circuit = circuit(basis, distance, error_rate)
        labels = {"basis": basis, "d": distance, "p": error_rate}
        task = sinter.Task(
            circuit=point_circuit,
            decoder=decoder,
            json_metadata=labels,
            collection_options=options,
        )
        custom_decoder = point_decoder(decoder, point_circuit)
        point = BaselinePoint(task, custom_decoder)
        points.append(point)
    return points


def main(arguments: Optional[list] = None) -> None:
    """Run what the command line asks: list, one point, all, or combine."""
    parser = _parser()
    parsed = parser.parse_args(arguments)
    points = baseline()
    if parsed.list:
        _print_points(points)
        return
    _refuse_no_workers(parser, parsed.workers)
    point_ids = _point_ids(parser, parsed.target, len(points))
    folder = _results_folder(parser, parsed)
    run_folder.refuse_another_tree(folder)
    every_id = list(range(len(points)))
    started_utc = run_folder.start_run(folder, SCRIPT, every_id)
    for point_id in point_ids:
        _collect_point(points[point_id], point_id, folder, parsed.workers)
    if parsed.target in (None, COMBINE):
        combine(folder, len(points))
    run_folder.finish_run(folder, SCRIPT, every_id, started_utc)


def combine(folder: pathlib.Path, point_count: int) -> None:
    """Every saved point's stats into stats.csv, sinter's combine."""
    paths = []
    for point_id in range(point_count):
        path = point_path(folder, point_id)
        if path.exists():
            paths.append(path)
    saved_statistics = sinter.read_stats_from_csv_files(*paths)
    lines = [sinter.CSV_HEADER]
    for point_statistics in saved_statistics:
        line = point_statistics.to_csv_line()
        lines.append(line)
    lines.append("")
    text = "\n".join(lines)
    stats_path = folder / STATS_FILE
    stats_path.write_text(text)


def point_path(folder: pathlib.Path, point_id: int) -> pathlib.Path:
    """Where a point's counts are saved, sinter's resume CSV."""
    return folder / POINTS_FOLDER / f"{point_id}.csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the decoder baseline's points with sinter."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="a point id from --list, or combine; all points when absent",
    )
    parser.add_argument(
        "--list", action="store_true", help="print every point's id"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="sinter's worker processes"
    )
    parser.add_argument(
        "--out",
        help="the results folder; a new dated one when every point runs",
    )
    return parser


def _print_points(points: list) -> None:
    for point_id, point in enumerate(points):
        label_pairs = []
        for key, value in point.task.json_metadata.items():
            label_pairs.append(f"{key}={value}")
        labels_text = " ".join(label_pairs)
        print(f"{point_id} decoder={point.task.decoder} {labels_text}")


def _refuse_no_workers(
    parser: argparse.ArgumentParser, worker_count: int
) -> None:
    """Sinter waits for its workers to answer, so none would hang it."""
    if worker_count < 1:
        parser.error(
            f"--workers is {worker_count}; sinter needs at least one worker "
            "process"
        )


def _point_ids(
    parser: argparse.ArgumentParser, target: Optional[str], point_count: int
) -> list:
    """Every point for no target, none for combine, else the one named."""
    if target is None:
        return list(range(point_count))
    if target == COMBINE:
        return []
    if not target.isdigit() or int(target) >= point_count:
        last_point_id = point_count - 1
        parser.error(
            f"{target!r} is no point id; --list shows the {point_count} "
            f"points, 0 to {last_point_id}, or give combine"
        )
    return [int(target)]


def _results_folder(
    parser: argparse.ArgumentParser, parsed: argparse.Namespace
) -> pathlib.Path:
    """--out, or a new dated folder for a run of every point.

    One point and combine name their folder, since every task of an
    array writes the one folder its points share.
    """
    if parsed.out is None and parsed.target is not None:
        parser.error(
            "one point or combine writes the folder its array shares; "
            "name it with --out"
        )
    return run_folder.run_dir_for(NAME, parsed.out)


def _collect_point(
    point: BaselinePoint,
    point_id: int,
    folder: pathlib.Path,
    worker_count: int,
) -> None:
    """One point collected to its stop rule, resumed from its own CSV."""
    decoders = {}
    if point.decoder is not None:
        decoders[point.task.decoder] = point.decoder
    path = point_path(folder, point_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    (point_statistics,) = sinter.collect(
        num_workers=worker_count,
        tasks=[point.task],
        custom_decoders=decoders,
        save_resume_filepath=path,
    )
    print(
        f"point {point_id}: {point_statistics.shots} shots, "
        f"{point_statistics.errors} errors, "
        f"{point_statistics.seconds:.0f} core seconds"
    )


# sinter's workers start by spawn and import this file again, so the
# points are built and run only when it is the script itself.
if __name__ == "__main__":
    main(sys.argv[1:])
