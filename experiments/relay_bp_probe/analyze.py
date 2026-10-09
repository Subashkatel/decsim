"""Why a Relay-BP-5 window does not converge: three checks on each one.

`python analyze.py <windows folder> <out folder>` writes windows.csv,
one row per captured window (size, detection events, status,
iterations), and failed_windows.csv, one row per window that did not
converge with:
  - whether BP+OSD-0 (the ldpc package) finds a correction that explains
    the syndrome, so a valid answer exists;
  - whether Relay-BP-5 converges with two other gamma seeds, 600 legs;
  - whether it converges with 3,000 legs, five times the budget.
The Relay-BP settings are Experiment 1's (switching_baseline.yaml
strong_decoder, 2506.01779 lines 303-344).
"""

import csv
import multiprocessing
import pathlib
import pickle
import sys

import numpy
import relay_bp
import scipy.sparse
from ldpc import BpOsdDecoder

GAMMA_INTERVAL = (-0.254, 0.985)
GAMMA0 = 0.35
PRE_ITERATIONS = 80
ITERATIONS_PER_LEG = 60
CONVERGED_SOLUTIONS = 5
# (column name, gamma seed, legs after the first)
RETRIES = (("seed_2_600_legs", 2, 600), ("seed_3_600_legs", 3, 600), ("seed_2_3000_legs", 2, 3000))


def relay_decoder(check, priors, seed: int, legs: int):
    generator = numpy.random.Generator(numpy.random.PCG64(seed))
    gamma_table = generator.uniform(*GAMMA_INTERVAL, size=(legs, check.shape[1]))
    return relay_bp.RelayDecoderF32(
        check,
        priors,
        gamma0=GAMMA0,
        pre_iter=PRE_ITERATIONS,
        num_sets=legs,
        set_max_iter=ITERATIONS_PER_LEG,
        gamma_dist_interval=GAMMA_INTERVAL,
        explicit_gammas=numpy.ascontiguousarray(gamma_table),
        stop_nconv=CONVERGED_SOLUTIONS,
        stopping_criterion="nconv",
        logging=False,
        seed=seed,
    )


def load(path: pathlib.Path) -> tuple:
    with open(path, "rb") as handle:
        window = pickle.load(handle)
    check = scipy.sparse.csr_matrix(window["check"]).astype(numpy.uint8)
    priors = numpy.asarray(window["priors"], dtype=float)
    syndrome = numpy.asarray(window["syndrome"], dtype=numpy.uint8)
    return window, check, priors, syndrome


def check_failed_window(path: pathlib.Path) -> dict:
    """The three checks on one window that did not converge."""
    _, check, priors, syndrome = load(path)
    osd = BpOsdDecoder(
        check, error_channel=list(priors), max_iter=1000,
        bp_method="minimum_sum", osd_method="osd0", osd_order=0,
    )
    correction = osd.decode(syndrome)
    explained = check @ correction % 2
    row = {
        "window": path.name,
        "osd_answer_explains_syndrome": bool(numpy.array_equal(explained, syndrome)),
        "osd_answer_weight": int(correction.sum()),
    }
    for column, seed, legs in RETRIES:
        result = relay_decoder(check, priors, seed, legs).decode_detailed(syndrome)
        row[f"{column}_converged"] = bool(result.success)
        row[f"{column}_iterations"] = int(result.iterations)
    return row


def main(windows_folder: pathlib.Path, out: pathlib.Path) -> None:
    window_rows = []
    failed_paths = []
    for path in sorted(windows_folder.glob("*.pkl")):
        window, check, _, syndrome = load(path)
        window_rows.append({
            "window": path.name,
            "detectors": check.shape[0],
            "fault_columns": check.shape[1],
            "detection_events": int(syndrome.sum()),
            "status": window["status"],
            "iterations": window["iterations"],
        })
        if window["status"] == "nonconverged":
            failed_paths.append(path)
    write(out / "windows.csv", window_rows)
    with multiprocessing.Pool(len(failed_paths) or 1) as pool:
        failed_rows = pool.map(check_failed_window, failed_paths)
    if failed_rows:
        write(out / "failed_windows.csv", failed_rows)


def write(path: pathlib.Path, rows: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
