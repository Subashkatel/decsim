"""Time NVIDIA's Relay-BP on one GPU, one decode at a time.

Each decode is one decoder.decode(syndrome) call from Python, timed on
the host clock: the binding, the syndrome's copy to the device, the
kernels and the answer's copy back, with no dispatcher and no link. This
is what decsim/decoders/measured_table/measurements.py calls a decode's
time. The decoder is nv-qldpc-decoder from cudaq-qec-cu12, built at the
Relay-BP keys the region file carries (make_regions.py).

Run with a venv holding cudaq-qec-cu12, numpy and scipy, on a GPU node:

    python tools/relay_gpu_timing/time_decodes.py --region FILE \
        --device a100 --partition whole --out FILE.csv
"""

import argparse
import csv
import socket
import subprocess
import time

import cudaq_qec
import numpy
import scipy.sparse


def load_region(path):
    """The region's arrays and its check and observable matrices."""
    region = numpy.load(path)
    check = scipy.sparse.csr_matrix(
        (region["check_data"], region["check_indices"], region["check_indptr"]),
        shape=tuple(region["check_shape"]),
    )
    observables = scipy.sparse.csr_matrix(
        (region["obs_data"], region["obs_indices"], region["obs_indptr"]),
        shape=tuple(region["obs_shape"]),
    )
    return region, check, observables


def decoder_keys(region):
    """nv-qldpc-decoder's keys for the region's Relay-BP settings.

    Mueller et al. 2506.01779's Relay-BP: memory BP (bp_method 3,
    composition 1) in fp32 with min-sum scale 1, pre_iterations then
    relay_set_count legs of iterations_per_set each, stopping after
    converged_solution_count solutions ("NConv" with stop_nconv). The
    plugin's leg k reads explicit_gammas row k - 1 where relay-bp's leg k
    reads row k % relay_set_count (relay.rs init_next_set), so the table
    rolls by one row to run decsim's legs in decsim's order.
    """
    gammas = numpy.roll(region["gammas"], -1, axis=0)
    gamma_rows = gammas.astype(float)
    explicit_gammas = gamma_rows.tolist()
    gamma_low, gamma_high = region["gamma_interval"]
    relay = {
        "pre_iter": int(region["pre_iterations"]),
        "num_sets": int(region["relay_set_count"]),
        "stopping_criterion": "NConv",
        "stop_nconv": int(region["converged_solution_count"]),
    }
    return dict(
        use_sparsity=True,
        bp_method=3,
        composition=1,
        max_iterations=int(region["iterations_per_set"]),
        gamma0=float(region["gamma0"]),
        gamma_dist=[float(gamma_low), float(gamma_high)],
        explicit_gammas=explicit_gammas,
        srelay_config=relay,
        scale_factor=1.0,
        proc_float="fp32",
        opt_results={"num_iter": True},
    )


def gpu_name():
    """The GPU's name as nvidia-smi reports it."""
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    output = subprocess.run(query, capture_output=True, text=True, timeout=30)
    stdout = output.stdout.strip()
    names = stdout.splitlines()
    if not names:
        return "unknown"
    return names[0]


def timed_decode(decoder, check, observables, syndrome, flips):
    """One decode's time, iterations and whether its answer holds."""
    syndrome_values = _decoder_input(syndrome)
    start = time.perf_counter_ns()
    result = decoder.decode(syndrome_values)
    end = time.perf_counter_ns()
    elapsed_ns = end - start
    answer = numpy.asarray(result.result)
    flipped = answer > 0.5
    correction = flipped.astype(numpy.uint8)
    residual = check.dot(correction) % 2
    predicted = observables.dot(correction) % 2
    reported_iterations = result.opt_results.get("num_iter", -1)
    residual_bits = residual.astype(bool)
    predicted_bits = predicted.astype(bool)
    matches_syndrome = numpy.array_equal(residual_bits, syndrome)
    logical_correct = numpy.array_equal(predicted_bits, flips)
    fired = syndrome.sum()
    return dict(
        time_ns=elapsed_ns,
        iterations=int(reported_iterations),
        converged=int(bool(result.converged)),
        matches_syndrome=int(matches_syndrome),
        logical_correct=int(logical_correct),
        fired=int(fired),
    )


def main():
    """Time the region's decodes and write one CSV row per decode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    region, check, observables = load_region(args.region)
    decoder = _build_decoder(region, check)
    syndromes = region["syndromes"]
    flips = region["observable_flips"]
    shots = syndromes.shape[0]
    _warm_up(decoder, syndromes, args.warmup)
    request_count = min(args.requests, shots)
    rows = []
    for shot in range(request_count):
        row = timed_decode(
            decoder, check, observables, syndromes[shot], flips[shot]
        )
        rows.append(dict(shot=shot, **row))
    fixed = _run_facts(args, region, check)
    _write_rows(args.out, fixed, rows)
    _print_summary(args.out, rows)


def _decoder_input(syndrome):
    syndrome_floats = syndrome.astype(numpy.float32)
    return syndrome_floats.tolist()


def _build_decoder(region, check):
    priors = region["priors"].tolist()
    keys = decoder_keys(region)
    return cudaq_qec.get_decoder(
        "nv-qldpc-decoder", check, error_rate_vec=priors, **keys
    )


def _warm_up(decoder, syndromes, warmup_count):
    # the first decodes pay the device's warm-up, not a decode's own time
    shots = syndromes.shape[0]
    for shot in range(warmup_count):
        syndrome = syndromes[shot % shots]
        syndrome_values = _decoder_input(syndrome)
        decoder.decode(syndrome_values)


def _run_facts(args, region, check):
    gpu = gpu_name()
    host = socket.gethostname()
    return dict(
        device=args.device,
        gpu=gpu,
        host=host,
        partition=args.partition,
        cudaq_qec=cudaq_qec.__version__,
        distance=int(region["distance"]),
        rounds=int(region["rounds"]),
        probability=float(region["probability"]),
        detectors=check.shape[0],
        faults=check.shape[1],
    )


def _write_rows(path, fixed, rows):
    fieldnames = list(fixed) + list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**fixed, **row})


def _print_summary(path, rows):
    times_ns = numpy.array([row["time_ns"] for row in rows])
    times = times_ns / 1e3
    median = numpy.median(times)
    p99 = numpy.percentile(times, 99)
    longest = times.max()
    print(
        f"{path}: n={len(rows)} median={median:.1f}us "
        f"p99={p99:.1f}us max={longest:.1f}us"
    )


if __name__ == "__main__":
    main()
