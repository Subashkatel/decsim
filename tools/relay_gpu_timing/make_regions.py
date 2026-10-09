"""Strong regions to time on a GPU, at Experiment 1's Relay-BP keys.

A region is decsim's memory circuit of 3d rounds, r_strong = r_com +
2 r_buf at r_com = r_buf = d (Toshio et al. arXiv:2510.25222 lines
1246-1250), its physical fault catalog as decsim's relay_bp row reads
it, Stim detector samples, and decsim's gamma table drawn at the keys
experiments/switching_baseline/run.py strong_decoder_pool() sets. The
keys travel in the file, so time_decodes.py runs the GPU at the same
keys the table row will price.

Run with decsim's interpreter, PYTHONPATH=<tree>:<pydeps>:

    python tools/relay_gpu_timing/make_regions.py --out DIR \
        --distances 5 7 9 11 13 --probabilities 0.003 0.005 --shots 2000
"""

import argparse
import dataclasses
import importlib.util
import os

import numpy
import scipy.sparse

import decsim
import decsim.decoders.relay_belief_propagation.window_decoder as window
import decsim.detector_error_model.stim_fault_catalog as catalog
import decsim.frontends.settings as frontend_settings
import decsim.records.fault_model_contracts as contracts

CODE_TASK = "surface_code:rotated_memory_z"
GAMMA_SEED = 20260923


def experiment_relay_settings():
    """The Relay-BP keys Experiment 1's strong tier decodes at."""
    package_dir = os.path.dirname(decsim.__file__)
    tree = os.path.dirname(package_dir)
    run_path = os.path.join(tree, "experiments", "switching_baseline", "run.py")
    spec = importlib.util.spec_from_file_location("switching_run", run_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool = module.strong_decoder_pool()
    return pool.algorithm


def sparse_columns(sets, row_count):
    """One column per fault, a one in each row the fault flips."""
    rows = []
    columns = []
    for column, members in enumerate(sets):
        rows.extend(members)
        member_columns = [column] * len(members)
        columns.extend(member_columns)
    data = numpy.ones(len(rows), dtype=numpy.uint8)
    shape = (row_count, len(sets))
    return scipy.sparse.csr_matrix((data, (rows, columns)), shape=shape)


def region_arrays(distance, probability, shots, settings):
    """One region's check matrix, priors, samples and gamma table."""
    rounds = 3 * distance
    circuit = frontend_settings.memory_circuit(
        CODE_TASK, rounds, distance, probability
    )
    check, observables, priors = _fault_matrices(circuit)
    sample_seed = 7 + distance
    sampler = circuit.compile_detector_sampler(seed=sample_seed)
    syndromes, flips = sampler.sample(shots, separate_observables=True)
    gammas = window._gamma_table(settings, GAMMA_SEED, check.shape[1])
    check_shape = numpy.asarray(check.shape)
    observable_shape = numpy.asarray(observables.shape)
    single_precision_gammas = gammas.astype(numpy.float32)
    arrays = dict(
        distance=distance,
        rounds=rounds,
        probability=probability,
        measurement_bits=circuit.num_measurements,
        check_data=check.data,
        check_indices=check.indices,
        check_indptr=check.indptr,
        check_shape=check_shape,
        obs_data=observables.data,
        obs_indices=observables.indices,
        obs_indptr=observables.indptr,
        obs_shape=observable_shape,
        priors=priors,
        syndromes=syndromes,
        observable_flips=flips,
        gammas=single_precision_gammas,
        gamma_seed=GAMMA_SEED,
        sample_seed=sample_seed,
    )
    relay_arrays = _relay_key_arrays(settings)
    arrays.update(relay_arrays)
    return arrays


def main():
    """Write one region file per distance and noise strength asked for."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[5, 7, 9, 11, 13]
    )
    parser.add_argument(
        "--probabilities", type=float, nargs="+", default=[0.003, 0.005]
    )
    parser.add_argument("--shots", type=int, default=2000)
    args = parser.parse_args()
    settings = experiment_relay_settings()
    relay_keys = dataclasses.asdict(settings)
    print(f"relay keys: {relay_keys}")
    os.makedirs(args.out, exist_ok=True)
    for probability in args.probabilities:
        for distance in args.distances:
            arrays = region_arrays(distance, probability, args.shots, settings)
            name = f"region_d{distance}_p{probability}.npz"
            path = os.path.join(args.out, name)
            numpy.savez(path, **arrays)
            print(
                f"{path}: detectors={arrays['check_shape'][0]} "
                f"faults={arrays['check_shape'][1]} shots={args.shots}"
            )


def _fault_matrices(circuit):
    catalogs = catalog.prepare_fault_catalogs(
        circuit, contracts.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = catalogs.by_representation[
        contracts.FaultRepresentation.PHYSICAL
    ]
    check = sparse_columns(physical.detector_sets, circuit.num_detectors)
    observables = sparse_columns(
        physical.observable_sets, circuit.num_observables
    )
    priors = numpy.asarray(physical.priors, dtype=numpy.float64)
    return check, observables, priors


def _relay_key_arrays(settings):
    relay_keys = dataclasses.asdict(settings)
    gamma_interval = numpy.asarray(relay_keys["gamma_interval"])
    return dict(
        gamma0=relay_keys["gamma0"],
        gamma_interval=gamma_interval,
        pre_iterations=relay_keys["pre_iterations"],
        relay_set_count=relay_keys["relay_set_count"],
        iterations_per_set=relay_keys["iterations_per_set"],
        converged_solution_count=relay_keys["converged_solution_count"],
    )


if __name__ == "__main__":
    main()
