"""The decoder baseline, offline: Stim circuits, sinter, each decoder's adapter.

Rotated surface code memory in X and Z, d 5 to 15, six error rates, 100
rounds, four decoders; a point stops at 100 errors. The Slurm time limit
is the budget: a task stopped by it keeps what it saved, and the same
submission run again goes on from there. `python run.py --list` prints
the points, `python run.py <id>` runs one, and slurm/run.sbatch runs one
per array task (docs/how-to/run_an_experiment.md).
"""

import itertools

import relay_bp.stim.sinter.decoders as relay_bp_sinter
import stim
import tesseract_decoder

import decsim.experiment_runner as experiment_runner
import decsim.sinter_adapters.union_find as union_find_adapter

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

DECODERS = {
    "union-find": union_find_adapter.UnionFindDecoder(),
    # built into sinter
    "pymatching": None,
    # Relay-BP-1: the relay_bp package's own sinter adapter with the
    # surface code values of Mueller et al. 2506.01779 (lines 307, 332,
    # 343). It decodes one combined problem on sinter's decomposed model
    # in 64-bit, gammas drawn by the package; decsim's relay_bp row
    # decodes X and Z apart on the full fault model (the paper's
    # XZ-decoding) with seeded gammas, so the two are different
    # decoders. On the same 2000 shots at d 5, p 0.005, 100 rounds,
    # basis X, measured 2026-09-27, the adapter failed 696 (34.8%) and
    # the row 625 (31.3%) (tmp/job_1f7e5d61/relay_compare/compare.py).
    "relay-bp-1": relay_bp_sinter.SinterDecoder_RelayBP(
        gamma0=0.35,
        gamma_dist_interval=(-0.254, 0.985),
        pre_iter=80,
        num_sets=300,
        set_max_iter=60,
        stop_nconv=1,
    ),
    "tesseract-short-beam": TESSERACT_PROFILES["tesseract-short-beam"],
}


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


def baseline() -> experiment_runner.Experiment:
    """Every point, basis by distance by rate by decoder."""
    experiment = experiment_runner.Experiment("decoder_baseline")
    points = itertools.product(BASES, DISTANCES, ERROR_RATES, DECODERS)
    for basis, distance, error_rate, decoder in points:
        point_circuit = circuit(basis, distance, error_rate)
        labels = {"basis": basis, "d": distance, "p": error_rate}
        experiment.add_offline(
            point_circuit,
            decoder=decoder,
            labels=labels,
            max_errors=MAX_ERRORS,
            max_shots=MAX_SHOTS,
        )
    return experiment


# sinter's workers start by spawn and import this file again, so the
# points are built and run only when it is the script itself.
if __name__ == "__main__":
    experiment = baseline()
    experiment.main(custom_decoders=DECODERS)
