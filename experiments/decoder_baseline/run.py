"""The decoder baseline, offline: Stim circuits, sinter, each decoder's adapter.

Rotated surface code memory in X and Z, d 5 to 15, six error rates, 100
rounds, four decoders; a point stops at 100 errors. The Slurm time limit
is the budget: a task stopped by it keeps what it saved, and the same
submission run again goes on from there. `python run.py --list` prints
the points, `python run.py <id>` runs one, and slurm/run.sbatch runs one
per array task (docs/how-to/run_an_experiment.md).
"""

import itertools
from typing import Optional

import sinter
import stim
import tesseract_decoder

import decsim.experiment_runner as experiment_runner
import decsim.sinter_adapters.relay_bp as relay_bp_adapter
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

# Relay-BP-1 with the surface code values of Mueller et al. 2506.01779
# (lines 307, 332, 343), X and Z decoded apart as the paper's
# XZ-decoding, in the machine's relay_bp row's own keys.
RELAY_BP = "relay-bp-1"
RELAY_BP_SETTINGS = {
    "gamma0": 0.35,
    "gamma_interval": [-0.254, 0.985],
    "pre_iterations": 80,
    "relay_set_count": 300,
    "iterations_per_set": 60,
    "converged_solution_count": 1,
    "bases": "apart",
}
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


def baseline() -> experiment_runner.Experiment:
    """Every point, basis by distance by rate by decoder."""
    experiment = experiment_runner.Experiment("decoder_baseline")
    points = itertools.product(BASES, DISTANCES, ERROR_RATES, DECODERS)
    for basis, distance, error_rate, decoder in points:
        point_circuit = circuit(basis, distance, error_rate)
        labels = {"basis": basis, "d": distance, "p": error_rate}
        custom_decoder = point_decoder(decoder, point_circuit)
        experiment.add_offline(
            point_circuit,
            decoder=decoder,
            labels=labels,
            max_errors=MAX_ERRORS,
            max_shots=MAX_SHOTS,
            custom_decoder=custom_decoder,
        )
    return experiment


# sinter's workers start by spawn and import this file again, so the
# points are built and run only when it is the script itself.
if __name__ == "__main__":
    experiment = baseline()
    experiment.main()
