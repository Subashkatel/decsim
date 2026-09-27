"""The decoder baseline, offline: Stim circuits, sinter, each decoder's adapter.

Rotated surface code memory in X and Z, d 5 to 15, six error rates, 100
rounds, four decoders; a point stops at 100 errors or at the shots 24
core-hours buy. `python run.py --list` prints the points, `python run.py
<id>` runs one, and slurm/run.sbatch runs one per array task
(docs/how-to/run_an_experiment.md).
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
# the owner's cap: 24 core-hours a point
CORE_SECONDS_PER_POINT = 86400
SHOT_CAP = 1_000_000_000
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

# Decode seconds a shot through each adapter above, from its
# decode_shots_bit_packed at p 0.005, the slowest rate here, 100 rounds,
# memory Z, one core of della-vis1, measured 2026-09-27 in process time
# with the sample and the compile left out. The shots behind each
# number: PyMatching 1000 to 4000, Union-Find 100 to 400, Relay-BP and
# Tesseract 3 to 20, so the slow two are rough. The slowest rate makes
# the cap safe at every rate and tight only at p 0.005.
SECONDS_PER_SHOT = {
    "union-find": {
        5: 0.00671,
        7: 0.0152,
        9: 0.0298,
        11: 0.0486,
        13: 0.0741,
        15: 0.102,
    },
    "pymatching": {
        5: 0.0000730,
        7: 0.000179,
        9: 0.000351,
        11: 0.000608,
        13: 0.000993,
        15: 0.00138,
    },
    "relay-bp-1": {
        5: 0.0486,
        7: 1.88,
        9: 18.5,
        11: 34.5,
        13: 49.0,
        15: 66.0,
    },
    "tesseract-short-beam": {
        5: 0.311,
        7: 1.64,
        9: 5.00,
        11: 16.6,
        13: 34.0,
        15: 40.5,
    },
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


def max_shots(decoder: str, distance: int) -> int:
    """The shots 24 core-hours buy at the decoder's speed, at most SHOT_CAP."""
    seconds_per_shot = SECONDS_PER_SHOT[decoder][distance]
    shot_count = CORE_SECONDS_PER_POINT / seconds_per_shot
    affordable = int(shot_count)
    return min(affordable, SHOT_CAP)


def baseline() -> experiment_runner.Experiment:
    """Every point, basis by distance by rate by decoder."""
    experiment = experiment_runner.Experiment("decoder_baseline")
    points = itertools.product(BASES, DISTANCES, ERROR_RATES, DECODERS)
    for basis, distance, error_rate, decoder in points:
        point_circuit = circuit(basis, distance, error_rate)
        labels = {"basis": basis, "d": distance, "p": error_rate}
        shot_cap = max_shots(decoder, distance)
        experiment.add_offline(
            point_circuit,
            decoder=decoder,
            labels=labels,
            max_errors=MAX_ERRORS,
            max_shots=shot_cap,
        )
    return experiment


# sinter's workers start by spawn and import this file again, so the
# points are built and run only when it is the script itself.
if __name__ == "__main__":
    experiment = baseline()
    experiment.main(custom_decoders=DECODERS)
