"""Check the Ising row against PyMatching alone on the same SD6 shots.

One whole-circuit window of a rotated surface code Z memory, d rounds,
under SD6 at p. Every shot is decoded by the ising_pymatching row and by
decsim's pymatching row; both are scored against the sampled
observable. Prints the failures of each, the shots where they disagree,
and the network's average predicted and applied flips. Runs on a GPU
node from the decsim checkout:

    PYTHONPATH=. python experiments/switching_study/ising_check.py \\
        REPOSITORY CHECKPOINT SHOTS [DEVICE]
"""

import pathlib
import sys
import time

import numpy
import stim

import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.frontends.deltakit as deltakit
from tests.decoders import windows

# ising_row sits beside this file; experiments/ is no package
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ising_row  # noqa: E402

POINTS = ((9, 0.003), (9, 0.006), (13, 0.003), (13, 0.006))


def main() -> None:
    repository, checkpoint, shot_text = sys.argv[1:4]
    shots = int(shot_text)
    device = "cuda"
    if len(sys.argv) > 4:
        device = sys.argv[4]
    settings = ising_row.IsingMatchingDecoder.Settings(
        repository_path=repository, checkpoint_path=checkpoint, device=device
    )
    for distance, probability in POINTS:
        check_point(settings, distance, probability, shots)


def check_point(settings, distance: int, probability: float, shots: int):
    noiseless = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=distance
    )
    noise = deltakit.SD6Noise()
    circuit = noise.noisy(noiseless, probability)
    requirement = ising_row.IsingMatchingDecoder.fault_model_requirement
    model = windows.whole_circuit_window(circuit, distance, requirement)
    detection_events, observables = windows.sampled_shots(circuit, shots, 3)
    ising = settings.build()
    plain_settings = matching.PyMatchingDecoder.Settings()
    plain = plain_settings.build()
    counts = {"ising": 0, "pymatching": 0, "disagree": 0}
    predicted = 0
    applied = 0
    started = time.time()
    for shot, observable in zip(detection_events, observables, strict=True):
        job = windows.job_for(model, shot)
        ising_result = ising.decode(job)
        plain_result = plain.decode(job)
        truth = (int(observable[0]),)
        ising_wrong = ising_result.logical_observables != truth
        plain_wrong = plain_result.logical_observables != truth
        counts["ising"] += int(ising_wrong)
        counts["pymatching"] += int(plain_wrong)
        counts["disagree"] += int(ising_wrong != plain_wrong)
        predicted += ising_result.evidence[ising_row.PREDICTED_FLIPS]
        applied += ising_result.evidence[ising_row.APPLIED_FLIPS]
    seconds = time.time() - started
    print(
        f"d={distance} p={probability} shots={shots} "
        f"ising_failures={counts['ising']} "
        f"pymatching_failures={counts['pymatching']} "
        f"disagree={counts['disagree']} "
        f"predicted_per_shot={predicted / shots:.2f} "
        f"applied_per_shot={applied / shots:.2f} "
        f"seconds={seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
