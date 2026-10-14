"""Diagnose the Ising row: each grid layout, and each output channel.

One whole-circuit d = 9 window under SD6 at p = 0.006, seeds apart from
ising_check.py's. For every layout that fits the window's stabilizers,
the row's failures with all four channels applied; then, on layout 0,
with no channel (which must equal PyMatching alone), data flips only and
measurement flips only. Run like ising_check.py:

    PYTHONPATH=. python experiments/switching_study/ising_layouts.py \\
        REPOSITORY CHECKPOINT SHOTS DEVICE
"""

import dataclasses
import pathlib
import sys

import stim

import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.frontends.deltakit as deltakit
from tests.decoders import windows

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ising_row  # noqa: E402

DISTANCE = 9
PROBABILITY = 0.006
SEED = 5


def main() -> None:
    repository, checkpoint, shot_text, device = sys.argv[1:5]
    shots = int(shot_text)
    base = ising_row.IsingMatchingDecoder.Settings(
        repository_path=repository, checkpoint_path=checkpoint, device=device
    )
    noiseless = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=DISTANCE, rounds=DISTANCE
    )
    circuit = deltakit.SD6Noise().noisy(noiseless, PROBABILITY)
    requirement = ising_row.IsingMatchingDecoder.fault_model_requirement
    model = windows.whole_circuit_window(circuit, DISTANCE, requirement)
    events, observables = windows.sampled_shots(circuit, shots, SEED)
    plain = matching.PyMatchingDecoder.Settings().build()
    print(f"pymatching failures={failures(plain, model, events, observables)}")
    fitting = ising_row.layouts(repository, model)
    print(f"layouts that fit: {fitting}", flush=True)
    for index, layout in enumerate(fitting):
        settings = dataclasses.replace(base, layout=index)
        count = failures(settings.build(), model, events, observables)
        print(f"layout {index} {layout}: failures={count}", flush=True)
    for channels in ((), (0, 1), (2, 3)):
        settings = dataclasses.replace(base, applied_channels=channels)
        count = failures(settings.build(), model, events, observables)
        print(f"layout 0 channels {channels}: failures={count}", flush=True)


def failures(row, model, events, observables) -> int:
    count = 0
    for shot, observable in zip(events, observables, strict=True):
        job = windows.job_for(model, shot)
        result = row.decode(job)
        truth = (int(observable[0]),)
        count += int(result.logical_observables != truth)
    return count


if __name__ == "__main__":
    main()
