r"""A machine with the fitted correlated matching time runs at Table IX's mean.

Runs a few shots of the five results' correlated matching alone (sliding
windows of 2 d rounds) and of switching with the double window, every
window escalated (regions of 3 d rounds, which the campaign could only
stand in for), each with the fitted strong time
(five_results.fitted_strong_time), at p = 0.003 on Stim's live source,
on memories of rounds_per_distance d rounds, long so that a memory's
first and last windows, which read fewer rounds, weigh little. It
prints the mean strong decode the shots charged beside Table IX's
average for the rounds those decodes read (2604.12841 p.22, the five
results' CORRELATED_MATCHING_ROUND_MICROSECONDS_BY_DISTANCE), and the
standard error of their ratio over the shots: the fitted law's scale
makes the two agree over the campaign's calm full windows.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md),
through Slurm:

```
export PYTHONPATH=$PWD/../..
python check_machine.py --distances 5 9 --shots 8 --rounds-per-distance 40
```
"""

import argparse
import dataclasses
import statistics

import decsim
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.stim_device as stim_device
import decsim.windows.built_window_models as built_window_models

five_results = sweeps.load_beside(__file__, "../five_results/config.py")
PROBABILITY = 0.003
SCHEMES = ("sliding", "double")
# a cluster gap is finite, so a threshold this high escalates every
# window whose gap is finite
EVERY_WINDOW_DECIBELS = 1e9
TABLE_ROUND_MICROSECONDS = (
    five_results.CORRELATED_MATCHING_ROUND_MICROSECONDS_BY_DISTANCE
)


def fitted_machine(scheme: str, distance: int, rounds_per_distance: int):
    """The scheme's machine at p = 0.003 with the fitted strong time."""
    if scheme == "sliding":
        machine = five_results.alone("strong_alone", distance, PROBABILITY)
    else:
        machine = five_results.switching(
            scheme, distance, PROBABILITY, EVERY_WINDOW_DECIBELS
        )
    fitted = five_results.fitted_strong_time(machine, distance, scheme)
    workload = fitted.held("workload")
    longer = dataclasses.replace(
        workload, rounds_per_distance=rounds_per_distance
    )
    fitted = fitted.with_held("workload", longer)
    qpu = fitted.held(machines.QPU)
    source = stim_device.StimDevice.Settings()
    live = dataclasses.replace(qpu, source=source)
    return fitted.with_held(machines.QPU, live)


def check(
    scheme: str, distance: int, shots: int, rounds_per_distance: int
) -> None:
    """Run the shots; print the charged mean beside Table IX's."""
    machine = fitted_machine(scheme, distance, rounds_per_distance)
    task = decsim.Task(f"check_{scheme}_d{distance}", machine)
    built_models = built_window_models.BuiltWindowModels()
    windows = 0
    rounds = 0
    service_microseconds = 0.0
    shot_ratios = []
    for seed in range(shots):
        shot = collect.run_shot(task, seed, built_models=built_models)
        measured = measure.measure_shot(shot)
        windows += measured.escalated_windows
        rounds += measured.strong_decoded_rounds
        service_microseconds += measured.strong_service_sum_us
        shot_ratio = _charged_over_table(measured, distance)
        shot_ratios.append(shot_ratio)
    table_microseconds = TABLE_ROUND_MICROSECONDS[distance] * rounds
    charged_mean = service_microseconds / windows
    table_mean = table_microseconds / windows
    mean_rounds = rounds / windows
    ratio = charged_mean / table_mean
    spread = statistics.stdev(shot_ratios)
    standard_error = spread / shots**0.5
    print(
        f"d={distance} {scheme}: {windows} decodes of {mean_rounds:.1f} "
        f"rounds on average; charged mean {charged_mean:.2f} us, Table IX "
        f"{table_mean:.2f} us, ratio {ratio:.4f} (standard error over "
        f"shots {standard_error:.4f})"
    )


def main() -> None:
    """Check each distance and scheme asked for."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--rounds-per-distance", type=int, default=40)
    arguments = parser.parse_args()
    for distance in arguments.distances:
        for scheme in SCHEMES:
            check(
                scheme,
                distance,
                arguments.shots,
                arguments.rounds_per_distance,
            )


def _charged_over_table(measured, distance: int) -> float:
    """One shot's charged strong time over Table IX's for its rounds."""
    round_microseconds = TABLE_ROUND_MICROSECONDS[distance]
    table_microseconds = round_microseconds * measured.strong_decoded_rounds
    return measured.strong_service_sum_us / table_microseconds


if __name__ == "__main__":
    main()
