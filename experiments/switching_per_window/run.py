"""Switching per window: each window's answer beside its true label.

Four configurations on the same shots: union-find alone, Relay-BP-5
alone, switching between them, and Tesseract alone as a reference
outside switching. Each runs Experiment 1's machine
(decsim/presets/experiment_one.py), so every latency is the one value
that preset holds. Switching keeps Relay-BP-5's answer, also
when it does not converge. Shots are drawn from the error model with
the errors that fired (error_model_stim), and every window that
delivers a shot's answer is written with its true label
(record_window_outcomes, window_outcomes.csv), beside Experiment 1's
switching and backlog records. A shot's samples depend on its seed
alone, so shot n is the same shot in all four configurations.

Every setting runs one fixed number of shots, the same for all four,
as decoder comparisons do (Toshio et al. 2510.25222 lines 741 and
1690; Dentelski et al. 2606.08758 lines 381-383 and 1165-1167):
the shots at which union-find alone makes about 100 failures,
100 / its rate in Experiment 1. A rate is shown only with at least 20
failures, with 95% Wilson intervals.

seconds_per_shot.csv beside this file sizes the Slurm jobs only: each
task's sim_wall_seconds_per_shot in Experiment 1 (switching, union-find
alone) or Experiment 3 (Relay-BP-5 and Tesseract alone) at the same
setting; where Experiment 3 has no row, the next higher physical error
rate's, an upper estimate. No result reads it.

Usage
-----

```
decsim run experiments/switching_per_window/run.py --slurm --dry-run
```
"""

import dataclasses
import math

import decsim
import decsim.decoders.tesseract.decoder as tesseract
import decsim.presets.experiment_one as experiment_one
import decsim.qpu.stim_device as stim_device
import decsim.topology.graph as graph

NAME = "switching_per_window"
DISTANCES = (5, 7, 9, 11)
PHYSICAL_ERROR_PROBABILITIES = (0.002, 0.003, 0.004, 0.005)
# the failures union-find alone is expected to make in each setting
UNION_FIND_FAILURES = 100
# Experiment 1's union-find alone, (shots, failures) at each
# (distance, physical error rate), every row at its 100-failure stop
UNION_FIND_EXPERIMENT_ONE = {
    (5, 0.002): (4445, 110),
    (5, 0.003): (2000, 133),
    (5, 0.004): (2000, 284),
    (5, 0.005): (2000, 477),
    (7, 0.002): (18727, 105),
    (7, 0.003): (4000, 106),
    (7, 0.004): (2000, 157),
    (7, 0.005): (2000, 325),
    (9, 0.002): (200000, 226),
    (9, 0.003): (11765, 114),
    (9, 0.004): (4000, 159),
    (9, 0.005): (2000, 201),
    (11, 0.002): (392157, 100),
    (11, 0.003): (35844, 103),
    (11, 0.004): (6445, 123),
    (11, 0.005): (2000, 131),
}


def configurations(distance: int, physical_error_probability: float) -> dict:
    """The four machines of one setting, by configuration name."""
    union_find_alone = experiment_one.weak_alone(
        distance, physical_error_probability
    )
    switching = experiment_one.switching(distance, physical_error_probability)
    relay_bp5 = experiment_one.strong_decoder_pool()
    # the package's tesseract-short-beam profile, order seed included
    # (tesseract-decoder src/tesseract_sinter_compat.pybind.h:468-474);
    # it has no measured device time, so its unit is held for its own
    # call's wall clock and only its error rate is a result
    short_beam = tesseract.TesseractDecoder.Settings(
        merge_errors=True, detector_order_seed=2384753
    )
    tesseract_pool = dataclasses.replace(relay_bp5, algorithm=short_beam)
    relay_bp5_alone = experiment_one.strong_alone(
        distance, physical_error_probability, relay_bp5
    )
    tesseract_alone = experiment_one.strong_alone(
        distance, physical_error_probability, tesseract_pool
    )
    machines = {
        "union_find_alone": union_find_alone,
        "switching": switching,
        "relay_bp5_alone": relay_bp5_alone,
        "tesseract_alone": tesseract_alone,
    }
    return {name: with_labels(machine) for name, machine in machines.items()}


def with_labels(machine: graph.Machine) -> graph.Machine:
    """The machine drawing shots with their errors, writing window labels."""
    error_model_source = stim_device.ErrorModelStimDevice.Settings()
    labelled = machine.with_settings("qpu", source=error_model_source)
    observation = dataclasses.replace(
        machine.observation, record_window_outcomes=True
    )
    return dataclasses.replace(labelled, observation=observation)


def shot_count(distance: int, physical_error_probability: float) -> int:
    """The shots at which union-find alone makes about 100 failures."""
    setting = (distance, physical_error_probability)
    shots, failures = UNION_FIND_EXPERIMENT_ONE[setting]
    expected_shots = UNION_FIND_FAILURES * shots / failures
    return math.ceil(expected_shots)


def switching_per_window_tasks() -> list:
    """Every configuration at every setting, the distance fastest."""
    grid = decsim.grid(
        physical_error_probability=PHYSICAL_ERROR_PROBABILITIES,
        distance=DISTANCES,
    )
    tasks = []
    for values in grid:
        probability = values["physical_error_probability"]
        distance = values["distance"]
        metadata = {
            "workload.arguments.physical_error_probability": probability,
            "qpu.distance": distance,
        }
        max_shots = shot_count(distance, probability)
        machines = configurations(distance, probability)
        collection = decsim.CollectionSettings(max_shots=max_shots)
        for name, machine in machines.items():
            task_name = f"{name}_d{distance}_p{probability}"
            task = decsim.Task(task_name, machine, metadata, collection)
            tasks.append(task)
    return tasks


tasks = switching_per_window_tasks()
experiment = decsim.Experiment(NAME, tasks)
