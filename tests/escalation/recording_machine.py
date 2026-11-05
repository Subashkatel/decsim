"""The recording switching machine the replay and truth verdict tests run.

The test switching machine (tests/experiments/run_files.py) on Toshio's
double window, union-find and its cluster gap at stated times, writing
each window's record; and its shots, run and measured as a collect job
measures them.
"""

import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.parameters.assignments as assignments_module
import decsim.parameters.tree as tree
import tests.experiments.run_files as run_files


def switching_machine(*pairs, source="error_model_stim"):
    """The test switching machine on Toshio's double window, recording."""
    machine = run_files.switching_machine(rounds_per_shot=30)
    base = (
        ("qpu.source.type", source),
        ("observation.record_window_outcomes", "true"),
        ("switching.strong_window.type", "double_window"),
        ("windows.boundary_policy.type", "Eager"),
        ("windows.terminal_policy", "lookahead"),
        ("workload.arguments.physical_error_probability", "0.01"),
        # union-find at a stated time, its gap at a stated time, so a
        # recorded run's own times repeat
        ("weak.algorithm.type", "union_find"),
        ("weak.algorithm.timing.type", "PresetTime"),
        ("weak.algorithm.timing.latency", "0.3us"),
        ("switching.confidence.type", "cluster_gap"),
        ("switching.confidence.walk", "1us"),
    )
    texts = [f"{path}={value}" for path, value in base + pairs]
    written = assignments_module.from_texts(texts, "a test")
    applied = tree.apply(machine, written)
    return applied.settings


def measured_shots(name: str, machine, seeds) -> list:
    """Each seed's shot of the machine, run and measured."""
    task = collect.Task(name, machine)
    return [measured_shot(task, seed) for seed in seeds]


def measured_shot(task, seed: int):
    """One shot of the task, run and measured."""
    shot = collect.run_shot(task, seed)
    return measure.measure_shot(shot)
