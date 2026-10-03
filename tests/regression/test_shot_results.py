"""Whole shots against stored digests, so a result that moves fails here.

A characterization test (Feathers' golden master): it pins what the
machine does today, not what it should do. Each shape is one seed of a
small machine, and three things of its shot are compared with
shot_digests.json: the result's json, every seed the run derived with
its path, and the engine's final tick with the number of events it
scheduled. The shapes cover the decode slots (weak, strong, switching,
none), both strong windows, the three threshold sources, the cluster
gap, bounded links and buffers, the data path read in place, and
Experiment 1's two machines, each at d = 3 from the shipped base
functions and run files.

The decoders' and confidence walks' host clocks are a counter stepping
20 microseconds a reading, as the fingerprint harness steps them, so a
host-timed decoder gives the same ticks on any host. Log and trace text
are formats, not results, and are left out.

A change that moves results on purpose writes new digests with
`python -m tests.regression.test_shot_results` and says so in its
commit message.
"""

import dataclasses
import functools
import hashlib
import importlib.util
import itertools
import json
import pathlib
import types

import pytest

import decsim.collect as collect
import decsim.confidence.cluster as cluster
import decsim.confidence.extra_cluster as extra_cluster
import decsim.decoders.decoder as decoder_module
import decsim.engine as engine_module
import decsim.escalation.threshold_sources as threshold_sources
import decsim.links.link_profiles as link_profiles
import decsim.seeding as seeding
import decsim.settings as machine_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

THIS_FILE = pathlib.Path(__file__)
HERE = THIS_FILE.parent
DIGESTS_PATH = HERE / "shot_digests.json"
# one row: 40 dB at d = 3 and p = 0.008, twice the fixed threshold
THRESHOLD_TABLE = HERE / "threshold_table.csv"
EXPERIMENTS = HERE.parents[1] / "experiments"
DISTANCE = 3
# the rotated surface code's checks a round
SYNDROME_BITS_PER_ROUND = DISTANCE * DISTANCE - 1
ROUND_PERIOD_MICROSECONDS = 1.0
SEED = 0
HOST_CLOCK_READERS = (decoder_module, cluster, extra_cluster)


def weak_alone() -> machine_settings.MachineSettings:
    return machine_settings.weak_decoder_baseline(
        DISTANCE, 0.003, ROUND_PERIOD_MICROSECONDS
    )


def strong_alone() -> machine_settings.MachineSettings:
    return machine_settings.strong_decoder_baseline(
        DISTANCE, 0.003, ROUND_PERIOD_MICROSECONDS
    )


def redo_window_switching() -> machine_settings.MachineSettings:
    run_file = _run_file("switching")
    return run_file.redo_window_switching(DISTANCE)


def cluster_gap_switching() -> machine_settings.MachineSettings:
    run_file = _run_file("switching")
    return run_file.cluster_gap_switching(DISTANCE)


def table_threshold() -> machine_settings.MachineSettings:
    table = threshold_sources.TableThreshold.Settings(table=THRESHOLD_TABLE)
    return _redo_window_switching_on(table)


def online_threshold() -> machine_settings.MachineSettings:
    # an audit rate high enough that a short shot audits, so the
    # calibrator moves its threshold within the shot
    online = threshold_sources.OnlineThreshold.Settings(
        threshold_decibels=20.0,
        target_escalation_rate=0.01,
        step_decibels=0.5,
        audit_rate=0.7,
        kept_bad_budget=0.001,
        adjust_factor=2.0,
        min_escalation_rate=0.0001,
        max_escalation_rate=0.9,
    )
    return _redo_window_switching_on(online)


def bounded_links() -> machine_settings.MachineSettings:
    # every path carries exactly its nominal traffic, so jitter queues
    base = weak_alone()
    links = link_profiles.bandwidth_limited_profile(
        syndrome_bits_per_round=SYNDROME_BITS_PER_ROUND,
        round_microseconds=ROUND_PERIOD_MICROSECONDS,
        commit_rounds=DISTANCE,
        buffer_rounds=DISTANCE,
    )
    return dataclasses.replace(base, links=links)


def bounded_buffer() -> machine_settings.MachineSettings:
    # a 64-bit store behind a 5 microsecond unit fills, and the
    # controller holds rounds
    base = weak_alone()
    store = dataclasses.replace(base.weak_syndrome_buffer, bits=64)
    slow = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=5.0
    )
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=slow)
    return dataclasses.replace(
        base, weak_syndrome_buffer=store, weak_decoder=weak_decoder
    )


def data_path_in_place() -> machine_settings.MachineSettings:
    run_file = _run_file("data_movement")
    return run_file.weak_input_in_place(DISTANCE)


def experiment_one_switching() -> machine_settings.MachineSettings:
    run_file = _run_file("switching_baseline")
    return run_file.switching(DISTANCE, 0.003)


def experiment_one_weak_alone() -> machine_settings.MachineSettings:
    run_file = _run_file("switching_baseline")
    return run_file.weak_alone(DISTANCE, 0.003)


def no_decoder() -> machine_settings.MachineSettings:
    base = weak_alone()
    workload = dataclasses.replace(base.workload, decode_operations=())
    return dataclasses.replace(base, weak_decoder=None, workload=workload)


SHAPES = {
    "weak_alone": weak_alone,
    "strong_alone": strong_alone,
    "redo_window_switching": redo_window_switching,
    "cluster_gap_switching": cluster_gap_switching,
    "table_threshold": table_threshold,
    "online_threshold": online_threshold,
    "bounded_links": bounded_links,
    "bounded_buffer": bounded_buffer,
    "data_path_in_place": data_path_in_place,
    "experiment_one_switching": experiment_one_switching,
    "experiment_one_weak_alone": experiment_one_weak_alone,
    "no_decoder": no_decoder,
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_a_shot_gives_its_recorded_results(name):
    shape = SHAPES[name]
    digests_text = DIGESTS_PATH.read_text()
    recorded = json.loads(digests_text)

    digest = shot_digest(shape)

    assert digest == recorded[name]


def shot_digest(shape) -> dict:
    """One shot of the shape: its result, seeds, final tick and events."""
    settings = shape()
    task = collect.Task(settings, {})
    seed_rows = []
    event_counter = itertools.count()
    derive = _recording_derive(seeding.derive_component_seed, seed_rows)
    schedule = _counting_schedule(engine_module.Engine.schedule, event_counter)
    with pytest.MonkeyPatch.context() as patch:
        for module in HOST_CLOCK_READERS:
            clock = _CountingClock()
            patch.setattr(module, "time", clock)
        patch.setattr(seeding, "derive_component_seed", derive)
        patch.setattr(engine_module.Engine, "schedule", schedule)
        shot = collect.run_shot(task, SEED)
    result = collect.json_value(shot.result, record_classes=False)
    result_text = json.dumps(result, sort_keys=True)
    seeds_text = "\n".join(seed_rows)
    return {
        "result_sha256": _sha256(result_text),
        "seeds_sha256": _sha256(seeds_text),
        "final_tick": shot.machine.engine.now,
        "event_count": next(event_counter),
    }


def write_digests() -> None:
    """Record every shape's digest, for a change that moves results."""
    digests = {}
    for name, shape in SHAPES.items():
        digests[name] = shot_digest(shape)
    text = json.dumps(digests, indent=2)
    DIGESTS_PATH.write_text(f"{text}\n")


class _CountingClock:
    """perf_counter_ns stepping 20 microseconds a reading, from zero."""

    def __init__(self) -> None:
        self.readings = itertools.count(0, 20_000)

    def perf_counter_ns(self) -> int:
        return next(self.readings)


@functools.cache
def _run_file(name: str) -> types.ModuleType:
    """experiments/<name>/run.py as a module, for its machine functions."""
    path = EXPERIMENTS / name / "run.py"
    module_name = f"regression_run_file_{name}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _redo_window_switching_on(threshold) -> machine_settings.MachineSettings:
    """The redo window switching point with its threshold source swapped."""
    machine = redo_window_switching()
    switching = dataclasses.replace(machine.switching, threshold=threshold)
    return dataclasses.replace(machine, switching=switching)


def _recording_derive(derive, seed_rows: list):
    """derive_component_seed, writing each path's bytes and seed down."""

    def recording_derive(root_seed: int, path) -> int:
        seed = derive(root_seed, path)
        path_bytes = bytearray()
        for segment in path:
            segment_bytes = segment.canonical_bytes()
            path_bytes.extend(segment_bytes)
        path_text = path_bytes.hex()
        seed_rows.append(f"{path_text} {seed}")
        return seed

    return recording_derive


def _counting_schedule(schedule, event_counter):
    """Engine.schedule, counting each event it queues."""

    def counting_schedule(engine, *arguments, **keywords):
        next(event_counter)
        return schedule(engine, *arguments, **keywords)

    return counting_schedule


def _sha256(text: str) -> str:
    encoded = text.encode()
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


if __name__ == "__main__":
    write_digests()
