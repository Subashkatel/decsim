"""Whole shots against stored digests, so a result that moves fails here.

A characterization test (Feathers' golden master): it pins what the
machine does today, not what it should do. Each shape is one seed of a
small machine, and three things of its shot are compared with
shot_digests.json: the result's json, every seed the run derived with
its path, and the engine's final tick with the number of events it
scheduled. The shapes cover the decode slots (weak, strong, switching,
none), both strong windows, the three threshold sources, both cluster
gaps, the four window schemes, bounded links and buffers, the ported
store, the credit and reliable protocols, the chip side's priced
stages, the data path read in place, and Experiment 1's two machines,
each at d = 3 from the shipped base functions and run files.

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

import decsim.confidence.cluster as cluster
import decsim.confidence.extra_cluster as extra_cluster
import decsim.decoders.decoder as decoder_module
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.engine as engine_module
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.links.credit_channel as credit_channel
import decsim.links.framings as framings
import decsim.links.link_profiles as link_profiles
import decsim.links.reliable_channel as reliable_channel
import decsim.seeding as seeding
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.naive_online as naive_online
import decsim.windows.schemes.parallel as parallel
import decsim.windows.schemes.sandwich as sandwich
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


def extra_cluster_gap_switching() -> machine_settings.MachineSettings:
    machine = cluster_gap_switching()
    confidence = extra_cluster.ExtraClusterGap.Settings()
    switching = dataclasses.replace(machine.switching, confidence=confidence)
    return dataclasses.replace(machine, switching=switching)


def parallel_windows() -> machine_settings.MachineSettings:
    scheme = parallel.ParallelWindowScheme.Settings(
        commit_rounds=2, buffer_rounds=2
    )
    sparse = boundary_payloads.SparseSeamList.Settings()
    return _weak_alone_on_windows(scheme, sparse, None)


def sandwich_windows() -> machine_settings.MachineSettings:
    scheme = sandwich.TanSandwichScheme.Settings(
        commit_rounds=2, buffer_rounds=3
    )
    return _weak_alone_on_windows(scheme, None, None)


def naive_online_windows() -> machine_settings.MachineSettings:
    scheme = naive_online.NaiveOnlineScheme.Settings()
    sparse = boundary_payloads.SparseSeamList.Settings()
    held = boundary_policies.Held.Settings()
    return _weak_alone_on_windows(scheme, sparse, held)


def ported_buffer() -> machine_settings.MachineSettings:
    base = weak_alone()
    store = ported_syndrome_buffer.PortedSyndromeBufferSettings(
        read_write_ports=1, cycles_per_access=2, access_latency_cycles=1
    )
    return dataclasses.replace(base, weak_syndrome_buffer=store)


def protocol_links() -> machine_settings.MachineSettings:
    # a reliable RoCE hop that drops bits and retransmits, then a credit
    # hop in flits, so both protocols queue and the drops draw seeds
    base = weak_alone()
    roce = framings.RoceV2.Settings(path_mtu_bytes=256)
    reliable = reliable_channel.ReliableChannel.Settings(
        framing=roce,
        receive_buffer_frames=8,
        credit_latency_cycles=1,
        window_packets=4,
        ack_every_packets=2,
        retransmit_timeout_cycles=400,
        retry_count=7,
        bit_error_rate=0.0005,
        clock=machine_settings.FRIDGE_CLOCK,
    )
    flits = framings.Flits.Settings(flit_bits=8)
    credit = credit_channel.CreditChannel.Settings(
        framing=flits,
        receive_buffer_frames=2,
        credit_latency_cycles=1,
        clock=machine_settings.FRIDGE_CLOCK,
    )
    to_store = _protocol_path(base.links, "controller_to_weak_buffer", reliable)
    to_unit = _protocol_path(base.links, "weak_buffer_to_weak_decoder", credit)
    links = dataclasses.replace(
        base.links,
        controller_to_weak_buffer=to_store,
        weak_buffer_to_weak_decoder=to_unit,
    )
    return dataclasses.replace(base, links=links)


def chip_side_costs() -> machine_settings.MachineSettings:
    # the chip-side prices the bases leave free: two units with a bounded
    # memory read in words, held until their result is read, every stage
    # priced per job and per round, events formed at the decoder, and a
    # priced dispatch
    base = weak_alone()
    engine = decoder_settings.EngineSettings(
        fetch_cycles_per_round=3,
        fetch_cycles_per_job=2,
        release_cycles_per_job=5,
        release_cycles_per_round=7,
    )
    memory = decoder_settings.UnitMemorySettings(bits=4096, word_bits=8)
    weak_decoder = dataclasses.replace(
        base.weak_decoder,
        unit_count=2,
        engine=engine,
        unit_memory=memory,
        result_blocks_unit=True,
    )
    events = detection_event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",), latency_cycles=2, cycles_per_round=1
    )
    manager = dataclasses.replace(base.decoder_manager, dispatch_cycles=1)
    return dataclasses.replace(
        base,
        weak_decoder=weak_decoder,
        detection_events=events,
        decoder_manager=manager,
    )


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
    "extra_cluster_gap_switching": extra_cluster_gap_switching,
    "parallel_windows": parallel_windows,
    "sandwich_windows": sandwich_windows,
    "naive_online_windows": naive_online_windows,
    "ported_buffer": ported_buffer,
    "protocol_links": protocol_links,
    "chip_side_costs": chip_side_costs,
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
    task = collect.Task("task", settings, {})
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


def _weak_alone_on_windows(
    scheme, boundary_payload, boundary_policy
) -> machine_settings.MachineSettings:
    """The weak base on another scheme; None keeps the base's choice."""
    base = weak_alone()
    windows = dataclasses.replace(base.windows, scheme=scheme)
    if boundary_payload is not None:
        windows = dataclasses.replace(
            windows, boundary_payload=boundary_payload
        )
    if boundary_policy is not None:
        windows = dataclasses.replace(windows, boundary_policy=boundary_policy)
    return dataclasses.replace(base, windows=windows)


def _protocol_path(links, path_name: str, protocol):
    """One path of the card, one fridge cycle at 8 bits, on a protocol."""
    return link_profiles.path_card(
        links,
        path_name,
        clock=machine_settings.FRIDGE_CLOCK,
        latency_cycles=1,
        bits_per_cycle=8.0,
        source="the regression lock's protocol hop",
        protocol=protocol,
    )


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
