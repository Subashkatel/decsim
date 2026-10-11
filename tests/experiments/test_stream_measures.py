"""A stream's measures folded window by window (stream_measures.py).

The referent is measure_shot over the same run's whole ledgers: every
number both compute must be the same, the growth rates to 1e-9 (an
exact fit in ticks against statistics.linear_regression in floats). A
run that lets each window go must measure what the whole run does, and
leave no window, stage, frame write or round event behind.
"""

import collections
import dataclasses
import math

import pytest

import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.stream_measures as stream_measures
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

SEED = 17
# a window of three rounds commits every 3 us; 4 us of decoding a window
# overloads the tier, so its reaction grows and misses the deadline
DEADLINE_MICROSECONDS = 3.0
# (rounds, weak decode us, first recorded round, escalates)
RUNS = [
    (60, 0.5, 1, False),
    (120, 4.0, 20, False),
    (90, 0.5, 1, True),
]
# measure_shot's names for the pool numbers the stream keeps together
POOL_FIELDS = (
    "weak_queue_max",
    "strong_queue_max",
    "weak_busy_fraction",
    "strong_busy_fraction",
    "weak_offered_load",
    "strong_offered_load",
)


def _settings(rounds, latency, first_recorded, escalates, folds):
    """A d=3 flip stream protected from round 1 and read out at rounds."""
    program = memory_programs.memory_program(3, "Z", 0.01)
    workload = _stream_workload(program, rounds)
    source = flip_stream_device.FlipStreamDevice.Settings()
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source, round_period_microseconds=1.0
    )
    weak = _pool(latency)
    observation = observe_settings.ObservationSettings(
        first_recorded_round=first_recorded,
        reaction_deadline_microseconds=DEADLINE_MICROSECONDS,
        backlog_trace=True,
        record_switching_windows=escalates,
        folds_windows=folds,
    )
    frame = pauli_frame.PauliFrameConfig()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak,
        pauli_frame=frame,
        observation=observation,
    )
    if not escalates:
        return settings
    return _switching(settings)


def _stream_workload(program, rounds: int):
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    begin = program_records.Operation(
        1, "protect", (0,), patches=(0,), emits_detector_data=False
    )
    finish = dataclasses.replace(
        begin,
        id=2,
        name="readout",
        predecessors=(1,),
        scheduled_start_round=rounds,
    )
    region = program_records.ProtectedRegion(100, 1, 2)
    policy = round_policies.PerOperationRounds(((100, 0), (1, 0), (2, 0)))
    return workload_settings.WorkloadSettings(
        operations=(begin, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
        physical_circuits=((100, program),),
    )


def _pool(latency: float):
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=latency
    )
    return decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )


def _switching(settings):
    """A strong tier redoing the windows whose gap is under 12 dB."""
    strong_window = strong_window_shapes.RedoWindow.Settings()
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=12.0
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    strong = _pool(6.0)
    windows = window_settings.switching_windows(settings.windows, strong_window)
    return dataclasses.replace(
        settings, switching=switching, strong_decoder=strong, windows=windows
    )


def _run(settings):
    """(machine, result, the stream's measurement) of one run."""
    machine = machine_module.Machine.build(settings, SEED)
    measures = stream_measures.StreamMeasures(machine)
    result = machine.run()
    folded = measures.measurement(result)
    return machine, result, folded


def _differences(folded, shot) -> list:
    """Every number the stream and the shot both read, where they differ."""
    pairs = [
        ("decoded", folded.decoded_windows, shot.decoded_windows),
        ("failure", folded.logical_failure, shot.logical_failure),
        ("rounds", folded.rounds, shot.executed_rounds),
        ("recorded", folded.recorded_windows, len(shot.window_tiers)),
        (
            "missed",
            folded.deadline_missed_windows,
            shot.deadline_missed_windows,
        ),
        (
            "first miss",
            folded.first_deadline_miss_us,
            shot.first_deadline_miss_us,
        ),
        ("escalated", folded.escalated_windows, shot.escalated_windows),
        (
            "strong rounds",
            folded.strong_decoded_rounds,
            shot.strong_decoded_rounds,
        ),
        ("service", folded.strong_service_sum_us, shot.strong_service_sum_us),
        (
            "windows/us",
            folded.throughput_windows_per_us,
            shot.throughput_windows_per_us,
        ),
        (
            "rounds/us",
            folded.throughput_rounds_per_us,
            shot.throughput_rounds_per_us,
        ),
        ("queued", folded.max_queued_windows, shot.max_queued_windows),
        ("backlog", folded.backlog_peak_rounds, shot.backlog_peak_rounds),
        ("statuses", folded.window_statuses, shot.window_statuses),
        ("unscored", folded.unscored_reason, shot.unscored_reason),
        ("links", folded.link_totals, shot.link_totals),
        ("statistics", folded.statistics, shot.statistics),
    ]
    value_pairs = _value_pairs(folded, shot)
    pool_pairs = _pool_pairs(folded, shot)
    pairs.extend(value_pairs)
    pairs.extend(pool_pairs)
    tiers = collections.Counter(shot.window_tiers)
    pairs.append(("tiers", folded.window_tiers, dict(tiers)))
    end_samples = shot.samples["end_of_stream_reaction"]
    pairs.append(("end", [folded.end_of_stream_reaction_us], end_samples))
    differences = [(name, a, b) for name, a, b in pairs if a != b]
    growth_differences = _growth_differences(folded, shot)
    differences.extend(growth_differences)
    return differences


def _value_pairs(folded, shot) -> list:
    pairs = []
    for name, summary in folded.values.items():
        samples = shot.samples[name]
        pairs.append((f"{name} count", summary["count"], len(samples)))
        pairs.append((f"{name} mean", summary["mean"], shot.means[name]))
        pairs.append((f"{name} maximum", summary["maximum"], shot.maxes[name]))
    return pairs


def _pool_pairs(folded, shot) -> list:
    return [
        (name, folded.pools[name], getattr(shot, name)) for name in POOL_FIELDS
    ]


def _growth_differences(folded, shot) -> list:
    """The growth rates, each None in both or the same to 1e-9."""
    pairs = [
        (folded.weak_reaction_growth_rate, shot.weak_reaction_growth_rate),
        (folded.strong_reaction_growth_rate, shot.strong_reaction_growth_rate),
    ]
    return [
        ("growth", folded_rate, shot_rate)
        for folded_rate, shot_rate in pairs
        if not _same_rate(folded_rate, shot_rate)
    ]


def _same_rate(folded_rate, shot_rate) -> bool:
    if folded_rate is None or shot_rate is None:
        return folded_rate is shot_rate
    return math.isclose(folded_rate, shot_rate, rel_tol=1e-9)


@pytest.mark.parametrize("rounds, latency, first_recorded, escalates", RUNS)
def test_the_stream_measures_what_measure_shot_reads_off_the_whole_run(
    rounds, latency, first_recorded, escalates
):
    settings = _settings(rounds, latency, first_recorded, escalates, False)
    machine, result, folded = _run(settings)
    task = collect.Task("stream", settings, {})
    shot = collect.Shot(task, SEED, machine, result, 0.0)

    measured = measure.measure_shot(shot)

    assert _differences(folded, measured) == []
    assert folded.decoded_windows > 10
    assert folded.escalated_windows > 0 or not escalates
    assert folded.deadline_missed_windows > 0 or latency < 1.0


@pytest.mark.parametrize("rounds, latency, first_recorded, escalates", RUNS)
def test_a_stream_that_lets_each_window_go_measures_the_same_and_keeps_none(
    rounds, latency, first_recorded, escalates
):
    whole = _settings(rounds, latency, first_recorded, escalates, False)
    lean = _settings(rounds, latency, first_recorded, escalates, True)
    _machine, _result, kept = _run(whole)

    machine, _result, folded = _run(lean)

    observation = machine.observation
    assert folded == kept
    assert observation.windows.windows == {}
    assert observation.stages.records_for(100, 0) == ()
    assert observation.frame_corrections.committed == []
    assert observation.round_events.events == []
