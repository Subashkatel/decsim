"""The five results' machines: the basic case's, on the paper's values.

    alone("weak_alone", 9, 0.003)
    alone("strong_alone", 9, 0.003)
    switching("double", 9, 0.003, threshold_decibels=12.0,
              strong_unit_count=2)
    recording_gap(alone("weak_alone", 9, 0.003))
    streaming(alone("weak_alone", 9, 0.003))
    fitted_strong_time(alone("strong_alone", 9, 0.003), 9, "sliding")

The basic case's machine (experiments/basic_case/config.py) with the
values of specs/adaptive-switching/values.md where they differ: rounds of
1.1 us, the chip's syndrome buffer moving a 32-bit word a cycle each
way, and the strong decoder correlated matching on one core of a CPU
host, over Backline's RoCE v2 link, at the paper's average time per
round. A fixed time has no tail and does not slow down in a burst, so it
serves the pilots at steady noise only (experiments.md, decision D4);
fitted_strong_time prices each strong decode by a law in its window's
detection events instead, whose calm mean is that average (B1 stage 2).
"""

import dataclasses

import decsim
import decsim.confidence.cluster as cluster
import decsim.decoders.settings as decoder_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.topology.graph as graph
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)
from decsim.decoders.minimum_weight_perfect_matching import (
    detection_event_time,
)

basic_case = sweeps.load_beside(__file__, "../basic_case/config.py")
# values.md section 1, round period: Willow (2408.13687 p.1)
ROUND_PERIOD_MICROSECONDS = 1.1
# values.md section 5: correlated matching's time per round on a Grace
# CPU, batch 1, at p = 0.003 (2604.12841 p.22, Table IX); at other p the
# same times are assumed
CORRELATED_MATCHING_ROUND_MICROSECONDS_BY_DISTANCE = {
    5: 1.15,
    9: 3.35,
    13: 9.14,
}
# values.md section 2: Backline's RoCE v2 echo to a CPU, 2.305 us a round
# trip (2609.09270 p.13)
STRONG_LINK = "roce_cpu"
# values.md section 2, syndrome buffer: the bus to the decoder moves
# 32-bit words (Caune et al. 2410.05202 p.10, a 32-bit WISHBONE bus), one
# a cycle in and one a cycle out (the meeting's rule)
BUFFER_WORD = "32 bits"
BUFFER_ACCESS = "1 cycle"
# the rounds one strong decode reads, in code distances: a sliding
# window's d committed and d looked ahead (values.md section 6), also
# the window speculation redoes; the double window's r_com + 2 r_buf
# (Toshio et al. 2510.25222 line 1250)
STRONG_DECODE_DISTANCES_BY_SCHEME = {
    "sliding": 2,
    "speculation": 2,
    "double": 3,
}
# B1 stage 2 (experiments.md, decision D4): correlated matching's time
# as a law in each window's detection events, fitted to decsim's own
# decodes on one core of an AMD EPYC 9654 at p = 0.003 times 1, 2, 4
# and 10, and scaled so its mean over calm full windows is the average
# above; keyed by distance and the rounds a decode reads, in distances.
# The law with the least held-out error: intercept and power at d = 5,
# power at d = 9 and 13 (experiments/correlated_matching_timing)
FITTED_SOURCE = (
    "results/correlated_matching_timing/2026-10-29, timing_genoa_d5_9 "
    "and timing_genoa_d13 fit.json"
)
FITTED_CORRELATED_MATCHING_TIMES = {
    (5, 2): detection_event_time.DetectionEventTime(
        intercept_microseconds=5.3707,
        coefficient_microseconds=0.078224,
        exponent=1.4832,
        scale=1.1092,
        source=FITTED_SOURCE,
    ),
    (5, 3): detection_event_time.DetectionEventTime(
        intercept_microseconds=11.9653,
        coefficient_microseconds=0.005502,
        exponent=2.0209,
        scale=1.0986,
        source=FITTED_SOURCE,
    ),
    (9, 2): detection_event_time.DetectionEventTime(
        intercept_microseconds=0.0,
        coefficient_microseconds=0.006981,
        exponent=1.9456,
        scale=1.1000,
        source=FITTED_SOURCE,
    ),
    (9, 3): detection_event_time.DetectionEventTime(
        intercept_microseconds=0.0,
        coefficient_microseconds=0.007211,
        exponent=1.9014,
        scale=0.9058,
        source=FITTED_SOURCE,
    ),
    (13, 2): detection_event_time.DetectionEventTime(
        intercept_microseconds=0.0,
        coefficient_microseconds=0.009304,
        exponent=1.8236,
        scale=0.7339,
        source=FITTED_SOURCE,
    ),
    (13, 3): detection_event_time.DetectionEventTime(
        intercept_microseconds=0.0,
        coefficient_microseconds=0.006731,
        exponent=1.8480,
        scale=0.6230,
        source=FITTED_SOURCE,
    ),
}
# the saved datasets' seeds (experiments.md, How we run them): the
# calibration seeds, 0 to 9,999, set the thresholds, and every result
# reads the evaluation seeds from 10,000, so the two never meet
CALIBRATION_SHOTS = 10_000
EVALUATION_FIRST_SEED = CALIBRATION_SHOTS
EVALUATION_SHOTS = 10_000
# the cells metadata() writes under the names the reports read
POINT_CELLS = ("distance", "physical_error_probability")


def alone(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One decoder alone, weak_alone or strong_alone, on the paper's values.

    Correlated matching alone decodes sliding windows.
    """
    strong_pool = correlated_matching_pool(distance, "sliding")
    basic = basic_case.machine(
        decoding,
        distance,
        physical_error_probability,
        strong_link=STRONG_LINK,
        strong_pool=strong_pool,
    )
    return on_paper_values(basic, distance)


def switching(
    scheme: str,
    distance: int,
    physical_error_probability: float,
    threshold_decibels: float,
    strong_unit_count: int = 1,
) -> graph.Machine:
    """Union-find first, correlated matching redoing the unsure windows.

    scheme is how an escalated window is handled (basic_case.SCHEMES):
    sliding, speculation or double. A window whose cluster gap is under
    threshold_decibels escalates, and the strong pool holds
    strong_unit_count cores.
    """
    strong_pool = correlated_matching_pool(distance, scheme)
    basic = basic_case.machine(
        "switching",
        distance,
        physical_error_probability,
        strong_link=STRONG_LINK,
        strong_pool=strong_pool,
    )
    window_rule = basic_case.SCHEMES[scheme]
    sized_rule = dataclasses.replace(
        window_rule, strong_units=strong_unit_count
    )
    schemed = basic_case.with_scheme(basic, sized_rule)
    threshold = threshold_sources.FixedThreshold.Settings(threshold_decibels)
    thresholded = schemed.with_settings(machines.SWITCHING, threshold=threshold)
    return on_paper_values(thresholded, distance)


def correlated_matching_pool(
    distance: int, scheme: str
) -> decoder_settings.DecoderPoolSettings:
    """Correlated matching at its average time for the scheme's decode.

    Toshio et al. price a decode as its time per round times its rounds
    (2510.25222 p.10), an assumption the fitted time replaces
    (experiments.md, B1).
    """
    round_microseconds = CORRELATED_MATCHING_ROUND_MICROSECONDS_BY_DISTANCE[
        distance
    ]
    decode_round_count = STRONG_DECODE_DISTANCES_BY_SCHEME[scheme] * distance
    decode_microseconds = round_microseconds * decode_round_count
    return basic_case.correlated_matching_pool(f"{decode_microseconds:g}us")


def fitted_strong_time(
    machine: graph.Machine, distance: int, scheme: str
) -> graph.Machine:
    """The machine's correlated matching priced by its fitted time.

    Each decode takes the law at its window's own detection events, so a
    window in a burst is slow; over calm windows at p = 0.003 the law's
    mean is the average correlated_matching_pool charges (experiments.md,
    B1 stage 2). The answer is still correlated matching's own.
    """
    read_distances = STRONG_DECODE_DISTANCES_BY_SCHEME[scheme]
    key = (distance, read_distances)
    timing = FITTED_CORRELATED_MATCHING_TIMES[key]
    row = minimum_weight_perfect_matching.CorrelatedPyMatchingDecoder
    algorithm = row.Settings(timing=timing)
    return machine.with_settings(machines.STRONG_POOL, algorithm=algorithm)


def on_paper_values(machine: graph.Machine, distance: int) -> graph.Machine:
    """The basic case's machine on the paper's rounds and buffer."""
    with_rounds = with_paper_rounds(machine, distance)
    return with_chip_buffer(with_rounds)


def with_paper_rounds(machine: graph.Machine, distance: int) -> graph.Machine:
    """The machine with rounds of ROUND_PERIOD_MICROSECONDS.

    Its reaction deadline is one logical cycle, d rounds, which the
    reports count the windows over; it is reported, not a pass mark
    (experiments.md, decision D1).
    """
    qpu = machine.held("qpu")
    paper_rounds = dataclasses.replace(
        qpu, round_period_microseconds=ROUND_PERIOD_MICROSECONDS
    )
    with_rounds = machine.with_held("qpu", paper_rounds)
    logical_cycle_microseconds = distance * ROUND_PERIOD_MICROSECONDS
    observation = dataclasses.replace(
        with_rounds.observation,
        reaction_deadline_microseconds=logical_cycle_microseconds,
    )
    return dataclasses.replace(with_rounds, observation=observation)


def with_chip_buffer(machine: graph.Machine) -> graph.Machine:
    """The weak pool's syndrome buffer on the paper's bus, if it has one.

    One write port and one read port, on the chip's clock, each moving
    a BUFFER_WORD word every BUFFER_ACCESS, so a round of a few words is
    written in as many cycles. It is unbounded and its peak use is logged, from
    which the pilot sets its size (values.md section 2). The strong
    pool's store is the host's memory, landed by the link whose measured
    round trip holds the loading.
    """
    part_names = [part.name for part in machine.parts]
    if machines.WEAK_STORE not in part_names:
        return machine
    chip_buffer = basic_case.record(
        ported_syndrome_buffer.PortedSyndromeBufferSettings,
        clock=basic_case.constants.clocks.chip,
        word=BUFFER_WORD,
        per_access=BUFFER_ACCESS,
    )
    return machine.with_record(machines.WEAK_STORE, chip_buffer)


def recording_gap(machine: graph.Machine) -> graph.Machine:
    """A machine with no switching, its cluster gap recorded on every window.

    The gap is the one a switching run reads (the basic case's
    escalation.confidence), and the run's confidence_histogram.csv
    counts every window's gap. Its work is charged as on a switching
    run, so the run's times are no result of a decoder alone.
    """
    confidence = basic_case.constants.escalation.confidence
    gap = basic_case.record(cluster.ClusterGap.Settings, confidence)
    windows = machine.held(machines.WINDOWS)
    recording = dataclasses.replace(windows, recorded_confidence=gap)
    return machine.with_held(machines.WINDOWS, recording)


def streaming(machine: graph.Machine) -> graph.Machine:
    """The machine sampling its memory a round at a time, for decsim stream.

    The saved datasets hold whole shots, so a stream samples the same
    circuit with the flip stream source instead.
    """
    qpu = machine.held("qpu")
    source = flip_stream_device.FlipStreamDevice.Settings()
    sampled = dataclasses.replace(qpu, source=source)
    return machine.with_held("qpu", sampled)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry.

    The point's cells take the names the reports read; every other cell
    keeps its own.
    """
    named = {
        name: value for name, value in cell.items() if name not in POINT_CELLS
    }
    qpu = task.machine.held("qpu")
    return {
        **named,
        "workload.arguments.physical_error_probability": cell[
            "physical_error_probability"
        ],
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": qpu.round_period_microseconds,
    }
