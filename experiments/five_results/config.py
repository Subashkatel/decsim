"""The five results' machines: the basic case's, on the paper's values.

    alone("weak_alone", 9, 0.003)
    alone("strong_alone", 9, 0.003)
    switching("double", 9, 0.003, threshold_decibels=12.0,
              strong_unit_count=2)
    recording_gap(alone("weak_alone", 9, 0.003))
    streaming(alone("weak_alone", 9, 0.003))

The basic case's machine (experiments/basic_case/config.py) with the
values of specs/adaptive-switching/values.md where they differ: rounds of
1.1 us, and the strong decoder correlated matching on one core of a CPU
host, over Backline's RoCE v2 link, at the paper's average time per
round. A fixed time has no tail and does not slow down in a burst, so it
serves the pilots at steady noise only (experiments.md, decision D4).
"""

import dataclasses

import decsim
import decsim.confidence.cluster as cluster
import decsim.decoders.settings as decoder_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.topology.graph as graph

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
# the rounds one strong decode reads, in code distances: a sliding
# window's d committed and d looked ahead (values.md section 6), also
# the window speculation redoes; the double window's r_com + 2 r_buf
# (Toshio et al. 2510.25222 line 1250)
STRONG_DECODE_DISTANCES_BY_SCHEME = {
    "sliding": 2,
    "speculation": 2,
    "double": 3,
}
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
    return with_paper_rounds(basic, distance)


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
    return with_paper_rounds(thresholded, distance)


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
