"""The switching experiment's machines: two weak tiers, each escalating.

    machine("cluster_gap", distance=3)
    machine("redo_window", distance=5)

A weak result is kept at 20 dB or more and any other window is redone
on the redo window, the strong decode starting after the weak verdict,
which is free. Every number is in constants.yaml beside this file.
"""

import pathlib

import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.ports as ports
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.topology.constants as constants_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.machine_file as machine_file

HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)


def machine(configuration: str, distance: int) -> graph.Machine:
    """One configuration at one distance."""
    make = CONFIGURATIONS[configuration]
    return make(distance)


def cluster_gap_switching(distance: int) -> graph.Machine:
    """Union-find's cluster gap decides; Toshio's strong decoder redoes.

    The union-find decode is charged the host's measured time, and the
    escalation's rounds arrive T_strong_comm after it sends them.
    """
    base = weak_base(distance)
    host_time = cycle_count.HostMeasuredTime()
    union_find_decoder = union_find.UnionFindDecoder.Settings(timing=host_time)
    weak = base.with_settings(machines.WEAK_POOL, algorithm=union_find_decoder)
    per_round = constants.strong_decode_microseconds_per_round
    strong = decoder_settings.linear_decoder_pool(
        per_round, clocks.ROOM_CLOCK, solves_per_window=1
    )
    cluster_gap = {"type": "cluster_gap", **constants.cluster_gap}
    link = machine_file.record(fabrics.ClockedLink, constants.escalation_link)
    return switching(weak, cluster_gap, strong, link)


def redo_window_switching(distance: int) -> graph.Machine:
    """PyMatching's complementary gap decides; the strong base redoes."""
    base = weak_base(distance)
    pymatching = matching.PyMatchingDecoder.Settings()
    weak = base.with_settings(machines.WEAK_POOL, algorithm=pymatching)
    strong = machines.STRONG_BASELINE_POOL
    complementary_gap = {"type": "complementary_gap"}
    return switching(weak, complementary_gap, strong)


def weak_base(distance: int) -> graph.Machine:
    """The weak decoder baseline at this experiment's rate and period."""
    probability = constants.physical_error_probability
    period = constants.round_period_microseconds
    return machines.weak_baseline(distance, probability, period)


def switching(
    weak: graph.Machine,
    confidence: dict,
    strong: decoder_settings.DecoderPoolSettings,
    escalation: ports.LinkCard = link_cards.ONE_ROOM_CYCLE,
) -> graph.Machine:
    """The weak machine switching on confidence to a strong pool on the host.

    The strong side's hops are one room cycle each, the escalation as
    given.
    """
    threshold = {"type": "FixedThreshold", "threshold": constants.threshold}
    edge = machine_file.record(
        escalation_settings.SwitchingSettings,
        confidence=confidence,
        threshold=threshold,
    )
    fabric = link_cards.strong_side(weak.fabric, escalation)
    return machines.with_strong_tier(weak, strong, edge, fabric)


CONFIGURATIONS = {
    "cluster_gap": cluster_gap_switching,
    "redo_window": redo_window_switching,
}
