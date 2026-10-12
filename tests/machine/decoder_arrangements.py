"""Sixteen decoder arrangements on one machine.

Each of pymatching, union find, Relay-BP and BP-OSD as the weak tier
alone and as the strong tier alone (belief matching too), and switching
from a weak tier that reports a confidence to each strong tier: the
arrangements of the decoder experiments that ran in September 2026,
at their distance-3 task, p = 0.001 and one microsecond rounds. The
arrangements differ only in their switching part and decoder rows, so a
check that holds on one machine holds on every decoder the tree ships.
"""

import dataclasses

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.belief_propagation_osd.decoder as bposd
import decsim.decoders.relay_belief_propagation.decoder as relay_bp
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

DISTANCE = 3
PHYSICAL_ERROR_PROBABILITY = 0.001
ROUND_PERIOD_MICROSECONDS = 1.0
FRIDGE_CLOCK = clocks.FRIDGE_CLOCK
ROOM_CLOCK = clocks.ROOM_CLOCK
HOST_TIME = cycle_count.HostMeasuredTime()
# every row at its own settings, union-find charged the host's time
ALGORITHMS = {
    "pymatching": minimum_weight_perfect_matching.PyMatchingDecoder.Settings(),
    "union_find": union_find.UnionFindDecoder.Settings(timing=HOST_TIME),
    "relay_bp": relay_bp.RelayBeliefPropagationDecoder.Settings(),
    "bposd": bposd.BeliefPropagationOsdDecoder.Settings(),
    "belief_matching": belief_matching.BeliefMatchingDecoder.Settings(),
}
# the confidence each weak row reports
CONFIDENCES = {
    "pymatching": complementary.ComplementaryGap.Settings(),
    "union_find": cluster.ClusterGap.Settings(),
}


def machine() -> graph.Machine:
    """Every part but the switching part and the decoder rows.

    The weak base with every hop one cycle of its side's clock. One
    window plan for all sixteen: switching needs the lookahead tail
    (Toshio 2510.25222 Sec. III C, a strong recovery reads past the
    commit) and the single tier arrangements take the same plan so the
    comparison is fair.
    """
    weak_graph = machines.weak_baseline(
        DISTANCE, PHYSICAL_ERROR_PROBABILITY, ROUND_PERIOD_MICROSECONDS
    )
    fabric = link_cards.strong_side(weak_graph.fabric)
    base = dataclasses.replace(weak_graph, fabric=fabric)
    windows = base.held(machines.WINDOWS)
    lookahead = dataclasses.replace(windows, terminal_policy="lookahead")
    return base.with_held(machines.WINDOWS, lookahead)


def decoder_unit(kind: str, clock) -> decoder_settings.DecoderPoolSettings:
    """One tier's decoder row: one unit, its engine on the tier's clock."""
    engine = decoder_settings.EngineSettings(
        clock=clock, release_cycles_per_job=10
    )
    return decoder_settings.DecoderPoolSettings(
        algorithm=ALGORITHMS[kind], engine=engine
    )


def weak_only(kind: str) -> graph.Machine:
    """The weak tier alone, decoding every window in the fridge."""
    weak_decoder = decoder_unit(kind, FRIDGE_CLOCK)
    weak = graph.Part(machines.WEAK_POOL, weak_decoder, fabrics.FRIDGE)
    return machine().with_part(weak)


def strong_only(kind: str) -> graph.Machine:
    """The strong tier alone, decoding every window at room temperature."""
    strong_decoder = decoder_unit(kind, ROOM_CLOCK)
    return machines.strong_alone(machine(), strong_decoder)


def switching(weak_kind: str, strong_kind: str):
    """A weak tier escalating to a strong tier below 20 dB.

    The paper's 20 dB threshold (Toshio 2510.25222 Sec. IV), the
    escalated window's near face pinned on its neighbour's committed
    correction.
    """
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    strong_window = strong_window_shapes.RedoWindow.Settings()
    switching_slot = escalation_settings.SwitchingSettings(
        confidence=CONFIDENCES[weak_kind],
        threshold=threshold,
        strong_window=strong_window,
    )
    strong_decoder = decoder_unit(strong_kind, ROOM_CLOCK)
    weak = weak_only(weak_kind)
    return machines.with_strong_tier(weak, strong_decoder, switching_slot)


ARRANGEMENTS = {
    "pymatching_weak": lambda: weak_only("pymatching"),
    "union_find_weak": lambda: weak_only("union_find"),
    "relay_bp_weak": lambda: weak_only("relay_bp"),
    "bposd_weak": lambda: weak_only("bposd"),
    "pymatching_strong": lambda: strong_only("pymatching"),
    "union_find_strong": lambda: strong_only("union_find"),
    "belief_matching_strong": lambda: strong_only("belief_matching"),
    "relay_bp_strong": lambda: strong_only("relay_bp"),
    "bposd_strong": lambda: strong_only("bposd"),
    "pymatching_belief_matching_switching": lambda: switching(
        "pymatching", "belief_matching"
    ),
    "pymatching_relay_bp_switching": lambda: switching(
        "pymatching", "relay_bp"
    ),
    "pymatching_bposd_switching": lambda: switching("pymatching", "bposd"),
    "union_find_pymatching_switching": lambda: switching(
        "union_find", "pymatching"
    ),
    "union_find_belief_matching_switching": lambda: switching(
        "union_find", "belief_matching"
    ),
    "union_find_relay_bp_switching": lambda: switching(
        "union_find", "relay_bp"
    ),
    "union_find_bposd_switching": lambda: switching("union_find", "bposd"),
}
