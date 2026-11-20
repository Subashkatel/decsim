"""Toshio et al.'s two tiers (2510.25222) on the weak baseline, for tests.

    redo_window_switching(distance=3)
    cluster_gap_switching(distance=3)
    weak_input_in_place(distance=3)

The two switching machines keep a weak result at 20 dB or more and redo
any other window on the redo window, the strong decode starting after
the weak verdict, which is free. They run at p = 0.008 and a one
microsecond round. The data path machines are the paper's linear
decoder on priced cards at p = 0.003, the weak unit decoding a copy of
its rounds or reading them where the store keeps them. The regression
digests (tests/regression) pin a shot of each.
"""

import dataclasses

import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.ports as ports
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.machine_file as machine_file

SWITCHING_ERROR_PROBABILITY = 0.008
DATA_PATH_ERROR_PROBABILITY = 0.003
# the paper's tau_gen
ROUND_PERIOD_MICROSECONDS = 1.0
# tau_weak_dec = 0.4 tau_gen, one swept value (2510.25222 lines 1125-1126)
WEAK_DECODE_MICROSECONDS_PER_ROUND = 0.4
# tau_strong_dec = 10 tau_gen (2510.25222 lines 1109-1114)
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10.0
# T_weak_comm = tau_gen, 1 us, on the fridge clock (lines 1109-1114)
WEAK_LINK = {
    "clock": "250MHz",
    "latency": "1us",
    "source": "T_weak_comm, 2510.25222 lines 1109-1114",
}
# T_strong_comm = 10 tau_gen, 10 us, on the room clock (lines 1109-1114)
ESCALATION_LINK = {
    "clock": "250MHz",
    "latency": "10us",
    "source": "T_strong_comm, 2510.25222 lines 1109-1114",
}
# the keep threshold of 2510.25222 Sec. IV
THRESHOLD = "20dB"
CLUSTER_GAP_WALK = "12us"


def cluster_gap_switching(distance: int) -> graph.Machine:
    """Union-find's cluster gap decides; Toshio's strong decoder redoes.

    The union-find decode is charged the host's measured time, and the
    escalation's rounds arrive T_strong_comm after it sends them.
    """
    base = _switching_weak_base(distance)
    host_time = cycle_count.HostMeasuredTime()
    union_find_decoder = union_find.UnionFindDecoder.Settings(timing=host_time)
    weak = base.with_settings(machines.WEAK_POOL, algorithm=union_find_decoder)
    strong = decoder_settings.linear_decoder_pool(
        STRONG_DECODE_MICROSECONDS_PER_ROUND,
        clocks.ROOM_CLOCK,
        solves_per_window=1,
    )
    cluster_gap = {"type": "cluster_gap", "walk": CLUSTER_GAP_WALK}
    link = machine_file.record(fabrics.ClockedLink, ESCALATION_LINK)
    return _switching(weak, cluster_gap, strong, link)


def redo_window_switching(distance: int) -> graph.Machine:
    """PyMatching's complementary gap decides; the strong base redoes."""
    base = _switching_weak_base(distance)
    pymatching = matching.PyMatchingDecoder.Settings()
    weak = base.with_settings(machines.WEAK_POOL, algorithm=pymatching)
    strong = machines.STRONG_BASELINE_POOL
    complementary_gap = {"type": "complementary_gap"}
    return _switching(weak, complementary_gap, strong)


def every_hop_copies(distance: int) -> graph.Machine:
    """The weak tier on copies, every copy counted and traced."""
    base = machines.weak_baseline(
        distance, DATA_PATH_ERROR_PROBABILITY, ROUND_PERIOD_MICROSECONDS
    )
    weak_decoder = decoder_settings.linear_decoder_pool(
        WEAK_DECODE_MICROSECONDS_PER_ROUND,
        clocks.FRIDGE_CLOCK,
        solves_per_window=1,
    )
    observation = dataclasses.replace(
        base.observation, trace="chrome", data_movement=True
    )
    with_pool = base.with_record(machines.WEAK_POOL, weak_decoder)
    weak_link = machine_file.record(fabrics.ClockedLink, WEAK_LINK)
    fabric = fabrics.RoleLinks(
        write_written=weak_link, base=link_cards.WEAK_BASELINE_FABRIC
    )
    return dataclasses.replace(
        with_pool, fabric=fabric, observation=observation
    )


def weak_input_in_place(distance: int) -> graph.Machine:
    """The weak unit reads its rounds where the store keeps them."""
    copies = every_hop_copies(distance)
    return copies.with_settings(machines.WEAK_POOL, copies_input=False)


def _switching_weak_base(distance: int) -> graph.Machine:
    return machines.weak_baseline(
        distance, SWITCHING_ERROR_PROBABILITY, ROUND_PERIOD_MICROSECONDS
    )


def _switching(
    weak: graph.Machine,
    confidence: dict,
    strong: decoder_settings.DecoderPoolSettings,
    escalation: ports.LinkCard = link_cards.ONE_ROOM_CYCLE,
) -> graph.Machine:
    """The weak machine switching on confidence to a strong pool on the host.

    The strong side's hops are one room cycle each, the escalation as
    given.
    """
    threshold = {"type": "FixedThreshold", "threshold": THRESHOLD}
    edge = machine_file.record(
        escalation_settings.SwitchingSettings,
        confidence=confidence,
        threshold=threshold,
    )
    fabric = link_cards.strong_side(weak.fabric, escalation)
    return machines.with_strong_tier(weak, strong, edge, fabric)
