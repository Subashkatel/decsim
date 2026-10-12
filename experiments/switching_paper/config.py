"""Toshio et al.'s Fig. 15 machines: one window a shot, three decodings.

    machine("weak_alone", "complementary_gap", 5, 0.002)
    machine("strong_alone", "complementary_gap", 5, 0.002)
    machine("switching", "cluster_gap", 5, 0.002)

The memory is Stim's rotated surface code under SD6 at p (Deltakit's
row, frontends/deltakit.py SD6Noise), d rounds a shot with its data
prepared and read out without noise (Sec. IV, lines 1407-1412). Each shot is one window, as the paper
decodes offline: the base's sliding windows commit d rounds, so a
d-round shot is a single window under either terminal policy
(windows/schemes/sliding.py). The weak decoder is
PyMatching with the complementary gap, or union-find with the cluster
gap; the strong decoder is belief matching. Every number is in
constants.yaml beside this file.
"""

import dataclasses
import pathlib

import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.frontends.deltakit as deltakit
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.topology.constants as constants_module
import decsim.topology.graph as graph
import decsim.topology.machine_file as machine_file

HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)
# the weak decoder that computes each signal (lines 1393-1397); union-find
# is charged the host's measured time, which offline decoding does not read
# Deltakit's SD6 at each memory's p (Gidney et al. 2108.10457, Table 2);
# the paper names only a uniform circuit-level noise model (line 1682)
SD6 = deltakit.SD6Noise()
HOST_TIME = cycle_count.HostMeasuredTime()
WEAK_ALGORITHMS = {
    "complementary_gap": matching.PyMatchingDecoder.Settings(),
    "cluster_gap": union_find.UnionFindDecoder.Settings(timing=HOST_TIME),
}


def machine(
    decoding: str,
    signal: str,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """One cell: weak alone, strong alone, or switching on signal."""
    weak = weak_alone(signal, distance, physical_error_probability)
    if decoding == "weak_alone":
        return weak
    strong_pool = belief_matching_pool()
    if decoding == "strong_alone":
        return machines.strong_alone(weak, strong_pool)
    return switching(weak, signal, strong_pool)


def weak_alone(
    signal: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """The weak decoder on the paper's memory, one d-round window a shot."""
    period = constants.round_period_microseconds
    base = machines.weak_baseline(distance, physical_error_probability, period)
    memory = producers.MemorySettings(
        physical_error_probability,
        rounds_per_distance=1,
        noise=SD6,
        noiseless_time_boundaries=True,
    )
    paper = base.with_held("workload", memory)
    algorithm = WEAK_ALGORITHMS[signal]
    return paper.with_settings(machines.WEAK_POOL, algorithm=algorithm)


def belief_matching_pool():
    """Belief matching on the strong base's host unit."""
    iterations = constants.belief_propagation_iterations
    algorithm = belief_matching.BeliefMatchingDecoder.Settings(
        max_iterations=iterations, belief_propagation_method="product_sum"
    )
    pool = machines.STRONG_BASELINE_POOL
    return dataclasses.replace(pool, algorithm=algorithm)


def switching(
    weak: graph.Machine, signal: str, strong_pool
) -> graph.Machine:
    """The weak machine switching to belief matching below the threshold."""
    threshold = {"type": "FixedThreshold", "threshold": constants.threshold}
    confidence = {"type": signal}
    edge = machine_file.record(
        escalation_settings.SwitchingSettings,
        confidence=confidence,
        threshold=threshold,
    )
    fabric = link_cards.strong_side(weak.fabric, link_cards.ONE_ROOM_CYCLE)
    return machines.with_strong_tier(weak, strong_pool, edge, fabric)
