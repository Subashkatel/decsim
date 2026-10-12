"""The data-movement study's machines: four blocks on the weak baseline.

    machine("every_hop_copies", distance=3)

1. Every hop copies: the weak unit decodes a copy of its rounds and
   folds the boundary into a duplicate of that copy.
2. The weak input read in place, where the weak store keeps it.
3. The boundary folded in place, into the unit's own copy.
4. The strong tier and its links: switching at 20 dB.

Both tiers are Toshio et al.'s linear decoder on priced cards; every
number is in constants.yaml beside this file.
"""

import dataclasses
import pathlib

import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
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


def machine(block: str, distance: int) -> graph.Machine:
    """One block at one distance."""
    make = BLOCKS[block]
    return make(distance)


def every_hop_copies(distance: int) -> graph.Machine:
    """Block 1: the weak tier on copies, every copy counted and traced."""
    probability = constants.physical_error_probability
    period = constants.round_period_microseconds
    base = machines.weak_baseline(distance, probability, period)
    weak_decoder = linear_pool(
        constants.weak_decode_microseconds_per_round, clocks.FRIDGE_CLOCK, 1
    )
    observation = dataclasses.replace(
        base.observation, trace="chrome", data_movement=True
    )
    weak_pool = graph.Part(machines.WEAK_POOL, weak_decoder, fabrics.FRIDGE)
    with_pool = base.with_part(weak_pool)
    fabric = weak_tier_fabric()
    return dataclasses.replace(
        with_pool, fabric=fabric, observation=observation
    )


def weak_input_in_place(distance: int) -> graph.Machine:
    """Block 2: the weak unit reads its rounds where the store keeps them."""
    copies = every_hop_copies(distance)
    return copies.with_settings(machines.WEAK_POOL, copies_input=False)


def boundary_folded_in_place(distance: int) -> graph.Machine:
    """Block 3: the boundary is XORed into the unit's own copy."""
    copies = every_hop_copies(distance)
    return copies.with_settings(machines.WEAK_POOL, copies_boundary_fold=False)


def switching(distance: int) -> graph.Machine:
    """Block 4: weak first, escalating below 20 dB to the strong tier.

    The complementary gap decodes each weak window twice.
    """
    copies = every_hop_copies(distance)
    weak_decoder = linear_pool(
        constants.weak_decode_microseconds_per_round, clocks.FRIDGE_CLOCK, 2
    )
    strong_decoder = linear_pool(
        constants.strong_decode_microseconds_per_round, clocks.ROOM_CLOCK, 1
    )
    weak_pool = graph.Part(machines.WEAK_POOL, weak_decoder, fabrics.FRIDGE)
    weak_tier = copies.with_part(weak_pool)
    strong_link = machine_file.record(
        fabrics.ClockedLink, constants.strong_link
    )
    base_fabric = weak_tier_fabric()
    fabric = link_cards.strong_side(base_fabric, strong_link)
    threshold = {"type": "FixedThreshold", "threshold": constants.threshold}
    edge = machine_file.record(
        escalation_settings.SwitchingSettings,
        confidence={"type": "complementary_gap"},
        threshold=threshold,
    )
    return machines.with_strong_tier(weak_tier, strong_decoder, edge, fabric)


def linear_pool(
    microseconds_per_round: float, clock: object, solves_per_window: int
) -> decoder_settings.DecoderPoolSettings:
    """Toshio's linear decoder: tau_dec a round of each window decoded."""
    return decoder_settings.linear_decoder_pool(
        microseconds_per_round, clock, solves_per_window=solves_per_window
    )


def weak_tier_fabric() -> fabrics.PathLinks:
    """The weak base's hops, the controller's write a round period long."""
    weak_link = machine_file.record(fabrics.ClockedLink, constants.weak_link)
    return fabrics.PathLinks(
        controller_to_weak_buffer=weak_link,
        base=link_cards.WEAK_BASELINE_FABRIC,
    )


BLOCKS = {
    "every_hop_copies": every_hop_copies,
    "weak_input_in_place": weak_input_in_place,
    "boundary_folded_in_place": boundary_folded_in_place,
    "switching": switching,
}
