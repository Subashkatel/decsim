"""Links that price a wire, and fabric rules that pick a link per route.

A route is one hop a transfer takes between two parts, with the site of
each end, the fridge or the host. A link prices a route: it puts a
channel, a latency and a rate, on the route's own payload rule, the
reference card's (links/link_profiles.py), so what a hop carries stays
with the hop and only its wire changes. A wire may carry its own link;
every other route takes the link the machine's fabric rule picks for
it, or the reference card's path where the rule picks none.

    ClockedLink    latency in cycles of a clock, lanes of bits a cycle
    TimedLink      latency in time, a rate or an unbounded wire
    RoundTripLeg   one leg of a measured round trip: half of it, less
                   the echo payload's time on the cable
    FixedPath      a path written out whole, a packet protocol and all

    ReferenceFabric  the reference card on every route
    PathLinks        a link by the route's path name, else a base rule
    FridgeHostFabric a cable between the sites, a host link on the host,
                     else a base rule
    CardFabric       today's card written out, a path per route

A link times a transfer as its latency plus its bits over its rate,
ns-3's point-to-point law (point-to-point-net-device.cc:243); the rule
by sites is qecboard's FridgeHostFabric (notes/qecboard_prototype,
components/fabrics.py), whose prices reproduced Experiment 1's card.
"""

import dataclasses
import fractions
from typing import Optional, Protocol

import decsim.config as config
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings

FRIDGE = "fridge"
HOST = "host"
# the name a card built from a graph goes by; a label, no part of an id
PROFILE_NAME = "a graph's fabric"
# Each path of today's card, by the MachineSettings slots at its ends.
# decoder_to_decoder carries a window's boundary to the next window on
# the pool that decodes the windows.
LEGACY_ENDS = {
    "qpu_to_controller": ("qpu", "controller"),
    "controller_to_weak_buffer": ("controller", "weak_syndrome_buffer"),
    "weak_buffer_to_weak_decoder": ("weak_syndrome_buffer", "weak_decoder"),
    "weak_decoder_to_strong_decoder": ("weak_decoder", "strong_decoder"),
    "strong_buffer_to_strong_decoder": (
        "strong_syndrome_buffer",
        "strong_decoder",
    ),
    "weak_decoder_to_frame": ("weak_decoder", "pauli_frame"),
    "decoder_to_decoder": ("weak_decoder", "weak_decoder"),
    "strong_decoder_to_frame": ("strong_decoder", "pauli_frame"),
    "frame_to_controller": ("pauli_frame", "controller"),
    "controller_to_qpu": ("controller", "qpu"),
    "controller_to_strong_buffer": ("controller", "strong_syndrome_buffer"),
    "strong_decoder_to_weak_decoder": ("strong_decoder", "weak_decoder"),
}
# Where each slot's part sits when its part says nothing: the controller,
# the frame and the weak tier on the chip in the fridge, the strong tier
# on the host, as the reference card prices them.
LEGACY_SITES = {
    "qpu": FRIDGE,
    "controller": FRIDGE,
    "pauli_frame": FRIDGE,
    "weak_syndrome_buffer": FRIDGE,
    "weak_decoder": FRIDGE,
    "strong_syndrome_buffer": HOST,
    "strong_decoder": HOST,
}


class LinkCard(Protocol):
    """A wire's price: a route's path from its reference path."""

    def priced(
        self,
        name: str,
        reference: link_settings.PathSettings,
        machine_clock: Optional[config.Clock],
    ) -> link_settings.PathSettings:
        """The path named name: the reference's payload rule on this wire."""


@dataclasses.dataclass(frozen=True)
class Route:
    """One hop between two parts: its path's name and each end's site."""

    path: str
    source_site: Optional[str]
    target_site: Optional[str]


class FabricRule(Protocol):
    """A fabric: the link each route takes, None for the reference path."""

    def link_for(self, route: Route) -> Optional[LinkCard]:
        """The link this route takes, or None for the reference path."""


@dataclasses.dataclass(frozen=True)
class ClockedLink:
    """A wire in cycles of a clock: a latency, and lanes of bits a cycle.

    A cycle is the clock's period in whole ticks, gem5's cyclesToTicks
    (src/sim/clocked_object.hh:227), so a link's cycles and every other
    part's cycles on one clock are the same ticks; clock None counts on
    the machine's. Each of lane_count lanes moves bits_per_cycle each
    cycle, None an unbounded wire. excludes_receiver_processing says the
    latency times the wire alone, so the receiver prices its own work.
    """

    clock: Optional[config.Clock] = None
    latency_cycles: int = 0
    bits_per_cycle: Optional[float] = None
    lane_count: int = 1
    setup_cycles_per_transfer: int = 0
    header_bits_per_transfer: int = 0
    excludes_receiver_processing: bool = True
    # where the numbers come from, a label: no part of a task's id
    source: str = dataclasses.field(default="", compare=False)

    def priced(
        self,
        name: str,
        reference: link_settings.PathSettings,
        machine_clock: Optional[config.Clock],
    ) -> link_settings.PathSettings:
        """The path named name: the reference's payload on this wire."""
        clock = self.clock or machine_clock
        period_ticks = clock.period_ticks
        latency_ticks = self.latency_cycles * period_ticks
        setup_ticks = self.setup_cycles_per_transfer * period_ticks
        capacity = self._capacity(period_ticks)
        channel = link_settings.ChannelSettings(
            name, latency_ticks, capacity, self.source
        )
        return dataclasses.replace(
            reference,
            channel=channel,
            setup_ticks=setup_ticks,
            header_bits_per_transfer=self.header_bits_per_transfer,
            excludes_receiver_processing=self.excludes_receiver_processing,
        )

    def _capacity(
        self, period_ticks: int
    ) -> Optional[link_settings.CapacitySettings]:
        """Every lane's bits a cycle, in bits per microsecond, exactly."""
        if self.bits_per_cycle is None:
            return None
        lane_bits_per_cycle = fractions.Fraction(str(self.bits_per_cycle))
        bits_per_period = lane_bits_per_cycle * self.lane_count
        bits_per_tick = bits_per_period / period_ticks
        bits_per_microsecond = bits_per_tick * config.TICKS_PER_MICROSECOND
        return link_settings.CapacitySettings(bits_per_microsecond, self.source)


@dataclasses.dataclass(frozen=True)
class TimedLink:
    """A wire in time: a latency, and a rate or none for an unbounded wire.

    The latency is rounded to whole ticks once. A measured number is end
    to end, so by default it holds the receiver's processing too.
    """

    latency_microseconds: float = 0.0
    capacity_bits_per_microsecond: Optional[float] = None
    excludes_receiver_processing: bool = False
    # where the numbers come from, a label: no part of a task's id
    source: str = dataclasses.field(default="", compare=False)

    def priced(
        self,
        name: str,
        reference: link_settings.PathSettings,
        machine_clock: Optional[config.Clock],
    ) -> link_settings.PathSettings:
        """The path named name: the reference's payload on this wire."""
        del machine_clock
        latency_ticks = config.microseconds_to_ticks(self.latency_microseconds)
        capacity = self._capacity()
        channel = link_settings.ChannelSettings(
            name, latency_ticks, capacity, self.source
        )
        return dataclasses.replace(
            reference,
            channel=channel,
            setup_ticks=0,
            header_bits_per_transfer=0,
            excludes_receiver_processing=self.excludes_receiver_processing,
        )

    def _capacity(self) -> Optional[link_settings.CapacitySettings]:
        if self.capacity_bits_per_microsecond is None:
            return None
        return link_settings.CapacitySettings(
            self.capacity_bits_per_microsecond, self.source
        )


@dataclasses.dataclass(frozen=True)
class RoundTripLeg:
    """One leg of a measured round trip over a cable.

    A round trip measured as one number (Backline 2609.09270 Table III,
    NVQLink 2510.25213 line 485) gives no per-direction split, so each
    leg is half of it, less the echo payload's time on the cable, which
    every transfer pays for its own bits: an echo then takes the
    measured round trip exactly.
    """

    round_trip_microseconds: float
    echo_payload_bits: int
    capacity_bits_per_microsecond: float
    # where the numbers come from, a label: no part of a task's id
    source: str = dataclasses.field(default="", compare=False)

    def priced(
        self,
        name: str,
        reference: link_settings.PathSettings,
        machine_clock: Optional[config.Clock],
    ) -> link_settings.PathSettings:
        """The path named name: the reference's payload on one leg."""
        leg = self.leg()
        return leg.priced(name, reference, machine_clock)

    def leg(self) -> TimedLink:
        """The leg as a timed wire on the cable."""
        rate = fractions.Fraction(str(self.capacity_bits_per_microsecond))
        echo_fraction = self.echo_payload_bits / rate
        echo_on_the_cable = float(echo_fraction)
        half_microseconds = self.round_trip_microseconds / 2
        leg_microseconds = half_microseconds - echo_on_the_cable
        return TimedLink(
            leg_microseconds,
            self.capacity_bits_per_microsecond,
            False,
            self.source,
        )


@dataclasses.dataclass(frozen=True)
class FixedPath:
    """A path written out whole, as today's cards write each one.

    It is the path as given, its channel's name and a packet protocol
    included: the link for a wire no other link describes.
    """

    path: link_settings.PathSettings

    def priced(
        self,
        name: str,
        reference: link_settings.PathSettings,
        machine_clock: Optional[config.Clock],
    ) -> link_settings.PathSettings:
        """The path as written."""
        del name, reference, machine_clock
        return self.path


@dataclasses.dataclass(frozen=True)
class ReferenceFabric:
    """Every route on the reference card (link_profiles.py)."""

    def link_for(self, route: Route) -> Optional[LinkCard]:
        """No link: the reference path."""
        del route
        return None


@dataclasses.dataclass(frozen=True)
class PathLinks:
    """A link per path, by the path's name; any other route, the base's."""

    qpu_to_controller: Optional[LinkCard] = None
    controller_to_weak_buffer: Optional[LinkCard] = None
    weak_buffer_to_weak_decoder: Optional[LinkCard] = None
    weak_decoder_to_strong_decoder: Optional[LinkCard] = None
    strong_buffer_to_strong_decoder: Optional[LinkCard] = None
    weak_decoder_to_frame: Optional[LinkCard] = None
    decoder_to_decoder: Optional[LinkCard] = None
    strong_decoder_to_frame: Optional[LinkCard] = None
    frame_to_controller: Optional[LinkCard] = None
    controller_to_qpu: Optional[LinkCard] = None
    controller_to_strong_buffer: Optional[LinkCard] = None
    strong_decoder_to_weak_decoder: Optional[LinkCard] = None
    base: FabricRule = ReferenceFabric()

    def link_for(self, route: Route) -> Optional[LinkCard]:
        """This path's link, else the base's."""
        link = getattr(self, route.path)
        if link is not None:
            return link
        return self.base.link_for(route)


@dataclasses.dataclass(frozen=True)
class FridgeHostFabric:
    """A cable between the fridge and the host, a host link on the host.

    A route whose ends sit on different sites crosses the cable, and one
    with both ends on the host takes the host's own link; a route inside
    the fridge takes the base's.
    """

    cable: LinkCard
    host_local: LinkCard
    base: FabricRule = ReferenceFabric()

    def link_for(self, route: Route) -> Optional[LinkCard]:
        """The cable, the host link, or the base's link for the route."""
        sites = {route.source_site, route.target_site}
        if sites == {FRIDGE, HOST}:
            return self.cable
        if sites == {HOST}:
            return self.host_local
        return self.base.link_for(route)


@dataclasses.dataclass(frozen=True)
class CardFabric:
    """Today's card written out: each route takes the card's own path."""

    card: link_settings.FabricSettings

    def link_for(self, route: Route) -> Optional[LinkCard]:
        """The card's path for the route, as written."""
        path = getattr(self.card, route.path)
        return FixedPath(path)


def card(
    rule: FabricRule,
    routes: tuple,
    wire_links: dict,
    machine_clock: Optional[config.Clock],
) -> link_settings.FabricSettings:
    """Today's card from a rule: each route's path, a wire's link first.

    wire_links holds the link a wire carries, by its route's path name;
    a route with no link of its own takes the rule's, and one the rule
    gives none the reference path.
    """
    reference = link_profiles.logical_reference_profile()
    paths = {}
    for route in routes:
        reference_path = getattr(reference, route.path)
        link = wire_links.get(route.path)
        if link is None:
            link = rule.link_for(route)
        paths[route.path] = reference_path
        if link is not None:
            paths[route.path] = link.priced(
                route.path, reference_path, machine_clock
            )
    return dataclasses.replace(reference, **paths, profile_name=PROFILE_NAME)


def legacy_routes(sites: dict) -> tuple:
    """Every path of today's card as a route, its ends at these sites.

    sites maps a slot to its part's site; a slot it leaves out, or maps
    to None, sits where LEGACY_SITES puts it.
    """
    routes = []
    for path, (source_slot, target_slot) in LEGACY_ENDS.items():
        source_site = _site(sites, source_slot)
        target_site = _site(sites, target_slot)
        route = Route(path, source_site, target_site)
        routes.append(route)
    return tuple(routes)


def _site(sites: dict, slot: str) -> Optional[str]:
    site = sites.get(slot)
    if site is None:
        return LEGACY_SITES.get(slot)
    return site
