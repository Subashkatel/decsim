"""Links that price a wire, and fabric rules that pick a link per route.

A route is one hop a transfer takes between two parts, with the site of
each end: the fridge, the host, or any site a part is placed at. A link
prices a route: it puts a
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
    SharedChannel  any link, its routes riding one named channel

    ReferenceFabric  the reference card on every route
    RoleLinks        a link by the route's role, else a base rule
    FridgeHostFabric a cable between the sites, a host link on the host,
                     else a base rule
    SiteLinks        a link by the route's pair of sites, else a base rule

Transfers contend only where they ride one channel. A device is a site
whose links are shared: every hop into or out of it rides one channel
each way, so its pools wait for one another there, as the components
on one SST Link share it (sst-core src/sst/core/link.h) and ns-3's net
devices on one channel share its wire.

A link times a transfer as its latency plus its bits over its rate,
ns-3's point-to-point law (point-to-point-net-device.cc:243); the rule
by sites is qecboard's FridgeHostFabric (notes/qecboard_prototype,
components/fabrics.py), whose prices reproduced Experiment 1's card.
"""

import dataclasses
import fractions
from typing import Optional

import decsim.config as config
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.ports as ports
import decsim.records.routes as route_records

FRIDGE = route_records.FRIDGE
HOST = route_records.HOST


@dataclasses.dataclass(frozen=True)
class Route:
    """One hop between two parts: its path's name, each end's site, its role.

    role is what the hop carries (records/routes.py HopRole): the
    reference card prices the hop by it, and a fabric may pick its link
    by it.
    """

    path: str
    source_site: str
    target_site: str
    role: route_records.HopRole


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
class SharedChannel:
    """Any link, every route it prices riding one channel of this name.

    The routes then wait for one another on it in both directions, a
    half-duplex wire; SiteLinks shares a full-duplex one a channel each
    way.
    """

    link: ports.LinkCard
    channel: str

    def priced(
        self,
        name: str,
        reference: link_settings.PathSettings,
        machine_clock: Optional[config.Clock],
    ) -> link_settings.PathSettings:
        """The link's path, on the shared channel instead of its own."""
        del name
        return self.link.priced(self.channel, reference, machine_clock)


@dataclasses.dataclass(frozen=True)
class ReferenceFabric:
    """Every route on the reference card (link_profiles.py)."""

    def link_for(self, route: Route) -> Optional[ports.LinkCard]:
        """No link: the reference path."""
        del route
        return None


@dataclasses.dataclass(frozen=True)
class RoleLinks:
    """A link per role (records/routes.py HopRole.key); any other, the base's.

    A write or a read is split by how its store takes its rounds, an
    answer or a same-pool hop by the pool's site, so a base that prices
    the weak chain on one clock and the strong side on another names
    each by its role.
    """

    readout: Optional[ports.LinkCard] = None
    write_written: Optional[ports.LinkCard] = None
    write_landed: Optional[ports.LinkCard] = None
    read_written: Optional[ports.LinkCard] = None
    read_landed: Optional[ports.LinkCard] = None
    forward: Optional[ports.LinkCard] = None
    reverse: Optional[ports.LinkCard] = None
    same_pool_fridge: Optional[ports.LinkCard] = None
    same_pool_host: Optional[ports.LinkCard] = None
    answer_fridge: Optional[ports.LinkCard] = None
    answer_host: Optional[ports.LinkCard] = None
    decision: Optional[ports.LinkCard] = None
    command: Optional[ports.LinkCard] = None
    base: ports.FabricRule = ReferenceFabric()

    def link_for(self, route: Route) -> Optional[ports.LinkCard]:
        """This role's link, else the base's."""
        link = getattr(self, route.role.key)
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

    cable: ports.LinkCard
    host_local: ports.LinkCard
    base: ports.FabricRule = ReferenceFabric()

    def link_for(self, route: Route) -> Optional[ports.LinkCard]:
        """The cable, the host link, or the base's link for the route."""
        sites = {route.source_site, route.target_site}
        if sites == {FRIDGE, HOST}:
            return self.cable
        if sites == {HOST}:
            return self.host_local
        return self.base.link_for(route)


@dataclasses.dataclass(frozen=True)
class SiteLink:
    """The link between two sites, or inside one when both are the same.

    sites is a pair, in either order. shared puts every hop between the
    two sites on one channel each way, named by the sites, so the parts
    at a device's site contend for its link (a GPU's PCIe, an FPGA's
    serial lanes); unshared, each hop rides a channel of its own.
    """

    sites: tuple
    link: ports.LinkCard
    shared: bool = False

    def __post_init__(self) -> None:
        if len(self.sites) != 2:
            raise ValueError(
                f"a site link joins two sites, or one site to itself; got "
                f"{self.sites}"
            )

    def joins(self, route: Route) -> bool:
        """Whether the route runs between this link's two sites."""
        ends = {route.source_site, route.target_site}
        return ends == set(self.sites)

    def link_for(self, route: Route) -> ports.LinkCard:
        """The link, on the shared channel of the route's way if shared."""
        if not self.shared:
            return self.link
        channel = f"{route.source_site}_to_{route.target_site}"
        return SharedChannel(self.link, channel)


@dataclasses.dataclass(frozen=True)
class SiteLinks:
    """A link per pair of sites; a route between no listed pair, the base's.

    A co-design is a graph of parts at sites joined by these links, as
    gem5-SALAM's accelerator cluster is components and the links between
    them: an ASIC predecoder in the fridge and a GPU in a rack, a cable
    between them and a PCIe link into the GPU. Two entries for one pair
    are refused, since a route would take whichever came first.
    """

    links: tuple = ()
    base: ports.FabricRule = ReferenceFabric()

    def __post_init__(self) -> None:
        pairs = []
        for site_link in self.links:
            pair = frozenset(site_link.sites)
            if pair in pairs:
                raise ValueError(
                    f"two site links join {sorted(pair)}; give each pair "
                    "of sites one link"
                )
            pairs.append(pair)

    def link_for(self, route: Route) -> Optional[ports.LinkCard]:
        """The link of the route's pair of sites, else the base's."""
        for site_link in self.links:
            if site_link.joins(route):
                return site_link.link_for(route)
        return self.base.link_for(route)


def card(
    rule: ports.FabricRule,
    routes: tuple,
    wire_links: dict,
    machine_clock: Optional[config.Clock],
    readout_routes: tuple = (),
) -> link_settings.FabricSettings:
    """The card from a rule: each route's path, a wire's link first.

    wire_links holds the link a wire carries, by its route's path name;
    a route with no link of its own takes the rule's, and one the rule
    gives none the reference path of its role. readout_routes ride
    beside the readout's own path.
    """
    paths = []
    for route in routes:
        priced = _priced(route, rule, wire_links, machine_clock)
        path = link_settings.NamedPath(
            route.path,
            priced,
            route.role,
            route.source_site,
            route.target_site,
        )
        paths.append(path)
    return link_settings.FabricSettings(tuple(paths), readout_routes)


def _priced(
    route: Route,
    rule: ports.FabricRule,
    wire_links: dict,
    machine_clock: Optional[config.Clock],
) -> link_settings.PathSettings:
    """One route's path: its wire's link, else the rule's, else the reference.

    Every hop rides a channel of its own name.
    """
    reference_path = link_profiles.reference_path(route.role, route.path)
    link = wire_links.get(route.path)
    if link is None:
        link = rule.link_for(route)
    if link is None:
        return reference_path
    return link.priced(route.path, reference_path, machine_clock)
