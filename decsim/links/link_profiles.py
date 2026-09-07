"""The link number cards: the two reference fabrics and the optional hops.

logical_reference_profile is the default when no links card is given:
every channel unbounded, so it prices propagation only and no transfer
ever queues; latencies from Khalid et al. Table II. bandwidth_limited_
profile is the same fabric with finite calibrated rates so contention
becomes measurable; capacity_scale sweeps the whole fabric. The with_
functions add the optional store hops and a setup cost to either.

Every number carries a source string that travels into the traffic
report; paper locators are line numbers in tmp/references/papers/. To
change a number, copy a card into your own file and edit it, then pass
it as the machine's links setting.
"""

import dataclasses
from collections.abc import Mapping
from typing import Callable, Optional

import decsim.config as config
import decsim.links.settings as settings
import decsim.message as message

# A decoder result reaches the frame as one bit per logical observable, the
# logical-frame convention: Caune et al. 2410.05202 return one Boolean per
# decode, Google 2408.13687 an observable bitmask per block, and PECOS's
# frame XORs an observable mask per cycle. A decoder that feeds a physical
# frame instead emits a per-qubit or per-edge correction vector (LILLIPUT's
# error log, Helios's correction port); that is a different card. The
# window manager supplies the count from the result itself.
RESULT_PAYLOAD_SOURCE = "DecodeResult.logical_observables bits"

# A decision crosses the control fabric as one bus word: the decoder
# sequencer's 32-bit WISHBONE interface (Caune et al., arXiv 2410.05202,
# Methods). A command to the pulse controller is one instruction word:
# QubiC's distributed processor implements every instruction as a 128-bit
# word (Fruitwala et al., arXiv 2404.15260, Sec. III and IV).
BUS_WORD_BITS = 32
BUS_WORD_SOURCE = (
    "one 32-bit control bus word (Caune et al. 2410.05202, WISHBONE)"
)
INSTRUCTION_WORD_BITS = 128
INSTRUCTION_WORD_SOURCE = (
    "one 128-bit control-processor instruction word "
    "(QubiC distributed processor, Fruitwala et al. 2404.15260)"
)

ROUND_PAYLOAD_SOURCE = "SyndromeRoundPacket.fragment_size_sum"


def logical_reference_profile() -> settings.FabricSettings:
    """The default card: Khalid's latencies, unbounded bandwidth.

    Propagation only, nothing ever queues. Actual-payload paths price the
    runtime's own bit counts; default-payload paths price Khalid's Table
    II sizes.
    """
    qpu_to_controller = _actual_path(
        "qpu_to_controller",
        0.15,
        "Khalid qc effective time",
        "SyndromePayload.size_bits",
    )
    weak_buffer_to_weak_decoder = _actual_path(
        "weak_buffer_to_weak_decoder",
        2.0,
        "Khalid cd latency; logical_reference integrated weak-input transfer",
        ROUND_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = _actual_path(
        "weak_decoder_to_strong_decoder",
        0.5,
        "repository weak-to-strong model choice",
        "switching decision payload_bits",
    )
    strong_buffer_to_strong_decoder = _actual_path(
        "strong_buffer_to_strong_decoder",
        2.0,
        "Khalid cd mapped to the strong input",
        "DecodeJob.retained_payload_size_bits",
    )
    weak_decoder_to_frame = _actual_path(
        "weak_decoder_to_frame",
        1.0,
        "Khalid do latency mapped to the weak output",
        RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = _default_path(
        "decoder_to_decoder",
        0.5,
        100,
        "Khalid dd representative aggregate transaction",
    )
    strong_decoder_to_frame = _actual_path(
        "strong_decoder_to_frame",
        1.0,
        "Khalid do latency",
        RESULT_PAYLOAD_SOURCE,
    )
    frame_to_controller = _default_path(
        "frame_to_controller", 4.0, BUS_WORD_BITS, BUS_WORD_SOURCE
    )
    controller_to_qpu = _default_path(
        "controller_to_qpu",
        0.15,
        INSTRUCTION_WORD_BITS,
        INSTRUCTION_WORD_SOURCE,
    )
    return settings.FabricSettings(
        qpu_to_controller=qpu_to_controller,
        weak_buffer_to_weak_decoder=weak_buffer_to_weak_decoder,
        weak_decoder_to_strong_decoder=weak_decoder_to_strong_decoder,
        strong_buffer_to_strong_decoder=strong_buffer_to_strong_decoder,
        weak_decoder_to_frame=weak_decoder_to_frame,
        decoder_to_decoder=decoder_to_decoder,
        strong_decoder_to_frame=strong_decoder_to_frame,
        frame_to_controller=frame_to_controller,
        controller_to_qpu=controller_to_qpu,
        profile_name="logical_reference",
    )


def bandwidth_limited_profile(
    *,
    syndrome_bits_per_round: int,
    round_microseconds: float,
    commit_rounds: int,
    buffer_rounds: int,
    capacity_scale: float = 1.0,
) -> settings.FabricSettings:
    """The reference card with finite rates from the run's own geometry.

    Same latencies, paths and payload rules as logical_reference_profile,
    so switching cards changes bandwidth and nothing else. Capacity is
    bits per microsecond.

    Every path is provisioned to carry exactly its nominal traffic in one
    commit region (qc: one round's syndrome bits per round period), so at
    capacity_scale 1 each link runs at utilization one, and a scale of s
    runs it at utilization 1/s. State the utilization when reporting
    results from this card: a single-server queue at utilization one
    waits zero only under perfectly periodic arrivals, and any jitter
    accumulates (Little's law). The rates are an explicit per-link
    provisioning, the way ns-3 declares a DataRate per point-to-point
    device, never a floor borrowed from another path.
    """
    commit_region_microseconds = commit_rounds * round_microseconds
    weak_window_rounds = commit_rounds + buffer_rounds
    weak_window_bits = weak_window_rounds * syndrome_bits_per_round
    strong_window_rounds = commit_rounds + 2 * buffer_rounds
    strong_window_bits = strong_window_rounds * syndrome_bits_per_round
    one_per_region = 1 / commit_region_microseconds
    round_bits_per_microsecond = syndrome_bits_per_round / round_microseconds
    weak_window_bits_per_microsecond = (
        weak_window_bits / commit_region_microseconds
    )
    strong_window_bits_per_microsecond = (
        strong_window_bits / commit_region_microseconds
    )
    boundary_bits_per_microsecond = 100 / commit_region_microseconds
    bus_word_bits_per_microsecond = BUS_WORD_BITS / commit_region_microseconds
    instruction_word_bits_per_microsecond = (
        INSTRUCTION_WORD_BITS / commit_region_microseconds
    )
    bus_word_source = BUS_WORD_SOURCE + ", one per commit region"
    instruction_word_source = (
        INSTRUCTION_WORD_SOURCE + ", one per commit region"
    )
    provisioning = _Provisioning(capacity_scale)
    qpu_to_controller = provisioning.path(
        "qpu_to_controller",
        0.15,
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one syndrome round per round period",
        "SyndromePayload.size_bits",
    )
    weak_buffer_to_weak_decoder = provisioning.path(
        "weak_buffer_to_weak_decoder",
        2.0,
        weak_window_bits,
        weak_window_bits_per_microsecond,
        "one weak window of rcom+rbuf rounds per commit region "
        "(tmp/references/papers/2510.25222v1.txt:1150-1152, "
        '"rcom = rbuf = d")',
        ROUND_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = provisioning.path(
        "weak_decoder_to_strong_decoder",
        0.5,
        1,
        one_per_region,
        "one escalation decision per commit region",
        "switching decision payload_bits",
    )
    strong_buffer_to_strong_decoder = provisioning.path(
        "strong_buffer_to_strong_decoder",
        2.0,
        strong_window_bits,
        strong_window_bits_per_microsecond,
        "one strong window of rcom+2rbuf rounds per commit region "
        "(tmp/references/papers/2510.25222v1.txt:1155, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        "DecodeJob.retained_payload_size_bits",
    )
    weak_decoder_to_frame = provisioning.path(
        "weak_decoder_to_frame",
        1.0,
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = provisioning.path(
        "decoder_to_decoder",
        0.5,
        100,
        boundary_bits_per_microsecond,
        "one boundary transaction per commit region",
        None,
    )
    strong_decoder_to_frame = provisioning.path(
        "strong_decoder_to_frame",
        1.0,
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        RESULT_PAYLOAD_SOURCE,
    )
    frame_to_controller = provisioning.path(
        "frame_to_controller",
        4.0,
        BUS_WORD_BITS,
        bus_word_bits_per_microsecond,
        bus_word_source,
        None,
    )
    controller_to_qpu = provisioning.path(
        "controller_to_qpu",
        0.15,
        INSTRUCTION_WORD_BITS,
        instruction_word_bits_per_microsecond,
        instruction_word_source,
        None,
    )
    return settings.FabricSettings(
        qpu_to_controller=qpu_to_controller,
        weak_buffer_to_weak_decoder=weak_buffer_to_weak_decoder,
        weak_decoder_to_strong_decoder=weak_decoder_to_strong_decoder,
        strong_buffer_to_strong_decoder=strong_buffer_to_strong_decoder,
        weak_decoder_to_frame=weak_decoder_to_frame,
        decoder_to_decoder=decoder_to_decoder,
        strong_decoder_to_frame=strong_decoder_to_frame,
        frame_to_controller=frame_to_controller,
        controller_to_qpu=controller_to_qpu,
        profile_name="bandwidth_limited",
    )


def from_yaml(
    section: Mapping, clocks: config.ClockSettings, name: str
) -> settings.FabricSettings:
    """The yaml's `links` section: one card per path over the reference.

    A card prices its path in cycles of a named clock domain: latency,
    bits per cycle per lane (null is unbounded), the lane count, and an
    optional per-transfer setup cost. A null card keeps the reference
    card's numbers for that path. The two controller-to-buffer hops are
    free until a card wires them. The config prices readout
    classification on its own line, so its qpu_to_controller card is
    link propagation only, and the fabric says so.
    """
    source = f"configs/{name}.yaml links"
    path_names = []
    for path in message.LinkPath:
        path_names.append(path.value)
    for path_name in section:
        if path_name not in path_names:
            raise ValueError(
                f"links names {path_name!r}, which is not a path; the "
                f"paths are {path_names}"
            )
    profile = logical_reference_profile()
    cards = dict(section)
    weak_store_card = cards.pop("controller_to_weak_buffer", None)
    if weak_store_card is not None:
        profile = _priced_store_hop(
            profile,
            with_controller_to_weak_buffer_path,
            weak_store_card,
            clocks,
            source,
        )
    strong_store_card = cards.pop("controller_to_strong_buffer", None)
    if strong_store_card is not None:
        profile = _priced_store_hop(
            profile,
            with_controller_to_strong_buffer_path,
            strong_store_card,
            clocks,
            source,
        )
    replacements = {}
    for path_name, card in cards.items():
        if card is None:
            continue
        path_settings = getattr(profile, path_name)
        replacements[path_name] = _carded_path(
            path_name, path_settings, card, clocks, source
        )
    return dataclasses.replace(
        profile,
        **replacements,
        profile_name=f"{name}.yaml",
        is_controller_processing_outside_qpu_to_controller=True,
    )


def with_transfer_overhead(
    profile: settings.FabricSettings,
    *,
    overhead_microseconds: float,
    paths: tuple = (
        "weak_buffer_to_weak_decoder",
        "strong_buffer_to_strong_decoder",
    ),
) -> settings.FabricSettings:
    """The profile with a fixed per-transfer setup cost on the listed paths.

    The default paths are the two decoder-input DMA paths. The wire keeps
    streaming during a setup, but successive setups on one channel
    serialize (gem5-Aladdin's one delayed-DMA event).
    """
    setup_ticks = config.microseconds_to_ticks(overhead_microseconds)
    replacements = {}
    for path in paths:
        path_settings = getattr(profile, path)
        if path_settings is None:
            raise ValueError(f"{path} is not wired on this card")
        replacements[path] = dataclasses.replace(
            path_settings, setup_ticks=setup_ticks
        )
    return dataclasses.replace(
        profile,
        **replacements,
        profile_name=f"{profile.profile_name}+transfer_overhead",
    )


def with_controller_to_strong_buffer_path(
    profile: settings.FabricSettings,
    *,
    latency_microseconds: float,
    aggregate_bits_per_microsecond: Optional[float],
    source: str,
) -> settings.FabricSettings:
    """The profile with the optional priced controller_to_strong_buffer hop.

    The caller supplies both experiment-card numbers and their provenance;
    aggregate_bits_per_microsecond of None means unbounded bandwidth (the hop
    charges propagation latency only); a profile without this hop stores
    rounds in syndrome buffer 1 for free.
    """
    store_path = _store_path(
        "controller_to_strong_buffer",
        latency_microseconds,
        aggregate_bits_per_microsecond,
        source,
    )
    return dataclasses.replace(
        profile,
        controller_to_strong_buffer=store_path,
        profile_name=f"{profile.profile_name}+priced_controller_to_strong_buffer",
    )


def with_controller_to_weak_buffer_path(
    profile: settings.FabricSettings,
    *,
    latency_microseconds: float,
    aggregate_bits_per_microsecond: Optional[float],
    source: str,
) -> settings.FabricSettings:
    """The profile with the optional priced controller_to_weak_buffer hop.

    The caller supplies both experiment-card numbers and their provenance;
    aggregate_bits_per_microsecond of None means unbounded bandwidth (the hop
    charges propagation latency only); a profile without this hop
    publishes rounds to buffer 0 for free.
    """
    store_path = _store_path(
        "controller_to_weak_buffer",
        latency_microseconds,
        aggregate_bits_per_microsecond,
        source,
    )
    return dataclasses.replace(
        profile,
        controller_to_weak_buffer=store_path,
        profile_name=f"{profile.profile_name}+priced_controller_to_weak_buffer",
    )


def _card_microseconds(card: Mapping, clocks: config.ClockSettings) -> tuple:
    """(latency, aggregate rate, setup) of one card in microseconds."""
    megahertz = clocks.megahertz(card["clock"])
    latency_microseconds = card["latency_cycles"] / megahertz
    bits_per_cycle = card["bits_per_cycle"]
    lane_count = card.get("channels", 1)
    bits_per_microsecond = None
    if bits_per_cycle is not None:
        bits_per_microsecond = bits_per_cycle * lane_count * megahertz
    setup_cycles = card.get("setup_cycles_per_transfer")
    setup_microseconds = None
    if setup_cycles is not None:
        setup_microseconds = setup_cycles / megahertz
    return latency_microseconds, bits_per_microsecond, setup_microseconds


def _priced_store_hop(
    profile: settings.FabricSettings,
    with_store_path: Callable,
    card: Mapping,
    clocks: config.ClockSettings,
    source: str,
) -> settings.FabricSettings:
    latency_microseconds, bits_per_microsecond, _setup = _card_microseconds(
        card, clocks
    )
    return with_store_path(
        profile,
        latency_microseconds=latency_microseconds,
        aggregate_bits_per_microsecond=bits_per_microsecond,
        source=source,
    )


def _carded_path(
    path_name: str,
    path_settings: settings.PathSettings,
    card: Mapping,
    clocks: config.ClockSettings,
    source: str,
) -> settings.PathSettings:
    """The reference path with the card's channel and setup cost."""
    latency_microseconds, bits_per_microsecond, setup_microseconds = (
        _card_microseconds(card, clocks)
    )
    capacity = None
    if bits_per_microsecond is not None:
        capacity = settings.CapacitySettings(
            bits_per_microsecond,
            settings.QuantityBasis.AGGREGATE,
            None,
            source,
        )
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    channel = settings.ChannelSettings(
        path_name, latency_ticks, capacity, source
    )
    setup_ticks = 0
    if setup_microseconds:
        setup_ticks = config.microseconds_to_ticks(setup_microseconds)
    return dataclasses.replace(
        path_settings, channel=channel, setup_ticks=setup_ticks
    )


class _Provisioning:
    """The bounded card's paths: one channel per path at a scaled rate."""

    def __init__(self, capacity_scale: float):
        self._capacity_scale = capacity_scale

    def path(
        self,
        name: str,
        latency_microseconds: float,
        bits: int,
        nominal_bits_per_microsecond: float,
        source: str,
        actual_payload_source: Optional[str],
    ) -> settings.PathSettings:
        rate = nominal_bits_per_microsecond * self._capacity_scale
        capacity = settings.CapacitySettings(
            rate, settings.QuantityBasis.AGGREGATE, None, source
        )
        latency_ticks = config.microseconds_to_ticks(latency_microseconds)
        channel = settings.ChannelSettings(
            name, latency_ticks, capacity, source
        )
        payload = _aggregate_payload(bits, source)
        return settings.PathSettings(channel, payload, actual_payload_source)


def _aggregate_payload(bits: int, source: str) -> settings.PayloadSettings:
    return settings.PayloadSettings(
        bits, settings.QuantityBasis.AGGREGATE, None, source
    )


def _unbounded_channel(
    name: str, latency_microseconds: float, source: str
) -> settings.ChannelSettings:
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    return settings.ChannelSettings(name, latency_ticks, None, source)


def _actual_path(
    name: str,
    latency_microseconds: float,
    source: str,
    actual_payload_source: str,
) -> settings.PathSettings:
    channel = _unbounded_channel(name, latency_microseconds, source)
    return settings.PathSettings(channel, None, actual_payload_source)


def _default_path(
    name: str, latency_microseconds: float, bits: int, source: str
) -> settings.PathSettings:
    channel = _unbounded_channel(name, latency_microseconds, source)
    payload = _aggregate_payload(bits, source)
    return settings.PathSettings(channel, payload, None)


def _store_path(
    name: str,
    latency_microseconds: float,
    aggregate_bits_per_microsecond: Optional[float],
    source: str,
) -> settings.PathSettings:
    capacity = None
    if aggregate_bits_per_microsecond is not None:
        capacity = settings.CapacitySettings(
            aggregate_bits_per_microsecond,
            settings.QuantityBasis.AGGREGATE,
            None,
            source,
        )
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    channel = settings.ChannelSettings(name, latency_ticks, capacity, source)
    return settings.PathSettings(channel, None, ROUND_PAYLOAD_SOURCE)
