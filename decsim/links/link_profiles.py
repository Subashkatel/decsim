"""The link number cards: the two reference fabrics and the optional hops.

logical_reference_profile is the default when RunSpec.links is unset:
every channel unbounded, so it prices propagation only and no transfer
ever queues; latencies from Khalid et al. Table II. bandwidth_limited_
profile is the same fabric with finite calibrated rates so contention
becomes measurable; capacity_scale sweeps the whole fabric. The with_
functions add the optional store hops and a setup cost to either.

Every number carries a source string that travels into the traffic
report; paper locators are line numbers in tmp/references/papers/. To
change a number, copy a card into your own file and edit it, then pass
it as RunSpec(links=...).
"""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.links.settings as settings

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
        "qc", 0.15, "Khalid qc effective time", "SyndromePayload.size_bits"
    )
    weak_buffer_to_weak_decoder = _actual_path(
        "wbd",
        2.0,
        "Khalid cd latency; logical_reference integrated weak-input transfer",
        ROUND_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = _actual_path(
        "wsd",
        0.5,
        "repository weak-to-strong model choice",
        "switching decision payload_bits",
    )
    strong_buffer_to_strong_decoder = _actual_path(
        "sbd",
        2.0,
        "Khalid cd mapped to the strong input",
        "DecodeJob.retained_payload_size_bits",
    )
    weak_decoder_to_frame = _actual_path(
        "wdo",
        1.0,
        "Khalid do latency mapped to the weak output",
        RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = _default_path(
        "dd", 0.5, 100, "Khalid dd representative aggregate transaction"
    )
    strong_decoder_to_frame = _actual_path(
        "do", 1.0, "Khalid do latency", RESULT_PAYLOAD_SOURCE
    )
    frame_to_controller = _default_path(
        "oc", 4.0, BUS_WORD_BITS, BUS_WORD_SOURCE
    )
    controller_to_qpu = _default_path(
        "cq", 0.15, INSTRUCTION_WORD_BITS, INSTRUCTION_WORD_SOURCE
    )
    return settings.FabricSettings(
        qc=qpu_to_controller,
        wbd=weak_buffer_to_weak_decoder,
        wsd=weak_decoder_to_strong_decoder,
        sbd=strong_buffer_to_strong_decoder,
        wdo=weak_decoder_to_frame,
        dd=decoder_to_decoder,
        do=strong_decoder_to_frame,
        oc=frame_to_controller,
        cq=controller_to_qpu,
        profile_name="logical_reference",
    )


def bandwidth_limited_profile(
    *,
    syndrome_bits_per_round: int,
    round_us: float,
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
    commit_region_us = commit_rounds * round_us
    weak_window_rounds = commit_rounds + buffer_rounds
    weak_window_bits = weak_window_rounds * syndrome_bits_per_round
    strong_window_rounds = commit_rounds + 2 * buffer_rounds
    strong_window_bits = strong_window_rounds * syndrome_bits_per_round
    one_per_region = 1 / commit_region_us
    round_bits_per_us = syndrome_bits_per_round / round_us
    weak_window_bits_per_us = weak_window_bits / commit_region_us
    strong_window_bits_per_us = strong_window_bits / commit_region_us
    boundary_bits_per_us = 100 / commit_region_us
    bus_word_bits_per_us = BUS_WORD_BITS / commit_region_us
    instruction_word_bits_per_us = INSTRUCTION_WORD_BITS / commit_region_us
    bus_word_source = BUS_WORD_SOURCE + ", one per commit region"
    instruction_word_source = (
        INSTRUCTION_WORD_SOURCE + ", one per commit region"
    )
    provisioning = _Provisioning(capacity_scale)
    qpu_to_controller = provisioning.path(
        "qc",
        0.15,
        syndrome_bits_per_round,
        round_bits_per_us,
        "one syndrome round per round period",
        "SyndromePayload.size_bits",
    )
    weak_buffer_to_weak_decoder = provisioning.path(
        "wbd",
        2.0,
        weak_window_bits,
        weak_window_bits_per_us,
        "one weak window of rcom+rbuf rounds per commit region "
        "(tmp/references/papers/2510.25222v1.txt:1150-1152, "
        '"rcom = rbuf = d")',
        ROUND_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = provisioning.path(
        "wsd",
        0.5,
        1,
        one_per_region,
        "one escalation decision per commit region",
        "switching decision payload_bits",
    )
    strong_buffer_to_strong_decoder = provisioning.path(
        "sbd",
        2.0,
        strong_window_bits,
        strong_window_bits_per_us,
        "one strong window of rcom+2rbuf rounds per commit region "
        "(tmp/references/papers/2510.25222v1.txt:1155, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        "DecodeJob.retained_payload_size_bits",
    )
    weak_decoder_to_frame = provisioning.path(
        "wdo",
        1.0,
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = provisioning.path(
        "dd",
        0.5,
        100,
        boundary_bits_per_us,
        "one boundary transaction per commit region",
        None,
    )
    strong_decoder_to_frame = provisioning.path(
        "do",
        1.0,
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        RESULT_PAYLOAD_SOURCE,
    )
    frame_to_controller = provisioning.path(
        "oc",
        4.0,
        BUS_WORD_BITS,
        bus_word_bits_per_us,
        bus_word_source,
        None,
    )
    controller_to_qpu = provisioning.path(
        "cq",
        0.15,
        INSTRUCTION_WORD_BITS,
        instruction_word_bits_per_us,
        instruction_word_source,
        None,
    )
    return settings.FabricSettings(
        qc=qpu_to_controller,
        wbd=weak_buffer_to_weak_decoder,
        wsd=weak_decoder_to_strong_decoder,
        sbd=strong_buffer_to_strong_decoder,
        wdo=weak_decoder_to_frame,
        dd=decoder_to_decoder,
        do=strong_decoder_to_frame,
        oc=frame_to_controller,
        cq=controller_to_qpu,
        profile_name="bandwidth_limited",
    )


def with_transfer_overhead(
    profile: settings.FabricSettings,
    *,
    overhead_us: float,
    paths: tuple = ("wbd", "sbd"),
) -> settings.FabricSettings:
    """The profile with a fixed per-transfer setup cost on the listed paths.

    The default paths are the two decoder-input DMA paths. The wire keeps
    streaming during a setup, but successive setups on one channel
    serialize (gem5-Aladdin's one delayed-DMA event).
    """
    setup_ticks = config.microseconds_to_ticks(overhead_us)
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


def with_csb_edge(
    profile: settings.FabricSettings,
    *,
    latency_us: float,
    aggregate_bits_per_us: Optional[float],
    source: str,
) -> settings.FabricSettings:
    """The profile with the optional priced csb hop to syndrome buffer 1.

    The caller supplies both experiment-card numbers and their provenance;
    aggregate_bits_per_us of None means unbounded bandwidth (the hop
    charges propagation latency only); a profile without this hop stores
    rounds in syndrome buffer 1 for free.
    """
    store_path = _store_path("csb", latency_us, aggregate_bits_per_us, source)
    return dataclasses.replace(
        profile,
        csb=store_path,
        profile_name=f"{profile.profile_name}+priced_csb",
    )


def with_controller_to_buffer_edge(
    profile: settings.FabricSettings,
    *,
    latency_us: float,
    aggregate_bits_per_us: Optional[float],
    source: str,
) -> settings.FabricSettings:
    """The profile with the optional priced cwb hop to syndrome buffer 0.

    The caller supplies both experiment-card numbers and their provenance;
    aggregate_bits_per_us of None means unbounded bandwidth (the hop
    charges propagation latency only); a profile without this hop
    publishes rounds to buffer 0 for free.
    """
    store_path = _store_path("cwb", latency_us, aggregate_bits_per_us, source)
    return dataclasses.replace(
        profile,
        cwb=store_path,
        profile_name=f"{profile.profile_name}+priced_cwb",
    )


class _Provisioning:
    """The bounded card's paths: one channel per path at a scaled rate."""

    def __init__(self, capacity_scale: float):
        self._capacity_scale = capacity_scale

    def path(
        self,
        name: str,
        latency_us: float,
        bits: int,
        nominal_bits_per_us: float,
        source: str,
        actual_payload_source: Optional[str],
    ) -> settings.PathSettings:
        rate = nominal_bits_per_us * self._capacity_scale
        capacity = settings.CapacitySettings(
            rate, settings.QuantityBasis.AGGREGATE, None, source
        )
        latency_ticks = config.microseconds_to_ticks(latency_us)
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
    name: str, latency_us: float, source: str
) -> settings.ChannelSettings:
    latency_ticks = config.microseconds_to_ticks(latency_us)
    return settings.ChannelSettings(name, latency_ticks, None, source)


def _actual_path(
    name: str, latency_us: float, source: str, actual_payload_source: str
) -> settings.PathSettings:
    channel = _unbounded_channel(name, latency_us, source)
    return settings.PathSettings(channel, None, actual_payload_source)


def _default_path(
    name: str, latency_us: float, bits: int, source: str
) -> settings.PathSettings:
    channel = _unbounded_channel(name, latency_us, source)
    payload = _aggregate_payload(bits, source)
    return settings.PathSettings(channel, payload, None)


def _store_path(
    name: str,
    latency_us: float,
    aggregate_bits_per_us: Optional[float],
    source: str,
) -> settings.PathSettings:
    capacity = None
    if aggregate_bits_per_us is not None:
        capacity = settings.CapacitySettings(
            aggregate_bits_per_us,
            settings.QuantityBasis.AGGREGATE,
            None,
            source,
        )
    latency_ticks = config.microseconds_to_ticks(latency_us)
    channel = settings.ChannelSettings(name, latency_ticks, capacity, source)
    return settings.PathSettings(channel, None, ROUND_PAYLOAD_SOURCE)
