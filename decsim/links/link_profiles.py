"""The link number cards: the two reference fabrics and one optional edge.

``logical_reference_profile`` is the default when ``RunSpec.links`` is unset:
every channel unbounded, so it prices propagation only and no transfer ever
queues; latencies from Khalid et al. Table II. ``bandwidth_limited_profile``
is the same fabric with finite calibrated rates so contention becomes
measurable; ``capacity_scale`` sweeps the whole fabric.
``with_controller_to_buffer_edge`` adds the priced CWB hop to either.

Every number carries a ``source`` string that travels into the topology and
traffic reports; paper locators are line numbers in tmp/references/papers/.
To change a number, copy a card into your own file and edit it, then pass it
as ``RunSpec(links=...)``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..config import microseconds_to_ticks
from .links import (
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkQuantityBasis,
    PayloadSizeConfig,
    TransferOverheadConfig,
)


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
BUS_WORD_SOURCE = "one 32-bit control bus word (Caune et al. 2410.05202, WISHBONE)"
INSTRUCTION_WORD_BITS = 128
INSTRUCTION_WORD_SOURCE = ("one 128-bit control-processor instruction word "
                           "(QubiC distributed processor, Fruitwala et al. 2404.15260)")


def _aggregate_payload(bits: int, source: str) -> PayloadSizeConfig:
    return PayloadSizeConfig(bits, LinkQuantityBasis.DIRECT_AGGREGATE, None, source)


def logical_reference_profile() -> LinkModelConfig:
    """The default card: Khalid's latencies, unbounded bandwidth (propagation
    only, nothing ever queues). Actual-payload edges price the runtime's own
    bit counts; default-payload edges price Khalid's Table II sizes."""
    def unbounded(latency_us: float, source: str) -> LinkConfig:
        return LinkConfig(microseconds_to_ticks(latency_us), None, source)

    def actual_edge(latency_us: float, source: str, actual: str) -> LinkEdgeConfig:
        return LinkEdgeConfig(unbounded(latency_us, source), None, actual)

    def default_edge(latency_us: float, payload: PayloadSizeConfig) -> LinkEdgeConfig:
        return LinkEdgeConfig(unbounded(latency_us, payload.source), payload, None)

    return LinkModelConfig(
        qc=actual_edge(0.15, "Khalid qc effective time", "SyndromePayload.size_bits"),
        wbd=actual_edge(2.0, "Khalid cd latency; logical_reference integrated weak-input transfer",
                        "SyndromeRoundPacket.fragment_size_sum"),
        wsd=actual_edge(0.5, "repository weak-to-strong model choice", "switching decision payload_bits"),
        sbd=actual_edge(2.0, "Khalid cd mapped to the strong input", "DecodeJob.retained_payload_size_bits"),
        wdo=actual_edge(1.0, "Khalid do latency mapped to the weak output",
                        RESULT_PAYLOAD_SOURCE),
        dd=default_edge(0.5, _aggregate_payload(100, "Khalid dd representative aggregate transaction")),
        do=actual_edge(1.0, "Khalid do latency", RESULT_PAYLOAD_SOURCE),
        oc=default_edge(4.0, _aggregate_payload(BUS_WORD_BITS, BUS_WORD_SOURCE)),
        cq=default_edge(0.15, _aggregate_payload(INSTRUCTION_WORD_BITS,
                                                 INSTRUCTION_WORD_SOURCE)),
        profile_name="logical_reference",
    )


def bandwidth_limited_profile(*, syndrome_bits_per_round: int, round_us: float,
                              commit_rounds: int, buffer_rounds: int,
                              capacity_scale: float = 1.0) -> LinkModelConfig:
    """The reference card with finite rates provisioned from the run's own
    geometry: same latencies, paths and payload rules as
    logical_reference_profile, so switching cards changes bandwidth and
    nothing else. Capacity is bits per microsecond.

    Every path is provisioned to carry exactly its nominal traffic in one
    commit region (qc: one round's syndrome bits per round period), so at
    ``capacity_scale`` 1 each link runs at utilization one, and a scale of
    s runs it at utilization 1/s. State the utilization when reporting
    results from this card: a single-server queue at utilization one waits
    zero only under perfectly periodic arrivals, and any jitter
    accumulates (Little's law). The rates are an explicit per-link
    provisioning, the way ns-3 declares a DataRate per point-to-point
    device, never a floor borrowed from another path.
    """
    commit_region_us = commit_rounds * round_us
    weak_window_bits = (commit_rounds + buffer_rounds) * syndrome_bits_per_round
    strong_window_bits = (commit_rounds + 2 * buffer_rounds) * syndrome_bits_per_round

    def aggregate_edge(latency_us: float, bits: int, nominal_bits_per_us: float,
                       source: str, actual_payload_source) -> LinkEdgeConfig:
        rate = nominal_bits_per_us * capacity_scale
        capacity = LinkCapacityConfig(rate, LinkQuantityBasis.DIRECT_AGGREGATE, None, source)
        channel = LinkConfig(microseconds_to_ticks(latency_us), capacity, source)
        return LinkEdgeConfig(channel, _aggregate_payload(bits, source), actual_payload_source)

    return LinkModelConfig(
        qc=aggregate_edge(
            0.15,
            syndrome_bits_per_round,
            syndrome_bits_per_round / round_us,
            "one syndrome round per round period",
            "SyndromePayload.size_bits",
        ),
        wbd=aggregate_edge(
            2.0,
            weak_window_bits,
            weak_window_bits / commit_region_us,
            "one weak window of rcom+rbuf rounds per commit region "
            "(tmp/references/papers/2510.25222v1.txt:1150-1152, "
            "\"rcom = rbuf = d\")",
            "SyndromeRoundPacket.fragment_size_sum",
        ),
        wsd=aggregate_edge(
            0.5,
            1,
            1 / commit_region_us,
            "one escalation decision per commit region",
            "switching decision payload_bits",
        ),
        sbd=aggregate_edge(
            2.0,
            strong_window_bits,
            strong_window_bits / commit_region_us,
            "one strong window of rcom+2rbuf rounds per commit region "
            "(tmp/references/papers/2510.25222v1.txt:1155, "
            "\"In this paper, we assume that rstrong = rcom + 2rbuf.\")",
            "DecodeJob.retained_payload_size_bits",
        ),
        wdo=aggregate_edge(
            1.0, 1, 1 / commit_region_us,
            "one frame-update bit per logical observable per commit region",
            RESULT_PAYLOAD_SOURCE,
        ),
        dd=aggregate_edge(
            0.5,
            100,
            100 / commit_region_us,
            "one boundary transaction per commit region",
            None,
        ),
        do=aggregate_edge(
            1.0, 1, 1 / commit_region_us,
            "one frame-update bit per logical observable per commit region",
            RESULT_PAYLOAD_SOURCE,
        ),
        oc=aggregate_edge(
            4.0, BUS_WORD_BITS, BUS_WORD_BITS / commit_region_us,
            BUS_WORD_SOURCE + ", one per commit region",
            None,
        ),
        cq=aggregate_edge(
            0.15, INSTRUCTION_WORD_BITS, INSTRUCTION_WORD_BITS / commit_region_us,
            INSTRUCTION_WORD_SOURCE + ", one per commit region",
            None,
        ),
        profile_name="bandwidth_limited",
    )


def with_transfer_overhead(
    profile: LinkModelConfig,
    *,
    overhead_us: float,
    source: str,
    paths: tuple = ("wbd", "sbd"),
) -> LinkModelConfig:
    """Return ``profile`` with a fixed per-transfer setup cost on the listed
    paths (default: the two decoder-input DMA paths).

    Engine-side: the wire keeps streaming during a setup, but successive
    setups on one channel serialize (gem5-Aladdin's one delayed-DMA event;
    see ``TransferOverheadConfig`` for the measured numbers), so paths
    sharing a channel reach its wire in request order.
    """
    overhead = TransferOverheadConfig(microseconds_to_ticks(overhead_us), source)
    replacements = {}
    for path in paths:
        edge = getattr(profile, path)
        if edge is None:
            raise ValueError(f"{path} is not wired on this card")
        replacements[path] = replace(edge, transfer_overhead=overhead)
    return replace(
        profile,
        **replacements,
        profile_name=f"{profile.profile_name}+transfer_overhead",
    )


def with_csb_edge(
    profile: LinkModelConfig,
    *,
    latency_us: float,
    aggregate_bits_per_us: Optional[float],
    source: str,
) -> LinkModelConfig:
    """Return ``profile`` with the optional priced csb edge to syndrome
    buffer 1.

    The caller supplies both experiment-card numbers and their provenance;
    ``aggregate_bits_per_us`` of ``None`` means unbounded bandwidth (the edge
    charges propagation latency only); a profile without this edge stores
    rounds in syndrome buffer 1 for free.
    """
    capacity = None
    if aggregate_bits_per_us is not None:
        capacity = LinkCapacityConfig(
            aggregate_bits_per_us,
            LinkQuantityBasis.DIRECT_AGGREGATE,
            None,
            source,
        )
    channel = LinkConfig(microseconds_to_ticks(latency_us), capacity, source)
    edge = LinkEdgeConfig(
        channel,
        None,
        "SyndromeRoundPacket.fragment_size_sum",
    )
    return replace(
        profile,
        csb=edge,
        profile_name=f"{profile.profile_name}+priced_csb",
    )


def with_controller_to_buffer_edge(
    profile: LinkModelConfig,
    *,
    latency_us: float,
    aggregate_bits_per_us: Optional[float],
    source: str,
) -> LinkModelConfig:
    """Return ``profile`` with the optional priced CWB round-transfer edge.

    The caller supplies both experiment-card numbers and their provenance;
    ``aggregate_bits_per_us`` of ``None`` means unbounded bandwidth (the edge
    charges propagation latency only); a profile without this edge publishes
    rounds to Buffer 0 for free.
    """
    capacity = None
    if aggregate_bits_per_us is not None:
        capacity = LinkCapacityConfig(
            aggregate_bits_per_us,
            LinkQuantityBasis.DIRECT_AGGREGATE,
            None,
            source,
        )
    channel = LinkConfig(microseconds_to_ticks(latency_us), capacity, source)
    edge = LinkEdgeConfig(
        channel,
        None,
        "SyndromeRoundPacket.fragment_size_sum",
    )
    return replace(
        profile,
        cwb=edge,
        profile_name=f"{profile.profile_name}+priced_cwb",
    )
