"""The link number cards: the two reference fabrics and one optional edge.

``logical_reference_profile`` is the default when ``RunSpec.links`` is unset:
every channel unbounded, so it prices propagation only and no transfer ever
queues; latencies from Khalid et al. Table II. ``bandwidth_limited_profile``
is the same fabric with finite calibrated rates so contention becomes
measurable; ``capacity_scale`` sweeps the whole fabric.
``with_controller_to_buffer_edge`` adds the priced C2B hop to either.

Every number carries a ``source`` string that travels into the topology and
traffic reports; paper locators are line numbers in tmp/references/papers/.
To change a number, copy a card into your own file and edit it, then pass it
as ``RunSpec(links=...)``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..config import us
from .links import (
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkQuantityBasis,
    PayloadSizeConfig,
)


def _aggregate_payload(bits: int, source: str) -> PayloadSizeConfig:
    return PayloadSizeConfig(bits, LinkQuantityBasis.DIRECT_AGGREGATE, None, source)


def _per_channel_payload(bits: int, count: int, source: str) -> PayloadSizeConfig:
    return PayloadSizeConfig(bits, LinkQuantityBasis.PER_CHANNEL, count, source)


def logical_reference_profile() -> LinkModelConfig:
    """The default card: Khalid's latencies, unbounded bandwidth (propagation
    only, nothing ever queues). Actual-payload edges price the runtime's own
    bit counts; default-payload edges price Khalid's Table II sizes."""
    def unbounded(latency_us: float, source: str) -> LinkConfig:
        return LinkConfig(us(latency_us), None, source)

    def actual_edge(latency_us: float, source: str, actual: str) -> LinkEdgeConfig:
        return LinkEdgeConfig(unbounded(latency_us, source), None, actual)

    def default_edge(latency_us: float, payload: PayloadSizeConfig) -> LinkEdgeConfig:
        return LinkEdgeConfig(unbounded(latency_us, payload.source), payload, None)

    return LinkModelConfig(
        qc=actual_edge(0.15, "Khalid qc effective time", "SyndromePayload.size_bits"),
        cwd=actual_edge(2.0, "Khalid cd latency; logical_reference integrated weak-input transfer",
                        "SyndromeRoundPacket.fragment_size_sum"),
        wsd=actual_edge(0.5, "repository weak-to-strong model choice", "switching decision payload_bits"),
        csd=actual_edge(2.0, "Khalid cd mapped to controller-to-strong", "DecodeJob.retained_payload_size_bits"),
        wdo=default_edge(1.0, _per_channel_payload(50_000, 100, "Khalid do mapped to weak output")),
        dd=default_edge(0.5, _aggregate_payload(100, "Khalid dd representative aggregate transaction")),
        do=default_edge(1.0, _per_channel_payload(50_000, 100, "Khalid do")),
        oc=default_edge(4.0, _per_channel_payload(20_000, 1000, "Khalid oc")),
        cq=default_edge(0.15, _per_channel_payload(1, 5_000_000, "Khalid cq")),
        profile_name="logical_reference",
    )


def bandwidth_limited_profile(*, capacity_scale: float = 1.0) -> LinkModelConfig:
    """The reference card with finite, calibrated rates: same latencies, paths
    and default payloads as logical_reference_profile, so switching cards
    changes bandwidth and nothing else. Capacity is bits per microsecond.

    Calibration point: one distance-5 patch, 24 syndrome bits per 1.0 us
    round, 24 bits/us; a repository modelling choice (no reference states the
    bit width; the setting is 2408.13687v1.txt:68-71 with a 1.1 us cycle at
    :87, and 2303.00054.txt:141-143 puts syndrome data at "a few tens of
    Mbps" per logical qubit). Each channel carries its nominal transfer once
    per commit region and no channel is provisioned below the 24 bits/us
    anchor. ``capacity_scale`` multiplies every channel.
    """
    syndrome_bits_per_round = 24
    round_us = 1.0
    commit_rounds = 5
    buffer_rounds = 5
    commit_region_us = commit_rounds * round_us
    weak_window_bits = (commit_rounds + buffer_rounds) * syndrome_bits_per_round
    strong_window_bits = (commit_rounds + 2 * buffer_rounds) * syndrome_bits_per_round
    anchor_bits_per_us = syndrome_bits_per_round / round_us

    def aggregate_edge(latency_us: float, bits: int, nominal_bits_per_us: float,
                       source: str, actual_payload_source) -> LinkEdgeConfig:
        rate = max(anchor_bits_per_us, nominal_bits_per_us) * capacity_scale
        capacity = LinkCapacityConfig(rate, LinkQuantityBasis.DIRECT_AGGREGATE, None, source)
        channel = LinkConfig(us(latency_us), capacity, source)
        return LinkEdgeConfig(channel, _aggregate_payload(bits, source), actual_payload_source)

    def per_channel_edge(latency_us: float, bits: int, count: int, source: str) -> LinkEdgeConfig:
        rate = max(anchor_bits_per_us / count, bits / commit_region_us) * capacity_scale
        capacity = LinkCapacityConfig(rate, LinkQuantityBasis.PER_CHANNEL, count, source)
        channel = LinkConfig(us(latency_us), capacity, source)
        return LinkEdgeConfig(channel, _per_channel_payload(bits, count, source), None)

    return LinkModelConfig(
        qc=aggregate_edge(
            0.15,
            syndrome_bits_per_round,
            syndrome_bits_per_round / round_us,
            "one 24-bit distance-5 syndrome round per 1.0 us calibration "
            "round period; both figures are repository modelling choices "
            "with no cited source. Setting: "
            "tmp/references/papers/2408.13687v1.txt:68-71. Reported cadence: "
            "tmp/references/papers/2408.13687v1.txt:87 \"fast 1.1 µs cycle duration.\". "
            "Envelope: tmp/references/papers/2303.00054.txt:141-143 "
            "\"QEC rounds were performed every ∼1 µs\", \"a few tens of Mbps of syndrome data\" per logical qubit",
            "SyndromePayload.size_bits",
        ),
        cwd=aggregate_edge(
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
            "one escalation decision per commit region, floored at the "
            "24 Mbps syndrome anchor",
            "switching decision payload_bits",
        ),
        csd=aggregate_edge(
            2.0,
            strong_window_bits,
            strong_window_bits / commit_region_us,
            "one strong window of rcom+2rbuf rounds per commit region "
            "(tmp/references/papers/2510.25222v1.txt:1155, "
            "\"In this paper, we assume that rstrong = rcom + 2rbuf.\")",
            "DecodeJob.retained_payload_size_bits",
        ),
        wdo=per_channel_edge(
            1.0, 50_000, 100,
            "one weak decoder output payload per commit region",
        ),
        dd=aggregate_edge(
            0.5,
            100,
            100 / commit_region_us,
            "one boundary transaction per commit region, floored at the "
            "24 Mbps syndrome anchor",
            None,
        ),
        do=per_channel_edge(
            1.0, 50_000, 100,
            "one decoder output payload per commit region",
        ),
        oc=per_channel_edge(
            4.0, 20_000, 1000,
            "one output-to-controller payload per commit region",
        ),
        cq=per_channel_edge(
            0.15, 1, 5_000_000,
            "one controller-to-QPU payload per commit region",
        ),
        profile_name="bandwidth_limited",
    )


def with_controller_to_buffer_edge(
    profile: LinkModelConfig,
    *,
    latency_us: float,
    aggregate_bits_per_us: Optional[float],
    source: str,
) -> LinkModelConfig:
    """Return ``profile`` with the optional priced C2B round-transfer edge.

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
    channel = LinkConfig(us(latency_us), capacity, source)
    edge = LinkEdgeConfig(
        channel,
        None,
        "SyndromeRoundPacket.fragment_size_sum",
    )
    return replace(
        profile,
        c2b=edge,
        profile_name=f"{profile.profile_name}+priced_c2b",
    )
