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

from ..config import us
from .links import (
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkQuantityBasis,
    PayloadSizeConfig,
)


def logical_reference_profile() -> LinkModelConfig:
    """Return the default fabric card: propagation only, no finite bandwidth."""
    def channel(latency_us: float, source: str) -> LinkConfig:
        return LinkConfig(us(latency_us), None, source)

    def actual_edge(latency_us: float, source: str, actual: str):
        return LinkEdgeConfig(channel(latency_us, source), None, actual)

    def direct_default(latency_us: float, bits: int, source: str):
        return LinkEdgeConfig(
            channel(latency_us, source),
            PayloadSizeConfig(
                bits,
                LinkQuantityBasis.DIRECT_AGGREGATE,
                None,
                source,
            ),
            None,
        )

    def per_channel_default(
        latency_us: float,
        bits: int,
        count: int,
        source: str,
    ):
        return LinkEdgeConfig(
            channel(latency_us, source),
            PayloadSizeConfig(
                bits,
                LinkQuantityBasis.PER_CHANNEL,
                count,
                source,
            ),
            None,
        )

    return LinkModelConfig(
        qc=actual_edge(0.15, "Khalid qc effective time", "SyndromePayload.size_bits"),
        cwd=actual_edge(
            2.0,
            "Khalid cd latency; logical_reference integrated weak-input transfer",
            "SyndromeRoundPacket.fragment_size_sum",
        ),
        wsd=actual_edge(
            0.5,
            "repository weak-to-strong model choice",
            "switching decision payload_bits",
        ),
        csd=actual_edge(
            2.0,
            "Khalid cd mapped to controller-to-strong",
            "DecodeJob.retained_payload_size_bits",
        ),
        wdo=per_channel_default(
            1.0,
            50_000,
            100,
            "Khalid do mapped to weak output",
        ),
        dd=direct_default(
            0.5,
            100,
            "Khalid dd representative aggregate transaction",
        ),
        do=per_channel_default(1.0, 50_000, 100, "Khalid do"),
        oc=per_channel_default(4.0, 20_000, 1000, "Khalid oc"),
        cq=per_channel_default(0.15, 1, 5_000_000, "Khalid cq"),
        profile_name="logical_reference",
    )


def bandwidth_limited_profile(*, capacity_scale: float = 1.0) -> LinkModelConfig:
    """Return the reference fabric with finite, calibrated channel rates.

    Propagation latencies, path mapping and the configured default payloads
    match logical_reference_profile, so switching profiles changes bandwidth
    and nothing else. Capacity is bits per microsecond, which equals Mbps.

    The calibration point is one distance-5 surface-code logical qubit whose
    syndrome round is modelled as 24 bits, produced once per 1.0 us, that is
    24 Mbps. Both figures are REPOSITORY MODELLING CHOICES with no cited
    source: no reference here states a bit width for a distance-5 syndrome
    round, and none uses a 1.0 us period.

    What the references do fix is the setting and the envelope. The
    distance-5 surface code with an integrated real-time decoder is
    tmp/references/papers/2408.13687v1.txt:68-71, and its reported cadence is
    tmp/references/papers/2408.13687v1.txt:87, "fast 1.1 µs cycle duration.",
    which is also the decsim default (config.py TimingConfig.round_us = 1.1),
    so the 1.0 us used here is a round number slightly faster than the
    reported cadence and is conservative in the direction of more bandwidth
    per round than a run consumes.
    tmp/references/papers/2303.00054.txt:141-143 reports that "QEC rounds were performed every ∼1 µs"
    and estimates "a few tens of Mbps of syndrome data" per logical qubit, the
    envelope the 24 Mbps anchor sits inside.

    Each channel carries its nominal transfer at the cadence that transfer
    occurs and no channel is provisioned below that 24 Mbps anchor.
    capacity_scale multiplies every channel, so sweeping it moves the whole
    fabric through its contention regimes.
    """
    syndrome_bits_per_round = 24
    round_us = 1.0
    commit_rounds = 5
    buffer_rounds = 5
    commit_region_us = commit_rounds * round_us
    weak_window_bits = (
        (commit_rounds + buffer_rounds) * syndrome_bits_per_round
    )
    strong_window_bits = (
        (commit_rounds + 2 * buffer_rounds) * syndrome_bits_per_round
    )
    anchor_bits_per_us = syndrome_bits_per_round / round_us

    def aggregate_edge(
        latency_us: float,
        bits: int,
        nominal_bits_per_us: float,
        source: str,
        actual_payload_source,
    ):
        capacity = LinkCapacityConfig(
            max(anchor_bits_per_us, nominal_bits_per_us) * capacity_scale,
            LinkQuantityBasis.DIRECT_AGGREGATE,
            None,
            source,
        )
        return LinkEdgeConfig(
            LinkConfig(us(latency_us), capacity, source),
            PayloadSizeConfig(
                bits,
                LinkQuantityBasis.DIRECT_AGGREGATE,
                None,
                source,
            ),
            actual_payload_source,
        )

    def per_channel_edge(
        latency_us: float,
        bits: int,
        count: int,
        source: str,
    ):
        capacity = LinkCapacityConfig(
            max(anchor_bits_per_us / count, bits / commit_region_us)
            * capacity_scale,
            LinkQuantityBasis.PER_CHANNEL,
            count,
            source,
        )
        return LinkEdgeConfig(
            LinkConfig(us(latency_us), capacity, source),
            PayloadSizeConfig(
                bits,
                LinkQuantityBasis.PER_CHANNEL,
                count,
                source,
            ),
            None,
        )

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
    aggregate_bits_per_us: float,
    source: str,
) -> LinkModelConfig:
    """Return ``profile`` with the optional priced C2B round-transfer edge.

    The caller supplies both experiment-card numbers and their provenance;
    a profile without this edge publishes rounds to Buffer 0 for free.
    """
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
