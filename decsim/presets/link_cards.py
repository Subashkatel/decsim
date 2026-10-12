"""The link cards and fabrics the presets and the shipped studies use.

Every number carries its source on the card it sets. A fabric prices
each hop no wire's own link prices (decsim/topology/fabrics.py); a
card here is a link for one hop or a fabric for the whole machine.
"""

import dataclasses
import fractions
from typing import Optional

import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.ports as ports
import decsim.presets.clocks as clocks
import decsim.records.windows as window_records
import decsim.topology.fabrics as fabrics

# An idealized wire: one cycle of the hop's clock, unbounded.
_ONE_CYCLE_SOURCE = "one cycle of the hop's clock, an idealized wire"
ONE_FRIDGE_CYCLE = fabrics.ClockedLink(
    clock=clocks.FRIDGE_CLOCK, latency_cycles=1, source=_ONE_CYCLE_SOURCE
)
ONE_ROOM_CYCLE = fabrics.ClockedLink(
    clock=clocks.ROOM_CLOCK, latency_cycles=1, source=_ONE_CYCLE_SOURCE
)
# The weak base's hops: the weak chain, the decision and the pulse, each
# one fridge cycle. The strong side keeps the reference card.
WEAK_BASELINE_FABRIC = fabrics.PathLinks(
    qpu_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_weak_buffer=ONE_FRIDGE_CYCLE,
    weak_buffer_to_weak_decoder=ONE_FRIDGE_CYCLE,
    decoder_to_decoder=ONE_FRIDGE_CYCLE,
    weak_decoder_to_frame=ONE_FRIDGE_CYCLE,
    frame_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_qpu=ONE_FRIDGE_CYCLE,
)
# The strong base's hops: the readout, the weak store's write, the
# decision and the pulse on the chip; the strong store's write and the
# strong chain on the host, whose boundaries ride the host's own memory
# read. The weak chain and the escalation keep the reference card.
STRONG_BASELINE_FABRIC = fabrics.PathLinks(
    qpu_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_weak_buffer=ONE_FRIDGE_CYCLE,
    controller_to_strong_buffer=ONE_ROOM_CYCLE,
    strong_buffer_to_strong_decoder=ONE_ROOM_CYCLE,
    strong_decoder_to_frame=ONE_ROOM_CYCLE,
    frame_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_qpu=ONE_FRIDGE_CYCLE,
)

# Backline (Xanadu and AMD), arXiv 2609.09270, Sec. V-C and Table III.
# An FPGA controller posts a 16-byte payload (an 8-byte syndrome) as a
# one-sided RDMA write into the coprocessor's memory over RoCE v2; the
# coprocessor polls that buffer and writes the reply back with a
# one-sided write, and on the GPU path it signals a CPU thread which
# writes back. The FPGA times the round trip in its own clock, so the
# measurement is one number per round trip and the paper states no
# per-direction split. These are the medians of the echo rows, which
# carry no decoding work and price the fabric alone.
ROCE_V2_CPU_ROUND_TRIP_MICROSECONDS = 2.305  # 2609.09270 line 1627
ROCE_V2_GPU_ROUND_TRIP_MICROSECONDS = 4.5  # 2609.09270 line 1632
ROCE_V2_CPU_SOURCE = (
    "Backline 2609.09270 Table III CPU echo, 2.305 us median round trip, "
    "FPGA controller to CPU coprocessor over RoCE v2"
)
ROCE_V2_GPU_SOURCE = (
    "Backline 2609.09270 Table III GPU echo, 4.5 us median round trip, "
    "FPGA controller to GPU coprocessor over RoCE v2"
)
# the payload each echo carried, whose time on the cable the measured
# round trip holds: 16 bytes (2609.09270 line 1611)
ROCE_V2_ECHO_PAYLOAD_BITS = 128
# NVQLink (NVIDIA), arXiv 2510.25213, Sec. 2.4. An FPGA sends 32-byte
# payloads as RoCE packets over 100 Gb Ethernet to a ConnectX-7 NIC,
# which writes them into GPU memory by RDMA; a persistent GPU kernel
# waits for each packet and loops it back through the NIC with no host
# processor involved (lines 391-403, 526-533; Fig. 2, lines 438-439),
# and the FPGA times the round trip (line 400). The connection is
# unreliable by choice: a dropped packet is not retransmitted (lines
# 376-388), on cables engineered under 1e-15 bit errors, so the ideal
# protocol; the measured round trip already holds its echo's framing
# (line 403). The steady-state mean and median are 3.839 us (line 485).
NVQLINK_ROUND_TRIP_MICROSECONDS = 3.839  # 2510.25213 line 485
# 32 bytes (2510.25213 lines 402-403)
NVQLINK_ECHO_PAYLOAD_BITS = 256
NVQLINK_SOURCE = (
    "NVQLink 2510.25213 line 485, 3.839 us steady-state median round "
    "trip, FPGA to a persistent GPU kernel and back over RoCE"
)
# Both measure the round trip on a 100 Gb/s cable: Backline's FPGA
# direct-attach cable (2609.09270 lines 1229-1230) and NVQLink's
# Ethernet link to the GPU host's NIC (2510.25213 line 382).
CABLE_BITS_PER_MICROSECOND = 100_000
# The coprocessor polls the buffer the write landed in, in its own
# memory: no cable, no time (Backline 2609.09270 lines 1616-1617).
_POLL_LEG = (
    "zero: the coprocessor polls the buffer in its own memory that the "
    "write landed in"
)
ROCE_V2_CPU_POLL = fabrics.TimedLink(
    0.0, source=f"{ROCE_V2_CPU_SOURCE}; {_POLL_LEG}"
)
ROCE_V2_GPU_POLL = fabrics.TimedLink(
    0.0, source=f"{ROCE_V2_GPU_SOURCE}; {_POLL_LEG}"
)
NVQLINK_GPU_POLL = fabrics.TimedLink(
    0.0, source=f"{NVQLINK_SOURCE}; {_POLL_LEG}"
)
ROCE_V2_CPU_LEG = fabrics.RoundTripLeg(
    ROCE_V2_CPU_ROUND_TRIP_MICROSECONDS,
    ROCE_V2_ECHO_PAYLOAD_BITS,
    CABLE_BITS_PER_MICROSECOND,
    ROCE_V2_CPU_SOURCE,
)
ROCE_V2_GPU_LEG = fabrics.RoundTripLeg(
    ROCE_V2_GPU_ROUND_TRIP_MICROSECONDS,
    ROCE_V2_ECHO_PAYLOAD_BITS,
    CABLE_BITS_PER_MICROSECOND,
    ROCE_V2_GPU_SOURCE,
)
NVQLINK_GPU_LEG = fabrics.RoundTripLeg(
    NVQLINK_ROUND_TRIP_MICROSECONDS,
    NVQLINK_ECHO_PAYLOAD_BITS,
    CABLE_BITS_PER_MICROSECOND,
    NVQLINK_SOURCE,
)
# The reference card with the strong side on a measured round trip: the
# write into the strong store, the escalation and both replies cross the
# cable, each a leg of the round trip, and the strong store's read is the
# host's poll. Neither paper gives a per-direction number, so the split
# is decsim's rule (fabrics.RoundTripLeg).
ROCE_V2_CPU_FABRIC = fabrics.FridgeHostFabric(
    cable=ROCE_V2_CPU_LEG, host_local=ROCE_V2_CPU_POLL
)
ROCE_V2_GPU_FABRIC = fabrics.FridgeHostFabric(
    cable=ROCE_V2_GPU_LEG, host_local=ROCE_V2_GPU_POLL
)
NVQLINK_GPU_FABRIC = fabrics.FridgeHostFabric(
    cable=NVQLINK_GPU_LEG, host_local=NVQLINK_GPU_POLL
)


def strong_side(
    base: ports.FabricRule,
    weak_to_strong: ports.LinkCard = ONE_ROOM_CYCLE,
) -> fabrics.PathLinks:
    """A base fabric with the strong side's five hops one room cycle.

    A run that switches on the weak base reaches the strong decoder over
    these hops, each an unbounded wire; weak_to_strong prices the
    escalation instead, where a study sets it; every other hop keeps
    the base's.
    """
    return fabrics.PathLinks(
        controller_to_strong_buffer=ONE_ROOM_CYCLE,
        weak_decoder_to_strong_decoder=weak_to_strong,
        strong_buffer_to_strong_decoder=ONE_ROOM_CYCLE,
        strong_decoder_to_frame=ONE_ROOM_CYCLE,
        strong_decoder_to_weak_decoder=ONE_ROOM_CYCLE,
        base=base,
    )


def bandwidth_limited_card(
    *,
    syndrome_bits_per_round: int,
    round_microseconds: float,
    commit_rounds: int,
    buffer_rounds: int,
    capacity_scale: float = 1.0,
) -> link_settings.FabricSettings:
    """The reference card with finite rates from the run's own geometry.

    The latencies are the reference card's, so switching cards changes
    bandwidth alone. Capacity is bits per microsecond. Every path carries
    exactly its nominal traffic in one commit region, so at capacity_scale 1
    each link runs at utilization one and at scale s at 1/s. State the
    utilization when reporting: a single-server queue at utilization one
    waits zero only under perfectly periodic arrivals (Little's law). The
    rates are an explicit per-link provisioning, as ns-3 declares a DataRate
    per device. A graph takes it as fabrics.CardFabric(card).

    Each rate is an exact fraction, so a nominal payload serializes in
    exactly its period: a float such as 8 / 1.1 is not the true rate, and
    its error would reach the tick count.
    """
    round_period_microseconds = fractions.Fraction(str(round_microseconds))
    commit_region_microseconds = commit_rounds * round_period_microseconds
    weak_window_rounds = commit_rounds + buffer_rounds
    weak_window_bits = weak_window_rounds * syndrome_bits_per_round
    strong_window_rounds = window_records.strong_region_round_count(
        commit_rounds, buffer_rounds
    )
    strong_window_bits = strong_window_rounds * syndrome_bits_per_round
    # one escalation is a selection and a region, each behind the
    # request's name, and its answer is one flip behind the same name;
    # on the default direct route the selection also carries the weak
    # crossing commit, one bit per logical observable
    name_bits = window_records.REQUEST_KEY_WIRE_BITS
    region_message_bits = name_bits + strong_window_bits
    selection_bits = name_bits + 1
    escalation_bits = selection_bits + region_message_bits
    strong_answer_bits = name_bits + 1
    one_per_region = 1 / commit_region_microseconds
    round_bits_per_microsecond = (
        syndrome_bits_per_round / round_period_microseconds
    )
    weak_window_bits_per_microsecond = (
        weak_window_bits / commit_region_microseconds
    )
    strong_window_bits_per_microsecond = (
        strong_window_bits / commit_region_microseconds
    )
    escalation_bits_per_microsecond = (
        escalation_bits / commit_region_microseconds
    )
    strong_answer_bits_per_microsecond = (
        strong_answer_bits / commit_region_microseconds
    )
    # one dense seam layer per commit region: the layer's detectors are
    # the round's syndrome bits (section 2.2 of the layer arithmetic:
    # d*d-1 ancilla measurements is one bulk layer's detector count)
    boundary_bits_per_microsecond = (
        syndrome_bits_per_round / commit_region_microseconds
    )
    bus_word_bits = link_profiles.BUS_WORD_BITS
    instruction_word_bits = link_profiles.INSTRUCTION_WORD_BITS
    bus_word_bits_per_microsecond = bus_word_bits / commit_region_microseconds
    instruction_word_bits_per_microsecond = (
        instruction_word_bits / commit_region_microseconds
    )
    bus_word_source = link_profiles.BUS_WORD_SOURCE + ", one per commit region"
    instruction_word_source = (
        link_profiles.INSTRUCTION_WORD_SOURCE + ", one per commit region"
    )
    reference = link_profiles.logical_reference_profile()
    provisioning = _Provisioning(capacity_scale, reference)
    qpu_to_controller = provisioning.path(
        "qpu_to_controller",
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one syndrome round per round period",
        link_profiles.READOUT_PAYLOAD_SOURCE,
    )
    controller_to_weak_buffer = provisioning.path(
        "controller_to_weak_buffer",
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one packed round per round period",
        link_profiles.ROUND_PAYLOAD_SOURCE,
    )
    controller_to_strong_buffer = provisioning.path(
        "controller_to_strong_buffer",
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one packed round per round period",
        link_profiles.ROUND_PAYLOAD_SOURCE,
    )
    weak_buffer_to_weak_decoder = provisioning.path(
        "weak_buffer_to_weak_decoder",
        weak_window_bits,
        weak_window_bits_per_microsecond,
        "one weak window of rcom+rbuf rounds per commit region "
        "(Toshio 2510.25222 Sec. III C, "
        '"rcom = rbuf = d")',
        link_profiles.DECODER_INPUT_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = provisioning.path(
        "weak_decoder_to_strong_decoder",
        region_message_bits,
        escalation_bits_per_microsecond,
        "one selection and one strong window of rcom+2rbuf rounds per "
        "commit region, each behind the request's name "
        "(Toshio 2510.25222 Sec. III C, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        link_profiles.ESCALATION_PAYLOAD_SOURCE,
    )
    strong_buffer_to_strong_decoder = provisioning.path(
        "strong_buffer_to_strong_decoder",
        strong_window_bits,
        strong_window_bits_per_microsecond,
        "one strong window of rcom+2rbuf rounds per commit region "
        "(Toshio 2510.25222 Sec. III C, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        link_profiles.DECODER_INPUT_PAYLOAD_SOURCE,
    )
    weak_decoder_to_frame = provisioning.path(
        "weak_decoder_to_frame",
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        link_profiles.RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = provisioning.path(
        "decoder_to_decoder",
        syndrome_bits_per_round,
        boundary_bits_per_microsecond,
        "one dense seam layer per commit region",
        link_profiles.BOUNDARY_PAYLOAD_SOURCE,
    )
    strong_decoder_to_frame = provisioning.path(
        "strong_decoder_to_frame",
        strong_answer_bits,
        strong_answer_bits_per_microsecond,
        "one frame-update bit per logical observable per commit region, "
        "behind the request's name",
        link_profiles.STRONG_RESULT_PAYLOAD_SOURCE,
    )
    strong_decoder_to_weak_decoder = provisioning.path(
        "strong_decoder_to_weak_decoder",
        strong_answer_bits,
        strong_answer_bits_per_microsecond,
        "one frame-update bit per logical observable per commit region, "
        "behind the request's name",
        link_profiles.STRONG_RESULT_PAYLOAD_SOURCE,
    )
    frame_to_controller = provisioning.path(
        "frame_to_controller",
        bus_word_bits,
        bus_word_bits_per_microsecond,
        bus_word_source,
        None,
    )
    controller_to_qpu = provisioning.path(
        "controller_to_qpu",
        instruction_word_bits,
        instruction_word_bits_per_microsecond,
        instruction_word_source,
        None,
    )
    return link_settings.FabricSettings(
        qpu_to_controller=qpu_to_controller,
        controller_to_weak_buffer=controller_to_weak_buffer,
        weak_buffer_to_weak_decoder=weak_buffer_to_weak_decoder,
        weak_decoder_to_strong_decoder=weak_decoder_to_strong_decoder,
        strong_buffer_to_strong_decoder=strong_buffer_to_strong_decoder,
        weak_decoder_to_frame=weak_decoder_to_frame,
        decoder_to_decoder=decoder_to_decoder,
        strong_decoder_to_frame=strong_decoder_to_frame,
        frame_to_controller=frame_to_controller,
        controller_to_qpu=controller_to_qpu,
        controller_to_strong_buffer=controller_to_strong_buffer,
        strong_decoder_to_weak_decoder=strong_decoder_to_weak_decoder,
        profile_name="bandwidth_limited",
    )


class _Provisioning:
    """The bounded card's paths: the reference latency at a scaled rate."""

    def __init__(
        self, capacity_scale: float, reference: link_settings.FabricSettings
    ):
        self._capacity_scale = fractions.Fraction(str(capacity_scale))
        self._reference = reference

    def path(
        self,
        name: str,
        bits: int,
        nominal_bits_per_microsecond: fractions.Fraction,
        source: str,
        actual_payload_source: Optional[str],
    ) -> link_settings.PathSettings:
        rate = nominal_bits_per_microsecond * self._capacity_scale
        capacity = link_settings.CapacitySettings(rate, source)
        reference_path = getattr(self._reference, name)
        reference_channel = reference_path.channel
        latency_ticks = reference_channel.propagation_latency_ticks
        channel = link_settings.ChannelSettings(
            name, latency_ticks, capacity, source
        )
        payload = link_settings.PayloadSettings(bits, source)
        return dataclasses.replace(
            reference_path,
            channel=channel,
            default_payload=payload,
            actual_payload_source=actual_payload_source,
        )
