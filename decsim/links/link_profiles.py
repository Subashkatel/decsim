"""The link number cards: the four shipped rows and the yaml's own.

logical_reference_profile is the default when no links card is given:
each hop priced from a system of decsim's scale, a surface-code patch on
one control rack with a decoder FPGA and a strong node one hop away. A
card is a latency and a rate, so a transfer takes its latency plus its
bits over the rate, the law ns-3's point-to-point device times a packet
by (point-to-point-net-device.cc:243). bandwidth_limited_profile keeps
those latencies and provisions finite rates from the run's geometry so
contention becomes measurable; capacity_scale sweeps the whole fabric.
roce_v2_measured_profile is the reference card with the strong tier's
off-board path priced by Backline's measured RoCE v2 round trip; it is
the roce_v2_cpu and roce_v2_gpu rows. from_yaml puts the yaml's own card
on any path; with_transfer_overhead adds a setup cost to any fabric.

Every number carries a source string on the settings record it sets,
and a payload's source also travels into the traffic report with every
transfer; paper locators are arXiv numbers and text lines. To
change a number, copy a card into your own file and edit it, then pass
it as the machine's links setting.
"""

import dataclasses
import fractions
import math
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.fabric as fabric
import decsim.links.settings as settings
import decsim.ports as ports
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.tables as tables

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

# The readout hop carries the QPU-side readout at the width the QPU
# reports for it (controller/controller.py sends readout.size_bits).
READOUT_PAYLOAD_SOURCE = "QPUReadout.size_bits"

# The escalation hop carries the strong window's rounds up from the
# weak syndrome buffer at the width each round left the controller
# (Toshio et al. 2510.25222 lines 1247 to 1250 assign the region's
# syndrome data to the strong decoder at the switch; decoders/
# decoder_output.py send_region). The selection that names the strong
# request rides the same hop first with no bits.
ESCALATION_PAYLOAD_SOURCE = "EscalatedRegion.wire_bits"

# The two controller-to-store hops carry the packed round at the width
# it leaves the controller: the detection events where the controller
# forms them and the raw measurement outcomes where the decoder does
# (controller.detection_events_formed_at,
# controller/round_assembly.py's wire_bits of the fragments that leave).
ROUND_PAYLOAD_SOURCE = "PackedRound.wire_bits"

# Both decoder-input hops carry the job's own payload count, read by the
# store's output port while the rounds are still the job's
# (syndrome_buffer/round_output.py).
DECODER_INPUT_PAYLOAD_SOURCE = "DecodeJob.payload_bits()"

# A window's hand-off updates the detectors of its neighbour's oldest
# round layer and nothing else (Tan et al. 2209.09219 lines 936-946;
# quits sliding_window.py:164-174; cuda-q QEC sliding_window.cpp:325-344;
# Bombin et al. 2303.04846 lines 784-786). The window side counts that
# seam in the representation windows.boundary_payload names, dense or
# sparse (decsim/windows/boundary_payloads.py), and passes the count with
# the send.
BOUNDARY_PAYLOAD_SOURCE = "DependencyResidual seam-layer detectors"

# Every default card counts on one 250 MHz clock, a 4 ns cycle: the clock
# Yang et al. break their closed loop down at (arXiv 2605.04892 Table I,
# text lines 1047-1063) and the cycle of IBM's decoder FPGA (Maurer et
# al., arXiv 2510.21600, line 404). A card's rate is bits per cycle of it.
_REFERENCE_CLOCK_MEGAHERTZ = 250
_REFERENCE_CYCLE_MICROSECONDS = 1 / _REFERENCE_CLOCK_MEGAHERTZ

# Yang et al., arXiv 2605.04892, Table I, is the one loop of decsim's scale
# broken down hop by hop: a d=3 surface code whose readout modules send
# bit strings over low-latency links to a decoder FPGA, which sends branch
# control over further links to the pulse generators (lines 184-199).
# The readout hop starts where Table I's acquisition window ends, which
# decsim's round period already holds, and it includes turning the signal
# into bits, which is why the card is the reference number a yaml's own
# readout cost must not repeat. Each qubit has its own demodulation
# channel (QubiC, Fruitwala et al. 2404.15260, lines 709-715), so a
# round's bits arrive in parallel and the hop serializes nothing.
_READOUT_LATENCY_MICROSECONDS = 0.048
_READOUT_SOURCE = (
    "Yang 2605.04892 Table I lines 1053-1055: ADC chip 12 ns, IQ "
    "demodulation 32 ns, qubit-state classification 4 ns"
)

# Table I gives one 36 ns digital-communication line for the loop's two
# links, readout module to decoder FPGA and decoder FPGA to the pulse
# side, and no split, so each link is charged half, the way the RoCE rows
# split a round trip. The weak syndrome buffer sits on the decoder chip,
# so the controller's write into it is the first link; a weak result
# reaches the frame in the controller over the second.
WEAK_STORE_LATENCY_MICROSECONDS = 0.018
WEAK_STORE_SOURCE = (
    "Yang 2605.04892 Table I line 1056, digital communication 36 ns over "
    "the loop's two links; half, readout module to decoder FPGA"
)
_WEAK_RESULT_SOURCE = (
    "Yang 2605.04892 Table I line 1056, digital communication 36 ns over "
    "the loop's two links; half, decoder FPGA to the pulse side"
)

# The strong node is one hop from the control rack, the way Caune et al.,
# arXiv 2410.05202, keep every chassis one inter-chassis hop from the hub
# (lines 1013-1017). Fig. 1a measures that hop, stage F, "inter-node delay
# time for broadcasting between control system chassis" (240 to 260 ns,
# lines 176-177), taken at the stated worst case for every hop that
# crosses to or from the strong node.
STRONG_STORE_LATENCY_MICROSECONDS = 0.26
STRONG_STORE_SOURCE = (
    "Caune 2410.05202 Fig. 1a F, inter-node broadcast between control "
    "system chassis, 240 to 260 ns at the stated worst case"
)

# No referent measures a wire inside one chip. Yang keep the syndrome in
# FPGA registers, fully pipelined (lines 1273-1275), and a compiled
# on-chip memory moves one 32-bit word per access (the sky130 1rw1r 32x256
# SRAM macro, VLSIDA sky130_sram_macros, configuration lines 7-18), so an
# on-chip hop is one cycle and a word per cycle, a stated assumption.
_ON_CHIP_SOURCE = (
    "one 250 MHz cycle on chip, a stated assumption: Yang 2605.04892 "
    "lines 1273-1275 keep the syndrome in registers, fully pipelined"
)
# A seam layer crosses one link per graph edge in one clock domain
# (Helios, Liyanage et al. 2301.08419, lines 764-769 and 801), so all its
# detectors move in parallel.
_BOUNDARY_SOURCE = (
    "one 250 MHz cycle on chip, a stated assumption: Helios 2301.08419 "
    "lines 764-769 and 801, a link per graph edge in one clock domain"
)
# The frame sits in the controller and hands the decision to the core as
# one 32-bit word with a ready signal (QubiC 2404.15260 lines 186-190),
# so the hop adds nothing beyond that word's one cycle.
_DECISION_SOURCE = (
    "no propagation: the frame hands the core one 32-bit word with a "
    "ready signal (QubiC 2404.15260 lines 186-190)"
)
# From the pulse trigger to the pulse leaving the rack. The coax down to
# the QPU is outside Yang's measurement, which ends at the pulse
# generator's output port (lines 1699-1702), and no source gives it.
_PULSE_LATENCY_MICROSECONDS = 0.088
_PULSE_SOURCE = (
    "Yang 2605.04892 Table I lines 1058-1060: trigger propagation to the "
    "pulse generator 16 ns, waveform generation 32 ns, DAC chip 40 ns; "
    "QICK 2110.00557 lines 884-886 measure a 45 ns DAC"
)

# The rates. A word per cycle on the 32-bit bus the decoder is written
# and read over (Caune 2410.05202 lines 998-1008 and 1252-1254; the paper
# gives the width, not a transaction rate) and on the on-chip memory
# word above; one 128-bit instruction word per cycle into the pulse
# generator; and off board, the 100 Gb direct-attach cable from an FPGA
# controller to its coprocessor host (Backline, arXiv 2609.09270, lines
# 1229-1230), the one off-board line rate a referent of this scale gives.
_WORD_BITS_PER_MICROSECOND = BUS_WORD_BITS * _REFERENCE_CLOCK_MEGAHERTZ
_WORD_RATE_SOURCE = (
    "one 32-bit word per 250 MHz cycle (Caune 2410.05202 lines 998-1008, "
    "the decoder's 32-bit bus)"
)
_INSTRUCTION_BITS_PER_MICROSECOND = (
    INSTRUCTION_WORD_BITS * _REFERENCE_CLOCK_MEGAHERTZ
)
_INSTRUCTION_RATE_SOURCE = (
    "one 128-bit instruction word per 250 MHz cycle (QubiC 2404.15260 "
    "lines 176-179)"
)
_OFF_BOARD_BITS_PER_MICROSECOND = 100_000
_OFF_BOARD_RATE_SOURCE = (
    "100 Gb/s, Backline 2609.09270 lines 1229-1230, the FPGA "
    "controller's direct-attach cable to its coprocessor host"
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
ROCE_V2_CPU_ROUND_TRIP_MICROSECONDS = 2.305
ROCE_V2_GPU_ROUND_TRIP_MICROSECONDS = 4.5
ROCE_V2_CPU_SOURCE = (
    "Backline 2609.09270 Table III CPU echo, 2.305 us median round trip, "
    "FPGA controller to CPU coprocessor over RoCE v2"
)
ROCE_V2_GPU_SOURCE = (
    "Backline 2609.09270 Table III GPU echo, 4.5 us median round trip, "
    "FPGA controller to GPU coprocessor over RoCE v2"
)

# decsim prices one number per hop, so each leg of the round trip is
# charged half of it and the coprocessor's own poll is charged nothing.
# The legs the measurement covers are the controller's one-sided write
# into the coprocessor's memory (paper lines 1610-1612), the coprocessor's
# poll "on the expected memory buffer" (1616-1617) and the reply's
# one-sided write back (1617-1620).
ROCE_V2_WRITE_LEG = (
    "the controller's one-sided write into the coprocessor's memory, one "
    "half of the measured round trip; the paper gives no per-direction "
    "split"
)
ROCE_V2_ESCALATION_LEG = (
    "the escalation request, the same one-sided write from the "
    "controller side, one half of the measured round trip; the paper "
    "gives no per-direction split"
)
ROCE_V2_POLL_LEG = (
    "zero: the coprocessor polls the buffer in its own memory that the "
    "write landed in"
)
ROCE_V2_REPLY_LEG = (
    "the reply's one-sided write back to the controller, one half of the "
    "measured round trip; the paper gives no per-direction split"
)


# The keys a path's card writes: the first three on every card, the rest
# only when the path has lanes, a setup or a header to price.
_REQUIRED_CARD_KEYS = ("latency_cycles", "clock", "bits_per_cycle")
_CARD_KEYS = _REQUIRED_CARD_KEYS + (
    "channels",
    "setup_cycles_per_transfer",
    "header_bits_per_transfer",
)


def logical_reference_profile() -> settings.FabricSettings:
    """The default card: every hop from a system of decsim's scale.

    Each hop has a latency and a rate, or is unbounded where its
    referent moves every bit in parallel, so a transfer costs its
    latency plus its bits over the rate. The weak loop is Yang et al.'s
    closed loop (arXiv 2605.04892 Table I) hop for hop, the strong node
    is Caune et al.'s inter-chassis hop away (arXiv 2410.05202 Fig. 1a),
    and the rates are the referents' bus words and Backline's cable.
    Actual-payload paths price the runtime's own bit counts;
    default-payload paths price a stated word width.
    """
    word_rate = settings.CapacitySettings(
        _WORD_BITS_PER_MICROSECOND, _WORD_RATE_SOURCE
    )
    instruction_rate = settings.CapacitySettings(
        _INSTRUCTION_BITS_PER_MICROSECOND, _INSTRUCTION_RATE_SOURCE
    )
    off_board_rate = settings.CapacitySettings(
        _OFF_BOARD_BITS_PER_MICROSECOND, _OFF_BOARD_RATE_SOURCE
    )
    qpu_to_controller = _actual_path(
        "qpu_to_controller",
        _READOUT_LATENCY_MICROSECONDS,
        _READOUT_SOURCE,
        READOUT_PAYLOAD_SOURCE,
    )
    controller_to_weak_buffer = _actual_path(
        "controller_to_weak_buffer",
        WEAK_STORE_LATENCY_MICROSECONDS,
        WEAK_STORE_SOURCE,
        ROUND_PAYLOAD_SOURCE,
        word_rate,
    )
    controller_to_strong_buffer = _actual_path(
        "controller_to_strong_buffer",
        STRONG_STORE_LATENCY_MICROSECONDS,
        STRONG_STORE_SOURCE,
        ROUND_PAYLOAD_SOURCE,
        off_board_rate,
    )
    weak_buffer_to_weak_decoder = _actual_path(
        "weak_buffer_to_weak_decoder",
        _REFERENCE_CYCLE_MICROSECONDS,
        _ON_CHIP_SOURCE,
        DECODER_INPUT_PAYLOAD_SOURCE,
        word_rate,
    )
    weak_decoder_to_strong_decoder = _actual_path(
        "weak_decoder_to_strong_decoder",
        STRONG_STORE_LATENCY_MICROSECONDS,
        STRONG_STORE_SOURCE,
        ESCALATION_PAYLOAD_SOURCE,
        off_board_rate,
    )
    strong_buffer_to_strong_decoder = _actual_path(
        "strong_buffer_to_strong_decoder",
        _REFERENCE_CYCLE_MICROSECONDS,
        _ON_CHIP_SOURCE,
        DECODER_INPUT_PAYLOAD_SOURCE,
        word_rate,
    )
    weak_decoder_to_frame = _actual_path(
        "weak_decoder_to_frame",
        WEAK_STORE_LATENCY_MICROSECONDS,
        _WEAK_RESULT_SOURCE,
        RESULT_PAYLOAD_SOURCE,
        word_rate,
    )
    decoder_to_decoder = _actual_path(
        "decoder_to_decoder",
        _REFERENCE_CYCLE_MICROSECONDS,
        _BOUNDARY_SOURCE,
        BOUNDARY_PAYLOAD_SOURCE,
    )
    strong_decoder_to_frame = _actual_path(
        "strong_decoder_to_frame",
        STRONG_STORE_LATENCY_MICROSECONDS,
        STRONG_STORE_SOURCE,
        RESULT_PAYLOAD_SOURCE,
        off_board_rate,
    )
    frame_to_controller = _default_path(
        "frame_to_controller",
        0.0,
        _DECISION_SOURCE,
        BUS_WORD_BITS,
        BUS_WORD_SOURCE,
        word_rate,
    )
    controller_to_qpu = _default_path(
        "controller_to_qpu",
        _PULSE_LATENCY_MICROSECONDS,
        _PULSE_SOURCE,
        INSTRUCTION_WORD_BITS,
        INSTRUCTION_WORD_SOURCE,
        instruction_rate,
    )
    return settings.FabricSettings(
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
        profile_name="logical_reference",
        kind="logical_reference",
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

    Same paths and payload rules as logical_reference_profile, and its
    latencies read from that card, so switching cards changes bandwidth
    and nothing else. Capacity is bits per microsecond.

    Every path is provisioned to carry exactly its nominal traffic in one
    commit region (qc: one round's syndrome bits per round period), so at
    capacity_scale 1 each link runs at utilization one, and a scale of s
    runs it at utilization 1/s. State the utilization when reporting
    results from this card: a single-server queue at utilization one
    waits zero only under perfectly periodic arrivals, and any jitter
    accumulates (Little's law). The rates are an explicit per-link
    provisioning, the way ns-3 declares a DataRate per point-to-point
    device, never a floor borrowed from another path.

    Each rate is an exact fraction of the decimals given, so a nominal
    payload serializes in exactly its period: a float such as 8 / 1.1
    lands below the true rate, and the rounded-up serialization then
    runs one tick past the period and a periodic stream queues one tick
    more every round (ns-3 times a packet from its size and the device's
    stated DataRate, point-to-point-net-device.cc:243).
    """
    round_period_microseconds = fractions.Fraction(str(round_microseconds))
    commit_region_microseconds = commit_rounds * round_period_microseconds
    weak_window_rounds = commit_rounds + buffer_rounds
    weak_window_bits = weak_window_rounds * syndrome_bits_per_round
    strong_window_rounds = window_records.strong_region_round_count(
        commit_rounds, buffer_rounds
    )
    strong_window_bits = strong_window_rounds * syndrome_bits_per_round
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
    # one dense seam layer per commit region: the layer's detectors are
    # the round's syndrome bits (section 2.2 of the layer arithmetic:
    # d*d-1 ancilla measurements is one bulk layer's detector count)
    boundary_bits_per_microsecond = (
        syndrome_bits_per_round / commit_region_microseconds
    )
    bus_word_bits_per_microsecond = BUS_WORD_BITS / commit_region_microseconds
    instruction_word_bits_per_microsecond = (
        INSTRUCTION_WORD_BITS / commit_region_microseconds
    )
    bus_word_source = BUS_WORD_SOURCE + ", one per commit region"
    instruction_word_source = (
        INSTRUCTION_WORD_SOURCE + ", one per commit region"
    )
    reference = logical_reference_profile()
    provisioning = _Provisioning(capacity_scale, reference)
    qpu_to_controller = provisioning.path(
        "qpu_to_controller",
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one syndrome round per round period",
        READOUT_PAYLOAD_SOURCE,
    )
    controller_to_weak_buffer = provisioning.path(
        "controller_to_weak_buffer",
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one packed round per round period",
        ROUND_PAYLOAD_SOURCE,
    )
    controller_to_strong_buffer = provisioning.path(
        "controller_to_strong_buffer",
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one packed round per round period",
        ROUND_PAYLOAD_SOURCE,
    )
    weak_buffer_to_weak_decoder = provisioning.path(
        "weak_buffer_to_weak_decoder",
        weak_window_bits,
        weak_window_bits_per_microsecond,
        "one weak window of rcom+rbuf rounds per commit region "
        "(Toshio 2510.25222 Sec. III C, "
        '"rcom = rbuf = d")',
        DECODER_INPUT_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = provisioning.path(
        "weak_decoder_to_strong_decoder",
        strong_window_bits,
        strong_window_bits_per_microsecond,
        "one strong window of rcom+2rbuf rounds per commit region "
        "(Toshio 2510.25222 Sec. III C, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        ESCALATION_PAYLOAD_SOURCE,
    )
    strong_buffer_to_strong_decoder = provisioning.path(
        "strong_buffer_to_strong_decoder",
        strong_window_bits,
        strong_window_bits_per_microsecond,
        "one strong window of rcom+2rbuf rounds per commit region "
        "(Toshio 2510.25222 Sec. III C, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        DECODER_INPUT_PAYLOAD_SOURCE,
    )
    weak_decoder_to_frame = provisioning.path(
        "weak_decoder_to_frame",
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = provisioning.path(
        "decoder_to_decoder",
        syndrome_bits_per_round,
        boundary_bits_per_microsecond,
        "one dense seam layer per commit region",
        BOUNDARY_PAYLOAD_SOURCE,
    )
    strong_decoder_to_frame = provisioning.path(
        "strong_decoder_to_frame",
        1,
        one_per_region,
        "one frame-update bit per logical observable per commit region",
        RESULT_PAYLOAD_SOURCE,
    )
    frame_to_controller = provisioning.path(
        "frame_to_controller",
        BUS_WORD_BITS,
        bus_word_bits_per_microsecond,
        bus_word_source,
        None,
    )
    controller_to_qpu = provisioning.path(
        "controller_to_qpu",
        INSTRUCTION_WORD_BITS,
        instruction_word_bits_per_microsecond,
        instruction_word_source,
        None,
    )
    return settings.FabricSettings(
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
        profile_name="bandwidth_limited",
        kind="bandwidth_limited",
    )


def roce_v2_measured_profile(coprocessor: str) -> settings.FabricSettings:
    """The reference card with the strong path on a measured round trip.

    Backline (arXiv 2609.09270, Sec. V-C and Table III) measures one
    number: an FPGA controller writes into a coprocessor's memory over
    RoCE v2 with a one-sided RDMA write, the coprocessor polls that
    buffer and writes the reply back, and the controller times the whole
    round trip in its own clock. The paper gives no per-direction
    number, so the split below is decsim's rule rather than a
    measurement: the controller's write into strong syndrome buffer, the
    escalation request and the strong decoder's reply to the frame are
    each one half of the round trip, and the
    strong store's read into the strong decoder is zero because the
    coprocessor polls a buffer in its own memory. The escalation round
    trip on this card, weak_decoder_to_strong_decoder plus
    strong_buffer_to_strong_decoder plus strong_decoder_to_frame, is
    therefore the measured median exactly.

    The card prices that median. It does not cover the first, warm-up
    round trip, 4.64 us on the CPU path and 9.27 us on the GPU path, nor
    the tails, which on the CPU path reach 2.420 us at P99, 2.475 us at
    P99.999 and 2.590 us at the maximum, and on the GPU path 4.985 us,
    5.255 us and 5.285 us. Every other hop keeps the number and the
    source of logical_reference_profile.

    Args:
        coprocessor: "cpu" or "gpu", the two paths Backline measured.
    """
    round_trip_microseconds, measurement_source = _roce_v2_measurement(
        coprocessor
    )
    strong_paths = _roce_v2_strong_paths(
        round_trip_microseconds, measurement_source
    )
    reference = logical_reference_profile()
    row_name = f"roce_v2_{coprocessor}"
    return dataclasses.replace(
        reference, **strong_paths, profile_name=row_name, kind=row_name
    )


class LogicalReferenceFabric:
    """The default row: the reference card's latencies and rates.

    A transfer costs its latency plus its bits over the hop's rate, and
    transfers on one hop queue for its wire. This is the row a yaml gets
    when it names no kind, and the numbers its per-path cards override.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The numbers a yaml's per-path cards override."""
        return logical_reference_profile()

    @staticmethod
    def build(
        card: settings.FabricSettings, engine: decsim.engine.Engine
    ) -> ports.Link:
        """The object that carries this run's transfers."""
        return fabric.LinkFabric(card, engine, channel_module.Channel)


class BandwidthLimitedFabric:
    """The same fabric with finite rates, provisioned from the geometry.

    Every channel carries exactly its nominal traffic in one commit
    region (the readout and store hops: one round's bits in one round
    period), so contention becomes measurable and capacity_scale sweeps
    the whole fabric. Its numbers come from the run's own geometry, not from the
    links section, which is why base_card refuses a yaml and names what
    a caller has to give it.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """Refused: this row provisions itself from the run's geometry."""
        raise ValueError(
            "links.kind bandwidth_limited provisions every channel from "
            "the sweep point's own geometry (the syndrome bits per round, "
            "the round period, the commit and buffer rounds), which the "
            "links section does not carry and which is known only per "
            "point; build the card with "
            "link_profiles.bandwidth_limited_profile(...) and pass it as "
            "the machine's links setting"
        )

    @staticmethod
    def build(
        card: settings.FabricSettings, engine: decsim.engine.Engine
    ) -> ports.Link:
        """The object that carries this run's transfers."""
        return fabric.LinkFabric(card, engine, channel_module.Channel)


class RoceV2CpuFabric:
    """The reference card with the strong path on Backline's CPU round trip.

    The four strong-side hops are priced by Backline's measured
    FPGA-to-CPU round trip over RoCE v2, 2.305 us in the median
    (arXiv 2609.09270, Table III): half of it on each leg the write
    crosses, and nothing for the coprocessor's poll of its own memory.
    Every other hop keeps the reference numbers.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The numbers a yaml's per-path cards override."""
        return roce_v2_measured_profile("cpu")

    @staticmethod
    def build(
        card: settings.FabricSettings, engine: decsim.engine.Engine
    ) -> ports.Link:
        """The object that carries this run's transfers."""
        return fabric.LinkFabric(card, engine, channel_module.Channel)


class RoceV2GpuFabric:
    """The reference card with the strong path on Backline's GPU round trip.

    The four strong-side hops are priced by Backline's measured
    FPGA-to-GPU round trip over RoCE v2, 4.5 us in the median
    (arXiv 2609.09270, Table III): half of it on each leg the write
    crosses, and nothing for the coprocessor's poll of its own memory.
    The GPU path is the slower and the wider of the two Backline
    measured, because the reply is written by a CPU thread the GPU
    signals. Every other hop keeps the reference numbers.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The numbers a yaml's per-path cards override."""
        return roce_v2_measured_profile("gpu")

    @staticmethod
    def build(
        card: settings.FabricSettings, engine: decsim.engine.Engine
    ) -> ports.Link:
        """The object that carries this run's transfers."""
        return fabric.LinkFabric(card, engine, channel_module.Channel)


# links.kind names one of these rows: which fabric model carries the
# transfers and which numbers the section's per-path cards override. A
# row answers base_card at the yaml boundary and build at the root.
LINK_FABRICS = {
    "logical_reference": LogicalReferenceFabric,
    "bandwidth_limited": BandwidthLimitedFabric,
    "roce_v2_cpu": RoceV2CpuFabric,
    "roce_v2_gpu": RoceV2GpuFabric,
}


def from_yaml(
    section: Mapping, clocks: config.ClockSettings, name: str
) -> settings.FabricSettings:
    """The yaml's `links` section: a kind, and one card per path over it.

    A card prices its path in cycles of a named clock domain: latency,
    bits per cycle per lane (null is unbounded), the lane count, an
    optional per-transfer setup cost, and an optional per-transfer
    header in bits. A null card keeps the chosen row's
    numbers for that path. The config prices readout classification on
    its own line, so its qpu_to_controller card is link propagation only,
    and the fabric says so.
    """
    source = f"configs/{name}.yaml links"
    kind = section.get("kind", "logical_reference")
    row = tables.row(LINK_FABRICS, "links.kind", kind)
    _check_section_names(section)
    profile = row.base_card()
    replacements = {}
    for path_name, card in section.items():
        if path_name == "kind":
            continue
        if card is None:
            continue
        _check_card(path_name, card)
        path_settings = getattr(profile, path_name)
        carded = _carded_path(path_name, path_settings, card, clocks, source)
        replacements[path_name] = dataclasses.replace(
            carded, excludes_receiver_processing=True
        )
    return dataclasses.replace(
        profile,
        **replacements,
        kind=kind,
        profile_name=f"{name}.yaml",
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


def _roce_v2_measurement(coprocessor: str) -> tuple:
    """(round trip in microseconds, source) of one echo row of Table III."""
    if coprocessor == "cpu":
        return ROCE_V2_CPU_ROUND_TRIP_MICROSECONDS, ROCE_V2_CPU_SOURCE
    if coprocessor == "gpu":
        return ROCE_V2_GPU_ROUND_TRIP_MICROSECONDS, ROCE_V2_GPU_SOURCE
    raise ValueError(
        f"a measured RoCE v2 card names the coprocessor Backline echoed "
        f"from, 'cpu' or 'gpu', not {coprocessor!r}"
    )


def _roce_v2_strong_paths(
    round_trip_microseconds: float, measurement_source: str
) -> dict:
    """The four strong-side paths, priced from one measured round trip."""
    leg_microseconds = round_trip_microseconds / 2
    write_source = f"{measurement_source}; {ROCE_V2_WRITE_LEG}"
    escalation_source = f"{measurement_source}; {ROCE_V2_ESCALATION_LEG}"
    poll_source = f"{measurement_source}; {ROCE_V2_POLL_LEG}"
    reply_source = f"{measurement_source}; {ROCE_V2_REPLY_LEG}"
    controller_to_strong_buffer = _actual_path(
        "controller_to_strong_buffer",
        leg_microseconds,
        write_source,
        ROUND_PAYLOAD_SOURCE,
    )
    weak_decoder_to_strong_decoder = _actual_path(
        "weak_decoder_to_strong_decoder",
        leg_microseconds,
        escalation_source,
        ESCALATION_PAYLOAD_SOURCE,
    )
    strong_buffer_to_strong_decoder = _actual_path(
        "strong_buffer_to_strong_decoder",
        0.0,
        poll_source,
        DECODER_INPUT_PAYLOAD_SOURCE,
    )
    strong_decoder_to_frame = _actual_path(
        "strong_decoder_to_frame",
        leg_microseconds,
        reply_source,
        RESULT_PAYLOAD_SOURCE,
    )
    return {
        "controller_to_strong_buffer": controller_to_strong_buffer,
        "weak_decoder_to_strong_decoder": weak_decoder_to_strong_decoder,
        "strong_buffer_to_strong_decoder": strong_buffer_to_strong_decoder,
        "strong_decoder_to_frame": strong_decoder_to_frame,
    }


def _check_section_names(section: Mapping) -> None:
    """The links section names the kind and paths, and nothing else."""
    path_names = []
    for path in transfer_records.LinkPath:
        path_names.append(path.value)
    for section_name in section:
        if section_name == "kind":
            continue
        if section_name not in path_names:
            raise ValueError(
                f"links names {section_name!r}, which is not a path; the "
                f"paths are {path_names}"
            )


def _check_card(path_name: str, card) -> None:
    """A path's card is a mapping of the card keys, the first three written.

    Refused here, once, with the card's yaml name, so a card never reaches
    the arithmetic below as a list or with a key missing, and a misspelt
    key is never read as the default of the key it meant.
    """
    card_name = f"links.{path_name}"
    if not isinstance(card, Mapping):
        raise ValueError(
            f"{card_name} holds {card!r}; a path's card is a mapping of "
            f"{list(_CARD_KEYS)}, or null for the row's own numbers"
        )
    unknown = [key for key in card if key not in _CARD_KEYS]
    if unknown:
        raise ValueError(
            f"{card_name} does not know {unknown}; its keys are "
            f"{list(_CARD_KEYS)}"
        )
    missing = [key for key in _REQUIRED_CARD_KEYS if key not in card]
    if missing:
        raise ValueError(
            f"{card_name} needs {missing}; a card writes "
            f"{list(_REQUIRED_CARD_KEYS)}, with bits_per_cycle null for an "
            f"unbounded wire"
        )
    _check_bits_per_cycle(card_name, card["bits_per_cycle"])
    lane_count = card.get("channels", 1)
    _check_lane_count(card_name, lane_count)
    _check_cycle_counts(card_name, card)


def _check_cycle_counts(card_name: str, card: Mapping) -> None:
    """The latency and the setup, when written, are whole cycle counts."""
    config.check_cycles(f"{card_name}.latency_cycles", card["latency_cycles"])
    setup_cycles = card.get("setup_cycles_per_transfer")
    if setup_cycles is None:
        return
    setup_name = f"{card_name}.setup_cycles_per_transfer"
    config.check_cycles(setup_name, setup_cycles)


def _check_bits_per_cycle(card_name: str, bits_per_cycle) -> None:
    """A lane's rate is a positive finite number, or null for no bound.

    A yaml `true` is a boolean, which Python would read as the number 1,
    so it is refused with the rest rather than priced as one bit.
    """
    if bits_per_cycle is None:
        return
    if _is_positive_number(bits_per_cycle):
        return
    raise ValueError(
        f"{card_name}.bits_per_cycle is {bits_per_cycle!r}; it is the "
        f"positive number of bits each lane moves per cycle, or null for "
        f"an unbounded wire"
    )


def _check_lane_count(card_name: str, lane_count) -> None:
    """A card's lanes are a positive whole number, never a yaml boolean."""
    if _is_positive_whole_number(lane_count):
        return
    raise ValueError(
        f"{card_name}.channels is {lane_count!r}; it is the positive whole "
        f"number of parallel lanes the path's wire has"
    )


def _is_positive_whole_number(value) -> bool:
    """A whole number above zero, never a yaml boolean."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, int):
        return False
    return value > 0


def _is_positive_number(value) -> bool:
    """A finite number above zero, never a yaml boolean."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return 0 < value < math.inf


def _card_ticks(card: Mapping, clocks: config.ClockSettings) -> tuple:
    """(latency ticks, aggregate rate, setup ticks) of one card.

    A cycle costs its domain's period in whole ticks, gem5's
    cyclesToTicks (src/sim/clocked_object.hh:227, clockPeriod() * c),
    so a link's cycles and every other component's cycles on one domain
    are the same ticks. The rate moves bits_per_cycle on each lane every
    period, kept as an exact fraction of the card's decimals.
    """
    clock = clocks.clock(card["clock"])
    period_ticks = clock.period_ticks
    latency_cycles = card["latency_cycles"]
    latency_ticks = latency_cycles * period_ticks
    bits_per_cycle = card["bits_per_cycle"]
    lane_count = card.get("channels", 1)
    bits_per_microsecond = None
    if bits_per_cycle is not None:
        lane_bits_per_cycle = fractions.Fraction(str(bits_per_cycle))
        bits_per_period = lane_bits_per_cycle * lane_count
        bits_per_tick = bits_per_period / period_ticks
        bits_per_microsecond = bits_per_tick * config.TICKS_PER_MICROSECOND
    setup_cycles = card.get("setup_cycles_per_transfer")
    setup_ticks = 0
    if setup_cycles is not None:
        setup_ticks = setup_cycles * period_ticks
    return latency_ticks, bits_per_microsecond, setup_ticks


def _carded_path(
    path_name: str,
    path_settings: settings.PathSettings,
    card: Mapping,
    clocks: config.ClockSettings,
    source: str,
) -> settings.PathSettings:
    """The reference path with the card's channel and setup cost."""
    latency_ticks, bits_per_microsecond, setup_ticks = _card_ticks(card, clocks)
    capacity = None
    if bits_per_microsecond is not None:
        capacity = settings.CapacitySettings(bits_per_microsecond, source)
    channel = settings.ChannelSettings(
        path_name, latency_ticks, capacity, source
    )
    header_bits = card.get("header_bits_per_transfer", 0)
    return dataclasses.replace(
        path_settings,
        channel=channel,
        setup_ticks=setup_ticks,
        header_bits=header_bits,
    )


class _Provisioning:
    """The bounded card's paths: the reference latency at a scaled rate."""

    def __init__(
        self, capacity_scale: float, reference: settings.FabricSettings
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
    ) -> settings.PathSettings:
        rate = nominal_bits_per_microsecond * self._capacity_scale
        capacity = settings.CapacitySettings(rate, source)
        reference_path = getattr(self._reference, name)
        reference_channel = reference_path.channel
        latency_ticks = reference_channel.propagation_latency_ticks
        channel = settings.ChannelSettings(
            name, latency_ticks, capacity, source
        )
        payload = settings.PayloadSettings(bits, source)
        return settings.PathSettings(channel, payload, actual_payload_source)


def _channel(
    name: str,
    latency_microseconds: float,
    source: str,
    capacity: Optional[settings.CapacitySettings],
) -> settings.ChannelSettings:
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    return settings.ChannelSettings(name, latency_ticks, capacity, source)


def _actual_path(
    name: str,
    latency_microseconds: float,
    source: str,
    actual_payload_source: str,
    capacity: Optional[settings.CapacitySettings] = None,
) -> settings.PathSettings:
    channel = _channel(name, latency_microseconds, source, capacity)
    return settings.PathSettings(channel, None, actual_payload_source)


def _default_path(
    name: str,
    latency_microseconds: float,
    latency_source: str,
    bits: int,
    payload_source: str,
    capacity: settings.CapacitySettings,
) -> settings.PathSettings:
    channel = _channel(name, latency_microseconds, latency_source, capacity)
    payload = settings.PayloadSettings(bits, payload_source)
    return settings.PathSettings(channel, payload, None)
