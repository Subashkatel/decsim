"""The link number cards decsim ships.

A card is a latency and a rate per hop, so a transfer takes its latency
plus its bits over the rate, the law ns-3's point-to-point device times
a packet by (point-to-point-net-device.cc:243). Every number carries a
source string on the record it sets, and a payload's source travels into
the traffic report; paper locators are arXiv numbers and text lines. To
change a number, copy a card into your own file and pass it as the
machine's links setting.
"""

import dataclasses
import fractions
from typing import Optional

import decsim.config as config
import decsim.links.settings as settings
import decsim.records.windows as window_records

# A decoder result reaches the frame as one bit per logical observable, the
# logical-frame convention: Caune et al. 2410.05202 return one Boolean per
# decode, Google 2408.13687 an observable bitmask per block, and PECOS's
# frame XORs an observable mask per cycle. A decoder that feeds a physical
# frame instead emits a per-qubit or per-edge correction vector (LILLIPUT's
# error log, Helios's correction port); that is a different card. The
# window manager supplies the count from the result itself.
RESULT_PAYLOAD_SOURCE = "DecodeResult.logical_observables bits"
# A strong answer crosses the wall while other requests are open, so it
# carries the name of the request it answers in front of those bits
# (decoders/decoder_output.py ANSWER_NAME_BITS_BY_TIER).
STRONG_RESULT_PAYLOAD_SOURCE = (
    "DecodeResult.logical_observables bits behind the request's name"
)

# A decision crosses the control fabric as one bus word: the decoder
# sequencer's 32-bit WISHBONE interface (Caune et al., arXiv 2410.05202,
# Methods). A command to the pulse controller is one instruction word:
# QubiC's distributed processor implements every instruction as a 128-bit
# word (Fruitwala et al., arXiv 2404.15260, Sec. III and IV).
BUS_WORD_BITS = 32  # 2410.05202 line 999
BUS_WORD_SOURCE = (
    "one 32-bit control bus word (Caune et al. 2410.05202, WISHBONE)"
)
INSTRUCTION_WORD_BITS = 128  # 2404.15260 line 177
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
# decoder_output.py send_region). The selection rides the same hop first
# as the request's name alone, and the region carries that name in front
# of its rounds (records/windows.py REQUEST_KEY_WIRE_BITS).
ESCALATION_PAYLOAD_SOURCE = "EscalatedRegion.message_bits()"

# The two controller-to-store hops carry the packed round at the width
# it leaves the controller: the detection events where the controller
# forms them and the raw measurement outcomes where a later seat does
# (detection_events.formed_at,
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
_REFERENCE_CLOCK_MEGAHERTZ = 250  # 2605.04892 line 1063
_REFERENCE_CYCLE_MICROSECONDS = 1 / _REFERENCE_CLOCK_MEGAHERTZ

# Yang et al., arXiv 2605.04892, Table I, is the one loop of decsim's scale
# broken down hop by hop: a d=3 surface code whose readout modules send
# bit strings over low-latency links to a decoder FPGA, which sends branch
# control over further links to the pulse generators (lines 184-199).
# The readout hop starts where Table I's acquisition window ends, which
# decsim's round period already holds, and it includes turning the signal
# into bits, which is why the card is the reference number a caller's own
# readout cost must not repeat. Each qubit has its own demodulation
# channel (QubiC, Fruitwala et al. 2404.15260, lines 709-715), so a
# round's bits arrive in parallel and the hop serializes nothing.
_READOUT_LATENCY_MICROSECONDS = 0.048  # 2605.04892 lines 1053-1055
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
WEAK_STORE_LATENCY_MICROSECONDS = 0.018  # half 36 ns, 2605.04892 line 1056
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
STRONG_STORE_LATENCY_MICROSECONDS = 0.26  # 2410.05202 line 177
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
_PULSE_LATENCY_MICROSECONDS = 0.088  # 2605.04892 lines 1058-1060
_PULSE_SOURCE = (
    "Yang 2605.04892 Table I lines 1058-1060: trigger propagation to the "
    "pulse generator 16 ns, waveform generation 32 ns, DAC chip 40 ns; "
    "QICK 2110.00557 lines 884-886 measure a 45 ns DAC"
)

# The rates. A word per cycle on the 32-bit bus the decoder is written
# and read over (Caune 2410.05202 lines 998-1008 and 1252-1254) and on
# the on-chip memory word above: the width is Caune's, and one word a
# 250 MHz cycle is an estimate, since Caune clock that bus at 156.25 MHz
# (lines 1021-1027) and give no transaction rate. One 128-bit
# instruction word per cycle into the pulse generator: the width is
# QubiC's, and decsim runs it on its 250 MHz controller clock, while
# QubiC's processors run at 500 MHz (2404.15260 lines 716-717). Off
# board, the 100 Gb direct-attach cable from an FPGA controller to its
# coprocessor host (Backline, arXiv 2609.09270, lines 1229-1230), the
# one off-board line rate a referent of this scale gives.
_WORD_BITS_PER_MICROSECOND = BUS_WORD_BITS * _REFERENCE_CLOCK_MEGAHERTZ
_WORD_RATE_SOURCE = (
    "one 32-bit word per 250 MHz cycle, an estimate: the width is Caune "
    "2410.05202 line 999's bus, which Caune clock at 156.25 MHz (lines "
    "1021-1027)"
)
_INSTRUCTION_BITS_PER_MICROSECOND = (
    INSTRUCTION_WORD_BITS * _REFERENCE_CLOCK_MEGAHERTZ
)
_INSTRUCTION_RATE_SOURCE = (
    "one 128-bit instruction word per 250 MHz controller cycle, an "
    "estimate: the width is QubiC 2404.15260 lines 176-179's, whose "
    "processors run at 500 MHz (lines 716-717)"
)
_OFF_BOARD_BITS_PER_MICROSECOND = 100_000  # 2609.09270 lines 1229-1230
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
# 376-388). The steady-state mean and median are 3.839 us (line 485).
NVQLINK_ROUND_TRIP_MICROSECONDS = 3.839  # 2510.25213 line 485
# 32 bytes (2510.25213 lines 402-403)
NVQLINK_ECHO_PAYLOAD_BITS = 256
NVQLINK_SOURCE = (
    "NVQLink 2510.25213 line 485, 3.839 us steady-state median round "
    "trip, FPGA to a persistent GPU kernel and back over RoCE"
)
_NVQLINK_RATE_SOURCE = (
    "100 Gb/s, NVQLink 2510.25213 line 382 and Fig. 2 (lines 438-439), "
    "the FPGA's Ethernet link to the GPU host's NIC"
)

# decsim prices one number per hop, so each leg of the round trip is
# charged half of it, less the echo payload's time on the cable, which
# every transfer pays for its own bits; the coprocessor's own poll is
# charged nothing.
# The legs the measurement covers are the controller's one-sided write
# into the coprocessor's memory (paper lines 1610-1612), the coprocessor's
# poll "on the expected memory buffer" (1616-1617) and the reply's
# one-sided write back (1617-1620).
ROCE_V2_WRITE_LEG = (
    "the controller's one-sided write into the coprocessor's memory, one "
    "half of the measured round trip less the echo payload's time on the "
    "cable; the paper gives no per-direction split"
)
ROCE_V2_ESCALATION_LEG = (
    "the escalation request, the same one-sided write from the "
    "controller side, one half of the measured round trip less the echo "
    "payload's time on the cable; the paper gives no per-direction split"
)
ROCE_V2_POLL_LEG = (
    "zero: the coprocessor polls the buffer in its own memory that the "
    "write landed in"
)
ROCE_V2_REPLY_LEG = (
    "the reply's one-sided write back to the controller, one half of the "
    "measured round trip less the echo payload's time on the cable; the "
    "paper gives no per-direction split"
)


def logical_reference_profile() -> settings.FabricSettings:
    """The default card: every hop from a system of decsim's scale.

    A surface-code patch on one control rack, a decoder FPGA, and a strong
    node one hop away. A hop is unbounded where its referent moves every bit
    in parallel. The weak loop is Yang et al.'s closed loop (arXiv
    2605.04892 Table I) hop for hop, the strong node is Caune et al.'s
    inter-chassis hop away (arXiv 2410.05202 Fig. 1a), and the rates are the
    referents' bus words and Backline's cable.
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
        STRONG_RESULT_PAYLOAD_SOURCE,
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

    The latencies are the reference card's, so switching cards changes
    bandwidth alone. Capacity is bits per microsecond. Every path carries
    exactly its nominal traffic in one commit region, so at capacity_scale 1
    each link runs at utilization one and at scale s at 1/s. State the
    utilization when reporting: a single-server queue at utilization one
    waits zero only under perfectly periodic arrivals (Little's law). The
    rates are an explicit per-link provisioning, as ns-3 declares a DataRate
    per device.

    Each rate is an exact fraction, so a nominal payload serializes in
    exactly its period: a float such as 8 / 1.1 lands below the true rate,
    and the rounded-up serialization then queues one tick more every round.
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
    # request's name, and its answer is one flip behind the same name
    name_bits = window_records.REQUEST_KEY_WIRE_BITS
    region_message_bits = name_bits + strong_window_bits
    escalation_bits = name_bits + region_message_bits
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
        region_message_bits,
        escalation_bits_per_microsecond,
        "one selection and one strong window of rcom+2rbuf rounds per "
        "commit region, each behind the request's name "
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
        strong_answer_bits,
        strong_answer_bits_per_microsecond,
        "one frame-update bit per logical observable per commit region, "
        "behind the request's name",
        STRONG_RESULT_PAYLOAD_SOURCE,
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
    )


def _measured_round_trip_card(
    round_trip_microseconds: float,
    measurement_source: str,
    rate_source: str,
    echo_payload_bits: int,
    row_name: str,
) -> settings.FabricSettings:
    """The reference card with the strong path on a measured round trip.

    Backline (arXiv 2609.09270, Sec. V-C and Table III) and NVQLink (arXiv
    2510.25213, Sec. 2.4) each measure one FPGA-to-coprocessor round trip
    over RoCE, with the coprocessor polling its own memory. Neither gives a
    per-direction number, so the split is decsim's rule: the write into the
    strong syndrome buffer, the escalation request and the reply to the
    frame each carry half the round trip, and the strong store's read is
    zero because the coprocessor polls its own memory. The three cable legs
    serialize at 100 Gb/s, and each leg's latency is its half less the echo
    payload's time on the cable, so an echo takes the measured median
    exactly: not a warm-up round trip nor the tail.
    """
    cable_rate = settings.CapacitySettings(
        _OFF_BOARD_BITS_PER_MICROSECOND, rate_source
    )
    strong_paths = _roce_v2_strong_paths(
        round_trip_microseconds,
        measurement_source,
        cable_rate,
        echo_payload_bits,
    )
    reference = logical_reference_profile()
    return dataclasses.replace(reference, **strong_paths, profile_name=row_name)


class RoceV2CpuFabric:
    """The reference card with the strong path on Backline's CPU round trip.

    Backline's FPGA-to-CPU round trip over RoCE v2, 2.305 us median (arXiv
    2609.09270, Table III), split as _measured_round_trip_card says.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The reference card with this row's strong paths."""
        return _measured_round_trip_card(
            ROCE_V2_CPU_ROUND_TRIP_MICROSECONDS,
            ROCE_V2_CPU_SOURCE,
            _OFF_BOARD_RATE_SOURCE,
            ROCE_V2_ECHO_PAYLOAD_BITS,
            "roce_v2_cpu",
        )


class RoceV2GpuFabric:
    """The reference card with the strong path on Backline's GPU round trip.

    Backline's FPGA-to-GPU round trip over RoCE v2, 4.5 us median (arXiv
    2609.09270, Table III), split as _measured_round_trip_card says. It is
    slower and wider than the CPU path because a CPU thread the GPU signals
    writes the reply.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The reference card with this row's strong paths."""
        return _measured_round_trip_card(
            ROCE_V2_GPU_ROUND_TRIP_MICROSECONDS,
            ROCE_V2_GPU_SOURCE,
            _OFF_BOARD_RATE_SOURCE,
            ROCE_V2_ECHO_PAYLOAD_BITS,
            "roce_v2_gpu",
        )


class NvqlinkGpuFabric:
    """The reference card with the strong path on NVQLink's GPU round trip.

    NVQLink's FPGA to persistent GPU kernel round trip over RoCE, 3.839 us
    steady-state median (arXiv 2510.25213, line 485), split as
    _measured_round_trip_card says.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The reference card with this row's strong paths.

        The connection is unreliable by choice (2510.25213 lines 376-388) on
        cables engineered under 1e-15 bit errors, so the ideal protocol row; the
        measured round trip already holds its 32-byte echo's framing (line 403),
        so no framing row.
        """
        return _measured_round_trip_card(
            NVQLINK_ROUND_TRIP_MICROSECONDS,
            NVQLINK_SOURCE,
            _NVQLINK_RATE_SOURCE,
            NVQLINK_ECHO_PAYLOAD_BITS,
            "nvqlink_gpu",
        )


def path_card(
    links: settings.FabricSettings,
    path_name: str,
    *,
    clock: config.Clock,
    latency_cycles: int,
    bits_per_cycle: Optional[float],
    source: str,
    lane_count: int = 1,
    setup_cycles_per_transfer: int = 0,
    header_bits_per_transfer: int = 0,
    protocol: Optional[settings.PacketProtocolSettings] = None,
) -> settings.PathSettings:
    """One path priced in cycles of a clock, over its payload rule in links.

    A cycle is the clock's period in whole ticks, gem5's cyclesToTicks
    (src/sim/clocked_object.hh:227), so a link's cycles and every other
    component's cycles on one clock are the same ticks. Each of lane_count
    lanes moves bits_per_cycle every period; None is an unbounded wire. The
    path gets a wire of its own, and its latency times the wire alone, so
    the receiver prices its own processing.
    """
    period_ticks = clock.period_ticks
    latency_ticks = latency_cycles * period_ticks
    setup_ticks = setup_cycles_per_transfer * period_ticks
    capacity = None
    if bits_per_cycle is not None:
        lane_bits_per_cycle = fractions.Fraction(str(bits_per_cycle))
        bits_per_period = lane_bits_per_cycle * lane_count
        bits_per_tick = bits_per_period / period_ticks
        bits_per_microsecond = bits_per_tick * config.TICKS_PER_MICROSECOND
        capacity = settings.CapacitySettings(bits_per_microsecond, source)
    channel = settings.ChannelSettings(
        path_name, latency_ticks, capacity, source, protocol
    )
    path_settings = getattr(links, path_name)
    return dataclasses.replace(
        path_settings,
        channel=channel,
        setup_ticks=setup_ticks,
        header_bits_per_transfer=header_bits_per_transfer,
        excludes_receiver_processing=True,
    )


def with_path_latency(
    links: settings.FabricSettings,
    path_name: str,
    latency_microseconds: float,
) -> settings.FabricSettings:
    """The card with one path's wire at a latency, the rest as it was.

    The latency is rounded to whole ticks once; the path keeps its rate,
    setup, framing and what its latency covers.
    """
    config.check_duration("latency_microseconds", latency_microseconds)
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    path = getattr(links, path_name)
    channel = dataclasses.replace(
        path.channel, propagation_latency_ticks=latency_ticks
    )
    changed = dataclasses.replace(path, channel=channel)
    return dataclasses.replace(links, **{path_name: changed})


def _roce_v2_strong_paths(
    round_trip_microseconds: float,
    measurement_source: str,
    cable_rate: settings.CapacitySettings,
    echo_payload_bits: int,
) -> dict:
    """The four strong-side paths, priced from one measured round trip.

    The three cable legs serialize at its rate, so the echo payload's time
    on the cable comes out of each leg's half; the poll crosses no cable and
    stays unbounded.
    """
    rate = cable_rate.input_bits_per_microsecond
    echo_fraction = echo_payload_bits / rate
    echo_on_the_cable = float(echo_fraction)
    half_microseconds = round_trip_microseconds / 2
    leg_microseconds = half_microseconds - echo_on_the_cable
    write_source = f"{measurement_source}; {ROCE_V2_WRITE_LEG}"
    escalation_source = f"{measurement_source}; {ROCE_V2_ESCALATION_LEG}"
    poll_source = f"{measurement_source}; {ROCE_V2_POLL_LEG}"
    reply_source = f"{measurement_source}; {ROCE_V2_REPLY_LEG}"
    controller_to_strong_buffer = _actual_path(
        "controller_to_strong_buffer",
        leg_microseconds,
        write_source,
        ROUND_PAYLOAD_SOURCE,
        cable_rate,
    )
    weak_decoder_to_strong_decoder = _actual_path(
        "weak_decoder_to_strong_decoder",
        leg_microseconds,
        escalation_source,
        ESCALATION_PAYLOAD_SOURCE,
        cable_rate,
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
        STRONG_RESULT_PAYLOAD_SOURCE,
        cable_rate,
    )
    return {
        "controller_to_strong_buffer": controller_to_strong_buffer,
        "weak_decoder_to_strong_decoder": weak_decoder_to_strong_decoder,
        "strong_buffer_to_strong_decoder": strong_buffer_to_strong_decoder,
        "strong_decoder_to_frame": strong_decoder_to_frame,
    }


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
        reference_excludes_processing = (
            reference_path.excludes_receiver_processing
        )
        reference_channel = reference_path.channel
        latency_ticks = reference_channel.propagation_latency_ticks
        channel = settings.ChannelSettings(
            name, latency_ticks, capacity, source
        )
        payload = settings.PayloadSettings(bits, source)
        return settings.PathSettings(
            channel,
            payload,
            actual_payload_source,
            excludes_receiver_processing=reference_excludes_processing,
        )


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
    # a preset's latency is its referent's end-to-end number
    return settings.PathSettings(
        channel, None, actual_payload_source, excludes_receiver_processing=False
    )


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
    # a preset's latency is its referent's end-to-end number
    return settings.PathSettings(
        channel, payload, None, excludes_receiver_processing=False
    )
