"""The reference card: every hop's latency, rate and payload, cited.

A card is a latency and a rate per hop, so a transfer takes its latency
plus its bits over the rate, the law ns-3's point-to-point device times
a packet by (point-to-point-net-device.cc:243). Every number carries a
source string on the record it sets, and a payload's source travels into
the traffic report; paper locators are arXiv numbers and text lines. A
graph's fabric (decsim/topology/fabrics.py) prices each hop it names
and leaves the rest on this card; the shipped links and fabrics are in
decsim/presets/link_cards.py.
"""

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
# A strong answer crosses the wall while other requests are open, so it
# carries the name of the request it answers in front of those bits
# (build/routes.py names a hop across the wall).
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
# as the request's name, with the weak crossing commit behind it when the
# strong host joins it, and the region carries that name in front of its
# rounds (records/windows.py REQUEST_KEY_WIRE_BITS).
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
    # the frame is in the controller beside the weak chip, so the strong
    # answer's hop to either is the one inter-chassis hop
    strong_decoder_to_weak_decoder = _actual_path(
        "strong_decoder_to_weak_decoder",
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
        strong_decoder_to_weak_decoder=strong_decoder_to_weak_decoder,
        profile_name="logical_reference",
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
