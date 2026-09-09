"""The link number cards: the two reference fabrics and the yaml's own.

logical_reference_profile is the default when no links card is given:
every channel unbounded, so it prices propagation only and no transfer
ever queues; latencies from Khalid et al., arXiv 2511.10633, Table I,
and the two controller-to-buffer hops from Caune et al. Fig. 1a. Khalid
Table I also gives a size per channel and a channel count beside each
latency; decsim takes the latencies only, because it wires one channel
per path and counts each transfer's own payload from the record that
carries it, so the paper's channel counts would price nothing here.
bandwidth_limited_profile is the same fabric with finite calibrated rates
so contention becomes measurable; capacity_scale sweeps the whole fabric.
from_yaml puts the yaml's own card on any path; with_transfer_overhead
adds a setup cost to either fabric.

Every number carries a source string that travels into the traffic
report; paper locators are arXiv numbers and sections. To
change a number, copy a card into your own file and edit it, then pass
it as the machine's links setting.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.links.fabric as fabric
import decsim.links.settings as settings
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

# The two controller-to-store hops carry the packed round as the
# controller assembled it: the raw measurement bits, counted before the
# detection events are formed (controller/round_assembly.py, wire_bits
# of the merged raw fragments). The store then holds the formed events,
# so a store's contents and the bits this card charges are two different
# counts on the same round.
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

# The controller's write into a syndrome buffer is a hop of the control
# system, and Caune et al., arXiv:2410.05202, Fig. 1a is the referent that
# measures such hops one by one, with worst-case values where measured.
# Syndrome buffer 0 sits with the controller, so its write is stage D,
# "time required to handle result message and prepare for broadcast"
# (40 ns). Syndrome buffer 1 sits at room temperature, so its write leaves
# the chassis: stage F, "inter-node delay time for broadcasting between
# control system chassis" (240 to 260 ns), taken at the stated worst case.
# Google, arXiv:2408.13687, gives the same topology without a per-hop
# number: bits go to a workstation over low-latency Ethernet and are then
# streamed to the decoder through a shared memory buffer. Toshio et al.,
# arXiv:2510.25222, Table I price the whole controller-to-decoder path as
# T_comm^weak = tau_gen and T_comm^strong = 10 tau_gen, which decsim
# splits into this hop, the buffer-to-decoder hop and the
# decoder-to-frame hop, so those symbols bound the sum of three cards
# rather than either card here.
WEAK_STORE_LATENCY_MICROSECONDS = 0.04
WEAK_STORE_SOURCE = (
    "Caune 2410.05202 Fig. 1a D, result message handled and prepared for "
    "broadcast, 40 ns"
)
STRONG_STORE_LATENCY_MICROSECONDS = 0.26
STRONG_STORE_SOURCE = (
    "Caune 2410.05202 Fig. 1a F, inter-node broadcast between control "
    "system chassis, 240 to 260 ns at the stated worst case"
)


def logical_reference_profile() -> settings.FabricSettings:
    """The default card: Khalid's latencies, unbounded bandwidth.

    Propagation only, nothing ever queues. Actual-payload paths price the
    runtime's own bit counts; default-payload paths price a stated word
    width, not one of Khalid's per-channel sizes.
    """
    qpu_to_controller = _actual_path(
        "qpu_to_controller",
        0.15,
        "Khalid 2511.10633 Table I tqc, syndrome transfer from QPU "
        "to controller",
        "SyndromePayload.size_bits",
    )
    controller_to_weak_buffer = _actual_path(
        "controller_to_weak_buffer",
        WEAK_STORE_LATENCY_MICROSECONDS,
        WEAK_STORE_SOURCE,
        ROUND_PAYLOAD_SOURCE,
    )
    controller_to_strong_buffer = _actual_path(
        "controller_to_strong_buffer",
        STRONG_STORE_LATENCY_MICROSECONDS,
        STRONG_STORE_SOURCE,
        ROUND_PAYLOAD_SOURCE,
    )
    weak_buffer_to_weak_decoder = _actual_path(
        "weak_buffer_to_weak_decoder",
        2.0,
        "Khalid 2511.10633 Table I tcd, syndrome transfer from "
        "controller to decoders; the integrated weak-input transfer",
        DECODER_INPUT_PAYLOAD_SOURCE,
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
        "Khalid 2511.10633 Table I tcd mapped to the strong input",
        DECODER_INPUT_PAYLOAD_SOURCE,
    )
    weak_decoder_to_frame = _actual_path(
        "weak_decoder_to_frame",
        1.0,
        "Khalid 2511.10633 Table I tdo, decoding results transfer, "
        "mapped to the weak output",
        RESULT_PAYLOAD_SOURCE,
    )
    decoder_to_decoder = _actual_path(
        "decoder_to_decoder",
        0.5,
        "Khalid 2511.10633 Table I tdd, decoder-to-decoder exchange",
        BOUNDARY_PAYLOAD_SOURCE,
    )
    strong_decoder_to_frame = _actual_path(
        "strong_decoder_to_frame",
        1.0,
        "Khalid 2511.10633 Table I tdo, decoding results transfer",
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
    strong_window_rounds = window_records.strong_region_round_count(
        commit_rounds, buffer_rounds
    )
    strong_window_bits = strong_window_rounds * syndrome_bits_per_round
    one_per_region = 1 / commit_region_microseconds
    round_bits_per_microsecond = syndrome_bits_per_round / round_microseconds
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
    provisioning = _Provisioning(capacity_scale)
    qpu_to_controller = provisioning.path(
        "qpu_to_controller",
        0.15,
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one syndrome round per round period",
        "SyndromePayload.size_bits",
    )
    controller_to_weak_buffer = provisioning.path(
        "controller_to_weak_buffer",
        WEAK_STORE_LATENCY_MICROSECONDS,
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one packed round per round period",
        ROUND_PAYLOAD_SOURCE,
    )
    controller_to_strong_buffer = provisioning.path(
        "controller_to_strong_buffer",
        STRONG_STORE_LATENCY_MICROSECONDS,
        syndrome_bits_per_round,
        round_bits_per_microsecond,
        "one packed round per round period",
        ROUND_PAYLOAD_SOURCE,
    )
    weak_buffer_to_weak_decoder = provisioning.path(
        "weak_buffer_to_weak_decoder",
        2.0,
        weak_window_bits,
        weak_window_bits_per_microsecond,
        "one weak window of rcom+rbuf rounds per commit region "
        "(Toshio 2510.25222 Sec. III C, "
        '"rcom = rbuf = d")',
        DECODER_INPUT_PAYLOAD_SOURCE,
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
        "(Toshio 2510.25222 Sec. III C, "
        '"In this paper, we assume that rstrong = rcom + 2rbuf.")',
        DECODER_INPUT_PAYLOAD_SOURCE,
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
        syndrome_bits_per_round,
        boundary_bits_per_microsecond,
        "one dense seam layer per commit region",
        BOUNDARY_PAYLOAD_SOURCE,
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


class LogicalReferenceFabric:
    """The default row: Khalid's latencies on unbounded channels.

    Every channel is unbounded, so the fabric prices propagation only and
    no transfer ever queues. This is the row a yaml gets when it names no
    kind, and the numbers its per-path cards override.
    """

    @staticmethod
    def base_card() -> settings.FabricSettings:
        """The numbers a yaml's per-path cards override."""
        return logical_reference_profile()

    @staticmethod
    def build(card: settings.FabricSettings, engine):
        """The object that carries this run's transfers."""
        return fabric.LinkFabric(card, engine)


class BandwidthLimitedFabric:
    """The same fabric with finite rates, provisioned from the geometry.

    Every channel carries exactly its nominal traffic in one round, so
    contention becomes measurable and capacity_scale sweeps the whole
    fabric. Its numbers come from the run's own geometry, not from the
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
    def build(card: settings.FabricSettings, engine):
        """The object that carries this run's transfers."""
        return fabric.LinkFabric(card, engine)


# links.kind names one of these rows: which fabric model carries the
# transfers and which numbers the section's per-path cards override. A
# row answers base_card at the yaml boundary and build at the root.
LINK_FABRICS = {
    "logical_reference": LogicalReferenceFabric,
    "bandwidth_limited": BandwidthLimitedFabric,
}


def from_yaml(
    section: Mapping, clocks: config.ClockSettings, name: str
) -> settings.FabricSettings:
    """The yaml's `links` section: a kind, and one card per path over it.

    A card prices its path in cycles of a named clock domain: latency,
    bits per cycle per lane (null is unbounded), the lane count, and an
    optional per-transfer setup cost. A null card keeps the chosen row's
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
