"""The reference card carries its sources' numbers.

Sources: Yang et al. 2605.04892 Table I for the reference card's weak
loop, Caune et al. 2410.05202 Fig. 1a F for its strong node, and ns-3's
txTime, bits over the DataRate (point-to-point-net-device.cc:243), for
its law; Caune et al. 2410.05202 (a 32-bit bus word) and Fruitwala et al.
2404.15260 (a 128-bit instruction word) for the default payloads. The
presets' cards are tested in tests/presets/test_link_cards.py.
"""

import fractions
import math
import re

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings

# the four hops a strong-node card prices as a cable or chassis crossing
STRONG_NODE_CROSSINGS = (
    transfer_records.CONTROLLER_TO_STRONG_BUFFER,
    transfer_records.WEAK_DECODER_TO_STRONG_DECODER,
    transfer_records.STRONG_DECODER_TO_FRAME,
    transfer_records.STRONG_DECODER_TO_WEAK_DECODER,
)
INSTRUCTION_PATHS = (
    transfer_records.FRAME_TO_CONTROLLER,
    transfer_records.CONTROLLER_TO_QPU,
)


def channel_names_of(profile):
    names = []
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        names.append(path_settings.channel.name)
    return names


def path_names():
    names = []
    for path in transfer_records.LEGACY_PATHS:
        names.append(path)
    return names


def test_a_reference_hop_delivers_its_latency_after_its_bits_serialize():
    """ns-3: delivery is txTime, bits over the rate, plus the delay.

    point-to-point-net-device.cc:243 times the packet from its size and
    the DataRate, and the receiver has it the channel delay later. A d=11
    round of 120 bits on the weak store's hop, rounded to the nearest tick.
    """
    profile = link_profiles.logical_reference_profile()
    weak_store = profile.controller_to_weak_buffer.channel
    engine = decsim.engine.Engine()
    channel = channel_module.Channel(weak_store, engine)
    framed = transfer_records.FramedPayload(120)
    delivered = []
    channel.send(framed, 0, 0, delivered.append)
    engine.run()
    transfer = delivered[0]
    rate = weak_store.capacity.input_bits_per_microsecond
    exact_ticks = 120 * config.TICKS_PER_MICROSECOND / rate
    half_tick = fractions.Fraction(1, 2)
    half_up_ticks = exact_ticks + half_tick
    serialization_ticks = math.floor(half_up_ticks)
    latency_ticks = weak_store.propagation_latency_ticks
    assert transfer.delivery_ticks == serialization_ticks + latency_ticks


def test_the_reference_card_leaves_unbounded_only_the_parallel_hops():
    """Readout and the seam move every bit at once; the rest serialize.

    Each qubit has its own demodulation channel (QubiC 2404.15260 lines
    709-715) and each graph edge its own link (Helios 2301.08419 lines
    764-769), so those two hops carry no rate.
    """
    profile = link_profiles.logical_reference_profile()
    unbounded = _unbounded_paths(profile)
    assert unbounded == ["qpu_to_controller", "decoder_to_decoder"]
    assert profile.profile_name == "logical_reference"
    assert profile.qpu_to_controller.excludes_receiver_processing is False


def test_the_reference_card_prices_a_bus_word_and_an_instruction_word():
    profile = link_profiles.logical_reference_profile()
    assert profile.frame_to_controller.default_payload.input_bits == 32
    assert profile.controller_to_qpu.default_payload.input_bits == 128


def test_the_reference_card_prices_a_boundary_on_the_seam_it_updates():
    profile = link_profiles.logical_reference_profile()
    assert profile.decoder_to_decoder.default_payload is None
    assert profile.decoder_to_decoder.actual_payload_source == (
        "DependencyResidual seam-layer detectors"
    )


def test_the_reference_card_names_the_runtime_quantity_of_each_actual_path():
    profile = link_profiles.logical_reference_profile()
    assert (
        profile.qpu_to_controller.actual_payload_source
        == "QPUReadout.size_bits"
    )
    assert profile.controller_to_weak_buffer.actual_payload_source == (
        "PackedRound.wire_bits"
    )
    assert profile.controller_to_strong_buffer.actual_payload_source == (
        "PackedRound.wire_bits"
    )
    assert profile.weak_buffer_to_weak_decoder.actual_payload_source == (
        "DecodeJob.payload_bits()"
    )
    assert (
        profile.weak_decoder_to_strong_decoder.actual_payload_source
        == "EscalatedRegion.message_bits()"
    )
    assert profile.strong_buffer_to_strong_decoder.actual_payload_source == (
        "DecodeJob.payload_bits()"
    )
    assert profile.weak_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )
    assert profile.strong_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits behind the request's name"
    )


def _unbounded_paths(profile) -> list:
    """The names of the paths whose channel carries no rate."""
    unbounded = []
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        if path_settings.channel.capacity is None:
            unbounded.append(path)
    return unbounded


def test_every_hop_of_the_reaction_path_has_a_reference_card():
    """A null card is the reference numbers, so no hop is ever free.

    The traffic report lists one row per hop of the reaction path, and
    each row is priced: walking LEGACY_PATHS finds a card on every one, and
    each card's channel carries the hop's own name.
    """
    profile = link_profiles.logical_reference_profile()
    names = channel_names_of(profile)
    assert names == path_names()


def test_the_reference_weak_loop_is_yangs_control_electronics():
    """Yang 2605.04892 Table I lines 1053-1061, the loop's links.

    Readout 12 + 32 + 4 ns, the two digital links 36 ns, and trigger,
    waveform and DAC 16 + 32 + 40 ns: the 180 ns control-electronics
    subtotal less the 8 ns backplane logic, which is the controller's
    own decision work and not a hop.
    """
    profile = link_profiles.logical_reference_profile()
    loop_paths = (
        profile.qpu_to_controller,
        profile.controller_to_weak_buffer,
        profile.weak_decoder_to_frame,
        profile.controller_to_qpu,
    )
    loop_ticks = sum(
        path_settings.channel.propagation_latency_ticks
        for path_settings in loop_paths
    )
    control_electronics_microseconds = 0.180
    backplane_logic_microseconds = 0.008
    loop_microseconds = (
        control_electronics_microseconds - backplane_logic_microseconds
    )
    assert loop_ticks == config.microseconds_to_ticks(loop_microseconds)


@pytest.mark.parametrize("path", STRONG_NODE_CROSSINGS)
def test_every_hop_to_or_from_the_strong_node_is_one_chassis_crossing(path):
    """Caune 2410.05202 Fig. 1a F, the inter-chassis hop, at 260 ns."""
    profile = link_profiles.logical_reference_profile()
    path_settings = profile.path_settings(path)
    channel = path_settings.channel
    crossing_ticks = config.microseconds_to_ticks(0.26)
    assert channel.propagation_latency_ticks == crossing_ticks
    assert "Caune 2410.05202 Fig. 1a F" in channel.configuration_source


def test_a_run_without_a_card_uses_the_reference_card():
    default_settings = machine_settings.MachineSettings()
    default_machine = machine.Machine.build(default_settings)
    default_result = default_machine.run()
    reference = link_profiles.logical_reference_profile()
    explicit_settings = machine_settings.MachineSettings(links=reference)
    explicit_machine = machine.Machine.build(explicit_settings)
    explicit_result = explicit_machine.run()
    assert default_result == explicit_result


# an arXiv identifier as the cards write it: four digits, a dot, four or
# five digits
ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")


def sources_of(profile):
    """The configuration source of each path's channel, by path name."""
    sources = {}
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        sources[path] = path_settings.channel.configuration_source
    return sources


def _capacity_sources(profile) -> list:
    """The source of each rate the card declares, in path order."""
    sources = []
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        capacity = path_settings.channel.capacity
        if capacity is not None:
            sources.append(capacity.source)
    return sources


def _uncited(sources: list) -> list:
    uncited = []
    for source in sources:
        if not _cites_a_paper(source):
            uncited.append(source)
    return uncited


def _cites_a_paper(source: str) -> bool:
    """Whether one source carries an arXiv identifier."""
    found = ARXIV.search(source)
    return found is not None


def test_every_reference_number_cites_a_paper():
    """Each latency and each rate on the card names its arXiv source."""
    profile = link_profiles.logical_reference_profile()
    sources_by_path = sources_of(profile)
    path_sources = sources_by_path.values()
    capacity_sources = _capacity_sources(profile)
    sources = [*path_sources, *capacity_sources]
    uncited = _uncited(sources)
    assert uncited == []


@pytest.mark.parametrize("path", INSTRUCTION_PATHS)
def test_an_instruction_hop_moves_its_word_in_one_cycle(path):
    """QubiC 2404.15260: a 32-bit result word, a 128-bit instruction.

    Each word crosses its hop whole, one per 250 MHz cycle, 4000 ticks.
    """
    profile = link_profiles.logical_reference_profile()
    path_settings = profile.path_settings(path)
    word_bits = path_settings.default_payload.input_bits
    capacity = path_settings.channel.capacity
    rate = capacity.input_bits_per_microsecond
    word_ticks = word_bits * config.TICKS_PER_MICROSECOND / rate
    assert word_ticks == 4000
