"""The reference card carries its sources' numbers.

Sources: Yang et al. 2605.04892 Table I for the reference card's weak
loop, Caune et al. 2410.05202 Fig. 1a F for its strong node, and ns-3's
txTime, bits over the DataRate (point-to-point-net-device.cc:243), for
its law; Caune et al. 2410.05202 (a 32-bit bus word) and Fruitwala et al.
2404.15260 (a 128-bit instruction word) for the default payloads. The
presets' cards are tested in tests/presets/test_link_cards.py.
"""

import dataclasses
import fractions
import math
import re

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine
import decsim.presets.machines as machines
import decsim.records.transfers as transfer_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph_layout as graph_layout
import tests.experiments.run_files as run_files

# the four hops a strong-node card prices as a cable or chassis crossing
STRONG_NODE_CROSSINGS = (
    "controller_to_strong_buffer",
    "weak_to_strong",
    "strong_to_frame",
    "strong_to_weak",
)
INSTRUCTION_PATHS = (
    transfer_records.FRAME_TO_CONTROLLER,
    transfer_records.CONTROLLER_TO_QPU,
)


def reference_card():
    """The reference card of a weak pool escalating to one on the host."""
    switching = run_files.switching_machine()
    reference = fabrics.ReferenceFabric()
    on_the_reference = dataclasses.replace(switching, fabric=reference)
    at_task = on_the_reference.at_task()
    layout = graph_layout.layout_of(at_task)
    return layout.card


def channel_names_of(profile):
    names = []
    for path in profile.paths:
        names.append(path.settings.channel.name)
    return names


def test_a_reference_hop_delivers_its_latency_after_its_bits_serialize():
    """ns-3: delivery is txTime, bits over the rate, plus the delay.

    point-to-point-net-device.cc:243 times the packet from its size and
    the DataRate, and the receiver has it the channel delay later. A d=11
    round of 120 bits on the weak store's hop, rounded to the nearest tick.
    """
    profile = reference_card()
    weak_store_path = profile.path_settings("controller_to_weak_buffer")
    weak_store = weak_store_path.channel
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
    profile = reference_card()
    unbounded = _unbounded_paths(profile)
    readout = profile.path_settings("qpu_to_controller")
    assert unbounded == ["qpu_to_controller", "weak_to_weak"]
    assert readout.excludes_receiver_processing is False


def test_the_reference_card_prices_a_bus_word_and_an_instruction_word():
    profile = reference_card()
    decision = profile.path_settings("frame_to_controller")
    command = profile.path_settings("controller_to_qpu")
    assert decision.default_payload.input_bits == 32
    assert command.default_payload.input_bits == 128


def test_the_reference_card_prices_a_boundary_on_the_seam_it_updates():
    profile = reference_card()
    seam = profile.path_settings("weak_to_weak")
    assert seam.default_payload is None
    assert seam.actual_payload_source == (
        "DependencyResidual seam-layer detectors"
    )


def test_the_reference_card_names_the_runtime_quantity_of_each_actual_path():
    profile = reference_card()
    sources = {}
    for path in profile.paths:
        sources[path.name] = path.settings.actual_payload_source
    assert sources["qpu_to_controller"] == "QPUReadout.size_bits"
    assert sources["controller_to_weak_buffer"] == "PackedRound.wire_bits"
    assert sources["controller_to_strong_buffer"] == "PackedRound.wire_bits"
    assert sources["weak_buffer_to_weak"] == "DecodeJob.payload_bits()"
    assert sources["weak_to_strong"] == "EscalatedRegion.message_bits()"
    assert sources["strong_buffer_to_strong"] == "DecodeJob.payload_bits()"
    assert sources["weak_to_frame"] == "DecodeResult.logical_observables bits"
    assert sources["strong_to_frame"] == (
        "DecodeResult.logical_observables bits behind the request's name"
    )


def _unbounded_paths(profile) -> list:
    """The names of the paths whose channel carries no rate."""
    unbounded = []
    for path in profile.paths:
        if path.settings.channel.capacity is None:
            unbounded.append(path.name)
    return unbounded


def test_every_hop_of_the_reaction_path_has_a_reference_card():
    """A null card is the reference numbers, so no hop is ever free.

    The traffic report lists one row per hop of the reaction path, and
    each row is priced: walking the card's hops finds a card on every one,
    and each card's channel carries the hop's own name.
    """
    profile = reference_card()
    names = channel_names_of(profile)
    path_names = profile.path_names()
    assert names == list(path_names)


def test_the_reference_weak_loop_is_yangs_control_electronics():
    """Yang 2605.04892 Table I lines 1053-1061, the loop's links.

    Readout 12 + 32 + 4 ns, the two digital links 36 ns, and trigger,
    waveform and DAC 16 + 32 + 40 ns: the 180 ns control-electronics
    subtotal less the 8 ns backplane logic, which is the controller's
    own decision work and not a hop.
    """
    profile = reference_card()
    loop_names = (
        "qpu_to_controller",
        "controller_to_weak_buffer",
        "weak_to_frame",
        "controller_to_qpu",
    )
    loop_paths = [profile.path_settings(name) for name in loop_names]
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
    profile = reference_card()
    path_settings = profile.path_settings(path)
    channel = path_settings.channel
    crossing_ticks = config.microseconds_to_ticks(0.26)
    assert channel.propagation_latency_ticks == crossing_ticks
    assert "Caune 2410.05202 Fig. 1a F" in channel.configuration_source


def test_a_run_without_a_card_uses_the_reference_card():
    """Each hop written out as its reference path runs as the default."""
    default_graph = machines.base()
    default_machine = machine.Machine.build(default_graph)
    default_result = default_machine.run()
    explicit_fabric = _reference_paths_written_out(default_graph)
    explicit_graph = dataclasses.replace(default_graph, fabric=explicit_fabric)
    explicit_machine = machine.Machine.build(explicit_graph)
    explicit_result = explicit_machine.run()
    assert default_result == explicit_result


def _reference_paths_written_out(graph) -> fabrics.RoleLinks:
    """Each hop of the graph's card as its reference path, by its role."""
    at_task = graph.at_task()
    layout = graph_layout.layout_of(at_task)
    links = {}
    for path in layout.card.paths:
        reference = link_profiles.reference_path(path.role, path.name)
        links[path.role.key] = fabrics.FixedPath(reference)
    return fabrics.RoleLinks(**links)


# an arXiv identifier as the cards write it: four digits, a dot, four or
# five digits
ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")


def sources_of(profile):
    """The configuration source of each path's channel, by path name."""
    sources = {}
    for path in profile.paths:
        sources[path.name] = path.settings.channel.configuration_source
    return sources


def _capacity_sources(profile) -> list:
    """The source of each rate the card declares, in path order."""
    sources = []
    for path in profile.paths:
        capacity = path.settings.channel.capacity
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
    profile = reference_card()
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
    profile = reference_card()
    path_settings = profile.path_settings(path)
    word_bits = path_settings.default_payload.input_bits
    capacity = path_settings.channel.capacity
    rate = capacity.input_bits_per_microsecond
    word_ticks = word_bits * config.TICKS_PER_MICROSECOND / rate
    assert word_ticks == 4000
