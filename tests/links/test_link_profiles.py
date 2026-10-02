"""The number cards carry their sources' numbers.

Sources: Yang et al. 2605.04892 Table I for the reference card's weak
loop, Caune et al. 2410.05202 Fig. 1a F for its strong node, and ns-3's
txTime, bits over the DataRate (point-to-point-net-device.cc:243), for
its law; Caune et al. 2410.05202 (a 32-bit bus word) and Fruitwala et al.
2404.15260 (a 128-bit instruction word) for the default payloads; the
provisioning rule of the bandwidth card (each path carries its nominal
traffic in one commit region, ns-3's per-device DataRate); Backline
2609.09270 Table III, the CPU and GPU echo rows, for the two measured
RoCE v2 rows.
"""

import ast
import fractions
import math
import pathlib
import re

import pytest

import decsim.collect as collect
import decsim.config as config
import decsim.engine
import decsim.experiments.experiment as experiment
import decsim.links.channel as channel_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings
import tests.experiments.yaml_configs as yaml_configs

# the three hops a strong-node card prices as a cable or chassis crossing
STRONG_NODE_CROSSINGS = (
    transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER,
    transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
    transfer_records.LinkPath.STRONG_DECODER_TO_FRAME,
)
INSTRUCTION_PATHS = (
    transfer_records.LinkPath.FRAME_TO_CONTROLLER,
    transfer_records.LinkPath.CONTROLLER_TO_QPU,
)

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent.parent
DECSIM_ROOT = PACKAGE_ROOT / "decsim"

# A payload source written as Record.field, or Record.method(), states
# where a transfer's bit count came from; anything else is prose about
# what the card assumes.
NAMED_FIELD = re.compile(r"^([A-Z]\w+)\.(\w+)(\(\))?$")

# One distance-5 patch: 24 syndrome bits per 1.0 us round, commit and
# buffer regions of 5 rounds.
DISTANCE_5_GEOMETRY = dict(
    syndrome_bits_per_round=24,
    round_microseconds=1.0,
    commit_rounds=5,
    buffer_rounds=5,
)


def latencies_of(profile):
    latencies = {}
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        channel = path_settings.channel
        latencies[path.value] = channel.propagation_latency_ticks
    return latencies


def capacities_of(profile):
    capacities = {}
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        capacity = path_settings.channel.capacity
        rate = capacity.exact_aggregate_bits_per_microsecond()
        capacities[path.value] = rate
    return capacities


def channel_names_of(profile):
    names = []
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        names.append(path_settings.channel.name)
    return names


def path_names():
    names = []
    for path in transfer_records.LinkPath:
        names.append(path.value)
    return names


def test_a_reference_hop_delivers_its_latency_after_its_bits_serialize():
    """ns-3: delivery is txTime, bits over the rate, plus the delay.

    point-to-point-net-device.cc:243 times the packet from its size and
    the DataRate, and the receiver has it the channel delay later. A d=11
    round of 120 bits on the weak store's hop, rounded up to whole ticks.
    """
    profile = link_profiles.logical_reference_profile()
    weak_store = profile.controller_to_weak_buffer.channel
    engine = decsim.engine.Engine()
    channel = channel_module.Channel(weak_store, engine)
    framed = channel_module.FramedPayload(120)
    delivered = []
    channel.send(framed, 0, 0, delivered.append)
    engine.run()
    transfer = delivered[0]
    rate = weak_store.capacity.exact_aggregate_bits_per_microsecond()
    exact_ticks = 120 * config.TICKS_PER_MICROSECOND / rate
    serialization_ticks = math.ceil(exact_ticks)
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
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        if path_settings.channel.capacity is None:
            unbounded.append(path.value)
    return unbounded


def payload_sources_of(profile) -> list:
    """Every payload source string one card states, path by path."""
    sources = []
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        _add_payload_sources(path_settings, sources)
    return sources


def _add_payload_sources(path_settings, sources: list) -> None:
    """Add one path's actual source and its default payload's source."""
    actual = path_settings.actual_payload_source
    if actual is not None:
        sources.append(actual)
    default_payload = path_settings.default_payload
    if default_payload is not None:
        sources.append(default_payload.source)


def names_by_class() -> dict:
    """Every class decsim declares, with the names it carries."""
    declared = {}
    modules = DECSIM_ROOT.rglob("*.py")
    for module in sorted(modules):
        text = module.read_text()
        tree = ast.parse(text)
        _add_module_classes(tree, declared)
    return declared


def _add_module_classes(tree: ast.Module, declared: dict) -> None:
    """Add one module's classes to the map of declared names."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            carried = declared.setdefault(node.name, set())
            carried |= _names_of_class(node)


def _names_of_class(node: ast.ClassDef) -> set:
    """The fields and methods one class body declares."""
    names = set()
    for statement in node.body:
        _add_statement_name(statement, names)
    return names


def _add_statement_name(statement, names: set) -> None:
    """Add the name one class-body statement declares, if it declares one."""
    if isinstance(statement, ast.FunctionDef):
        names.add(statement.name)
        return
    if isinstance(statement, ast.AnnAssign):
        target = statement.target
        if isinstance(target, ast.Name):
            names.add(target.id)
        return
    if isinstance(statement, ast.Assign):
        _add_assigned_names(statement, names)


def _add_assigned_names(statement: ast.Assign, names: set) -> None:
    """Add the plain names one assignment in a class body writes to."""
    for target in statement.targets:
        if isinstance(target, ast.Name):
            names.add(target.id)


def _unknown_fields(sources: list, declared: dict) -> dict:
    """Each source that names a field the tree does not have: its class."""
    unknown = {}
    for source in sources:
        _add_unknown_field(source, declared, unknown)
    return unknown


def _add_unknown_field(source: str, declared: dict, unknown: dict) -> None:
    """Record a payload source that names a field the tree does not have."""
    match = NAMED_FIELD.match(source)
    if match is None:
        return
    class_name, field_name = match.group(1), match.group(2)
    carried = declared.get(class_name, set())
    if field_name in carried:
        return
    unknown[source] = class_name


def test_every_payload_source_that_names_a_field_names_a_real_one():
    """A card that names a runtime quantity names one the tree carries.

    The string travels into the traffic ledger as the transfer's
    payload_source, so a record or field the tree does not have is a
    false statement about what crossed. A card that carries no runtime
    count says so in prose instead, and this law leaves prose alone.
    """
    declared = names_by_class()
    reference = link_profiles.logical_reference_profile()
    stated = payload_sources_of(reference)
    bandwidth = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    provisioned = payload_sources_of(bandwidth)
    stated.extend(provisioned)
    measured_cpu = link_profiles.roce_v2_measured_profile("cpu")
    measured_on_the_cpu_path = payload_sources_of(measured_cpu)
    stated.extend(measured_on_the_cpu_path)
    measured_gpu = link_profiles.roce_v2_measured_profile("gpu")
    measured_on_the_gpu_path = payload_sources_of(measured_gpu)
    stated.extend(measured_on_the_gpu_path)
    unknown = _unknown_fields(stated, declared)
    assert unknown == {}


def test_every_hop_of_the_reaction_path_has_a_reference_card():
    """A null card is the reference numbers, so no hop is ever free.

    The traffic report lists one row per hop of the reaction path, and
    each row is priced: walking LinkPath finds a card on every one, and
    each card's channel carries the hop's own name.
    """
    profile = link_profiles.logical_reference_profile()
    names = channel_names_of(profile)
    assert names == path_names()


def test_the_bandwidth_card_provisions_each_path_for_one_commit_region():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert capacities_of(profile) == {
        "qpu_to_controller": 24.0,
        "controller_to_weak_buffer": 24.0,
        "controller_to_strong_buffer": 24.0,
        "weak_buffer_to_weak_decoder": 48.0,
        # a selection and a region of 360 bits, each behind a 64-bit name
        "weak_decoder_to_strong_decoder": fractions.Fraction("97.6"),
        "strong_buffer_to_strong_decoder": 72.0,
        "weak_decoder_to_frame": fractions.Fraction("0.2"),
        "decoder_to_decoder": fractions.Fraction("4.8"),
        # one flip behind the same name
        "strong_decoder_to_frame": 13,
        "frame_to_controller": fractions.Fraction("6.4"),
        "controller_to_qpu": fractions.Fraction("25.6"),
    }
    assert profile.profile_name == "bandwidth_limited"


def test_a_nominal_round_serializes_in_exactly_one_round_period():
    """ns-3 txTime = bits / rate (point-to-point-net-device.cc:243).

    At 8 bits per 1.1 us a float rate falls below 80/11 bits per
    microsecond and the round would take one tick more than its period.
    """
    profile = link_profiles.bandwidth_limited_profile(
        syndrome_bits_per_round=8,
        round_microseconds=1.1,
        commit_rounds=3,
        buffer_rounds=3,
    )
    engine = decsim.engine.Engine()
    readout = profile.qpu_to_controller.channel
    channel = channel_module.Channel(readout, engine)
    framed = channel_module.FramedPayload(8)
    delivered = []
    channel.send(framed, 0, 0, delivered.append)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 1_100_000


def test_the_bandwidth_card_keeps_the_reference_latencies():
    reference = link_profiles.logical_reference_profile()
    bandwidth = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert latencies_of(bandwidth) == latencies_of(reference)


def test_the_bandwidth_cards_default_payloads_are_one_regions_traffic():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert profile.qpu_to_controller.default_payload.input_bits == 24
    assert profile.weak_buffer_to_weak_decoder.default_payload.input_bits == 240
    assert (
        profile.strong_buffer_to_strong_decoder.default_payload.input_bits
        == 360
    )
    assert (
        profile.weak_decoder_to_strong_decoder.default_payload.input_bits
        == 64 + 360
    )
    assert profile.decoder_to_decoder.default_payload.input_bits == 24


def test_the_capacity_scale_multiplies_every_rate():
    halved = link_profiles.bandwidth_limited_profile(
        **DISTANCE_5_GEOMETRY, capacity_scale=0.5
    )
    capacities = capacities_of(halved)
    assert capacities["qpu_to_controller"] == 12.0
    assert capacities["controller_to_qpu"] == fractions.Fraction("12.8")


def test_a_capacity_scale_of_zero_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        link_profiles.bandwidth_limited_profile(
            **DISTANCE_5_GEOMETRY, capacity_scale=0.0
        )


def test_a_cards_cycles_cost_its_clocks_period_in_whole_ticks():
    """gem5 cyclesToTicks, clockPeriod() * c (clocked_object.hh:227).

    At 300 MHz the period is 3333 ticks, so three cycles are 9999 ticks
    and eight bits at eight bits per cycle take one period, the same
    ticks every other component on the domain charges.
    """
    clocks = config.ClockSettings({"fridge": 300.0})
    card = {
        "latency_cycles": 3,
        "clock": "fridge",
        "bits_per_cycle": 8.0,
        "setup_cycles_per_transfer": 3,
    }
    section = {"controller_to_weak_buffer": card}
    profile = link_profiles.from_yaml(section, clocks, "clocked")
    path = profile.controller_to_weak_buffer
    engine = decsim.engine.Engine()
    channel = channel_module.Channel(path.channel, engine)
    framed = channel_module.FramedPayload(8)
    delivered = []
    channel.send(framed, 0, 0, delivered.append)
    engine.run()
    transfer = delivered[0]
    assert path.channel.propagation_latency_ticks == 9999
    assert path.setup_ticks == 9999
    assert transfer.serialization_ticks == 3333


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


def test_a_links_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "not_a_row"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    with pytest.raises(ValueError, match="links.kind 'not_a_row' is not"):
        experiment.load_experiment(config_path)


def test_the_bandwidth_row_says_what_it_needs_instead_of_a_yaml(tmp_path):
    """Its channels come from the sweep point's geometry, not the section."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "bandwidth_limited"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    with pytest.raises(ValueError, match="provisions every channel"):
        experiment.load_experiment(config_path)


# an arXiv identifier as the cards write it: four digits, a dot, four or
# five digits
ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")


def sources_of(profile):
    """The configuration source of each path's channel, by path name."""
    sources = {}
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        sources[path.value] = path_settings.channel.configuration_source
    return sources


def _capacity_sources(profile) -> list:
    """The source of each rate the card declares, in path order."""
    sources = []
    for path in transfer_records.LinkPath:
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


def _declares_a_choice(source: str) -> bool:
    """Whether one source says the number is decsim's own."""
    return "repository" in source


def _named_where(sources: dict, holds) -> list:
    """The path names whose source satisfies a predicate, in order."""
    names = []
    carded = sources.items()
    items = sorted(carded)
    for name, source in items:
        if holds(source):
            names.append(name)
    return names


def test_every_reference_number_cites_a_paper():
    """Each latency and each rate on the card names its arXiv source."""
    profile = link_profiles.logical_reference_profile()
    sources_by_path = sources_of(profile)
    path_sources = sources_by_path.values()
    capacity_sources = _capacity_sources(profile)
    sources = [*path_sources, *capacity_sources]
    uncited = _uncited(sources)
    assert uncited == []


# The hops the measured RoCE v2 rows reprice: the write into syndrome
# strong syndrome buffer, the escalation, the strong window's input, and the
# reply.
STRONG_SIDE_PATHS = (
    "controller_to_strong_buffer",
    "weak_decoder_to_strong_decoder",
    "strong_buffer_to_strong_decoder",
    "strong_decoder_to_frame",
)


def _priced(path_settings) -> tuple:
    """One path's latency, capacity and payload rule."""
    channel = path_settings.channel
    return (
        channel.propagation_latency_ticks,
        channel.capacity,
        path_settings.default_payload,
        path_settings.actual_payload_source,
    )


def paths_outside_the_strong_side(profile) -> dict:
    """What every hop but the four strong-side ones is priced at."""
    priced = {}
    for path in transfer_records.LinkPath:
        if path.value in STRONG_SIDE_PATHS:
            continue
        path_settings = profile.path_settings(path)
        priced[path.value] = _priced(path_settings)
    return priced


def _cites_backline(source: str) -> bool:
    """Whether one source carries Backline's arXiv identifier."""
    found = ARXIV.search(source)
    if found is None:
        return False
    identifier = found.group()
    return identifier == "2609.09270"


@pytest.mark.parametrize(
    ("measured_profile", "arguments", "echo_bits", "round_trip_microseconds"),
    [
        (link_profiles.roce_v2_measured_profile, ("cpu",), 128, 2.305),
        (link_profiles.roce_v2_measured_profile, ("gpu",), 128, 4.5),
        (link_profiles.nvqlink_measured_profile, (), 256, 3.839),
    ],
)
def test_an_echo_of_the_measured_payload_takes_the_measured_round_trip(
    measured_profile, arguments, echo_bits, round_trip_microseconds
):
    """The paper's own echo, replayed on the card, is the paper's median.

    Backline echoes a 16-byte payload (2609.09270 line 1611) and NVQLink
    a 32-byte one (2510.25213 lines 402-403), and each times the whole
    round trip, the payload's own time on the 100 Gb/s cable included.
    An echo is the escalation out, the poll of the coprocessor's own
    memory, and the reply back, each its latency and its bits' time on
    its wire.
    """
    profile = measured_profile(*arguments)
    out = profile.weak_decoder_to_strong_decoder
    poll = profile.strong_buffer_to_strong_decoder
    back = profile.strong_decoder_to_frame
    echo_ticks = (
        _crossing_ticks(out, echo_bits)
        + _crossing_ticks(poll, echo_bits)
        + _crossing_ticks(back, echo_bits)
    )
    measured_ticks = config.microseconds_to_ticks(round_trip_microseconds)
    write = profile.controller_to_strong_buffer

    assert echo_ticks == measured_ticks
    assert write.channel.propagation_latency_ticks == (
        out.channel.propagation_latency_ticks
    )
    assert poll.channel.propagation_latency_ticks == 0


def _crossing_ticks(path_settings, bits: int) -> int:
    """One transfer of `bits` alone on the path: latency and wire time."""
    channel = path_settings.channel
    latency = channel.propagation_latency_ticks
    if channel.capacity is None:
        return latency
    wire_ticks = channel_module.serialization_ticks(bits, channel.capacity)
    return latency + wire_ticks


@pytest.mark.parametrize("coprocessor", ["cpu", "gpu"])
def test_a_measured_rows_cable_legs_serialize_at_backlines_100_gbps(
    coprocessor,
):
    """Backline 2609.09270 lines 1229-1230: a 100 Gb direct-attach cable.

    The write, the escalation and the reply cross it at 100000 bits per
    microsecond; the poll reads the coprocessor's own memory, so it
    crosses no cable and is unbounded.
    """
    profile = link_profiles.roce_v2_measured_profile(coprocessor)
    write = profile.controller_to_strong_buffer.channel.capacity
    escalation = profile.weak_decoder_to_strong_decoder.channel.capacity
    reply = profile.strong_decoder_to_frame.channel.capacity
    poll = profile.strong_buffer_to_strong_decoder.channel.capacity
    write_rate = write.exact_aggregate_bits_per_microsecond()
    escalation_rate = escalation.exact_aggregate_bits_per_microsecond()
    reply_rate = reply.exact_aggregate_bits_per_microsecond()

    assert (write_rate, escalation_rate, reply_rate) == (100000,) * 3
    assert poll is None


def test_the_measured_rows_keep_the_reference_card_off_the_strong_side():
    """Only the four strong-side hops move: the rest is the default card."""
    reference = link_profiles.logical_reference_profile()
    measured_cpu = link_profiles.roce_v2_measured_profile("cpu")
    measured_gpu = link_profiles.roce_v2_measured_profile("gpu")
    unchanged = paths_outside_the_strong_side(reference)
    assert paths_outside_the_strong_side(measured_cpu) == unchanged
    assert paths_outside_the_strong_side(measured_gpu) == unchanged


def test_the_cpu_rows_strong_side_sources_cite_backline():
    """Four hops read one measurement, and none of them is a choice now.

    On the reference card the weak-to-strong hop names no paper and says
    "repository weak-to-strong model choice"; on this row it is a leg of
    a measured round trip and cites the paper it was read from.
    """
    profile = link_profiles.roce_v2_measured_profile("cpu")
    sources = sources_of(profile)
    cited = _named_where(sources, _cites_backline)
    assert cited == [
        "controller_to_strong_buffer",
        "strong_buffer_to_strong_decoder",
        "strong_decoder_to_frame",
        "weak_decoder_to_strong_decoder",
    ]
    declared = _named_where(sources, _declares_a_choice)
    assert declared == []


def test_the_gpu_rows_strong_side_sources_cite_backline():
    """The same four hops, read from the GPU echo row."""
    profile = link_profiles.roce_v2_measured_profile("gpu")
    sources = sources_of(profile)
    cited = _named_where(sources, _cites_backline)
    assert cited == [
        "controller_to_strong_buffer",
        "strong_buffer_to_strong_decoder",
        "strong_decoder_to_frame",
        "weak_decoder_to_strong_decoder",
    ]
    declared = _named_where(sources, _declares_a_choice)
    assert declared == []


@pytest.mark.parametrize("path", INSTRUCTION_PATHS)
def test_an_instruction_hop_moves_its_word_in_one_cycle(path):
    """QubiC 2404.15260: a 32-bit result word, a 128-bit instruction.

    Each word crosses its hop whole, one per 250 MHz cycle, 4000 ticks.
    """
    profile = link_profiles.logical_reference_profile()
    path_settings = profile.path_settings(path)
    word_bits = path_settings.default_payload.input_bits
    capacity = path_settings.channel.capacity
    rate = capacity.exact_aggregate_bits_per_microsecond()
    word_ticks = word_bits * config.TICKS_PER_MICROSECOND / rate
    assert word_ticks == 4000


def test_a_coprocessor_backline_did_not_echo_from_is_refused():
    """The measurement covers two paths, and the refusal names them."""
    with pytest.raises(ValueError, match="'cpu' or 'gpu'"):
        link_profiles.roce_v2_measured_profile("fpga")


def strong_buffer_source_of(built) -> str:
    """The source a run's traffic report kept for the strong-buffer hop."""
    snapshot = built.observation.traffic.snapshot()
    wanted = transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER
    for channel in snapshot.channels:
        if wanted in channel.member_paths:
            return channel.settings.configuration_source
    return ""


def test_the_measured_cpu_row_runs_from_a_yaml(tmp_path):
    """One links.kind is the whole edit, and the run carries the source."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "roce_v2_cpu"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    experiment_config = experiment.load_experiment(config_path)
    point = experiment_config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert result.terminal_status == "complete"
    assert "2609.09270" in strong_buffer_source_of(built)


def test_the_measured_gpu_row_runs_from_a_yaml(tmp_path):
    """The same, on the row that prices the GPU coprocessor's path."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "roce_v2_gpu"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    experiment_config = experiment.load_experiment(config_path)
    point = experiment_config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert result.terminal_status == "complete"
    assert "2609.09270" in strong_buffer_source_of(built)


def test_the_nvqlink_rows_strong_side_sources_cite_nvqlink():
    """The four strong-side hops read one measurement, 2510.25213."""
    profile = link_profiles.nvqlink_measured_profile()
    sources = sources_of(profile)
    cited = _named_where(sources, _cites_nvqlink)
    assert cited == [
        "controller_to_strong_buffer",
        "strong_buffer_to_strong_decoder",
        "strong_decoder_to_frame",
        "weak_decoder_to_strong_decoder",
    ]


def test_the_nvqlink_rows_strong_side_retransmits_nothing():
    """An unreliable connection by choice (2510.25213 lines 376-388)."""
    profile = link_profiles.nvqlink_measured_profile()
    strong_side = (
        profile.controller_to_strong_buffer,
        profile.weak_decoder_to_strong_decoder,
        profile.strong_buffer_to_strong_decoder,
        profile.strong_decoder_to_frame,
    )
    protocols = [path.channel.protocol for path in strong_side]
    assert protocols == [None] * 4


@pytest.mark.parametrize("path", STRONG_NODE_CROSSINGS)
def test_the_nvqlink_rows_cable_legs_serialize_at_100_gbps(path):
    """NVQLink's 100 Gb Ethernet link (2510.25213 line 382, Fig. 2)."""
    profile = link_profiles.nvqlink_measured_profile()
    path_settings = profile.path_settings(path)
    capacity = path_settings.channel.capacity
    rate = capacity.exact_aggregate_bits_per_microsecond()
    assert rate == 100000


def test_the_nvqlink_row_keeps_the_reference_card_off_the_strong_side():
    reference = link_profiles.logical_reference_profile()
    measured = link_profiles.nvqlink_measured_profile()
    unchanged = paths_outside_the_strong_side(reference)
    assert paths_outside_the_strong_side(measured) == unchanged


def test_a_python_path_card_is_the_yaml_card_of_the_same_numbers():
    """One way to price a path: the yaml's card goes through path_card.

    The rate stays the exact fraction of the decimal, so the two cards
    name one point (collect.json_value, which a point's id hashes).
    """
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    card = {
        "latency_cycles": 40,
        "clock": "fridge",
        "bits_per_cycle": 38.79,
        "channels": 4,
        "setup_cycles_per_transfer": 2,
        "header_bits_per_transfer": 16,
    }
    section = {"controller_to_weak_buffer": card}
    from_yaml = link_profiles.from_yaml(section, clocks, "mine")
    reference = link_profiles.logical_reference_profile()
    fridge = config.Clock.from_megahertz(250.0)

    python_card = link_profiles.path_card(
        reference,
        "controller_to_weak_buffer",
        clock=fridge,
        latency_cycles=40,
        bits_per_cycle=38.79,
        source="2603.16203 lines 895-897",
        lane_count=4,
        setup_cycles_per_transfer=2,
        header_bits_per_transfer=16,
    )

    yaml_card = from_yaml.controller_to_weak_buffer
    python_value = collect.json_value(python_card, keep_labels=False)
    yaml_value = collect.json_value(yaml_card, keep_labels=False)
    rate = python_card.channel.capacity.input_bits_per_microsecond
    assert python_card == yaml_card
    assert python_value == yaml_value
    assert rate == fractions.Fraction(38790)
    assert python_card.excludes_receiver_processing is True


def test_a_path_latency_in_microseconds_moves_that_path_alone():
    reference = link_profiles.logical_reference_profile()

    links = link_profiles.with_path_latency(
        reference, "frame_to_controller", 0.5
    )

    changed = links.frame_to_controller.channel
    assert changed.propagation_latency_ticks == 500_000
    assert links.controller_to_qpu == reference.controller_to_qpu


def test_a_path_latency_that_rounds_to_no_ticks_is_refused():
    reference = link_profiles.logical_reference_profile()
    sentence = "latency_microseconds is positive but rounds to zero ticks"

    with pytest.raises(ValueError, match=sentence):
        link_profiles.with_path_latency(reference, "frame_to_controller", 1e-7)


def _cites_nvqlink(source: str) -> bool:
    """Whether one source carries NVQLink's arXiv identifier."""
    found = ARXIV.search(source)
    if found is None:
        return False
    identifier = found.group()
    return identifier == "2510.25213"
