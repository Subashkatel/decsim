"""The number cards carry their sources' numbers.

Sources: Yang et al. 2605.04892 Table I for the reference card's weak
loop, Caune et al. 2410.05202 Fig. 1a F for its strong node, and ns-3's
txTime, bits over the DataRate (point-to-point-net-device.cc:243), for
its law; Caune et al. 2410.05202 (a 32-bit bus word) and Fruitwala et al.
2404.15260 (a 128-bit instruction word) for the default payloads; the
provisioning rule of the bandwidth card (each path carries its nominal
traffic in one commit region, ns-3's per-device DataRate); gem5-Aladdin's
setup cost (Shao et al., MICRO 2016) for with_transfer_overhead; Backline
2609.09270 Table III, the CPU and GPU echo rows, for the two measured
RoCE v2 rows.
"""

import ast
import fractions
import math
import pathlib
import re

import pytest

import decsim.config as config
import decsim.engine
import decsim.experiments.experiment as experiment
import decsim.links.channel as channel_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings
import tests.experiments.yaml_configs as yaml_configs

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
    unbounded = []
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        if path_settings.channel.capacity is None:
            unbounded.append(path.value)
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
        == "EscalatedRegion.wire_bits"
    )
    assert profile.strong_buffer_to_strong_decoder.actual_payload_source == (
        "DecodeJob.payload_bits()"
    )
    assert profile.weak_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )
    assert profile.strong_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )


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
    unknown = {}
    for source in stated:
        _add_unknown_field(source, declared, unknown)
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
        "weak_decoder_to_strong_decoder": 72.0,
        "strong_buffer_to_strong_decoder": 72.0,
        "weak_decoder_to_frame": fractions.Fraction("0.2"),
        "decoder_to_decoder": fractions.Fraction("4.8"),
        "strong_decoder_to_frame": fractions.Fraction("0.2"),
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
        profile.weak_decoder_to_strong_decoder.default_payload.input_bits == 360
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


def test_the_setup_cost_lands_on_the_two_decoder_input_paths():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_transfer_overhead(
        reference, overhead_microseconds=0.4
    )
    assert (
        profile.weak_buffer_to_weak_decoder.setup_ticks
        == config.microseconds_to_ticks(0.4)
    )
    assert (
        profile.strong_buffer_to_strong_decoder.setup_ticks
        == config.microseconds_to_ticks(0.4)
    )
    assert profile.qpu_to_controller.setup_ticks == 0
    assert profile.profile_name == "logical_reference+transfer_overhead"


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
    loop_ticks = 0
    for path_settings in loop_paths:
        loop_ticks += path_settings.channel.propagation_latency_ticks
    control_electronics_microseconds = 0.180
    backplane_logic_microseconds = 0.008
    loop_microseconds = (
        control_electronics_microseconds - backplane_logic_microseconds
    )
    assert loop_ticks == config.microseconds_to_ticks(loop_microseconds)


def test_every_hop_to_or_from_the_strong_node_is_one_chassis_crossing():
    """Caune 2410.05202 Fig. 1a F, the inter-chassis hop, at 260 ns."""
    profile = link_profiles.logical_reference_profile()
    crossings = (
        profile.controller_to_strong_buffer,
        profile.weak_decoder_to_strong_decoder,
        profile.strong_decoder_to_frame,
    )
    crossing_ticks = config.microseconds_to_ticks(0.26)
    for path_settings in crossings:
        channel = path_settings.channel
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


class CountingFabric:
    """A fabric row written outside decsim: it counts what it carried.

    Its base card is the reference row's, so a run on it prices every hop
    exactly as the shipped row does and the count is the only difference.
    """

    sent = []

    @staticmethod
    def base_card():
        """The numbers the section's per-path cards override."""
        return link_profiles.logical_reference_profile()

    @staticmethod
    def build(card, engine):
        """One LinkFabric, with every send counted on the way through."""
        return _CountingLinkFabric(card, engine, channel_module.Channel)


class _CountingLinkFabric(fabric_module.LinkFabric):
    """The reference fabric, counting the transfers it carried."""

    def send(self, path, payload_bits, now_ticks, attribution, on_delivered):
        """Count the send, then carry it."""
        CountingFabric.sent.append(path)
        reference = super()
        reference.send(path, payload_bits, now_ticks, attribution, on_delivered)


def test_a_fabric_row_written_outside_decsim_runs_from_a_yaml(
    monkeypatch, tmp_path
):
    """One LINK_FABRICS row and one links.kind is the whole edit."""
    monkeypatch.setitem(link_profiles.LINK_FABRICS, "counting", CountingFabric)
    monkeypatch.setattr(CountingFabric, "sent", [])
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "counting"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    experiment_config = experiment.load_experiment(config_path)
    settings = experiment_config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert settings.links.kind == "counting"
    assert isinstance(built.links, _CountingLinkFabric)
    assert result.terminal_status == "complete"
    assert CountingFabric.sent != []


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
    sources = list(path_sources)
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        capacity = path_settings.channel.capacity
        if capacity is not None:
            sources.append(capacity.source)
    uncited = []
    for source in sources:
        if not _cites_a_paper(source):
            uncited.append(source)
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


def test_the_cpu_row_charges_half_the_measured_round_trip_on_each_leg():
    """Backline times one round trip; decsim needs a number per hop.

    The controller's one-sided write into the strong syndrome buffer, the
    escalation request and the reply to the frame are each half of the
    2.305 us median (2609.09270 Table III, CPU echo); the strong store's
    read into the strong decoder is free, because the coprocessor polls
    a slot in its own memory. The escalation round trip is therefore the
    measured median exactly.
    """
    profile = link_profiles.roce_v2_measured_profile("cpu")
    latencies = latencies_of(profile)
    half = config.microseconds_to_ticks(1.1525)
    assert latencies["controller_to_strong_buffer"] == half
    assert latencies["weak_decoder_to_strong_decoder"] == half
    assert latencies["strong_buffer_to_strong_decoder"] == 0
    assert latencies["strong_decoder_to_frame"] == half
    escalation_round_trip = (
        latencies["weak_decoder_to_strong_decoder"]
        + latencies["strong_buffer_to_strong_decoder"]
        + latencies["strong_decoder_to_frame"]
    )
    assert escalation_round_trip == config.microseconds_to_ticks(2.305)


def test_the_gpu_row_charges_half_the_measured_round_trip_on_each_leg():
    """The same split on the 4.5 us GPU echo row of Table III."""
    profile = link_profiles.roce_v2_measured_profile("gpu")
    latencies = latencies_of(profile)
    half = config.microseconds_to_ticks(2.25)
    assert latencies["controller_to_strong_buffer"] == half
    assert latencies["weak_decoder_to_strong_decoder"] == half
    assert latencies["strong_buffer_to_strong_decoder"] == 0
    assert latencies["strong_decoder_to_frame"] == half
    escalation_round_trip = (
        latencies["weak_decoder_to_strong_decoder"]
        + latencies["strong_buffer_to_strong_decoder"]
        + latencies["strong_decoder_to_frame"]
    )
    assert escalation_round_trip == config.microseconds_to_ticks(4.5)


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


def test_an_instruction_hop_moves_its_word_in_one_cycle():
    """QubiC 2404.15260: a 32-bit result word, a 128-bit instruction.

    Each word crosses its hop whole, one per 250 MHz cycle, 4000 ticks.
    """
    profile = link_profiles.logical_reference_profile()
    instruction_paths = (
        profile.frame_to_controller,
        profile.controller_to_qpu,
    )
    for path_settings in instruction_paths:
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
    settings = experiment_config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert settings.links.kind == "roce_v2_cpu"
    assert result.terminal_status == "complete"
    assert "2609.09270" in strong_buffer_source_of(built)


def test_the_measured_gpu_row_runs_from_a_yaml(tmp_path):
    """The same, on the row that prices the GPU coprocessor's path."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "roce_v2_gpu"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    experiment_config = experiment.load_experiment(config_path)
    settings = experiment_config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert settings.links.kind == "roce_v2_gpu"
    assert result.terminal_status == "complete"
    assert "2609.09270" in strong_buffer_source_of(built)
