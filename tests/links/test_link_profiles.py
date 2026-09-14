"""The number cards carry their sources' numbers.

Sources: Khalid et al. 2511.10633 Table I for the reference latencies;
Caune et al.
2410.05202 (a 32-bit bus word) and Fruitwala et al. 2404.15260 (a 128-bit
instruction word) for the default payloads; the provisioning rule of the
bandwidth card (each path carries its nominal traffic in one commit
region, ns-3's per-device DataRate); gem5-Aladdin's setup cost (Shao et
al., MICRO 2016) for with_transfer_overhead; Backline 2609.09270
Table III, the CPU and GPU echo rows, for the two measured RoCE v2 rows.
"""

import ast
import pathlib
import re

import pytest

import decsim.config as config
import decsim.front.experiment as experiment
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings
import tests.front.yaml_configs as yaml_configs

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
        capacities[path.value] = capacity.aggregate_bits_per_microsecond
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


def test_the_reference_card_carries_its_sources_latencies():
    profile = link_profiles.logical_reference_profile()
    assert latencies_of(profile) == {
        "qpu_to_controller": config.microseconds_to_ticks(0.15),
        "controller_to_weak_buffer": config.microseconds_to_ticks(0.04),
        "controller_to_strong_buffer": config.microseconds_to_ticks(0.26),
        "weak_buffer_to_weak_decoder": config.microseconds_to_ticks(2.0),
        "weak_decoder_to_strong_decoder": config.microseconds_to_ticks(0.5),
        "strong_buffer_to_strong_decoder": config.microseconds_to_ticks(2.0),
        "weak_decoder_to_frame": config.microseconds_to_ticks(1.0),
        "decoder_to_decoder": config.microseconds_to_ticks(0.5),
        "strong_decoder_to_frame": config.microseconds_to_ticks(1.0),
        "frame_to_controller": config.microseconds_to_ticks(4.0),
        "controller_to_qpu": config.microseconds_to_ticks(0.15),
    }


def test_the_reference_card_is_unbounded_on_every_path():
    profile = link_profiles.logical_reference_profile()
    assert profile.qpu_to_controller.channel.capacity is None
    assert profile.controller_to_weak_buffer.channel.capacity is None
    assert profile.controller_to_strong_buffer.channel.capacity is None
    assert profile.weak_buffer_to_weak_decoder.channel.capacity is None
    assert profile.weak_decoder_to_strong_decoder.channel.capacity is None
    assert profile.strong_buffer_to_strong_decoder.channel.capacity is None
    assert profile.weak_decoder_to_frame.channel.capacity is None
    assert profile.decoder_to_decoder.channel.capacity is None
    assert profile.strong_decoder_to_frame.channel.capacity is None
    assert profile.frame_to_controller.channel.capacity is None
    assert profile.controller_to_qpu.channel.capacity is None
    assert profile.profile_name == "logical_reference"
    assert profile.qpu_to_controller.excludes_receiver_processing is False


def test_the_reference_card_prices_a_bus_word_and_an_instruction_word():
    profile = link_profiles.logical_reference_profile()
    assert profile.frame_to_controller.default_payload.aggregate_bits == 32
    assert profile.controller_to_qpu.default_payload.aggregate_bits == 128


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
        == "no payload; the escalation names the strong request"
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
        "weak_decoder_to_strong_decoder": 0.2,
        "strong_buffer_to_strong_decoder": 72.0,
        "weak_decoder_to_frame": 0.2,
        "decoder_to_decoder": 4.8,
        "strong_decoder_to_frame": 0.2,
        "frame_to_controller": 6.4,
        "controller_to_qpu": 25.6,
    }
    assert profile.profile_name == "bandwidth_limited"


def test_the_bandwidth_card_keeps_the_reference_latencies():
    reference = link_profiles.logical_reference_profile()
    bandwidth = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert latencies_of(bandwidth) == latencies_of(reference)


def test_the_bandwidth_cards_default_payloads_are_one_regions_traffic():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert profile.qpu_to_controller.default_payload.aggregate_bits == 24
    assert (
        profile.weak_buffer_to_weak_decoder.default_payload.aggregate_bits
        == 240
    )
    assert (
        profile.strong_buffer_to_strong_decoder.default_payload.aggregate_bits
        == 360
    )
    assert (
        profile.weak_decoder_to_strong_decoder.default_payload.aggregate_bits
        == 1
    )
    assert profile.decoder_to_decoder.default_payload.aggregate_bits == 24


def test_the_capacity_scale_multiplies_every_rate():
    halved = link_profiles.bandwidth_limited_profile(
        **DISTANCE_5_GEOMETRY, capacity_scale=0.5
    )
    capacities = capacities_of(halved)
    assert capacities["qpu_to_controller"] == 12.0
    assert capacities["controller_to_qpu"] == 12.8


def test_a_capacity_scale_of_zero_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        link_profiles.bandwidth_limited_profile(
            **DISTANCE_5_GEOMETRY, capacity_scale=0.0
        )


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


def test_the_reference_card_prices_the_two_controller_to_buffer_hops():
    """Caune's per-hop stages: the intra-unit write, and the crossing.

    Syndrome buffer 0 sits with the controller, so its write is Fig. 1a
    stage D, the 40 ns the control system takes to handle a result
    message and prepare it for broadcast; syndrome buffer 1 sits at room
    temperature, so its write leaves the chassis on stage F, the 240 to
    260 ns inter-node broadcast, taken at the caption's stated worst
    case (arXiv:2410.05202, Fig. 1a).
    """
    profile = link_profiles.logical_reference_profile()
    weak_store = profile.controller_to_weak_buffer
    strong_store = profile.controller_to_strong_buffer
    weak_ticks = config.microseconds_to_ticks(0.04)
    strong_ticks = config.microseconds_to_ticks(0.26)
    weak_source = weak_store.channel.configuration_source
    strong_source = strong_store.channel.configuration_source
    assert weak_store.channel.propagation_latency_ticks == weak_ticks
    assert strong_store.channel.propagation_latency_ticks == strong_ticks
    assert "Caune 2410.05202 Fig. 1a D" in weak_source
    assert "Caune 2410.05202 Fig. 1a F" in strong_source


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
        return _CountingLinkFabric(card, engine)


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
        physical_error_probability=0.001, distance=3, round_period_us=1.0
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


def test_every_reference_latency_cites_a_paper_or_says_it_is_a_choice():
    """A number on the card carries where it came from.

    Ten of the eleven latencies are card facts read off a published
    table, so each names its arXiv identifier; the weak-to-strong
    selection hop has no referent on disk and says so instead of
    borrowing the authority of one.
    """
    profile = link_profiles.logical_reference_profile()
    sources = sources_of(profile)
    cited = _named_where(sources, _cites_a_paper)
    declared = _named_where(sources, _declares_a_choice)
    assert declared == ["weak_decoder_to_strong_decoder"]
    assert len(cited) == 10


KHALID_TABLE = "Khalid 2511.10633 Table I "


def _khalid_symbol(source: str) -> str:
    """The Table I row symbol one source reads, or an empty string."""
    if not source.startswith(KHALID_TABLE):
        return ""
    rest = source[len(KHALID_TABLE) :]
    words = rest.split()
    return words[0].rstrip(",")


def test_the_khalid_latencies_name_the_table_row_they_are_read_from():
    """Five hops take a Khalid Table I row; each names the row's symbol."""
    profile = link_profiles.logical_reference_profile()
    sources = sources_of(profile)
    symbols = {}
    for name, source in sources.items():
        symbols[name] = _khalid_symbol(source)
    assert symbols["qpu_to_controller"] == "tqc"
    assert symbols["weak_buffer_to_weak_decoder"] == "tcd"
    assert symbols["strong_buffer_to_strong_decoder"] == "tcd"
    assert symbols["weak_decoder_to_frame"] == "tdo"
    assert symbols["strong_decoder_to_frame"] == "tdo"
    assert symbols["decoder_to_decoder"] == "tdd"


# The hops the measured RoCE v2 rows reprice: the write into syndrome
# buffer 1, the escalation, the strong window's input, and the reply.
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

    The controller's one-sided write into syndrome buffer 1, the
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
        physical_error_probability=0.001, distance=3, round_period_us=1.0
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
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert settings.links.kind == "roce_v2_gpu"
    assert result.terminal_status == "complete"
    assert "2609.09270" in strong_buffer_source_of(built)
